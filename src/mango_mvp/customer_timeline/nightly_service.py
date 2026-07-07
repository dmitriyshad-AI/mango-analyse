from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    run_nightly_incremental,
    summarize_report,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path, is_customer_timeline_prod_path


NIGHTLY_SERVICE_SCHEMA_VERSION = "customer_timeline_nightly_service_v1"


@dataclass(frozen=True)
class NightlyServiceStep:
    name: str
    kind: str
    enabled: bool = True
    required: bool = True
    config: Optional[NightlyIncrementalConfig] = None
    monitor_config: Optional[Mapping[str, Any]] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class NightlyServiceConfig:
    timeline_db: Path
    allowed_root: Path
    out_root: Path
    publish_dir: Path
    steps: Sequence[NightlyServiceStep]
    tenant_id: str = "foton"
    lock_timeout_seconds: float = 30.0
    actor: str = "customer_timeline_nightly_service"


def run_nightly_service(config: NightlyServiceConfig) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    timeline_db, allowed_root, out_root, publish_dir = validated_service_paths(config)
    out_root.mkdir(parents=True, exist_ok=True)
    publish_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": NIGHTLY_SERVICE_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started.isoformat(),
        "timeline_db": str(timeline_db),
        "allowed_root": str(allowed_root),
        "out_root": str(out_root),
        "publish_dir": str(publish_dir),
        "tenant_id": config.tenant_id,
        "steps": [],
        "safety": {
            "writes_prod_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "sends_messages": False,
            "writes_staging_db": True,
            "installs_launchd": False,
        },
    }
    with service_lock(timeline_db, timeout_seconds=config.lock_timeout_seconds) as lock_info:
        report["lock"] = lock_info
        failed_required_steps: list[str] = []
        for index, step in enumerate(config.steps, start=1):
            step_started = time.monotonic()
            if not step.enabled:
                status = "failed_required_disabled" if step.required else "skipped_disabled"
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "reason": step.reason,
                        "duration_seconds": 0.0,
                    }
                )
                continue
            if step.kind == "local_freshness_monitor":
                step_report = run_local_freshness_monitor(
                    step,
                    timeline_db=timeline_db,
                    allowed_root=allowed_root,
                    tenant_id=config.tenant_id,
                    actor=config.actor,
                )
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("status") == "ok" else (
                    "failed" if step.required else "skipped_optional_failed"
                )
                if status == "failed":
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": step_report,
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind != "nightly_incremental":
                reason = f"unsupported nightly service step kind: {step.kind}"
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    failed_step_report(
                        index=index,
                        step=step,
                        reason=reason,
                        duration_seconds=round(time.monotonic() - step_started, 3),
                    )
                )
                continue
            if step.config is None:
                reason = f"enabled step {step.name} requires config"
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    failed_step_report(
                        index=index,
                        step=step,
                        reason=reason,
                        duration_seconds=round(time.monotonic() - step_started, 3),
                    )
                )
                continue
            try:
                step_report = run_nightly_incremental(step.config)
            except Exception as exc:  # service-level fail-soft: report and keep manifest writing.
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    failed_step_report(
                        index=index,
                        step=step,
                        reason=f"step_exception:{type(exc).__name__}",
                        duration_seconds=round(time.monotonic() - step_started, 3),
                        error=exc,
                    )
                )
                continue
            step_path = run_dir / f"{index:02d}_{step.name}.json"
            write_json(step_path, step_report)
            summary = summarize_report(step_report)
            step_failed_required_sources = summary.get("failed_required_sources") or ()
            step_gate_passed = bool(summary.get("gate_passed", True))
            status = "ok" if step_gate_passed else "failed_required_source"
            if not step_gate_passed and step.required:
                failed_required_steps.append(step.name)
            report["steps"].append(
                {
                    "index": index,
                    "name": step.name,
                    "kind": step.kind,
                    "status": status,
                    "required": step.required,
                    "report_path": str(step_path),
                    "summary": summary,
                    "failed_required_sources": list(step_failed_required_sources),
                    "duration_seconds": round(time.monotonic() - step_started, 3),
                }
            )
        manifest = build_snapshot_manifest(timeline_db, tenant_id=config.tenant_id)
        manifest["run_id"] = run_id
        manifest["service_report_path"] = str(run_dir / "service_report.json")
        manifest["published_at"] = datetime.now(timezone.utc).isoformat()
        failed_required_steps = list(dict.fromkeys(failed_required_steps))
        report["failed_required_steps"] = failed_required_steps
        report["partial_failure"] = bool(failed_required_steps)
        report["overall_status"] = "partial" if failed_required_steps else "ok"
        manifest_path = publish_dir / f"customer_timeline_snapshot_{run_id}.json"
        write_json(manifest_path, manifest)
        latest_path = publish_dir / "latest_customer_timeline_snapshot.json"
        latest_published = not failed_required_steps
        if latest_published:
            shutil.copyfile(manifest_path, latest_path)
        report["snapshot_manifest"] = {
            "path": str(manifest_path),
            "latest_path": str(latest_path) if latest_published else None,
            "latest_published": latest_published,
            "sha256": manifest["files"]["sqlite"]["sha256"],
            "counts": manifest["counts"],
        }
        finished = datetime.now(timezone.utc)
        report["finished_at"] = finished.isoformat()
        report["duration_seconds"] = round((finished - started).total_seconds(), 3)
        write_json(run_dir / "service_report.json", report)
    return report


def service_config_from_json(path: Path) -> NightlyServiceConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("nightly service config must be a JSON object")
    timeline_db = Path(str(payload["timeline_db"]))
    allowed_root = Path(str(payload.get("allowed_root") or timeline_db.parent))
    out_root = Path(str(payload["out_root"]))
    publish_dir = Path(str(payload["publish_dir"]))
    tenant_id = str(payload.get("tenant_id") or "foton")
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("nightly service config must contain non-empty steps")
    steps = tuple(
        service_step_from_json(
            item,
            timeline_db=timeline_db,
            allowed_root=allowed_root,
            tenant_id=tenant_id,
            actor=str(payload.get("actor") or "customer_timeline_nightly_service"),
        )
        for item in steps_payload
    )
    config = NightlyServiceConfig(
        timeline_db=timeline_db,
        allowed_root=allowed_root,
        out_root=out_root,
        publish_dir=publish_dir,
        steps=steps,
        tenant_id=tenant_id,
        lock_timeout_seconds=float(payload.get("lock_timeout_seconds", 30.0)),
        actor=str(payload.get("actor") or "customer_timeline_nightly_service"),
    )
    validated_service_paths(config)
    return config


def service_step_from_json(
    payload: Any,
    *,
    timeline_db: Path,
    allowed_root: Path,
    tenant_id: str,
    actor: str,
) -> NightlyServiceStep:
    if not isinstance(payload, Mapping):
        raise ValueError("nightly service step must be an object")
    name = str(payload.get("name") or "")
    kind = str(payload.get("kind") or "")
    if not name or not kind:
        raise ValueError("nightly service step requires name and kind")
    enabled = bool(payload.get("enabled", True))
    required = bool(payload.get("required", True))
    reason = str(payload.get("reason")) if payload.get("reason") else None
    config = None
    monitor_config = None
    if kind == "nightly_incremental":
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError(f"step {name} requires config")
        sources_payload = config_payload.get("sources")
        if not isinstance(sources_payload, list) or not sources_payload:
            raise ValueError(f"step {name} requires non-empty sources")
        config = NightlyIncrementalConfig(
            timeline_db=Path(str(config_payload.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(config_payload.get("allowed_root") or allowed_root)),
            sources=tuple(source_from_json(item, tenant_id=tenant_id) for item in sources_payload),
            journal_path=Path(str(config_payload["journal_path"])),
            tenant_id=str(config_payload.get("tenant_id") or tenant_id),
            safety_margin_seconds=int(config_payload.get("safety_margin_seconds", 300)),
            lock_timeout_seconds=float(config_payload.get("lock_timeout_seconds", 30.0)),
            actor=str(config_payload.get("actor") or actor),
        )
    elif kind == "local_freshness_monitor":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        monitor_config = dict(raw_config)
    elif enabled:
        raise ValueError(f"unsupported enabled step kind: {kind}")
    return NightlyServiceStep(
        name=name,
        kind=kind,
        enabled=enabled,
        required=required,
        config=config,
        monitor_config=monitor_config,
        reason=reason,
    )


def source_from_json(payload: Any, *, tenant_id: str) -> IncrementalSourceConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("source config item must be an object")
    return IncrementalSourceConfig(
        name=str(payload.get("name") or payload["source_system"]),
        source_system=str(payload["source_system"]),
        path=Path(str(payload["path"])),
        tenant_id=str(payload.get("tenant_id") or tenant_id),
        source_ref=str(payload["source_ref"]) if payload.get("source_ref") else None,
        normalizer=str(payload.get("normalizer") or "jsonl"),
        required=bool(payload.get("required", True)),
    )


def failed_step_report(
    *,
    index: int,
    step: NightlyServiceStep,
    reason: str,
    duration_seconds: float,
    error: Exception | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "index": index,
        "name": step.name,
        "kind": step.kind,
        "status": "failed" if step.required else "skipped_optional_failed",
        "required": step.required,
        "reason": reason,
        "duration_seconds": duration_seconds,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
    return payload


def validated_service_paths(config: NightlyServiceConfig) -> tuple[Path, Path, Path, Path]:
    allowed_root = Path(config.allowed_root).expanduser().resolve(strict=False)
    timeline_db = guard_customer_timeline_output_path(config.timeline_db, allowed_root)
    if is_customer_timeline_prod_path(timeline_db):
        raise ValueError(f"nightly service must not target prod DB paths: {timeline_db}")
    out_root = guard_customer_timeline_output_path(config.out_root, allowed_root)
    publish_dir = guard_customer_timeline_output_path(config.publish_dir, allowed_root)
    for step in config.steps:
        if step.config is None:
            if step.monitor_config is not None:
                for raw_path in step.monitor_config.get("paths") or ():
                    guard_customer_timeline_output_path(Path(str(raw_path)), allowed_root)
                if step.monitor_config.get("metrics_path"):
                    guard_customer_timeline_output_path(Path(str(step.monitor_config["metrics_path"])), allowed_root)
            continue
        guard_customer_timeline_output_path(step.config.timeline_db, allowed_root)
        guard_customer_timeline_output_path(step.config.allowed_root, allowed_root)
        guard_customer_timeline_output_path(step.config.journal_path, allowed_root)
        for source in step.config.sources:
            guard_customer_timeline_output_path(source.path, allowed_root)
    return timeline_db, allowed_root, out_root, publish_dir


def run_local_freshness_monitor(
    step: NightlyServiceStep,
    *,
    timeline_db: Path,
    allowed_root: Path,
    tenant_id: str,
    actor: str,
) -> Mapping[str, Any]:
    from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
    from mango_mvp.customer_timeline.nightly_incremental import parse_datetime

    config = dict(step.monitor_config or {})
    paths = [guard_customer_timeline_output_path(Path(str(item)), allowed_root) for item in config.get("paths") or ()]
    metrics_path = (
        guard_customer_timeline_output_path(Path(str(config["metrics_path"])), allowed_root)
        if config.get("metrics_path")
        else None
    )
    existing_paths = [path for path in paths if path.exists()]
    metrics: Mapping[str, Any] = {}
    if metrics_path and metrics_path.exists():
        try:
            parsed = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            metrics = {"error": "invalid_json"}
    status = "ok" if existing_paths or metrics else str(config.get("empty_status") or "skipped")
    cursor_source_system = str(config.get("cursor_source_system") or "").strip()
    cursor_ts_raw = str(config.get("cursor_ts") or "").strip()
    cursor_written = None
    if cursor_source_system and cursor_ts_raw:
        cursor_ts = parse_datetime(cursor_ts_raw, "cursor_ts")
        with CustomerTimelineSQLiteStore(timeline_db, allowed_root=allowed_root) as store:
            cursor = store.upsert_ingestion_cursor(
                tenant_id,
                cursor_source_system,
                last_cursor_ts=cursor_ts,
                metadata={
                    "last_status": status,
                    "monitor_step": step.name,
                    "reason": config.get("reason"),
                    "metrics_path": str(metrics_path) if metrics_path else None,
                },
                actor=actor,
            )
            cursor_written = cursor.to_json_dict()
    return {
        "schema_version": "customer_timeline_local_freshness_monitor_v1",
        "name": step.name,
        "kind": step.kind,
        "required": step.required,
        "status": status,
        "reason": config.get("reason"),
        "paths": [
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
            }
            for path in paths
        ],
        "metrics_path": str(metrics_path) if metrics_path else None,
        "metrics": metrics,
        "cursor": cursor_written,
        "safety": {
            "network_calls": False,
            "writes_timeline_events": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "runs_asr": False,
        },
    }


def build_snapshot_manifest(db_path: Path, *, tenant_id: str) -> dict[str, Any]:
    import sqlite3

    db = Path(db_path)
    files = {}
    for suffix, label in (("", "sqlite"), ("-wal", "wal"), ("-shm", "shm")):
        path = Path(str(db) + suffix)
        files[label] = file_fingerprint(path)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        counts = {
            "customer_identities": table_count(con, "customer_identities"),
            "timeline_events": table_count(con, "timeline_events"),
            "bot_context_chunks": table_count(con, "bot_context_chunks"),
            "identity_links": table_count(con, "identity_links"),
            "timeline_conflicts": table_count(con, "timeline_conflicts"),
            "derived_signals": table_count(con, "derived_signals"),
            "ingestion_cursors": table_count(con, "ingestion_cursors"),
        }
        source_counts = [
            dict(row)
            for row in con.execute(
                """
                SELECT source_system, COUNT(*) AS count, MAX(event_at) AS max_event_at
                FROM timeline_events
                WHERE tenant_id = ?
                GROUP BY source_system
                ORDER BY source_system
                """,
                (tenant_id,),
            )
        ]
        cursors = [
            dict(row)
            for row in con.execute(
                """
                SELECT source_system, last_cursor_ts, updated_at
                FROM ingestion_cursors
                WHERE tenant_id = ?
                ORDER BY source_system
                """,
                (tenant_id,),
            )
        ]
    return {
        "schema_version": "customer_timeline_snapshot_manifest_v1",
        "timeline_db": str(db),
        "quick_check": quick_check,
        "files": files,
        "counts": counts,
        "source_counts": source_counts,
        "ingestion_cursors": cursors,
    }


def table_count(con: Any, table: str) -> int:
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return 0


def file_fingerprint(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": 0, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


@contextmanager
def service_lock(db_path: Path, *, timeout_seconds: float) -> Iterator[Mapping[str, Any]]:
    lock_path = Path(str(db_path) + ".nightly_service.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                waited = time.monotonic() - started
                if waited >= timeout_seconds:
                    raise TimeoutError(f"nightly service lock timeout: {lock_path}")
                time.sleep(0.2)
        yield {"path": str(lock_path), "waited_seconds": round(time.monotonic() - started, 3)}
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
