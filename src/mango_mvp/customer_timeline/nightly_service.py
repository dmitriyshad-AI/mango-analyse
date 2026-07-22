from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.amo_incremental import AmoIncrementalConfig, run_amo_incremental
from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    run_nightly_incremental,
    summarize_report,
)
from mango_mvp.customer_timeline.mail_link_enrich import MailLinkEnrichConfig, run_mail_link_enrich
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
    mail_link_config: Optional[MailLinkEnrichConfig] = None
    mango_sweep_config: Optional[Mapping[str, Any]] = None
    amo_incremental_config: Optional[AmoIncrementalConfig] = None
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
    timeline_db.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    publish_dir.mkdir(parents=True, exist_ok=True)
    timeline_db.parent.chmod(0o700)
    out_root.chmod(0o700)
    publish_dir.chmod(0o700)
    if timeline_db.exists():
        timeline_db.chmod(0o600)
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.chmod(0o700)
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
            if step.kind == "mango_processed_sweep":
                step_report = run_mango_processed_sweep(
                    step,
                    timeline_db=timeline_db,
                    allowed_root=allowed_root,
                    tenant_id=config.tenant_id,
                )
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("status") == "ready" else (
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
                        "summary": {
                            "status": step_report.get("status"),
                            "events_written": step_report.get("events_written"),
                            "dbs_scanned": step_report.get("dbs_scanned"),
                            "dbs_selected_after_cursor": step_report.get("dbs_selected_after_cursor"),
                            "cursor": step_report.get("cursor"),
                            "output_jsonl": step_report.get("output_jsonl"),
                            "manifest_path": step_report.get("manifest_path"),
                            "known_rerun_noise": step_report.get("known_rerun_noise"),
                            "safety": step_report.get("safety"),
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "mail_link_enrich":
                if step.mail_link_config is None:
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
                    step_report = run_mail_link_enrich(step.mail_link_config)
                except Exception as exc:  # optional enrichment must fail-soft through the service report.
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
                safety = step_report.get("safety") if isinstance(step_report.get("safety"), Mapping) else {}
                changed_visibility = bool(safety.get("allowed_for_bot_changed")) or bool(
                    safety.get("mail_stage2_allowed_for_bot_changed")
                )
                status = "failed_visibility_changed" if changed_visibility else "ok"
                if changed_visibility and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "target_events": step_report.get("target_events"),
                            "counts": step_report.get("counts"),
                            "apply_counts": (step_report.get("apply") or {}).get("counts")
                            if isinstance(step_report.get("apply"), Mapping)
                            else {},
                            "safety": safety,
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "amo_incremental":
                try:
                    if step.amo_incremental_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = run_amo_incremental(step.amo_incremental_config)
                    step_ok = amo_incremental_report_ok(step_report)
                except Exception as exc:
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
                status = "ok" if step_ok else ("failed" if step.required else "skipped_optional_failed")
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
                        "summary": {
                            "cursor_before": step_report.get("cursor_before"),
                            "cursor_after": step_report.get("cursor_after"),
                            "fetch": step_report.get("fetch"),
                            "repeat_run_duplicates": step_report.get("repeat_run_duplicates"),
                            "safety": step_report.get("safety"),
                        },
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
    mail_link_config = None
    mango_sweep_config = None
    amo_incremental_config = None
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
    elif kind == "mail_link_enrich":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        mail_link_config = MailLinkEnrichConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            out_dir=Path(str(raw_config.get("out_dir") or Path(allowed_root) / "mail_link_enrich")),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            apply=bool(raw_config.get("apply", True)),
            max_events=int(raw_config["max_events"]) if raw_config.get("max_events") is not None else None,
        )
    elif kind == "mango_processed_sweep":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        mango_sweep_config = dict(raw_config)
    elif kind == "amo_incremental":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        amo_incremental_config = AmoIncrementalConfig(
            source_db=Path(str(raw_config.get("source_db") or timeline_db)),
            out_root=Path(str(raw_config["out_root"])),
            mcp_env=Path(str(raw_config["mcp_env"])),
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            mcp_transport=str(raw_config["mcp_transport"]) if raw_config.get("mcp_transport") else None,
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            safety_overlap_seconds=int(raw_config.get("safety_overlap_seconds", 300)),
            page_limit=int(raw_config.get("page_limit", 250)),
            max_pages=int(raw_config.get("max_pages", 20)),
            sleep_sec=float(raw_config.get("sleep_sec", 1.05)),
            copy_db=False,
        )
    elif enabled:
        raise ValueError(f"unsupported enabled step kind: {kind}")
    return NightlyServiceStep(
        name=name,
        kind=kind,
        enabled=enabled,
        required=required,
        config=config,
        monitor_config=monitor_config,
        mail_link_config=mail_link_config,
        mango_sweep_config=mango_sweep_config,
        amo_incremental_config=amo_incremental_config,
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
            if step.mail_link_config is not None:
                guard_customer_timeline_output_path(step.mail_link_config.timeline_db, allowed_root)
                guard_customer_timeline_output_path(step.mail_link_config.allowed_root, allowed_root)
                guard_customer_timeline_output_path(step.mail_link_config.out_dir, allowed_root)
            if step.mango_sweep_config is not None:
                for key in ("out_jsonl", "report_out", "manifest_path", "inventory_out"):
                    if step.mango_sweep_config.get(key):
                        guard_customer_timeline_output_path(Path(str(step.mango_sweep_config[key])), allowed_root)
            if step.amo_incremental_config is not None:
                guard_customer_timeline_output_path(step.amo_incremental_config.timeline_db, allowed_root)
                guard_customer_timeline_output_path(step.amo_incremental_config.allowed_root, allowed_root)
                guard_customer_timeline_output_path(step.amo_incremental_config.out_root, allowed_root)
            continue
        guard_customer_timeline_output_path(step.config.timeline_db, allowed_root)
        guard_customer_timeline_output_path(step.config.allowed_root, allowed_root)
        guard_customer_timeline_output_path(step.config.journal_path, allowed_root)
        for source in step.config.sources:
            guard_customer_timeline_output_path(source.path, allowed_root)
    return timeline_db, allowed_root, out_root, publish_dir


def amo_incremental_report_ok(report: Mapping[str, Any]) -> bool:
    safety = report.get("safety") if isinstance(report.get("safety"), Mapping) else {}
    if any(safety.get(key) is not False for key in ("amo_write", "tallanto_write", "crm_write")):
        return False
    fetch = report.get("fetch") if isinstance(report.get("fetch"), Mapping) else {}
    if any(isinstance(item, Mapping) and item.get("page_cap_hit") for item in fetch.values()):
        return False
    reports = [report.get("second_run")]
    first = report.get("first_run") if isinstance(report.get("first_run"), Mapping) else {}
    reports.extend((first.get("cards"), first.get("events")))
    return not any(
        isinstance(item, Mapping) and item.get("source_errors")
        for item in reports
    )


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
    deprecated_removed: list[str] = []
    if cursor_source_system and cursor_ts_raw:
        cursor_ts = parse_datetime(cursor_ts_raw, "cursor_ts")
        with CustomerTimelineSQLiteStore(timeline_db, allowed_root=allowed_root) as store:
            for raw_source in config.get("deprecated_cursor_source_systems") or ():
                legacy_source = str(raw_source or "").strip()
                if not legacy_source or legacy_source == cursor_source_system:
                    continue
                removed = store._con.execute(
                    "DELETE FROM ingestion_cursors WHERE tenant_id = ? AND source_system = ?",
                    (tenant_id, legacy_source),
                )
                if removed.rowcount:
                    deprecated_removed.append(legacy_source)
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
            store._con.commit()
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
        "deprecated_cursors_removed": deprecated_removed,
        "safety": {
            "network_calls": False,
            "writes_timeline_events": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "runs_asr": False,
        },
    }


def run_mango_processed_sweep(
    step: NightlyServiceStep,
    *,
    timeline_db: Path,
    allowed_root: Path,
    tenant_id: str,
) -> Mapping[str, Any]:
    config = dict(step.mango_sweep_config or {})
    out_jsonl = guard_customer_timeline_output_path(
        Path(str(config.get("out_jsonl") or Path(allowed_root) / "nightly_dv2_sources" / "mango_processed_sweep.jsonl")),
        allowed_root,
    )
    report_out = guard_customer_timeline_output_path(
        Path(str(config.get("report_out") or out_jsonl.with_suffix(".producer_report.json"))),
        allowed_root,
    )
    manifest_path = guard_customer_timeline_output_path(
        Path(str(config.get("manifest_path") or out_jsonl.with_suffix(".manifest.json"))),
        allowed_root,
    )
    inventory_out = guard_customer_timeline_output_path(
        Path(str(config.get("inventory_out") or out_jsonl.with_suffix(".inventory.json"))),
        allowed_root,
    )
    producer_script = Path(str(config.get("producer_script") or Path.cwd() / "scripts" / "build_mango_call_timeline_increment.py"))
    if not producer_script.exists():
        return mango_sweep_manifest(
            status="failed",
            reason="producer_script_missing",
            timeline_db=timeline_db,
            out_jsonl=out_jsonl,
            report_out=report_out,
            manifest_path=manifest_path,
            inventory_out=inventory_out,
            cursor={},
            inventory=[],
            producer_report={},
            command=(),
            rc=127,
        )
    cursor = mango_processed_cursor(timeline_db, tenant_id=tenant_id)
    since = str(config.get("since") or cursor.get("max_source_ts") or cursor.get("last_cursor_ts") or "").strip()
    scan_roots = tuple(Path(str(item)).expanduser() for item in config.get("scan_roots") or ())
    package_globs = tuple(str(item) for item in config.get("package_globs") or ("mango_update_after_*",))
    inventory = discover_mango_processed_call_dbs(scan_roots, package_globs=package_globs, since=since)
    seen_package_dbs = {
        Path(str(item["db_path"])).expanduser().resolve(strict=False)
        for item in inventory
        if item.get("db_path")
    }
    since_dt = parse_iso_datetime(since) if since else None
    for raw_db in config.get("package_dbs") or ():
        db_path = Path(str(raw_db)).expanduser().resolve(strict=False)
        if db_path in seen_package_dbs:
            continue
        seen_package_dbs.add(db_path)
        inventory.append(inspect_mango_call_db(db_path.parent, db_path, since_dt=since_dt))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    inventory_out.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    explicit_package_dbs = {
        Path(str(item)).expanduser().resolve(strict=False)
        for item in config.get("package_dbs") or ()
    }
    unusable_explicit = [
        item
        for item in inventory
        if Path(str(item.get("db_path") or "")).expanduser().resolve(strict=False) in explicit_package_dbs
        and not item.get("usable")
    ]
    if unusable_explicit:
        manifest = mango_sweep_manifest(
            status="failed",
            reason="explicit_package_db_unusable",
            timeline_db=timeline_db,
            out_jsonl=out_jsonl,
            report_out=report_out,
            manifest_path=manifest_path,
            inventory_out=inventory_out,
            cursor=cursor,
            inventory=inventory,
            producer_report={"unusable_explicit_package_dbs": unusable_explicit},
            command=(),
            rc=78,
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    package_dbs = [item["db_path"] for item in inventory if item.get("usable") and int(item.get("selected_after_cursor") or 0) > 0]
    command = [
        sys.executable,
        str(producer_script),
        "--timeline-db",
        str(timeline_db),
        "--out-jsonl",
        str(out_jsonl),
        "--report-out",
        str(report_out),
        "--tenant-id",
        tenant_id,
    ]
    if since:
        command.extend(["--since", since])
    for db_path in package_dbs:
        command.extend(["--package-db", str(db_path)])
    proc = subprocess.run(command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    producer_report: Mapping[str, Any] = {}
    if report_out.exists():
        try:
            parsed = json.loads(report_out.read_text(encoding="utf-8"))
            producer_report = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            producer_report = {"error": "invalid_producer_report_json"}
    status = "ready" if proc.returncode == 0 and out_jsonl.exists() else "failed"
    manifest = mango_sweep_manifest(
        status=status,
        reason="" if status == "ready" else "producer_failed",
        timeline_db=timeline_db,
        out_jsonl=out_jsonl,
        report_out=report_out,
        manifest_path=manifest_path,
        inventory_out=inventory_out,
        cursor=cursor,
        inventory=inventory,
        producer_report=producer_report,
        command=tuple(command),
        rc=proc.returncode,
        stdout_tail="\n".join((proc.stdout or "").splitlines()[-40:]),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def mango_sweep_manifest(
    *,
    status: str,
    reason: str,
    timeline_db: Path,
    out_jsonl: Path,
    report_out: Path,
    manifest_path: Path,
    inventory_out: Path,
    cursor: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    producer_report: Mapping[str, Any],
    command: Sequence[str],
    rc: int,
    stdout_tail: str = "",
) -> Mapping[str, Any]:
    return {
        "schema_version": "mango_processed_sweep_v1",
        "status": status,
        "reason": reason,
        "timeline_db": str(timeline_db),
        "cursor": dict(cursor),
        "dbs_scanned": len(inventory),
        "dbs_selected_after_cursor": sum(1 for item in inventory if int(item.get("selected_after_cursor") or 0) > 0),
        "rows_selected_after_cursor": sum(int(item.get("selected_after_cursor") or 0) for item in inventory),
        "events_written": int(producer_report.get("events_written") or 0),
        "source_counts": dict(producer_report.get("source_counts") or {}),
        "identity_resolution_counts": dict(producer_report.get("identity_resolution_counts") or {}),
        "call_type_counts": dict(producer_report.get("call_type_counts") or {}),
        "output_jsonl": str(out_jsonl),
        "producer_report": str(report_out),
        "manifest_path": str(manifest_path),
        "inventory_out": str(inventory_out),
        "producer_rc": rc,
        "producer_command": list(command),
        "producer_stdout_tail": stdout_tail,
        "known_rerun_noise": {
            "amocrm_snapshot_safety_window_updated": True,
            "note": "calls rerun acceptance must use mango_processed_summary event/chunk deltas; AMO contact snapshots may report updated rows inside the 5-minute overlap.",
        },
        "safety": {
            "network_calls": False,
            "writes_timeline_db": False,
            "writes_prod_db": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "sends_messages": False,
            "runs_asr": False,
            "runs_analyze": False,
        },
    }


def mango_processed_cursor(db_path: Path, *, tenant_id: str) -> Mapping[str, Any]:
    row = None
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        row = con.execute(
            """
            SELECT last_cursor_ts, updated_at, metadata_json
            FROM ingestion_cursors
            WHERE tenant_id = ? AND source_system = 'mango_processed_summary'
            """,
            (tenant_id,),
        ).fetchone()
    if row is None:
        return {"source_system": "mango_processed_summary", "last_cursor_ts": None, "max_source_ts": None}
    metadata = {}
    try:
        parsed = json.loads(str(row["metadata_json"] or "{}"))
        metadata = parsed.get("metadata") if isinstance(parsed, Mapping) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
    except json.JSONDecodeError:
        metadata = {}
    return {
        "source_system": "mango_processed_summary",
        "last_cursor_ts": row["last_cursor_ts"],
        "updated_at": row["updated_at"],
        "max_source_ts": metadata.get("max_source_ts"),
    }


def discover_mango_processed_call_dbs(
    scan_roots: Sequence[Path],
    *,
    package_globs: Sequence[str],
    since: str,
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[Path] = set()
    since_dt = parse_iso_datetime(since) if since else None
    for scan_root in scan_roots:
        for pattern in package_globs:
            for root in sorted(scan_root.glob(pattern)):
                if not root.is_dir():
                    continue
                for db_path in sorted((root / "asr_ui_batch").glob("*.sqlite")):
                    resolved = db_path.expanduser().resolve(strict=False)
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    result.append(inspect_mango_call_db(root, resolved, since_dt=since_dt))
    return result


def inspect_mango_call_db(root: Path, db_path: Path, *, since_dt: datetime | None) -> Mapping[str, Any]:
    if ".before_" in db_path.name or "before_" in db_path.name:
        return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "backup_db_name"}
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only=ON")
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='call_records'").fetchone() is None:
                return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "missing_call_records"}
            cols = {str(row[1]) for row in con.execute("PRAGMA table_info(call_records)")}
            if "analysis_status" not in cols or "analysis_json" not in cols:
                return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "missing_analysis_columns"}
            date_col = next((col for col in ("started_at", "call_at", "event_at") if col in cols), None)
            if not date_col:
                return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "missing_call_datetime"}
            rows = con.execute(
                f"""
                SELECT {date_col} AS call_at, analysis_status, analysis_json
                FROM call_records
                WHERE {date_col} IS NOT NULL AND TRIM({date_col}) != ''
                """
            ).fetchall()
    except sqlite3.Error as exc:
        return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": f"sqlite_error:{type(exc).__name__}"}
    total = len(rows)
    done_rows = [
        row
        for row in rows
        if str(row["analysis_status"] or "") == "done" and str(row["analysis_json"] or "").strip()
    ]
    selected = 0
    min_at = None
    max_at = None
    for row in done_rows:
        raw = str(row["call_at"] or "")
        parsed = parse_iso_datetime(raw)
        if parsed is None:
            continue
        min_at = parsed if min_at is None else min(min_at, parsed)
        max_at = parsed if max_at is None else max(max_at, parsed)
        if since_dt is None or parsed >= since_dt:
            selected += 1
    return {
        "root": str(root),
        "db_path": str(db_path),
        "usable": True,
        "rows_total": total,
        "analysis_done": len(done_rows),
        "min_started_at": min_at.isoformat() if min_at else None,
        "max_started_at": max_at.isoformat() if max_at else None,
        "selected_after_cursor": selected,
        "has_ra_final_summary": any(root.rglob("RA_FINAL_SUMMARY.json")),
    }


def parse_iso_datetime(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_snapshot_manifest(db_path: Path, *, tenant_id: str) -> dict[str, Any]:
    import sqlite3

    db = Path(db_path)
    now = datetime.now(timezone.utc)
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
        for row in source_counts:
            row["max_event_age_days"] = iso_age_days(row.get("max_event_at"), now=now)
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
        for row in cursors:
            row["last_cursor_age_days"] = iso_age_days(row.get("last_cursor_ts"), now=now)
            row["updated_age_days"] = iso_age_days(row.get("updated_at"), now=now)
        mail_link_enrich = build_mail_link_enrich_manifest_metrics(con, tenant_id=tenant_id)
    return {
        "schema_version": "customer_timeline_snapshot_manifest_v1",
        "timeline_db": str(db),
        "quick_check": quick_check,
        "files": files,
        "counts": counts,
        "source_counts": source_counts,
        "ingestion_cursors": cursors,
        "mail_link_enrich": mail_link_enrich,
    }


def build_mail_link_enrich_manifest_metrics(con: Any, *, tenant_id: str) -> Mapping[str, Any]:
    try:
        outcome_rows = con.execute(
            """
            SELECT COALESCE(json_extract(record_json, '$.metadata.mail_link_enrich.outcome'), 'not_processed') AS outcome,
                   COUNT(*) AS count
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = 'mail_archive_stage2'
            GROUP BY outcome
            ORDER BY outcome
            """,
            (tenant_id,),
        ).fetchall()
        brand_rows = con.execute(
            """
            SELECT COALESCE(json_extract(record_json, '$.metadata.brand'), 'unknown') AS brand,
                   COUNT(*) AS count
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = 'mail_archive_stage2'
            GROUP BY brand
            ORDER BY brand
            """,
            (tenant_id,),
        ).fetchall()
        pending_null = con.execute(
            """
            SELECT COUNT(*)
            FROM timeline_events
            WHERE tenant_id = ?
              AND source_system = 'mail_archive_stage2'
              AND match_status = 'unmatched'
              AND (customer_id IS NULL OR customer_id = '')
              AND json_extract(record_json, '$.metadata.pending_attribution') = 1
              AND json_extract(record_json, '$.metadata.pending_reason') IS NULL
            """,
            (tenant_id,),
        ).fetchone()[0]
    except Exception:
        return {"status": "unavailable"}
    outcomes = {str(row["outcome"]): int(row["count"]) for row in outcome_rows}
    brands = {str(row["brand"]): int(row["count"]) for row in brand_rows}
    return {
        "status": "ok",
        "outcomes": outcomes,
        "linked_strong": outcomes.get("strong", 0),
        "weak_email": outcomes.get("weak_email", 0),
        "unmatched": outcomes.get("unmatched", 0),
        "blocked": outcomes.get("blocked", 0),
        "brand_counts": brands,
        "unknown_brand": brands.get("unknown", 0),
        "pending_without_reason": int(pending_null),
    }


def iso_age_days(raw: Any, *, now: datetime) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return round((now - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0, 3)


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
