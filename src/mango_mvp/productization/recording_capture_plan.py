from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


RECORDING_CAPTURE_PLAN_SCHEMA_VERSION = "recording_capture_plan_v1"
PLAN_DOWNLOAD_DRY_RUN = "PLAN_DOWNLOAD_DRY_RUN"
SKIP_DUPLICATE_RECORDING = "SKIP_DUPLICATE_RECORDING"
SKIP_EXISTING_FILE = "SKIP_EXISTING_FILE"
BLOCK_MISSING_RECORDING_REF = "BLOCK_MISSING_RECORDING_REF"


@dataclass(frozen=True)
class RecordingCapturePlanSummary:
    schema_version: str
    product_db_path: str
    manifest_path: str
    recordings_dir: str
    inbox_items_seen: int
    manifest_items: int
    plan_download_dry_run: int
    skip_duplicate_recording: int
    skip_existing_file: int
    blocked_missing_recording_ref: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_recording_capture_plan(
    product_db_path: Path,
    product_root: Path,
    recordings_dir: Path,
    manifest_path: Path,
    out_path: Optional[Path] = None,
    limit: Optional[int] = None,
    manager_ref: Optional[str] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root, recordings_dir, manifest_path, out_path = resolve_plan_paths(
        product_db_path=product_db_path,
        product_root=product_root,
        recordings_dir=recordings_dir,
        manifest_path=manifest_path,
        out_path=out_path,
    )
    rows = read_ready_inbox_items(product_db_path, limit=limit, manager_ref=manager_ref)
    items = build_plan_items(rows=rows, recordings_dir=recordings_dir)
    write_manifest(manifest_path, items)
    action_counts = Counter(clean(item["action"]) for item in items)
    blocked = int(action_counts[BLOCK_MISSING_RECORDING_REF])
    warnings = int(action_counts[SKIP_DUPLICATE_RECORDING] + action_counts[SKIP_EXISTING_FILE])
    summary = RecordingCapturePlanSummary(
        schema_version=RECORDING_CAPTURE_PLAN_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        manifest_path=str(manifest_path),
        recordings_dir=str(recordings_dir),
        inbox_items_seen=len(rows),
        manifest_items=len(items),
        plan_download_dry_run=int(action_counts[PLAN_DOWNLOAD_DRY_RUN]),
        skip_duplicate_recording=int(action_counts[SKIP_DUPLICATE_RECORDING]),
        skip_existing_file=int(action_counts[SKIP_EXISTING_FILE]),
        blocked_missing_recording_ref=int(action_counts[BLOCK_MISSING_RECORDING_REF]),
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=warnings,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": dict(sorted(action_counts.items())),
        "manager_ref_counts": dict(sorted(Counter(clean(item.get("manager_ref")) for item in items).items())),
        "samples": {"items": items[:30]},
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def audit_recording_capture_plan(
    manifest_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    manifest_path = manifest_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_plan_path(manifest_path, product_root, "recording capture manifest")
    if out_path:
        guard_plan_path(out_path, product_root, "recording capture audit output")
    rows = read_manifest(manifest_path)
    action_counts = Counter(clean(row.get("action")) for row in rows)
    target_paths = [clean(row.get("target_audio_path")) for row in rows if clean(row.get("target_audio_path"))]
    duplicate_targets = duplicate_values(target_paths)
    target_paths_outside_root = [
        path for path in target_paths if not path_is_relative_to(Path(path).resolve(strict=False), product_root)
    ]
    existing_files = [path for path in target_paths if Path(path).exists()]
    blocked = len(duplicate_targets) + len(target_paths_outside_root) + int(action_counts[BLOCK_MISSING_RECORDING_REF])
    report = {
        "summary": {
            "schema_version": RECORDING_CAPTURE_PLAN_SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "items": len(rows),
            "plan_download_dry_run": int(action_counts[PLAN_DOWNLOAD_DRY_RUN]),
            "skip_duplicate_recording": int(action_counts[SKIP_DUPLICATE_RECORDING]),
            "skip_existing_file": int(action_counts[SKIP_EXISTING_FILE]),
            "blocked_missing_recording_ref": int(action_counts[BLOCK_MISSING_RECORDING_REF]),
            "existing_audio_files": len(existing_files),
            "duplicate_target_paths": len(duplicate_targets),
            "target_paths_outside_root": len(target_paths_outside_root),
            "validation_ok": blocked == 0,
            "blocked": blocked,
            "warnings": len(existing_files) + int(action_counts[SKIP_DUPLICATE_RECORDING]),
        },
        "action_counts": dict(sorted(action_counts.items())),
        "duplicate_target_paths": duplicate_targets[:50],
        "target_paths_outside_root": target_paths_outside_root[:50],
        "existing_audio_files": existing_files[:50],
        "samples": {"items": rows[:30]},
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def read_ready_inbox_items(
    product_db_path: Path,
    limit: Optional[int],
    manager_ref: Optional[str],
) -> Sequence[Mapping[str, Any]]:
    params: list[Any] = []
    where = ["status = 'ready_for_capture'"]
    if manager_ref:
        where.append("manager_ref = ?")
        params.append(clean(manager_ref))
    limit_sql = ""
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        limit_sql = "LIMIT ?"
        params.append(int(limit))
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        ensure_capture_inbox_schema(con)
        rows = con.execute(
            f"""
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   source_job_run_id, source_report_ref, raw_payload_ref,
                   started_at, ended_at, direction, client_phone, manager_ref,
                   recording_ref, recording_url, audio_ref, decision_reason,
                   first_seen_at, last_seen_at, enqueue_count
              FROM capture_inbox_items
             WHERE {' AND '.join(where)}
             ORDER BY started_at, id
             {limit_sql}
            """,
            params,
        ).fetchall()
    return tuple(dict(row) for row in rows)


def ensure_capture_inbox_schema(con: sqlite3.Connection) -> None:
    row = con.execute(
        """
        SELECT 1
          FROM sqlite_master
         WHERE type = 'table'
           AND name = 'capture_inbox_items'
        """
    ).fetchone()
    if not row:
        raise RuntimeError("product DB does not contain capture_inbox_items; run the capture inbox stage first")


def build_plan_items(rows: Sequence[Mapping[str, Any]], recordings_dir: Path) -> list[Mapping[str, Any]]:
    seen_recordings: dict[str, Mapping[str, Any]] = {}
    items = []
    for row in rows:
        recording_id = clean(row.get("recording_ref")) or clean(row.get("audio_ref"))
        target_audio_path = recordings_dir / build_recording_filename(row, recording_id)
        action = PLAN_DOWNLOAD_DRY_RUN
        reason = "ready_for_recording_download_dry_run"
        canonical_event_key = None
        canonical_target_audio_path = None
        if not recording_id:
            action = BLOCK_MISSING_RECORDING_REF
            reason = "recording_reference_missing"
        elif recording_id in seen_recordings:
            action = SKIP_DUPLICATE_RECORDING
            reason = "recording_id_already_planned"
            canonical = seen_recordings[recording_id]
            canonical_event_key = clean(canonical.get("event_key"))
            canonical_target_audio_path = clean(canonical.get("target_audio_path"))
        elif target_audio_path.exists() and target_audio_path.stat().st_size > 0:
            action = SKIP_EXISTING_FILE
            reason = "target_audio_file_already_exists"
        item = {
            "schema_version": RECORDING_CAPTURE_PLAN_SCHEMA_VERSION,
            "action": action,
            "reason": reason,
            "tenant_id": clean(row.get("tenant_id")),
            "provider": clean(row.get("provider")),
            "capture_inbox_item_id": int(row["id"]),
            "event_key": clean(row.get("event_key")),
            "provider_call_id": clean(row.get("provider_call_id")),
            "source_job_run_id": optional_int(row.get("source_job_run_id")),
            "source_report_ref": clean(row.get("source_report_ref")) or None,
            "raw_payload_ref": clean(row.get("raw_payload_ref")) or None,
            "started_at": clean(row.get("started_at")) or None,
            "ended_at": clean(row.get("ended_at")) or None,
            "direction": clean(row.get("direction")) or None,
            "client_phone": clean(row.get("client_phone")) or None,
            "manager_ref": clean(row.get("manager_ref")) or None,
            "recording_ref": clean(row.get("recording_ref")) or None,
            "recording_url": clean(row.get("recording_url")) or None,
            "audio_ref": clean(row.get("audio_ref")) or None,
            "recording_id": recording_id or None,
            "target_audio_path": str(target_audio_path),
            "canonical_event_key": canonical_event_key,
            "canonical_target_audio_path": canonical_target_audio_path,
            "download_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_runtime_db": False,
            "write_crm": False,
        }
        items.append(item)
        if recording_id and action in {PLAN_DOWNLOAD_DRY_RUN, SKIP_EXISTING_FILE}:
            seen_recordings[recording_id] = item
    return items


def build_recording_filename(row: Mapping[str, Any], recording_id: str) -> str:
    started = parse_datetime(clean(row.get("started_at")))
    stamp = started.strftime("%Y%m%dT%H%M%SZ") if started else "unknown_time"
    manager = safe_slug(clean(row.get("manager_ref")) or "no_manager")
    call_hash = short_hash(clean(row.get("event_key")) or clean(row.get("provider_call_id")))
    recording_hash = short_hash(recording_id or clean(row.get("audio_ref")) or call_hash)
    return f"{stamp}__mgr_{manager}__call_{call_hash}__rec_{recording_hash}.mp3"


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._-")[:40] or "value"


def short_hash(value: str) -> str:
    return hashlib.sha256(clean(value).encode("utf-8")).hexdigest()[:12]


def resolve_plan_paths(
    product_db_path: Path,
    product_root: Path,
    recordings_dir: Path,
    manifest_path: Path,
    out_path: Optional[Path],
) -> tuple[Path, Path, Path, Path, Optional[Path]]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    recordings_dir = recordings_dir.resolve(strict=False)
    manifest_path = manifest_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    guard_plan_path(recordings_dir, product_root, "recordings directory")
    guard_plan_path(manifest_path, product_root, "recording capture manifest")
    if out_path:
        guard_plan_path(out_path, product_root, "recording capture audit output")
    return product_db_path, product_root, recordings_dir, manifest_path, out_path


def guard_plan_path(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def write_manifest(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def read_manifest(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"recording capture manifest not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def duplicate_values(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safety_contract() -> Mapping[str, bool]:
    return {
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "download_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
