from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from mango_mvp.productization.capture_staging import file_sha256, optional_float, optional_int
from mango_mvp.productization.recording_capture_plan import (
    PLAN_DOWNLOAD_DRY_RUN,
    read_manifest,
)
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


RECORDING_CAPTURE_DOWNLOAD_SCHEMA_VERSION = "recording_capture_download_v1"
PLAN_RECORDING_DOWNLOAD = "PLAN_RECORDING_DOWNLOAD"
DOWNLOADED_RECORDING = "DOWNLOADED_RECORDING"
SKIP_ALREADY_DOWNLOADED = "SKIP_ALREADY_DOWNLOADED"
SKIP_NON_DOWNLOAD_PLAN = "SKIP_NON_DOWNLOAD_PLAN"
BLOCK_MISSING_RECORDING_ID = "BLOCK_MISSING_RECORDING_ID"
FAILED_DOWNLOAD = "FAILED_DOWNLOAD"
AVAILABLE_RECORDING_ACTIONS = {DOWNLOADED_RECORDING, SKIP_ALREADY_DOWNLOADED}


class RecordingDownloader(Protocol):
    def download(self, recording_id: str, target_path: Path) -> int:
        ...


@dataclass(frozen=True)
class RecordingCaptureDownloadSummary:
    schema_version: str
    source_plan_manifest_path: str
    download_manifest_path: str
    recordings_dir: str
    execute: bool
    plan_items_seen: int
    eligible_items: int
    selected_items: int
    plan_recording_download: int
    downloaded_recording: int
    skip_already_downloaded: int
    skip_non_download_plan: int
    blocked_missing_recording_id: int
    failed_download: int
    downloaded_bytes_total: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def run_recording_capture_download(
    source_plan_manifest_path: Path,
    product_root: Path,
    recordings_dir: Path,
    download_manifest_path: Path,
    out_path: Optional[Path] = None,
    downloader: Optional[RecordingDownloader] = None,
    execute: bool = False,
    limit: Optional[int] = None,
    manager_ref: Optional[str] = None,
    sleep_sec: float = 0.0,
) -> Mapping[str, Any]:
    product_root, source_plan_manifest_path, recordings_dir, download_manifest_path, out_path = resolve_download_paths(
        product_root=product_root,
        source_plan_manifest_path=source_plan_manifest_path,
        recordings_dir=recordings_dir,
        download_manifest_path=download_manifest_path,
        out_path=out_path,
    )
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if sleep_sec < 0:
        raise ValueError("sleep_sec must not be negative")
    if execute and downloader is None:
        raise ValueError("downloader is required when execute=true")

    source_rows = read_manifest(source_plan_manifest_path)
    existing_entries = read_download_manifest(download_manifest_path)
    downloaded_by_event = latest_downloaded_by(existing_entries, "event_key")
    downloaded_by_recording = latest_downloaded_by(existing_entries, "recording_id")
    selected_source_rows = select_source_rows(source_rows, limit=limit, manager_ref=manager_ref)

    items = []
    for row in selected_source_rows:
        item = build_download_item(
            row=row,
            product_root=product_root,
            recordings_dir=recordings_dir,
            downloaded_by_event=downloaded_by_event,
            downloaded_by_recording=downloaded_by_recording,
            execute=execute,
        )
        if item["action"] == PLAN_RECORDING_DOWNLOAD and execute:
            item = execute_download_item(item=item, downloader=downloader, sleep_sec=sleep_sec)
            if item["action"] == DOWNLOADED_RECORDING:
                downloaded_by_event[clean(item.get("event_key"))] = item
                downloaded_by_recording[clean(item.get("recording_id"))] = item
        items.append(item)

    append_download_manifest(download_manifest_path, items)
    report = build_download_report(
        source_plan_manifest_path=source_plan_manifest_path,
        download_manifest_path=download_manifest_path,
        recordings_dir=recordings_dir,
        execute=execute,
        source_rows=source_rows,
        selected_rows=selected_source_rows,
        items=items,
    )
    if out_path:
        write_json(out_path, report)
    return report


def audit_recording_capture_download(
    download_manifest_path: Path,
    product_root: Path,
    recordings_dir: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    download_manifest_path = download_manifest_path.resolve(strict=False)
    recordings_dir = recordings_dir.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_download_path(download_manifest_path, product_root, "download manifest")
    guard_download_path(recordings_dir, product_root, "recordings dir")
    if out_path:
        guard_download_path(out_path, product_root, "download audit output")

    rows = read_download_manifest(download_manifest_path)
    latest_by_event: dict[str, Mapping[str, Any]] = {}
    latest_available_by_event: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        event_key = clean(row.get("event_key"))
        if event_key:
            latest_by_event[event_key] = row
            if clean(row.get("action")) in AVAILABLE_RECORDING_ACTIONS:
                latest_available_by_event[event_key] = row

    status_counts = Counter(clean(row.get("action")) for row in rows)
    latest_status_counts = Counter(clean(row.get("action")) for row in latest_by_event.values())
    available_rows = list(latest_available_by_event.values())
    downloaded_rows = [row for row in available_rows if clean(row.get("action")) == DOWNLOADED_RECORDING]

    missing_files = []
    zero_size_files = []
    checksum_mismatches = []
    local_paths_outside_root = []
    for row in available_rows:
        local_path = clean(row.get("local_audio_path"))
        if not local_path:
            missing_files.append({"event_key": row.get("event_key"), "path": None})
            continue
        path = Path(local_path).resolve(strict=False)
        if not path_is_relative_to(path, product_root):
            local_paths_outside_root.append(local_path)
            continue
        if not path.exists():
            missing_files.append({"event_key": row.get("event_key"), "path": local_path})
            continue
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            zero_size_files.append({"event_key": row.get("event_key"), "path": local_path})
            continue
        expected_checksum = clean(row.get("checksum_sha256"))
        if expected_checksum and file_sha256(path) != expected_checksum:
            checksum_mismatches.append({"event_key": row.get("event_key"), "path": local_path})

    referenced_paths = {
        str(Path(clean(row.get("local_audio_path"))).resolve(strict=False))
        for row in latest_by_event.values()
        if clean(row.get("local_audio_path"))
    }
    audio_files = sorted(recordings_dir.glob("*.mp3")) if recordings_dir.exists() else []
    unreferenced_audio_files = [
        str(path)
        for path in audio_files
        if str(path.resolve(strict=False)) not in referenced_paths
    ]
    failed_latest = int(latest_status_counts[FAILED_DOWNLOAD])
    blocked = (
        failed_latest
        + len(missing_files)
        + len(zero_size_files)
        + len(checksum_mismatches)
        + len(local_paths_outside_root)
    )
    report = {
        "summary": {
            "schema_version": RECORDING_CAPTURE_DOWNLOAD_SCHEMA_VERSION,
            "download_manifest_path": str(download_manifest_path),
            "recordings_dir": str(recordings_dir),
            "manifest_rows": len(rows),
            "latest_unique_events": len(latest_by_event),
            "available_latest_events": len(available_rows),
            "downloaded_latest_events": len(downloaded_rows),
            "failed_latest_events": failed_latest,
            "missing_files": len(missing_files),
            "zero_size_files": len(zero_size_files),
            "checksum_mismatches": len(checksum_mismatches),
            "local_paths_outside_root": len(local_paths_outside_root),
            "recordings_dir_mp3_files": len(audio_files),
            "recordings_dir_total_bytes": sum(path.stat().st_size for path in audio_files),
            "unreferenced_audio_files": len(unreferenced_audio_files),
            "validation_ok": blocked == 0,
            "blocked": blocked,
            "warnings": len(unreferenced_audio_files) + int(latest_status_counts[SKIP_ALREADY_DOWNLOADED]),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "latest_status_counts": dict(sorted(latest_status_counts.items())),
        "samples": {
            "available": available_rows[:20],
            "downloaded": downloaded_rows[:20],
            "missing_files": missing_files[:20],
            "zero_size_files": zero_size_files[:20],
            "checksum_mismatches": checksum_mismatches[:20],
            "local_paths_outside_root": local_paths_outside_root[:20],
            "unreferenced_audio_files": unreferenced_audio_files[:20],
        },
        "safety": safety_contract(download_audio=any(bool(row.get("download_audio")) for row in rows)),
    }
    if out_path:
        write_json(out_path, report)
    return report


def select_source_rows(
    source_rows: Sequence[Mapping[str, Any]],
    limit: Optional[int],
    manager_ref: Optional[str],
) -> list[Mapping[str, Any]]:
    selected = []
    wanted_manager = clean(manager_ref)
    for row in source_rows:
        if wanted_manager and clean(row.get("manager_ref")) != wanted_manager:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def build_download_item(
    row: Mapping[str, Any],
    product_root: Path,
    recordings_dir: Path,
    downloaded_by_event: Mapping[str, Mapping[str, Any]],
    downloaded_by_recording: Mapping[str, Mapping[str, Any]],
    execute: bool,
) -> Mapping[str, Any]:
    event_key = clean(row.get("event_key"))
    recording_id = clean(row.get("recording_id")) or clean(row.get("recording_ref")) or clean(row.get("audio_ref"))
    target_path = build_download_target_path(row=row, recordings_dir=recordings_dir)
    action = PLAN_RECORDING_DOWNLOAD
    reason = "ready_for_recording_download"
    canonical_event_key = None
    canonical_audio_path = None

    if clean(row.get("action")) != PLAN_DOWNLOAD_DRY_RUN:
        action = SKIP_NON_DOWNLOAD_PLAN
        reason = "source_plan_action_not_downloadable"
    elif not recording_id:
        action = BLOCK_MISSING_RECORDING_ID
        reason = "recording_id_missing"
    elif event_key in downloaded_by_event:
        action = SKIP_ALREADY_DOWNLOADED
        reason = "event_already_downloaded"
        canonical = downloaded_by_event[event_key]
        canonical_event_key = clean(canonical.get("event_key")) or None
        canonical_audio_path = clean(canonical.get("local_audio_path")) or None
    elif recording_id in downloaded_by_recording:
        action = SKIP_ALREADY_DOWNLOADED
        reason = "recording_id_already_downloaded"
        canonical = downloaded_by_recording[recording_id]
        canonical_event_key = clean(canonical.get("event_key")) or None
        canonical_audio_path = clean(canonical.get("local_audio_path")) or None
    elif target_path.exists() and target_path.stat().st_size > 0:
        action = SKIP_ALREADY_DOWNLOADED
        reason = "target_audio_file_already_exists"
        canonical_audio_path = str(target_path)

    guard_download_path(target_path.resolve(strict=False), product_root, "target audio path")
    local_audio_path = canonical_audio_path or str(target_path)
    item = {
        "schema_version": RECORDING_CAPTURE_DOWNLOAD_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "reason": reason,
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")),
        "event_key": event_key,
        "provider_call_id": clean(row.get("provider_call_id")),
        "capture_inbox_item_id": optional_int(row.get("capture_inbox_item_id")),
        "source_plan_action": clean(row.get("action")),
        "source_plan_manifest_path": clean(row.get("source_plan_manifest_path")) or None,
        "started_at": clean(row.get("started_at")) or None,
        "ended_at": clean(row.get("ended_at")) or None,
        "direction": clean(row.get("direction")) or None,
        "client_phone": clean(row.get("client_phone")) or None,
        "manager_ref": clean(row.get("manager_ref")) or None,
        "recording_id": recording_id or None,
        "recording_ref": clean(row.get("recording_ref")) or None,
        "recording_url": clean(row.get("recording_url")) or None,
        "local_audio_path": local_audio_path,
        "canonical_event_key": canonical_event_key,
        "canonical_audio_path": canonical_audio_path,
        "execute": execute,
        "download_audio": execute and action == PLAN_RECORDING_DOWNLOAD,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "write_product_db": False,
        "write_crm": False,
    }
    if action == SKIP_ALREADY_DOWNLOADED:
        path = Path(local_audio_path).resolve(strict=False)
        if path.exists() and path.stat().st_size > 0:
            item.update(validate_downloaded_recording_file(path))
    return item


def execute_download_item(
    item: Mapping[str, Any],
    downloader: Optional[RecordingDownloader],
    sleep_sec: float,
) -> Mapping[str, Any]:
    assert downloader is not None
    target_path = Path(clean(item.get("local_audio_path")))
    recording_id = clean(item.get("recording_id"))
    try:
        downloader.download(recording_id=recording_id, target_path=target_path)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        validation = validate_downloaded_recording_file(target_path)
        return {
            **item,
            "action": DOWNLOADED_RECORDING,
            "reason": "recording_downloaded",
            "download_audio": True,
            **validation,
        }
    except Exception as exc:
        return {
            **item,
            "action": FAILED_DOWNLOAD,
            "reason": "recording_download_failed",
            "download_audio": True,
            "error": str(exc),
        }


def validate_downloaded_recording_file(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"downloaded recording is empty: {path}")
    result: dict[str, Any] = {
        "size_bytes": size_bytes,
        "checksum_sha256": file_sha256(path),
    }
    try:
        from mango_mvp.utils.audio import probe_audio

        meta = probe_audio(path)
        result.update(
            {
                "duration_sec": optional_float(meta.get("duration_sec")),
                "codec_name": clean(meta.get("codec_name")) or None,
                "channels": optional_int(meta.get("channels")),
                "sample_rate": optional_int(meta.get("sample_rate")),
                "audio_probe_error": None,
            }
        )
    except Exception as exc:
        result["audio_probe_error"] = str(exc)
    return result


def build_download_report(
    source_plan_manifest_path: Path,
    download_manifest_path: Path,
    recordings_dir: Path,
    execute: bool,
    source_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    action_counts = Counter(clean(item.get("action")) for item in items)
    blocked = int(action_counts[BLOCK_MISSING_RECORDING_ID] + action_counts[FAILED_DOWNLOAD])
    warnings = int(action_counts[SKIP_ALREADY_DOWNLOADED] + action_counts[SKIP_NON_DOWNLOAD_PLAN])
    summary = RecordingCaptureDownloadSummary(
        schema_version=RECORDING_CAPTURE_DOWNLOAD_SCHEMA_VERSION,
        source_plan_manifest_path=str(source_plan_manifest_path),
        download_manifest_path=str(download_manifest_path),
        recordings_dir=str(recordings_dir),
        execute=execute,
        plan_items_seen=len(source_rows),
        eligible_items=sum(1 for row in source_rows if clean(row.get("action")) == PLAN_DOWNLOAD_DRY_RUN),
        selected_items=len(selected_rows),
        plan_recording_download=int(action_counts[PLAN_RECORDING_DOWNLOAD]),
        downloaded_recording=int(action_counts[DOWNLOADED_RECORDING]),
        skip_already_downloaded=int(action_counts[SKIP_ALREADY_DOWNLOADED]),
        skip_non_download_plan=int(action_counts[SKIP_NON_DOWNLOAD_PLAN]),
        blocked_missing_recording_id=int(action_counts[BLOCK_MISSING_RECORDING_ID]),
        failed_download=int(action_counts[FAILED_DOWNLOAD]),
        downloaded_bytes_total=sum(int(item.get("size_bytes") or 0) for item in items),
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=warnings,
    )
    return {
        "summary": summary.to_json_dict(),
        "action_counts": dict(sorted(action_counts.items())),
        "manager_ref_counts": dict(sorted(Counter(clean(item.get("manager_ref")) for item in items).items())),
        "samples": {"items": list(items[:30])},
        "safety": safety_contract(download_audio=any(bool(item.get("download_audio")) for item in items)),
    }


def build_download_target_path(row: Mapping[str, Any], recordings_dir: Path) -> Path:
    source_target = clean(row.get("target_audio_path"))
    if source_target:
        return recordings_dir / Path(source_target).name
    event_key = clean(row.get("event_key")) or clean(row.get("provider_call_id")) or "unknown"
    return recordings_dir / f"{event_key.replace('/', '_')}.mp3"


def resolve_download_paths(
    product_root: Path,
    source_plan_manifest_path: Path,
    recordings_dir: Path,
    download_manifest_path: Path,
    out_path: Optional[Path],
) -> tuple[Path, Path, Path, Path, Optional[Path]]:
    product_root = product_root.resolve(strict=False)
    source_plan_manifest_path = source_plan_manifest_path.resolve(strict=False)
    recordings_dir = recordings_dir.resolve(strict=False)
    download_manifest_path = download_manifest_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_download_path(source_plan_manifest_path, product_root, "source plan manifest")
    guard_download_path(recordings_dir, product_root, "recordings dir")
    guard_download_path(download_manifest_path, product_root, "download manifest")
    if out_path:
        guard_download_path(out_path, product_root, "download audit output")
    return product_root, source_plan_manifest_path, recordings_dir, download_manifest_path, out_path


def guard_download_path(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def latest_downloaded_by(
    entries: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for entry in entries:
        if clean(entry.get("action")) not in AVAILABLE_RECORDING_ACTIONS:
            continue
        value = clean(entry.get(key))
        if value:
            result[value] = entry
    return result


def append_download_manifest(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def read_download_manifest(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def safety_contract(download_audio: bool) -> Mapping[str, bool]:
    return {
        "download_audio": download_audio,
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
