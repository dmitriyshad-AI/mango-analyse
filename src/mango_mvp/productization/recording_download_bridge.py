from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.capture_staging import CAPTURE_MANIFEST_SCHEMA_VERSION, file_sha256
from mango_mvp.productization.pipeline_bridge import build_pipeline_bridge_plan, write_bridge_plan_csv
from mango_mvp.productization.recording_capture_download import (
    AVAILABLE_RECORDING_ACTIONS,
    DOWNLOADED_RECORDING,
    RECORDING_CAPTURE_DOWNLOAD_SCHEMA_VERSION,
    SKIP_ALREADY_DOWNLOADED,
    guard_download_path,
    read_download_manifest,
)
from mango_mvp.productization.test_ingest import clean


RECORDING_DOWNLOAD_BRIDGE_SCHEMA_VERSION = "recording_download_bridge_v1"


@dataclass(frozen=True)
class RecordingDownloadBridgeSummary:
    schema_version: str
    download_manifest_path: str
    capture_manifest_path: str
    bridge_plan_path: str
    csv_path: Optional[str]
    source_dir: str
    db_paths: Sequence[str]
    download_manifest_rows: int
    latest_available_events: int
    converted_capture_manifest_rows: int
    bridge_total_manifest_events: int
    would_import: int
    blocked: int
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_recording_download_bridge_dry_run(
    download_manifest_path: Path,
    product_root: Path,
    capture_manifest_path: Path,
    bridge_plan_path: Path,
    source_dir: Path,
    csv_path: Optional[Path] = None,
    db_paths: Optional[Sequence[Path]] = None,
    tolerance_sec: int = 120,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    product_root, download_manifest_path, capture_manifest_path, bridge_plan_path, csv_path = resolve_bridge_paths(
        product_root=product_root,
        download_manifest_path=download_manifest_path,
        capture_manifest_path=capture_manifest_path,
        bridge_plan_path=bridge_plan_path,
        csv_path=csv_path,
    )
    source_dir = source_dir.resolve(strict=False)
    guarded_db_paths = tuple(guard_readonly_db_path(path) for path in (db_paths or ()))

    rows = read_download_manifest(download_manifest_path)
    available_rows = latest_available_download_rows(rows)
    capture_rows = [capture_manifest_row_from_download(row, product_root) for row in available_rows]
    write_jsonl(capture_manifest_path, capture_rows)

    bridge_plan = build_pipeline_bridge_plan(
        manifest_path=capture_manifest_path,
        source_dir=source_dir,
        db_paths=guarded_db_paths,
        tolerance_sec=tolerance_sec,
        verify_checksum=verify_checksum,
    )
    if csv_path:
        write_bridge_plan_csv(bridge_plan, csv_path)
    write_json(bridge_plan_path, build_report_payload(
        download_manifest_path=download_manifest_path,
        capture_manifest_path=capture_manifest_path,
        bridge_plan_path=bridge_plan_path,
        csv_path=csv_path,
        source_dir=source_dir,
        db_paths=guarded_db_paths,
        download_rows=rows,
        capture_rows=capture_rows,
        bridge_plan=bridge_plan,
    ))
    return json.loads(bridge_plan_path.read_text(encoding="utf-8"))


def latest_available_download_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    latest_by_event: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        event_key = clean(row.get("event_key"))
        if not event_key:
            continue
        if clean(row.get("action")) in AVAILABLE_RECORDING_ACTIONS:
            latest_by_event[event_key] = row
    return sorted(latest_by_event.values(), key=lambda row: (clean(row.get("started_at")), clean(row.get("event_key"))))


def capture_manifest_row_from_download(row: Mapping[str, Any], product_root: Path) -> Mapping[str, Any]:
    local_audio_path = Path(clean(row.get("local_audio_path"))).resolve(strict=False)
    guard_download_path(local_audio_path, product_root, "downloaded audio path")
    if not local_audio_path.exists():
        raise FileNotFoundError(local_audio_path)
    size_bytes = local_audio_path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"downloaded audio file is empty: {local_audio_path}")
    checksum = clean(row.get("checksum_sha256")) or file_sha256(local_audio_path)
    return {
        "schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
        "created_at": clean(row.get("created_at")),
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")) or "mango",
        "event_key": clean(row.get("event_key")),
        "provider_call_id": clean(row.get("provider_call_id")),
        "recording_id": clean(row.get("recording_id")) or None,
        "started_at": clean(row.get("started_at")),
        "ended_at": clean(row.get("ended_at")) or None,
        "direction": clean(row.get("direction")) or "unknown",
        "client_phone": clean(row.get("client_phone")) or None,
        "manager_ref": clean(row.get("manager_ref")) or None,
        "status": "downloaded",
        "local_audio_path": str(local_audio_path),
        "size_bytes": int(row.get("size_bytes") or size_bytes),
        "checksum_sha256": checksum,
        "duration_sec": optional_float(row.get("duration_sec")),
        "codec_name": clean(row.get("codec_name")) or None,
        "channels": optional_int(row.get("channels")),
        "sample_rate": optional_int(row.get("sample_rate")),
        "source_download_action": clean(row.get("action")),
        "source_download_schema_version": clean(row.get("schema_version")) or RECORDING_CAPTURE_DOWNLOAD_SCHEMA_VERSION,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "write_crm": False,
    }


def build_report_payload(
    download_manifest_path: Path,
    capture_manifest_path: Path,
    bridge_plan_path: Path,
    csv_path: Optional[Path],
    source_dir: Path,
    db_paths: Sequence[Path],
    download_rows: Sequence[Mapping[str, Any]],
    capture_rows: Sequence[Mapping[str, Any]],
    bridge_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    bridge_audit = bridge_plan.get("audit", {})
    bridge_summary = bridge_plan.get("summary", {})
    bridge_status_counts = bridge_summary.get("bridge_status_counts", {})
    blocked = int(bridge_audit.get("blocked") or 0)
    warnings = int(bridge_status_counts.get("already_present_audio", 0) or 0) + int(
        bridge_status_counts.get("already_present_db", 0) or 0
    )
    summary = RecordingDownloadBridgeSummary(
        schema_version=RECORDING_DOWNLOAD_BRIDGE_SCHEMA_VERSION,
        download_manifest_path=str(download_manifest_path),
        capture_manifest_path=str(capture_manifest_path),
        bridge_plan_path=str(bridge_plan_path),
        csv_path=str(csv_path) if csv_path else None,
        source_dir=str(source_dir),
        db_paths=tuple(str(path) for path in db_paths),
        download_manifest_rows=len(download_rows),
        latest_available_events=len(capture_rows),
        converted_capture_manifest_rows=len(capture_rows),
        bridge_total_manifest_events=int(bridge_summary.get("total_manifest_events") or 0),
        would_import=int(bridge_audit.get("would_import") or 0),
        blocked=blocked,
        validation_ok=blocked == 0,
        warnings=warnings,
    )
    return {
        "summary": summary.to_json_dict(),
        "download_action_counts": dict(sorted(Counter(clean(row.get("action")) for row in download_rows).items())),
        "converted_action_counts": dict(sorted(Counter(clean(row.get("source_download_action")) for row in capture_rows).items())),
        "bridge": bridge_plan,
        "safety": {
            "copy_audio_to_legacy_source": False,
            "product_db_writes": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
        },
    }


def write_import_candidates_csv(report: Mapping[str, Any], path: Path) -> None:
    items = report.get("bridge", {}).get("items", [])
    fieldnames = [
        "status",
        "reason",
        "started_at_msk",
        "client_phone",
        "manager_ref",
        "duration_sec",
        "provider_call_id",
        "recording_id",
        "local_audio_path",
        "proposed_filename",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def resolve_bridge_paths(
    product_root: Path,
    download_manifest_path: Path,
    capture_manifest_path: Path,
    bridge_plan_path: Path,
    csv_path: Optional[Path],
) -> tuple[Path, Path, Path, Path, Optional[Path]]:
    product_root = product_root.resolve(strict=False)
    download_manifest_path = download_manifest_path.resolve(strict=False)
    capture_manifest_path = capture_manifest_path.resolve(strict=False)
    bridge_plan_path = bridge_plan_path.resolve(strict=False)
    csv_path = csv_path.resolve(strict=False) if csv_path else None
    guard_download_path(download_manifest_path, product_root, "download manifest")
    guard_download_path(capture_manifest_path, product_root, "capture bridge manifest")
    guard_download_path(bridge_plan_path, product_root, "bridge plan")
    if csv_path:
        guard_download_path(csv_path, product_root, "bridge CSV")
    return product_root, download_manifest_path, capture_manifest_path, bridge_plan_path, csv_path


def guard_readonly_db_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("bridge DB read must not target stable_runtime")
    return resolved


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
