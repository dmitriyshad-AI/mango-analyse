from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.capture_staging import file_sha256, optional_float, optional_int


QUARANTINE_PLAN_SCHEMA_VERSION = "quarantine_import_plan_v1"
QUARANTINE_MATERIALIZATION_SCHEMA_VERSION = "quarantine_materialization_v1"
IMPORTABLE_BRIDGE_STATUS = "would_import"
MATERIALIZE_MODES = ("copy", "hardlink")


@dataclass(frozen=True)
class QuarantinePlanItem:
    schema_version: str
    status: str
    reason: str
    event_key: str
    provider_call_id: str
    recording_id: Optional[str]
    source_audio_path: Optional[str]
    target_filename: Optional[str]
    target_audio_path: Optional[str]
    started_at_msk: Optional[str]
    phone: Optional[str]
    manager_ref: Optional[str]
    direction: Optional[str]
    duration_sec: Optional[float]
    checksum_sha256: Optional[str]
    size_bytes: Optional[int]
    metadata: Mapping[str, Any]
    bridge_status: str
    bridge_reason: Optional[str] = None
    collision_original_filename: Optional[str] = None

    def to_json_dict(self) -> Mapping[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class QuarantinePlanSummary:
    bridge_plan_path: str
    quarantine_dir: str
    metadata_csv_path: str
    copy_mode: str
    total_bridge_items: int
    ready: int
    skipped_non_import_status: int
    blocked: int
    status_counts: Mapping[str, int]
    ready_total_mb: float
    ready_by_day: Mapping[str, int]
    unique_target_filenames: int
    target_filename_collisions: int
    metadata_rows: int
    quarantine_audio_files: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuarantineMaterializeItem:
    schema_version: str
    status: str
    reason: str
    event_key: str
    source_audio_path: Optional[str]
    target_audio_path: Optional[str]
    checksum_sha256: Optional[str]
    size_bytes: Optional[int]
    action: Optional[str] = None

    def to_json_dict(self) -> Mapping[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class QuarantineMaterializeSummary:
    plan_path: str
    quarantine_dir: str
    materialize_mode: str
    total_plan_items: int
    ready_plan_items: int
    copied: int
    hardlinked: int
    already_present: int
    skipped_non_ready: int
    blocked: int
    status_counts: Mapping[str, int]
    target_audio_files: int
    target_total_mb: float
    expected_ready_files: int
    missing_expected_files: int
    checksum_mismatch_files: int
    zero_size_files: int
    unreferenced_audio_files: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_quarantine_import_plan(
    bridge_plan_path: Path,
    quarantine_dir: Path,
    metadata_csv_path: Path,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    bridge_plan = json.loads(bridge_plan_path.read_text(encoding="utf-8"))
    bridge_items = bridge_plan.get("items", [])
    used_filenames: dict[str, str] = {}
    items: list[QuarantinePlanItem] = []
    for bridge_item in bridge_items:
        item = plan_quarantine_item(
            bridge_item=bridge_item,
            quarantine_dir=quarantine_dir,
            used_filenames=used_filenames,
            verify_checksum=verify_checksum,
        )
        items.append(item)

    write_metadata_csv(items, metadata_csv_path)
    audit = audit_quarantine_items(items, quarantine_dir=quarantine_dir, metadata_csv_path=metadata_csv_path)
    summary = QuarantinePlanSummary(
        bridge_plan_path=str(bridge_plan_path),
        quarantine_dir=str(quarantine_dir),
        metadata_csv_path=str(metadata_csv_path),
        copy_mode="dry_run",
        total_bridge_items=len(bridge_items),
        ready=audit["ready"],
        skipped_non_import_status=audit["status_counts"].get("skipped_non_import_status", 0),
        blocked=audit["blocked"],
        status_counts=audit["status_counts"],
        ready_total_mb=audit["ready_total_mb"],
        ready_by_day=audit["ready_by_day"],
        unique_target_filenames=audit["unique_target_filenames"],
        target_filename_collisions=audit["target_filename_collisions"],
        metadata_rows=audit["metadata_rows"],
        quarantine_audio_files=audit["quarantine_audio_files"],
    )
    return {
        "summary": summary.to_json_dict(),
        "items": [item.to_json_dict() for item in items],
        "audit": audit,
    }


def materialize_quarantine_package(
    plan_path: Path,
    mode: str = "copy",
    verify_checksum: bool = True,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    if mode not in MATERIALIZE_MODES:
        raise ValueError(f"mode must be one of: {', '.join(MATERIALIZE_MODES)}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_summary = plan.get("summary") or {}
    quarantine_dir = Path(str(plan_summary.get("quarantine_dir") or ""))
    if not str(quarantine_dir):
        raise ValueError("plan summary does not contain quarantine_dir")

    items: list[QuarantineMaterializeItem] = []
    for plan_item in plan.get("items", []):
        items.append(
            materialize_plan_item(
                plan_item=plan_item,
                quarantine_dir=quarantine_dir,
                mode=mode,
                verify_checksum=verify_checksum,
                overwrite=overwrite,
            )
        )

    audit = audit_materialized_items(
        plan_items=plan.get("items", []),
        materialized_items=items,
        quarantine_dir=quarantine_dir,
        verify_checksum=verify_checksum,
    )
    status_counts = Counter(item.status for item in items)
    summary = QuarantineMaterializeSummary(
        plan_path=str(plan_path),
        quarantine_dir=str(quarantine_dir),
        materialize_mode=mode,
        total_plan_items=len(plan.get("items", [])),
        ready_plan_items=len([item for item in plan.get("items", []) if item.get("status") == "ready"]),
        copied=status_counts.get("copied", 0),
        hardlinked=status_counts.get("hardlinked", 0),
        already_present=status_counts.get("already_present", 0),
        skipped_non_ready=status_counts.get("skipped_non_ready", 0),
        blocked=sum(count for status, count in status_counts.items() if status.startswith("blocked_")),
        status_counts=dict(sorted(status_counts.items())),
        target_audio_files=audit["target_audio_files"],
        target_total_mb=audit["target_total_mb"],
        expected_ready_files=audit["expected_ready_files"],
        missing_expected_files=audit["missing_expected_files"],
        checksum_mismatch_files=audit["checksum_mismatch_files"],
        zero_size_files=audit["zero_size_files"],
        unreferenced_audio_files=audit["unreferenced_audio_files"],
    )
    return {
        "summary": summary.to_json_dict(),
        "items": [item.to_json_dict() for item in items],
        "audit": audit,
    }


def materialize_plan_item(
    plan_item: Mapping[str, Any],
    quarantine_dir: Path,
    mode: str,
    verify_checksum: bool,
    overwrite: bool,
) -> QuarantineMaterializeItem:
    base = materialize_base_kwargs(plan_item)
    if plan_item.get("status") != "ready":
        return QuarantineMaterializeItem(
            **base,
            status="skipped_non_ready",
            reason=f"plan_status_is_{plan_item.get('status') or 'empty'}",
        )

    source_path = Path(str(plan_item.get("source_audio_path") or ""))
    target_path = Path(str(plan_item.get("target_audio_path") or ""))
    if not str(source_path):
        return blocked_materialize_item(base, "blocked_missing_source", "source_audio_path_missing")
    if not str(target_path):
        return blocked_materialize_item(base, "blocked_missing_target", "target_audio_path_missing")
    if not is_target_under_quarantine(target_path, quarantine_dir):
        return blocked_materialize_item(base, "blocked_unsafe_target", "target_outside_quarantine_dir")
    if not source_path.exists():
        return blocked_materialize_item(base, "blocked_missing_source", "source_audio_file_missing")
    if not source_path.is_file():
        return blocked_materialize_item(base, "blocked_invalid_source", "source_audio_path_not_file")
    if source_path.stat().st_size <= 0:
        return blocked_materialize_item(base, "blocked_zero_size_source", "source_audio_file_zero_size")

    expected_checksum = optional_str(plan_item.get("checksum_sha256"))
    if verify_checksum and expected_checksum and file_sha256(source_path) != expected_checksum:
        return blocked_materialize_item(base, "blocked_source_checksum_mismatch", "source_checksum_mismatch")

    if target_path.exists():
        if not target_path.is_file():
            return blocked_materialize_item(base, "blocked_invalid_target", "target_audio_path_not_file")
        if verify_checksum and expected_checksum and file_sha256(target_path) != expected_checksum:
            if not overwrite:
                return blocked_materialize_item(
                    base,
                    "blocked_existing_target_checksum_mismatch",
                    "existing_target_checksum_mismatch",
                )
            target_path.unlink()
        else:
            return QuarantineMaterializeItem(
                **base,
                status="already_present",
                reason="target_audio_already_present_and_valid",
                action="none",
            )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "copy":
            shutil.copy2(source_path, target_path)
            status = "copied"
            action = "copy2"
        else:
            os.link(source_path, target_path)
            status = "hardlinked"
            action = "hardlink"
    except OSError as exc:
        return blocked_materialize_item(base, "blocked_write_failed", f"{type(exc).__name__}:{exc}")

    if target_path.stat().st_size <= 0:
        return blocked_materialize_item(base, "blocked_zero_size_target", "target_audio_file_zero_size")
    if verify_checksum and expected_checksum and file_sha256(target_path) != expected_checksum:
        return blocked_materialize_item(base, "blocked_written_checksum_mismatch", "written_target_checksum_mismatch")

    return QuarantineMaterializeItem(
        **base,
        status=status,
        reason="target_audio_materialized_and_verified",
        action=action,
    )


def audit_materialized_items(
    plan_items: Sequence[Mapping[str, Any]],
    materialized_items: Sequence[QuarantineMaterializeItem],
    quarantine_dir: Path,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    ready_plan_items = [item for item in plan_items if item.get("status") == "ready"]
    expected_paths = [Path(str(item.get("target_audio_path") or "")) for item in ready_plan_items]
    expected_path_set = {str(path) for path in expected_paths if str(path)}
    target_files = sorted(quarantine_dir.glob("*.mp3")) if quarantine_dir.exists() else []
    target_file_set = {str(path) for path in target_files}

    missing_expected_files = 0
    checksum_mismatch_files = 0
    zero_size_files = 0
    verified_files = 0
    for item in ready_plan_items:
        target_path = Path(str(item.get("target_audio_path") or ""))
        expected_checksum = optional_str(item.get("checksum_sha256"))
        if not target_path.exists():
            missing_expected_files += 1
            continue
        if target_path.stat().st_size <= 0:
            zero_size_files += 1
        if verify_checksum and expected_checksum:
            if file_sha256(target_path) != expected_checksum:
                checksum_mismatch_files += 1
            else:
                verified_files += 1

    target_total_bytes = sum(path.stat().st_size for path in target_files if path.is_file())
    status_counts = Counter(item.status for item in materialized_items)
    return {
        "quarantine_dir": str(quarantine_dir),
        "target_audio_files": len(target_files),
        "target_total_mb": round(target_total_bytes / 1024 / 1024, 2),
        "expected_ready_files": len(ready_plan_items),
        "missing_expected_files": missing_expected_files,
        "checksum_mismatch_files": checksum_mismatch_files,
        "zero_size_files": zero_size_files,
        "checksum_verified_files": verified_files,
        "unreferenced_audio_files": len(target_file_set - expected_path_set),
        "status_counts": dict(sorted(status_counts.items())),
        "samples": {
            "blocked": [sample_materialized_item(item) for item in materialized_items if item.status.startswith("blocked_")][:20],
            "changed": [
                sample_materialized_item(item)
                for item in materialized_items
                if item.status in {"copied", "hardlinked", "already_present"}
            ][:20],
        },
    }


def blocked_materialize_item(
    base: Mapping[str, Any],
    status: str,
    reason: str,
) -> QuarantineMaterializeItem:
    return QuarantineMaterializeItem(
        **base,
        status=status,
        reason=reason,
    )


def materialize_base_kwargs(plan_item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": QUARANTINE_MATERIALIZATION_SCHEMA_VERSION,
        "event_key": str(plan_item.get("event_key") or ""),
        "source_audio_path": optional_str(plan_item.get("source_audio_path")),
        "target_audio_path": optional_str(plan_item.get("target_audio_path")),
        "checksum_sha256": optional_str(plan_item.get("checksum_sha256")),
        "size_bytes": optional_int(plan_item.get("size_bytes")),
    }


def is_target_under_quarantine(target_path: Path, quarantine_dir: Path) -> bool:
    try:
        return target_path.resolve(strict=False).is_relative_to(quarantine_dir.resolve(strict=False))
    except AttributeError:
        try:
            target_path.resolve(strict=False).relative_to(quarantine_dir.resolve(strict=False))
            return True
        except ValueError:
            return False


def sample_materialized_item(item: QuarantineMaterializeItem) -> Mapping[str, Any]:
    return {
        "status": item.status,
        "reason": item.reason,
        "event_key": item.event_key,
        "source_audio_path": item.source_audio_path,
        "target_audio_path": item.target_audio_path,
    }


def plan_quarantine_item(
    bridge_item: Mapping[str, Any],
    quarantine_dir: Path,
    used_filenames: dict[str, str],
    verify_checksum: bool,
) -> QuarantinePlanItem:
    base = base_item_kwargs(bridge_item)
    bridge_status = str(bridge_item.get("status") or "")
    if bridge_status != IMPORTABLE_BRIDGE_STATUS:
        return QuarantinePlanItem(
            **base,
            status="skipped_non_import_status",
            reason=f"bridge_status_is_{bridge_status or 'empty'}",
            target_filename=None,
            target_audio_path=None,
            metadata={},
        )

    source_audio_path = optional_str(bridge_item.get("local_audio_path"))
    if not source_audio_path:
        return blocked_item(base, "blocked_missing_source", "local_audio_path_missing")
    source_path = Path(source_audio_path)
    if not source_path.exists():
        return blocked_item(base, "blocked_missing_source", "local_audio_file_missing")
    if source_path.stat().st_size <= 0:
        return blocked_item(base, "blocked_zero_size_source", "local_audio_file_zero_size")

    expected_checksum = optional_str(bridge_item.get("checksum_sha256"))
    if verify_checksum and expected_checksum:
        actual_checksum = file_sha256(source_path)
        if actual_checksum != expected_checksum:
            return blocked_item(base, "blocked_checksum_mismatch", "source_checksum_mismatch")

    metadata = build_metadata_row(bridge_item)
    missing_metadata = required_metadata_gaps(metadata)
    if missing_metadata:
        return blocked_item(
            base,
            "blocked_missing_metadata",
            f"missing_metadata:{','.join(missing_metadata)}",
        )

    proposed_filename = optional_str(bridge_item.get("proposed_filename"))
    if not proposed_filename:
        return blocked_item(base, "blocked_missing_metadata", "proposed_filename_missing")

    target_filename, collision_original = unique_target_filename(
        proposed_filename=proposed_filename,
        event_key=str(bridge_item.get("event_key") or ""),
        used_filenames=used_filenames,
    )
    target_audio_path = quarantine_dir / target_filename
    metadata = {**metadata, "filename": target_filename, "target_audio_path": str(target_audio_path)}
    return QuarantinePlanItem(
        **base,
        status="ready",
        reason="ready_for_quarantine_import_package",
        target_filename=target_filename,
        target_audio_path=str(target_audio_path),
        metadata=metadata,
        collision_original_filename=collision_original,
    )


def blocked_item(
    base: Mapping[str, Any],
    status: str,
    reason: str,
) -> QuarantinePlanItem:
    return QuarantinePlanItem(
        **base,
        status=status,
        reason=reason,
        target_filename=None,
        target_audio_path=None,
        metadata={},
    )


def build_metadata_row(bridge_item: Mapping[str, Any]) -> Mapping[str, Any]:
    proposed_metadata = bridge_item.get("proposed_metadata") or {}
    started_at = optional_str(proposed_metadata.get("started_at_msk")) or optional_str(
        bridge_item.get("started_at_msk")
    ) or optional_str(
        bridge_item.get("started_at")
    )
    phone = optional_str(proposed_metadata.get("client_phone")) or optional_str(
        bridge_item.get("client_phone")
    )
    manager = optional_str(proposed_metadata.get("manager_ref")) or optional_str(
        bridge_item.get("manager_ref")
    )
    provider_call_id = optional_str(proposed_metadata.get("provider_call_id")) or optional_str(
        bridge_item.get("provider_call_id")
    )
    recording_id = optional_str(proposed_metadata.get("recording_id")) or optional_str(
        bridge_item.get("recording_id")
    )
    checksum = optional_str(proposed_metadata.get("checksum_sha256")) or optional_str(
        bridge_item.get("checksum_sha256")
    )
    source_audio_path = optional_str(bridge_item.get("local_audio_path"))
    return {
        "filename": optional_str(bridge_item.get("proposed_filename")),
        "source_audio_path": source_audio_path,
        "target_audio_path": None,
        "phone": phone,
        "client_phone": phone,
        "manager": manager,
        "manager_name": f"mango_{manager or 'unknown'}",
        "started_at": started_at,
        "start_time": started_at,
        "direction": optional_str(proposed_metadata.get("direction")) or optional_str(bridge_item.get("direction")),
        "call_id": provider_call_id,
        "record_id": recording_id,
        "event_key": optional_str(proposed_metadata.get("event_key")) or optional_str(bridge_item.get("event_key")),
        "provider_call_id": provider_call_id,
        "recording_id": recording_id,
        "duration_sec": optional_float(bridge_item.get("duration_sec")),
        "checksum_sha256": checksum,
        "source_size_bytes": optional_int(bridge_item.get("size_bytes")),
        "source": "mango_api_capture",
        "tenant_id": optional_str(proposed_metadata.get("tenant_id")),
        "provider": optional_str(proposed_metadata.get("provider")) or "mango",
    }


def required_metadata_gaps(metadata: Mapping[str, Any]) -> Sequence[str]:
    required = ("source_audio_path", "phone", "started_at", "call_id", "recording_id")
    return tuple(key for key in required if not metadata.get(key))


def unique_target_filename(
    proposed_filename: str,
    event_key: str,
    used_filenames: dict[str, str],
) -> tuple[str, Optional[str]]:
    normalized = normalize_target_filename(proposed_filename)
    existing_event = used_filenames.get(normalized)
    if existing_event is None or existing_event == event_key:
        used_filenames[normalized] = event_key
        return normalized, None

    suffix = short_event_suffix(event_key)
    stem = Path(normalized).stem
    extension = Path(normalized).suffix or ".mp3"
    candidate = f"{stem}__event_{suffix}{extension}"
    counter = 2
    while candidate in used_filenames and used_filenames[candidate] != event_key:
        candidate = f"{stem}__event_{suffix}_{counter}{extension}"
        counter += 1
    used_filenames[candidate] = event_key
    return candidate, normalized


def normalize_target_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        name = "unknown.mp3"
    if not Path(name).suffix:
        name = f"{name}.mp3"
    return name


def short_event_suffix(event_key: str) -> str:
    return "".join(ch for ch in event_key if ch.isalnum())[-12:] or "unknown"


def write_metadata_csv(items: Sequence[QuarantinePlanItem], metadata_csv_path: Path) -> None:
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "source_audio_path",
        "target_audio_path",
        "phone",
        "client_phone",
        "manager",
        "manager_name",
        "started_at",
        "start_time",
        "direction",
        "call_id",
        "record_id",
        "event_key",
        "provider_call_id",
        "recording_id",
        "duration_sec",
        "checksum_sha256",
        "source_size_bytes",
        "source",
        "tenant_id",
        "provider",
    ]
    ready_items = [item for item in items if item.status == "ready"]
    with metadata_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in ready_items:
            writer.writerow({field: item.metadata.get(field, "") for field in fieldnames})


def audit_quarantine_items(
    items: Sequence[QuarantinePlanItem],
    quarantine_dir: Path,
    metadata_csv_path: Path,
) -> Mapping[str, Any]:
    status_counts = Counter(item.status for item in items)
    ready_items = [item for item in items if item.status == "ready"]
    blocked_items = [item for item in items if item.status.startswith("blocked_")]
    target_counts = Counter(item.target_filename for item in ready_items if item.target_filename)
    target_collisions = {name: count for name, count in target_counts.items() if count > 1}
    source_paths = [Path(item.source_audio_path) for item in ready_items if item.source_audio_path]
    ready_total_bytes = sum(path.stat().st_size for path in source_paths if path.exists())
    by_day = Counter(item_day(item) for item in ready_items)
    quarantine_mp3_files = list(quarantine_dir.glob("*.mp3")) if quarantine_dir.exists() else []
    return {
        "quarantine_dir": str(quarantine_dir),
        "quarantine_audio_files": len(quarantine_mp3_files),
        "copy_mode": "dry_run",
        "metadata_csv_path": str(metadata_csv_path),
        "metadata_csv_exists": metadata_csv_path.exists(),
        "metadata_rows": len(ready_items),
        "total_items": len(items),
        "ready": len(ready_items),
        "blocked": len(blocked_items),
        "status_counts": dict(sorted(status_counts.items())),
        "ready_by_day": dict(sorted(by_day.items())),
        "ready_total_mb": round(ready_total_bytes / 1024 / 1024, 2),
        "unique_target_filenames": len(target_counts),
        "target_filename_collisions": len(target_collisions),
        "samples": {
            "ready": [sample_item(item) for item in ready_items[:20]],
            "blocked": [sample_item(item) for item in blocked_items[:20]],
        },
    }


def item_day(item: QuarantinePlanItem) -> str:
    started_at = item.started_at_msk or optional_str(item.metadata.get("started_at")) or ""
    return started_at[:10] or "unknown"


def sample_item(item: QuarantinePlanItem) -> Mapping[str, Any]:
    return {
        "status": item.status,
        "reason": item.reason,
        "started_at_msk": item.started_at_msk,
        "phone": item.phone,
        "manager_ref": item.manager_ref,
        "provider_call_id": item.provider_call_id,
        "target_filename": item.target_filename,
    }


def base_item_kwargs(bridge_item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": QUARANTINE_PLAN_SCHEMA_VERSION,
        "event_key": str(bridge_item.get("event_key") or ""),
        "provider_call_id": str(bridge_item.get("provider_call_id") or ""),
        "recording_id": optional_str(bridge_item.get("recording_id")),
        "source_audio_path": optional_str(bridge_item.get("local_audio_path")),
        "started_at_msk": optional_str(bridge_item.get("started_at_msk")),
        "phone": optional_str(bridge_item.get("client_phone")),
        "manager_ref": optional_str(bridge_item.get("manager_ref")),
        "direction": optional_str(bridge_item.get("direction")),
        "duration_sec": optional_float(bridge_item.get("duration_sec")),
        "checksum_sha256": optional_str(bridge_item.get("checksum_sha256")),
        "size_bytes": optional_int(bridge_item.get("size_bytes")),
        "bridge_status": str(bridge_item.get("status") or ""),
        "bridge_reason": optional_str(bridge_item.get("reason")),
    }


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
