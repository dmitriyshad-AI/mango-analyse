from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.processing_handoff import (
    ASR_HANDOFF_STATUS,
    PROCESSING_HANDOFF_SCHEMA_VERSION,
)
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_PACK_SCHEMA_VERSION = "asr_worker_pack_v1"


@dataclass(frozen=True)
class AsrWorkerPackSummary:
    schema_version: str
    product_root: str
    source_manifest_path: str
    pack_root: str
    pack_manifest_path: str
    dry_run: bool
    materialize_mode: str
    source_manifest_rows: int
    selected_items: int
    manifest_rows: int
    copied: int
    hardlinked: int
    already_present: int
    blocked: int
    skipped_not_ready: int
    warnings: int
    pack_audio_files: int
    pack_total_bytes: int
    manifest_sha256: str
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_pack(
    source_manifest_path: Path,
    product_root: Path,
    pack_root: Path,
    pack_manifest_path: Path,
    out_path: Optional[Path] = None,
    dry_run: bool = False,
    mode: str = "copy",
    overwrite: bool = False,
    verify_checksum: bool = True,
    limit: Optional[int] = None,
) -> Mapping[str, Any]:
    paths = resolve_pack_paths(
        source_manifest_path=source_manifest_path,
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest_path,
        out_path=out_path,
    )
    source_manifest_path = paths["source_manifest_path"]
    product_root = paths["product_root"]
    pack_root = paths["pack_root"]
    pack_manifest_path = paths["pack_manifest_path"]
    out_path = paths.get("out_path")
    if mode not in {"copy", "hardlink"}:
        raise ValueError("mode must be one of: copy, hardlink")

    source_rows = read_jsonl(source_manifest_path, limit=limit)
    items = [
        plan_pack_item(row, row_number=index, product_root=product_root, pack_root=pack_root, verify_checksum=verify_checksum)
        for index, row in enumerate(source_rows, start=1)
    ]
    items = apply_duplicate_pack_blocks(items)
    if not dry_run:
        items = materialize_pack_items(items, pack_root=pack_root, mode=mode, overwrite=overwrite, verify_checksum=verify_checksum)
    manifest_rows = [worker_manifest_item(item) for item in items if item["action"] in {"PACK_ASR_WORKER_ITEM", "SKIP_ALREADY_PACKED", "PLAN_ASR_WORKER_PACK_ITEM"}]
    if dry_run:
        manifest_rows = [worker_manifest_item(item) for item in items if item["action"] == "PLAN_ASR_WORKER_PACK_ITEM"]
    write_jsonl(pack_manifest_path, manifest_rows)
    manifest_sha = sha256_file(pack_manifest_path)
    pack_audit = audit_pack_audio(pack_root=pack_root, expected_items=items, verify_checksum=verify_checksum)
    action_counts = action_counts_for(items)
    blocked = int(action_counts.get("BLOCK_ASR_WORKER_PACK") or 0)
    summary = AsrWorkerPackSummary(
        schema_version=ASR_WORKER_PACK_SCHEMA_VERSION,
        product_root=str(product_root),
        source_manifest_path=str(source_manifest_path),
        pack_root=str(pack_root),
        pack_manifest_path=str(pack_manifest_path),
        dry_run=dry_run,
        materialize_mode=mode,
        source_manifest_rows=len(source_rows),
        selected_items=len(items),
        manifest_rows=len(manifest_rows),
        copied=int(action_counts.get("PACK_ASR_WORKER_ITEM") or 0) if mode == "copy" else 0,
        hardlinked=int(action_counts.get("PACK_ASR_WORKER_ITEM") or 0) if mode == "hardlink" else 0,
        already_present=int(action_counts.get("SKIP_ALREADY_PACKED") or 0),
        blocked=blocked,
        skipped_not_ready=int(action_counts.get("SKIP_NOT_READY_FOR_ASR") or 0),
        warnings=count_warnings(items),
        pack_audio_files=int(pack_audit["pack_audio_files"]),
        pack_total_bytes=int(pack_audit["pack_total_bytes"]),
        manifest_sha256=manifest_sha,
        validation_ok=blocked == 0
        and len(manifest_rows) == sum(1 for item in items if item["action"] in {"PACK_ASR_WORKER_ITEM", "SKIP_ALREADY_PACKED", "PLAN_ASR_WORKER_PACK_ITEM"})
        and int(pack_audit["blocked"]) == 0,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts,
        "pack_audit": pack_audit,
        "items": items,
        "manifest_samples": manifest_rows[:20],
        "contract": worker_pack_contract(),
        "safety": {
            "product_db_writes": False,
            "asset_db_writes": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "downloads_audio": False,
            "copies_audio": bool(int(action_counts.get("PACK_ASR_WORKER_ITEM") or 0) and mode == "copy"),
            "hardlinks_audio": bool(int(action_counts.get("PACK_ASR_WORKER_ITEM") or 0) and mode == "hardlink"),
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_tallanto": False,
        },
    }
    if out_path:
        write_json(out_path, report)
    return report


def plan_pack_item(
    row: Mapping[str, Any],
    row_number: int,
    product_root: Path,
    pack_root: Path,
    verify_checksum: bool,
) -> dict[str, Any]:
    audio_path = Path(clean(row.get("audio_path"))).resolve(strict=False)
    queue_item_id = clean(row.get("queue_item_id"))
    target_audio_rel_path = f"audio/{queue_item_id[:16]}__{safe_filename(clean(row.get('source_filename')) or audio_path.name)}"
    target_audio_path = (pack_root / target_audio_rel_path).resolve(strict=False)
    planned_outputs_rel = planned_output_rel_paths(row, queue_item_id)
    item: dict[str, Any] = {
        "action": "PLAN_ASR_WORKER_PACK_ITEM",
        "reason": "ready_for_asr_worker_pack",
        "row_number": row_number,
        "schema_version": clean(row.get("schema_version")),
        "queue_status": clean(row.get("queue_status")),
        "queue_item_id": queue_item_id,
        "asset_id": optional_int(row.get("asset_id")),
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")),
        "event_key": clean(row.get("event_key")),
        "provider_call_id": clean(row.get("provider_call_id")),
        "recording_id": clean(row.get("recording_id")),
        "package_ref": clean(row.get("package_ref")),
        "source_audio_path": str(audio_path),
        "pack_audio_rel_path": target_audio_rel_path,
        "pack_audio_path": str(target_audio_path),
        "source_filename": clean(row.get("source_filename")) or audio_path.name,
        "checksum_sha256": clean(row.get("checksum_sha256")).lower(),
        "size_bytes": optional_int(row.get("size_bytes")),
        "duration_sec": optional_float(row.get("duration_sec")),
        "started_at": clean(row.get("started_at")) or None,
        "direction": clean(row.get("direction")) or None,
        "client_phone": clean(row.get("client_phone")) or None,
        "manager_ref": clean(row.get("manager_ref")) or None,
        "manager_name": clean(row.get("manager_name")) or None,
        "planned_outputs_rel": planned_outputs_rel,
        "blocked_reasons": [],
        "warnings": [],
    }
    if item["queue_status"] != ASR_HANDOFF_STATUS:
        item["action"] = "SKIP_NOT_READY_FOR_ASR"
        item["reason"] = f"queue_status_is_{item['queue_status'] or 'empty'}"
        return item
    if item["schema_version"] != PROCESSING_HANDOFF_SCHEMA_VERSION:
        item["warnings"].append("unexpected_source_schema_version")
    for field in ("queue_item_id", "tenant_id", "provider", "event_key", "recording_id", "checksum_sha256"):
        if not clean(item.get(field)):
            item["blocked_reasons"].append(f"missing_{field}")
    item["blocked_reasons"].extend(validate_source_audio(audio_path, product_root=product_root))
    item["blocked_reasons"].extend(validate_pack_target(target_audio_path, pack_root=pack_root))
    if verify_checksum and not item["blocked_reasons"]:
        actual_checksum = sha256_file(audio_path)
        item["actual_checksum_sha256"] = actual_checksum
        if actual_checksum != item["checksum_sha256"]:
            item["blocked_reasons"].append("checksum_sha256_mismatch")
    if item["blocked_reasons"]:
        item["action"] = "BLOCK_ASR_WORKER_PACK"
        item["reason"] = ",".join(sorted(set(item["blocked_reasons"])))
    return item


def materialize_pack_items(
    items: Sequence[Mapping[str, Any]],
    pack_root: Path,
    mode: str,
    overwrite: bool,
    verify_checksum: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        if item["action"] != "PLAN_ASR_WORKER_PACK_ITEM":
            result.append(item)
            continue
        source_audio_path = Path(clean(item.get("source_audio_path"))).resolve(strict=False)
        target_audio_path = Path(clean(item.get("pack_audio_path"))).resolve(strict=False)
        if not path_is_relative_to(target_audio_path, pack_root):
            item["action"] = "BLOCK_ASR_WORKER_PACK"
            item["reason"] = "pack_audio_outside_pack_root"
            item["blocked_reasons"] = ["pack_audio_outside_pack_root"]
            result.append(item)
            continue
        target_audio_path.parent.mkdir(parents=True, exist_ok=True)
        if target_audio_path.exists() and not overwrite:
            if verify_checksum and sha256_file(target_audio_path) != item["checksum_sha256"]:
                item["action"] = "BLOCK_ASR_WORKER_PACK"
                item["reason"] = "existing_pack_audio_checksum_mismatch"
                item["blocked_reasons"] = ["existing_pack_audio_checksum_mismatch"]
            else:
                item["action"] = "SKIP_ALREADY_PACKED"
                item["reason"] = "pack_audio_already_present"
            result.append(item)
            continue
        if mode == "hardlink":
            if target_audio_path.exists() and overwrite:
                target_audio_path.unlink()
            target_audio_path.hardlink_to(source_audio_path)
        else:
            shutil.copy2(source_audio_path, target_audio_path)
        if verify_checksum and sha256_file(target_audio_path) != item["checksum_sha256"]:
            item["action"] = "BLOCK_ASR_WORKER_PACK"
            item["reason"] = "copied_pack_audio_checksum_mismatch"
            item["blocked_reasons"] = ["copied_pack_audio_checksum_mismatch"]
        else:
            item["action"] = "PACK_ASR_WORKER_ITEM"
            item["reason"] = "pack_audio_materialized"
        result.append(item)
    return result


def apply_duplicate_pack_blocks(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in items if clean(item.get("queue_item_id")))
    target_counts = Counter(clean(item.get("pack_audio_rel_path")) for item in items if clean(item.get("pack_audio_rel_path")))
    result: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        if item["action"] == "PLAN_ASR_WORKER_PACK_ITEM":
            blocked = list(item.get("blocked_reasons") or [])
            if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
                blocked.append("duplicate_queue_item_id")
            if target_counts.get(clean(item.get("pack_audio_rel_path")), 0) > 1:
                blocked.append("duplicate_pack_audio_rel_path")
            if blocked:
                item["blocked_reasons"] = sorted(set(blocked))
                item["action"] = "BLOCK_ASR_WORKER_PACK"
                item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def worker_manifest_item(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": ASR_WORKER_PACK_SCHEMA_VERSION,
        "queue_status": ASR_HANDOFF_STATUS,
        "queue_item_id": item["queue_item_id"],
        "asset_id": item["asset_id"],
        "tenant_id": item["tenant_id"],
        "provider": item["provider"],
        "event_key": item["event_key"],
        "provider_call_id": item["provider_call_id"],
        "recording_id": item["recording_id"],
        "package_ref": item["package_ref"],
        "audio_rel_path": item["pack_audio_rel_path"],
        "audio_sha256": item["checksum_sha256"],
        "size_bytes": item["size_bytes"],
        "duration_sec": item["duration_sec"],
        "started_at": item["started_at"],
        "direction": item["direction"],
        "client_phone": item["client_phone"],
        "manager_ref": item["manager_ref"],
        "manager_name": item["manager_name"],
        "planned_outputs_rel": item["planned_outputs_rel"],
        "source_refs": {
            "source_handoff_schema_version": item["schema_version"],
            "source_audio_path": item["source_audio_path"],
            "source_filename": item["source_filename"],
        },
    }


def planned_output_rel_paths(row: Mapping[str, Any], queue_item_id: str) -> Mapping[str, str]:
    tenant = safe_slug(clean(row.get("tenant_id")) or "tenant")
    provider = safe_slug(clean(row.get("provider")) or "provider")
    stem = f"{clean(row.get('started_at'))[:10] or 'undated'}__{queue_item_id[:16]}"
    return {
        "transcript_json": f"outputs/{tenant}/{provider}/{stem}.transcript.json",
        "transcript_txt": f"outputs/{tenant}/{provider}/{stem}.transcript.txt",
        "asr_audit_json": f"outputs/{tenant}/{provider}/{stem}.asr_audit.json",
    }


def validate_source_audio(audio_path: Path, product_root: Path) -> list[str]:
    blocked: list[str] = []
    if "stable_runtime" in audio_path.parts:
        blocked.append("source_audio_under_stable_runtime")
    if not path_is_relative_to(audio_path, product_root):
        blocked.append("source_audio_outside_product_root")
    if audio_path.suffix.lower() != ".mp3":
        blocked.append("unsupported_audio_extension")
    if not audio_path.exists():
        blocked.append("source_audio_missing")
    elif not audio_path.is_file():
        blocked.append("source_audio_not_file")
    elif audio_path.stat().st_size <= 0:
        blocked.append("zero_size_source_audio")
    return blocked


def validate_pack_target(target_audio_path: Path, pack_root: Path) -> list[str]:
    blocked: list[str] = []
    if "stable_runtime" in target_audio_path.parts:
        blocked.append("pack_audio_under_stable_runtime")
    if not path_is_relative_to(target_audio_path, pack_root):
        blocked.append("pack_audio_outside_pack_root")
    if target_audio_path.suffix.lower() != ".mp3":
        blocked.append("pack_audio_unsupported_extension")
    return blocked


def audit_pack_audio(pack_root: Path, expected_items: Sequence[Mapping[str, Any]], verify_checksum: bool) -> Mapping[str, Any]:
    expected = {
        clean(item.get("pack_audio_path")): item
        for item in expected_items
        if item["action"] in {"PACK_ASR_WORKER_ITEM", "SKIP_ALREADY_PACKED"}
    }
    audio_dir = pack_root / "audio"
    actual_files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []
    missing_expected_files = []
    checksum_mismatch_files = []
    zero_size_files = []
    for path_text, item in expected.items():
        path = Path(path_text)
        if not path.exists():
            missing_expected_files.append(path_text)
            continue
        if path.stat().st_size <= 0:
            zero_size_files.append(path_text)
        if verify_checksum and sha256_file(path) != clean(item.get("checksum_sha256")):
            checksum_mismatch_files.append(path_text)
    expected_paths = {str(Path(path).resolve(strict=False)) for path in expected}
    unreferenced_audio_files = [str(path.resolve(strict=False)) for path in actual_files if str(path.resolve(strict=False)) not in expected_paths]
    blocked_reasons = {
        "missing_expected_files": len(missing_expected_files),
        "checksum_mismatch_files": len(checksum_mismatch_files),
        "zero_size_files": len(zero_size_files),
        "unreferenced_audio_files": len(unreferenced_audio_files),
    }
    return {
        "pack_root": str(pack_root),
        "pack_audio_files": len(actual_files),
        "pack_total_bytes": sum(path.stat().st_size for path in actual_files),
        "expected_audio_files": len(expected),
        "blocked": sum(blocked_reasons.values()),
        "blocked_reasons": blocked_reasons,
        "missing_expected_files": missing_expected_files[:100],
        "checksum_mismatch_files": checksum_mismatch_files[:100],
        "zero_size_files": zero_size_files[:100],
        "unreferenced_audio_files": unreferenced_audio_files[:100],
    }


def worker_pack_contract() -> Mapping[str, Any]:
    return {
        "schema_version": ASR_WORKER_PACK_SCHEMA_VERSION,
        "input_manifest": "asr_worker_input_manifest.jsonl",
        "audio_location": "audio/*.mp3",
        "output_location": "outputs/<tenant>/<provider>/*",
        "worker_must_verify": ["audio_rel_path_exists", "audio_sha256", "runtime_target_approval"],
        "worker_must_not_do_in_pack_build": ["run_asr", "write_runtime_db", "write_crm"],
    }


def read_jsonl(path: Path, limit: Optional[int] = None) -> list[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_pack_paths(
    source_manifest_path: Path,
    product_root: Path,
    pack_root: Path,
    pack_manifest_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "source_manifest_path": source_manifest_path.resolve(strict=False),
        "product_root": product_root.resolve(strict=False),
        "pack_root": pack_root.resolve(strict=False),
        "pack_manifest_path": pack_manifest_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_pack_paths(**paths)
    return paths


def guard_pack_paths(
    source_manifest_path: Path,
    product_root: Path,
    pack_root: Path,
    pack_manifest_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("source handoff manifest", source_manifest_path),
        ("ASR worker pack root", pack_root),
        ("ASR worker pack manifest", pack_manifest_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not source_manifest_path.exists() or not source_manifest_path.is_file():
        raise FileNotFoundError(f"source handoff manifest not found: {source_manifest_path}")
    if not path_is_relative_to(pack_manifest_path, pack_root):
        raise ValueError(f"ASR worker pack manifest must stay under pack root: {pack_root}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR worker pack audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR worker pack audit must stay under product root: {product_root}")


def action_counts_for(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(item.get("action")) for item in items).items()))


def count_warnings(items: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(item.get("warnings") or []) for item in items)


def optional_int(value: Any) -> Optional[int]:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def optional_float(value: Any) -> Optional[float]:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_filename(value: str) -> str:
    name = Path(value).name
    result = []
    for char in name:
        if char.isalnum() or char in {"-", "_", ".", "="}:
            result.append(char)
        else:
            result.append("_")
    text = "".join(result).strip("._")
    return text if text.endswith(".mp3") else f"{text or 'audio'}.mp3"


def safe_slug(value: str) -> str:
    result = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_"}:
            result.append(char)
        else:
            result.append("_")
    return "".join(result).strip("_") or "unknown"
