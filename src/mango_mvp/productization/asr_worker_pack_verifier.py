from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_worker_pack import ASR_WORKER_PACK_SCHEMA_VERSION
from mango_mvp.productization.processing_handoff import ASR_HANDOFF_STATUS
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_PACK_VERIFY_SCHEMA_VERSION = "asr_worker_pack_verify_v1"
REQUIRED_OUTPUT_KEYS = ("transcript_json", "transcript_txt", "asr_audit_json")


@dataclass(frozen=True)
class AsrWorkerPackVerifySummary:
    schema_version: str
    product_root: str
    pack_root: str
    pack_manifest_path: str
    manifest_rows: int
    ready_items: int
    blocked: int
    warnings: int
    pack_audio_files: int
    pack_total_bytes: int
    manifest_sha256: str
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def verify_asr_worker_pack(
    product_root: Path,
    pack_root: Path,
    pack_manifest_path: Path,
    out_path: Optional[Path] = None,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    paths = resolve_verify_paths(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    pack_root = paths["pack_root"]
    pack_manifest_path = paths["pack_manifest_path"]
    out_path = paths.get("out_path")

    rows = read_jsonl(pack_manifest_path)
    items = [verify_manifest_row(row, row_number=index, pack_root=pack_root, verify_checksum=verify_checksum) for index, row in enumerate(rows, start=1)]
    items = apply_duplicate_blocks(items)
    action_counts = action_counts_for(items)
    pack_audit = audit_pack_files(pack_root=pack_root, items=items)
    blocked = int(action_counts.get("BLOCK_ASR_WORKER_PACK_VERIFY") or 0) + int(pack_audit["blocked"])
    warnings = count_warnings(items)
    ready = int(action_counts.get("VERIFY_ASR_WORKER_PACK_ITEM") or 0)
    manifest_sha = sha256_file(pack_manifest_path)
    summary = AsrWorkerPackVerifySummary(
        schema_version=ASR_WORKER_PACK_VERIFY_SCHEMA_VERSION,
        product_root=str(product_root),
        pack_root=str(pack_root),
        pack_manifest_path=str(pack_manifest_path),
        manifest_rows=len(rows),
        ready_items=ready,
        blocked=blocked,
        warnings=warnings,
        pack_audio_files=int(pack_audit["pack_audio_files"]),
        pack_total_bytes=int(pack_audit["pack_total_bytes"]),
        manifest_sha256=manifest_sha,
        validation_ok=blocked == 0 and ready == len(rows),
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts,
        "pack_audit": pack_audit,
        "items": items,
        "readiness_gate": {
            "ready_for_worker": summary.validation_ok,
            "worker_may_run_asr": False,
            "requires_explicit_runtime_target_approval": True,
            "verified": [
                "manifest_schema",
                "queue_status",
                "relative_audio_paths",
                "relative_planned_output_paths",
                "audio_file_exists",
                "audio_size",
                "audio_sha256",
                "duplicate_queue_ids",
                "unreferenced_audio_files",
            ],
        },
        "safety": {
            "read_only": True,
            "product_db_writes": False,
            "asset_db_writes": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "downloads_audio": False,
            "copies_audio": False,
            "hardlinks_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_tallanto": False,
        },
    }
    if out_path:
        write_json(out_path, report)
    return report


def verify_manifest_row(
    row: Mapping[str, Any],
    row_number: int,
    pack_root: Path,
    verify_checksum: bool,
) -> dict[str, Any]:
    audio_rel_path = clean(row.get("audio_rel_path"))
    audio_path = resolve_relative_pack_path(pack_root, audio_rel_path)
    item: dict[str, Any] = {
        "action": "VERIFY_ASR_WORKER_PACK_ITEM",
        "reason": "worker_pack_item_ready",
        "row_number": row_number,
        "schema_version": clean(row.get("schema_version")),
        "queue_status": clean(row.get("queue_status")),
        "queue_item_id": clean(row.get("queue_item_id")),
        "asset_id": optional_int(row.get("asset_id")),
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")),
        "event_key": clean(row.get("event_key")),
        "recording_id": clean(row.get("recording_id")),
        "audio_rel_path": audio_rel_path,
        "audio_path": str(audio_path) if audio_path else None,
        "audio_sha256": clean(row.get("audio_sha256")).lower(),
        "size_bytes": optional_int(row.get("size_bytes")),
        "planned_outputs_rel": row.get("planned_outputs_rel") if isinstance(row.get("planned_outputs_rel"), Mapping) else {},
        "blocked_reasons": [],
        "warnings": [],
    }
    if item["schema_version"] != ASR_WORKER_PACK_SCHEMA_VERSION:
        item["blocked_reasons"].append("unexpected_schema_version")
    if item["queue_status"] != ASR_HANDOFF_STATUS:
        item["blocked_reasons"].append("unexpected_queue_status")
    for field in ("queue_item_id", "tenant_id", "provider", "event_key", "recording_id", "audio_rel_path", "audio_sha256"):
        if not clean(item.get(field)):
            item["blocked_reasons"].append(f"missing_{field}")
    item["blocked_reasons"].extend(validate_audio_rel_path(audio_rel_path, audio_path=audio_path, pack_root=pack_root))
    item["blocked_reasons"].extend(validate_planned_outputs(item["planned_outputs_rel"], pack_root=pack_root))
    item["blocked_reasons"].extend(validate_source_refs(row.get("source_refs")))
    if audio_path and not item["blocked_reasons"]:
        actual_size = audio_path.stat().st_size
        item["actual_size_bytes"] = actual_size
        if item["size_bytes"] is not None and item["size_bytes"] != actual_size:
            item["blocked_reasons"].append("size_bytes_mismatch")
        if verify_checksum:
            actual_sha = sha256_file(audio_path)
            item["actual_sha256"] = actual_sha
            if actual_sha != item["audio_sha256"]:
                item["blocked_reasons"].append("audio_sha256_mismatch")
    if item["blocked_reasons"]:
        item["blocked_reasons"] = sorted(set(item["blocked_reasons"]))
        item["action"] = "BLOCK_ASR_WORKER_PACK_VERIFY"
        item["reason"] = ",".join(item["blocked_reasons"])
    return item


def validate_audio_rel_path(audio_rel_path: str, audio_path: Optional[Path], pack_root: Path) -> list[str]:
    blocked: list[str] = []
    rel = Path(audio_rel_path)
    if not audio_rel_path:
        return ["audio_rel_path_required"]
    if rel.is_absolute():
        blocked.append("audio_rel_path_must_be_relative")
    if ".." in rel.parts:
        blocked.append("audio_rel_path_must_not_traverse")
    if not rel.parts or rel.parts[0] != "audio":
        blocked.append("audio_rel_path_must_start_with_audio")
    if rel.suffix.lower() != ".mp3":
        blocked.append("unsupported_audio_extension")
    if audio_path is None:
        blocked.append("audio_path_unresolvable")
        return blocked
    if "stable_runtime" in audio_path.parts:
        blocked.append("audio_under_stable_runtime")
    if not path_is_relative_to(audio_path, pack_root):
        blocked.append("audio_outside_pack_root")
    if not audio_path.exists():
        blocked.append("audio_missing")
    elif not audio_path.is_file():
        blocked.append("audio_not_file")
    elif audio_path.stat().st_size <= 0:
        blocked.append("zero_size_audio")
    return blocked


def validate_planned_outputs(planned_outputs: Mapping[str, Any], pack_root: Path) -> list[str]:
    blocked: list[str] = []
    for key in REQUIRED_OUTPUT_KEYS:
        value = clean(planned_outputs.get(key))
        if not value:
            blocked.append(f"missing_planned_output_{key}")
            continue
        rel = Path(value)
        target = resolve_relative_pack_path(pack_root, value)
        if rel.is_absolute():
            blocked.append(f"planned_output_{key}_must_be_relative")
        if ".." in rel.parts:
            blocked.append(f"planned_output_{key}_must_not_traverse")
        if not rel.parts or rel.parts[0] != "outputs":
            blocked.append(f"planned_output_{key}_must_start_with_outputs")
        if target is None or not path_is_relative_to(target, pack_root):
            blocked.append(f"planned_output_{key}_outside_pack_root")
        if "stable_runtime" in rel.parts:
            blocked.append(f"planned_output_{key}_under_stable_runtime")
    extra = sorted(set(planned_outputs) - set(REQUIRED_OUTPUT_KEYS))
    for key in extra:
        blocked.append(f"unexpected_planned_output_{key}")
    return blocked


def validate_source_refs(source_refs: Any) -> list[str]:
    if not isinstance(source_refs, Mapping):
        return []
    blocked = []
    for key, value in source_refs.items():
        text = clean(value)
        if not text:
            continue
        if "stable_runtime" in Path(text).parts:
            blocked.append(f"source_ref_{key}_under_stable_runtime")
    return blocked


def apply_duplicate_blocks(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in items if clean(item.get("queue_item_id")))
    audio_counts = Counter(clean(item.get("audio_rel_path")) for item in items if clean(item.get("audio_rel_path")))
    result: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        blocked = list(item.get("blocked_reasons") or [])
        if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
            blocked.append("duplicate_queue_item_id")
        if audio_counts.get(clean(item.get("audio_rel_path")), 0) > 1:
            blocked.append("duplicate_audio_rel_path")
        if blocked:
            item["blocked_reasons"] = sorted(set(blocked))
            item["action"] = "BLOCK_ASR_WORKER_PACK_VERIFY"
            item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def audit_pack_files(pack_root: Path, items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    expected = {
        clean(item.get("audio_path"))
        for item in items
        if clean(item.get("audio_path"))
    }
    audio_dir = pack_root / "audio"
    actual_files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []
    actual = {str(path.resolve(strict=False)) for path in actual_files}
    missing_expected_files = sorted(path for path in expected if path not in actual)
    unreferenced_audio_files = sorted(path for path in actual if path not in expected)
    blocked_reasons = {
        "missing_expected_files": len(missing_expected_files),
        "unreferenced_audio_files": len(unreferenced_audio_files),
    }
    return {
        "pack_audio_files": len(actual_files),
        "pack_total_bytes": sum(path.stat().st_size for path in actual_files),
        "expected_audio_files": len(expected),
        "blocked": sum(blocked_reasons.values()),
        "blocked_reasons": blocked_reasons,
        "missing_expected_files": missing_expected_files[:100],
        "unreferenced_audio_files": unreferenced_audio_files[:100],
    }


def resolve_verify_paths(
    product_root: Path,
    pack_root: Path,
    pack_manifest_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "pack_root": pack_root.resolve(strict=False),
        "pack_manifest_path": pack_manifest_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_verify_paths(**paths)
    return paths


def guard_verify_paths(
    product_root: Path,
    pack_root: Path,
    pack_manifest_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR worker pack root", pack_root),
        ("ASR worker pack manifest", pack_manifest_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not pack_root.exists() or not pack_root.is_dir():
        raise FileNotFoundError(f"ASR worker pack root not found: {pack_root}")
    if not pack_manifest_path.exists() or not pack_manifest_path.is_file():
        raise FileNotFoundError(f"ASR worker pack manifest not found: {pack_manifest_path}")
    if not path_is_relative_to(pack_manifest_path, pack_root):
        raise ValueError(f"ASR worker pack manifest must stay under pack root: {pack_root}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR worker pack verify audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR worker pack verify audit must stay under product root: {product_root}")


def resolve_relative_pack_path(pack_root: Path, rel_path: str) -> Optional[Path]:
    text = clean(rel_path)
    if not text:
        return None
    return (pack_root / text).resolve(strict=False)


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
