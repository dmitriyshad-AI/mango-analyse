from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE
from mango_mvp.productization.asr_scheduler_dry_run import (
    ASR_SCHEDULER_DRY_RUN_SCHEMA_VERSION,
    DANGEROUS_HARD_GUARDS,
    load_json_object,
    mapping_or_empty,
    optional_int,
    scheduler_dry_run_safety,
)
from mango_mvp.productization.asr_worker_pack import ASR_WORKER_PACK_SCHEMA_VERSION, read_jsonl
from mango_mvp.productization.asr_worker_pack_verifier import REQUIRED_OUTPUT_KEYS
from mango_mvp.productization.processing_handoff import ASR_HANDOFF_STATUS
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_EXECUTION_PLAN_SCHEMA_VERSION = "asr_execution_plan_v1"


@dataclass(frozen=True)
class AsrExecutionPlanSummary:
    schema_version: str
    product_root: str
    scheduler_plan_path: str
    out_dir: str
    execution_plan_path: str
    scheduler_plan_sha256: str
    pack_manifest_path: Optional[str]
    pack_manifest_sha256: Optional[str]
    approval_ref: Optional[str]
    manifest_rows: int
    planned_items: int
    blocked_items: int
    skipped_items: int
    technical_blocked: int
    warnings: int
    execution_allowed: bool
    scheduler_dispatch: bool
    run_asr: bool
    execution_plan_sha256: str
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_execution_plan(
    product_root: Path,
    scheduler_plan_path: Path,
    out_dir: Path,
    execution_plan_path: Path,
    out_path: Optional[Path] = None,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    paths = resolve_execution_plan_paths(
        product_root=product_root,
        scheduler_plan_path=scheduler_plan_path,
        out_dir=out_dir,
        execution_plan_path=execution_plan_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    scheduler_plan_path = paths["scheduler_plan_path"]
    out_dir = paths["out_dir"]
    execution_plan_path = paths["execution_plan_path"]
    out_path = paths.get("out_path")

    scheduler_plan = load_json_object(scheduler_plan_path)
    scheduler_plan_sha = sha256_file(scheduler_plan_path)
    scheduler_reasons = validate_approved_scheduler_plan(scheduler_plan)
    pack_refs = resolve_pack_refs(product_root=product_root, scheduler_plan=scheduler_plan)
    technical_reasons = list(scheduler_reasons) + list(pack_refs["reasons"])
    rows: list[Mapping[str, Any]] = []
    items: list[Mapping[str, Any]] = []
    manifest_sha: Optional[str] = None
    if not technical_reasons:
        pack_manifest_path = pack_refs["pack_manifest_path"]
        pack_root = pack_refs["pack_root"]
        rows = read_jsonl(pack_manifest_path)
        manifest_sha = sha256_file(pack_manifest_path)
        expected_sha = clean(mapping_or_empty(scheduler_plan.get("input_refs")).get("pack_manifest_sha256"))
        if manifest_sha != expected_sha:
            technical_reasons.append("pack_manifest_sha_mismatch")
        else:
            items = [
                plan_execution_item(row, row_number=index, pack_root=pack_root, verify_checksum=verify_checksum)
                for index, row in enumerate(rows, start=1)
            ]
            items = apply_duplicate_execution_blocks(items)
    execution_plan = build_execution_plan_payload(
        scheduler_plan_path=scheduler_plan_path,
        scheduler_plan=scheduler_plan,
        scheduler_plan_sha256=scheduler_plan_sha,
        manifest_sha256=manifest_sha or clean(mapping_or_empty(scheduler_plan.get("input_refs")).get("pack_manifest_sha256")) or None,
        technical_reasons=sorted(set(technical_reasons)),
        items=items,
        manifest_rows=len(rows),
    )
    write_json(execution_plan_path, execution_plan)
    execution_plan_sha = sha256_file(execution_plan_path)
    action_counts = action_counts_for(items) if items else action_counts_for(execution_plan["actions"])
    blocked_items = int(action_counts.get("BLOCK_ASR_EXECUTION_PLAN_ITEM") or 0)
    planned_items = int(action_counts.get("PLAN_ASR_EXECUTION_ITEM") or 0)
    skipped_items = int(action_counts.get("SKIP_ASR_EXECUTION_ITEM") or 0)
    warnings = count_warnings(items)
    validation_ok = not technical_reasons and blocked_items == 0 and planned_items == len(rows) and len(rows) > 0
    summary = AsrExecutionPlanSummary(
        schema_version=ASR_EXECUTION_PLAN_SCHEMA_VERSION,
        product_root=str(product_root),
        scheduler_plan_path=str(scheduler_plan_path),
        out_dir=str(out_dir),
        execution_plan_path=str(execution_plan_path),
        scheduler_plan_sha256=scheduler_plan_sha,
        pack_manifest_path=str(pack_refs.get("pack_manifest_path")) if pack_refs.get("pack_manifest_path") else None,
        pack_manifest_sha256=execution_plan["input_refs"]["pack_manifest_sha256"],
        approval_ref=clean(scheduler_plan.get("approval_ref")) or None,
        manifest_rows=len(rows),
        planned_items=planned_items,
        blocked_items=blocked_items,
        skipped_items=skipped_items,
        technical_blocked=len(set(technical_reasons)),
        warnings=warnings,
        execution_allowed=False,
        scheduler_dispatch=False,
        run_asr=False,
        execution_plan_sha256=execution_plan_sha,
        validation_ok=validation_ok,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts,
        "actions": execution_plan["actions"],
        "execution_plan": execution_plan,
        "item_samples": items[:20],
        "source_scheduler_plan": {
            "schema_version": clean(scheduler_plan.get("schema_version")) or None,
            "job_type": clean(scheduler_plan.get("job_type")) or None,
            "mode": clean(scheduler_plan.get("mode")) or None,
            "status": clean(scheduler_plan.get("status")) or None,
            "approval_ref": clean(scheduler_plan.get("approval_ref")) or None,
            "sha256": scheduler_plan_sha,
        },
        "safety": execution_plan_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def validate_approved_scheduler_plan(scheduler_plan: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if clean(scheduler_plan.get("schema_version")) != ASR_SCHEDULER_DRY_RUN_SCHEMA_VERSION:
        reasons.append("scheduler_plan_schema_unexpected")
    if clean(scheduler_plan.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("scheduler_plan_job_type_unexpected")
    if clean(scheduler_plan.get("mode")) != "scheduler_approval_dry_run":
        reasons.append("scheduler_plan_mode_unexpected")
    if clean(scheduler_plan.get("status")) != "approved_dry_run_not_dispatched":
        reasons.append("scheduler_plan_not_approved_dry_run")
    if bool(scheduler_plan.get("scheduler_may_dispatch")):
        reasons.append("scheduler_plan_must_not_dispatch")
    if bool(scheduler_plan.get("execution_allowed")):
        reasons.append("scheduler_plan_must_not_allow_execution")
    if not clean(scheduler_plan.get("approval_ref")):
        reasons.append("scheduler_plan_approval_ref_missing")
    approval_gate = mapping_or_empty(scheduler_plan.get("approval_gate"))
    if not bool(approval_gate.get("approval_present")):
        reasons.append("scheduler_plan_approval_missing")
    if not bool(approval_gate.get("approval_valid")):
        reasons.append("scheduler_plan_approval_invalid")
    if approval_gate.get("approval_reasons"):
        reasons.append("scheduler_plan_approval_reasons_present")
    next_stage = mapping_or_empty(scheduler_plan.get("next_stage_contract"))
    if not bool(next_stage.get("may_create_execution_plan_after_valid_approval")):
        reasons.append("scheduler_plan_next_stage_execution_plan_not_allowed")
    if bool(next_stage.get("may_dispatch_asr_in_this_stage")):
        reasons.append("scheduler_plan_next_stage_must_not_dispatch_asr")
    hard_guards = mapping_or_empty(scheduler_plan.get("hard_guards"))
    if not hard_guards:
        reasons.append("scheduler_plan_hard_guards_missing")
    for guard in DANGEROUS_HARD_GUARDS:
        if bool(hard_guards.get(guard)):
            reasons.append(f"scheduler_plan_hard_guard_{guard}_must_be_false")
    workload = mapping_or_empty(scheduler_plan.get("workload"))
    if (optional_int(workload.get("ready_items")) or 0) <= 0:
        reasons.append("scheduler_plan_ready_items_empty")
    if (optional_int(workload.get("manifest_rows")) or 0) != (optional_int(workload.get("ready_items")) or 0):
        reasons.append("scheduler_plan_ready_items_do_not_match_manifest")
    return sorted(set(reasons))


def resolve_pack_refs(product_root: Path, scheduler_plan: Mapping[str, Any]) -> Mapping[str, Any]:
    input_refs = mapping_or_empty(scheduler_plan.get("input_refs"))
    reasons: list[str] = []
    pack_root_text = clean(input_refs.get("pack_root"))
    pack_manifest_text = clean(input_refs.get("pack_manifest_path"))
    if not pack_root_text:
        reasons.append("scheduler_plan_pack_root_missing")
    if not pack_manifest_text:
        reasons.append("scheduler_plan_pack_manifest_path_missing")
    pack_root = Path(pack_root_text).resolve(strict=False) if pack_root_text else None
    pack_manifest_path = Path(pack_manifest_text).resolve(strict=False) if pack_manifest_text else None
    for label, path in (("pack_root", pack_root), ("pack_manifest_path", pack_manifest_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            reasons.append(f"{label}_under_stable_runtime")
        if not path_is_relative_to(path, product_root):
            reasons.append(f"{label}_outside_product_root")
    if pack_root and pack_manifest_path and not path_is_relative_to(pack_manifest_path, pack_root):
        reasons.append("pack_manifest_outside_pack_root")
    if pack_root and not pack_root.exists():
        reasons.append("pack_root_missing")
    if pack_manifest_path and (not pack_manifest_path.exists() or not pack_manifest_path.is_file()):
        reasons.append("pack_manifest_missing")
    if not clean(input_refs.get("pack_manifest_sha256")):
        reasons.append("pack_manifest_sha_missing")
    return {
        "pack_root": pack_root,
        "pack_manifest_path": pack_manifest_path,
        "reasons": sorted(set(reasons)),
    }


def plan_execution_item(
    row: Mapping[str, Any],
    row_number: int,
    pack_root: Path,
    verify_checksum: bool,
) -> Mapping[str, Any]:
    audio_rel_path = clean(row.get("audio_rel_path"))
    audio_path = resolve_relative_pack_path(pack_root, audio_rel_path)
    planned_outputs = mapping_or_empty(row.get("planned_outputs_rel"))
    output_paths = {
        key: str(target) if (target := resolve_relative_pack_path(pack_root, clean(planned_outputs.get(key)))) else None
        for key in REQUIRED_OUTPUT_KEYS
    }
    item: dict[str, Any] = {
        "action": "PLAN_ASR_EXECUTION_ITEM",
        "reason": "ready_for_asr_execution_plan",
        "row_number": row_number,
        "schema_version": clean(row.get("schema_version")),
        "queue_status": clean(row.get("queue_status")),
        "queue_item_id": clean(row.get("queue_item_id")),
        "asset_id": optional_int(row.get("asset_id")),
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")),
        "event_key": clean(row.get("event_key")),
        "provider_call_id": clean(row.get("provider_call_id")),
        "recording_id": clean(row.get("recording_id")),
        "started_at": clean(row.get("started_at")) or None,
        "direction": clean(row.get("direction")) or None,
        "client_phone": clean(row.get("client_phone")) or None,
        "manager_ref": clean(row.get("manager_ref")) or None,
        "manager_name": clean(row.get("manager_name")) or None,
        "audio_rel_path": audio_rel_path,
        "audio_path": str(audio_path) if audio_path else None,
        "audio_sha256": clean(row.get("audio_sha256")).lower(),
        "size_bytes": optional_int(row.get("size_bytes")),
        "duration_sec": optional_float(row.get("duration_sec")),
        "planned_outputs_rel": dict(planned_outputs),
        "planned_output_paths": output_paths,
        "blocked_reasons": [],
        "warnings": [],
        "execution": {
            "run_asr": False,
            "write_runtime_db": False,
            "write_crm": False,
            "dispatch_allowed_in_stage18": False,
        },
    }
    if item["schema_version"] != ASR_WORKER_PACK_SCHEMA_VERSION:
        item["blocked_reasons"].append("unexpected_manifest_schema")
    if item["queue_status"] != ASR_HANDOFF_STATUS:
        item["blocked_reasons"].append("unexpected_queue_status")
    for field in ("queue_item_id", "tenant_id", "provider", "event_key", "recording_id", "audio_rel_path", "audio_sha256"):
        if not clean(item.get(field)):
            item["blocked_reasons"].append(f"missing_{field}")
    item["blocked_reasons"].extend(validate_audio_ref(audio_rel_path=audio_rel_path, audio_path=audio_path, pack_root=pack_root))
    item["blocked_reasons"].extend(validate_output_refs(planned_outputs, pack_root=pack_root))
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
        item["action"] = "BLOCK_ASR_EXECUTION_PLAN_ITEM"
        item["reason"] = ",".join(item["blocked_reasons"])
    return item


def validate_audio_ref(audio_rel_path: str, audio_path: Optional[Path], pack_root: Path) -> list[str]:
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


def validate_output_refs(planned_outputs: Mapping[str, Any], pack_root: Path) -> list[str]:
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
        if target and "stable_runtime" in target.parts:
            blocked.append(f"planned_output_{key}_under_stable_runtime")
    return blocked


def apply_duplicate_execution_blocks(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in items if clean(item.get("queue_item_id")))
    audio_counts = Counter(clean(item.get("audio_rel_path")) for item in items if clean(item.get("audio_rel_path")))
    output_counts = Counter(
        clean(value)
        for item in items
        for value in mapping_or_empty(item.get("planned_outputs_rel")).values()
        if clean(value)
    )
    result: list[Mapping[str, Any]] = []
    for source in items:
        item = dict(source)
        blocked = list(item.get("blocked_reasons") or [])
        if item["action"] == "PLAN_ASR_EXECUTION_ITEM":
            if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
                blocked.append("duplicate_queue_item_id")
            if audio_counts.get(clean(item.get("audio_rel_path")), 0) > 1:
                blocked.append("duplicate_audio_rel_path")
            duplicates = [
                clean(value)
                for value in mapping_or_empty(item.get("planned_outputs_rel")).values()
                if clean(value) and output_counts.get(clean(value), 0) > 1
            ]
            if duplicates:
                blocked.append("duplicate_planned_output_rel_path")
            if blocked:
                item["blocked_reasons"] = sorted(set(blocked))
                item["action"] = "BLOCK_ASR_EXECUTION_PLAN_ITEM"
                item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def build_execution_plan_payload(
    scheduler_plan_path: Path,
    scheduler_plan: Mapping[str, Any],
    scheduler_plan_sha256: str,
    manifest_sha256: Optional[str],
    technical_reasons: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    manifest_rows: int,
) -> Mapping[str, Any]:
    input_refs = mapping_or_empty(scheduler_plan.get("input_refs"))
    blocked = sorted(set(technical_reasons))
    actions: list[Mapping[str, Any]]
    if blocked:
        status = "blocked_execution_plan_not_ready"
        actions = [
            {
                "action": "BLOCK_ASR_EXECUTION_PLAN",
                "reason": "execution_plan_source_not_ready",
                "technical_reasons": blocked,
                "run_asr": False,
                "execution_allowed": False,
            }
        ]
    else:
        status = "planned_not_dispatched"
        actions = [
            {
                "action": "PLAN_ASR_EXECUTION_BATCH_DRY_RUN",
                "reason": "approved_scheduler_plan_expanded_to_items_without_dispatch",
                "item_count": len(items),
                "run_asr": False,
                "execution_allowed": False,
            }
        ]
    return {
        "schema_version": ASR_EXECUTION_PLAN_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "execution_plan_dry_run",
        "status": status,
        "approval_ref": clean(scheduler_plan.get("approval_ref")) or None,
        "run_asr": False,
        "scheduler_dispatch": False,
        "execution_allowed": False,
        "input_refs": {
            "scheduler_plan_path": str(scheduler_plan_path),
            "scheduler_plan_sha256": scheduler_plan_sha256,
            "approved_scheduler_status": clean(scheduler_plan.get("status")) or None,
            "approval_path": mapping_or_empty(scheduler_plan.get("approval_gate")).get("approval_path"),
            "job_plan_path": clean(input_refs.get("job_plan_path")) or None,
            "job_plan_sha256": clean(input_refs.get("job_plan_sha256")) or None,
            "pack_root": clean(input_refs.get("pack_root")) or None,
            "pack_manifest_path": clean(input_refs.get("pack_manifest_path")) or None,
            "pack_manifest_sha256": manifest_sha256,
        },
        "workload": {
            "manifest_rows": manifest_rows,
            "planned_items": len(items),
            "blocked_items": sum(1 for item in items if item["action"] == "BLOCK_ASR_EXECUTION_PLAN_ITEM"),
            "ready_items": sum(1 for item in items if item["action"] == "PLAN_ASR_EXECUTION_ITEM"),
            "pack_audio_files": optional_int(mapping_or_empty(scheduler_plan.get("workload")).get("pack_audio_files")) or 0,
            "pack_total_bytes": optional_int(mapping_or_empty(scheduler_plan.get("workload")).get("pack_total_bytes")) or 0,
        },
        "actions": actions,
        "items": list(items),
        "hard_guards": {
            "download_audio": False,
            "copy_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_runtime_db": False,
            "write_product_db": False,
            "write_asset_db": False,
            "write_crm": False,
            "write_tallanto": False,
            "touch_stable_runtime": False,
            "scheduler_dispatch": False,
        },
        "next_stage_contract": {
            "may_run_asr_in_this_stage": False,
            "may_dispatch_worker_in_this_stage": False,
            "may_create_worker_execution_dry_run": True,
            "must_verify_audio_before_execution": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
        },
    }


def resolve_execution_plan_paths(
    product_root: Path,
    scheduler_plan_path: Path,
    out_dir: Path,
    execution_plan_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "scheduler_plan_path": scheduler_plan_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "execution_plan_path": execution_plan_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_execution_plan_paths(**paths)
    return paths


def guard_execution_plan_paths(
    product_root: Path,
    scheduler_plan_path: Path,
    out_dir: Path,
    execution_plan_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR scheduler plan", scheduler_plan_path),
        ("ASR execution plan output directory", out_dir),
        ("ASR execution plan", execution_plan_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not scheduler_plan_path.exists() or not scheduler_plan_path.is_file():
        raise FileNotFoundError(f"ASR scheduler plan not found: {scheduler_plan_path}")
    if not path_is_relative_to(execution_plan_path, out_dir):
        raise ValueError(f"ASR execution plan must stay under output directory: {out_dir}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR execution plan audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR execution plan audit must stay under product root: {product_root}")
        if not path_is_relative_to(out_path, out_dir):
            raise ValueError(f"ASR execution plan audit must stay under output directory: {out_dir}")


def execution_plan_safety() -> Mapping[str, bool]:
    safety = dict(scheduler_dry_run_safety())
    safety.update(
        {
            "reads_worker_pack_manifest": True,
            "writes_execution_plan": True,
            "scheduler_dispatch": False,
            "run_asr": False,
        }
    )
    return safety


def resolve_relative_pack_path(pack_root: Path, value: str) -> Optional[Path]:
    text = clean(value)
    if not text:
        return None
    return (pack_root / text).resolve(strict=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(item.get("action")) for item in items).items()))


def count_warnings(items: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(item.get("warnings") or []) for item in items)


def optional_float(value: Any) -> Optional[float]:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
