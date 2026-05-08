from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE
from mango_mvp.productization.asr_execution_plan import ASR_EXECUTION_PLAN_SCHEMA_VERSION
from mango_mvp.productization.asr_scheduler_dry_run import DANGEROUS_HARD_GUARDS, load_json_object, mapping_or_empty, optional_int
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_EXECUTION_DRY_RUN_SCHEMA_VERSION = "asr_worker_execution_dry_run_v1"
WORKER_CONTRACT_VERSION = "asr_worker_command_envelope_v1"


@dataclass(frozen=True)
class AsrWorkerExecutionDryRunSummary:
    schema_version: str
    product_root: str
    execution_plan_path: str
    out_dir: str
    worker_plan_path: str
    execution_plan_sha256: str
    worker_plan_sha256: str
    approval_ref: Optional[str]
    source_items: int
    envelopes: int
    blocked_envelopes: int
    skipped_items: int
    technical_blocked: int
    total_audio_bytes: int
    total_duration_sec: float
    estimated_tmp_bytes: int
    estimated_timeout_sec: int
    dispatch_allowed: bool
    run_asr: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_execution_dry_run(
    product_root: Path,
    execution_plan_path: Path,
    out_dir: Path,
    worker_plan_path: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_worker_dry_run_paths(
        product_root=product_root,
        execution_plan_path=execution_plan_path,
        out_dir=out_dir,
        worker_plan_path=worker_plan_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    execution_plan_path = paths["execution_plan_path"]
    out_dir = paths["out_dir"]
    worker_plan_path = paths["worker_plan_path"]
    out_path = paths.get("out_path")

    execution_plan = load_json_object(execution_plan_path)
    execution_plan_sha = sha256_file(execution_plan_path)
    technical_reasons = validate_execution_plan_for_worker_dry_run(execution_plan)
    source_items = execution_plan.get("items") if isinstance(execution_plan.get("items"), SequenceABC) and not isinstance(execution_plan.get("items"), (str, bytes)) else []
    envelopes: list[Mapping[str, Any]] = []
    if not technical_reasons:
        envelopes = [build_worker_envelope(item, row_number=index) for index, item in enumerate(source_items, start=1)]
        envelopes = apply_duplicate_envelope_blocks(envelopes)
    worker_plan = build_worker_plan_payload(
        execution_plan_path=execution_plan_path,
        execution_plan=execution_plan,
        execution_plan_sha256=execution_plan_sha,
        technical_reasons=technical_reasons,
        envelopes=envelopes,
        source_items=len(source_items),
    )
    write_json(worker_plan_path, worker_plan)
    worker_plan_sha = sha256_file(worker_plan_path)
    action_counts = action_counts_for(envelopes) if envelopes else action_counts_for(worker_plan["actions"])
    blocked_envelopes = int(action_counts.get("BLOCK_ASR_WORKER_ENVELOPE") or 0)
    planned_envelopes = int(action_counts.get("PLAN_ASR_WORKER_ENVELOPE") or 0)
    skipped_items = int(action_counts.get("SKIP_ASR_WORKER_ENVELOPE") or 0)
    resource_totals = worker_plan["resource_estimate"]
    validation_ok = not technical_reasons and blocked_envelopes == 0 and planned_envelopes == len(source_items) and len(source_items) > 0
    summary = AsrWorkerExecutionDryRunSummary(
        schema_version=ASR_WORKER_EXECUTION_DRY_RUN_SCHEMA_VERSION,
        product_root=str(product_root),
        execution_plan_path=str(execution_plan_path),
        out_dir=str(out_dir),
        worker_plan_path=str(worker_plan_path),
        execution_plan_sha256=execution_plan_sha,
        worker_plan_sha256=worker_plan_sha,
        approval_ref=clean(execution_plan.get("approval_ref")) or None,
        source_items=len(source_items),
        envelopes=planned_envelopes,
        blocked_envelopes=blocked_envelopes,
        skipped_items=skipped_items,
        technical_blocked=len(set(technical_reasons)),
        total_audio_bytes=int(resource_totals["total_audio_bytes"]),
        total_duration_sec=float(resource_totals["total_duration_sec"]),
        estimated_tmp_bytes=int(resource_totals["estimated_tmp_bytes"]),
        estimated_timeout_sec=int(resource_totals["estimated_timeout_sec"]),
        dispatch_allowed=False,
        run_asr=False,
        validation_ok=validation_ok,
        warnings=count_warnings(envelopes),
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts,
        "actions": worker_plan["actions"],
        "worker_plan": worker_plan,
        "envelope_samples": envelopes[:20],
        "source_execution_plan": {
            "schema_version": clean(execution_plan.get("schema_version")) or None,
            "job_type": clean(execution_plan.get("job_type")) or None,
            "mode": clean(execution_plan.get("mode")) or None,
            "status": clean(execution_plan.get("status")) or None,
            "approval_ref": clean(execution_plan.get("approval_ref")) or None,
            "sha256": execution_plan_sha,
        },
        "safety": worker_dry_run_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def validate_execution_plan_for_worker_dry_run(execution_plan: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if clean(execution_plan.get("schema_version")) != ASR_EXECUTION_PLAN_SCHEMA_VERSION:
        reasons.append("execution_plan_schema_unexpected")
    if clean(execution_plan.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("execution_plan_job_type_unexpected")
    if clean(execution_plan.get("mode")) != "execution_plan_dry_run":
        reasons.append("execution_plan_mode_unexpected")
    if clean(execution_plan.get("status")) != "planned_not_dispatched":
        reasons.append("execution_plan_status_unexpected")
    if not clean(execution_plan.get("approval_ref")):
        reasons.append("execution_plan_approval_ref_missing")
    if bool(execution_plan.get("run_asr")):
        reasons.append("execution_plan_must_not_run_asr")
    if bool(execution_plan.get("scheduler_dispatch")):
        reasons.append("execution_plan_must_not_dispatch")
    if bool(execution_plan.get("execution_allowed")):
        reasons.append("execution_plan_must_not_allow_execution")
    hard_guards = mapping_or_empty(execution_plan.get("hard_guards"))
    if not hard_guards:
        reasons.append("execution_plan_hard_guards_missing")
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("scheduler_dispatch",):
        if bool(hard_guards.get(guard)):
            reasons.append(f"execution_plan_hard_guard_{guard}_must_be_false")
    next_stage = mapping_or_empty(execution_plan.get("next_stage_contract"))
    if not bool(next_stage.get("may_create_worker_execution_dry_run")):
        reasons.append("execution_plan_next_stage_worker_dry_run_not_allowed")
    if bool(next_stage.get("may_dispatch_worker_in_this_stage")):
        reasons.append("execution_plan_next_stage_must_not_dispatch_worker")
    if bool(next_stage.get("may_run_asr_in_this_stage")):
        reasons.append("execution_plan_next_stage_must_not_run_asr")
    workload = mapping_or_empty(execution_plan.get("workload"))
    if (optional_int(workload.get("planned_items")) or 0) <= 0:
        reasons.append("execution_plan_planned_items_empty")
    if optional_int(workload.get("blocked_items")):
        reasons.append("execution_plan_has_blocked_items")
    items = execution_plan.get("items")
    if not isinstance(items, SequenceABC) or isinstance(items, (str, bytes)):
        reasons.append("execution_plan_items_missing")
    elif len(items) != (optional_int(workload.get("planned_items")) or 0):
        reasons.append("execution_plan_items_do_not_match_workload")
    return sorted(set(reasons))


def build_worker_envelope(item: Mapping[str, Any], row_number: int) -> Mapping[str, Any]:
    audio_path = Path(clean(item.get("audio_path"))).resolve(strict=False)
    duration = optional_float(item.get("duration_sec")) or 0.0
    size_bytes = optional_int(item.get("size_bytes")) or 0
    resource = estimate_item_resources(duration_sec=duration, size_bytes=size_bytes)
    envelope: dict[str, Any] = {
        "action": "PLAN_ASR_WORKER_ENVELOPE",
        "reason": "execution_plan_item_ready_for_worker_dry_run_envelope",
        "row_number": row_number,
        "contract_version": WORKER_CONTRACT_VERSION,
        "queue_item_id": clean(item.get("queue_item_id")),
        "asset_id": optional_int(item.get("asset_id")),
        "tenant_id": clean(item.get("tenant_id")),
        "provider": clean(item.get("provider")),
        "event_key": clean(item.get("event_key")),
        "recording_id": clean(item.get("recording_id")),
        "audio": {
            "path": str(audio_path),
            "rel_path": clean(item.get("audio_rel_path")),
            "sha256": clean(item.get("audio_sha256")).lower(),
            "size_bytes": size_bytes,
            "duration_sec": duration,
        },
        "outputs": {
            "transcript_json": clean(mapping_or_empty(item.get("planned_output_paths")).get("transcript_json")) or None,
            "transcript_txt": clean(mapping_or_empty(item.get("planned_output_paths")).get("transcript_txt")) or None,
            "asr_audit_json": clean(mapping_or_empty(item.get("planned_output_paths")).get("asr_audit_json")) or None,
        },
        "worker_command_envelope": {
            "executable": None,
            "argv": [],
            "dry_run_only": True,
            "dispatch_allowed": False,
            "run_asr": False,
            "write_outputs": False,
            "write_runtime_db": False,
            "write_crm": False,
            "adapter_contract": "future_asr_worker_adapter",
        },
        "resource_estimate": resource,
        "blocked_reasons": [],
        "warnings": [],
    }
    if clean(item.get("action")) != "PLAN_ASR_EXECUTION_ITEM":
        envelope["blocked_reasons"].append("source_item_not_planned")
    for field in ("queue_item_id", "tenant_id", "provider", "event_key", "recording_id"):
        if not clean(envelope.get(field)):
            envelope["blocked_reasons"].append(f"missing_{field}")
    if not audio_path.exists():
        envelope["blocked_reasons"].append("audio_missing")
    elif not audio_path.is_file():
        envelope["blocked_reasons"].append("audio_not_file")
    elif audio_path.stat().st_size <= 0:
        envelope["blocked_reasons"].append("zero_size_audio")
    if bool(mapping_or_empty(item.get("execution")).get("run_asr")):
        envelope["blocked_reasons"].append("source_item_unexpectedly_allows_asr")
    if bool(mapping_or_empty(item.get("execution")).get("dispatch_allowed_in_stage18")):
        envelope["blocked_reasons"].append("source_item_unexpectedly_allows_dispatch")
    for key, value in envelope["outputs"].items():
        if not clean(value):
            envelope["blocked_reasons"].append(f"missing_output_{key}")
        elif "stable_runtime" in Path(clean(value)).parts:
            envelope["blocked_reasons"].append(f"output_{key}_under_stable_runtime")
    if envelope["blocked_reasons"]:
        envelope["blocked_reasons"] = sorted(set(envelope["blocked_reasons"]))
        envelope["action"] = "BLOCK_ASR_WORKER_ENVELOPE"
        envelope["reason"] = ",".join(envelope["blocked_reasons"])
    return envelope


def estimate_item_resources(duration_sec: float, size_bytes: int) -> Mapping[str, Any]:
    normalized_duration = max(float(duration_sec), 0.0)
    normalized_size = max(int(size_bytes), 0)
    timeout = max(60, int(normalized_duration * 4) + 45)
    tmp_bytes = normalized_size * 3 + 32768
    return {
        "audio_duration_sec": round(normalized_duration, 3),
        "audio_bytes": normalized_size,
        "estimated_tmp_bytes": tmp_bytes,
        "estimated_timeout_sec": timeout,
        "estimated_cpu_seconds_min": round(max(5.0, normalized_duration * 0.5), 3),
        "estimated_cpu_seconds_max": round(max(15.0, normalized_duration * 3.0), 3),
        "size_class": duration_size_class(normalized_duration),
    }


def duration_size_class(duration_sec: float) -> str:
    if duration_sec < 60:
        return "short"
    if duration_sec < 300:
        return "medium"
    return "long"


def apply_duplicate_envelope_blocks(envelopes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in envelopes if clean(item.get("queue_item_id")))
    output_counts = Counter(
        clean(value)
        for item in envelopes
        for value in mapping_or_empty(item.get("outputs")).values()
        if clean(value)
    )
    result: list[Mapping[str, Any]] = []
    for source in envelopes:
        item = dict(source)
        blocked = list(item.get("blocked_reasons") or [])
        if item["action"] == "PLAN_ASR_WORKER_ENVELOPE":
            if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
                blocked.append("duplicate_queue_item_id")
            duplicate_outputs = [
                clean(value)
                for value in mapping_or_empty(item.get("outputs")).values()
                if clean(value) and output_counts.get(clean(value), 0) > 1
            ]
            if duplicate_outputs:
                blocked.append("duplicate_output_path")
            if blocked:
                item["blocked_reasons"] = sorted(set(blocked))
                item["action"] = "BLOCK_ASR_WORKER_ENVELOPE"
                item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def build_worker_plan_payload(
    execution_plan_path: Path,
    execution_plan: Mapping[str, Any],
    execution_plan_sha256: str,
    technical_reasons: Sequence[str],
    envelopes: Sequence[Mapping[str, Any]],
    source_items: int,
) -> Mapping[str, Any]:
    blocked = sorted(set(technical_reasons))
    if blocked:
        status = "blocked_worker_dry_run_not_ready"
        actions: list[Mapping[str, Any]] = [
            {
                "action": "BLOCK_ASR_WORKER_DRY_RUN",
                "reason": "source_execution_plan_not_ready",
                "technical_reasons": blocked,
                "dispatch_allowed": False,
                "run_asr": False,
            }
        ]
    else:
        status = "worker_envelopes_planned_not_dispatched"
        actions = [
            {
                "action": "PLAN_ASR_WORKER_DRY_RUN_BATCH",
                "reason": "execution_plan_expanded_to_worker_envelopes_without_dispatch",
                "envelope_count": len(envelopes),
                "dispatch_allowed": False,
                "run_asr": False,
            }
        ]
    resource_estimate = aggregate_resource_estimates(envelopes)
    return {
        "schema_version": ASR_WORKER_EXECUTION_DRY_RUN_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "worker_execution_dry_run",
        "status": status,
        "approval_ref": clean(execution_plan.get("approval_ref")) or None,
        "dispatch_allowed": False,
        "run_asr": False,
        "execution_allowed": False,
        "write_outputs": False,
        "input_refs": {
            "execution_plan_path": str(execution_plan_path),
            "execution_plan_sha256": execution_plan_sha256,
            "approval_path": mapping_or_empty(execution_plan.get("input_refs")).get("approval_path"),
            "scheduler_plan_path": mapping_or_empty(execution_plan.get("input_refs")).get("scheduler_plan_path"),
            "pack_manifest_path": mapping_or_empty(execution_plan.get("input_refs")).get("pack_manifest_path"),
            "pack_manifest_sha256": mapping_or_empty(execution_plan.get("input_refs")).get("pack_manifest_sha256"),
        },
        "workload": {
            "source_items": source_items,
            "envelopes": len(envelopes),
            "blocked_envelopes": sum(1 for item in envelopes if item["action"] == "BLOCK_ASR_WORKER_ENVELOPE"),
            "planned_envelopes": sum(1 for item in envelopes if item["action"] == "PLAN_ASR_WORKER_ENVELOPE"),
        },
        "resource_estimate": resource_estimate,
        "actions": actions,
        "worker_envelopes": list(envelopes),
        "hard_guards": {
            "dispatch_worker": False,
            "download_audio": False,
            "copy_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_transcripts": False,
            "write_runtime_db": False,
            "write_product_db": False,
            "write_asset_db": False,
            "write_crm": False,
            "write_tallanto": False,
            "touch_stable_runtime": False,
        },
        "next_stage_contract": {
            "may_run_asr_in_this_stage": False,
            "may_dispatch_worker_in_this_stage": False,
            "may_create_worker_sandbox_plan": True,
            "must_choose_asr_engine_before_execution": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
        },
    }


def aggregate_resource_estimates(envelopes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    planned = [item for item in envelopes if item.get("action") == "PLAN_ASR_WORKER_ENVELOPE"]
    estimates = [mapping_or_empty(item.get("resource_estimate")) for item in planned]
    total_duration = sum(float(estimate.get("audio_duration_sec") or 0.0) for estimate in estimates)
    total_audio_bytes = sum(int(estimate.get("audio_bytes") or 0) for estimate in estimates)
    total_tmp = sum(int(estimate.get("estimated_tmp_bytes") or 0) for estimate in estimates)
    timeout = sum(int(estimate.get("estimated_timeout_sec") or 0) for estimate in estimates)
    size_counts = Counter(clean(estimate.get("size_class")) for estimate in estimates if clean(estimate.get("size_class")))
    return {
        "planned_envelopes": len(planned),
        "total_duration_sec": round(total_duration, 3),
        "total_audio_bytes": total_audio_bytes,
        "estimated_tmp_bytes": total_tmp,
        "estimated_timeout_sec": timeout,
        "estimated_cpu_seconds_min": round(sum(float(estimate.get("estimated_cpu_seconds_min") or 0.0) for estimate in estimates), 3),
        "estimated_cpu_seconds_max": round(sum(float(estimate.get("estimated_cpu_seconds_max") or 0.0) for estimate in estimates), 3),
        "size_class_counts": dict(sorted(size_counts.items())),
    }


def resolve_worker_dry_run_paths(
    product_root: Path,
    execution_plan_path: Path,
    out_dir: Path,
    worker_plan_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "execution_plan_path": execution_plan_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "worker_plan_path": worker_plan_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_worker_dry_run_paths(**paths)
    return paths


def guard_worker_dry_run_paths(
    product_root: Path,
    execution_plan_path: Path,
    out_dir: Path,
    worker_plan_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR execution plan", execution_plan_path),
        ("ASR worker dry-run output directory", out_dir),
        ("ASR worker dry-run plan", worker_plan_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not execution_plan_path.exists() or not execution_plan_path.is_file():
        raise FileNotFoundError(f"ASR execution plan not found: {execution_plan_path}")
    if not path_is_relative_to(worker_plan_path, out_dir):
        raise ValueError(f"ASR worker dry-run plan must stay under output directory: {out_dir}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR worker dry-run audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR worker dry-run audit must stay under product root: {product_root}")
        if not path_is_relative_to(out_path, out_dir):
            raise ValueError(f"ASR worker dry-run audit must stay under output directory: {out_dir}")


def worker_dry_run_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "reads_execution_plan": True,
        "writes_worker_plan": True,
        "product_db_writes": False,
        "asset_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "copies_audio": False,
        "hardlinks_audio": False,
        "dispatch_worker": False,
        "run_asr": False,
        "run_ra": False,
        "write_transcripts": False,
        "write_crm": False,
        "write_tallanto": False,
    }


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
