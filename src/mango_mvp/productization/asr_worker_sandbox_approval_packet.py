from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE
from mango_mvp.productization.asr_scheduler_dry_run import DANGEROUS_HARD_GUARDS, load_json_object, mapping_or_empty, optional_int
from mango_mvp.productization.asr_worker_sandbox_execution_contract import (
    ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION,
)
from mango_mvp.productization.asr_worker_sandbox_preflight import ASR_WORKER_SANDBOX_PREFLIGHT_SCHEMA_VERSION
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_SANDBOX_APPROVAL_PACKET_SCHEMA_VERSION = "asr_worker_sandbox_approval_packet_v1"
APPROVAL_PACKET_REF_PREFIX = "stage23-pending-approval"
REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS = (
    "explicit_asr_sandbox_execution",
    "selected_engine_acknowledged",
    "audio_sha_preflight_acknowledged",
    "expected_outputs_acknowledged",
    "resource_limits_acknowledged",
    "sandbox_output_root_acknowledged",
    "no_runtime_db_writes",
    "no_crm_or_tallanto_writes",
    "stable_runtime_not_touched",
    "human_operator_accepts_execution_risk",
)
REQUIRED_APPROVAL_PHRASE = "APPROVE_ASR_SANDBOX_EXECUTION_STAGE23"


@dataclass(frozen=True)
class AsrWorkerSandboxApprovalPacketSummary:
    schema_version: str
    product_root: str
    preflight_report_path: str
    contract_path: str
    out_dir: str
    approval_packet_path: str
    preflight_report_sha256: str
    contract_sha256: str
    approval_packet_sha256: str
    approval_packet_ref: str
    selected_engine: Optional[str]
    tasks: int
    audio_files_checked: int
    audio_sha_ok: int
    output_collisions: int
    required_acknowledgements: int
    approval_required: bool
    execution_approved: bool
    dispatch_allowed: bool
    run_asr: bool
    write_transcripts: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_sandbox_approval_packet(
    product_root: Path,
    preflight_report_path: Path,
    out_dir: Path,
    approval_packet_path: Path,
    out_path: Optional[Path] = None,
    contract_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_approval_packet_paths(
        product_root=product_root,
        preflight_report_path=preflight_report_path,
        out_dir=out_dir,
        approval_packet_path=approval_packet_path,
        out_path=out_path,
        contract_path=contract_path,
    )
    product_root = paths["product_root"]
    preflight_report_path = paths["preflight_report_path"]
    out_dir = paths["out_dir"]
    approval_packet_path = paths["approval_packet_path"]
    out_path = paths.get("out_path")

    preflight_report = load_json_object(preflight_report_path)
    preflight_sha = sha256_file(preflight_report_path)
    contract_path = paths.get("contract_path") or infer_contract_path(preflight_report=preflight_report, product_root=product_root)
    guard_inferred_contract_path(product_root=product_root, contract_path=contract_path)
    contract = load_json_object(contract_path)
    contract_sha = sha256_file(contract_path)
    source_reasons = validate_preflight_for_approval_packet(preflight_report, preflight_sha, contract_path, contract_sha)
    source_reasons.extend(validate_contract_for_approval_packet(contract, contract_sha, preflight_report))
    source_reasons = sorted(set(source_reasons))
    items: list[Mapping[str, Any]] = []
    if not source_reasons:
        items = build_approval_items(preflight_report=preflight_report, contract=contract)
        item_reasons = validate_approval_items(items)
        if item_reasons:
            source_reasons = sorted(set(item_reasons))
    actions = build_approval_packet_actions(source_reasons=source_reasons, items=items)
    packet_ref = f"{APPROVAL_PACKET_REF_PREFIX}-{preflight_sha[:16]}"
    approval_packet = build_approval_packet_payload(
        preflight_report_path=preflight_report_path,
        preflight_report=preflight_report,
        preflight_report_sha256=preflight_sha,
        contract_path=contract_path,
        contract=contract,
        contract_sha256=contract_sha,
        packet_ref=packet_ref,
        source_reasons=source_reasons,
        items=items,
        actions=actions,
    )
    write_json(approval_packet_path, approval_packet)
    packet_sha = sha256_file(approval_packet_path)
    workload = mapping_or_empty(approval_packet.get("workload"))
    validation_ok = not source_reasons and bool(approval_packet.get("approval_required")) and not bool(approval_packet.get("execution_approved"))
    summary = AsrWorkerSandboxApprovalPacketSummary(
        schema_version=ASR_WORKER_SANDBOX_APPROVAL_PACKET_SCHEMA_VERSION,
        product_root=str(product_root),
        preflight_report_path=str(preflight_report_path),
        contract_path=str(contract_path),
        out_dir=str(out_dir),
        approval_packet_path=str(approval_packet_path),
        preflight_report_sha256=preflight_sha,
        contract_sha256=contract_sha,
        approval_packet_sha256=packet_sha,
        approval_packet_ref=packet_ref,
        selected_engine=clean(approval_packet.get("selected_engine")) or None,
        tasks=optional_int(workload.get("tasks")) or 0,
        audio_files_checked=optional_int(workload.get("audio_files_checked")) or 0,
        audio_sha_ok=optional_int(workload.get("audio_sha_ok")) or 0,
        output_collisions=optional_int(workload.get("output_collisions")) or 0,
        required_acknowledgements=len(REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS),
        approval_required=True,
        execution_approved=False,
        dispatch_allowed=False,
        run_asr=False,
        write_transcripts=False,
        validation_ok=validation_ok,
        warnings=0 if validation_ok else 1,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts_for(actions),
        "actions": actions,
        "approval_packet": approval_packet,
        "item_samples": items[:20],
        "source_preflight_report": {
            "schema_version": clean(preflight_report.get("schema_version")) or None,
            "status": clean(preflight_report.get("status")) or None,
            "selected_engine": clean(preflight_report.get("selected_engine")) or None,
            "sha256": preflight_sha,
        },
        "source_contract": {
            "schema_version": clean(contract.get("schema_version")) or None,
            "status": clean(contract.get("status")) or None,
            "selected_engine": clean(contract.get("selected_engine")) or None,
            "sha256": contract_sha,
        },
        "safety": approval_packet_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def validate_preflight_for_approval_packet(
    preflight_report: Mapping[str, Any],
    preflight_sha256: str,
    contract_path: Path,
    contract_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    if clean(preflight_report.get("schema_version")) != ASR_WORKER_SANDBOX_PREFLIGHT_SCHEMA_VERSION:
        reasons.append("preflight_schema_unexpected")
    if clean(preflight_report.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("preflight_job_type_unexpected")
    if clean(preflight_report.get("mode")) != "sandbox_final_preflight_dry_run":
        reasons.append("preflight_mode_unexpected")
    if clean(preflight_report.get("status")) != "preflight_passed_not_dispatched":
        reasons.append("preflight_status_unexpected")
    if not bool(preflight_report.get("preflight_ready")):
        reasons.append("preflight_not_ready")
    if not clean(preflight_report.get("selected_engine")):
        reasons.append("preflight_selected_engine_missing")
    for key in ("dispatch_allowed", "run_asr", "execution_allowed", "write_outputs", "write_transcripts", "write_runtime_db", "write_crm"):
        if bool(preflight_report.get(key)):
            reasons.append(f"preflight_{key}_must_be_false")
    input_refs = mapping_or_empty(preflight_report.get("input_refs"))
    if clean(input_refs.get("contract_path")) != str(contract_path):
        reasons.append("preflight_contract_path_mismatch")
    if clean(input_refs.get("contract_sha256")) != contract_sha256:
        reasons.append("preflight_contract_sha_mismatch")
    workload = mapping_or_empty(preflight_report.get("workload"))
    tasks = optional_int(workload.get("tasks")) or 0
    passed = optional_int(workload.get("passed_tasks")) or 0
    blocked = optional_int(workload.get("blocked_tasks")) or 0
    audio_checked = optional_int(workload.get("audio_files_checked")) or 0
    audio_sha_ok = optional_int(workload.get("audio_sha_ok")) or 0
    output_collisions = optional_int(workload.get("output_collisions")) or 0
    if tasks <= 0:
        reasons.append("preflight_tasks_empty")
    if passed != tasks:
        reasons.append("preflight_passed_tasks_do_not_match_tasks")
    if blocked:
        reasons.append("preflight_has_blocked_tasks")
    if audio_checked != tasks:
        reasons.append("preflight_audio_files_checked_do_not_match_tasks")
    if audio_sha_ok != tasks:
        reasons.append("preflight_audio_sha_ok_do_not_match_tasks")
    if output_collisions:
        reasons.append("preflight_has_output_collisions")
    for section_name in ("engine_preflight", "disk_preflight", "directory_preflight"):
        if not bool(mapping_or_empty(preflight_report.get(section_name)).get("ok")):
            reasons.append(f"preflight_{section_name}_not_ok")
    directory = mapping_or_empty(preflight_report.get("directory_preflight"))
    if bool(directory.get("creates_dirs")):
        reasons.append("preflight_directory_must_not_create_dirs")
    hard_guards = mapping_or_empty(preflight_report.get("hard_guards"))
    if not hard_guards:
        reasons.append("preflight_hard_guards_missing")
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("dispatch_worker", "load_asr_model", "write_transcripts"):
        if bool(hard_guards.get(guard)):
            reasons.append(f"preflight_hard_guard_{guard}_must_be_false")
    next_stage = mapping_or_empty(preflight_report.get("next_stage_contract"))
    if not bool(next_stage.get("may_request_explicit_asr_execution_approval")):
        reasons.append("preflight_next_stage_approval_not_allowed")
    if bool(next_stage.get("may_dispatch_worker_in_this_stage")):
        reasons.append("preflight_next_stage_must_not_dispatch_worker")
    if bool(next_stage.get("may_run_asr_in_this_stage")):
        reasons.append("preflight_next_stage_must_not_run_asr")
    if not preflight_sha256:
        reasons.append("preflight_sha_missing")
    return sorted(set(reasons))


def validate_contract_for_approval_packet(
    contract: Mapping[str, Any],
    contract_sha256: str,
    preflight_report: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if clean(contract.get("schema_version")) != ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION:
        reasons.append("contract_schema_unexpected")
    if clean(contract.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("contract_job_type_unexpected")
    if clean(contract.get("mode")) != "sandbox_execution_contract_dry_run":
        reasons.append("contract_mode_unexpected")
    if clean(contract.get("status")) != "sandbox_execution_contract_planned_not_dispatched":
        reasons.append("contract_status_unexpected")
    if clean(contract.get("selected_engine")) != clean(preflight_report.get("selected_engine")):
        reasons.append("contract_selected_engine_mismatch")
    for key in ("dispatch_allowed", "run_asr", "execution_allowed", "write_outputs", "write_transcripts", "write_runtime_db", "write_crm"):
        if bool(contract.get(key)):
            reasons.append(f"contract_{key}_must_be_false")
    workload = mapping_or_empty(contract.get("workload"))
    if (optional_int(workload.get("planned_tasks")) or 0) != (optional_int(mapping_or_empty(preflight_report.get("workload")).get("tasks")) or 0):
        reasons.append("contract_planned_tasks_mismatch")
    if optional_int(workload.get("blocked_tasks")):
        reasons.append("contract_has_blocked_tasks")
    hard_guards = mapping_or_empty(contract.get("hard_guards"))
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("dispatch_worker", "load_asr_model", "write_transcripts"):
        if bool(hard_guards.get(guard)):
            reasons.append(f"contract_hard_guard_{guard}_must_be_false")
    if not contract_sha256:
        reasons.append("contract_sha_missing")
    return sorted(set(reasons))


def build_approval_items(preflight_report: Mapping[str, Any], contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    contract_tasks = {
        (optional_int(task.get("row_number")) or 0, clean(task.get("queue_item_id"))): task
        for task in contract.get("tasks", [])
        if isinstance(task, MappingABC)
    }
    items: list[Mapping[str, Any]] = []
    for row_number, preflight_item in enumerate(preflight_report.get("task_preflights", []), start=1):
        if not isinstance(preflight_item, MappingABC):
            items.append(
                {
                    "action": "BLOCK_ASR_SANDBOX_APPROVAL_ITEM",
                    "row_number": row_number,
                    "blocked_reasons": ["preflight_item_not_object"],
                }
            )
            continue
        key = (optional_int(preflight_item.get("row_number")) or row_number, clean(preflight_item.get("queue_item_id")))
        contract_task = mapping_or_empty(contract_tasks.get(key))
        audio = mapping_or_empty(preflight_item.get("audio"))
        sandbox_paths = mapping_or_empty(preflight_item.get("sandbox_paths"))
        item = {
            "action": "INCLUDE_ASR_SANDBOX_APPROVAL_ITEM",
            "reason": "preflight_passed_item_included_for_human_approval",
            "row_number": key[0],
            "queue_item_id": clean(preflight_item.get("queue_item_id")) or None,
            "asset_id": optional_int(preflight_item.get("asset_id")),
            "selected_engine": clean(preflight_report.get("selected_engine")) or None,
            "audio": {
                "path": clean(audio.get("path")) or None,
                "sha256": clean(audio.get("actual_sha256")) or None,
                "size_bytes": optional_int(audio.get("actual_size_bytes")) or 0,
                "sha256_ok": bool(audio.get("sha256_ok")),
                "size_ok": bool(audio.get("size_ok")),
            },
            "expected_outputs": {
                "transcript_json": clean(sandbox_paths.get("transcript_json")) or None,
                "transcript_txt": clean(sandbox_paths.get("transcript_txt")) or None,
                "asr_audit_json": clean(sandbox_paths.get("asr_audit_json")) or None,
                "engine_stdout_log": clean(sandbox_paths.get("engine_stdout_log")) or None,
                "engine_stderr_log": clean(sandbox_paths.get("engine_stderr_log")) or None,
            },
            "task_tmp_dir": clean(sandbox_paths.get("task_tmp_dir")) or None,
            "resource_limits": dict(mapping_or_empty(contract_task.get("resource_limits"))),
            "preflight": {
                "action": clean(preflight_item.get("action")) or None,
                "reason": clean(preflight_item.get("reason")) or None,
                "blocked_reasons": list(preflight_item.get("blocked_reasons") or []),
                "warnings": list(preflight_item.get("warnings") or []),
            },
            "approval_required": True,
            "execution_approved": False,
            "dispatch_allowed": False,
            "run_asr": False,
            "write_transcripts": False,
            "blocked_reasons": [],
        }
        if not contract_task:
            item["blocked_reasons"].append("matching_contract_task_missing")
        if clean(preflight_item.get("action")) != "PASS_ASR_SANDBOX_PREFLIGHT_TASK":
            item["blocked_reasons"].append("preflight_item_not_passed")
        if not item["audio"]["sha256_ok"]:
            item["blocked_reasons"].append("audio_sha_not_ok")
        if not item["audio"]["size_ok"]:
            item["blocked_reasons"].append("audio_size_not_ok")
        if item["blocked_reasons"]:
            item["action"] = "BLOCK_ASR_SANDBOX_APPROVAL_ITEM"
            item["reason"] = ",".join(sorted(set(item["blocked_reasons"])))
            item["blocked_reasons"] = sorted(set(item["blocked_reasons"]))
        items.append(item)
    return items


def validate_approval_items(items: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if not items:
        reasons.append("approval_items_empty")
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in items if clean(item.get("queue_item_id")))
    output_counts = Counter(
        clean(value)
        for item in items
        for value in mapping_or_empty(item.get("expected_outputs")).values()
        if clean(value)
    )
    for index, item in enumerate(items, start=1):
        if clean(item.get("action")) != "INCLUDE_ASR_SANDBOX_APPROVAL_ITEM":
            reasons.append(f"approval_item_{index}_not_included")
        if not clean(item.get("queue_item_id")):
            reasons.append(f"approval_item_{index}_queue_item_id_missing")
        if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
            reasons.append(f"approval_item_{index}_duplicate_queue_item_id")
        audio = mapping_or_empty(item.get("audio"))
        if not bool(audio.get("sha256_ok")):
            reasons.append(f"approval_item_{index}_audio_sha_not_ok")
        if not bool(audio.get("size_ok")):
            reasons.append(f"approval_item_{index}_audio_size_not_ok")
        for output_key, value in mapping_or_empty(item.get("expected_outputs")).items():
            text = clean(value)
            if not text:
                reasons.append(f"approval_item_{index}_{output_key}_missing")
            elif "stable_runtime" in Path(text).parts:
                reasons.append(f"approval_item_{index}_{output_key}_under_stable_runtime")
            elif output_counts.get(text, 0) > 1:
                reasons.append(f"approval_item_{index}_{output_key}_duplicate")
        if bool(item.get("execution_approved")):
            reasons.append(f"approval_item_{index}_must_not_be_approved")
    return sorted(set(reasons))


def build_approval_packet_actions(source_reasons: Sequence[str], items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if source_reasons:
        return [
            {
                "action": "BLOCK_ASR_SANDBOX_APPROVAL_PACKET",
                "reason": "source_preflight_or_items_failed_approval_packet_contract",
                "technical_reasons": list(source_reasons),
                "approval_required": True,
                "execution_approved": False,
                "dispatch_allowed": False,
                "run_asr": False,
            }
        ]
    return [
        {
            "action": "PREPARE_ASR_SANDBOX_EXECUTION_APPROVAL_PACKET",
            "reason": "preflight_passed_and_human_approval_packet_prepared_without_execution",
            "item_count": len(items),
            "approval_required": True,
            "execution_approved": False,
            "dispatch_allowed": False,
            "run_asr": False,
        }
    ]


def build_approval_packet_payload(
    preflight_report_path: Path,
    preflight_report: Mapping[str, Any],
    preflight_report_sha256: str,
    contract_path: Path,
    contract: Mapping[str, Any],
    contract_sha256: str,
    packet_ref: str,
    source_reasons: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    preflight_workload = mapping_or_empty(preflight_report.get("workload"))
    contract_resources = mapping_or_empty(contract.get("batch_resource_limits"))
    return {
        "schema_version": ASR_WORKER_SANDBOX_APPROVAL_PACKET_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "sandbox_execution_approval_packet",
        "status": "pending_human_approval" if not source_reasons else "blocked_approval_packet",
        "approval_packet_ref": packet_ref,
        "approval_status": "pending_human_approval",
        "approval_required": True,
        "execution_approved": False,
        "dispatch_allowed": False,
        "run_asr": False,
        "execution_allowed": False,
        "write_outputs": False,
        "write_transcripts": False,
        "write_runtime_db": False,
        "write_crm": False,
        "selected_engine": clean(preflight_report.get("selected_engine")) or None,
        "input_refs": {
            "preflight_report_path": str(preflight_report_path),
            "preflight_report_sha256": preflight_report_sha256,
            "contract_path": str(contract_path),
            "contract_sha256": contract_sha256,
            "readiness_report_path": mapping_or_empty(preflight_report.get("input_refs")).get("readiness_report_path"),
            "readiness_report_sha256": mapping_or_empty(preflight_report.get("input_refs")).get("readiness_report_sha256"),
            "worker_plan_path": mapping_or_empty(preflight_report.get("input_refs")).get("worker_plan_path"),
            "worker_plan_sha256": mapping_or_empty(preflight_report.get("input_refs")).get("worker_plan_sha256"),
        },
        "workload": {
            "tasks": optional_int(preflight_workload.get("tasks")) or 0,
            "audio_files_checked": optional_int(preflight_workload.get("audio_files_checked")) or 0,
            "audio_sha_ok": optional_int(preflight_workload.get("audio_sha_ok")) or 0,
            "output_collisions": optional_int(preflight_workload.get("output_collisions")) or 0,
        },
        "batch_resource_limits": dict(contract_resources),
        "preflight_summary": {
            "status": clean(preflight_report.get("status")) or None,
            "preflight_ready": bool(preflight_report.get("preflight_ready")),
            "engine_preflight_ok": bool(mapping_or_empty(preflight_report.get("engine_preflight")).get("ok")),
            "disk_space_ok": bool(mapping_or_empty(preflight_report.get("disk_preflight")).get("ok")),
            "dir_preflight_ok": bool(mapping_or_empty(preflight_report.get("directory_preflight")).get("ok")),
            "required_free_bytes": optional_int(mapping_or_empty(preflight_report.get("disk_preflight")).get("required_free_bytes")) or 0,
            "available_free_bytes": optional_int(mapping_or_empty(preflight_report.get("disk_preflight")).get("available_free_bytes")) or 0,
        },
        "required_approval_phrase": REQUIRED_APPROVAL_PHRASE,
        "required_acknowledgements": list(REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS),
        "acknowledgement_template": {key: False for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS},
        "approval_record_template": {
            "decision": "pending",
            "approved_by": None,
            "approved_at": None,
            "approval_phrase": None,
            "approval_packet_ref": packet_ref,
            "preflight_report_sha256": preflight_report_sha256,
            "contract_sha256": contract_sha256,
            "acknowledgements": {key: False for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS},
            "valid_for_asr_execution_dispatch": False,
        },
        "actions": list(actions),
        "items": list(items),
        "source_validation_reasons": list(source_reasons),
        "hard_guards": {
            "dispatch_worker": False,
            "load_asr_model": False,
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
            "may_record_human_execution_approval": not source_reasons,
            "may_run_asr_in_this_stage": False,
            "may_dispatch_worker_in_this_stage": False,
            "must_require_exact_approval_phrase": True,
            "must_require_all_acknowledgements_true": True,
            "must_match_preflight_report_sha256": True,
            "must_match_contract_sha256": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
            "must_not_touch_stable_runtime": True,
        },
    }


def resolve_approval_packet_paths(
    product_root: Path,
    preflight_report_path: Path,
    out_dir: Path,
    approval_packet_path: Path,
    out_path: Optional[Path],
    contract_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "preflight_report_path": preflight_report_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "approval_packet_path": approval_packet_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    if contract_path is not None:
        paths["contract_path"] = contract_path.resolve(strict=False)
    guard_approval_packet_paths(**paths)
    return paths


def guard_approval_packet_paths(
    product_root: Path,
    preflight_report_path: Path,
    out_dir: Path,
    approval_packet_path: Path,
    out_path: Optional[Path] = None,
    contract_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR sandbox preflight report", preflight_report_path),
        ("ASR sandbox approval output directory", out_dir),
        ("ASR sandbox approval packet", approval_packet_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not preflight_report_path.exists() or not preflight_report_path.is_file():
        raise FileNotFoundError(f"ASR sandbox preflight report not found: {preflight_report_path}")
    if not path_is_relative_to(approval_packet_path, out_dir):
        raise ValueError(f"ASR sandbox approval packet must stay under output directory: {out_dir}")
    for label, path in (("ASR sandbox contract", contract_path), ("ASR sandbox approval audit", out_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if out_path is not None and not path_is_relative_to(out_path, out_dir):
        raise ValueError(f"ASR sandbox approval audit must stay under output directory: {out_dir}")


def infer_contract_path(preflight_report: Mapping[str, Any], product_root: Path) -> Path:
    raw = clean(mapping_or_empty(preflight_report.get("input_refs")).get("contract_path"))
    if not raw:
        raise ValueError("ASR sandbox contract path is missing from preflight input_refs")
    path = Path(raw).resolve(strict=False)
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"ASR sandbox contract must stay under product root: {product_root}")
    return path


def guard_inferred_contract_path(product_root: Path, contract_path: Path) -> None:
    if "stable_runtime" in contract_path.parts:
        raise ValueError("refusing ASR sandbox contract under stable_runtime")
    if not path_is_relative_to(contract_path, product_root):
        raise ValueError(f"ASR sandbox contract must stay under product root: {product_root}")
    if not contract_path.exists() or not contract_path.is_file():
        raise FileNotFoundError(f"ASR sandbox contract not found: {contract_path}")


def approval_packet_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "reads_preflight_report": True,
        "reads_contract": True,
        "writes_approval_packet": True,
        "creates_sandbox_output_dirs": False,
        "creates_sandbox_tmp_dirs": False,
        "imports_asr_modules": False,
        "loads_models": False,
        "dispatch_worker": False,
        "downloads_audio": False,
        "copies_audio": False,
        "hardlinks_audio": False,
        "reads_audio": False,
        "product_db_writes": False,
        "asset_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
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
