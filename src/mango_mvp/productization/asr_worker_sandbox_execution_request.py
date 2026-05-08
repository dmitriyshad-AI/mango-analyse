from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE
from mango_mvp.productization.asr_scheduler_dry_run import DANGEROUS_HARD_GUARDS, load_json_object, mapping_or_empty, optional_int
from mango_mvp.productization.asr_worker_sandbox_approval_packet import (
    ASR_WORKER_SANDBOX_APPROVAL_PACKET_SCHEMA_VERSION,
    REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS,
)
from mango_mvp.productization.asr_worker_sandbox_execution_contract import (
    ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION,
)
from mango_mvp.productization.asr_worker_sandbox_human_approval_record import (
    ASR_WORKER_SANDBOX_HUMAN_APPROVAL_RECORD_SCHEMA_VERSION,
    validate_approval_packet_for_human_record,
    validate_human_approval_record_payload,
)
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_SANDBOX_EXECUTION_REQUEST_SCHEMA_VERSION = "asr_worker_sandbox_execution_request_v1"


@dataclass(frozen=True)
class AsrWorkerSandboxExecutionRequestSummary:
    schema_version: str
    product_root: str
    approval_packet_path: str
    approval_record_path: Optional[str]
    contract_path: Optional[str]
    out_dir: str
    request_path: str
    out_path: Optional[str]
    approval_packet_sha256: str
    approval_record_sha256: Optional[str]
    contract_sha256: Optional[str]
    approval_packet_ref: Optional[str]
    selected_engine: Optional[str]
    tasks: int
    requested_tasks: int
    approval_packet_valid: bool
    approval_record_present: bool
    approval_record_valid: bool
    contract_valid: bool
    execution_request_ready: bool
    execution_approved_by_human_record: bool
    dispatch_allowed: bool
    run_asr: bool
    write_transcripts: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_sandbox_execution_request(
    product_root: Path,
    approval_packet_path: Path,
    out_dir: Path,
    request_path: Path,
    out_path: Optional[Path] = None,
    approval_record_path: Optional[Path] = None,
    contract_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_execution_request_paths(
        product_root=product_root,
        approval_packet_path=approval_packet_path,
        approval_record_path=approval_record_path,
        contract_path=contract_path,
        out_dir=out_dir,
        request_path=request_path,
        out_path=out_path,
    )
    packet = load_json_object(paths["approval_packet_path"])
    packet_sha = sha256_file(paths["approval_packet_path"])
    packet_reasons = validate_approval_packet_for_execution_request(packet=packet, packet_sha256=packet_sha)

    resolved_contract_path = paths.get("contract_path")
    contract: Optional[Mapping[str, Any]] = None
    contract_sha: Optional[str] = None
    contract_reasons: list[str] = []
    if resolved_contract_path is None:
        resolved_contract_path, contract_reasons = infer_contract_path_from_packet(packet=packet, product_root=paths["product_root"])
    else:
        contract_reasons.extend(validate_contract_path(paths["product_root"], resolved_contract_path))
    if resolved_contract_path is not None and not contract_reasons:
        contract = load_json_object(resolved_contract_path)
        contract_sha = sha256_file(resolved_contract_path)
        contract_reasons.extend(validate_contract_for_execution_request(contract=contract, contract_sha256=contract_sha, packet=packet))

    record: Optional[Mapping[str, Any]] = None
    record_sha: Optional[str] = None
    record_reasons: list[str] = []
    record_path = paths.get("approval_record_path")
    if record_path is None:
        record_reasons.append("approval_record_path_missing")
    elif not record_path.exists():
        record_reasons.append("approval_record_missing")
    elif not record_path.is_file():
        record_reasons.append("approval_record_not_file")
    else:
        record = load_json_object(record_path)
        record_sha = sha256_file(record_path)
        record_reasons.extend(
            validate_human_approval_record_for_execution_request(
                packet=packet,
                packet_sha256=packet_sha,
                record=record,
                record_sha256=record_sha,
            )
        )

    packet_valid = not packet_reasons
    contract_valid = not contract_reasons
    record_present = record is not None
    record_valid = record_present and not record_reasons
    request_ready = packet_valid and contract_valid and record_valid and contract is not None
    request_items = build_execution_request_items(contract=contract, packet=packet) if request_ready else []
    request_reasons = sorted(set(packet_reasons + contract_reasons + record_reasons))
    action = execution_request_action(
        packet_valid=packet_valid,
        contract_valid=contract_valid,
        record_present=record_present,
        record_valid=record_valid,
        request_ready=request_ready,
    )
    actions = build_execution_request_actions(action=action, request_ready=request_ready, request_reasons=request_reasons, request_items=request_items)
    request_payload = build_execution_request_payload(
        product_root=paths["product_root"],
        approval_packet_path=paths["approval_packet_path"],
        approval_packet=packet,
        approval_packet_sha256=packet_sha,
        approval_record_path=record_path,
        approval_record=record,
        approval_record_sha256=record_sha,
        contract_path=resolved_contract_path,
        contract=contract,
        contract_sha256=contract_sha,
        out_dir=paths["out_dir"],
        request_path=paths["request_path"],
        request_ready=request_ready,
        request_reasons=request_reasons,
        actions=actions,
        request_items=request_items,
    )
    write_json(paths["request_path"], request_payload)
    request_sha = sha256_file(paths["request_path"])
    benign_missing_approval = not record_present and set(record_reasons).issubset({"approval_record_path_missing", "approval_record_missing"})
    validation_ok = packet_valid and contract_valid and (request_ready or benign_missing_approval)
    summary = AsrWorkerSandboxExecutionRequestSummary(
        schema_version=ASR_WORKER_SANDBOX_EXECUTION_REQUEST_SCHEMA_VERSION,
        product_root=str(paths["product_root"]),
        approval_packet_path=str(paths["approval_packet_path"]),
        approval_record_path=str(record_path) if record_path else None,
        contract_path=str(resolved_contract_path) if resolved_contract_path else None,
        out_dir=str(paths["out_dir"]),
        request_path=str(paths["request_path"]),
        out_path=str(paths.get("out_path")) if paths.get("out_path") else None,
        approval_packet_sha256=packet_sha,
        approval_record_sha256=record_sha,
        contract_sha256=contract_sha,
        approval_packet_ref=clean(packet.get("approval_packet_ref")) or None,
        selected_engine=clean(packet.get("selected_engine")) or None,
        tasks=optional_int(mapping_or_empty(packet.get("workload")).get("tasks")) or 0,
        requested_tasks=len(request_items),
        approval_packet_valid=packet_valid,
        approval_record_present=record_present,
        approval_record_valid=record_valid,
        contract_valid=contract_valid,
        execution_request_ready=request_ready,
        execution_approved_by_human_record=record_valid,
        dispatch_allowed=False,
        run_asr=False,
        write_transcripts=False,
        validation_ok=validation_ok,
        warnings=0 if validation_ok and request_ready else max(1, len(request_reasons)),
    )
    report = {
        "summary": {
            **summary.to_json_dict(),
            "request_sha256": request_sha,
        },
        "action_counts": action_counts_for(actions),
        "actions": actions,
        "approval": {
            "approval_packet_present": True,
            "approval_packet_valid": packet_valid,
            "approval_record_present": record_present,
            "approval_record_valid": record_valid,
            "approval_packet_ref": clean(packet.get("approval_packet_ref")) or None,
            "required_acknowledgements": list(REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS),
            "execution_approved_by_human_record": record_valid,
            "execution_request_ready": request_ready,
            "dispatch_allowed": False,
            "run_asr": False,
            "missing_or_invalid_reasons": request_reasons,
        },
        "execution_request": request_payload,
        "source_approval_packet": {
            "schema_version": clean(packet.get("schema_version")) or None,
            "status": clean(packet.get("status")) or None,
            "approval_status": clean(packet.get("approval_status")) or None,
            "selected_engine": clean(packet.get("selected_engine")) or None,
            "sha256": packet_sha,
        },
        "source_approval_record": {
            "schema_version": clean(record.get("schema_version")) if record else None,
            "decision": clean(record.get("decision")) if record else None,
            "sha256": record_sha,
        },
        "source_contract": {
            "schema_version": clean(contract.get("schema_version")) if contract else None,
            "status": clean(contract.get("status")) if contract else None,
            "selected_engine": clean(contract.get("selected_engine")) if contract else None,
            "sha256": contract_sha,
        },
        "request_item_samples": request_items[:20],
        "safety": execution_request_safety(),
    }
    if out_path:
        write_json(paths["out_path"], report)
    return report


def validate_approval_packet_for_execution_request(packet: Mapping[str, Any], packet_sha256: str) -> list[str]:
    reasons = list(validate_approval_packet_for_human_record(packet=packet, packet_sha256=packet_sha256))
    if clean(packet.get("schema_version")) != ASR_WORKER_SANDBOX_APPROVAL_PACKET_SCHEMA_VERSION:
        reasons.append("approval_packet_schema_unexpected")
    next_stage = mapping_or_empty(packet.get("next_stage_contract"))
    if not bool(next_stage.get("may_record_human_execution_approval")):
        reasons.append("approval_packet_next_stage_human_approval_not_allowed")
    for key in ("may_dispatch_worker_in_this_stage", "may_run_asr_in_this_stage"):
        if bool(next_stage.get(key)):
            reasons.append(f"approval_packet_next_stage_{key}_must_be_false")
    return sorted(set(reasons))


def infer_contract_path_from_packet(packet: Mapping[str, Any], product_root: Path) -> tuple[Optional[Path], list[str]]:
    raw = clean(mapping_or_empty(packet.get("input_refs")).get("contract_path"))
    if not raw:
        return None, ["approval_packet_contract_path_missing"]
    path = Path(raw).resolve(strict=False)
    return path, validate_contract_path(product_root=product_root, contract_path=path)


def validate_contract_path(product_root: Path, contract_path: Path) -> list[str]:
    reasons: list[str] = []
    if "stable_runtime" in contract_path.parts:
        reasons.append("contract_path_under_stable_runtime")
    if not path_is_relative_to(contract_path, product_root):
        reasons.append("contract_path_outside_product_root")
    if not contract_path.exists():
        reasons.append("contract_missing")
    elif not contract_path.is_file():
        reasons.append("contract_not_file")
    return reasons


def validate_contract_for_execution_request(
    contract: Mapping[str, Any],
    contract_sha256: str,
    packet: Mapping[str, Any],
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
    if clean(contract.get("selected_engine")) != clean(packet.get("selected_engine")):
        reasons.append("contract_selected_engine_mismatch")
    packet_refs = mapping_or_empty(packet.get("input_refs"))
    if clean(packet_refs.get("contract_sha256")) != contract_sha256:
        reasons.append("contract_sha_mismatch_packet_ref")
    for key in ("dispatch_allowed", "run_asr", "execution_allowed", "write_outputs", "write_transcripts", "write_runtime_db", "write_crm"):
        if bool(contract.get(key)):
            reasons.append(f"contract_{key}_must_be_false")
    hard_guards = mapping_or_empty(contract.get("hard_guards"))
    if not hard_guards:
        reasons.append("contract_hard_guards_missing")
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("dispatch_worker", "load_asr_model", "write_transcripts"):
        if bool(hard_guards.get(guard)):
            reasons.append(f"contract_hard_guard_{guard}_must_be_false")
    next_stage = mapping_or_empty(contract.get("next_stage_contract"))
    if not bool(next_stage.get("may_request_explicit_asr_execution_approval")):
        reasons.append("contract_next_stage_execution_approval_not_allowed")
    if bool(next_stage.get("may_dispatch_worker_in_this_stage")):
        reasons.append("contract_next_stage_must_not_dispatch")
    if bool(next_stage.get("may_run_asr_in_this_stage")):
        reasons.append("contract_next_stage_must_not_run_asr")
    workload = mapping_or_empty(contract.get("workload"))
    planned = optional_int(workload.get("planned_tasks")) or 0
    blocked = optional_int(workload.get("blocked_tasks")) or 0
    packet_workload = mapping_or_empty(packet.get("workload"))
    if planned <= 0:
        reasons.append("contract_planned_tasks_empty")
    if blocked:
        reasons.append("contract_has_blocked_tasks")
    if planned != (optional_int(packet_workload.get("tasks")) or 0):
        reasons.append("contract_task_count_mismatch_packet")
    tasks = contract.get("tasks")
    if not isinstance(tasks, SequenceABC) or isinstance(tasks, (str, bytes)):
        reasons.append("contract_tasks_missing")
    elif len(tasks) != planned:
        reasons.append("contract_tasks_count_mismatch")
    else:
        reasons.extend(validate_contract_tasks_for_execution_request(tasks))
    return sorted(set(reasons))


def validate_contract_tasks_for_execution_request(tasks: Sequence[Any]) -> list[str]:
    reasons: list[str] = []
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in tasks if isinstance(item, MappingABC) and clean(item.get("queue_item_id")))
    output_counts = Counter(
        clean(value)
        for item in tasks
        if isinstance(item, MappingABC)
        for value in mapping_or_empty(item.get("sandbox_paths")).values()
        if clean(value)
    )
    for index, item in enumerate(tasks, start=1):
        if not isinstance(item, MappingABC):
            reasons.append(f"contract_task_{index}_not_object")
            continue
        if clean(item.get("action")) != "PLAN_ASR_SANDBOX_TASK":
            reasons.append(f"contract_task_{index}_action_unexpected")
        if not clean(item.get("queue_item_id")):
            reasons.append(f"contract_task_{index}_queue_item_id_missing")
        if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
            reasons.append(f"contract_task_{index}_duplicate_queue_item_id")
        if item.get("blocked_reasons"):
            reasons.append(f"contract_task_{index}_has_blocked_reasons")
        audio = mapping_or_empty(item.get("audio"))
        if not clean(audio.get("path")):
            reasons.append(f"contract_task_{index}_audio_path_missing")
        if not clean(audio.get("sha256")):
            reasons.append(f"contract_task_{index}_audio_sha_missing")
        for key in ("dispatch_allowed", "run_asr", "write_outputs", "write_transcripts", "write_runtime_db", "write_crm"):
            if bool(mapping_or_empty(item.get("worker_command_contract")).get(key)):
                reasons.append(f"contract_task_{index}_{key}_must_be_false")
        for output_key, value in mapping_or_empty(item.get("sandbox_paths")).items():
            text = clean(value)
            if not text:
                reasons.append(f"contract_task_{index}_{output_key}_missing")
            elif "stable_runtime" in Path(text).parts:
                reasons.append(f"contract_task_{index}_{output_key}_under_stable_runtime")
            elif output_counts.get(text, 0) > 1:
                reasons.append(f"contract_task_{index}_{output_key}_duplicate")
    return reasons


def validate_human_approval_record_for_execution_request(
    packet: Mapping[str, Any],
    packet_sha256: str,
    record: Mapping[str, Any],
    record_sha256: str,
) -> list[str]:
    reasons = list(
        validate_human_approval_record_payload(
            packet=packet,
            packet_sha256=packet_sha256,
            record=record,
            record_sha256=record_sha256,
        )
    )
    if clean(record.get("schema_version")) != ASR_WORKER_SANDBOX_HUMAN_APPROVAL_RECORD_SCHEMA_VERSION:
        reasons.append("record_schema_unexpected")
    scope = mapping_or_empty(record.get("scope"))
    if not bool(scope.get("valid_for_asr_sandbox_execution_request")):
        reasons.append("record_scope_execution_request_not_allowed")
    for key in ("valid_for_immediate_worker_dispatch", "valid_for_runtime_db_writes", "valid_for_crm_writes"):
        if bool(scope.get(key)):
            reasons.append(f"record_scope_{key}_must_be_false")
    stage_contract = mapping_or_empty(record.get("stage_contract"))
    if not bool(stage_contract.get("valid_for_next_stage_execution_request")):
        reasons.append("record_stage_contract_next_stage_not_allowed")
    for key in ("valid_for_asr_execution_in_stage24", "valid_for_immediate_worker_dispatch"):
        if bool(stage_contract.get(key)):
            reasons.append(f"record_stage_contract_{key}_must_be_false")
    return sorted(set(reasons))


def build_execution_request_items(contract: Mapping[str, Any], packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    packet_items_by_queue = {
        clean(item.get("queue_item_id")): item
        for item in packet.get("items", [])
        if isinstance(item, MappingABC) and clean(item.get("queue_item_id"))
    }
    items: list[Mapping[str, Any]] = []
    for index, task in enumerate(contract.get("tasks", []), start=1):
        if not isinstance(task, MappingABC):
            continue
        queue_item_id = clean(task.get("queue_item_id"))
        approval_item = mapping_or_empty(packet_items_by_queue.get(queue_item_id))
        item = {
            "action": "PLAN_ASR_SANDBOX_EXECUTION_REQUEST_TASK",
            "reason": "human_approval_record_validated_request_planned_without_dispatch",
            "row_number": index,
            "queue_item_id": queue_item_id or None,
            "asset_id": optional_int(task.get("asset_id")),
            "tenant_id": clean(task.get("tenant_id")) or None,
            "provider": clean(task.get("provider")) or None,
            "event_key": clean(task.get("event_key")) or None,
            "recording_id": clean(task.get("recording_id")) or None,
            "selected_engine": clean(task.get("selected_engine")) or clean(packet.get("selected_engine")) or None,
            "audio": dict(mapping_or_empty(task.get("audio"))),
            "sandbox_paths": dict(mapping_or_empty(task.get("sandbox_paths"))),
            "resource_limits": dict(mapping_or_empty(task.get("resource_limits"))),
            "approval_item": {
                "approval_required": bool(approval_item.get("approval_required")),
                "queue_item_id": clean(approval_item.get("queue_item_id")) or None,
                "audio_sha256_ok": bool(mapping_or_empty(approval_item.get("audio")).get("sha256_ok")),
            },
            "execution_request_ready": True,
            "dispatch_allowed": False,
            "run_asr": False,
            "write_outputs": False,
            "write_transcripts": False,
            "write_runtime_db": False,
            "write_crm": False,
        }
        items.append(item)
    return items


def execution_request_action(
    packet_valid: bool,
    contract_valid: bool,
    record_present: bool,
    record_valid: bool,
    request_ready: bool,
) -> str:
    if request_ready:
        return "PLAN_ASR_SANDBOX_EXECUTION_REQUEST_NOT_DISPATCHED"
    if not packet_valid or not contract_valid:
        return "BLOCK_ASR_SANDBOX_EXECUTION_REQUEST_SOURCE_INVALID"
    if not record_present:
        return "BLOCK_ASR_SANDBOX_EXECUTION_REQUEST_PENDING_HUMAN_APPROVAL"
    if not record_valid:
        return "BLOCK_ASR_SANDBOX_EXECUTION_REQUEST_INVALID_HUMAN_APPROVAL"
    return "BLOCK_ASR_SANDBOX_EXECUTION_REQUEST"


def build_execution_request_actions(
    action: str,
    request_ready: bool,
    request_reasons: Sequence[str],
    request_items: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        {
            "action": action,
            "reason": "execution_request_ready_but_not_dispatched" if request_ready else "execution_request_blocked_without_asr",
            "technical_reasons": list(request_reasons),
            "requested_tasks": len(request_items),
            "execution_request_ready": request_ready,
            "dispatch_allowed": False,
            "run_asr": False,
            "write_outputs": False,
            "write_transcripts": False,
            "write_runtime_db": False,
            "write_crm": False,
        }
    ]


def build_execution_request_payload(
    product_root: Path,
    approval_packet_path: Path,
    approval_packet: Mapping[str, Any],
    approval_packet_sha256: str,
    approval_record_path: Optional[Path],
    approval_record: Optional[Mapping[str, Any]],
    approval_record_sha256: Optional[str],
    contract_path: Optional[Path],
    contract: Optional[Mapping[str, Any]],
    contract_sha256: Optional[str],
    out_dir: Path,
    request_path: Path,
    request_ready: bool,
    request_reasons: Sequence[str],
    actions: Sequence[Mapping[str, Any]],
    request_items: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    workload = mapping_or_empty(approval_packet.get("workload"))
    return {
        "schema_version": ASR_WORKER_SANDBOX_EXECUTION_REQUEST_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "sandbox_execution_request_dry_run",
        "status": "execution_request_planned_not_dispatched" if request_ready else "blocked_execution_request",
        "product_root": str(product_root),
        "approval_packet_ref": clean(approval_packet.get("approval_packet_ref")) or None,
        "selected_engine": clean(approval_packet.get("selected_engine")) or None,
        "execution_request_ready": request_ready,
        "execution_approved_by_human_record": request_ready,
        "dispatch_allowed": False,
        "run_asr": False,
        "execution_allowed": False,
        "write_outputs": False,
        "write_transcripts": False,
        "write_runtime_db": False,
        "write_crm": False,
        "create_dirs": False,
        "input_refs": {
            "approval_packet_path": str(approval_packet_path),
            "approval_packet_sha256": approval_packet_sha256,
            "approval_record_path": str(approval_record_path) if approval_record_path else None,
            "approval_record_sha256": approval_record_sha256,
            "contract_path": str(contract_path) if contract_path else None,
            "contract_sha256": contract_sha256,
        },
        "approval_record": {
            "present": bool(approval_record),
            "schema_version": clean(approval_record.get("schema_version")) if approval_record else None,
            "decision": clean(approval_record.get("decision")) if approval_record else None,
            "approved_by": clean(approval_record.get("approved_by")) if approval_record else None,
            "approved_at": clean(approval_record.get("approved_at")) if approval_record else None,
            "valid_for_execution_request": request_ready,
        },
        "workload": {
            "tasks": optional_int(workload.get("tasks")) or 0,
            "requested_tasks": len(request_items),
            "audio_files_checked": optional_int(workload.get("audio_files_checked")) or 0,
            "audio_sha_ok": optional_int(workload.get("audio_sha_ok")) or 0,
            "output_collisions": optional_int(workload.get("output_collisions")) or 0,
        },
        "batch_resource_limits": dict(mapping_or_empty(approval_packet.get("batch_resource_limits"))),
        "out_dir": str(out_dir),
        "request_path": str(request_path),
        "actions": list(actions),
        "request_items": list(request_items),
        "source_validation_reasons": list(request_reasons),
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
            "may_create_worker_launcher_dry_run": request_ready,
            "may_dispatch_worker_in_this_stage": False,
            "may_run_asr_in_this_stage": False,
            "must_validate_approval_record_sha256": True,
            "must_validate_approval_packet_sha256": True,
            "must_validate_contract_sha256": True,
            "must_run_final_preflight_before_dispatch": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
            "must_not_touch_stable_runtime": True,
        },
        "source_contract": {
            "schema_version": clean(contract.get("schema_version")) if contract else None,
            "status": clean(contract.get("status")) if contract else None,
        },
        "safety": execution_request_safety(),
    }


def resolve_execution_request_paths(
    product_root: Path,
    approval_packet_path: Path,
    approval_record_path: Optional[Path],
    contract_path: Optional[Path],
    out_dir: Path,
    request_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "approval_packet_path": approval_packet_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "request_path": request_path.resolve(strict=False),
    }
    if approval_record_path is not None:
        paths["approval_record_path"] = approval_record_path.resolve(strict=False)
    if contract_path is not None:
        paths["contract_path"] = contract_path.resolve(strict=False)
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_execution_request_paths(**paths)
    return paths


def guard_execution_request_paths(
    product_root: Path,
    approval_packet_path: Path,
    out_dir: Path,
    request_path: Path,
    approval_record_path: Optional[Path] = None,
    contract_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR sandbox approval packet", approval_packet_path),
        ("ASR sandbox execution request output directory", out_dir),
        ("ASR sandbox execution request", request_path),
        ("ASR sandbox human approval record", approval_record_path),
        ("ASR sandbox contract", contract_path),
        ("ASR sandbox execution request audit", out_path),
    ):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not approval_packet_path.exists() or not approval_packet_path.is_file():
        raise FileNotFoundError(f"ASR sandbox approval packet not found: {approval_packet_path}")
    if not path_is_relative_to(request_path, out_dir):
        raise ValueError(f"ASR sandbox execution request must stay under output directory: {out_dir}")
    if out_path is not None and not path_is_relative_to(out_path, out_dir):
        raise ValueError(f"ASR sandbox execution request audit must stay under output directory: {out_dir}")
    if out_path is not None and out_path == request_path:
        raise ValueError("ASR sandbox execution request audit must differ from request payload")


def execution_request_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "reads_approval_packet": True,
        "reads_human_approval_record_if_present": True,
        "reads_contract": True,
        "writes_execution_request": True,
        "writes_audit_json": True,
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
