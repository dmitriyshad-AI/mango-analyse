from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE
from mango_mvp.productization.asr_scheduler_dry_run import DANGEROUS_HARD_GUARDS, load_json_object, mapping_or_empty, optional_int
from mango_mvp.productization.asr_worker_sandbox_approval_packet import (
    ASR_WORKER_SANDBOX_APPROVAL_PACKET_SCHEMA_VERSION,
    REQUIRED_APPROVAL_PHRASE,
    REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS,
)
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_SANDBOX_HUMAN_APPROVAL_RECORD_SCHEMA_VERSION = "asr_worker_sandbox_human_approval_record_v1"
ASR_WORKER_SANDBOX_HUMAN_APPROVAL_WRITER_SCHEMA_VERSION = "asr_worker_sandbox_human_approval_writer_v1"


@dataclass(frozen=True)
class AsrWorkerSandboxHumanApprovalRecordSummary:
    schema_version: str
    product_root: str
    operation: str
    approval_packet_path: str
    approval_record_path: Optional[str]
    out_path: Optional[str]
    approval_packet_sha256: str
    approval_record_sha256: Optional[str]
    approval_packet_ref: Optional[str]
    selected_engine: Optional[str]
    tasks: int
    required_acknowledgements: int
    acknowledgement_true_count: int
    approval_packet_valid: bool
    approval_record_present: bool
    approval_record_valid: bool
    execution_approved: bool
    dispatch_allowed: bool
    run_asr: bool
    write_transcripts: bool
    validation_ok: bool
    written: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_sandbox_human_approval_requirements(
    product_root: Path,
    approval_packet_path: Path,
    out_path: Optional[Path] = None,
    approval_record_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_human_approval_paths(
        product_root=product_root,
        approval_packet_path=approval_packet_path,
        approval_record_path=approval_record_path,
        out_path=out_path,
        require_record_exists=False,
    )
    packet_path = paths["approval_packet_path"]
    record_path = paths.get("approval_record_path")
    packet = load_json_object(packet_path)
    packet_sha = sha256_file(packet_path)
    packet_reasons = validate_approval_packet_for_human_record(packet=packet, packet_sha256=packet_sha)
    report = build_human_approval_report(
        product_root=paths["product_root"],
        operation="requirements",
        approval_packet_path=packet_path,
        approval_record_path=record_path,
        out_path=paths.get("out_path"),
        packet=packet,
        packet_sha256=packet_sha,
        packet_reasons=packet_reasons,
        approval_record=None,
        approval_record_sha256=None,
        record_reasons=[],
        written=0,
    )
    if out_path:
        write_json(paths["out_path"], report)
    return report


def write_asr_worker_sandbox_human_approval_record(
    product_root: Path,
    approval_packet_path: Path,
    approval_record_path: Path,
    out_path: Optional[Path] = None,
    approved_by: str = "",
    approval_phrase: str = "",
    acknowledgements: Optional[Mapping[str, bool]] = None,
    approved_at: Optional[str] = None,
    reason: str = "",
    replace_existing: bool = False,
) -> Mapping[str, Any]:
    paths = resolve_human_approval_paths(
        product_root=product_root,
        approval_packet_path=approval_packet_path,
        approval_record_path=approval_record_path,
        out_path=out_path,
        require_record_exists=False,
    )
    packet_path = paths["approval_packet_path"]
    record_path = paths["approval_record_path"]
    if record_path.exists() and not replace_existing:
        raise FileExistsError(f"ASR sandbox human approval record already exists: {record_path}")
    packet = load_json_object(packet_path)
    packet_sha = sha256_file(packet_path)
    packet_reasons = validate_approval_packet_for_human_record(packet=packet, packet_sha256=packet_sha)
    input_reasons = validate_human_approval_inputs(
        packet=packet,
        approved_by=approved_by,
        approval_phrase=approval_phrase,
        acknowledgements=acknowledgements or {},
        approved_at=approved_at,
    )
    source_reasons = sorted(set(packet_reasons + input_reasons))
    if source_reasons:
        report = build_human_approval_report(
            product_root=paths["product_root"],
            operation="write",
            approval_packet_path=packet_path,
            approval_record_path=record_path,
            out_path=paths.get("out_path"),
            packet=packet,
            packet_sha256=packet_sha,
            packet_reasons=packet_reasons,
            approval_record=None,
            approval_record_sha256=None,
            record_reasons=source_reasons,
            written=0,
        )
        if out_path:
            write_json(paths["out_path"], report)
        return report
    record = build_human_approval_record(
        packet=packet,
        packet_sha256=packet_sha,
        approved_by=approved_by,
        approved_at=normalize_approved_at(approved_at),
        approval_phrase=approval_phrase,
        acknowledgements=acknowledgements or {},
        reason=reason,
    )
    write_json(record_path, record)
    record_sha = sha256_file(record_path)
    record_reasons = validate_human_approval_record_payload(
        packet=packet,
        packet_sha256=packet_sha,
        record=record,
        record_sha256=record_sha,
    )
    report = build_human_approval_report(
        product_root=paths["product_root"],
        operation="write",
        approval_packet_path=packet_path,
        approval_record_path=record_path,
        out_path=paths.get("out_path"),
        packet=packet,
        packet_sha256=packet_sha,
        packet_reasons=packet_reasons,
        approval_record=record,
        approval_record_sha256=record_sha,
        record_reasons=record_reasons,
        written=1,
    )
    if out_path:
        write_json(paths["out_path"], report)
    return report


def validate_asr_worker_sandbox_human_approval_record(
    product_root: Path,
    approval_packet_path: Path,
    approval_record_path: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_human_approval_paths(
        product_root=product_root,
        approval_packet_path=approval_packet_path,
        approval_record_path=approval_record_path,
        out_path=out_path,
        require_record_exists=True,
    )
    packet = load_json_object(paths["approval_packet_path"])
    packet_sha = sha256_file(paths["approval_packet_path"])
    record = load_json_object(paths["approval_record_path"])
    record_sha = sha256_file(paths["approval_record_path"])
    packet_reasons = validate_approval_packet_for_human_record(packet=packet, packet_sha256=packet_sha)
    record_reasons = validate_human_approval_record_payload(
        packet=packet,
        packet_sha256=packet_sha,
        record=record,
        record_sha256=record_sha,
    )
    report = build_human_approval_report(
        product_root=paths["product_root"],
        operation="validate",
        approval_packet_path=paths["approval_packet_path"],
        approval_record_path=paths["approval_record_path"],
        out_path=paths.get("out_path"),
        packet=packet,
        packet_sha256=packet_sha,
        packet_reasons=packet_reasons,
        approval_record=record,
        approval_record_sha256=record_sha,
        record_reasons=record_reasons,
        written=0,
    )
    if out_path:
        write_json(paths["out_path"], report)
    return report


def validate_approval_packet_for_human_record(packet: Mapping[str, Any], packet_sha256: str) -> list[str]:
    reasons: list[str] = []
    if clean(packet.get("schema_version")) != ASR_WORKER_SANDBOX_APPROVAL_PACKET_SCHEMA_VERSION:
        reasons.append("approval_packet_schema_unexpected")
    if clean(packet.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("approval_packet_job_type_unexpected")
    if clean(packet.get("mode")) != "sandbox_execution_approval_packet":
        reasons.append("approval_packet_mode_unexpected")
    if clean(packet.get("status")) != "pending_human_approval":
        reasons.append("approval_packet_status_unexpected")
    if clean(packet.get("approval_status")) != "pending_human_approval":
        reasons.append("approval_packet_approval_status_unexpected")
    if not bool(packet.get("approval_required")):
        reasons.append("approval_packet_approval_required_missing")
    if bool(packet.get("execution_approved")):
        reasons.append("approval_packet_must_not_be_preapproved")
    for key in ("dispatch_allowed", "run_asr", "execution_allowed", "write_outputs", "write_transcripts", "write_runtime_db", "write_crm"):
        if bool(packet.get(key)):
            reasons.append(f"approval_packet_{key}_must_be_false")
    if clean(packet.get("required_approval_phrase")) != REQUIRED_APPROVAL_PHRASE:
        reasons.append("approval_packet_required_phrase_unexpected")
    required_ack = packet.get("required_acknowledgements")
    if not isinstance(required_ack, SequenceABC) or isinstance(required_ack, (str, bytes)):
        reasons.append("approval_packet_required_acknowledgements_missing")
    elif tuple(clean(item) for item in required_ack) != REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS:
        reasons.append("approval_packet_required_acknowledgements_unexpected")
    template = mapping_or_empty(packet.get("acknowledgement_template"))
    for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS:
        if key not in template:
            reasons.append(f"approval_packet_ack_template_missing:{key}")
        elif bool(template.get(key)):
            reasons.append(f"approval_packet_ack_template_must_be_false:{key}")
    record_template = mapping_or_empty(packet.get("approval_record_template"))
    if clean(record_template.get("decision")) != "pending":
        reasons.append("approval_packet_record_template_decision_unexpected")
    if bool(record_template.get("valid_for_asr_execution_dispatch")):
        reasons.append("approval_packet_record_template_must_not_allow_dispatch")
    input_refs = mapping_or_empty(packet.get("input_refs"))
    if clean(record_template.get("preflight_report_sha256")) != clean(input_refs.get("preflight_report_sha256")):
        reasons.append("approval_packet_record_template_preflight_sha_mismatch")
    if clean(record_template.get("contract_sha256")) != clean(input_refs.get("contract_sha256")):
        reasons.append("approval_packet_record_template_contract_sha_mismatch")
    workload = mapping_or_empty(packet.get("workload"))
    tasks = optional_int(workload.get("tasks")) or 0
    if tasks <= 0:
        reasons.append("approval_packet_tasks_empty")
    if optional_int(workload.get("audio_files_checked")) != tasks:
        reasons.append("approval_packet_audio_files_checked_mismatch")
    if optional_int(workload.get("audio_sha_ok")) != tasks:
        reasons.append("approval_packet_audio_sha_ok_mismatch")
    if optional_int(workload.get("output_collisions")):
        reasons.append("approval_packet_has_output_collisions")
    items = packet.get("items")
    if not isinstance(items, SequenceABC) or isinstance(items, (str, bytes)):
        reasons.append("approval_packet_items_missing")
    elif len(items) != tasks:
        reasons.append("approval_packet_item_count_mismatch")
    else:
        reasons.extend(validate_packet_items_for_human_record(items))
    hard_guards = mapping_or_empty(packet.get("hard_guards"))
    if not hard_guards:
        reasons.append("approval_packet_hard_guards_missing")
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("dispatch_worker", "load_asr_model", "write_transcripts"):
        if bool(hard_guards.get(guard)):
            reasons.append(f"approval_packet_hard_guard_{guard}_must_be_false")
    next_stage = mapping_or_empty(packet.get("next_stage_contract"))
    if not bool(next_stage.get("may_record_human_execution_approval")):
        reasons.append("approval_packet_next_stage_record_not_allowed")
    if bool(next_stage.get("may_dispatch_worker_in_this_stage")):
        reasons.append("approval_packet_next_stage_must_not_dispatch")
    if bool(next_stage.get("may_run_asr_in_this_stage")):
        reasons.append("approval_packet_next_stage_must_not_run_asr")
    for key in (
        "must_require_exact_approval_phrase",
        "must_require_all_acknowledgements_true",
        "must_match_preflight_report_sha256",
        "must_match_contract_sha256",
        "must_not_write_runtime_db",
        "must_not_write_crm",
        "must_not_touch_stable_runtime",
    ):
        if not bool(next_stage.get(key)):
            reasons.append(f"approval_packet_next_stage_{key}_missing")
    if not packet_sha256:
        reasons.append("approval_packet_sha_missing")
    return sorted(set(reasons))


def validate_packet_items_for_human_record(items: Sequence[Any]) -> list[str]:
    reasons: list[str] = []
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in items if isinstance(item, MappingABC) and clean(item.get("queue_item_id")))
    for index, item in enumerate(items, start=1):
        if not isinstance(item, MappingABC):
            reasons.append(f"approval_packet_item_{index}_not_object")
            continue
        if clean(item.get("action")) != "INCLUDE_ASR_SANDBOX_APPROVAL_ITEM":
            reasons.append(f"approval_packet_item_{index}_action_unexpected")
        if not clean(item.get("queue_item_id")):
            reasons.append(f"approval_packet_item_{index}_queue_item_id_missing")
        if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
            reasons.append(f"approval_packet_item_{index}_duplicate_queue_item_id")
        audio = mapping_or_empty(item.get("audio"))
        if not bool(audio.get("sha256_ok")):
            reasons.append(f"approval_packet_item_{index}_audio_sha_not_ok")
        if not bool(audio.get("size_ok")):
            reasons.append(f"approval_packet_item_{index}_audio_size_not_ok")
        if bool(item.get("execution_approved")):
            reasons.append(f"approval_packet_item_{index}_must_not_be_approved")
        for key, value in mapping_or_empty(item.get("expected_outputs")).items():
            if not clean(value):
                reasons.append(f"approval_packet_item_{index}_{key}_missing")
            elif "stable_runtime" in Path(clean(value)).parts:
                reasons.append(f"approval_packet_item_{index}_{key}_under_stable_runtime")
    return reasons


def validate_human_approval_inputs(
    packet: Mapping[str, Any],
    approved_by: str,
    approval_phrase: str,
    acknowledgements: Mapping[str, bool],
    approved_at: Optional[str],
) -> list[str]:
    reasons: list[str] = []
    if not clean(approved_by):
        reasons.append("approved_by_missing")
    if clean(approval_phrase) != clean(packet.get("required_approval_phrase")):
        reasons.append("approval_phrase_mismatch")
    required = tuple(clean(item) for item in packet.get("required_acknowledgements", []))
    if required != REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS:
        reasons.append("required_acknowledgements_unexpected")
    unknown = sorted(set(acknowledgements) - set(REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS))
    for key in unknown:
        reasons.append(f"acknowledgement_unknown:{key}")
    for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS:
        if not bool(acknowledgements.get(key)):
            reasons.append(f"acknowledgement_missing_or_false:{key}")
    if approved_at:
        try:
            datetime.fromisoformat(clean(approved_at).replace("Z", "+00:00"))
        except ValueError:
            reasons.append("approved_at_invalid_iso")
    return sorted(set(reasons))


def build_human_approval_record(
    packet: Mapping[str, Any],
    packet_sha256: str,
    approved_by: str,
    approved_at: str,
    approval_phrase: str,
    acknowledgements: Mapping[str, bool],
    reason: str,
) -> Mapping[str, Any]:
    input_refs = mapping_or_empty(packet.get("input_refs"))
    workload = mapping_or_empty(packet.get("workload"))
    resources = mapping_or_empty(packet.get("batch_resource_limits"))
    return {
        "schema_version": ASR_WORKER_SANDBOX_HUMAN_APPROVAL_RECORD_SCHEMA_VERSION,
        "decision": "approved",
        "approval_packet_ref": clean(packet.get("approval_packet_ref")) or None,
        "approval_packet_sha256": packet_sha256,
        "approval_phrase": clean(approval_phrase),
        "approved_by": clean(approved_by),
        "approved_at": approved_at,
        "reason": clean(reason) or "stage24_human_sandbox_execution_approval",
        "selected_engine": clean(packet.get("selected_engine")) or None,
        "preflight_report_sha256": clean(input_refs.get("preflight_report_sha256")) or None,
        "contract_sha256": clean(input_refs.get("contract_sha256")) or None,
        "acknowledgements": {key: bool(acknowledgements.get(key)) for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS},
        "scope": {
            "job_type": ASR_JOB_TYPE,
            "allowed_task_count": optional_int(workload.get("tasks")) or 0,
            "audio_sha_ok": optional_int(workload.get("audio_sha_ok")) or 0,
            "output_collisions": optional_int(workload.get("output_collisions")) or 0,
            "total_audio_bytes": optional_int(resources.get("total_audio_bytes")) or 0,
            "total_duration_sec": float(resources.get("total_duration_sec") or 0.0),
            "estimated_tmp_bytes": optional_int(resources.get("estimated_tmp_bytes")) or 0,
            "estimated_timeout_sec": optional_int(resources.get("estimated_timeout_sec")) or 0,
            "valid_for_asr_sandbox_execution_request": True,
            "valid_for_immediate_worker_dispatch": False,
            "valid_for_runtime_db_writes": False,
            "valid_for_crm_writes": False,
        },
        "safety": human_approval_record_safety(writes_approval_record=True),
        "stage_contract": {
            "stage": "stage24_human_approval_record",
            "valid_for_next_stage_execution_request": True,
            "valid_for_asr_execution_in_stage24": False,
            "valid_for_immediate_worker_dispatch": False,
            "must_not_run_asr": True,
            "must_not_write_transcripts": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
            "must_not_touch_stable_runtime": True,
        },
    }


def validate_human_approval_record_payload(
    packet: Mapping[str, Any],
    packet_sha256: str,
    record: Mapping[str, Any],
    record_sha256: Optional[str],
) -> list[str]:
    reasons: list[str] = []
    if clean(record.get("schema_version")) != ASR_WORKER_SANDBOX_HUMAN_APPROVAL_RECORD_SCHEMA_VERSION:
        reasons.append("record_schema_unexpected")
    if clean(record.get("decision")) != "approved":
        reasons.append("record_decision_unexpected")
    if clean(record.get("approval_packet_ref")) != clean(packet.get("approval_packet_ref")):
        reasons.append("record_packet_ref_mismatch")
    if clean(record.get("approval_packet_sha256")) != packet_sha256:
        reasons.append("record_packet_sha_mismatch")
    if clean(record.get("approval_phrase")) != clean(packet.get("required_approval_phrase")):
        reasons.append("record_approval_phrase_mismatch")
    if not clean(record.get("approved_by")):
        reasons.append("record_approved_by_missing")
    if not parse_iso_datetime(clean(record.get("approved_at"))):
        reasons.append("record_approved_at_invalid")
    input_refs = mapping_or_empty(packet.get("input_refs"))
    if clean(record.get("preflight_report_sha256")) != clean(input_refs.get("preflight_report_sha256")):
        reasons.append("record_preflight_sha_mismatch")
    if clean(record.get("contract_sha256")) != clean(input_refs.get("contract_sha256")):
        reasons.append("record_contract_sha_mismatch")
    acknowledgements = mapping_or_empty(record.get("acknowledgements"))
    if set(acknowledgements) != set(REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS):
        reasons.append("record_acknowledgement_keys_mismatch")
    for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS:
        if not bool(acknowledgements.get(key)):
            reasons.append(f"record_acknowledgement_false:{key}")
    scope = mapping_or_empty(record.get("scope"))
    workload = mapping_or_empty(packet.get("workload"))
    if optional_int(scope.get("allowed_task_count")) != (optional_int(workload.get("tasks")) or 0):
        reasons.append("record_allowed_task_count_mismatch")
    if optional_int(scope.get("audio_sha_ok")) != (optional_int(workload.get("audio_sha_ok")) or 0):
        reasons.append("record_audio_sha_ok_mismatch")
    if optional_int(scope.get("output_collisions")):
        reasons.append("record_output_collisions_must_be_zero")
    if not bool(scope.get("valid_for_asr_sandbox_execution_request")):
        reasons.append("record_scope_execution_request_not_allowed")
    for key in ("valid_for_immediate_worker_dispatch", "valid_for_runtime_db_writes", "valid_for_crm_writes"):
        if bool(scope.get(key)):
            reasons.append(f"record_scope_{key}_must_be_false")
    safety = mapping_or_empty(record.get("safety"))
    for key in ("run_asr", "write_transcripts", "runtime_db_writes", "stable_runtime_writes", "write_crm", "write_tallanto", "dispatch_worker"):
        if bool(safety.get(key)):
            reasons.append(f"record_safety_{key}_must_be_false")
    stage_contract = mapping_or_empty(record.get("stage_contract"))
    if not bool(stage_contract.get("valid_for_next_stage_execution_request")):
        reasons.append("record_stage_contract_next_stage_not_allowed")
    for key in ("valid_for_asr_execution_in_stage24", "valid_for_immediate_worker_dispatch"):
        if bool(stage_contract.get(key)):
            reasons.append(f"record_stage_contract_{key}_must_be_false")
    if not record_sha256:
        reasons.append("record_sha_missing")
    return sorted(set(reasons))


def build_human_approval_report(
    product_root: Path,
    operation: str,
    approval_packet_path: Path,
    approval_record_path: Optional[Path],
    out_path: Optional[Path],
    packet: Mapping[str, Any],
    packet_sha256: str,
    packet_reasons: Sequence[str],
    approval_record: Optional[Mapping[str, Any]],
    approval_record_sha256: Optional[str],
    record_reasons: Sequence[str],
    written: int,
) -> Mapping[str, Any]:
    packet_valid = not packet_reasons
    record_valid = bool(approval_record) and not record_reasons
    ack_count = count_true_acknowledgements(approval_record)
    validation_ok = packet_valid if operation == "requirements" else packet_valid and record_valid
    action = action_for_report(operation=operation, packet_valid=packet_valid, record_valid=record_valid)
    summary = AsrWorkerSandboxHumanApprovalRecordSummary(
        schema_version=ASR_WORKER_SANDBOX_HUMAN_APPROVAL_WRITER_SCHEMA_VERSION,
        product_root=str(product_root),
        operation=operation,
        approval_packet_path=str(approval_packet_path),
        approval_record_path=str(approval_record_path) if approval_record_path else None,
        out_path=str(out_path) if out_path else None,
        approval_packet_sha256=packet_sha256,
        approval_record_sha256=approval_record_sha256,
        approval_packet_ref=clean(packet.get("approval_packet_ref")) or None,
        selected_engine=clean(packet.get("selected_engine")) or None,
        tasks=optional_int(mapping_or_empty(packet.get("workload")).get("tasks")) or 0,
        required_acknowledgements=len(REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS),
        acknowledgement_true_count=ack_count,
        approval_packet_valid=packet_valid,
        approval_record_present=bool(approval_record),
        approval_record_valid=record_valid,
        execution_approved=record_valid,
        dispatch_allowed=False,
        run_asr=False,
        write_transcripts=False,
        validation_ok=validation_ok,
        written=int(written),
        warnings=0 if validation_ok else 1,
    )
    actions = [
        {
            "action": action,
            "operation": operation,
            "approval_packet_valid": packet_valid,
            "approval_record_present": bool(approval_record),
            "approval_record_valid": record_valid,
            "execution_approved": record_valid,
            "dispatch_allowed": False,
            "run_asr": False,
            "technical_reasons": sorted(set(packet_reasons + list(record_reasons))),
        }
    ]
    return {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts_for(actions),
        "actions": actions,
        "approval": {
            "approval_packet_present": True,
            "approval_packet_valid": packet_valid,
            "approval_record_present": bool(approval_record),
            "approval_record_valid": record_valid,
            "approval_packet_ref": clean(packet.get("approval_packet_ref")) or None,
            "required_approval_phrase": clean(packet.get("required_approval_phrase")) or None,
            "required_acknowledgements": list(REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS),
            "missing_or_invalid_reasons": sorted(set(packet_reasons + list(record_reasons))),
            "execution_approved": record_valid,
            "dispatch_allowed": False,
            "run_asr": False,
        },
        "approval_record": approval_record,
        "source_approval_packet": {
            "schema_version": clean(packet.get("schema_version")) or None,
            "status": clean(packet.get("status")) or None,
            "approval_status": clean(packet.get("approval_status")) or None,
            "selected_engine": clean(packet.get("selected_engine")) or None,
            "sha256": packet_sha256,
        },
        "safety": human_approval_record_safety(writes_approval_record=bool(written)),
    }


def action_for_report(operation: str, packet_valid: bool, record_valid: bool) -> str:
    if operation == "requirements":
        return "REPORT_ASR_SANDBOX_HUMAN_APPROVAL_REQUIREMENTS" if packet_valid else "BLOCK_ASR_SANDBOX_HUMAN_APPROVAL_REQUIREMENTS"
    if operation == "write":
        return "WRITE_ASR_SANDBOX_HUMAN_APPROVAL_RECORD" if record_valid else "BLOCK_ASR_SANDBOX_HUMAN_APPROVAL_RECORD"
    if operation == "validate":
        return "VALIDATE_ASR_SANDBOX_HUMAN_APPROVAL_RECORD" if record_valid else "BLOCK_ASR_SANDBOX_HUMAN_APPROVAL_RECORD"
    return "BLOCK_ASR_SANDBOX_HUMAN_APPROVAL_RECORD"


def count_true_acknowledgements(record: Optional[Mapping[str, Any]]) -> int:
    if not record:
        return 0
    return sum(1 for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS if bool(mapping_or_empty(record.get("acknowledgements")).get(key)))


def human_approval_record_safety(writes_approval_record: bool) -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "reads_approval_packet": True,
        "writes_approval_record": bool(writes_approval_record),
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


def resolve_human_approval_paths(
    product_root: Path,
    approval_packet_path: Path,
    approval_record_path: Optional[Path],
    out_path: Optional[Path],
    require_record_exists: bool,
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "approval_packet_path": approval_packet_path.resolve(strict=False),
    }
    if approval_record_path is not None:
        paths["approval_record_path"] = approval_record_path.resolve(strict=False)
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_human_approval_paths(**paths, require_record_exists=require_record_exists)
    return paths


def guard_human_approval_paths(
    product_root: Path,
    approval_packet_path: Path,
    require_record_exists: bool,
    approval_record_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR sandbox approval packet", approval_packet_path),
        ("ASR sandbox human approval record", approval_record_path),
        ("ASR sandbox human approval audit", out_path),
    ):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not approval_packet_path.exists() or not approval_packet_path.is_file():
        raise FileNotFoundError(f"ASR sandbox approval packet not found: {approval_packet_path}")
    if require_record_exists:
        if approval_record_path is None:
            raise ValueError("approval_record_path is required for validation")
        if not approval_record_path.exists() or not approval_record_path.is_file():
            raise FileNotFoundError(f"ASR sandbox human approval record not found: {approval_record_path}")
    if out_path is not None and approval_record_path is not None and out_path == approval_record_path:
        raise ValueError("ASR sandbox human approval audit must differ from approval record")


def normalize_approved_at(value: Optional[str]) -> str:
    text = clean(value)
    if not text:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if not parse_iso_datetime(text):
        raise ValueError(f"approved_at must be ISO datetime: {value}")
    return text


def parse_iso_datetime(value: str) -> bool:
    if not clean(value):
        return False
    try:
        datetime.fromisoformat(clean(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(item.get("action")) for item in items).items()))
