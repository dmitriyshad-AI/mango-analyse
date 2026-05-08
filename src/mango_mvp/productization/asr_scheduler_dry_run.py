from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import (
    ASR_EXECUTION_APPROVAL_SCHEMA_VERSION,
    ASR_JOB_TYPE,
    required_approvals,
)
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_SCHEDULER_DRY_RUN_SCHEMA_VERSION = "asr_scheduler_dry_run_v1"
ASR_APPROVAL_RECORD_SCHEMA_VERSION = "asr_execution_approval_record_v1"

DANGEROUS_HARD_GUARDS = (
    "download_audio",
    "copy_audio",
    "run_asr",
    "run_ra",
    "write_runtime_db",
    "write_product_db",
    "write_asset_db",
    "write_crm",
    "write_tallanto",
    "touch_stable_runtime",
)
REQUIRED_ACKNOWLEDGEMENTS = (
    "explicit_asr_execution",
    "runtime_target_db_selected",
    "stable_runtime_write_policy_acknowledged",
    "asr_worker_resource_approved",
    "no_crm_or_tallanto_writes",
)


@dataclass(frozen=True)
class AsrSchedulerDryRunSummary:
    schema_version: str
    product_root: str
    job_plan_path: str
    out_dir: str
    scheduler_plan_path: str
    approval_path: Optional[str]
    job_plan_sha256: str
    scheduler_plan_sha256: str
    pack_manifest_sha256: Optional[str]
    ready_items: int
    approval_present: bool
    approval_valid: bool
    pending_approval: int
    invalid_approval: int
    technical_blocked: int
    scheduler_may_dispatch: bool
    execution_allowed: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_scheduler_dry_run(
    product_root: Path,
    job_plan_path: Path,
    out_dir: Path,
    scheduler_plan_path: Path,
    out_path: Optional[Path] = None,
    approval_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_scheduler_dry_run_paths(
        product_root=product_root,
        job_plan_path=job_plan_path,
        out_dir=out_dir,
        scheduler_plan_path=scheduler_plan_path,
        out_path=out_path,
        approval_path=approval_path,
    )
    product_root = paths["product_root"]
    job_plan_path = paths["job_plan_path"]
    out_dir = paths["out_dir"]
    scheduler_plan_path = paths["scheduler_plan_path"]
    out_path = paths.get("out_path")
    approval_path = paths.get("approval_path")

    job_plan = load_json_object(job_plan_path)
    job_plan_sha = sha256_file(job_plan_path)
    technical_reasons = validate_source_job_plan(job_plan)
    approval = evaluate_approval_record(
        approval_path=approval_path,
        job_plan=job_plan,
        job_plan_sha256=job_plan_sha,
    )
    actions = build_scheduler_actions(
        technical_reasons=technical_reasons,
        approval_present=bool(approval["approval_present"]),
        approval_valid=bool(approval["approval_valid"]),
        approval_reasons=approval["reasons"],
    )
    scheduler_plan = build_scheduler_plan(
        job_plan_path=job_plan_path,
        job_plan=job_plan,
        job_plan_sha256=job_plan_sha,
        approval=approval,
        status=planned_status(actions),
    )
    write_json(scheduler_plan_path, scheduler_plan)
    scheduler_plan_sha = sha256_file(scheduler_plan_path)

    action_counts = action_counts_for(actions)
    technical_blocked = len(technical_reasons)
    approval_present = bool(approval["approval_present"])
    approval_valid = bool(approval["approval_valid"])
    pending_approval = int(action_counts.get("BLOCK_ASR_SCHEDULER_PENDING_APPROVAL") or 0)
    invalid_approval = int(action_counts.get("BLOCK_ASR_SCHEDULER_INVALID_APPROVAL") or 0)
    validation_ok = technical_blocked == 0 and invalid_approval == 0
    warnings = pending_approval
    summary = AsrSchedulerDryRunSummary(
        schema_version=ASR_SCHEDULER_DRY_RUN_SCHEMA_VERSION,
        product_root=str(product_root),
        job_plan_path=str(job_plan_path),
        out_dir=str(out_dir),
        scheduler_plan_path=str(scheduler_plan_path),
        approval_path=str(approval_path) if approval_path else None,
        job_plan_sha256=job_plan_sha,
        scheduler_plan_sha256=scheduler_plan_sha,
        pack_manifest_sha256=pack_manifest_sha256(job_plan),
        ready_items=optional_int(mapping_or_empty(job_plan.get("workload")).get("ready_items")) or 0,
        approval_present=approval_present,
        approval_valid=approval_valid,
        pending_approval=pending_approval,
        invalid_approval=invalid_approval,
        technical_blocked=technical_blocked,
        scheduler_may_dispatch=False,
        execution_allowed=False,
        validation_ok=validation_ok,
        warnings=warnings,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts,
        "actions": actions,
        "scheduler_plan": scheduler_plan,
        "source_job_plan": {
            "schema_version": clean(job_plan.get("schema_version")) or None,
            "job_type": clean(job_plan.get("job_type")) or None,
            "mode": clean(job_plan.get("mode")) or None,
            "status": clean(job_plan.get("status")) or None,
            "execution_allowed": bool(job_plan.get("execution_allowed")),
            "sha256": job_plan_sha,
        },
        "approval": approval,
        "approval_contract": approval_record_contract(job_plan, job_plan_sha),
        "safety": scheduler_dry_run_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def validate_source_job_plan(job_plan: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if clean(job_plan.get("schema_version")) != ASR_EXECUTION_APPROVAL_SCHEMA_VERSION:
        reasons.append("job_plan_schema_unexpected")
    if clean(job_plan.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("job_plan_type_unexpected")
    if clean(job_plan.get("mode")) != "approval_gate_dry_run":
        reasons.append("job_plan_mode_unexpected")
    if bool(job_plan.get("execution_allowed")):
        reasons.append("job_plan_must_not_allow_execution")
    status = clean(job_plan.get("status"))
    if status not in {"blocked_pending_explicit_approval", "approval_recorded_dry_run"}:
        reasons.append("job_plan_status_unexpected")

    input_refs = job_plan.get("input_refs") if isinstance(job_plan.get("input_refs"), Mapping) else {}
    if not clean(input_refs.get("pack_manifest_sha256")):
        reasons.append("job_plan_pack_manifest_sha_missing")

    workload = job_plan.get("workload") if isinstance(job_plan.get("workload"), Mapping) else {}
    manifest_rows = optional_int(workload.get("manifest_rows")) or 0
    ready_items = optional_int(workload.get("ready_items")) or 0
    if manifest_rows <= 0:
        reasons.append("job_plan_manifest_empty")
    if ready_items != manifest_rows:
        reasons.append("job_plan_ready_items_do_not_match_manifest")

    readiness = job_plan.get("readiness_gate") if isinstance(job_plan.get("readiness_gate"), Mapping) else {}
    if not bool(readiness.get("ready_for_worker")):
        reasons.append("job_plan_not_ready_for_worker")
    if bool(readiness.get("worker_may_run_asr")):
        reasons.append("job_plan_readiness_must_not_allow_asr")
    if not bool(readiness.get("requires_explicit_runtime_target_approval")):
        reasons.append("job_plan_missing_runtime_target_approval_requirement")

    hard_guards = job_plan.get("hard_guards") if isinstance(job_plan.get("hard_guards"), Mapping) else {}
    if not hard_guards:
        reasons.append("job_plan_hard_guards_missing")
    for guard in DANGEROUS_HARD_GUARDS:
        if bool(hard_guards.get(guard)):
            reasons.append(f"job_plan_hard_guard_{guard}_must_be_false")

    declared_approvals = sequence_of_clean_strings(job_plan.get("required_approvals"))
    missing_approvals = sorted(set(required_approvals()) - set(declared_approvals))
    if missing_approvals:
        reasons.append("job_plan_required_approvals_missing")
    return sorted(set(reasons))


def evaluate_approval_record(
    approval_path: Optional[Path],
    job_plan: Mapping[str, Any],
    job_plan_sha256: str,
) -> Mapping[str, Any]:
    if approval_path is None:
        return {
            "approval_present": False,
            "approval_valid": False,
            "approval_path": None,
            "approval_ref": None,
            "reasons": [],
            "record_summary": None,
        }
    if not approval_path.exists() or not approval_path.is_file():
        return invalid_approval(approval_path, ["approval_record_not_found"], record=None)
    try:
        record = load_json_object(approval_path)
    except (json.JSONDecodeError, ValueError) as exc:
        return invalid_approval(approval_path, [f"approval_record_invalid_json:{clean(str(exc))}"], record=None)

    reasons: list[str] = []
    if clean(record.get("schema_version")) != ASR_APPROVAL_RECORD_SCHEMA_VERSION:
        reasons.append("approval_schema_unexpected")
    if clean(record.get("decision")) != "approved":
        reasons.append("approval_decision_not_approved")
    if not clean(record.get("approval_ref")):
        reasons.append("approval_ref_missing")
    if not clean(record.get("approved_by")):
        reasons.append("approved_by_missing")
    if not parse_iso_datetime(clean(record.get("approved_at"))):
        reasons.append("approved_at_invalid")
    if clean(record.get("job_plan_sha256")) != job_plan_sha256:
        reasons.append("approval_job_plan_sha_mismatch")
    if clean(record.get("pack_manifest_sha256")) != pack_manifest_sha256(job_plan):
        reasons.append("approval_pack_manifest_sha_mismatch")

    approved_approvals = sequence_of_clean_strings(record.get("approved_approvals"))
    missing_approved_approvals = sorted(set(required_approvals()) - set(approved_approvals))
    if missing_approved_approvals:
        reasons.append("approval_required_approvals_missing")

    acknowledgements = record.get("acknowledgements") if isinstance(record.get("acknowledgements"), Mapping) else {}
    missing_acknowledgements = sorted(key for key in REQUIRED_ACKNOWLEDGEMENTS if not bool(acknowledgements.get(key)))
    if missing_acknowledgements:
        reasons.append("approval_acknowledgements_missing")

    scope = record.get("scope") if isinstance(record.get("scope"), Mapping) else {}
    allowed_item_count = optional_int(scope.get("allowed_item_count"))
    ready_items = optional_int(mapping_or_empty(job_plan.get("workload")).get("ready_items")) or 0
    if allowed_item_count is None:
        reasons.append("approval_scope_allowed_item_count_missing")
    elif allowed_item_count != ready_items:
        reasons.append("approval_scope_allowed_item_count_mismatch")

    reasons = sorted(set(reasons))
    return {
        "approval_present": True,
        "approval_valid": not reasons,
        "approval_path": str(approval_path),
        "approval_ref": clean(record.get("approval_ref")) or None,
        "reasons": reasons,
        "record_summary": summarize_approval_record(record),
    }


def invalid_approval(approval_path: Path, reasons: Sequence[str], record: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "approval_present": True,
        "approval_valid": False,
        "approval_path": str(approval_path),
        "approval_ref": clean(record.get("approval_ref")) if record else None,
        "reasons": list(reasons),
        "record_summary": summarize_approval_record(record) if record else None,
    }


def summarize_approval_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": clean(record.get("schema_version")) or None,
        "approval_ref": clean(record.get("approval_ref")) or None,
        "decision": clean(record.get("decision")) or None,
        "approved_by": clean(record.get("approved_by")) or None,
        "approved_at": clean(record.get("approved_at")) or None,
        "job_plan_sha256": clean(record.get("job_plan_sha256")) or None,
        "pack_manifest_sha256": clean(record.get("pack_manifest_sha256")) or None,
    }


def build_scheduler_actions(
    technical_reasons: Sequence[str],
    approval_present: bool,
    approval_valid: bool,
    approval_reasons: Sequence[str],
) -> list[Mapping[str, Any]]:
    if technical_reasons:
        return [
            {
                "action": "BLOCK_ASR_SCHEDULER_JOB_PLAN_NOT_READY",
                "reason": "source_job_plan_failed_scheduler_readiness",
                "technical_reasons": list(technical_reasons),
                "scheduler_may_dispatch": False,
                "execution_allowed": False,
            }
        ]
    if not approval_present:
        return [
            {
                "action": "BLOCK_ASR_SCHEDULER_PENDING_APPROVAL",
                "reason": "approval_record_required_before_dispatch",
                "required_approvals": required_approvals(),
                "scheduler_may_dispatch": False,
                "execution_allowed": False,
            }
        ]
    if not approval_valid:
        return [
            {
                "action": "BLOCK_ASR_SCHEDULER_INVALID_APPROVAL",
                "reason": "approval_record_failed_contract_validation",
                "approval_reasons": list(approval_reasons),
                "scheduler_may_dispatch": False,
                "execution_allowed": False,
            }
        ]
    return [
        {
            "action": "PLAN_ASR_SCHEDULER_APPROVED_DRY_RUN",
            "reason": "approval_record_valid_but_stage16_never_dispatches_execution",
            "scheduler_may_dispatch": False,
            "execution_allowed": False,
        }
    ]


def build_scheduler_plan(
    job_plan_path: Path,
    job_plan: Mapping[str, Any],
    job_plan_sha256: str,
    approval: Mapping[str, Any],
    status: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": ASR_SCHEDULER_DRY_RUN_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "scheduler_approval_dry_run",
        "status": status,
        "scheduler_may_dispatch": False,
        "execution_allowed": False,
        "approval_ref": approval.get("approval_ref"),
        "input_refs": {
            "job_plan_path": str(job_plan_path),
            "job_plan_sha256": job_plan_sha256,
            "pack_manifest_sha256": pack_manifest_sha256(job_plan),
            "pack_root": clean(mapping_or_empty(job_plan.get("input_refs")).get("pack_root")) or None,
            "pack_manifest_path": clean(mapping_or_empty(job_plan.get("input_refs")).get("pack_manifest_path")) or None,
        },
        "workload": {
            "manifest_rows": optional_int(mapping_or_empty(job_plan.get("workload")).get("manifest_rows")) or 0,
            "ready_items": optional_int(mapping_or_empty(job_plan.get("workload")).get("ready_items")) or 0,
            "pack_audio_files": optional_int(mapping_or_empty(job_plan.get("workload")).get("pack_audio_files")) or 0,
            "pack_total_bytes": optional_int(mapping_or_empty(job_plan.get("workload")).get("pack_total_bytes")) or 0,
        },
        "approval_gate": {
            "approval_present": bool(approval.get("approval_present")),
            "approval_valid": bool(approval.get("approval_valid")),
            "approval_path": approval.get("approval_path"),
            "approval_reasons": approval.get("reasons") or [],
        },
        "hard_guards": {guard: False for guard in DANGEROUS_HARD_GUARDS},
        "next_stage_contract": {
            "may_record_approval": True,
            "may_create_execution_plan_after_valid_approval": True,
            "may_dispatch_asr_in_this_stage": False,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
        },
    }


def approval_record_contract(job_plan: Mapping[str, Any], job_plan_sha256: str) -> Mapping[str, Any]:
    return {
        "schema_version": ASR_APPROVAL_RECORD_SCHEMA_VERSION,
        "template_only": True,
        "valid_for_execution": False,
        "required_fields": [
            "schema_version",
            "approval_ref",
            "decision",
            "approved_by",
            "approved_at",
            "job_plan_sha256",
            "pack_manifest_sha256",
            "approved_approvals",
            "scope.allowed_item_count",
            "acknowledgements",
        ],
        "expected_values": {
            "decision": "approved",
            "job_plan_sha256": job_plan_sha256,
            "pack_manifest_sha256": pack_manifest_sha256(job_plan),
            "approved_approvals": required_approvals(),
            "scope": {
                "job_type": ASR_JOB_TYPE,
                "allowed_item_count": optional_int(mapping_or_empty(job_plan.get("workload")).get("ready_items")) or 0,
            },
            "acknowledgements": {key: True for key in REQUIRED_ACKNOWLEDGEMENTS},
        },
    }


def planned_status(actions: Sequence[Mapping[str, Any]]) -> str:
    action = clean(actions[0].get("action")) if actions else ""
    if action == "BLOCK_ASR_SCHEDULER_JOB_PLAN_NOT_READY":
        return "blocked_job_plan_not_ready"
    if action == "BLOCK_ASR_SCHEDULER_PENDING_APPROVAL":
        return "blocked_pending_approval"
    if action == "BLOCK_ASR_SCHEDULER_INVALID_APPROVAL":
        return "blocked_invalid_approval"
    return "approved_dry_run_not_dispatched"


def scheduler_dry_run_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
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
        "scheduler_dispatch": False,
    }


def resolve_scheduler_dry_run_paths(
    product_root: Path,
    job_plan_path: Path,
    out_dir: Path,
    scheduler_plan_path: Path,
    out_path: Optional[Path],
    approval_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "job_plan_path": job_plan_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "scheduler_plan_path": scheduler_plan_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    if approval_path is not None:
        paths["approval_path"] = approval_path.resolve(strict=False)
    guard_scheduler_dry_run_paths(**paths)
    return paths


def guard_scheduler_dry_run_paths(
    product_root: Path,
    job_plan_path: Path,
    out_dir: Path,
    scheduler_plan_path: Path,
    out_path: Optional[Path] = None,
    approval_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR job plan", job_plan_path),
        ("ASR scheduler dry-run output directory", out_dir),
        ("ASR scheduler dry-run plan", scheduler_plan_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not job_plan_path.exists() or not job_plan_path.is_file():
        raise FileNotFoundError(f"ASR job plan not found: {job_plan_path}")
    if not path_is_relative_to(scheduler_plan_path, out_dir):
        raise ValueError(f"ASR scheduler dry-run plan must stay under output directory: {out_dir}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR scheduler dry-run audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR scheduler dry-run audit must stay under product root: {product_root}")
        if not path_is_relative_to(out_path, out_dir):
            raise ValueError(f"ASR scheduler dry-run audit must stay under output directory: {out_dir}")
    if approval_path is not None:
        if "stable_runtime" in approval_path.parts:
            raise ValueError("refusing ASR approval record under stable_runtime")
        if not path_is_relative_to(approval_path, product_root):
            raise ValueError(f"ASR approval record must stay under product root: {product_root}")


def load_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(actions: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(action.get("action")) for action in actions).items()))


def sequence_of_clean_strings(value: Any) -> list[str]:
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        return []
    return [text for text in (clean(item) for item in value) if text]


def mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def pack_manifest_sha256(job_plan: Mapping[str, Any]) -> Optional[str]:
    input_refs = job_plan.get("input_refs") if isinstance(job_plan.get("input_refs"), Mapping) else {}
    return clean(input_refs.get("pack_manifest_sha256")) or None


def parse_iso_datetime(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def optional_int(value: Any) -> Optional[int]:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None
