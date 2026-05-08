from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_worker_pack_verifier import ASR_WORKER_PACK_VERIFY_SCHEMA_VERSION
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_EXECUTION_APPROVAL_SCHEMA_VERSION = "asr_execution_approval_gate_v1"
ASR_JOB_TYPE = "asr_execution"


@dataclass(frozen=True)
class AsrExecutionApprovalSummary:
    schema_version: str
    product_root: str
    verify_audit_path: str
    out_dir: str
    job_plan_path: str
    source_manifest_rows: int
    ready_items: int
    readiness_ok: bool
    approval_required: bool
    approval_present: bool
    execution_allowed: bool
    approval_blocked: int
    technical_blocked: int
    warnings: int
    job_plan_sha256: str
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_execution_approval_gate(
    product_root: Path,
    verify_audit_path: Path,
    out_dir: Path,
    job_plan_path: Path,
    out_path: Optional[Path] = None,
    approval_ref: Optional[str] = None,
) -> Mapping[str, Any]:
    paths = resolve_gate_paths(
        product_root=product_root,
        verify_audit_path=verify_audit_path,
        out_dir=out_dir,
        job_plan_path=job_plan_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    verify_audit_path = paths["verify_audit_path"]
    out_dir = paths["out_dir"]
    job_plan_path = paths["job_plan_path"]
    out_path = paths.get("out_path")
    approval_ref = clean(approval_ref) or None

    verify_report = json.loads(verify_audit_path.read_text(encoding="utf-8"))
    verify_summary = verify_report.get("summary") if isinstance(verify_report.get("summary"), Mapping) else {}
    readiness = verify_report.get("readiness_gate") if isinstance(verify_report.get("readiness_gate"), Mapping) else {}
    safety = verify_report.get("safety") if isinstance(verify_report.get("safety"), Mapping) else {}

    technical_reasons = verify_technical_blockers(verify_summary=verify_summary, readiness=readiness, safety=safety)
    approval_required = True
    approval_present = approval_ref is not None
    readiness_ok = not technical_reasons
    execution_allowed = False
    actions = build_actions(
        readiness_ok=readiness_ok,
        technical_reasons=technical_reasons,
        approval_present=approval_present,
        approval_ref=approval_ref,
    )
    job_plan = build_job_plan(
        verify_audit_path=verify_audit_path,
        verify_summary=verify_summary,
        readiness=readiness,
        approval_ref=approval_ref,
        status=planned_status(actions),
    )
    write_json(job_plan_path, job_plan)
    job_plan_sha = sha256_file(job_plan_path)
    action_counts = action_counts_for(actions)
    technical_blocked = len(technical_reasons)
    approval_blocked = int(action_counts.get("BLOCK_ASR_EXECUTION_PENDING_APPROVAL") or 0)
    warnings = 0 if approval_present else 1
    summary = AsrExecutionApprovalSummary(
        schema_version=ASR_EXECUTION_APPROVAL_SCHEMA_VERSION,
        product_root=str(product_root),
        verify_audit_path=str(verify_audit_path),
        out_dir=str(out_dir),
        job_plan_path=str(job_plan_path),
        source_manifest_rows=optional_int(verify_summary.get("manifest_rows")) or 0,
        ready_items=optional_int(verify_summary.get("ready_items")) or 0,
        readiness_ok=readiness_ok,
        approval_required=approval_required,
        approval_present=approval_present,
        execution_allowed=execution_allowed,
        approval_blocked=approval_blocked,
        technical_blocked=technical_blocked,
        warnings=warnings,
        job_plan_sha256=job_plan_sha,
        validation_ok=technical_blocked == 0,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts,
        "actions": actions,
        "job_plan": job_plan,
        "source_verify_summary": dict(verify_summary),
        "source_readiness_gate": dict(readiness),
        "approval_gate": {
            "execution_allowed": False,
            "approval_required": True,
            "approval_present": approval_present,
            "approval_ref": approval_ref,
            "required_approvals": required_approvals(),
            "next_allowed_step": "record_explicit_approval_and_build_execution_plan" if readiness_ok else "repair_worker_pack_and_verify_again",
        },
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def verify_technical_blockers(
    verify_summary: Mapping[str, Any],
    readiness: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if clean(verify_summary.get("schema_version")) != ASR_WORKER_PACK_VERIFY_SCHEMA_VERSION:
        reasons.append("verify_audit_schema_unexpected")
    if not bool(verify_summary.get("validation_ok")):
        reasons.append("verify_audit_validation_not_ok")
    if optional_int(verify_summary.get("blocked")):
        reasons.append("verify_audit_has_blocked_items")
    manifest_rows = optional_int(verify_summary.get("manifest_rows")) or 0
    ready_items = optional_int(verify_summary.get("ready_items")) or 0
    if manifest_rows <= 0:
        reasons.append("verify_audit_manifest_empty")
    if ready_items != manifest_rows:
        reasons.append("ready_items_do_not_match_manifest_rows")
    if not bool(readiness.get("ready_for_worker")):
        reasons.append("readiness_gate_not_ready_for_worker")
    if bool(readiness.get("worker_may_run_asr")):
        reasons.append("readiness_gate_unexpectedly_allows_asr")
    if not bool(readiness.get("requires_explicit_runtime_target_approval")):
        reasons.append("readiness_gate_missing_runtime_approval_requirement")
    for key in ("runtime_db_writes", "stable_runtime_writes", "run_asr", "run_ra", "write_crm", "write_tallanto"):
        if bool(safety.get(key)):
            reasons.append(f"source_safety_{key}_must_be_false")
    if not bool(safety.get("read_only")):
        reasons.append("source_safety_read_only_missing")
    return sorted(set(reasons))


def build_actions(
    readiness_ok: bool,
    technical_reasons: Sequence[str],
    approval_present: bool,
    approval_ref: Optional[str],
) -> list[Mapping[str, Any]]:
    if not readiness_ok:
        return [
            {
                "action": "BLOCK_ASR_EXECUTION_PACK_NOT_READY",
                "reason": "technical_readiness_gate_failed",
                "technical_reasons": list(technical_reasons),
                "execution_allowed": False,
            }
        ]
    if not approval_present:
        return [
            {
                "action": "BLOCK_ASR_EXECUTION_PENDING_APPROVAL",
                "reason": "explicit_runtime_target_approval_required",
                "required_approvals": required_approvals(),
                "execution_allowed": False,
            }
        ]
    return [
        {
            "action": "PLAN_ASR_EXECUTION_APPROVAL_RECORDED_DRY_RUN",
            "reason": "approval_ref_recorded_but_execution_not_run_in_stage15",
            "approval_ref": approval_ref,
            "execution_allowed": False,
        }
    ]


def build_job_plan(
    verify_audit_path: Path,
    verify_summary: Mapping[str, Any],
    readiness: Mapping[str, Any],
    approval_ref: Optional[str],
    status: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": ASR_EXECUTION_APPROVAL_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "approval_gate_dry_run",
        "status": status,
        "execution_allowed": False,
        "approval_ref": approval_ref,
        "input_refs": {
            "verify_audit_path": str(verify_audit_path),
            "pack_root": clean(verify_summary.get("pack_root")) or None,
            "pack_manifest_path": clean(verify_summary.get("pack_manifest_path")) or None,
            "pack_manifest_sha256": clean(verify_summary.get("manifest_sha256")) or None,
        },
        "workload": {
            "manifest_rows": optional_int(verify_summary.get("manifest_rows")) or 0,
            "ready_items": optional_int(verify_summary.get("ready_items")) or 0,
            "pack_audio_files": optional_int(verify_summary.get("pack_audio_files")) or 0,
            "pack_total_bytes": optional_int(verify_summary.get("pack_total_bytes")) or 0,
        },
        "readiness_gate": {
            "ready_for_worker": bool(readiness.get("ready_for_worker")),
            "worker_may_run_asr": False,
            "requires_explicit_runtime_target_approval": True,
        },
        "required_approvals": required_approvals(),
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
        },
        "next_stage_contract": {
            "may_build_execution_plan_after_approval": True,
            "may_run_asr_in_this_stage": False,
            "must_verify_pack_before_execution": True,
        },
    }


def planned_status(actions: Sequence[Mapping[str, Any]]) -> str:
    action = clean(actions[0].get("action")) if actions else ""
    if action == "BLOCK_ASR_EXECUTION_PACK_NOT_READY":
        return "blocked_pack_not_ready"
    if action == "BLOCK_ASR_EXECUTION_PENDING_APPROVAL":
        return "blocked_pending_explicit_approval"
    return "approval_recorded_dry_run"


def required_approvals() -> list[str]:
    return [
        "explicit_asr_execution_approval",
        "runtime_target_db_approval",
        "stable_runtime_write_policy_acknowledgement",
        "asr_worker_resource_approval",
    ]


def safety_contract() -> Mapping[str, bool]:
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
    }


def resolve_gate_paths(
    product_root: Path,
    verify_audit_path: Path,
    out_dir: Path,
    job_plan_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "verify_audit_path": verify_audit_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "job_plan_path": job_plan_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_gate_paths(**paths)
    return paths


def guard_gate_paths(
    product_root: Path,
    verify_audit_path: Path,
    out_dir: Path,
    job_plan_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("verify audit", verify_audit_path),
        ("approval gate output directory", out_dir),
        ("approval gate job plan", job_plan_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not verify_audit_path.exists() or not verify_audit_path.is_file():
        raise FileNotFoundError(f"verify audit not found: {verify_audit_path}")
    if not path_is_relative_to(job_plan_path, out_dir):
        raise ValueError(f"job plan must stay under approval gate output directory: {out_dir}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing approval gate audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"approval gate audit must stay under product root: {product_root}")
        if not path_is_relative_to(out_path, out_dir):
            raise ValueError(f"approval gate audit must stay under approval gate output directory: {out_dir}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(actions: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(action.get("action")) for action in actions).items()))


def optional_int(value: Any) -> Optional[int]:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None
