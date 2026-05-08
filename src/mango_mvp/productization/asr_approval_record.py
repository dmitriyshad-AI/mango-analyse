from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE, required_approvals
from mango_mvp.productization.asr_scheduler_dry_run import (
    ASR_APPROVAL_RECORD_SCHEMA_VERSION,
    ASR_SCHEDULER_DRY_RUN_SCHEMA_VERSION,
    REQUIRED_ACKNOWLEDGEMENTS,
    evaluate_approval_record,
    load_json_object,
    mapping_or_empty,
    optional_int,
    pack_manifest_sha256,
    scheduler_dry_run_safety,
    write_json,
)
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_APPROVAL_RECORD_WRITER_SCHEMA_VERSION = "asr_approval_record_writer_v1"


@dataclass(frozen=True)
class AsrApprovalRecordSummary:
    schema_version: str
    product_root: str
    operation: str
    job_plan_path: str
    approval_path: str
    out_path: Optional[str]
    approval_ref: Optional[str]
    approved_by: Optional[str]
    job_plan_sha256: str
    approval_sha256: Optional[str]
    pack_manifest_sha256: Optional[str]
    ready_items: int
    approval_valid: bool
    validation_ok: bool
    written: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def write_asr_approval_record(
    product_root: Path,
    job_plan_path: Path,
    approval_path: Path,
    out_path: Optional[Path] = None,
    approval_ref: str = "",
    approved_by: str = "",
    approved_at: Optional[str] = None,
    reason: str = "",
    replace_existing: bool = False,
) -> Mapping[str, Any]:
    paths = resolve_approval_record_paths(
        product_root=product_root,
        job_plan_path=job_plan_path,
        approval_path=approval_path,
        out_path=out_path,
        require_approval_exists=False,
    )
    product_root = paths["product_root"]
    job_plan_path = paths["job_plan_path"]
    approval_path = paths["approval_path"]
    out_path = paths.get("out_path")

    approval_ref = clean(approval_ref)
    approved_by = clean(approved_by)
    approved_at = normalize_approved_at(approved_at)
    if not approval_ref:
        raise ValueError("approval_ref must not be empty")
    if not approved_by:
        raise ValueError("approved_by must not be empty")
    if approval_path.exists() and not replace_existing:
        raise FileExistsError(f"approval record already exists: {approval_path}")

    job_plan = load_json_object(job_plan_path)
    job_plan_sha = sha256_file(job_plan_path)
    record = build_approval_record(
        job_plan=job_plan,
        job_plan_path=job_plan_path,
        job_plan_sha256=job_plan_sha,
        approval_ref=approval_ref,
        approved_by=approved_by,
        approved_at=approved_at,
        reason=reason,
    )
    write_json(approval_path, record)
    report = build_approval_record_report(
        product_root=product_root,
        operation="write",
        job_plan_path=job_plan_path,
        approval_path=approval_path,
        out_path=out_path,
        job_plan=job_plan,
        job_plan_sha256=job_plan_sha,
        written=1,
    )
    if out_path:
        write_json(out_path, report)
    return report


def validate_asr_approval_record(
    product_root: Path,
    job_plan_path: Path,
    approval_path: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_approval_record_paths(
        product_root=product_root,
        job_plan_path=job_plan_path,
        approval_path=approval_path,
        out_path=out_path,
        require_approval_exists=True,
    )
    product_root = paths["product_root"]
    job_plan_path = paths["job_plan_path"]
    approval_path = paths["approval_path"]
    out_path = paths.get("out_path")

    job_plan = load_json_object(job_plan_path)
    report = build_approval_record_report(
        product_root=product_root,
        operation="validate",
        job_plan_path=job_plan_path,
        approval_path=approval_path,
        out_path=out_path,
        job_plan=job_plan,
        job_plan_sha256=sha256_file(job_plan_path),
        written=0,
    )
    if out_path:
        write_json(out_path, report)
    return report


def build_approval_record(
    job_plan: Mapping[str, Any],
    job_plan_path: Path,
    job_plan_sha256: str,
    approval_ref: str,
    approved_by: str,
    approved_at: str,
    reason: str,
) -> Mapping[str, Any]:
    workload = mapping_or_empty(job_plan.get("workload"))
    input_refs = mapping_or_empty(job_plan.get("input_refs"))
    return {
        "schema_version": ASR_APPROVAL_RECORD_SCHEMA_VERSION,
        "approval_ref": approval_ref,
        "decision": "approved",
        "approved_by": approved_by,
        "approved_at": approved_at,
        "reason": clean(reason) or "stage17_scheduler_dry_run_approval",
        "job_plan_sha256": job_plan_sha256,
        "pack_manifest_sha256": pack_manifest_sha256(job_plan),
        "approved_approvals": required_approvals(),
        "scope": {
            "job_type": ASR_JOB_TYPE,
            "allowed_item_count": optional_int(workload.get("ready_items")) or 0,
            "manifest_rows": optional_int(workload.get("manifest_rows")) or 0,
            "pack_audio_files": optional_int(workload.get("pack_audio_files")) or 0,
            "scheduler_dry_run_only": True,
            "execution_dispatch_allowed": False,
        },
        "acknowledgements": {key: True for key in REQUIRED_ACKNOWLEDGEMENTS},
        "source_refs": {
            "job_plan_path": str(job_plan_path),
            "pack_root": clean(input_refs.get("pack_root")) or None,
            "pack_manifest_path": clean(input_refs.get("pack_manifest_path")) or None,
        },
        "safety": scheduler_dry_run_safety(),
        "stage_contract": {
            "schema_version": ASR_SCHEDULER_DRY_RUN_SCHEMA_VERSION,
            "stage": "stage17_approval_record",
            "valid_for_scheduler_dry_run": True,
            "valid_for_asr_execution_dispatch": False,
            "must_not_run_asr": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
        },
    }


def build_approval_record_report(
    product_root: Path,
    operation: str,
    job_plan_path: Path,
    approval_path: Path,
    out_path: Optional[Path],
    job_plan: Mapping[str, Any],
    job_plan_sha256: str,
    written: int,
) -> Mapping[str, Any]:
    approval = evaluate_approval_record(
        approval_path=approval_path,
        job_plan=job_plan,
        job_plan_sha256=job_plan_sha256,
    )
    approval_sha = sha256_file(approval_path) if approval_path.exists() and approval_path.is_file() else None
    record_summary = approval.get("record_summary") if isinstance(approval.get("record_summary"), Mapping) else {}
    summary = AsrApprovalRecordSummary(
        schema_version=ASR_APPROVAL_RECORD_WRITER_SCHEMA_VERSION,
        product_root=str(product_root),
        operation=operation,
        job_plan_path=str(job_plan_path),
        approval_path=str(approval_path),
        out_path=str(out_path) if out_path else None,
        approval_ref=clean(record_summary.get("approval_ref")) or None,
        approved_by=clean(record_summary.get("approved_by")) or None,
        job_plan_sha256=job_plan_sha256,
        approval_sha256=approval_sha,
        pack_manifest_sha256=pack_manifest_sha256(job_plan),
        ready_items=optional_int(mapping_or_empty(job_plan.get("workload")).get("ready_items")) or 0,
        approval_valid=bool(approval.get("approval_valid")),
        validation_ok=bool(approval.get("approval_valid")),
        written=int(written),
        warnings=0,
    )
    return {
        "summary": summary.to_json_dict(),
        "approval": approval,
        "source_job_plan": {
            "schema_version": clean(job_plan.get("schema_version")) or None,
            "job_type": clean(job_plan.get("job_type")) or None,
            "mode": clean(job_plan.get("mode")) or None,
            "status": clean(job_plan.get("status")) or None,
            "execution_allowed": bool(job_plan.get("execution_allowed")),
            "sha256": job_plan_sha256,
        },
        "safety": approval_record_safety(writes_approval_record=bool(written)),
    }


def approval_record_safety(writes_approval_record: bool) -> Mapping[str, bool]:
    return {
        "reads_job_plan": True,
        "writes_approval_record": bool(writes_approval_record),
        "writes_audit_json": True,
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


def resolve_approval_record_paths(
    product_root: Path,
    job_plan_path: Path,
    approval_path: Path,
    out_path: Optional[Path],
    require_approval_exists: bool,
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "job_plan_path": job_plan_path.resolve(strict=False),
        "approval_path": approval_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_approval_record_paths(**paths, require_approval_exists=require_approval_exists)
    return paths


def guard_approval_record_paths(
    product_root: Path,
    job_plan_path: Path,
    approval_path: Path,
    require_approval_exists: bool,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR job plan", job_plan_path),
        ("ASR approval record", approval_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not job_plan_path.exists() or not job_plan_path.is_file():
        raise FileNotFoundError(f"ASR job plan not found: {job_plan_path}")
    if require_approval_exists and (not approval_path.exists() or not approval_path.is_file()):
        raise FileNotFoundError(f"ASR approval record not found: {approval_path}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR approval audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR approval audit must stay under product root: {product_root}")


def normalize_approved_at(value: Optional[str]) -> str:
    text = clean(value)
    if not text:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"approved_at must be ISO datetime: {value}") from exc
    return text
