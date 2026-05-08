from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_execution_approval_gate import (
    build_asr_execution_approval_gate,
    required_approvals,
)
from mango_mvp.productization.asr_scheduler_dry_run import (
    ASR_APPROVAL_RECORD_SCHEMA_VERSION,
    REQUIRED_ACKNOWLEDGEMENTS,
    build_asr_scheduler_dry_run,
)
from mango_mvp.productization.asr_worker_pack_verifier import verify_asr_worker_pack
from mango_mvp.productization.recording_asset_ingest import sha256_file
from scripts import mango_office_asr_scheduler_dry_run
from tests.test_productization_asr_worker_pack_verifier import build_pack


def test_asr_scheduler_dry_run_blocks_without_approval(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=2)
    out_dir = product_root / "asr_scheduler_dry_run_stage16"
    scheduler_plan = out_dir / "scheduler_plan.json"

    report = build_asr_scheduler_dry_run(
        product_root=product_root,
        job_plan_path=job_plan,
        out_dir=out_dir,
        scheduler_plan_path=scheduler_plan,
        out_path=out_dir / "audit.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["ready_items"] == 2
    assert report["summary"]["approval_present"] is False
    assert report["summary"]["approval_valid"] is False
    assert report["summary"]["pending_approval"] == 1
    assert report["summary"]["scheduler_may_dispatch"] is False
    assert report["summary"]["execution_allowed"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SCHEDULER_PENDING_APPROVAL": 1}
    assert report["scheduler_plan"]["status"] == "blocked_pending_approval"
    assert report["safety"]["run_asr"] is False
    assert scheduler_plan.exists()


def test_asr_scheduler_dry_run_accepts_valid_approval_but_still_does_not_dispatch(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    approval = write_valid_approval(product_root, job_plan)
    out_dir = product_root / "asr_scheduler_dry_run_stage16"

    report = build_asr_scheduler_dry_run(
        product_root=product_root,
        job_plan_path=job_plan,
        out_dir=out_dir,
        scheduler_plan_path=out_dir / "scheduler_plan.json",
        approval_path=approval,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["approval_present"] is True
    assert report["summary"]["approval_valid"] is True
    assert report["summary"]["pending_approval"] == 0
    assert report["summary"]["invalid_approval"] == 0
    assert report["summary"]["scheduler_may_dispatch"] is False
    assert report["action_counts"] == {"PLAN_ASR_SCHEDULER_APPROVED_DRY_RUN": 1}
    assert report["scheduler_plan"]["status"] == "approved_dry_run_not_dispatched"


def test_asr_scheduler_dry_run_blocks_invalid_approval(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    approval = write_valid_approval(product_root, job_plan)
    data = json.loads(approval.read_text(encoding="utf-8"))
    data["job_plan_sha256"] = "bad-sha"
    approval.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = product_root / "asr_scheduler_dry_run_stage16"

    report = build_asr_scheduler_dry_run(
        product_root=product_root,
        job_plan_path=job_plan,
        out_dir=out_dir,
        scheduler_plan_path=out_dir / "scheduler_plan.json",
        approval_path=approval,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["invalid_approval"] == 1
    assert report["action_counts"] == {"BLOCK_ASR_SCHEDULER_INVALID_APPROVAL": 1}
    assert "approval_job_plan_sha_mismatch" in report["approval"]["reasons"]
    assert report["scheduler_plan"]["scheduler_may_dispatch"] is False


def test_asr_scheduler_dry_run_blocks_unsafe_job_plan(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    data = json.loads(job_plan.read_text(encoding="utf-8"))
    data["hard_guards"]["run_asr"] = True
    job_plan.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = product_root / "asr_scheduler_dry_run_stage16"

    report = build_asr_scheduler_dry_run(
        product_root=product_root,
        job_plan_path=job_plan,
        out_dir=out_dir,
        scheduler_plan_path=out_dir / "scheduler_plan.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["technical_blocked"] >= 1
    assert report["action_counts"] == {"BLOCK_ASR_SCHEDULER_JOB_PLAN_NOT_READY": 1}
    assert "job_plan_hard_guard_run_asr_must_be_false" in report["actions"][0]["technical_reasons"]


def test_asr_scheduler_dry_run_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_scheduler_dry_run(
            product_root=product_root,
            job_plan_path=job_plan,
            out_dir=tmp_path / "outside",
            scheduler_plan_path=tmp_path / "outside" / "scheduler_plan.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_scheduler_dry_run(
            product_root=product_root,
            job_plan_path=product_root / "stable_runtime" / "job_plan.json",
            out_dir=product_root / "asr_scheduler_dry_run_stage16",
            scheduler_plan_path=product_root / "asr_scheduler_dry_run_stage16" / "scheduler_plan.json",
        )
    with pytest.raises(ValueError, match="output directory"):
        build_asr_scheduler_dry_run(
            product_root=product_root,
            job_plan_path=job_plan,
            out_dir=product_root / "asr_scheduler_dry_run_stage16",
            scheduler_plan_path=product_root / "asr_scheduler_dry_run_stage16" / "scheduler_plan.json",
            out_path=product_root / "other_stage" / "audit.json",
        )
    with pytest.raises(ValueError, match="approval record"):
        build_asr_scheduler_dry_run(
            product_root=product_root,
            job_plan_path=job_plan,
            out_dir=product_root / "asr_scheduler_dry_run_stage16",
            scheduler_plan_path=product_root / "asr_scheduler_dry_run_stage16" / "scheduler_plan.json",
            approval_path=tmp_path / "approval.json",
        )


def test_asr_scheduler_dry_run_cli_writes_audit(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    out_dir = product_root / "asr_scheduler_dry_run_stage16"
    out = out_dir / "audit.json"
    scheduler_plan = out_dir / "scheduler_plan.json"

    rc = mango_office_asr_scheduler_dry_run.main(
        [
            "--product-root",
            str(product_root),
            "--job-plan",
            str(job_plan),
            "--out-dir",
            str(out_dir),
            "--scheduler-plan",
            str(scheduler_plan),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["pending_approval"] == 1
    assert data["safety"]["scheduler_dispatch"] is False
    assert scheduler_plan.exists()


def build_stage15_job_plan(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=count)
    verify_audit = pack_root / "verify.json"
    verify_report = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=verify_audit,
    )
    assert verify_report["summary"]["validation_ok"] is True
    out_dir = product_root / "asr_execution_approval_stage15"
    job_plan = out_dir / "asr_execution_job_plan.json"
    gate_report = build_asr_execution_approval_gate(
        product_root=product_root,
        verify_audit_path=verify_audit,
        out_dir=out_dir,
        job_plan_path=job_plan,
        out_path=out_dir / "audit.json",
    )
    assert gate_report["summary"]["validation_ok"] is True
    return product_root, job_plan


def write_valid_approval(product_root: Path, job_plan: Path) -> Path:
    job_plan_data = json.loads(job_plan.read_text(encoding="utf-8"))
    approval = product_root / "approvals" / "asr_approval.json"
    approval.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": ASR_APPROVAL_RECORD_SCHEMA_VERSION,
        "approval_ref": "manual-test-approval",
        "decision": "approved",
        "approved_by": "test-operator",
        "approved_at": "2026-05-07T12:00:00+00:00",
        "job_plan_sha256": sha256_file(job_plan),
        "pack_manifest_sha256": job_plan_data["input_refs"]["pack_manifest_sha256"],
        "approved_approvals": required_approvals(),
        "scope": {
            "job_type": "asr_execution",
            "allowed_item_count": job_plan_data["workload"]["ready_items"],
        },
        "acknowledgements": {key: True for key in REQUIRED_ACKNOWLEDGEMENTS},
    }
    approval.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return approval
