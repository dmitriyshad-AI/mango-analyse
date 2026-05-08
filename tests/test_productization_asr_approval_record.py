from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_approval_record import (
    validate_asr_approval_record,
    write_asr_approval_record,
)
from mango_mvp.productization.asr_scheduler_dry_run import build_asr_scheduler_dry_run
from scripts import mango_office_asr_approval_record
from tests.test_productization_asr_scheduler_dry_run import build_stage15_job_plan


def test_asr_approval_record_writer_creates_valid_dry_run_record(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=2)
    approval = product_root / "asr_approval_record_stage17" / "approval.json"
    out = product_root / "asr_approval_record_stage17" / "audit.json"

    report = write_asr_approval_record(
        product_root=product_root,
        job_plan_path=job_plan,
        approval_path=approval,
        out_path=out,
        approval_ref="stage17-test-approval",
        approved_by="test-operator",
        approved_at="2026-05-07T19:20:00+00:00",
        reason="unit-test",
    )

    record = json.loads(approval.read_text(encoding="utf-8"))
    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["written"] == 1
    assert report["summary"]["approval_valid"] is True
    assert record["decision"] == "approved"
    assert record["scope"]["scheduler_dry_run_only"] is True
    assert record["scope"]["execution_dispatch_allowed"] is False
    assert record["stage_contract"]["valid_for_asr_execution_dispatch"] is False
    assert record["safety"]["run_asr"] is False
    assert out.exists()


def test_asr_approval_record_validator_accepts_written_record(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    approval = product_root / "asr_approval_record_stage17" / "approval.json"
    write_asr_approval_record(
        product_root=product_root,
        job_plan_path=job_plan,
        approval_path=approval,
        approval_ref="stage17-test-approval",
        approved_by="test-operator",
        approved_at="2026-05-07T19:20:00+00:00",
    )

    report = validate_asr_approval_record(
        product_root=product_root,
        job_plan_path=job_plan,
        approval_path=approval,
        out_path=product_root / "asr_approval_record_stage17" / "validation.json",
    )

    assert report["summary"]["operation"] == "validate"
    assert report["summary"]["written"] == 0
    assert report["summary"]["approval_valid"] is True
    assert report["approval"]["reasons"] == []
    assert report["safety"]["writes_approval_record"] is False


def test_asr_approval_record_unlocks_scheduler_approved_dry_run_only(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    approval = product_root / "asr_approval_record_stage17" / "approval.json"
    write_asr_approval_record(
        product_root=product_root,
        job_plan_path=job_plan,
        approval_path=approval,
        approval_ref="stage17-test-approval",
        approved_by="test-operator",
        approved_at="2026-05-07T19:20:00+00:00",
    )
    out_dir = product_root / "asr_scheduler_approved_dry_run_stage17"

    report = build_asr_scheduler_dry_run(
        product_root=product_root,
        job_plan_path=job_plan,
        out_dir=out_dir,
        scheduler_plan_path=out_dir / "scheduler_plan.json",
        approval_path=approval,
    )

    assert report["action_counts"] == {"PLAN_ASR_SCHEDULER_APPROVED_DRY_RUN": 1}
    assert report["summary"]["approval_valid"] is True
    assert report["summary"]["pending_approval"] == 0
    assert report["summary"]["scheduler_may_dispatch"] is False
    assert report["summary"]["execution_allowed"] is False


def test_asr_approval_record_writer_refuses_overwrite_by_default(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    approval = product_root / "asr_approval_record_stage17" / "approval.json"
    kwargs = {
        "product_root": product_root,
        "job_plan_path": job_plan,
        "approval_path": approval,
        "approval_ref": "stage17-test-approval",
        "approved_by": "test-operator",
        "approved_at": "2026-05-07T19:20:00+00:00",
    }
    write_asr_approval_record(**kwargs)

    with pytest.raises(FileExistsError, match="already exists"):
        write_asr_approval_record(**kwargs)


def test_asr_approval_record_writer_refuses_unsafe_paths_and_bad_metadata(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)

    with pytest.raises(ValueError, match="approval_ref"):
        write_asr_approval_record(
            product_root=product_root,
            job_plan_path=job_plan,
            approval_path=product_root / "asr_approval_record_stage17" / "approval.json",
            approval_ref="",
            approved_by="test-operator",
        )
    with pytest.raises(ValueError, match="product root"):
        write_asr_approval_record(
            product_root=product_root,
            job_plan_path=job_plan,
            approval_path=tmp_path / "outside" / "approval.json",
            approval_ref="stage17-test-approval",
            approved_by="test-operator",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        write_asr_approval_record(
            product_root=product_root,
            job_plan_path=job_plan,
            approval_path=product_root / "stable_runtime" / "approval.json",
            approval_ref="stage17-test-approval",
            approved_by="test-operator",
        )


def test_asr_approval_record_cli_write_and_validate(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    approval = product_root / "asr_approval_record_stage17" / "approval.json"
    write_out = product_root / "asr_approval_record_stage17" / "audit.json"
    validate_out = product_root / "asr_approval_record_stage17" / "validation.json"

    write_rc = mango_office_asr_approval_record.main(
        [
            "--product-root",
            str(product_root),
            "--job-plan",
            str(job_plan),
            "write",
            "--approval",
            str(approval),
            "--out",
            str(write_out),
            "--approval-ref",
            "stage17-test-approval",
            "--approved-by",
            "test-operator",
            "--approved-at",
            "2026-05-07T19:20:00+00:00",
        ]
    )
    validate_rc = mango_office_asr_approval_record.main(
        [
            "--product-root",
            str(product_root),
            "--job-plan",
            str(job_plan),
            "validate",
            "--approval",
            str(approval),
            "--out",
            str(validate_out),
        ]
    )

    assert write_rc == 0
    assert validate_rc == 0
    assert json.loads(write_out.read_text(encoding="utf-8"))["summary"]["approval_valid"] is True
    assert json.loads(validate_out.read_text(encoding="utf-8"))["summary"]["approval_valid"] is True
    assert json.loads(validate_out.read_text(encoding="utf-8"))["safety"]["writes_approval_record"] is False
