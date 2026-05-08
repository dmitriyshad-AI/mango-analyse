from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_execution_approval_gate import build_asr_execution_approval_gate
from mango_mvp.productization.asr_worker_pack_verifier import verify_asr_worker_pack
from scripts import mango_office_asr_execution_approval_gate
from tests.test_productization_asr_worker_pack_verifier import build_pack


def test_asr_execution_approval_gate_blocks_without_explicit_approval(tmp_path: Path) -> None:
    product_root, verify_audit = build_verify_audit(tmp_path, count=2)
    out_dir = product_root / "asr_execution_approval_stage15"
    job_plan = out_dir / "asr_execution_job_plan.json"

    report = build_asr_execution_approval_gate(
        product_root=product_root,
        verify_audit_path=verify_audit,
        out_dir=out_dir,
        job_plan_path=job_plan,
        out_path=out_dir / "audit.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["readiness_ok"] is True
    assert report["summary"]["approval_required"] is True
    assert report["summary"]["approval_present"] is False
    assert report["summary"]["execution_allowed"] is False
    assert report["summary"]["approval_blocked"] == 1
    assert report["summary"]["technical_blocked"] == 0
    assert report["action_counts"] == {"BLOCK_ASR_EXECUTION_PENDING_APPROVAL": 1}
    assert report["job_plan"]["status"] == "blocked_pending_explicit_approval"
    assert report["job_plan"]["hard_guards"]["run_asr"] is False
    assert job_plan.exists()


def test_asr_execution_approval_gate_is_idempotent(tmp_path: Path) -> None:
    product_root, verify_audit = build_verify_audit(tmp_path, count=1)
    out_dir = product_root / "asr_execution_approval_stage15"
    job_plan = out_dir / "asr_execution_job_plan.json"

    first = build_asr_execution_approval_gate(
        product_root=product_root,
        verify_audit_path=verify_audit,
        out_dir=out_dir,
        job_plan_path=job_plan,
        out_path=out_dir / "audit.json",
    )
    second = build_asr_execution_approval_gate(
        product_root=product_root,
        verify_audit_path=verify_audit,
        out_dir=out_dir,
        job_plan_path=job_plan,
        out_path=out_dir / "audit_idempotency.json",
    )

    assert first["summary"]["job_plan_sha256"] == second["summary"]["job_plan_sha256"]
    assert second["summary"]["approval_blocked"] == 1


def test_asr_execution_approval_gate_records_approval_ref_but_does_not_allow_execution(tmp_path: Path) -> None:
    product_root, verify_audit = build_verify_audit(tmp_path, count=1)
    out_dir = product_root / "asr_execution_approval_stage15"

    report = build_asr_execution_approval_gate(
        product_root=product_root,
        verify_audit_path=verify_audit,
        out_dir=out_dir,
        job_plan_path=out_dir / "asr_execution_job_plan.json",
        approval_ref="manual-test-approval",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["approval_present"] is True
    assert report["summary"]["approval_blocked"] == 0
    assert report["summary"]["execution_allowed"] is False
    assert report["action_counts"] == {"PLAN_ASR_EXECUTION_APPROVAL_RECORDED_DRY_RUN": 1}
    assert report["job_plan"]["status"] == "approval_recorded_dry_run"
    assert report["job_plan"]["hard_guards"]["run_asr"] is False


def test_asr_execution_approval_gate_blocks_failed_verify_audit(tmp_path: Path) -> None:
    product_root, verify_audit = build_verify_audit(tmp_path, count=1)
    data = json.loads(verify_audit.read_text(encoding="utf-8"))
    data["summary"]["validation_ok"] = False
    data["summary"]["blocked"] = 1
    verify_audit.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_asr_execution_approval_gate(
        product_root=product_root,
        verify_audit_path=verify_audit,
        out_dir=product_root / "asr_execution_approval_stage15",
        job_plan_path=product_root / "asr_execution_approval_stage15" / "asr_execution_job_plan.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["readiness_ok"] is False
    assert report["summary"]["technical_blocked"] >= 1
    assert report["summary"]["approval_blocked"] == 0
    assert report["action_counts"] == {"BLOCK_ASR_EXECUTION_PACK_NOT_READY": 1}
    assert report["job_plan"]["status"] == "blocked_pack_not_ready"


def test_asr_execution_approval_gate_refuses_outside_and_stable_runtime_paths(tmp_path: Path) -> None:
    product_root, verify_audit = build_verify_audit(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_execution_approval_gate(
            product_root=product_root,
            verify_audit_path=verify_audit,
            out_dir=tmp_path / "outside",
            job_plan_path=tmp_path / "outside" / "job.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_execution_approval_gate(
            product_root=product_root,
            verify_audit_path=product_root / "stable_runtime" / "verify.json",
            out_dir=product_root / "asr_execution_approval_stage15",
            job_plan_path=product_root / "asr_execution_approval_stage15" / "job.json",
        )
    with pytest.raises(ValueError, match="approval gate output directory"):
        build_asr_execution_approval_gate(
            product_root=product_root,
            verify_audit_path=verify_audit,
            out_dir=product_root / "asr_execution_approval_stage15",
            job_plan_path=product_root / "asr_execution_approval_stage15" / "job.json",
            out_path=product_root / "other_stage" / "audit.json",
        )


def test_asr_execution_approval_gate_script_writes_report(tmp_path: Path) -> None:
    product_root, verify_audit = build_verify_audit(tmp_path, count=1)
    out_dir = product_root / "asr_execution_approval_stage15"
    out = out_dir / "audit.json"
    job_plan = out_dir / "asr_execution_job_plan.json"

    rc = mango_office_asr_execution_approval_gate.main(
        [
            "--product-root",
            str(product_root),
            "--verify-audit",
            str(verify_audit),
            "--out-dir",
            str(out_dir),
            "--job-plan",
            str(job_plan),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["approval_blocked"] == 1
    assert data["safety"]["run_asr"] is False
    assert data["safety"]["runtime_db_writes"] is False
    assert job_plan.exists()


def build_verify_audit(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=count)
    verify_audit = pack_root / "verify.json"
    report = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=verify_audit,
    )
    assert report["summary"]["validation_ok"] is True
    return product_root, verify_audit
