from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_approval_record import write_asr_approval_record
from mango_mvp.productization.asr_execution_plan import build_asr_execution_plan
from mango_mvp.productization.asr_scheduler_dry_run import build_asr_scheduler_dry_run
from mango_mvp.productization.recording_asset_ingest import sha256_file
from scripts import mango_office_asr_execution_plan
from tests.test_productization_asr_scheduler_dry_run import build_stage15_job_plan


def test_asr_execution_plan_expands_approved_scheduler_plan(tmp_path: Path) -> None:
    product_root, scheduler_plan = build_stage17_approved_scheduler_plan(tmp_path, count=2)
    out_dir = product_root / "asr_execution_plan_stage18"
    execution_plan = out_dir / "asr_execution_plan.json"

    report = build_asr_execution_plan(
        product_root=product_root,
        scheduler_plan_path=scheduler_plan,
        out_dir=out_dir,
        execution_plan_path=execution_plan,
        out_path=out_dir / "audit.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["manifest_rows"] == 2
    assert report["summary"]["planned_items"] == 2
    assert report["summary"]["blocked_items"] == 0
    assert report["summary"]["technical_blocked"] == 0
    assert report["summary"]["run_asr"] is False
    assert report["summary"]["scheduler_dispatch"] is False
    assert report["summary"]["execution_allowed"] is False
    assert report["action_counts"] == {"PLAN_ASR_EXECUTION_ITEM": 2}
    assert report["execution_plan"]["status"] == "planned_not_dispatched"
    assert report["execution_plan"]["hard_guards"]["run_asr"] is False
    assert len(report["execution_plan"]["items"]) == 2
    assert execution_plan.exists()


def test_asr_execution_plan_blocks_pending_scheduler_plan(tmp_path: Path) -> None:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=1)
    out_dir = product_root / "asr_scheduler_dry_run_stage16"
    scheduler_plan = out_dir / "scheduler_plan.json"
    build_asr_scheduler_dry_run(
        product_root=product_root,
        job_plan_path=job_plan,
        out_dir=out_dir,
        scheduler_plan_path=scheduler_plan,
    )

    report = build_asr_execution_plan(
        product_root=product_root,
        scheduler_plan_path=scheduler_plan,
        out_dir=product_root / "asr_execution_plan_stage18",
        execution_plan_path=product_root / "asr_execution_plan_stage18" / "execution_plan.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["technical_blocked"] >= 1
    assert report["action_counts"] == {"BLOCK_ASR_EXECUTION_PLAN": 1}
    assert "scheduler_plan_not_approved_dry_run" in report["execution_plan"]["actions"][0]["technical_reasons"]
    assert report["execution_plan"]["execution_allowed"] is False


def test_asr_execution_plan_blocks_manifest_sha_mismatch(tmp_path: Path) -> None:
    product_root, scheduler_plan = build_stage17_approved_scheduler_plan(tmp_path, count=1)
    data = json.loads(scheduler_plan.read_text(encoding="utf-8"))
    data["input_refs"]["pack_manifest_sha256"] = "bad-sha"
    scheduler_plan.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_asr_execution_plan(
        product_root=product_root,
        scheduler_plan_path=scheduler_plan,
        out_dir=product_root / "asr_execution_plan_stage18",
        execution_plan_path=product_root / "asr_execution_plan_stage18" / "execution_plan.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert "pack_manifest_sha_mismatch" in report["execution_plan"]["actions"][0]["technical_reasons"]
    assert report["summary"]["run_asr"] is False


def test_asr_execution_plan_blocks_bad_manifest_item_checksum(tmp_path: Path) -> None:
    product_root, scheduler_plan = build_stage17_approved_scheduler_plan(tmp_path, count=1)
    scheduler = json.loads(scheduler_plan.read_text(encoding="utf-8"))
    manifest = Path(scheduler["input_refs"]["pack_manifest_path"])
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    row["audio_sha256"] = "0" * 64
    manifest.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    scheduler["input_refs"]["pack_manifest_sha256"] = sha256_file(manifest)
    scheduler_plan.write_text(json.dumps(scheduler, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_asr_execution_plan(
        product_root=product_root,
        scheduler_plan_path=scheduler_plan,
        out_dir=product_root / "asr_execution_plan_stage18",
        execution_plan_path=product_root / "asr_execution_plan_stage18" / "execution_plan.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked_items"] == 1
    assert report["action_counts"] == {"BLOCK_ASR_EXECUTION_PLAN_ITEM": 1}
    assert "audio_sha256_mismatch" in report["execution_plan"]["items"][0]["blocked_reasons"]


def test_asr_execution_plan_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, scheduler_plan = build_stage17_approved_scheduler_plan(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_execution_plan(
            product_root=product_root,
            scheduler_plan_path=scheduler_plan,
            out_dir=tmp_path / "outside",
            execution_plan_path=tmp_path / "outside" / "execution_plan.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_execution_plan(
            product_root=product_root,
            scheduler_plan_path=product_root / "stable_runtime" / "scheduler_plan.json",
            out_dir=product_root / "asr_execution_plan_stage18",
            execution_plan_path=product_root / "asr_execution_plan_stage18" / "execution_plan.json",
        )
    with pytest.raises(ValueError, match="output directory"):
        build_asr_execution_plan(
            product_root=product_root,
            scheduler_plan_path=scheduler_plan,
            out_dir=product_root / "asr_execution_plan_stage18",
            execution_plan_path=product_root / "asr_execution_plan_stage18" / "execution_plan.json",
            out_path=product_root / "other_stage" / "audit.json",
        )


def test_asr_execution_plan_cli_writes_audit(tmp_path: Path) -> None:
    product_root, scheduler_plan = build_stage17_approved_scheduler_plan(tmp_path, count=1)
    out_dir = product_root / "asr_execution_plan_stage18"
    out = out_dir / "audit.json"
    execution_plan = out_dir / "execution_plan.json"

    rc = mango_office_asr_execution_plan.main(
        [
            "--product-root",
            str(product_root),
            "--scheduler-plan",
            str(scheduler_plan),
            "--out-dir",
            str(out_dir),
            "--execution-plan",
            str(execution_plan),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["planned_items"] == 1
    assert data["safety"]["run_asr"] is False
    assert execution_plan.exists()


def build_stage17_approved_scheduler_plan(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, job_plan = build_stage15_job_plan(tmp_path, count=count)
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
    scheduler_plan = out_dir / "scheduler_plan.json"
    report = build_asr_scheduler_dry_run(
        product_root=product_root,
        job_plan_path=job_plan,
        out_dir=out_dir,
        scheduler_plan_path=scheduler_plan,
        approval_path=approval,
    )
    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["approval_valid"] is True
    return product_root, scheduler_plan
