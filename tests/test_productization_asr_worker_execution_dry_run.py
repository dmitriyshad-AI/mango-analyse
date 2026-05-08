from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_execution_plan import build_asr_execution_plan
from mango_mvp.productization.asr_worker_execution_dry_run import build_asr_worker_execution_dry_run
from scripts import mango_office_asr_worker_dry_run
from tests.test_productization_asr_execution_plan import build_stage17_approved_scheduler_plan


def test_asr_worker_dry_run_builds_command_envelopes(tmp_path: Path) -> None:
    product_root, execution_plan = build_stage18_execution_plan(tmp_path, count=2)
    out_dir = product_root / "asr_worker_dry_run_stage19"
    worker_plan = out_dir / "worker_plan.json"

    report = build_asr_worker_execution_dry_run(
        product_root=product_root,
        execution_plan_path=execution_plan,
        out_dir=out_dir,
        worker_plan_path=worker_plan,
        out_path=out_dir / "audit.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["source_items"] == 2
    assert report["summary"]["envelopes"] == 2
    assert report["summary"]["blocked_envelopes"] == 0
    assert report["summary"]["dispatch_allowed"] is False
    assert report["summary"]["run_asr"] is False
    assert report["action_counts"] == {"PLAN_ASR_WORKER_ENVELOPE": 2}
    assert report["worker_plan"]["status"] == "worker_envelopes_planned_not_dispatched"
    assert report["worker_plan"]["write_outputs"] is False
    assert report["worker_plan"]["hard_guards"]["run_asr"] is False
    first = report["worker_plan"]["worker_envelopes"][0]
    assert first["worker_command_envelope"]["executable"] is None
    assert first["worker_command_envelope"]["argv"] == []
    assert first["worker_command_envelope"]["dispatch_allowed"] is False
    assert first["worker_command_envelope"]["run_asr"] is False
    assert first["resource_estimate"]["estimated_timeout_sec"] >= 60
    assert worker_plan.exists()


def test_asr_worker_dry_run_blocks_bad_execution_plan_status(tmp_path: Path) -> None:
    product_root, execution_plan = build_stage18_execution_plan(tmp_path, count=1)
    data = json.loads(execution_plan.read_text(encoding="utf-8"))
    data["status"] = "planned_and_dispatched"
    data["run_asr"] = True
    execution_plan.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_asr_worker_execution_dry_run(
        product_root=product_root,
        execution_plan_path=execution_plan,
        out_dir=product_root / "asr_worker_dry_run_stage19",
        worker_plan_path=product_root / "asr_worker_dry_run_stage19" / "worker_plan.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["technical_blocked"] >= 1
    assert report["action_counts"] == {"BLOCK_ASR_WORKER_DRY_RUN": 1}
    reasons = report["worker_plan"]["actions"][0]["technical_reasons"]
    assert "execution_plan_status_unexpected" in reasons
    assert "execution_plan_must_not_run_asr" in reasons
    assert report["worker_plan"]["run_asr"] is False


def test_asr_worker_dry_run_blocks_duplicate_outputs(tmp_path: Path) -> None:
    product_root, execution_plan = build_stage18_execution_plan(tmp_path, count=2)
    data = json.loads(execution_plan.read_text(encoding="utf-8"))
    duplicate = data["items"][0]["planned_output_paths"]["transcript_json"]
    data["items"][1]["planned_output_paths"]["transcript_json"] = duplicate
    execution_plan.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_asr_worker_execution_dry_run(
        product_root=product_root,
        execution_plan_path=execution_plan,
        out_dir=product_root / "asr_worker_dry_run_stage19",
        worker_plan_path=product_root / "asr_worker_dry_run_stage19" / "worker_plan.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked_envelopes"] == 2
    assert report["action_counts"] == {"BLOCK_ASR_WORKER_ENVELOPE": 2}
    assert "duplicate_output_path" in report["worker_plan"]["worker_envelopes"][0]["blocked_reasons"]
    assert report["worker_plan"]["dispatch_allowed"] is False


def test_asr_worker_dry_run_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, execution_plan = build_stage18_execution_plan(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_worker_execution_dry_run(
            product_root=product_root,
            execution_plan_path=execution_plan,
            out_dir=tmp_path / "outside",
            worker_plan_path=tmp_path / "outside" / "worker_plan.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_execution_dry_run(
            product_root=product_root,
            execution_plan_path=product_root / "stable_runtime" / "execution_plan.json",
            out_dir=product_root / "asr_worker_dry_run_stage19",
            worker_plan_path=product_root / "asr_worker_dry_run_stage19" / "worker_plan.json",
        )
    with pytest.raises(ValueError, match="output directory"):
        build_asr_worker_execution_dry_run(
            product_root=product_root,
            execution_plan_path=execution_plan,
            out_dir=product_root / "asr_worker_dry_run_stage19",
            worker_plan_path=product_root / "asr_worker_dry_run_stage19" / "worker_plan.json",
            out_path=product_root / "other_stage" / "audit.json",
        )


def test_asr_worker_dry_run_cli_writes_audit(tmp_path: Path) -> None:
    product_root, execution_plan = build_stage18_execution_plan(tmp_path, count=1)
    out_dir = product_root / "asr_worker_dry_run_stage19"
    out = out_dir / "audit.json"
    worker_plan = out_dir / "worker_plan.json"

    rc = mango_office_asr_worker_dry_run.main(
        [
            "--product-root",
            str(product_root),
            "--execution-plan",
            str(execution_plan),
            "--out-dir",
            str(out_dir),
            "--worker-plan",
            str(worker_plan),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["envelopes"] == 1
    assert data["safety"]["run_asr"] is False
    assert data["safety"]["write_transcripts"] is False
    assert worker_plan.exists()


def build_stage18_execution_plan(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, scheduler_plan = build_stage17_approved_scheduler_plan(tmp_path, count=count)
    out_dir = product_root / "asr_execution_plan_stage18"
    execution_plan = out_dir / "execution_plan.json"
    report = build_asr_execution_plan(
        product_root=product_root,
        scheduler_plan_path=scheduler_plan,
        out_dir=out_dir,
        execution_plan_path=execution_plan,
        out_path=out_dir / "audit.json",
    )
    assert report["summary"]["validation_ok"] is True
    return product_root, execution_plan
