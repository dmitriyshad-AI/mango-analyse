from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_worker_sandbox_readiness import build_asr_worker_sandbox_readiness
from scripts import mango_office_asr_worker_sandbox_readiness
from tests.test_productization_asr_worker_execution_dry_run import build_stage18_execution_plan
from mango_mvp.productization.asr_worker_execution_dry_run import build_asr_worker_execution_dry_run


def test_asr_worker_sandbox_readiness_reports_ready_when_real_engine_detected(tmp_path: Path) -> None:
    product_root, worker_plan = build_stage19_worker_plan(tmp_path, count=2)
    out_dir = product_root / "asr_worker_sandbox_readiness_stage20"
    readiness = out_dir / "readiness.json"

    report = build_asr_worker_sandbox_readiness(
        product_root=product_root,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        readiness_report_path=readiness,
        out_path=out_dir / "audit.json",
        env={"TRANSCRIBE_PROVIDER": "mlx", "MLX_WHISPER_MODEL": "test-model"},
        module_checker=lambda name: name == "mlx_whisper",
        binary_checker=lambda _name: None,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["source_plan_valid"] is True
    assert report["summary"]["worker_sandbox_ready"] is True
    assert report["summary"]["asr_engine_ready"] is True
    assert report["summary"]["ready_real_engines"] == 1
    assert report["summary"]["envelopes"] == 2
    assert report["summary"]["run_asr"] is False
    assert report["action_counts"] == {"PLAN_ASR_WORKER_SANDBOX_READY_DRY_RUN": 1}
    assert report["readiness_report"]["status"] == "sandbox_ready_dry_run"
    assert report["readiness_report"]["hard_guards"]["run_asr"] is False
    assert readiness.exists()


def test_asr_worker_sandbox_readiness_blocks_missing_real_engine_without_failing_source_validation(tmp_path: Path) -> None:
    product_root, worker_plan = build_stage19_worker_plan(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_readiness_stage20"

    report = build_asr_worker_sandbox_readiness(
        product_root=product_root,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        readiness_report_path=out_dir / "readiness.json",
        env={"TRANSCRIBE_PROVIDER": "mock"},
        module_checker=lambda _name: False,
        binary_checker=lambda _name: None,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["source_plan_valid"] is True
    assert report["summary"]["worker_sandbox_ready"] is False
    assert report["summary"]["asr_engine_ready"] is False
    assert report["summary"]["warnings"] == 1
    assert report["action_counts"] == {"BLOCK_ASR_WORKER_SANDBOX_MISSING_ENGINE": 1}
    assert report["readiness_report"]["status"] == "blocked_missing_asr_engine"
    assert report["safety"]["imports_asr_modules"] is False
    assert report["safety"]["loads_models"] is False


def test_asr_worker_sandbox_readiness_blocks_invalid_worker_plan(tmp_path: Path) -> None:
    product_root, worker_plan = build_stage19_worker_plan(tmp_path, count=1)
    data = json.loads(worker_plan.read_text(encoding="utf-8"))
    data["run_asr"] = True
    data["hard_guards"]["run_asr"] = True
    worker_plan.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_dir = product_root / "asr_worker_sandbox_readiness_stage20"

    report = build_asr_worker_sandbox_readiness(
        product_root=product_root,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        readiness_report_path=out_dir / "readiness.json",
        env={"TRANSCRIBE_PROVIDER": "mlx"},
        module_checker=lambda name: name == "mlx_whisper",
        binary_checker=lambda _name: None,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["source_plan_valid"] is False
    assert report["summary"]["worker_sandbox_ready"] is False
    assert report["action_counts"] == {"BLOCK_ASR_WORKER_SANDBOX_SOURCE_PLAN": 1}
    reasons = report["readiness_report"]["actions"][0]["technical_reasons"]
    assert "worker_plan_run_asr_must_be_false" in reasons
    assert "worker_plan_hard_guard_run_asr_must_be_false" in reasons


def test_asr_worker_sandbox_readiness_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, worker_plan = build_stage19_worker_plan(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_worker_sandbox_readiness(
            product_root=product_root,
            worker_plan_path=worker_plan,
            out_dir=tmp_path / "outside",
            readiness_report_path=tmp_path / "outside" / "readiness.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_sandbox_readiness(
            product_root=product_root,
            worker_plan_path=product_root / "stable_runtime" / "worker_plan.json",
            out_dir=product_root / "asr_worker_sandbox_readiness_stage20",
            readiness_report_path=product_root / "asr_worker_sandbox_readiness_stage20" / "readiness.json",
        )
    with pytest.raises(ValueError, match="output directory"):
        build_asr_worker_sandbox_readiness(
            product_root=product_root,
            worker_plan_path=worker_plan,
            out_dir=product_root / "asr_worker_sandbox_readiness_stage20",
            readiness_report_path=product_root / "asr_worker_sandbox_readiness_stage20" / "readiness.json",
            out_path=product_root / "other_stage" / "audit.json",
        )


def test_asr_worker_sandbox_readiness_cli_writes_audit(tmp_path: Path) -> None:
    product_root, worker_plan = build_stage19_worker_plan(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_readiness_stage20"
    out = out_dir / "audit.json"
    readiness = out_dir / "readiness.json"

    rc = mango_office_asr_worker_sandbox_readiness.main(
        [
            "--product-root",
            str(product_root),
            "--worker-plan",
            str(worker_plan),
            "--out-dir",
            str(out_dir),
            "--readiness-report",
            str(readiness),
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
    assert readiness.exists()


def build_stage19_worker_plan(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, execution_plan = build_stage18_execution_plan(tmp_path, count=count)
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
    return product_root, worker_plan
