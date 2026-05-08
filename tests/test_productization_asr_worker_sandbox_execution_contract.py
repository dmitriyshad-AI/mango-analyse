from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_worker_sandbox_execution_contract import build_asr_worker_sandbox_execution_contract
from mango_mvp.productization.asr_worker_sandbox_readiness import build_asr_worker_sandbox_readiness
from scripts import mango_office_asr_worker_sandbox_contract
from tests.test_productization_asr_worker_sandbox_readiness import build_stage19_worker_plan


def test_asr_worker_sandbox_contract_builds_dry_run_tasks(tmp_path: Path) -> None:
    product_root, readiness_report, worker_plan = build_stage20_readiness(tmp_path, count=2)
    out_dir = product_root / "asr_worker_sandbox_contract_stage21"
    contract = out_dir / "contract.json"

    report = build_asr_worker_sandbox_execution_contract(
        product_root=product_root,
        readiness_report_path=readiness_report,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        contract_path=contract,
        out_path=out_dir / "audit.json",
        preferred_engine="mlx",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["selected_engine"] == "mlx"
    assert report["summary"]["tasks"] == 2
    assert report["summary"]["blocked_tasks"] == 0
    assert report["summary"]["dispatch_allowed"] is False
    assert report["summary"]["run_asr"] is False
    assert report["summary"]["write_transcripts"] is False
    assert report["action_counts"] == {"PLAN_ASR_SANDBOX_TASK": 2}
    assert report["contract"]["status"] == "sandbox_execution_contract_planned_not_dispatched"
    assert report["contract"]["write_outputs"] is False
    assert report["contract"]["hard_guards"]["run_asr"] is False
    assert report["contract"]["hard_guards"]["write_transcripts"] is False
    first = report["contract"]["tasks"][0]
    assert first["worker_command_contract"]["executable"] is None
    assert first["worker_command_contract"]["argv"] == []
    assert first["worker_command_contract"]["run_asr"] is False
    assert first["worker_command_contract"]["write_outputs"] is False
    assert first["sandbox_paths"]["transcript_json"].startswith(str(out_dir))
    assert first["preflight_checks"]["audio_file_exists"] is True
    assert first["preflight_checks"]["audio_file_size_matches_plan"] is True
    assert contract.exists()


def test_asr_worker_sandbox_contract_auto_prefers_mlx_before_gigaam(tmp_path: Path) -> None:
    product_root, readiness_report, worker_plan = build_stage20_readiness(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_contract_stage21"

    report = build_asr_worker_sandbox_execution_contract(
        product_root=product_root,
        readiness_report_path=readiness_report,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        contract_path=out_dir / "contract.json",
        preferred_engine="auto",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["selected_engine"] == "mlx"
    assert report["contract"]["engine_selection"]["selection_reason"] == "auto_preference_order"
    assert report["contract"]["engine_selection"]["ready_real_engines"] == ["mlx", "gigaam"]


def test_asr_worker_sandbox_contract_blocks_unready_preferred_engine(tmp_path: Path) -> None:
    product_root, readiness_report, worker_plan = build_stage20_readiness(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_contract_stage21"

    report = build_asr_worker_sandbox_execution_contract(
        product_root=product_root,
        readiness_report_path=readiness_report,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        contract_path=out_dir / "contract.json",
        preferred_engine="openai",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["selected_engine"] is None
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_EXECUTION_CONTRACT": 1}
    assert "engine_not_ready:openai" in report["contract"]["source_validation_reasons"]
    assert report["contract"]["run_asr"] is False


def test_asr_worker_sandbox_contract_blocks_invalid_readiness(tmp_path: Path) -> None:
    product_root, readiness_report, worker_plan = build_stage20_readiness(tmp_path, count=1)
    data = json.loads(readiness_report.read_text(encoding="utf-8"))
    data["run_asr"] = True
    data["hard_guards"]["run_asr"] = True
    readiness_report.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_dir = product_root / "asr_worker_sandbox_contract_stage21"

    report = build_asr_worker_sandbox_execution_contract(
        product_root=product_root,
        readiness_report_path=readiness_report,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        contract_path=out_dir / "contract.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_EXECUTION_CONTRACT": 1}
    reasons = report["contract"]["source_validation_reasons"]
    assert "readiness_run_asr_must_be_false" in reasons
    assert "readiness_hard_guard_run_asr_must_be_false" in reasons


def test_asr_worker_sandbox_contract_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, readiness_report, worker_plan = build_stage20_readiness(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_worker_sandbox_execution_contract(
            product_root=product_root,
            readiness_report_path=readiness_report,
            worker_plan_path=worker_plan,
            out_dir=tmp_path / "outside",
            contract_path=tmp_path / "outside" / "contract.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_sandbox_execution_contract(
            product_root=product_root,
            readiness_report_path=readiness_report,
            worker_plan_path=product_root / "stable_runtime" / "worker_plan.json",
            out_dir=product_root / "asr_worker_sandbox_contract_stage21",
            contract_path=product_root / "asr_worker_sandbox_contract_stage21" / "contract.json",
        )
    with pytest.raises(ValueError, match="output directory"):
        build_asr_worker_sandbox_execution_contract(
            product_root=product_root,
            readiness_report_path=readiness_report,
            worker_plan_path=worker_plan,
            out_dir=product_root / "asr_worker_sandbox_contract_stage21",
            contract_path=product_root / "asr_worker_sandbox_contract_stage21" / "contract.json",
            out_path=product_root / "other_stage" / "audit.json",
        )
    with pytest.raises(ValueError, match="output root"):
        build_asr_worker_sandbox_execution_contract(
            product_root=product_root,
            readiness_report_path=readiness_report,
            worker_plan_path=worker_plan,
            out_dir=product_root / "asr_worker_sandbox_contract_stage21",
            contract_path=product_root / "asr_worker_sandbox_contract_stage21" / "contract.json",
            sandbox_output_root=product_root / "other_output_root",
        )


def test_asr_worker_sandbox_contract_cli_writes_audit(tmp_path: Path) -> None:
    product_root, readiness_report, worker_plan = build_stage20_readiness(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_contract_stage21"
    out = out_dir / "audit.json"
    contract = out_dir / "contract.json"

    rc = mango_office_asr_worker_sandbox_contract.main(
        [
            "--product-root",
            str(product_root),
            "--readiness-report",
            str(readiness_report),
            "--worker-plan",
            str(worker_plan),
            "--out-dir",
            str(out_dir),
            "--contract",
            str(contract),
            "--out",
            str(out),
            "--engine",
            "mlx",
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["selected_engine"] == "mlx"
    assert data["summary"]["tasks"] == 1
    assert data["safety"]["run_asr"] is False
    assert data["safety"]["write_transcripts"] is False
    assert contract.exists()


def build_stage20_readiness(tmp_path: Path, count: int) -> tuple[Path, Path, Path]:
    product_root, worker_plan = build_stage19_worker_plan(tmp_path, count=count)
    out_dir = product_root / "asr_worker_sandbox_readiness_stage20"
    readiness_report = out_dir / "readiness.json"
    report = build_asr_worker_sandbox_readiness(
        product_root=product_root,
        worker_plan_path=worker_plan,
        out_dir=out_dir,
        readiness_report_path=readiness_report,
        out_path=out_dir / "audit.json",
        env={"TRANSCRIBE_PROVIDER": "mock", "MLX_WHISPER_MODEL": "test-mlx", "GIGAAM_MODEL": "test-gigaam"},
        module_checker=lambda name: name in {"mlx_whisper", "gigaam"},
        binary_checker=lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["worker_sandbox_ready"] is True
    return product_root, readiness_report, worker_plan
