from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mango_mvp.productization.asr_worker_sandbox_execution_contract import build_asr_worker_sandbox_execution_contract
import mango_mvp.productization.asr_worker_sandbox_preflight as preflight_module
from mango_mvp.productization.asr_worker_sandbox_preflight import build_asr_worker_sandbox_preflight
from scripts import mango_office_asr_worker_sandbox_preflight
from tests.test_productization_asr_worker_sandbox_execution_contract import build_stage20_readiness


def test_asr_worker_sandbox_preflight_passes_contract_without_execution(tmp_path: Path) -> None:
    product_root, contract = build_stage21_contract(tmp_path, count=2)
    out_dir = product_root / "asr_worker_sandbox_preflight_stage22"
    preflight = out_dir / "preflight.json"

    report = build_asr_worker_sandbox_preflight(
        product_root=product_root,
        contract_path=contract,
        out_dir=out_dir,
        preflight_report_path=preflight,
        out_path=out_dir / "audit.json",
        module_checker=lambda name: name == "mlx_whisper",
        disk_usage_provider=large_disk,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["selected_engine"] == "mlx"
    assert report["summary"]["tasks"] == 2
    assert report["summary"]["passed_tasks"] == 2
    assert report["summary"]["blocked_tasks"] == 0
    assert report["summary"]["audio_files_checked"] == 2
    assert report["summary"]["audio_sha_ok"] == 2
    assert report["summary"]["dir_preflight_ok"] is True
    assert report["summary"]["disk_space_ok"] is True
    assert report["summary"]["engine_preflight_ok"] is True
    assert report["summary"]["run_asr"] is False
    assert report["summary"]["write_transcripts"] is False
    assert report["action_counts"] == {"PASS_ASR_SANDBOX_PREFLIGHT_TASK": 2}
    assert report["preflight_report"]["status"] == "preflight_passed_not_dispatched"
    assert report["preflight_report"]["hard_guards"]["run_asr"] is False
    assert report["safety"]["reads_audio_for_sha256"] is True
    assert report["safety"]["write_transcripts"] is False
    assert preflight.exists()
    assert not (out_dir / "sandbox_outputs").exists()
    assert not (out_dir / "sandbox_tmp").exists()


def test_asr_worker_sandbox_preflight_blocks_audio_sha_mismatch(tmp_path: Path) -> None:
    product_root, contract = build_stage21_contract(tmp_path, count=1)
    data = json.loads(contract.read_text(encoding="utf-8"))
    data["tasks"][0]["audio"]["sha256"] = "0" * 64
    contract.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_dir = product_root / "asr_worker_sandbox_preflight_stage22"

    report = build_asr_worker_sandbox_preflight(
        product_root=product_root,
        contract_path=contract,
        out_dir=out_dir,
        preflight_report_path=out_dir / "preflight.json",
        module_checker=lambda name: name == "mlx_whisper",
        disk_usage_provider=large_disk,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked_tasks"] == 1
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_FINAL_PREFLIGHT": 1}
    assert "audio_sha256_mismatch" in report["preflight_report"]["task_preflights"][0]["blocked_reasons"]
    assert report["preflight_report"]["run_asr"] is False


def test_asr_worker_sandbox_preflight_blocks_output_collision(tmp_path: Path) -> None:
    product_root, contract = build_stage21_contract(tmp_path, count=1)
    data = json.loads(contract.read_text(encoding="utf-8"))
    collision = Path(data["tasks"][0]["sandbox_paths"]["transcript_json"])
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_text("existing transcript", encoding="utf-8")
    out_dir = product_root / "asr_worker_sandbox_preflight_stage22"

    report = build_asr_worker_sandbox_preflight(
        product_root=product_root,
        contract_path=contract,
        out_dir=out_dir,
        preflight_report_path=out_dir / "preflight.json",
        module_checker=lambda name: name == "mlx_whisper",
        disk_usage_provider=large_disk,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["output_collisions"] == 1
    assert "output_collision:transcript_json" in report["preflight_report"]["task_preflights"][0]["blocked_reasons"]
    assert report["preflight_report"]["write_transcripts"] is False


def test_asr_worker_sandbox_preflight_blocks_missing_engine(tmp_path: Path) -> None:
    product_root, contract = build_stage21_contract(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_preflight_stage22"

    report = build_asr_worker_sandbox_preflight(
        product_root=product_root,
        contract_path=contract,
        out_dir=out_dir,
        preflight_report_path=out_dir / "preflight.json",
        module_checker=lambda _name: False,
        disk_usage_provider=large_disk,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["engine_preflight_ok"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_FINAL_PREFLIGHT": 1}
    assert "missing_python_module:mlx_whisper" in report["preflight_report"]["engine_preflight"]["blocked_reasons"]
    assert report["preflight_report"]["run_asr"] is False


def test_asr_worker_sandbox_preflight_blocks_insufficient_disk(tmp_path: Path) -> None:
    product_root, contract = build_stage21_contract(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_preflight_stage22"

    report = build_asr_worker_sandbox_preflight(
        product_root=product_root,
        contract_path=contract,
        out_dir=out_dir,
        preflight_report_path=out_dir / "preflight.json",
        module_checker=lambda name: name == "mlx_whisper",
        disk_usage_provider=lambda _path: SimpleNamespace(total=100, used=90, free=10),
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["disk_space_ok"] is False
    assert "insufficient_disk_space" in report["preflight_report"]["disk_preflight"]["blocked_reasons"]


def test_asr_worker_sandbox_preflight_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, contract = build_stage21_contract(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_worker_sandbox_preflight(
            product_root=product_root,
            contract_path=contract,
            out_dir=tmp_path / "outside",
            preflight_report_path=tmp_path / "outside" / "preflight.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_sandbox_preflight(
            product_root=product_root,
            contract_path=product_root / "stable_runtime" / "contract.json",
            out_dir=product_root / "asr_worker_sandbox_preflight_stage22",
            preflight_report_path=product_root / "asr_worker_sandbox_preflight_stage22" / "preflight.json",
        )
    with pytest.raises(ValueError, match="output directory"):
        build_asr_worker_sandbox_preflight(
            product_root=product_root,
            contract_path=contract,
            out_dir=product_root / "asr_worker_sandbox_preflight_stage22",
            preflight_report_path=product_root / "asr_worker_sandbox_preflight_stage22" / "preflight.json",
            out_path=product_root / "other_stage" / "audit.json",
        )


def test_asr_worker_sandbox_preflight_cli_writes_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    product_root, contract = build_stage21_contract(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_preflight_stage22"
    out = out_dir / "audit.json"
    preflight = out_dir / "preflight.json"
    monkeypatch.setattr(preflight_module, "module_available", lambda name: name == "mlx_whisper")

    rc = mango_office_asr_worker_sandbox_preflight.main(
        [
            "--product-root",
            str(product_root),
            "--contract",
            str(contract),
            "--out-dir",
            str(out_dir),
            "--preflight-report",
            str(preflight),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["tasks"] == 1
    assert data["summary"]["audio_sha_ok"] == 1
    assert data["safety"]["run_asr"] is False
    assert data["safety"]["write_transcripts"] is False
    assert preflight.exists()


def build_stage21_contract(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, readiness_report, worker_plan = build_stage20_readiness(tmp_path, count=count)
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
    return product_root, contract


def large_disk(_path: Path) -> SimpleNamespace:
    return SimpleNamespace(total=10**12, used=0, free=10**12)
