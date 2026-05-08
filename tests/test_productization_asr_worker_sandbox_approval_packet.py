from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mango_mvp.productization.asr_worker_sandbox_approval_packet import (
    REQUIRED_APPROVAL_PHRASE,
    REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS,
    build_asr_worker_sandbox_approval_packet,
)
from mango_mvp.productization.asr_worker_sandbox_preflight import build_asr_worker_sandbox_preflight
from scripts import mango_office_asr_worker_sandbox_approval_packet
from tests.test_productization_asr_worker_sandbox_preflight import build_stage21_contract


def test_asr_worker_sandbox_approval_packet_builds_pending_human_packet(tmp_path: Path) -> None:
    product_root, preflight_report, contract = build_stage22_preflight(tmp_path, count=2)
    out_dir = product_root / "asr_worker_sandbox_approval_stage23"
    packet = out_dir / "packet.json"

    report = build_asr_worker_sandbox_approval_packet(
        product_root=product_root,
        preflight_report_path=preflight_report,
        contract_path=contract,
        out_dir=out_dir,
        approval_packet_path=packet,
        out_path=out_dir / "audit.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["selected_engine"] == "mlx"
    assert report["summary"]["tasks"] == 2
    assert report["summary"]["audio_sha_ok"] == 2
    assert report["summary"]["approval_required"] is True
    assert report["summary"]["execution_approved"] is False
    assert report["summary"]["run_asr"] is False
    assert report["summary"]["write_transcripts"] is False
    assert report["action_counts"] == {"PREPARE_ASR_SANDBOX_EXECUTION_APPROVAL_PACKET": 1}
    approval_packet = report["approval_packet"]
    assert approval_packet["status"] == "pending_human_approval"
    assert approval_packet["approval_status"] == "pending_human_approval"
    assert approval_packet["execution_approved"] is False
    assert approval_packet["required_approval_phrase"] == REQUIRED_APPROVAL_PHRASE
    assert approval_packet["acknowledgement_template"] == {
        key: False for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS
    }
    assert approval_packet["approval_record_template"]["valid_for_asr_execution_dispatch"] is False
    assert len(approval_packet["items"]) == 2
    first = approval_packet["items"][0]
    assert first["action"] == "INCLUDE_ASR_SANDBOX_APPROVAL_ITEM"
    assert first["audio"]["sha256_ok"] is True
    assert first["expected_outputs"]["transcript_json"].endswith(".transcript.json")
    assert first["resource_limits"]["timeout_sec"] >= 60
    assert first["execution_approved"] is False
    assert packet.exists()


def test_asr_worker_sandbox_approval_packet_blocks_invalid_preflight(tmp_path: Path) -> None:
    product_root, preflight_report, contract = build_stage22_preflight(tmp_path, count=1)
    data = json.loads(preflight_report.read_text(encoding="utf-8"))
    data["run_asr"] = True
    data["hard_guards"]["run_asr"] = True
    preflight_report.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_dir = product_root / "asr_worker_sandbox_approval_stage23"

    report = build_asr_worker_sandbox_approval_packet(
        product_root=product_root,
        preflight_report_path=preflight_report,
        contract_path=contract,
        out_dir=out_dir,
        approval_packet_path=out_dir / "packet.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_APPROVAL_PACKET": 1}
    reasons = report["approval_packet"]["source_validation_reasons"]
    assert "preflight_run_asr_must_be_false" in reasons
    assert "preflight_hard_guard_run_asr_must_be_false" in reasons
    assert report["approval_packet"]["execution_approved"] is False


def test_asr_worker_sandbox_approval_packet_blocks_contract_sha_mismatch(tmp_path: Path) -> None:
    product_root, preflight_report, contract = build_stage22_preflight(tmp_path, count=1)
    data = json.loads(contract.read_text(encoding="utf-8"))
    data["batch_resource_limits"]["estimated_timeout_sec"] += 1
    contract.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_dir = product_root / "asr_worker_sandbox_approval_stage23"

    report = build_asr_worker_sandbox_approval_packet(
        product_root=product_root,
        preflight_report_path=preflight_report,
        contract_path=contract,
        out_dir=out_dir,
        approval_packet_path=out_dir / "packet.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert "preflight_contract_sha_mismatch" in report["approval_packet"]["source_validation_reasons"]
    assert report["approval_packet"]["run_asr"] is False


def test_asr_worker_sandbox_approval_packet_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, preflight_report, contract = build_stage22_preflight(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_worker_sandbox_approval_packet(
            product_root=product_root,
            preflight_report_path=preflight_report,
            contract_path=contract,
            out_dir=tmp_path / "outside",
            approval_packet_path=tmp_path / "outside" / "packet.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_sandbox_approval_packet(
            product_root=product_root,
            preflight_report_path=product_root / "stable_runtime" / "preflight.json",
            contract_path=contract,
            out_dir=product_root / "asr_worker_sandbox_approval_stage23",
            approval_packet_path=product_root / "asr_worker_sandbox_approval_stage23" / "packet.json",
        )
    with pytest.raises(ValueError, match="output directory"):
        build_asr_worker_sandbox_approval_packet(
            product_root=product_root,
            preflight_report_path=preflight_report,
            contract_path=contract,
            out_dir=product_root / "asr_worker_sandbox_approval_stage23",
            approval_packet_path=product_root / "asr_worker_sandbox_approval_stage23" / "packet.json",
            out_path=product_root / "other_stage" / "audit.json",
        )


def test_asr_worker_sandbox_approval_packet_cli_writes_audit(tmp_path: Path) -> None:
    product_root, preflight_report, contract = build_stage22_preflight(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_approval_stage23"
    out = out_dir / "audit.json"
    packet = out_dir / "packet.json"

    rc = mango_office_asr_worker_sandbox_approval_packet.main(
        [
            "--product-root",
            str(product_root),
            "--preflight-report",
            str(preflight_report),
            "--contract",
            str(contract),
            "--out-dir",
            str(out_dir),
            "--approval-packet",
            str(packet),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["execution_approved"] is False
    assert data["approval_packet"]["status"] == "pending_human_approval"
    assert data["safety"]["run_asr"] is False
    assert data["safety"]["write_transcripts"] is False
    assert packet.exists()


def build_stage22_preflight(tmp_path: Path, count: int) -> tuple[Path, Path, Path]:
    product_root, contract = build_stage21_contract(tmp_path, count=count)
    out_dir = product_root / "asr_worker_sandbox_preflight_stage22"
    preflight_report = out_dir / "preflight.json"
    report = build_asr_worker_sandbox_preflight(
        product_root=product_root,
        contract_path=contract,
        out_dir=out_dir,
        preflight_report_path=preflight_report,
        out_path=out_dir / "audit.json",
        module_checker=lambda name: name == "mlx_whisper",
        disk_usage_provider=lambda _path: SimpleNamespace(total=10**12, used=0, free=10**12),
    )
    assert report["summary"]["validation_ok"] is True
    return product_root, preflight_report, contract
