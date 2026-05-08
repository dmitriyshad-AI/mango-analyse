from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_worker_sandbox_approval_packet import (
    REQUIRED_APPROVAL_PHRASE,
    REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS,
    build_asr_worker_sandbox_approval_packet,
)
from mango_mvp.productization.asr_worker_sandbox_human_approval_record import (
    build_asr_worker_sandbox_human_approval_requirements,
    validate_asr_worker_sandbox_human_approval_record,
    write_asr_worker_sandbox_human_approval_record,
)
from scripts import mango_office_asr_worker_sandbox_human_approval
from tests.test_productization_asr_worker_sandbox_approval_packet import build_stage22_preflight


def test_human_approval_requirements_reports_pending_without_approval(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=2)
    out = product_root / "asr_worker_sandbox_human_approval_stage24" / "requirements.json"

    report = build_asr_worker_sandbox_human_approval_requirements(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json",
        out_path=out,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["operation"] == "requirements"
    assert report["summary"]["approval_packet_valid"] is True
    assert report["summary"]["approval_record_present"] is False
    assert report["summary"]["execution_approved"] is False
    assert report["summary"]["run_asr"] is False
    assert report["action_counts"] == {"REPORT_ASR_SANDBOX_HUMAN_APPROVAL_REQUIREMENTS": 1}
    assert report["approval"]["required_approval_phrase"] == REQUIRED_APPROVAL_PHRASE
    assert report["approval"]["missing_or_invalid_reasons"] == []
    assert report["safety"]["writes_approval_record"] is False
    assert out.exists()


def test_human_approval_record_write_and_validate_accepts_strict_approval(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_human_approval_stage24"
    record = out_dir / "approval.json"

    write_report = write_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=record,
        out_path=out_dir / "write_audit.json",
        approved_by="test-operator",
        approved_at="2026-05-08T12:00:00+00:00",
        approval_phrase=REQUIRED_APPROVAL_PHRASE,
        acknowledgements=all_acknowledged(),
        reason="unit-test",
    )
    validate_report = validate_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=record,
        out_path=out_dir / "validate_audit.json",
    )

    saved = json.loads(record.read_text(encoding="utf-8"))
    assert write_report["summary"]["validation_ok"] is True
    assert write_report["summary"]["written"] == 1
    assert write_report["summary"]["execution_approved"] is True
    assert write_report["summary"]["dispatch_allowed"] is False
    assert write_report["action_counts"] == {"WRITE_ASR_SANDBOX_HUMAN_APPROVAL_RECORD": 1}
    assert saved["decision"] == "approved"
    assert saved["scope"]["valid_for_asr_sandbox_execution_request"] is True
    assert saved["scope"]["valid_for_immediate_worker_dispatch"] is False
    assert saved["stage_contract"]["valid_for_asr_execution_in_stage24"] is False
    assert saved["safety"]["run_asr"] is False
    assert validate_report["summary"]["validation_ok"] is True
    assert validate_report["summary"]["approval_record_valid"] is True
    assert validate_report["safety"]["writes_approval_record"] is False


def test_human_approval_record_blocks_phrase_mismatch(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_human_approval_stage24"
    record = out_dir / "approval.json"

    report = write_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=record,
        out_path=out_dir / "write_audit.json",
        approved_by="test-operator",
        approved_at="2026-05-08T12:00:00+00:00",
        approval_phrase="wrong phrase",
        acknowledgements=all_acknowledged(),
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["written"] == 0
    assert report["summary"]["execution_approved"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_HUMAN_APPROVAL_RECORD": 1}
    assert "approval_phrase_mismatch" in report["approval"]["missing_or_invalid_reasons"]
    assert not record.exists()


def test_human_approval_record_blocks_missing_acknowledgement(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)
    acknowledgements = all_acknowledged()
    acknowledgements["stable_runtime_not_touched"] = False

    report = write_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json",
        approved_by="test-operator",
        approved_at="2026-05-08T12:00:00+00:00",
        approval_phrase=REQUIRED_APPROVAL_PHRASE,
        acknowledgements=acknowledgements,
    )

    assert report["summary"]["validation_ok"] is False
    assert "acknowledgement_missing_or_false:stable_runtime_not_touched" in report["approval"]["missing_or_invalid_reasons"]
    assert report["summary"]["run_asr"] is False


def test_human_approval_record_validator_blocks_tampered_record(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_human_approval_stage24"
    record = out_dir / "approval.json"
    write_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=record,
        approved_by="test-operator",
        approved_at="2026-05-08T12:00:00+00:00",
        approval_phrase=REQUIRED_APPROVAL_PHRASE,
        acknowledgements=all_acknowledged(),
    )
    data = json.loads(record.read_text(encoding="utf-8"))
    data["approval_packet_sha256"] = "bad-sha"
    record.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = validate_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=record,
        out_path=out_dir / "validate_audit.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_HUMAN_APPROVAL_RECORD": 1}
    assert "record_packet_sha_mismatch" in report["approval"]["missing_or_invalid_reasons"]
    assert report["summary"]["dispatch_allowed"] is False


def test_human_approval_record_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_worker_sandbox_human_approval_requirements(
            product_root=product_root,
            approval_packet_path=tmp_path / "outside" / "packet.json",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_sandbox_human_approval_requirements(
            product_root=product_root,
            approval_packet_path=product_root / "stable_runtime" / "packet.json",
        )
    with pytest.raises(ValueError, match="approval audit"):
        write_asr_worker_sandbox_human_approval_record(
            product_root=product_root,
            approval_packet_path=packet,
            approval_record_path=product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json",
            out_path=product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json",
            approved_by="test-operator",
            approval_phrase=REQUIRED_APPROVAL_PHRASE,
            acknowledgements=all_acknowledged(),
        )


def test_human_approval_cli_requirements_and_write_validate(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_human_approval_stage24"
    requirements_out = out_dir / "requirements.json"
    record = out_dir / "approval.json"
    write_out = out_dir / "write.json"
    validate_out = out_dir / "validate.json"

    requirements_rc = mango_office_asr_worker_sandbox_human_approval.main(
        [
            "--product-root",
            str(product_root),
            "--approval-packet",
            str(packet),
            "requirements",
            "--approval-record",
            str(record),
            "--out",
            str(requirements_out),
        ]
    )
    write_rc = mango_office_asr_worker_sandbox_human_approval.main(
        [
            "--product-root",
            str(product_root),
            "--approval-packet",
            str(packet),
            "write",
            "--approval-record",
            str(record),
            "--out",
            str(write_out),
            "--approved-by",
            "test-operator",
            "--approved-at",
            "2026-05-08T12:00:00+00:00",
            "--approval-phrase",
            REQUIRED_APPROVAL_PHRASE,
            "--acknowledge-all",
        ]
    )
    validate_rc = mango_office_asr_worker_sandbox_human_approval.main(
        [
            "--product-root",
            str(product_root),
            "--approval-packet",
            str(packet),
            "validate",
            "--approval-record",
            str(record),
            "--out",
            str(validate_out),
        ]
    )

    assert requirements_rc == 0
    assert write_rc == 0
    assert validate_rc == 0
    assert json.loads(requirements_out.read_text(encoding="utf-8"))["summary"]["execution_approved"] is False
    assert json.loads(write_out.read_text(encoding="utf-8"))["summary"]["execution_approved"] is True
    assert json.loads(validate_out.read_text(encoding="utf-8"))["summary"]["approval_record_valid"] is True


def build_stage23_packet(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, preflight_report, contract = build_stage22_preflight(tmp_path, count=count)
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
    return product_root, packet


def all_acknowledged() -> dict[str, bool]:
    return {key: True for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS}
