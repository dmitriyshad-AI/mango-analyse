from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_worker_sandbox_approval_packet import REQUIRED_APPROVAL_PHRASE
from mango_mvp.productization.asr_worker_sandbox_execution_request import (
    build_asr_worker_sandbox_execution_request,
)
from mango_mvp.productization.asr_worker_sandbox_human_approval_record import (
    write_asr_worker_sandbox_human_approval_record,
)
from scripts import mango_office_asr_worker_sandbox_execution_request
from tests.test_productization_asr_worker_sandbox_human_approval_record import all_acknowledged, build_stage23_packet


def test_execution_request_blocks_absent_human_approval_without_failure(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=2)
    out_dir = product_root / "asr_worker_sandbox_execution_request_stage25"
    request = out_dir / "request.json"
    audit = out_dir / "audit.json"
    approval_record = product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json"

    report = build_asr_worker_sandbox_execution_request(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=approval_record,
        out_dir=out_dir,
        request_path=request,
        out_path=audit,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["execution_request_ready"] is False
    assert report["summary"]["approval_packet_valid"] is True
    assert report["summary"]["contract_valid"] is True
    assert report["summary"]["approval_record_present"] is False
    assert report["summary"]["approval_record_valid"] is False
    assert report["summary"]["requested_tasks"] == 0
    assert report["summary"]["dispatch_allowed"] is False
    assert report["summary"]["run_asr"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_EXECUTION_REQUEST_PENDING_HUMAN_APPROVAL": 1}
    assert report["approval"]["missing_or_invalid_reasons"] == ["approval_record_missing"]
    assert report["execution_request"]["status"] == "blocked_execution_request"
    assert report["execution_request"]["next_stage_contract"]["may_create_worker_launcher_dry_run"] is False
    assert report["safety"]["creates_sandbox_output_dirs"] is False
    assert report["safety"]["reads_audio"] is False
    assert request.exists()
    assert audit.exists()


def test_execution_request_accepts_valid_record_but_never_dispatches(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=2)
    out_dir = product_root / "asr_worker_sandbox_execution_request_stage25"
    request = out_dir / "request.json"
    approval_record = product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json"
    write_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=approval_record,
        approved_by="test-operator",
        approved_at="2026-05-08T12:00:00+00:00",
        approval_phrase=REQUIRED_APPROVAL_PHRASE,
        acknowledgements=all_acknowledged(),
    )

    report = build_asr_worker_sandbox_execution_request(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=approval_record,
        out_dir=out_dir,
        request_path=request,
        out_path=out_dir / "audit.json",
    )

    payload = json.loads(request.read_text(encoding="utf-8"))
    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["execution_request_ready"] is True
    assert report["summary"]["execution_approved_by_human_record"] is True
    assert report["summary"]["requested_tasks"] == 2
    assert report["summary"]["dispatch_allowed"] is False
    assert report["summary"]["run_asr"] is False
    assert report["action_counts"] == {"PLAN_ASR_SANDBOX_EXECUTION_REQUEST_NOT_DISPATCHED": 1}
    assert payload["status"] == "execution_request_planned_not_dispatched"
    assert payload["execution_allowed"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["run_asr"] is False
    assert payload["create_dirs"] is False
    assert payload["request_items"][0]["action"] == "PLAN_ASR_SANDBOX_EXECUTION_REQUEST_TASK"
    assert payload["request_items"][0]["dispatch_allowed"] is False
    assert payload["request_items"][0]["write_transcripts"] is False
    assert payload["next_stage_contract"]["may_create_worker_launcher_dry_run"] is True
    assert payload["next_stage_contract"]["may_dispatch_worker_in_this_stage"] is False


def test_execution_request_blocks_tampered_approval_record(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_execution_request_stage25"
    approval_record = product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json"
    write_asr_worker_sandbox_human_approval_record(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=approval_record,
        approved_by="test-operator",
        approved_at="2026-05-08T12:00:00+00:00",
        approval_phrase=REQUIRED_APPROVAL_PHRASE,
        acknowledgements=all_acknowledged(),
    )
    data = json.loads(approval_record.read_text(encoding="utf-8"))
    data["scope"]["valid_for_asr_sandbox_execution_request"] = False
    approval_record.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_asr_worker_sandbox_execution_request(
        product_root=product_root,
        approval_packet_path=packet,
        approval_record_path=approval_record,
        out_dir=out_dir,
        request_path=out_dir / "request.json",
        out_path=out_dir / "audit.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["execution_request_ready"] is False
    assert report["summary"]["approval_record_present"] is True
    assert report["summary"]["approval_record_valid"] is False
    assert report["action_counts"] == {"BLOCK_ASR_SANDBOX_EXECUTION_REQUEST_INVALID_HUMAN_APPROVAL": 1}
    assert "record_scope_execution_request_not_allowed" in report["approval"]["missing_or_invalid_reasons"]
    assert report["summary"]["dispatch_allowed"] is False
    assert report["summary"]["run_asr"] is False


def test_execution_request_refuses_unsafe_paths(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)

    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_sandbox_execution_request(
            product_root=product_root,
            approval_packet_path=packet,
            approval_record_path=product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json",
            out_dir=product_root / "asr_worker_sandbox_execution_request_stage25",
            request_path=product_root / "stable_runtime" / "request.json",
        )
    with pytest.raises(ValueError, match="human approval record"):
        build_asr_worker_sandbox_execution_request(
            product_root=product_root,
            approval_packet_path=packet,
            approval_record_path=tmp_path / "outside" / "approval.json",
            out_dir=product_root / "asr_worker_sandbox_execution_request_stage25",
            request_path=product_root / "asr_worker_sandbox_execution_request_stage25" / "request.json",
        )
    with pytest.raises(ValueError, match="audit must differ"):
        build_asr_worker_sandbox_execution_request(
            product_root=product_root,
            approval_packet_path=packet,
            approval_record_path=product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json",
            out_dir=product_root / "asr_worker_sandbox_execution_request_stage25",
            request_path=product_root / "asr_worker_sandbox_execution_request_stage25" / "request.json",
            out_path=product_root / "asr_worker_sandbox_execution_request_stage25" / "request.json",
        )


def test_execution_request_cli_writes_blocked_request(tmp_path: Path) -> None:
    product_root, packet = build_stage23_packet(tmp_path, count=1)
    out_dir = product_root / "asr_worker_sandbox_execution_request_stage25"
    request = out_dir / "request.json"
    audit = out_dir / "audit.json"
    approval_record = product_root / "asr_worker_sandbox_human_approval_stage24" / "approval.json"

    rc = mango_office_asr_worker_sandbox_execution_request.main(
        [
            "--product-root",
            str(product_root),
            "--approval-packet",
            str(packet),
            "--approval-record",
            str(approval_record),
            "--out-dir",
            str(out_dir),
            "--request",
            str(request),
            "--out",
            str(audit),
        ]
    )

    saved = json.loads(audit.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["validation_ok"] is True
    assert saved["summary"]["execution_request_ready"] is False
    assert saved["action_counts"] == {"BLOCK_ASR_SANDBOX_EXECUTION_REQUEST_PENDING_HUMAN_APPROVAL": 1}
    assert request.exists()
