from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pytest

from mango_mvp.customer_timeline import (
    CustomerTimelineApprovedContextPackConfig,
    CustomerTimelineApprovalWorkspaceConfig,
    CustomerTimelineSQLiteStore,
    build_customer_timeline_approval_workspace,
    build_customer_timeline_approved_context_pack,
)
from mango_mvp.customer_timeline.approval_decisions import (
    CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION,
    build_decision_template_rows,
    load_workspace_json,
    validate_customer_timeline_approval_decisions,
)
from mango_mvp.customer_timeline.approved_context_pack import main
from tests.test_customer_timeline_read_api import seed_timeline_db


FIXED_TIME = datetime(2026, 5, 13, 9, 0, tzinfo=timezone.utc)
DECIDED_AT = "2026-05-13T09:05:00+03:00"


def test_approved_context_pack_builds_from_valid_approve_decision(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path)

    pack = build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
        ),
        generated_at=FIXED_TIME,
    )
    serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True)

    assert pack["validation_ok"] is True
    assert pack["status"] == "approved_read_only_context_pack"
    assert pack["summary"]["blocked_reasons"] == []
    assert pack["summary"]["context_chunks"] == 1
    assert pack["summary"]["bot_context_total_chunks"] == 2
    assert pack["summary"]["bot_context_review_required_chunks"] == 1
    assert pack["approval"]["workflow_status"] == "approved_for_next_dry_run"
    assert pack["approval"]["approved"] == 1
    assert pack["channel_context"]["can_build_draft"] is True
    assert pack["channel_context"]["can_send"] is False
    assert pack["approved_context"]["items"][0]["allowed_for_bot"] is True
    assert pack["approved_context"]["items"][0]["requires_manager_review"] is False
    assert "customer_id" not in pack["approved_context"]["items"][0]
    assert "opportunity_id" not in pack["approved_context"]["items"][0]
    assert "event_id" not in pack["approved_context"]["items"][0]
    assert "Клиент спрашивал стоимость" in serialized
    assert "Этот фрагмент требует проверки менеджера" not in serialized
    assert "raw_payload" not in serialized
    assert "provider_raw_payload" not in serialized
    assert "record_json" not in serialized
    assert "/not/read/transcript.json" not in serialized
    assert "/secret/audio.mp3" not in serialized
    assert "+79161234567" not in serialized
    assert "parent@example.com" not in serialized
    assert str(workspace_path) not in serialized
    assert pack["safety"]["write_crm"] is False
    assert pack["safety"]["write_tallanto"] is False
    assert pack["safety"]["live_send"] is False
    assert pack["safety"]["network_calls"] is False
    assert pack["safety"]["subprocess_calls"] is False
    assert pack["safety"]["llm_calls"] is False
    assert pack["safety"]["rag_used"] is False


def test_approved_context_pack_self_validates_decisions_without_cached_report(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, _report_path = build_approval_artifacts(tmp_path)

    pack = build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert pack["validation_ok"] is True
    assert pack["source_refs"]["approval_report_sha256"] is None
    assert pack["source_refs"]["decisions_jsonl_sha256"] == file_sha256(decisions_path)


@pytest.mark.parametrize(
    ("decision", "expected_reason"),
    (
        ("reject", "approval_workflow_not_approved:rejected"),
        ("needs_rework", "approval_workflow_not_approved:needs_rework"),
    ),
)
def test_reject_and_needs_rework_block_context_pack(tmp_path: Path, decision: str, expected_reason: str) -> None:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path, decision=decision)

    pack = build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert pack["validation_ok"] is False
    assert pack["status"] == "blocked"
    assert expected_reason in pack["summary"]["blocked_reasons"]
    assert pack["approved_context"]["items"] == []
    assert pack["channel_context"]["can_build_draft"] is False
    assert pack["channel_context"]["can_send"] is False


def test_invalid_pending_decision_blocks_context_pack(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, _report_path = build_approval_artifacts(tmp_path)
    rows = [dict(row) for row in load_decision_rows(decisions_path)]
    rows[0]["decision"] = "pending"
    write_decision_rows(decisions_path, rows)

    pack = build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert pack["validation_ok"] is False
    assert "approval_report_not_valid" in pack["summary"]["blocked_reasons"]
    assert pack["approved_context"]["items"] == []


def test_current_db_conflict_after_approval_blocks_context_pack(tmp_path: Path) -> None:
    db_path, customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path)
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.record_conflict(
        "foton",
        conflict_type="ambiguous_identity",
        entity_refs=(customer_id, "customer:new-conflict"),
        actor="test",
    )
    store.close()

    pack = build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert pack["validation_ok"] is False
    assert "current_db_open_conflicts" in pack["summary"]["blocked_reasons"]
    assert pack["approved_context"]["items"] == []


def test_cached_report_mismatch_blocks_context_pack(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["approved"] = 99
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pack = build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert pack["validation_ok"] is False
    assert "cached_approval_report_summary_approved_mismatch" in pack["summary"]["blocked_reasons"]


def test_context_pack_cli_is_artifact_only_no_network_or_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("approved context pack must not use network/subprocess")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(os, "system", fail)
    monkeypatch.setattr(socket, "socket", fail)
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path)
    out_pack = tmp_path / "approved" / "context_pack.json"

    rc = main(
        [
            "--timeline-db",
            str(db_path),
            "--allowed-root",
            str(tmp_path),
            "--workspace-json",
            str(workspace_path),
            "--decisions-jsonl",
            str(decisions_path),
            "--approval-report-json",
            str(report_path),
            "--out-pack-json",
            str(out_pack),
        ]
    )
    pack = json.loads(out_pack.read_text(encoding="utf-8"))

    assert rc == 0
    assert pack["validation_ok"] is True
    assert pack["safety"]["network_calls"] is False
    assert pack["safety"]["subprocess_calls"] is False
    assert pack["safety"]["write_runtime_db"] is False


def test_context_pack_cli_returns_one_for_business_blockers(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path, decision="reject")
    out_pack = tmp_path / "approved" / "blocked_context_pack.json"

    rc = main(
        [
            "--timeline-db",
            str(db_path),
            "--allowed-root",
            str(tmp_path),
            "--workspace-json",
            str(workspace_path),
            "--decisions-jsonl",
            str(decisions_path),
            "--approval-report-json",
            str(report_path),
            "--out-pack-json",
            str(out_pack),
        ]
    )
    pack = json.loads(out_pack.read_text(encoding="utf-8"))

    assert rc == 1
    assert pack["validation_ok"] is False
    assert pack["status"] == "blocked"


def test_context_pack_is_readonly_and_does_not_mutate_db_or_stable_runtime(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path)
    stable_dir = tmp_path / "stable_runtime"
    stable_dir.mkdir()
    sentinel = stable_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    before_db_sha = file_sha256(db_path)
    before_stable = sorted(path.relative_to(stable_dir) for path in stable_dir.rglob("*"))
    out_pack = tmp_path / "approved" / "context_pack.json"

    pack = build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=out_pack,
        ),
        generated_at=FIXED_TIME,
    )

    assert pack["validation_ok"] is True
    assert file_sha256(db_path) == before_db_sha
    assert sorted(path.relative_to(stable_dir) for path in stable_dir.rglob("*")) == before_stable
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_context_pack_path_guards(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path)
    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=tmp_path / "stable_runtime" / "pack.json",
        )
    with pytest.raises(ValueError, match="allowed root"):
        CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=tmp_path.parent / "outside_pack.json",
        )
    with pytest.raises(ValueError, match="overwrite timeline DB"):
        CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=db_path,
        )
    with pytest.raises(ValueError, match="overwrite workspace JSON"):
        CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=workspace_path,
        )
    with pytest.raises(ValueError, match="overwrite decisions JSONL"):
        CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=decisions_path,
        )
    with pytest.raises(ValueError, match="overwrite approval report JSON"):
        CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=report_path,
        )


def test_context_pack_is_deterministic_with_fixed_generated_at(tmp_path: Path) -> None:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path)
    config = CustomerTimelineApprovedContextPackConfig(
        timeline_db=db_path,
        allowed_root=tmp_path,
        workspace_json=workspace_path,
        decisions_jsonl=decisions_path,
        approval_report_json=report_path,
    )

    first = build_customer_timeline_approved_context_pack(config=config, generated_at=FIXED_TIME)
    second = build_customer_timeline_approved_context_pack(config=config, generated_at=FIXED_TIME)

    assert first == second


def build_approval_artifacts(
    tmp_path: Path,
    *,
    decision: str = "approve",
) -> tuple[Path, str, Path, Path, Path]:
    db_path, customer_id = seed_timeline_db(tmp_path)
    remove_open_conflicts(db_path)
    workspace_path = tmp_path / "approval" / "workspace.json"
    decisions_path = tmp_path / "approval" / f"{decision}_decisions.jsonl"
    report_path = tmp_path / "approval" / f"{decision}_validation_report.json"
    build_customer_timeline_approval_workspace(
        config=CustomerTimelineApprovalWorkspaceConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_json=workspace_path,
        ),
        tenant_id="foton",
        customer_id=customer_id,
        generated_at=FIXED_TIME,
    )
    persisted_workspace = load_workspace_json(workspace_path)
    rows = [
        final_decision(row, decision=decision, reason=f"Operator decision: {decision}")
        for row in build_decision_template_rows(persisted_workspace)
    ]
    write_decision_rows(decisions_path, rows)
    report = validate_customer_timeline_approval_decisions(
        persisted_workspace,
        rows,
        generated_at=FIXED_TIME,
        out_report_json=report_path,
    )
    assert report["validation_ok"] is True
    return db_path, customer_id, workspace_path, decisions_path, report_path


def final_decision(template_row: Mapping[str, object], *, decision: str, reason: str) -> dict[str, object]:
    row = deepcopy(dict(template_row))
    row["schema_version"] = CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION
    row["decision"] = decision
    row["reviewer"] = "operator@example.com"
    row["decided_by"] = "operator@example.com"
    row["reason"] = reason
    row["reason_codes"] = (
        ["reviewed_ok"]
        if decision == "approve"
        else ["needs_rework"]
        if decision == "needs_rework"
        else ["operator_rejected"]
    )
    row["comment"] = reason
    row["decided_at"] = DECIDED_AT
    row["acknowledgements"] = {
        "reviewed_customer": True,
        "reviewed_timeline": True,
        "reviewed_bot_context": True,
        "reviewed_conflicts": True,
        "understands_no_live_writes": True,
    }
    row["rework_items"] = ["operator_rework"] if decision == "needs_rework" else []
    row["live_write"] = False
    return row


def remove_open_conflicts(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    con.execute("DELETE FROM timeline_conflicts")
    con.commit()
    con.close()


def load_decision_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_decision_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
