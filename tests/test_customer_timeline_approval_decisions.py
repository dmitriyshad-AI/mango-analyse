from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pytest

from mango_mvp.customer_timeline import (
    CustomerTimelineApprovalDecisionConfig,
    CustomerTimelineApprovalWorkspaceConfig,
    build_customer_timeline_approval_decision_template,
    build_customer_timeline_approval_workspace,
    run_customer_timeline_approval_decisions,
    validate_customer_timeline_approval_decisions,
)
from mango_mvp.customer_timeline.approval_decisions import (
    CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION,
    build_decision_template_rows,
    load_decision_jsonl_rows,
    main,
)
from tests.test_customer_timeline_read_api import seed_timeline_db


FIXED_TIME = datetime(2026, 5, 12, 21, 0, tzinfo=timezone.utc)
DECIDED_AT = "2026-05-12T21:10:00+00:00"


def test_approval_decision_template_writes_jsonl_report_and_blocks_live_actions(tmp_path: Path) -> None:
    workspace = blocked_workspace(tmp_path)
    out_template = tmp_path / "approval" / "decisions.template.jsonl"
    out_report = tmp_path / "approval" / "decisions_template_report.json"

    report = build_customer_timeline_approval_decision_template(
        workspace,
        generated_at=FIXED_TIME,
        out_template_jsonl=out_template,
        out_report_json=out_report,
    )
    rows = load_decision_jsonl_rows(out_template)
    from_disk = json.loads(out_report.read_text(encoding="utf-8"))
    serialized = out_template.read_text(encoding="utf-8") + out_report.read_text(encoding="utf-8")

    assert from_disk == report
    assert len(rows) == 2
    assert rows[0]["decision"] == "pending"
    assert rows[0]["record_type"] == "customer_timeline_approval_decision"
    assert all(row["approval_allowed"] is False for row in rows)
    assert all(row["allowed_decisions"] == ["reject", "needs_rework"] for row in rows)
    assert rows[0]["acknowledgements"]["understands_no_live_writes"] is False
    assert rows[0]["workspace_summary_snapshot"]["status"] == "blocked_by_conflict"
    assert rows[0]["queue_item"]["live_write"] is False
    assert rows[0]["live_write"] is False
    assert report["summary"]["pending_rows"] == 2
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["write_product_timeline_db"] is False
    assert report["safety"]["network_calls"] is False
    assert "raw_payload" not in serialized
    assert "/not/read/transcript.json" not in serialized


def test_approval_decision_validate_accepts_needs_rework_for_blocked_workspace(tmp_path: Path) -> None:
    workspace = blocked_workspace(tmp_path)
    rows = [
        final_decision(row, decision="needs_rework", reason="Нужно закрыть блокер workspace.")
        for row in build_decision_template_rows(workspace)
    ]

    report = validate_customer_timeline_approval_decisions(
        workspace,
        rows,
        generated_at=FIXED_TIME,
    )

    assert report["validation_ok"] is True
    assert report["summary"]["workflow_status"] == "needs_rework"
    assert report["summary"]["accepted_rows"] == 2
    assert report["summary"]["ready_for_live"] is False
    assert report["accepted_rows"][0]["decision"] == "needs_rework"
    assert report["next_safe_step"] == "fix_workspace_blockers_and_regenerate_template"


def test_approval_decision_validate_accepts_approve_only_for_ready_workspace(tmp_path: Path) -> None:
    workspace = ready_workspace()
    row = final_decision(build_decision_template_rows(workspace)[0], decision="approve", reason="Проверено оператором.")

    report = validate_customer_timeline_approval_decisions(workspace, [row], generated_at=FIXED_TIME)

    assert report["validation_ok"] is True
    assert report["summary"]["workflow_status"] == "approved_for_next_dry_run"
    assert report["summary"]["approved"] == 1
    assert report["next_safe_step"] == "prepare_read_only_dry_run_pack"
    assert report["safety"]["live_send"] is False


def test_approval_decision_validate_rejects_pending_unknown_duplicate_and_bad_final_rows(tmp_path: Path) -> None:
    workspace = ready_workspace()
    template_row = build_decision_template_rows(workspace)[0]
    pending_row = dict(template_row)
    unknown_row = final_decision(template_row, decision="approve", reason="Неизвестная строка.")
    unknown_row["decision_id"] = "approval_decision:unknown"
    bad_final_row = final_decision(template_row, decision="approve", reason="")
    bad_final_row["reviewer"] = ""
    bad_final_row["decided_by"] = ""
    duplicate_row = final_decision(template_row, decision="approve", reason="Дубликат.")

    report = validate_customer_timeline_approval_decisions(
        workspace,
        [pending_row, unknown_row, bad_final_row, duplicate_row],
        generated_at=FIXED_TIME,
    )

    errors = "\n".join(error for item in report["invalid_rows"] for error in item["errors"])
    assert report["validation_ok"] is False
    assert report["summary"]["pending_rows"] == 1
    assert report["summary"]["unknown_rows"] == 1
    assert report["summary"]["duplicate_rows"] == 2
    assert "decision_still_pending" in errors
    assert "unknown_decision_id" in errors
    assert "reviewer_required" in errors
    assert "reason_required" in errors
    assert "duplicate_decision_id" in errors


def test_approval_decision_validate_rejects_approve_for_blocked_workspace(tmp_path: Path) -> None:
    workspace = blocked_workspace(tmp_path)
    row = final_decision(build_decision_template_rows(workspace)[0], decision="approve", reason="Пытаюсь одобрить конфликт.")

    report = validate_customer_timeline_approval_decisions(workspace, [row], generated_at=FIXED_TIME)

    assert report["validation_ok"] is False
    assert report["summary"]["invalid_rows"] == 1
    assert "decision_not_allowed_for_workspace_state" in report["invalid_rows"][0]["errors"]


def test_approval_decision_requires_timezone_aware_decided_at(tmp_path: Path) -> None:
    workspace = ready_workspace()
    row = final_decision(build_decision_template_rows(workspace)[0], decision="approve", reason="Проверено.")
    row["decided_at"] = "2026-05-12T21:10:00"

    report = validate_customer_timeline_approval_decisions(workspace, [row], generated_at=FIXED_TIME)

    assert report["validation_ok"] is False
    assert "decided_at_must_be_timezone_aware" in report["invalid_rows"][0]["errors"]


def test_approval_decision_cli_template_and_validate_are_artifact_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("approval decisions must not use network/subprocess")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(os, "system", fail)
    monkeypatch.setattr(socket, "socket", fail)
    db_path, customer_id = seed_timeline_db(tmp_path)
    workspace_path = tmp_path / "approval" / "workspace.json"
    template_path = tmp_path / "approval" / "template.jsonl"
    template_report_path = tmp_path / "approval" / "template_report.json"
    validation_report_path = tmp_path / "approval" / "validation_report.json"
    workspace = build_customer_timeline_approval_workspace(
        config=CustomerTimelineApprovalWorkspaceConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_json=workspace_path,
        ),
        tenant_id="foton",
        customer_id=customer_id,
        generated_at=FIXED_TIME,
    )

    rc_template = main(
        [
            "template",
            "--allowed-root",
            str(tmp_path),
            "--workspace-json",
            str(workspace_path),
            "--out-template-jsonl",
            str(template_path),
            "--out-report-json",
            str(template_report_path),
        ]
    )
    filled_rows = [
        final_decision(row, decision="needs_rework", reason="Закрыть блокер workspace.")
        for row in load_decision_jsonl_rows(template_path)
    ]
    template_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in filled_rows),
        encoding="utf-8",
    )
    rc_validate = main(
        [
            "validate",
            "--allowed-root",
            str(tmp_path),
            "--workspace-json",
            str(workspace_path),
            "--decisions-jsonl",
            str(template_path),
            "--out-report-json",
            str(validation_report_path),
        ]
    )
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))

    assert rc_template == 0
    assert rc_validate == 0
    assert validation_report["validation_ok"] is True
    assert validation_report["safety"]["write_crm"] is False
    assert validation_report["safety"]["run_asr"] is False
    assert validation_report["safety"]["run_ra"] is False


def test_approval_decision_can_build_workspace_from_readonly_timeline_db(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    out_template = tmp_path / "approval" / "from_db.template.jsonl"
    report = run_customer_timeline_approval_decisions(
        config=CustomerTimelineApprovalDecisionConfig(
            allowed_root=tmp_path,
            timeline_db=db_path,
            out_template_jsonl=out_template,
        ),
        mode="template",
        tenant_id="foton",
        customer_id=customer_id,
        generated_at=FIXED_TIME,
    )

    assert report["summary"]["decision_rows"] == 2
    assert out_template.exists()
    assert load_decision_jsonl_rows(out_template)[0]["tenant_id"] == "foton"


def test_approval_decision_does_not_mutate_timeline_db_or_stable_runtime(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    stable_dir = tmp_path / "stable_runtime"
    stable_dir.mkdir()
    sentinel = stable_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    before_db_sha = file_sha256(db_path)
    before_stable = sorted(path.relative_to(stable_dir) for path in stable_dir.rglob("*"))
    out_template = tmp_path / "approval" / "safe.template.jsonl"
    out_report = tmp_path / "approval" / "safe.report.json"

    report = run_customer_timeline_approval_decisions(
        config=CustomerTimelineApprovalDecisionConfig(
            allowed_root=tmp_path,
            timeline_db=db_path,
            out_template_jsonl=out_template,
            out_report_json=out_report,
        ),
        mode="template",
        tenant_id="foton",
        customer_id=customer_id,
        generated_at=FIXED_TIME,
    )

    assert report["safety"]["write_product_timeline_db"] is False
    assert file_sha256(db_path) == before_db_sha
    assert sorted(path.relative_to(stable_dir) for path in stable_dir.rglob("*")) == before_stable
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_approval_decision_path_guards(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text(json.dumps(ready_workspace(), ensure_ascii=False), encoding="utf-8")
    stable_input = tmp_path / "stable_runtime" / "workspace.json"
    stable_input.parent.mkdir()
    stable_input.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineApprovalDecisionConfig(allowed_root=tmp_path, workspace_json=stable_input)
    with pytest.raises(ValueError, match="allowed root"):
        CustomerTimelineApprovalDecisionConfig(allowed_root=tmp_path, workspace_json=tmp_path.parent / "outside.json")
    with pytest.raises(ValueError, match="does not exist"):
        CustomerTimelineApprovalDecisionConfig(allowed_root=tmp_path, workspace_json=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="overwrite workspace JSON"):
        CustomerTimelineApprovalDecisionConfig(
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            out_report_json=workspace_path,
        )
    with pytest.raises(ValueError, match="outputs must be separate"):
        CustomerTimelineApprovalDecisionConfig(
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            out_template_jsonl=tmp_path / "same.json",
            out_report_json=tmp_path / "same.json",
        )


def ready_workspace() -> dict[str, object]:
    return {
        "schema_version": "customer_timeline_approval_workspace_v1",
        "read_api_schema_version": "customer_timeline_read_api_v1",
        "generated_at": FIXED_TIME.isoformat(),
        "tenant_id": "foton",
        "inputs": {},
        "summary": {
            "validation_ok": True,
            "status": "ready_for_review",
            "selected_customer_id": "customer:ready",
            "selected_customer_found": True,
            "customers_visible": 1,
            "open_conflicts": 0,
            "bot_allowed_chunks": 2,
            "bot_review_required_chunks": 0,
            "live_actions_available": False,
            "warnings": 0,
            "blocked": 0,
        },
        "panels": {},
        "review_queue": [
            {
                "action": "READY_FOR_OPERATOR_APPROVAL_REVIEW",
                "priority": "normal",
                "label": "Customer timeline is ready for read-only approval review",
                "live_write": False,
            }
        ],
        "actions": {"read_only": True},
        "safety": {"write_crm": False},
        "validation_ok": True,
    }


def blocked_workspace(tmp_path: Path) -> dict[str, object]:
    db_path, customer_id = seed_timeline_db(tmp_path)
    workspace = build_customer_timeline_approval_workspace(
        config=CustomerTimelineApprovalWorkspaceConfig(timeline_db=db_path, allowed_root=tmp_path),
        tenant_id="foton",
        customer_id=customer_id,
        generated_at=FIXED_TIME,
    )
    return dict(workspace)


def final_decision(template_row: Mapping[str, object], *, decision: str, reason: str) -> dict[str, object]:
    row = deepcopy(dict(template_row))
    row["schema_version"] = CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION
    row["decision"] = decision
    row["reviewer"] = "operator@example.com"
    row["decided_by"] = "operator@example.com"
    row["reason"] = reason
    row["reason_codes"] = ["reviewed_ok"] if decision == "approve" else ["needs_rework"] if decision == "needs_rework" else ["operator_rejected"]
    row["comment"] = reason
    row["decided_at"] = DECIDED_AT
    row["acknowledgements"] = {
        "reviewed_customer": True,
        "reviewed_timeline": True,
        "reviewed_bot_context": True,
        "reviewed_conflicts": True,
        "understands_no_live_writes": True,
    }
    row["rework_items"] = ["identity_conflict"] if decision == "needs_rework" else []
    row["live_write"] = False
    return row


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
