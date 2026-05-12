from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    CustomerIdentity,
    CustomerTimelineApprovalWorkspaceConfig,
    CustomerTimelineSQLiteStore,
    build_customer_timeline_approval_workspace,
)
from mango_mvp.customer_timeline.approval_workspace import main, render_customer_timeline_approval_workspace_html
from tests.test_customer_timeline_read_api import seed_timeline_db


FIXED_TIME = datetime(2026, 5, 12, 18, 0, tzinfo=timezone.utc)


def test_approval_workspace_builds_json_and_escaped_html(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    add_html_named_customer(tmp_path, db_path)
    out_json = tmp_path / "workspace" / "approval.json"
    out_html = tmp_path / "workspace" / "approval.html"

    workspace = build_customer_timeline_approval_workspace(
        config=CustomerTimelineApprovalWorkspaceConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_json=out_json,
            out_html=out_html,
        ),
        tenant_id="foton",
        customer_id=customer_id,
        query="Иванова",
        limit=10,
        generated_at=FIXED_TIME,
    )
    html = out_html.read_text(encoding="utf-8")
    from_disk = json.loads(out_json.read_text(encoding="utf-8"))

    assert from_disk == workspace
    assert workspace["validation_ok"] is True
    assert workspace["summary"]["selected_customer_found"] is True
    assert workspace["summary"]["open_conflicts"] == 1
    assert workspace["summary"]["status"] == "blocked_by_conflict"
    assert workspace["panels"]["selected_customer"]["customer"]["primary_phone"] == "+***4567"
    assert workspace["panels"]["safety_gates"]["write_crm"] is False
    assert "Customer Timeline Approval Workspace" in html
    assert "Write CRM blocked" in html
    assert "Run ASR/R+A blocked" in html
    assert "/not/read/transcript.json" not in html
    assert "raw_payload" not in html
    assert "provider_raw_payload" not in html
    assert "record_json" not in html
    assert "hidden" not in html


def test_approval_workspace_escapes_customer_text_and_handles_no_selection(tmp_path: Path) -> None:
    db_path, _ = seed_timeline_db(tmp_path)
    add_html_named_customer(tmp_path, db_path)

    workspace = build_customer_timeline_approval_workspace(
        config=CustomerTimelineApprovalWorkspaceConfig(timeline_db=db_path, allowed_root=tmp_path),
        tenant_id="foton",
        query="<script>alert(1)</script>",
        generated_at=FIXED_TIME,
    )
    html = render_customer_timeline_approval_workspace_html(workspace)

    assert workspace["validation_ok"] is True
    assert workspace["summary"]["selected_customer_found"] is True
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_approval_workspace_without_matching_customer_is_still_readonly_report(tmp_path: Path) -> None:
    db_path, _ = seed_timeline_db(tmp_path)

    workspace = build_customer_timeline_approval_workspace(
        config=CustomerTimelineApprovalWorkspaceConfig(timeline_db=db_path, allowed_root=tmp_path),
        tenant_id="foton",
        query="no such customer",
        generated_at=FIXED_TIME,
    )

    assert workspace["validation_ok"] is True
    assert workspace["summary"]["selected_customer_found"] is False
    assert workspace["summary"]["warnings"] >= 1
    assert workspace["actions"]["read_only"] is True
    assert workspace["actions"]["blocked"]["write_crm"] is False


def test_approval_workspace_cli_and_no_network_or_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("approval workspace must not use network/subprocess")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(os, "system", fail)
    monkeypatch.setattr(socket, "socket", fail)
    db_path, customer_id = seed_timeline_db(tmp_path)
    out_json = tmp_path / "out" / "workspace.json"
    out_html = tmp_path / "out" / "workspace.html"

    rc = main(
        [
            "--tenant-id",
            "foton",
            "--timeline-db",
            str(db_path),
            "--allowed-root",
            str(tmp_path),
            "--customer-id",
            customer_id,
            "--query",
            "стоимость",
            "--out-json",
            str(out_json),
            "--out-html",
            str(out_html),
        ]
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert out_html.exists()
    assert report["safety"]["network_calls"] is False
    assert report["safety"]["subprocess_calls"] is False
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["run_ra"] is False


def test_approval_workspace_output_guards(tmp_path: Path) -> None:
    db_path, _ = seed_timeline_db(tmp_path)
    with pytest.raises(ValueError, match="overwrite timeline DB"):
        CustomerTimelineApprovalWorkspaceConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_json=db_path,
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineApprovalWorkspaceConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_html=tmp_path / "stable_runtime" / "workspace.html",
        )
    with pytest.raises(ValueError, match="allowed root"):
        CustomerTimelineApprovalWorkspaceConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_json=tmp_path.parent / "outside_workspace.json",
        )


def test_approval_workspace_is_deterministic_with_fixed_generated_at(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    config = CustomerTimelineApprovalWorkspaceConfig(timeline_db=db_path, allowed_root=tmp_path)

    first = build_customer_timeline_approval_workspace(
        config=config,
        tenant_id="foton",
        customer_id=customer_id,
        query="стоимость",
        generated_at=FIXED_TIME,
    )
    second = build_customer_timeline_approval_workspace(
        config=config,
        tenant_id="foton",
        customer_id=customer_id,
        query="стоимость",
        generated_at=FIXED_TIME,
    )

    assert first == second


def add_html_named_customer(tmp_path: Path, db_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            identity_status="partial",
            display_name="<script>alert(1)</script>",
            primary_phone="+79160000001",
            first_seen_at=FIXED_TIME,
            last_seen_at=FIXED_TIME,
            touch_count=1,
            created_at=FIXED_TIME,
            updated_at=FIXED_TIME,
        )
    )
    store.close()
