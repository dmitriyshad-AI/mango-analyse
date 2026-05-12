from __future__ import annotations

import json
import subprocess
import sys

from mango_mvp.channels import build_channel_demo_workspace


def test_channel_demo_workspace_is_idempotent_and_redacted(tmp_path) -> None:
    db_path = tmp_path / "channel_demo.sqlite"

    first = build_channel_demo_workspace(db_path)
    second = build_channel_demo_workspace(db_path)

    workspace = second["workspace"]
    assert first["workspace"]["metrics"]["sessions"] == 3
    assert workspace["metrics"]["sessions"] == 3
    assert workspace["metrics"]["drafts_needing_review"] == 3
    assert workspace["metrics"]["hot_leads"] == 1
    assert workspace["metrics"]["risk_feedback_events"] == 3
    assert workspace["safety"]["live_send"] is False
    assert workspace["safety"]["write_crm"] is False
    assert second["safety"]["demo_only"] is True
    assert "redacted" not in str(second)


def test_channel_demo_script_prints_json(tmp_path) -> None:
    db_path = tmp_path / "channel_demo.sqlite"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_channel_workspace_demo.py",
            "--db-path",
            str(db_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "channel_demo_workspace_v1"
    assert payload["workspace"]["metrics"]["sessions"] == 3
    assert payload["workspace"]["safety"]["read_only_workspace"] is True
