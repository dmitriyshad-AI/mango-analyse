from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import mango_office_pipeline_bridge_dry_run


def test_pipeline_bridge_script_writes_json_and_csv(tmp_path: Path) -> None:
    out = tmp_path / "plan.json"
    csv_out = tmp_path / "plan.csv"
    fake_plan = {
        "summary": {"bridge_status_counts": {"would_import": 1}},
        "audit": {"blocked": 0},
        "items": [{"status": "would_import", "reason": "ok"}],
    }

    with patch.object(
        mango_office_pipeline_bridge_dry_run,
        "build_pipeline_bridge_plan",
        return_value=fake_plan,
    ), patch.object(mango_office_pipeline_bridge_dry_run, "write_bridge_plan_csv") as write_csv:
        rc = mango_office_pipeline_bridge_dry_run.main(
            [
                "--manifest",
                str(tmp_path / "manifest.jsonl"),
                "--source-dir",
                str(tmp_path / "source"),
                "--db",
                str(tmp_path / "calls.db"),
                "--out",
                str(out),
                "--csv-out",
                str(csv_out),
            ]
        )

    assert rc == 0
    assert out.exists()
    write_csv.assert_called_once()
