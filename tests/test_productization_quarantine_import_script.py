from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import mango_office_quarantine_import_plan


def test_quarantine_import_script_writes_plan(tmp_path: Path) -> None:
    out = tmp_path / "plan.json"
    fake_plan = {
        "summary": {"ready": 1},
        "audit": {"blocked": 0},
        "items": [{"status": "ready"}],
    }

    with patch.object(
        mango_office_quarantine_import_plan,
        "build_quarantine_import_plan",
        return_value=fake_plan,
    ):
        rc = mango_office_quarantine_import_plan.main(
            [
                "--bridge-plan",
                str(tmp_path / "bridge.json"),
                "--out-root",
                str(tmp_path / "out"),
                "--out",
                str(out),
            ]
        )

    assert rc == 0
    assert out.exists()
