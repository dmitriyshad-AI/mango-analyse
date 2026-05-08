from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import mango_office_quarantine_materialize


def test_quarantine_materialize_script_writes_audit(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    fake_report = {
        "summary": {"blocked": 0, "copied": 1},
        "audit": {"target_audio_files": 1},
        "items": [{"status": "copied"}],
    }

    with patch.object(
        mango_office_quarantine_materialize,
        "materialize_quarantine_package",
        return_value=fake_report,
    ):
        rc = mango_office_quarantine_materialize.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--out",
                str(out),
                "--mode",
                "copy",
            ]
        )

    assert rc == 0
    assert out.exists()
