from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_export_wappi_replay_dialogs_refuses_without_live_read_flag(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_wappi_replay_dialogs.py",
            "--raw-root",
            str(Path.home() / ".mango_local/replay_exam/raw"),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Refusing live Wappi read" in result.stderr
