from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from mango_mvp.replay_exam.m1_adapter import build_replay_m1_manifest


def test_build_replay_m1_manifest_records_chat_only_metric(tmp_path: Path) -> None:
    set_path = tmp_path / "set.jsonl"
    set_path.write_text("{}\n", encoding="utf-8")
    out = build_replay_m1_manifest(
        set_path=set_path,
        out_path=tmp_path / "manifest.json",
        command=["python3", "scripts/run_wappi_replay_exam.py"],
        repo_root=Path.cwd(),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metric"] == "chat_only_replay"
    assert payload["live_writes_allowed"] is False
    assert payload["parallelism"] == "dialogs_only"


def test_build_replay_m1_manifest_cli_accepts_command_flags(tmp_path: Path) -> None:
    set_path = tmp_path / "set.jsonl"
    out_path = tmp_path / "manifest.json"
    set_path.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_wappi_replay_m1_manifest.py",
            "--set",
            str(set_path),
            "--out",
            str(out_path),
            "--command",
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--out-dir",
            "out",
            "--fake-provider",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "manifest=" in result.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "--out-dir" in payload["command"]
    assert "--fake-provider" in payload["command"]
