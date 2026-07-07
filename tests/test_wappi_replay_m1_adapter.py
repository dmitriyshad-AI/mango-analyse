from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from mango_mvp.replay_exam.m1_adapter import build_replay_m1_manifest


def test_build_replay_m1_manifest_records_chat_only_metric(tmp_path: Path) -> None:
    set_path = tmp_path / "set.jsonl"
    set_path.write_text('{"dialog_id":"d","segment":"chat_only","brand":"foton"}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    source_head = tmp_path / "SOURCE_HEAD.txt"
    source_head.write_text("abc\n", encoding="utf-8")
    pii_report = tmp_path / "pii_scan_v2.json"
    pii_report.write_text('{"leak_count":0}\n', encoding="utf-8")
    out = build_replay_m1_manifest(
        set_path=set_path,
        out_path=tmp_path / "manifest.json",
        command=["python3", "scripts/run_wappi_replay_exam.py"],
        live_head="live-sha",
        snapshot_path=snapshot,
        source_head_path=source_head,
        pii_report_path=pii_report,
        budgets={"max_bot_calls": 1, "max_judge_calls": 1},
        repo_root=Path.cwd(),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "wappi_replay_m1_manifest_v2"
    assert payload["metric"] == "chat_only_replay"
    assert payload["live_head"] == "live-sha"
    assert payload["live_writes_allowed"] is False
    assert payload["raw_included"] is False
    assert payload["scrubbed_only"] is True
    assert payload["parallelism"] == "dialogs_only"
    assert payload["real_provider_parallel_cap"] == 2
    assert payload["snapshot_sha256"]
    assert payload["pii_leak_count"] == 0
    assert payload["budgets"]["max_bot_calls"] == 1
    assert payload["case_stats"]["turns"] == 1
    assert payload["case_stats"]["dialogs"] == 1


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
            "--live-head",
            "live-sha",
            "--max-bot-calls",
            "5",
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
    assert payload["live_head"] == "live-sha"
    assert payload["budgets"]["max_bot_calls"] == 5
    assert "--out-dir" in payload["command"]
    assert "--fake-provider" in payload["command"]
