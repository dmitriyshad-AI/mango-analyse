from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from mango_mvp.replay_exam.m1_adapter import build_replay_m1_manifest
from mango_mvp.replay_exam.provider_adapter import SCRUBBED_ROOT


def _m1_scrubbed_set(tmp_path: Path) -> Path:
    return SCRUBBED_ROOT / "pytest_m1_adapter" / tmp_path.name / "replay_exam_set_v1.jsonl"


def test_build_replay_m1_manifest_records_chat_only_metric(tmp_path: Path) -> None:
    set_path = tmp_path / "set.jsonl"
    set_path.write_text('{"dialog_id":"d","segment":"chat_only","brand":"foton"}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    source_head = tmp_path / "SOURCE_HEAD.txt"
    source_head.write_text("abc\n", encoding="utf-8")
    pii_report = tmp_path / "pii_scan_v2.json"
    pii_report.write_text('{"leak_count":0}\n', encoding="utf-8")
    retention = tmp_path / "RETENTION_MANIFEST.json"
    retention.write_text(
        json.dumps({"suggested_delete_paths_after_exam": ["/tmp/raw-local", "/tmp/scrubbed-local", "/tmp/package"]}),
        encoding="utf-8",
    )
    m1_set_path = _m1_scrubbed_set(tmp_path)
    out = build_replay_m1_manifest(
        set_path=set_path,
        out_path=tmp_path / "manifest.json",
        command=["python3", "scripts/run_wappi_replay_exam.py", "--set", str(m1_set_path)],
        live_head="live-sha",
        snapshot_path=snapshot,
        source_head_path=source_head,
        pii_report_path=pii_report,
        retention_manifest_path=retention,
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
    assert payload["m1_scrubbed_set_path"] == str(m1_set_path.resolve())
    assert payload["m1_prompt_requirements"]["copy_package_set_to_scrubbed_path_before_run"] is True
    assert payload["m1_prompt_requirements"]["verify_copied_set_sha256"] is True
    assert payload["m1_prompt_requirements"]["include_scrubbed_set_copy_in_retention"] is True
    assert payload["parallelism"] == "dialogs_only"
    assert payload["real_provider_parallel_cap"] == 2
    assert payload["snapshot_sha256"]
    assert payload["pii_leak_count"] == 0
    assert "/tmp/raw-local" in payload["retention_delete_command"]
    assert "/tmp/scrubbed-local" in payload["retention_delete_command"]
    assert "/tmp/package" in payload["retention_delete_command"]
    assert str(m1_set_path.parent.resolve()) in payload["retention_delete_command"]
    assert payload["budgets"]["max_bot_calls"] == 1
    assert payload["case_stats"]["turns"] == 1
    assert payload["case_stats"]["dialogs"] == 1


def test_build_replay_m1_manifest_rejects_package_set_command(tmp_path: Path) -> None:
    set_path = tmp_path / "set" / "replay_exam_set_v1.jsonl"
    set_path.parent.mkdir()
    set_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="real replay cases must stay under"):
        build_replay_m1_manifest(
            set_path=set_path,
            out_path=tmp_path / "manifest.json",
            command=["python3", "scripts/run_wappi_replay_exam.py", "--set", str(set_path)],
            repo_root=Path.cwd(),
        )


def test_build_replay_m1_manifest_cli_accepts_command_flags(tmp_path: Path) -> None:
    set_path = tmp_path / "set.jsonl"
    out_path = tmp_path / "manifest.json"
    set_path.write_text("{}\n", encoding="utf-8")
    m1_set_path = _m1_scrubbed_set(tmp_path)
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
            "--set",
            str(m1_set_path),
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
    assert payload["m1_scrubbed_set_path"] == str(m1_set_path.resolve())
    assert "--out-dir" in payload["command"]
    assert "--fake-provider" in payload["command"]
