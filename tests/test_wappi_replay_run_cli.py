from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_case(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dialog_id": "wappi_replay_dialog",
                "profile_id": "[profile_id:id_aaaaaaaaaaaa]",
                "chat_id": "[chat_id:id_bbbbbbbbbbbb]",
                "turn_id": "turn-1",
                "brand": "foton",
                "client_message": "Есть места?",
                "manager_reference": "Ответ менеджера",
                "segment": "chat_only",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_replay_cli_requires_exactly_one_provider_mode(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_case(cases)
    result = subprocess.run(
        [sys.executable, "scripts/run_wappi_replay_exam.py", "--set", str(cases), "--out-dir", str(tmp_path / "out")],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Choose exactly one provider mode" in result.stderr


def test_run_replay_cli_real_provider_requires_explicit_llm_permission(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_case(cases)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(tmp_path / "out"),
            "--real-provider",
            "--snapshot",
            str(snapshot),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--allow-llm-calls" in result.stderr


def test_run_replay_cli_judge_requires_explicit_llm_permission_and_budget(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_case(cases)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(tmp_path / "out"),
            "--fake-provider",
            "--run-judge",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--allow-judge-llm-calls" in result.stderr


def test_run_replay_cli_real_provider_rejects_cases_outside_scrubbed_root(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_case(cases)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(tmp_path / "out"),
            "--real-provider",
            "--allow-llm-calls",
            "--snapshot",
            str(snapshot),
            "--parallel",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "real replay cases must stay under" in result.stderr


def test_run_replay_cli_real_provider_rejects_outside_set_before_json_parse(tmp_path: Path) -> None:
    cases = tmp_path / "bad_cases.jsonl"
    cases.write_text("{not-json", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(tmp_path / "out"),
            "--real-provider",
            "--allow-llm-calls",
            "--snapshot",
            str(snapshot),
            "--parallel",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "real replay cases must stay under" in result.stderr
    assert "JSONDecodeError" not in result.stderr


def test_run_replay_cli_real_provider_rejects_runtime_output_before_provider(tmp_path: Path) -> None:
    cases = Path("~/.mango_local/replay_exam/scrubbed/pytest_missing_cases.jsonl").expanduser()
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(tmp_path / "stable_runtime" / "replay_out"),
            "--real-provider",
            "--allow-llm-calls",
            "--snapshot",
            str(snapshot),
            "--parallel",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "stable_runtime" in result.stderr


def test_run_replay_cli_fake_provider_writes_pii_scan(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_case(cases)
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(out_dir),
            "--fake-provider",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    scan = json.loads((out_dir / "pii_scan_v2.json").read_text(encoding="utf-8"))
    assert scan["leak_count"] == 0
    progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["done_cases"] == 1
    assert (out_dir / "replay_results.partial.jsonl").exists()
    assert "pii_scan=" in result.stdout


def test_run_replay_cli_fake_provider_max_bot_calls_and_resume(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    first = {
        "dialog_id": "d",
        "profile_id": "[profile_id:id_aaaaaaaaaaaa]",
        "chat_id": "[chat_id:id_bbbbbbbbbbbb]",
        "turn_id": "d#1",
        "brand": "foton",
        "client_message": "Первый вопрос",
        "manager_reference": "Ответ",
        "segment": "chat_only",
    }
    second = {**first, "turn_id": "d#2", "client_message": "Второй вопрос"}
    cases.write_text(
        json.dumps(first, ensure_ascii=False) + "\n" + json.dumps(second, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    first_run = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(out_dir),
            "--fake-provider",
            "--max-bot-calls",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert first_run.returncode == 0, first_run.stderr
    assert json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))["done_cases"] == 1

    second_run = subprocess.run(
        [
            sys.executable,
            "scripts/run_wappi_replay_exam.py",
            "--set",
            str(cases),
            "--out-dir",
            str(out_dir),
            "--fake-provider",
            "--resume",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert second_run.returncode == 0, second_run.stderr
    assert json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))["done_cases"] == 2
    assert len((out_dir / "replay_results.jsonl").read_text(encoding="utf-8").splitlines()) == 2
