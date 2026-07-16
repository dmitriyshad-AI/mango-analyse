from __future__ import annotations

import json
from pathlib import Path

from scripts.skills import fail_raw_export, inventory_before_build, live_truth, tz_lint, wappi_draft_loop_replay
from scripts.wappi_draft_loop_ops import ProcessInfo


def test_tz_lint_reuses_preflight_header_and_flags_common_tz_defects(tmp_path: Path) -> None:
    tz = tmp_path / "TZ.md"
    tz.write_text(
        "\n".join(
            [
                "Ветка: main",
                "Зоны: scripts/",
                "Тест-команда: pytest tests/test_x.py",
                "Семантический-аудит: нет",
                "",
                "Путь коллеги: /Users/dmitriy/Projects/Mango analyse",
                "Старый sha abc1234 был в прошлом ТЗ.",
                "Смотри src/foo.py:123.",
            ]
        ),
        encoding="utf-8",
    )

    result = tz_lint.lint_tz(tz)

    codes = {issue.code for issue in result.issues}
    assert result.header["branch"] == "main"
    assert "foreign_user_path" in codes
    assert "old_sha_tail" in codes
    assert "bare_line_number" in codes
    assert "missing_acceptance" in codes
    assert "missing_stop" in codes


def test_tz_lint_passes_minimal_well_formed_tz(tmp_path: Path) -> None:
    tz = tmp_path / "TZ.md"
    tz.write_text(
        "\n".join(
            [
                "Ветка: main",
                "Зоны: scripts/, tests/",
                "Тест-команда: PYTHONPATH=src python3 -m pytest -q tests/test_x.py",
                "Семантический-аудит: нет",
                "",
                "## Приёмка",
                "- тест зелёный",
                "",
                "## СТОП",
                "- live-write",
            ]
        ),
        encoding="utf-8",
    )

    assert tz_lint.lint_tz(tz).status == "PASS"


def test_fail_raw_export_masks_pii_and_exports_only_fail_rows(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "dynamic_dialog_transcripts.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dialog_id": "case_fail",
                        "turns": [
                            {
                                "client_text": "Мой телефон +7 999 123-45-67",
                                "bot_text": "Напишите на user@example.com",
                                "bot_route": "draft_for_manager",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"dialog_id": "case_ok", "turns": [{"client_text": "ok", "bot_text": "ok"}]}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "dynamic_judge_results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"dialog_id": "case_fail", "verdict": "FAIL", "rationale": "bad +7 999 123-45-67"}, ensure_ascii=False),
                json.dumps({"dialog_id": "case_ok", "verdict": "PASS"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = fail_raw_export.write_export(run, tmp_path / "out")

    assert result["fail_count"] == 1
    text = Path(result["md"]).read_text(encoding="utf-8")
    assert "+7 999 123-45-67" not in text
    assert "user@example.com" not in text
    assert "[redacted_phone]" in text
    assert "[redacted_email]" in text
    assert "case_ok" not in text


def test_wappi_replay_checks_four_profiles_mapping_and_brand_mismatch(tmp_path: Path) -> None:
    stop_file = tmp_path / "STOP_DRAFT_LOOP"
    stop_file.write_text("stop", encoding="utf-8")
    rows = [
        {"brand": "foton", "expected_brand": "foton", "channel": "telegram", "lead_id": "1", "contact_id": "11"},
        {"brand": "unpk", "expected_brand": "unpk", "channel": "telegram", "lead_id": "2", "contact_id": "22"},
        {"brand": "foton", "expected_brand": "foton", "channel": "max", "lead_id": "3", "contact_id": "33"},
        {"brand": "unpk", "expected_brand": "unpk", "channel": "max", "lead_id": "4", "contact_id": "44"},
    ]

    result = wappi_draft_loop_replay.validate_replay(rows, stop_file=stop_file)

    assert result.status == "PASS"
    by_code = {check.code: check for check in result.checks}
    assert by_code["four_profile_brand_channel_split"].status == "PASS"
    assert by_code["stop_file_guard"].status == "PASS"

    bad = wappi_draft_loop_replay.validate_replay(
        [{**rows[0], "expected_brand": "unpk"}],
        stop_file=stop_file,
        require_four_profiles=False,
    )
    assert bad.status == "FAIL"


def test_live_truth_snapshot_redacts_env_and_reports_head_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    process = ProcessInfo(pid=42, ppid=1, command=f"python3 {repo}/scripts/run_amo_wappi_draft_loop.py --loop")

    snapshot = live_truth.build_snapshot(
        repo_root=repo,
        processes=[process],
        env_reader=lambda _pid: ({"TELEGRAM_BOT_TOKEN": "secret", "TELEGRAM_FACT_VENUE_SCOPE": "1"}, "test"),
        lsof_reader=lambda _pid: [str(repo / "customer_timeline.sqlite")],
        expected_heads={"run_amo_wappi_draft_loop.py": "expected"},
    )

    assert snapshot.status == "WARN"
    row = snapshot.processes[0]
    assert row.env["TELEGRAM_BOT_TOKEN"] == "[REDACTED]"
    assert row.env["TELEGRAM_FACT_VENUE_SCOPE"] == "1"
    assert row.db_paths == [str(repo / "customer_timeline.sqlite")]


def test_live_truth_ignores_test_process_that_only_mentions_live_script(tmp_path: Path) -> None:
    process = ProcessInfo(
        pid=43,
        ppid=1,
        command="python3 -m pytest tests/test_run_amo_wappi_draft_loop.py --expect-head run_amo_wappi_draft_loop.py=abc",
    )

    snapshot = live_truth.build_snapshot(repo_root=tmp_path, processes=[process])

    assert snapshot.status == "PASS"
    assert snapshot.processes == []


def test_inventory_before_build_uses_git_log_and_inventory_summary(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(_root: Path, command: list[str], *, timeout: int = 60):
        calls.append(command)

        class Result:
            returncode = 0
            stdout = "abc123 existing symbol\n" if "log" in command else ""
            stderr = ""

        return Result()

    monkeypatch.setattr(inventory_before_build, "_run", fake_run)
    monkeypatch.setattr(
        inventory_before_build,
        "build_project_inventory",
        lambda _config: {"db_files": 0, "archive_candidate_rows": 0},
    )

    result = inventory_before_build.run_inventory(tmp_path, keywords=["memory step guard"], symbols=["apply_bot_safe_memory_step_guard"], graph=tmp_path / "missing.json")

    assert result.status == "FOUND"
    assert any(candidate.source == "git_log_S" for candidate in result.candidates)
    assert any("log" in command for command in calls)
