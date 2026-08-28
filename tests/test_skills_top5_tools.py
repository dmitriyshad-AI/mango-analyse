from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.skills import fail_raw_export, inventory_before_build, live_truth, tz_lint, wappi_draft_loop_replay
from scripts.wappi_draft_loop_ops import ProcessInfo


def _write_wappi_attestation(root: Path, *, head: str, started_at: str = "2026-07-19T08:55:41Z") -> tuple[Path, Path]:
    manifest = root / "phase1b_startup_manifest.json"
    heartbeat = root / "heartbeat.json"
    manifest.write_text(
        json.dumps({"status": "ready", "head": head, "started_at": started_at, "cwd": str(root)}),
        encoding="utf-8",
    )
    heartbeat.write_text(
        json.dumps({"git_sha": head, "last_cycle_at": "2026-07-19T09:00:00Z", "code_root": str(root)}),
        encoding="utf-8",
    )
    return manifest, heartbeat


def _fixed_process_start(_pid: int) -> datetime:
    return datetime(2026, 7, 19, 8, 55, 41, tzinfo=timezone.utc)


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
                "Feature-ID: feature.test",
                "Problem-ID: problem.test",
                "Изменение: extend",
                "Ключевые-символы: run_test",
                "Ключевые-слова: test process",
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

    assert result["status"] == "FAILS_EXPORTED"
    assert result["fail_count"] == 1
    text = Path(result["md"]).read_text(encoding="utf-8")
    assert "+7 999 123-45-67" not in text
    assert "user@example.com" not in text
    assert "[redacted_phone]" in text
    assert "[redacted_email]" in text
    assert "case_ok" not in text


def test_fail_raw_export_discovers_nested_runs_without_merging_legs(tmp_path: Path) -> None:
    run = tmp_path / "package"
    for leg in ("B", "ON"):
        dry = run / "dry" / leg
        dry.mkdir(parents=True)
        (dry / "dynamic_dialog_transcripts.jsonl").write_text('{"dialog_id":"dry"}\n', encoding="utf-8")
        (dry / "dynamic_judge_results.jsonl").write_text('{"dialog_id":"dry","verdict":"FAIL"}\n', encoding="utf-8")
    for leg, fail_count in (("B", 4), ("ON", 5)):
        leg_dir = run / "exam_full" / leg
        leg_dir.mkdir(parents=True)
        transcripts = [
            {"dialog_id": f"case_{index}", "turns": [{"client_text": "вопрос", "bot_text": "ответ"}]}
            for index in range(fail_count)
        ]
        judges = [
            {"dialog_id": f"case_{index}", "verdict": "FAIL", "rationale": f"{leg}-{index}"}
            for index in range(fail_count)
        ]
        (leg_dir / "dynamic_dialog_transcripts.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in transcripts) + "\n",
            encoding="utf-8",
        )
        (leg_dir / "dynamic_judge_results.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in judges) + "\n",
            encoding="utf-8",
        )

    first = fail_raw_export.write_export(run, tmp_path / "out-first")
    second = fail_raw_export.write_export(run, tmp_path / "out-second")

    rows = [json.loads(line) for line in Path(first["jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert first["status"] == "FAILS_EXPORTED"
    assert first["fail_count"] == 9
    assert first["source_runs"] == ["exam_full/B", "exam_full/ON"]
    assert [row["source_run"] for row in rows].count("exam_full/B") == 4
    assert [row["source_run"] for row in rows].count("exam_full/ON") == 5
    assert Path(first["jsonl"]).read_text(encoding="utf-8") == Path(second["jsonl"]).read_text(encoding="utf-8")


def test_fail_raw_export_rejects_unknown_layout(tmp_path: Path) -> None:
    run = tmp_path / "empty"
    run.mkdir()

    with pytest.raises(FileNotFoundError, match="expected exactly one complete"):
        fail_raw_export.write_export(run, tmp_path / "out")


def test_fail_raw_export_uses_first_failing_turn_and_m1_fields(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "dynamic_dialog_transcripts.jsonl").write_text(
        json.dumps({"dialog_id": "case", "turns": [
            {"turn": 1, "client_message": "первый", "bot_text": "ошибка", "judge_fact_audit": {"ok": False}, "bot_safe_context_items": ["fact"]},
            {"turn": 2, "client_message": "второй", "bot_text": "ответ"},
        ]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run / "dynamic_judge_results.jsonl").write_text(
        json.dumps({"dialog_id": "case", "verdict": "FAIL", "first_failing_turn": 1}) + "\n",
        encoding="utf-8",
    )

    result = fail_raw_export.write_export(run, tmp_path / "out")
    row = json.loads(Path(result["jsonl"]).read_text(encoding="utf-8"))

    assert row["client_text"] == "первый"
    assert row["fact_audit"] == {"ok": False}
    assert row["context_items"] == ["fact"]
    assert row["first_failing_turn"] == 1


def test_fail_raw_export_rejects_malformed_jsonl(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "dynamic_dialog_transcripts.jsonl").write_text("{broken\n", encoding="utf-8")
    (run / "dynamic_judge_results.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSONL"):
        fail_raw_export.write_export(run, tmp_path / "out")


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
    manifest, heartbeat = _write_wappi_attestation(repo, head="actual")

    snapshot = live_truth.build_snapshot(
        repo_root=repo,
        processes=[process],
        env_reader=lambda _pid: ({"TELEGRAM_BOT_TOKEN": "secret", "TELEGRAM_FACT_VENUE_SCOPE": "1"}, "test"),
        lsof_reader=lambda _pid: [str(repo / "customer_timeline.sqlite")],
        cwd_reader=lambda _pid: repo,
        process_started_reader=_fixed_process_start,
        wappi_pid_reader=lambda: 42,
        expected_heads={"run_amo_wappi_draft_loop.py": "expected"},
        wappi_manifest_path=manifest,
        wappi_heartbeat_path=heartbeat,
    )

    assert snapshot.status == "WARN"
    row = snapshot.processes[0]
    assert row.env["TELEGRAM_BOT_TOKEN"] == "[REDACTED]"
    assert row.env["TELEGRAM_FACT_VENUE_SCOPE"] == "1"
    assert row.db_paths == [str(repo / "customer_timeline.sqlite")]


def test_live_truth_accepts_full_expected_sha_for_short_runtime_head(
    tmp_path: Path, monkeypatch
) -> None:
    process = ProcessInfo(pid=45, ppid=1, command="python3 scripts/run_amo_wappi_draft_loop.py --loop")
    monkeypatch.setattr(live_truth, "_git_value", lambda *_args: "3fbe8d90")
    manifest, heartbeat = _write_wappi_attestation(tmp_path, head="3fbe8d90")

    snapshot = live_truth.build_snapshot(
        repo_root=tmp_path,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: tmp_path,
        process_started_reader=_fixed_process_start,
        wappi_pid_reader=lambda: 45,
        expected_heads={"run_amo_wappi_draft_loop.py": "3fbe8d90b0973e9aecfb3d4db61eee0f562a4404"},
        wappi_manifest_path=manifest,
        wappi_heartbeat_path=heartbeat,
    )

    assert snapshot.status == "PASS"


def test_live_truth_ignores_test_process_that_only_mentions_live_script(tmp_path: Path) -> None:
    process = ProcessInfo(
        pid=43,
        ppid=1,
        command="python3 -m pytest tests/test_run_amo_wappi_draft_loop.py --expect-head run_amo_wappi_draft_loop.py=abc",
    )

    snapshot = live_truth.build_snapshot(repo_root=tmp_path, processes=[process])

    assert snapshot.status == "NO_PROCESS"
    assert snapshot.processes == []


def test_live_truth_rejects_removed_public_bot_from_old_worktree(tmp_path: Path) -> None:
    process = ProcessInfo(pid=44, ppid=1, command="python3 scripts/run_telegram_public_pilot_bots.py --mode poll")

    snapshot = live_truth.build_snapshot(
        repo_root=tmp_path,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: tmp_path,
        process_started_reader=_fixed_process_start,
    )

    assert snapshot.status == "WARN"
    assert snapshot.processes[0].kind == "run_telegram_public_pilot_bots.py"
    assert snapshot.processes[0].warnings[0] == "forbidden_process marker=run_telegram_public_pilot_bots.py"


def test_live_truth_recognizes_thin_telegram_ai_agent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(live_truth, "_git_value", lambda *_args: "abc")
    process = ProcessInfo(pid=45, ppid=1, command="python3 scripts/run_telegram_ai_agent.py --brand foton")

    snapshot = live_truth.build_snapshot(
        repo_root=tmp_path,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: tmp_path,
        process_started_reader=_fixed_process_start,
    )

    assert snapshot.status == "PASS"
    assert snapshot.processes[0].kind == "run_telegram_ai_agent.py"


def test_live_truth_uses_process_cwd_for_relative_live_command(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    live = tmp_path / "live"
    repo.mkdir()
    live.mkdir()
    process = ProcessInfo(pid=44, ppid=1, command="python3 scripts/run_amo_wappi_draft_loop.py --loop")
    manifest, heartbeat = _write_wappi_attestation(live, head="abc")

    snapshot = live_truth.build_snapshot(
        repo_root=repo,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: live,
        process_started_reader=_fixed_process_start,
        wappi_pid_reader=lambda: 44,
        wappi_manifest_path=manifest,
        wappi_heartbeat_path=heartbeat,
    )

    assert snapshot.status == "WARN"
    assert snapshot.processes[0].worktree == str(live)


def test_live_truth_rejects_startup_manifest_from_previous_pid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(live_truth, "_git_value", lambda *_args: "abc")
    process = ProcessInfo(pid=46, ppid=1, command="python3 scripts/run_amo_wappi_draft_loop.py --loop")
    manifest, heartbeat = _write_wappi_attestation(tmp_path, head="abc", started_at="2026-07-19T08:00:00Z")
    heartbeat.write_text("{}", encoding="utf-8")

    snapshot = live_truth.build_snapshot(
        repo_root=tmp_path,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: tmp_path,
        process_started_reader=_fixed_process_start,
        wappi_pid_reader=lambda: 46,
        wappi_manifest_path=manifest,
        wappi_heartbeat_path=heartbeat,
    )

    assert snapshot.status == "WARN"
    assert snapshot.processes[0].head == ""
    assert any("startup_manifest_pid_mismatch" in item for item in snapshot.processes[0].warnings)


def test_live_truth_rejects_startup_manifest_that_is_not_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(live_truth, "_git_value", lambda *_args: "abc")
    process = ProcessInfo(pid=46, ppid=1, command="python3 scripts/run_amo_wappi_draft_loop.py --loop")
    manifest, heartbeat = _write_wappi_attestation(tmp_path, head="abc")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "starting"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    heartbeat.write_text("{}", encoding="utf-8")

    snapshot = live_truth.build_snapshot(
        repo_root=tmp_path,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: tmp_path,
        process_started_reader=_fixed_process_start,
        wappi_pid_reader=lambda: 46,
        wappi_manifest_path=manifest,
        wappi_heartbeat_path=heartbeat,
    )

    assert snapshot.processes[0].head == ""
    assert any("startup_manifest_not_ready" in item for item in snapshot.processes[0].warnings)


def test_live_truth_ignores_stale_heartbeat_when_startup_manifest_matches_process(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(live_truth, "_git_value", lambda *_args: "abc")
    process = ProcessInfo(pid=47, ppid=1, command="python3 scripts/run_amo_wappi_draft_loop.py --loop")
    manifest, heartbeat = _write_wappi_attestation(tmp_path, head="abc")
    heartbeat.write_text(
        json.dumps(
            {
                "git_sha": "old",
                "last_cycle_at": "2026-07-19T08:00:00Z",
                "code_root": str(tmp_path / "old-worktree"),
            }
        ),
        encoding="utf-8",
    )

    snapshot = live_truth.build_snapshot(
        repo_root=tmp_path,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: tmp_path,
        process_started_reader=_fixed_process_start,
        wappi_pid_reader=lambda: 47,
        expected_heads={"run_amo_wappi_draft_loop.py": "abc"},
        wappi_manifest_path=manifest,
        wappi_heartbeat_path=heartbeat,
    )

    assert snapshot.status == "PASS"
    assert snapshot.processes[0].head_source == "startup_manifest"
    assert snapshot.processes[0].warnings == []


def test_live_truth_uses_current_heartbeat_when_manifest_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(live_truth, "_git_value", lambda *_args: "abc")
    process = ProcessInfo(pid=47, ppid=1, command="python3 scripts/run_amo_wappi_draft_loop.py --loop")
    manifest = tmp_path / "missing.json"
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps({"git_sha": "abc", "last_cycle_at": "2026-07-19T09:00:00Z", "code_root": str(tmp_path)}),
        encoding="utf-8",
    )

    snapshot = live_truth.build_snapshot(
        repo_root=tmp_path,
        processes=[process],
        env_reader=lambda _pid: ({}, "test"),
        lsof_reader=lambda _pid: [],
        cwd_reader=lambda _pid: tmp_path,
        process_started_reader=_fixed_process_start,
        wappi_pid_reader=lambda: 47,
        wappi_manifest_path=manifest,
        wappi_heartbeat_path=heartbeat,
    )

    assert snapshot.processes[0].head == "abc"
    assert snapshot.processes[0].head_source == "heartbeat"


def _mock_inventory_surface(
    tmp_path: Path,
    monkeypatch,
    *,
    graph_revision: str = "head123",
    worktrees: str | None = None,
    dirty: dict[str, set[str]] | None = None,
    hits: dict[tuple[str, str], list[tuple[str, int]]] | None = None,
    hints: list[str] | None = None,
) -> None:
    worktrees = worktrees or f"worktree {tmp_path}\nHEAD head123\nbranch refs/heads/main\n"
    entries = inventory_before_build.parse_worktrees_porcelain(worktrees)
    monkeypatch.setattr(inventory_before_build, "load_output_manifest", lambda _graph: {"revision": graph_revision})
    monkeypatch.setattr(inventory_before_build, "graph_source_hints", lambda *_args, **_kwargs: list(hints or []))
    monkeypatch.setattr(inventory_before_build, "stale_banner", lambda *_args: "graph banner")
    monkeypatch.setattr(inventory_before_build, "_surface", lambda _root: (entries, worktrees, dirty or {}, {}))
    monkeypatch.setattr(
        inventory_before_build,
        "_rg_hits",
        lambda root, term, paths=inventory_before_build.CODE_ROOTS, **_kwargs: list((hits or {}).get((str(root), term), [])),
    )
    monkeypatch.setattr(inventory_before_build, "_history", lambda *_args: [])
    monkeypatch.setattr(inventory_before_build, "_metadata_hits", lambda *_args: [])
    monkeypatch.setattr(inventory_before_build, "_fingerprint", lambda *_args: "sha256:surface")
    monkeypatch.setattr(inventory_before_build, "_owns_symbol", lambda *_args: True)

    def fake_git(root: Path, *args: str) -> str:
        if args[:2] == ("rev-parse", "--abbrev-ref"):
            return "main\n"
        if args[:2] == ("rev-parse", "HEAD"):
            return "head123\n" if Path(root) == tmp_path else "donor456\n"
        if args[:3] == ("show", "-s", "--format=%cI"):
            return "2026-08-23T12:00:00+03:00\n"
        return ""

    monkeypatch.setattr(inventory_before_build, "_git", fake_git)


def _run_test_inventory(tmp_path: Path) -> inventory_before_build.InventoryResult:
    graph = tmp_path / "graph.json"
    graph.write_text("{}", encoding="utf-8")
    return inventory_before_build.run_inventory(
        tmp_path,
        feature_id="feature.test",
        problem_id="problem.test",
        change="extend",
        keywords=["business capability"],
        symbols=["exact_owner"],
        graph=graph,
    )


def test_inventory_before_build_reuses_exact_current_owner_deterministically(tmp_path: Path, monkeypatch) -> None:
    hits = {
        (str(tmp_path), "exact_owner"): [("src/owner.py", 17)],
        (str(tmp_path), "business capability"): [("src/related.py", 4)],
    }
    _mock_inventory_surface(tmp_path, monkeypatch, hits=hits)

    first = _run_test_inventory(tmp_path)
    second = _run_test_inventory(tmp_path)

    assert first.status_fingerprint == second.status_fingerprint
    assert first.candidates == second.candidates
    assert first.decision == second.decision
    assert first.decision == "extend"
    assert first.selected_owner["path"] == "src/owner.py"
    assert any(candidate.classification == "ACTIVE_EXTEND" for candidate in first.candidates)
    assert set(first.coverage) == {"graphify", "worktrees", "raw_rg", "git_refs", "tasks", "audits", "decisions"}


def test_inventory_maps_every_symbol_and_keeps_later_definition_in_same_file(tmp_path: Path) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {
        "src/one.py": "REFERENCE = 'first_owner'\n\ndef first_owner():\n    return 1\n",
        "src/two.py": "def second_owner():\n    return 2\n",
    })

    result = _real_inventory(repo, graph, symbols=["first_owner", "second_owner"], keywords=[])

    assert result.decision == "reuse"
    assert result.owner_map == {
        "first_owner": {"path": "src/one.py", "symbol": "first_owner", "sha": result.repo_head},
        "second_owner": {"path": "src/two.py", "symbol": "second_owner", "sha": result.repo_head},
    }
    assert any(
        item.symbol == "first_owner" and item.path == "src/one.py" and item.line == 3
        and item.classification == "ACTIVE_REUSE"
        for item in result.candidates
    )


def test_inventory_before_build_stops_on_modified_or_untracked_other_worktree(tmp_path: Path, monkeypatch) -> None:
    other = tmp_path / "other"
    other.mkdir()
    worktrees = (
        f"worktree {tmp_path}\nHEAD head123\nbranch refs/heads/main\n\n"
        f"worktree {other}\nHEAD donor456\nbranch refs/heads/codex/d3\n"
    )
    hits = {(str(other), "exact_owner"): [("src/partial.py", 3), ("scripts/untracked.py", 4)]}
    dirty = {str(other): {"src/partial.py", "scripts/untracked.py"}}
    _mock_inventory_surface(tmp_path, monkeypatch, worktrees=worktrees, dirty=dirty, hits=hits)

    result = _run_test_inventory(tmp_path)

    assert result.decision == "stop"
    assert result.unresolved == ["dirty_code_unclassified"]
    assert [item.classification for item in result.candidates].count("PARTIAL_WORKTREE") == 2


def test_inventory_metadata_ignores_generated_claude_context_manifest(tmp_path: Path) -> None:
    audit = tmp_path / "audits/_inbox/context"
    audit.mkdir(parents=True)
    (audit / "manifest.json").write_text(json.dumps({
        "schema_version": "mango_claude_context_pack_v1", "problem_id": "problem.test",
    }), encoding="utf-8")
    (audit / "implementation_notes.md").write_text("problem.test is implemented\n", encoding="utf-8")

    hits = inventory_before_build._metadata_hits(tmp_path, "problem.test")

    assert [item.path for item in hits] == ["audits/_inbox/context/implementation_notes.md"]


def test_inventory_before_build_does_not_promote_graph_noise_to_found(tmp_path: Path, monkeypatch) -> None:
    _mock_inventory_surface(tmp_path, monkeypatch, hints=["src/unrelated.py"])

    result = _run_test_inventory(tmp_path)

    assert result.decision == "new"
    assert any(item.classification == "FALSE_MATCH" and not item.verified_in_raw_source for item in result.candidates)
    assert any(item.classification == "ABSENT_PROVEN" for item in result.candidates)


def test_inventory_before_build_ports_donor_and_never_restores_removed_code(tmp_path: Path, monkeypatch) -> None:
    _mock_inventory_surface(tmp_path, monkeypatch)
    donor = inventory_before_build.InventoryCandidate("DONOR_REF", "git_history", "src/donor.py", 8, "donor456", "exact_owner", "donor", True)
    monkeypatch.setattr(inventory_before_build, "_history", lambda *_args: [donor])
    result = _run_test_inventory(tmp_path)
    assert result.decision == "port"
    assert result.selected_owner["sha"] == "donor456"

    removed = inventory_before_build.InventoryCandidate("REMOVED_INTENTIONALLY", "git_history", "src/old.py", 8, "old789", "exact_owner", "removed", True)
    monkeypatch.setattr(inventory_before_build, "_history", lambda *_args: [removed])
    result = _run_test_inventory(tmp_path)
    assert result.decision == "stop"
    assert "removed_intentionally_requires_owner_decision" in result.unresolved


def test_inventory_before_build_stale_graph_blocks_only_absence(tmp_path: Path, monkeypatch) -> None:
    _mock_inventory_surface(tmp_path, monkeypatch, graph_revision="old000")
    assert _run_test_inventory(tmp_path).decision == "stop"

    hits = {(str(tmp_path), "exact_owner"): [("src/owner.py", 17)]}
    _mock_inventory_surface(tmp_path, monkeypatch, graph_revision="old000", hits=hits)
    assert _run_test_inventory(tmp_path).decision == "extend"


def test_inventory_before_build_avoids_mass_or_sensitive_readers_and_writes_contract(tmp_path: Path, monkeypatch) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {"src/owner.py": "def exact_owner(): pass\n"})
    for name in ("stable_runtime", "product_data", "runtime", "data", "audio", "mail", "calls"):
        path = repo / name
        path.mkdir()
        (path / "sensitive.txt").write_text("must not be read\n", encoding="utf-8")
    assert not hasattr(inventory_before_build, "build_project_inventory")
    source = Path(inventory_before_build.__file__).read_text(encoding="utf-8")
    assert "sqlite" not in source.casefold()
    assert "stable_runtime" not in source
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def assert_safe_path(path: Path) -> None:
        try:
            relative = path.relative_to(repo)
        except ValueError:
            return
        assert not relative.parts or relative.parts[0] not in {"stable_runtime", "product_data", "runtime", "data", "audio", "mail", "calls"}

    def guarded_read_text(path: Path, *args, **kwargs):
        assert_safe_path(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path: Path, *args, **kwargs):
        assert_safe_path(path)
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sqlite forbidden")))

    result = _real_inventory(repo, graph, symbols=["exact_owner"], keywords=[])
    out = tmp_path / "audit"
    inventory_before_build._write_outputs(result, out)
    payload = json.loads((out / "prebuild_inventory.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "mango_prebuild_inventory_v1"
    assert payload["generator_command_sha256"].startswith("sha256:")
    assert payload["queries"][:2] == ["feature.real", "problem.real"]
    assert (out / "prebuild_inventory.md").exists()


def _git_test_repo(path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    for name, text in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)
    graph = path.parent / f"{path.name}_graph" / "graph.json"
    graph.parent.mkdir(parents=True)
    graph.write_text('{"nodes": []}', encoding="utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
    (graph.parent / "mango_structural_manifest.json").write_text(json.dumps({"revision": head}), encoding="utf-8")
    return path, graph


def _real_inventory(repo: Path, graph: Path, *, symbols: list[str], keywords: list[str]):
    return inventory_before_build.run_inventory(
        repo,
        feature_id="feature.real",
        problem_id="problem.real",
        change="new",
        keywords=keywords,
        symbols=symbols,
        graph=graph,
    )


def test_inventory_real_git_blocks_broad_keyword_and_staged_rename(tmp_path: Path) -> None:
    broad, broad_graph = _git_test_repo(tmp_path / "broad", {"src/unrelated.py": "data = 1\n"})
    broad_result = _real_inventory(broad, broad_graph, symbols=[], keywords=["data"])
    assert broad_result.decision == "stop"
    assert broad_result.unresolved == ["keyword_lead_requires_classification"]

    renamed, rename_graph = _git_test_repo(tmp_path / "renamed", {"src/owner.py": "def exact_owner():\n    return 1\n"})
    before = _real_inventory(renamed, rename_graph, symbols=["exact_owner"], keywords=[])
    repeated = _real_inventory(renamed, rename_graph, symbols=["exact_owner"], keywords=[])
    assert before.selected_owner["path"] == "src/owner.py"
    assert before.candidates == repeated.candidates
    (renamed / "docs").mkdir()
    subprocess.run(["git", "mv", "src/owner.py", "docs/owner.md"], cwd=renamed, check=True)
    rename_result = _real_inventory(renamed, rename_graph, symbols=["exact_owner"], keywords=[])
    assert rename_result.decision == "stop"
    assert any(item.classification == "PARTIAL_WORKTREE" and item.path == "src/owner.py" for item in rename_result.candidates)


def test_inventory_real_git_removed_owner_wins_over_donor(tmp_path: Path) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {"src/owner.py": "def exact_owner():\n    return 1\n"})
    subprocess.run(["git", "switch", "-qc", "donor"], cwd=repo, check=True)
    (repo / "src/owner.py").write_text("def exact_owner():\n    return 2\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "donor change"], cwd=repo, check=True)
    subprocess.run(["git", "switch", "-q", "main"], cwd=repo, check=True)
    subprocess.run(["git", "rm", "-q", "src/owner.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "remove owner"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    (graph.parent / "mango_structural_manifest.json").write_text(json.dumps({"revision": head}), encoding="utf-8")

    result = _real_inventory(repo, graph, symbols=["exact_owner"], keywords=[])

    assert {item.classification for item in result.candidates} >= {"DONOR_REF", "REMOVED_INTENTIONALLY"}
    assert result.decision == "stop"
    assert result.unresolved == ["removed_intentionally_requires_owner_decision"]


def test_inventory_fingerprint_does_not_follow_untracked_symlink_and_stops(tmp_path: Path, monkeypatch) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {"README.md": "clean\n"})
    (repo / "src").mkdir()
    external = tmp_path / "external.txt"
    external.write_text("def exact_owner(): pass\n", encoding="utf-8")
    (repo / "src/link.py").symlink_to(external)
    monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sqlite forbidden")))

    first = _real_inventory(repo, graph, symbols=["exact_owner"], keywords=[])
    external.write_text("changed sensitive content\n", encoding="utf-8")
    second = _real_inventory(repo, graph, symbols=["exact_owner"], keywords=[])

    assert first.status_fingerprint == second.status_fingerprint
    assert first.decision == second.decision == "stop"
    assert any(item.classification == "PARTIAL_WORKTREE" and item.source == "dirty_symlink" for item in first.candidates)


def test_inventory_real_git_stops_on_conflicting_active_owners(tmp_path: Path) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {
        "src/one.py": "def exact_owner(): pass\n",
        "scripts/two.py": "def exact_owner(): pass\n",
    })

    result = _real_inventory(repo, graph, symbols=["exact_owner"], keywords=[])

    assert result.decision == "stop"
    assert result.unresolved == ["active_owner_conflict:exact_owner"]


def test_inventory_fingerprint_covers_refs_graph_metadata_and_queries(tmp_path: Path) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {"src/owner.py": "def exact_owner(): pass\n"})
    fingerprints = [_real_inventory(repo, graph, symbols=["exact_owner"], keywords=[]).status_fingerprint]

    subprocess.run(["git", "branch", "unmerged-candidate"], cwd=repo, check=True)
    fingerprints.append(_real_inventory(repo, graph, symbols=["exact_owner"], keywords=[]).status_fingerprint)

    manifest = graph.parent / "mango_structural_manifest.json"
    manifest.write_text(json.dumps({"revision": "different"}), encoding="utf-8")
    fingerprints.append(_real_inventory(repo, graph, symbols=["exact_owner"], keywords=[]).status_fingerprint)

    task = repo / "tasks/_inbox_codex/task.md"
    task.parent.mkdir(parents=True)
    task.write_text("Feature-ID: feature.real\n", encoding="utf-8")
    fingerprints.append(_real_inventory(repo, graph, symbols=["exact_owner"], keywords=[]).status_fingerprint)

    audit = repo / "audits/_inbox/check/implementation_notes.md"
    audit.parent.mkdir(parents=True)
    audit.write_text("problem.real\n", encoding="utf-8")
    fingerprints.append(_real_inventory(repo, graph, symbols=["exact_owner"], keywords=[]).status_fingerprint)

    fingerprints.append(_real_inventory(repo, graph, symbols=["exact_owner"], keywords=["new query"]).status_fingerprint)
    assert len(set(fingerprints)) == len(fingerprints)


def test_inventory_fails_closed_when_rg_stage_errors(tmp_path: Path, monkeypatch) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {"src/owner.py": "def exact_owner(): pass\n"})
    real_run = inventory_before_build._run

    def fail_rg(root: Path, command, *, timeout: int = 60):
        if command and command[0] == "rg":
            return subprocess.CompletedProcess(command, 2, "", "broken rg")
        return real_run(root, command, timeout=timeout)

    monkeypatch.setattr(inventory_before_build, "_run", fail_rg)
    with pytest.raises(RuntimeError, match="rg failed"):
        _real_inventory(repo, graph, symbols=["exact_owner"], keywords=[])


def test_inventory_is_deterministic_across_python_hash_seeds(tmp_path: Path) -> None:
    repo, graph = _git_test_repo(tmp_path / "repo", {
        "src/owner.py": "def exact_owner(): pass\n",
        "scripts/reference.py": "value = 'exact_owner'\n",
    })

    def run(seed: str) -> dict[str, object]:
        result = subprocess.run(
            [
                sys.executable, str(Path(inventory_before_build.__file__)), "--root", str(repo),
                "--feature-id", "feature.real", "--problem-id", "problem.real", "--change", "extend",
                "--symbols", "exact_owner", "--graph", str(graph), "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        payload = json.loads(result.stdout)
        payload.pop("generated_at")
        return payload

    assert run("1") == run("random")
