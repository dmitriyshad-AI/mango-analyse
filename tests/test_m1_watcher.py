import hashlib
import json
import os
from pathlib import Path

from scripts import m1_watcher as watcher
from scripts.build_mango_clean_bundle import write_bundle_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(root: Path, bundle_id: str = "mango_clean_12345678") -> Path:
    bundle = root / bundle_id
    (bundle / "scripts").mkdir(parents=True)
    (bundle / "src" / "mango_mvp").mkdir(parents=True)
    (bundle / "src" / "mango_mvp" / "__init__.py").write_text("", encoding="utf-8")
    (bundle / "scripts" / "run_telegram_dynamic_client_sim.py").write_text("print('ok')\n", encoding="utf-8")
    (bundle / "BUNDLE_INFO.txt").write_text(
        "\n".join(
            [
                bundle_id,
                "head: 1234567890abcdef",
                "kb_snapshot: product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/kb_release_v3_snapshot.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_bundle_manifest(bundle, bundle_id, "1234567890abcdef")
    return bundle


def _make_set(root: Path, rel: str = "sets/smoke.jsonl") -> tuple[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"type":"persona","dialog_id":"d1"}\n', encoding="utf-8")
    return rel, _sha(path)


def _write_task(root: Path, name: str, *, task_id: str = "task1", bundle: str = "mango_clean_12345678", set_rel: str, set_sha: str, parallel: int = 4, brain: str = "codex", env: str = '  TELEGRAM_TEST_FLAG: "1"\n', max_hours: str = "1") -> Path:
    inbox = root / "tasks" / "_inbox_m1"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(
        "\n".join(
            [
                f"id: {task_id}",
                f"requires_bundle: {bundle}",
                f"set: {set_rel}",
                f"set_sha256: {set_sha}",
                f"brain: {brain}",
                f"parallel: {parallel}",
                f"max_hours: {max_hours}",
                "env:",
                env.rstrip("\n"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".ready").write_text(_sha(path), encoding="utf-8")
    return path


def _fake_runner(spec, deploy_dir, out_dir, command, env):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dynamic_summary.json").write_text(
        json.dumps({"overall_verdict": "PASS", "hard_gates_passed": True, "total_turns": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "dynamic_dialog_transcripts.jsonl").write_text("{}\n", encoding="utf-8")
    return watcher.RunOutcome(0, True, False, command=tuple(command))


def _fake_probe(args, cwd, env, timeout):
    args = tuple(args)
    if args[:2] in {("codex", "--version"), ("claude", "--version")}:
        return 0, f"{args[0]} test-version\n", ""
    if args[:3] == ("claude", "auth", "status"):
        return 0, '{"authenticated":true}\n', ""
    if len(args) >= 3 and args[1] == "-c" and args[2] == "import mango_mvp":
        return 0, "", ""
    return 0, "", ""


def _new_watcher(tmp_path: Path, **kwargs) -> watcher.M1Watcher:
    return watcher.M1Watcher(
        tmp_path,
        tmp_path / "work",
        tmp_path / "state",
        bundle_wait_seconds=kwargs.pop("bundle_wait_seconds", 0),
        min_free_bytes=kwargs.pop("min_free_bytes", 0),
        runner=kwargs.pop("runner", _fake_runner),
        command_probe=kwargs.pop("command_probe", _fake_probe),
        **kwargs,
    )


def _run_ready_cycle(w: watcher.M1Watcher) -> str:
    assert w.process_once() == "idle"
    return w.process_once()


def test_watcher_creates_failed_dir_and_executes_valid_task(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_valid.task.yaml", set_rel=set_rel, set_sha=set_sha)

    w = _new_watcher(tmp_path)
    assert not (tmp_path / "tasks" / "_failed").exists()
    assert _run_ready_cycle(w) == "success"

    assert (tmp_path / "tasks" / "_failed").is_dir()
    assert (tmp_path / "tasks" / "_done" / "task1.report.md").exists()
    assert (tmp_path / "runs" / "task1" / "dynamic_summary.json").exists()


def test_success_report_includes_readiness_cli_versions_and_heartbeat(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_valid.task.yaml", set_rel=set_rel, set_sha=set_sha)

    assert _run_ready_cycle(_new_watcher(tmp_path)) == "success"

    report = (tmp_path / "tasks" / "_done" / "task1.report.md").read_text(encoding="utf-8")
    assert "codex_cli_version: codex test-version" in report
    assert "claude_cli_version: claude test-version" in report
    assert "readiness:" in report
    assert "--judge-prompt-version v9.1" in report
    assert "import_mango_mvp" in report
    assert "disk_free_bytes" in report
    heartbeat = json.loads((tmp_path / "tasks" / "_done" / "task1.heartbeat").read_text(encoding="utf-8"))
    assert heartbeat["task_id"] == "task1"
    assert heartbeat["counter"] >= 4
    assert heartbeat["stage"] == "success"
    assert heartbeat["transcript_exists"] is True
    assert heartbeat["stall"] is False


def test_task_schema_accepts_explicit_judge_v91(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    task = _write_task(tmp_path, "2026-06-07_judge_v91.task.yaml", set_rel=set_rel, set_sha=set_sha)
    task.write_text(task.read_text(encoding="utf-8") + "judge_prompt_version: v9.1\n", encoding="utf-8")
    task.with_suffix(task.suffix + ".ready").write_text(_sha(task), encoding="utf-8")

    assert _run_ready_cycle(_new_watcher(tmp_path)) == "success"
    report = (tmp_path / "tasks" / "_done" / "task1.report.md").read_text(encoding="utf-8")
    assert "--judge-prompt-version v9.1" in report


def test_claude_task_uses_longer_llm_timeout(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_claude_timeout.task.yaml", brain="claude", set_rel=set_rel, set_sha=set_sha)

    assert _run_ready_cycle(_new_watcher(tmp_path)) == "success"

    report = (tmp_path / "tasks" / "_done" / "task1.report.md").read_text(encoding="utf-8")
    assert "--timeout-sec 270" in report


def test_empty_task_env_uses_production_stack_and_reports_llm_breakdown(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_empty_env.task.yaml", set_rel=set_rel, set_sha=set_sha, env="")
    seen_env: dict[str, str] = {}

    def runner(spec, deploy_dir, out_dir, command, env):
        seen_env.update(dict(env))
        assert env["TELEGRAM_DIALOGUE_CONTRACT_PIPELINE"] == "1"
        assert env["TELEGRAM_SEMANTIC_OUTPUT_VERIFIER"] == "1"
        assert env["TELEGRAM_TONE_SELL_PROMPT"] == "1"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dynamic_summary.json").write_text(
            json.dumps(
                {
                    "overall_verdict": "PASS",
                    "hard_gates_passed": True,
                    "total_turns": 2,
                    "llm_calls": {
                        "bot_faithfulness": 3,
                        "bot_semantic_output_verifier": 2,
                        "total": 5,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (out_dir / "dynamic_dialog_transcripts.jsonl").write_text("{}\n", encoding="utf-8")
        return watcher.RunOutcome(0, True, False, command=tuple(command))

    assert _run_ready_cycle(_new_watcher(tmp_path, runner=runner)) == "success"

    report = (tmp_path / "tasks" / "_done" / "task1.report.md").read_text(encoding="utf-8")
    assert "task_delta: {}" in report
    assert '"TELEGRAM_DIALOGUE_CONTRACT_PIPELINE": "1"' in report
    assert '"TELEGRAM_SEMANTIC_OUTPUT_VERIFIER": "1"' in report
    assert '"bot_faithfulness": 3' in report
    assert seen_env["PYTHONPATH"] == "src"


def test_task_env_delta_overrides_default_production_stack(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(
        tmp_path,
        "2026-06-07_env_override.task.yaml",
        set_rel=set_rel,
        set_sha=set_sha,
        env='  TELEGRAM_TONE_SELL_PROMPT: "0"\n',
    )

    def runner(spec, deploy_dir, out_dir, command, env):
        assert env["TELEGRAM_TONE_SELL_PROMPT"] == "0"
        assert env["TELEGRAM_DIALOGUE_CONTRACT_PIPELINE"] == "1"
        assert env["TELEGRAM_STEP4_KEEP_ANSWER"] == "1"
        return _fake_runner(spec, deploy_dir, out_dir, command, env)

    assert _run_ready_cycle(_new_watcher(tmp_path, runner=runner)) == "success"

    report = (tmp_path / "tasks" / "_done" / "task1.report.md").read_text(encoding="utf-8")
    assert 'task_delta: {"TELEGRAM_TONE_SELL_PROMPT": "0"}' in report
    assert '"TELEGRAM_TONE_SELL_PROMPT": "0"' in report
    assert '"TELEGRAM_STEP4_KEEP_ANSWER": "1"' in report


def test_effective_task_env_adds_canonical_profile_only_when_enforced() -> None:
    plain = watcher.effective_task_env({})
    enforced = watcher.effective_task_env({"ENFORCE_CANONICAL_PROFILE": "1"})

    assert "TELEGRAM_DIRECT_PATH_PILOT_CONFIG" not in plain
    assert enforced["TELEGRAM_DIRECT_PATH_PILOT_CONFIG"] == "pilot_gold_v1"


def test_status_writes_date_rotated_log_tail(tmp_path):
    w = _new_watcher(tmp_path)
    assert w.process_once() == "idle"

    logs = sorted((tmp_path / "state" / "logs").glob("*.log"))
    assert len(logs) == 1
    assert logs[0].name.endswith(".log")
    status = (tmp_path / "tasks" / "watcher_status.txt").read_text(encoding="utf-8")
    assert f"log_path: {logs[0]}" in status
    assert "log_tail:" in status
    assert '"event": "status"' in status


def test_heartbeat_marks_stale_transcript_as_stalled(tmp_path):
    w = _new_watcher(tmp_path, stall_seconds=1)
    (tmp_path / "tasks" / "_running").mkdir(parents=True)
    out_dir = tmp_path / "work" / "mango_clean_12345678" / "runs" / "task1"
    out_dir.mkdir(parents=True)
    transcript = out_dir / "dynamic_dialog_transcripts.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    old = transcript.stat().st_mtime - 10
    os.utime(transcript, (old, old))

    w._write_heartbeat("task1", "running", out_dir)

    heartbeat = json.loads((tmp_path / "tasks" / "_running" / "task1.heartbeat").read_text(encoding="utf-8"))
    assert heartbeat["counter"] == 1
    assert heartbeat["stage"] == "running"
    assert heartbeat["transcript_exists"] is True
    assert heartbeat["transcript_mtime_age_seconds"] >= 1
    assert heartbeat["stall"] is True


def test_readiness_rejects_import_failure_disk_shortage_and_claude_auth_failure(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)

    def import_fail_probe(args, cwd, env, timeout):
        if len(args) >= 3 and args[1] == "-c" and args[2] == "import mango_mvp":
            return 1, "", "missing mango_mvp"
        return _fake_probe(args, cwd, env, timeout)

    _write_task(tmp_path, "2026-06-07_import.task.yaml", task_id="import_bad", set_rel=set_rel, set_sha=set_sha)
    assert _run_ready_cycle(_new_watcher(tmp_path, command_probe=import_fail_probe)) == "readiness_failed"
    assert "missing mango_mvp" in (tmp_path / "tasks" / "_failed" / "import_bad.report.md").read_text(encoding="utf-8")

    case_disk = tmp_path / "disk"
    _make_bundle(case_disk)
    disk_set_rel, disk_set_sha = _make_set(case_disk)
    _write_task(case_disk, "2026-06-07_disk.task.yaml", task_id="disk_bad", set_rel=disk_set_rel, set_sha=disk_set_sha)
    assert _run_ready_cycle(_new_watcher(case_disk, min_free_bytes=10**30)) == "readiness_failed"
    assert "free disk below threshold" in (case_disk / "tasks" / "_failed" / "disk_bad.report.md").read_text(encoding="utf-8")

    case_claude = tmp_path / "claude"
    _make_bundle(case_claude)
    claude_set_rel, claude_set_sha = _make_set(case_claude)
    _write_task(case_claude, "2026-06-07_claude.task.yaml", task_id="claude_bad", brain="claude", set_rel=claude_set_rel, set_sha=claude_set_sha)

    def claude_fail_probe(args, cwd, env, timeout):
        if tuple(args[:3]) == ("claude", "auth", "status"):
            return 1, "", "not logged in"
        return _fake_probe(args, cwd, env, timeout)

    assert _run_ready_cycle(_new_watcher(case_claude, command_probe=claude_fail_probe)) == "readiness_failed"
    assert "claude auth status failed" in (case_claude / "tasks" / "_failed" / "claude_bad.report.md").read_text(encoding="utf-8")


def test_parallel_env_and_set_path_are_rejected(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)

    _write_task(tmp_path, "2026-06-07_parallel.task.yaml", task_id="bad_parallel", set_rel=set_rel, set_sha=set_sha, parallel=8)
    assert _run_ready_cycle(_new_watcher(tmp_path)) == "task_schema_mismatch"

    _write_task(tmp_path, "2026-06-07_max_hours.task.yaml", task_id="nine_hours", set_rel=set_rel, set_sha=set_sha, max_hours="9")
    assert _run_ready_cycle(_new_watcher(tmp_path)) == "success"

    _write_task(tmp_path, "2026-06-07_env.task.yaml", task_id="bad_env", set_rel=set_rel, set_sha=set_sha, env='  PATH: "/tmp"\n')
    assert _run_ready_cycle(_new_watcher(tmp_path)) == "task_schema_mismatch"

    _write_task(tmp_path, "2026-06-07_path.task.yaml", task_id="bad_path", set_rel="../outside.jsonl", set_sha=set_sha)
    assert _run_ready_cycle(_new_watcher(tmp_path)) == "task_schema_mismatch"


def test_conflict_copy_is_ignored_and_fifo_is_lexicographic(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_second.task.yaml", task_id="second", set_rel=set_rel, set_sha=set_sha)
    _write_task(tmp_path, "2026-06-07_first.task.yaml", task_id="first", set_rel=set_rel, set_sha=set_sha)
    _write_task(tmp_path, "2026-06-07_first (1).task.yaml", task_id="conflict", set_rel=set_rel, set_sha=set_sha)

    w = _new_watcher(tmp_path)
    assert _run_ready_cycle(w) == "success"
    assert (tmp_path / "tasks" / "_done" / "first.report.md").exists()
    assert not (tmp_path / "tasks" / "_done" / "conflict.report.md").exists()


def test_stop_blocks_then_allows_after_removal(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_stop.task.yaml", set_rel=set_rel, set_sha=set_sha)
    stop = tmp_path / "tasks" / "STOP"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text("stop\n", encoding="utf-8")

    w = _new_watcher(tmp_path)
    assert w.process_once() == "stopped"
    stop.unlink()
    assert _run_ready_cycle(w) == "success"


def test_single_flight_live_orphan_waits_and_dead_orphan_fails(tmp_path):
    (tmp_path / "tasks" / "_running").mkdir(parents=True)
    (tmp_path / "tasks" / "_running" / "live.task.yaml").write_text("id: live\n", encoding="utf-8")
    (tmp_path / "tasks" / "_running" / "live.claimed").write_text('{"pgid": 999}\n', encoding="utf-8")

    w_live = _new_watcher(tmp_path, process_alive=lambda pid: True)
    assert w_live.process_once() == "running_present"
    assert "orphan_run_alive" in (tmp_path / "tasks" / "watcher_status.txt").read_text(encoding="utf-8")

    w_dead = _new_watcher(tmp_path, process_alive=lambda pid: False)
    assert w_dead.process_once() == "running_present"
    assert (tmp_path / "tasks" / "_failed" / "live.report.md").exists()
    assert "interrupted" in (tmp_path / "tasks" / "_failed" / "live.report.md").read_text(encoding="utf-8")


def test_bundle_manifest_missing_or_mismatch_waits_then_fails(tmp_path):
    set_rel, set_sha = _make_set(tmp_path)
    bundle = tmp_path / "mango_clean_12345678"
    bundle.mkdir()
    (bundle / "BUNDLE_INFO.txt").write_text("kb_snapshot: x\n", encoding="utf-8")
    _write_task(tmp_path, "2026-06-07_bundle.task.yaml", set_rel=set_rel, set_sha=set_sha)

    assert _run_ready_cycle(_new_watcher(tmp_path, bundle_wait_seconds=0)) == "bundle_incomplete"

    # Existing manifest with changed payload is also incomplete, not deployed.
    tmp2 = tmp_path / "case2"
    _make_bundle(tmp2)
    (tmp2 / "mango_clean_12345678" / "BUNDLE_INFO.txt").write_text("kb_snapshot: changed\n", encoding="utf-8")
    set_rel2, set_sha2 = _make_set(tmp2)
    _write_task(tmp2, "2026-06-07_bundle.task.yaml", set_rel=set_rel2, set_sha=set_sha2)
    assert _run_ready_cycle(_new_watcher(tmp2, bundle_wait_seconds=0)) == "bundle_incomplete"


def test_ready_marker_required_before_task_is_taken(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    path = _write_task(tmp_path, "2026-06-07_ready.task.yaml", set_rel=set_rel, set_sha=set_sha)
    path.with_suffix(path.suffix + ".ready").unlink()

    w = _new_watcher(tmp_path)
    assert w.process_once() == "idle"
    assert not (tmp_path / "tasks" / "_done" / "task1.report.md").exists()
    path.with_suffix(path.suffix + ".ready").write_text(_sha(path), encoding="utf-8")
    assert _run_ready_cycle(w) == "success"


def test_fresh_yaml_parse_error_waits_three_cycles_before_failed(tmp_path):
    inbox = tmp_path / "tasks" / "_inbox_m1"
    inbox.mkdir(parents=True)
    path = inbox / "2026-06-07_bad_yaml.task.yaml"
    path.write_text("id task1\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".ready").write_text(_sha(path), encoding="utf-8")

    w = _new_watcher(tmp_path)
    assert w.process_once() == "idle"  # first stable-file cycle
    assert w.process_once() == "task_parse_waiting"
    assert w.process_once() == "task_parse_waiting"
    assert w.process_once() == "task_parse_waiting"
    assert path.exists()
    assert w.process_once() == "task_schema_mismatch"
    assert not path.exists()
    assert (tmp_path / "tasks" / "_failed" / "2026-06-07_bad_yaml.report.md").exists()

    case_fixed = tmp_path / "fixed"
    _make_bundle(case_fixed)
    set_rel, set_sha = _make_set(case_fixed)
    fixed_path = _write_task(case_fixed, "2026-06-07_later_fixed.task.yaml", set_rel=set_rel, set_sha=set_sha)
    fixed_path.write_text("id task1\n", encoding="utf-8")
    fixed_path.with_suffix(fixed_path.suffix + ".ready").write_text(_sha(fixed_path), encoding="utf-8")

    w_fixed = _new_watcher(case_fixed)
    assert w_fixed.process_once() == "idle"
    assert w_fixed.process_once() == "task_parse_waiting"
    fixed_path.write_text(
        "\n".join(
            [
                "id: task1",
                "requires_bundle: mango_clean_12345678",
                f"set: {set_rel}",
                f"set_sha256: {set_sha}",
                "brain: codex",
                "parallel: 4",
                "max_hours: 1",
                "env:",
                '  TELEGRAM_TEST_FLAG: "1"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    fixed_path.with_suffix(fixed_path.suffix + ".ready").write_text(_sha(fixed_path), encoding="utf-8")
    assert w_fixed.process_once() == "idle"
    assert w_fixed.process_once() == "success"


def test_duplicate_id_rejected(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    done = tmp_path / "tasks" / "_done"
    done.mkdir(parents=True)
    (done / "same.report.md").write_text("executed: true\n", encoding="utf-8")
    _write_task(tmp_path, "2026-06-07_dup.task.yaml", task_id="same", set_rel=set_rel, set_sha=set_sha)

    assert _run_ready_cycle(_new_watcher(tmp_path)) == "duplicate_id"


def test_timeout_marks_failed(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_timeout.task.yaml", set_rel=set_rel, set_sha=set_sha)

    def timeout_runner(spec, deploy_dir, out_dir, command, env):
        out_dir.mkdir(parents=True, exist_ok=True)
        return watcher.RunOutcome(-15, True, True, command=tuple(command))

    assert _run_ready_cycle(_new_watcher(tmp_path, runner=timeout_runner)) == "timeout"
    assert "timeout" in (tmp_path / "tasks" / "_failed" / "task1.report.md").read_text(encoding="utf-8")


def test_config_invalid_summary_gets_named_watcher_status(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)
    _write_task(tmp_path, "2026-06-07_config_invalid.task.yaml", set_rel=set_rel, set_sha=set_sha)

    def invalid_runner(spec, deploy_dir, out_dir, command, env):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dynamic_summary.json").write_text(
            json.dumps({"config_validity": {"invalid": True, "reason": "config_invalid"}}, ensure_ascii=False),
            encoding="utf-8",
        )
        return watcher.RunOutcome(2, True, False, command=tuple(command))

    assert _run_ready_cycle(_new_watcher(tmp_path, runner=invalid_runner)) == "config_invalid"
    assert (tmp_path / "tasks" / "_failed" / "task1.report.md").exists()


def test_ack_counter_blocks_fourth_unconfirmed_executed_task(tmp_path):
    for idx in range(3):
        done = tmp_path / "tasks" / "_done"
        done.mkdir(parents=True, exist_ok=True)
        (done / f"done{idx}.report.md").write_text("executed: true\n", encoding="utf-8")

    w = _new_watcher(tmp_path)
    assert w.process_once() == "awaiting_ack"
    assert "awaiting_ack" in (tmp_path / "tasks" / "watcher_status.txt").read_text(encoding="utf-8")

    (tmp_path / "tasks" / "ACK_2026-06-07.md").write_text("ok\n", encoding="utf-8")
    assert w.process_once() == "idle"


def test_manifest_verification_requires_exact_file_count_sha_and_size(tmp_path):
    bundle = _make_bundle(tmp_path)
    assert watcher.verify_bundle_manifest(bundle)["bundle_id"] == "mango_clean_12345678"
    (bundle / "extra.txt").write_text("extra\n", encoding="utf-8")
    try:
        watcher.verify_bundle_manifest(bundle)
    except watcher.WatcherError as exc:
        assert exc.code == "bundle_waiting"
    else:
        raise AssertionError("extra unmanifested file must fail verification")
