import hashlib
import json
from pathlib import Path

from scripts import m1_watcher as watcher
from scripts.build_mango_clean_bundle import write_bundle_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(root: Path, bundle_id: str = "mango_clean_12345678") -> Path:
    bundle = root / bundle_id
    (bundle / "scripts").mkdir(parents=True)
    (bundle / "scripts" / "run_telegram_dynamic_client_sim.py").write_text("print('ok')\n", encoding="utf-8")
    (bundle / "BUNDLE_INFO.txt").write_text(
        "\n".join(
            [
                bundle_id,
                "head: 1234567890abcdef",
                "kb_snapshot: product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/kb_release_v3_snapshot.json",
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


def _new_watcher(tmp_path: Path, **kwargs) -> watcher.M1Watcher:
    return watcher.M1Watcher(
        tmp_path,
        tmp_path / "work",
        tmp_path / "state",
        bundle_wait_seconds=kwargs.pop("bundle_wait_seconds", 0),
        runner=kwargs.pop("runner", _fake_runner),
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


def test_parallel_env_and_set_path_are_rejected(tmp_path):
    _make_bundle(tmp_path)
    set_rel, set_sha = _make_set(tmp_path)

    _write_task(tmp_path, "2026-06-07_parallel.task.yaml", task_id="bad_parallel", set_rel=set_rel, set_sha=set_sha, parallel=8)
    assert _run_ready_cycle(_new_watcher(tmp_path)) == "task_schema_mismatch"

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
