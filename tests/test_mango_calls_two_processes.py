from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.calls_two_processes import (
    CallsTwoProcessesConfig,
    LockBusy,
    PARALLEL_PIPELINE_STAGES,
    assert_no_pdn,
    call_event_source_systems,
    capture_mango_window,
    codex_network_available,
    dead_letter_total,
    dead_letter_mass_failure,
    prepare_ingest_inputs,
    prepare_codex_home,
    process_lease,
    pipeline_stages,
    publish_ready_db,
    read_known_processed_ids,
    run_parallel_pipeline_workers,
    run_process_b,
    safe_daily_payload,
    worker_command,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.productization.contracts import Direction, TelephonyCallEvent, TenantRef


def config_for(tmp_path: Path, *, timeline_name: str = "customer_timeline_staging.sqlite") -> CallsTwoProcessesConfig:
    allowed = tmp_path / "staging"
    allowed.mkdir(parents=True, exist_ok=True)
    return CallsTwoProcessesConfig(
        pipeline_root=tmp_path / "pipeline",
        timeline_db=allowed / timeline_name,
        timeline_allowed_root=allowed,
        python_executable=Path(sys.executable),
        codex_binary=Path(sys.executable),
        codex_home_root=tmp_path / "codex_home",
        min_free_gib=1,
    )


def test_config_refuses_prod_and_stable_runtime_paths(tmp_path: Path) -> None:
    prod = config_for(tmp_path, timeline_name="customer_timeline_prod_20260709.sqlite")
    with pytest.raises(ValueError, match="prod"):
        prod.validate()

    stable = CallsTwoProcessesConfig(
        pipeline_root=tmp_path / "stable_runtime" / "calls",
        timeline_db=tmp_path / "staging" / "customer_timeline.sqlite",
        timeline_allowed_root=tmp_path / "staging",
        python_executable=Path(sys.executable),
        codex_binary=Path(sys.executable),
        codex_home_root=tmp_path / "codex_home",
    )
    with pytest.raises(ValueError, match="stable_runtime"):
        stable.validate()


def test_process_a_lock_is_nonblocking_and_reports_holder(tmp_path: Path) -> None:
    lock = tmp_path / "process_a.lock"
    with process_lease(lock, stale_seconds=60) as first:
        assert first["pid"]
        with pytest.raises(LockBusy) as caught:
            with process_lease(lock, stale_seconds=60):
                pass
        assert caught.value.metadata["pid"] == first["pid"]


def test_capture_does_not_commit_calls_without_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)
    tenant = TenantRef("foton")
    no_recording = TelephonyCallEvent(
        tenant=tenant,
        provider="mango",
        provider_call_id="late",
        started_at=datetime(2026, 7, 9, tzinfo=timezone.utc),
        ended_at=None,
        direction=Direction.INBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref=None,
        raw_payload={},
    )
    ready = TelephonyCallEvent(
        tenant=tenant,
        provider="mango",
        provider_call_id="ready",
        started_at=datetime(2026, 7, 9, 1, tzinfo=timezone.utc),
        ended_at=None,
        direction=Direction.OUTBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref="recording-1",
        raw_payload={},
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            self.calls = 0

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            self.calls += 1
            return [{"id": "late"}, {"id": "ready"}] if self.calls == 1 else []

    class FakeMapper:
        def __init__(self) -> None:
            self.items = iter((no_recording, ready))

        def from_payload(self, **_: object) -> TelephonyCallEvent:
            return next(self.items)

    captured: list[TelephonyCallEvent] = []

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0}

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        captured.extend(events)
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficePayloadMapper", FakeMapper)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 2, tzinfo=timezone.utc),
    )

    assert [event.provider_call_id for event in captured] == ["ready"]
    assert report["api_requests"] == 2
    assert report["api_events_without_recording"] == 1


def test_prepare_ingest_inputs_is_idempotent(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    source = config.recordings_dir / "call.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    config.capture_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.capture_manifest.write_text(
        json.dumps(
            {
                "schema_version": "capture_manifest_v1",
                "created_at": "2026-07-09T00:00:00+00:00",
                "tenant_id": "foton",
                "provider": "mango",
                "event_key": "event:1",
                "provider_call_id": "call-1",
                "recording_id": "recording-1",
                "started_at": "2026-07-09T00:00:00+00:00",
                "direction": "inbound",
                "status": "downloaded",
                "local_audio_path": str(source),
                "size_bytes": source.stat().st_size,
                "checksum_sha256": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = prepare_ingest_inputs(config)
    second = prepare_ingest_inputs(config)

    assert first["audio_files"] == second["audio_files"] == 1
    assert second["link_actions"] == {"exists_same_hash": 1}


def test_known_processed_ids_only_accept_successful_downloads(tmp_path: Path) -> None:
    root = tmp_path / "product_data"
    package = root / "mango_update_after_test"
    package.mkdir(parents=True)
    rows = [
        {"action": "DOWNLOADED_RECORDING", "recording_id": "ready", "provider_call_id": "call-ready"},
        {"action": "FAILED_DOWNLOAD", "recording_id": "retry", "provider_call_id": "call-retry"},
    ]
    (package / "recording_download_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    recordings, calls = read_known_processed_ids(root)

    assert recordings == {"ready"}
    assert calls == {"call-ready"}


def test_worker_command_is_drain_and_never_sync(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    command = worker_command(config, "resolve,analyze")
    assert "--poll-sec" in command
    assert "--max-idle-cycles" in command
    assert command[command.index("--stage-limit") + 1] == "20"
    assert "sync" not in command


def test_pipeline_matches_ui_one_stage_at_a_time(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_runner(command, env, cwd):
        del cwd
        calls.append((list(command), dict(env)))
        return {"rc": 0}

    result = run_parallel_pipeline_workers(config, {}, fake_runner)

    assert len(result) == len(PARALLEL_PIPELINE_STAGES) == 4
    assert [command[command.index("--stages") + 1] for command, _ in calls] == list(
        PARALLEL_PIPELINE_STAGES
    )
    assert calls[0][1]["DUAL_TRANSCRIBE_ENABLED"] == "0"
    assert calls[1][1]["DUAL_TRANSCRIBE_ENABLED"] == "1"
    assert calls[2][1]["DUAL_TRANSCRIBE_ENABLED"] == "1"
    assert calls[3][1]["DUAL_TRANSCRIBE_ENABLED"] == "1"
    assert all("sync" not in command for command, _ in calls)


def test_single_asr_fallback_mode_is_rejected(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), asr_mode="gigaam_fallback")
    with pytest.raises(ValueError, match="single-ASR fallback is disabled"):
        config.validate()


def test_publish_ready_db_handles_space_path_wal_tmp(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), pipeline_root=tmp_path / "Mango analyse" / "pipeline")
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records(id) VALUES (1)")

    manifest = publish_ready_db(config, {"total": 1})

    assert manifest["quick_check"] == "ok"
    assert config.ready_db.exists()
    assert not config.ready_db.with_suffix(".sqlite.tmp-shm").exists()


def test_network_outage_runs_only_local_asr_stages(tmp_path: Path) -> None:
    normal = config_for(tmp_path)

    assert pipeline_stages(normal, include_llm=False) == ("transcribe", "backfill-second-asr")


def test_codex_network_probe_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError("dns unavailable")

    monkeypatch.setattr("socket.getaddrinfo", fail)
    assert codex_network_available() is False


def test_dead_letter_total_ignores_empty_stage_and_counts_failures() -> None:
    assert dead_letter_total({"dead_letter_stage": {"": 200, "transcribe": 2, "analyze": 1}}) == 3
    assert dead_letter_mass_failure({"total": 241, "dead_letter_stage": {"transcribe": 3}}) is False
    assert dead_letter_mass_failure({"total": 241, "dead_letter_stage": {"transcribe": 13}}) is True


def test_codex_runtime_does_not_copy_desktop_mcp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    source = home / ".codex"
    source.mkdir(parents=True)
    (source / "auth.json").write_text('{"auth":"masked"}', encoding="utf-8")
    (source / "config.toml").write_text('[mcp_servers.live]\ncommand="unsafe"\n', encoding="utf-8")
    (source / "AGENTS.md").write_text("desktop personality", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    target = prepare_codex_home(tmp_path / "runtime")

    assert "mcp_servers" not in (target / "config.toml").read_text(encoding="utf-8")
    assert "desktop personality" not in (target / "AGENTS.md").read_text(encoding="utf-8")
    assert (target / "auth.json").is_file()


def test_codex_wrapper_disables_desktop_tools(tmp_path: Path) -> None:
    captured = tmp_path / "args.txt"
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/bin/zsh\nprintf '%s\\n' \"$@\" > \"$CAPTURED\"\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    wrapper = Path(__file__).resolve().parents[1] / "scripts" / "run_codex_cli_isolated.sh"
    env = {**os.environ, "MANGO_CODEX_REAL_BIN": str(fake), "CAPTURED": str(captured)}

    subprocess.run([str(wrapper), "exec", "--model", "test", "prompt"], env=env, check=True)

    args = captured.read_text(encoding="utf-8").splitlines()
    assert args[0] == "exec"
    assert args.count("--disable") == 5
    assert "apps" in args and "plugins" in args and "browser_use" in args
    assert args[-3:] == ["--model", "test", "prompt"]


def create_ready_call_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE call_records (
                id INTEGER PRIMARY KEY,
                source_call_id TEXT,
                source_filename TEXT,
                source_file TEXT,
                started_at TEXT,
                phone TEXT,
                manager_name TEXT,
                direction TEXT,
                duration_sec REAL,
                analysis_status TEXT,
                analysis_json TEXT,
                amocrm_contact_id TEXT,
                amocrm_lead_id TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO call_records (
                id, source_call_id, source_filename, source_file, started_at,
                phone, manager_name, direction, duration_sec,
                analysis_status, analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "provider-1",
                "masked.mp3",
                "/ignored/masked.mp3",
                "2026-07-09T10:00:00+00:00",
                "",
                "manager",
                "inbound",
                60.0,
                "done",
                json.dumps({"call_type": "sales_call", "history_summary": "Обсуждался курс."}),
            ),
        )


def test_process_b_returns_locked_instead_of_traceback(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    holder = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root)
    try:
        def fake_producer(_: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None) -> dict[str, object]:
            del since
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")
            return {"status": "ok", "events_written": 0}

        result = run_process_b(config, producer_runner=fake_producer)
    finally:
        holder.close()

    assert result["status"] == "locked"
    assert result["stop_reason"] == "timeline_writer_locked"


def test_process_b_is_idempotent_and_keeps_one_source_system(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root)
    store.close()

    first = run_process_b(config)
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        first_count = int(
            con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type='mango_call'").fetchone()[0]
        )
    second = run_process_b(config)
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        second_count = int(
            con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type='mango_call'").fetchone()[0]
        )

    assert first["status"] == second["status"] == "ok"
    assert first_count == second_count == 1
    assert call_event_source_systems(config.timeline_db) == ["mango_processed_summary"]


def test_process_b_does_not_skip_late_old_call_by_timestamp_cursor(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root)
    store.close()
    first = run_process_b(config)
    assert first["status"] == "ok"
    seen_since: list[str | None] = []

    def fake_producer(_: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None) -> dict[str, object]:
        seen_since.append(since)
        out.write_text("", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return {"status": "ok", "events_written": 0}

    second = run_process_b(config, producer_runner=fake_producer)

    assert second["status"] == "ok"
    assert seen_since == [None]
    assert second["counters"]["producer_scan_mode"] == "full_drop_dedupe"


def test_foton_pdn_sweep_blocks_phone() -> None:
    with pytest.raises(RuntimeError, match="pdn-sweep"):
        assert_no_pdn({"text": "Позвонить +7 999 123-45-67"})
    assert_no_pdn({"calls": 22, "status": "ok"})


def test_locked_report_does_not_claim_work_or_publish_pid() -> None:
    payload = safe_daily_payload(
        {
            "schema_version": "v1",
            "run_id": "masked",
            "process": "process_a",
            "status": "locked",
            "stop_reason": "process_a_locked",
            "counters": {"lock": {"pid": 12345, "previous_pid": 111}},
            "safety": {"runs_asr": False, "runs_resolve_analyze": False},
        }
    )

    assert payload["counters"]["lock"] == {}
    assert payload["safety"]["runs_asr"] is False


def test_launchd_installer_defaults_to_near_realtime_900_seconds(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config_path = tmp_path / "config.json"
    env_path = tmp_path / "mango.env"
    plist_path = tmp_path / "calls.plist"
    config_path.write_text(
        json.dumps(
            {
                "pipeline_root": str(config.pipeline_root),
                "timeline_db": str(config.timeline_db),
                "timeline_allowed_root": str(config.timeline_allowed_root),
                "python_executable": str(config.python_executable),
                "codex_binary": str(config.codex_binary),
                "codex_home_root": str(config.codex_home_root),
                "poll_overlap_minutes": 30,
            }
        ),
        encoding="utf-8",
    )
    env_path.write_text("MANGO_OFFICE_API_KEY=x\nMANGO_OFFICE_API_SALT=y\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/install_mango_calls_two_processes_service.py",
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "--out",
            str(plist_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    assert payload["StartInterval"] == 900
