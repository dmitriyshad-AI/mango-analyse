from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp import cli as mango_cli
import mango_mvp.customer_timeline.calls_two_processes as calls_runtime
from mango_mvp.config import get_settings
from mango_mvp.customer_timeline.calls_two_processes import (
    CallsTwoProcessesConfig,
    LockBusy,
    PARALLEL_PIPELINE_STAGES,
    assert_no_pdn,
    call_db_has_open_work,
    call_event_source_systems,
    command_path,
    capture_mango_window,
    codex_network_available,
    dead_letter_total,
    dead_letter_mass_failure,
    environment_preflight,
    finalize_report,
    hardlink_or_copy,
    module_probe_command,
    missing_capture_recovery_events,
    new_calls_run_id,
    prepare_ingest_inputs,
    prepare_codex_home,
    process_lease,
    pipeline_stages,
    publish_ready_db,
    publish_ready_db_if_changed,
    read_json,
    read_known_processed_ids,
    run_capture,
    run_parallel_pipeline_workers,
    run_pipeline,
    run_process_a,
    run_process_b,
    run_cycle,
    run_command,
    compact_command_reports,
    pipeline_freshness,
    safe_daily_payload,
    sha256_file,
    sqlite_check,
    stage_timeout_deadline,
    terminate_process_group,
    worker_command,
    write_json,
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


def create_empty_capture_manifest(config: CallsTwoProcessesConfig) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore

    CaptureManifestStore(config.capture_manifest).ensure_exists()


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


def test_pipeline_and_direct_process_a_share_one_lock_while_capture_is_independent(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    assert config.process_a_lock == config.pipeline_lock

    with process_lease(config.pipeline_lock, stale_seconds=60):
        pipeline = run_pipeline(
            config,
            command_runner=lambda *_args: pytest.fail("worker must not start"),
        )
        direct = run_process_a(
            config,
            command_runner=lambda *_args: pytest.fail("worker must not start"),
        )
        for _ in range(3):
            with process_lease(config.capture_lock, stale_seconds=60):
                pass

    assert pipeline["status"] == "locked"
    assert pipeline["stop_reason"] == "pipeline_locked"
    assert direct["status"] == "locked"
    assert direct["stop_reason"] == "process_a_locked"


def test_run_capture_writes_cursor_only_for_complete_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calls_runtime, "cutover_authority_report", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(calls_runtime, "disk_preflight", lambda *_a, **_k: {"ok": True})
    complete = config_for(tmp_path / "complete")
    report = run_capture(
        complete,
        since="2026-08-11T00:00:00+00:00",
        until="2026-08-11T01:00:00+00:00",
        capture_runner=lambda *_a: {
            "status": "ok",
            "mango_enumeration_complete": True,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": "a" * 64,
        },
    )
    assert report["status"] == "ok"
    assert read_json(complete.cursor_path)["mango_enumeration_complete"] is True

    incomplete = config_for(tmp_path / "incomplete")
    failed = run_capture(
        incomplete,
        since="2026-08-11T00:00:00+00:00",
        until="2026-08-11T01:00:00+00:00",
        capture_runner=lambda *_a: {
            "status": "partial",
            "mango_enumeration_complete": False,
        },
    )
    assert failed["status"] == "failed"
    assert failed["stop_reason"] == "capture_or_enumeration_failed"
    assert not incomplete.cursor_path.exists()


def test_run_pipeline_reuses_unchanged_frozen_snapshot_without_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    config.capture_manifest.parent.mkdir(parents=True)
    config.capture_manifest.write_bytes(b"")
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest, end_offset=0
    )
    write_json(
        config.cursor_path,
        {
            "mango_enumeration_complete": True,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": snapshot["sha256"],
        },
    )
    monkeypatch.setattr(calls_runtime, "cutover_authority_report", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(
        calls_runtime,
        "run_process_a",
        lambda *_a, **_k: {
            "status": "ok",
            "stop_reason": "",
            "downstream_ready": True,
            "counters": {
                "metadata": {
                    "audio_files": 0,
                    "db_open_work": False,
                    "skipped": {"already_ingested": 3},
                },
                "drop": {"reused": True},
            },
        },
    )
    report = run_pipeline(
        config,
        command_runner=lambda *_a: pytest.fail("worker must not start"),
        process_b_runner=lambda _config: {
            "status": "idle",
            "stop_reason": "drop_unchanged",
        },
    )

    assert report["status"] == "idle"
    assert report["stop_reason"] == "unchanged_snapshot"
    assert report["new"] == 0
    assert report["reused"] == 3


def test_run_pipeline_rejects_changed_capture_snapshot_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    config.capture_manifest.parent.mkdir(parents=True)
    config.capture_manifest.write_bytes(b"")
    write_json(
        config.cursor_path,
        {
            "mango_enumeration_complete": True,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(calls_runtime, "cutover_authority_report", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(
        calls_runtime,
        "run_process_a",
        lambda *_a, **_k: pytest.fail("Process A must not start"),
    )

    report = run_pipeline(config)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "capture_snapshot_sha256_mismatch"


def test_process_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("must remain unchanged", encoding="utf-8")
    lock = tmp_path / "process_a.lock"
    lock.symlink_to(victim)

    with pytest.raises(RuntimeError, match="lock is unsafe"):
        with process_lease(lock, stale_seconds=60):
            pass

    assert lock.is_symlink()
    assert victim.read_text(encoding="utf-8") == "must remain unchanged"


def test_terminate_process_group_kills_children_after_parent_exits_on_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"parent": True, "group": True}
    delivered: list[int] = []

    class FakeProcess:
        pid = 424242

        def poll(self) -> int | None:
            return None if state["parent"] else 0

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            state["parent"] = False
            return 0

    def killpg(_pid: int, sent_signal: int) -> None:
        if sent_signal == 0:
            if not state["group"]:
                raise ProcessLookupError
            return
        delivered.append(sent_signal)
        if sent_signal == 15:
            state["parent"] = False
        elif sent_signal == 9:
            state["group"] = False

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.os.killpg", killpg
    )
    terminate_process_group(FakeProcess(), grace_seconds=0.1)  # type: ignore[arg-type]

    assert delivered == [15, 9]
    assert state == {"parent": False, "group": False}


def test_hardlink_race_rejects_a_concurrent_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"synthetic audio")
    target = tmp_path / "target.mp3"
    real_link = os.link

    def race_link(src: object, dst: object, **_kwargs: object) -> None:
        assert Path(src) == source
        Path(dst).symlink_to(source)
        raise FileExistsError

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.os.link", race_link
    )
    with pytest.raises(RuntimeError, match="concurrent audio target is unsafe"):
        hardlink_or_copy(source, target)

    assert target.is_symlink()
    assert source.read_bytes() == b"synthetic audio"
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.os.link", real_link
    )


def test_run_cycle_imports_partial_ready_drop_and_keeps_partial_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.run_process_a",
        lambda *_args, **_kwargs: {
            "status": "partial",
            "stop_reason": "capture_audio_incomplete",
            "downstream_ready": True,
        },
    )

    def fake_b(*_args, **_kwargs):
        calls.append("b")
        return {"status": "ok"}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.run_process_b", fake_b)
    report = run_cycle(config_for(tmp_path))
    assert calls == ["b"]
    assert report["status"] == "partial"
    assert report["stop_reason"] == "capture_audio_incomplete"


@pytest.mark.parametrize("first", [
    {"status": "partial", "downstream_ready": False},
    {"status": "failed", "downstream_ready": False},
])
def test_run_cycle_does_not_import_without_ready_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: dict[str, object]
) -> None:
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.run_process_a",
        lambda *_args, **_kwargs: first,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.run_process_b",
        lambda *_args, **_kwargs: pytest.fail("Process B must not start"),
    )
    assert run_cycle(config_for(tmp_path))["process_b"] is None


def test_capture_keeps_calls_without_recording_in_retry_queue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        skipped_no_recording = 1

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 1}

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

    assert [event.provider_call_id for event in captured] == ["late", "ready"]
    assert report["status"] == "ok"
    assert report["api_requests"] == 2
    assert report["api_events_without_recording"] == 1


def test_capture_reports_partial_when_one_download_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return []

    class Summary:
        failed = 1

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 1}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 1, tzinfo=timezone.utc),
    )

    assert report["status"] == "partial"
    assert config.capture_manifest.exists()
    assert config.capture_manifest.stat().st_mode & 0o777 == 0o600


def test_capture_refuses_to_recreate_missing_manifest_for_prior_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": "ok",
            "checked_through": "2026-07-09T09:00:00+00:00",
            "data_through": "2026-07-09T09:00:00+00:00",
        },
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            pytest.fail("API poll must not run when a prior manifest is missing")

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    with pytest.raises(RuntimeError, match="missing for an existing runtime"):
        capture_mango_window(
            config,
            datetime(2026, 7, 9, 9, tzinfo=timezone.utc),
            datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        )

    assert not config.capture_manifest.exists()


@pytest.mark.parametrize("marker_kind", ["fifo", "dangling_symlink"])
def test_capture_prior_status_special_file_fails_closed_without_hanging(
    tmp_path: Path,
    marker_kind: str,
) -> None:
    config = config_for(tmp_path)
    marker = config.process_a_status_path
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker_kind == "fifo":
        os.mkfifo(marker)
    else:
        marker.symlink_to(tmp_path / "missing-status-target.json")
    child_code = """
import sys
from pathlib import Path
from mango_mvp.customer_timeline.calls_two_processes import (
    CallsTwoProcessesConfig, capture_runtime_has_prior_state,
)
root = Path(sys.argv[1])
config = CallsTwoProcessesConfig(
    pipeline_root=root / "pipeline",
    timeline_db=root / "staging" / "customer_timeline_staging.sqlite",
    timeline_allowed_root=root / "staging",
    python_executable=Path(sys.executable),
    codex_binary=Path(sys.executable),
    codex_home_root=root / "codex_home",
    min_free_gib=1,
)
print("blocked" if capture_runtime_has_prior_state(config) else "accepted")
"""

    completed = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked"


def test_capture_prior_status_swap_after_open_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.customer_timeline import calls_two_processes as module

    config = config_for(tmp_path)
    marker = config.process_a_status_path
    replacement = tmp_path / "replacement-status.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")
    replacement.write_text(
        json.dumps({"checked_through": "2026-08-08T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    real_open = module.os.open
    swapped = False

    def swap_after_open(path: object, flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode)
        if Path(path) == marker and not swapped:
            os.replace(replacement, marker)
            swapped = True
        return descriptor

    monkeypatch.setattr(module.os, "open", swap_after_open)

    assert module.read_regular_json_marker(marker) is None
    assert swapped is True
    assert module.capture_runtime_has_prior_state(config) is True


def test_capture_recovers_torn_manifest_when_api_window_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
        acknowledge_capture_recovery,
    )

    config = replace(config_for(tmp_path), api_window_hours=1)
    store = CaptureManifestStore(config.capture_manifest)
    store.append(
        ManifestEntry(
            schema_version="capture_manifest_v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:old-call",
            provider_call_id="old-call",
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
                manager_ref=None,
                status="recording_retry_expired",
                error="recording_missing_after_retry_ttl",
            )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return []

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", EmptyClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    since = datetime(2026, 7, 9, 9, tzinfo=timezone.utc)
    until = datetime(2026, 7, 9, 10, tzinfo=timezone.utc)

    first = capture_mango_window(config, since, until)

    assert first["status"] == "partial"
    assert first["incomplete_trailing_manifest_records"] == 0
    assert first["recovered_trailing_manifest_records"] == 1
    acknowledge_capture_recovery(
        config.capture_manifest,
        expected_count=1,
        expected_incident_sha256=str(first["recovery_incident_sha256"]),
    )

    second = capture_mango_window(config, since, until)

    assert second["status"] == "ok"
    assert second["incomplete_trailing_manifest_records"] == 0
    assert second["recovered_trailing_manifest_records"] == 0


def test_capture_reports_tail_recovered_during_new_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), api_window_hours=1)
    store = CaptureManifestStore(config.capture_manifest)
    store.append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:expired",
            provider_call_id="expired",
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
            error="recording_missing_after_retry_ttl",
            remediation_code="manual_review_or_retry_if_recording_appears",
        )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    new_event = TelephonyCallEvent(
        tenant=TenantRef("foton"),
        provider="mango",
        provider_call_id="new-call",
        started_at=datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        ended_at=None,
        direction=Direction.INBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref="new-recording",
        raw_payload={},
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return [{"id": "new-call"}]

    class FakeMapper:
        def from_payload(self, **_: object) -> TelephonyCallEvent:
            return new_event

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 0}

    def fake_stage(*, manifest_store: CaptureManifestStore, **_: object) -> Summary:
        manifest_store.append(
            ManifestEntry(
                schema_version="v1",
                created_at="2026-07-09T10:00:00+00:00",
                tenant_id="foton",
                provider="mango",
                event_key="foton:mango:new-call",
                provider_call_id="new-call",
                recording_id="new-recording",
                started_at="2026-07-09T10:00:00+00:00",
                ended_at=None,
                direction="inbound",
                client_phone=None,
                manager_ref=None,
                status="downloaded",
                local_audio_path="/synthetic/new-call.mp3",
            )
        )
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficePayloadMapper", FakeMapper)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
    )

    assert report["status"] == "partial"
    assert report["incomplete_trailing_manifest_records"] == 0
    assert report["recovered_trailing_manifest_records"] == 1
    assert len(CaptureManifestStore(config.capture_manifest).read_entries()) == 2


def test_pending_recording_widens_poll_window_beyond_normal_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=12, pending_recording_retry_hours=24)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:pending-1",
            provider_call_id="pending-1",
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="skipped_no_recording",
        )
    )
    requested: list[datetime] = []

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, *, since: datetime, **_: object) -> list[dict[str, str]]:
            requested.append(since)
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0, "skipped_no_recording": 0}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 11, tzinfo=timezone.utc),
    )

    assert requested[0] == datetime(2026, 7, 9, 7, 30, tzinfo=timezone.utc)
    assert report["pending_recording_retries"] == 1
    assert report["status"] == "ok"


def test_recording_retry_ttl_uses_first_seen_and_expires_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), pending_recording_retry_hours=24)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    store = CaptureManifestStore(config.capture_manifest)
    store.append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-10T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:old-call-newly-seen",
            provider_call_id="old-call-newly-seen",
            recording_id=None,
            started_at="2025-01-01T00:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="skipped_no_recording",
        )
    )
    requested: list[tuple[datetime, datetime]] = []

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, *, since: datetime, until: datetime) -> list[dict[str, str]]:
            requested.append((since, until))
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0, "skipped_no_recording": 0}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    first = capture_mango_window(
        config,
        datetime(2026, 7, 12, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 12, 11, tzinfo=timezone.utc),
    )
    lines_after_first = len(store.read_entries())
    second = capture_mango_window(
        config,
        datetime(2026, 7, 12, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    )

    assert first["api_requests"] == 2
    assert first["status"] == "ok"
    assert first["pending_recording_expired"] == 1
    assert store.latest_by_event_key()["foton:mango:old-call-newly-seen"].status == "recording_retry_expired"
    assert second["status"] == "ok"
    assert len(store.read_entries()) == lines_after_first
    assert len(requested) == 3


def test_expired_recording_is_recovered_on_last_bounded_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=12, pending_recording_retry_hours=24)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    store = CaptureManifestStore(config.capture_manifest)
    pending = ManifestEntry(
        schema_version="v1",
        created_at="2026-07-10T10:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key="foton:mango:recovered",
        provider_call_id="recovered",
        recording_id=None,
        started_at="2025-01-01T00:00:00+00:00",
        ended_at="2025-01-01T00:20:00+00:00",
        direction="inbound",
        client_phone=None,
        manager_ref=None,
        status="skipped_no_recording",
    )
    store.append(pending)

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, *, since: datetime, **_: object) -> list[dict[str, str]]:
            if since.year < 2026:
                return [{
                    "id": "recovered",
                    "started_at": "2025-01-01T00:00:00+00:00",
                    "ended_at": "2025-01-01T00:20:00+00:00",
                    "direction": "inbound",
                    "recording_ref": "recording-late",
                }]
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 0}

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        assert [event.recording_ref for event in events] == ["recording-late"]
        store.append(replace(pending, status="downloaded", recording_id="recording-late"))
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 12, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 12, 11, tzinfo=timezone.utc),
    )

    assert report["api_requests"] == 2
    assert report["status"] == "ok"
    assert report["pending_recording_expired"] == 0
    assert store.latest_by_event_key()[pending.event_key].status == "downloaded"


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
                "recording_ids": ["recording-1"],
                "started_at": "2026-07-09T00:00:00+00:00",
                "direction": "inbound",
                "status": "downloaded",
                "local_audio_path": str(source),
                "size_bytes": source.stat().st_size,
                    "checksum_sha256": sha256_file(source),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = prepare_ingest_inputs(config)
    second = prepare_ingest_inputs(config)

    assert first["audio_files"] == second["audio_files"] == 1
    assert second["link_actions"] == {"exists_same_hash": 1}


def test_prepare_ingest_inputs_waits_for_recording_set_stabilization(tmp_path: Path) -> None:
    config = replace(
        config_for(tmp_path),
        pending_recording_retry_hours=24,
        recording_set_stabilization_minutes=120,
    )
    source = config.recordings_dir / "recent.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    started = datetime.now(timezone.utc) - timedelta(hours=1)
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at=started.isoformat(),
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:recent",
            provider_call_id="recent",
            recording_id="rec-1",
            recording_ids=("rec-1",),
            started_at=started.isoformat(),
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(source),
        )
    )

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 0
    assert result["skipped"] == {"recording_set_stabilizing": 1}


def test_prepare_ingest_inputs_skips_call_already_present_in_working_db(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    source = config.recordings_dir / "known.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-01T00:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:known", provider_call_id="known",
            recording_id="rec-1", started_at="2026-07-01T00:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(source),
        )
    )
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(source_call_id TEXT)")
        con.execute("INSERT INTO call_records VALUES ('known')")

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 0
    assert result["skipped"] == {"already_in_working": 1}
    assert result["incomplete_total"] == 0


def test_prepare_ingest_inputs_keeps_missing_known_audio_visible(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-01T00:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:known", provider_call_id="known",
            recording_id="rec-1", started_at="2026-07-01T00:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(config.recordings_dir / "missing.mp3"),
        )
    )
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(source_call_id TEXT)")
        con.execute("INSERT INTO call_records VALUES ('known')")

    result = prepare_ingest_inputs(config)

    assert result["skipped"] == {"audio_file_missing": 1}
    assert result["incomplete_total"] == 1


def test_prepare_ingest_inputs_restores_missing_working_audio_for_known_call(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    source = config.recordings_dir / "known.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-01T00:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:known", provider_call_id="known",
            recording_id="rec-1", started_at="2026-07-01T00:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(source),
        )
    )
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(source_call_id TEXT)")
        con.execute("INSERT INTO call_records VALUES ('known')")

    result = prepare_ingest_inputs(config)

    assert (config.working_audio_dir / source.name).read_bytes() == b"audio"
    assert result["incomplete_total"] == 0
    assert sum(result["link_actions"].values()) == 1


def test_existing_manifest_event_is_not_hidden_by_external_known_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    audio = config.recordings_dir / "existing.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"existing")
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:call-multi",
            provider_call_id="call-multi",
            recording_id="rec-1",
            recording_ids=("rec-1",),
            started_at="2026-07-09T10:00:00+00:00",
            ended_at="2026-07-09T10:10:00+00:00",
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(audio),
        )
    )
    captured: list[TelephonyCallEvent] = []

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, object]]:
            base = {"id": "call-multi", "started_at": "2026-07-09T10:00:00+00:00", "ended_at": "2026-07-09T10:10:00+00:00"}
            return [{**base, "records": ["rec-1", "rec-2"]}, {**base, "records": ["rec-1"]}]

    class Summary:
        failed = 0
        skipped_no_recording = 0
        needs_review_multiple_recordings = 1

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0, "skipped_no_recording": 0, "needs_review_multiple_recordings": 1}

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        captured.extend(events)
        store = CaptureManifestStore(config.capture_manifest)
        current = store.latest_by_event_key()["foton:mango:call-multi"]
        store.append(replace(current, status="multiple_recordings_needs_review", recording_ids=("rec-1", "rec-2"), recording_paths=(str(audio), str(audio))))
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.read_known_processed_ids",
        lambda *_args: (set(), {"call-multi"}),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 11, tzinfo=timezone.utc),
    )

    assert [event.recording_refs for event in captured] == [("rec-1", "rec-2")]
    assert report["api_events_already_known_external"] == 0
    assert report["status"] == "ok"
    assert report["open_multiple_recordings_needs_review"] == 1


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


def test_each_stage_timeout_is_capped_by_shared_four_hour_cycle() -> None:
    timeout = 4 * 60 * 60
    cycle_deadline = 100.0 + timeout

    assert stage_timeout_deadline(
        started_at=100.0,
        timeout_seconds=timeout,
        cycle_deadline=cycle_deadline,
    ) == (
        cycle_deadline
    )
    assert stage_timeout_deadline(
        started_at=14_000.0,
        timeout_seconds=timeout,
        cycle_deadline=cycle_deadline,
    ) == (
        cycle_deadline
    )


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
    assert config.ready_db.stat().st_mode & 0o777 == 0o600
    assert config.ready_db.parent.stat().st_mode & 0o777 == 0o700
    assert config.ready_manifest.stat().st_mode & 0o777 == 0o600
    assert not config.ready_db.with_suffix(".sqlite.tmp-shm").exists()


def test_publish_ready_db_is_private_under_open_umask(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
    previous = os.umask(0)
    try:
        publish_ready_db(config, {"total": 0})
    finally:
        os.umask(previous)

    assert config.ready_db.stat().st_mode & 0o777 == 0o600
    assert config.ready_db.parent.stat().st_mode & 0o777 == 0o700
    assert config.ready_manifest.stat().st_mode & 0o777 == 0o600


def test_strict_cli_creates_owner_only_sqlite_under_open_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "strict-runtime" / "working.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("MANGO_STRICT_ASR_RUNTIME", "1")
    get_settings.cache_clear()
    previous = os.umask(0)
    try:
        assert mango_cli.main(["init-db"]) == 0
    finally:
        os.umask(previous)
        get_settings.cache_clear()

    assert db.stat().st_mode & 0o777 == 0o600
    assert db.parent.stat().st_mode & 0o777 == 0o700


def test_unchanged_working_db_rebuilds_modified_ready_copy(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    config.ready_db.write_bytes(b"damaged")

    result = publish_ready_db_if_changed(config, {"total": 1}, changed=False)

    assert result["reused"] is False
    assert sqlite_check(config.ready_db, "quick_check") == "ok"


def test_network_outage_runs_only_local_asr_stages(tmp_path: Path) -> None:
    normal = config_for(tmp_path)

    assert pipeline_stages(normal, include_llm=False) == ("transcribe", "backfill-second-asr")


def test_codex_network_probe_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError("dns unavailable")

    monkeypatch.setattr("socket.getaddrinfo", fail)
    assert codex_network_available() is False


def test_environment_preflight_lists_failed_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MANGO_OFFICE_API_KEY", raising=False)
    monkeypatch.delenv("MANGO_OFFICE_API_SALT", raising=False)
    config = replace(
        config_for(tmp_path),
        python_executable=tmp_path / "missing-python",
        codex_binary=tmp_path / "missing-codex",
    )
    report = environment_preflight(config, run_commands=True, require_mango_credentials=True)
    assert report["ok"] is False
    assert set(report["failed_checks"]) >= {
        "mango_credentials",
        "python_executable",
        "asr_modules",
        "codex_binary",
        "codex_auth",
    }


def test_module_preflight_checks_presence_without_loading_heavy_models(tmp_path: Path) -> None:
    command = module_probe_command(config_for(tmp_path))
    assert "find_spec" in command[-1]
    assert "import mlx_whisper" not in command[-1]
    assert "import gigaam" not in command[-1]


def test_command_path_includes_codex_binary_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(config_for(tmp_path), codex_binary=tmp_path / "homebrew" / "bin" / "codex")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert command_path(config).split(os.pathsep)[0] == str(config.codex_binary.parent)


def test_pipeline_freshness_marks_old_data_stale(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    old = "2026-07-10T01:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(path, {"process": process, "status": "ok", "checked_through": old, "data_through": old})
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc))
    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "stale"
    assert report["stages"]["process_b"]["status"] == "stale"


def test_pipeline_freshness_does_not_call_missing_drop_fresh(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    write_json(
        config.process_b_status_path,
        {
            "process": "process_b",
            "status": "idle",
            "stop_reason": "drop_missing",
            "checked_at": "2026-07-10T02:00:00+00:00",
        },
    )
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert report["stages"]["process_b"]["status"] == "missing"


def test_pipeline_freshness_uses_recent_success_during_a_quiet_period(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": "ok",
            "checked_through": "2026-07-10T02:00:00+00:00",
            "data_through": "2026-07-10T01:00:00+00:00",
        },
    )
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 5, tzinfo=timezone.utc))
    assert report["stages"]["process_a"]["status"] == "fresh"
    assert report["stages"]["process_a"]["age_seconds"] == 3900.0


def test_pipeline_freshness_missing_data_is_not_fresh(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": "2026-07-10T02:00:00+00:00"},
    )
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert report["stages"]["process_a"]["status"] == "missing"


def test_pipeline_freshness_missing_manifest_overrides_fresh_stage(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    fresh = "2026-07-10T02:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(
            path,
            {"process": process, "status": "ok", "checked_through": fresh, "data_through": fresh},
        )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "missing"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_manifest_missing"
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_keeps_partial_day_red_with_fresh_timestamps(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": "partial",
            "stop_reason": "capture_manifest_tail_incomplete",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )
    write_json(
        config.process_b_status_path,
        {
            "process": "process_b",
            "status": "ok",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "partial"
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_fails_closed_on_unresolved_capture_recovery(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    store = CaptureManifestStore(config.capture_manifest)
    entry = ManifestEntry(
        schema_version="v1",
        created_at="2026-07-10T02:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key="foton:mango:before-crash",
        provider_call_id="before-crash",
        recording_id=None,
        started_at="2026-07-10T02:00:00+00:00",
        ended_at=None,
        direction="inbound",
        client_phone=None,
        manager_ref=None,
        status="recording_retry_expired",
    )
    store.append(entry)
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    store.append(replace(entry, event_key="foton:mango:after-crash", provider_call_id="after-crash"))
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )
    write_json(
        config.process_b_status_path,
        {"process": "process_b", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "partial"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_manifest_tail_incomplete"
    assert report["stages"]["process_a"]["capture_recovery_unresolved_count"] == 1
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_fails_closed_on_torn_tail_before_recovery(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
        capture_recovery_path,
    )

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-10T02:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:before-crash",
            provider_call_id="before-crash",
            recording_id=None,
            started_at="2026-07-10T02:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
        )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    assert not capture_recovery_path(config.capture_manifest).exists()
    fresh = "2026-07-10T02:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(
            path,
            {"process": process, "status": "ok", "checked_through": fresh, "data_through": fresh},
        )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "partial"
    assert report["stages"]["process_a"]["capture_manifest_tail_status"] == "incomplete"
    assert report["stages"]["process_a"]["capture_recovery_status"] == "resolved"
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_fails_closed_on_invalid_capture_recovery(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import capture_recovery_path

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    recovery_path = capture_recovery_path(config.capture_manifest)
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text("{invalid", encoding="utf-8")
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "failed"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_recovery_ledger_invalid"


@pytest.mark.parametrize("manifest_kind", ["directory", "fifo"])
def test_pipeline_freshness_fails_closed_on_non_regular_manifest(
    tmp_path: Path,
    manifest_kind: str,
) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    config.capture_manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest_kind == "directory":
        config.capture_manifest.mkdir()
    else:
        os.mkfifo(config.capture_manifest)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "failed"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_manifest_tail_invalid"


def test_ok_stage_is_not_published_before_durable_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import atomic_write_private_json as real_atomic_write

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_b_status_path,
        {"process": "process_b", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    def crash_on_report(path: Path, payload: object, **kwargs: object) -> None:
        if path.parent == config.reports_dir:
            raise SystemExit(77)
        real_atomic_write(path, payload, **kwargs)

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.atomic_write_private_json",
        crash_on_report,
    )

    with pytest.raises(SystemExit, match="77"):
        finalize_report(config, "synthetic-ok", "process_a", "ok", "", {})

    assert not config.process_a_status_path.exists()
    assert not list(config.reports_dir.glob("*_process_a.json"))
    freshness = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert freshness["stages"]["process_a"]["status"] == "missing"
    assert freshness["status"] == "stale"


@pytest.mark.parametrize("red_status", ["failed", "partial"])
def test_red_stage_is_published_before_broken_report_directory(
    tmp_path: Path,
    red_status: str,
) -> None:
    config = config_for(tmp_path)
    config.reports_dir.parent.mkdir(parents=True, exist_ok=True)
    config.reports_dir.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(FileExistsError):
        finalize_report(config, "synthetic-red", "process_a", red_status, "synthetic", {})

    assert read_json(config.process_a_status_path)["status"] == red_status


def test_broken_reports_directory_cannot_leave_process_a_fresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1, freshness_max_age_minutes=30)
    config.reports_dir.parent.mkdir(parents=True, exist_ok=True)
    config.reports_dir.write_text("not-a-directory", encoding="utf-8")
    fresh = "2026-07-10T02:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(
            path,
            {"process": process, "status": "ok", "checked_through": fresh, "data_through": fresh},
        )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    with pytest.raises(FileExistsError):
        run_process_a(config, skip_capture=True, skip_workers=True)

    assert read_json(config.process_a_status_path)["status"] == "failed"
    freshness = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert freshness["stages"]["process_a"]["status"] == "failed"
    assert freshness["status"] == "stale"


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
                transcription_status TEXT,
                resolve_status TEXT,
                analysis_status TEXT,
                analysis_json TEXT,
                dead_letter_stage TEXT,
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
    write_json(
        path.with_suffix(".manifest.json"),
        {
            "schema_version": "mango_calls_two_processes_v1",
            "status": "ready",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "quick_check": "ok",
        },
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


def test_process_b_has_its_own_nonblocking_process_lock(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)

    with process_lease(config.process_b_lock, stale_seconds=60):
        result = run_process_b(config)

    assert result["status"] == "locked"
    assert result["stop_reason"] == "process_b_locked"


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

    assert first["status"] == "ok"
    assert second["status"] == "idle"
    assert second["stop_reason"] == "drop_unchanged"
    assert first_count == second_count == 1
    assert call_event_source_systems(config.timeline_db) == ["mango_processed_summary"]


def test_process_b_fails_loud_when_import_validation_fails(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    def fake_producer(_: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None) -> dict[str, object]:
        del since
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return {"status": "ok", "events_written": 0}

    def invalid_import(_: object) -> dict[str, object]:
        return {
            "validation_ok": False,
            "summary": {"records_read": 1, "records_accepted": 1, "writes_applied": 1},
            "writes": {"status_counts": {"updated": 1}},
            "source_system": "mango_processed_summary",
        }

    report = run_process_b(config, producer_runner=fake_producer, import_runner=invalid_import)
    assert report["status"] == "failed"
    assert report["stop_reason"] == "import_validation_failed"


def test_process_b_rejects_producer_event_count_mismatch(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    def incomplete_producer(
        _: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None
    ) -> dict[str, object]:
        del since
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return {"status": "ok", "rows_selected": 1, "events_written": 0}

    result = run_process_b(config, producer_runner=incomplete_producer)

    assert result["status"] == "failed"
    assert result["stop_reason"] == "producer_event_count_mismatch"
    assert read_json(config.process_b_cursor_path) == {}


def test_process_b_normalizes_unexpected_exception_and_keeps_cursor(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    write_json(config.process_b_cursor_path, {"sha256": "previous"})

    def broken_producer(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("unexpected local producer failure")

    report = run_process_b(config, producer_runner=broken_producer)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "process_b_exception:ValueError"
    assert report["counters"]["diagnostic"]["type"] == "ValueError"
    assert read_json(config.process_b_cursor_path) == {"sha256": "previous"}


def test_process_b_invalid_config_returns_normalized_failure(tmp_path: Path) -> None:
    config = config_for(tmp_path, timeline_name="customer_timeline_prod_20260713.sqlite")

    report = run_process_b(config)

    assert report["status"] == "failed"
    assert report["stop_reason"].startswith("process_b_config_exception:")
    assert report["safety"]["writes_timeline_staging"] is False


def test_process_b_returns_in_memory_failure_when_report_path_is_broken(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    config.reports_dir.parent.mkdir(parents=True, exist_ok=True)
    config.reports_dir.write_text("not a directory", encoding="utf-8")

    def broken_producer(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("producer failed")

    report = run_process_b(config, producer_runner=broken_producer)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "process_b_finalize_exception:FileExistsError"
    assert report["counters"]["original_stop_reason"] == "process_b_exception:ValueError"
    assert report["safety"]["writes_timeline_staging"] is False


def test_ingest_failure_counts_are_visible_in_compact_worker_report(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'processed': 2, 'inserted': 1, 'failed': 1, 'failure_types': {'ValueError': 1}}))",
        "ingest",
    ]

    raw = run_command(command, os.environ, tmp_path)
    compact = compact_command_reports([raw])

    assert compact[0]["command"] == "ingest"
    assert compact[0]["metrics"]["failed"] == 1
    assert compact[0]["metrics"]["failure_types"] == {"ValueError": 1}


def test_process_b_does_not_skip_late_old_call_by_timestamp_cursor(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root)
    store.close()
    first = run_process_b(config)
    assert first["status"] == "ok"
    old_sha = read_json(config.process_b_cursor_path)["sha256"]
    with sqlite3.connect(config.ready_db) as con:
        con.execute("UPDATE call_records SET duration_sec=61 WHERE id=1")
    # Re-seal the drop the way publish_ready_db does: the manifest describes the
    # republished sqlite, while the process B cursor still holds the old sha.
    write_json(
        config.ready_db.with_suffix(".manifest.json"),
        {
            "status": "ready",
            "quick_check": "ok",
            "sha256": sha256_file(config.ready_db),
            "size_bytes": config.ready_db.stat().st_size,
        },
    )
    assert read_json(config.process_b_cursor_path)["sha256"] == old_sha
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


def test_run_id_uuid_prefix_cannot_be_mistaken_for_phone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_uuid = type("FixedUuid", (), {"hex": "81234567890abcdef01234567890abcd"})()
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.uuid.uuid4",
        lambda: fixed_uuid,
    )

    run_id = new_calls_run_id(datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc))

    assert run_id.endswith("-u81234567890a")
    assert_no_pdn({"run_id": run_id})


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
    config_path.chmod(0o600)
    env_path.write_text("MANGO_OFFICE_API_KEY=x\nMANGO_OFFICE_API_SALT=y\n", encoding="utf-8")
    env_path.chmod(0o600)

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


def test_process_b_fails_loud_on_stale_drop_manifest(tmp_path: Path) -> None:
    """A drop manifest that no longer matches the sealed sqlite must stop the
    import instead of passing as success: `ready_drop_fingerprint` already
    computes `manifest_mismatch`, and process B must honour it."""
    config = replace(config_for(tmp_path), manifest_recheck_sleep_sec=0.0)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    write_json(
        config.ready_db.with_suffix(".manifest.json"),
        {
            "status": "ready",
            "sha256": "0" * 64,
            "size_bytes": config.ready_db.stat().st_size + 1,
            "quick_check": "ok",
        },
    )

    report = run_process_b(config)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "drop_manifest_mismatch"
    assert report["counters"]["drop"]["manifest_mismatch"] is True
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        written = int(
            con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type='mango_call'").fetchone()[0]
        )
    assert written == 0
    assert read_json(config.process_b_cursor_path) == {}


@pytest.mark.parametrize("manifest", [None, {"status": "ready", "quick_check": "ok"}])
def test_process_b_rejects_missing_or_incomplete_manifest(
    tmp_path: Path, manifest: dict[str, str] | None
) -> None:
    config = replace(config_for(tmp_path), manifest_recheck_sleep_sec=0.0)
    create_ready_call_db(config.ready_db)
    manifest_path = config.ready_db.with_suffix(".manifest.json")
    if manifest is None:
        manifest_path.unlink()
    else:
        write_json(manifest_path, manifest)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    report = run_process_b(config)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "drop_manifest_invalid"
    assert report["counters"]["drop"]["manifest_valid"] is False
    assert read_json(config.process_b_cursor_path) == {}


def test_prepare_ingest_inputs_counts_missing_capture_audio(tmp_path: Path) -> None:
    """A manifest row marked `downloaded` whose audio no longer exists must be
    counted in the report, not silently dropped."""
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = config_for(tmp_path)
    config.recordings_dir.mkdir(parents=True, exist_ok=True)
    present = config.recordings_dir / "present.mp3"
    present.write_bytes(b"audio-bytes")
    empty = config.recordings_dir / "empty.mp3"
    empty.write_bytes(b"")

    store = CaptureManifestStore(config.capture_manifest)

    def entry(event_key: str, audio_path: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=event_key,
            provider_call_id=event_key,
            recording_id=event_key,
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=audio_path,
        )

    store.append(entry("ok-1", str(present)))
    store.append(entry("gone-1", str(config.recordings_dir / "vanished.mp3")))
    store.append(entry("empty-1", str(empty)))

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 1
    assert result["skipped"] == {"audio_file_missing": 1, "audio_file_empty": 1}
    assert result["skipped_total"] == 2


def test_prepare_ingest_inputs_keeps_torn_manifest_tail_incomplete(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = config_for(tmp_path)
    audio = config.recordings_dir / "present.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio-bytes")
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="present-1",
            provider_call_id="present-1",
            recording_id="present-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(audio),
        )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 1
    assert result["incomplete_trailing_manifest_records"] == 1
    assert result["incomplete_total"] == 1


def test_process_a_processes_available_audio_then_marks_missing_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), min_free_gib=1)
    store = CaptureManifestStore(config.capture_manifest)

    def entry(event_key: str, audio_path: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=event_key,
            provider_call_id=event_key,
            recording_id=event_key,
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=audio_path,
        )
    present = config.recordings_dir / "present.mp3"
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_bytes(b"audio")
    store.append(entry("present-1", str(present)))
    store.append(entry("gone-1", str(config.recordings_dir / "missing.mp3")))
    create_ready_call_db(config.working_db)
    commands: list[str] = []
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    def fake_command(command: list[str], _env: dict[str, str], _cwd: Path) -> dict[str, object]:
        commands.append(" ".join(command))
        return {"rc": 0, "command": command[-1]}

    report = run_process_a(
        config,
        skip_capture=True,
        skip_workers=False,
        command_runner=fake_command,
    )

    assert report["status"] == "partial"
    assert report["stop_reason"] == "capture_asset_integrity_failed"
    assert report["counters"]["metadata"]["skipped_total"] == 1
    assert report["counters"]["drop"]["status"] == "blocked"
    assert not config.ready_db.exists()
    assert any(" ingest " in f" {command} " for command in commands)
    assert read_json(config.cursor_path) == {}


def test_process_a_partial_capture_publishes_available_work_and_advances_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.prepare_ingest_inputs",
        lambda _config: {"audio_files": 0, "skipped_total": 0},
    )

    report = run_process_a(
        config,
        since="2026-07-09T09:00:00+00:00",
        until="2026-07-09T10:00:00+00:00",
        skip_workers=True,
        capture_runner=lambda *_args: {"status": "partial", "downloaded": 1, "failed": 1},
    )

    assert report["status"] == "partial"
    assert report["counters"]["drop"]["status"] == "ready"
    assert report["downstream_ready"] is True
    assert config.ready_db.exists()
    assert read_json(config.cursor_path)["until"] == "2026-07-09T10:00:00+00:00"


def test_process_a_reports_recovered_manifest_tail_and_blocks_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    store = CaptureManifestStore(config.capture_manifest)

    def terminal_entry(call_id: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=f"foton:mango:{call_id}",
            provider_call_id=call_id,
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
            error="recording_missing_after_retry_ttl",
            remediation_code="manual_review_or_retry_if_recording_appears",
        )

    store.append(terminal_entry("before-crash"))
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    def recover_during_capture(
        _config: CallsTwoProcessesConfig,
        _since: datetime,
        _until: datetime,
    ) -> dict[str, object]:
        CaptureManifestStore(config.capture_manifest).append(terminal_entry("after-restart"))
        return {"status": "ok"}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    report = run_process_a(
        config,
        since="2026-07-09T09:00:00+00:00",
        until="2026-07-09T10:00:00+00:00",
        skip_workers=True,
        capture_runner=recover_during_capture,
    )

    assert report["status"] == "partial"
    assert report["stop_reason"] == "capture_manifest_tail_incomplete"
    assert report["downstream_ready"] is False
    assert report["counters"]["drop"] == {
        "status": "blocked",
        "reason": "capture_manifest_tail_incomplete",
    }
    assert report["counters"]["metadata"]["recovered_trailing_manifest_records"] == 1
    assert not config.ready_db.exists()
    assert read_json(config.cursor_path) == {}
    assert CaptureManifestStore(config.capture_manifest).recovered_trailing_records == 0
    partial_report_path = Path(str(report["report_path"]))
    partial_report_bytes = partial_report_path.read_bytes()

    clean_retry = run_process_a(config, skip_capture=True, skip_workers=True)

    assert clean_retry["status"] == "ok"
    assert clean_retry["downstream_ready"] is True
    assert len(CaptureManifestStore(config.capture_manifest).read_entries()) == 2
    assert Path(str(clean_retry["report_path"])) != partial_report_path
    assert partial_report_path.read_bytes() == partial_report_bytes
    assert read_json(partial_report_path)["stop_reason"] == "capture_manifest_tail_incomplete"


def test_process_a_report_failure_leaves_recovery_unacknowledged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)

    def terminal_entry(call_id: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=f"foton:mango:{call_id}",
            provider_call_id=call_id,
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
        )

    store = CaptureManifestStore(config.capture_manifest)
    store.append(terminal_entry("before-crash"))
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    def recover_during_capture(*_args: object) -> dict[str, object]:
        CaptureManifestStore(config.capture_manifest).append(terminal_entry("after-restart"))
        return {"status": "ok"}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    from mango_mvp.productization.capture_staging import atomic_write_private_json as real_atomic_write

    def fail_report_write(path: Path, payload: object, **kwargs: object) -> None:
        if path.parent == config.reports_dir:
            raise OSError("synthetic report failure")
        real_atomic_write(path, payload, **kwargs)

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.atomic_write_private_json",
        fail_report_write,
    )

    with pytest.raises(OSError, match="synthetic report failure"):
        run_process_a(
            config,
            since="2026-07-09T09:00:00+00:00",
            until="2026-07-09T10:00:00+00:00",
            skip_workers=True,
            capture_runner=recover_during_capture,
        )

    restarted = CaptureManifestStore(config.capture_manifest)
    assert restarted.recovered_trailing_records == 1
    assert restarted.recovery_incident_sha256
    assert read_json(config.process_a_status_path)["status"] == "failed"
    assert pipeline_freshness(config)["stages"]["process_a"]["status"] == "failed"
    assert read_json(config.cursor_path) == {}
    assert not config.ready_db.exists()


@pytest.mark.parametrize("raise_after_ack", [False, True])
def test_process_a_ack_failure_preserves_partial_incident_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raise_after_ack: bool,
) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
        acknowledge_capture_recovery as real_acknowledge,
    )

    config = replace(config_for(tmp_path), min_free_gib=1, foton_daily_dir=tmp_path / "daily")
    create_ready_call_db(config.working_db)

    def terminal_entry(call_id: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=f"foton:mango:{call_id}",
            provider_call_id=call_id,
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
        )

    store = CaptureManifestStore(config.capture_manifest)
    store.append(terminal_entry("before-crash"))
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    def recover_during_capture(*_args: object) -> dict[str, object]:
        CaptureManifestStore(config.capture_manifest).append(terminal_entry("after-restart"))
        return {"status": "ok"}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    def fail_acknowledge(*args: object, **kwargs: object) -> int:
        if raise_after_ack:
            real_acknowledge(*args, **kwargs)
        raise RuntimeError("synthetic acknowledgement failure")

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.acknowledge_capture_recovery",
        fail_acknowledge,
    )

    failed = run_process_a(
        config,
        since="2026-07-09T09:00:00+00:00",
        until="2026-07-09T10:00:00+00:00",
        skip_workers=True,
        capture_runner=recover_during_capture,
    )

    reports = [read_json(path) for path in sorted(config.reports_dir.glob("*_process_a.json"))]
    partial = [report for report in reports if report.get("status") == "partial"]
    failures = [report for report in reports if report.get("status") == "failed"]
    restarted = CaptureManifestStore(config.capture_manifest)

    assert failed["status"] == "failed"
    assert len(reports) == 2
    assert len(partial) == 1
    assert len(failures) == 1
    assert failed["run_id"] != partial[0]["run_id"]
    assert partial[0]["stop_reason"] == "capture_manifest_tail_incomplete"
    assert failures[0]["counters"]["diagnostic"]["preserved_report_run_id"] == partial[0]["run_id"]
    daily_failure = read_json(Path(str(failed["daily_report_path"])))
    assert daily_failure["counters"]["preserved_report"]["run_id"] == partial[0]["run_id"]
    assert "diagnostic" not in daily_failure["counters"]
    assert restarted.recovered_trailing_records == (0 if raise_after_ack else 1)
    assert read_json(config.cursor_path) == {}
    assert not config.ready_db.exists()


def test_process_a_runs_workers_for_existing_open_db_work_without_reingest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute("UPDATE call_records SET transcription_status='pending', analysis_status='pending'")
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.prepare_ingest_inputs",
        lambda _config: {"audio_files": 0, "skipped_total": 0},
    )
    commands: list[str] = []

    def command_runner(command: list[str], *_args: object) -> dict[str, object]:
        commands.append(" ".join(command))
        return {"rc": 0, "command": command[-1]}

    report = run_process_a(config, skip_capture=True, command_runner=command_runner)

    assert report["status"] == "ok"
    assert report["counters"]["metadata"]["db_open_work"] is True
    assert any("worker" in command for command in commands)
    assert not any(" init-db" in command or " ingest" in command for command in commands)


def test_process_a_complete_existing_db_runs_no_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    source = config.recordings_dir / "known.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-01T00:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:provider-1", provider_call_id="provider-1",
            recording_id="rec-1", started_at="2026-07-01T00:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(source),
        )
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    first = run_process_a(
        config,
        skip_capture=True,
        command_runner=lambda *_args: pytest.fail("complete second run must stay empty"),
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.publish_ready_db",
        lambda *_args, **_kwargs: pytest.fail("unchanged second run must reuse ready DB"),
    )
    report = run_process_a(
        config,
        skip_capture=True,
        command_runner=lambda *_args: pytest.fail("complete second run must stay empty"),
    )

    assert first["status"] == "ok"
    assert report["status"] == "ok"
    assert report["counters"]["metadata"]["db_open_work"] is False
    assert report["counters"]["drop"]["reused"] is True


def test_call_db_open_work_includes_interrupted_secondary_backfill(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute("ALTER TABLE call_records ADD COLUMN pipeline_stage TEXT")
        con.execute(
            "UPDATE call_records SET transcription_status='done', resolve_status='done', "
            "analysis_status='done', pipeline_stage='backfill-second-asr'"
        )

    assert call_db_has_open_work(config.working_db) is True


def test_call_db_open_work_excludes_future_retry_and_exhausted_attempts(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute("ALTER TABLE call_records ADD COLUMN transcribe_attempts INTEGER")
        con.execute("ALTER TABLE call_records ADD COLUMN next_retry_at TEXT")
        con.execute(
            "UPDATE call_records SET transcription_status='failed', transcribe_attempts=3, "
            "next_retry_at='2099-01-01T00:00:00+00:00'"
        )

    assert call_db_has_open_work(config.working_db) is False


def test_call_db_open_work_includes_stale_analyze_claim(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute("ALTER TABLE call_records ADD COLUMN analysis_claimed_at TEXT")
        con.execute(
            "UPDATE call_records SET transcription_status='done', resolve_status='done', "
            "analysis_status='in_progress', analysis_claimed_at='2020-01-01T00:00:00+00:00'"
        )

    assert call_db_has_open_work(config.working_db) is True


def test_call_db_open_work_includes_secondary_asr_retry(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    payload = {
        "mode": "mono_or_fallback",
        "secondary_provider": "gigaam",
        "full": {"variant_a": "текст", "variant_b": ""},
    }
    with sqlite3.connect(config.working_db) as con:
        con.execute("ALTER TABLE call_records ADD COLUMN transcript_variants_json TEXT")
        con.execute(
            "UPDATE call_records SET transcription_status='done', resolve_status='done', "
            "analysis_status='done', transcript_variants_json=?",
            (json.dumps(payload, ensure_ascii=False),),
        )

    assert call_db_has_open_work(config.working_db) is True


def test_missing_downloaded_capture_is_returned_for_recovery(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:gone-1",
            provider_call_id="gone-1",
            recording_id="recording-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(config.recordings_dir / "missing.mp3"),
        )
    )

    recovered = missing_capture_recovery_events(config)

    assert len(recovered) == 1
    assert recovered[0].provider_call_id == "gone-1"
    assert recovered[0].recording_ref == "recording-1"


def test_failed_capture_stays_in_recovery_queue(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:failed-1",
            provider_call_id="failed-1",
            recording_id="recording-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="failed",
            local_audio_path=str(config.recordings_dir / "failed.mp3"),
        )
    )

    recovered = missing_capture_recovery_events(config)

    assert [event.provider_call_id for event in recovered] == ["failed-1"]


def test_expired_capture_with_known_recording_is_rechecked_at_most_daily(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    attempted_at = datetime.now(timezone.utc) - timedelta(hours=25)
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at=attempted_at.isoformat(),
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:expired-known",
            provider_call_id="expired-known",
            recording_id="recording-known",
            recording_ids=("recording-known",),
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
            error="recording_missing_after_retry_ttl",
        )
    )

    assert missing_capture_recovery_events(
        config, now=attempted_at + timedelta(hours=23)
    ) == ()
    due = missing_capture_recovery_events(
        config, now=attempted_at + timedelta(hours=24)
    )
    assert [event.provider_call_id for event in due] == ["expired-known"]


def test_recovery_is_not_filtered_by_known_processed_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:failed-1",
            provider_call_id="failed-1",
            recording_id="recording-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="failed",
            local_audio_path=str(config.recordings_dir / "failed.mp3"),
        )
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 0}

    captured: list[TelephonyCallEvent] = []

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        captured.extend(events)
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.read_known_processed_ids",
        lambda _root: ({"recording-1"}, {"failed-1"}),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    capture_mango_window(
        config,
        datetime(2026, 7, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 1, tzinfo=timezone.utc),
    )

    assert [event.provider_call_id for event in captured] == ["failed-1"]


def test_process_b_registers_call_audio_artifact(tmp_path: Path) -> None:
    """The recording path is the only pointer a manager has back to the call;
    it must land in `event_artifacts`, not only inside record_json."""
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    report = run_process_b(config)
    assert report["status"] == "ok"

    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT artifact_type, path FROM event_artifacts WHERE source_system='mango_processed_summary'"
        ).fetchall()
    assert [row[0] for row in rows] == ["call_audio"]
    assert rows[0][1] == "/ignored/masked.mp3"


def test_call_audio_artifact_path_never_reaches_a_projection(tmp_path: Path) -> None:
    """Capture filenames embed the client phone, so the artifact path is PDn:
    the read projection must expose only `has_path`, never the path itself."""
    from mango_mvp.customer_timeline.read_api import project_artifact

    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    assert run_process_b(config)["status"] == "ok"

    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        stored = [dict(row) for row in con.execute("SELECT * FROM event_artifacts")]
    assert len(stored) == 1
    assert stored[0]["path"]

    projected = project_artifact(stored[0])
    assert "path" not in projected
    assert projected["has_path"] is True
    assert stored[0]["path"] not in json.dumps(projected, ensure_ascii=False)


def test_mlx_cache_is_released_once_after_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mango_mvp.services import transcribe

    releases: list[str] = []
    monkeypatch.setattr(
        transcribe,
        "release_mlx_free_cache",
        lambda: releases.append("released") or True,
    )

    assert transcribe.run_with_mlx_cache_release(lambda: "ok", mlx_executed=True) == "ok"
    assert releases == ["released"]

    def fail() -> str:
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        transcribe.run_with_mlx_cache_release(fail, mlx_executed=True)
    assert releases == ["released", "released"]


def test_transcribe_does_not_apply_a_global_mlx_cache_limit() -> None:
    import inspect

    from mango_mvp.services import transcribe

    assert "set_cache_limit" not in inspect.getsource(transcribe)
