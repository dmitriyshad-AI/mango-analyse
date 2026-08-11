from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.productization.capture_staging import (
    AudioValidation,
    CaptureManifestStore,
    ManifestEntry,
    acknowledge_capture_recovery,
    atomic_write_private_json,
    audit_capture_manifest,
    build_capture_audio_filename,
    capture_recovery_incident_sha256,
    capture_recovery_path,
    recording_part_paths,
    stage_capture_events,
)
from mango_mvp.productization.contracts import Direction, TelephonyCallEvent, TenantRef


class FakeDownloader:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls = []
        self.fail_first = fail_first

    def download(self, recording_id: str, target_path: Path) -> int:
        self.calls.append((recording_id, target_path))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("temporary link failure")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = f"fake-audio:{recording_id}".encode("utf-8")
        target_path.write_bytes(data)
        return len(data)

def fake_validator(path: Path) -> AudioValidation:
    size = path.stat().st_size
    return AudioValidation(
        size_bytes=size,
        checksum_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        duration_sec=12.5,
        codec_name="mp3",
        channels=2,
        sample_rate=8000,
    )


def event(
    call_id: str,
    recording_ref: str | None = "rec-1",
    phone: str = "+79990000000",
    started_offset_sec: int = 0,
    recording_refs: tuple[str, ...] = (),
) -> TelephonyCallEvent:
    started = datetime(2026, 5, 7, 9, 0, tzinfo=timezone.utc) + timedelta(seconds=started_offset_sec)
    return TelephonyCallEvent(
        tenant=TenantRef("foton"),
        provider="mango",
        provider_call_id=call_id,
        started_at=started,
        ended_at=started + timedelta(seconds=60),
        direction=Direction.INBOUND,
        client_phone=phone,
        manager_ref="101",
        recording_ref=recording_ref,
        recording_refs=recording_refs,
        raw_payload={},
    )


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_entry(call_id: str) -> ManifestEntry:
    return ManifestEntry(
        schema_version="capture_manifest_v1",
        created_at="2026-07-09T10:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key=f"foton:mango:{call_id}",
        provider_call_id=call_id,
        recording_id=f"recording-{call_id}",
        recording_ids=(f"recording-{call_id}",),
        started_at="2026-07-09T10:00:00+00:00",
        ended_at=None,
        direction="inbound",
        client_phone=None,
        manager_ref='менеджер "тест" \\ line\n tab\t control\x02 😀',
        status="downloaded",
        local_audio_path=f"/synthetic/{call_id}.mp3",
        size_bytes=12345,
        duration_sec=12.34,
        dry_run=False,
    )


def test_manifest_recovers_only_unterminated_final_record_before_append(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    first_event = event("CALL-1", "rec-1")
    stage_capture_events([first_event], store, tmp_path / "recordings", FakeDownloader(), validator=fake_validator)
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    assert len(store.read_entries()) == 1
    assert store.incomplete_trailing_records == 1

    summary = stage_capture_events(
        [event("CALL-2", "rec-2", started_offset_sec=1)],
        store,
        tmp_path / "recordings",
        FakeDownloader(),
        validator=fake_validator,
    )

    assert [row["provider_call_id"] for row in read_manifest(manifest)] == ["CALL-1", "CALL-2"]
    assert manifest.read_bytes().endswith(b"\n")
    assert store.incomplete_trailing_records == 0
    assert store.recovered_trailing_records == 1
    assert summary.recovered_trailing_manifest_records == 1
    assert store.recovery_path.stat().st_mode & 0o777 == 0o600
    recovery = json.loads(store.recovery_path.read_text(encoding="utf-8"))
    assert recovery["status"] == "unresolved"
    assert recovery["unresolved_count"] == 1
    assert recovery["incident_sha256"] == store.recovery_incident_sha256
    assert "unfinished" not in store.recovery_path.read_text(encoding="utf-8")
    assert CaptureManifestStore(manifest).recovered_trailing_records == 1

    assert acknowledge_capture_recovery(
        manifest,
        expected_count=1,
        expected_incident_sha256=str(store.recovery_incident_sha256),
    ) == 1
    assert CaptureManifestStore(manifest).recovered_trailing_records == 0
    resolved = json.loads(store.recovery_path.read_text(encoding="utf-8"))
    assert resolved["acknowledged_incident_sha256"] == store.recovery_incident_sha256


def test_manifest_recovers_unterminated_final_record_with_partial_utf8(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    first_event = event("CALL-1", "rec-1")
    stage_capture_events([first_event], store, tmp_path / "recordings", FakeDownloader(), validator=fake_validator)
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"\xd0')

    assert len(store.read_entries()) == 1
    assert store.incomplete_trailing_records == 1

    stage_capture_events(
        [event("CALL-2", "rec-2", started_offset_sec=1)],
        store,
        tmp_path / "recordings",
        FakeDownloader(),
        validator=fake_validator,
    )

    assert [row["provider_call_id"] for row in read_manifest(manifest)] == ["CALL-1", "CALL-2"]


def test_manifest_reader_refreshes_recovery_written_after_store_creation(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    writer = CaptureManifestStore(manifest)
    writer.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    stale_reader = CaptureManifestStore(manifest)

    CaptureManifestStore(manifest).append(manifest_entry("CALL-2"))

    assert len(stale_reader.read_entries()) == 2
    assert stale_reader.recovered_trailing_records == 1


def test_manifest_reader_accepts_peer_recovery_only_with_matching_ledger(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    writer = CaptureManifestStore(manifest)
    writer.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    stale_reader = CaptureManifestStore(manifest)
    assert [entry.provider_call_id for entry in stale_reader.read_entries()] == ["CALL-1"]
    assert stale_reader.incomplete_trailing_records == 1

    CaptureManifestStore(manifest).append(manifest_entry("CALL-2"))

    assert [entry.provider_call_id for entry in stale_reader.read_entries()] == [
        "CALL-1",
        "CALL-2",
    ]
    assert stale_reader.recovered_trailing_records == 1


def test_manifest_recovers_incomplete_tail_without_appending_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    restarted = CaptureManifestStore(manifest)
    assert restarted.recover_incomplete_tail() == 1
    assert [entry.provider_call_id for entry in restarted.read_entries()] == ["CALL-1"]
    assert restarted.incomplete_trailing_records == 0
    assert restarted.recovered_trailing_records == 1
    assert capture_recovery_path(manifest).exists()
    before_bytes = manifest.read_bytes()
    before_mtime = manifest.stat().st_mtime_ns

    assert CaptureManifestStore(manifest).recover_incomplete_tail() == 0
    assert manifest.read_bytes() == before_bytes
    assert manifest.stat().st_mtime_ns == before_mtime


def test_manifest_recovery_resumes_after_crash_with_durable_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization import capture_staging as module

    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    before = manifest.read_bytes()
    real_record = module.record_capture_recovery

    def crash_after_ledger(*args: object, **kwargs: object) -> tuple[int, str]:
        result = real_record(*args, **kwargs)
        raise SystemExit(77)

    monkeypatch.setattr(module, "record_capture_recovery", crash_after_ledger)
    with pytest.raises(SystemExit, match="77"):
        CaptureManifestStore(manifest).recover_incomplete_tail()

    assert manifest.read_bytes() == before
    ledger = json.loads(capture_recovery_path(manifest).read_text(encoding="utf-8"))
    assert ledger["status"] == "unresolved"
    assert ledger["unresolved_count"] == 1

    monkeypatch.setattr(module, "record_capture_recovery", real_record)
    restarted = CaptureManifestStore(manifest)
    assert restarted.recover_incomplete_tail() == 1
    assert [entry.provider_call_id for entry in restarted.read_entries()] == ["CALL-1"]
    assert restarted.recovered_trailing_records == 1


def test_manifest_reader_rejects_tail_replacement_without_matching_ledger(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    writer = CaptureManifestStore(manifest)
    writer.append(manifest_entry("CALL-1"))
    valid_prefix = manifest.read_bytes()
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    stale_reader = CaptureManifestStore(manifest)
    assert len(stale_reader.read_entries()) == 1
    replacement = (
        json.dumps(manifest_entry("CALL-2").to_json_dict(), ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    manifest.write_bytes(valid_prefix + replacement)

    with pytest.raises(RuntimeError, match="without recovery record"):
        stale_reader.read_entries()


def test_manifest_recovery_ack_rejects_newer_unreported_tail(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"')
    store.append(manifest_entry("CALL-2"))
    assert store.recovered_trailing_records == 1
    first_incident_sha256 = str(store.recovery_incident_sha256)

    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"')
    CaptureManifestStore(manifest).append(manifest_entry("CALL-3"))

    with pytest.raises(RuntimeError, match="changed before acknowledgement"):
        acknowledge_capture_recovery(
            manifest,
            expected_count=1,
            expected_incident_sha256=first_incident_sha256,
        )
    assert CaptureManifestStore(manifest).recovered_trailing_records == 2


def test_manifest_recovery_ack_rejects_same_count_aba_incident(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"incident-a"')
    store.append(manifest_entry("CALL-2"))
    first_sha256 = str(store.recovery_incident_sha256)
    acknowledge_capture_recovery(
        manifest,
        expected_count=1,
        expected_incident_sha256=first_sha256,
    )

    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"incident-b"')
    second_store = CaptureManifestStore(manifest)
    second_store.append(manifest_entry("CALL-3"))
    assert second_store.recovered_trailing_records == 1
    assert second_store.recovery_incident_sha256 != first_sha256

    with pytest.raises(RuntimeError, match="changed before acknowledgement"):
        acknowledge_capture_recovery(
            manifest,
            expected_count=1,
            expected_incident_sha256=first_sha256,
        )
    assert CaptureManifestStore(manifest).recovered_trailing_records == 1


def test_manifest_recovery_ledger_survives_crash_after_durable_append(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    child_code = """
import os
import sys
from mango_mvp.productization import capture_staging as module

real_fsync = os.fsync
calls = 0
def crash_after_manifest_fsync(descriptor):
    global calls
    calls += 1
    real_fsync(descriptor)
    if calls == 3:
        os._exit(77)
module.os.fsync = crash_after_manifest_fsync
entry = module.ManifestEntry(
    schema_version="v1", created_at="2026-07-09T10:00:00+00:00",
    tenant_id="foton", provider="mango", event_key="foton:mango:CALL-2",
    provider_call_id="CALL-2", recording_id=None,
    started_at="2026-07-09T10:00:00+00:00", ended_at=None,
    direction="inbound", client_phone=None, manager_ref=None,
    status="recording_retry_expired",
)
module.CaptureManifestStore(module.Path(sys.argv[1])).append(entry)
"""

    completed = subprocess.run(
        [sys.executable, "-c", child_code, str(manifest)],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
    )

    assert completed.returncode == 77
    restarted = CaptureManifestStore(manifest)
    assert len(restarted.read_entries()) == 2
    assert restarted.recovered_trailing_records == 1


def test_capture_recovery_ledger_rejects_duplicate_fingerprints(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    store.append(manifest_entry("CALL-2"))
    payload = json.loads(store.recovery_path.read_text(encoding="utf-8"))
    payload["tails"] = [payload["tails"][0], payload["tails"][0]]
    payload["unresolved_count"] = 2
    payload["incident_sha256"] = capture_recovery_incident_sha256(payload["tails"])
    store.recovery_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ledger is invalid"):
        CaptureManifestStore(manifest)


def test_private_json_and_recovery_ledger_reject_dangling_symlinks(tmp_path: Path) -> None:
    dangling_target = tmp_path / "missing-target.json"
    private_path = tmp_path / "private.json"
    private_path.symlink_to(dangling_target)

    with pytest.raises(RuntimeError, match="symlink"):
        atomic_write_private_json(private_path, {"safe": True})
    assert private_path.is_symlink()

    manifest = tmp_path / "capture_manifest.jsonl"
    recovery_path = capture_recovery_path(manifest)
    recovery_path.symlink_to(dangling_target)
    with pytest.raises(RuntimeError, match="symlink"):
        CaptureManifestStore(manifest)


@pytest.mark.parametrize("ledger_kind", ["directory", "fifo"])
def test_capture_manifest_health_rejects_non_regular_recovery_without_hanging(
    tmp_path: Path,
    ledger_kind: str,
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    recovery_path = capture_recovery_path(manifest)
    if ledger_kind == "directory":
        recovery_path.mkdir()
    else:
        os.mkfifo(recovery_path)
    child_code = """
import json
import sys
from pathlib import Path
from mango_mvp.productization.capture_staging import capture_manifest_health
print(json.dumps(capture_manifest_health(Path(sys.argv[1])), sort_keys=True))
"""

    completed = subprocess.run(
        [sys.executable, "-c", child_code, str(manifest)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    health = json.loads(completed.stdout)
    assert health["tail_status"] == "clean"
    assert health["recovery_status"] == "invalid"


@pytest.mark.parametrize("locked_file", ["manifest", "recovery"])
def test_capture_manifest_health_fails_closed_without_waiting_on_busy_lock(
    tmp_path: Path,
    locked_file: str,
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    if locked_file == "recovery":
        with manifest.open("ab") as handle:
            handle.write(b'{"event_key":"unfinished"')
        store.recover_incomplete_tail()
    target = manifest if locked_file == "manifest" else capture_recovery_path(manifest)
    child_code = """
import fcntl
import sys
path = sys.argv[1]
with open(path, "rb") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    print("locked", flush=True)
    sys.stdin.read(1)
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", child_code, str(target)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        health_code = """
import json
import sys
from pathlib import Path
from mango_mvp.productization.capture_staging import capture_manifest_health
print(json.dumps(capture_manifest_health(Path(sys.argv[1])), sort_keys=True))
"""
        completed = subprocess.run(
            [sys.executable, "-c", health_code, str(manifest)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
        )
        assert completed.returncode == 0, completed.stderr
        health = json.loads(completed.stdout)
    finally:
        holder.communicate("x", timeout=2)

    if locked_file == "manifest":
        assert health["tail_status"] == "invalid"
    else:
        assert health["tail_status"] == "clean"
        assert health["recovery_status"] == "invalid"


@pytest.mark.parametrize("manifest_kind", ["directory", "fifo"])
@pytest.mark.parametrize("operation", ["read", "append", "ack"])
def test_capture_manifest_operations_reject_non_regular_file_without_hanging(
    tmp_path: Path,
    manifest_kind: str,
    operation: str,
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    incident_sha256 = "unused"
    if operation == "ack":
        store = CaptureManifestStore(manifest)
        store.append(manifest_entry("CALL-1"))
        with manifest.open("ab") as handle:
            handle.write(b'{"event_key":"unfinished"')
        store.append(manifest_entry("CALL-2"))
        incident_sha256 = str(store.recovery_incident_sha256)
        manifest.unlink()
    if manifest_kind == "directory":
        manifest.mkdir()
    else:
        os.mkfifo(manifest)
    child_code = """
import sys
from pathlib import Path
from mango_mvp.productization.capture_staging import (
    CaptureManifestStore, ManifestEntry, acknowledge_capture_recovery,
)
path = Path(sys.argv[1])
operation = sys.argv[2]
try:
    if operation == "read":
        CaptureManifestStore(path).read_entries()
    elif operation == "append":
        CaptureManifestStore(path).append(ManifestEntry(
            schema_version="v1", created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton", provider="mango", event_key="foton:mango:child",
            provider_call_id="child", recording_id=None,
            started_at="2026-07-09T10:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None,
            status="recording_retry_expired",
        ))
    else:
        acknowledge_capture_recovery(
            path, expected_count=1, expected_incident_sha256=sys.argv[3]
        )
except (OSError, RuntimeError):
    print("rejected")
else:
    raise SystemExit(3)
"""

    completed = subprocess.run(
        [sys.executable, "-c", child_code, str(manifest), operation, incident_sha256],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "rejected"


def test_manifest_adds_separator_after_valid_final_record_without_newline(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    recordings = tmp_path / "recordings"
    stage_capture_events([event("CALL-1", "rec-1")], store, recordings, FakeDownloader(), validator=fake_validator)
    manifest.write_bytes(manifest.read_bytes().removesuffix(b"\n"))

    with pytest.raises(RuntimeError, match="shrank"):
        store.read_entries()
    store = CaptureManifestStore(manifest)
    assert len(store.read_entries()) == 1
    assert store.incomplete_trailing_records == 0

    stage_capture_events(
        [event("CALL-2", "rec-2", started_offset_sec=1)],
        store,
        recordings,
        FakeDownloader(),
        validator=fake_validator,
    )

    assert [row["provider_call_id"] for row in read_manifest(manifest)] == ["CALL-1", "CALL-2"]


def test_manifest_ensure_exists_is_owner_only_and_idempotent(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)

    store.ensure_exists()
    first = manifest.stat()
    store.ensure_exists()

    assert manifest.read_bytes() == b""
    assert manifest.stat().st_mode & 0o777 == 0o600
    assert manifest.stat().st_ino == first.st_ino
    assert manifest.stat().st_mtime_ns == first.st_mtime_ns


def test_manifest_ensure_exists_refuses_to_mask_missing_recovered_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    store.append(manifest_entry("CALL-2"))
    incident_sha256 = str(store.recovery_incident_sha256)
    acknowledge_capture_recovery(
        manifest,
        expected_count=1,
        expected_incident_sha256=incident_sha256,
    )
    manifest.unlink()

    with pytest.raises(RuntimeError, match="missing after recorded recovery"):
        CaptureManifestStore(manifest).ensure_exists()

    assert not manifest.exists()
    assert capture_recovery_path(manifest).exists()


@pytest.mark.parametrize("operation", ["ensure", "append"])
def test_manifest_store_does_not_recreate_file_removed_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
) -> None:
    from mango_mvp.productization import capture_staging as module

    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    real_open = module._open_regular_file
    removed = False

    def remove_before_open(path: Path, flags: int, **kwargs: object) -> int:
        nonlocal removed
        if path == manifest and not removed:
            manifest.unlink()
            removed = True
        return real_open(path, flags, **kwargs)

    monkeypatch.setattr(module, "_open_regular_file", remove_before_open)

    with pytest.raises(FileNotFoundError):
        if operation == "ensure":
            store.ensure_exists()
        else:
            store.append(manifest_entry("CALL-2"))

    assert removed is True
    assert not manifest.exists()


def test_manifest_store_marks_peer_created_file_seen_and_never_recreates(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    stale_store = CaptureManifestStore(manifest)
    CaptureManifestStore(manifest).append(manifest_entry("CALL-1"))

    assert [entry.provider_call_id for entry in stale_store.read_entries()] == ["CALL-1"]
    manifest.unlink()

    with pytest.raises(RuntimeError, match="disappeared"):
        stale_store.read_entries()
    with pytest.raises(FileNotFoundError):
        stale_store.append(manifest_entry("CALL-2"))
    assert not manifest.exists()


def test_manifest_store_rejects_inode_replacement_but_new_store_accepts_it(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    CaptureManifestStore(replacement).append(manifest_entry("CALL-2"))
    os.replace(replacement, manifest)

    with pytest.raises(RuntimeError, match="inode changed"):
        store.append(manifest_entry("CALL-3"))
    assert [entry.provider_call_id for entry in CaptureManifestStore(manifest).read_entries()] == [
        "CALL-2"
    ]

    relocated_store = CaptureManifestStore(manifest)
    relocated_store.append(manifest_entry("CALL-3"))
    assert [entry.provider_call_id for entry in relocated_store.read_entries()] == [
        "CALL-2",
        "CALL-3",
    ]


def test_manifest_store_rejects_same_inode_shrink(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    store.append(manifest_entry("CALL-2"))
    first_record_end = manifest.read_bytes().index(b"\n") + 1
    with manifest.open("r+b") as handle:
        handle.truncate(first_record_end)
        handle.flush()
        os.fsync(handle.fileno())

    with pytest.raises(RuntimeError, match="shrank"):
        store.append(manifest_entry("CALL-3"))
    assert [row["provider_call_id"] for row in read_manifest(manifest)] == ["CALL-1"]


def test_manifest_store_accepts_peer_append_with_unchanged_prefix(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    CaptureManifestStore(manifest).append(manifest_entry("CALL-2"))

    store.append(manifest_entry("CALL-3"))

    assert [entry.provider_call_id for entry in store.read_entries()] == [
        "CALL-1",
        "CALL-2",
        "CALL-3",
    ]


def test_manifest_store_rejects_path_swap_after_open_without_writing_unlinked_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization import capture_staging as module

    manifest = tmp_path / "capture_manifest.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    CaptureManifestStore(replacement).append(manifest_entry("CALL-X"))
    real_open = module._open_regular_file
    swapped = False

    def swap_after_open(path: Path, flags: int, **kwargs: object) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, **kwargs)
        if path == manifest and not swapped:
            os.replace(replacement, manifest)
            swapped = True
        return descriptor

    monkeypatch.setattr(module, "_open_regular_file", swap_after_open)

    with pytest.raises(RuntimeError, match="changed while open"):
        store.append(manifest_entry("CALL-2"))

    assert swapped is True
    assert [row["provider_call_id"] for row in read_manifest(manifest)] == ["CALL-X"]


def test_manifest_cache_rejects_same_inode_rewrite_between_identity_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization import capture_staging as module

    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    first = manifest_entry("CALL-A")
    second = manifest_entry("CALL-B")
    store.append(first)
    assert store.read_entries()[0].event_key == first.event_key
    replacement = (
        json.dumps(second.to_json_dict(), ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert len(replacement) == manifest.stat().st_size
    real_identity = module._assert_open_path_identity
    checks = 0

    def rewrite_after_first_check(path: Path, descriptor: int, **kwargs: object) -> os.stat_result:
        nonlocal checks
        result = real_identity(path, descriptor, **kwargs)
        checks += 1
        if path == manifest and checks == 1:
            manifest.write_bytes(replacement)
        return result

    monkeypatch.setattr(module, "_assert_open_path_identity", rewrite_after_first_check)

    with pytest.raises(RuntimeError, match="changed while reading"):
        store.read_entries()
    assert checks == 2
    assert [row["provider_call_id"] for row in read_manifest(manifest)] == ["CALL-B"]


def test_manifest_rejects_corruption_before_final_record(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    manifest.write_text('{"broken":\n{"event_key":"also-unfinished"', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        CaptureManifestStore(manifest).read_entries()


@pytest.mark.parametrize(
    "broken_tail,expected_error",
    [
        (b"{BAD", json.JSONDecodeError),
        (b'{"event_key":"unfinished"\x00\x00', json.JSONDecodeError),
        (b'{"event_key":"\xff"}', UnicodeDecodeError),
        (b"{}", ValueError),
        (b"[]", ValueError),
        (b'{"event_key":"only"}', ValueError),
        (b'{"event_key":[]}', ValueError),
    ],
)
def test_manifest_rejects_ambiguous_final_corruption_without_mutation(
    tmp_path: Path,
    broken_tail: bytes,
    expected_error: type[Exception],
) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    store.append(manifest_entry("CALL-1"))
    with manifest.open("ab") as handle:
        handle.write(broken_tail)
    before = manifest.read_bytes()

    with pytest.raises(expected_error):
        store.append(manifest_entry("CALL-2"))

    assert manifest.read_bytes() == before


def test_every_byte_prefix_of_canonical_entry_is_recoverable(tmp_path: Path) -> None:
    encoded = json.dumps(
        manifest_entry("CALL-RU").to_json_dict(),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    for cut in range(1, len(encoded)):
        manifest = tmp_path / f"capture-{cut}.jsonl"
        manifest.write_bytes(encoded[:cut])
        store = CaptureManifestStore(manifest)
        assert store.read_entries() == (), f"cut={cut}"
        assert store.incomplete_trailing_records == 1, f"cut={cut}"


def test_manifest_append_is_locked_owner_only_and_linear_per_store(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    parse_calls = 0
    original_parse = store._parse_raw

    def counted_parse(raw: bytes) -> None:
        nonlocal parse_calls
        parse_calls += 1
        original_parse(raw)

    store._parse_raw = counted_parse  # type: ignore[method-assign]
    store.append(manifest_entry("CALL-0"))
    cache_identity = id(store._cached_entries)
    for index in range(1, 20):
        store.append(manifest_entry(f"CALL-{index}"))

    assert parse_calls == 1
    assert id(store._cached_entries) == cache_identity
    assert manifest.stat().st_mode & 0o777 == 0o600

    def concurrent_append(index: int) -> None:
        CaptureManifestStore(manifest).append(manifest_entry(f"CONCURRENT-{index}"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(concurrent_append, range(24)))

    keys = [entry.event_key for entry in CaptureManifestStore(manifest).read_entries()]
    assert len(keys) == len(set(keys)) == 44


def test_manifest_cache_rejects_same_size_rewrite_with_restored_mtime(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    store = CaptureManifestStore(manifest)
    first = manifest_entry("CALL-A")
    second = manifest_entry("CALL-B")
    store.append(first)
    assert store.read_entries()[0].event_key == first.event_key
    before = manifest.stat()
    replacement = (
        json.dumps(second.to_json_dict(), ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert len(replacement) == before.st_size

    manifest.write_bytes(replacement)
    os.utime(manifest, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert manifest.stat().st_ctime_ns != before.st_ctime_ns
    with pytest.raises(RuntimeError, match="rewritten"):
        store.read_entries()
    assert CaptureManifestStore(manifest).read_entries()[0].event_key == second.event_key


def test_stage_capture_events_downloads_and_writes_validated_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    downloader = FakeDownloader()

    summary = stage_capture_events(
        events=[event("CALL-1", "rec-1")],
        manifest_store=CaptureManifestStore(manifest),
        recordings_dir=recordings,
        downloader=downloader,
        validator=fake_validator,
    )

    rows = read_manifest(manifest)
    assert summary.downloaded == 1
    assert summary.failed == 0
    assert len(downloader.calls) == 1
    assert len(rows) == 1
    assert rows[0]["status"] == "downloaded"
    assert rows[0]["recording_id"] == "rec-1"
    assert rows[0]["checksum_sha256"] == hashlib.sha256(
        Path(rows[0]["local_audio_path"]).read_bytes()
    ).hexdigest()
    assert rows[0]["duration_sec"] == 12.5
    assert Path(rows[0]["local_audio_path"]).exists()


def test_stage_capture_events_rejects_existing_audio_symlink_without_download(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    item = event("CALL-SYMLINK", "rec-symlink")
    target = recordings / build_capture_audio_filename(item, "rec-symlink")
    victim = tmp_path / "victim.mp3"
    victim.write_bytes(b"synthetic victim")
    target.symlink_to(victim)
    downloader = FakeDownloader()

    summary = stage_capture_events(
        [item],
        CaptureManifestStore(tmp_path / "capture_manifest.jsonl"),
        recordings,
        downloader,
        validator=fake_validator,
    )

    assert summary.failed == 1
    assert summary.reused_existing_file == 0
    assert downloader.calls == []
    assert target.is_symlink()
    assert victim.read_bytes() == b"synthetic victim"


def test_stage_capture_events_rejects_multi_part_symlink_without_download(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    item = event(
        "CALL-MULTI-SYMLINK",
        "rec-1",
        recording_refs=("rec-1", "rec-2"),
    )
    base = recordings / build_capture_audio_filename(item, "rec-1")
    first_part, _second_part = recording_part_paths(base, ("rec-1", "rec-2"))
    victim = tmp_path / "victim-part.mp3"
    victim.write_bytes(b"synthetic victim")
    first_part.symlink_to(victim)
    downloader = FakeDownloader()

    summary = stage_capture_events(
        [item],
        CaptureManifestStore(tmp_path / "capture_manifest.jsonl"),
        recordings,
        downloader,
        validator=fake_validator,
    )

    assert summary.failed == 1
    assert summary.needs_review_multiple_recordings == 0
    assert downloader.calls == []
    assert first_part.is_symlink()
    assert victim.read_bytes() == b"synthetic victim"


def test_stage_capture_events_is_idempotent_on_second_run(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    first_downloader = FakeDownloader()
    store = CaptureManifestStore(manifest)
    events = [event("CALL-1", "rec-1")]

    first = stage_capture_events(events, store, recordings, first_downloader, validator=fake_validator)
    second_downloader = FakeDownloader()
    second = stage_capture_events(events, store, recordings, second_downloader, validator=fake_validator)

    assert first.downloaded == 1
    assert second.already_manifested == 1
    assert len(second_downloader.calls) == 0
    assert len(read_manifest(manifest)) == 1


def test_stage_capture_events_quarantines_late_part_and_keeps_monotonic_set(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    store = CaptureManifestStore(manifest)
    first_downloader = FakeDownloader()
    first = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1",))],
        store,
        recordings,
        first_downloader,
        validator=fake_validator,
    )
    second_downloader = FakeDownloader()
    second = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1", "rec-2"))],
        store,
        recordings,
        second_downloader,
        validator=fake_validator,
    )
    third_downloader = FakeDownloader()
    third = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1", "rec-2"))],
        store,
        recordings,
        third_downloader,
        validator=fake_validator,
    )

    rows = read_manifest(manifest)
    assert first.downloaded == 1
    assert second.downloaded == 0
    assert second.needs_review_multiple_recordings == 1
    assert [call[0] for call in second_downloader.calls] == ["rec-2"]
    assert rows[-1]["status"] == "multiple_recordings_needs_review"
    assert rows[-1]["recording_ids"] == ["rec-1", "rec-2"]
    assert len(rows[-1]["recording_paths"]) == 2
    assert len(list(recordings.glob("*.mp3"))) == 2
    assert third.already_manifested == 1
    assert third_downloader.calls == []

    shrinking = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1",))],
        store,
        recordings,
        FakeDownloader(),
        validator=fake_validator,
    )
    assert shrinking.already_manifested == 1
    assert store.latest_by_event_key()[event("CALL-MULTI").event_key].recording_ids == ("rec-1", "rec-2")

    Path(rows[-1]["recording_paths"][1]).write_bytes(b"")
    repair_downloader = FakeDownloader()
    repaired = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1", "rec-2"))],
        store,
        recordings,
        repair_downloader,
        validator=fake_validator,
    )
    assert repaired.needs_review_multiple_recordings == 1
    assert [call[0] for call in repair_downloader.calls] == ["rec-2"]


def test_old_scalar_manifest_reads_as_single_recording_tuple(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "created_at": "2026-05-07T00:00:00+00:00",
            "tenant_id": "foton",
            "provider": "mango",
            "event_key": "foton:mango:old",
            "provider_call_id": "old",
            "recording_id": "rec-old",
            "recording_ids": [],
            "started_at": "2026-05-07T00:00:00+00:00",
            "direction": "inbound",
            "status": "failed",
        }) + "\n",
        encoding="utf-8",
    )
    assert CaptureManifestStore(manifest).read_entries()[0].recording_ids == ("rec-old",)


def test_late_part_failure_preserves_first_part_and_sanitizes_error(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    store = CaptureManifestStore(manifest)
    first_event = event("CALL-MULTI", "rec-1", recording_refs=("rec-1",))
    stage_capture_events([first_event], store, recordings, FakeDownloader(), validator=fake_validator)
    target = Path(read_manifest(manifest)[0]["local_audio_path"])
    previous = target.read_bytes()

    failed = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1", "rec-2"))],
        store,
        recordings,
        FakeDownloader(fail_first=True),
        validator=fake_validator,
    )

    assert failed.failed == 1
    assert target.read_bytes() == previous
    latest = CaptureManifestStore(manifest).latest_by_event_key()[first_event.event_key]
    assert latest.status == "failed"
    assert latest.recording_ids == ("rec-1", "rec-2")
    assert latest.recording_paths == (str(target),)
    assert latest.error == "RuntimeError:capture_failed"
    assert "temporary" not in latest.error

    retry_downloader = FakeDownloader()
    retry = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1", "rec-2"))],
        store,
        recordings,
        retry_downloader,
        validator=fake_validator,
    )
    assert retry.needs_review_multiple_recordings == 1
    assert [call[0] for call in retry_downloader.calls] == ["rec-2"]


def test_corrupt_nonempty_part_is_removed_and_downloaded_on_retry(tmp_path: Path) -> None:
    store = CaptureManifestStore(tmp_path / "manifest.jsonl")
    recordings = tmp_path / "recordings"
    stage_capture_events([event("CALL-MULTI", "rec-1")], store, recordings, FakeDownloader(), validator=fake_validator)

    def reject_second(path: Path) -> AudioValidation:
        if "__part-" in path.name:
            raise ValueError("corrupt secret /tmp/+79990000000")
        return fake_validator(path)

    failed = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1", "rec-2"))],
        store,
        recordings,
        FakeDownloader(),
        validator=reject_second,
    )
    retry_downloader = FakeDownloader()
    retried = stage_capture_events(
        [event("CALL-MULTI", "rec-1", recording_refs=("rec-1", "rec-2"))],
        store,
        recordings,
        retry_downloader,
        validator=fake_validator,
    )

    assert failed.failed == 1
    assert retried.needs_review_multiple_recordings == 1
    assert [call[0] for call in retry_downloader.calls] == ["rec-2"]
    assert "+79990000000" not in read_manifest(store.path)[-2]["error"]


def test_stage_capture_events_redownloads_missing_downloaded_asset(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    store = CaptureManifestStore(manifest)
    events = [event("CALL-1", "rec-1")]
    first = stage_capture_events(events, store, recordings, FakeDownloader(), validator=fake_validator)
    Path(read_manifest(manifest)[0]["local_audio_path"]).unlink()
    retry_downloader = FakeDownloader()

    retry = stage_capture_events(events, store, recordings, retry_downloader, validator=fake_validator)

    assert first.downloaded == 1
    assert retry.downloaded == 1
    assert len(retry_downloader.calls) == 1
    assert [row["status"] for row in read_manifest(manifest)] == ["downloaded", "downloaded"]


def test_stage_capture_events_retries_failed_download_without_duplicate_asset(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    store = CaptureManifestStore(manifest)
    events = [event("CALL-1", "rec-1")]

    first = stage_capture_events(
        events,
        store,
        recordings,
        FakeDownloader(fail_first=True),
        validator=fake_validator,
    )
    second = stage_capture_events(events, store, recordings, FakeDownloader(), validator=fake_validator)
    third_downloader = FakeDownloader()
    third = stage_capture_events(events, store, recordings, third_downloader, validator=fake_validator)

    assert first.failed == 1
    assert second.downloaded == 1
    assert third.already_manifested == 1
    assert third_downloader.calls == []
    assert [row["status"] for row in read_manifest(manifest)] == ["failed", "downloaded"]
    assert len(list(recordings.iterdir())) == 1


def test_stage_capture_events_reuses_file_when_retry_validation_now_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    store = CaptureManifestStore(manifest)
    events = [event("CALL-1", "rec-1")]

    def reject(_path: Path) -> AudioValidation:
        raise ValueError("corrupt audio")

    first = stage_capture_events(events, store, recordings, FakeDownloader(), validator=reject)
    corrupt_path = Path(read_manifest(manifest)[0]["local_audio_path"])
    corrupt_path.write_bytes(b"still-nonempty-but-corrupt")
    retry_downloader = FakeDownloader()
    second = stage_capture_events(events, store, recordings, retry_downloader, validator=fake_validator)

    assert first.failed == 1
    assert second.reused_existing_file == 1
    assert retry_downloader.calls == []
    assert corrupt_path.read_bytes() == b"still-nonempty-but-corrupt"


def test_stage_capture_events_links_duplicate_recording_to_canonical_asset(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    downloader = FakeDownloader()

    summary = stage_capture_events(
        events=[
            event("CALL-1", "rec-shared", phone="+79990000001"),
            event("CALL-2", "rec-shared", phone="+79990000002", started_offset_sec=30),
        ],
        manifest_store=CaptureManifestStore(manifest),
        recordings_dir=recordings,
        downloader=downloader,
        validator=fake_validator,
    )

    rows = read_manifest(manifest)
    assert summary.downloaded == 1
    assert summary.duplicate_recording == 1
    assert len(downloader.calls) == 1
    assert rows[0]["status"] == "downloaded"
    assert rows[1]["status"] == "duplicate_recording"
    assert rows[1]["canonical_recording_id"] == "rec-shared"
    assert rows[1]["canonical_audio_path"] == rows[0]["local_audio_path"]


def test_duplicate_multi_recording_parts_are_downloaded_once(tmp_path: Path) -> None:
    downloader = FakeDownloader()
    summary = stage_capture_events(
        events=[
            event("CALL-1", "rec-1", recording_refs=("rec-1", "rec-2")),
            event("CALL-2", "rec-2", recording_refs=("rec-2", "rec-1"), started_offset_sec=1),
        ],
        manifest_store=CaptureManifestStore(tmp_path / "manifest.jsonl"),
        recordings_dir=tmp_path / "recordings",
        downloader=downloader,
        validator=fake_validator,
    )

    assert [call[0] for call in downloader.calls] == ["rec-1", "rec-2"]
    assert summary.needs_review_multiple_recordings == 1
    assert summary.duplicate_recording == 1


def test_stage_capture_events_records_no_recording_without_downloading(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    downloader = FakeDownloader()

    summary = stage_capture_events(
        events=[event("CALL-1", None)],
        manifest_store=CaptureManifestStore(manifest),
        recordings_dir=tmp_path / "recordings",
        downloader=downloader,
        validator=fake_validator,
    )

    rows = read_manifest(manifest)
    assert summary.skipped_no_recording == 1
    assert len(downloader.calls) == 0
    assert rows[0]["status"] == "skipped_no_recording"


def test_stage_capture_events_dry_run_records_plan_without_file(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    downloader = FakeDownloader()

    summary = stage_capture_events(
        events=[event("CALL-1", "rec-1")],
        manifest_store=CaptureManifestStore(manifest),
        recordings_dir=tmp_path / "recordings",
        downloader=downloader,
        dry_run=True,
        validator=fake_validator,
    )

    rows = read_manifest(manifest)
    assert summary.dry_run_download == 1
    assert len(downloader.calls) == 0
    assert rows[0]["status"] == "dry_run_download"
    assert rows[0]["dry_run"] is True
    assert not Path(rows[0]["local_audio_path"]).exists()


def test_stage_capture_events_retries_after_failed_manifest_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    store = CaptureManifestStore(manifest)
    events = [event("CALL-1", "rec-1")]

    failed = stage_capture_events(
        events=events,
        manifest_store=store,
        recordings_dir=recordings,
        downloader=FakeDownloader(fail_first=True),
        validator=fake_validator,
    )
    retry = stage_capture_events(
        events=events,
        manifest_store=store,
        recordings_dir=recordings,
        downloader=FakeDownloader(),
        validator=fake_validator,
    )

    rows = read_manifest(manifest)
    assert failed.failed == 1
    assert retry.downloaded == 1
    assert [row["status"] for row in rows] == ["failed", "downloaded"]


def test_audit_capture_manifest_reports_missing_integrity_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    missing_file = recordings / "missing.mp3"
    zero_file = recordings / "zero.mp3"
    zero_file.write_bytes(b"")

    rows = [
        {
            "schema_version": "capture_manifest_v1",
            "created_at": "2026-05-07T00:00:00+00:00",
            "tenant_id": "foton",
            "provider": "mango",
            "event_key": "foton:mango:missing",
            "provider_call_id": "missing",
            "recording_id": "rec-missing",
            "started_at": "2026-05-07T00:00:00+00:00",
            "direction": "inbound",
            "status": "downloaded",
            "local_audio_path": str(missing_file),
        },
        {
            "schema_version": "capture_manifest_v1",
            "created_at": "2026-05-07T00:00:00+00:00",
            "tenant_id": "foton",
            "provider": "mango",
            "event_key": "foton:mango:zero",
            "provider_call_id": "zero",
            "recording_id": "rec-zero",
            "started_at": "2026-05-07T00:00:00+00:00",
            "direction": "inbound",
            "status": "downloaded",
            "local_audio_path": str(zero_file),
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    audit = audit_capture_manifest(manifest, recordings)

    assert audit["missing_files"] == 1
    assert audit["zero_size_files"] == 1
    assert audit["checksum_missing"] == 1
    assert audit["duration_missing"] == 1


def test_audit_capture_manifest_reports_unreferenced_audio_files(tmp_path: Path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    referenced = recordings / "referenced.mp3"
    unreferenced = recordings / "unreferenced.mp3"
    referenced.write_bytes(b"ok")
    unreferenced.write_bytes(b"orphan")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "capture_manifest_v1",
                "created_at": "2026-05-07T00:00:00+00:00",
                "tenant_id": "foton",
                "provider": "mango",
                "event_key": "foton:mango:referenced",
                "provider_call_id": "referenced",
                "recording_id": "rec-referenced",
                "started_at": "2026-05-07T00:00:00+00:00",
                "direction": "inbound",
                "status": "downloaded",
                "local_audio_path": str(referenced),
                "checksum_sha256": "sha",
                "duration_sec": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_capture_manifest(manifest, recordings)

    assert audit["unreferenced_audio_files"] == 1
    assert str(unreferenced) in audit["samples"]["unreferenced_audio_files"]
