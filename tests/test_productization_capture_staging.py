from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.productization.capture_staging import (
    AudioValidation,
    CaptureManifestStore,
    audit_capture_manifest,
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
        checksum_sha256=f"sha-{size}",
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
        raw_payload={},
    )


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    assert rows[0]["checksum_sha256"].startswith("sha-")
    assert rows[0]["duration_sec"] == 12.5
    assert Path(rows[0]["local_audio_path"]).exists()


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
