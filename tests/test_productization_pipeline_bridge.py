from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.productization.capture_staging import file_sha256
from mango_mvp.productization.pipeline_bridge import (
    BridgeStatus,
    build_pipeline_bridge_plan,
    write_bridge_plan_csv,
)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def manifest_row(
    event_id: str,
    audio_path: Path | None,
    status: str = "downloaded",
    checksum: str | None = None,
    duration_sec: float | None = 30.0,
    started_at: str = "2026-05-07T06:00:00+00:00",
    phone: str = "+79990000000",
) -> dict:
    return {
        "schema_version": "capture_manifest_v1",
        "created_at": "2026-05-07T07:00:00+00:00",
        "tenant_id": "foton",
        "provider": "mango",
        "event_key": f"foton:mango:{event_id}",
        "provider_call_id": event_id,
        "recording_id": f"rec-{event_id}",
        "started_at": started_at,
        "ended_at": "2026-05-07T06:00:30+00:00",
        "direction": "inbound",
        "client_phone": phone,
        "manager_ref": "101",
        "status": status,
        "local_audio_path": str(audio_path) if audio_path else None,
        "size_bytes": audio_path.stat().st_size if audio_path and audio_path.exists() else None,
        "checksum_sha256": checksum,
        "duration_sec": duration_sec,
        "codec_name": "mp3",
        "channels": 2,
        "sample_rate": 8000,
    }


def make_audio(path: Path, data: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def statuses(plan: dict) -> list[str]:
    return [item["status"] for item in plan["items"]]


def test_bridge_marks_valid_unseen_capture_as_would_import(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [manifest_row("CALL-1", audio, checksum=file_sha256(audio))])

    plan = build_pipeline_bridge_plan(
        manifest_path=manifest,
        source_dir=tmp_path / "source",
        db_paths=(),
    )

    assert statuses(plan) == [BridgeStatus.WOULD_IMPORT.value]
    assert plan["summary"]["bridge_status_counts"] == {"would_import": 1}
    item = plan["items"][0]
    assert item["proposed_filename"].startswith("2026-05-07__09-00-00__79990000000__mango_101_CALL-1")
    assert item["proposed_metadata"]["recording_id"] == "rec-CALL-1"


def test_bridge_marks_existing_audio_by_phone_and_time_fuzzy_match(tmp_path: Path) -> None:
    capture = make_audio(tmp_path / "capture" / "call.mp3")
    source_dir = tmp_path / "source"
    make_audio(source_dir / "2026-05-07__09-00-45__79990000000__Manager.mp3")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [manifest_row("CALL-1", capture, checksum=file_sha256(capture))])

    plan = build_pipeline_bridge_plan(manifest, source_dir, db_paths=(), tolerance_sec=120)

    assert statuses(plan) == [BridgeStatus.ALREADY_PRESENT_AUDIO.value]
    assert plan["items"][0]["matched_audio_delta_sec"] == 45


def test_bridge_marks_existing_db_row_read_only(tmp_path: Path) -> None:
    capture = make_audio(tmp_path / "capture" / "call.mp3")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [manifest_row("CALL-1", capture, checksum=file_sha256(capture))])
    db_path = tmp_path / "calls.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE call_records ("
            "id INTEGER PRIMARY KEY, source_filename TEXT, source_file TEXT, "
            "source_call_id TEXT, phone TEXT, started_at TEXT)"
        )
        conn.execute(
            "INSERT INTO call_records (source_filename, source_file, source_call_id, phone, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("existing.mp3", "/tmp/existing.mp3", "old", "+79990000000", "2026-05-07 09:01:00"),
        )
        conn.commit()

    plan = build_pipeline_bridge_plan(
        manifest_path=manifest,
        source_dir=tmp_path / "source",
        db_paths=(db_path,),
        tolerance_sec=120,
    )

    assert statuses(plan) == [BridgeStatus.ALREADY_PRESENT_DB.value]
    assert plan["items"][0]["matched_db_source_filename"] == "existing.mp3"
    assert plan["items"][0]["matched_db_delta_sec"] == 60


def test_bridge_blocks_checksum_mismatch(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [manifest_row("CALL-1", audio, checksum="bad")])

    plan = build_pipeline_bridge_plan(manifest, tmp_path / "source", db_paths=())

    assert statuses(plan) == [BridgeStatus.BLOCKED_CHECKSUM_MISMATCH.value]
    assert plan["audit"]["blocked"] == 1


def test_bridge_blocks_non_downloaded_manifest_status(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            manifest_row(
                "CALL-1",
                None,
                status="skipped_no_recording",
                checksum=None,
                duration_sec=None,
            )
        ],
    )

    plan = build_pipeline_bridge_plan(manifest, tmp_path / "source", db_paths=())

    assert statuses(plan) == [BridgeStatus.BLOCKED_MANIFEST_STATUS.value]
    assert plan["summary"]["manifest_status_counts"] == {"skipped_no_recording": 1}


def test_bridge_blocks_missing_file_and_missing_duration(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            manifest_row("CALL-1", tmp_path / "capture" / "missing.mp3", checksum="sha"),
            manifest_row("CALL-2", audio, checksum=file_sha256(audio), duration_sec=None),
        ],
    )

    plan = build_pipeline_bridge_plan(manifest, tmp_path / "source", db_paths=())

    assert statuses(plan) == [
        BridgeStatus.BLOCKED_MISSING_FILE.value,
        BridgeStatus.BLOCKED_DURATION_MISSING.value,
    ]


def test_write_bridge_plan_csv_writes_import_plan_rows(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    manifest = tmp_path / "manifest.jsonl"
    csv_path = tmp_path / "plan.csv"
    write_manifest(manifest, [manifest_row("CALL-1", audio, checksum=file_sha256(audio))])
    plan = build_pipeline_bridge_plan(manifest, tmp_path / "source", db_paths=())

    write_bridge_plan_csv(plan, csv_path)

    text = csv_path.read_text(encoding="utf-8")
    assert "status,reason,started_at_msk" in text
    assert "would_import" in text
