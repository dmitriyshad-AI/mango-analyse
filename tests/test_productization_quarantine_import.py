from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.productization.capture_staging import file_sha256
from mango_mvp.productization.quarantine_import import (
    build_quarantine_import_plan,
    materialize_quarantine_package,
)


def make_audio(path: Path, data: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def bridge_item(
    event_id: str,
    audio_path: Path | None,
    status: str = "would_import",
    checksum: str | None = None,
    proposed_filename: str | None = None,
    phone: str | None = "+79990000000",
) -> dict:
    return {
        "event_key": f"foton:mango:{event_id}",
        "provider_call_id": event_id,
        "recording_id": f"rec-{event_id}",
        "status": status,
        "reason": "validated_capture_not_found_in_source_dir_or_db",
        "started_at": "2026-05-07T06:00:00+00:00",
        "started_at_msk": "2026-05-07T09:00:00+03:00",
        "direction": "inbound",
        "client_phone": phone,
        "manager_ref": "101",
        "local_audio_path": str(audio_path) if audio_path else None,
        "size_bytes": audio_path.stat().st_size if audio_path and audio_path.exists() else None,
        "checksum_sha256": checksum,
        "duration_sec": 30.0,
        "proposed_filename": proposed_filename or f"2026-05-07__09-00-00__79990000000__mango_101_{event_id}.mp3",
        "proposed_metadata": {
            "source": "mango_api_capture",
            "tenant_id": "foton",
            "provider": "mango",
            "event_key": f"foton:mango:{event_id}",
            "provider_call_id": event_id,
            "recording_id": f"rec-{event_id}",
            "started_at_msk": "2026-05-07T09:00:00+03:00",
            "client_phone": phone,
            "manager_ref": "101",
            "direction": "inbound",
            "duration_sec": 30.0,
            "checksum_sha256": checksum,
        },
    }


def write_bridge_plan(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps({"summary": {}, "audit": {}, "items": items}), encoding="utf-8")


def write_plan(path: Path, plan: dict) -> None:
    path.write_text(json.dumps(plan), encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_quarantine_import_plan_writes_ready_metadata_csv(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    bridge_plan = tmp_path / "bridge.json"
    metadata_csv = tmp_path / "quarantine" / "metadata.csv"
    write_bridge_plan(bridge_plan, [bridge_item("CALL-1", audio, checksum=file_sha256(audio))])

    plan = build_quarantine_import_plan(
        bridge_plan_path=bridge_plan,
        quarantine_dir=tmp_path / "quarantine" / "audio",
        metadata_csv_path=metadata_csv,
    )

    rows = read_csv(metadata_csv)
    assert plan["summary"]["ready"] == 1
    assert plan["summary"]["metadata_rows"] == 1
    assert plan["summary"]["target_filename_collisions"] == 0
    assert plan["summary"]["quarantine_audio_files"] == 0
    assert plan["summary"]["ready_by_day"] == {"2026-05-07": 1}
    assert plan["audit"]["blocked"] == 0
    assert len(rows) == 1
    assert rows[0]["filename"].endswith("CALL-1.mp3")
    assert rows[0]["source_audio_path"] == str(audio)
    assert rows[0]["phone"] == "+79990000000"
    assert rows[0]["call_id"] == "CALL-1"
    assert rows[0]["recording_id"] == "rec-CALL-1"


def test_quarantine_import_plan_skips_non_import_bridge_status(tmp_path: Path) -> None:
    bridge_plan = tmp_path / "bridge.json"
    metadata_csv = tmp_path / "metadata.csv"
    write_bridge_plan(bridge_plan, [bridge_item("CALL-1", None, status="already_present_audio")])

    plan = build_quarantine_import_plan(bridge_plan, tmp_path / "audio", metadata_csv)

    assert plan["summary"]["skipped_non_import_status"] == 1
    assert plan["audit"]["status_counts"] == {"skipped_non_import_status": 1}
    assert read_csv(metadata_csv) == []


def test_quarantine_import_plan_blocks_missing_source_file(tmp_path: Path) -> None:
    bridge_plan = tmp_path / "bridge.json"
    metadata_csv = tmp_path / "metadata.csv"
    write_bridge_plan(
        bridge_plan,
        [bridge_item("CALL-1", tmp_path / "capture" / "missing.mp3", checksum="sha")],
    )

    plan = build_quarantine_import_plan(bridge_plan, tmp_path / "audio", metadata_csv)

    assert plan["audit"]["blocked"] == 1
    assert plan["audit"]["status_counts"] == {"blocked_missing_source": 1}


def test_quarantine_import_plan_blocks_checksum_mismatch(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    bridge_plan = tmp_path / "bridge.json"
    metadata_csv = tmp_path / "metadata.csv"
    write_bridge_plan(bridge_plan, [bridge_item("CALL-1", audio, checksum="bad")])

    plan = build_quarantine_import_plan(bridge_plan, tmp_path / "audio", metadata_csv)

    assert plan["audit"]["blocked"] == 1
    assert plan["audit"]["status_counts"] == {"blocked_checksum_mismatch": 1}


def test_quarantine_import_plan_blocks_missing_required_metadata(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    bridge_plan = tmp_path / "bridge.json"
    metadata_csv = tmp_path / "metadata.csv"
    write_bridge_plan(bridge_plan, [bridge_item("CALL-1", audio, checksum=file_sha256(audio), phone=None)])

    plan = build_quarantine_import_plan(bridge_plan, tmp_path / "audio", metadata_csv)

    assert plan["audit"]["blocked"] == 1
    assert plan["audit"]["status_counts"] == {"blocked_missing_metadata": 1}


def test_quarantine_import_plan_makes_filename_collisions_deterministic(tmp_path: Path) -> None:
    audio1 = make_audio(tmp_path / "capture" / "call1.mp3", b"audio1")
    audio2 = make_audio(tmp_path / "capture" / "call2.mp3", b"audio2")
    bridge_plan = tmp_path / "bridge.json"
    metadata_csv = tmp_path / "metadata.csv"
    shared_name = "same.mp3"
    write_bridge_plan(
        bridge_plan,
        [
            bridge_item("CALL-1", audio1, checksum=file_sha256(audio1), proposed_filename=shared_name),
            bridge_item("CALL-2", audio2, checksum=file_sha256(audio2), proposed_filename=shared_name),
        ],
    )

    plan = build_quarantine_import_plan(bridge_plan, tmp_path / "audio", metadata_csv)
    rows = read_csv(metadata_csv)
    filenames = [row["filename"] for row in rows]

    assert plan["audit"]["ready"] == 2
    assert plan["audit"]["target_filename_collisions"] == 0
    assert plan["summary"]["target_filename_collisions"] == 0
    assert filenames[0] == "same.mp3"
    assert filenames[1].startswith("same__event_")
    assert filenames[1].endswith(".mp3")


def test_quarantine_import_plan_accepts_utc_start_when_msk_missing(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3")
    item = bridge_item("CALL-1", audio, checksum=file_sha256(audio))
    item.pop("started_at_msk")
    item["proposed_metadata"].pop("started_at_msk")
    bridge_plan = tmp_path / "bridge.json"
    metadata_csv = tmp_path / "metadata.csv"
    write_bridge_plan(bridge_plan, [item])

    plan = build_quarantine_import_plan(bridge_plan, tmp_path / "audio", metadata_csv)

    assert plan["audit"]["blocked"] == 0
    assert plan["audit"]["ready_by_day"] == {"2026-05-07": 1}
    assert read_csv(metadata_csv)[0]["started_at"] == "2026-05-07T06:00:00+00:00"


def test_materialize_quarantine_package_copies_and_is_idempotent(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3", b"audio-body")
    bridge_plan = tmp_path / "bridge.json"
    import_plan_path = tmp_path / "quarantine" / "plan.json"
    quarantine_dir = tmp_path / "quarantine" / "audio"
    metadata_csv = tmp_path / "quarantine" / "metadata.csv"
    write_bridge_plan(bridge_plan, [bridge_item("CALL-1", audio, checksum=file_sha256(audio))])
    plan = build_quarantine_import_plan(bridge_plan, quarantine_dir, metadata_csv)
    write_plan(import_plan_path, plan)

    first = materialize_quarantine_package(import_plan_path, mode="copy")
    target = Path(plan["items"][0]["target_audio_path"])

    assert first["summary"]["copied"] == 1
    assert first["summary"]["blocked"] == 0
    assert first["audit"]["checksum_verified_files"] == 1
    assert target.read_bytes() == b"audio-body"

    second = materialize_quarantine_package(import_plan_path, mode="copy")

    assert second["summary"]["already_present"] == 1
    assert second["summary"]["copied"] == 0
    assert second["summary"]["blocked"] == 0


def test_materialize_quarantine_package_blocks_existing_checksum_mismatch(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3", b"audio-body")
    bridge_plan = tmp_path / "bridge.json"
    import_plan_path = tmp_path / "quarantine" / "plan.json"
    quarantine_dir = tmp_path / "quarantine" / "audio"
    metadata_csv = tmp_path / "quarantine" / "metadata.csv"
    write_bridge_plan(bridge_plan, [bridge_item("CALL-1", audio, checksum=file_sha256(audio))])
    plan = build_quarantine_import_plan(bridge_plan, quarantine_dir, metadata_csv)
    write_plan(import_plan_path, plan)
    target = Path(plan["items"][0]["target_audio_path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong")

    report = materialize_quarantine_package(import_plan_path, mode="copy")

    assert report["summary"]["blocked"] == 1
    assert report["summary"]["status_counts"] == {"blocked_existing_target_checksum_mismatch": 1}


def test_materialize_quarantine_package_blocks_target_outside_quarantine(tmp_path: Path) -> None:
    audio = make_audio(tmp_path / "capture" / "call.mp3", b"audio-body")
    bridge_plan = tmp_path / "bridge.json"
    import_plan_path = tmp_path / "quarantine" / "plan.json"
    quarantine_dir = tmp_path / "quarantine" / "audio"
    metadata_csv = tmp_path / "quarantine" / "metadata.csv"
    write_bridge_plan(bridge_plan, [bridge_item("CALL-1", audio, checksum=file_sha256(audio))])
    plan = build_quarantine_import_plan(bridge_plan, quarantine_dir, metadata_csv)
    plan["items"][0]["target_audio_path"] = str(tmp_path / "outside.mp3")
    write_plan(import_plan_path, plan)

    report = materialize_quarantine_package(import_plan_path, mode="copy")

    assert report["summary"]["blocked"] == 1
    assert report["summary"]["status_counts"] == {"blocked_unsafe_target": 1}
