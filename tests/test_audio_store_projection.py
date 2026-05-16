from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from argparse import Namespace
from pathlib import Path

from mango_mvp.audio_store import AudioStoreIndex
from scripts.build_audio_store_downstream_projection import build_projection


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_audio_store_index_resolves_current_records(tmp_path: Path) -> None:
    audio = tmp_path / "store" / "audio" / "aa" / "a.mp3"
    audio.parent.mkdir(parents=True)
    data = b"audio-bytes"
    audio.write_bytes(data)
    digest = sha256_bytes(data)
    mapping = tmp_path / "store" / "audio_store_mapping.csv"
    write_csv(
        mapping,
        [
            {
                "record_type": "canonical_call",
                "record_id": "42",
                "source_set": "current_canonical_master",
                "source_audio_path": "old/call.mp3",
                "source_audio_basename": "call.mp3",
                "canonical_audio_path": "store/audio/aa/a.mp3",
                "sha256": digest,
                "size_bytes": len(data),
                "ext": ".mp3",
                "queue_item_id": "",
                "event_key": "",
                "provider_call_id": "",
            },
            {
                "record_type": "new_mango_queue",
                "record_id": "mango_new_00001",
                "source_set": "new_mango_asr_handoff_queue",
                "source_audio_path": "old/mango.mp3",
                "source_audio_basename": "mango.mp3",
                "canonical_audio_path": "store/audio/aa/a.mp3",
                "sha256": digest,
                "size_bytes": len(data),
                "ext": ".mp3",
                "queue_item_id": "mango_new_00001",
                "event_key": "foton:mango:1",
                "provider_call_id": "1",
            },
        ],
        [
            "record_type",
            "record_id",
            "source_set",
            "source_audio_path",
            "source_audio_basename",
            "canonical_audio_path",
            "sha256",
            "size_bytes",
            "ext",
            "queue_item_id",
            "event_key",
            "provider_call_id",
        ],
    )

    index = AudioStoreIndex(mapping, project_root=tmp_path)

    assert index.resolve(record_type="canonical_call", record_id="42") is not None
    by_event = index.resolve(event_key="foton:mango:1")
    assert by_event is not None
    assert index.canonical_path(by_event) == audio
    assert index.unresolved_canonical_paths() == []


def test_downstream_projection_builds_canonical_and_asr_handoff(tmp_path: Path) -> None:
    old_audio = tmp_path / "old" / "call.mp3"
    old_audio.parent.mkdir(parents=True)
    old_audio.write_bytes(b"source-audio")
    store_audio = tmp_path / "store" / "audio" / "11" / "audio.mp3"
    store_audio.parent.mkdir(parents=True)
    store_audio.write_bytes(b"source-audio")
    digest = sha256_bytes(b"source-audio")

    db_path = tmp_path / "canonical.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        create table canonical_calls (
            canonical_call_id integer,
            source_filename text,
            source_file text,
            started_at text,
            phone text,
            manager_name text,
            duration_sec real,
            is_actionable integer,
            canonical_status text
        )
        """
    )
    con.execute(
        "insert into canonical_calls values (1, 'call.mp3', 'old/call.mp3', '2026-05-01 10:00:00', '+79990000000', 'Manager', 12.0, 1, 'actionable')"
    )
    con.commit()
    con.close()

    mapping = tmp_path / "store" / "audio_store_mapping.csv"
    fields = [
        "record_type",
        "record_id",
        "source_set",
        "source_audio_path",
        "source_audio_basename",
        "canonical_audio_path",
        "sha256",
        "size_bytes",
        "ext",
        "queue_item_id",
        "event_key",
        "provider_call_id",
    ]
    write_csv(
        mapping,
        [
            {
                "record_type": "canonical_call",
                "record_id": "1",
                "source_set": "current_canonical_master",
                "source_audio_path": "old/call.mp3",
                "source_audio_basename": "call.mp3",
                "canonical_audio_path": "store/audio/11/audio.mp3",
                "sha256": digest,
                "size_bytes": len(b"source-audio"),
                "ext": ".mp3",
                "queue_item_id": "",
                "event_key": "",
                "provider_call_id": "",
            },
            {
                "record_type": "new_mango_queue",
                "record_id": "mango_new_00001",
                "source_set": "new_mango_asr_handoff_queue",
                "source_audio_path": "old/call.mp3",
                "source_audio_basename": "call.mp3",
                "canonical_audio_path": "store/audio/11/audio.mp3",
                "sha256": digest,
                "size_bytes": len(b"source-audio"),
                "ext": ".mp3",
                "queue_item_id": "mango_new_00001",
                "event_key": "foton:mango:1",
                "provider_call_id": "1",
            },
        ],
        fields,
    )
    queue = tmp_path / "queue.csv"
    write_csv(
        queue,
        [
            {
                "queue_item_id": "mango_new_00001",
                "event_key": "foton:mango:1",
                "provider_call_id": "1",
                "recording_id": "rec-1",
                "recording_ref": "rec-1",
                "started_at_utc": "2026-05-01T07:00:00+00:00",
                "client_phone": "+79990000000",
                "manager_ref": "23",
                "audio_path": "old/call.mp3",
                "audio_size_bytes": len(b"source-audio"),
                "audio_sha256": digest,
                "source_manifest": "manifest.jsonl",
            }
        ],
        [
            "queue_item_id",
            "event_key",
            "provider_call_id",
            "recording_id",
            "recording_ref",
            "started_at_utc",
            "client_phone",
            "manager_ref",
            "audio_path",
            "audio_size_bytes",
            "audio_sha256",
            "source_manifest",
        ],
    )
    out_dir = tmp_path / "projection"

    report = build_projection(
        Namespace(
            project_root=str(tmp_path),
            audio_store_mapping=str(mapping),
            canonical_db=str(db_path),
            new_queue_csv=str(queue),
            out_dir=str(out_dir),
            verify_checksum=True,
        )
    )

    assert report["validation_ok"] is True
    assert report["canonical_rows_projected"] == 1
    assert report["new_queue_handoff_rows"] == 1
    handoff = [json.loads(line) for line in (out_dir / "new_mango_processing_handoff_audio_store.jsonl").read_text(encoding="utf-8").splitlines()]
    assert handoff[0]["queue_status"] == "ready_for_asr"
    assert handoff[0]["audio_path"] == "store/audio/11/audio.mp3"
    assert handoff[0]["checksum_sha256"] == digest
