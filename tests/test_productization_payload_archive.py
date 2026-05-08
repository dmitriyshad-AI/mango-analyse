from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.payload_archive import archive_mango_payloads_and_update_sidecar
from mango_mvp.productization.provider_metadata import install_provider_metadata_sidecar
from tests.test_productization_provider_metadata import build_disposable_db, make_row


def write_source_payload_jsonl(path: Path, event_id: str = "CALL-1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": "mango_shadow_poll_raw_payload_v1",
        "tenant_id": "foton",
        "provider": "mango",
        "provider_call_id": event_id,
        "recording_id": f"REC-{event_id}",
        "raw_payload": {
            "entry_id": event_id,
            "records": f"[REC-{event_id}]",
            "start": "1778133600",
            "finish": "1778133900",
            "from_number": "+79990000000",
            "to_extension": "101",
        },
    }
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


def test_payload_archive_writes_jsonl_and_updates_sidecar_ref(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_row(audio_dir, "CALL-1")
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, [row])
    install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=True)
    source_payload = tmp_path / "quarantine" / "raw_payload_archive" / "shadow_poll.jsonl"
    archive_root = tmp_path / "quarantine" / "raw_payload_archive" / "by_call"
    write_source_payload_jsonl(source_payload)

    report = archive_mango_payloads_and_update_sidecar(
        db_path=db_path,
        metadata_csv_path=metadata_csv,
        source_payload_path=source_payload,
        archive_root=archive_root,
        out_allowed_root=tmp_path / "quarantine",
        replace_existing=True,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["archived_entries"] == 1
    assert report["summary"]["sidecar_refs_updated"] == 1
    assert report["summary"]["sidecar_refs_present"] == 1
    archive_files = list(archive_root.rglob("*.jsonl"))
    assert len(archive_files) == 1
    archived = json.loads(archive_files[0].read_text(encoding="utf-8").strip())
    assert archived["raw_payload"]["entry_id"] == "CALL-1"
    with sqlite3.connect(db_path) as con:
        ref = con.execute("select raw_payload_ref from provider_call_metadata").fetchone()[0]
    assert ref.startswith("raw_payload_archive/by_call/")
    assert "#entry=" in ref


def test_payload_archive_is_idempotent(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_row(audio_dir, "CALL-1")
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, [row])
    install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=True)
    source_payload = tmp_path / "quarantine" / "raw_payload_archive" / "shadow_poll.jsonl"
    archive_root = tmp_path / "quarantine" / "raw_payload_archive" / "by_call"
    write_source_payload_jsonl(source_payload)

    first = archive_mango_payloads_and_update_sidecar(
        db_path, metadata_csv, source_payload, archive_root, tmp_path / "quarantine", replace_existing=True
    )
    second = archive_mango_payloads_and_update_sidecar(
        db_path, metadata_csv, source_payload, archive_root, tmp_path / "quarantine", replace_existing=False
    )

    assert first["summary"]["validation_ok"] is True
    assert second["summary"]["validation_ok"] is True
    assert second["summary"]["archived_entries"] == 1
    assert second["summary"]["sidecar_refs_updated"] == 1


def test_payload_archive_blocks_missing_source_payload(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_row(audio_dir, "CALL-1")
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, [row])
    install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=True)
    source_payload = tmp_path / "quarantine" / "raw_payload_archive" / "shadow_poll.jsonl"
    archive_root = tmp_path / "quarantine" / "raw_payload_archive" / "by_call"
    write_source_payload_jsonl(source_payload, event_id="OTHER")

    report = archive_mango_payloads_and_update_sidecar(
        db_path, metadata_csv, source_payload, archive_root, tmp_path / "quarantine", replace_existing=True
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 2
    assert report["audit"]["blocked_reasons"]["missing_source_payloads"] == 1
    assert report["audit"]["blocked_reasons"]["sidecar_refs_missing"] == 1
    assert report["audit"]["blocked_reasons"]["archive_file_row_mismatch"] == 0


def test_payload_archive_refuses_runtime_db_name(tmp_path: Path) -> None:
    db_path = tmp_path / "mango_mvp.db"
    db_path.write_bytes(b"not-real-db")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("filename\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime-looking DB"):
        archive_mango_payloads_and_update_sidecar(
            db_path=db_path,
            metadata_csv_path=metadata,
            source_payload_path=source,
            archive_root=tmp_path / "archive",
            out_allowed_root=tmp_path,
        )
