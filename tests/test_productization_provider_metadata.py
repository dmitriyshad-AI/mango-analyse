from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.provider_metadata import install_provider_metadata_sidecar
from mango_mvp.productization.test_ingest import run_quarantine_test_ingest
from tests.test_dialogue_format import make_settings


FIELDS = [
    "filename",
    "source_audio_path",
    "target_audio_path",
    "phone",
    "client_phone",
    "manager",
    "manager_name",
    "started_at",
    "start_time",
    "direction",
    "call_id",
    "record_id",
    "event_key",
    "provider_call_id",
    "recording_id",
    "duration_sec",
    "checksum_sha256",
    "source_size_bytes",
    "source",
    "tenant_id",
    "provider",
]


def write_metadata(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def make_row(audio_dir: Path, event_id: str, provider_call_id: str | None = None) -> dict:
    filename = f"2026-05-07__09-00-00__79990000000__mango_101_{event_id}.mp3"
    audio_path = audio_dir / filename
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-mp3")
    call_id = provider_call_id or event_id
    return {
        "filename": filename,
        "target_audio_path": str(audio_path),
        "phone": "+79990000000",
        "client_phone": "+79990000000",
        "manager": "101",
        "manager_name": "mango_101",
        "started_at": "2026-05-07T09:00:00+03:00",
        "start_time": "2026-05-07T09:00:00+03:00",
        "direction": "inbound",
        "call_id": call_id,
        "record_id": f"REC-{event_id}",
        "recording_id": f"REC-{event_id}",
        "event_key": f"foton:mango:{event_id}",
        "provider_call_id": call_id,
        "duration_sec": "10.0",
        "checksum_sha256": "a" * 64,
        "source_size_bytes": "8",
        "source": "mango_api_capture",
        "tenant_id": "foton",
        "provider": "mango",
    }


def build_disposable_db(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
    root = tmp_path / "quarantine"
    audio_dir = root / "audio"
    metadata_csv = root / "metadata.csv"
    write_metadata(metadata_csv, rows)
    out_root = root / "test_ingest"
    db_path = out_root / "test.sqlite"
    run_quarantine_test_ingest(
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        out_allowed_root=out_root,
        replace_existing=True,
        base_settings=make_settings(),
    )
    return db_path, metadata_csv, out_root


def test_provider_metadata_sidecar_installs_rows_and_audits(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_row(audio_dir, "CALL-1")
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, [row])

    report = install_provider_metadata_sidecar(
        db_path=db_path,
        metadata_csv_path=metadata_csv,
        out_allowed_root=out_root,
        replace_existing=True,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["inserted"] == 1
    assert report["summary"]["sidecar_rows"] == 1
    assert report["summary"]["warnings"] == 1
    assert report["audit"]["tenant_provider_counts"] == {"foton|mango": 1}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        sidecar = dict(con.execute("select * from provider_call_metadata").fetchone())
    assert sidecar["provider_call_id"] == "CALL-1"
    assert sidecar["recording_id"] == "REC-CALL-1"
    assert sidecar["event_key"] == "foton:mango:CALL-1"
    assert sidecar["manager_extension"] == "101"


def test_provider_metadata_sidecar_is_idempotent_update(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_row(audio_dir, "CALL-1")
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, [row])

    first = install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=True)
    second = install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=False)

    assert first["summary"]["inserted"] == 1
    assert second["summary"]["inserted"] == 0
    assert second["summary"]["updated"] == 1
    assert second["summary"]["sidecar_rows"] == 1
    assert second["summary"]["validation_ok"] is True


def test_provider_metadata_sidecar_preserves_existing_raw_payload_ref(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_row(audio_dir, "CALL-1")
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, [row])
    install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=True)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "update provider_call_metadata set raw_payload_ref = ? where source_filename = ?",
            ("raw_payload_archive/file.jsonl#entry=abc", row["filename"]),
        )
        con.commit()

    install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=False)

    with sqlite3.connect(db_path) as con:
        ref = con.execute("select raw_payload_ref from provider_call_metadata").fetchone()[0]
    assert ref == "raw_payload_archive/file.jsonl#entry=abc"


def test_provider_metadata_sidecar_blocks_duplicate_provider_call_key(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    rows = [
        make_row(audio_dir, "CALL-1", provider_call_id="DUP"),
        make_row(audio_dir, "CALL-2", provider_call_id="DUP"),
    ]
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, rows)

    report = install_provider_metadata_sidecar(db_path, metadata_csv, out_root, replace_existing=True)

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 1
    assert report["audit"]["blocked_reasons"]["duplicate_provider_call_keys"] == 1


def test_provider_metadata_sidecar_refuses_runtime_db_name(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "metadata.csv"
    write_metadata(metadata_csv, [])
    db_path = tmp_path / "mango_mvp.db"
    db_path.write_bytes(b"not-real-db")

    with pytest.raises(ValueError, match="runtime-looking DB"):
        install_provider_metadata_sidecar(
            db_path=db_path,
            metadata_csv_path=metadata_csv,
            out_allowed_root=tmp_path,
        )
