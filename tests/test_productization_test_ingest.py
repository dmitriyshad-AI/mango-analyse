from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mango_mvp.productization.test_ingest import run_quarantine_test_ingest
from tests.test_dialogue_format import make_settings


METADATA_FIELDS = [
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
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})


def make_package(tmp_path: Path) -> tuple[Path, Path, str]:
    audio_dir = tmp_path / "quarantine" / "audio"
    audio_dir.mkdir(parents=True)
    filename = "2026-05-07__09-00-00__79990000000__mango_101_CALL-1.mp3"
    audio_path = audio_dir / filename
    audio_path.write_bytes(b"fake-mp3")
    metadata_csv = tmp_path / "quarantine" / "metadata.csv"
    write_metadata(
        metadata_csv,
        [
            {
                "filename": filename,
                "target_audio_path": str(audio_path),
                "phone": "+79990000000",
                "client_phone": "+79990000000",
                "manager": "101",
                "manager_name": "mango_101",
                "started_at": "2026-05-07T09:00:00+03:00",
                "start_time": "2026-05-07T09:00:00+03:00",
                "direction": "inbound",
                "call_id": "CALL-1",
                "record_id": "REC-1",
                "recording_id": "REC-1",
                "event_key": "foton:mango:CALL-1",
                "provider": "mango",
                "tenant_id": "foton",
            }
        ],
    )
    return audio_dir, metadata_csv, filename


def test_quarantine_test_ingest_writes_disposable_db_and_audit(tmp_path: Path) -> None:
    audio_dir, metadata_csv, filename = make_package(tmp_path)
    out_root = tmp_path / "quarantine" / "test_ingest"
    db_path = out_root / "test.sqlite"

    report = run_quarantine_test_ingest(
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        out_allowed_root=out_root,
        replace_existing=False,
        base_settings=make_settings(),
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["ingest_inserted"] == 1
    assert report["summary"]["db_call_records"] == 1
    assert report["audit"]["blocked"] == 0
    assert report["audit"]["status_counts"]["transcription_status"] == {"pending": 1}
    assert report["audit"]["status_counts"]["resolve_status"] == {"pending": 1}
    assert report["audit"]["status_counts"]["analysis_status"] == {"pending": 1}
    assert report["audit"]["samples"]["db_rows"][0]["source_filename"] == filename
    assert report["audit"]["samples"]["db_rows"][0]["source_call_id"] == "CALL-1"


def test_quarantine_test_ingest_refuses_existing_db_without_replace(tmp_path: Path) -> None:
    audio_dir, metadata_csv, _filename = make_package(tmp_path)
    out_root = tmp_path / "quarantine" / "test_ingest"
    db_path = out_root / "test.sqlite"
    out_root.mkdir(parents=True)
    db_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        run_quarantine_test_ingest(
            audio_dir=audio_dir,
            metadata_csv_path=metadata_csv,
            db_path=db_path,
            out_allowed_root=out_root,
            base_settings=make_settings(),
        )


def test_quarantine_test_ingest_replace_is_idempotent_for_disposable_db(tmp_path: Path) -> None:
    audio_dir, metadata_csv, _filename = make_package(tmp_path)
    out_root = tmp_path / "quarantine" / "test_ingest"
    db_path = out_root / "test.sqlite"

    first = run_quarantine_test_ingest(
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        out_allowed_root=out_root,
        replace_existing=True,
        base_settings=make_settings(),
    )
    second = run_quarantine_test_ingest(
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        out_allowed_root=out_root,
        replace_existing=True,
        base_settings=make_settings(),
    )

    assert first["summary"]["ingest_inserted"] == 1
    assert second["summary"]["replaced_existing_db"] is True
    assert second["summary"]["ingest_inserted"] == 1
    assert second["summary"]["validation_ok"] is True


def test_quarantine_test_ingest_allow_existing_skips_existing_rows(tmp_path: Path) -> None:
    audio_dir, metadata_csv, _filename = make_package(tmp_path)
    out_root = tmp_path / "quarantine" / "test_ingest"
    db_path = out_root / "test.sqlite"

    first = run_quarantine_test_ingest(
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        out_allowed_root=out_root,
        replace_existing=True,
        base_settings=make_settings(),
    )
    second = run_quarantine_test_ingest(
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        out_allowed_root=out_root,
        allow_existing=True,
        base_settings=make_settings(),
    )

    assert first["summary"]["ingest_inserted"] == 1
    assert second["summary"]["ingest_inserted"] == 0
    assert second["summary"]["ingest_skipped"] == 1
    assert second["summary"]["db_call_records"] == 1
    assert second["summary"]["validation_ok"] is True


def test_quarantine_test_ingest_refuses_runtime_db_name(tmp_path: Path) -> None:
    audio_dir, metadata_csv, _filename = make_package(tmp_path)
    out_root = tmp_path / "quarantine" / "test_ingest"

    with pytest.raises(ValueError, match="runtime-looking DB"):
        run_quarantine_test_ingest(
            audio_dir=audio_dir,
            metadata_csv_path=metadata_csv,
            db_path=out_root / "mango_mvp.db",
            out_allowed_root=out_root,
            base_settings=make_settings(),
        )


def test_quarantine_test_ingest_refuses_db_outside_allowed_root(tmp_path: Path) -> None:
    audio_dir, metadata_csv, _filename = make_package(tmp_path)

    with pytest.raises(ValueError, match="allowed root"):
        run_quarantine_test_ingest(
            audio_dir=audio_dir,
            metadata_csv_path=metadata_csv,
            db_path=tmp_path / "outside.sqlite",
            out_allowed_root=tmp_path / "quarantine" / "test_ingest",
            base_settings=make_settings(),
        )
