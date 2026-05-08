from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.recording_asset_ingest import run_recording_asset_ingest
from scripts import mango_office_recording_asset_ingest


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


def test_recording_asset_ingest_imports_and_skips_idempotently(tmp_path: Path) -> None:
    product_root, package_root, audio_dir, metadata_csv = make_package(tmp_path, count=2)
    db_path = product_root / "recording_quarantine_stage10" / "recording_asset_ingest.sqlite"

    first = run_recording_asset_ingest(
        package_root=package_root,
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        product_root=product_root,
    )
    second = run_recording_asset_ingest(
        package_root=package_root,
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=db_path,
        product_root=product_root,
        allow_existing_db=True,
    )

    assert first["summary"]["validation_ok"] is True
    assert first["summary"]["inserted"] == 2
    assert first["summary"]["db_assets_for_package"] == 2
    assert first["action_counts"] == {"INGEST_RECORDING_ASSET": 2}
    assert second["summary"]["validation_ok"] is True
    assert second["summary"]["already_present"] == 2
    assert second["action_counts"] == {"SKIP_ALREADY_INGESTED": 2}
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT tenant_id, provider, event_key, recording_id, status
              FROM captured_recording_assets
             ORDER BY id
            """
        ).fetchall()
    assert rows == [
        ("foton", "mango", "foton:mango:CALL-1", "REC-1", "quarantined_ready"),
        ("foton", "mango", "foton:mango:CALL-2", "REC-2", "quarantined_ready"),
    ]


def test_recording_asset_ingest_blocks_checksum_mismatch(tmp_path: Path) -> None:
    product_root, package_root, audio_dir, metadata_csv = make_package(tmp_path, count=1, checksum="bad")

    report = run_recording_asset_ingest(
        package_root=package_root,
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=product_root / "recording_quarantine_stage10" / "recording_asset_ingest.sqlite",
        product_root=product_root,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 1
    assert report["action_counts"] == {"BLOCK_RECORDING_ASSET_INGEST": 1}
    assert "checksum_sha256_mismatch" in report["items"][0]["blocked_reasons"]
    assert report["summary"]["db_assets_for_package"] == 0


def test_recording_asset_ingest_blocks_duplicate_metadata_keys(tmp_path: Path) -> None:
    product_root, package_root, audio_dir, metadata_csv = make_package(tmp_path, count=2, duplicate_event=True)

    report = run_recording_asset_ingest(
        package_root=package_root,
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=product_root / "recording_quarantine_stage10" / "recording_asset_ingest.sqlite",
        product_root=product_root,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 2
    assert {item["action"] for item in report["items"]} == {"BLOCK_RECORDING_ASSET_INGEST"}
    assert all("duplicate_event_key_in_metadata" in item["blocked_reasons"] for item in report["items"])


def test_recording_asset_ingest_refuses_runtime_db_and_outside_paths(tmp_path: Path) -> None:
    product_root, package_root, audio_dir, metadata_csv = make_package(tmp_path, count=1)

    with pytest.raises(ValueError, match="runtime-looking"):
        run_recording_asset_ingest(
            package_root=package_root,
            audio_dir=audio_dir,
            metadata_csv_path=metadata_csv,
            db_path=product_root / "mango_mvp.db",
            product_root=product_root,
        )

    with pytest.raises(ValueError, match="product root"):
        run_recording_asset_ingest(
            package_root=package_root,
            audio_dir=audio_dir,
            metadata_csv_path=metadata_csv,
            db_path=tmp_path / "outside.sqlite",
            product_root=product_root,
        )


def test_recording_asset_ingest_script_writes_audit(tmp_path: Path) -> None:
    product_root, package_root, audio_dir, metadata_csv = make_package(tmp_path, count=1)
    out = product_root / "recording_quarantine_stage10" / "audit.json"
    db_path = product_root / "recording_quarantine_stage10" / "recording_asset_ingest.sqlite"

    rc = mango_office_recording_asset_ingest.main(
        [
            "--product-root",
            str(product_root),
            "--package-root",
            str(package_root),
            "--audio-dir",
            str(audio_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--db",
            str(db_path),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["inserted"] == 1
    assert data["safety"]["runtime_db_writes"] is False
    assert data["safety"]["run_asr"] is False


def make_package(
    tmp_path: Path,
    count: int,
    checksum: str | None = None,
    duplicate_event: bool = False,
) -> tuple[Path, Path, Path, Path]:
    product_root = tmp_path / "product_appliance"
    package_root = product_root / "recording_quarantine_stage9"
    audio_dir = package_root / "audio"
    audio_dir.mkdir(parents=True)
    rows = []
    for index in range(1, count + 1):
        payload = f"fake-mp3-{index}".encode()
        filename = f"2026-05-07__17-0{index}-00__7999000000{index}__mango_10_CALL-{index}.mp3"
        audio_path = audio_dir / filename
        audio_path.write_bytes(payload)
        digest = checksum if checksum is not None else hashlib.sha256(payload).hexdigest()
        event_key = "foton:mango:CALL-1" if duplicate_event else f"foton:mango:CALL-{index}"
        rows.append(
            {
                "filename": filename,
                "source_audio_path": str(product_root / "recording_capture_downloads" / filename),
                "target_audio_path": str(audio_path),
                "phone": f"7999000000{index}",
                "client_phone": f"7999000000{index}",
                "manager": "10",
                "manager_name": "mango_10",
                "started_at": f"2026-05-07T17:0{index}:00+03:00",
                "start_time": f"2026-05-07T17:0{index}:00+03:00",
                "direction": "outbound",
                "call_id": f"CALL-{index}",
                "record_id": f"REC-{index}",
                "event_key": event_key,
                "provider_call_id": f"CALL-{index}",
                "recording_id": f"REC-{index}",
                "duration_sec": "1.5",
                "checksum_sha256": digest,
                "source_size_bytes": str(len(payload)),
                "source": "mango_api_capture",
                "tenant_id": "foton",
                "provider": "mango",
            }
        )
    metadata_csv = package_root / "metadata.csv"
    with metadata_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return product_root, package_root, audio_dir, metadata_csv
