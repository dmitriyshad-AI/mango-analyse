from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.processing_handoff import build_processing_handoff_dry_run
from mango_mvp.productization.recording_asset_ingest import run_recording_asset_ingest
from scripts import mango_office_processing_handoff
from tests.test_productization_recording_asset_ingest import make_package


def test_processing_handoff_plans_ready_assets_and_manifest(tmp_path: Path) -> None:
    product_root, asset_db = build_asset_db(tmp_path, count=2)
    out_dir = product_root / "processing_handoff_stage12"
    manifest = out_dir / "asr_handoff_manifest.jsonl"

    report = build_processing_handoff_dry_run(
        asset_db_path=asset_db,
        product_root=product_root,
        out_dir=out_dir,
        manifest_path=manifest,
        out_path=out_dir / "audit.json",
        package_ref="recording_quarantine_stage9",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["ready_for_asr"] == 2
    assert report["summary"]["blocked"] == 0
    assert report["action_counts"] == {"PLAN_ASR_HANDOFF": 2}
    rows = read_jsonl(manifest)
    assert len(rows) == 2
    assert rows[0]["queue_status"] == "ready_for_asr"
    assert rows[0]["audio_path"].endswith(".mp3")
    assert "transcript_json" in rows[0]["planned_outputs"]
    assert report["safety"]["runtime_db_writes"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["write_crm"] is False


def test_processing_handoff_repeated_run_keeps_manifest_hash(tmp_path: Path) -> None:
    product_root, asset_db = build_asset_db(tmp_path, count=2)
    out_dir = product_root / "processing_handoff_stage12"
    manifest = out_dir / "asr_handoff_manifest.jsonl"

    first = build_processing_handoff_dry_run(
        asset_db_path=asset_db,
        product_root=product_root,
        out_dir=out_dir,
        manifest_path=manifest,
        out_path=out_dir / "audit.json",
    )
    second = build_processing_handoff_dry_run(
        asset_db_path=asset_db,
        product_root=product_root,
        out_dir=out_dir,
        manifest_path=manifest,
        out_path=out_dir / "audit_idempotency.json",
    )

    assert first["summary"]["manifest_sha256"] == second["summary"]["manifest_sha256"]
    assert second["summary"]["manifest_rows"] == 2


def test_processing_handoff_blocks_missing_audio(tmp_path: Path) -> None:
    product_root, asset_db = build_asset_db(tmp_path, count=1)
    audio_path = Path(
        sqlite_scalar(asset_db, "select audio_path from captured_recording_assets limit 1")
    )
    audio_path.unlink()

    report = build_processing_handoff_dry_run(
        asset_db_path=asset_db,
        product_root=product_root,
        out_dir=product_root / "processing_handoff_stage12",
        manifest_path=product_root / "processing_handoff_stage12" / "asr_handoff_manifest.jsonl",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["ready_for_asr"] == 0
    assert report["summary"]["blocked"] == 1
    assert report["action_counts"] == {"BLOCK_ASR_HANDOFF": 1}
    assert "audio_missing" in report["items"][0]["blocked_reasons"]


def test_processing_handoff_blocks_checksum_mismatch(tmp_path: Path) -> None:
    product_root, asset_db = build_asset_db(tmp_path, count=1)
    audio_path = Path(
        sqlite_scalar(asset_db, "select audio_path from captured_recording_assets limit 1")
    )
    audio_path.write_bytes(b"changed-audio")

    report = build_processing_handoff_dry_run(
        asset_db_path=asset_db,
        product_root=product_root,
        out_dir=product_root / "processing_handoff_stage12",
        manifest_path=product_root / "processing_handoff_stage12" / "asr_handoff_manifest.jsonl",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 1
    assert "checksum_sha256_mismatch" in report["items"][0]["blocked_reasons"]


def test_processing_handoff_blocks_duplicate_provider_call_ids(tmp_path: Path) -> None:
    product_root, asset_db = build_asset_db(tmp_path, count=2)
    with sqlite3.connect(asset_db) as con:
        con.execute("UPDATE captured_recording_assets SET provider_call_id = 'CALL-1' WHERE event_key = 'foton:mango:CALL-2'")
        con.commit()

    report = build_processing_handoff_dry_run(
        asset_db_path=asset_db,
        product_root=product_root,
        out_dir=product_root / "processing_handoff_stage12",
        manifest_path=product_root / "processing_handoff_stage12" / "asr_handoff_manifest.jsonl",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 2
    assert report["idempotency"]["provider_call_id_duplicate_groups"] == 1
    assert report["action_counts"] == {"BLOCK_ASR_HANDOFF": 2}
    assert all("duplicate_provider_call_id" in item["blocked_reasons"] for item in report["items"])


def test_processing_handoff_refuses_runtime_and_outside_paths(tmp_path: Path) -> None:
    product_root, asset_db = build_asset_db(tmp_path, count=1)

    with pytest.raises(ValueError, match="runtime-looking"):
        build_processing_handoff_dry_run(
            asset_db_path=product_root / "mango_mvp.db",
            product_root=product_root,
            out_dir=product_root / "processing_handoff_stage12",
            manifest_path=product_root / "processing_handoff_stage12" / "asr_handoff_manifest.jsonl",
        )
    with pytest.raises(ValueError, match="product root"):
        build_processing_handoff_dry_run(
            asset_db_path=asset_db,
            product_root=product_root,
            out_dir=tmp_path / "outside",
            manifest_path=product_root / "processing_handoff_stage12" / "asr_handoff_manifest.jsonl",
        )


def test_processing_handoff_script_writes_report(tmp_path: Path) -> None:
    product_root, asset_db = build_asset_db(tmp_path, count=1)
    out_dir = product_root / "processing_handoff_stage12"
    out = out_dir / "audit.json"
    manifest = out_dir / "asr_handoff_manifest.jsonl"

    rc = mango_office_processing_handoff.main(
        [
            "--product-root",
            str(product_root),
            "--asset-db",
            str(asset_db),
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--package-ref",
            "recording_quarantine_stage9",
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["ready_for_asr"] == 1
    assert data["summary"]["manifest_rows"] == 1
    assert manifest.exists()


def build_asset_db(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, package_root, audio_dir, metadata_csv = make_package(tmp_path, count=count)
    asset_db = product_root / "recording_quarantine_stage10" / "recording_asset_ingest.sqlite"
    report = run_recording_asset_ingest(
        package_root=package_root,
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=asset_db,
        product_root=product_root,
        package_ref="recording_quarantine_stage9",
    )
    assert report["summary"]["validation_ok"] is True
    return product_root, asset_db


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sqlite_scalar(db_path: Path, sql: str) -> str:
    with sqlite3.connect(db_path) as con:
        row = con.execute(sql).fetchone()
    return str(row[0])
