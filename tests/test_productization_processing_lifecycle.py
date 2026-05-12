from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.processing_lifecycle import (
    BLOCK_DUPLICATE_RECORDING_ID,
    CANDIDATE_ASR_HANDOFF_DRY_RUN,
    SKIP_ALREADY_IN_HANDOFF_MANIFEST,
    WAIT_RECORDING_ASSET,
    build_processing_lifecycle_report,
)
from mango_mvp.productization.product_db import initialize_product_db
from mango_mvp.productization.recording_asset_ingest import run_recording_asset_ingest
from scripts import mango_office_processing_lifecycle
from tests.test_productization_recording_asset_ingest import make_package


def test_processing_lifecycle_reports_asr_handoff_candidates(tmp_path: Path) -> None:
    product_root, product_db, asset_db = build_product_and_asset_db(tmp_path, count=1)
    insert_capture_item(product_db, event_key="foton:mango:CALL-1", provider_call_id="CALL-1", recording_ref="REC-1")

    report = build_processing_lifecycle_report(
        product_db_path=product_db,
        product_root=product_root,
        asset_db_path=asset_db,
        out_path=product_root / "processing_lifecycle_stage5" / "report.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["action_counts"] == {CANDIDATE_ASR_HANDOFF_DRY_RUN: 1}
    assert report["items"][0]["run_asr"] is False
    assert report["items"][0]["write_runtime_db"] is False
    assert report["safety"]["write_crm"] is False


def test_processing_lifecycle_detects_existing_handoff_manifest(tmp_path: Path) -> None:
    product_root, product_db, asset_db = build_product_and_asset_db(tmp_path, count=1)
    insert_capture_item(product_db, event_key="foton:mango:CALL-1", provider_call_id="CALL-1", recording_ref="REC-1")
    manifest = product_root / "processing_handoff_stage12" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"event_key": "foton:mango:CALL-1", "recording_id": "REC-1"}) + "\n", encoding="utf-8")

    report = build_processing_lifecycle_report(product_db, product_root, asset_db, manifest)

    assert report["summary"]["validation_ok"] is True
    assert report["action_counts"] == {SKIP_ALREADY_IN_HANDOFF_MANIFEST: 1}


def test_processing_lifecycle_blocks_duplicate_recording_id(tmp_path: Path) -> None:
    product_root, product_db, _asset_db = build_product_and_asset_db(tmp_path, count=1)
    insert_capture_item(product_db, event_key="foton:mango:CALL-1", provider_call_id="CALL-1", recording_ref="REC-1")
    insert_capture_item(product_db, event_key="foton:mango:CALL-2", provider_call_id="CALL-2", recording_ref="REC-1")

    report = build_processing_lifecycle_report(product_db, product_root)

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked_duplicate_recording_id"] == 2
    assert report["action_counts"] == {BLOCK_DUPLICATE_RECORDING_ID: 2}


def test_processing_lifecycle_cli_without_asset_db_waits_for_assets(tmp_path: Path) -> None:
    product_root, product_db, _asset_db = build_product_and_asset_db(tmp_path, count=1)
    insert_capture_item(product_db, event_key="foton:mango:CALL-1", provider_call_id="CALL-1", recording_ref="REC-1")
    out = product_root / "processing_lifecycle_stage5" / "cli.json"

    rc = mango_office_processing_lifecycle.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--out",
            str(out),
            "--no-asset-db",
            "--no-handoff-manifest",
        ]
    )

    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["action_counts"] == {WAIT_RECORDING_ASSET: 1}
    assert saved["safety"]["run_asr"] is False


def test_processing_lifecycle_refuses_runtime_db_names(tmp_path: Path) -> None:
    product_root, product_db, _asset_db = build_product_and_asset_db(tmp_path, count=1)

    with pytest.raises(ValueError, match="runtime-looking"):
        build_processing_lifecycle_report(product_db, product_root, product_root / "mango_mvp.db")


def build_product_and_asset_db(tmp_path: Path, count: int) -> tuple[Path, Path, Path]:
    product_root, package_root, audio_dir, metadata_csv = make_package(tmp_path, count=count)
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    asset_db = product_root / "recording_quarantine_stage11" / "recording_asset_ingest_stage11.sqlite"
    report = run_recording_asset_ingest(
        package_root=package_root,
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv,
        db_path=asset_db,
        product_root=product_root,
        package_ref="recording_quarantine_stage11",
    )
    assert report["summary"]["validation_ok"] is True
    return product_root, product_db, asset_db


def insert_capture_item(
    product_db: Path,
    *,
    event_key: str,
    provider_call_id: str,
    recording_ref: str,
) -> None:
    now = "2026-05-07T06:00:00+00:00"
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            INSERT INTO capture_inbox_items (
                tenant_id, provider, event_key, provider_call_id, status,
                started_at, direction, manager_ref, recording_ref, audio_ref,
                first_seen_at, last_seen_at, enqueue_count
            )
            VALUES ('foton', 'mango', ?, ?, 'ready_for_capture', ?, 'outbound', '101', ?, ?, ?, ?, 1)
            """,
            (event_key, provider_call_id, now, recording_ref, recording_ref, now, now),
        )
        con.commit()
