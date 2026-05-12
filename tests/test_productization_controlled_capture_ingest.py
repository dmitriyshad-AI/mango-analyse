from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.productization.controlled_capture_ingest import (
    INGEST_ENQUEUE_CAPTURE,
    SKIP_DUPLICATE_PRODUCT_CALL,
    SKIP_NO_RECORDING,
    WAIT_DELAYED_RECORDING,
    build_controlled_capture_ingest_report,
)
from scripts import mango_office_controlled_capture_ingest
from tests.test_productization_product_db import bootstrap_sample_product_db


def test_controlled_capture_ingest_classifies_shadow_poll_without_writes(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    report_path = write_shadow_report(product_root)

    report = build_controlled_capture_ingest_report(
        product_db_path=product_db,
        product_root=product_root,
        report_path=report_path,
        out_path=product_root / "controlled_capture_ingest_stage4" / "plan.json",
        now=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )

    assert report["summary"]["apply"] is False
    assert report["summary"]["decisions_seen"] == 4
    assert report["action_counts"] == {
        INGEST_ENQUEUE_CAPTURE: 1,
        SKIP_DUPLICATE_PRODUCT_CALL: 1,
        SKIP_NO_RECORDING: 1,
        WAIT_DELAYED_RECORDING: 1,
    }
    assert report["safety"]["product_db_writes"] is False
    assert report["safety"]["downloads_audio"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["write_crm"] is False
    with sqlite3.connect(product_db) as con:
        assert con.execute("SELECT count(*) FROM capture_inbox_items").fetchone()[0] == 0


def test_controlled_capture_ingest_apply_writes_only_enqueue_rows_idempotently(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    report_path = write_shadow_report(product_root)

    first = build_controlled_capture_ingest_report(product_db, product_root, report_path, apply=True)
    second = build_controlled_capture_ingest_report(product_db, product_root, report_path, apply=True)

    assert first["summary"]["apply"] is True
    assert first["apply_result"]["summary"]["inserted"] == 1
    assert second["apply_result"]["summary"]["inserted"] == 0
    assert second["summary"]["skip_duplicate_capture_inbox"] == 1
    with sqlite3.connect(product_db) as con:
        rows = con.execute("SELECT event_key, recording_ref FROM capture_inbox_items").fetchall()
    assert rows == [("foton:mango:CALL-4", "rec-4")]


def test_controlled_capture_ingest_cli_plan_writes_report(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    report_path = write_shadow_report(product_root)
    out = product_root / "controlled_capture_ingest_stage4" / "cli.json"

    rc = mango_office_controlled_capture_ingest.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--report",
            str(report_path),
            "--out",
            str(out),
            "plan",
        ]
    )

    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["ingest_enqueue_capture"] == 1
    assert saved["safety"]["product_db_writes"] is False


def test_controlled_capture_ingest_refuses_report_outside_product_root(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="shadow poll report"):
        build_controlled_capture_ingest_report(product_db, product_root, outside)


def write_shadow_report(product_root: Path) -> Path:
    report = product_root / "shadow_poll" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "mango_live_shadow_poll_v1",
                "decisions": [
                    enqueue_decision("CALL-4", "rec-4", "2026-05-09T10:00:00+00:00"),
                    enqueue_decision("CALL-1", "rec-1", "2026-05-09T10:00:00+00:00"),
                    no_recording_decision("CALL-5", "2026-05-09T11:30:00+00:00"),
                    no_recording_decision("CALL-6", "2026-05-07T09:00:00+00:00"),
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def enqueue_decision(call_id: str, recording_ref: str, started_at: str) -> dict:
    return {
        "action": "enqueue_shadow_capture",
        "action_code": "ENQUEUE_SHADOW_CAPTURE",
        "reason": "ready_for_shadow_capture",
        "event": {
            "event_key": f"foton:mango:{call_id}",
            "provider_call_id": call_id,
            "started_at": started_at,
            "direction": "outbound",
            "manager_ref": "101",
            "recording_ref": recording_ref,
        },
        "candidate": {
            "event_key": f"foton:mango:{call_id}",
            "tenant_id": "foton",
            "provider": "mango",
            "provider_call_id": call_id,
            "started_at": started_at,
            "audio_ref": recording_ref,
            "manager_ref": "101",
        },
    }


def no_recording_decision(call_id: str, started_at: str) -> dict:
    return {
        "action": "skip_no_recording",
        "action_code": "SKIP_NO_RECORDING",
        "reason": "records_empty",
        "event": {
            "event_key": f"foton:mango:{call_id}",
            "provider_call_id": call_id,
            "started_at": started_at,
            "direction": "outbound",
            "manager_ref": "101",
        },
        "candidate": None,
    }
