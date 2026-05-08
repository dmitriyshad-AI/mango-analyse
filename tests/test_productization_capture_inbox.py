from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.capture_inbox import (
    apply_shadow_poll_report_to_capture_inbox,
    audit_capture_inbox,
)
from scripts import mango_office_capture_inbox
from tests.test_productization_product_db import bootstrap_sample_product_db


def write_shadow_report(product_root: Path) -> tuple[Path, Path]:
    raw_payload = product_root / "raw_payload_archive" / "live" / "poll.jsonl"
    raw_payload.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = [
        {
            "schema_version": "mango_shadow_poll_raw_payload_v1",
            "event_key": None,
            "provider_call_id": "CALL-4",
            "recording_id": "rec-4",
            "raw_payload": {"entry_id": "CALL-4", "records": "[rec-4]"},
        },
        {
            "schema_version": "mango_shadow_poll_raw_payload_v1",
            "event_key": None,
            "provider_call_id": "CALL-5",
            "recording_id": "rec-5",
            "raw_payload": {"entry_id": "CALL-5", "records": "[rec-5]"},
        },
    ]
    raw_payload.write_text("\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8")
    report = product_root / "scheduler_outputs" / "shadow_poll_job_000009.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "mango_live_shadow_poll_v1",
                "mode": "live_read_only",
                "raw_payload_archive": {"path": str(raw_payload), "rows": 2},
                "decisions": [
                    enqueue_decision("CALL-4", "rec-4", manager_ref="101"),
                    enqueue_decision("CALL-5", "rec-5", manager_ref="102"),
                    skip_decision("CALL-6", "SKIP_NO_RECORDING"),
                    skip_decision("CALL-1", "SKIP_DUPLICATE"),
                ],
                "validation_ok": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report, raw_payload


def enqueue_decision(call_id: str, recording_ref: str, manager_ref: str) -> dict:
    return {
        "action": "enqueue_shadow_capture",
        "action_code": "ENQUEUE_SHADOW_CAPTURE",
        "reason": "ready_for_shadow_capture",
        "event": {
            "event_key": f"foton:mango:{call_id}",
            "provider_call_id": call_id,
            "started_at": "2026-05-07T06:00:00+00:00",
            "ended_at": "2026-05-07T06:05:00+00:00",
            "direction": "outbound",
            "client_phone": "+79990000000",
            "manager_ref": manager_ref,
            "recording_ref": recording_ref,
            "recording_url": None,
        },
        "candidate": {
            "event_key": f"foton:mango:{call_id}",
            "tenant_id": "foton",
            "provider": "mango",
            "provider_call_id": call_id,
            "started_at": "2026-05-07T06:00:00+00:00",
            "direction": "outbound",
            "audio_ref": recording_ref,
            "client_phone": "+79990000000",
            "manager_ref": manager_ref,
            "raw_payload": {"entry_id": call_id},
        },
    }


def skip_decision(call_id: str, action_code: str) -> dict:
    return {
        "action": action_code.lower(),
        "action_code": action_code,
        "reason": "not_ready",
        "event": {
            "event_key": f"foton:mango:{call_id}",
            "provider_call_id": call_id,
            "started_at": "2026-05-07T06:00:00+00:00",
            "direction": "outbound",
        },
        "candidate": None,
    }


def test_capture_inbox_applies_shadow_report_idempotently(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    report, _raw_payload = write_shadow_report(product_root)
    insert_job_run(product_db, 9)

    first = apply_shadow_poll_report_to_capture_inbox(
        product_db_path=product_db,
        product_root=product_root,
        report_path=report,
        out_path=product_root / "capture_inbox_apply.json",
    )
    second = apply_shadow_poll_report_to_capture_inbox(product_db, product_root, report)
    audit = audit_capture_inbox(product_db, product_root, out_path=product_root / "capture_inbox_audit.json")

    assert first["summary"]["decisions_seen"] == 4
    assert first["summary"]["enqueue_decisions"] == 2
    assert first["summary"]["inserted"] == 2
    assert first["summary"]["updated_existing"] == 0
    assert first["summary"]["already_present"] == 0
    assert first["summary"]["skipped_non_enqueue"] == 2
    assert second["summary"]["inserted"] == 0
    assert second["summary"]["updated_existing"] == 0
    assert second["summary"]["already_present"] == 2
    assert audit["summary"]["items"] == 2
    assert audit["summary"]["ready_for_capture"] == 2
    assert audit["summary"]["validation_ok"] is True
    assert audit["status_counts"] == {"ready_for_capture": 2}
    assert (product_root / "capture_inbox_apply.json").exists()
    assert (product_root / "capture_inbox_audit.json").exists()
    with sqlite3.connect(product_db) as con:
        rows = con.execute(
            "select event_key, source_job_run_id, raw_payload_ref, enqueue_count from capture_inbox_items order by event_key"
        ).fetchall()
    assert rows[0][0] == "foton:mango:CALL-4"
    assert rows[0][1] == 9
    assert "#line=1" in rows[0][2]
    assert rows[0][3] == 1


def test_capture_inbox_accepts_scheduler_tick_report_shape(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    report, _raw_payload = write_shadow_report(product_root)
    insert_job_run(product_db, 42)
    body = json.loads(report.read_text(encoding="utf-8"))
    tick_report = product_root / "scheduler_tick.json"
    tick_report.write_text(
        json.dumps({"results": [{"job_id": 42, "status": "succeeded", "result": body}]}),
        encoding="utf-8",
    )

    applied = apply_shadow_poll_report_to_capture_inbox(product_db, product_root, tick_report)

    assert applied["summary"]["source_reports"] == 1
    assert applied["summary"]["inserted"] == 2
    with sqlite3.connect(product_db) as con:
        source_ids = {row[0] for row in con.execute("select source_job_run_id from capture_inbox_items")}
    assert source_ids == {42}


def test_capture_inbox_refuses_report_outside_product_root(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="shadow poll report"):
        apply_shadow_poll_report_to_capture_inbox(product_db, product_root, outside)


def test_capture_inbox_cli_apply_and_audit(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    report, _raw_payload = write_shadow_report(product_root)

    apply_rc = mango_office_capture_inbox.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "apply-report",
            "--report",
            str(report),
            "--out",
            str(product_root / "apply_cli.json"),
        ]
    )
    audit_rc = mango_office_capture_inbox.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "audit",
            "--out",
            str(product_root / "audit_cli.json"),
        ]
    )

    assert apply_rc == 0
    assert audit_rc == 0
    assert json.loads((product_root / "audit_cli.json").read_text(encoding="utf-8"))["summary"]["items"] == 2


def insert_job_run(product_db: Path, job_id: int) -> None:
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            INSERT INTO job_runs (id, job_type, tenant_id, status, planned_at, input_ref)
            VALUES (?, 'shadow_poll', 'foton', 'succeeded', '2026-05-07T00:00:00+00:00', '{}')
            """,
            (job_id,),
        )
        con.commit()
