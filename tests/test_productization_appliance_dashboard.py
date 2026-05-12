from __future__ import annotations

import sqlite3
from pathlib import Path

from mango_mvp.productization.product_api import ProductApiFacade
from mango_mvp.productization.product_db import initialize_product_db


def test_appliance_dashboard_aggregates_product_db_capture_and_scheduler(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    _seed_dashboard_fixture(product_db)

    api = ProductApiFacade(product_root=product_root, product_db_path=product_db, workspace_root=Path.cwd())
    dashboard = api.appliance_dashboard(capture_limit=5, scheduler_limit=5)

    assert dashboard["schema_version"] == "appliance_dashboard_v1"
    assert dashboard["status"] == "ready"
    assert dashboard["summary"]["product_calls"] == 1
    assert dashboard["summary"]["capture_inbox_ready"] == 1
    assert dashboard["summary"]["job_runs"] == 1
    assert dashboard["summary"]["scheduler_failed_jobs"] == 0
    assert dashboard["summary"]["writeback_mode"] == "preview_only"
    assert dashboard["panels"]["capture"]["items"][0]["event_key"] == "foton:mango:CALL-1"
    assert dashboard["panels"]["scheduler"]["items"][0]["job_type"] == "shadow_poll"
    assert dashboard["panels"]["scheduler"]["health"]["summary"]["validation_ok"] is True
    assert dashboard["panels"]["lifecycle"]["mode"] == "waiting_for_lifecycle_report"
    assert dashboard["panels"]["demo_readiness"]["summary"]["required_panels"] == 9
    assert dashboard["panels"]["demo_readiness"]["summary"]["panels_present"] == 9
    assert dashboard["safety"]["write_crm"] is False
    assert dashboard["safety"]["run_asr"] is False


def test_appliance_dashboard_filters_capture_and_scheduler(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    _seed_dashboard_fixture(product_db)

    api = ProductApiFacade(product_root=product_root, product_db_path=product_db, workspace_root=Path.cwd())
    capture = api.capture_recent(limit=5, status="ready_for_capture", manager_ref="101", q="CALL-1")
    empty_capture = api.capture_recent(limit=5, status="blocked_no_recording")
    scheduler = api.scheduler_runs(limit=5, status="succeeded", job_type="shadow_poll")
    dashboard = api.appliance_dashboard(
        capture_limit=5,
        scheduler_limit=5,
        capture_status="ready_for_capture",
        manager_ref="101",
        q="REC-1",
        scheduler_status="succeeded",
        scheduler_job_type="shadow_poll",
    )

    assert len(capture["items"]) == 1
    assert empty_capture["items"] == []
    assert len(scheduler["items"]) == 1
    assert dashboard["filters"]["q"] == "REC-1"
    assert dashboard["panels"]["capture"]["total_visible"] == 1


def _seed_dashboard_fixture(product_db: Path) -> None:
    now = "2026-05-09T10:00:00+00:00"
    with sqlite3.connect(str(product_db)) as con:
        con.execute(
            """
            INSERT INTO tenants (tenant_id, display_name, status, created_at, updated_at)
            VALUES ('foton', 'Foton', 'active', ?, ?)
            ON CONFLICT(tenant_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (now, now),
        )
        con.execute(
            """
            INSERT INTO tenant_manager_owner_map (
              tenant_id, telephony_provider, manager_extension, mango_name,
              crm_provider, crm_owner_id, crm_owner_name, decision_status,
              match_status, created_at, updated_at
            ) VALUES (
              'foton', 'mango', '101', 'Иванов Иван',
              'amocrm', 77, 'Иванов Иван', 'confirmed_candidate',
              'manual_confirmed', ?, ?
            )
            """,
            (now, now),
        )
        con.execute(
            """
            INSERT INTO product_calls (
              tenant_id, telephony_provider, provider_call_id, event_key,
              recording_id, source_filename, started_at, duration_sec,
              manager_extension, manager_display_name, crm_owner_id,
              crm_owner_name, crm_match_status, raw_payload_ref,
              source_repository_ref, imported_at, updated_at
            ) VALUES (
              'foton', 'mango', 'CALL-1', 'foton:mango:CALL-1',
              'REC-1', 'CALL-1.mp3', ?, 180.0,
              '101', 'Иванов Иван', 77,
              'Иванов Иван', 'manual_confirmed', 'raw/payload.jsonl#CALL-1',
              'fixture', ?, ?
            )
            """,
            (now, now, now),
        )
        con.execute(
            """
            INSERT OR IGNORE INTO job_types (job_type, description, default_mode, created_at)
            VALUES ('shadow_poll', 'Shadow poll Mango calls', 'dry_run', ?)
            """,
            (now,),
        )
        con.execute(
            """
            INSERT INTO job_runs (
              job_type, tenant_id, status, planned_at, started_at, finished_at,
              input_ref, output_ref, scheduled_for, next_run_at, attempt_count, max_attempts
            ) VALUES (
              'shadow_poll', 'foton', 'succeeded', ?, ?, ?,
              'fixture-input', 'fixture-output', ?, NULL, 1, 3
            )
            """,
            (now, now, now, now),
        )
        con.execute(
            """
            INSERT INTO capture_inbox_items (
              tenant_id, provider, event_key, provider_call_id, status,
              source_report_ref, raw_payload_ref, started_at, ended_at,
              direction, client_phone, manager_ref, recording_ref,
              recording_url, audio_ref, decision_reason, candidate_json,
              event_json, first_seen_at, last_seen_at, enqueue_count
            ) VALUES (
              'foton', 'mango', 'foton:mango:CALL-1', 'CALL-1', 'ready_for_capture',
              'fixture-report', 'raw/payload.jsonl#CALL-1', ?, ?,
              'inbound', '+79990000000', '101', 'REC-1',
              NULL, 'REC-1', 'ready_for_shadow_capture', '{}',
              '{}', ?, ?, 1
            )
            """,
            (now, now, now, now),
        )
        con.commit()
