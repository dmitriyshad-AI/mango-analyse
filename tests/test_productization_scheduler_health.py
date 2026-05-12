from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.product_db import initialize_product_db, seed_job_types
from mango_mvp.productization.scheduler_health import build_scheduler_health_report
from scripts import mango_office_scheduler_health


def test_scheduler_health_reports_due_failed_locked_and_stale_jobs(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    _seed_scheduler_rows(product_db)

    report = build_scheduler_health_report(
        product_db_path=product_db,
        product_root=product_root,
        out_path=product_root / "scheduler_health" / "report.json",
        stale_after_minutes=30,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["due_jobs"] == 1
    assert report["summary"]["failed_jobs"] == 1
    assert report["summary"]["stale_running_jobs"] == 1
    assert report["summary"]["locked_jobs"] == 1
    assert report["safety"]["write_crm"] is False
    assert report["due_jobs"][0]["job_type"] == "shadow_poll"


def test_scheduler_health_cli_writes_clean_report_for_empty_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    out = product_root / "scheduler_health" / "report.json"
    initialize_product_db(product_db, product_root)

    rc = mango_office_scheduler_health.main(
        ["--product-root", str(product_root), "--product-db", str(product_db), "--out", str(out)]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["job_rows"] == 0
    assert saved["safety"]["product_db_writes"] is False


def test_scheduler_health_refuses_invalid_output_path(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)

    with pytest.raises(ValueError, match="scheduler health output"):
        build_scheduler_health_report(
            product_db_path=product_db,
            product_root=product_root,
            out_path=tmp_path / "outside.json",
        )


def _seed_scheduler_rows(product_db: Path) -> None:
    now = "2026-05-09T10:00:00+00:00"
    old = "2000-01-01T08:00:00+00:00"
    future = "2999-01-01T12:00:00+00:00"
    with sqlite3.connect(str(product_db)) as con:
        con.row_factory = sqlite3.Row
        seed_job_types(con, now)
        con.execute(
            """
            INSERT INTO tenants (tenant_id, display_name, status, created_at, updated_at)
            VALUES ('foton', 'Foton', 'active', ?, ?)
            """,
            (now, now),
        )
        con.execute(
            """
            INSERT INTO job_runs (
              job_type, tenant_id, status, planned_at, started_at, finished_at,
              input_ref, output_ref, error, scheduled_for, next_run_at,
              lock_owner, lock_expires_at, heartbeat_at, attempt_count, max_attempts
            ) VALUES
              ('shadow_poll', 'foton', 'planned', ?, NULL, NULL, '{}', NULL, NULL, ?, ?, NULL, NULL, NULL, 0, 3),
              ('shadow_poll', 'foton', 'failed', ?, ?, ?, '{}', NULL, 'sample failure', ?, NULL, NULL, NULL, NULL, 3, 3),
              ('shadow_poll', 'foton', 'running', ?, ?, NULL, '{}', NULL, NULL, ?, NULL, 'worker-1', ?, ?, 1, 3),
              ('shadow_poll', 'foton', 'planned', ?, NULL, NULL, '{}', NULL, NULL, ?, ?, NULL, NULL, NULL, 0, 3)
            """,
            (
                old,
                old,
                old,
                old,
                old,
                old,
                old,
                old,
                old,
                old,
                future,
                old,
                future,
                future,
                future,
            ),
        )
        con.commit()
