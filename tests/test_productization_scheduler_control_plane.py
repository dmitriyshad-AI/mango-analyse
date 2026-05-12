from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.product_db import initialize_product_db, seed_job_types
from mango_mvp.productization.scheduler_control_plane import build_scheduler_control_plane_report
from scripts import mango_office_scheduler_control_plane


def test_scheduler_control_plane_recommends_schedule_for_empty_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)

    report = build_scheduler_control_plane_report(product_db, product_root)

    assert report["summary"]["validation_ok"] is True
    assert report["recommended_actions"][0]["action"] == "SCHEDULE_SHADOW_POLL"
    assert "plan-live-shadow-poll" in report["recommended_actions"][0]["command"]
    assert report["safety"]["executes_jobs"] is False


def test_scheduler_control_plane_reports_due_and_failed_jobs(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    _seed_control_plane_rows(product_db)

    report = build_scheduler_control_plane_report(
        product_db,
        product_root,
        out_path=product_root / "scheduler_control_plane" / "report.json",
    )

    actions = {item["action"] for item in report["recommended_actions"]}
    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["due_jobs"] == 1
    assert report["summary"]["failed_jobs"] == 1
    assert "RUN_SCHEDULER_TICK" in actions
    assert "REVIEW_FAILED_JOBS" in actions


def test_scheduler_control_plane_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    out = product_root / "scheduler_control_plane" / "report.json"
    initialize_product_db(product_db, product_root)

    rc = mango_office_scheduler_control_plane.main(
        ["--product-root", str(product_root), "--product-db", str(product_db), "--out", str(out)]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["recommended_actions"] == 1
    assert saved["safety"]["run_asr"] is False


def test_scheduler_control_plane_refuses_output_outside_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)

    with pytest.raises(ValueError, match="scheduler control-plane output"):
        build_scheduler_control_plane_report(product_db, product_root, out_path=tmp_path / "outside.json")


def _seed_control_plane_rows(product_db: Path) -> None:
    now = "2026-05-09T10:00:00+00:00"
    old = "2000-01-01T08:00:00+00:00"
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
              attempt_count, max_attempts
            ) VALUES
              ('shadow_poll', 'foton', 'planned', ?, NULL, NULL, '{}', NULL, NULL, ?, ?, 0, 3),
              ('shadow_poll', 'foton', 'failed', ?, ?, ?, '{}', NULL, 'sample failure', ?, NULL, 3, 3)
            """,
            (old, old, old, old, old, old, old),
        )
        con.commit()
