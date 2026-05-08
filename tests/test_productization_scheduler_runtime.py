from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.scheduler_runtime import (
    audit_scheduler_runtime,
    run_scheduler_tick,
    schedule_live_shadow_poll_job,
    schedule_shadow_poll_job,
)
from scripts import mango_office_scheduler_runtime
from tests.test_productization_product_db import bootstrap_sample_product_db


def write_raw_payload(product_root: Path) -> Path:
    raw = product_root / "raw_payload_archive" / "shadow.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('{"entry_id":"CALL-1"}\n{"entry_id":"CALL-2"}\n', encoding="utf-8")
    return raw


def test_scheduler_plans_ticks_and_audits_shadow_poll(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw = write_raw_payload(product_root)

    plan = schedule_shadow_poll_job(
        product_db_path=product_db,
        product_root=product_root,
        tenant_id="foton",
        raw_payload_path=raw,
        output_dir=product_root / "custom_scheduler_outputs",
        out_path=product_root / "plan.json",
    )
    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker", out_path=product_root / "tick.json")
    audit = audit_scheduler_runtime(product_db, product_root, out_path=product_root / "scheduler_audit.json")

    assert plan["summary"]["planned"] == 1
    assert tick["summary"]["succeeded"] == 1
    assert tick["results"][0]["result"]["raw_payload_stats"]["rows"] == 2
    assert Path(tick["results"][0]["output_ref"]).exists()
    assert Path(tick["results"][0]["output_ref"]).parent == product_root / "custom_scheduler_outputs"
    assert audit["summary"]["status_counts"] == {"shadow_poll|succeeded": 1}
    assert (product_root / "plan.json").exists()
    assert (product_root / "tick.json").exists()
    assert (product_root / "scheduler_audit.json").exists()


def test_scheduler_retries_failed_shadow_poll(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw = write_raw_payload(product_root)
    schedule_shadow_poll_job(product_db, product_root, "foton", raw, product_root / "scheduler_outputs", max_attempts=2)

    def failing_executor(_payload: dict) -> dict:
        raise RuntimeError("temporary provider error")

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker", executor=failing_executor)

    assert tick["summary"]["retry_wait"] == 1
    with sqlite3.connect(product_db) as con:
        row = con.execute("select status, attempt_count, error, next_run_at from job_runs").fetchone()
    assert row[0] == "retry_wait"
    assert row[1] == 1
    assert "temporary provider error" in row[2]
    assert row[3]


def test_scheduler_retries_executor_validation_failure(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    missing_raw = product_root / "raw_payload_archive" / "missing.jsonl"
    schedule_shadow_poll_job(
        product_db,
        product_root,
        "foton",
        missing_raw,
        product_root / "scheduler_outputs",
        max_attempts=2,
    )

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker")

    assert tick["summary"]["retry_wait"] == 1
    with sqlite3.connect(product_db) as con:
        row = con.execute("select status, attempt_count, error from job_runs").fetchone()
    assert row == ("retry_wait", 1, "executor_validation_failed")


def test_scheduler_blocks_disabled_job_types(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            INSERT INTO job_runs (job_type, tenant_id, status, planned_at, input_ref, next_run_at)
            VALUES ('asr', 'foton', 'planned', '2000-01-01T00:00:00+00:00', '{}', '2000-01-01T00:00:00+00:00')
            """
        )
        con.commit()

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker")

    assert tick["summary"]["blocked"] == 1
    with sqlite3.connect(product_db) as con:
        row = con.execute("select status, error from job_runs").fetchone()
    assert row[0] == "blocked"
    assert "job_type_disabled_for_stage3" in row[1]


def test_scheduler_blocks_unsafe_shadow_poll_input_from_db(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    outside = tmp_path / "outside.jsonl"
    payload = {
        "mode": "dry_run",
        "tenant_id": "foton",
        "provider": "mango",
        "raw_payload_path": str(outside),
        "output_dir": str(product_root / "scheduler_outputs"),
        "hard_guards": {
            "download_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_runtime_db": False,
        },
    }
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            INSERT INTO job_runs (job_type, tenant_id, status, planned_at, input_ref, next_run_at)
            VALUES ('shadow_poll', 'foton', 'planned', '2000-01-01T00:00:00+00:00', ?, '2000-01-01T00:00:00+00:00')
            """,
            (json.dumps(payload),),
        )
        con.commit()

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker")

    assert tick["summary"]["blocked"] == 1
    with sqlite3.connect(product_db) as con:
        row = con.execute("select status, error from job_runs").fetchone()
    assert row[0] == "blocked"
    assert "raw payload must stay under product root" in row[1]


def test_scheduler_skips_unexpired_locks(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw = write_raw_payload(product_root)
    plan = schedule_shadow_poll_job(product_db, product_root, "foton", raw, product_root / "scheduler_outputs")
    job_id = plan["summary"]["job_id"]
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            UPDATE job_runs
               SET lock_owner = 'other-worker',
                   lock_expires_at = '2999-01-01T00:00:00+00:00'
             WHERE id = ?
            """,
            (job_id,),
        )
        con.commit()

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker")

    assert tick["summary"]["claimed"] == 0
    with sqlite3.connect(product_db) as con:
        row = con.execute("select status, lock_owner from job_runs where id = ?", (job_id,)).fetchone()
    assert row == ("planned", "other-worker")


def test_scheduler_requeues_expired_running_lock(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw = write_raw_payload(product_root)
    plan = schedule_shadow_poll_job(product_db, product_root, "foton", raw, product_root / "scheduler_outputs")
    job_id = plan["summary"]["job_id"]
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            UPDATE job_runs
               SET status = 'running',
                   attempt_count = 1,
                   max_attempts = 3,
                   lock_owner = 'dead-worker',
                   lock_expires_at = '2000-01-01T00:00:00+00:00'
             WHERE id = ?
            """,
            (job_id,),
        )
        con.commit()

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker")

    assert tick["maintenance"]["requeued_expired_locks"] == 1
    assert tick["summary"]["succeeded"] == 1
    with sqlite3.connect(product_db) as con:
        row = con.execute("select status, attempt_count, lock_owner from job_runs where id = ?", (job_id,)).fetchone()
    assert row == ("succeeded", 2, None)


def test_scheduler_fails_expired_running_lock_after_max_attempts(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw = write_raw_payload(product_root)
    plan = schedule_shadow_poll_job(
        product_db,
        product_root,
        "foton",
        raw,
        product_root / "scheduler_outputs",
        max_attempts=1,
    )
    job_id = plan["summary"]["job_id"]
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            UPDATE job_runs
               SET status = 'running',
                   attempt_count = 1,
                   max_attempts = 1,
                   lock_owner = 'dead-worker',
                   lock_expires_at = '2000-01-01T00:00:00+00:00'
             WHERE id = ?
            """,
            (job_id,),
        )
        con.commit()

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker")

    assert tick["summary"]["claimed"] == 0
    assert tick["summary"]["failed"] == 1
    assert tick["summary"]["validation_ok"] is False
    assert tick["maintenance"]["failed_expired_locks"] == 1
    with sqlite3.connect(product_db) as con:
        row = con.execute("select status, error, lock_owner from job_runs where id = ?", (job_id,)).fetchone()
    assert row == ("failed", "lock_expired_max_attempts_exhausted", None)


def test_scheduler_refuses_paths_outside_product_root(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="raw payload"):
        schedule_shadow_poll_job(product_db, product_root, "foton", outside, product_root / "scheduler_outputs")


def test_scheduler_cli_plan_tick_and_audit(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw = write_raw_payload(product_root)

    plan_rc = mango_office_scheduler_runtime.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "plan-shadow-poll",
            "--tenant",
            "foton",
            "--raw-payload",
            str(raw),
            "--out",
            str(product_root / "plan_cli.json"),
        ]
    )
    tick_rc = mango_office_scheduler_runtime.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "tick",
            "--worker-id",
            "cli-worker",
            "--out",
            str(product_root / "tick_cli.json"),
        ]
    )
    audit_rc = mango_office_scheduler_runtime.main(
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

    assert plan_rc == 0
    assert tick_rc == 0
    assert audit_rc == 0
    assert json.loads((product_root / "audit_cli.json").read_text(encoding="utf-8"))["summary"]["job_rows"] == 1


def test_scheduler_plans_live_shadow_poll_without_secret_material(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)

    plan = schedule_live_shadow_poll_job(
        product_db_path=product_db,
        product_root=product_root,
        tenant_id="foton",
        raw_payload_dir=product_root / "raw_payload_archive" / "live",
        output_dir=product_root / "scheduler_outputs",
        out_path=product_root / "plan_live.json",
    )

    job_input = plan["job"]["input"]
    assert plan["summary"]["planned"] == 1
    assert job_input["mode"] == "live_read_only"
    assert job_input["credentials_ref"] == {
        "api_key": "env:MANGO_OFFICE_API_KEY",
        "api_salt": "env:MANGO_OFFICE_API_SALT",
    }
    serialized = json.dumps(job_input)
    assert "api_salt" in serialized
    assert "MANGO_OFFICE_API_SALT" in serialized
    assert "secret" not in serialized.lower()
    assert (product_root / "plan_live.json").exists()


def test_scheduler_executes_live_shadow_poll_with_injected_executor(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    schedule_live_shadow_poll_job(
        product_db_path=product_db,
        product_root=product_root,
        tenant_id="foton",
        raw_payload_dir=product_root / "raw_payload_archive" / "live",
        output_dir=product_root / "scheduler_outputs",
    )

    def fake_live_executor(payload: dict) -> dict:
        assert payload["mode"] == "live_read_only"
        assert payload["raw_payload_dir"].startswith(str(product_root))
        return {
            "schema_version": "mango_live_shadow_poll_v1",
            "mode": "live_read_only",
            "summary": {
                "source_rows": 1,
                "enqueue_shadow_capture": 1,
                "validation_ok": True,
            },
            "decisions": [{"event": {"event_key": "foton:mango:CALL-4"}}],
            "safety": {"download_audio": False, "write_runtime_db": False},
            "validation_ok": True,
        }

    tick = run_scheduler_tick(product_db, product_root, worker_id="test-worker", executor=fake_live_executor)

    assert tick["summary"]["succeeded"] == 1
    assert tick["results"][0]["result"]["mode"] == "live_read_only"
    assert Path(tick["results"][0]["output_ref"]).exists()


def test_scheduler_cli_plans_live_shadow_poll(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)

    rc = mango_office_scheduler_runtime.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "plan-live-shadow-poll",
            "--tenant",
            "foton",
            "--raw-payload-dir",
            str(product_root / "raw_payload_archive" / "live"),
            "--out",
            str(product_root / "plan_live_cli.json"),
        ]
    )

    assert rc == 0
    data = json.loads((product_root / "plan_live_cli.json").read_text(encoding="utf-8"))
    assert data["summary"]["operation"] == "schedule_live_shadow_poll"
