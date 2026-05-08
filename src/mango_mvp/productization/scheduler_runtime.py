from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.productization.mango_live_shadow_poll import build_mango_live_shadow_poll_report
from mango_mvp.productization.mango_office_client import DEFAULT_MANGO_BASE_URL
from mango_mvp.productization.product_db import (
    PRODUCT_DB_ADMIN_SCHEMA_VERSION,
    audit_product_db,
    apply_product_db_migrations,
    guard_product_db_path,
    now_utc,
)
from mango_mvp.productization.supervisor import payload_file_stats
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


SCHEDULER_RUNTIME_SCHEMA_VERSION = "scheduler_runtime_v1"
ALLOWED_RUNTIME_JOB_TYPES = {"shadow_poll"}
TERMINAL_STATUSES = {"succeeded", "failed", "blocked", "skipped"}


@dataclass(frozen=True)
class SchedulerJobSummary:
    schema_version: str
    product_db_path: str
    operation: str
    planned: int
    claimed: int
    succeeded: int
    failed: int
    blocked: int
    retry_wait: int
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def schedule_shadow_poll_job(
    product_db_path: Path,
    product_root: Path,
    tenant_id: str,
    raw_payload_path: Path,
    output_dir: Path,
    window_hours: float = 2.0,
    planned_at: Optional[datetime] = None,
    max_attempts: int = 3,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root = resolve_product_paths(product_db_path, product_root)
    raw_payload_path = raw_payload_path.resolve(strict=False)
    output_dir = output_dir.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_scheduler_path(raw_payload_path, product_root, "raw payload")
    guard_scheduler_path(output_dir, product_root, "scheduler output directory")
    if out_path:
        guard_scheduler_path(out_path, product_root, "scheduler audit output")
    tenant_id = clean(tenant_id)
    if not tenant_id:
        raise ValueError("tenant_id must not be empty")
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    planned = planned_at or datetime.now(timezone.utc)
    if planned.tzinfo is None or planned.utcoffset() is None:
        planned = planned.replace(tzinfo=timezone.utc)
    input_payload = {
        "schema_version": SCHEDULER_RUNTIME_SCHEMA_VERSION,
        "mode": "dry_run",
        "tenant_id": tenant_id,
        "provider": "mango",
        "window_hours": window_hours,
        "raw_payload_path": str(raw_payload_path),
        "output_dir": str(output_dir),
        "hard_guards": {
            "download_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_runtime_db": False,
        },
    }
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        apply_product_db_migrations(con)
        assert_job_type_allowed(con, "shadow_poll")
        now = now_utc()
        cur = con.execute(
            """
            INSERT INTO job_runs (
              job_type, tenant_id, status, planned_at,
              input_ref, scheduled_for, next_run_at,
              attempt_count, max_attempts
            ) VALUES ('shadow_poll', ?, 'planned', ?, ?, ?, ?, 0, ?)
            """,
            (
                tenant_id,
                now,
                json.dumps(input_payload, ensure_ascii=False, sort_keys=True),
                planned.isoformat(),
                planned.isoformat(),
                int(max_attempts),
            ),
        )
        job_id = int(cur.lastrowid)
        con.commit()
    report = {
        "summary": SchedulerJobSummary(
            schema_version=SCHEDULER_RUNTIME_SCHEMA_VERSION,
            product_db_path=str(product_db_path),
            operation="schedule_shadow_poll",
            planned=1,
            claimed=0,
            succeeded=0,
            failed=0,
            blocked=0,
            retry_wait=0,
            validation_ok=True,
            warnings=0,
        ).to_json_dict()
        | {"job_id": job_id},
        "job": {"id": job_id, "input": input_payload},
    }
    if out_path:
        write_json(out_path, report)
    return report


def schedule_live_shadow_poll_job(
    product_db_path: Path,
    product_root: Path,
    tenant_id: str,
    raw_payload_dir: Path,
    output_dir: Path,
    window_hours: float = 2.0,
    planned_at: Optional[datetime] = None,
    max_attempts: int = 3,
    base_url: str = DEFAULT_MANGO_BASE_URL,
    allow_metadata_only: bool = False,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root = resolve_product_paths(product_db_path, product_root)
    raw_payload_dir = raw_payload_dir.resolve(strict=False)
    output_dir = output_dir.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_scheduler_path(raw_payload_dir, product_root, "raw payload directory")
    guard_scheduler_path(output_dir, product_root, "scheduler output directory")
    if out_path:
        guard_scheduler_path(out_path, product_root, "scheduler audit output")
    tenant_id = clean(tenant_id)
    if not tenant_id:
        raise ValueError("tenant_id must not be empty")
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    planned = planned_at or datetime.now(timezone.utc)
    if planned.tzinfo is None or planned.utcoffset() is None:
        planned = planned.replace(tzinfo=timezone.utc)
    input_payload = {
        "schema_version": SCHEDULER_RUNTIME_SCHEMA_VERSION,
        "mode": "live_read_only",
        "tenant_id": tenant_id,
        "provider": "mango",
        "window_hours": window_hours,
        "product_db_path": str(product_db_path),
        "base_url": clean(base_url) or DEFAULT_MANGO_BASE_URL,
        "raw_payload_dir": str(raw_payload_dir),
        "output_dir": str(output_dir),
        "allow_metadata_only": bool(allow_metadata_only),
        "credentials_ref": {
            "api_key": "env:MANGO_OFFICE_API_KEY",
            "api_salt": "env:MANGO_OFFICE_API_SALT",
        },
        "hard_guards": {
            "download_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_runtime_db": False,
        },
    }
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        apply_product_db_migrations(con)
        assert_job_type_allowed(con, "shadow_poll")
        now = now_utc()
        cur = con.execute(
            """
            INSERT INTO job_runs (
              job_type, tenant_id, status, planned_at,
              input_ref, scheduled_for, next_run_at,
              attempt_count, max_attempts
            ) VALUES ('shadow_poll', ?, 'planned', ?, ?, ?, ?, 0, ?)
            """,
            (
                tenant_id,
                now,
                json.dumps(input_payload, ensure_ascii=False, sort_keys=True),
                planned.isoformat(),
                planned.isoformat(),
                int(max_attempts),
            ),
        )
        job_id = int(cur.lastrowid)
        con.commit()
    report = {
        "summary": SchedulerJobSummary(
            schema_version=SCHEDULER_RUNTIME_SCHEMA_VERSION,
            product_db_path=str(product_db_path),
            operation="schedule_live_shadow_poll",
            planned=1,
            claimed=0,
            succeeded=0,
            failed=0,
            blocked=0,
            retry_wait=0,
            validation_ok=True,
            warnings=0,
        ).to_json_dict()
        | {"job_id": job_id},
        "job": {"id": job_id, "input": input_payload},
    }
    if out_path:
        write_json(out_path, report)
    return report


def run_scheduler_tick(
    product_db_path: Path,
    product_root: Path,
    worker_id: str,
    limit: int = 1,
    lock_seconds: int = 300,
    out_path: Optional[Path] = None,
    executor: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root = resolve_product_paths(product_db_path, product_root)
    out_path = out_path.resolve(strict=False) if out_path else None
    if out_path:
        guard_scheduler_path(out_path, product_root, "scheduler tick output")
    worker_id = clean(worker_id)
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    if limit < 1:
        raise ValueError("limit must be positive")
    if lock_seconds < 1:
        raise ValueError("lock_seconds must be positive")

    results = []
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        apply_product_db_migrations(con)
        maintenance = requeue_expired_running_jobs(con)
        for _ in range(limit):
            job = claim_next_due_job(con, worker_id=worker_id, lock_seconds=lock_seconds)
            if job is None:
                break
            result = execute_claimed_job(con, job=job, product_root=product_root, executor=executor)
            results.append(result)
        con.commit()
    counts = count_tick_results(results)
    report = {
        "summary": SchedulerJobSummary(
            schema_version=SCHEDULER_RUNTIME_SCHEMA_VERSION,
            product_db_path=str(product_db_path),
            operation="tick",
            planned=0,
            claimed=len(results),
            succeeded=counts["succeeded"],
            failed=counts["failed"] + int(maintenance["failed_expired_locks"]),
            blocked=counts["blocked"],
            retry_wait=counts["retry_wait"],
            validation_ok=counts["failed"] == 0 and counts["blocked"] == 0 and int(maintenance["failed_expired_locks"]) == 0,
            warnings=counts["retry_wait"] + int(maintenance["requeued_expired_locks"]) + int(maintenance["failed_expired_locks"]),
        ).to_json_dict(),
        "maintenance": maintenance,
        "results": results,
        "safety": scheduler_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def audit_scheduler_runtime(
    product_db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root = resolve_product_paths(product_db_path, product_root)
    out_path = out_path.resolve(strict=False) if out_path else None
    if out_path:
        guard_scheduler_path(out_path, product_root, "scheduler audit output")
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        apply_product_db_migrations(con)
        rows = con.execute(
            """
            SELECT job_type, status, count(*) AS n
              FROM job_runs
             GROUP BY job_type, status
             ORDER BY job_type, status
            """
        ).fetchall()
        recent = con.execute(
            """
            SELECT id, job_type, tenant_id, status, planned_at, started_at, finished_at,
                   attempt_count, max_attempts, next_run_at, output_ref, error
              FROM job_runs
             ORDER BY id DESC
             LIMIT 20
            """
        ).fetchall()
    product_integrity = audit_product_db(product_db_path, product_root)
    status_counts = {f"{row['job_type']}|{row['status']}": int(row["n"] or 0) for row in rows}
    terminal = sum(count for key, count in status_counts.items() if key.rsplit("|", 1)[-1] in TERMINAL_STATUSES)
    failed = sum(count for key, count in status_counts.items() if key.endswith("|failed"))
    blocked = sum(count for key, count in status_counts.items() if key.endswith("|blocked"))
    retry_wait = sum(count for key, count in status_counts.items() if key.endswith("|retry_wait"))
    running = sum(count for key, count in status_counts.items() if key.endswith("|running"))
    report = {
        "summary": {
            "schema_version": SCHEDULER_RUNTIME_SCHEMA_VERSION,
            "product_db_path": str(product_db_path),
            "operation": "audit",
            "job_rows": sum(status_counts.values()),
            "terminal_job_rows": terminal,
            "failed_job_rows": failed,
            "blocked_job_rows": blocked,
            "retry_wait_job_rows": retry_wait,
            "running_job_rows": running,
            "status_counts": status_counts,
            "validation_ok": bool(product_integrity["summary"]["validation_ok"]) and failed == 0 and blocked == 0,
            "warnings": int(product_integrity["summary"]["warnings"]) + retry_wait + running,
        },
        "recent_jobs": [dict(row) for row in recent],
        "product_integrity": product_integrity,
        "safety": scheduler_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def requeue_expired_running_jobs(con: sqlite3.Connection) -> Mapping[str, int]:
    now = datetime.now(timezone.utc)
    now_text = now.isoformat()
    rows = con.execute(
        """
        SELECT id, attempt_count, max_attempts
          FROM job_runs
         WHERE status = 'running'
           AND lock_expires_at IS NOT NULL
           AND lock_expires_at <= ?
         ORDER BY id
        """,
        (now_text,),
    ).fetchall()
    requeued = 0
    failed = 0
    for row in rows:
        attempt_count = int(row["attempt_count"] or 0)
        max_attempts = int(row["max_attempts"] or 1)
        if attempt_count >= max_attempts:
            con.execute(
                """
                UPDATE job_runs
                   SET status = 'failed',
                       finished_at = ?,
                       next_run_at = NULL,
                       error = 'lock_expired_max_attempts_exhausted',
                       lock_owner = NULL,
                       lock_expires_at = NULL,
                       heartbeat_at = ?
                 WHERE id = ?
                """,
                (now_text, now_text, int(row["id"])),
            )
            failed += 1
        else:
            con.execute(
                """
                UPDATE job_runs
                   SET status = 'retry_wait',
                       next_run_at = ?,
                       error = 'lock_expired_requeued',
                       lock_owner = NULL,
                       lock_expires_at = NULL,
                       heartbeat_at = ?
                 WHERE id = ?
                """,
                (now_text, now_text, int(row["id"])),
            )
            requeued += 1
    return {"requeued_expired_locks": requeued, "failed_expired_locks": failed}


def claim_next_due_job(con: sqlite3.Connection, worker_id: str, lock_seconds: int) -> Optional[Mapping[str, Any]]:
    now = datetime.now(timezone.utc)
    now_text = now.isoformat()
    lock_until = (now + timedelta(seconds=lock_seconds)).isoformat()
    row = con.execute(
        """
        SELECT *
          FROM job_runs
         WHERE status IN ('planned', 'retry_wait')
           AND coalesce(next_run_at, planned_at) <= ?
           AND (lock_expires_at IS NULL OR lock_expires_at <= ?)
         ORDER BY coalesce(next_run_at, planned_at), id
         LIMIT 1
        """,
        (now_text, now_text),
    ).fetchone()
    if row is None:
        return None
    cur = con.execute(
        """
        UPDATE job_runs
           SET status = 'running',
               started_at = coalesce(started_at, ?),
               lock_owner = ?,
               lock_expires_at = ?,
               heartbeat_at = ?,
               attempt_count = attempt_count + 1
         WHERE id = ?
           AND status IN ('planned', 'retry_wait')
           AND (lock_expires_at IS NULL OR lock_expires_at <= ?)
        """,
        (now_text, worker_id, lock_until, now_text, int(row["id"]), now_text),
    )
    if cur.rowcount != 1:
        return None
    claimed = con.execute("SELECT * FROM job_runs WHERE id = ?", (int(row["id"]),)).fetchone()
    return dict(claimed)


def execute_claimed_job(
    con: sqlite3.Connection,
    job: Mapping[str, Any],
    product_root: Path,
    executor: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]],
) -> Mapping[str, Any]:
    job_id = int(job["id"])
    job_type = clean(job.get("job_type"))
    try:
        if job_type not in ALLOWED_RUNTIME_JOB_TYPES:
            return finish_job_blocked(con, job, f"job_type_disabled_for_stage3:{job_type}")
        assert_job_type_allowed(con, job_type)
        input_payload = parse_job_input(job.get("input_ref"))
        mode = clean(input_payload.get("mode"))
        if mode not in {"dry_run", "live_read_only"}:
            return finish_job_blocked(con, job, f"mode_disabled_for_stage4:{mode}")
        try:
            if mode == "live_read_only":
                input_payload = validate_live_shadow_poll_input(input_payload, product_root)
            else:
                input_payload = validate_shadow_poll_input(input_payload, product_root)
        except ValueError as exc:
            return finish_job_blocked(con, job, str(exc))
        job_executor = executor or default_executor_for_mode(mode)
        result = dict(job_executor(input_payload))
        if not bool(result.get("validation_ok", True)):
            raise RuntimeError("executor_validation_failed")
        output_ref = write_job_output(Path(clean(input_payload.get("output_dir"))), job_id=job_id, result=result)
        now = now_utc()
        con.execute(
            """
            UPDATE job_runs
               SET status = 'succeeded',
                   finished_at = ?,
                   output_ref = ?,
                   result_json = ?,
                   error = NULL,
                   lock_owner = NULL,
                   lock_expires_at = NULL,
                   heartbeat_at = ?
             WHERE id = ?
            """,
            (now, str(output_ref), json.dumps(result, ensure_ascii=False, sort_keys=True), now, job_id),
        )
        return {"job_id": job_id, "job_type": job_type, "status": "succeeded", "output_ref": str(output_ref), "result": result}
    except Exception as exc:
        return finish_job_error(con, job, str(exc))


def validate_shadow_poll_input(input_payload: Mapping[str, Any], product_root: Path) -> Mapping[str, Any]:
    tenant_id = clean(input_payload.get("tenant_id"))
    if not tenant_id:
        raise ValueError("tenant_id_required")
    raw_payload_path = Path(clean(input_payload.get("raw_payload_path"))).resolve(strict=False)
    output_dir = Path(clean(input_payload.get("output_dir"))).resolve(strict=False)
    guard_scheduler_path(raw_payload_path, product_root, "raw payload")
    guard_scheduler_path(output_dir, product_root, "scheduler output directory")

    hard_guards = input_payload.get("hard_guards") or {}
    if not isinstance(hard_guards, Mapping):
        raise ValueError("hard_guards_must_be_object")
    dangerous_actions = {
        "download_audio",
        "run_asr",
        "run_ra",
        "write_crm",
        "write_runtime_db",
    }
    enabled = sorted(action for action in dangerous_actions if bool(hard_guards.get(action)))
    if enabled:
        raise ValueError(f"dangerous_action_enabled:{','.join(enabled)}")

    normalized = dict(input_payload)
    normalized["tenant_id"] = tenant_id
    normalized["provider"] = clean(normalized.get("provider")) or "mango"
    normalized["raw_payload_path"] = str(raw_payload_path)
    normalized["output_dir"] = str(output_dir)
    return normalized


def validate_live_shadow_poll_input(input_payload: Mapping[str, Any], product_root: Path) -> Mapping[str, Any]:
    tenant_id = clean(input_payload.get("tenant_id"))
    if not tenant_id:
        raise ValueError("tenant_id_required")
    product_db_path = Path(clean(input_payload.get("product_db_path"))).resolve(strict=False)
    raw_payload_dir = Path(clean(input_payload.get("raw_payload_dir"))).resolve(strict=False)
    output_dir = Path(clean(input_payload.get("output_dir"))).resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    guard_scheduler_path(raw_payload_dir, product_root, "raw payload directory")
    guard_scheduler_path(output_dir, product_root, "scheduler output directory")
    if float(input_payload.get("window_hours") or 0) <= 0:
        raise ValueError("window_hours_must_be_positive")

    hard_guards = input_payload.get("hard_guards") or {}
    if not isinstance(hard_guards, Mapping):
        raise ValueError("hard_guards_must_be_object")
    dangerous_actions = {
        "download_audio",
        "run_asr",
        "run_ra",
        "write_crm",
        "write_runtime_db",
    }
    enabled = sorted(action for action in dangerous_actions if bool(hard_guards.get(action)))
    if enabled:
        raise ValueError(f"dangerous_action_enabled:{','.join(enabled)}")

    normalized = dict(input_payload)
    normalized["tenant_id"] = tenant_id
    normalized["provider"] = clean(normalized.get("provider")) or "mango"
    normalized["product_db_path"] = str(product_db_path)
    normalized["product_root"] = str(product_root)
    normalized["raw_payload_dir"] = str(raw_payload_dir)
    normalized["output_dir"] = str(output_dir)
    normalized["base_url"] = clean(normalized.get("base_url")) or DEFAULT_MANGO_BASE_URL
    normalized["window_hours"] = float(normalized.get("window_hours") or 0)
    normalized["allow_metadata_only"] = bool(normalized.get("allow_metadata_only"))
    return normalized


def default_executor_for_mode(mode: str) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    if mode == "live_read_only":
        return execute_live_shadow_poll_job
    return execute_shadow_poll_dry_run_job


def execute_shadow_poll_dry_run_job(input_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_payload_path = Path(clean(input_payload.get("raw_payload_path"))).resolve(strict=False)
    stats = payload_file_stats(raw_payload_path)
    return {
        "schema_version": SCHEDULER_RUNTIME_SCHEMA_VERSION,
        "job_type": "shadow_poll",
        "mode": "dry_run",
        "tenant_id": clean(input_payload.get("tenant_id")),
        "provider": clean(input_payload.get("provider")) or "mango",
        "window_hours": input_payload.get("window_hours"),
        "raw_payload_stats": stats,
        "actions": {
            "would_poll_provider": True,
            "download_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_runtime_db": False,
        },
        "validation_ok": bool(stats.get("exists")),
    }


def execute_live_shadow_poll_job(input_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    until = datetime.now(timezone.utc)
    since = until - timedelta(hours=float(input_payload.get("window_hours") or 2.0))
    raw_payload_dir = Path(clean(input_payload.get("raw_payload_dir"))).resolve(strict=False)
    raw_payload_path = raw_payload_dir / build_live_shadow_payload_filename(
        tenant_id=clean(input_payload.get("tenant_id")),
        since=since,
        until=until,
    )
    report = build_mango_live_shadow_poll_report(
        product_db_path=Path(clean(input_payload.get("product_db_path"))),
        product_root=Path(clean(input_payload.get("product_root"))).resolve(strict=False),
        tenant_id=clean(input_payload.get("tenant_id")),
        since=since,
        until=until,
        raw_payload_path=raw_payload_path,
        base_url=clean(input_payload.get("base_url")) or DEFAULT_MANGO_BASE_URL,
        allow_metadata_only=bool(input_payload.get("allow_metadata_only")),
    )
    return {
        **report,
        "job_type": "shadow_poll",
        "mode": "live_read_only",
    }


def build_live_shadow_payload_filename(tenant_id: str, since: datetime, until: datetime) -> str:
    start = compact_timestamp(since)
    end = compact_timestamp(until)
    tenant = clean(tenant_id).replace("/", "_") or "tenant"
    return f"live_shadow_poll_{tenant}_{start}_{end}.jsonl"


def compact_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def finish_job_blocked(con: sqlite3.Connection, job: Mapping[str, Any], reason: str) -> Mapping[str, Any]:
    job_id = int(job["id"])
    now = now_utc()
    con.execute(
        """
        UPDATE job_runs
           SET status = 'blocked',
               finished_at = ?,
               error = ?,
               lock_owner = NULL,
               lock_expires_at = NULL,
               heartbeat_at = ?
         WHERE id = ?
        """,
        (now, reason, now, job_id),
    )
    return {"job_id": job_id, "job_type": clean(job.get("job_type")), "status": "blocked", "error": reason}


def finish_job_error(con: sqlite3.Connection, job: Mapping[str, Any], error: str) -> Mapping[str, Any]:
    job_id = int(job["id"])
    attempt_count = int(job.get("attempt_count") or 0)
    max_attempts = int(job.get("max_attempts") or 1)
    now = datetime.now(timezone.utc)
    if attempt_count < max_attempts:
        status = "retry_wait"
        delay_seconds = min(3600, 60 * (2 ** max(0, attempt_count - 1)))
        next_run_at = (now + timedelta(seconds=delay_seconds)).isoformat()
    else:
        status = "failed"
        next_run_at = None
    con.execute(
        """
        UPDATE job_runs
           SET status = ?,
               finished_at = CASE WHEN ? = 'failed' THEN ? ELSE finished_at END,
               next_run_at = ?,
               error = ?,
               lock_owner = NULL,
               lock_expires_at = NULL,
               heartbeat_at = ?
         WHERE id = ?
        """,
        (status, status, now.isoformat(), next_run_at, error, now.isoformat(), job_id),
    )
    return {"job_id": job_id, "job_type": clean(job.get("job_type")), "status": status, "error": error}


def assert_job_type_allowed(con: sqlite3.Connection, job_type: str) -> None:
    row = con.execute("SELECT default_mode FROM job_types WHERE job_type = ?", (job_type,)).fetchone()
    if row is None:
        raise ValueError(f"unknown job_type: {job_type}")
    if clean(row["default_mode"]) == "disabled":
        raise ValueError(f"job_type disabled: {job_type}")


def parse_job_input(input_ref: Any) -> Mapping[str, Any]:
    payload = json.loads(clean(input_ref))
    if not isinstance(payload, Mapping):
        raise ValueError("job input_ref must be a JSON object")
    return payload


def write_job_output(output_dir: Path, job_id: int, result: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ref = output_dir / f"shadow_poll_job_{job_id:06d}.json"
    write_json(output_ref, result)
    return output_ref


def count_tick_results(results: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts = {"succeeded": 0, "failed": 0, "blocked": 0, "retry_wait": 0}
    for result in results:
        status = clean(result.get("status"))
        if status in counts:
            counts[status] += 1
    return counts


def resolve_product_paths(product_db_path: Path, product_root: Path) -> tuple[Path, Path]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    return product_db_path, product_root


def guard_scheduler_path(path: Path, product_root: Path, label: str) -> None:
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def scheduler_safety() -> Mapping[str, bool]:
    return {
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "download_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "product_db_job_runs_writes": True,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
