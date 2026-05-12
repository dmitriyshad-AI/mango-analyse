from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


SCHEDULER_HEALTH_SCHEMA_VERSION = "scheduler_health_v1"


@dataclass(frozen=True)
class SchedulerHealthSummary:
    schema_version: str
    product_db_path: str
    job_rows: int
    due_jobs: int
    failed_jobs: int
    blocked_jobs: int
    running_jobs: int
    stale_running_jobs: int
    locked_jobs: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_scheduler_health_report(
    product_db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
    *,
    stale_after_minutes: int = 30,
    recent_limit: int = 50,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if out_path:
        if "stable_runtime" in out_path.parts:
            raise ValueError("scheduler health output must not be under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"scheduler health output must stay under product root: {product_root}")
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be positive")
    if recent_limit < 1:
        raise ValueError("recent_limit must be positive")
    now = datetime.now(timezone.utc)
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        ensure_job_runs(con)
        rows = [dict(row) for row in con.execute("SELECT * FROM job_runs ORDER BY id DESC").fetchall()]
    due = [row for row in rows if job_is_due(row, now)]
    failed = [row for row in rows if clean(row.get("status")) == "failed"]
    blocked_rows = [row for row in rows if clean(row.get("status")) == "blocked"]
    running = [row for row in rows if clean(row.get("status")) == "running"]
    stale = [row for row in running if job_is_stale(row, now, stale_after_minutes=stale_after_minutes)]
    locked = [row for row in rows if clean(row.get("lock_owner")) or clean(row.get("lock_expires_at"))]
    blocked_count = len(failed) + len(blocked_rows) + len(stale)
    warnings = len(due) + len(running)
    status_counts = Counter(clean(row.get("status")) for row in rows)
    report = {
        "summary": SchedulerHealthSummary(
            schema_version=SCHEDULER_HEALTH_SCHEMA_VERSION,
            product_db_path=str(product_db_path),
            job_rows=len(rows),
            due_jobs=len(due),
            failed_jobs=len(failed),
            blocked_jobs=len(blocked_rows),
            running_jobs=len(running),
            stale_running_jobs=len(stale),
            locked_jobs=len(locked),
            validation_ok=blocked_count == 0,
            blocked=blocked_count,
            warnings=warnings,
        ).to_json_dict(),
        "status_counts": dict(sorted(status_counts.items())),
        "due_jobs": compact_jobs(due[:recent_limit]),
        "failed_jobs": compact_jobs(failed[:recent_limit]),
        "blocked_jobs": compact_jobs(blocked_rows[:recent_limit]),
        "stale_running_jobs": compact_jobs(stale[:recent_limit]),
        "locked_jobs": compact_jobs(locked[:recent_limit]),
        "recent_jobs": compact_jobs(rows[:recent_limit]),
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def job_is_due(row: Mapping[str, Any], now: datetime) -> bool:
    if clean(row.get("status")) not in {"planned", "retry_wait"}:
        return False
    due_at = parse_datetime(clean(row.get("next_run_at")) or clean(row.get("scheduled_for")) or clean(row.get("planned_at")))
    return due_at is not None and due_at <= now


def job_is_stale(row: Mapping[str, Any], now: datetime, *, stale_after_minutes: int) -> bool:
    heartbeat = parse_datetime(clean(row.get("heartbeat_at")) or clean(row.get("started_at")))
    if heartbeat is None:
        return True
    return (now - heartbeat).total_seconds() > stale_after_minutes * 60


def compact_jobs(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "job_type": clean(row.get("job_type")),
            "tenant_id": clean(row.get("tenant_id")) or None,
            "status": clean(row.get("status")),
            "planned_at": clean(row.get("planned_at")) or None,
            "scheduled_for": clean(row.get("scheduled_for")) or None,
            "next_run_at": clean(row.get("next_run_at")) or None,
            "attempt_count": row.get("attempt_count"),
            "max_attempts": row.get("max_attempts"),
            "lock_owner": clean(row.get("lock_owner")) or None,
            "lock_expires_at": clean(row.get("lock_expires_at")) or None,
            "heartbeat_at": clean(row.get("heartbeat_at")) or None,
            "output_ref": clean(row.get("output_ref")) or None,
            "error": clean(row.get("error")) or None,
        }
        for row in rows
    ]


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ensure_job_runs(con: sqlite3.Connection) -> None:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'job_runs'").fetchone()
    if row is None:
        raise ValueError("product DB does not contain job_runs")


def safety_contract() -> Mapping[str, bool]:
    return {
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(path_payload(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_payload(payload: object) -> object:
    return payload
