from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.scheduler_health import build_scheduler_health_report
from mango_mvp.productization.scheduler_runtime import audit_scheduler_runtime
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


SCHEDULER_CONTROL_PLANE_SCHEMA_VERSION = "scheduler_control_plane_v1"


@dataclass(frozen=True)
class SchedulerControlPlaneSummary:
    schema_version: str
    product_db_path: str
    due_jobs: int
    failed_jobs: int
    stale_running_jobs: int
    locked_jobs: int
    recommended_actions: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_scheduler_control_plane_report(
    product_db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
    *,
    worker_id: str = "appliance-supervisor",
    tick_limit: int = 1,
    stale_after_minutes: int = 30,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if out_path:
        guard_under_root(out_path, product_root, "scheduler control-plane output")
    worker_id = clean(worker_id) or "appliance-supervisor"
    if tick_limit < 1:
        raise ValueError("tick_limit must be positive")
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be positive")

    health = build_scheduler_health_report(
        product_db_path=product_db_path,
        product_root=product_root,
        stale_after_minutes=stale_after_minutes,
    )
    runtime = audit_scheduler_runtime(product_db_path=product_db_path, product_root=product_root)
    health_summary = health["summary"]
    actions = recommended_actions(
        product_root=product_root,
        product_db_path=product_db_path,
        worker_id=worker_id,
        tick_limit=tick_limit,
        health_summary=health_summary,
    )
    blocked = int(health_summary.get("failed_jobs") or 0) + int(health_summary.get("stale_running_jobs") or 0)
    warnings = int(health_summary.get("due_jobs") or 0) + int(health_summary.get("locked_jobs") or 0)
    report = {
        "summary": SchedulerControlPlaneSummary(
            schema_version=SCHEDULER_CONTROL_PLANE_SCHEMA_VERSION,
            product_db_path=str(product_db_path),
            due_jobs=int(health_summary.get("due_jobs") or 0),
            failed_jobs=int(health_summary.get("failed_jobs") or 0),
            stale_running_jobs=int(health_summary.get("stale_running_jobs") or 0),
            locked_jobs=int(health_summary.get("locked_jobs") or 0),
            recommended_actions=len(actions),
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict(),
        "recommended_actions": actions,
        "scheduler_health": health,
        "scheduler_runtime_audit": runtime,
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def recommended_actions(
    *,
    product_root: Path,
    product_db_path: Path,
    worker_id: str,
    tick_limit: int,
    health_summary: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    actions = []
    due = int(health_summary.get("due_jobs") or 0)
    failed = int(health_summary.get("failed_jobs") or 0)
    stale = int(health_summary.get("stale_running_jobs") or 0)
    locked = int(health_summary.get("locked_jobs") or 0)
    rows = int(health_summary.get("job_rows") or 0)
    if failed:
        actions.append(
            action(
                "REVIEW_FAILED_JOBS",
                "blocked",
                "Open scheduler health failed_jobs before retrying automation.",
                None,
            )
        )
    if stale:
        actions.append(
            action(
                "REQUEUE_OR_FAIL_STALE_LOCKS",
                "blocked",
                "Run one scheduler tick to apply expired-lock maintenance.",
                tick_command(product_root, product_db_path, worker_id, tick_limit),
            )
        )
    if due:
        actions.append(
            action(
                "RUN_SCHEDULER_TICK",
                "ready",
                "Due jobs are ready for an explicit scheduler tick.",
                tick_command(product_root, product_db_path, worker_id, tick_limit),
            )
        )
    if locked and not stale:
        actions.append(
            action(
                "WAIT_FOR_LOCK_OR_REVIEW_WORKER",
                "warning",
                "Locked jobs exist but are not stale yet.",
                None,
            )
        )
    if rows == 0:
        actions.append(
            action(
                "SCHEDULE_SHADOW_POLL",
                "ready",
                "No scheduler jobs exist. Schedule a read-only Mango shadow poll when credentials are ready.",
                schedule_command(product_root, product_db_path),
            )
        )
    if not actions:
        actions.append(action("NO_ACTION_REQUIRED", "ready", "Scheduler has no due, failed, locked or stale jobs.", None))
    return actions


def action(action_id: str, status: str, reason: str, command_text: Optional[str]) -> Mapping[str, Any]:
    return {
        "action": action_id,
        "status": status,
        "reason": reason,
        "command": command_text,
        "executes_now": False,
    }


def tick_command(product_root: Path, product_db_path: Path, worker_id: str, tick_limit: int) -> str:
    return (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_scheduler_runtime.py "
        f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} "
        f"tick --worker-id {worker_id} --limit {tick_limit}"
    )


def schedule_command(product_root: Path, product_db_path: Path) -> str:
    return (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_scheduler_runtime.py "
        f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} "
        "plan-live-shadow-poll --tenant <tenant_id>"
    )


def shell_path(path: Path) -> str:
    text = str(path)
    if not text or all(ch not in text for ch in (" ", "'", '"', "(", ")")):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "executes_jobs": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
