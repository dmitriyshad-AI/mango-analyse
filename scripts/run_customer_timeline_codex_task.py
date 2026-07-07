#!/usr/bin/env python3
"""Run one Customer Timeline dev-schedule task with log and daily summary.

This is the single command used by Codex dev automations and, later, by the
launchd templates while the pipeline is still being stabilized.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FOTON_DAILY = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/_daily")
LOG_ROOT = ROOT / ".codex_local" / "staging" / "codex_dev_tasks"
NIGHTLY_DV2_CONFIG = ROOT / ".codex_local" / "staging" / "nightly_service" / "customer_timeline_nightly_service_dv2_config.json"


@dataclass(frozen=True)
class TaskSpec:
    task: str
    command: tuple[str, ...]
    expected_output: Path | None = None
    stop_reason: str = ""


def build_task_spec(task: str, *, tallanto_phone_limit: int) -> TaskSpec:
    if task == "mail-capture":
        return TaskSpec(
            task=task,
            command=("bash", "scripts/run_customer_timeline_mail_capture_daily.sh", "--apply"),
            expected_output=ROOT / ".codex_local" / "staging" / "daily_capture" / "mail_capture_manifest.json",
        )
    if task == "mango-capture":
        command_file = os.getenv("MANGO_CAPTURE_COMMAND_FILE", "").strip()
        if command_file:
            command = ("bash", "scripts/run_customer_timeline_mango_capture_daily.sh", "--apply")
        else:
            command = ("bash", "scripts/run_customer_timeline_mango_capture_daily.sh")
        return TaskSpec(
            task=task,
            command=command,
            expected_output=ROOT / ".codex_local" / "staging" / "daily_capture" / "mango_capture_manifest.json",
        )
    if task == "tallanto-api-capture":
        if os.getenv("TALLANTO_API_CAPTURE_ENABLED", "0").strip() != "1":
            return TaskSpec(
                task=task,
                command=(),
                stop_reason="TALLANTO_API_CAPTURE_ENABLED is not 1; read-only Tallanto API capture is not configured",
            )
        out_root = ROOT / ".codex_local" / "staging" / "daily_capture" / "tallanto_api_capture"
        output_path = out_root / "product" / "tallanto_entities.json"
        return TaskSpec(
            task=task,
            command=(
                sys.executable,
                "scripts/mango_office_tallanto_snapshot_export.py",
                "--product-root",
                str(out_root / "product"),
                "--product-db",
                str(out_root / "product" / "mango_product_appliance.sqlite"),
                "--out",
                str(output_path),
                "--phone-limit",
                str(tallanto_phone_limit),
                "--max-contacts-per-phone",
                "5",
            ),
            expected_output=output_path,
        )
    if task == "nightly-warehouse":
        return TaskSpec(
            task=task,
            command=(
                sys.executable,
                "scripts/run_customer_timeline_nightly_service.py",
                "--config",
                str(NIGHTLY_DV2_CONFIG),
                "--summary-only",
            ),
            expected_output=ROOT
            / ".codex_local"
            / "staging"
            / "nightly_service"
            / "published"
            / "latest_customer_timeline_snapshot.json",
            stop_reason="" if NIGHTLY_DV2_CONFIG.exists() else f"nightly config is missing: {NIGHTLY_DV2_CONFIG}",
        )
    raise ValueError(f"Unknown task: {task}")


def parse_last_json(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped:
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, Mapping):
                return value
    decoder = json.JSONDecoder()
    last: Mapping[str, Any] = {}
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            last = value
    return last


def status_from_payload(payload: Mapping[str, Any], rc: int, stop_reason: str) -> tuple[str, str]:
    if stop_reason:
        return "stopped", stop_reason
    if rc != 0:
        return "failed", f"command_rc={rc}"
    status = str(payload.get("status") or payload.get("overall_status") or "").strip()
    if status in {"locked", "not_configured", "failed", "error"}:
        return "stopped", status
    if payload.get("partial_failure") is True:
        return "stopped", "partial_failure"
    failed_required = payload.get("failed_required_steps")
    if isinstance(failed_required, Sequence) and not isinstance(failed_required, (str, bytes)) and failed_required:
        return "stopped", "failed_required_steps"
    return "ok", ""


def compact_metrics(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "no_json_payload"
    parts: list[str] = []
    for key in (
        "rows_written",
        "linked_rows",
        "pending_rows",
        "preserved_mail_link_state_rows",
        "max_event_at",
        "run_id",
        "duration_seconds",
    ):
        if key in payload:
            parts.append(f"{key}={payload.get(key)}")
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        for key in ("validation_ok", "entities", "phones", "contacts"):
            if key in summary:
                parts.append(f"summary.{key}={summary.get(key)}")
    steps = payload.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        statuses = []
        for step in steps:
            if isinstance(step, Mapping):
                statuses.append(f"{step.get('name')}:{step.get('status')}")
        if statuses:
            parts.append("steps=" + ",".join(statuses[:12]))
    return "; ".join(parts) if parts else "payload_keys=" + ",".join(sorted(str(k) for k in payload.keys())[:12])


def write_summary(
    *,
    task: str,
    started: datetime,
    finished: datetime,
    command: Sequence[str],
    log_path: Path,
    rc: int,
    status: str,
    stop_reason: str,
    metrics: str,
    expected_output: Path | None,
) -> Path:
    FOTON_DAILY.mkdir(parents=True, exist_ok=True)
    stamp = finished.strftime("%Y%m%dT%H%M%SZ")
    path = FOTON_DAILY / f"{stamp}_{task}.md"
    command_text = " ".join(command) if command else "not_run"
    lines = [
        f"Задача: {task}; статус: {status}; rc={rc}; старт={started.isoformat()}; финиш={finished.isoformat()}.",
        f"Команда: `{command_text}`.",
        f"Новое/курсоры/метрики: {metrics}.",
        f"Аномалия/stop-причина: {stop_reason or 'нет'}.",
        f"Лог: `{log_path}`; ожидаемый артефакт: `{expected_output}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_task(task: str, *, tallanto_phone_limit: int) -> int:
    spec = build_task_spec(task, tallanto_phone_limit=tallanto_phone_limit)
    started = datetime.now(timezone.utc)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_ROOT / f"{stamp}_{task}.log"
    payload: Mapping[str, Any] = {}
    rc = 78 if spec.stop_reason else 0
    combined = ""
    if spec.command:
        proc = subprocess.run(
            spec.command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        rc = proc.returncode
        combined = proc.stdout or ""
    else:
        combined = json.dumps(
            {
                "schema_version": "customer_timeline_codex_task_v1",
                "task": task,
                "status": "not_configured",
                "reason": spec.stop_reason,
                "writes_prod": False,
                "writes_crm": False,
                "writes_tallanto": False,
                "runs_asr": False,
                "runs_llm": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    log_path.write_text(combined, encoding="utf-8")
    payload = parse_last_json(combined)
    if not payload and spec.expected_output and spec.expected_output.exists():
        try:
            payload = json.loads(spec.expected_output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    status, reason = status_from_payload(payload, rc, spec.stop_reason)
    finished = datetime.now(timezone.utc)
    summary_path = write_summary(
        task=task,
        started=started,
        finished=finished,
        command=spec.command,
        log_path=log_path,
        rc=rc,
        status=status,
        stop_reason=reason,
        metrics=compact_metrics(payload),
        expected_output=spec.expected_output,
    )
    print(
        json.dumps(
            {
                "schema_version": "customer_timeline_codex_task_run_v1",
                "task": task,
                "status": status,
                "rc": rc,
                "stop_reason": reason,
                "log_path": str(log_path),
                "daily_summary_path": str(summary_path),
                "writes_prod": False,
                "writes_crm": False,
                "writes_tallanto": False,
                "runs_asr": False,
                "runs_llm": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "ok" else rc or 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Customer Timeline Codex dev-schedule task.")
    parser.add_argument(
        "--task",
        required=True,
        choices=("mail-capture", "mango-capture", "tallanto-api-capture", "nightly-warehouse"),
    )
    parser.add_argument("--tallanto-phone-limit", type=int, default=250)
    args = parser.parse_args(argv)
    return run_task(args.task, tallanto_phone_limit=args.tallanto_phone_limit)


if __name__ == "__main__":
    raise SystemExit(main())
