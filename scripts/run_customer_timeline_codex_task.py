#!/usr/bin/env python3
"""Run one Customer Timeline dev-schedule task with log and daily summary.

This is the single command used by Codex dev automations and, later, by the
launchd templates while the pipeline is still being stabilized.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
FOTON_DAILY = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/_daily")
MANGO_READY_PACKAGE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/"
    "mango_calls_two_processes/drop/mango_calls_ready.sqlite"
)
NIGHTLY_HOME = Path(
    os.getenv("CUSTOMER_TIMELINE_NIGHTLY_HOME", "~/.mango_local/customer_timeline_nightly")
).expanduser()
STAGING_ROOT = NIGHTLY_HOME / ".codex_local" / "staging"
LOG_ROOT = STAGING_ROOT / "codex_dev_tasks"
TASK_STATE_ROOT = STAGING_ROOT / "task_state"
NIGHTLY_SERVICE_ROOT = STAGING_ROOT / "nightly_service"
NIGHTLY_DV2_CONFIG = NIGHTLY_SERVICE_ROOT / "customer_timeline_nightly_service_dv2_config.json"
NIGHTLY_BASE_CONFIG = NIGHTLY_SERVICE_ROOT / "customer_timeline_nightly_service_config.json"
STAGING_TIMELINE_DB = STAGING_ROOT / "customer_timeline_staging.sqlite"
MAIL_STATE_DIR = STAGING_ROOT / "mail_pipeline"
MAIL_DATA_ROOT = Path(
    os.getenv("MANGO_MAIL_DATA_ROOT", "/Users/dmitrijfabarisov/Mango_Data")
).expanduser()
PROD_TIMELINE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/"
    "customer_timeline_prod_20260621/customer_timeline.sqlite"
)
PROD_SNAPSHOT_STALE_HOURS = 7 * 24
TASK_SUCCESS_STALE_HOURS = 30
REQUIRED_NIGHTLY_STEPS = {
    "mango_processed_sweep",
    "calls_and_amo_incremental",
    "mail_archive_incremental",
}
REQUIRED_CALL_SOURCES = {"mango_processed_summary": "mango_processed_summary"}


@dataclass(frozen=True)
class TaskSpec:
    task: str
    command: tuple[str, ...]
    expected_output: Path | None = None
    stop_reason: str = ""


def current_runtime() -> Mapping[str, str]:
    return {
        "head": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "worktree": str(ROOT),
    }


def stage_manifest_stop_reason(path: Path, *, max_age_hours: float = 4.0) -> str:
    if not path.is_file():
        return f"stage manifest is missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        finished = datetime.fromisoformat(str(payload["finished_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError, json.JSONDecodeError):
        return "stage manifest is invalid"
    if payload.get("status") != "ok":
        return "stage manifest status is not ok"
    if payload.get("runtime") != current_runtime():
        return "stage manifest runtime does not match current HEAD/worktree"
    age = datetime.now(timezone.utc) - finished.astimezone(timezone.utc)
    if age < timedelta(0) or age > timedelta(hours=max_age_hours):
        return "stage manifest is stale"
    return ""


def validate_nightly_config(path: Path | None = None) -> str:
    path = path or NIGHTLY_DV2_CONFIG
    if not path.is_file():
        return f"nightly config is missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"nightly config is invalid: {path}"
    if Path(str(payload.get("timeline_db") or "")).expanduser().resolve(strict=False) != STAGING_TIMELINE_DB.resolve(
        strict=False
    ):
        return "nightly config timeline_db does not match persistent staging DB"
    if Path(str(payload.get("allowed_root") or "")).expanduser().resolve(strict=False) != STAGING_ROOT.resolve(
        strict=False
    ):
        return "nightly config allowed_root does not match persistent staging root"
    steps = {
        str(item.get("name") or ""): item
        for item in payload.get("steps") or ()
        if isinstance(item, Mapping)
    }
    missing = sorted(REQUIRED_NIGHTLY_STEPS - steps.keys())
    if missing:
        return "nightly config misses required steps: " + ",".join(missing)
    inactive = sorted(
        name
        for name in REQUIRED_NIGHTLY_STEPS
        if steps[name].get("enabled") is not True or steps[name].get("required") is not True
    )
    if inactive:
        return "nightly config has inactive required steps: " + ",".join(inactive)
    calls_config = steps["calls_and_amo_incremental"].get("config")
    if not isinstance(calls_config, Mapping):
        return "calls_and_amo_incremental has no config"
    journal_path = Path(str(calls_config.get("journal_path") or "")).expanduser().resolve(strict=False)
    if not path_is_within(journal_path, STAGING_ROOT):
        return "calls_and_amo_incremental journal is outside persistent staging root"
    call_sources = {
        str(item.get("source_system") or ""): item
        for item in calls_config.get("sources") or ()
        if isinstance(item, Mapping)
    }
    missing_call_sources = sorted(REQUIRED_CALL_SOURCES.keys() - call_sources.keys())
    if missing_call_sources:
        return "calls_and_amo_incremental misses required sources: " + ",".join(missing_call_sources)
    for source_system, normalizer in REQUIRED_CALL_SOURCES.items():
        source = call_sources[source_system]
        if source.get("normalizer") != normalizer or source.get("required") is not True:
            return f"calls_and_amo_incremental source contract is invalid: {source_system}"
        source_path = Path(str(source.get("path") or "")).expanduser().resolve(strict=False)
        if not path_is_within(source_path, STAGING_ROOT):
            return f"calls_and_amo_incremental source is outside persistent staging root: {source_system}"
    sweep_script = Path(
        str(steps["mango_processed_sweep"].get("config", {}).get("producer_script") or "")
    ).resolve(strict=False)
    expected_sweep_script = (ROOT / "scripts" / "build_mango_call_timeline_increment.py").resolve(strict=False)
    if sweep_script != expected_sweep_script:
        return "nightly config points mango sweep at another code root"
    package_dbs = {
        Path(str(item)).expanduser().resolve(strict=False)
        for item in steps["mango_processed_sweep"].get("config", {}).get("package_dbs") or ()
    }
    ready_package_db = MANGO_READY_PACKAGE_DB.resolve(strict=False)
    if ready_package_db not in package_dbs:
        return "mango_processed_sweep misses required mango_calls_ready.sqlite package DB"
    if not ready_package_db.is_file():
        return f"required mango_calls_ready.sqlite package DB is missing: {ready_package_db}"
    try:
        with sqlite3.connect(f"file:{ready_package_db}?mode=ro", uri=True, timeout=5) as con:
            columns = {str(row[1]) for row in con.execute("PRAGMA table_info(call_records)")}
    except sqlite3.Error as exc:
        return f"required mango_calls_ready.sqlite package DB is unreadable: {type(exc).__name__}"
    required_columns = {"analysis_status", "analysis_json"}
    if not required_columns <= columns:
        return "required mango_calls_ready.sqlite package DB misses analyzed call columns"
    return ""


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def ensure_nightly_config() -> str:
    current_reason = validate_nightly_config()
    if not current_reason:
        return ""
    if not STAGING_TIMELINE_DB.is_file():
        return f"cannot rebuild nightly config: staging DB is missing: {STAGING_TIMELINE_DB}"
    out_root = STAGING_ROOT / "nightly_dv2_sources"
    command = (
        sys.executable,
        "scripts/build_customer_timeline_nightly_dv2_sources.py",
        "--out-root",
        str(out_root),
        "--timeline-db",
        str(STAGING_TIMELINE_DB),
        "--base-service-config",
        str(NIGHTLY_BASE_CONFIG),
        "--service-config-out",
        str(NIGHTLY_DV2_CONFIG),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "CUSTOMER_TIMELINE_NIGHTLY_HOME": str(NIGHTLY_HOME)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        return f"nightly config rebuild failed: command_rc={completed.returncode}"
    rebuilt_reason = validate_nightly_config()
    return rebuilt_reason or ""


def build_task_spec(
    task: str,
    *,
    tallanto_phone_limit: int,
    nightly_stop_reason: str | None = None,
) -> TaskSpec:
    if task == "mail-download":
        return TaskSpec(
            task=task,
            command=(
                sys.executable,
                "scripts/run_customer_timeline_mail_download.py",
                "--apply",
                "--data-root",
                str(MAIL_DATA_ROOT),
                "--state-dir",
                str(MAIL_STATE_DIR),
            ),
            expected_output=MAIL_STATE_DIR / "mail_download_manifest.json",
        )
    if task == "mail-process":
        download_manifest = MAIL_STATE_DIR / "mail_download_manifest.json"
        return TaskSpec(
            task=task,
            command=(
                sys.executable,
                "scripts/run_customer_timeline_mail_process.py",
                "--data-root",
                str(MAIL_DATA_ROOT),
                "--state-dir",
                str(MAIL_STATE_DIR),
                "--timeline-db",
                str(STAGING_TIMELINE_DB),
            ),
            expected_output=MAIL_STATE_DIR / "mail_process_manifest.json",
            stop_reason=stage_manifest_stop_reason(download_manifest),
        )
    if task == "mail-import":
        process_manifest = MAIL_STATE_DIR / "mail_process_manifest.json"
        stop_reason = stage_manifest_stop_reason(process_manifest)
        return TaskSpec(
            task=task,
            command=(
                sys.executable,
                "scripts/run_customer_timeline_mail_import.py",
                "--state-dir",
                str(MAIL_STATE_DIR),
            ),
            expected_output=MAIL_STATE_DIR / "mail_import_manifest.json",
            stop_reason=stop_reason,
        )
    if task == "mail-capture":
        return TaskSpec(
            task=task,
            command=("bash", "scripts/run_customer_timeline_mail_capture_daily.sh", "--apply"),
            expected_output=STAGING_ROOT / "daily_capture" / "mail_capture_manifest.json",
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
            expected_output=STAGING_ROOT / "daily_capture" / "mango_capture_manifest.json",
        )
    if task == "tallanto-api-capture":
        if os.getenv("TALLANTO_API_CAPTURE_ENABLED", "0").strip() != "1":
            return TaskSpec(
                task=task,
                command=(),
                stop_reason="TALLANTO_API_CAPTURE_ENABLED is not 1; read-only Tallanto API capture is not configured",
            )
        out_root = STAGING_ROOT / "daily_capture" / "tallanto_api_capture"
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
            expected_output=NIGHTLY_SERVICE_ROOT / "published" / "latest_customer_timeline_snapshot.json",
            stop_reason=validate_nightly_config() if nightly_stop_reason is None else nightly_stop_reason,
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
    if payload.get("gate_passed") is False:
        return "stopped", "gate_failed"
    if status == "partial":
        return "stopped", "partial"
    failed_required = payload.get("failed_required_steps")
    if isinstance(failed_required, Sequence) and not isinstance(failed_required, (str, bytes)) and failed_required:
        return "stopped", "failed_required_steps"
    failed_sources = payload.get("failed_required_sources")
    if isinstance(failed_sources, Sequence) and not isinstance(failed_sources, (str, bytes)) and failed_sources:
        return "stopped", "failed_required_sources"
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


def prod_snapshot_staleness_metric(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if not PROD_TIMELINE_DB.exists():
        return "prod_snapshot_alert=missing"
    mtime = datetime.fromtimestamp(PROD_TIMELINE_DB.stat().st_mtime, timezone.utc)
    age_hours = max(0.0, (now - mtime).total_seconds() / 3600.0)
    status = "alert" if age_hours > PROD_SNAPSHOT_STALE_HOURS else "ok"
    return f"prod_snapshot_age_hours={age_hours:.1f}; prod_snapshot_staleness={status}"


def task_success_age_metric(task: str, *, status: str, finished: datetime) -> str:
    TASK_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = TASK_STATE_ROOT / f"{task}.json"
    if status == "ok":
        payload = {"task": task, "last_success_at": finished.isoformat()}
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(state_path)
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        last_success = datetime.fromisoformat(str(payload["last_success_at"]).replace("Z", "+00:00"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return "last_success_age_hours=no_data; last_success_status=alert"
    age_hours = max(0.0, (finished - last_success.astimezone(timezone.utc)).total_seconds() / 3600.0)
    freshness = "alert" if age_hours > TASK_SUCCESS_STALE_HOURS else "ok"
    return f"last_success_age_hours={age_hours:.1f}; last_success_status={freshness}"


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
    extra_metrics: str = "",
) -> Path:
    FOTON_DAILY.mkdir(parents=True, exist_ok=True)
    stamp = finished.strftime("%Y%m%dT%H%M%SZ")
    path = FOTON_DAILY / f"{stamp}_{task}.md"
    command_text = " ".join(command) if command else "not_run"
    metrics_text = metrics if not extra_metrics else f"{metrics}; {extra_metrics}"
    lines = [
        f"Задача: {task}; статус: {status}; rc={rc}; старт={started.isoformat()}; финиш={finished.isoformat()}.",
        f"Команда: `{command_text}`.",
        f"Новое/курсоры/метрики: {metrics_text}.",
        f"Аномалия/stop-причина: {stop_reason or 'нет'}.",
        f"Лог: `{log_path}`; ожидаемый артефакт: `{expected_output}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_task(task: str, *, tallanto_phone_limit: int) -> int:
    nightly_stop_reason = ensure_nightly_config() if task == "nightly-warehouse" else None
    spec = build_task_spec(
        task,
        tallanto_phone_limit=tallanto_phone_limit,
        nightly_stop_reason=nightly_stop_reason,
    )
    started = datetime.now(timezone.utc)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_ROOT / f"{stamp}_{task}.log"
    payload: Mapping[str, Any] = {}
    rc = 78 if spec.stop_reason else 0
    combined = ""
    if spec.command and not spec.stop_reason:
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
    extra_metrics = [task_success_age_metric(task, status=status, finished=finished)]
    if task == "nightly-warehouse":
        extra_metrics.append(prod_snapshot_staleness_metric(finished))
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
        extra_metrics="; ".join(extra_metrics),
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
        choices=(
            "mail-download",
            "mail-process",
            "mail-import",
            "mail-capture",
            "mango-capture",
            "tallanto-api-capture",
            "nightly-warehouse",
        ),
    )
    parser.add_argument("--tallanto-phone-limit", type=int, default=250)
    args = parser.parse_args(argv)
    return run_task(args.task, tallanto_phone_limit=args.tallanto_phone_limit)


if __name__ == "__main__":
    raise SystemExit(main())
