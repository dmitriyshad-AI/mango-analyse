#!/usr/bin/env python3
"""Run one Customer Timeline dev-schedule task with log and daily summary.

This is the single command used by Codex dev automations and, later, by the
launchd templates while the pipeline is still being stabilized.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.nightly_service import (  # noqa: E402
    DEFAULT_TALLANTO_CARDS_MAX_PAGES,
    DEFAULT_TOTAL_RUNTIME_BUDGET_SECONDS,
    NIGHTLY_SERVICE_CONFIG_SCHEMA_VERSION,
    REQUIRED_MANIFEST_SOURCE_STEP_MAP,
)

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
REQUIRED_MUTATING_NIGHTLY_CHAIN = (
    ("wappi_history_incremental", "wappi_history"),
    ("family_graph_refresh", "family_graph"),
    ("derived_signals_refresh", "derived_signals"),
    ("bot_safe_rebuild", "bot_safe_rebuild"),
)
REQUIRED_NIGHTLY_STEPS = {
    "mango_processed_sweep",
    "calls_and_amo_incremental",
    "wappi_history_incremental",
    "mail_archive_incremental",
    "mail_link_enrich",
    "tallanto_money_api_incremental",
    "tallanto_cards_sync",
    "tallanto_attendance_api_incremental",
    "family_graph_refresh",
    "derived_signals_refresh",
    "bot_safe_rebuild",
}
REQUIRED_CALL_SOURCES = {"mango_processed_summary": "mango_processed_summary"}
EXPECTED_NIGHTLY_CONFIG_SCHEMA_VERSION = NIGHTLY_SERVICE_CONFIG_SCHEMA_VERSION
REQUIRED_MANIFEST_SOURCES = frozenset(REQUIRED_MANIFEST_SOURCE_STEP_MAP)
DEFAULT_NIGHTLY_RUNTIME_BUDGET_SECONDS = DEFAULT_TOTAL_RUNTIME_BUDGET_SECONDS
# B3: bounded wait after SIGTERM before escalating to SIGKILL.
NIGHTLY_TERM_GRACE_SECONDS = 20.0
GIT_CONTEXT_ENV_KEYS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


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
            env=repo_python_env(),
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
    # B1: an old config built before this field (or before
    # required_manifest_sources) existed must never pass preflight as if it
    # were current -- that is exactly the "old nightly.json without
    # required_manifest_sources still validates" false-green bug. Checked
    # first, ahead of any other field, so a stale config is rejected
    # regardless of what else it happens to still get right.
    if payload.get("config_schema_version") != EXPECTED_NIGHTLY_CONFIG_SCHEMA_VERSION:
        return (
            "nightly config schema version is stale: "
            f"{payload.get('config_schema_version')!r} != {EXPECTED_NIGHTLY_CONFIG_SCHEMA_VERSION!r}"
        )
    required_sources_raw = payload.get("required_manifest_sources")
    required_sources = (
        {str(item) for item in required_sources_raw} if isinstance(required_sources_raw, list) else set()
    )
    if required_sources != REQUIRED_MANIFEST_SOURCES:
        missing_sources = sorted(REQUIRED_MANIFEST_SOURCES - required_sources)
        extra_sources = sorted(required_sources - REQUIRED_MANIFEST_SOURCES)
        return (
            "nightly config required_manifest_sources mismatch: "
            f"missing={','.join(missing_sources) or '-'} extra={','.join(extra_sources) or '-'}"
        )
    if Path(str(payload.get("timeline_db") or "")).expanduser().resolve(strict=False) != STAGING_TIMELINE_DB.resolve(
        strict=False
    ):
        return "nightly config timeline_db does not match persistent staging DB"
    if Path(str(payload.get("allowed_root") or "")).expanduser().resolve(strict=False) != STAGING_ROOT.resolve(
        strict=False
    ):
        return "nightly config allowed_root does not match persistent staging root"
    chain_reason = validate_mutating_nightly_chain(payload)
    if chain_reason:
        return chain_reason
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
    amo_config = steps.get("amo_incremental_shadow", {}).get("config")
    if not isinstance(amo_config, Mapping) or amo_config.get("max_pages") != 200:
        return "amo_incremental_shadow must read up to 200 pages per run"
    if amo_config.get("page_limit") != 20:
        return "amo_incremental_shadow page_limit must be 20 to avoid truncated MCP responses"
    wappi_config = steps["wappi_history_incremental"].get("config")
    if not isinstance(wappi_config, Mapping) or wappi_config.get("require_widget_linkage") is not False:
        return "wappi_history_incremental must quarantine incomplete identity linkage"
    checkpoint_dir = Path(str(wappi_config.get("checkpoint_dir") or "")).expanduser().resolve(strict=False)
    if not path_is_within(checkpoint_dir, STAGING_ROOT):
        return "wappi_history_incremental checkpoint is outside persistent staging root"
    mail_config = steps["mail_link_enrich"].get("config")
    if not isinstance(mail_config, Mapping) or mail_config.get("reconsider_pending") is not True:
        return "mail_link_enrich must reconsider pending after Tallanto refresh"
    identity_dbs = tuple(
        Path(str(item)).expanduser().resolve(strict=False) for item in mail_config.get("tallanto_identity_dbs", ())
    )
    if not identity_dbs or any(not item.is_file() for item in identity_dbs):
        return "mail_link_enrich requires existing Tallanto identity DBs"
    money_step = steps["tallanto_money_api_incremental"]
    money_config = money_step.get("config")
    if money_step.get("kind") != "tallanto_money_api" or not isinstance(money_config, Mapping):
        return "tallanto_money_api_incremental has invalid kind or config"
    expected_money_importer = (ROOT / "scripts" / "import_tallanto_payments_to_timeline.py").resolve(
        strict=False
    )
    money_importer = Path(str(money_config.get("importer_script") or "")).resolve(strict=False)
    if money_importer != expected_money_importer:
        return "nightly config points Tallanto money import at another code root"
    money_db = Path(str(money_config.get("timeline_db") or "")).expanduser().resolve(strict=False)
    if money_config.get("apply") is not True or money_db != STAGING_TIMELINE_DB.resolve(strict=False):
        return "tallanto_money_api_incremental must apply to persistent staging DB"
    try:
        money_timeout = float(money_config.get("timeout_seconds") or 0)
    except (TypeError, ValueError):
        money_timeout = 0
    if money_timeout < 5400:
        return "tallanto_money_api_incremental timeout must allow the 90-minute first full backfill"
    attendance_step = steps["tallanto_attendance_api_incremental"]
    attendance_config = attendance_step.get("config")
    if attendance_step.get("kind") != "tallanto_attendance_api" or not isinstance(attendance_config, Mapping):
        return "tallanto_attendance_api_incremental has invalid kind or config"
    attendance_db = Path(str(attendance_config.get("timeline_db") or "")).expanduser().resolve(strict=False)
    if attendance_config.get("apply") is not True or attendance_db != STAGING_TIMELINE_DB.resolve(strict=False):
        return "tallanto_attendance_api_incremental must apply to persistent staging DB"
    cards_step = steps["tallanto_cards_sync"]
    cards_config = cards_step.get("config")
    if cards_step.get("kind") != "tallanto_cards" or not isinstance(cards_config, Mapping):
        return "tallanto_cards_sync has invalid kind or config"
    cards_db = Path(str(cards_config.get("timeline_db") or "")).expanduser().resolve(strict=False)
    if cards_db != STAGING_TIMELINE_DB.resolve(strict=False):
        return "tallanto_cards_sync must target persistent staging DB"
    if cards_config.get("max_pages") != DEFAULT_TALLANTO_CARDS_MAX_PAGES:
        return f"tallanto_cards_sync must read up to {DEFAULT_TALLANTO_CARDS_MAX_PAGES} pages per run"
    raw_steps = payload.get("steps") or ()
    positions_by_name = {
        str(step.get("name") or ""): index
        for index, step in enumerate(raw_steps)
        if isinstance(step, Mapping)
    }
    tallanto_chain = (
        positions_by_name["tallanto_cards_sync"],
        positions_by_name["tallanto_attendance_api_incremental"],
        positions_by_name["tallanto_money_api_incremental"],
        positions_by_name["wappi_history_incremental"],
    )
    if tallanto_chain != tuple(sorted(tallanto_chain)):
        return "nightly config requires Tallanto cards -> attendance -> money before Wappi"
    if positions_by_name["mail_link_enrich"] < positions_by_name["tallanto_cards_sync"]:
        return "nightly config requires Tallanto cards before mail pending reconsider"
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
        if source_system == "mango_processed_summary" and (
            source.get("ignore_cursor") is not True or source.get("preserve_cursor") is not True
        ):
            return "mango_processed_summary late sweep must ignore and preserve the shared cursor"
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


def validate_mutating_nightly_chain(payload: Mapping[str, Any]) -> str:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return "nightly config must contain a steps list"
    declared_db = Path(str(payload.get("timeline_db") or "")).expanduser().resolve(strict=False)
    positions: list[int] = []
    for name, expected_kind in REQUIRED_MUTATING_NIGHTLY_CHAIN:
        matches = [
            (index, step)
            for index, step in enumerate(raw_steps)
            if isinstance(step, Mapping) and step.get("name") == name
        ]
        if len(matches) != 1:
            return f"nightly config requires exactly one {name} step"
        position, step = matches[0]
        if step.get("enabled") is not True or step.get("required") is not True:
            return f"nightly config {name} must be enabled and required"
        if step.get("kind") != expected_kind:
            return f"nightly config {name} kind must be {expected_kind}"
        config = step.get("config")
        if not isinstance(config, Mapping):
            return f"nightly config {name} requires config"
        if config.get("apply") is not True:
            return f"nightly config {name} apply must be true"
        step_db = Path(str(config.get("timeline_db") or "")).expanduser().resolve(strict=False)
        if step_db != declared_db:
            return f"nightly config {name} timeline_db must match declared staging DB"
        positions.append(position)
    if positions != sorted(positions):
        names = " -> ".join(name for name, _kind in REQUIRED_MUTATING_NIGHTLY_CHAIN)
        return f"nightly config required step order must be {names}"
    return ""


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def repo_python_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in GIT_CONTEXT_ENV_KEYS:
        env.pop(key, None)
    existing = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(ROOT / "src"), existing) if part)
    return env


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
        env={**repo_python_env(), "CUSTOMER_TIMELINE_NIGHTLY_HOME": str(NIGHTLY_HOME)},
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


@dataclass(frozen=True)
class BoundedRunResult:
    rc: int
    stdout: str
    timed_out: bool


def nightly_runtime_budget_seconds(path: Path | None = None) -> float:
    """B3: read the total wall-clock budget for one nightly run straight out
    of the same JSON config the nightly service itself reads (so it is part
    of NightlyServiceConfig.total_runtime_budget_seconds and therefore of
    service_config_fingerprint -- a budget change forces a fresh run instead
    of resuming under stale timing assumptions). Falls back to a safe
    generous default for a config that predates this field.
    """
    path = path or NIGHTLY_DV2_CONFIG
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_NIGHTLY_RUNTIME_BUDGET_SECONDS
    try:
        return float(payload.get("total_runtime_budget_seconds", DEFAULT_NIGHTLY_RUNTIME_BUDGET_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_NIGHTLY_RUNTIME_BUDGET_SECONDS


def _safe_getpgid(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


def _safe_killpg(pgid: Optional[int], sig: int) -> None:
    if pgid is None:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def run_with_runtime_budget(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    budget_seconds: float,
    term_grace_seconds: float = NIGHTLY_TERM_GRACE_SECONDS,
) -> BoundedRunResult:
    """B3: run `command` as the leader of its own process group and enforce
    a hard wall-clock budget from *outside* the process -- never by
    interrupting a Python thread inside it. subprocess.run(..., timeout=...)
    alone is not enough here: on timeout it only ever targets the single
    direct child pid, so a step that itself shells out to another process
    (Tallanto/Mango importer scripts) would be orphaned and keep running.
    start_new_session=True makes the child (pid == its own new pgid) the
    group leader, so any descendant it spawns without its own setsid() call
    inherits that same pgid; killing the *group* (os.killpg) reaches all of
    them. On timeout: SIGTERM the whole group, wait a bounded grace period,
    then SIGKILL the whole group.
    """
    proc = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=budget_seconds)
        return BoundedRunResult(rc=proc.returncode, stdout=stdout or "", timed_out=False)
    except subprocess.TimeoutExpired:
        pass
    pgid = _safe_getpgid(proc.pid)
    _safe_killpg(pgid, signal.SIGTERM)
    try:
        stdout, _ = proc.communicate(timeout=term_grace_seconds)
    except subprocess.TimeoutExpired:
        _safe_killpg(pgid, signal.SIGKILL)
        try:
            stdout, _ = proc.communicate(timeout=term_grace_seconds)
        except subprocess.TimeoutExpired:
            stdout = ""
    return BoundedRunResult(rc=proc.returncode, stdout=stdout or "", timed_out=True)


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
    timed_out = False
    runtime_budget_seconds: Optional[float] = None
    if spec.command and not spec.stop_reason:
        if task == "nightly-warehouse":
            # B3: this is the "external wrapper waits forever" gap -- nightly
            # chains several steps that call out to external
            # APIs/subprocesses in-process (no per-step timeout of their
            # own, e.g. the Wappi history step), so only a wrapper-level,
            # process-group-wide budget can guarantee this task ever returns.
            runtime_budget_seconds = nightly_runtime_budget_seconds()
            result = run_with_runtime_budget(
                spec.command,
                cwd=ROOT,
                env=repo_python_env(),
                budget_seconds=runtime_budget_seconds,
            )
            rc = result.rc
            combined = result.stdout
            timed_out = result.timed_out
        else:
            proc = subprocess.run(
                spec.command,
                cwd=ROOT,
                env=repo_python_env(),
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
    # B3: a timeout always reports as "stopped"/timeout, regardless of
    # whatever partial JSON the killed process happened to print before it
    # was terminated -- latest was never reached (see
    # run_nightly_service/atomic_publish_latest), and progress.json from the
    # in-flight run is left exactly as the process last wrote it.
    effective_stop_reason = spec.stop_reason
    if timed_out:
        effective_stop_reason = (
            f"nightly_runtime_budget_exceeded_seconds={runtime_budget_seconds:.0f}"
        )
    status, reason = status_from_payload(payload, rc, effective_stop_reason)
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
