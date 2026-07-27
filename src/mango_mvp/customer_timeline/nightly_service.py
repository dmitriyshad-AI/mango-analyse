from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NoReturn, Optional, Sequence

from mango_mvp.customer_timeline.amo_incremental import AmoIncrementalConfig, run_amo_incremental
from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    run_nightly_incremental,
    summarize_report,
)
from mango_mvp.customer_timeline.mail_link_enrich import MailLinkEnrichConfig, run_mail_link_enrich
from mango_mvp.customer_timeline.bot_safe_summary import BotSafeSummaryBuildConfig, build_bot_safe_summaries
from mango_mvp.customer_timeline.family_graph import FamilyGraphConfig, build_family_graph
from mango_mvp.customer_timeline.tallanto_attendance_import import (
    TallantoAttendanceApiIncrementConfig,
    TallantoAttendanceImportConfig,
    run_tallanto_attendance_api_increment,
    run_tallanto_attendance_import,
)
from mango_mvp.customer_timeline.tallanto_cards_sync import (
    TallantoCardsSyncConfig,
    run_tallanto_cards_sync,
)
from mango_mvp.customer_timeline.wappi_history_import import (
    WappiFetchLimits,
    WappiHistoryImportConfig,
    run_wappi_history_import,
)

GIT_CONTEXT_ENV_KEYS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def _repo_python_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in GIT_CONTEXT_ENV_KEYS:
        env.pop(key, None)
    src = str(repo_root.resolve(strict=False) / "src")
    existing = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (src, existing) if part)
    return env


from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path, is_customer_timeline_prod_path


NIGHTLY_SERVICE_SCHEMA_VERSION = "customer_timeline_nightly_service_v1"

# B1: schema version of the *input* nightly service JSON config (as opposed
# to NIGHTLY_SERVICE_SCHEMA_VERSION above, which stamps the *output* run
# report). scripts/run_customer_timeline_codex_task.py:validate_nightly_config
# rejects any on-disk config whose "config_schema_version" does not match
# this constant, so a config produced by an older builder (e.g. one written
# before required_manifest_sources existed) fails validation instead of
# silently passing, and ensure_nightly_config() rebuilds it. Bump this
# whenever a field validate_nightly_config() now requires is added.
NIGHTLY_SERVICE_CONFIG_SCHEMA_VERSION = "customer_timeline_nightly_service_config_v3"

# B3: safe default total wall-clock budget for one full nightly run. Enforced
# by the *external* process-group wrapper
# (scripts/run_customer_timeline_codex_task.py:run_with_runtime_budget), not
# by this module -- run_nightly_service has no general way to bound a hang
# inside a step that never shells out to a subprocess (e.g. an API call with
# no read timeout). Carried on NightlyServiceConfig purely so a change to the
# budget is part of service_config_fingerprint() and therefore forces a fresh
# run instead of resuming under stale timing assumptions.
DEFAULT_TOTAL_RUNTIME_BUDGET_SECONDS = 6.0 * 3600.0

# B4: how long a source's ingestion cursor may go without a fresh check-in
# (ingestion_cursors.updated_at, refreshed on every clean run even when zero
# new records were found -- see CustomerTimelineSQLiteStore.upsert_ingestion_cursor)
# before a "successful no-op" stops counting as proof the source is healthy.
# Set generously above the ~24h nightly cadence so one missed/late run does
# not immediately flip a healthy, quiet source to "stale".
SOURCE_PROOF_STALE_AFTER_HOURS = 36.0


@dataclass(frozen=True)
class NightlyServiceStep:
    name: str
    kind: str
    enabled: bool = True
    required: bool = True
    config: Optional[NightlyIncrementalConfig] = None
    monitor_config: Optional[Mapping[str, Any]] = None
    mail_link_config: Optional[MailLinkEnrichConfig] = None
    mango_sweep_config: Optional[Mapping[str, Any]] = None
    amo_incremental_config: Optional[AmoIncrementalConfig] = None
    attendance_config: Optional[TallantoAttendanceImportConfig] = None
    tallanto_money_api_config: Optional[Mapping[str, Any]] = None
    tallanto_cards_config: Optional[TallantoCardsSyncConfig] = None
    tallanto_attendance_api_config: Optional[TallantoAttendanceApiIncrementConfig] = None
    wappi_history_config: Optional[WappiHistoryImportConfig] = None
    family_graph_config: Optional[FamilyGraphConfig] = None
    bot_safe_rebuild_config: Optional[BotSafeSummaryBuildConfig] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class NightlyServiceConfig:
    timeline_db: Path
    allowed_root: Path
    out_root: Path
    publish_dir: Path
    steps: Sequence[NightlyServiceStep]
    tenant_id: str = "foton"
    lock_timeout_seconds: float = 30.0
    actor: str = "customer_timeline_nightly_service"
    # B2: bounds any single step's external subprocess call so a stalled
    # external read (Tallanto API importer, Mango sweep producer) cannot hang
    # the whole nightly run forever. Backward compatible: old configs get the
    # same generous default and behave exactly as before for fast steps.
    step_timeout_seconds: float = 1800.0
    # B2: opt-in list of business-source labels (see
    # REQUIRED_MANIFEST_SOURCE_STEP_MAP) that must show status "ok" in this
    # run before the manifest is allowed to publish as latest. Empty by
    # default so existing narrow/test configs are unaffected.
    required_manifest_sources: Sequence[str] = ()
    # B3: see DEFAULT_TOTAL_RUNTIME_BUDGET_SECONDS above.
    total_runtime_budget_seconds: float = DEFAULT_TOTAL_RUNTIME_BUDGET_SECONDS


def run_nightly_service(config: NightlyServiceConfig) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    timeline_db, allowed_root, out_root, publish_dir = validated_service_paths(config)
    timeline_db.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    publish_dir.mkdir(parents=True, exist_ok=True)
    timeline_db.parent.chmod(0o700)
    out_root.chmod(0o700)
    publish_dir.chmod(0o700)
    if timeline_db.exists():
        timeline_db.chmod(0o600)
    # B2: fingerprint of the full normalized, immutable service config (every
    # step field, target DB/allowed/out/publish paths, tenant, required
    # sources, timeouts -- never secrets or env *values*, only env *paths*).
    # Computed once and reused for both resume matching and progress.json, so
    # a config edit (even one that only touches a single step's parameters)
    # is guaranteed to invalidate any interrupted run instead of silently
    # resuming under stale assumptions.
    config_fingerprint = service_config_fingerprint(
        config,
        timeline_db=timeline_db,
        allowed_root=allowed_root,
        out_root=out_root,
        publish_dir=publish_dir,
    )
    with service_lock(timeline_db, timeout_seconds=config.lock_timeout_seconds) as lock_info:
        # B2 resume: reuse an interrupted run's directory/run_id and carry its
        # already-"ok" leading steps forward instead of redoing them from
        # step 1. Selection happens only now, *after* the lock is held, so
        # two concurrent starts can never both pick the same run (the loser
        # blocks on the lock above and re-evaluates from scratch once it
        # acquires it, by which point a finished run has a service_report.json
        # and is no longer resumable). See find_resumable_run for the full
        # eligibility rules (fingerprint match, leading "ok" prefix, DB
        # checkpoint + quick_check match).
        resumed_run_id, resumed_run_dir, resumed_steps = find_resumable_run(
            out_root, config_fingerprint, timeline_db=timeline_db
        )
        if resumed_run_id is not None and resumed_run_dir is not None:
            run_id = resumed_run_id
            run_dir = resumed_run_dir
        else:
            run_dir = out_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir.chmod(0o700)
        report: dict[str, Any] = {
            "schema_version": NIGHTLY_SERVICE_SCHEMA_VERSION,
            "run_id": run_id,
            "started_at": started.isoformat(),
            "timeline_db": str(timeline_db),
            "allowed_root": str(allowed_root),
            "out_root": str(out_root),
            "publish_dir": str(publish_dir),
            "tenant_id": config.tenant_id,
            "resumed_from_run_id": resumed_run_id,
            "steps": list(resumed_steps),
            "safety": {
                "writes_prod_db": False,
                "writes_crm": False,
                "writes_tallanto": False,
                "sends_messages": False,
                "writes_staging_db": True,
                "installs_launchd": False,
            },
        }
        report["lock"] = lock_info
        failed_required_steps: list[str] = []
        skip_count = len(resumed_steps)
        for index, step in enumerate(config.steps, start=1):
            if index <= skip_count:
                continue
            # B2 progress: durable checkpoint reflecting every step completed
            # so far (plus a lightweight DB+WAL stat checkpoint), written
            # atomically before the next step starts (and once more after the
            # loop) so a killed/hung process leaves a resumable, inspectable
            # trail instead of silence.
            write_progress(
                run_dir,
                run_id=run_id,
                total_steps=len(config.steps),
                completed_steps=report["steps"],
                config_fingerprint=config_fingerprint,
                timeline_db=timeline_db,
            )
            step_started = time.monotonic()
            if not step.enabled:
                status = "failed_required_disabled" if step.required else "skipped_disabled"
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "reason": step.reason,
                        "duration_seconds": 0.0,
                    }
                )
                continue
            if step.kind == "local_freshness_monitor":
                try:
                    step_report = run_local_freshness_monitor(
                        step,
                        timeline_db=timeline_db,
                        allowed_root=allowed_root,
                        tenant_id=config.tenant_id,
                        actor=config.actor,
                    )
                except Exception as exc:  # service-level fail-soft: report and keep manifest writing.
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("status") == "ok" else (
                    "failed" if step.required else "skipped_optional_failed"
                )
                if status == "failed":
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": step_report,
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "mango_processed_sweep":
                try:
                    step_report = run_mango_processed_sweep(
                        step,
                        timeline_db=timeline_db,
                        allowed_root=allowed_root,
                        tenant_id=config.tenant_id,
                        step_timeout_seconds=config.step_timeout_seconds,
                    )
                except Exception as exc:  # service-level fail-soft: report and keep manifest writing.
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("status") == "ready" else (
                    "failed" if step.required else "skipped_optional_failed"
                )
                if status == "failed":
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "status": step_report.get("status"),
                            "events_written": step_report.get("events_written"),
                            "dbs_scanned": step_report.get("dbs_scanned"),
                            "dbs_selected_after_cursor": step_report.get("dbs_selected_after_cursor"),
                            "cursor": step_report.get("cursor"),
                            "output_jsonl": step_report.get("output_jsonl"),
                            "manifest_path": step_report.get("manifest_path"),
                            "known_rerun_noise": step_report.get("known_rerun_noise"),
                            "safety": step_report.get("safety"),
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "mail_link_enrich":
                if step.mail_link_config is None:
                    reason = f"enabled step {step.name} requires config"
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=reason,
                            duration_seconds=round(time.monotonic() - step_started, 3),
                        )
                    )
                    continue
                try:
                    step_report = run_mail_link_enrich(step.mail_link_config)
                except Exception as exc:  # optional enrichment must fail-soft through the service report.
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                safety = step_report.get("safety") if isinstance(step_report.get("safety"), Mapping) else {}
                changed_visibility = bool(safety.get("allowed_for_bot_changed")) or bool(
                    safety.get("mail_stage2_allowed_for_bot_changed")
                )
                status = "failed_visibility_changed" if changed_visibility else "ok"
                if changed_visibility and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "target_events": step_report.get("target_events"),
                            "counts": step_report.get("counts"),
                            "apply_counts": (step_report.get("apply") or {}).get("counts")
                            if isinstance(step_report.get("apply"), Mapping)
                            else {},
                            "safety": safety,
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "amo_incremental":
                try:
                    if step.amo_incremental_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = run_amo_incremental(step.amo_incremental_config)
                    step_ok = amo_incremental_report_ok(step_report)
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_ok else ("failed" if step.required else "skipped_optional_failed")
                if status == "failed":
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "cursor_before": step_report.get("cursor_before"),
                            "cursor_after": step_report.get("cursor_after"),
                            "fetch": step_report.get("fetch"),
                            "repeat_run_duplicates": step_report.get("repeat_run_duplicates"),
                            "safety": step_report.get("safety"),
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "tallanto_attendance":
                try:
                    if step.attendance_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = run_tallanto_attendance_import(step.attendance_config)
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("validation_ok") else "failed"
                if status == "failed" and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": step_report,
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "tallanto_money_api":
                try:
                    if step.tallanto_money_api_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = run_tallanto_money_api_step(
                        step,
                        timeline_db=timeline_db,
                        allowed_root=allowed_root,
                        tenant_id=config.tenant_id,
                        step_timeout_seconds=config.step_timeout_seconds,
                    )
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("validation_ok") else "failed"
                if status == "failed" and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "status": step_report.get("summary", {}).get("status"),
                            "records_loaded": step_report.get("summary", {}).get("records_loaded"),
                            "api": step_report.get("api"),
                            "safety": step_report.get("safety"),
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "tallanto_cards":
                try:
                    if step.tallanto_cards_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = run_tallanto_cards_sync(step.tallanto_cards_config)
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("validation_ok") else (
                    "partial" if step_report.get("apply_blocked") else "failed"
                )
                if status != "ok" and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "checked": step_report.get("checked"),
                            "updated": step_report.get("updated"),
                            "unchanged": step_report.get("unchanged"),
                            "unmatched": step_report.get("unmatched"),
                            "blocked_reason": step_report.get("blocked_reason"),
                            "safety": step_report.get("safety"),
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "tallanto_attendance_api":
                try:
                    if step.tallanto_attendance_api_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = run_tallanto_attendance_api_increment(step.tallanto_attendance_api_config)
                    step_partial = step_report.get("status") == "partial"
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "partial" if step_partial else (
                    "ok" if step_report.get("validation_ok") else "failed"
                )
                if status in {"failed", "partial"} and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "status": step_report.get("status"),
                            "unresolved_count": step_report.get("unresolved_count"),
                            "cursor_before": step_report.get("cursor_before"),
                            "cursor_after": step_report.get("cursor_after"),
                            "counts": step_report.get("counts"),
                            "safety": step_report.get("safety"),
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "wappi_history":
                try:
                    if step.wappi_history_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = run_wappi_history_import(step.wappi_history_config)
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("validation_ok") else "failed"
                if status == "failed" and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": step_report.get("summary"),
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "family_graph":
                try:
                    if step.family_graph_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = build_family_graph(step.family_graph_config)
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                status = "ok" if step_report.get("quick_check") == "ok" else "failed"
                if status == "failed" and step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": status,
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": step_report,
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind == "bot_safe_rebuild":
                try:
                    if step.bot_safe_rebuild_config is None:
                        raise ValueError(f"enabled step {step.name} requires config")
                    step_report = build_bot_safe_summaries(step.bot_safe_rebuild_config).to_json_dict()
                except Exception as exc:
                    if step.required:
                        failed_required_steps.append(step.name)
                    report["steps"].append(
                        failed_step_report(
                            index=index,
                            step=step,
                            reason=f"step_exception:{type(exc).__name__}",
                            duration_seconds=round(time.monotonic() - step_started, 3),
                            error=exc,
                        )
                    )
                    continue
                step_path = run_dir / f"{index:02d}_{step.name}.json"
                write_json(step_path, step_report)
                report["steps"].append(
                    {
                        "index": index,
                        "name": step.name,
                        "kind": step.kind,
                        "status": "ok",
                        "required": step.required,
                        "report_path": str(step_path),
                        "summary": {
                            "considered_customers": step_report.get("considered_customers"),
                            "customers_with_summary": step_report.get("customers_with_summary"),
                            "created": step_report.get("created"),
                            "updated": step_report.get("updated"),
                            "duplicate": step_report.get("duplicate"),
                            "retired_stale": step_report.get("retired_stale"),
                        },
                        "duration_seconds": round(time.monotonic() - step_started, 3),
                    }
                )
                continue
            if step.kind != "nightly_incremental":
                reason = f"unsupported nightly service step kind: {step.kind}"
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    failed_step_report(
                        index=index,
                        step=step,
                        reason=reason,
                        duration_seconds=round(time.monotonic() - step_started, 3),
                    )
                )
                continue
            if step.config is None:
                reason = f"enabled step {step.name} requires config"
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    failed_step_report(
                        index=index,
                        step=step,
                        reason=reason,
                        duration_seconds=round(time.monotonic() - step_started, 3),
                    )
                )
                continue
            try:
                step_report = run_nightly_incremental(step.config)
            except Exception as exc:  # service-level fail-soft: report and keep manifest writing.
                if step.required:
                    failed_required_steps.append(step.name)
                report["steps"].append(
                    failed_step_report(
                        index=index,
                        step=step,
                        reason=f"step_exception:{type(exc).__name__}",
                        duration_seconds=round(time.monotonic() - step_started, 3),
                        error=exc,
                    )
                )
                continue
            step_path = run_dir / f"{index:02d}_{step.name}.json"
            write_json(step_path, step_report)
            summary = summarize_report(step_report)
            step_failed_required_sources = summary.get("failed_required_sources") or ()
            step_gate_passed = bool(summary.get("gate_passed", True))
            status = "ok" if step_gate_passed else "failed_required_source"
            if not step_gate_passed and step.required:
                failed_required_steps.append(step.name)
            report["steps"].append(
                {
                    "index": index,
                    "name": step.name,
                    "kind": step.kind,
                    "status": status,
                    "required": step.required,
                    "report_path": str(step_path),
                    "summary": summary,
                    "failed_required_sources": list(step_failed_required_sources),
                    "duration_seconds": round(time.monotonic() - step_started, 3),
                }
            )
        write_progress(
            run_dir,
            run_id=run_id,
            total_steps=len(config.steps),
            completed_steps=report["steps"],
            config_fingerprint=config_fingerprint,
            timeline_db=timeline_db,
        )
        # B5: PRAGMA quick_check runs exactly once here, right before the
        # publish decision (the only other place it runs is find_resumable_run,
        # at most once, before accepting a resume). Built before the required
        # sources check below so that check can use real DB-observed counts
        # and cursors (manifest["source_counts"]/["ingestion_cursors"]) as
        # proof, instead of trusting each step's self-reported status alone.
        manifest = build_snapshot_manifest(timeline_db, tenant_id=config.tenant_id)
        # B4 fail-loud: the 10 mandatory business sources are checked against
        # *proof* -- real timeline_events counts/ingestion_cursors freshness
        # and each step's own reported numbers -- not merely whether a
        # mapped step's name/status says "ok" (opt-in via
        # config.required_manifest_sources). This is what stops one shared
        # step (e.g. wappi_history_incremental) from silently vouching for
        # two independent sources (Telegram and MAX) when only one of them
        # actually has fresh data, and what stops a step that always
        # self-reports "ok" from covering for a source whose cursor has not
        # moved in weeks.
        required_sources_check = check_required_manifest_sources(
            report["steps"],
            config.required_manifest_sources,
            source_counts=manifest["source_counts"],
            ingestion_cursors=manifest["ingestion_cursors"],
            mail_link_enrich=manifest["mail_link_enrich"],
            now=datetime.now(timezone.utc),
        )
        report["required_sources_check"] = required_sources_check
        failed_required_steps.extend(
            f"required_manifest_source:{label}" for label in required_sources_check["missing"]
        )
        # B5: a corrupted staging DB must never publish "latest", even if
        # every individual step reported ok -- integrity of the file the bot
        # reads for freshness is a harder requirement than any one step's
        # self-reported status.
        if manifest.get("quick_check") != "ok":
            failed_required_steps.append("timeline_db_quick_check")
        manifest["run_id"] = run_id
        manifest["service_report_path"] = str(run_dir / "service_report.json")
        manifest["published_at"] = datetime.now(timezone.utc).isoformat()
        manifest["required_sources_check"] = required_sources_check
        failed_required_steps = list(dict.fromkeys(failed_required_steps))
        report["failed_required_steps"] = failed_required_steps
        report["partial_failure"] = bool(failed_required_steps)
        report["overall_status"] = "partial" if failed_required_steps else "ok"
        manifest_path = publish_dir / f"customer_timeline_snapshot_{run_id}.json"
        write_json(manifest_path, manifest)
        latest_path = publish_dir / "latest_customer_timeline_snapshot.json"
        latest_published = not failed_required_steps
        if latest_published:
            # B5: never shutil.copyfile() straight over the readable "latest"
            # path -- that writes into the destination in place, so a reader
            # (or a crash mid-copy) can observe a truncated/corrupt file.
            # Write to a temp file in the same directory, fsync it, then
            # os.replace() it over the destination so "latest" is always
            # either the previous good snapshot or the new one, never
            # something in between. When latest_published is False, this
            # branch never runs at all, so a partial run leaves the previous
            # "latest" byte-for-byte untouched.
            atomic_publish_latest(manifest_path, latest_path)
        report["snapshot_manifest"] = {
            "path": str(manifest_path),
            "latest_path": str(latest_path) if latest_published else None,
            "latest_published": latest_published,
            "sha256": manifest["files"]["sqlite"]["sha256"],
            "counts": manifest["counts"],
        }
        finished = datetime.now(timezone.utc)
        report["finished_at"] = finished.isoformat()
        report["duration_seconds"] = round((finished - started).total_seconds(), 3)
        write_json(run_dir / "service_report.json", report)
    return report


def service_config_from_json(path: Path) -> NightlyServiceConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("nightly service config must be a JSON object")
    timeline_db = Path(str(payload["timeline_db"]))
    allowed_root = Path(str(payload.get("allowed_root") or timeline_db.parent))
    out_root = Path(str(payload["out_root"]))
    publish_dir = Path(str(payload["publish_dir"]))
    tenant_id = str(payload.get("tenant_id") or "foton")
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("nightly service config must contain non-empty steps")
    steps = tuple(
        service_step_from_json(
            item,
            timeline_db=timeline_db,
            allowed_root=allowed_root,
            tenant_id=tenant_id,
            actor=str(payload.get("actor") or "customer_timeline_nightly_service"),
        )
        for item in steps_payload
    )
    config = NightlyServiceConfig(
        timeline_db=timeline_db,
        allowed_root=allowed_root,
        out_root=out_root,
        publish_dir=publish_dir,
        steps=steps,
        tenant_id=tenant_id,
        lock_timeout_seconds=float(payload.get("lock_timeout_seconds", 30.0)),
        actor=str(payload.get("actor") or "customer_timeline_nightly_service"),
        step_timeout_seconds=float(payload.get("step_timeout_seconds", 1800.0)),
        required_manifest_sources=tuple(
            str(item) for item in payload.get("required_manifest_sources") or ()
        ),
        total_runtime_budget_seconds=float(
            payload.get("total_runtime_budget_seconds", DEFAULT_TOTAL_RUNTIME_BUDGET_SECONDS)
        ),
    )
    validated_service_paths(config)
    return config


def service_step_from_json(
    payload: Any,
    *,
    timeline_db: Path,
    allowed_root: Path,
    tenant_id: str,
    actor: str,
) -> NightlyServiceStep:
    if not isinstance(payload, Mapping):
        raise ValueError("nightly service step must be an object")
    name = str(payload.get("name") or "")
    kind = str(payload.get("kind") or "")
    if not name or not kind:
        raise ValueError("nightly service step requires name and kind")
    enabled = bool(payload.get("enabled", True))
    required = bool(payload.get("required", True))
    reason = str(payload.get("reason")) if payload.get("reason") else None
    config = None
    monitor_config = None
    mail_link_config = None
    mango_sweep_config = None
    amo_incremental_config = None
    attendance_config = None
    tallanto_money_api_config = None
    tallanto_cards_config = None
    tallanto_attendance_api_config = None
    wappi_history_config = None
    family_graph_config = None
    bot_safe_rebuild_config = None
    if kind == "nightly_incremental":
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError(f"step {name} requires config")
        sources_payload = config_payload.get("sources")
        if not isinstance(sources_payload, list) or not sources_payload:
            raise ValueError(f"step {name} requires non-empty sources")
        config = NightlyIncrementalConfig(
            timeline_db=Path(str(config_payload.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(config_payload.get("allowed_root") or allowed_root)),
            sources=tuple(source_from_json(item, tenant_id=tenant_id) for item in sources_payload),
            journal_path=Path(str(config_payload["journal_path"])),
            tenant_id=str(config_payload.get("tenant_id") or tenant_id),
            safety_margin_seconds=int(config_payload.get("safety_margin_seconds", 300)),
            lock_timeout_seconds=float(config_payload.get("lock_timeout_seconds", 30.0)),
            actor=str(config_payload.get("actor") or actor),
        )
    elif kind == "local_freshness_monitor":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        monitor_config = dict(raw_config)
    elif kind == "mail_link_enrich":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        mail_link_config = MailLinkEnrichConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            out_dir=Path(str(raw_config.get("out_dir") or Path(allowed_root) / "mail_link_enrich")),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            apply=bool(raw_config.get("apply", True)),
            max_events=int(raw_config["max_events"]) if raw_config.get("max_events") is not None else None,
            tallanto_identity_dbs=tuple(Path(str(path)) for path in raw_config.get("tallanto_identity_dbs", ())),
        )
    elif kind == "mango_processed_sweep":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        mango_sweep_config = dict(raw_config)
    elif kind == "amo_incremental":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        amo_incremental_config = AmoIncrementalConfig(
            source_db=Path(str(raw_config.get("source_db") or timeline_db)),
            out_root=Path(str(raw_config["out_root"])),
            mcp_env=Path(str(raw_config["mcp_env"])),
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            mcp_transport=str(raw_config["mcp_transport"]) if raw_config.get("mcp_transport") else None,
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            safety_overlap_seconds=int(raw_config.get("safety_overlap_seconds", 300)),
            page_limit=int(raw_config.get("page_limit", 50)),
            max_pages=int(raw_config.get("max_pages", 20)),
            sleep_sec=float(raw_config.get("sleep_sec", 1.05)),
            copy_db=False,
        )
    elif kind == "tallanto_attendance":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        attendance_config = TallantoAttendanceImportConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            contacts_workbook=Path(str(raw_config["contacts_workbook"])),
            attendance_report=Path(str(raw_config["attendance_report"])),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            apply=bool(raw_config.get("apply", True)),
            actor=str(raw_config.get("actor") or actor),
            tallanto_env_file=Path(str(raw_config["tallanto_env_file"])) if raw_config.get("tallanto_env_file") else None,
        )
    elif kind == "tallanto_attendance_api":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        initial_since = datetime.fromisoformat(str(raw_config["initial_since"]).replace("Z", "+00:00"))
        if initial_since.tzinfo is None:
            raise ValueError(f"step {name} initial_since must be timezone-aware")
        tallanto_attendance_api_config = TallantoAttendanceApiIncrementConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            tallanto_env_file=Path(str(raw_config["tallanto_env_file"])).expanduser(),
            initial_since=initial_since,
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            apply=bool(raw_config.get("apply", True)),
            actor=str(raw_config.get("actor") or actor),
        )
    elif kind == "tallanto_cards":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        tallanto_cards_config = TallantoCardsSyncConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            out_root=Path(str(raw_config["out_root"])),
            tallanto_env_file=Path(str(raw_config["tallanto_env_file"])).expanduser(),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            select_fields=tuple(str(item) for item in raw_config.get("select_fields") or ()),
            max_pages=int(raw_config.get("max_pages", 5)),
            safety_margin_seconds=int(raw_config.get("safety_margin_seconds", 300)),
            actor=str(raw_config.get("actor") or actor),
        )
    elif kind == "tallanto_money_api":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        tallanto_money_api_config = dict(raw_config)
    elif kind == "wappi_history":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        wappi_history_config = WappiHistoryImportConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            env_file=Path(str(raw_config["env_file"])),
            phase1_config=Path(str(raw_config["phase1_config"])),
            pairs_file=Path(str(raw_config["pairs_file"])) if raw_config.get("pairs_file") else None,
            auto_pairs_file=Path(str(raw_config["auto_pairs_file"])) if raw_config.get("auto_pairs_file") else None,
            amo_auto_resolver_enabled=bool(raw_config.get("amo_auto_resolver_enabled", False)),
            amo_mcp_env_file=Path(str(raw_config["amo_mcp_env_file"])) if raw_config.get("amo_mcp_env_file") else None,
            shared_phone_stoplist=Path(str(raw_config["shared_phone_stoplist"])) if raw_config.get("shared_phone_stoplist") else None,
            apply=bool(raw_config.get("apply", True)),
            require_nonempty_profiles=bool(raw_config.get("require_nonempty_profiles", True)),
            require_widget_linkage=bool(raw_config.get("require_widget_linkage", True)),
            widget_link_db=Path(str(raw_config["widget_link_db"])) if raw_config.get("widget_link_db") else None,
            refresh_widget_links=bool(raw_config.get("refresh_widget_links", True)),
            actor=str(raw_config.get("actor") or actor),
            out_path=Path(str(raw_config["out_path"])) if raw_config.get("out_path") else None,
            checkpoint_dir=Path(str(raw_config["checkpoint_dir"])) if raw_config.get("checkpoint_dir") else None,
            limits=WappiFetchLimits(
                chat_limit_per_profile=int(raw_config.get("chat_limit_per_profile", 5000)),
                messages_per_chat=int(raw_config.get("messages_per_chat", 100)),
                message_limit_total=int(raw_config.get("message_limit_total", 50000)),
                request_limit_total=int(raw_config.get("request_limit_total", 10000)),
                page_size=int(raw_config.get("page_size", 100)),
                sleep_seconds=float(raw_config.get("sleep_seconds", 0.2)),
                show_all_chats=bool(raw_config.get("show_all_chats", True)),
                complete_message_history=bool(raw_config.get("complete_message_history", False)),
            ),
        )
    elif kind == "family_graph":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        family_graph_config = FamilyGraphConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            out_path=Path(str(raw_config["out_path"])) if raw_config.get("out_path") else None,
            profiles_db=Path(str(raw_config["profiles_db"])) if raw_config.get("profiles_db") else None,
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            apply=bool(raw_config.get("apply", True)),
        )
    elif kind == "bot_safe_rebuild":
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"step {name} requires config")
        bot_safe_rebuild_config = BotSafeSummaryBuildConfig(
            timeline_db=Path(str(raw_config.get("timeline_db") or timeline_db)),
            allowed_root=Path(str(raw_config.get("allowed_root") or allowed_root)),
            tenant_id=str(raw_config.get("tenant_id") or tenant_id),
            apply=bool(raw_config.get("apply", True)),
            limit=int(raw_config["limit"]) if raw_config.get("limit") is not None else None,
        )
    elif enabled:
        raise ValueError(f"unsupported enabled step kind: {kind}")
    return NightlyServiceStep(
        name=name,
        kind=kind,
        enabled=enabled,
        required=required,
        config=config,
        monitor_config=monitor_config,
        mail_link_config=mail_link_config,
        mango_sweep_config=mango_sweep_config,
        amo_incremental_config=amo_incremental_config,
        attendance_config=attendance_config,
        tallanto_money_api_config=tallanto_money_api_config,
        tallanto_cards_config=tallanto_cards_config,
        tallanto_attendance_api_config=tallanto_attendance_api_config,
        wappi_history_config=wappi_history_config,
        family_graph_config=family_graph_config,
        bot_safe_rebuild_config=bot_safe_rebuild_config,
        reason=reason,
    )


def source_from_json(payload: Any, *, tenant_id: str) -> IncrementalSourceConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("source config item must be an object")
    return IncrementalSourceConfig(
        name=str(payload.get("name") or payload["source_system"]),
        source_system=str(payload["source_system"]),
        path=Path(str(payload["path"])),
        tenant_id=str(payload.get("tenant_id") or tenant_id),
        source_ref=str(payload["source_ref"]) if payload.get("source_ref") else None,
        normalizer=str(payload.get("normalizer") or "jsonl"),
        required=bool(payload.get("required", True)),
        ignore_cursor=bool(payload.get("ignore_cursor", False)),
        preserve_cursor=bool(payload.get("preserve_cursor", False)),
    )


def failed_step_report(
    *,
    index: int,
    step: NightlyServiceStep,
    reason: str,
    duration_seconds: float,
    error: Exception | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "index": index,
        "name": step.name,
        "kind": step.kind,
        "status": "failed" if step.required else "skipped_optional_failed",
        "required": step.required,
        "reason": reason,
        "duration_seconds": duration_seconds,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        diagnostics_path = getattr(error, "diagnostics_path", None)
        if diagnostics_path:
            payload["error_diagnostics_path"] = str(diagnostics_path)
    return payload


def validated_service_paths(config: NightlyServiceConfig) -> tuple[Path, Path, Path, Path]:
    allowed_root = Path(config.allowed_root).expanduser().resolve(strict=False)
    timeline_db = guard_customer_timeline_output_path(config.timeline_db, allowed_root)
    if is_customer_timeline_prod_path(timeline_db):
        raise ValueError(f"nightly service must not target prod DB paths: {timeline_db}")
    out_root = guard_customer_timeline_output_path(config.out_root, allowed_root)
    publish_dir = guard_customer_timeline_output_path(config.publish_dir, allowed_root)
    for step in config.steps:
        if step.config is None:
            if step.monitor_config is not None:
                for raw_path in step.monitor_config.get("paths") or ():
                    guard_customer_timeline_output_path(Path(str(raw_path)), allowed_root)
                if step.monitor_config.get("metrics_path"):
                    guard_customer_timeline_output_path(Path(str(step.monitor_config["metrics_path"])), allowed_root)
            if step.mail_link_config is not None:
                guard_customer_timeline_output_path(step.mail_link_config.timeline_db, allowed_root)
                guard_customer_timeline_output_path(step.mail_link_config.allowed_root, allowed_root)
                guard_customer_timeline_output_path(step.mail_link_config.out_dir, allowed_root)
            if step.mango_sweep_config is not None:
                for key in ("out_jsonl", "report_out", "manifest_path", "inventory_out"):
                    if step.mango_sweep_config.get(key):
                        guard_customer_timeline_output_path(Path(str(step.mango_sweep_config[key])), allowed_root)
            if step.amo_incremental_config is not None:
                guard_customer_timeline_output_path(step.amo_incremental_config.timeline_db, allowed_root)
                guard_customer_timeline_output_path(step.amo_incremental_config.allowed_root, allowed_root)
                guard_customer_timeline_output_path(step.amo_incremental_config.out_root, allowed_root)
            if step.attendance_config is not None:
                guard_customer_timeline_output_path(step.attendance_config.timeline_db, allowed_root)
            if step.tallanto_attendance_api_config is not None:
                guard_customer_timeline_output_path(step.tallanto_attendance_api_config.timeline_db, allowed_root)
                guard_customer_timeline_output_path(step.tallanto_attendance_api_config.allowed_root, allowed_root)
            if step.tallanto_cards_config is not None:
                guard_customer_timeline_output_path(step.tallanto_cards_config.timeline_db, allowed_root)
                guard_customer_timeline_output_path(step.tallanto_cards_config.allowed_root, allowed_root)
                guard_customer_timeline_output_path(step.tallanto_cards_config.out_root, allowed_root)
            if step.wappi_history_config is not None:
                guard_customer_timeline_output_path(step.wappi_history_config.timeline_db, allowed_root)
                if step.wappi_history_config.widget_link_db is not None:
                    guard_customer_timeline_output_path(step.wappi_history_config.widget_link_db, allowed_root)
                if step.wappi_history_config.checkpoint_dir is not None:
                    guard_customer_timeline_output_path(step.wappi_history_config.checkpoint_dir, allowed_root)
            if step.family_graph_config is not None:
                guard_customer_timeline_output_path(step.family_graph_config.timeline_db, allowed_root)
                if step.family_graph_config.out_path is not None:
                    guard_customer_timeline_output_path(step.family_graph_config.out_path, allowed_root)
            if step.bot_safe_rebuild_config is not None:
                guard_customer_timeline_output_path(step.bot_safe_rebuild_config.timeline_db, allowed_root)
                guard_customer_timeline_output_path(step.bot_safe_rebuild_config.allowed_root, allowed_root)
            continue
        guard_customer_timeline_output_path(step.config.timeline_db, allowed_root)
        guard_customer_timeline_output_path(step.config.allowed_root, allowed_root)
        guard_customer_timeline_output_path(step.config.journal_path, allowed_root)
        for source in step.config.sources:
            guard_customer_timeline_output_path(source.path, allowed_root)
    return timeline_db, allowed_root, out_root, publish_dir


def amo_incremental_report_ok(report: Mapping[str, Any]) -> bool:
    if report.get("validation_ok") is not True or report.get("complete") is not True or report.get("apply_blocked"):
        return False
    safety = report.get("safety") if isinstance(report.get("safety"), Mapping) else {}
    if any(safety.get(key) is not False for key in ("amo_write", "tallanto_write", "crm_write")):
        return False
    fetch = report.get("fetch") if isinstance(report.get("fetch"), Mapping) else {}
    if len(fetch) != 3 or any(
        not isinstance(item, Mapping)
        or item.get("complete") is not True
        or item.get("page_cap_hit")
        or item.get("pagination_drift_detected")
        for item in fetch.values()
    ):
        return False
    reports = [report.get("second_run")]
    first = report.get("first_run") if isinstance(report.get("first_run"), Mapping) else {}
    reports.extend((first.get("cards"), first.get("events")))
    return not any(
        isinstance(item, Mapping) and item.get("source_errors")
        for item in reports
    )


def run_local_freshness_monitor(
    step: NightlyServiceStep,
    *,
    timeline_db: Path,
    allowed_root: Path,
    tenant_id: str,
    actor: str,
) -> Mapping[str, Any]:
    from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
    from mango_mvp.customer_timeline.nightly_incremental import parse_datetime

    config = dict(step.monitor_config or {})
    paths = [guard_customer_timeline_output_path(Path(str(item)), allowed_root) for item in config.get("paths") or ()]
    metrics_path = (
        guard_customer_timeline_output_path(Path(str(config["metrics_path"])), allowed_root)
        if config.get("metrics_path")
        else None
    )
    existing_paths = [path for path in paths if path.exists()]
    metrics: Mapping[str, Any] = {}
    if metrics_path and metrics_path.exists():
        try:
            parsed = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            metrics = {"error": "invalid_json"}
    status = "ok" if existing_paths or metrics else str(config.get("empty_status") or "skipped")
    cursor_source_system = str(config.get("cursor_source_system") or "").strip()
    cursor_ts_raw = str(config.get("cursor_ts") or "").strip()
    cursor_written = None
    deprecated_removed: list[str] = []
    if cursor_source_system and cursor_ts_raw:
        cursor_ts = parse_datetime(cursor_ts_raw, "cursor_ts")
        with CustomerTimelineSQLiteStore(timeline_db, allowed_root=allowed_root) as store:
            for raw_source in config.get("deprecated_cursor_source_systems") or ():
                legacy_source = str(raw_source or "").strip()
                if not legacy_source or legacy_source == cursor_source_system:
                    continue
                removed = store._con.execute(
                    "DELETE FROM ingestion_cursors WHERE tenant_id = ? AND source_system = ?",
                    (tenant_id, legacy_source),
                )
                if removed.rowcount:
                    deprecated_removed.append(legacy_source)
            cursor = store.upsert_ingestion_cursor(
                tenant_id,
                cursor_source_system,
                last_cursor_ts=cursor_ts,
                metadata={
                    "last_status": status,
                    "monitor_step": step.name,
                    "reason": config.get("reason"),
                    "metrics_path": str(metrics_path) if metrics_path else None,
                },
                actor=actor,
            )
            store._con.commit()
            cursor_written = cursor.to_json_dict()
    return {
        "schema_version": "customer_timeline_local_freshness_monitor_v1",
        "name": step.name,
        "kind": step.kind,
        "required": step.required,
        "status": status,
        "reason": config.get("reason"),
        "paths": [
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
            }
            for path in paths
        ],
        "metrics_path": str(metrics_path) if metrics_path else None,
        "metrics": metrics,
        "cursor": cursor_written,
        "deprecated_cursors_removed": deprecated_removed,
        "safety": {
            "network_calls": False,
            "writes_timeline_events": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "runs_asr": False,
        },
    }


def run_tallanto_money_api_step(
    step: NightlyServiceStep,
    *,
    timeline_db: Path,
    allowed_root: Path,
    tenant_id: str,
    step_timeout_seconds: float = 1800.0,
) -> Mapping[str, Any]:
    config = dict(step.tallanto_money_api_config or {})
    importer_script = Path(str(config.get("importer_script") or "")).expanduser().resolve(strict=False)
    env_file = Path(str(config.get("tallanto_env_file") or "")).expanduser().resolve(strict=False)
    configured_db = Path(str(config.get("timeline_db") or timeline_db)).expanduser().resolve(strict=False)
    configured_root = Path(str(config.get("allowed_root") or allowed_root)).expanduser().resolve(strict=False)
    if configured_db != timeline_db.resolve(strict=False) or configured_root != allowed_root.resolve(strict=False):
        raise ValueError("Tallanto money API step must use the service staging DB and allowed root")
    if config.get("apply") is not True:
        raise ValueError("Tallanto money API step must explicitly apply to staging")
    if not importer_script.is_file() or not env_file.is_file():
        raise FileNotFoundError("Tallanto money API importer or read-only env is missing")
    command = [
        sys.executable,
        str(importer_script),
        "--tallanto-api-env",
        str(env_file),
        "--timeline-db",
        str(timeline_db),
        "--allowed-root",
        str(allowed_root),
        "--tenant-id",
        tenant_id,
        "--apply",
        "--actor",
        str(config.get("actor") or "customer_timeline_nightly_tallanto_money"),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=importer_script.parents[1],
            env=_repo_python_env(importer_script.parents[1]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=step_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        _raise_tallanto_money_failure(
            timeline_db,
            allowed_root,
            failure_kind="timeout",
            stdout=exc.stdout,
            stderr=exc.stderr,
            cause=exc,
        )
    if proc.returncode != 0:
        _raise_tallanto_money_failure(
            timeline_db, allowed_root, failure_kind="nonzero_exit",
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        _raise_tallanto_money_failure(
            timeline_db, allowed_root, failure_kind="invalid_json",
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        )
    if not isinstance(report, Mapping):
        _raise_tallanto_money_failure(
            timeline_db, allowed_root, failure_kind="non_object_report",
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        )
    safety = report.get("safety") if isinstance(report.get("safety"), Mapping) else {}
    if safety.get("write_tallanto") is not False or safety.get("write_product_timeline_db") is not True:
        _raise_tallanto_money_failure(
            timeline_db, allowed_root, failure_kind="safety_contract_failed",
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        )
    return report


def _raise_tallanto_money_failure(
    timeline_db: Path,
    allowed_root: Path,
    *,
    failure_kind: str,
    returncode: int | None = None,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    cause: Exception | None = None,
) -> NoReturn:
    stdout_bytes = stdout if isinstance(stdout, bytes) else str(stdout or "").encode("utf-8", errors="replace")
    stderr_bytes = stderr if isinstance(stderr, bytes) else str(stderr or "").encode("utf-8", errors="replace")
    diagnostics_path = guard_customer_timeline_output_path(
        timeline_db.parent / "tallanto_money_api_failure.json", allowed_root
    )
    write_json(
        diagnostics_path,
        {
            "schema_version": "tallanto_money_api_failure_v1",
            "failure_kind": failure_kind,
            "returncode": returncode,
            "stdout_bytes": len(stdout_bytes),
            "stderr_bytes": len(stderr_bytes),
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
            "raw_output_persisted": False,
        },
    )
    error = cause or RuntimeError(f"Tallanto money API importer failed: {failure_kind}")
    error.diagnostics_path = diagnostics_path  # type: ignore[attr-defined]
    raise error


def run_mango_processed_sweep(
    step: NightlyServiceStep,
    *,
    timeline_db: Path,
    allowed_root: Path,
    tenant_id: str,
    step_timeout_seconds: float = 1800.0,
) -> Mapping[str, Any]:
    config = dict(step.mango_sweep_config or {})
    out_jsonl = guard_customer_timeline_output_path(
        Path(str(config.get("out_jsonl") or Path(allowed_root) / "nightly_dv2_sources" / "mango_processed_sweep.jsonl")),
        allowed_root,
    )
    report_out = guard_customer_timeline_output_path(
        Path(str(config.get("report_out") or out_jsonl.with_suffix(".producer_report.json"))),
        allowed_root,
    )
    manifest_path = guard_customer_timeline_output_path(
        Path(str(config.get("manifest_path") or out_jsonl.with_suffix(".manifest.json"))),
        allowed_root,
    )
    inventory_out = guard_customer_timeline_output_path(
        Path(str(config.get("inventory_out") or out_jsonl.with_suffix(".inventory.json"))),
        allowed_root,
    )
    producer_script = Path(str(config.get("producer_script") or Path.cwd() / "scripts" / "build_mango_call_timeline_increment.py"))
    if not producer_script.exists():
        return mango_sweep_manifest(
            status="failed",
            reason="producer_script_missing",
            timeline_db=timeline_db,
            out_jsonl=out_jsonl,
            report_out=report_out,
            manifest_path=manifest_path,
            inventory_out=inventory_out,
            cursor={},
            inventory=[],
            producer_report={},
            command=(),
            rc=127,
        )
    cursor = mango_processed_cursor(timeline_db, tenant_id=tenant_id)
    # ponytail: source timestamps cannot reveal a call analyzed after its old call date.
    # Re-read analyzed rows and rely on stable event deduplication instead.
    since = ""
    scan_roots = tuple(Path(str(item)).expanduser() for item in config.get("scan_roots") or ())
    package_globs = tuple(str(item) for item in config.get("package_globs") or ("mango_update_after_*",))
    inventory = discover_mango_processed_call_dbs(scan_roots, package_globs=package_globs, since=since)
    seen_package_dbs = {
        Path(str(item["db_path"])).expanduser().resolve(strict=False)
        for item in inventory
        if item.get("db_path")
    }
    since_dt = parse_iso_datetime(since) if since else None
    for raw_db in config.get("package_dbs") or ():
        db_path = Path(str(raw_db)).expanduser().resolve(strict=False)
        if db_path in seen_package_dbs:
            continue
        seen_package_dbs.add(db_path)
        inventory.append(inspect_mango_call_db(db_path.parent, db_path, since_dt=since_dt))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    inventory_out.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    explicit_package_dbs = {
        Path(str(item)).expanduser().resolve(strict=False)
        for item in config.get("package_dbs") or ()
    }
    unusable_explicit = [
        item
        for item in inventory
        if Path(str(item.get("db_path") or "")).expanduser().resolve(strict=False) in explicit_package_dbs
        and not item.get("usable")
    ]
    if unusable_explicit:
        manifest = mango_sweep_manifest(
            status="failed",
            reason="explicit_package_db_unusable",
            timeline_db=timeline_db,
            out_jsonl=out_jsonl,
            report_out=report_out,
            manifest_path=manifest_path,
            inventory_out=inventory_out,
            cursor=cursor,
            inventory=inventory,
            producer_report={"unusable_explicit_package_dbs": unusable_explicit},
            command=(),
            rc=78,
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    package_dbs = [item["db_path"] for item in inventory if item.get("usable") and int(item.get("selected_after_cursor") or 0) > 0]
    command = [
        sys.executable,
        str(producer_script),
        "--timeline-db",
        str(timeline_db),
        "--out-jsonl",
        str(out_jsonl),
        "--report-out",
        str(report_out),
        "--tenant-id",
        tenant_id,
    ]
    if since:
        command.extend(["--since", since])
    for db_path in package_dbs:
        command.extend(["--package-db", str(db_path)])
    repo_root = producer_script.parents[1]
    proc = subprocess.run(
        command,
        cwd=repo_root,
        env=_repo_python_env(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=step_timeout_seconds,
    )
    producer_report: Mapping[str, Any] = {}
    if report_out.exists():
        try:
            parsed = json.loads(report_out.read_text(encoding="utf-8"))
            producer_report = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            producer_report = {"error": "invalid_producer_report_json"}
    status = "ready" if proc.returncode == 0 and out_jsonl.exists() else "failed"
    manifest = mango_sweep_manifest(
        status=status,
        reason="" if status == "ready" else "producer_failed",
        timeline_db=timeline_db,
        out_jsonl=out_jsonl,
        report_out=report_out,
        manifest_path=manifest_path,
        inventory_out=inventory_out,
        cursor=cursor,
        inventory=inventory,
        producer_report=producer_report,
        command=tuple(command),
        rc=proc.returncode,
        stdout_tail="\n".join((proc.stdout or "").splitlines()[-40:]),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def mango_sweep_manifest(
    *,
    status: str,
    reason: str,
    timeline_db: Path,
    out_jsonl: Path,
    report_out: Path,
    manifest_path: Path,
    inventory_out: Path,
    cursor: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    producer_report: Mapping[str, Any],
    command: Sequence[str],
    rc: int,
    stdout_tail: str = "",
) -> Mapping[str, Any]:
    return {
        "schema_version": "mango_processed_sweep_v1",
        "status": status,
        "reason": reason,
        "timeline_db": str(timeline_db),
        "cursor": dict(cursor),
        "dbs_scanned": len(inventory),
        "dbs_selected_after_cursor": sum(1 for item in inventory if int(item.get("selected_after_cursor") or 0) > 0),
        "rows_selected_after_cursor": sum(int(item.get("selected_after_cursor") or 0) for item in inventory),
        "events_written": int(producer_report.get("events_written") or 0),
        "source_counts": dict(producer_report.get("source_counts") or {}),
        "identity_resolution_counts": dict(producer_report.get("identity_resolution_counts") or {}),
        "call_type_counts": dict(producer_report.get("call_type_counts") or {}),
        "output_jsonl": str(out_jsonl),
        "producer_report": str(report_out),
        "manifest_path": str(manifest_path),
        "inventory_out": str(inventory_out),
        "producer_rc": rc,
        "producer_command": list(command),
        "producer_stdout_tail": stdout_tail,
        "known_rerun_noise": {
            "amocrm_snapshot_safety_window_updated": True,
            "note": "calls rerun acceptance must use mango_processed_summary event/chunk deltas; AMO contact snapshots may report updated rows inside the 5-minute overlap.",
        },
        "safety": {
            "network_calls": False,
            "writes_timeline_db": False,
            "writes_prod_db": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "sends_messages": False,
            "runs_asr": False,
            "runs_analyze": False,
        },
    }


def mango_processed_cursor(db_path: Path, *, tenant_id: str) -> Mapping[str, Any]:
    row = None
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        row = con.execute(
            """
            SELECT last_cursor_ts, updated_at, metadata_json
            FROM ingestion_cursors
            WHERE tenant_id = ? AND source_system = 'mango_processed_summary'
            """,
            (tenant_id,),
        ).fetchone()
    if row is None:
        return {"source_system": "mango_processed_summary", "last_cursor_ts": None, "max_source_ts": None}
    metadata = {}
    try:
        parsed = json.loads(str(row["metadata_json"] or "{}"))
        metadata = parsed.get("metadata") if isinstance(parsed, Mapping) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
    except json.JSONDecodeError:
        metadata = {}
    return {
        "source_system": "mango_processed_summary",
        "last_cursor_ts": row["last_cursor_ts"],
        "updated_at": row["updated_at"],
        "max_source_ts": metadata.get("max_source_ts"),
    }


def discover_mango_processed_call_dbs(
    scan_roots: Sequence[Path],
    *,
    package_globs: Sequence[str],
    since: str,
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[Path] = set()
    since_dt = parse_iso_datetime(since) if since else None
    for scan_root in scan_roots:
        for pattern in package_globs:
            for root in sorted(scan_root.glob(pattern)):
                if not root.is_dir():
                    continue
                for db_path in sorted((root / "asr_ui_batch").glob("*.sqlite")):
                    resolved = db_path.expanduser().resolve(strict=False)
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    result.append(inspect_mango_call_db(root, resolved, since_dt=since_dt))
    return result


def inspect_mango_call_db(root: Path, db_path: Path, *, since_dt: datetime | None) -> Mapping[str, Any]:
    if ".before_" in db_path.name or "before_" in db_path.name:
        return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "backup_db_name"}
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only=ON")
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='call_records'").fetchone() is None:
                return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "missing_call_records"}
            cols = {str(row[1]) for row in con.execute("PRAGMA table_info(call_records)")}
            if "analysis_status" not in cols or "analysis_json" not in cols:
                return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "missing_analysis_columns"}
            date_col = next((col for col in ("started_at", "call_at", "event_at") if col in cols), None)
            if not date_col:
                return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "missing_call_datetime"}
            rows = con.execute(
                f"""
                SELECT {date_col} AS call_at, analysis_status, analysis_json
                FROM call_records
                WHERE {date_col} IS NOT NULL AND TRIM({date_col}) != ''
                """
            ).fetchall()
    except sqlite3.Error as exc:
        return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": f"sqlite_error:{type(exc).__name__}"}
    total = len(rows)
    done_rows = [
        row
        for row in rows
        if str(row["analysis_status"] or "") == "done" and str(row["analysis_json"] or "").strip()
    ]
    for row in done_rows:
        if parse_iso_datetime(str(row["call_at"] or "")) is None:
            return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "invalid_done_call_datetime"}
        try:
            analysis = json.loads(str(row["analysis_json"] or ""))
        except json.JSONDecodeError:
            analysis = None
        if not isinstance(analysis, Mapping):
            return {"root": str(root), "db_path": str(db_path), "usable": False, "skip_reason": "invalid_done_analysis_json"}
    selected = 0
    min_at = None
    max_at = None
    for row in done_rows:
        raw = str(row["call_at"] or "")
        parsed = parse_iso_datetime(raw)
        if parsed is None:
            continue
        min_at = parsed if min_at is None else min(min_at, parsed)
        max_at = parsed if max_at is None else max(max_at, parsed)
        if since_dt is None or parsed >= since_dt:
            selected += 1
    return {
        "root": str(root),
        "db_path": str(db_path),
        "usable": True,
        "rows_total": total,
        "analysis_done": len(done_rows),
        "min_started_at": min_at.isoformat() if min_at else None,
        "max_started_at": max_at.isoformat() if max_at else None,
        "selected_after_cursor": selected,
        "has_ra_final_summary": any(root.rglob("RA_FINAL_SUMMARY.json")),
    }


def parse_iso_datetime(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_snapshot_manifest(db_path: Path, *, tenant_id: str) -> dict[str, Any]:
    import sqlite3

    db = Path(db_path)
    now = datetime.now(timezone.utc)
    files = {}
    for suffix, label in (("", "sqlite"), ("-wal", "wal"), ("-shm", "shm")):
        path = Path(str(db) + suffix)
        files[label] = file_fingerprint(path)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        counts = {
            "customer_identities": table_count(con, "customer_identities"),
            "timeline_events": table_count(con, "timeline_events"),
            "bot_context_chunks": table_count(con, "bot_context_chunks"),
            "identity_links": table_count(con, "identity_links"),
            "timeline_conflicts": table_count(con, "timeline_conflicts"),
            "derived_signals": table_count(con, "derived_signals"),
            "ingestion_cursors": table_count(con, "ingestion_cursors"),
        }
        source_counts = [
            dict(row)
            for row in con.execute(
                """
                SELECT source_system, COUNT(*) AS count, MAX(event_at) AS max_event_at
                FROM timeline_events
                WHERE tenant_id = ?
                GROUP BY source_system
                ORDER BY source_system
                """,
                (tenant_id,),
            )
        ]
        for row in source_counts:
            row["max_event_age_days"] = iso_age_days(row.get("max_event_at"), now=now)
        cursors = [
            dict(row)
            for row in con.execute(
                """
                SELECT source_system, last_cursor_ts, updated_at
                FROM ingestion_cursors
                WHERE tenant_id = ?
                ORDER BY source_system
                """,
                (tenant_id,),
            )
        ]
        for row in cursors:
            row["last_cursor_age_days"] = iso_age_days(row.get("last_cursor_ts"), now=now)
            row["updated_age_days"] = iso_age_days(row.get("updated_at"), now=now)
        mail_link_enrich = build_mail_link_enrich_manifest_metrics(con, tenant_id=tenant_id)
    return {
        "schema_version": "customer_timeline_snapshot_manifest_v1",
        "timeline_db": str(db),
        "quick_check": quick_check,
        "files": files,
        "counts": counts,
        "source_counts": source_counts,
        "ingestion_cursors": cursors,
        "mail_link_enrich": mail_link_enrich,
    }


def build_mail_link_enrich_manifest_metrics(con: Any, *, tenant_id: str) -> Mapping[str, Any]:
    try:
        outcome_rows = con.execute(
            """
            SELECT COALESCE(json_extract(record_json, '$.metadata.mail_link_enrich.outcome'), 'not_processed') AS outcome,
                   COUNT(*) AS count
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = 'mail_archive_stage2'
            GROUP BY outcome
            ORDER BY outcome
            """,
            (tenant_id,),
        ).fetchall()
        brand_rows = con.execute(
            """
            SELECT COALESCE(json_extract(record_json, '$.metadata.brand'), 'unknown') AS brand,
                   COUNT(*) AS count
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = 'mail_archive_stage2'
            GROUP BY brand
            ORDER BY brand
            """,
            (tenant_id,),
        ).fetchall()
        pending_null = con.execute(
            """
            SELECT COUNT(*)
            FROM timeline_events
            WHERE tenant_id = ?
              AND source_system = 'mail_archive_stage2'
              AND match_status = 'unmatched'
              AND (customer_id IS NULL OR customer_id = '')
              AND json_extract(record_json, '$.metadata.pending_attribution') = 1
              AND json_extract(record_json, '$.metadata.pending_reason') IS NULL
            """,
            (tenant_id,),
        ).fetchone()[0]
    except Exception:
        return {"status": "unavailable"}
    outcomes = {str(row["outcome"]): int(row["count"]) for row in outcome_rows}
    brands = {str(row["brand"]): int(row["count"]) for row in brand_rows}
    return {
        "status": "ok",
        "outcomes": outcomes,
        "linked_strong": outcomes.get("strong", 0),
        "weak_email": outcomes.get("weak_email", 0),
        "unmatched": outcomes.get("unmatched", 0),
        "blocked": outcomes.get("blocked", 0),
        "brand_counts": brands,
        "unknown_brand": brands.get("unknown", 0),
        "pending_without_reason": int(pending_null),
    }


def iso_age_days(raw: Any, *, now: datetime) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return round((now - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0, 3)


def table_count(con: Any, table: str) -> int:
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return 0


def file_fingerprint(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": 0, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


@contextmanager
def service_lock(db_path: Path, *, timeout_seconds: float) -> Iterator[Mapping[str, Any]]:
    lock_path = Path(str(db_path) + ".nightly_service.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                waited = time.monotonic() - started
                if waited >= timeout_seconds:
                    raise TimeoutError(f"nightly service lock timeout: {lock_path}")
                time.sleep(0.2)
        yield {"path": str(lock_path), "waited_seconds": round(time.monotonic() - started, 3)}
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory, flush+fsync,
    then os.replace() over the destination. Used for progress.json (B2),
    per-step reports, and the numbered snapshot manifest (B5) so a killed
    process never leaves a half-written file at the final path -- readers
    always see either the previous complete file or the new complete one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_publish_latest(source_path: Path, dest_path: Path) -> None:
    """B5: promote a just-written, self-consistent manifest to be the
    "latest" snapshot bots/tools read for freshness, without ever letting a
    concurrent reader (or a mid-write crash) observe a truncated file.
    shutil.copyfile() writes directly into the destination path; a temp file
    in the same directory, fsynced then os.replace()-ed over the
    destination, keeps dest_path always either the previous good file or the
    new one, never something in between.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest_path.name}.", suffix=".tmp", dir=str(dest_path.parent))
    try:
        with os.fdopen(fd, "wb") as tmp_handle, source_path.open("rb") as src_handle:
            shutil.copyfileobj(src_handle, tmp_handle)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
        os.replace(tmp_name, dest_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# B2: business-source label -> the step name(s) that must report status "ok"
# in a given run for that source to count as fresh. A label mapped to a step
# name that no current step config uses is reported missing instead of being
# silently accepted.
REQUIRED_MANIFEST_SOURCE_STEP_MAP: Mapping[str, tuple[str, ...]] = {
    "amo_contacts_leads_events": ("amo_incremental_shadow",),
    "tallanto_cards": ("tallanto_cards_sync",),
    "tallanto_payments_subscriptions": ("tallanto_money_api_incremental",),
    "tallanto_attendance": ("tallanto_attendance_api_incremental",),
    "calls": ("mango_processed_sweep", "calls_and_amo_incremental"),
    "email": ("mail_archive_incremental", "mail_link_enrich"),
    "wappi_telegram": ("wappi_history_incremental",),
    "wappi_max": ("wappi_history_incremental",),
    "family_child_graph": ("family_graph_refresh",),
    "bot_safe_chunks_and_dossier": ("bot_safe_rebuild",),
}


def _fingerprint_value(value: Any) -> Any:
    """Recursively turn a (possibly frozen, possibly nested) dataclass /
    Path / Mapping / sequence into a plain JSON-safe structure for hashing.
    Generic over every field of NightlyServiceConfig and every step config
    dataclass, so a new field added to any of them is automatically covered
    by service_config_fingerprint() without having to remember to update a
    hand-written field list. Only ever sees paths, ids, booleans, numbers and
    already-safe raw JSON step config dicts -- never secret *values* (secrets
    live in env files referenced only by path).
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _fingerprint_value(getattr(value, field.name)) for field in dataclass_fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _fingerprint_value(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def service_config_fingerprint(
    config: NightlyServiceConfig,
    *,
    timeline_db: Path,
    allowed_root: Path,
    out_root: Path,
    publish_dir: Path,
) -> str:
    """B2: fingerprint of the full normalized, immutable service config used
    to gate resume. Covers target DB path, allowed/out/publish roots,
    tenant, required sources, every timeout, and *every field* of every
    step (not just name/kind/required/enabled) -- so changing a source path,
    a step parameter, a timeout, or the required-sources set forbids
    resuming on top of a run started under the old assumptions. The four
    path fields use the already-guarded/resolved values (not the raw config
    fields) so equivalent paths reached via different relative forms hash
    the same. Never includes secrets or env *values* -- only paths to env
    files, which is all these dataclasses ever carry.
    """
    payload = dict(_fingerprint_value(config))
    payload["timeline_db"] = str(timeline_db)
    payload["allowed_root"] = str(allowed_root)
    payload["out_root"] = str(out_root)
    payload["publish_dir"] = str(publish_dir)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _file_state(path: Path) -> Mapping[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def db_lightweight_checkpoint(db_path: Path) -> Mapping[str, Any]:
    """B2: cheap (no sqlite connection) fingerprint of the target DB's
    on-disk state -- resolved path, inode, size, mtime_ns for the main file
    and its -wal sidecar. Recomputed on every write_progress() call (i.e.
    effectively after every step), so the *last* progress.json written
    before a crash always reflects DB state as of the last recorded step.
    Deliberately not a full PRAGMA quick_check (too expensive to run after
    every step); see db_quick_check_ok for the one-time checks around resume
    accept and publish.
    """
    resolved = Path(db_path).resolve(strict=False)
    return {
        "path": str(resolved),
        "main": _file_state(resolved),
        "wal": _file_state(Path(str(resolved) + "-wal")),
    }


def db_quick_check_ok(db_path: Path) -> bool:
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5) as con:
            con.execute("PRAGMA query_only=ON")
            result = str(con.execute("PRAGMA quick_check").fetchone()[0])
    except sqlite3.Error:
        return False
    return result == "ok"


def check_required_manifest_sources(
    steps_report: Sequence[Mapping[str, Any]],
    required_labels: Sequence[str],
    *,
    source_counts: Sequence[Mapping[str, Any]] = (),
    ingestion_cursors: Sequence[Mapping[str, Any]] = (),
    mail_link_enrich: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """B4: decide each required source's freshness from *proof* -- real
    DB-observed timeline_events counts/max_event_at and ingestion_cursors
    freshness, plus each step's own reported numbers -- never from a bare
    step name/status lookup. This is what stops one shared step (e.g.
    wappi_history_incremental) from silently vouching for two independent
    business sources (Telegram and MAX) when only one of them actually has
    fresh data, and what stops a step that always self-reports "ok" from
    covering for a source whose underlying cursor has gone stale.
    """
    ctx = _SourceProofContext(
        steps_by_name={str(item.get("name")): item for item in steps_report},
        source_counts=source_counts,
        cursors=ingestion_cursors,
        mail_link_enrich=mail_link_enrich or {},
        now=now or datetime.now(timezone.utc),
    )
    missing: list[str] = []
    proofs: dict[str, Mapping[str, Any]] = {}
    detail: dict[str, Any] = {}
    for label in required_labels:
        builder = SOURCE_PROOF_BUILDERS.get(label)
        if builder is None:
            proof = _proof(
                label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                reason=f"unknown required_manifest_sources label: {label}",
            )
        else:
            proof = builder(ctx)
        proofs[label] = proof
        satisfied = proof["status"] == "ok"
        detail[label] = {
            "steps": list(REQUIRED_MANIFEST_SOURCE_STEP_MAP.get(label, ())),
            "proof": proof,
            "satisfied": satisfied,
        }
        if not satisfied:
            missing.append(label)
    return {
        "schema_version": "customer_timeline_required_manifest_sources_v2",
        "required": list(required_labels),
        "missing": missing,
        "satisfied": [label for label in required_labels if label not in missing],
        "detail": detail,
        "proofs": proofs,
    }


@dataclass(frozen=True)
class _SourceProofContext:
    steps_by_name: Mapping[str, Mapping[str, Any]]
    source_counts: Sequence[Mapping[str, Any]]
    cursors: Sequence[Mapping[str, Any]]
    mail_link_enrich: Mapping[str, Any]
    now: datetime


def _step_status(ctx: "_SourceProofContext", name: str) -> Optional[str]:
    step = ctx.steps_by_name.get(name)
    return str(step.get("status")) if step is not None else None


def _proof(
    label: str,
    ctx: "_SourceProofContext",
    *,
    status: str,
    records: int,
    cursor_or_max_event_at: Optional[str],
    reason: str,
) -> Mapping[str, Any]:
    return {
        "source_label": label,
        "checked_at": ctx.now.isoformat(),
        "status": status,
        "records_seen_or_written": records,
        "cursor_or_max_event_at": cursor_or_max_event_at,
        "source_specific_reason": reason,
    }


def _parse_iso_or_none(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _max_iso(*values: Optional[str]) -> Optional[str]:
    parsed = [(parsed_value, raw) for raw in values if raw for parsed_value in (_parse_iso_or_none(raw),) if parsed_value]
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def _cursor_row(cursors: Sequence[Mapping[str, Any]], source_system: str) -> Optional[Mapping[str, Any]]:
    for row in cursors:
        if str(row.get("source_system")) == source_system:
            return row
    return None


def _count_row(counts: Sequence[Mapping[str, Any]], source_system: str) -> tuple[int, Optional[str]]:
    for row in counts:
        if str(row.get("source_system")) == source_system:
            max_event_at = row.get("max_event_at")
            return int(row.get("count") or 0), (str(max_event_at) if max_event_at else None)
    return 0, None


def _cursor_is_fresh(cursor: Optional[Mapping[str, Any]], *, now: datetime) -> bool:
    if cursor is None:
        return False
    checked = _parse_iso_or_none(cursor.get("updated_at"))
    if checked is None:
        return False
    age_hours = (now - checked).total_seconds() / 3600.0
    return 0 <= age_hours <= SOURCE_PROOF_STALE_AFTER_HOURS


def _proof_amo_contacts_leads_events(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "amo_contacts_leads_events"
    status = _step_status(ctx, "amo_incremental_shadow")
    if status is None:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="step amo_incremental_shadow did not run in this run")
    if status != "ok":
        return _proof(label, ctx, status="error", records=0, cursor_or_max_event_at=None,
                      reason=f"amo_incremental_shadow step status={status}")
    contacts, contacts_ts = _count_row(ctx.source_counts, "amocrm_snapshot")
    events, events_ts = _count_row(ctx.source_counts, "amocrm_event")
    return _proof(
        label, ctx, status="ok", records=contacts + events,
        cursor_or_max_event_at=_max_iso(contacts_ts, events_ts),
        reason="amo_incremental_shadow ok; timeline_events counts for amocrm_snapshot+amocrm_event",
    )


def _proof_tallanto_cards(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "tallanto_cards"
    status = _step_status(ctx, "tallanto_cards_sync")
    records, event_ts = _count_row(ctx.source_counts, "tallanto_snapshot")
    cursor = _cursor_row(ctx.cursors, "tallanto_cards_daily")
    cursor_ts = str(cursor["last_cursor_ts"]) if cursor and cursor.get("last_cursor_ts") else event_ts
    if status is None:
        return _proof(
            label, ctx, status="missing", records=records, cursor_or_max_event_at=cursor_ts,
            reason="step tallanto_cards_sync did not run in this run",
        )
    if status != "ok":
        return _proof(
            label, ctx, status="error", records=records, cursor_or_max_event_at=cursor_ts,
            reason=f"tallanto_cards_sync step status={status}",
        )
    if not records or not _cursor_is_fresh(cursor, now=ctx.now):
        return _proof(
            label, ctx, status="stale", records=records, cursor_or_max_event_at=cursor_ts,
            reason="tallanto_snapshot is empty or tallanto_cards_daily cursor is stale",
        )
    return _proof(
        label, ctx, status="ok", records=records, cursor_or_max_event_at=cursor_ts,
        reason="tallanto_cards_sync ok; tallanto snapshot present and cursor fresh",
    )


def _proof_tallanto_payments_subscriptions(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "tallanto_payments_subscriptions"
    status = _step_status(ctx, "tallanto_money_api_incremental")
    if status is None:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="step tallanto_money_api_incremental did not run in this run")
    records, event_ts = _count_row(ctx.source_counts, "tallanto_crm_call")
    if status != "ok":
        return _proof(label, ctx, status="error", records=records, cursor_or_max_event_at=event_ts,
                      reason=f"tallanto_money_api_incremental step status={status}")
    if not records:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=event_ts,
                      reason="tallanto_money_api_incremental produced no Tallanto payment or subscription events")
    return _proof(
        label, ctx, status="ok", records=records, cursor_or_max_event_at=event_ts,
        reason="tallanto_money_api_incremental ok; timeline_events count for tallanto_crm_call",
    )


def _proof_tallanto_attendance(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "tallanto_attendance"
    status = _step_status(ctx, "tallanto_attendance_api_incremental")
    if status is None:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="step tallanto_attendance_api_incremental did not run in this run")
    records, event_ts = _count_row(ctx.source_counts, "tallanto_attendance_api")
    cursor = _cursor_row(ctx.cursors, "tallanto_attendance_api")
    cursor_ts = str(cursor["last_cursor_ts"]) if cursor and cursor.get("last_cursor_ts") else event_ts
    if status != "ok":
        return _proof(label, ctx, status="error", records=records, cursor_or_max_event_at=cursor_ts,
                      reason=f"tallanto_attendance_api_incremental step status={status}")
    if not records:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=cursor_ts,
                      reason="tallanto_attendance_api_incremental has no persisted attendance events")
    if not _cursor_is_fresh(cursor, now=ctx.now):
        return _proof(
            label, ctx, status="stale", records=records, cursor_or_max_event_at=cursor_ts,
            reason=(
                "ingestion_cursors.tallanto_attendance_api.updated_at is missing or older than "
                f"{SOURCE_PROOF_STALE_AFTER_HOURS:.0f}h"
            ),
        )
    return _proof(label, ctx, status="ok", records=records, cursor_or_max_event_at=cursor_ts,
                  reason="tallanto_attendance_api_incremental ok; ingestion cursor fresh")


def _proof_calls(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "calls"
    sweep_status = _step_status(ctx, "mango_processed_sweep")
    incremental_status = _step_status(ctx, "calls_and_amo_incremental")
    if sweep_status is None or incremental_status is None:
        missing_steps = [
            name
            for name, status in (
                ("mango_processed_sweep", sweep_status),
                ("calls_and_amo_incremental", incremental_status),
            )
            if status is None
        ]
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="steps did not run in this run: " + ",".join(missing_steps))
    records, event_ts = _count_row(ctx.source_counts, "mango_processed_summary")
    cursor = _cursor_row(ctx.cursors, "mango_processed_summary")
    cursor_ts = str(cursor["last_cursor_ts"]) if cursor and cursor.get("last_cursor_ts") else event_ts
    if sweep_status != "ok" or incremental_status != "ok":
        return _proof(
            label, ctx, status="error", records=records, cursor_or_max_event_at=cursor_ts,
            reason=f"mango_processed_sweep status={sweep_status}; calls_and_amo_incremental status={incremental_status}",
        )
    return _proof(label, ctx, status="ok", records=records, cursor_or_max_event_at=cursor_ts,
                  reason="mango_processed_sweep and calls_and_amo_incremental both ok")


def _proof_email(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "email"
    archive_status = _step_status(ctx, "mail_archive_incremental")
    enrich_status = _step_status(ctx, "mail_link_enrich")
    if archive_status is None or enrich_status is None:
        missing_steps = [
            name
            for name, status in (
                ("mail_archive_incremental", archive_status),
                ("mail_link_enrich", enrich_status),
            )
            if status is None
        ]
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="steps did not run in this run: " + ",".join(missing_steps))
    records, event_ts = _count_row(ctx.source_counts, "mail_archive_stage2")
    cursor = _cursor_row(ctx.cursors, "mail_archive_stage2")
    cursor_ts = str(cursor["last_cursor_ts"]) if cursor and cursor.get("last_cursor_ts") else event_ts
    if archive_status != "ok":
        return _proof(label, ctx, status="error", records=records, cursor_or_max_event_at=cursor_ts,
                      reason=f"mail_archive_incremental step status={archive_status}")
    if enrich_status != "ok":
        return _proof(label, ctx, status="error", records=records, cursor_or_max_event_at=cursor_ts,
                      reason=f"mail_link_enrich step status={enrich_status}")
    enrich_metrics_status = str(ctx.mail_link_enrich.get("status") or "")
    if enrich_metrics_status != "ok":
        return _proof(
            label, ctx, status="error", records=records, cursor_or_max_event_at=cursor_ts,
            reason=f"mail_link_enrich manifest metrics unavailable: status={enrich_metrics_status or 'unknown'}",
        )
    return _proof(
        label, ctx, status="ok", records=records, cursor_or_max_event_at=cursor_ts,
        reason="mail_archive_incremental and mail_link_enrich both ok; archive+link-enrich metrics present",
    )


def _proof_wappi_channel(ctx: "_SourceProofContext", *, label: str, source_system: str) -> Mapping[str, Any]:
    status = _step_status(ctx, "wappi_history_incremental")
    if status is None:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="step wappi_history_incremental did not run in this run")
    records, event_ts = _count_row(ctx.source_counts, source_system)
    if status != "ok":
        return _proof(label, ctx, status="error", records=records, cursor_or_max_event_at=event_ts,
                      reason=f"wappi_history_incremental step status={status}")
    if records <= 0:
        return _proof(
            label, ctx, status="missing", records=records, cursor_or_max_event_at=event_ts,
            reason=f"wappi_history_incremental ok, but timeline_events has zero {source_system} rows",
        )
    return _proof(label, ctx, status="ok", records=records, cursor_or_max_event_at=event_ts,
                  reason=f"wappi_history_incremental ok; timeline_events count for {source_system}")


def _proof_wappi_telegram(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    return _proof_wappi_channel(ctx, label="wappi_telegram", source_system="wappi_telegram")


def _proof_wappi_max(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    return _proof_wappi_channel(ctx, label="wappi_max", source_system="wappi_max")


def _proof_family_child_graph(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "family_child_graph"
    step = ctx.steps_by_name.get("family_graph_refresh")
    if step is None:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="step family_graph_refresh did not run in this run")
    status = str(step.get("status"))
    summary = step.get("summary") if isinstance(step.get("summary"), Mapping) else {}
    quick_check = summary.get("quick_check")
    if status != "ok":
        return _proof(label, ctx, status="error", records=0, cursor_or_max_event_at=None,
                      reason=f"family_graph_refresh step status={status}; quick_check={quick_check}")
    records = 0
    for key in ("edges_written", "pairs_written", "family_edges", "written", "rows_written"):
        candidate = summary.get(key)
        if isinstance(candidate, int):
            records = candidate
            break
    return _proof(label, ctx, status="ok", records=records, cursor_or_max_event_at=None,
                  reason=f"family_graph_refresh ok; quick_check={quick_check}")


def _proof_bot_safe_chunks_and_dossier(ctx: "_SourceProofContext") -> Mapping[str, Any]:
    label = "bot_safe_chunks_and_dossier"
    step = ctx.steps_by_name.get("bot_safe_rebuild")
    if step is None:
        return _proof(label, ctx, status="missing", records=0, cursor_or_max_event_at=None,
                      reason="step bot_safe_rebuild did not run in this run")
    status = str(step.get("status"))
    summary = step.get("summary") if isinstance(step.get("summary"), Mapping) else {}
    if status != "ok":
        return _proof(label, ctx, status="error", records=0, cursor_or_max_event_at=None,
                      reason=f"bot_safe_rebuild step status={status}")
    records = int(summary.get("customers_with_summary") or 0)
    return _proof(label, ctx, status="ok", records=records, cursor_or_max_event_at=None,
                  reason="bot_safe_rebuild ok; customers_with_summary from step summary")


SOURCE_PROOF_BUILDERS: Mapping[str, Callable[["_SourceProofContext"], Mapping[str, Any]]] = {
    "amo_contacts_leads_events": _proof_amo_contacts_leads_events,
    "tallanto_cards": _proof_tallanto_cards,
    "tallanto_payments_subscriptions": _proof_tallanto_payments_subscriptions,
    "tallanto_attendance": _proof_tallanto_attendance,
    "calls": _proof_calls,
    "email": _proof_email,
    "wappi_telegram": _proof_wappi_telegram,
    "wappi_max": _proof_wappi_max,
    "family_child_graph": _proof_family_child_graph,
    "bot_safe_chunks_and_dossier": _proof_bot_safe_chunks_and_dossier,
}


def find_resumable_run(
    out_root: Path,
    config_fingerprint: str,
    *,
    timeline_db: Path,
) -> tuple[Optional[str], Optional[Path], list[Mapping[str, Any]]]:
    """Find the newest interrupted run_dir under out_root that can be resumed.

    Must be called only while holding the service lock (see
    run_nightly_service) so two concurrent starts can never both pick the
    same run. A run is resumable only if:

    - it never finished (no service_report.json). A finished run -- even one
      that ended "partial" -- is never reused: every step is designed to be
      idempotent, so a fresh run is always safe and simplest.
    - it left a progress.json whose config_fingerprint matches the full
      normalized *current* service config exactly (target DB, allowed/out/
      publish roots, tenant, required sources, every timeout, and every
      field of every step -- not just step name/kind/required/enabled).
    - it has at least one leading step recorded with status "ok". Only that
      leading "ok" prefix is trusted; anything after the first non-"ok"
      entry is re-attempted rather than carried forward.
    - its saved db_checkpoint (lightweight stat of the DB file + WAL, taken
      right after that leading "ok" prefix completed) matches the DB file's
      *current* on-disk state exactly, and the DB currently passes
      PRAGMA quick_check (run at most once here, only when there is an
      otherwise-eligible candidate to accept). A progress.json predating
      this checkpoint field, any stat mismatch, or a failed quick_check all
      mean "something touched the DB since" and force a fresh, independently
      idempotent run instead of resuming on top of an unknown DB state.
    """
    if not out_root.is_dir():
        return None, None, []
    for candidate in sorted((path for path in out_root.glob("run_*") if path.is_dir()), reverse=True):
        if (candidate / "service_report.json").exists():
            continue
        progress_path = candidate / "progress.json"
        if not progress_path.exists():
            continue
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(progress, Mapping) or progress.get("config_fingerprint") != config_fingerprint:
            continue
        run_id = str(progress.get("run_id") or "")
        completed = progress.get("steps")
        if not run_id or not isinstance(completed, list):
            continue
        resumable_prefix: list[Mapping[str, Any]] = []
        for item in completed:
            if isinstance(item, Mapping) and item.get("status") == "ok":
                resumable_prefix.append(item)
            else:
                break
        if not resumable_prefix:
            continue
        saved_checkpoint = progress.get("db_checkpoint")
        if not isinstance(saved_checkpoint, Mapping):
            continue  # pre-checkpoint progress.json: fail safe, do not resume
        if saved_checkpoint != db_lightweight_checkpoint(timeline_db):
            continue
        if not db_quick_check_ok(timeline_db):
            continue
        return run_id, candidate, resumable_prefix
    return None, None, []


def write_progress(
    run_dir: Path,
    *,
    run_id: str,
    total_steps: int,
    completed_steps: Sequence[Mapping[str, Any]],
    config_fingerprint: str,
    timeline_db: Path,
) -> None:
    completed_count = len(completed_steps)
    payload = {
        "schema_version": "customer_timeline_nightly_service_progress_v2",
        "run_id": run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_steps": total_steps,
        "completed_steps": completed_count,
        "next_step_index": completed_count + 1 if completed_count < total_steps else None,
        "config_fingerprint": config_fingerprint,
        "steps": list(completed_steps),
        # B2: lightweight (stat-only) DB+WAL checkpoint taken *now*, i.e.
        # reflecting DB state as of the most recently completed step in
        # `steps` above. Compared byte-for-byte against a fresh checkpoint at
        # resume time; see find_resumable_run.
        "db_checkpoint": db_lightweight_checkpoint(timeline_db),
    }
    write_json(run_dir / "progress.json", payload)
