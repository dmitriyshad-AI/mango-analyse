#!/usr/bin/env python3
"""Import a fresh mail-only increment into the configured staging timeline."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_customer_timeline_mail_download import (  # noqa: E402
    atomic_write_json,
    exclusive_lock,
    runtime_identity,
    sha256_file,
    utc_now,
)
from scripts.run_customer_timeline_mail_process import staging_root_for  # noqa: E402
from mango_mvp.customer_timeline.mail_link_enrich import (  # noqa: E402
    MailLinkEnrichConfig,
    run_mail_link_enrich,
)
from mango_mvp.productization.mail_archive import (  # noqa: E402
    CANONICAL_MAIL_CURRENT_IDENTITY_DB,
    CANONICAL_MAIL_IDENTITY_DB,
    DEFAULT_MAIL_DATA_ROOT,
)


def parse_time(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp_without_timezone")
    return parsed.astimezone(timezone.utc)


def read_mail_cursor_state(timeline_db: Path) -> Mapping[str, Any] | None:
    if not timeline_db.is_file():
        return None
    try:
        immutable = not Path(f"{timeline_db.resolve()}-wal").exists()
        suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
        with sqlite3.connect(timeline_db.resolve().as_uri() + suffix, uri=True) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                SELECT tenant_id, source_system, last_cursor_ts, updated_at, metadata_json
                FROM ingestion_cursors
                WHERE tenant_id=? AND source_system=?
                """,
                ("foton", "mail_archive_stage2"),
            ).fetchone()
    except sqlite3.Error:
        return None
    return dict(row) if row else None


def read_mail_cursor(timeline_db: Path) -> str | None:
    state = read_mail_cursor_state(timeline_db)
    return str(state["last_cursor_ts"]) if state and state.get("last_cursor_ts") else None


def restore_mail_cursor(timeline_db: Path, previous: Mapping[str, Any] | None) -> None:
    with sqlite3.connect(timeline_db) as con:
        if previous is None:
            con.execute(
                "DELETE FROM ingestion_cursors WHERE tenant_id=? AND source_system=?",
                ("foton", "mail_archive_stage2"),
            )
        else:
            con.execute(
                """
                INSERT INTO ingestion_cursors (
                  tenant_id, source_system, last_cursor_ts, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, source_system) DO UPDATE SET
                  last_cursor_ts=excluded.last_cursor_ts,
                  updated_at=excluded.updated_at,
                  metadata_json=excluded.metadata_json
                """,
                (
                    previous["tenant_id"],
                    previous["source_system"],
                    previous["last_cursor_ts"],
                    previous["updated_at"],
                    previous["metadata_json"],
                ),
            )


def load_inputs(
    *, state_dir: Path, runtime: Mapping[str, str], max_age_hours: float
) -> tuple[Mapping[str, Any], Path, Mapping[str, Any]]:
    manifest_path = state_dir / "mail_process_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("process_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok" or manifest.get("runtime") != runtime:
        raise RuntimeError("process_manifest_not_current")
    age = datetime.now(timezone.utc) - parse_time(manifest.get("finished_at"))
    if age < timedelta(0) or age > timedelta(hours=max_age_hours):
        raise RuntimeError("process_manifest_stale")
    config_path = Path(str(manifest.get("config") or "")).resolve()
    process_root = (state_dir / "process").resolve()
    if process_root not in config_path.parents or not config_path.is_file():
        raise RuntimeError("process_config_missing_or_outside_state")
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise RuntimeError("process_config_sha_mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("runtime") != runtime:
        raise RuntimeError("process_config_runtime_mismatch")
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise RuntimeError("process_config_not_mail_only")
    source = sources[0]
    if (
        source.get("source_system") != "mail_archive_stage2"
        or source.get("normalizer") != "mail_archive_stage2"
        or source.get("required") is not True
    ):
        raise RuntimeError("process_config_not_mail_only")
    timeline_db = Path(str(config.get("timeline_db") or "")).resolve()
    staging_root = staging_root_for(state_dir=state_dir, timeline_db=timeline_db)
    if Path(str(config.get("allowed_root") or "")).resolve() != staging_root:
        raise RuntimeError("process_config_allowed_root_not_staging")
    source_path = Path(str(source.get("path") or "")).resolve()
    journal_path = Path(str(config.get("journal_path") or "")).resolve()
    if process_root not in source_path.parents or not source_path.is_file():
        raise RuntimeError("process_source_missing_or_outside_state")
    if process_root not in journal_path.parents:
        raise RuntimeError("process_journal_outside_state")
    if str(source_path) != str(Path(str(manifest.get("output_jsonl") or "")).resolve()):
        raise RuntimeError("process_source_manifest_mismatch")
    if sha256_file(source_path) != manifest.get("output_sha256"):
        raise RuntimeError("process_source_sha_mismatch")
    return manifest, config_path, config


def run_incremental(code_root: Path, config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(code_root / "scripts/run_customer_timeline_nightly_incremental.py"),
            "--config",
            str(config_path),
            "--summary-only",
        ],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )


def enrich_mail_links(*, timeline_db: Path, allowed_root: Path, out_dir: Path) -> Mapping[str, Any]:
    return run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=timeline_db,
            allowed_root=allowed_root,
            out_dir=out_dir,
            tenant_id="foton",
            apply=True,
            tallanto_identity_dbs=(
                DEFAULT_MAIL_DATA_ROOT / CANONICAL_MAIL_CURRENT_IDENTITY_DB,
                DEFAULT_MAIL_DATA_ROOT / CANONICAL_MAIL_IDENTITY_DB,
            ),
        )
    )


def execute(args: argparse.Namespace) -> Mapping[str, Any]:
    code_root = Path(args.code_root).resolve()
    state_dir = Path(args.state_dir).resolve()
    runtime = runtime_identity(code_root)
    started_at = utc_now()
    with exclusive_lock(state_dir / "mail_pipeline.lock"):
        process_manifest, config_path, config = load_inputs(
            state_dir=state_dir,
            runtime=runtime,
            max_age_hours=args.max_process_age_hours,
        )
        timeline_db = Path(str(config["timeline_db"])).resolve()
        cursor_before_state = read_mail_cursor_state(timeline_db)
        cursor_before = (
            str(cursor_before_state["last_cursor_ts"])
            if cursor_before_state and cursor_before_state.get("last_cursor_ts")
            else None
        )
        backfill_missing_only = bool(process_manifest.get("backfill_missing_only"))
        skip_link_enrich = backfill_missing_only or bool(args.skip_link_enrich)
        completed = run_incremental(code_root, config_path)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = {"overall_status": "failed", "gate_passed": False}
        gate_passed = result.get("gate_passed") is True
        failed_required = result.get("failed_required_sources") or []
        incremental_ok = (
            completed.returncode == 0
            and gate_passed
            and not failed_required
            and result.get("overall_status") != "partial"
        )
        cursor_preserved = read_mail_cursor_state(timeline_db) == cursor_before_state
        if backfill_missing_only and not cursor_preserved:
            restore_mail_cursor(timeline_db, cursor_before_state)
            incremental_ok = False
            failed_required = [*failed_required, "mail_archive_stage2:cursor_changed"]
        enrich_report: Mapping[str, Any] = {}
        enrich_error = ""
        if incremental_ok and not skip_link_enrich:
            try:
                enrich_report = enrich_mail_links(
                    timeline_db=timeline_db,
                    allowed_root=Path(str(config["allowed_root"])).resolve(),
                    out_dir=state_dir / "mail_link_enrich",
                )
            except Exception as exc:  # noqa: BLE001
                enrich_error = type(exc).__name__
        enrich_safety = enrich_report.get("safety") if isinstance(enrich_report, Mapping) else {}
        if not isinstance(enrich_safety, Mapping):
            enrich_safety = {}
        visibility_changed = bool(
            enrich_safety.get("allowed_for_bot_changed")
            or enrich_safety.get("mail_stage2_allowed_for_bot_changed")
        )
        enrich_ok = incremental_ok and (
            skip_link_enrich or (not enrich_error and not visibility_changed)
        )
        counts = enrich_report.get("counts") if isinstance(enrich_report, Mapping) else {}
        apply_counts = enrich_report.get("apply") if isinstance(enrich_report, Mapping) else {}
        if not isinstance(counts, Mapping):
            counts = {}
        if not isinstance(apply_counts, Mapping):
            apply_counts = {}
        applied = apply_counts.get("counts") if isinstance(apply_counts.get("counts"), Mapping) else {}
        status = "ok" if enrich_ok else "failed"
        if incremental_ok and not enrich_ok:
            restore_mail_cursor(timeline_db, cursor_before_state)
        report = {
            "schema_version": "mail_import_manifest_v1",
            "status": status,
            "started_at": started_at,
            "finished_at": utc_now(),
            "runtime": runtime,
            "process_manifest": str(state_dir / "mail_process_manifest.json"),
            "process_manifest_sha256": sha256_file(state_dir / "mail_process_manifest.json"),
            "config": str(config_path),
            "command_rc": completed.returncode,
            "overall_status": result.get("overall_status"),
            "gate_passed": result.get("gate_passed"),
            "failed_required_sources": failed_required,
            "mail_link_enrich": {
                "status": "skipped" if skip_link_enrich and incremental_ok else ("ok" if enrich_ok else "failed"),
                "error": enrich_error or None,
                "target_events": int(enrich_report.get("target_events") or 0),
                "planned": {
                    "strong": int(counts.get("planned.strong") or 0),
                    "weak_email": int(counts.get("planned.weak_email") or 0),
                    "unmatched": int(counts.get("planned.unmatched") or 0),
                    "blocked": int(counts.get("planned.blocked") or 0),
                },
                "updated_events": int(applied.get("updated_events") or 0),
                "created_chunks": int(applied.get("created_chunks") or 0),
                "visibility_changed": visibility_changed,
            },
            "cursor_before": cursor_before,
            "cursor_after": read_mail_cursor(timeline_db),
            "cursor_preserved": read_mail_cursor_state(timeline_db) == cursor_before_state,
            "backfill_missing_only": backfill_missing_only,
            "writes_prod_db": False,
            "write_external_systems": False,
            "timeline_db": str(timeline_db),
            "input_rows": process_manifest.get("rows_written"),
        }
        atomic_write_json(state_dir / "mail_import_manifest.json", report)
        return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", default=str(ROOT))
    parser.add_argument("--state-dir", default=str(ROOT / ".codex_local/staging/mail_pipeline"))
    parser.add_argument("--max-process-age-hours", type=float, default=4.0)
    parser.add_argument("--skip-link-enrich", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = execute(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        if str(exc) == "mail_download_already_running":
            print(json.dumps({"status": "already_running", "stop_reason": "already_running"}, sort_keys=True))
            return 75
        print(json.dumps({"status": "failed", "error": type(exc).__name__}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
