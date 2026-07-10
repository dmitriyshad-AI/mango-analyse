#!/usr/bin/env python3
"""Run a repeatable staging-only Customer Timeline E2E probe.

The probe writes only under .codex_local/staging. It does not open prod DB,
does not call CRM/Tallanto, and treats any allowed_for_bot delta as a failure.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.nightly_service import run_nightly_service, service_config_from_json  # noqa: E402
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path  # noqa: E402


DEFAULT_STAGING_ROOT = ROOT / ".codex_local" / "staging"
DEFAULT_TIMELINE_DB = DEFAULT_STAGING_ROOT / "customer_timeline_staging.sqlite"
DEFAULT_OUT_ROOT = DEFAULT_STAGING_ROOT / "e2e_probe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staging-only Customer Timeline E2E probe.")
    parser.add_argument("--timeline-db", type=Path, default=DEFAULT_TIMELINE_DB)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--skip-service", action="store_true", help="Only render metrics/report skeleton.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    staging_root = DEFAULT_STAGING_ROOT.resolve(strict=False)
    timeline_db = guard_customer_timeline_output_path(args.timeline_db, staging_root)
    out_root = guard_customer_timeline_output_path(args.out_root, staging_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    before = collect_metrics(timeline_db, tenant_id=args.tenant_id)
    source_builder = run_source_builder(run_dir, timeline_db=timeline_db) if not args.skip_service else {}
    service_reports: list[Mapping[str, Any]] = []
    e2e_config_path = None
    if not args.skip_service:
        generated_config = Path(str(source_builder["service_config"]))
        e2e_config_path = run_dir / "customer_timeline_e2e_service_config.json"
        write_e2e_service_config(generated_config, e2e_config_path, run_dir=run_dir, timeline_db=timeline_db)
        first = run_nightly_service(service_config_from_json(e2e_config_path))
        second = run_nightly_service(service_config_from_json(e2e_config_path))
        service_reports = [first, second]
        (run_dir / "service_first.json").write_text(json_dumps(first), encoding="utf-8")
        (run_dir / "service_second.json").write_text(json_dumps(second), encoding="utf-8")
    after = collect_metrics(timeline_db, tenant_id=args.tenant_id)
    allowed_delta = {
        key: int(after["allowed_for_bot"].get(key, 0)) - int(before["allowed_for_bot"].get(key, 0))
        for key in sorted(set(before["allowed_for_bot"]) | set(after["allowed_for_bot"]))
    }
    rerun_summary = summarize_rerun(service_reports[1] if len(service_reports) > 1 else {})
    delta = metric_delta(before, after)
    event_chunk_link_delta_zero = all(
        int(delta["counts"].get(key, 0)) == 0
        for key in ("timeline_events", "bot_context_chunks", "identity_links", "timeline_conflicts")
    )
    report = {
        "schema_version": "customer_timeline_e2e_staging_probe_v1",
        "run_id": run_id,
        "timeline_db": str(timeline_db),
        "out_dir": str(run_dir),
        "e2e_config": str(e2e_config_path) if e2e_config_path else None,
        "before": before,
        "after": after,
        "delta": delta,
        "allowed_for_bot_delta": allowed_delta,
        "allowed_for_bot_unchanged": all(value == 0 for value in allowed_delta.values()),
        "source_builder": source_builder,
        "service": {
            "first_status": service_reports[0].get("overall_status") if service_reports else "skipped",
            "second_status": service_reports[1].get("overall_status") if len(service_reports) > 1 else "skipped",
            "rerun_summary": rerun_summary,
            "rerun_zero_new": event_chunk_link_delta_zero and rerun_summary.get("mail_link_target_events", 0) == 0,
            "rerun_note": (
                "Rerun criterion is based on timeline_events/bot_context_chunks/identity_links/timeline_conflicts deltas. "
                "Cursor timestamps may still refresh on repeated service runs."
            ),
        },
        "readers": {
            "manager_view_has_linked_mail": after["mail_archive_stage2"]["linked"] > 0,
            "bot_view_not_opened": all(value == 0 for value in allowed_delta.values()),
        },
        "safety": {
            "writes_prod_db": False,
            "writes_staging_db": not args.skip_service,
            "writes_crm": False,
            "writes_tallanto": False,
            "sends_messages": False,
            "allowed_for_bot_changed": not all(value == 0 for value in allowed_delta.values()),
        },
    }
    (run_dir / "report.json").write_text(json_dumps(report), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json_dumps({"report": str(run_dir / "report.json"), "status": "ok" if report["allowed_for_bot_unchanged"] else "failed"}))
    return 0 if report["allowed_for_bot_unchanged"] else 2


def run_source_builder(run_dir: Path, *, timeline_db: Path) -> Mapping[str, Any]:
    out_root = run_dir / "sources"
    config_out = run_dir / "customer_timeline_nightly_service_dv2_config.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_customer_timeline_nightly_dv2_sources.py"),
            "--out-root",
            str(out_root),
            "--timeline-db",
            str(timeline_db),
            "--service-config-out",
            str(config_out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "source_builder.stdout.json").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "source_builder.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"source builder failed: rc={proc.returncode}")
    payload = json.loads(proc.stdout)
    payload["service_config"] = str(config_out)
    return payload


def write_e2e_service_config(source_config: Path, out_config: Path, *, run_dir: Path, timeline_db: Path) -> None:
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    payload["timeline_db"] = str(timeline_db)
    payload["allowed_root"] = str(DEFAULT_STAGING_ROOT.resolve(strict=False))
    payload["out_root"] = str(run_dir / "nightly_service_runs")
    payload["publish_dir"] = str(DEFAULT_STAGING_ROOT / "nightly_service" / "published")
    steps = list(payload.get("steps") or [])
    steps.append(
        {
            "name": "mail_link_enrich",
            "kind": "mail_link_enrich",
            "enabled": True,
            "required": False,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(DEFAULT_STAGING_ROOT.resolve(strict=False)),
                "out_dir": str(run_dir / "mail_link_enrich"),
                "apply": True,
            },
        }
    )
    payload["steps"] = steps
    out_config.write_text(json_dumps(payload), encoding="utf-8")


def collect_metrics(db_path: Path, *, tenant_id: str) -> Mapping[str, Any]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        counts = {
            table: int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("timeline_events", "bot_context_chunks", "identity_links", "timeline_conflicts", "ingestion_cursors")
        }
        allowed = {
            str(row["source_system"] or ""): int(row["count"])
            for row in con.execute(
                """
                SELECT source_system, COUNT(*) AS count
                FROM bot_context_chunks
                WHERE tenant_id = ? AND allowed_for_bot = 1
                GROUP BY source_system
                ORDER BY source_system
                """,
                (tenant_id,),
            )
        }
        mail = dict(
            con.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN customer_id IS NOT NULL AND customer_id <> '' THEN 1 ELSE 0 END) AS linked,
                  SUM(CASE WHEN customer_id IS NULL OR customer_id = '' THEN 1 ELSE 0 END) AS pending,
                  MAX(event_at) AS max_event_at
                FROM timeline_events
                WHERE tenant_id = ? AND source_system = 'mail_archive_stage2'
                """,
                (tenant_id,),
            ).fetchone()
        )
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
    return {
        "quick_check": quick_check,
        "counts": counts,
        "allowed_for_bot": allowed,
        "mail_archive_stage2": mail,
        "source_counts": source_counts,
    }


def metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "counts": {
            key: int(after["counts"].get(key, 0)) - int(before["counts"].get(key, 0))
            for key in sorted(set(before["counts"]) | set(after["counts"]))
        },
        "mail_archive_stage2_total": int(after["mail_archive_stage2"].get("total") or 0)
        - int(before["mail_archive_stage2"].get("total") or 0),
        "mail_archive_stage2_linked": int(after["mail_archive_stage2"].get("linked") or 0)
        - int(before["mail_archive_stage2"].get("linked") or 0),
    }


def summarize_rerun(report: Mapping[str, Any]) -> Mapping[str, Any]:
    changed = 0
    mail_link_target_events = 0
    for step in report.get("steps") or ():
        summary = step.get("summary") if isinstance(step, Mapping) else {}
        if not isinstance(summary, Mapping):
            continue
        changed += int(summary.get("changed_customer_count") or 0)
        if step.get("kind") == "mail_link_enrich":
            mail_link_target_events += int(summary.get("target_events") or 0)
    return {"changed_customer_count_total": changed, "mail_link_target_events": mail_link_target_events}


def render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Customer Timeline E2E staging probe",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- timeline_db: `{report['timeline_db']}`",
            f"- first_status: `{report['service']['first_status']}`",
            f"- second_status: `{report['service']['second_status']}`",
            f"- rerun_zero_new: `{report['service']['rerun_zero_new']}`",
            f"- allowed_for_bot_unchanged: `{report['allowed_for_bot_unchanged']}`",
            f"- manager_view_has_linked_mail: `{report['readers']['manager_view_has_linked_mail']}`",
            f"- bot_view_not_opened: `{report['readers']['bot_view_not_opened']}`",
            "",
            "## Delta",
            "",
            "```json",
            json.dumps(report["delta"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    ) + "\n"


def json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
