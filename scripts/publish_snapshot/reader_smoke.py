#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig  # noqa: E402
from scripts.publish_snapshot.common import (  # noqa: E402
    add_common_args,
    finish_cli,
    load_config,
    quick_check,
    render_command,
    report_base,
    run_command,
)


def control_customer_counts(db_path: Path, tenant_id: str, customer_id: str) -> dict[str, int]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as con:
        return {
            "events_total": int(
                con.execute(
                    "SELECT COUNT(*) FROM timeline_events WHERE tenant_id = ? AND customer_id = ?",
                    (tenant_id, customer_id),
                ).fetchone()[0]
            ),
            "bot_context_chunks_total": int(
                con.execute(
                    "SELECT COUNT(*) FROM bot_context_chunks WHERE tenant_id = ? AND customer_id = ?",
                    (tenant_id, customer_id),
                ).fetchone()[0]
            ),
            "allowed_chunks": int(
                con.execute(
                    "SELECT COUNT(*) FROM bot_context_chunks WHERE tenant_id = ? AND customer_id = ? AND allowed_for_bot = 1",
                    (tenant_id, customer_id),
                ).fetchone()[0]
            ),
            "review_required_chunks": int(
                con.execute(
                    "SELECT COUNT(*) FROM bot_context_chunks WHERE tenant_id = ? AND customer_id = ? AND requires_manager_review = 1",
                    (tenant_id, customer_id),
                ).fetchone()[0]
            ),
            "derived_signals_total": int(
                con.execute(
                    "SELECT COUNT(*) FROM derived_signals WHERE tenant_id = ? AND customer_id = ?",
                    (tenant_id, customer_id),
                ).fetchone()[0]
            ),
        }


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def mail_allowed_safety_gate(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as con:
        if not _table_exists(con, "a2v3_mail_event_facts") or not _table_exists(con, "bot_context_chunks"):
            return {"ok": True, "skipped": True, "reason": "mail_facts_or_chunks_table_missing"}
        counts = {
            "allowed_mail_chunks": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks
                    WHERE source_system = 'mail_archive_stage2' AND allowed_for_bot = 1
                    """
                ).fetchone()[0]
            ),
            "allowed_mail_with_manager_tags": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    JOIN a2v3_mail_event_facts AS f ON f.event_id = b.event_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND (
                        f.sensitivity_tags_json LIKE '%manager_action_required%'
                        OR f.sensitivity_tags_json LIKE '%has_manager_note%'
                      )
                    """
                ).fetchone()[0]
            ),
            "allowed_mail_client_safe_false": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    JOIN a2v3_mail_event_facts AS f ON f.event_id = b.event_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND f.client_safe = 0
                    """
                ).fetchone()[0]
            ),
            "allowed_mail_bot_visible_false": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    JOIN a2v3_mail_event_facts AS f ON f.event_id = b.event_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND f.bot_visible = 0
                    """
                ).fetchone()[0]
            ),
        }
    violations = {
        key: value
        for key, value in counts.items()
        if key != "allowed_mail_chunks" and int(value) > 0
    }
    return {
        "ok": not violations,
        "skipped": False,
        "counts": counts,
        "violations": violations,
        "policy": "opened mail chunks must not contradict A2 facts client_safe/bot_visible/manager tags",
    }


def run_internal_smoke(db_path: Path, allowed_root: Path, tenant_id: str, control_customers: tuple[dict, ...]) -> list[dict]:
    results = []
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=allowed_root)) as api:
        for item in control_customers:
            customer_id = str(item.get("customer_id") or "")
            profile = api.customer_profile(tenant_id, customer_id, event_limit=5, bot_context_limit=5)
            observed_counts = control_customer_counts(db_path, tenant_id, customer_id) if profile.get("found") else {}
            expected_counts = {str(key): int(value) for key, value in dict(item.get("expected_counts") or {}).items()}
            count_mismatches = {
                key: {"expected": expected, "actual": observed_counts.get(key)}
                for key, expected in expected_counts.items()
                if observed_counts.get(key) != expected
            }
            found_ok = bool(profile.get("found")) == bool(item.get("expected_found", True))
            results.append(
                {
                    "customer_id": customer_id,
                    "label": item.get("label"),
                    "expected_found": bool(item.get("expected_found", True)),
                    "found": bool(profile.get("found")),
                    "events": len(((profile.get("timeline") or {}).get("items") or [])),
                    "bot_visible": int(((profile.get("bot_context") or {}).get("summary") or {}).get("visible_chunks") or 0),
                    "observed_counts": observed_counts,
                    "expected_counts": expected_counts,
                    "count_mismatches": count_mismatches,
                    "ok": found_ok and not count_mismatches,
                }
            )
    return results


def smoke(config_path: Path, *, snapshot_db: Path) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "reader_smoke")
    db_path = snapshot_db.expanduser().resolve(strict=False)
    variables = {"db": db_path, "allowed_root": db_path.parent, "tenant_id": cfg.tenant_id}
    reader_results = []
    ok = quick_check(db_path) == "ok"
    for reader in cfg.readers:
        command = reader.get("smoke_command")
        if command:
            worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
            result = run_command(render_command(command, variables), cwd=worktree, timeout=int(reader.get("smoke_timeout_seconds") or 120))
            result["name"] = reader.get("name")
            result["ok"] = result.get("rc") == 0
            ok = ok and bool(result["ok"])
            reader_results.append(result)
    internal_results = run_internal_smoke(db_path, db_path.parent, cfg.tenant_id, tuple(dict(x) for x in cfg.control_customers))
    if internal_results:
        ok = ok and all(item["ok"] for item in internal_results)
    mail_safety = mail_allowed_safety_gate(db_path)
    ok = ok and bool(mail_safety["ok"])
    report.update(
        {
            "snapshot_db": str(db_path),
            "quick_check": quick_check(db_path),
            "reader_results": reader_results,
            "internal_control_customers": internal_results,
            "mail_allowed_safety_gate": mail_safety,
            "status": "ok" if ok else "failed",
        }
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Customer Timeline snapshot through reader APIs.")
    add_common_args(parser)
    parser.add_argument("--snapshot-db", type=Path, required=True)
    args = parser.parse_args()
    report, ok = smoke(args.config, snapshot_db=args.snapshot_db)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
