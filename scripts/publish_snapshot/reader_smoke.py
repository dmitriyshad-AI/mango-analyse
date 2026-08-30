#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig  # noqa: E402
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore  # noqa: E402
from mango_mvp.customer_timeline.stage4b_bot_opening import (  # noqa: E402
    _MAIL_OUTPUT_SECRET_TAGS,
)
from scripts.publish_snapshot.common import (  # noqa: E402
    add_common_args,
    finish_cli,
    load_config,
    quick_check,
    render_command,
    report_base,
    run_command,
)

_MAIL_FORBIDDEN_PRIMARY_REASONS = frozenset({"manager_action_required", "has_manager_note"})
_KNOWN_BRANDS = frozenset({"foton", "unpk"})


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
        primary_reason_placeholders = ",".join("?" for _ in _MAIL_FORBIDDEN_PRIMARY_REASONS)
        secret_tag_placeholders = ",".join("?" for _ in _MAIL_OUTPUT_SECRET_TAGS)
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
            "allowed_mail_without_a2_fact": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    LEFT JOIN a2v3_mail_event_facts AS f
                      ON f.event_id = b.event_id
                     AND f.tenant_id = b.tenant_id
                     AND f.customer_id = b.customer_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND f.event_id IS NULL
                    """
                ).fetchone()[0]
            ),
            "allowed_mail_forbidden_primary_reason": int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    JOIN a2v3_mail_event_facts AS f
                      ON f.event_id = b.event_id
                     AND f.tenant_id = b.tenant_id
                     AND f.customer_id = b.customer_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND f.client_safe_reason IN ({primary_reason_placeholders})
                    """,
                    tuple(_MAIL_FORBIDDEN_PRIMARY_REASONS),
                ).fetchone()[0]
            ),
            "allowed_mail_bot_visible_false": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    JOIN a2v3_mail_event_facts AS f
                      ON f.event_id = b.event_id
                     AND f.tenant_id = b.tenant_id
                     AND f.customer_id = b.customer_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND f.bot_visible = 0
                    """
                ).fetchone()[0]
            ),
            "allowed_mail_secret_tags": int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    JOIN a2v3_mail_event_facts AS f
                      ON f.event_id = b.event_id
                     AND f.tenant_id = b.tenant_id
                     AND f.customer_id = b.customer_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND EXISTS (
                        SELECT 1
                        FROM json_each(COALESCE(f.sensitivity_tags_json, '[]')) AS tag
                        WHERE tag.value IN ({secret_tag_placeholders})
                      )
                    """,
                    tuple(_MAIL_OUTPUT_SECRET_TAGS),
                ).fetchone()[0]
            ),
            "allowed_mail_client_unsafe": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    JOIN a2v3_mail_event_facts AS f
                      ON f.event_id = b.event_id
                     AND f.tenant_id = b.tenant_id
                     AND f.customer_id = b.customer_id
                    WHERE b.source_system = 'mail_archive_stage2'
                      AND b.allowed_for_bot = 1
                      AND f.client_safe != 1
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
        "policy": (
            "opened mail chunks require A2 client_safe=1 and bot_visible=1; missing facts, "
            "client_safe=0, bot_visible=0, secret tags, and manager-review reasons block publish"
        ),
    }


def mango_processed_allowed_safety_gate(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as con:
        if not _table_exists(con, "timeline_events") or not _table_exists(con, "bot_context_chunks"):
            return {"ok": True, "skipped": True, "reason": "timeline_events_or_chunks_table_missing"}
        counts = {
            "allowed_mango_processed_chunks": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks
                    WHERE source_system = 'mango_processed_summary'
                      AND allowed_for_bot = 1
                      AND requires_manager_review = 0
                      AND superseded_by IS NULL
                    """
                ).fetchone()[0]
            ),
            "allowed_mango_processed_missing_or_superseded_event": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    LEFT JOIN timeline_events AS e ON e.event_id = b.event_id
                    WHERE b.source_system = 'mango_processed_summary'
                      AND b.allowed_for_bot = 1
                      AND b.requires_manager_review = 0
                      AND b.superseded_by IS NULL
                      AND (e.event_id IS NULL OR e.superseded_by IS NOT NULL)
                    """
                ).fetchone()[0]
            ),
            "allowed_mango_processed_non_strong_match": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    LEFT JOIN timeline_events AS e ON e.event_id = b.event_id
                    WHERE b.source_system = 'mango_processed_summary'
                      AND b.allowed_for_bot = 1
                      AND b.requires_manager_review = 0
                      AND b.superseded_by IS NULL
                      AND COALESCE(e.match_status, '') != 'strong_unique'
                    """
                ).fetchone()[0]
            ),
            "allowed_mango_processed_wrong_chunk_type": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    WHERE b.source_system = 'mango_processed_summary'
                      AND b.allowed_for_bot = 1
                      AND b.requires_manager_review = 0
                      AND b.superseded_by IS NULL
                      AND COALESCE(b.chunk_type, '') != 'mango_call_summary'
                    """
                ).fetchone()[0]
            ),
            "allowed_mango_processed_customer_mismatch": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    LEFT JOIN timeline_events AS e ON e.event_id = b.event_id
                    WHERE b.source_system = 'mango_processed_summary'
                      AND b.allowed_for_bot = 1
                      AND b.requires_manager_review = 0
                      AND b.superseded_by IS NULL
                      AND (
                        e.event_id IS NULL
                        OR e.superseded_by IS NOT NULL
                        OR COALESCE(b.customer_id, '') != COALESCE(e.customer_id, '')
                      )
                    """
                ).fetchone()[0]
            ),
            "allowed_mango_processed_missing_identity": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    LEFT JOIN timeline_events AS e ON e.event_id = b.event_id
                    LEFT JOIN customer_identities AS ci
                      ON ci.tenant_id = b.tenant_id AND ci.customer_id = b.customer_id
                    WHERE b.source_system = 'mango_processed_summary'
                      AND b.allowed_for_bot = 1
                      AND b.requires_manager_review = 0
                      AND b.superseded_by IS NULL
                      AND (
                        b.customer_id IS NULL OR b.customer_id = ''
                        OR e.customer_id IS NULL OR e.customer_id = ''
                        OR e.superseded_by IS NOT NULL
                        OR b.customer_id != e.customer_id
                        OR ci.identity_status IS NULL
                        OR ci.identity_status NOT IN ('strong', 'partial')
                      )
                    """
                ).fetchone()[0]
            ),
            "allowed_mango_processed_unknown_brand_metric": int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks AS b
                    WHERE b.source_system = 'mango_processed_summary'
                      AND b.allowed_for_bot = 1
                      AND b.requires_manager_review = 0
                      AND b.superseded_by IS NULL
                      AND LOWER(COALESCE(json_extract(b.record_json, '$.metadata.content_brand'), '')) NOT IN ('foton', 'unpk')
                    """
                ).fetchone()[0]
            ),
        }
    violations = {
        key: value
        for key, value in counts.items()
        if key != "allowed_mango_processed_chunks" and int(value) > 0
    }
    return {
        "ok": not violations,
        "skipped": False,
        "counts": counts,
        "violations": violations,
        "policy": (
            "active opened mango_processed_summary chunks require strong_unique event match, "
            "resolved customer identity, and an explicit foton/unpk content_brand"
        ),
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


def bot_visibility_counts(
    db_path: Path,
    *,
    tenant_id: str,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Count stored and effective visibility through the canonical reader policy."""

    cutoff = as_of or datetime.now(timezone.utc)
    with CustomerTimelineSQLiteStore.open_read_only(
        db_path,
        allowed_root=db_path.parent,
        clock=lambda: cutoff,
    ) as store:
        stored = int(
            store._con.execute(  # noqa: SLF001 - acceptance probes the canonical store policy.
                """
                SELECT COUNT(*) FROM bot_context_chunks
                WHERE tenant_id=? AND allowed_for_bot=1
                  AND (superseded_by IS NULL OR superseded_by='')
                """,
                (tenant_id,),
            ).fetchone()[0]
        )
        clauses = ["tenant_id = ?"]
        params: list[object] = [tenant_id]
        store._append_chunk_filters(  # noqa: SLF001 - one canonical bot-safe boundary.
            clauses,
            params,
            tenant_id=tenant_id,
            customer_id=None,
            opportunity_id=None,
            since=None,
            until=cutoff,
            allowed_for_bot=True,
        )
        effective = int(
            store._con.execute(  # noqa: SLF001 - read-only acceptance count.
                "SELECT COUNT(*) FROM bot_context_chunks WHERE " + " AND ".join(clauses),
                tuple(params),
            ).fetchone()[0]
        )
        rows = store._con.execute(  # noqa: SLF001 - same canonical boundary, grouped.
            "SELECT source_system,chunk_type,COUNT(*) FROM bot_context_chunks WHERE "
            + " AND ".join(clauses)
            + " GROUP BY source_system,chunk_type ORDER BY source_system,chunk_type",
            tuple(params),
        )
        effective_pairs = [
            {"source_system": str(row[0] or ""), "chunk_type": str(row[1] or ""), "count": int(row[2])}
            for row in rows
        ]
    return {
        "as_of": cutoff.isoformat(),
        "bot_visible_stored": stored,
        "bot_visible_after_reader_policy": effective,
        "reader_policy_blocked": stored - effective,
        "effective_pairs": effective_pairs,
    }


def bot_visibility_gate(cfg, visibility: dict[str, object]) -> dict[str, object]:
    policy = cfg.bot_visibility_policy
    allowed = {
        (str(item["source_system"]), str(item["chunk_type"]))
        for item in policy["allowed_effective_pairs"]
    }
    effective = int(visibility["bot_visible_after_reader_policy"])
    unexpected = [
        item
        for item in visibility.get("effective_pairs", [])
        if (str(item["source_system"]), str(item["chunk_type"])) not in allowed
    ]
    minimum, maximum = int(policy["min_effective_chunks"]), int(policy["max_effective_chunks"])
    reasons = []
    for failed, reason in ((effective < minimum, "effective_visibility_below_minimum"),
                           (effective > maximum, "effective_visibility_above_maximum"),
                           (bool(unexpected), "unexpected_effective_source_or_chunk_type")):
        if failed:
            reasons.append(reason)
    return {
        "ok": not reasons,
        "policy": dict(policy),
        "unexpected_effective_pairs": unexpected,
        "reasons": reasons,
    }


def smoke(
    config_path: Path,
    *,
    snapshot_db: Path,
    as_of: datetime | None = None,
) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "reader_smoke")
    db_path = snapshot_db.expanduser().resolve(strict=False)
    variables = {"db": db_path, "allowed_root": db_path.parent, "tenant_id": cfg.tenant_id}
    reader_results = []
    snapshot_quick_check = quick_check(db_path)
    ok = snapshot_quick_check == "ok"
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
    mango_safety = mango_processed_allowed_safety_gate(db_path)
    ok = ok and bool(mango_safety["ok"])
    visibility = bot_visibility_counts(db_path, tenant_id=cfg.tenant_id, as_of=as_of)
    visibility_gate = bot_visibility_gate(cfg, visibility)
    ok = ok and bool(visibility_gate["ok"])
    report.update(
        {
            "snapshot_db": str(db_path),
            "quick_check": snapshot_quick_check,
            "reader_results": reader_results,
            "internal_control_customers": internal_results,
            "mail_allowed_safety_gate": mail_safety,
            "mango_processed_allowed_safety_gate": mango_safety,
            "bot_visibility": visibility,
            "bot_visibility_gate": visibility_gate,
            "status": "ok" if ok else "failed",
        }
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Customer Timeline snapshot through reader APIs.")
    add_common_args(parser)
    parser.add_argument("--snapshot-db", type=Path, required=True)
    parser.add_argument("--as-of", help="UTC cutoff used by the canonical reader visibility gate.")
    args = parser.parse_args()
    as_of = None
    if args.as_of:
        as_of = datetime.fromisoformat(args.as_of[:-1] + "+00:00" if args.as_of.endswith("Z") else args.as_of)
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise SystemExit("--as-of must include a timezone")
        as_of = as_of.astimezone(timezone.utc)
    report, ok = smoke(args.config, snapshot_db=args.snapshot_db, as_of=as_of)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
