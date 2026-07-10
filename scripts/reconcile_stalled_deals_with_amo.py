#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.existing_clients.amo_step1_snapshot import (
    DEFAULT_ENV_PATH,
    AmoMcpClient,
    AmoMcpError,
    read_mcp_env,
)


DEFAULT_STAGING_DB = Path(".codex_local/staging/customer_timeline_staging.sqlite")
DEFAULT_OUT_JSON = Path(".codex_local/review/f9_amo_actuality/stalled_deals_amo_reconcile.json")
SCHEMA_VERSION = "marathon2_f9_stalled_deals_amo_reconcile_v1"
CLOSED_AMO_STATUS_IDS = {"142", "143"}
CLOSED_STATUS_WORDS = ("закрыто", "успешно")


class AmoReadClient(Protocol):
    calls: int

    def amo_api_get(self, *, path: str, params: Mapping[str, Any] | None = None, limit: int = 50) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class OpenOpportunity:
    customer_id: str
    opportunity_id: str
    source_id: str
    snapshot_status: str
    snapshot_closed_at: str
    snapshot_title: str
    fact_total_in: float


def connect_ro(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def selected_stalled_customers(con: sqlite3.Connection, *, limit: int) -> list[str]:
    rows = con.execute(
        """
        SELECT
          s.customer_id,
          COALESCE(MAX(p.total_in), 0) AS fact_total_in,
          MAX(s.created_at) AS latest_signal_at
        FROM derived_signals s
        LEFT JOIN customer_purchases_v1 p
          ON p.tenant_id = s.tenant_id
         AND p.customer_id = s.customer_id
         AND p.period = 'all_time'
         AND p.money_kind = 'fact'
        WHERE s.signal_type = 'deal_stalling'
          AND s.status = 'active'
          AND s.customer_id IS NOT NULL
          AND s.customer_id != ''
        GROUP BY s.customer_id
        ORDER BY fact_total_in DESC, latest_signal_at DESC, s.customer_id
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    return [str(row["customer_id"]) for row in rows]


def open_opportunities_for_customers(con: sqlite3.Connection, customer_ids: Sequence[str]) -> list[OpenOpportunity]:
    if not customer_ids:
        return []
    placeholders = ",".join("?" for _ in customer_ids)
    rows = con.execute(
        f"""
        SELECT
          o.customer_id,
          o.opportunity_id,
          o.source_id,
          o.status,
          o.closed_at,
          o.title,
          COALESCE(p.total_in, 0) AS fact_total_in
        FROM customer_opportunities o
        LEFT JOIN customer_purchases_v1 p
          ON p.tenant_id = o.tenant_id
         AND p.customer_id = o.customer_id
         AND p.period = 'all_time'
         AND p.money_kind = 'fact'
        WHERE o.opportunity_type = 'amo_deal'
          AND o.customer_id IN ({placeholders})
          AND o.source_id IS NOT NULL
          AND o.source_id != ''
        ORDER BY fact_total_in DESC, o.customer_id, o.source_id
        """,
        tuple(customer_ids),
    ).fetchall()
    result: list[OpenOpportunity] = []
    for row in rows:
        if _snapshot_opportunity_closed(row["status"], row["closed_at"]):
            continue
        result.append(
            OpenOpportunity(
                customer_id=str(row["customer_id"]),
                opportunity_id=str(row["opportunity_id"]),
                source_id=str(row["source_id"]),
                snapshot_status=str(row["status"] or ""),
                snapshot_closed_at=str(row["closed_at"] or ""),
                snapshot_title=str(row["title"] or ""),
                fact_total_in=float(row["fact_total_in"] or 0),
            )
        )
    return result


def _snapshot_opportunity_closed(status: Any, closed_at: Any) -> bool:
    if str(closed_at or "").strip():
        return True
    value = str(status or "").strip().casefold()
    if value in CLOSED_AMO_STATUS_IDS:
        return True
    return any(word in value for word in CLOSED_STATUS_WORDS)


def fetch_amo_lead(client: AmoReadClient, lead_id: str, *, page_limit: int = 50) -> Mapping[str, Any] | None:
    payload = client.amo_api_get(path="leads", params={"filter[id]": str(lead_id), "with": "contacts"}, limit=page_limit)
    embedded = payload.get("_embedded") if isinstance(payload, Mapping) else None
    leads = embedded.get("leads") if isinstance(embedded, Mapping) else None
    if isinstance(leads, list) and leads:
        first = leads[0]
        return dict(first) if isinstance(first, Mapping) else None
    return None


def live_lead_changed(snapshot: OpenOpportunity, lead: Mapping[str, Any] | None) -> tuple[bool, str, Mapping[str, Any]]:
    if lead is None:
        return True, "live_lead_not_found", {"live_found": False}
    live_status = str(lead.get("status_id") or "")
    live_pipeline = str(lead.get("pipeline_id") or "")
    live_closed_at = str(lead.get("closed_at") or "")
    snapshot_status = str(snapshot.snapshot_status or "")
    if live_closed_at or live_status in CLOSED_AMO_STATUS_IDS:
        return True, "live_lead_closed", _live_payload(lead)
    if snapshot_status and snapshot_status.isdigit() and snapshot_status != live_status:
        return True, "status_id_changed", _live_payload(lead)
    return False, "unchanged_or_snapshot_status_name", _live_payload(lead)


def _live_payload(lead: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "live_found": True,
        "live_status_id": str(lead.get("status_id") or ""),
        "live_pipeline_id": str(lead.get("pipeline_id") or ""),
        "live_updated_at": lead.get("updated_at"),
        "live_closed_at": lead.get("closed_at"),
    }


def reconcile(
    *,
    timeline_db: Path,
    limit: int,
    mcp_env: Path,
    page_limit: int,
    sleep_sec: float,
    client: AmoReadClient | None = None,
) -> Mapping[str, Any]:
    db = timeline_db.expanduser().resolve(strict=False)
    with connect_ro(db) as con:
        customer_ids = selected_stalled_customers(con, limit=limit)
        opportunities = open_opportunities_for_customers(con, customer_ids)
    generated_at = datetime.now(timezone.utc).isoformat()
    if not mcp_env.expanduser().exists() and client is None:
        return _unavailable_report(
            generated_at=generated_at,
            timeline_db=db,
            customer_ids=customer_ids,
            opportunities=opportunities,
            reason="mcp_env_missing",
            mcp_env=mcp_env,
        )
    try:
        live_client = client or AmoMcpClient(read_mcp_env(mcp_env))
    except Exception as exc:
        return _unavailable_report(
            generated_at=generated_at,
            timeline_db=db,
            customer_ids=customer_ids,
            opportunities=opportunities,
            reason=f"mcp_env_unavailable:{type(exc).__name__}",
            mcp_env=mcp_env,
        )
    rows: list[Mapping[str, Any]] = []
    changed_customers: set[str] = set()
    checked_customers: set[str] = set()
    errors: list[Mapping[str, Any]] = []
    for opportunity in opportunities:
        checked_customers.add(opportunity.customer_id)
        try:
            lead = fetch_amo_lead(live_client, opportunity.source_id, page_limit=page_limit)
            changed, reason, live = live_lead_changed(opportunity, lead)
        except (AmoMcpError, OSError, TimeoutError, ValueError) as exc:
            errors.append({"lead_ref": _mask_id(opportunity.source_id), "error": type(exc).__name__})
            changed = False
            reason = f"live_read_error:{type(exc).__name__}"
            live = {"live_found": None}
        if changed:
            changed_customers.add(opportunity.customer_id)
        rows.append(
            {
                "customer_hash": _mask_id(opportunity.customer_id),
                "lead_hash": _mask_id(opportunity.source_id),
                "snapshot_status": opportunity.snapshot_status,
                "snapshot_status_is_id": opportunity.snapshot_status.isdigit(),
                "fact_total_in_bucket": _money_bucket(opportunity.fact_total_in),
                "changed": changed,
                "reason": reason,
                **live,
            }
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    checked = len(checked_customers)
    changed_count = len(changed_customers)
    divergence_rate = (changed_count / checked) if checked else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "checked",
        "timeline_db": str(db),
        "source_open_mode": "sqlite_mode_ro_immutable",
        "mcp_env": str(mcp_env.expanduser()),
        "top_customer_limit": limit,
        "customers_selected": len(customer_ids),
        "open_opportunities_checked": len(opportunities),
        "customers_checked": checked,
        "customers_changed": changed_count,
        "divergence_rate": divergence_rate,
        "snapshot_stale": divergence_rate > 0.15,
        "amo_api_calls": getattr(live_client, "calls", 0),
        "errors": errors,
        "rows": rows,
    }


def _unavailable_report(
    *,
    generated_at: str,
    timeline_db: Path,
    customer_ids: Sequence[str],
    opportunities: Sequence[OpenOpportunity],
    reason: str,
    mcp_env: Path,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "unavailable",
        "reason": reason,
        "timeline_db": str(timeline_db),
        "source_open_mode": "sqlite_mode_ro_immutable",
        "mcp_env": str(mcp_env.expanduser()),
        "customers_selected": len(customer_ids),
        "open_opportunities_planned": len(opportunities),
        "customers_checked": 0,
        "customers_changed": 0,
        "divergence_rate": None,
        "snapshot_stale": None,
        "rows": [],
    }


def _mask_id(value: Any) -> str:
    import hashlib

    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def _money_bucket(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 50_000:
        return "1..49k"
    if value < 150_000:
        return "50k..149k"
    if value < 500_000:
        return "150k..499k"
    return "500k+"


def write_report(report: Mapping[str, Any], out_json: Path, *, allowed_root: Path) -> Path:
    out = guard_customer_timeline_output_path(out_json, allowed_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only AMO actuality check for top stalled customers.")
    parser.add_argument("--timeline-db", type=Path, default=DEFAULT_STAGING_DB)
    parser.add_argument("--allowed-root", type=Path, default=Path.cwd())
    parser.add_argument("--mcp-env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--page-limit", type=int, default=50)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = reconcile(
        timeline_db=args.timeline_db,
        limit=args.limit,
        mcp_env=args.mcp_env,
        page_limit=args.page_limit,
        sleep_sec=args.sleep_sec,
    )
    path = write_report(report, args.out_json, allowed_root=args.allowed_root)
    print(json.dumps({"status": report["status"], "out_json": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
