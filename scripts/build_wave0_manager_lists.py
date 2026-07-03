#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path


DEFAULT_STAGING_DB = Path(".codex_local/staging/customer_timeline_staging.sqlite")
DEFAULT_RECONCILE_JSON = Path(".codex_local/review/f9_amo_actuality/stalled_deals_amo_reconcile.json")
DEFAULT_OUT_XLSX = Path(".codex_local/review/f9_amo_actuality/2026-07-03_VOLNA0_dengi_na_polu_refresh.xlsx")
SCHEMA_VERSION = "marathon2_f9_wave0_manager_lists_v1"


def connect_ro(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def source_freshness(con: sqlite3.Connection) -> list[Mapping[str, Any]]:
    return [
        dict(row)
        for row in con.execute(
            """
            SELECT source_system, MAX(event_at) AS max_event_at, COUNT(*) AS events
            FROM timeline_events
            GROUP BY source_system
            ORDER BY max_event_at DESC, source_system
            """
        ).fetchall()
    ]


def build_rows(con: sqlite3.Connection, signal_types: Sequence[str], *, limit: int) -> list[Mapping[str, Any]]:
    placeholders = ",".join("?" for _ in signal_types)
    params: list[Any] = [*signal_types]
    sql = f"""
        SELECT
          s.customer_id,
          ci.display_name,
          ci.primary_phone,
          ci.primary_email,
          s.signal_type,
          s.severity,
          s.expires_at,
          s.created_at AS signal_created_at,
          s.record_json AS signal_record_json,
          COALESCE(pf.total_in, 0) AS fact_total_in,
          COALESCE(pp.total_in, 0) AS plan_total_in,
          COALESCE(pf.deals_cnt, 0) AS fact_deals_cnt,
          COALESCE(pp.deals_cnt, 0) AS plan_deals_cnt,
          MAX(e.event_at) AS latest_event_at,
          GROUP_CONCAT(DISTINCT o.source_id) AS open_amo_lead_ids,
          GROUP_CONCAT(DISTINCT o.status) AS open_amo_statuses
        FROM derived_signals s
        JOIN customer_identities ci
          ON ci.tenant_id = s.tenant_id
         AND ci.customer_id = s.customer_id
        LEFT JOIN customer_purchases_v1 pf
          ON pf.tenant_id = s.tenant_id
         AND pf.customer_id = s.customer_id
         AND pf.period = 'all_time'
         AND pf.money_kind = 'fact'
        LEFT JOIN customer_purchases_v1 pp
          ON pp.tenant_id = s.tenant_id
         AND pp.customer_id = s.customer_id
         AND pp.period = 'all_time'
         AND pp.money_kind = 'plan'
        LEFT JOIN timeline_events e
          ON e.tenant_id = s.tenant_id
         AND e.customer_id = s.customer_id
         AND (e.superseded_by IS NULL OR e.superseded_by = '')
        LEFT JOIN customer_opportunities o
          ON o.tenant_id = s.tenant_id
         AND o.customer_id = s.customer_id
         AND o.opportunity_type = 'amo_deal'
         AND o.source_id IS NOT NULL
         AND o.source_id != ''
         AND (o.closed_at IS NULL OR o.closed_at = '')
         AND COALESCE(o.status, '') NOT IN ('142', '143', 'Закрыто и не реализовано', 'Успешно')
        WHERE s.signal_type IN ({placeholders})
          AND s.status = 'active'
        GROUP BY s.signal_id
        ORDER BY fact_total_in DESC, plan_total_in DESC, latest_event_at DESC, s.customer_id
    """
    if limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    result: list[Mapping[str, Any]] = []
    for row in con.execute(sql, tuple(params)).fetchall():
        signal_record = _loads(row["signal_record_json"])
        result.append(
            {
                "customer_id": row["customer_id"],
                "display_name": row["display_name"],
                "primary_phone": row["primary_phone"],
                "primary_email": row["primary_email"],
                "signal_type": row["signal_type"],
                "severity": row["severity"],
                "fact_total_in": row["fact_total_in"],
                "plan_total_in": row["plan_total_in"],
                "fact_deals_cnt": row["fact_deals_cnt"],
                "plan_deals_cnt": row["plan_deals_cnt"],
                "latest_event_at": row["latest_event_at"],
                "signal_created_at": row["signal_created_at"],
                "expires_at": row["expires_at"],
                "recommended_action": signal_record.get("recommended_action") or "",
                "evidence_text": signal_record.get("evidence_text") or "",
                "open_amo_lead_ids": row["open_amo_lead_ids"] or "",
                "open_amo_statuses": row["open_amo_statuses"] or "",
            }
        )
    return result


def _loads(raw: object) -> Mapping[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, Mapping) else {}


def reconcile_header(reconcile: Mapping[str, Any]) -> str:
    if not reconcile:
        return "сверка с живым AMO: не проводилась"
    status = str(reconcile.get("status") or "unknown")
    if status == "checked":
        return (
            "сверка с живым AMO: "
            f"{reconcile.get('generated_at')}; "
            f"{reconcile.get('customers_changed')} расхождений из {reconcile.get('customers_checked')}; "
            f"snapshot_stale={reconcile.get('snapshot_stale')}"
        )
    return f"сверка с живым AMO: не проводилась ({reconcile.get('reason') or status})"


def build_workbook(
    *,
    timeline_db: Path,
    out_xlsx: Path,
    allowed_root: Path,
    reconcile_json: Path | None,
    limit_per_sheet: int,
) -> Mapping[str, Any]:
    db = timeline_db.expanduser().resolve(strict=False)
    out = _guard_local_wave0_output_path(out_xlsx, allowed_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    reconcile = _read_json(reconcile_json) if reconcile_json and reconcile_json.exists() else {}
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect_ro(db) as con:
        freshness = source_freshness(con)
        sheets = {
            "Зависшие факт LTV": build_rows(con, ("deal_stalling",), limit=limit_per_sheet),
            "Сезонные": build_rows(con, ("season_return_candidate",), limit=limit_per_sheet),
            "Вернулись и перезвон": build_rows(con, ("client_returned", "callback_due"), limit=limit_per_sheet),
        }
        quick_check = con.execute("PRAGMA quick_check").fetchone()[0]
    wb = Workbook()
    wb.remove(wb.active)
    freshness_text = "; ".join(f"{row['source_system']}={row['max_event_at']}" for row in freshness[:8])
    header_text = f"Данные: staging max event_at по источникам: {freshness_text}; собрано {generated_at}; {reconcile_header(reconcile)}"
    for sheet_name, rows in sheets.items():
        ws = wb.create_sheet(sheet_name)
        ws["A1"] = sheet_name
        ws["A1"].font = Font(bold=True, size=14)
        ws["A2"] = header_text
        ws["A2"].fill = PatternFill("solid", fgColor="FFF2CC")
        columns = [
            "customer_id",
            "display_name",
            "primary_phone",
            "primary_email",
            "signal_type",
            "severity",
            "fact_total_in",
            "plan_total_in",
            "fact_deals_cnt",
            "plan_deals_cnt",
            "latest_event_at",
            "signal_created_at",
            "expires_at",
            "recommended_action",
            "evidence_text",
            "open_amo_lead_ids",
            "open_amo_statuses",
        ]
        for col_idx, column in enumerate(columns, start=1):
            cell = ws.cell(row=4, column=col_idx, value=column)
            cell.font = Font(bold=True)
        for row_idx, row in enumerate(rows, start=5):
            for col_idx, column in enumerate(columns, start=1):
                ws.cell(row=row_idx, column=col_idx, value=row.get(column))
        ws.freeze_panes = "A5"
        ws.auto_filter.ref = f"A4:{get_column_letter(len(columns))}{max(4, len(rows) + 4)}"
        for col_idx, column in enumerate(columns, start=1):
            width = 18
            if column in {"display_name", "primary_email", "recommended_action", "evidence_text"}:
                width = 34
            ws.column_dimensions[get_column_letter(col_idx)].width = width
    wb.save(out)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "timeline_db": str(db),
        "source_open_mode": "sqlite_mode_ro_immutable",
        "quick_check": quick_check,
        "out_xlsx": str(out),
        "rows_by_sheet": {name: len(rows) for name, rows in sheets.items()},
        "signal_counts": dict(Counter(row["signal_type"] for rows in sheets.values() for row in rows)),
        "freshness_top": freshness[:12],
        "reconcile_status": reconcile.get("status") if reconcile else "missing",
        "reconcile_summary": {
            "customers_checked": reconcile.get("customers_checked"),
            "customers_changed": reconcile.get("customers_changed"),
            "snapshot_stale": reconcile.get("snapshot_stale"),
            "reason": reconcile.get("reason"),
        },
        "safety": {
            "prod_db_write": False,
            "crm_write": False,
            "send_messages": False,
            "pii_scope": "local_codex_local_only",
        },
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _guard_local_wave0_output_path(path: Path, allowed_root: Path) -> Path:
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    root = Path(allowed_root).expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - base guard normally catches this.
        raise ValueError("wave0 manager list output must stay under allowed root") from exc
    if not relative.parts or relative.parts[0] != ".codex_local":
        raise ValueError("wave0 manager list contains PII and must stay under .codex_local")
    return resolved


def _read_json(path: Path | None) -> Mapping[str, Any]:
    if not path:
        return {}
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local Wave-0 manager Excel lists from staging timeline.")
    parser.add_argument("--timeline-db", type=Path, default=DEFAULT_STAGING_DB)
    parser.add_argument("--allowed-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX)
    parser.add_argument("--reconcile-json", type=Path, default=DEFAULT_RECONCILE_JSON)
    parser.add_argument("--limit-per-sheet", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_workbook(
        timeline_db=args.timeline_db,
        out_xlsx=args.out_xlsx,
        allowed_root=args.allowed_root,
        reconcile_json=args.reconcile_json,
        limit_per_sheet=args.limit_per_sheet,
    )
    print(json.dumps({"out_xlsx": summary["out_xlsx"], "rows_by_sheet": summary["rows_by_sheet"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
