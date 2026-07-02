from __future__ import annotations

import sqlite3
from typing import Any, Mapping, Sequence


PURCHASE_MONEY_KIND_PLAN = "plan"
PURCHASE_MONEY_KIND_FACT = "fact"
PURCHASE_MONEY_KINDS = frozenset({PURCHASE_MONEY_KIND_PLAN, PURCHASE_MONEY_KIND_FACT})


def ensure_customer_purchases_v1_table(con: sqlite3.Connection) -> None:
    """Create or migrate customer_purchases_v1 to the plan/fact schema."""
    if not _table_exists(con, "customer_purchases_v1"):
        _create_customer_purchases_v1_table(con)
        return
    columns = _table_columns(con, "customer_purchases_v1")
    pk_columns = _primary_key_columns(con, "customer_purchases_v1")
    if "money_kind" not in columns or tuple(pk_columns) != ("tenant_id", "customer_id", "period", "money_kind"):
        _rebuild_customer_purchases_v1_table(con, columns=columns)
    _create_customer_purchases_v1_indexes(con)


def upsert_customer_purchase_rows(
    con: sqlite3.Connection,
    rows: Sequence[Mapping[str, Any]],
    *,
    protect_computed_plan: bool = False,
) -> None:
    if not rows:
        return
    ensure_customer_purchases_v1_table(con)
    where_clause = ""
    if protect_computed_plan:
        where_clause = """
                WHERE NOT (
                  customer_purchases_v1.money_kind = 'plan'
                  AND customer_purchases_v1.computability = 'computed'
                  AND excluded.computability <> 'computed'
                )
        """
    con.executemany(
        f"""
        INSERT INTO customer_purchases_v1 (
          tenant_id, customer_id, period, money_kind, total_in, total_out, deals_cnt,
          last_purchase_at, sources_json, computability, code_version
        )
        VALUES (
          :tenant_id, :customer_id, :period, :money_kind, :total_in, :total_out, :deals_cnt,
          :last_purchase_at, :sources_json, :computability, :code_version
        )
        ON CONFLICT(tenant_id, customer_id, period, money_kind) DO UPDATE SET
          total_in = excluded.total_in,
          total_out = excluded.total_out,
          deals_cnt = excluded.deals_cnt,
          last_purchase_at = excluded.last_purchase_at,
          sources_json = excluded.sources_json,
          computability = excluded.computability,
          code_version = excluded.code_version
        {where_clause}
        """,
        [normalize_customer_purchase_row(row) for row in rows],
    )


def normalize_customer_purchase_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    money_kind = str(row.get("money_kind") or PURCHASE_MONEY_KIND_PLAN)
    if money_kind not in PURCHASE_MONEY_KINDS:
        raise ValueError(f"unsupported customer_purchases_v1.money_kind: {money_kind}")
    return {
        "tenant_id": row["tenant_id"],
        "customer_id": row["customer_id"],
        "period": row.get("period") or "all_time",
        "money_kind": money_kind,
        "total_in": row.get("total_in"),
        "total_out": row.get("total_out"),
        "deals_cnt": int(row.get("deals_cnt") or 0),
        "last_purchase_at": row.get("last_purchase_at"),
        "sources_json": row.get("sources_json") or "{}",
        "computability": row.get("computability") or "unknown",
        "code_version": row.get("code_version") or "unknown",
    }


def _create_customer_purchases_v1_table(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS customer_purchases_v1 (
          tenant_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          period TEXT NOT NULL,
          money_kind TEXT NOT NULL DEFAULT 'plan'
            CHECK (money_kind IN ('plan', 'fact')),
          total_in REAL,
          total_out REAL,
          deals_cnt INTEGER NOT NULL DEFAULT 0,
          last_purchase_at TEXT,
          sources_json TEXT NOT NULL,
          computability TEXT NOT NULL,
          code_version TEXT NOT NULL,
          PRIMARY KEY (tenant_id, customer_id, period, money_kind)
        );
        """
    )
    _create_customer_purchases_v1_indexes(con)


def _create_customer_purchases_v1_indexes(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_customer_purchases_v1_customer
          ON customer_purchases_v1(tenant_id, customer_id, money_kind, deals_cnt);
        CREATE INDEX IF NOT EXISTS idx_customer_purchases_v1_computability
          ON customer_purchases_v1(tenant_id, money_kind, computability, last_purchase_at);
        """
    )


def _rebuild_customer_purchases_v1_table(con: sqlite3.Connection, *, columns: set[str]) -> None:
    money_kind_expr = "money_kind" if "money_kind" in columns else "'plan'"
    con.executescript(
        """
        DROP TABLE IF EXISTS customer_purchases_v1__v2;
        CREATE TABLE customer_purchases_v1__v2 (
          tenant_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          period TEXT NOT NULL,
          money_kind TEXT NOT NULL DEFAULT 'plan'
            CHECK (money_kind IN ('plan', 'fact')),
          total_in REAL,
          total_out REAL,
          deals_cnt INTEGER NOT NULL DEFAULT 0,
          last_purchase_at TEXT,
          sources_json TEXT NOT NULL,
          computability TEXT NOT NULL,
          code_version TEXT NOT NULL,
          PRIMARY KEY (tenant_id, customer_id, period, money_kind)
        );
        """
    )
    con.execute(
        f"""
        INSERT OR REPLACE INTO customer_purchases_v1__v2 (
          tenant_id, customer_id, period, money_kind, total_in, total_out, deals_cnt,
          last_purchase_at, sources_json, computability, code_version
        )
        SELECT
          tenant_id,
          customer_id,
          period,
          COALESCE(NULLIF({money_kind_expr}, ''), 'plan') AS money_kind,
          total_in,
          total_out,
          COALESCE(deals_cnt, 0),
          last_purchase_at,
          COALESCE(sources_json, '{{}}'),
          COALESCE(computability, 'unknown'),
          COALESCE(code_version, 'unknown')
        FROM customer_purchases_v1
        """
    )
    con.executescript(
        """
        DROP TABLE customer_purchases_v1;
        ALTER TABLE customer_purchases_v1__v2 RENAME TO customer_purchases_v1;
        """
    )
    _create_customer_purchases_v1_indexes(con)


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}


def _primary_key_columns(con: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    keyed = sorted((int(row[5]), str(row[1])) for row in rows if int(row[5] or 0) > 0)
    return tuple(name for _, name in keyed)


__all__ = [
    "PURCHASE_MONEY_KIND_FACT",
    "PURCHASE_MONEY_KIND_PLAN",
    "ensure_customer_purchases_v1_table",
    "normalize_customer_purchase_row",
    "upsert_customer_purchase_rows",
]
