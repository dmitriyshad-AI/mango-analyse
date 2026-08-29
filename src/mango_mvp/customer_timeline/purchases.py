from __future__ import annotations

import sqlite3
from typing import Any, Mapping, Sequence


PURCHASE_MONEY_KIND_PLAN = "plan"
PURCHASE_MONEY_KIND_FACT = "fact"
PURCHASE_MONEY_KINDS = frozenset({PURCHASE_MONEY_KIND_PLAN, PURCHASE_MONEY_KIND_FACT})
CANONICAL_PURCHASE_FACT_CODE_VERSION = "customer_purchases_v1_primary_money_v3"
CANONICAL_PURCHASE_FACT_IDENTITY_PROOF = "tallanto_student_id_exact_owner_v1"
CANONICAL_PURCHASE_FACT_REQUIRED_COLUMNS = frozenset(
    {
        "tenant_id",
        "customer_id",
        "period",
        "money_kind",
        "total_in",
        "total_out",
        "deals_cnt",
        "last_purchase_at",
        "sources_json",
        "computability",
        "code_version",
    }
)
EXPLICIT_REFUND_DIRECTIONS = frozenset({"refund", "return", "возврат"})


def is_explicit_refund_direction(value: Any) -> bool:
    return str(value or "").strip().casefold() in EXPLICIT_REFUND_DIRECTIONS


def _record_integrity_sql(table_alias: str, columns: Sequence[str]) -> str:
    return " AND ".join(
        (
            f"json_valid({table_alias}.record_json)=1",
            f"_mango_timeline_record_digest({table_alias}.record_json)={table_alias}.record_hash",
            *(
                f"json_extract({table_alias}.record_json, '$.{column}') IS {table_alias}.{column}"
                for column in columns
            ),
        )
    )


def _exact_tallanto_owner_sql(
    link_alias: str,
    event_alias: str,
    *,
    require_same_customer: bool,
) -> str:
    customer_clause = (
        f" AND {link_alias}.customer_id={event_alias}.customer_id"
        if require_same_customer
        else ""
    )
    return (
        f"{link_alias}.tenant_id={event_alias}.tenant_id{customer_clause} "
        f"AND {link_alias}.link_type='tallanto_student_id' "
        f"AND {link_alias}.link_value="
        f"json_extract({event_alias}.record_json, '$.record.contact_id') "
        f"AND {link_alias}.match_class IN ('strong_unique','manual') "
        f"AND {link_alias}.customer_id IS NOT NULL AND {link_alias}.customer_id!='' "
        f"AND {_record_integrity_sql(link_alias, ('link_id', 'tenant_id', 'customer_id', 'link_type', 'link_value', 'source_system', 'source_ref', 'match_class'))}"
    )


def canonical_purchase_fact_sql(table_alias: str = "purchase_fact") -> str:
    """One SQL eligibility contract shared by every customer-facing reader."""

    prefix = f"{table_alias}."
    return (
        f"{prefix}period='all_time' AND {prefix}money_kind='fact' "
        f"AND {prefix}computability='computed' "
        f"AND {prefix}code_version='{CANONICAL_PURCHASE_FACT_CODE_VERSION}' "
        f"AND typeof({prefix}total_in) IN ('integer','real') AND {prefix}total_in>0 "
        f"AND abs({prefix}total_in)<1e308 "
        f"AND typeof({prefix}total_out) IN ('integer','real') AND abs({prefix}total_out)<1e308 "
        f"AND {prefix}deals_cnt>0 "
        f"AND mango_tz_epoch({prefix}last_purchase_at) IS NOT NULL "
        f"AND mango_tz_at_or_before({prefix}last_purchase_at, ?)=1 "
        f"AND json_valid({prefix}sources_json)=1 "
        f"AND json_extract({prefix}sources_json, '$.source')='stage5_primary_money_events' "
        f"AND json_extract({prefix}sources_json, '$.money_source')='tallanto_payment' "
        f"AND json_type({prefix}sources_json, '$.email_amounts_used')='false' "
        f"AND json_extract({prefix}sources_json, '$.identity_owner_proof')="
        f"'{CANONICAL_PURCHASE_FACT_IDENTITY_PROOF}' "
        f"AND json_type({prefix}sources_json, '$.exact_owner_incoming_event_count')='integer' "
        f"AND json_extract({prefix}sources_json, '$.exact_owner_incoming_event_count')>0 "
        "AND EXISTS (SELECT 1 FROM timeline_events purchase_event "
        f"WHERE purchase_event.tenant_id={prefix}tenant_id "
        f"AND purchase_event.customer_id={prefix}customer_id "
        "AND purchase_event.source_system='tallanto_crm_call' "
        "AND purchase_event.event_type='tallanto_payment' "
        "AND purchase_event.match_status IN ('strong_unique','manual') "
        "AND purchase_event.superseded_by IS NULL "
        "AND json_valid(purchase_event.record_json)=1 "
        "AND _mango_timeline_record_digest(purchase_event.record_json)=purchase_event.record_hash "
        "AND json_extract(purchase_event.record_json, '$.event_id') IS purchase_event.event_id "
        "AND json_extract(purchase_event.record_json, '$.tenant_id') IS purchase_event.tenant_id "
        "AND json_extract(purchase_event.record_json, '$.customer_id') IS purchase_event.customer_id "
        "AND json_extract(purchase_event.record_json, '$.opportunity_id') IS purchase_event.opportunity_id "
        "AND json_extract(purchase_event.record_json, '$.event_type') IS purchase_event.event_type "
        "AND json_extract(purchase_event.record_json, '$.event_at') IS purchase_event.event_at "
        "AND json_extract(purchase_event.record_json, '$.source_system') IS purchase_event.source_system "
        "AND json_extract(purchase_event.record_json, '$.source_id') IS purchase_event.source_id "
        "AND json_extract(purchase_event.record_json, '$.source_ref') IS purchase_event.source_ref "
        "AND json_extract(purchase_event.record_json, '$.direction') IS purchase_event.direction "
        "AND json_extract(purchase_event.record_json, '$.match_status') IS purchase_event.match_status "
        "AND json_extract(purchase_event.record_json, '$.superseded_by') IS purchase_event.superseded_by "
        "AND mango_tz_at_or_before(purchase_event.event_at, "
        f"{prefix}last_purchase_at)=1 "
        "AND json_extract(purchase_event.record_json, '$.record.contact_id_source')='direct' "
        "AND json_type(purchase_event.record_json, '$.record.contact_id_conflict')='false' "
        "AND lower(trim(COALESCE("
        "json_extract(purchase_event.record_json, '$.record.direction'),"
        "json_extract(purchase_event.record_json, '$.record.payment_direction'),''))) "
        "IN ('in','поступление на баланс') "
        "AND EXISTS (SELECT 1 FROM identity_links purchase_owner "
        "INDEXED BY ix_identity_links_lookup WHERE "
        f"{_exact_tallanto_owner_sql('purchase_owner', 'purchase_event', require_same_customer=True)}) "
        "AND 1=(SELECT COUNT(DISTINCT exact_owner.customer_id) "
        "FROM identity_links exact_owner INDEXED BY ix_identity_links_lookup WHERE "
        f"{_exact_tallanto_owner_sql('exact_owner', 'purchase_event', require_same_customer=False)}) "
        "AND NOT EXISTS (SELECT 1 FROM timeline_conflicts purchase_conflict, "
        "json_each(purchase_conflict.record_json, '$.entity_refs') conflict_ref "
        "WHERE purchase_conflict.tenant_id=purchase_event.tenant_id "
        "AND purchase_conflict.conflict_type='tallanto_payment_owner_unresolved' "
        "AND purchase_conflict.status IN ('open','active') "
        "AND CAST(conflict_ref.value AS TEXT)=purchase_event.source_ref) "
        "AND NOT EXISTS (SELECT 1 FROM timeline_conflicts purchase_conflict, "
        "json_each(purchase_conflict.record_json, '$.entity_refs') conflict_ref "
        "WHERE purchase_conflict.tenant_id=purchase_event.tenant_id "
        "AND purchase_conflict.conflict_type='tallanto_identity_conflict' "
        "AND purchase_conflict.status IN ('open','active') "
        "AND CAST(conflict_ref.value AS TEXT) IN ("
        "'tallanto_student_id:' || json_extract(purchase_event.record_json, '$.record.contact_id'),"
        "'tallanto_student:' || json_extract(purchase_event.record_json, '$.record.contact_id'),"
        "'tallanto:student:' || json_extract(purchase_event.record_json, '$.record.contact_id')))"
        ")"
    )


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
) -> None:
    if not rows:
        return
    ensure_customer_purchases_v1_table(con)
    con.executemany(
        """
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
    con.execute(
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
        )
        """
    )
    _create_customer_purchases_v1_indexes(con)


def _create_customer_purchases_v1_indexes(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_customer_purchases_v1_customer
          ON customer_purchases_v1(tenant_id, customer_id, money_kind, deals_cnt)
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_customer_purchases_v1_computability
          ON customer_purchases_v1(tenant_id, money_kind, computability, last_purchase_at)
        """
    )


def _rebuild_customer_purchases_v1_table(con: sqlite3.Connection, *, columns: set[str]) -> None:
    money_kind_expr = "money_kind" if "money_kind" in columns else "'plan'"
    con.execute("DROP TABLE IF EXISTS customer_purchases_v1__v2")
    con.execute(
        """
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
        )
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
    con.execute("DROP TABLE customer_purchases_v1")
    con.execute("ALTER TABLE customer_purchases_v1__v2 RENAME TO customer_purchases_v1")
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
    "CANONICAL_PURCHASE_FACT_CODE_VERSION",
    "CANONICAL_PURCHASE_FACT_IDENTITY_PROOF",
    "CANONICAL_PURCHASE_FACT_REQUIRED_COLUMNS",
    "PURCHASE_MONEY_KIND_FACT",
    "PURCHASE_MONEY_KIND_PLAN",
    "canonical_purchase_fact_sql",
    "ensure_customer_purchases_v1_table",
    "is_explicit_refund_direction",
    "normalize_customer_purchase_row",
    "upsert_customer_purchase_rows",
]
