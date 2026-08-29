from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    IdentityMatchClass,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.purchases import (
    CANONICAL_PURCHASE_FACT_CODE_VERSION,
    CANONICAL_PURCHASE_FACT_IDENTITY_PROOF,
    PURCHASE_MONEY_KIND_FACT,
    PURCHASE_MONEY_KIND_PLAN,
    canonical_purchase_fact_sql,
    ensure_customer_purchases_v1_table,
    upsert_customer_purchase_rows,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.source_policy import (
    PURCHASE_HISTORY_BOT_TEXT,
    PURCHASE_HISTORY_CHUNK_TYPE,
    PURCHASE_HISTORY_PROJECTION_OWNER,
    PURCHASE_HISTORY_PROJECTION_VERSION,
    PURCHASE_HISTORY_SEMANTIC_SCOPE,
    PURCHASE_HISTORY_SOURCE_SYSTEM,
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    authoritative_tallanto_student_owners,
    customer_timeline_run_lock,
    json_dumps,
    json_loads,
)
from mango_mvp.customer_timeline.temporal import (
    normalize_aware_utc,
    parse_aware_utc,
    register_temporal_sql_functions,
)


STAGE5_MONEY_INGEST_SCHEMA_VERSION = "stage5_money_ingest_v1"
STAGE5_AMO_PRICE_SOURCE_SYSTEM = "amocrm_price_readonly"
STAGE5_MONEY_CODE_VERSION = CANONICAL_PURCHASE_FACT_CODE_VERSION
PAID_AMO_STATUSES = frozenset({"Оплата получена", "Успешно", "won", "success", "paid"})


@dataclass(frozen=True)
class Stage5MoneyIngestConfig:
    timeline_db_path: Path
    allowed_root: Path
    source_path: Path
    out_dir: Path
    tenant_id: str = "foton"
    apply: bool = False
    allow_test_paths: bool = False
    as_of: datetime | None = None
    lock_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db_path", Path(self.timeline_db_path).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "source_path", Path(self.source_path).expanduser())
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        if self.as_of is not None:
            object.__setattr__(self, "as_of", normalize_aware_utc(self.as_of))
        if self.lock_timeout_seconds < 0:
            raise ValueError("lock_timeout_seconds must not be negative")


@dataclass(frozen=True)
class PlannedMoneyEvent:
    tenant_id: str
    customer_id: str
    opportunity_id: str | None
    source_system: str
    source_id: str
    source_ref: str
    event_type: str
    event_at: datetime
    amount_rub: float
    direction: str
    summary: str
    subject: str
    record: Mapping[str, Any]


def run_stage5_money_ingest(config: Stage5MoneyIngestConfig) -> Mapping[str, Any]:
    db_path = guard_customer_timeline_output_path(config.timeline_db_path, config.allowed_root)
    if not config.apply:
        return _run_stage5_money_ingest_unlocked(config)
    with customer_timeline_run_lock(db_path, timeout_seconds=config.lock_timeout_seconds):
        return _run_stage5_money_ingest_unlocked(config)


def _run_stage5_money_ingest_unlocked(config: Stage5MoneyIngestConfig) -> Mapping[str, Any]:
    started = time.monotonic()
    generated_at = datetime.now(timezone.utc)
    run_as_of = normalize_aware_utc(config.as_of or generated_at)
    db_path = guard_customer_timeline_output_path(config.timeline_db_path, config.allowed_root)
    source_path = guard_customer_timeline_output_path(config.source_path, config.allowed_root)
    out_dir = guard_customer_timeline_output_path(config.out_dir, config.allowed_root)
    _assert_stage5_staging_path(db_path, config.allowed_root, allow_test_paths=config.allow_test_paths)
    _assert_stage5_staging_path(source_path, config.allowed_root, allow_test_paths=config.allow_test_paths)
    _assert_stage5_staging_path(out_dir, config.allowed_root, allow_test_paths=config.allow_test_paths)
    if not source_path.exists():
        raise FileNotFoundError(f"stage5 money source does not exist: {source_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    source = _load_source(source_path)
    store = (
        CustomerTimelineSQLiteStore(db_path, allowed_root=config.allowed_root)
        if config.apply
        else CustomerTimelineSQLiteStore.open_read_only(
            db_path,
            allowed_root=config.allowed_root,
        )
    )
    con = store._con
    try:
        before = _metrics(con)
        plans, skipped = _build_money_event_plan(
            con,
            source,
            tenant_id=config.tenant_id,
            as_of=run_as_of,
        )
        report: dict[str, Any] = {
            "schema_version": STAGE5_MONEY_INGEST_SCHEMA_VERSION,
            "mode": "apply" if config.apply else "dry_run",
            "timeline_db_path": str(db_path),
            "source_path": str(source_path),
            "tenant_id": config.tenant_id,
            "generated_at": generated_at.isoformat(),
            "as_of": run_as_of.isoformat(),
            "safety": {
                "prod_write": False,
                "crm_write": False,
                "tallanto_write": False,
                "client_send": False,
                "llm_calls_total": 0,
                "raw_source_payload_persisted": False,
            },
            "source": _source_summary(source),
            "before": before,
            "plan": _plan_summary(plans, skipped),
        }
        if config.apply:
            with store.bulk_write():
                write_result = _apply_plan(store, plans)
                purchases_result = _refresh_customer_purchases_v1(
                    con,
                    tenant_id=config.tenant_id,
                    as_of=run_as_of,
                )
                purchases_result = {
                    **purchases_result,
                    "purchase_history_reconciliation": _reconcile_purchase_history_chunks_on_store(
                        store,
                        tenant_id=config.tenant_id,
                        as_of=run_as_of,
                    ),
                }
                store._commit()  # noqa: SLF001 - include direct projection SQL in the same Store transaction.
            report["apply"] = {**write_result, "customer_purchases_v1": purchases_result}
        else:
            report["apply"] = {"events_written": 0, "customer_purchases_v1": {"rows_upserted": 0}, "dry_run": True}
        after = _metrics(con)
        report["after"] = after
        report["final_checks"] = {
            "quick_check": con.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_check_rows": len(con.execute("PRAGMA foreign_key_check").fetchall()),
            "bot_context_chunks_from_money_sources": _scalar(
                con,
                """
                SELECT count(*)
                FROM bot_context_chunks
                WHERE source_system IN (?, 'tallanto_crm_call')
                """,
                (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
            ),
        }
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (out_dir / "stage5_money_ingest_report.json").write_text(json_dumps(report), encoding="utf-8")
        return report
    finally:
        store.close()


def refresh_customer_purchases_v1(
    db_path: Path,
    *,
    allowed_root: Path,
    tenant_id: str,
    as_of: datetime | None = None,
    lock_timeout_seconds: float = 30.0,
) -> Mapping[str, Any]:
    db_path = guard_customer_timeline_output_path(db_path, allowed_root)
    _assert_stage5_staging_path(db_path, allowed_root, allow_test_paths=False)
    cutoff = normalize_aware_utc(as_of or datetime.now(timezone.utc))
    if lock_timeout_seconds < 0:
        raise ValueError("lock_timeout_seconds must not be negative")
    with customer_timeline_run_lock(db_path, timeout_seconds=lock_timeout_seconds):
        with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
            with store.bulk_write():
                result = _refresh_customer_purchases_v1(
                    store._con,
                    tenant_id=tenant_id,
                    as_of=cutoff,
                )
                result = {
                    **result,
                    "purchase_history_reconciliation": _reconcile_purchase_history_chunks_on_store(
                        store,
                        tenant_id=tenant_id,
                        as_of=cutoff,
                    ),
                }
                store._commit()  # noqa: SLF001 - defer projection SQL to the outer atomic bulk write.
            return result


def _refresh_customer_purchases_v1(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    as_of: datetime,
) -> Mapping[str, Any]:
    cutoff = normalize_aware_utc(as_of)
    register_temporal_sql_functions(con)
    ensure_customer_purchases_v1_table(con)
    aggregates = _purchase_aggregates(con, tenant_id=tenant_id, as_of=cutoff)
    rows = [_purchase_row(row) for row in aggregates.values()]
    desired_fact_keys = {
        (str(row["customer_id"]), str(row["period"]))
        for row in rows
        if row["money_kind"] == PURCHASE_MONEY_KIND_FACT
    }
    existing_fact_keys = {
        (str(row[0]), str(row[1]))
        for row in con.execute(
            "SELECT customer_id,period FROM customer_purchases_v1 WHERE tenant_id=? AND money_kind=?",
            (tenant_id, PURCHASE_MONEY_KIND_FACT),
        )
    }
    stale_fact_keys = existing_fact_keys - desired_fact_keys
    desired_plan_keys = {
        (str(row["customer_id"]), str(row["period"]))
        for row in rows
        if row["money_kind"] == PURCHASE_MONEY_KIND_PLAN
    }
    existing_plan_keys = {
        (str(row[0]), str(row[1]))
        for row in con.execute(
            "SELECT customer_id,period FROM customer_purchases_v1 "
            "WHERE tenant_id=? AND money_kind=?",
            (tenant_id, PURCHASE_MONEY_KIND_PLAN),
        )
    }
    stale_plan_keys = existing_plan_keys - desired_plan_keys
    con.executemany(
        "DELETE FROM customer_purchases_v1 "
        "WHERE tenant_id=? AND customer_id=? AND period=? AND money_kind=?",
        (
            (tenant_id, customer_id, period, money_kind)
            for money_kind, stale_keys in (
                (PURCHASE_MONEY_KIND_PLAN, stale_plan_keys),
                (PURCHASE_MONEY_KIND_FACT, stale_fact_keys),
            )
            for customer_id, period in stale_keys
        ),
    )
    upsert_customer_purchase_rows(con, rows)
    return {
        "as_of": cutoff.isoformat(),
        "rows_upserted": len(rows),
        "stale_fact_rows_deleted": len(stale_fact_keys),
        "stale_plan_rows_deleted": len(stale_plan_keys),
        "customers_with_money": sum(
            1 for row in rows if (row["total_in"] or 0) or (row["total_out"] or 0)
        ),
        "total_in": round(sum(float(row["total_in"] or 0) for row in rows), 2),
        "total_out": round(sum(float(row["total_out"] or 0) for row in rows), 2),
        "computability": dict(Counter(str(row["computability"]) for row in rows)),
        "money_kind": dict(Counter(str(row["money_kind"]) for row in rows)),
    }


def _apply_plan(
    store: CustomerTimelineSQLiteStore,
    plans: Sequence[PlannedMoneyEvent],
) -> Mapping[str, Any]:
    status_counts: Counter[str] = Counter()
    for plan in plans:
        result = store.upsert_event(_event_from_plan(plan), actor="stage5_money_ingest")
        status_counts[result.status] += 1
    return {"events_written": len(plans), "write_status_counts": dict(status_counts)}


def _reconcile_purchase_history_chunks_on_store(
    store: CustomerTimelineSQLiteStore,
    *,
    tenant_id: str,
    as_of: datetime,
) -> Mapping[str, Any]:
    """Own one neutral bot projection for every eligible canonical payment fact."""

    cutoff = normalize_aware_utc(as_of)
    con = store._con  # noqa: SLF001 - projection repair shares the Stage5 transaction.
    facts = {
        str(row["customer_id"]): row
        for row in con.execute(
            """
            SELECT p.customer_id,p.total_in,p.total_out,p.deals_cnt,p.last_purchase_at,
                   p.sources_json,p.computability,p.code_version,i.identity_status
            FROM customer_purchases_v1 p
            LEFT JOIN customer_identities i
              ON i.tenant_id=p.tenant_id AND i.customer_id=p.customer_id
            WHERE p.tenant_id=? AND p.period='all_time' AND p.money_kind=?
            """,
            (tenant_id, PURCHASE_MONEY_KIND_FACT),
        )
    }
    eligible_customer_ids = {
        str(row[0])
        for row in con.execute(
            f"""
            SELECT purchase_fact.customer_id
            FROM customer_purchases_v1 purchase_fact
            WHERE purchase_fact.tenant_id=?
              AND EXISTS (
                SELECT 1 FROM customer_identities purchase_identity
                WHERE purchase_identity.tenant_id=purchase_fact.tenant_id
                  AND purchase_identity.customer_id=purchase_fact.customer_id
                  AND purchase_identity.identity_status='strong'
              )
              AND {canonical_purchase_fact_sql('purchase_fact')}
            """,
            (tenant_id, cutoff.isoformat()),
        )
    }
    eligible: dict[str, tuple[sqlite3.Row, datetime]] = {}
    ineligible_reasons: dict[str, str] = {}
    for customer_id, fact in facts.items():
        fact_at = parse_aware_utc(fact["last_purchase_at"])
        total_in = _money_value(fact["total_in"])
        total_out = _money_value(fact["total_out"])
        if str(fact["identity_status"] or "") != "strong":
            ineligible_reasons[customer_id] = "identity_not_strong"
        elif str(fact["code_version"] or "") != CANONICAL_PURCHASE_FACT_CODE_VERSION:
            ineligible_reasons[customer_id] = "canonical_fact_wrong_code_version"
        elif not _canonical_purchase_sources(fact["sources_json"]):
            ineligible_reasons[customer_id] = "canonical_fact_wrong_provenance"
        elif str(fact["computability"] or "") != "computed":
            ineligible_reasons[customer_id] = "canonical_fact_not_computed"
        elif int(fact["deals_cnt"] or 0) <= 0:
            ineligible_reasons[customer_id] = "canonical_fact_zero_deals"
        elif total_in is None or total_in <= 0 or total_out is None:
            ineligible_reasons[customer_id] = "canonical_fact_invalid_amounts"
        elif fact_at is None or fact_at > cutoff:
            ineligible_reasons[customer_id] = "canonical_fact_invalid_date"
        elif customer_id in eligible_customer_ids:
            eligible[customer_id] = (fact, fact_at)
        else:
            ineligible_reasons[customer_id] = "canonical_fact_contract_mismatch"
    chunks = con.execute(
        """
        SELECT chunk_id,customer_id,source_ref,event_at,freshness_score,created_at,record_json
        FROM bot_context_chunks
        WHERE tenant_id=? AND source_system=?
          AND chunk_type=? AND superseded_by IS NULL
        ORDER BY chunk_id
        """,
        (tenant_id, PURCHASE_HISTORY_SOURCE_SYSTEM, PURCHASE_HISTORY_CHUNK_TYPE),
    ).fetchall()
    reasons: Counter[str] = Counter()
    retired = 0
    projected: Counter[str] = Counter()
    covered: set[str] = set()
    for chunk in chunks:
        customer_id = str(chunk["customer_id"])
        eligible_fact = eligible.get(customer_id)
        if eligible_fact is not None:
            fact, fact_at = eligible_fact
            expected = _purchase_history_chunk(
                tenant_id=tenant_id,
                customer_id=customer_id,
                fact=fact,
                fact_at=fact_at,
                created_at=parse_aware_utc(chunk["created_at"]),
            )
            if expected.chunk_id == str(chunk["chunk_id"]):
                result = store.upsert_bot_context_chunk(
                    expected,
                    actor=PURCHASE_HISTORY_PROJECTION_OWNER,
                )
                projected[result.status] += 1
                covered.add(customer_id)
                continue
            reason = "noncanonical_projection_id"
        else:
            reason = ineligible_reasons.get(customer_id, "canonical_fact_missing")
        reasons[reason] += 1
        result = store.retire_bot_context_chunk(
            str(chunk["chunk_id"]),
            reason="purchase_projection_not_eligible",
            actor=PURCHASE_HISTORY_PROJECTION_OWNER,
        )
        retired += int(result.status == "updated")

    for customer_id in sorted(set(eligible) - covered):
        fact, fact_at = eligible[customer_id]
        result = store.upsert_bot_context_chunk(
            _purchase_history_chunk(
                tenant_id=tenant_id,
                customer_id=customer_id,
                fact=fact,
                fact_at=fact_at,
            ),
            actor=PURCHASE_HISTORY_PROJECTION_OWNER,
        )
        projected[result.status] += 1
        covered.add(customer_id)

    active_after = int(
        con.execute(
            "SELECT COUNT(*) FROM bot_context_chunks WHERE tenant_id=? AND source_system=? "
            "AND chunk_type=? AND superseded_by IS NULL",
            (tenant_id, PURCHASE_HISTORY_SOURCE_SYSTEM, PURCHASE_HISTORY_CHUNK_TYPE),
        ).fetchone()[0]
    )
    active_mismatches_after = abs(active_after - len(eligible))
    if active_mismatches_after:
        raise RuntimeError(
            "purchase history projection reconciliation left active mismatches: "
            f"{active_mismatches_after}"
        )
    return {
        "active_chunks_seen": len(chunks),
        "canonical_fact_rows": len(facts),
        "eligible_fact_rows": len(eligible),
        "mismatched_chunks": retired,
        "retired_chunks": retired,
        "projected_chunk_status_counts": dict(sorted(projected.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "active_chunks_after": active_after,
        "active_mismatches_after": active_mismatches_after,
    }


def _purchase_history_chunk(
    *,
    tenant_id: str,
    customer_id: str,
    fact: sqlite3.Row,
    fact_at: datetime,
    created_at: datetime | None = None,
) -> BotContextChunk:
    """Build the sole brand-neutral purchase projection from one canonical fact."""

    total_in = _money_value(fact["total_in"])
    total_out = _money_value(fact["total_out"])
    if total_in is None or total_out is None:
        raise ValueError("canonical purchase fact must contain finite money values")

    return BotContextChunk(
        tenant_id=tenant_id,
        customer_id=customer_id,
        source_system=PURCHASE_HISTORY_SOURCE_SYSTEM,
        source_ref=f"purchases:{customer_id}",
        chunk_type=PURCHASE_HISTORY_CHUNK_TYPE,
        text=PURCHASE_HISTORY_BOT_TEXT,
        summary=PURCHASE_HISTORY_BOT_TEXT,
        event_at=fact_at,
        freshness_score=0.5,
        relevance_tags=("purchase", "bot_visible", PURCHASE_HISTORY_SOURCE_SYSTEM),
        allowed_for_bot=True,
        requires_manager_review=False,
        metadata={
            "client_safe": True,
            "client_safe_reason": "canonical_historical_payment_fact",
            "client_safe_policy_version": "cs_v1",
            "client_safe_provenance": PURCHASE_HISTORY_PROJECTION_OWNER,
            "projection_owner": PURCHASE_HISTORY_PROJECTION_OWNER,
            "projection_version": PURCHASE_HISTORY_PROJECTION_VERSION,
            "semantic_scope": PURCHASE_HISTORY_SEMANTIC_SCOPE,
            "memory_status": "usable_memory",
            "raw_text_used": False,
            "current_access_asserted": False,
            "last_purchase_at": fact_at.isoformat(),
            "total_in": round(total_in, 2),
            "total_out": round(total_out, 2),
            "deals_cnt": int(fact["deals_cnt"]),
        },
        created_at=created_at or fact_at,
    )


def _event_from_plan(plan: PlannedMoneyEvent) -> TimelineEvent:
    return TimelineEvent(
        tenant_id=plan.tenant_id,
        customer_id=plan.customer_id,
        opportunity_id=plan.opportunity_id,
        event_type=plan.event_type,
        event_at=plan.event_at,
        source_system=plan.source_system,
        source_id=plan.source_id,
        source_ref=plan.source_ref,
        direction=TimelineDirection.SYSTEM,
        subject=plan.subject,
        summary=plan.summary,
        text_preview=plan.summary,
        match_status=IdentityMatchClass.STRONG_UNIQUE,
        confidence=0.99,
        record=plan.record,
        metadata={
            "source_kind": "stage5_primary_money",
            "plan_active": plan.amount_rub > 0,
            "raw_payload_persisted": False,
            "bot_context_chunk_created": False,
        },
        created_at=plan.event_at,
    )


def _build_money_event_plan(
    con: sqlite3.Connection,
    source: Mapping[str, Any],
    *,
    tenant_id: str,
    as_of: datetime,
) -> tuple[tuple[PlannedMoneyEvent, ...], Mapping[str, int]]:
    skipped: Counter[str] = Counter()
    amo_by_id = _amo_leads_by_id(source)
    existing_price_events = {
        str(row["source_id"]): parse_aware_utc(row["event_at"])
        for row in con.execute(
            "SELECT source_id,event_at FROM timeline_events "
            "WHERE tenant_id=? AND source_system=? AND superseded_by IS NULL",
            (tenant_id, STAGE5_AMO_PRICE_SOURCE_SYSTEM),
        ).fetchall()
    }
    rows = con.execute(
        """
        SELECT tenant_id, opportunity_id, customer_id, source_id, status, title, opened_at, closed_at, record_json
        FROM customer_opportunities
        WHERE tenant_id = ?
          AND source_system = 'amocrm_snapshot'
          AND source_id GLOB '[0-9]*'
        ORDER BY CAST(source_id AS INTEGER), opportunity_id
        """,
        (tenant_id,),
    ).fetchall()
    plans: list[PlannedMoneyEvent] = []
    for row in rows:
        lead_id = str(row["source_id"])
        price_source_id = f"lead:{lead_id}:price"
        existing_event_at = existing_price_events.get(price_source_id)
        status = str(row["status"] or "")
        if not _is_paid_status(status):
            skipped["amo_not_paid_status"] += 1
            if existing_event_at is not None:
                plans.append(
                    _money_event_plan(
                        row,
                        amo_by_id.get(lead_id) or {},
                        amount=0.0,
                        event_at=existing_event_at,
                        inactive_reason="amo_not_paid_status",
                    )
                )
                skipped["amo_projection_reconciled_inactive"] += 1
            continue
        lead = amo_by_id.get(lead_id)
        if not lead:
            skipped["amo_missing_from_source"] += 1
            continue
        source_status = str(lead.get("status_name") or lead.get("status") or "").strip()
        if source_status and not _is_paid_status(source_status):
            skipped["amo_source_not_paid_status"] += 1
            if existing_event_at is not None:
                plans.append(
                    _money_event_plan(
                        row,
                        lead,
                        amount=0.0,
                        event_at=existing_event_at,
                        inactive_reason="amo_source_not_paid_status",
                    )
                )
                skipped["amo_projection_reconciled_inactive"] += 1
            continue
        amount = _money_value(lead.get("price"))
        if amount is None or amount <= 0:
            skipped["amo_empty_price"] += 1
            if existing_event_at is not None:
                plans.append(
                    _money_event_plan(
                        row,
                        lead,
                        amount=0.0,
                        event_at=existing_event_at,
                        inactive_reason="amo_empty_price",
                    )
                )
                skipped["amo_projection_reconciled_inactive"] += 1
            continue
        raw_event_at = lead.get("closed_at") or row["closed_at"] or lead.get("updated_at") or row["opened_at"]
        event_at = _event_datetime(raw_event_at, default=existing_event_at or as_of)
        if event_at is None:
            skipped["amo_invalid_event_at"] += 1
            if existing_event_at is not None:
                plans.append(
                    _money_event_plan(
                        row,
                        lead,
                        amount=0.0,
                        event_at=existing_event_at,
                        inactive_reason="amo_invalid_event_at",
                    )
                )
                skipped["amo_projection_reconciled_inactive"] += 1
            continue
        plans.append(
            _money_event_plan(
                row,
                lead,
                amount=amount,
                event_at=event_at,
            )
        )
    skipped.update(_tallanto_snapshot_skips(source))
    return tuple(plans), dict(skipped)


def _money_event_plan(
    row: sqlite3.Row,
    lead: Mapping[str, Any],
    *,
    amount: float,
    event_at: datetime,
    inactive_reason: str | None = None,
) -> PlannedMoneyEvent:
    lead_id = str(row["source_id"])
    active = amount > 0
    record = {
        "source": "amo_get_lead_or_amo_api_get",
        "source_payload": "safe_projection",
        "raw_payload_persisted": False,
        "lead_id": lead_id,
        "amount_rub": amount,
        "direction": "in",
        "currency": "RUB",
        "status": str(row["status"] or ""),
        "amo_status_id": lead.get("status_id"),
        "amo_status_name": lead.get("status_name"),
        "pipeline_id": lead.get("pipeline_id"),
        "pipeline_name": lead.get("pipeline_name"),
        "opportunity_title_hash": stable_digest({"title": row["title"] or ""}),
        "money_source": "amo_lead_price",
        "plan_active": active,
        "inactive_reason": inactive_reason,
    }
    return PlannedMoneyEvent(
        tenant_id=str(row["tenant_id"]),
        customer_id=str(row["customer_id"]),
        opportunity_id=str(row["opportunity_id"]),
        source_system=STAGE5_AMO_PRICE_SOURCE_SYSTEM,
        source_id=f"lead:{lead_id}:price",
        source_ref=f"amocrm:lead:{lead_id}",
        event_type=TimelineEventType.AMO_DEAL_STAGE.value,
        event_at=event_at,
        amount_rub=amount,
        direction="in",
        subject="AMO deal price",
        summary=(
            "AMO deal price imported from read-only price field"
            if active
            else "AMO deal price projection reconciled inactive"
        ),
        record=record,
    )


def _purchase_aggregates(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    as_of: datetime,
) -> dict[tuple[str, str], dict[str, Any]]:
    exact_tallanto_owners = authoritative_tallanto_student_owners(con, tenant_id)
    unresolved_payment_refs = {
        str(row[0])
        for row in con.execute(
            "SELECT CAST(ref.value AS TEXT) FROM timeline_conflicts conflict, "
            "json_each(conflict.record_json, '$.entity_refs') ref "
            "WHERE conflict.tenant_id=? AND conflict.status IN ('open','active') "
            "AND conflict.conflict_type='tallanto_payment_owner_unresolved'",
            (tenant_id,),
        )
        if row[0]
    }
    aggregates: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "tenant_id": tenant_id,
            "customer_id": "",
            "period": "all_time",
            "money_kind": PURCHASE_MONEY_KIND_PLAN,
            "total_in": 0.0,
            "total_out": 0.0,
            "deals": set(),
            "last_purchase_at": None,
            "last_purchase_key": None,
            "sources": Counter(),
            "exact_owner_incoming_event_count": 0,
        }
    )
    for row in con.execute(
        """
        SELECT event_id, customer_id, opportunity_id, event_at, source_system, source_id,
               source_ref, record_json
        FROM timeline_events
        WHERE tenant_id = ?
          AND customer_id IS NOT NULL
          AND customer_id != ''
          AND superseded_by IS NULL
          AND mango_tz_at_or_before(event_at, ?) = 1
          AND (
            source_system = ?
            OR (source_system = 'tallanto_crm_call' AND event_type = 'tallanto_payment'
                AND match_status IN ('strong_unique', 'manual'))
          )
        """,
        (tenant_id, as_of.isoformat(), STAGE5_AMO_PRICE_SOURCE_SYSTEM),
    ):
        payload = json_loads(row["record_json"])
        record = payload.get("record") or {}
        amount = _money_value(record.get("amount_rub") if "amount_rub" in record else record.get("amount"))
        if amount is None or amount <= 0:
            continue
        direction = _money_direction(record)
        customer_id = str(row["customer_id"])
        money_kind = PURCHASE_MONEY_KIND_FACT if row["source_system"] == "tallanto_crm_call" else PURCHASE_MONEY_KIND_PLAN
        if money_kind == PURCHASE_MONEY_KIND_FACT:
            contact_id = str(record.get("contact_id") or "").strip()
            if (
                str(row["source_ref"] or "") in unresolved_payment_refs
                or str(record.get("contact_id_source") or "").strip() != "direct"
                or record.get("contact_id_conflict") is not False
                or exact_tallanto_owners.get(contact_id) != customer_id
            ):
                continue
        item = aggregates[(customer_id, money_kind)]
        item["customer_id"] = customer_id
        item["money_kind"] = money_kind
        if direction == "out":
            item["total_out"] += amount
        elif direction == "in":
            item["total_in"] += amount
        # Tallanto `out` is normally a paired balance charge, not a refund or a
        # second purchase. Only confirmed incoming money advances purchase facts.
        if direction == "in":
            if money_kind == PURCHASE_MONEY_KIND_FACT:
                item["exact_owner_incoming_event_count"] += 1
            deal_or_payment_key = row["opportunity_id"] or (
                row["source_id"] if money_kind == PURCHASE_MONEY_KIND_FACT else None
            )
            if deal_or_payment_key:
                item["deals"].add(str(deal_or_payment_key))
            current_at = parse_aware_utc(row["event_at"])
            if current_at is None:
                raise ValueError("temporal predicate admitted invalid event_at")
            current_key = (current_at.timestamp(), str(row["event_id"]))
            if item["last_purchase_key"] is None or current_key > item["last_purchase_key"]:
                item["last_purchase_at"] = current_at.isoformat()
                item["last_purchase_key"] = current_key
        item["sources"][str(row["source_system"])] += 1
    return aggregates


def _purchase_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = row["sources"]
    source_provenance: dict[str, Any] = {
        "source": "stage5_primary_money_events",
        "email_amounts_used": False,
        "source_event_system_counts": dict(sorted(sources.items())),
        "money_source": (
            "tallanto_payment"
            if row["money_kind"] == PURCHASE_MONEY_KIND_FACT
            else "amo_lead_price"
        ),
    }
    if row["money_kind"] == PURCHASE_MONEY_KIND_FACT:
        source_provenance.update(
            {
                "identity_owner_proof": CANONICAL_PURCHASE_FACT_IDENTITY_PROOF,
                "exact_owner_incoming_event_count": int(
                    row["exact_owner_incoming_event_count"]
                ),
            }
        )
    return {
        "tenant_id": row["tenant_id"],
        "customer_id": row["customer_id"],
        "period": row["period"],
        "money_kind": row["money_kind"],
        "total_in": round(float(row["total_in"]), 2),
        "total_out": round(float(row["total_out"]), 2),
        "deals_cnt": len(row["deals"]),
        "last_purchase_at": row["last_purchase_at"],
        "sources_json": json_dumps(source_provenance),
        "computability": "computed",
        "code_version": STAGE5_MONEY_CODE_VERSION,
    }


def _load_source(source_path: Path) -> Mapping[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("stage5 money source must be a JSON object")
    return payload


def _amo_leads_by_id(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    leads: list[Any] = []
    for key in ("amo_leads", "amocrm_leads", "leads"):
        value = source.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            leads.extend(value)
    embedded = source.get("_embedded")
    if isinstance(embedded, Mapping) and isinstance(embedded.get("leads"), Sequence):
        leads.extend(embedded["leads"])
    result: dict[str, Mapping[str, Any]] = {}
    for item in leads:
        if not isinstance(item, Mapping):
            continue
        lead_id = str(item.get("id") or "").strip()
        if lead_id:
            result[lead_id] = dict(item)
    return result


def _source_summary(source: Mapping[str, Any]) -> Mapping[str, Any]:
    amo = _amo_leads_by_id(source)
    return {
        "amo_leads": len(amo),
        "tallanto_snapshot_present": any(key in source for key in ("tallanto_snapshot", "most_finances", "most_abonements")),
        "raw_payload_persisted": False,
    }


def _plan_summary(plans: Sequence[PlannedMoneyEvent], skipped: Mapping[str, int]) -> Mapping[str, Any]:
    active_plans = tuple(plan for plan in plans if plan.amount_rub > 0)
    by_source = Counter(plan.source_system for plan in active_plans)
    total_in = sum(plan.amount_rub for plan in active_plans if plan.direction == "in")
    total_out = sum(plan.amount_rub for plan in active_plans if plan.direction == "out")
    return {
        "events_planned": len(active_plans),
        "events_reconciled_inactive": len(plans) - len(active_plans),
        "customers": len({plan.customer_id for plan in active_plans}),
        "source_system_counts": dict(by_source),
        "total_in": round(total_in, 2),
        "total_out": round(total_out, 2),
        "skipped": dict(sorted(skipped.items())),
    }


def _metrics(con: sqlite3.Connection) -> Mapping[str, Any]:
    tables = {
        "timeline_events": _table_count(con, "timeline_events"),
        "bot_context_chunks": _table_count(con, "bot_context_chunks"),
        "customer_purchases_v1": _table_count(con, "customer_purchases_v1"),
        "stage5_amo_price_events": _scalar(
            con,
            "SELECT count(*) FROM timeline_events WHERE source_system = ?",
            (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
        )
        if _table_exists(con, "timeline_events")
        else 0,
    }
    purchases = {}
    if _table_exists(con, "customer_purchases_v1"):
        row = con.execute(
            """
            SELECT count(*) AS rows,
                   sum(CASE WHEN total_in IS NOT NULL OR total_out IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_totals,
                   coalesce(sum(total_in), 0) AS total_in,
                   coalesce(sum(total_out), 0) AS total_out
            FROM customer_purchases_v1
            """
        ).fetchone()
        purchases = dict(row)
        if "money_kind" in {str(item[1]) for item in con.execute("PRAGMA table_info(customer_purchases_v1)").fetchall()}:
            purchases["by_money_kind"] = {
                str(row["money_kind"]): int(row["cnt"])
                for row in con.execute(
                    """
                    SELECT money_kind, count(*) AS cnt
                    FROM customer_purchases_v1
                    GROUP BY money_kind
                    """
                ).fetchall()
            }
    return {"tables": tables, "purchases": purchases}


def _table_count(con: sqlite3.Connection, table: str) -> int:
    if not _table_exists(con, table):
        return 0
    return _scalar(con, f"SELECT count(*) FROM {table}")


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _scalar(con: sqlite3.Connection, query: str, params: Sequence[Any] = ()) -> int:
    return int(con.execute(query, tuple(params)).fetchone()[0])


def _money_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value).replace(" ", "").replace(",", "."))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return float(number)


def _canonical_purchase_sources(value: Any) -> bool:
    try:
        payload = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and (
        payload.get("source") == "stage5_primary_money_events"
        and payload.get("money_source") == "tallanto_payment"
        and payload.get("email_amounts_used") is False
        and payload.get("identity_owner_proof") == CANONICAL_PURCHASE_FACT_IDENTITY_PROOF
        and isinstance(payload.get("exact_owner_incoming_event_count"), int)
        and not isinstance(payload.get("exact_owner_incoming_event_count"), bool)
        and payload["exact_owner_incoming_event_count"] > 0
    )


def _event_datetime(value: Any, *, default: datetime) -> datetime | None:
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    if value not in (None, ""):
        return parse_aware_utc(value)
    return normalize_aware_utc(default, field_name="default_event_at")


def _is_paid_status(value: str) -> bool:
    normalized = value.strip()
    return normalized in PAID_AMO_STATUSES or normalized.casefold() in PAID_AMO_STATUSES


def _money_direction(record: Mapping[str, Any]) -> str:
    direction = str(record.get("direction") or record.get("payment_direction") or "").strip().casefold()
    if direction in {"school_out", "refund", "return", "возврат", "расход"}:
        return "out"
    if direction in {"in", "поступление на баланс"}:
        return "in"
    return "neutral"


def _tallanto_snapshot_skips(source: Mapping[str, Any]) -> Mapping[str, int]:
    if any(key in source for key in ("tallanto_snapshot", "most_finances", "most_abonements")):
        return {}
    return {"tallanto_source_not_supplied": 1}


def _assert_stage5_staging_path(path: Path, allowed_root: Path, *, allow_test_paths: bool) -> None:
    resolved = path.resolve(strict=False)
    if any("customer_timeline_prod_" in part for part in resolved.parts):
        raise ValueError(f"refusing to run stage5 money ingest on prod timeline path: {resolved}")
    if allow_test_paths:
        return
    root = allowed_root.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"stage5 money ingest DB must stay under allowed root: {root}") from exc
    parts = tuple(part.casefold() for part in resolved.parts)
    if not any(part == ".codex_local" and parts[index + 1] == "staging" for index, part in enumerate(parts[:-1])):
        raise ValueError("stage5 money ingest applies only to .codex_local/staging paths")


__all__ = [
    "STAGE5_AMO_PRICE_SOURCE_SYSTEM",
    "STAGE5_MONEY_INGEST_SCHEMA_VERSION",
    "Stage5MoneyIngestConfig",
    "refresh_customer_purchases_v1",
    "run_stage5_money_ingest",
]
