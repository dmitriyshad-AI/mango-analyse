from __future__ import annotations

import json
import sqlite3
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import mango_mvp.customer_timeline.stage5_money_ingest as stage5_module

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    OpportunityType,
)
from mango_mvp.customer_timeline.stage5_money_ingest import (
    STAGE5_AMO_PRICE_SOURCE_SYSTEM,
    STAGE5_MONEY_CODE_VERSION,
    Stage5MoneyIngestConfig,
    refresh_customer_purchases_v1,
    run_stage5_money_ingest,
)
from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore, customer_timeline_run_lock


NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def test_stage5_money_ingest_dry_run_does_not_write(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)

    report = run_stage5_money_ingest(
        Stage5MoneyIngestConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            source_path=source_path,
            out_dir=out_dir,
            apply=False,
        )
    )

    assert report["mode"] == "dry_run"
    assert report["plan"]["events_planned"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT count(*) FROM timeline_events WHERE source_system = ?",
            (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
        ).fetchone()[0] == 0
        assert con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='customer_purchases_v1'").fetchone() is None


def test_stage5_money_ingest_apply_is_idempotent_and_keeps_money_out_of_bot_context(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    config = Stage5MoneyIngestConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        source_path=source_path,
        out_dir=out_dir,
        apply=True,
    )

    first = run_stage5_money_ingest(config)
    second = run_stage5_money_ingest(config)

    assert first["final_checks"]["quick_check"] == "ok"
    assert second["final_checks"]["quick_check"] == "ok"
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        assert con.execute(
            "SELECT count(*) FROM timeline_events WHERE source_system = ?",
            (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
        ).fetchone()[0] == 1
        row = con.execute("SELECT * FROM customer_purchases_v1").fetchone()
        assert row["money_kind"] == "plan"
        assert row["total_in"] == 12000
        assert row["total_out"] == 0
        assert row["deals_cnt"] == 1
        assert row["computability"] == "computed"
        sources = json.loads(row["sources_json"])
        assert sources["email_amounts_used"] is False
        assert sources["source_event_system_counts"] == {STAGE5_AMO_PRICE_SOURCE_SYSTEM: 1}
        assert con.execute("SELECT count(*) FROM bot_context_chunks").fetchone()[0] == 0


def test_stage5_reconciles_invalid_plan_in_place_and_preserves_partial_snapshot(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    config = Stage5MoneyIngestConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        source_path=source_path,
        out_dir=out_dir,
        apply=True,
        as_of=NOW + timedelta(days=2),
    )
    run_stage5_money_ingest(config)

    def source_payload() -> dict:
        return json.loads(source_path.read_text(encoding="utf-8"))

    def write_source(payload: dict) -> None:
        source_path.write_text(json.dumps(payload), encoding="utf-8")

    with sqlite3.connect(db_path) as con:
        stable_event_id = con.execute(
            "SELECT event_id FROM timeline_events WHERE source_system=?",
            (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
        ).fetchone()[0]

    empty_price = source_payload()
    empty_price["amo_leads"][0]["price"] = None
    write_source(empty_price)
    retired = run_stage5_money_ingest(config)
    repeated = run_stage5_money_ingest(config)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        event = con.execute(
            "SELECT event_id,record_json FROM timeline_events WHERE source_system=?",
            (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
        ).fetchone()
        assert con.execute(
            "SELECT count(*) FROM customer_purchases_v1 WHERE tenant_id='foton' AND money_kind='plan'"
        ).fetchone()[0] == 0
    assert retired["plan"]["events_reconciled_inactive"] == 1
    assert repeated["apply"]["write_status_counts"] == {"duplicate": 1}
    assert event["event_id"] == stable_event_id
    assert json.loads(event["record_json"])["record"]["amount_rub"] == 0

    valid = source_payload()
    valid["amo_leads"][0]["price"] = 12000
    write_source(valid)
    run_stage5_money_ingest(config)
    partial = source_payload()
    partial["amo_leads"] = [lead for lead in partial["amo_leads"] if lead["id"] != 101]
    write_source(partial)
    missing = run_stage5_money_ingest(config)
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT total_in FROM customer_purchases_v1 WHERE tenant_id='foton' AND money_kind='plan'"
        ).fetchone()[0] == 12000
    assert missing["plan"]["skipped"]["amo_missing_from_source"] == 1

    write_source(valid)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id="customer-1",
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amocrm_snapshot",
                source_id="101",
                title="Paid deal",
                status="В работе",
                opened_at=NOW,
                confidence=0.99,
                product_context={"brand": "foton"},
            )
        )
    non_paid = run_stage5_money_ingest(config)
    assert non_paid["plan"]["events_reconciled_inactive"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT count(*) FROM customer_purchases_v1 WHERE tenant_id='foton' AND money_kind='plan'"
        ).fetchone()[0] == 0


def test_stage5_event_and_purchase_projection_roll_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    config = Stage5MoneyIngestConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        source_path=source_path,
        out_dir=out_dir,
        apply=True,
        as_of=NOW,
    )
    run_stage5_money_ingest(config)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["amo_leads"][0]["price"] = 15000
    source_path.write_text(json.dumps(source), encoding="utf-8")

    def rows() -> tuple[list[tuple], list[tuple]]:
        with sqlite3.connect(db_path) as con:
            events = con.execute(
                "SELECT event_id,event_at,record_hash,record_json FROM timeline_events "
                "WHERE source_system=? ORDER BY event_id",
                (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
            ).fetchall()
            purchases = con.execute(
                "SELECT * FROM customer_purchases_v1 ORDER BY tenant_id,customer_id,period,money_kind"
            ).fetchall()
        return events, purchases

    before = rows()
    real_refresh = stage5_module._refresh_customer_purchases_v1

    def fail_after_refresh(*args, **kwargs):
        real_refresh(*args, **kwargs)
        raise RuntimeError("fault after purchase refresh")

    monkeypatch.setattr(stage5_module, "_refresh_customer_purchases_v1", fail_after_refresh)
    with pytest.raises(RuntimeError, match="fault after purchase refresh"):
        run_stage5_money_ingest(config)

    assert rows() == before


def test_stage5_apply_entrypoints_share_the_nightly_run_lock(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    ready = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with customer_timeline_run_lock(db_path, timeout_seconds=1):
            ready.set()
            assert release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold_lock)
        assert ready.wait(timeout=2)
        with pytest.raises(TimeoutError, match="run lock timeout"):
            run_stage5_money_ingest(
                Stage5MoneyIngestConfig(
                    timeline_db_path=db_path,
                    allowed_root=tmp_path,
                    source_path=source_path,
                    out_dir=out_dir,
                    apply=True,
                    lock_timeout_seconds=0.01,
                )
            )
        with pytest.raises(TimeoutError, match="run lock timeout"):
            refresh_customer_purchases_v1(
                db_path,
                allowed_root=tmp_path,
                tenant_id="foton",
                lock_timeout_seconds=0.01,
            )
        dry_run = run_stage5_money_ingest(
            Stage5MoneyIngestConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                source_path=source_path,
                out_dir=out_dir,
                apply=False,
            )
        )
        release.set()
        future.result(timeout=2)

    assert dry_run["mode"] == "dry_run"
    with customer_timeline_run_lock(db_path, timeout_seconds=1):
        refresh_customer_purchases_v1(
            db_path,
            allowed_root=tmp_path,
            tenant_id="foton",
            lock_timeout_seconds=0.01,
        )


@pytest.mark.parametrize("match_status", ["strong_unique", "manual"])
def test_stage5_customer_purchases_splits_plan_and_tallanto_fact(
    tmp_path: Path,
    match_status: str,
) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    run_stage5_money_ingest(
        Stage5MoneyIngestConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            source_path=source_path,
            out_dir=out_dir,
            apply=True,
        )
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer-1",
                event_type=TimelineEventType.TALLANTO_PAYMENT,
                event_at=NOW,
                source_system="tallanto_crm_call",
                source_id="most_finances:pay-1",
                source_ref="tallanto:most_finances:pay-1",
                direction=TimelineDirection.SYSTEM,
                subject="Tallanto payment",
                summary="Оплата Tallanto",
                match_status=match_status,
                record={"amount": 7000, "payment_direction": "in"},
                created_at=NOW,
            )
        )

    result = refresh_customer_purchases_v1(db_path, allowed_root=tmp_path, tenant_id="foton")
    repeat = refresh_customer_purchases_v1(db_path, allowed_root=tmp_path, tenant_id="foton")

    assert result["money_kind"] == {"plan": 1, "fact": 1}
    assert repeat["stale_fact_rows_deleted"] == 0
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["money_kind"]: row
            for row in con.execute(
                "SELECT money_kind, total_in, deals_cnt, sources_json FROM customer_purchases_v1 ORDER BY money_kind"
            ).fetchall()
        }
    assert rows["plan"]["total_in"] == 12000
    assert rows["plan"]["deals_cnt"] == 1
    assert rows["fact"]["total_in"] == 7000
    assert rows["fact"]["deals_cnt"] == 1
    assert json.loads(rows["fact"]["sources_json"])["money_source"] == "tallanto_payment"


def test_purchase_refresh_replaces_stale_owner_after_payment_relink(tmp_path: Path) -> None:
    db_path, _, _ = _fixture(tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer-2",
                identity_status=IdentityStatus.STRONG,
                created_at=NOW,
                updated_at=NOW,
            )
        )

    def write_payment(customer_id: str, match_status: str) -> None:
        with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type=TimelineEventType.TALLANTO_PAYMENT,
                    event_at=NOW,
                    source_system="tallanto_crm_call",
                    source_id="most_finances:relinked-payment",
                    source_ref="tallanto:most_finances:relinked-payment",
                    direction=TimelineDirection.SYSTEM,
                    match_status=match_status,
                    record={"amount": 1000, "payment_direction": "in"},
                    created_at=NOW,
                )
            )

    write_payment("customer-1", "strong_unique")
    refresh_customer_purchases_v1(db_path, allowed_root=tmp_path, tenant_id="foton")
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT customer_id,total_in FROM customer_purchases_v1 WHERE money_kind='fact'"
        ).fetchall() == [("customer-1", 1000.0)]

    write_payment("customer-2", "ambiguous")
    refresh_customer_purchases_v1(db_path, allowed_root=tmp_path, tenant_id="foton")
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT customer_id,total_in FROM customer_purchases_v1 WHERE money_kind='fact'"
        ).fetchall() == []

    write_payment("customer-2", "strong_unique")
    refresh_customer_purchases_v1(db_path, allowed_root=tmp_path, tenant_id="foton")
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT customer_id,total_in FROM customer_purchases_v1 WHERE money_kind='fact'"
        ).fetchall() == [("customer-2", 1000.0)]


def test_stage5_tallanto_balance_charge_does_not_become_refund_or_new_purchase(tmp_path: Path) -> None:
    db_path, _, _ = _fixture(tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for source_id, event_at, direction in (
            ("pay-in", NOW, "in"),
            ("pay-out", NOW + timedelta(days=1), "out"),
            ("pay-school-out", NOW + timedelta(days=2), " school_out "),
        ):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id="customer-1",
                    event_type=TimelineEventType.TALLANTO_PAYMENT,
                    event_at=event_at,
                    source_system="tallanto_crm_call",
                    source_id=source_id,
                    source_ref=f"tallanto:most_finances:{source_id}",
                    direction=TimelineDirection.SYSTEM,
                    subject="Tallanto payment",
                    summary=direction,
                    match_status="strong_unique",
                    record={"amount": 7000, "payment_direction": direction},
                    created_at=event_at,
                )
            )

    refresh_customer_purchases_v1(db_path, allowed_root=tmp_path, tenant_id="foton")

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        fact = con.execute(
            "SELECT total_in,total_out,deals_cnt,last_purchase_at FROM customer_purchases_v1 WHERE money_kind='fact'"
        ).fetchone()
    assert fact["total_in"] == 7000
    assert fact["total_out"] == 7000
    assert fact["deals_cnt"] == 1
    assert fact["last_purchase_at"] == NOW.isoformat()


def test_purchase_refresh_uses_one_as_of_and_fails_closed(tmp_path: Path) -> None:
    db_path, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="as_of"):
        refresh_customer_purchases_v1(
            db_path,
            allowed_root=tmp_path,
            tenant_id="foton",
            as_of=NOW.replace(tzinfo=None),
        )
    mixed_earlier = datetime(2026, 7, 2, 13, 30, tzinfo=timezone(timedelta(hours=3)))
    mixed_later = datetime(2026, 7, 2, 11, 45, tzinfo=timezone.utc)
    events = (
        ("past", NOW - timedelta(days=1), 1000, "in"),
        ("mixed-earlier", mixed_earlier, 500, "in"),
        ("mixed-later", mixed_later, 700, "in"),
        ("boundary-utc", NOW, 2000, "in"),
        ("future-in", NOW + timedelta(days=1), 4000, "in"),
        ("future-out", NOW + timedelta(days=1, seconds=1), 5000, "school_out"),
        ("naive", NOW - timedelta(days=2), 6000, "in"),
        ("invalid", NOW - timedelta(days=3), 7000, "in"),
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for source_id, event_at, amount, direction in events:
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id="customer-1",
                    event_type=TimelineEventType.TALLANTO_PAYMENT,
                    event_at=event_at,
                    source_system="tallanto_crm_call",
                    source_id=source_id,
                    source_ref=f"tallanto:most_finances:{source_id}",
                    direction=TimelineDirection.SYSTEM,
                    match_status="strong_unique",
                    record={"amount": amount, "payment_direction": direction},
                    created_at=event_at,
                )
            )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE timeline_events SET event_at=? WHERE source_id=?",
            ("2026-06-30T12:00:00", "naive"),
        )
        con.execute(
            "UPDATE timeline_events SET event_at=? WHERE source_id=?",
            ("not-a-timestamp", "invalid"),
        )

    first = refresh_customer_purchases_v1(
        db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        as_of=NOW,
    )
    repeat = refresh_customer_purchases_v1(
        db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        as_of=NOW,
    )
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT total_in,total_out,deals_cnt,last_purchase_at "
            "FROM customer_purchases_v1 WHERE customer_id='customer-1' AND money_kind='fact'"
        ).fetchone()
    assert first == repeat
    assert first["as_of"] == NOW.isoformat()
    assert row[:3] == (4200.0, 0.0, 4)
    assert datetime.fromisoformat(row[3]).astimezone(timezone.utc) == NOW

    refresh_customer_purchases_v1(
        db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        as_of=NOW + timedelta(days=2),
    )
    with sqlite3.connect(db_path) as con:
        later = con.execute(
            "SELECT total_in,total_out,deals_cnt,last_purchase_at "
            "FROM customer_purchases_v1 WHERE customer_id='customer-1' AND money_kind='fact'"
        ).fetchone()
    assert later == (8200.0, 5000.0, 5, (NOW + timedelta(days=1)).isoformat())


def test_stage5_missing_amo_event_time_uses_run_as_of_boundary(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["amo_leads"][0]["updated_at"] = None
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE customer_opportunities SET opened_at=NULL, closed_at=NULL WHERE source_id='101'"
        )

    report = run_stage5_money_ingest(
        Stage5MoneyIngestConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            source_path=source_path,
            out_dir=out_dir,
            apply=True,
            as_of=NOW,
        )
    )
    with sqlite3.connect(db_path) as con:
        event_at = con.execute(
            "SELECT event_at FROM timeline_events WHERE source_system=?",
            (STAGE5_AMO_PRICE_SOURCE_SYSTEM,),
        ).fetchone()[0]
        plan = con.execute(
            "SELECT total_in,deals_cnt FROM customer_purchases_v1 WHERE money_kind='plan'"
        ).fetchone()
    assert event_at == NOW.isoformat()
    assert plan == (12000.0, 1)


def test_stage5_retires_stale_future_purchase_history_without_touching_payment_event(tmp_path: Path) -> None:
    db_path, _source_path, _out_dir = _fixture(tmp_path)
    payment_at = NOW - timedelta(days=30)
    false_future_at = NOW + timedelta(days=180)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        payment = TimelineEvent(
            tenant_id="foton",
            customer_id="customer-1",
            event_type=TimelineEventType.TALLANTO_PAYMENT,
            event_at=payment_at,
            source_system="tallanto_crm_call",
            source_id="confirmed-payment",
            source_ref="tallanto:most_finances:confirmed-payment",
            direction=TimelineDirection.SYSTEM,
            match_status="strong_unique",
            record={"amount": 5000, "payment_direction": "in"},
            created_at=payment_at,
        )
        store.upsert_event(payment)
        for chunk_id, projected_at in (
            ("legacy-future-purchase", false_future_at),
            ("canonical-purchase", payment_at),
        ):
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id="customer-1",
                    chunk_id=chunk_id,
                    source_system="customer_purchases_v1",
                    source_ref=f"customer_purchases_v1:customer-1:all_time:fact:{chunk_id}",
                    chunk_type="purchase_history",
                    text="Подтверждённая оплата.",
                    summary="Подтверждённая оплата.",
                    event_at=projected_at,
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata={
                        "client_safe": True,
                        "last_purchase_at": projected_at.isoformat(),
                        "total_in": 5000,
                        "total_out": 0,
                        "deals_cnt": 1,
                    },
                    created_at=NOW,
                )
            )

    first = refresh_customer_purchases_v1(
        db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        as_of=NOW,
    )
    repeat = refresh_customer_purchases_v1(
        db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        as_of=NOW,
    )
    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        bot_context = api.bot_context("foton", "customer-1", allowed_only=True, as_of=NOW)

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        chunk = con.execute(
            "SELECT allowed_for_bot,requires_manager_review,superseded_by FROM bot_context_chunks "
            "WHERE chunk_id='legacy-future-purchase'"
        ).fetchone()
        canonical_chunk = con.execute(
            "SELECT allowed_for_bot,requires_manager_review,superseded_by FROM bot_context_chunks "
            "WHERE chunk_id='canonical-purchase'"
        ).fetchone()
        payment = con.execute(
            "SELECT superseded_by FROM timeline_events WHERE source_id='confirmed-payment'"
        ).fetchone()
        fact = con.execute(
            "SELECT last_purchase_at FROM customer_purchases_v1 "
            "WHERE customer_id='customer-1' AND money_kind='fact'"
        ).fetchone()

    assert first["purchase_history_reconciliation"]["retired_chunks"] == 1
    assert first["purchase_history_reconciliation"]["active_mismatches_after"] == 0
    assert repeat["purchase_history_reconciliation"]["mismatched_chunks"] == 0
    assert repeat["purchase_history_reconciliation"]["retired_chunks"] == 0
    assert tuple(chunk) == (1, 0, "retired:purchase_projection_mismatch")
    assert tuple(canonical_chunk) == (1, 0, None)
    assert bot_context["summary"]["review_required_chunks"] == 0
    assert {item["chunk_id"] for item in bot_context["items"]} == {"canonical-purchase"}
    assert payment["superseded_by"] is None
    assert fact["last_purchase_at"] == payment_at.isoformat()


def test_stage5_is_sole_plan_owner_and_preserves_other_tenant(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    run_stage5_money_ingest(
        Stage5MoneyIngestConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            source_path=source_path,
            out_dir=out_dir,
            apply=True,
            as_of=NOW + timedelta(days=1),
        )
    )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE customer_purchases_v1 SET total_in=1, code_version='legacy-owner' "
            "WHERE tenant_id='foton' AND customer_id='customer-1' AND money_kind='plan'"
        )
        con.executemany(
            """
            INSERT INTO customer_purchases_v1 (
              tenant_id, customer_id, period, money_kind, total_in, total_out, deals_cnt,
              last_purchase_at, sources_json, computability, code_version
            ) VALUES (?, ?, 'all_time', 'plan', ?, 0, 0, NULL, '{}', ?, ?)
            """,
            (
                (
                    "foton",
                    "stale-a2",
                    None,
                    "not_computable_missing_primary_amounts",
                    "customer_purchases_v1_not_computable",
                ),
                ("foton", "stale-legacy", 5000, "computed", "legacy-v1"),
                ("unpk", "other-tenant", 7000, "computed", "legacy-v1"),
            ),
        )

    result = refresh_customer_purchases_v1(
        db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        as_of=NOW + timedelta(days=1),
    )
    repeat = refresh_customer_purchases_v1(
        db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        as_of=NOW + timedelta(days=1),
    )
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT tenant_id,customer_id,total_in,code_version FROM customer_purchases_v1 "
            "WHERE money_kind='plan' ORDER BY tenant_id,customer_id"
        ).fetchall()
    assert result["stale_plan_rows_deleted"] == 2
    assert repeat["stale_plan_rows_deleted"] == 0
    assert rows == [
        ("foton", "customer-1", 12000.0, STAGE5_MONEY_CODE_VERSION),
        ("unpk", "other-tenant", 7000.0, "legacy-v1"),
    ]


def test_stage5_apply_and_refresh_respect_store_writer_lock(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    config = Stage5MoneyIngestConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        source_path=source_path,
        out_dir=out_dir,
        apply=True,
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path):
        with pytest.raises(RuntimeError, match="writer lock"):
            run_stage5_money_ingest(config)
        with pytest.raises(RuntimeError, match="writer lock"):
            refresh_customer_purchases_v1(
                db_path,
                allowed_root=tmp_path,
                tenant_id="foton",
            )
        dry_run = run_stage5_money_ingest(
            Stage5MoneyIngestConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                source_path=source_path,
                out_dir=out_dir,
                apply=False,
            )
        )
    assert dry_run["mode"] == "dry_run"


def test_stage5_migrates_legacy_schema_and_removes_noncanonical_plan(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE customer_purchases_v1 (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              period TEXT NOT NULL,
              total_in REAL,
              total_out REAL,
              deals_cnt INTEGER NOT NULL DEFAULT 0,
              last_purchase_at TEXT,
              sources_json TEXT NOT NULL,
              computability TEXT NOT NULL,
              code_version TEXT NOT NULL,
              PRIMARY KEY (tenant_id, customer_id, period)
            );
            INSERT INTO customer_purchases_v1 VALUES (
              'foton', 'legacy-customer', 'all_time', 5000, 0, 1,
              '2026-01-01T00:00:00+00:00', '{}', 'computed', 'legacy'
            );
            """
        )

    run_stage5_money_ingest(
        Stage5MoneyIngestConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            source_path=source_path,
            out_dir=out_dir,
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT customer_id, money_kind, total_in, code_version
            FROM customer_purchases_v1
            ORDER BY customer_id, money_kind
            """
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("customer-1", "plan", 12000.0, STAGE5_MONEY_CODE_VERSION),
    ]


def test_stage5_money_ingest_refuses_prod_and_non_staging_paths(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    non_staging = tmp_path / "customer_timeline.sqlite"
    non_staging.write_bytes(db_path.read_bytes())
    prod_path = tmp_path / "customer_timeline_prod_20260621" / "customer_timeline.sqlite"
    prod_path.parent.mkdir(parents=True)
    prod_path.write_bytes(db_path.read_bytes())

    with pytest.raises(ValueError, match=".codex_local/staging"):
        run_stage5_money_ingest(
            Stage5MoneyIngestConfig(
                timeline_db_path=non_staging,
                allowed_root=tmp_path,
                source_path=source_path,
                out_dir=out_dir,
                apply=True,
            )
        )
    with pytest.raises(ValueError, match="prod timeline"):
        run_stage5_money_ingest(
            Stage5MoneyIngestConfig(
                timeline_db_path=prod_path,
                allowed_root=tmp_path,
                source_path=source_path,
                out_dir=out_dir,
                apply=True,
            )
        )


def test_stage5_money_ingest_refuses_source_and_outside_artifacts(tmp_path: Path) -> None:
    db_path, source_path, out_dir = _fixture(tmp_path)
    outside_source = tmp_path / "stage5_amo_prices.json"
    outside_source.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    outside_out = tmp_path / "reports"

    with pytest.raises(ValueError, match=".codex_local/staging"):
        run_stage5_money_ingest(
            Stage5MoneyIngestConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                source_path=outside_source,
                out_dir=out_dir,
                apply=False,
            )
        )
    with pytest.raises(ValueError, match=".codex_local/staging"):
        run_stage5_money_ingest(
            Stage5MoneyIngestConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                source_path=source_path,
                out_dir=outside_out,
                apply=False,
            )
        )


def test_fetch_stage5_script_guards_safe_projection_and_staging_paths(tmp_path: Path) -> None:
    module = _load_fetch_script()
    projected = module._safe_lead_projection(
        {
            "id": "101",
            "name": "must not persist",
            "price": "12000",
            "status_id": "1",
            "pipeline_id": "2",
            "_embedded": {"contacts": [{"id": 1}]},
            "custom_fields": [{"field_name": "phone", "values": ["secret"]}],
        }
    )

    assert projected == {
        "id": 101,
        "price": 12000,
        "status_id": 1,
        "status_name": None,
        "pipeline_id": 2,
        "pipeline_name": None,
        "created_at": None,
        "updated_at": None,
        "closed_at": None,
    }
    with pytest.raises(ValueError, match=".codex_local/staging"):
        module._guard_staging_path(tmp_path / "out.json", tmp_path, label="output")
    stage = tmp_path / ".codex_local" / "staging"
    stage.mkdir(parents=True)
    module._guard_staging_path(stage / "out.json", tmp_path, label="output")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    stage = tmp_path / ".codex_local" / "staging"
    stage.mkdir(parents=True)
    db_path = stage / "customer_timeline.sqlite"
    source_path = stage / "stage5_amo_prices.json"
    out_dir = stage / "reports"
    _seed_db(db_path, tmp_path)
    source_path.write_text(
        json.dumps(
            {
                "amo_leads": [
                    {"id": 101, "price": 12000, "updated_at": 1783000000, "status_id": 1, "pipeline_id": 2},
                    {"id": 102, "price": 99999, "updated_at": 1783000000, "status_id": 3, "pipeline_id": 2},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return db_path, source_path, out_dir


def _load_fetch_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "fetch_stage5_amo_prices_readonly.py"
    spec = importlib.util.spec_from_file_location("fetch_stage5_amo_prices_readonly", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_db(db_path: Path, allowed_root: Path) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            customer_id="customer-1",
            identity_status=IdentityStatus.STRONG,
            source_ref="seed",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(customer)
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id="customer-1",
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amocrm_snapshot",
                source_id="101",
                title="Paid deal",
                status="Оплата получена",
                opened_at=NOW,
                confidence=0.99,
                product_context={"brand": "foton"},
            )
        )
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id="customer-1",
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amocrm_snapshot",
                source_id="102",
                title="Open deal",
                status="В работе",
                opened_at=NOW,
                confidence=0.99,
                product_context={"brand": "foton"},
            )
        )
