from __future__ import annotations

import json
import sqlite3
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.contracts import (
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
    Stage5MoneyIngestConfig,
    refresh_customer_purchases_v1,
    run_stage5_money_ingest,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


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


def test_stage5_migrates_legacy_customer_purchases_to_plan(tmp_path: Path) -> None:
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
        row = con.execute(
            """
            SELECT money_kind, total_in
            FROM customer_purchases_v1
            WHERE customer_id = 'legacy-customer'
            """
        ).fetchone()
    assert row["money_kind"] == "plan"
    assert row["total_in"] == 5000


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
