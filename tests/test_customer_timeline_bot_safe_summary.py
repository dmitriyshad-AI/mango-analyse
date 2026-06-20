from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.customer_timeline import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    CustomerTimelineSQLiteStore,
    IdentityStatus,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.bot_safe_summary import (
    BOT_SAFE_SUMMARY_CHUNK_TYPE,
    BOT_SAFE_SUMMARY_SOURCE_SYSTEM,
    BotSafeSummaryBuildConfig,
    build_bot_safe_summaries,
    expected_bot_safe_chunk_id,
)


NOW = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def _open_store(tmp_path: Path) -> CustomerTimelineSQLiteStore:
    return CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=StepClock())


def _customer(customer_id: str = "customer:bot-safe-1") -> CustomerIdentity:
    return CustomerIdentity(
        tenant_id="foton",
        customer_id=customer_id,
        identity_status=IdentityStatus.STRONG,
        display_name="Иванова Мария",
        primary_phone="+79161234567",
        primary_email="parent@example.com",
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=2,
        created_at=NOW,
        updated_at=NOW,
    )


def _opportunity(
    customer: CustomerIdentity,
    *,
    brand: str = "foton",
    source_id: str = "lead-1",
    title: str = "8 класс математика онлайн +7 916 123-45-67",
    status: str = "Ожидание оплаты",
) -> CustomerOpportunity:
    return CustomerOpportunity(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        opportunity_type=OpportunityType.AMO_DEAL,
        source_system="amocrm_snapshot",
        source_id=source_id,
        title=title,
        status=status,
        product_context={"brand": brand, "products_of_interest": [{"title": title}]},
        opened_at=NOW,
        confidence=0.9,
    )


def _event(customer: CustomerIdentity, *, source_id: str = "call-1") -> TimelineEvent:
    return TimelineEvent(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        event_type=TimelineEventType.MANGO_CALL,
        event_at=NOW,
        source_system="mango_processed_summary",
        source_id=source_id,
        direction=TimelineDirection.INBOUND,
        match_status="strong_unique",
        confidence=0.9,
        importance=3,
        subject="Вопрос",
        text_preview="Сырой текст не должен быть использован",
        summary="RAW_SECRET_SUMMARY parent@example.com +7 916 123-45-67 менеджер Петров сообщил личные детали",
        record={"next_step": "Позвонить клиенту по оплате"},
        created_at=NOW,
    )


def _raw_chunk(customer: CustomerIdentity, event: TimelineEvent) -> BotContextChunk:
    return BotContextChunk(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        event_id=event.event_id,
        source_system="mango_processed_summary",
        source_ref=event.event_id,
        chunk_type="mango_call_summary",
        text="RAW_SECRET_SUMMARY parent@example.com +7 916 123-45-67 менеджер Петров сообщил личные детали",
        summary="RAW_SECRET_SUMMARY",
        event_at=NOW,
        allowed_for_bot=False,
        requires_manager_review=False,
        created_at=NOW,
    )


def _load_bot_safe_text(db_path: Path) -> str:
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT record_json FROM bot_context_chunks WHERE chunk_type = ?",
            (BOT_SAFE_SUMMARY_CHUNK_TYPE,),
        ).fetchone()
    assert row is not None
    return row[0]


def test_bot_safe_summary_uses_structural_fields_redacts_title_and_keeps_raw_blocked(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    opportunity = _opportunity(customer)
    event = _event(customer)
    store.upsert_customer(customer)
    store.upsert_opportunity(opportunity)
    store.upsert_event(event)
    store.upsert_bot_context_chunk(_raw_chunk(customer, event))
    store.close()

    report = build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    )
    dumped = _load_bot_safe_text(tmp_path / "customer_timeline.sqlite")

    assert report.created == 1
    assert report.raw_allowed_chunks_after == 0
    assert "RAW_SECRET_SUMMARY" not in dumped
    assert "parent@example.com" not in dumped
    assert "+7 916 123-45-67" not in dumped
    assert "<phone_masked>" in dumped
    assert "Бренд: Фотон" in dumped
    assert "Ожидание оплаты" in dumped
    assert "Позвонить клиенту по оплате" in dumped
    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        assert con.execute(
            "SELECT COUNT(*) FROM bot_context_chunks WHERE chunk_type != ? AND allowed_for_bot = 1",
            (BOT_SAFE_SUMMARY_CHUNK_TYPE,),
        ).fetchone()[0] == 0


def test_bot_safe_summary_is_idempotent_by_botsafe_source_ref(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    opportunity = _opportunity(customer)
    store.upsert_customer(customer)
    store.upsert_opportunity(opportunity)
    store.close()

    config = BotSafeSummaryBuildConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        tenant_id="foton",
        apply=True,
    )
    first = build_bot_safe_summaries(config)
    second = build_bot_safe_summaries(config)
    expected_id = expected_bot_safe_chunk_id(tenant_id="foton", customer_id=customer.customer_id)

    assert first.created == 1
    assert second.created == 0
    assert second.updated == 0
    assert second.duplicate == 1
    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        rows = con.execute(
            "SELECT chunk_id, source_ref, source_system, allowed_for_bot, requires_manager_review FROM bot_context_chunks WHERE chunk_type = ?",
            (BOT_SAFE_SUMMARY_CHUNK_TYPE,),
        ).fetchall()
    assert rows == [(expected_id, f"botsafe:{customer.customer_id}", BOT_SAFE_SUMMARY_SOURCE_SYSTEM, 1, 0)]


def test_bot_safe_summary_cross_brand_fails_closed_without_brand_wording(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(_opportunity(customer, brand="foton", source_id="lead-foton"))
    store.upsert_opportunity(_opportunity(customer, brand="unpk", source_id="lead-unpk", title="ОГЭ физика очно"))
    store.close()

    report = build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    )
    dumped = _load_bot_safe_text(tmp_path / "customer_timeline.sqlite")

    assert report.brand_counts["unknown"] == 1
    assert "Бренд: Фотон" not in dumped
    assert "Бренд: УНПК" not in dumped


def test_bot_safe_summary_drops_other_brand_title_for_known_customer_brand(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(_opportunity(customer, brand="foton", source_id="lead-foton", title="Фотон математика онлайн"))
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="unknown",
            source_id="mail-unpk",
            title="Вы записаны очно УНПК МФТИ 10 класс",
            status="open",
        )
    )
    store.close()

    build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    )
    dumped = _load_bot_safe_text(tmp_path / "customer_timeline.sqlite")

    assert "Бренд: Фотон" in dumped
    assert "Фотон математика онлайн" in dumped
    assert "УНПК" not in dumped
    assert "МФТИ" not in dumped


def test_bot_safe_summary_drops_finance_and_discount_titles_from_interest(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(_opportunity(customer, brand="foton", source_id="lead-main", title="Фотон физика 9 класс"))
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="foton",
            source_id="mail-discount",
            title="Открыта запись на 2026-2027 учебный год со скидкой",
            status="open",
        )
    )
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="foton",
            source_id="mail-payment",
            title="Задолженность за занятия и квитанция об оплате",
            status="open",
        )
    )
    store.close()

    build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    )
    dumped = _load_bot_safe_text(tmp_path / "customer_timeline.sqlite")

    assert "Фотон физика 9 класс" in dumped
    assert "скид" not in dumped.casefold()
    assert "Задолженность" not in dumped
    assert "квитанц" not in dumped.casefold()
    assert "оплат" not in dumped.casefold()


def test_bot_safe_summary_drops_attachment_file_names_from_interest(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(_opportunity(customer, brand="foton", source_id="lead-main", title="Летняя выездная школа 2026"))
    store.upsert_opportunity(_opportunity(customer, brand="foton", source_id="mail-image", title="image-28-01-26-09-39.jpeg"))
    store.upsert_opportunity(_opportunity(customer, brand="foton", source_id="mail-pdf", title="presentation.pdf"))
    store.close()

    build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    )
    dumped = _load_bot_safe_text(tmp_path / "customer_timeline.sqlite")

    assert "Летняя выездная школа 2026" in dumped
    assert "image-" not in dumped
    assert ".jpeg" not in dumped
    assert ".pdf" not in dumped
