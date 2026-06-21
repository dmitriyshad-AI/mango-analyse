from __future__ import annotations

import json
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


def _event(customer: CustomerIdentity, *, source_id: str = "call-1", brand: str = "") -> TimelineEvent:
    record = {"next_step": "Позвонить клиенту по оплате"}
    if brand:
        record["brand"] = brand
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
        record=record,
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


def _load_bot_safe_payload(db_path: Path) -> dict:
    return json.loads(_load_bot_safe_text(db_path))


def _load_bot_safe_records(db_path: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(db_path) as con:
        return con.execute(
            "SELECT source_ref, record_json FROM bot_context_chunks WHERE chunk_type = ? ORDER BY source_ref",
            (BOT_SAFE_SUMMARY_CHUNK_TYPE,),
        ).fetchall()


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


def test_bot_safe_summary_extracts_call_summary_next_step_and_scrubs_pii(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    opportunity = _opportunity(customer)
    event = TimelineEvent(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        event_type=TimelineEventType.MANGO_CALL,
        event_at=NOW,
        source_system="mango_processed_summary",
        source_id="call-summary-pii",
        direction=TimelineDirection.INBOUND,
        match_status="strong_unique",
        confidence=0.9,
        importance=3,
        summary=(
            "Менеджер Клычева Дарья обсудила условия. "
            "Согласован следующий шаг: менеджер Клычева Дарья отправит договор ученику Смирнову Арсению "
            "по брони 64-64-58 на parent@example.com."
        ),
        record={"brand": "foton", "contentful": "Да", "duration_sec": 360, "manual_review_required": "Нет"},
        created_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_opportunity(opportunity)
    store.upsert_event(event)
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
    payload = _load_bot_safe_payload(tmp_path / "customer_timeline.sqlite")
    next_step = payload["metadata"]["next_step"]

    assert report.next_step_status_counts["active"] == 1
    assert next_step["status"] == "active"
    assert next_step["source_event_id"] == event.event_id
    assert "Следующий шаг:" in dumped
    assert "договор" in dumped.casefold()
    assert "Клычева" not in dumped
    assert "Дарья" not in dumped
    assert "Смирнов" not in dumped
    assert "Арсени" not in dumped
    assert "64-64-58" not in dumped
    assert "parent@example.com" not in dumped
    assert "<number_masked>" in dumped
    assert "<email_masked>" in dumped


def test_bot_safe_summary_open_ambiguous_identity_blocks_extracted_step(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    opportunity = _opportunity(customer)
    event = TimelineEvent(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        event_type=TimelineEventType.MANGO_CALL,
        event_at=NOW,
        source_system="mango_processed_summary",
        source_id="call-summary-conflict",
        direction=TimelineDirection.INBOUND,
        match_status="strong_unique",
        confidence=0.9,
        importance=3,
        summary="Согласован следующий шаг: отправить договор и документы на почту.",
        record={"brand": "foton", "contentful": "Да", "duration_sec": 360, "manual_review_required": "Нет"},
        created_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_opportunity(opportunity)
    store.upsert_event(event)
    store.record_conflict(
        customer.tenant_id,
        conflict_type="ambiguous_identity",
        entity_refs=("phone:+79161234567", customer.customer_id, "customer:other"),
        actor="test",
    )
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
    payload = _load_bot_safe_payload(tmp_path / "customer_timeline.sqlite")
    next_step = payload["metadata"]["next_step"]

    assert report.next_step_status_counts["needs_manager_review"] == 1
    assert next_step["status"] == "needs_manager_review"
    assert next_step["reason_code"] == "ambiguous_identity_open"
    assert "Уточнить у менеджера: открыт конфликт идентичности" in dumped
    assert "отправить договор" not in dumped.casefold()


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
    expected_id = expected_bot_safe_chunk_id(tenant_id="foton", customer_id=customer.customer_id, brand="foton")

    assert first.created == 1
    assert second.created == 0
    assert second.updated == 0
    assert second.duplicate == 1
    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        rows = con.execute(
            "SELECT chunk_id, source_ref, source_system, allowed_for_bot, requires_manager_review FROM bot_context_chunks WHERE chunk_type = ?",
            (BOT_SAFE_SUMMARY_CHUNK_TYPE,),
        ).fetchall()
    assert rows == [(expected_id, f"botsafe:{customer.customer_id}:foton", BOT_SAFE_SUMMARY_SOURCE_SYSTEM, 1, 0)]


def test_bot_safe_summary_cross_brand_customer_gets_separate_brand_scoped_chunks(tmp_path: Path) -> None:
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
    rows = _load_bot_safe_records(tmp_path / "customer_timeline.sqlite")

    assert report.brand_counts["foton"] == 1
    assert report.brand_counts["unpk"] == 1
    assert [row[0] for row in rows] == [
        f"botsafe:{customer.customer_id}:foton",
        f"botsafe:{customer.customer_id}:unpk",
    ]
    foton_record = rows[0][1]
    unpk_record = rows[1][1]
    assert "Бренд: Фотон" in foton_record
    assert "Бренд: УНПК" not in foton_record
    assert "Бренд: УНПК" in unpk_record
    assert "Бренд: Фотон" not in unpk_record


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


def test_bot_safe_summary_uses_event_brand_when_deal_brand_unknown(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(_opportunity(customer, brand="unknown", source_id="lead-unknown", title="Заявка"))
    store.upsert_event(_event(customer, source_id="event-unpk", brand="unpk"))
    store.close()

    report = build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    )
    rows = _load_bot_safe_records(tmp_path / "customer_timeline.sqlite")

    assert report.brand_counts["unpk"] == 1
    assert report.brand_source_counts["timeline_events.metadata_or_record.brand"] == 1
    assert rows[0][0] == f"botsafe:{customer.customer_id}:unpk"
    assert "Бренд: УНПК" in rows[0][1]


def test_bot_safe_summary_retires_stale_unknown_chunk_after_brand_resolution(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    old_unknown = BotContextChunk(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        chunk_type=BOT_SAFE_SUMMARY_CHUNK_TYPE,
        text="Стадия: old. Интерес: old. Следующий шаг: Активный следующий шаг не найден.",
        summary="Стадия: old. Интерес: old. Следующий шаг: Активный следующий шаг не найден.",
        source_system=BOT_SAFE_SUMMARY_SOURCE_SYSTEM,
        source_ref=f"botsafe:{customer.customer_id}:unknown",
        event_at=NOW,
        freshness_score=1.0,
        relevance_tags=("bot_safe", "structured", "unknown"),
        allowed_for_bot=True,
        requires_manager_review=False,
        created_at=NOW,
    )
    store.upsert_bot_context_chunk(old_unknown)
    store.upsert_event(_event(customer, source_id="event-unpk", brand="unpk"))
    store.close()

    report = build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    )
    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        rows = con.execute(
            """
            SELECT source_ref, allowed_for_bot, requires_manager_review, record_json
            FROM bot_context_chunks
            WHERE source_system = ?
            ORDER BY source_ref
            """,
            (BOT_SAFE_SUMMARY_SOURCE_SYSTEM,),
        ).fetchall()

    assert report.retired_stale == 1
    assert [row[0] for row in rows] == [
        f"botsafe:{customer.customer_id}:unknown",
        f"botsafe:{customer.customer_id}:unpk",
    ]
    assert rows[0][1:3] == (0, 1)
    assert rows[1][1:3] == (1, 0)
    retired = json.loads(rows[0][3])
    assert retired["metadata"]["retired_reason"] == "bot_safe_source_ref_not_rebuilt"


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


def test_bot_safe_summary_scrubs_names_from_interest_title_and_keeps_program(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="foton",
            source_id="lead-person-program",
            title="Летняя Выездная Школа для ученика Смирнова Арсения",
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

    assert "Летняя Выездная Школа" in dumped
    assert "<name_masked>" in dumped
    assert "Смирнов" not in dumped
    assert "Арсени" not in dumped


def test_bot_safe_summary_scrubs_role_name_from_interest_title(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="foton",
            source_id="lead-manager-name",
            title="менеджер Клычева Дарья подобрала Фотон физика 9 класс",
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
    assert "<name_masked>" in dumped
    assert "Клычева" not in dumped
    assert "Дарья" not in dumped


def test_bot_safe_summary_keeps_known_programs_and_organizations_in_interest(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="foton",
            source_id="lead-safe-names",
            title="ЛВШ Летняя Выездная Школа Альфа Банк Фотон ЕГЭ М9",
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

    assert "ЛВШ" in dumped
    assert "Летняя Выездная Школа" in dumped
    assert "Альфа Банк" in dumped
    assert "Фотон" in dumped
    assert "ЕГЭ" in dumped
    assert "М9" in dumped
    assert "<name_masked>" not in dumped


def test_bot_safe_summary_keeps_winter_school_and_intensive_titles(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="unpk",
            source_id="lead-winter-school",
            title="Зимняя Выездная школа; Интенсив Мат 11 кл УНПК",
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

    assert "Зимняя Выездная школа" in dumped
    assert "Интенсив Мат 11 кл УНПК" in dumped
    assert "<name_masked>" not in dumped


def test_bot_safe_summary_drops_interest_that_is_only_person_name(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="foton",
            source_id="lead-only-person",
            title="Иванова Мария Петровна",
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

    assert "Интерес: не определён" in dumped
    assert "Иванова" not in dumped
    assert "Мария" not in dumped
    assert "Петровна" not in dumped
    assert "<name_masked>" not in dumped


def test_bot_safe_summary_scrubs_single_person_name_from_interest_title(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    customer = _customer()
    store.upsert_customer(customer)
    store.upsert_opportunity(
        _opportunity(
            customer,
            brand="foton",
            source_id="lead-single-person",
            title="Дарья подобрала Фотон информатика 10 класс",
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

    assert "Фотон информатика 10 класс" in dumped
    assert "<name_masked>" in dumped
    assert "Дарья" not in dumped
