from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline import CustomerIdentity, CustomerTimelineSQLiteStore, IdentityStatus
from mango_mvp.customer_timeline.contracts import TimelineDirection, TimelineEvent, TimelineEventType
from mango_mvp.customer_timeline.objections import (
    OBJECTION_EXTRACTOR_VERSION,
    backfill_customer_objections_v1,
    extract_objections_from_text,
)


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def test_extract_objections_detects_types_and_budget_without_pii() -> None:
    items = extract_objections_from_text(
        "Мария, телефон +7 916 123-45-67, нам дорого, бюджет до 70 000 руб. "
        "Время не подходит, ребёнок не хочет заниматься."
    )

    by_type = {item.objection_type: item for item in items}
    assert {"price", "schedule", "child_refusal"} <= set(by_type)
    assert by_type["price"].budget_hint_rub == 70_000
    assert by_type["price"].price_sensitivity == "high"
    assert len(by_type["price"].quote_preview) <= 120
    assert "+7" not in by_type["price"].quote_preview
    assert "[phone]" in by_type["price"].quote_preview


def test_extract_objections_price_question_is_medium_not_high_and_budget_formats() -> None:
    medium = extract_objections_from_text("Подскажите, сколько стоит курс? Бюджет примерно 50-60 тыс.")
    hundred = extract_objections_from_text("Можем рассмотреть 100к, если есть рассрочка.")

    assert medium[0].objection_type == "price"
    assert medium[0].price_sensitivity == "medium"
    assert medium[0].budget_hint_rub == 50_000
    assert hundred[0].budget_hint_rub == 100_000


def test_extract_objections_avoids_substring_false_positive() -> None:
    assert extract_objections_from_text("Есть место в группе и удобный кабинет.") == ()


def test_extract_objections_masks_bare_phone_and_trims_after_mask() -> None:
    items = extract_objections_from_text(
        "Мария, пишите @parent_chat на почту very-long-parent-address-for-test@example.com "
        "или телефон (916) 123-45-67, адрес: Москва, ул. Лесная, дом 5. "
        "lead_id abcdef123456. Это дорого, нам нужна скидка, иначе не потянем такой бюджет."
    )

    assert items
    preview = items[0].quote_preview
    assert len(preview) <= 120
    assert "Мария" not in preview
    assert "916" not in preview
    assert "123-45-67" not in preview
    assert "example.com" not in preview
    assert "@parent_chat" not in preview
    assert "abcdef123456" not in preview
    assert "Лесная" not in preview


def test_extract_objections_masks_leading_name_near_marker() -> None:
    items = extract_objections_from_text("Анна Иванова, это дорого, просит скидку.")

    assert items
    assert "Анна" not in items[0].quote_preview
    assert "Иванова" not in items[0].quote_preview
    assert "[name]" in items[0].quote_preview


def test_extract_objections_masks_labeled_full_names() -> None:
    items = extract_objections_from_text(
        "мама Анна Иванова говорит, что дорого. родитель Сергей Петрович Иванов просит скидку."
    )

    assert items
    preview = items[0].quote_preview
    assert "Анна" not in preview
    assert "Иванова" not in preview
    assert "Сергей" not in preview
    assert "Петрович" not in preview
    assert "Иванов" not in preview
    assert "мама [name]" in preview


def test_extract_objections_masks_full_name_before_preview_cut() -> None:
    items = extract_objections_from_text(
        "Длинное начало переписки без смысла. родитель Сергей Петрович Иванов просит скидку, это дорого."
    )

    assert items
    preview = items[0].quote_preview
    assert "Сергей" not in preview
    assert "Петрович" not in preview
    assert "Иванов" not in preview
    assert "родитель [name]" in preview


def test_backfill_customer_objections_dry_run_does_not_create_tables_and_apply_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-1",
                direction=TimelineDirection.INBOUND,
                subject="Цена",
                summary="Клиент пишет, что дорого и просит скидку.",
                match_status="strong_unique",
                record={"full_clean_text": "Дорого, не потянем 120 000 руб. Можно скидку?"},
                created_at=NOW,
            )
        )
    finally:
        store.close()

    dry = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=False, as_of=NOW)
    assert dry["objections"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='customer_objections_v1'"
        ).fetchone() is None

    first = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)
    second = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)
    assert first["objection_type_counts"] == {"price": 1}
    assert second["objection_type_counts"] == {"price": 1}
    with sqlite3.connect(db) as con:
        row_count = con.execute("SELECT count(*) FROM customer_objections_v1").fetchone()[0]
        summary = con.execute("SELECT top_objections_json, max_price_sensitivity FROM customer_objection_summary_v1").fetchone()
    assert row_count == 1
    assert json.loads(summary[0]) == [["price", 1]]
    assert summary[1] == "high"
    assert first["extractor_version"] == OBJECTION_EXTRACTOR_VERSION


def test_backfill_customer_objections_removes_stale_rows_on_rebuild(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-1",
                direction=TimelineDirection.INBOUND,
                subject="Цена",
                summary="Дорого, просит скидку.",
                match_status="strong_unique",
                record={"full_clean_text": "Дорого, просит скидку."},
                created_at=NOW,
            )
        )
    finally:
        store.close()

    first = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)
    assert first["objections"] == 1
    with sqlite3.connect(db) as con:
        con.execute(
            "UPDATE timeline_events SET subject = ?, summary = ?, record_json = ? WHERE source_id = ?",
            ("Общее", "Обычное письмо", "{}", "mail-1"),
        )
        con.commit()

    second = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)
    assert second["objections"] == 0
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT count(*) FROM customer_objections_v1").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM customer_objection_summary_v1").fetchone()[0] == 0


def test_backfill_customer_objections_uses_only_email_head_not_summary_or_thread(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-1",
                direction=TimelineDirection.INBOUND,
                subject="Re: стоимость",
                text_preview="Спасибо, получили.",
                summary="Цепочка раньше содержала стоимость и скидку.",
                match_status="strong_unique",
                record={
                    "full_clean_text": "Спасибо, получили.",
                    "thread_context": "Менеджер написал: стоимость 120 000 руб., можно рассрочку.",
                    "summary": "Прайс и скидка из старого письма.",
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0


def test_backfill_customer_objections_strips_quoted_price_tail(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-1",
                direction=TimelineDirection.INBOUND,
                subject="Re: курс",
                text_preview="",
                summary="",
                match_status="strong_unique",
                record={
                    "full_clean_text": (
                        "Здравствуйте, нам дорого, не потянем сейчас.\n\n"
                        "В пн, 1 июн. 2026 г. в 12:00, менеджер написал(а):\n"
                        "Стоимость обучения:\n120 000 руб.\nРассрочка доступна."
                    )
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 1
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT quote_preview, budget_hint_rub FROM customer_objections_v1").fetchone()
    assert "нам дорого" in row[0].casefold()
    assert "120 000" not in row[0]
    assert row[1] is None


def test_backfill_customer_objections_catches_declined_price_confirmation(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-price-confirm",
                direction=TimelineDirection.INBOUND,
                subject="Re: стоимость",
                summary="",
                match_status="strong_unique",
                record={
                    "full_clean_text": (
                        "Правильно я понимаю цену: 75 000 рублей минус скидка 5%? "
                        "Итого 71 250 рублей?"
                    )
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objection_type_counts"] == {"price": 1}
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT quote_preview, budget_hint_rub FROM customer_objections_v1").fetchone()
    assert "понимаю цену" in row[0].casefold()
    assert row[1] == 75_000


def test_backfill_customer_objections_skips_bare_est_li_u_vas_question(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-vacation-question",
                direction=TimelineDirection.INBOUND,
                subject="Re: курс",
                summary="",
                match_status="strong_unique",
                record={"full_clean_text": "Здравствуйте, есть ли у вас каникулы в августе?"},
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0


def test_backfill_customer_objections_skips_non_client_price_template_even_if_inbound(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-template",
                direction=TimelineDirection.INBOUND,
                subject="Вы записаны на курс",
                summary="",
                match_status="strong_unique",
                record={
                    "full_clean_text": (
                        "Подготовительные курсы в 2026-2027 учебном году\n"
                        "Здравствуйте! Вы записаны на Подготовительные курсы УНПК МФТИ.\n"
                        "Стоимость обучения: 120 000 руб. Возможна оплата Долями."
                    )
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0


def test_backfill_customer_objections_skips_summer_school_offer_even_if_inbound(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-offer",
                direction=TimelineDirection.INBOUND,
                subject="Летняя Выездная школа 2026",
                summary="",
                match_status="strong_unique",
                record={
                    "full_clean_text": (
                        "*Добрый день!*\n"
                        "Отправляем Вам информацию по Летним выездным школам.\n"
                        "Стоимость смены 98 000 руб. При оплате до 1 апреля предоставляется скидка 10%.\n"
                        "Акции «Приведи друга»."
                    )
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0


def test_backfill_customer_objections_skips_manager_reply_before_quote(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-manager-reply",
                direction=TimelineDirection.INBOUND,
                subject="Летняя школа",
                summary="",
                match_status="strong_unique",
                record={
                    "full_clean_text": (
                        "Доброе утро! Лицевой счет не указываете, он только у бюджетных организаций.\n"
                        "Оплата за летнюю выездную школу ФИО ученика.\n\n"
                        "Родитель писал(а) 02.06.2026 19:46:\n"
                        "> Как оплатить курс?"
                    )
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0


def test_backfill_customer_objections_skips_manager_followup_discount_deadline(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-followup",
                direction=TimelineDirection.INBOUND,
                subject="Re: Летняя Выездная школа",
                summary="",
                match_status="strong_unique",
                record={
                    "full_clean_text": (
                        "Добрый день, не смогли до Вас дозвониться. "
                        "Подскажите, пожалуйста, актуальна ли Ваша запись на смену? "
                        "Скидка действует до 11 октября, далее будет дороже."
                    )
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0


def test_backfill_customer_objections_skips_external_service_spam(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-spam",
                direction=TimelineDirection.INBOUND,
                subject="По поводу пожарной безопасности",
                summary="",
                match_status="strong_unique",
                record={
                    "full_clean_text": (
                        "Здравствуйте! Пишу чтобы предложить услуги по пожарной безопасности. "
                        "Мои услуги стоят не дорого."
                    )
                },
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0


def test_backfill_customer_objections_uses_client_transcript_not_call_summary(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    canonical_db = tmp_path / "canonical_calls_master.db"
    _seed_canonical_calls(
        canonical_db,
        [
            (
                101,
                "Клиент: нам дорого, можем ли получить скидку? Бюджет ограничен, но курс интересен. "
                "Родитель подробно объяснил, что без посильной цены не сможет продолжить оформление.",
                "outbound",
            ),
            (202, "", "outbound"),
        ],
    )
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        for source_id, summary in (
            ("101", "Менеджер сказал, что цена стандартная."),
            ("202", "Менеджер сказал клиенту: это дорого, но скидки нет."),
            ("303", "Слитая сводка: дорого и нужна скидка."),
        ):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id="customer:1",
                    event_type="mango_call",
                    event_at=NOW,
                    source_system="mango_processed_summary",
                    source_id=source_id,
                    direction="outbound",
                    summary=summary,
                    match_status="strong_unique",
                    confidence=0.95,
                    record={},
                    created_at=NOW,
                )
            )
    finally:
        store.close()

    result = backfill_customer_objections_v1(
        db,
        allowed_root=tmp_path,
        canonical_calls_db_path=canonical_db,
        apply=True,
        as_of=NOW,
    )

    assert result["call_events_total"] == 3
    assert result["call_events_matched"] == 2
    assert result["call_events_unmatched"] == 1
    assert result["objection_type_counts"] == {"price": 1}
    with sqlite3.connect(db) as con:
        row = con.execute(
            "SELECT source_event_id, speaker, confidence, quote_preview FROM customer_objections_v1"
        ).fetchone()
    assert row[0]
    assert row[1] == "client"
    assert row[2] == "high"
    assert "нам дорого" in row[3].casefold()
    assert "Слитая сводка" not in row[3]


def test_backfill_customer_objections_differentiates_call_confidence(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    canonical_db = tmp_path / "canonical_calls_master.db"
    _seed_canonical_calls(
        canonical_db,
        [
            (
                101,
                "Клиент: нам дорого, просим скидку. Готовы обсуждать курс, если цена будет посильной. "
                "Родитель отдельно отметил ограничение бюджета и попросил не предлагать самый дорогой вариант.",
                "outbound",
            ),
            (202, "Дорого.", "outbound"),
        ],
    )
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        for source_id, confidence in (("101", 0.95), ("202", 0.55)):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id="customer:1",
                    event_type="mango_call",
                    event_at=NOW,
                    source_system="mango_processed_summary",
                    source_id=source_id,
                    direction="outbound",
                    summary="summary не используется",
                    match_status="strong_unique",
                    confidence=confidence,
                    record={},
                    created_at=NOW,
                )
            )
    finally:
        store.close()

    result = backfill_customer_objections_v1(
        db,
        allowed_root=tmp_path,
        canonical_calls_db_path=canonical_db,
        apply=True,
        as_of=NOW,
    )

    assert result["confidence_counts"] == {"high": 1, "low": 1}


def test_backfill_customer_objections_skips_outbound_email(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-out",
                direction=TimelineDirection.OUTBOUND,
                subject="Цена",
                summary="Пишем клиенту, что курс дорогой.",
                match_status="strong_unique",
                record={"full_clean_text": "Курс дорогой, скидки нет."},
                created_at=NOW,
            )
        )
    finally:
        store.close()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 0
    assert result["email_events_skipped_non_client"] == 1


def test_backfill_customer_objections_records_coverage_gate(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    canonical_db = tmp_path / "canonical_calls_master.db"
    _seed_canonical_calls(canonical_db, [(101, "Клиент: дорого.", "outbound")])
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        for source_id in ("101", "202"):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id="customer:1",
                    event_type="mango_call",
                    event_at=NOW,
                    source_system="mango_processed_summary",
                    source_id=source_id,
                    direction="outbound",
                    summary="дорого",
                    match_status="strong_unique",
                    record={},
                    created_at=NOW,
                )
            )
    finally:
        store.close()

    result = backfill_customer_objections_v1(
        db,
        allowed_root=tmp_path,
        canonical_calls_db_path=canonical_db,
        apply=True,
        as_of=NOW,
    )

    assert result["call_match_coverage"] == 0.5
    assert result["coverage_gate_passed"] is False
    with sqlite3.connect(db) as con:
        row = con.execute(
            "SELECT crm_objections_enabled FROM customer_objection_extraction_runs_v1"
        ).fetchone()
    assert row[0] == 0


def test_backfill_customer_objections_migrates_old_table_and_removes_legacy_rows(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-1",
                direction=TimelineDirection.INBOUND,
                subject="Цена",
                summary="Клиент пишет, что дорого.",
                match_status="strong_unique",
                record={"full_clean_text": "Дорого, нужен бюджет 80 тыс."},
                created_at=NOW,
            )
        )
    finally:
        store.close()
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE customer_objections_v1 (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              source_event_id TEXT NOT NULL,
              source_channel TEXT NOT NULL,
              objection_type TEXT NOT NULL,
              quote_preview TEXT NOT NULL,
              budget_hint_rub INTEGER,
              price_sensitivity TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              PRIMARY KEY (tenant_id, customer_id, source_event_id, objection_type)
            );
            """
        )
        con.execute(
            "INSERT INTO customer_objections_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("foton", "customer:1", "legacy", "call", "price", "старое слитое summary", None, "high", NOW.isoformat(), "ob_v1"),
        )
        con.commit()

    result = backfill_customer_objections_v1(db, allowed_root=tmp_path, apply=True, as_of=NOW)

    assert result["objections"] == 1
    with sqlite3.connect(db) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(customer_objections_v1)").fetchall()}
        rows = con.execute(
            "SELECT source_event_id, speaker, direction, confidence FROM customer_objections_v1"
        ).fetchall()
        summary = con.execute("SELECT top_objections_json FROM customer_objection_summary_v1").fetchone()
    assert {"speaker", "direction", "confidence"} <= columns
    assert len(rows) == 1
    assert rows[0][0] != "legacy"
    assert rows[0][1:] == ("client", "inbound", "high")
    assert json.loads(summary[0]) == [["price", 1]]


def test_backfill_customer_objections_rejects_missing_db(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"

    try:
        backfill_customer_objections_v1(missing, allowed_root=tmp_path, apply=False, as_of=NOW)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing DB must not be created")

    assert not missing.exists()


def _seed_canonical_calls(db_path: Path, rows: list[tuple[int, str, str]]) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE canonical_calls (
              canonical_call_id INTEGER PRIMARY KEY,
              transcript_client TEXT,
              direction TEXT
            )
            """
        )
        con.executemany(
            "INSERT INTO canonical_calls (canonical_call_id, transcript_client, direction) VALUES (?, ?, ?)",
            rows,
        )
