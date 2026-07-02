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


def test_backfill_customer_objections_rejects_missing_db(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"

    try:
        backfill_customer_objections_v1(missing, allowed_root=tmp_path, apply=False, as_of=NOW)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing DB must not be created")

    assert not missing.exists()
