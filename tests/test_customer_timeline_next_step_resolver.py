from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.crm_card_aggregator import build_crm_card_projection
from mango_mvp.customer_timeline import (
    CustomerIdentity,
    CustomerTimelineReadApi,
    CustomerTimelineReadApiConfig,
    CustomerTimelineSQLiteStore,
    TimelineEvent,
    resolve_customer_next_step,
)
from mango_mvp.insights.sanitizers import has_personal_data_risk


NOW = datetime(2026, 6, 21, 9, 0, tzinfo=timezone.utc)
TENANT = "foton"
CUSTOMER = "customer:foton-next-step"


def test_documents_step_closed_by_later_email_and_deterministic() -> None:
    events = [
        event(
            "call-1",
            "mango_call",
            NOW,
            summary="Клиент попросил материалы.",
            record={"call_analysis": {"next_step": "Отправить документы на почту"}},
        ),
        event(
            "email-1",
            "email_message",
            NOW + timedelta(hours=1),
            source_system="mail_archive",
            summary="Документы и презентация отправлены клиенту на почту.",
        ),
    ]

    first = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)
    second = resolve_customer_next_step(list(reversed(events)), readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert first.to_json_dict() == second.to_json_dict()
    assert first.status == "closed"
    assert first.reason_code == "documents_closed_by_later_event"
    assert first.previous_step == "Отправить документы на почту"
    assert first.closing_channel == "почта"
    assert first.display_text == "Шаг закрыт: документы/материалы отправлены (от 21.06.2026, почта)"


def test_open_ambiguous_identity_blocks_shared_phone_closure_on_merged_customer() -> None:
    events = [
        event(
            "call-parent-a",
            "mango_call",
            NOW,
            summary="Клиент А: нужно отправить документы.",
            record={"call_analysis": {"next_step": "Отправить документы"}},
        ),
        event(
            "email-parent-b",
            "email_message",
            NOW + timedelta(hours=1),
            source_system="mail_archive",
            summary="Клиент Б с общего телефона получил документы.",
        ),
    ]
    conflicts = [
        {
            "conflict_type": "ambiguous_identity",
            "status": "open",
            "entity_refs": ["phone:+79990000000", CUSTOMER, "customer:other"],
        }
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 1}, conflicts=conflicts, customer_id=CUSTOMER)

    assert result.status == "needs_manager_review"
    assert result.reason_code == "ambiguous_identity_open"
    assert result.display_text == "Уточнить у менеджера: открыт конфликт идентичности"
    assert result.closing_event_id == ""


def test_campaign_service_and_bounce_do_not_close_step() -> None:
    events = [
        event(
            "call-1",
            "mango_call",
            NOW,
            summary="Клиент ждёт договор.",
            record={"call_analysis": {"next_step": "Отправить договор и документы"}},
        ),
        event(
            "campaign-1",
            "email_message",
            NOW + timedelta(minutes=30),
            source_system="outbound_campaign",
            summary="Массовая рассылка: документы доступны на сайте.",
            record={"outbound_campaign": True, "allowed_for_bot": False},
        ),
        event(
            "service-1",
            "system_note",
            NOW + timedelta(minutes=40),
            source_system="amocrm_snapshot",
            summary="Служебное уведомление: карточка обновлена.",
        ),
        event(
            "bounce-1",
            "email_message",
            NOW + timedelta(minutes=50),
            source_system="mail_archive",
            subject="Delivery Status Notification",
            summary="Bounce: письмо не доставлено.",
        ),
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "active"
    assert result.action == "Отправить договор и документы"
    assert set(result.ignored_event_ids) == {"campaign-1", "service-1", "bounce-1"}


def test_summary_next_step_extractor_uses_explicit_agreed_step_without_structured_field() -> None:
    events = [
        event(
            "call-summary-1",
            "mango_call",
            NOW,
            summary=(
                "Обсудили формат курса и оплату. "
                "Согласован следующий шаг: отправить клиентке подробные условия оплаты и список документов в WhatsApp."
            ),
            record={"brand": "foton", "contentful": "Да", "duration_sec": 360, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "active"
    assert result.reason_code == "latest_relevant_event_has_active_next_step"
    assert result.source_event_id == "call-summary-1"
    assert result.source_channel == "звонок"
    assert result.action == "Отправить клиентке подробные условия оплаты и список документов в WhatsApp"


def test_summary_without_explicit_next_step_does_not_invent_step() -> None:
    events = [
        event(
            "call-summary-2",
            "mango_call",
            NOW,
            summary="Обсудили оплату и договор, но следующий шаг не согласован.",
            record={"brand": "foton", "contentful": "Да", "duration_sec": 280, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "empty"
    assert result.action == ""
    assert result.reason_code == "no_explicit_next_step"


def test_summary_non_conversation_template_callback_does_not_create_step() -> None:
    events = [
        event(
            "call-summary-technical",
            "mango_call",
            NOW,
            summary=(
                "Номер оказался техническим номером для опросов населения и не используется для обратной связи. "
                "Контакт с потенциальным клиентом не состоялся. Договорились: Перезвонить клиенту."
            ),
            record={"brand": "foton", "contentful": "Да", "duration_sec": 90, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "empty"
    assert result.action == ""
    assert result.reason_code == "no_explicit_next_step"


def test_summary_extractor_does_not_treat_required_payment_condition_as_step() -> None:
    events = [
        event(
            "call-summary-condition",
            "mango_call",
            NOW,
            summary=(
                "Менеджер объяснил условия продления. "
                "Для продолжения затем потребуется оплата за следующий период, решение клиент пока не принял."
            ),
            record={"brand": "foton", "contentful": "Да", "duration_sec": 280, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "empty"
    assert result.action == ""
    assert result.reason_code == "no_explicit_next_step"


def test_summary_extractor_does_not_treat_customer_payment_question_as_step() -> None:
    events = [
        event(
            "call-summary-question",
            "mango_call",
            NOW,
            summary=(
                "Клиент уточнял, сколько еще нужно доплатить и до какого срока будут продолжаться занятия. "
                "Итоговое действие не согласовано."
            ),
            record={"brand": "foton", "contentful": "Да", "duration_sec": 280, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "empty"
    assert result.action == ""
    assert result.reason_code == "no_explicit_next_step"


def test_summary_extractor_drops_truncated_action_tail() -> None:
    events = [
        event(
            "call-summary-truncated",
            "mango_call",
            NOW,
            summary="Стороны договорились отправить на почту подробную и",
            record={"brand": "foton", "contentful": "Да", "duration_sec": 280, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "empty"
    assert result.action == ""
    assert result.reason_code == "no_explicit_next_step"


def test_summary_extracted_next_step_scrubs_manager_child_booking_and_email() -> None:
    events = [
        event(
            "call-summary-3",
            "mango_call",
            NOW,
            summary=(
                "Менеджер Клычева Дарья обсудила условия. "
                "Согласован следующий шаг: менеджер Клычева Дарья отправит договор ученику Смирнову Арсению "
                "по брони 64-64-58 на parent@example.com."
            ),
            record={"brand": "foton", "contentful": "Да", "duration_sec": 420, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "active"
    assert "Клычева" not in result.action
    assert "Дарья" not in result.action
    assert "Смирнов" not in result.action
    assert "Арсени" not in result.action
    assert "64-64-58" not in result.action
    assert "parent@example.com" not in result.action
    assert "<number_masked>" in result.action
    assert "<email_masked>" in result.action
    assert not has_personal_data_risk(result.action)


def test_summary_extracted_next_step_scrubs_single_named_target() -> None:
    events = [
        event(
            "call-summary-single-name",
            "mango_call",
            NOW,
            summary="Договорились передать Еве согласованный план для записи.",
            record={"brand": "foton", "contentful": "Да", "duration_sec": 300, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "active"
    assert result.action == "Передать клиенту согласованный план для записи"
    assert "Еве" not in result.action
    assert not has_personal_data_risk(result.action)


def test_summary_extracted_callback_with_new_year_phrase_falls_back_to_safe_action() -> None:
    events = [
        event(
            "call-summary-new-year",
            "mango_call",
            NOW,
            summary="Договорились после Нового года снова связаться и повторно обсудить обучение.",
            record={"brand": "foton", "contentful": "Да", "duration_sec": 300, "manual_review_required": "Нет"},
        )
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "active"
    assert result.action == "После праздников снова связаться и повторно обсудить обучение"
    assert not has_personal_data_risk(result.action)


def test_summary_extracted_documents_step_closed_by_later_email() -> None:
    events = [
        event(
            "call-summary-4",
            "mango_call",
            NOW,
            summary="Договорились, что менеджер отправит договор и документы на почту.",
            record={"brand": "foton", "contentful": "Да", "duration_sec": 300, "manual_review_required": "Нет"},
        ),
        event(
            "email-summary-4",
            "email_message",
            NOW + timedelta(hours=1),
            source_system="mail_archive",
            summary="Договор и документы отправлены клиенту на почту.",
        ),
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "closed"
    assert result.reason_code == "documents_closed_by_later_event"
    assert result.previous_step == "Менеджер отправит договор и документы на почту"
    assert result.closing_event_id == "email-summary-4"


def test_contradictory_later_event_requires_manager_review() -> None:
    events = [
        event(
            "call-1",
            "mango_call",
            NOW,
            record={"call_analysis": {"next_step": "Отправить документы"}},
        ),
        event(
            "email-1",
            "email_message",
            NOW + timedelta(hours=2),
            summary="Клиент пишет: документы не получили, проверьте почту?",
        ),
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "needs_manager_review"
    assert result.reason_code == "contradictory_later_event"
    assert result.previous_step == "Отправить документы"
    assert result.source_event_id == "email-1"


def test_later_imperative_request_does_not_close_documents_step() -> None:
    events = [
        event(
            "call-1",
            "mango_call",
            NOW,
            record={"call_analysis": {"next_step": "Отправить документы"}},
        ),
        event(
            "email-1",
            "email_message",
            NOW + timedelta(hours=2),
            summary="Клиент просит: отправьте документы ещё раз.",
        ),
    ]

    result = resolve_customer_next_step(events, readiness={"open_conflicts": 0}, customer_id=CUSTOMER)

    assert result.status == "active"
    assert result.action == "Отправить документы"
    assert result.closing_event_id == ""


def test_read_api_profile_and_crm_card_use_resolved_cross_channel_step(tmp_path: Path) -> None:
    db_path, customer_id = seed_next_step_db(tmp_path, with_conflict=False)
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        profile = api.customer_profile(TENANT, customer_id, event_limit=10, bot_context_limit=1)

    card = build_crm_card_projection(
        profile,
        manager_facts={"AMO contact IDs": "123", "selected_deal_id": "456"},
    )

    assert profile["next_step_resolution"]["status"] == "closed"
    assert profile["next_step_resolution"]["closing_channel"] == "почта"
    assert card["deal_card"]["fields"]["Следующий шаг"] == "Шаг закрыт: документы/материалы отправлены (от 21.06.2026, почта)"


def test_read_api_profile_blocks_closure_when_ambiguous_identity_is_open(tmp_path: Path) -> None:
    db_path, customer_id = seed_next_step_db(tmp_path, with_conflict=True)
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        profile = api.customer_profile(TENANT, customer_id, event_limit=10, bot_context_limit=1)

    card = build_crm_card_projection(
        profile,
        manager_facts={"AMO contact IDs": "123", "selected_deal_id": "456"},
    )

    assert profile["readiness"]["open_conflicts"] == 1
    assert profile["next_step_resolution"]["status"] == "needs_manager_review"
    assert profile["next_step_resolution"]["reason_code"] == "ambiguous_identity_open"
    assert card["deal_card"]["fields"]["Следующий шаг"] == "Уточнить у менеджера: открыт конфликт идентичности"


def seed_next_step_db(tmp_path: Path, *, with_conflict: bool) -> tuple[Path, str]:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path)
    customer = CustomerIdentity(
        tenant_id=TENANT,
        customer_id=CUSTOMER,
        identity_status="strong",
        display_name="Тестовый клиент",
        primary_phone="+79990000000",
        first_seen_at=NOW,
        last_seen_at=NOW + timedelta(hours=1),
        touch_count=2,
        created_at=NOW,
        updated_at=NOW + timedelta(hours=1),
    )
    store.upsert_customer(customer)
    store.upsert_event(
        TimelineEvent(
            tenant_id=TENANT,
            customer_id=customer.customer_id,
            event_type="mango_call",
            event_at=NOW,
            source_system="mango",
            source_id="call-1",
            direction="inbound",
            summary="Клиент попросил документы.",
            record={"call_analysis": {"next_step": "Отправить документы"}},
            created_at=NOW,
        )
    )
    store.upsert_event(
        TimelineEvent(
            tenant_id=TENANT,
            customer_id=customer.customer_id,
            event_type="email_message",
            event_at=NOW + timedelta(hours=1),
            source_system="mail_archive",
            source_id="email-1",
            direction="outbound",
            subject="Документы по курсу",
            summary="Документы отправлены клиенту.",
            created_at=NOW + timedelta(hours=1),
        )
    )
    if with_conflict:
        store.record_conflict(
            TENANT,
            conflict_type="ambiguous_identity",
            entity_refs=("phone:+79990000000", customer.customer_id, "customer:other"),
            actor="test",
        )
    store.close()
    return tmp_path / "customer_timeline.sqlite", customer.customer_id


def event(
    event_id: str,
    event_type: str,
    event_at: datetime,
    *,
    customer_id: str = CUSTOMER,
    source_system: str = "mango",
    subject: str = "",
    summary: str = "",
    record: dict | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "tenant_id": TENANT,
        "customer_id": customer_id,
        "event_type": event_type,
        "event_at": event_at.isoformat(),
        "source_system": source_system,
        "subject": subject,
        "summary": summary,
        "record": record or {},
    }
