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
