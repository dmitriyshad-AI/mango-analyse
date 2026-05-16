from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from mango_mvp.channels import (
    ChannelFeedbackMemoryStore,
    ChannelMemoryStore,
    TelegramReadOnlyAdapter,
    build_channel_draft_preview,
)
from mango_mvp.channels.telegram_manager_inbox import (
    MANAGER_ACTION_ACCEPT,
    MANAGER_ACTION_MANAGER_ONLY,
    MANAGER_ACTION_NEEDS_EDIT,
    MANAGER_DELIVERY_STATUS_BLOCKED,
    MANAGER_DELIVERY_STATUS_READY,
    MANAGER_DRAFT_STATUS_MANAGER_ONLY,
    MANAGER_DRAFT_STATUS_MARKED_NEEDS_EDIT,
    MANAGER_DRAFT_STATUS_MARKED_USEFUL,
    MANAGER_START_STATUS_BLOCKED,
    MANAGER_START_STATUS_REGISTERED,
    TELEGRAM_MANAGER_CHAT_IDS_ENV,
    TelegramManagerFeedbackCommand,
    TelegramManagerInboxConfig,
    TelegramManagerInboxMemoryStore,
    TelegramManagerInboxService,
    build_manager_draft_message,
    manager_action_from_callback_data,
    telegram_manager_inbox_safety_contract,
)


START = datetime(2026, 5, 16, 9, 0, tzinfo=timezone.utc)
NOW_TS = int(START.timestamp())
MANAGER_CHAT_ID = "700100"
UNAUTHORIZED_CHAT_ID = "999999"


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


def manager_start_update(chat_id: str = MANAGER_CHAT_ID) -> dict:
    return {
        "update_id": 5001,
        "message": {
            "message_id": 1,
            "date": NOW_TS,
            "chat": {"id": int(chat_id), "type": "private"},
            "from": {"id": int(chat_id), "username": "nastya"},
            "text": "/start",
        },
    }


def client_update() -> dict:
    return {
        "update_id": 6001,
        "message": {
            "message_id": 42,
            "date": NOW_TS,
            "chat": {"id": 300300, "type": "private"},
            "from": {"id": 300301, "username": "client"},
            "text": (
                "Здравствуйте, сколько стоит подготовка к ЕГЭ "
                "и можно ли оплатить частями?"
            ),
        },
    }


def manager_context() -> dict:
    return {
        "found_topic": "Стоимость обучения и порядок оплаты",
        "message_type": "question",
        "context_quality": {"customer_identity_found": True, "family_phone": False},
        "rop_decision": (
            "Можно отвечать только по утвержденным ценам, без обещаний скидки."
        ),
        "bot_must_ask": ["Класс ученика", "Предмет", "Формат обучения"],
        "risk_flags": ["commercial_risk", "requires_manager_review"],
        "crm_recommendations": [
            "Проверить активную сделку в AMO",
            "Проверить историю оплат и договоренности в CRM",
        ],
    }


def stored_telegram_draft():
    message = TelegramReadOnlyAdapter().parse_inbound(client_update())[0]
    preview = build_channel_draft_preview(
        message,
        context=manager_context()
        | {"safe_draft_text": "Здравствуйте! Уточните, пожалуйста, класс и предмет."},
    )
    channel_store = ChannelMemoryStore(clock=StepClock())
    channel_store.upsert_preview(preview, actor="telegram_manager_test")
    draft = channel_store.get_draft(preview.draft_id)
    assert draft is not None
    return preview, draft


def service_with_allowed_manager():
    clock = StepClock()
    state_store = TelegramManagerInboxMemoryStore(clock=clock)
    feedback_store = ChannelFeedbackMemoryStore(clock=clock)
    service = TelegramManagerInboxService(
        config=TelegramManagerInboxConfig.from_env({TELEGRAM_MANAGER_CHAT_IDS_ENV: MANAGER_CHAT_ID}),
        state_store=state_store,
        feedback_store=feedback_store,
        clock=clock,
    )
    return service, state_store, feedback_store


def test_manager_start_registers_allowed_chat() -> None:
    service, state_store, _ = service_with_allowed_manager()

    result = service.handle_start_update(manager_start_update())

    assert result.status == MANAGER_START_STATUS_REGISTERED
    assert result.authorized is True
    assert result.registered is True
    assert state_store.is_registered(MANAGER_CHAT_ID)
    assert result.rendered_payload is not None
    assert result.rendered_payload["chat_id"] == MANAGER_CHAT_ID
    assert result.client_send_attempted is False
    assert result.telegram_api_called is False


def test_manager_draft_message_contains_required_sections() -> None:
    service, _, _ = service_with_allowed_manager()
    preview, _ = stored_telegram_draft()

    delivery = service.build_delivery(preview, context=manager_context())
    message = delivery.message

    assert delivery.status == MANAGER_DELIVERY_STATUS_READY
    assert delivery.manager_chat_id == MANAGER_CHAT_ID
    assert message is not None
    for section in (
        "Откуда пришел клиент",
        "Текст клиента",
        "Найденная тема",
        "Тип сообщения",
        "Качество контекста",
        "Решение РОПа",
        "Что бот обязан спросить",
        "Черновик ответа",
        "Флаги риска",
        "Что проверить в AMO/CRM",
        "Напоминание менеджеру",
        "Статус",
    ):
        assert section in message.text
    assert "Клиенту не отправлено" in message.text
    assert "Стоимость обучения" in message.text
    assert "Класс ученика" in message.text
    assert delivery.rendered_payload is not None
    assert delivery.rendered_payload["telegram_api_called"] is False
    assert delivery.rendered_payload["client_send_enabled"] is False


def test_manager_draft_message_uses_preview_metadata_context_quality() -> None:
    service, _, _ = service_with_allowed_manager()
    preview, _ = stored_telegram_draft()
    preview = replace(
        preview,
        metadata={
            **dict(preview.metadata),
            "message_type": "question",
            "context_quality": {"customer_identity_found": True, "family_phone": True},
        },
    )

    delivery = service.build_delivery(
        preview,
        context={
            "rop_decision": "Пилотный режим: черновик только менеджеру.",
        },
    )

    assert delivery.message is not None
    assert "Тип сообщения" in delivery.message.text
    assert "question" in delivery.message.text
    assert "Качество контекста" in delivery.message.text
    assert "family_phone" in delivery.message.text


def test_unauthorized_manager_chat_blocked() -> None:
    service, _, _ = service_with_allowed_manager()
    preview, _ = stored_telegram_draft()

    start_result = service.handle_start_update(manager_start_update(UNAUTHORIZED_CHAT_ID))
    delivery = service.build_delivery(preview, manager_chat_id=UNAUTHORIZED_CHAT_ID, context=manager_context())

    assert start_result.status == MANAGER_START_STATUS_BLOCKED
    assert start_result.authorized is False
    assert start_result.registered is False
    assert start_result.blocked_reason == "manager_chat_not_allowed"
    assert delivery.status == MANAGER_DELIVERY_STATUS_BLOCKED
    assert delivery.message is None
    assert delivery.rendered_payload is None
    assert "сколько стоит" not in str(delivery.to_json_dict())
    assert delivery.client_send_attempted is False
    assert delivery.telegram_api_called is False


def test_feedback_buttons_update_local_state() -> None:
    service, state_store, feedback_store = service_with_allowed_manager()
    _, draft = stored_telegram_draft()

    accepted = service.handle_feedback(
        draft,
        TelegramManagerFeedbackCommand(
            manager_chat_id=MANAGER_CHAT_ID,
            draft_id=draft.draft_id,
            action=MANAGER_ACTION_ACCEPT,
            occurred_at=START,
        ),
    )
    needs_edit = service.handle_feedback(
        draft,
        TelegramManagerFeedbackCommand(
            manager_chat_id=MANAGER_CHAT_ID,
            draft_id=draft.draft_id,
            action=MANAGER_ACTION_NEEDS_EDIT,
            occurred_at=START + timedelta(seconds=1),
        ),
    )
    manager_only = service.handle_feedback(
        draft,
        TelegramManagerFeedbackCommand(
            manager_chat_id=MANAGER_CHAT_ID,
            draft_id=draft.draft_id,
            action=MANAGER_ACTION_MANAGER_ONLY,
            occurred_at=START + timedelta(seconds=2),
        ),
    )

    assert accepted.manager_status == MANAGER_DRAFT_STATUS_MARKED_USEFUL
    assert needs_edit.manager_status == MANAGER_DRAFT_STATUS_MARKED_NEEDS_EDIT
    assert manager_only.manager_status == MANAGER_DRAFT_STATUS_MANAGER_ONLY
    assert manager_only.client_send_attempted is False
    state = state_store.get_feedback_state(draft.draft_id)
    assert state is not None
    assert state.status == MANAGER_DRAFT_STATUS_MANAGER_ONLY
    assert state.client_send_attempted is False
    summary = feedback_store.summary(session_key=draft.session_key)
    assert summary["manager_review"]["approved"] == 1
    assert summary["manager_review"]["rejected"] == 2
    assert summary["total_events"] == 3


def test_no_client_send_button_in_phase1() -> None:
    preview, _ = stored_telegram_draft()
    manager_message = build_manager_draft_message(preview, context=manager_context())
    rendered = manager_message.to_json_dict()

    assert len(manager_message.buttons) == 3
    for button in manager_message.buttons:
        joined = f"{button.label} {button.action} {button.callback_data}".casefold()
        assert "отправить клиенту" not in joined
        assert "send_client" not in joined
        assert "client_send" not in joined
        assert manager_action_from_callback_data(button.callback_data) == button.action

    assert rendered["metadata"]["client_send_enabled"] is False
    assert rendered["metadata"]["client_send_button_included"] is False
    safety = telegram_manager_inbox_safety_contract()
    assert safety["client_send"] is False
    assert safety["telegram_api_called"] is False
    assert safety["write_crm"] is False


def test_manager_message_shows_followup_deadline() -> None:
    service, _, _ = service_with_allowed_manager()
    preview, _ = stored_telegram_draft()

    delivery = service.build_delivery(
        preview,
        context={
            **manager_context(),
            "safe_schedule_template": {
                "manager_followup_required": True,
                "manager_followup_deadline": "2026-05-17T18:00:00+03:00",
            },
        },
    )

    assert delivery.message is not None
    assert "Напоминание менеджеру" in delivery.message.text
    assert "Требуется follow-up до 2026-05-17T18:00:00+03:00" in delivery.message.text
    assert delivery.rendered_payload is not None
    assert delivery.rendered_payload["client_send_enabled"] is False
