from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mango_mvp.channels.telegram_bot_polling import (
    POLLING_STATUS_ACCEPTED,
    POLLING_STATUS_BLOCKED,
    POLLING_STATUS_DUPLICATE_UPDATE,
    TelegramBotPollingConfig,
    TelegramBotPollingRuntime,
    telegram_bot_polling_safety_contract,
)
from mango_mvp.channels.pilot_context import build_pilot_context
from mango_mvp.channels.preview_service import LlmChannelPreviewService
from mango_mvp.channels.subscription_llm import FakeDraftProvider


START = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value = self.value + timedelta(seconds=seconds)


def tg_update(update_id: int, message_id: int, text: str, *, chat_id: int = 123) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": int((START + timedelta(seconds=message_id)).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": chat_id, "username": "client"},
            "text": text,
        },
    }


def enabled_config() -> TelegramBotPollingConfig:
    return TelegramBotPollingConfig(enabled=True, kill_switch=False, bot_token="token", debounce_seconds=7)


def runtime_with_config(config: TelegramBotPollingConfig, *, clock: MutableClock | None = None) -> TelegramBotPollingRuntime:
    return TelegramBotPollingRuntime(config=config, clock=clock or MutableClock())


def test_polling_blocks_when_disabled() -> None:
    result = runtime_with_config(TelegramBotPollingConfig(enabled=False, bot_token="token")).process_update(
        tg_update(1, 1, "Здравствуйте")
    )

    assert result.status == POLLING_STATUS_BLOCKED
    assert result.blocked_reason == "telegram_pilot_disabled"


def test_polling_blocks_when_kill_switch_enabled() -> None:
    result = runtime_with_config(TelegramBotPollingConfig(enabled=True, kill_switch=True, bot_token="token")).process_update(
        tg_update(1, 1, "Здравствуйте")
    )

    assert result.status == POLLING_STATUS_BLOCKED
    assert result.blocked_reason == "telegram_pilot_kill_switch"


def test_bot_update_to_channel_message() -> None:
    result = runtime_with_config(enabled_config()).process_update(tg_update(1, 1, "Какая цена?"))

    assert result.status == POLLING_STATUS_ACCEPTED
    assert len(result.messages) == 1
    assert result.messages[0].text == "Какая цена?"
    assert result.messages[0].channel == "telegram_bot"
    assert result.client_send_attempted is False
    assert result.telegram_api_called is False


def test_duplicate_update_skipped() -> None:
    runtime = runtime_with_config(enabled_config())
    update = tg_update(1, 1, "Какая цена?")

    first = runtime.process_update(update)
    second = runtime.process_update(update)

    assert first.status == POLLING_STATUS_ACCEPTED
    assert second.status == POLLING_STATUS_DUPLICATE_UPDATE
    assert second.skipped_reason == "duplicate_update"


def test_client_send_disabled_by_default() -> None:
    result = runtime_with_config(enabled_config()).send_client_message(chat_id="123", text="Нельзя отправлять")

    assert result.sent is False
    assert result.status == "client_send_disabled"
    assert result.metadata["client_send_enabled"] is False
    assert result.metadata["telegram_api_called"] is False
    assert result.metadata["live_send"] is False


def test_debounce_groups_consecutive_messages_from_same_client() -> None:
    clock = MutableClock()
    runtime = runtime_with_config(enabled_config(), clock=clock)

    runtime.process_update(tg_update(1, 1, "Здравствуйте"))
    clock.advance(3)
    runtime.process_update(tg_update(2, 2, "У меня вопрос"))
    clock.advance(3)
    runtime.process_update(tg_update(3, 3, "Какая цена?"))

    clock.advance(2)
    assert runtime.flush_due() == ()

    clock.advance(6)
    drafts = runtime.flush_due()

    assert len(drafts) == 1
    combined = drafts[0].preview.source_message
    assert combined.text == "Здравствуйте\nУ меня вопрос\nКакая цена?"
    assert combined.metadata["telegram_debounce_message_count"] == 3
    assert len(drafts[0].source_message_idempotency_keys) == 3


def test_polling_safety_contract_blocks_live_side_effects() -> None:
    safety = telegram_bot_polling_safety_contract()

    assert safety["telegram_api_called_by_process_update"] is False
    assert safety["client_send"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["run_asr"] is False


def test_debounce_flush_can_use_contextual_llm_preview_service() -> None:
    clock = MutableClock()
    provider = FakeDraftProvider(
        {
            "message_type": "question",
            "broad_group": "commercial",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.88,
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Уточните, пожалуйста, класс и формат обучения.",
            "safety_flags": ["manager_approval_required", "no_auto_send"],
            "context_used": ["recent_messages", "rop_policy"],
        }
    )
    preview_service = LlmChannelPreviewService(
        draft_provider=provider,
        context_builder=lambda message: build_pilot_context(
            message,
            recent_messages=("Здравствуйте", message.text),
            client_identity={"channel_user_id": message.channel_user_id},
            rop_policy={"bot_permission": "draft_for_manager"},
        ).to_prompt_context(),
    )
    runtime = TelegramBotPollingRuntime(config=enabled_config(), clock=clock, preview_service=preview_service)

    runtime.process_update(tg_update(1, 1, "Какая цена?"))
    clock.advance(8)
    drafts = runtime.flush_due()

    assert len(drafts) == 1
    preview = drafts[0].preview
    assert preview.reply.text.startswith("Здравствуйте")
    assert preview.reply.metadata["preview_mode"] == "subscription_llm_draft"
    assert preview.reply.metadata["subscription_llm_result"]["message_type"] == "question"
    assert preview.metadata["context_quality"]["customer_identity_found"] is True
    assert "llm_used" in preview.reply.safety_flags
    assert provider.prompts and "context_quality" in provider.prompts[0]
