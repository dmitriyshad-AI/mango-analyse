#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from mango_mvp.channels.pilot_context import build_pilot_context
from mango_mvp.channels.preview_service import LlmChannelPreviewService
from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider
from mango_mvp.channels.telegram_bot_polling import (
    TelegramBotDraftResult,
    TelegramBotPollingConfig,
    TelegramBotPollingRuntime,
    telegram_bot_polling_safety_contract,
)
from mango_mvp.channels.telegram_manager_inbox import (
    TELEGRAM_MANAGER_CHAT_IDS_ENV,
    TelegramManagerInboxConfig,
    TelegramManagerInboxService,
    telegram_manager_inbox_safety_contract,
)


LONG_POLLING_CONFIRMATION = "YES_MANAGER_DRAFT_ONLY"
TELEGRAM_PILOT_LLM_ENABLED_ENV = "TELEGRAM_PILOT_LLM_ENABLED"
TELEGRAM_PILOT_CODEX_MODEL_ENV = "TELEGRAM_PILOT_CODEX_MODEL"
TELEGRAM_PILOT_CODEX_REASONING_ENV = "TELEGRAM_PILOT_CODEX_REASONING_EFFORT"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Безопасный запуск Telegram-пилота: черновики только менеджеру, без автоответов клиенту."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Обработать fake update без Telegram-сети.")
    mode.add_argument("--long-polling", action="store_true", help="Запустить реальный long polling Bot API.")
    parser.add_argument("--manager-chat-id", default="", help="Служебный chat_id менеджера для dry-run.")
    parser.add_argument(
        "--confirm-long-polling",
        default="",
        help=f"Для live long polling передать ровно {LONG_POLLING_CONFIRMATION!r}.",
    )
    args = parser.parse_args(argv)

    if args.long_polling:
        return run_long_polling(args.confirm_long_polling)
    return run_dry_run(manager_chat_id=args.manager_chat_id)


def run_dry_run(*, manager_chat_id: str = "") -> int:
    chat_id = str(manager_chat_id or os.environ.get(TELEGRAM_MANAGER_CHAT_IDS_ENV) or "700100").strip()
    config = TelegramBotPollingConfig(enabled=True, bot_token="dry-run-token", debounce_seconds=5)
    runtime = TelegramBotPollingRuntime(config=config)
    inbox = TelegramManagerInboxService(
        config=TelegramManagerInboxConfig(allowed_manager_chat_ids=(chat_id,)),
    )

    update = fake_client_update()
    inbound_result = runtime.process_update(update, actor="telegram_manager_draft_pilot_dry_run")
    draft_results = runtime.flush_all(actor="telegram_manager_draft_pilot_dry_run")
    deliveries = [
        inbox.build_delivery(
            draft.preview,
            manager_chat_id=chat_id,
            context={
                "found_topic": "Стоимость обучения",
                "rop_decision": "Пилотный режим: черновик только менеджеру.",
                "bot_must_ask": ["Класс ученика", "Предмет", "Формат обучения"],
                "risk_flags": ["dry_run", "manager_approval_required"],
            },
        ).to_json_dict()
        for draft in draft_results
    ]

    payload: Mapping[str, Any] = {
        "mode": "dry_run",
        "inbound_result": inbound_result.to_json_dict(include_message_text=True),
        "manager_deliveries": deliveries,
        "safety": {
            "bot_polling": telegram_bot_polling_safety_contract(),
            "manager_inbox": telegram_manager_inbox_safety_contract(),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_long_polling(confirmation: str) -> int:
    if confirmation != LONG_POLLING_CONFIRMATION:
        raise SystemExit(
            "Long polling не запущен: нужен явный --confirm-long-polling "
            f"{LONG_POLLING_CONFIRMATION!r}."
        )
    if os.environ.get("TELEGRAM_PILOT_ENABLED") not in {"1", "true", "TRUE", "yes", "YES"}:
        raise SystemExit("Long polling не запущен: TELEGRAM_PILOT_ENABLED должен быть включен.")
    config = TelegramBotPollingConfig.from_env()
    manager_config = TelegramManagerInboxConfig.from_env()
    if not manager_config.default_manager_chat_id:
        raise SystemExit(f"Long polling не запущен: нужно задать {TELEGRAM_MANAGER_CHAT_IDS_ENV}.")

    inbox = TelegramManagerInboxService(config=manager_config)
    bot = build_manager_bot(config.bot_token)

    async def deliver_manager_drafts(draft_results: Sequence[TelegramBotDraftResult]) -> None:
        for draft in draft_results:
            delivery = inbox.build_delivery(
                draft.preview,
                manager_chat_id=manager_config.default_manager_chat_id,
                context={
                    "rop_decision": "Пилотный режим: черновик только менеджеру.",
                    "risk_flags": ["manager_approval_required", "no_auto_send"],
                },
            )
            if delivery.blocked or delivery.message is None:
                print(json.dumps(delivery.to_json_dict(), ensure_ascii=False, sort_keys=True))
                continue
            await bot.send_message(
                chat_id=delivery.manager_chat_id,
                text=delivery.message.text,
                disable_web_page_preview=True,
                reply_markup=build_live_reply_markup(delivery.message.buttons),
            )

    runtime = TelegramBotPollingRuntime(config=config, preview_service=build_preview_service_from_env())
    runtime.start_long_polling(on_drafts_ready=deliver_manager_drafts)
    return 0


def build_preview_service_from_env() -> Any:
    if os.environ.get(TELEGRAM_PILOT_LLM_ENABLED_ENV) not in {"1", "true", "TRUE", "yes", "YES"}:
        return None
    provider = SubscriptionLlmDraftProvider(
        model=os.environ.get(TELEGRAM_PILOT_CODEX_MODEL_ENV, "gpt-5.5"),
        reasoning_effort=os.environ.get(TELEGRAM_PILOT_CODEX_REASONING_ENV, "xhigh"),
        timeout_sec=120,
        cache_dir=".codex_local/telegram_pilot/llm_cache",
    )
    return LlmChannelPreviewService(
        draft_provider=provider,
        context_builder=lambda message: build_pilot_context(
            message,
            recent_messages=(message.text,),
            client_identity={
                "channel": message.channel,
                "channel_thread_id": message.channel_thread_id,
                "channel_user_id": message.channel_user_id,
            },
            rop_policy={"bot_permission": "draft_for_manager"},
            risk_flags=("manager_approval_required", "no_auto_send"),
        ).to_prompt_context(),
    )


def build_manager_bot(token: str | None) -> Any:
    try:
        from telegram import Bot
    except ImportError as exc:
        raise RuntimeError("python-telegram-bot is required for live long polling") from exc
    return Bot(token=str(token or "").strip())


def build_live_reply_markup(buttons: Sequence[Any]) -> Any:
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    except ImportError as exc:
        raise RuntimeError("python-telegram-bot is required for live manager buttons") from exc
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text=button.label, callback_data=button.callback_data)] for button in buttons]
    )


def fake_client_update() -> Mapping[str, Any]:
    now_ts = int(datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc).timestamp())
    return {
        "update_id": 9001,
        "message": {
            "message_id": 101,
            "date": now_ts,
            "chat": {"id": 300300, "type": "private"},
            "from": {"id": 300301, "username": "client"},
            "text": "Здравствуйте, какая цена подготовки к ЕГЭ и можно ли оплатить частями?",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
