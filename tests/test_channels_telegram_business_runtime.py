from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mango_mvp.channels import (
    TELEGRAM_BUSINESS_CHANNEL,
    ChannelMemoryStore,
    TelegramBusinessRuntime,
    TelegramBusinessRuntimeMemoryStore,
    build_and_store_channel_draft_preview,
    project_business_message_for_report,
    scrub_telegram_business_payload,
    telegram_business_runtime_safety_contract,
)


START = datetime(2026, 5, 13, 9, 0, tzinfo=timezone.utc)
START_TS = int(START.timestamp())
LARGE_CHAT_ID = 9223372036854775807


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def business_message_update(*, update_id: int = 1001, message_id: int = 77, text: str = "Хочу записаться") -> dict:
    return {
        "update_id": update_id,
        "business_message": {
            "business_connection_id": "bc-123",
            "message_id": message_id,
            "date": START_TS,
            "chat": {"id": LARGE_CHAT_ID, "type": "private"},
            "from": {"id": LARGE_CHAT_ID, "first_name": "Client"},
            "text": text,
        },
    }


def test_business_connection_record_preserves_64_bit_ids_and_status() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    result = runtime.process_update(
        {
            "update_id": 9001,
            "business_connection": {
                "id": "bc-large",
                "user": {"id": LARGE_CHAT_ID, "is_bot": False},
                "user_chat_id": LARGE_CHAT_ID,
                "date": START_TS,
                "can_reply": True,
                "is_enabled": True,
                "rights": {"can_reply": True, "bot_token": "must-not-leak"},
            },
        }
    )

    record = result.update_record
    assert record.update_type == "business_connection"
    assert record.business_connection_id == "bc-large"
    assert record.chat_id == str(LARGE_CHAT_ID)
    assert record.connection is not None
    assert record.connection.user_id == str(LARGE_CHAT_ID)
    assert record.connection.can_reply is True
    assert record.connection.is_enabled is True
    assert record.connection.rights["bot_token"] == "[REDACTED]"
    assert result.messages == ()


def test_disabled_business_connection_is_record_only_and_safe() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    result = runtime.process_update(
        {
            "update_id": 9002,
            "business_connection": {
                "id": "bc-disabled",
                "user": {"id": LARGE_CHAT_ID},
                "user_chat_id": LARGE_CHAT_ID,
                "date": START_TS,
                "can_reply": False,
                "is_enabled": False,
            },
        }
    )

    assert result.messages == ()
    assert result.update_record.connection is not None
    assert result.update_record.connection.is_enabled is False
    assert result.update_record.connection.can_reply is False
    assert result.to_json_dict()["safety"]["live_send"] is False


def test_business_message_bridges_to_channel_message_and_store_idempotently() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    update = business_message_update()

    first = runtime.process_update(update)
    repeat = runtime.process_update(update)
    message = first.messages[0]

    assert first.update_record.idempotency_key == repeat.update_record.idempotency_key
    assert message.channel == TELEGRAM_BUSINESS_CHANNEL
    assert message.channel_message_id == "bc-123:77"
    assert message.channel_thread_id == f"bc-123:{LARGE_CHAT_ID}"
    assert message.channel_user_id == str(LARGE_CHAT_ID)
    assert message.metadata["business_connection_id"] == "bc-123"
    assert message.idempotency_key == repeat.messages[0].idempotency_key

    store = ChannelMemoryStore(clock=StepClock())
    preview, created = build_and_store_channel_draft_preview(store, message, actor="business_runtime_test")
    _, duplicate = build_and_store_channel_draft_preview(store, message, actor="business_runtime_test")

    assert created.created is True
    assert duplicate.created is False
    assert preview.session.channel_thread_id == message.channel_thread_id
    assert store.summary()["messages"] == 1
    assert store.summary()["drafts"] == 1


def test_same_message_id_in_different_business_connection_or_chat_does_not_collide() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    first = runtime.process_update(business_message_update(update_id=1001, message_id=77)).messages[0]
    second_update = business_message_update(update_id=1002, message_id=77)
    second_update["business_message"]["business_connection_id"] = "bc-other"
    second = runtime.process_update(second_update).messages[0]
    third_update = business_message_update(update_id=1003, message_id=77)
    third_update["business_message"]["chat"]["id"] = LARGE_CHAT_ID - 1
    third_update["business_message"]["from"]["id"] = LARGE_CHAT_ID - 1
    third = runtime.process_update(third_update).messages[0]

    assert first.channel_message_id == "bc-123:77"
    assert second.channel_message_id == "bc-other:77"
    assert third.channel_message_id == "bc-123:77"
    assert len({first.idempotency_key, second.idempotency_key, third.idempotency_key}) == 3


def test_edited_business_message_has_distinct_message_key_from_original() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    original = runtime.process_update(business_message_update(update_id=1001, message_id=77))
    edited = runtime.process_update(
        {
            "update_id": 1002,
            "edited_business_message": {
                "business_connection_id": "bc-123",
                "message_id": 77,
                "date": START_TS,
                "chat": {"id": LARGE_CHAT_ID, "type": "private"},
                "from": {"id": LARGE_CHAT_ID},
                "text": "Хочу записаться сегодня",
            },
        }
    )

    assert original.messages[0].channel_message_id == "bc-123:77"
    assert edited.messages[0].channel_message_id == "bc-123:edited_business_message:77"
    assert original.messages[0].idempotency_key != edited.messages[0].idempotency_key
    assert edited.update_record.update_type == "edited_business_message"


def test_deleted_business_messages_create_tombstone_update_without_channel_message() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    result = runtime.process_update(
        {
            "update_id": 1003,
            "deleted_business_messages": {
                "business_connection_id": "bc-123",
                "chat": {"id": LARGE_CHAT_ID, "type": "private"},
                "message_ids": [77, 78],
            },
        }
    )

    assert result.messages == ()
    assert result.update_record.update_type == "deleted_business_messages"
    assert result.update_record.business_connection_id == "bc-123"
    assert result.update_record.chat_id == str(LARGE_CHAT_ID)
    assert result.update_record.message_ids == ("77", "78")
    assert result.update_record.metadata["deleted_message_count"] == 2


def test_deleted_business_messages_rejects_non_sequence_message_ids() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    with pytest.raises(ValueError, match="message_ids must be a sequence"):
        runtime.process_update(
            {
                "update_id": 1004,
                "deleted_business_messages": {
                    "business_connection_id": "bc-123",
                    "chat": {"id": LARGE_CHAT_ID, "type": "private"},
                    "message_ids": "77,78",
                },
            }
        )


def test_runtime_memory_store_dedupes_update_records() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    update_store = TelegramBusinessRuntimeMemoryStore()
    record = runtime.process_update(business_message_update()).update_record

    first = update_store.upsert_update(record)
    duplicate = update_store.upsert_update(record)

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.status == "duplicate"
    assert update_store.summary()["by_type"] == {"business_message": 1}


def test_business_runtime_reports_are_safe_and_do_not_include_text_by_default() -> None:
    result = TelegramBusinessRuntime(clock=StepClock()).process_update(business_message_update())
    report = result.to_json_dict()
    message_report = project_business_message_for_report(result.messages[0])

    assert report["messages"][0]["text"] is None
    assert report["messages"][0]["text_redacted"] is True
    assert message_report["text_length"] == len("Хочу записаться")
    assert "Хочу записаться" not in str(report)
    assert scrub_telegram_business_payload({"api_hash": "secret", "nested": {"token": "x"}}) == {
        "api_hash": "[REDACTED]",
        "nested": {"token": "[REDACTED]"},
    }


def test_business_runtime_rejects_unsupported_or_ambiguous_update() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    with pytest.raises(ValueError, match="exactly one"):
        runtime.process_update({"update_id": 1})
    with pytest.raises(ValueError, match="exactly one"):
        runtime.process_update({"update_id": 1, "business_connection": {}, "business_message": {}})


def test_business_runtime_safety_contract_blocks_live_effects() -> None:
    safety = telegram_business_runtime_safety_contract()

    assert safety["network_calls"] is False
    assert safety["telegram_api_called"] is False
    assert safety["live_send"] is False
    assert safety["tdlib_used"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["run_asr"] is False
    assert safety["run_ra"] is False
