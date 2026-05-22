from __future__ import annotations

from datetime import datetime, timezone

from mango_mvp.channels.contracts import ChannelDirection, ChannelMessage
from mango_mvp.channels.pilot_context import build_pilot_context, pilot_context_safety_contract


def test_pilot_context_marks_family_phone_and_quality() -> None:
    message = ChannelMessage(
        channel="telegram_bot",
        channel_message_id="m1",
        channel_thread_id="chat-1",
        channel_user_id="u1",
        direction=ChannelDirection.INBOUND,
        text="Какая группа подойдет?",
        received_at=datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc),
    )

    context = build_pilot_context(
        message,
        recent_messages=("Здравствуйте", "У меня двое детей"),
        client_identity={"phone": "+79000000000"},
        amo_context={"family_phone": True, "deals_count": 2},
        tallanto_context={"students_count": 2},
        facts_context={"missing": True},
    )
    payload = context.to_prompt_context()

    assert payload["context_quality"]["phone_found"] is True
    assert payload["context_quality"]["family_phone"] is True
    assert payload["context_quality"]["multiple_students"] is True
    assert payload["context_quality"]["multiple_deals"] is True
    assert payload["context_quality"]["facts_missing"] is True
    assert "family_phone" in payload["context_warnings"]
    assert "multiple_students" in payload["risk_flags"]


def test_pilot_context_safety_contract_is_read_only() -> None:
    safety = pilot_context_safety_contract()

    assert safety["read_amo"] is True
    assert safety["read_tallanto"] is True
    assert safety["write_amo"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["send_client_message"] is False


def test_pilot_context_preserves_full_autonomy_topic_list() -> None:
    allowed = [f"theme:{index:03d}_test" for index in range(1, 25)]
    allowed.extend(["theme:014_format", "theme:015_address", "theme:026_camp_general"])

    context = build_pilot_context(
        "Можно заниматься онлайн?",
        active_brand="foton",
        rop_policy={
            "bot_permission": "bot_answer_self_for_pilot",
            "autonomy_policy": {
                "allow_autonomous": True,
                "allowed_topic_ids": allowed,
            },
        },
    )
    payload = context.to_prompt_context()

    preserved = payload["rop_policy"]["autonomy_policy"]["allowed_topic_ids"]
    assert "theme:014_format" in preserved
    assert "theme:015_address" in preserved
    assert "theme:026_camp_general" in preserved
    assert len(preserved) == len(allowed)
