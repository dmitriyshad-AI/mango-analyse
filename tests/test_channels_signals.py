from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mango_mvp.channels import (
    SIGNAL_ATTACHMENT_RECEIVED,
    SIGNAL_COMMERCIAL_RISK,
    SIGNAL_CUSTOMER_QUESTION,
    SIGNAL_FOLLOW_UP,
    SIGNAL_HOT_LEAD,
    SIGNAL_MANAGER_HANDOFF,
    SIGNAL_NEED_CRM_CONTEXT,
    SIGNAL_URGENCY,
    CRM_CHAT_CHANNEL,
    ChannelAttachment,
    ChannelMessage,
    ChannelSession,
    SafeAnswer,
    SignalPolicy,
    WebChatReadOnlyAdapter,
    build_channel_draft_preview,
    build_channel_signal_decision,
    default_signal_policy_map,
    extract_customer_signals,
    signal_engine_safety_contract,
)


NOW = datetime(2026, 5, 9, 18, 0, tzinfo=timezone.utc)


def message(text: str) -> ChannelMessage:
    return ChannelMessage(
        channel="site_chat",
        channel_message_id="m1",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction="inbound",
        text=text,
        received_at=NOW,
        raw_payload={"secret": "not exported by signal decision"},
    )


def signal_map(decision):
    return {signal.signal_type: signal for signal in decision.signals}


def test_default_signal_policies_are_safe() -> None:
    policies = default_signal_policy_map()

    assert policies[SIGNAL_NEED_CRM_CONTEXT]["autonomy_level"] == "L1"
    assert policies[SIGNAL_HOT_LEAD]["requires_notification"] is True
    assert policies[SIGNAL_COMMERCIAL_RISK]["requires_manager_review"] is True
    assert all(policy["allow_autonomous_reply"] is False for policy in policies.values())
    assert all(policy["allow_live_execution"] is False for policy in policies.values())


def test_signal_policy_refuses_autonomous_reply_and_live_execution() -> None:
    with pytest.raises(ValueError, match="must not allow autonomous"):
        SignalPolicy(
            signal_type="unsafe_reply",
            autonomy_level="L4",
            allow_autonomous_reply=True,
        )
    with pytest.raises(ValueError, match="must not allow live execution"):
        SignalPolicy(
            signal_type="unsafe_live_action",
            autonomy_level="L4",
            allow_live_execution=True,
        )


def test_extracts_question_urgency_commercial_handoff_followup_and_hot_lead_signals() -> None:
    msg = message("Срочно позвоните менеджеру завтра, хочу оплатить со скидкой. Сколько стоит курс?")
    preview = build_channel_draft_preview(msg, context={"priority": "hot", "follow_up_due_at": "2026-05-10"})

    decision = build_channel_signal_decision(message=msg, preview=preview, context={"priority": "hot"})
    by_type = signal_map(decision)

    assert SIGNAL_NEED_CRM_CONTEXT in by_type
    assert SIGNAL_CUSTOMER_QUESTION in by_type
    assert SIGNAL_URGENCY in by_type
    assert SIGNAL_COMMERCIAL_RISK in by_type
    assert SIGNAL_MANAGER_HANDOFF in by_type
    assert SIGNAL_FOLLOW_UP in by_type
    assert SIGNAL_HOT_LEAD in by_type
    assert by_type[SIGNAL_COMMERCIAL_RISK].severity == "high"
    assert by_type[SIGNAL_HOT_LEAD].policy.requires_notification is True
    assert decision.policy_flags["requires_manager_review"] is True
    assert decision.policy_flags["notify_rop"] is True
    assert "commercial_review_required" in decision.blocked_reasons
    assert decision.safe_answer.requires_approval is True


def test_context_only_signals_work_without_preview_or_actions() -> None:
    msg = message("Здравствуйте")
    session = ChannelSession.from_message(msg)

    signals = extract_customer_signals(
        message=msg,
        session=session,
        context={
            "crm_context_missing": True,
            "requires_commercial_review": True,
            "force_manager_handoff": True,
            "follow_up_due_at": "2026-05-10",
            "priority": "hot",
        },
    )
    types = {signal.signal_type for signal in signals}

    assert types == {
        SIGNAL_NEED_CRM_CONTEXT,
        SIGNAL_COMMERCIAL_RISK,
        SIGNAL_MANAGER_HANDOFF,
        SIGNAL_FOLLOW_UP,
        SIGNAL_HOT_LEAD,
    }
    assert all(signal.evidence[0].source_type == "read_only_context" for signal in signals)


def test_low_context_priority_does_not_create_hot_lead_signal() -> None:
    msg = message("Здравствуйте")
    session = ChannelSession.from_message(msg)

    signals = extract_customer_signals(
        message=msg,
        session=session,
        context={"priority": "low", "lead_priority": "normal"},
    )

    assert {signal.signal_type for signal in signals} == set()


def test_signal_decision_is_stable_and_does_not_export_raw_payload() -> None:
    msg = message("Сколько стоит подготовка к ЕГЭ?")
    preview = build_channel_draft_preview(msg)

    first = build_channel_signal_decision(message=msg, preview=preview)
    second = build_channel_signal_decision(message=msg, preview=preview)
    payload = first.to_json_dict()

    assert first.decision_id == second.decision_id
    assert [signal.idempotency_key for signal in first.signals] == [signal.idempotency_key for signal in second.signals]
    assert "raw_payload" not in str(payload)
    assert payload["safe_answer"]["safety_flags"] == [
        "draft_only",
        "requires_manager_approval",
        "live_send_disabled",
        "no_llm_used",
        "no_rag_used",
        "no_crm_write",
    ]


def test_signal_engine_does_not_turn_plain_draft_action_into_customer_question() -> None:
    msg = message("Здравствуйте")
    preview = build_channel_draft_preview(msg)

    decision = build_channel_signal_decision(message=msg, preview=preview)
    types = {signal.signal_type for signal in decision.signals}

    assert SIGNAL_NEED_CRM_CONTEXT in types
    assert SIGNAL_CUSTOMER_QUESTION not in types
    assert decision.safe_answer.requires_approval is True


def test_attachment_signal_requires_review() -> None:
    msg = ChannelMessage(
        channel="site_chat",
        channel_message_id="m-attach",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction="inbound",
        attachments=(ChannelAttachment(kind="document", uri="memory://contract.pdf"),),
        received_at=NOW,
    )
    preview = build_channel_draft_preview(msg)

    decision = build_channel_signal_decision(message=msg, preview=preview)
    by_type = signal_map(decision)

    assert SIGNAL_ATTACHMENT_RECEIVED in by_type
    assert by_type[SIGNAL_ATTACHMENT_RECEIVED].requires_manager_review is True
    assert "attachment_review_required" in decision.safe_answer.blocked_reasons


def test_signal_decision_rejects_session_mismatch() -> None:
    msg = message("Сколько стоит курс?")
    wrong_session = ChannelSession(channel="site_chat", channel_thread_id="other-thread", updated_at=NOW)

    with pytest.raises(ValueError, match="session thread must match"):
        build_channel_signal_decision(message=msg, session=wrong_session)


def test_safe_answer_refuses_non_draft_or_unapproved_answers() -> None:
    with pytest.raises(ValueError, match="draft answers only"):
        SafeAnswer(text="Ответ", answer_type="live")
    with pytest.raises(ValueError, match="must require approval"):
        SafeAnswer(text="Ответ", requires_approval=False)


def test_web_chat_adapter_to_signal_engine_e2e() -> None:
    adapter = WebChatReadOnlyAdapter(default_channel=CRM_CHAT_CHANNEL)
    payload = {
        "channel": CRM_CHAT_CHANNEL,
        "message_id": "crm-msg-1",
        "conversation_id": "lead-1",
        "contact_id": "contact-1",
        "body": "Срочно нужен договор и счет, можно связаться с менеджером?",
        "timestamp": int(NOW.timestamp()),
    }
    msg = adapter.parse_inbound(payload)[0]
    preview = build_channel_draft_preview(msg, context={"priority": "high"})

    decision = build_channel_signal_decision(message=msg, preview=preview, context={"priority": "high"})
    types = {signal.signal_type for signal in decision.signals}

    assert SIGNAL_CUSTOMER_QUESTION in types
    assert SIGNAL_URGENCY in types
    assert SIGNAL_COMMERCIAL_RISK in types
    assert SIGNAL_MANAGER_HANDOFF in types
    assert SIGNAL_HOT_LEAD in types
    assert decision.policy_flags["allow_autonomous_reply"] is False
    assert decision.policy_flags["write_crm"] is False
    assert decision.policy_flags["write_runtime_db"] is False


def test_signal_engine_safety_contract_blocks_external_effects() -> None:
    safety = signal_engine_safety_contract()

    assert safety["network_calls"] is False
    assert safety["llm_calls"] is False
    assert safety["rag_used"] is False
    assert safety["live_send"] is False
    assert safety["write_crm"] is False
    assert safety["write_runtime_db"] is False
