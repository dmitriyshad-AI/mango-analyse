from __future__ import annotations

from mango_mvp.channels.conversation_orchestrator import (
    P0_LEGAL_THEME,
    P0_NEGATIVE_FEEDBACK_THEME,
    P0_REFUND_THEME,
    conversation_orchestrator_safety_contract,
    route_question_catalog_theme,
)


def test_p0_refund_negative_and_legal_themes_force_manager_handoff() -> None:
    for theme_id in (P0_REFUND_THEME, P0_NEGATIVE_FEEDBACK_THEME, P0_LEGAL_THEME):
        decision = route_question_catalog_theme(theme_id, context={"channel": "telegram"})

        assert decision.priority == "P0"
        assert decision.route_type == "manager_handoff_p0"
        assert decision.notify_rop is True
        assert decision.bot_may_answer is False
        assert decision.handoff_target


def test_non_p0_theme_keeps_standard_policy() -> None:
    decision = route_question_catalog_theme("theme:001_pricing")

    assert decision.priority == "normal"
    assert decision.route_type == "standard_policy"
    assert decision.notify_rop is False
    assert decision.bot_may_answer is True


def test_conversation_orchestrator_safety_contract_is_read_only() -> None:
    contract = conversation_orchestrator_safety_contract()

    assert contract["routing_only"] is True
    assert contract["live_send"] is False
    assert contract["write_crm"] is False
    assert contract["write_tallanto"] is False
