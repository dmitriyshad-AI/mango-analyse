from __future__ import annotations

from dataclasses import replace

from mango_mvp.channels.manager_handoff_summary import build_manager_handoff_summary
from mango_mvp.channels.subscription_llm import (
    DEAL_ACTION_DECISION_ENV,
    SubscriptionDraftResult,
    apply_deal_action_decision_layer,
)


def _result(**overrides):
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        message_type="question",
        draft_text="Стоимость онлайн-курса физики для 8 класса — 47 250 ₽.",
        safety_flags=("direct_path_model", "draft_only"),
        metadata={
            "direct_path": {
                "retrieved_facts": {
                    "foton_online_price_physics_8": "Фотон: онлайн-курс физики для 8 класса стоит 47 250 ₽."
                },
                "wide_fact_exact_keys": ["foton_online_price_physics_8"],
            },
            "action_proposal": {"action": "answer_only", "confidence": 0.8},
        },
    )
    return replace(base, **overrides)


def _context(**extra):
    context = {
        DEAL_ACTION_DECISION_ENV: "1",
        "active_brand": "foton",
        "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
        "conversation_intent_plan": {"selling": {"objection": "none", "exit_signal": False}},
    }
    context.update(extra)
    return context


def _decision(result):
    return result.metadata["action_decision"]


def test_deal_action_decision_off_parity():
    result = _result()
    checked = apply_deal_action_decision_layer(result, client_message="Сколько стоит?", context={"active_brand": "foton"})
    assert checked == result
    assert "action_decision" not in checked.metadata


def test_manager_only_route_always_handoff_even_with_payment_text():
    result = _result(
        route="manager_only",
        safety_flags=("manager_approval_required", "no_auto_send"),
        metadata={
            "direct_path": {
                "retrieved_facts": {
                    "foton_online_price_physics_8": "Фотон: онлайн-курс физики для 8 класса стоит 47 250 ₽."
                },
                "wide_fact_exact_keys": ["foton_online_price_physics_8"],
            },
            "action_proposal": {"action": "send_payment_link", "confidence": 0.95},
        },
    )
    checked = apply_deal_action_decision_layer(result, client_message="Готова оплатить, давайте ссылку", context=_context())
    assert _decision(checked)["action"] == "handoff_manager"
    assert _decision(checked)["reason"] == "manager_only_route"


def test_p0_multitopic_complaint_latches_to_handoff_manager():
    result = _result(metadata={"action_proposal": {"action": "send_payment_link", "confidence": 0.95}})
    checked = apply_deal_action_decision_layer(
        result,
        client_message="Ребёнка унизили на занятии, сколько стоит физика?",
        context=_context(),
    )
    assert _decision(checked)["action"] == "handoff_manager"
    assert _decision(checked)["p0_latched"] is True
    assert _decision(checked)["requires_manager_approval"] is True


def test_answer_only_does_not_require_manager_approval():
    checked = apply_deal_action_decision_layer(
        _result(),
        client_message="Какая стоимость онлайн-физики?",
        context=_context(conversation_intent_plan={"primary_intent": "pricing"}),
    )
    assert _decision(checked)["action"] == "answer_only"
    assert _decision(checked)["requires_manager_approval"] is False


def test_send_payment_link_requires_explicit_payment_not_signup():
    result = _result(
        draft_text="Да, могу помочь с записью: передам заявку менеджеру.",
        metadata={**_result().metadata, "action_proposal": {"action": "send_payment_link", "confidence": 0.95}},
    )
    checked = apply_deal_action_decision_layer(result, client_message="Запишите нас", context=_context())
    assert _decision(checked)["action"] != "send_payment_link"
    assert _decision(checked)["action"] == "capture_lead"


def test_send_payment_link_requires_price_fact_and_product_slots():
    result = _result(
        draft_text="Стоимость онлайн-курса физики для 8 класса — 47 250 ₽. Менеджер может оформить ссылку на оплату.",
        metadata={**_result().metadata, "action_proposal": {"action": "send_payment_link", "confidence": 0.95}},
    )
    checked = apply_deal_action_decision_layer(result, client_message="Готова оплатить, оформляйте", context=_context())
    assert _decision(checked)["action"] == "send_payment_link"
    assert _decision(checked)["preconditions"]["price_backed_by_facts"] is True

    no_slots = apply_deal_action_decision_layer(
        result,
        client_message="Готова оплатить, оформляйте",
        context=_context(known_slots={}),
    )
    assert _decision(no_slots)["action"] == "unknown"
    assert _decision(no_slots)["reason"] == "payment_preconditions_missing"
    assert _decision(no_slots)["requires_manager_approval"] is False


def test_model_can_lower_payment_recommendation():
    result = _result(metadata={**_result().metadata, "action_proposal": {"action": "answer_only", "confidence": 0.95}})
    checked = apply_deal_action_decision_layer(result, client_message="Готова оплатить, оформляйте", context=_context())
    assert _decision(checked)["action"] == "answer_only"
    assert _decision(checked)["reason"] == "model_lowered_payment"
    assert _decision(checked)["requires_manager_approval"] is False


def test_crm_data_requires_strict_identity_and_brand_match():
    crm_context = _context(
        context_quality={"customer_identity_found": True},
        client_identity={"verified": True, "match_class": "exact"},
        amo_context={"brand": "foton"},
    )
    checked = apply_deal_action_decision_layer(_result(), client_message="Сколько у нас осталось занятий?", context=crm_context)
    assert _decision(checked)["action"] == "send_crm_data"

    wrong_brand = apply_deal_action_decision_layer(
        _result(),
        client_message="Сколько у нас осталось занятий?",
        context=_context(
            context_quality={"customer_identity_found": True},
            client_identity={"verified": True, "match_class": "exact"},
            amo_context={"brand": "unpk"},
        ),
    )
    assert _decision(wrong_brand)["action"] == "handoff_manager"
    assert "brand_mismatch" in _decision(wrong_brand)["reason"]


def test_unknown_action_does_not_use_old_funnel_recommendation():
    summary = build_manager_handoff_summary(
        brand="foton",
        client_message="Подскажите",
        answer_text="Передам менеджеру.",
        route="draft_for_manager",
        funnel_state={"next_best_question": "оставьте телефон", "next_step_type": "capture_contact"},
        context={"action_decision": {"action": "unknown"}},
    )
    assert "Рекомендуемый следующий шаг: обработать вручную" in summary
    assert "оставьте телефон" not in summary
