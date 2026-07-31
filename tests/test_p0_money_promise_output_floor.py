from __future__ import annotations

import subprocess

import pytest

from mango_mvp.channels.output_verification_floor import verify_output
from mango_mvp.channels.subscription_llm_parts import (
    DEAL_ACTION_DECISION_ENV,
    SEMANTIC_OUTPUT_VERIFIER_ENV,
    SubscriptionDraftResult,
    apply_deal_action_decision_layer,
    apply_authoritative_output_gate,
    apply_semantic_output_verifier,
    build_semantic_output_verifier_prompt,
)


DIRECT_PROMISES = (
    "вернём деньги",
    "мы вернём деньги в течение недели",
    "вернём вам деньги",
    "деньги вернём",
)
MODEL_PROMISES = (
    "сделаем возврат",
    "вернем средства",
    "вернём вам оплату",
    "Мы всё уладим и переведём сумму обратно на вашу карту.",
    "Верну вам оплату сам, без менеджера.",
)
SAFE_PHRASES = (
    "Вернёмся к этому вопросу завтра.",
    "Вернёмся к деньгам после проверки менеджера.",
    "Вернёмся к вопросу. Деньги менеджер проверит отдельно.",
    "Вернём условия договора. Деньги обсудит менеджер.",
    "Ребёнок сможет вернуться в группу.",
    "Передам менеджеру вопрос о порядке возврата.",
    "Условия возврата указаны в договоре.",
    "Менеджер проверит оплату и ответит.",
)


def _floor_codes(text: str) -> set[str]:
    return {
        finding.code
        for finding in verify_output(text, facts={}, active_brand="foton", client_message="")
    }


@pytest.mark.parametrize("text", DIRECT_PROMISES)
def test_money_promise_regex_remains_fail_closed_without_model(text: str) -> None:
    assert "p0_promise" in _floor_codes(text)


@pytest.mark.parametrize("text", SAFE_PHRASES)
def test_money_promise_regex_does_not_block_safe_return_language(text: str) -> None:
    assert "p0_promise" not in _floor_codes(text)


@pytest.mark.parametrize("text", MODEL_PROMISES)
def test_money_promise_model_code_blocks_real_authoritative_gate(text: str) -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=text,
        topic_id="theme:001_general",
        metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
    )
    checked = apply_semantic_output_verifier(
        base,
        client_message="Подскажите, пожалуйста",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=lambda _prompt: {
            "findings": [
                {
                    "code": "p0_money_promise",
                    "span": text,
                    "evidence": "центр берёт обязательство вернуть деньги",
                }
            ]
        },
    )
    gated = apply_authoritative_output_gate(
        checked,
        client_message="Подскажите, пожалуйста",
        context={"active_brand": "foton"},
    )

    assert checked.metadata["semantic_output_verifier"]["finding_codes"] == ["p0_money_promise"]
    assert checked.metadata["semantic_output_verifier"]["action"] == "block"
    assert gated.route == "manager_only"
    assert gated.draft_text != text
    assert gated.metadata["authoritative_output_gate"]["action"] == "block"
    assert "p0_money_promise" in {
        item["code"] for item in gated.metadata["authoritative_output_gate"]["findings"]
    }
    decided = apply_deal_action_decision_layer(
        gated,
        client_message="Подскажите, пожалуйста",
        context={DEAL_ACTION_DECISION_ENV: True, "active_brand": "foton"},
    )
    assert decided.metadata["action_decision"]["p0_latched"] is True
    assert decided.metadata["action_decision"]["action"] == "handoff_manager"


def test_money_promise_prompt_defines_positive_and_negative_boundary() -> None:
    prompt = build_semantic_output_verifier_prompt(bot_text="Сделаем возврат.")

    assert "p0_money_promise" in prompt
    assert "обещает вернуть" in prompt
    assert "описание порядка возврата" in prompt
    assert "вернёмся к вопросу/занятиям/обсуждению" in prompt


def test_money_promise_regex_remains_floor_when_model_times_out() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Мы вернём вам деньги в течение недели.",
        topic_id="theme:001_general",
        metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
    )

    def timeout(_prompt: str) -> object:
        raise subprocess.TimeoutExpired(cmd=["semantic"], timeout=30)

    checked = apply_semantic_output_verifier(
        base,
        client_message="Подскажите, пожалуйста",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=timeout,
    )
    gated = apply_authoritative_output_gate(
        checked,
        client_message="Подскажите, пожалуйста",
        context={"active_brand": "foton"},
    )

    assert checked.metadata["semantic_output_verifier"]["unavailable"] is True
    assert gated.route == "manager_only"
    assert gated.draft_text != base.draft_text
    assert "p0_promise" in {
        item["code"] for item in gated.metadata["authoritative_output_gate"]["findings"]
    }
