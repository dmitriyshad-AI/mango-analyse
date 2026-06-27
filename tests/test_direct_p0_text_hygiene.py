from __future__ import annotations

import pytest

import mango_mvp.channels.subscription_llm as subscription_llm
from mango_mvp.channels.subscription_llm import DIRECT_PATH_ENV, SubscriptionDraftResult, SubscriptionLlmDraftProvider
from mango_mvp.channels.subscription_llm_parts.text_hygiene import scrub_direct_path_p0_text
from mango_mvp.channels.tone_block import close_detect_enabled


class _DirectPathProvider(SubscriptionLlmDraftProvider):
    def __init__(self, result: SubscriptionDraftResult) -> None:
        super().__init__()
        self.result = result
        self.calls = 0
        self.last_prompt = ""

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        self.last_prompt = prompt
        return self.result


def _profile_context() -> dict[str, str]:
    return {subscription_llm.DIRECT_PATH_PILOT_CONFIG_ENV: subscription_llm.DIRECT_PATH_PILOT_CONFIG_VERSION}


def test_tone_close_detect_enabled_by_pilot_profile_with_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(subscription_llm.TONE_CLOSE_DETECT_ENV, raising=False)

    assert close_detect_enabled({}) is False
    assert subscription_llm.TONE_CLOSE_DETECT_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert close_detect_enabled(_profile_context()) is True
    assert close_detect_enabled({**_profile_context(), subscription_llm.TONE_CLOSE_DETECT_ENV: "0"}) is False
    assert close_detect_enabled({**_profile_context(), "tone_close_detect_enabled": False}) is False


def test_tone_close_detect_profile_on_stays_silent_on_p0() -> None:
    result = subscription_llm.apply_tone_close_detect_layer(
        SubscriptionDraftResult(
            route="manager_only",
            draft_text="Передам вопрос менеджеру.",
            safety_flags=("manager_approval_required", "refund"),
        ),
        client_message="Спасибо",
        context=_profile_context(),
    )

    assert result.route == "manager_only"
    assert result.draft_text == "Передам вопрос менеджеру."
    assert result.metadata["close_detect"]["status"] == "suppressed_p0"


def test_tone_close_detect_keeps_pending_after_recent_manager_handoff() -> None:
    result = subscription_llm.apply_tone_close_detect_layer(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Спасибо вам! Будем рады видеть вас на занятиях — возвращайтесь, если появятся вопросы.",
            safety_flags=("direct_path_model",),
        ),
        client_message="Хорошо, жду тогда менеджера.",
        context={
            **_profile_context(),
            "dialogue_memory_view": {
                "route_history": ["bot_answer_self_for_pilot", "manager_only"],
                "recent_turns": [
                    {
                        "role": "bot",
                        "text": (
                            "По возврату точную сумму и порядок действий должен подтвердить менеджер. "
                            "Передам ему ваш вопрос, чтобы он проверил ситуацию по данным записи и оплаты."
                        ),
                    }
                ],
            },
        },
    )

    lowered = result.draft_text.casefold()
    assert result.route == "manager_only"
    assert "tone_close_detect_pending" in result.safety_flags
    assert "рады видеть" not in lowered
    assert "занятиях" not in lowered
    assert "менеджер" in lowered
    assert result.metadata["close_detect"]["status"] == "suppressed_pending"


def test_direct_p0_text_hygiene_default_off_is_noop() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Да, возвращается остаток. Можно записаться и оплатить новый курс.",
        metadata={"direct_path_model_p0": {"is_p0": True, "p0_kind": "refund"}},
    )

    assert scrub_direct_path_p0_text(result, context={}, client_message="Нужно уточнить возврат.") is result


def test_direct_p0_text_hygiene_provider_level_scrubs_refund_sales_tail() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, возвращается остаток. Можно записаться и оплатить новый курс.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "refund",
                    "model_reason": "клиент спрашивает про возврат",
                }
            },
        )
    )

    result = provider.build_draft(
        "Подскажите, что делать дальше?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
        },
    )

    lowered = result.draft_text.casefold()
    assert provider.calls == 1
    assert result.route == "manager_only"
    assert "direct_p0_text_hygiene" in result.safety_flags
    assert "да, возвращается" not in lowered
    assert "оплатить" not in lowered
    assert "новый курс" not in lowered
    assert "записаться" not in lowered
    assert "менеджер" in lowered
    assert result.metadata["direct_p0_text_hygiene"]["kind"] == "refund"


def test_direct_p0_text_hygiene_payment_dispute_keeps_route_and_removes_sales_text() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можем посмотреть скидку, подобрать группу и перейти к оплате.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "payment_dispute",
                    "model_reason": "спорная оплата",
                }
            },
        )
    )

    result = provider.build_draft(
        "Подскажите, что делать дальше?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
        },
    )

    lowered = result.draft_text.casefold()
    assert result.route == "manager_only"
    assert "payment_dispute" in result.safety_flags
    assert "скидк" not in lowered
    assert "подобрать группу" not in lowered
    assert "оплате нужно сверить данные" in lowered


def test_direct_p0_text_hygiene_does_not_touch_presale_refund_policy() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Если передумаете до старта, менеджер подскажет порядок возврата по правилам.",
            risk_level="low",
            metadata={"direct_path_model_p0": {"is_p0": False, "p0_kind": "none"}},
        )
    )

    result = provider.build_draft(
        "Если передумаем до оплаты, какие условия возврата?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Если передумаете до старта, менеджер подскажет порядок возврата по правилам."
    assert "direct_p0_text_hygiene" not in result.safety_flags


def test_direct_p0_text_hygiene_prompt_instruction_is_flagged_only() -> None:
    context = {"active_brand": "foton", DIRECT_PATH_ENV: "1", subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1"}

    off_prompt = subscription_llm._build_direct_path_prompt("Верните оплату.", context=context)
    on_prompt = subscription_llm._build_direct_path_prompt(
        "Верните оплату.",
        context={**context, subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1"},
    )

    assert "P0-гигиена текста" not in off_prompt
    assert "P0-гигиена текста" in on_prompt
    assert "не обещай исход возврата" in on_prompt


def test_direct_p0_text_hygiene_contract_dispute_zero_collects_detail_request() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=(
                "Понимаю. Напишите, пожалуйста, что именно указано неверно в договоре: "
                "какое поле и как должно быть правильно. Передам менеджеру."
            ),
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "contract_dispute",
                    "model_reason": "клиент спорит с договором",
                }
            },
        )
    )

    result = provider.build_draft(
        "Договор составлен неправильно, там ошибка. Что делать?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
            subscription_llm.P0_MODEL_CLASSES_V2_ENV: "1",
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
        },
    )

    lowered = result.draft_text.casefold()
    assert result.route == "manager_only"
    assert "direct_p0_text_hygiene" in result.safety_flags
    assert "что именно" not in lowered
    assert "какое поле" not in lowered
    assert "как должно быть правильно" not in lowered
    assert "возврат" not in lowered
    assert "документы" in lowered
    assert result.metadata["direct_p0_text_hygiene"]["kind"] == "legal_threat"


def test_direct_p0_text_hygiene_final_hook_scrubs_post_gate_refund_manager_only() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=(
                "Да, если после оплаты передумаете, возвращается остаток неистраченных средств. "
                "По онлайн-математике можно спокойно переходить к следующему шагу, если условия подходят."
            ),
            message_type="question",
            topic_id="theme:014_format",
            risk_level="low",
            safety_flags=("refund",),
            metadata={"direct_path_model_p0": {"is_p0": False, "p0_kind": "none"}},
        )
    )

    result = provider.build_draft(
        "Хорошо, оплачу, но если передумаю - деньги вернете?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DEAL_ACTION_DECISION_ENV: "1",
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
            "client_safe_fact_verified": True,
        },
    )

    lowered = result.draft_text.casefold()
    assert result.route == "manager_only"
    assert "autonomy_blocked_high_risk" in result.safety_flags
    assert "direct_p0_text_hygiene" in result.safety_flags
    assert "возвращается остаток" not in lowered
    assert "следующему шагу" not in lowered
    assert "условия подходят" not in lowered
    assert "менеджер" in lowered


def test_direct_p0_text_hygiene_keeps_benign_presale_refund_without_high_risk_route() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Если передумаете до оплаты, менеджер подскажет порядок возврата по правилам.",
    )

    scrubbed = scrub_direct_path_p0_text(
        result,
        context={subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1"},
        client_message="Если передумаем до оплаты, какие условия возврата?",
    )

    assert scrubbed is result


def test_direct_p0_text_hygiene_keeps_benign_presale_refund_without_refund_word() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=(
            "До оплаты можно спокойно уточнить условия. Если передумаете до записи и оплаты, "
            "менеджер подтвердит порядок по выбранному курсу."
        ),
    )

    scrubbed = scrub_direct_path_p0_text(
        result,
        context={subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1"},
        client_message="А если передумаю ещё ДО оплаты — это ничем не грозит?",
    )

    assert scrubbed is result


def test_direct_p0_text_hygiene_keeps_benign_no_record_or_payment_clarification() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Если записи и оплаты ещё нет, можно спокойно уточнить правила заранее.",
    )

    scrubbed = scrub_direct_path_p0_text(
        result,
        context={subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1"},
        client_message="Я же про до оплаты, записи и оплаты ещё нет. Просто хочу понять, можно ли спокойно передумать.",
    )

    assert scrubbed is result


def test_direct_p0_text_hygiene_replaces_false_postpayment_handoff_on_presale_refund() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=(
            "По возврату точную сумму и порядок действий должен подтвердить менеджер. "
            "Передам ему ваш вопрос, чтобы он проверил ситуацию по данным записи и оплаты."
        ),
    )

    scrubbed = scrub_direct_path_p0_text(
        result,
        context={subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1"},
        client_message="А если передумаю ещё ДО оплаты — это ничем не грозит?",
    )

    lowered = scrubbed.draft_text.casefold()
    assert scrubbed is not result
    assert "данным записи и оплаты" not in lowered
    assert "точную сумму" not in lowered
    assert "до оплаты" in lowered
    assert "выбранному курсу" in lowered
    assert "direct_presale_policy_text_hygiene" in scrubbed.safety_flags
    risk_words = ("p0", "refund", "payment", "legal", "complaint", "high_risk")
    assert all(not any(word in flag.casefold() for word in risk_words) for flag in scrubbed.safety_flags)
    assert scrubbed.metadata["direct_presale_policy_text_hygiene"]["kind"] == "presale_policy"
