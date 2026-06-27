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
