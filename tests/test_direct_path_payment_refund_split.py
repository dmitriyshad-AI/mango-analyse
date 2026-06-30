from __future__ import annotations

import pytest

import mango_mvp.channels.subscription_llm as subscription_llm
from mango_mvp.channels.p0_recall_spec import codes_from_text
from mango_mvp.channels.subscription_llm import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    PAYMENT_REFUND_DISPUTE_SPLIT_ENV,
    SubscriptionDraftResult,
)
from mango_mvp.channels.subscription_llm_parts.policy_routing import PAYMENT_LINK_SAFE_TEXT
from mango_mvp.channels.subscription_llm_parts.post_layers import _direct_path_p0_text
from mango_mvp.channels.subscription_llm_parts.text_hygiene import scrub_direct_path_p0_text


def _draft(
    text: str,
    *,
    route: str = "bot_answer_self_for_pilot",
    safety_flags: tuple[str, ...] = (),
    metadata: dict[str, object] | None = None,
) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=route,
        draft_text=text,
        safety_flags=safety_flags,
        metadata=metadata or {},
    )


def _forward_payment_metadata() -> dict[str, object]:
    return {
        "semantic_frame_shadow": {
            "schema_version": "semantic_frame_shadow_v1_2026_06_30",
            "risk_class": "safe",
            "payment_readiness": "ready_to_pay",
            "requested_action": "send_payment_link",
            "must_handoff": False,
        },
        "action_decision": {
            "schema_version": "deal_action_decision_v1_2026_06_17",
            "action": "send_payment_link",
            "no_live_execution": True,
            "requires_manager_approval": True,
        },
    }


def test_payment_refund_dispute_split_flag_is_default_off_and_not_profile_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PAYMENT_REFUND_DISPUTE_SPLIT_ENV, raising=False)
    profile_context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}

    assert subscription_llm._payment_refund_dispute_split_enabled({}) is False
    assert subscription_llm._payment_refund_dispute_split_enabled(profile_context) is False
    assert PAYMENT_REFUND_DISPUTE_SPLIT_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._payment_refund_dispute_split_enabled({PAYMENT_REFUND_DISPUTE_SPLIT_ENV: "1"}) is True
    assert subscription_llm._payment_refund_dispute_split_enabled({"payment_refund_dispute_split": "1"}) is True
    assert subscription_llm._payment_refund_dispute_split_enabled({PAYMENT_REFUND_DISPUTE_SPLIT_ENV: "0"}) is False


def test_forward_payment_uses_payment_link_text_only_when_split_flag_on() -> None:
    result = _draft(
        "По возврату всё оформим, можно переходить к следующему шагу и оплате.",
        metadata=_forward_payment_metadata(),
    )

    off = scrub_direct_path_p0_text(
        result,
        context={subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1"},
        client_message="Готовы оплатить, пришлите ссылку на оплату",
    )
    on = scrub_direct_path_p0_text(
        result,
        context={
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
            PAYMENT_REFUND_DISPUTE_SPLIT_ENV: "1",
        },
        client_message="Готовы оплатить, пришлите ссылку на оплату",
    )

    assert off.draft_text != PAYMENT_LINK_SAFE_TEXT
    assert "По возврату точную сумму" in off.draft_text
    assert on.draft_text == PAYMENT_LINK_SAFE_TEXT
    assert "возврат" not in on.draft_text.casefold()
    assert on.metadata["direct_p0_text_hygiene"]["kind"] == "forward_payment"


def test_forward_payment_scrub_is_idempotent_on_inner_and_outer_paths() -> None:
    result = _draft(
        "Отлично, переходите к следующему шагу: менеджер пришлёт ссылку на оплату.",
        metadata=_forward_payment_metadata(),
    )
    context = {
        subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
        PAYMENT_REFUND_DISPUTE_SPLIT_ENV: "1",
    }

    inner = scrub_direct_path_p0_text(result, context=context, client_message="Куда оплатить?")
    outer = scrub_direct_path_p0_text(inner, context=context, client_message="Куда оплатить?")

    assert inner.draft_text == PAYMENT_LINK_SAFE_TEXT
    assert outer.draft_text == PAYMENT_LINK_SAFE_TEXT
    assert outer.safety_flags.count("direct_p0_text_hygiene") == 1
    assert outer.manager_checklist.count(
        "P0 text hygiene: убрать обещания исхода и продающий хвост перед ручной проверкой менеджером."
    ) == 1


def test_real_payment_dispute_stays_manager_only_and_not_payment_link_text() -> None:
    result = _draft(
        "Понимаю, деньги уже списались, можно переходить к оплате ещё раз.",
        route="manager_only",
        safety_flags=("payment_dispute", "direct_path_model_p0_payment_dispute"),
        metadata={
            "direct_path_model_p0": {"is_p0": True, "p0_kind": "payment_dispute"},
            **_forward_payment_metadata(),
        },
    )

    scrubbed = scrub_direct_path_p0_text(
        result,
        context={
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
            PAYMENT_REFUND_DISPUTE_SPLIT_ENV: "1",
        },
        client_message="Оплатили курс, а доступ так и не пришёл",
    )

    assert scrubbed.route == "manager_only"
    assert scrubbed.draft_text != PAYMENT_LINK_SAFE_TEXT
    assert "По оплате нужно сверить данные в системе" in scrubbed.draft_text
    assert scrubbed.metadata["direct_p0_text_hygiene"]["kind"] == "payment_dispute"


def test_presale_refund_benign_exception_stays_self_answer() -> None:
    result = _draft(
        "Да, если передумаете до оплаты, оформим возврат сразу.",
        metadata=_forward_payment_metadata(),
    )

    scrubbed = scrub_direct_path_p0_text(
        result,
        context={
            subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV: "1",
            PAYMENT_REFUND_DISPUTE_SPLIT_ENV: "1",
        },
        client_message="Перед оплатой хочу понять условия возврата, это не жалоба.",
    )

    assert scrubbed.route == "bot_answer_self_for_pilot"
    assert scrubbed.draft_text != PAYMENT_LINK_SAFE_TEXT
    assert "До оплаты можно спокойно уточнить условия заранее" in scrubbed.draft_text
    assert scrubbed.metadata["direct_presale_policy_text_hygiene"]["kind"] == "presale_policy"


def test_payment_keyword_no_longer_creates_dispute_without_floor_code() -> None:
    text, kind = _direct_path_p0_text("клиент спросил про оплату и ссылку", context={})
    dispute_text, dispute_kind = _direct_path_p0_text("payment_dispute", context={})

    assert kind == "legal"
    assert "По оплате нужно сверить данные" not in text
    assert dispute_kind == "payment_dispute"
    assert "оплате" in dispute_text.casefold()


def test_p0_floor_codes_keep_forward_payment_out_of_union() -> None:
    assert codes_from_text("Куда оплатить, пришлите ссылку на оплату") == ()
    assert "payment_dispute" in codes_from_text("Оплатили курс, логин и пароль не дали.")
    assert "refund" in codes_from_text("Я уже оплатил, хочу возврат.")
