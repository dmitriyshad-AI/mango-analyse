from __future__ import annotations

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.post_layers import (
    BOT_SAFE_CRM_CONTEXT_ENV,
    BOT_SAFE_MEMORY_STEP_GUARD_ENV,
    BOT_SAFE_MEMORY_STEP_GUARD_FLAG,
    apply_bot_safe_memory_step_guard,
    find_bot_safe_memory_disputed_step_claims,
)


def test_bot_safe_memory_step_guard_downgrades_review_step() -> None:
    result = _result(
        "Да, место уже забронировано, заявка подтверждена.",
        route="bot_answer_self_for_pilot",
        statuses=["needs_manager_review"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded.route == "draft_for_manager"
    assert "Уточню актуальный шаг с менеджером" in guarded.draft_text
    assert "забронировано" not in guarded.draft_text
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags
    assert guarded.metadata["bot_safe_memory_step_guard"]["review_statuses"] == ["needs_manager_review"]


def test_bot_safe_memory_step_guard_downgrades_empty_step_from_context_items() -> None:
    result = _result(
        "Место закреплено за вами, запись оформлена.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded.route == "draft_for_manager"
    assert "Уточню актуальный шаг с менеджером" in guarded.draft_text
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_bot_safe_memory_step_guard_keeps_active_confirmed_step() -> None:
    result = _result(
        "Да, место уже забронировано, заявка подтверждена.",
        route="bot_answer_self_for_pilot",
        statuses=["active"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded == result


def test_bot_safe_memory_step_guard_keeps_neutral_handoff_for_empty_status() -> None:
    result = _result(
        "Передам менеджеру, он свяжется и уточнит детали.",
        route="draft_for_manager",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded == result
    assert not find_bot_safe_memory_disputed_step_claims(result.draft_text, context=_context(flag=True))


def test_bot_safe_memory_step_guard_is_default_off() -> None:
    result = _result(
        "Да, место уже забронировано, заявка подтверждена.",
        route="bot_answer_self_for_pilot",
        statuses=["needs_manager_review"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=False))

    assert guarded is result
    assert guarded == result


def test_bot_safe_memory_step_guard_requires_separate_guard_flag() -> None:
    result = _result(
        "Да, место уже забронировано, заявка подтверждена.",
        route="bot_answer_self_for_pilot",
        statuses=["needs_manager_review"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, guard=False))

    assert guarded is result
    assert guarded.route == "bot_answer_self_for_pilot"


def test_bot_safe_memory_step_guard_off_is_noop_with_memory_context() -> None:
    result = _result(
        "Да, место уже забронировано, заявка подтверждена.",
        route="bot_answer_self_for_pilot",
        statuses=["needs_manager_review"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=False, statuses=["needs_manager_review"]))

    assert guarded is result
    assert guarded.route == "bot_answer_self_for_pilot"
    assert guarded.draft_text == "Да, место уже забронировано, заявка подтверждена."
    assert guarded.safety_flags == result.safety_flags
    assert guarded.manager_checklist == result.manager_checklist


def test_bot_safe_memory_step_guard_does_not_double_fire_followup_deadline() -> None:
    result = _result(
        "Менеджер свяжется завтра.",
        route="draft_for_manager",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded == result
    assert not find_bot_safe_memory_disputed_step_claims(result.draft_text, context=_context(flag=True))


def test_bot_safe_memory_step_guard_rewrites_soft_next_step_frame() -> None:
    result = _result(
        "Следующий шаг — уточнить класс ученика и предмет, чтобы подобрать группу.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "Следующий шаг" not in guarded.draft_text
    assert guarded.draft_text == "Уточните, пожалуйста, класс ученика, предмет, чтобы я не ошиблась с подбором."
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags
    assert guarded.metadata["bot_safe_memory_step_guard"]["claims"]


def test_bot_safe_memory_step_guard_matches_inserted_word_next_step_frame() -> None:
    result = _result(
        "Здравствуйте! Следующий шаг сейчас — уточнить класс ребёнка.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded.route == result.route
    assert guarded.draft_text == "Здравствуйте! Уточните, пожалуйста, класс ученика, чтобы я не ошиблась с подбором."
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_bot_safe_memory_step_guard_ignores_past_tense_next_step_payment() -> None:
    result = _result(
        "Следующим шагом была оплата.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded == result


def test_bot_safe_memory_step_guard_keeps_safe_current_payment_link_without_statuses() -> None:
    result = _result(
        "Следующий шаг — оплата по ссылке.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded == result


def test_bot_safe_memory_step_guard_keeps_payment_link_confirmation_without_risky_money() -> None:
    result = _result(
        "Следующий шаг — подтвердить оплату по ссылке.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded == result


def test_bot_safe_memory_step_guard_suppresses_concrete_unconfirmed_memory_step() -> None:
    result = _result(
        "Место уже забронировано, заявка подтверждена.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded.route == "draft_for_manager"
    assert guarded.draft_text == "Уточню актуальный шаг с менеджером и вернусь с ответом."
    assert "забронировано" not in guarded.draft_text


def test_bot_safe_memory_step_guard_suppresses_risky_payment_step_with_amount() -> None:
    result = _result(
        "Следующий шаг — вернуть оплату клиенту 5000 рублей.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded.route == "draft_for_manager"
    assert guarded.draft_text == "Уточню актуальный шаг с менеджером и вернусь с ответом."
    assert "5000" not in guarded.draft_text
    assert "вернуть оплату" not in guarded.draft_text


def test_bot_safe_memory_step_guard_suppresses_better_start_with_risky_payment() -> None:
    result = _result(
        "Лучше начать с возврата оплаты 5000 рублей.",
        route="bot_answer_self_for_pilot",
        statuses=["needs_manager_review"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded.route == "draft_for_manager"
    assert guarded.draft_text == "Уточню актуальный шаг с менеджером и вернусь с ответом."


def test_bot_safe_memory_step_guard_fail_open_when_rewrite_breaks(monkeypatch) -> None:
    result = _result(
        "Следующий шаг сейчас — уточнить класс ребёнка.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    def fail(_draft_text: str) -> str:
        raise RuntimeError("rewrite failed")

    monkeypatch.setattr(
        "mango_mvp.channels.subscription_llm_parts.post_layers._rewrite_bot_safe_memory_soft_step_frame",
        fail,
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statuses=["empty"]))

    assert guarded is result


def _result(
    draft_text: str,
    *,
    route: str,
    statuses: list[str],
) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=route,
        draft_text=draft_text,
        metadata={
            "direct_path": {
                "bot_safe_crm_context": {
                    "next_step_statuses": statuses,
                }
            }
        },
        safety_flags=(),
    )


def _context(*, flag: bool, statuses: list[str] | None = None, guard: bool | None = True) -> dict:
    context = {
        "active_brand": "foton",
        BOT_SAFE_CRM_CONTEXT_ENV: flag,
        "timeline_context": {
            "source": "customer_timeline_bot_context",
            "found": True,
            "bot_context": {
                "allowed_only": True,
                "items": [
                    {
                        "chunk_id": "chunk-foton",
                        "chunk_type": "bot_safe_summary",
                        "text": "Фотон: клиент обсуждал следующий шаг.",
                        "next_step_status": status,
                        "relevance_tags": ["bot_safe", "structured", "foton"],
                        "allowed_for_bot": True,
                        "requires_manager_review": False,
                    }
                    for status in (statuses or [])
                ],
            },
        },
    }
    if guard is not None:
        context[BOT_SAFE_MEMORY_STEP_GUARD_ENV] = guard
    return {
        **context,
    }
