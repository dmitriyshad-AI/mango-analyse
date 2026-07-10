from __future__ import annotations

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.post_layers import (
    BOT_SAFE_CRM_CONTEXT_ENV,
    BOT_SAFE_MEMORY_STEP_GUARD_ENV,
    BOT_SAFE_MEMORY_STEP_GUARD_FLAG,
    NO_MEMORY_STEP_FRAME_GUARD_FLAG,
    UNCONFIRMED_CONTACT_DATA_CLAIM_FLAG,
    apply_no_memory_step_frame_guard,
    apply_bot_safe_memory_step_guard,
    apply_unconfirmed_contact_data_claim_guard,
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


def test_bot_safe_memory_step_guard_reads_nested_next_step_metadata() -> None:
    result = _result(
        "Место закреплено за вами, запись оформлена.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, nested_statuses=["empty"]))

    assert guarded.route == "draft_for_manager"
    assert "Уточню актуальный шаг с менеджером" in guarded.draft_text
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_bot_safe_memory_step_guard_reads_result_next_step_metadata() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Место закреплено за вами, запись оформлена.",
        metadata={"next_step": {"status": "empty"}},
        safety_flags=(),
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded.route == "draft_for_manager"
    assert "Уточню актуальный шаг с менеджером" in guarded.draft_text
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_bot_safe_memory_step_guard_treats_missing_item_status_as_empty() -> None:
    result = _result(
        "Место закреплено за вами, запись оформлена.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True, statusless_items=True))

    assert guarded.route == "draft_for_manager"
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_bot_safe_memory_step_guard_rewrites_soft_next_step_without_manager_handoff() -> None:
    result = _result(
        "Следующий шаг — уточнить класс ученика, чтобы подобрать группу.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "следующий шаг" not in guarded.draft_text.casefold()
    assert "Уточните, пожалуйста, класс ученика" in guarded.draft_text
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_bot_safe_memory_step_guard_rewrites_soft_pick_or_check_step() -> None:
    result = _result(
        "Следующий шаг — подобрать вариант и проверить формат занятий.",
        route="bot_answer_self_for_pilot",
        statuses=["needs_manager_review"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "следующий шаг" not in guarded.draft_text.casefold()
    assert "Уточните, пожалуйста" in guarded.draft_text
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_bot_safe_memory_step_guard_keeps_soft_next_step_when_active() -> None:
    result = _result(
        "Следующий шаг — уточнить класс ученика, чтобы подобрать группу.",
        route="bot_answer_self_for_pilot",
        statuses=["active"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded == result


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


def test_bot_safe_memory_step_guard_requires_separate_step_guard_flag() -> None:
    result = _result(
        "Да, место уже забронировано, заявка подтверждена.",
        route="bot_answer_self_for_pilot",
        statuses=["needs_manager_review"],
    )

    guarded = apply_bot_safe_memory_step_guard(
        result,
        context={BOT_SAFE_CRM_CONTEXT_ENV: True, "active_brand": "foton"},
    )

    assert guarded == result


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


def test_bot_safe_memory_step_guard_blocks_risky_payment_step() -> None:
    result = _result(
        "Следующий шаг — вернуть 15000 рублей клиенту.",
        route="bot_answer_self_for_pilot",
        statuses=["empty"],
    )

    guarded = apply_bot_safe_memory_step_guard(result, context=_context(flag=True))

    assert guarded.route == "draft_for_manager"
    assert "Уточню актуальный шаг с менеджером" in guarded.draft_text
    assert BOT_SAFE_MEMORY_STEP_GUARD_FLAG in guarded.safety_flags


def test_unconfirmed_contact_data_claim_guard_rewrites_missing_phone_claim() -> None:
    result = _result(
        "Телефон повторно присылать не нужно, он уже есть в диалоге.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_unconfirmed_contact_data_claim_guard(
        result,
        client_message="Хочу записаться на курс",
        context={"recent_messages": ["Клиент: Хочу записаться на курс"]},
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert guarded.draft_text == "Повторно указывать не обязательно — менеджер сверит по системе."
    assert UNCONFIRMED_CONTACT_DATA_CLAIM_FLAG in guarded.safety_flags
    assert "телефон" not in guarded.draft_text.casefold()
    assert guarded.metadata["unconfirmed_contact_data_claim_guard"]["applied"] is True


def test_unconfirmed_contact_data_claim_guard_keeps_confirmed_phone_from_client_dialogue() -> None:
    result = _result(
        "Телефон повторно присылать не нужно, он уже есть в диалоге.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_unconfirmed_contact_data_claim_guard(
        result,
        client_message="Мой телефон +7 999 123-45-67",
        context={"recent_messages": ["Клиент: Мой телефон +7 999 123-45-67"]},
    )

    assert guarded == result


def test_unconfirmed_contact_data_claim_guard_ignores_school_contact_fact() -> None:
    result = _result(
        "Телефон повторно присылать не нужно, он уже есть в диалоге.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_unconfirmed_contact_data_claim_guard(
        result,
        client_message="Хочу записаться на курс",
        context={
            "client_safe_fact_verified": True,
            "confirmed_facts": {
                "foton.contacts": "Фотон: телефон центра +7 495 000-00-00, почта info@example.com."
            },
        },
    )

    assert guarded.draft_text == "Повторно указывать не обязательно — менеджер сверит по системе."
    assert UNCONFIRMED_CONTACT_DATA_CLAIM_FLAG in guarded.safety_flags


def test_no_memory_step_frame_guard_rewrites_baseline_better_start_frame() -> None:
    result = _result(
        "Лучше начать с класса ученика, чтобы подобрать группу.",
        route="bot_answer_self_for_pilot",
        statuses=[],
    )

    guarded = apply_no_memory_step_frame_guard(result, context={})

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "лучше начать" not in guarded.draft_text.casefold()
    assert "Предлагаю начать с класса ученика" in guarded.draft_text
    assert NO_MEMORY_STEP_FRAME_GUARD_FLAG in guarded.safety_flags


def test_no_memory_step_frame_guard_keeps_active_next_step_frame() -> None:
    result = _result(
        "Следующий шаг — уточнить класс ученика, чтобы подобрать группу.",
        route="bot_answer_self_for_pilot",
        statuses=["active"],
    )

    guarded = apply_no_memory_step_frame_guard(result, context=_context(flag=True))

    assert guarded == result


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


def _context(
    *,
    flag: bool,
    statuses: list[str] | None = None,
    nested_statuses: list[str] | None = None,
    statusless_items: bool = False,
) -> dict:
    items = [
        {
            "chunk_id": f"chunk-foton-{idx}",
            "chunk_type": "bot_safe_summary",
            "text": "Фотон: клиент обсуждал следующий шаг.",
            "next_step_status": status,
            "relevance_tags": ["bot_safe", "structured", "foton"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        }
        for idx, status in enumerate(statuses or [], 1)
    ]
    items.extend(
        {
            "chunk_id": f"chunk-foton-nested-{idx}",
            "chunk_type": "bot_safe_summary",
            "text": "Фотон: клиент обсуждал следующий шаг.",
            "metadata": {"next_step": {"status": status}},
            "relevance_tags": ["bot_safe", "structured", "foton"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        }
        for idx, status in enumerate(nested_statuses or [], 1)
    )
    if statusless_items:
        items.append(
            {
                "chunk_id": "chunk-foton-statusless",
                "chunk_type": "bot_safe_summary",
                "text": "Фотон: клиент обсуждал следующий шаг.",
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            }
        )
    return {
        BOT_SAFE_CRM_CONTEXT_ENV: flag,
        BOT_SAFE_MEMORY_STEP_GUARD_ENV: flag,
        "active_brand": "foton",
        "timeline_context": {
            "source": "customer_timeline_bot_context",
            "found": True,
            "bot_context": {
                "allowed_only": True,
                "items": items,
            },
        },
    }
