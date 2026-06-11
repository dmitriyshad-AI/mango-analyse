from __future__ import annotations

from datetime import datetime, timezone

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from scripts.check_public_bot_live import (
    LiveCheckTurn,
    llm_fallback_detected,
    retrieved_fact_keys,
    validate_turns,
    zero_drafts_alert,
)


def test_retrieved_fact_keys_collects_nested_direct_path_metadata() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Ответ",
        context_used=("fact.context",),
        metadata={
            "direct_path": {
                "fact_pack": {
                    "facts": {"fact.a": "A"},
                    "exact_keys": ["fact.b"],
                    "llm_retrieve": {"supplemented_exact_ids": ["fact.c"]},
                }
            }
        },
    )

    assert retrieved_fact_keys(result) == ("fact.a", "fact.b", "fact.c", "fact.context")


def test_live_check_validation_catches_memory_and_fact_failures() -> None:
    turns = [
        LiveCheckTurn(
            name="physics_online",
            input_text="онлайн",
            answer_text="Какой класс?",
            route="draft_for_manager",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        )
    ]

    failures = validate_turns(turns)

    reasons = {item["reason"] for item in failures}
    assert {"retrieved_facts", "memory_known_slots", "no_grade_reask"}.issubset(reasons)


def test_live_check_validation_accepts_required_public_bot_behaviour() -> None:
    turns = [
        LiveCheckTurn(
            name="greeting",
            input_text="привет",
            answer_text="Здравствуйте! Расскажите, какой класс и предмет интересуют?",
            route="bot_answer_self_for_pilot",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
        LiveCheckTurn(
            name="physics_first",
            input_text="8 класс физика",
            answer_text="Есть очный и онлайн-формат.",
            route="bot_answer_self_for_pilot",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=("schedule",),
            known_slots={},
        ),
        LiveCheckTurn(
            name="physics_online",
            input_text="онлайн",
            answer_text="Онлайн-группа: воскресенье 14:30–16:30, старт 20.09.2026. Семестр — 29 750 ₽, год — 47 250 ₽.",
            route="bot_answer_self_for_pilot",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=("schedule", "price"),
            known_slots={"grade": "8", "subject": "физика"},
        ),
        LiveCheckTurn(
            name="cross_brand",
            input_text="вы же одна контора с УНПК?",
            answer_text="Фотон и УНПК — разные бренды, по Фотону подскажу отдельно.",
            route="manager_only",
            safety_flags=("cross_brand",),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
        LiveCheckTurn(
            name="p0_double_charge",
            input_text="двойное списание!",
            answer_text="Передам менеджеру.",
            route="manager_only",
            safety_flags=("p0_deferral",),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
        LiveCheckTurn(
            name="pii_capture",
            input_text="запишите: Иванов Пётр 8-900-123-45-67",
            answer_text="Записала данные ребёнка, менеджер свяжется.",
            route="draft_for_manager",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
    ]

    assert validate_turns(turns) == ()


def test_llm_fallback_detected_uses_flags_error_and_text() -> None:
    assert llm_fallback_detected(SubscriptionDraftResult(route="manager_only", draft_text="", safety_flags=("llm_fallback",)), "")
    assert llm_fallback_detected(SubscriptionDraftResult(route="manager_only", draft_text="", error="timeout"), "")
    assert not llm_fallback_detected(
        SubscriptionDraftResult(route="manager_only", draft_text="", error="authoritative_output_gate_blocked"),
        "Чтобы не ошибиться, передам вопрос менеджеру.",
    )


def test_zero_drafts_alert_missing_store_is_non_fatal(tmp_path) -> None:
    result = zero_drafts_alert(tmp_path / "missing.sqlite", now=datetime(2026, 6, 12, 12, tzinfo=timezone.utc))

    assert result["drafts_since"] is None
    assert result["alert"] is False
