from __future__ import annotations

import pytest

from mango_mvp.channels.subscription_llm_parts.direct_path import (
    BOT_SAFE_CRM_CONTEXT_ENV,
    _build_direct_path_prompt,
    _direct_path_bot_safe_context_items,
    _direct_path_bot_safe_context_prompt_block,
    _direct_path_known_slots_next_step_prompt_enabled,
    _direct_path_select_gold_real_examples,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    BOT_GOLD_REAL_ENV,
    DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
)


def test_known_slots_next_step_prompt_flag_default_off_pilot_gold_on() -> None:
    assert _direct_path_known_slots_next_step_prompt_enabled({}) is False
    assert _direct_path_known_slots_next_step_prompt_enabled({DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}) is True
    assert (
        _direct_path_known_slots_next_step_prompt_enabled(
            {
                DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
                DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV: "0",
            }
        )
        is False
    )


def test_known_slots_prompt_off_does_not_change_direct_prompt() -> None:
    context = {
        "active_brand": "foton",
        "conversation_intent_plan": {
            "known_slots": {"grade": "6", "subject": "программирование"},
            "do_not_reask_slots": ["grade", "subject"],
        },
    }

    implicit_off = _build_direct_path_prompt("Интересует курс.", context=context)
    explicit_off = _build_direct_path_prompt(
        "Интересует курс.",
        context={**context, DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV: "0"},
    )

    assert implicit_off == explicit_off
    assert "Приоритет уже известного контекста" not in implicit_off
    assert "эти параметры клиент уже назвал" not in implicit_off


def test_known_slots_prompt_passes_planner_known_and_do_not_reask_without_pii() -> None:
    context = {
        "active_brand": "unpk",
        DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV: "1",
        BOT_GOLD_REAL_ENV: "1",
        "known_slots": {"client_name": "Ирина", "phone": "+7 999 123-45-67"},
        "conversation_intent_plan": {
            "known_slots": {"grade": "6", "subject": "программирование"},
            "do_not_reask_slots": ["grade", "subject", "phone"],
        },
    }

    gold_examples = _direct_path_select_gold_real_examples("Интересуют занятия.", context=context, active_brand="unpk")
    assert gold_examples

    prompt = _build_direct_path_prompt("Интересуют занятия.", context=context, gold_examples=gold_examples)

    assert "Приоритет уже известного контекста" in prompt
    assert "эти параметры клиент уже назвал — НЕ переспрашивай: класс: 6; предмет: программирование." in prompt
    assert "анкета — ошибка" in prompt
    assert "Ирина" not in prompt
    assert "+7 999" not in prompt
    assert prompt.index("Приоритет уже известного контекста") < prompt.index("Живые образцы менеджерского стиля")


def test_unknown_class_still_allows_one_qualification_question() -> None:
    prompt = _build_direct_path_prompt(
        "Хочу подобрать курс.",
        context={"active_brand": "foton", DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV: "1"},
    )

    assert "эти параметры клиент уже назвал" not in prompt
    assert "Если класс/предмет/формат действительно неизвестны" in prompt
    assert "допустим один короткий уточняющий вопрос" in prompt


def test_active_next_step_priority_is_above_gold_and_uses_bot_safe_context() -> None:
    context = _bot_safe_context(
        flag_value="1",
        gold_value="1",
    )

    gold_examples = _direct_path_select_gold_real_examples("Что дальше?", context=context, active_brand="foton")
    assert gold_examples

    prompt = _build_direct_path_prompt(
        "Что дальше?",
        context=context,
        facts={"fact:1": "Безопасный факт."},
        gold_examples=gold_examples,
    )

    assert "Если статус next_step active" in prompt
    assert "НЕ задавай квалифицирующих вопросов" in prompt
    assert "Безопасная выжимка клиента" in prompt
    assert "Следующий шаг: отправить расписание" in prompt
    assert "статус следующего шага: active" in prompt
    assert prompt.index("Приоритет уже известного контекста") < prompt.index("Живые образцы менеджерского стиля")


def test_pilot_gold_keeps_memory_off_without_extra_context_flag() -> None:
    context = _bot_safe_context(flag_value=None, bot_safe_value=None)
    context[DIRECT_PATH_PILOT_CONFIG_ENV] = DIRECT_PATH_PILOT_CONFIG_VERSION

    prompt = _build_direct_path_prompt("Что дальше?", context=context, facts={"fact:1": "Безопасный факт."})

    assert "Приоритет уже известного контекста" in prompt
    assert "Если статус next_step active" not in prompt
    assert "Безопасная выжимка клиента" not in prompt
    assert "Следующий шаг: отправить расписание" not in prompt


def test_bot_safe_context_prompt_drops_pii_and_keeps_fact_numbers() -> None:
    context = _bot_safe_context(
        flag_value="1",
        extra_items=[
            {
                "chunk_type": "bot_safe_summary",
                "text": "Менеджер Иванова Мария обсудила ребёнка Пётр и телефон +79991234567.",
                "event_at": "2026-06-21T12:00:00+00:00",
                "next_step_status": "active",
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            },
            {
                "chunk_type": "bot_safe_summary",
                "text": "Фотон: обсуждали расписание 2025/26, занятия 12:15-14:15, бюджет 94 500 ₽.",
                "event_at": "2026-06-20T12:00:00+00:00",
                "next_step_status": "active",
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            },
        ],
    )

    prompt = _build_direct_path_prompt(
        "Напомните расписание и цену?",
        context=context,
        facts={"fact:price": "Факт из базы: цена 47 250 ₽."},
    )

    assert "Иванова" not in prompt
    assert "Мария" not in prompt
    assert "Пётр" not in prompt
    assert "+79991234567" not in prompt
    assert "обсуждали расписание" in prompt
    assert "<точная деталь из памяти скрыта>" in prompt
    assert "2025/26" not in prompt
    assert "12:15-14:15" not in prompt
    assert "94 500" not in prompt
    assert "Факт из базы: цена 47 250 ₽." in prompt
    assert len(_direct_path_bot_safe_context_items(context)) == 2


def test_questionnaire_gold_is_suppressed_when_qualification_slots_known() -> None:
    off_examples = _direct_path_select_gold_real_examples(
        "Расскажите подробно какие есть условия обучения?",
        context={"active_brand": "unpk", BOT_GOLD_REAL_ENV: "1"},
        active_brand="unpk",
    )
    on_examples = _direct_path_select_gold_real_examples(
        "Расскажите подробно какие есть условия обучения?",
        context={
            "active_brand": "unpk",
            BOT_GOLD_REAL_ENV: "1",
            DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV: "1",
            "conversation_intent_plan": {
                "known_slots": {"grade": "6", "subject": "программирование", "format": "очно"},
                "do_not_reask_slots": ["grade", "subject", "format"],
            },
        },
        active_brand="unpk",
    )

    assert any(item["topic"] == "course_pick" for item in off_examples)
    assert all(item["topic"] != "course_pick" for item in on_examples)


@pytest.mark.parametrize("case_id", ["07", "13", "15", "21", "17"])
def test_measurement_cases_keep_active_step_priority_over_questionnaire_gold(case_id: str) -> None:
    context = _bot_safe_context(
        flag_value="1",
        gold_value="1",
        summary=f"Фотон: клиент уже назвал класс и предмет. Следующий шаг: отправить материалы по кейсу {case_id}.",
    )

    prompt = _build_direct_path_prompt("Спасибо, что дальше?", context=context, facts={"fact:1": "Безопасный факт."})
    examples = _direct_path_select_gold_real_examples("Спасибо, что дальше?", context=context, active_brand="foton")

    assert "Если статус next_step active" in prompt
    assert "Следующий шаг: отправить материалы по кейсу" in prompt
    assert all(item["topic"] != "course_pick" for item in examples)


def test_bot_safe_context_prompt_is_default_off_even_when_context_present() -> None:
    context = _bot_safe_context(flag_value="1", bot_safe_value="0")

    prompt = _build_direct_path_prompt("Что дальше?", context=context, facts={"fact:1": "Безопасный факт."})

    assert _direct_path_bot_safe_context_prompt_block(context) == ""
    assert "Безопасная выжимка клиента" not in prompt
    assert "Следующий шаг: отправить расписание" not in prompt


def _bot_safe_context(
    *,
    flag_value: str | None = "1",
    bot_safe_value: str | None = "1",
    gold_value: str = "0",
    summary: str = "Фотон: клиент уже спрашивал про онлайн-курс. Следующий шаг: отправить расписание.",
    extra_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    items = [
        {
            "chunk_type": "bot_safe_summary",
            "text": summary,
            "event_at": "2026-06-21T12:00:00+00:00",
            "next_step_status": "active",
            "relevance_tags": ["bot_safe", "structured", "foton"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        },
        {
            "chunk_type": "bot_safe_summary",
            "text": "УНПК: клиент интересовался выездной школой.",
            "event_at": "2026-06-21T12:00:00+00:00",
            "next_step_status": "active",
            "relevance_tags": ["bot_safe", "structured", "unpk"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        },
        *(extra_items or []),
    ]
    context = {
        "active_brand": "foton",
        BOT_GOLD_REAL_ENV: gold_value,
        "timeline_context": {
            "source": "customer_timeline_bot_context",
            "found": True,
            "bot_context": {
                "allowed_only": True,
                "items": items,
            },
        },
        "recent_messages": [],
    }
    if flag_value is not None:
        context[DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV] = flag_value
    if bot_safe_value is not None:
        context[BOT_SAFE_CRM_CONTEXT_ENV] = bot_safe_value
    return context
