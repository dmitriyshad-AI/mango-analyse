from __future__ import annotations

from mango_mvp.channels.subscription_llm_parts.direct_path import (
    BOT_SAFE_CRM_CONTEXT_ENV,
    _build_direct_path_prompt,
    _direct_path_bot_safe_memory_prompt_text,
    _direct_path_bot_safe_context_prompt_block,
    _direct_path_bot_safe_context_items,
)


def test_bot_safe_context_prompt_block_is_default_off() -> None:
    context = _context(flag=False)

    prompt = _build_direct_path_prompt("Что дальше?", context=context, facts={"fact:1": "Безопасный факт"})

    assert "Безопасная выжимка клиента" not in prompt
    assert _direct_path_bot_safe_context_items(context) == ()


def test_bot_safe_context_prompt_filters_by_active_brand_and_strips_ids() -> None:
    context = _context(flag=True)

    prompt = _build_direct_path_prompt("Что дальше?", context=context, facts={"fact:1": "Безопасный факт"})

    assert "Безопасная выжимка клиента" in prompt
    assert "Фотон: клиент уже спрашивал про онлайн-курс" in prompt
    assert "Без бренда: клиент ранее уточнял удобный формат" in prompt
    assert "УНПК: клиент интересовался выездной школой" not in prompt
    assert "customer:test-foton" not in prompt
    assert "botsafe:" not in prompt
    assert "chunk-foton" not in prompt
    assert "статус следующего шага: active" in prompt
    assert "статус следующего шага: needs_manager_review" in prompt


def test_bot_safe_context_prompt_drops_pii_items() -> None:
    context = _context(
        flag=True,
        extra_items=[
            {
                "chunk_id": "chunk-pii",
                "chunk_type": "bot_safe_summary",
                "text": "Фотон: телефон +79991234567, почта edu@example.com.",
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            }
        ],
    )

    prompt = _build_direct_path_prompt("Что дальше?", context=context, facts={"fact:1": "Безопасный факт"})

    assert "+79991234567" not in prompt
    assert "edu@example.com" not in prompt
    assert "Фотон: клиент уже спрашивал про онлайн-курс" in prompt


def test_bot_safe_context_prompt_requires_known_active_brand() -> None:
    context = _context(flag=True)
    context["active_brand"] = "unknown"

    prompt = _build_direct_path_prompt("Что дальше?", context=context, facts={"fact:1": "Безопасный факт"})

    assert "Безопасная выжимка клиента" not in prompt
    assert _direct_path_bot_safe_context_items(context) == ()


def test_bot_safe_context_prompt_marks_unconfirmed_dated_memory() -> None:
    context = _context(flag=True)

    block = _direct_path_bot_safe_context_prompt_block(context)

    assert "следующий шаг НЕ подтверждён" in block
    assert "по прежним заметкам, актуальность уточню" in block
    assert "статус следующего шага: needs_manager_review" in block
    assert "Без бренда: клиент ранее уточнял удобный формат. (2026-06-20)" in block


def test_bot_safe_context_prompt_does_not_overhedge_active_memory() -> None:
    context = _context(flag=True, include_unknown=False)

    block = _direct_path_bot_safe_context_prompt_block(context)

    assert "статус следующего шага: active" in block
    assert "следующий шаг НЕ подтверждён" not in block
    assert "по прежним заметкам, актуальность уточню" not in block


def test_bot_safe_context_prompt_hides_exact_numbers_from_memory_but_keeps_fact_numbers() -> None:
    context = _context(
        flag=True,
        include_unknown=False,
        extra_items=[
            {
                "chunk_id": "chunk-schedule-memory",
                "chunk_type": "bot_safe_summary",
                "text": "Фотон: обсуждали расписание 2025/26, занятия 12:15-14:15, бюджет 94 500 ₽.",
                "event_at": "2026-06-19T12:00:00+00:00",
                "next_step_status": "active",
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            }
        ],
    )

    prompt = _build_direct_path_prompt(
        "Напомните расписание и цену?",
        context=context,
        facts={"fact:price": "Факт из базы: цена 47 250 ₽."},
    )

    assert "обсуждали расписание" in prompt
    assert "<точная деталь из памяти скрыта>" in prompt
    assert "2025/26" not in prompt
    assert "12:15-14:15" not in prompt
    assert "94 500" not in prompt
    assert "Факт из базы: цена 47 250 ₽." in prompt
    assert "Числа, даты, проценты, цены, расписание и адреса из этой выжимки НЕ называй клиенту как факт" in prompt


def test_bot_safe_memory_prompt_text_preserves_thread_without_exact_schedule() -> None:
    text = _direct_path_bot_safe_memory_prompt_text(
        "Обсуждали учебный год 2025/26, расписание 12:15-14:15 и цену 94 500 ₽."
    )

    assert "Обсуждали учебный год" in text
    assert "расписание" in text
    assert "2025/26" not in text
    assert "12:15-14:15" not in text
    assert "94 500" not in text


def _context(*, flag: bool, extra_items=None, include_unknown: bool = True):
    items = [
        {
            "chunk_id": "chunk-foton",
            "customer_id": "customer:test-foton",
            "source_ref": "botsafe:customer:test-foton:foton",
            "chunk_type": "bot_safe_summary",
            "text": "Фотон: клиент уже спрашивал про онлайн-курс. Следующий шаг: отправить расписание.",
            "event_at": "2026-06-21T12:00:00+00:00",
            "next_step_status": "active",
            "relevance_tags": ["bot_safe", "structured", "foton"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        },
        {
            "chunk_id": "chunk-unpk",
            "customer_id": "customer:test-foton",
            "source_ref": "botsafe:customer:test-foton:unpk",
            "chunk_type": "bot_safe_summary",
            "text": "УНПК: клиент интересовался выездной школой.",
            "event_at": "2026-06-21T12:00:00+00:00",
            "next_step_status": "active",
            "relevance_tags": ["bot_safe", "structured", "unpk"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        },
        *((
            {
                "chunk_id": "chunk-unknown",
                "chunk_type": "bot_safe_summary",
                "text": "Без бренда: клиент ранее уточнял удобный формат.",
                "event_at": "2026-06-20T12:00:00+00:00",
                "next_step_status": "needs_manager_review",
                "relevance_tags": ["bot_safe", "structured", "unknown"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            },
        ) if include_unknown else ()),
        *(extra_items or []),
    ]
    return {
        "active_brand": "foton",
        BOT_SAFE_CRM_CONTEXT_ENV: flag,
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
