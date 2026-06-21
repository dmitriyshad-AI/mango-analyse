from __future__ import annotations

from mango_mvp.channels.subscription_llm_parts.direct_path import (
    BOT_SAFE_CRM_CONTEXT_ENV,
    _build_direct_path_prompt,
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


def _context(*, flag: bool, extra_items=None):
    items = [
        {
            "chunk_id": "chunk-foton",
            "customer_id": "customer:test-foton",
            "source_ref": "botsafe:customer:test-foton:foton",
            "chunk_type": "bot_safe_summary",
            "text": "Фотон: клиент уже спрашивал про онлайн-курс. Следующий шаг: отправить расписание.",
            "event_at": "2026-06-21T12:00:00+00:00",
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
            "relevance_tags": ["bot_safe", "structured", "unpk"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        },
        {
            "chunk_id": "chunk-unknown",
            "chunk_type": "bot_safe_summary",
            "text": "Без бренда: клиент ранее уточнял удобный формат.",
            "event_at": "2026-06-21T12:00:00+00:00",
            "relevance_tags": ["bot_safe", "structured", "unknown"],
            "allowed_for_bot": True,
            "requires_manager_review": False,
        },
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
