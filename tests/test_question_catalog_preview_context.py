from __future__ import annotations

from mango_mvp.question_catalog.preview_context import build_question_catalog_channel_context


def test_question_catalog_channel_context_blocks_until_rop_approval() -> None:
    context = build_question_catalog_channel_context(
        {
            "approved_for_bot": False,
            "final_approved_answer": "Черновой ответ",
            "required_fact_keys": (),
        }
    )

    assert context["question_catalog_safe_to_use"] is False
    assert context["question_catalog_blocked_reason"] == "answer_not_approved_by_rop"


def test_question_catalog_channel_context_requires_fresh_facts() -> None:
    context = build_question_catalog_channel_context(
        {
            "approved_for_bot": True,
            "final_approved_answer": "Стоимость: {price}",
            "required_fact_keys": ("price.current",),
        },
        facts_fresh=False,
    )

    assert context["question_catalog_safe_to_use"] is False
    assert context["question_catalog_blocked_reason"] == "required_facts_not_confirmed_fresh"

    ready = build_question_catalog_channel_context(
        {
            "approved_for_bot": True,
            "final_approved_answer": "Стоимость: {price}",
            "required_fact_keys": ("price.current",),
        },
        facts_fresh=True,
    )
    assert ready["question_catalog_safe_to_use"] is True
