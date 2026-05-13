from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from mango_mvp.question_catalog import (
    ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    SOURCE_CALL,
    CurrentFactSource,
    QuestionClass,
    QuestionItem,
    assert_question_catalog_safety_contract,
    question_catalog_contract_inventory,
    question_catalog_safety_contract,
)


NOW = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


def test_question_item_builds_stable_id_and_serializes() -> None:
    item = QuestionItem(
        tenant_id=" FOTON ",
        source_channel=SOURCE_CALL,
        source_ref="call:row=1",
        occurred_at=NOW,
        customer_text_redacted="Сколько стоит курс?",
        question_class_id="question_class:abc",
        intent="price",
        price_related=True,
        requires_dynamic_facts=True,
        dynamic_fact_types=("price",),
        fact_freshness_required="нужен свежий файл",
        answer_evidence_status=ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
    )

    payload = item.to_json_dict()

    assert item.tenant_id == "foton"
    assert item.question_item_id.startswith("question_item:")
    assert payload["occurred_at"] == NOW.isoformat()
    assert payload["dynamic_fact_types"] == ["price"]
    json.dumps(payload, ensure_ascii=False)


def test_question_item_rejects_unknown_source_and_naive_date() -> None:
    with pytest.raises(ValueError, match="unsupported source_channel"):
        QuestionItem(
            tenant_id="foton",
            source_channel="whatsapp",
            source_ref="x",
            customer_text_redacted="Вопрос?",
            question_class_id="question_class:x",
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        QuestionItem(
            tenant_id="foton",
            source_channel="call",
            source_ref="x",
            occurred_at=datetime(2026, 5, 13, 12, 0),
            customer_text_redacted="Вопрос?",
            question_class_id="question_class:x",
        )


def test_question_class_tracks_dynamic_fact_requirements() -> None:
    klass = QuestionClass(
        tenant_id="foton",
        canonical_question="стоимость / ЕГЭ / математика / 11 класс",
        narrow_scope="Цена ЕГЭ по математике для 11 класса.",
        class_key="intent=price|product=ege|subject=math|grade=11|format=any",
        examples_redacted=("Сколько стоит ЕГЭ по математике?",),
        count_total=3,
        count_calls=1,
        count_telegram=2,
        answer_status=ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
        required_fact_keys=("price.current",),
        bot_permission=BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    )

    payload = klass.to_json_dict()

    assert klass.question_class_id.startswith("question_class:")
    assert payload["count_total"] == 3
    assert payload["required_fact_keys"] == ["price.current"]
    json.dumps(payload, ensure_ascii=False)


def test_current_fact_source_and_safety_contract() -> None:
    source = CurrentFactSource(
        source_id="prices",
        fact_types=("price", "discount"),
        path="docs/prices.xlsx",
        usable_for_bot=False,
    )

    assert source.to_json_dict()["fact_types"] == ("price", "discount")
    assert_question_catalog_safety_contract(question_catalog_safety_contract())
    inventory = question_catalog_contract_inventory()
    assert "template_ready_needs_current_fact" in inventory["answer_statuses"]
