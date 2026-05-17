from __future__ import annotations

from mango_mvp.channels.telegram_pilot_context_builder import (
    NO_KNOWLEDGE_SNAPSHOT_VERSION,
    build_telegram_pilot_context,
)


def test_builder_wires_fresh_snapshot_into_pilot_context() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_night_20260517_v1",
        "sources": [
            {
                "source_id": "source:price_2026",
                "title": "Стоимость 2026/2027",
                "fact_types": ["price"],
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
            }
        ],
        "facts": [
            {
                "fact_id": "fact:price_grade_10",
                "fact_type": "price",
                "client_safe_text": "Стоимость курса для 10 класса: 120 000 рублей.",
                "source_id": "source:price_2026",
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            }
        ],
        "chunks": [
            {
                "chunk_id": "chunk:price_10",
                "source_id": "source:price_2026",
                "title": "Стоимость обучения",
                "text": "Стоимость курса для 10 класса: 120 000 рублей. Источник проверен на 2026/2027 год.",
                "fact_types": ["price"],
                "freshness_status": "fresh_verified",
            }
        ],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит курс для 10 класса?",
        theme={"topic_id": "theme:001_pricing", "topic_name": "Стоимость"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert payload["knowledge_base_version"] == "kb_night_20260517_v1"
    assert payload["facts_context"]["knowledge_base_version"] == "kb_night_20260517_v1"
    assert payload["facts_context"]["fresh"] is True
    assert payload["facts_context"]["facts_missing"] is False
    assert payload["confirmed_facts"]["fact:price_grade_10"] == "Стоимость курса для 10 класса: 120 000 рублей."
    assert payload["knowledge_snippets"]
    assert "Стоимость обучения" in payload["knowledge_snippets"][0]
    assert "missing_facts" not in payload


def test_builder_uses_safe_fallback_when_snapshot_missing() -> None:
    context = build_telegram_pilot_context(
        "Какая цена?",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=None,
    )
    payload = context.to_prompt_context()

    assert payload["knowledge_base_version"] == NO_KNOWLEDGE_SNAPSHOT_VERSION
    assert payload["facts_context"]["snapshot_found"] is False
    assert payload["facts_context"]["fresh"] is False
    assert payload["facts_context"]["facts_missing"] is True
    assert payload["missing_facts"] == ["prices.current"]
    assert "knowledge_snapshot_missing" in payload["context_warnings"]
    assert "precise_answer_blocked" in payload["context_warnings"]
    assert "knowledge_snippets" not in payload


def test_builder_keeps_metadata_only_price_as_missing_fact() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_night_20260517_v1",
        "sources": [
            {
                "source_id": "source:price_metadata_only",
                "title": "Стоимость 2026/2027",
                "fact_types": ["price"],
                "freshness_status": "metadata_only",
                "usable_for_precise_answer": False,
            }
        ],
        "chunks": [
            {
                "chunk_id": "chunk:price_exact_metadata_only",
                "source_id": "source:price_metadata_only",
                "title": "Стоимость обучения",
                "text": "Стоимость курса 120 000 рублей, но документ не прочитан и не подтвержден.",
                "fact_types": ["price"],
                "freshness_status": "metadata_only",
            },
            {
                "chunk_id": "chunk:price_warning",
                "source_id": "source:price_metadata_only",
                "title": "Стоимость обучения требует проверки",
                "text": "Документ по стоимости зарегистрирован, но точные цены требуют проверки менеджером.",
                "fact_types": ["price"],
                "freshness_status": "metadata_only",
            }
        ],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит обучение?",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert payload["facts_context"]["snapshot_found"] is True
    assert payload["facts_context"]["fresh"] is False
    assert payload["facts_context"]["facts_missing"] is True
    assert payload["missing_facts"] == ["prices.current"]
    assert "facts_stale" in payload["context_warnings"]
    assert "Стоимость обучения требует проверки" in payload["knowledge_snippets"][0]
    assert "120 000" not in " ".join(payload["knowledge_snippets"])
    assert "confirmed_facts" not in payload
