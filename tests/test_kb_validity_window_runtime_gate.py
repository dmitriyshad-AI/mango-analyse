from datetime import date

from mango_mvp.channels.subscription_llm_parts.support import (
    _direct_path_valid_until_ok,
)
from mango_mvp.channels.telegram_pilot_context_builder import (
    _chunk_records,
    _usable_for_precise_answer,
)
from mango_mvp.knowledge_base.fact_registry import fact_valid_until_ok
from scripts.build_kb_release_v3_from_claude_handoff import build_chunks


def _fact(**overrides):
    return {
        "fact_id": "fact:test",
        "brand": "foton",
        "title": "Тестовый факт",
        "client_safe_text": "Подтверждённый текст.",
        "freshness_status": "fresh",
        "usable_for_precise_answer": True,
        "allowed_for_client_answer": True,
        "requires_manager_confirmation": False,
        "forbidden_for_client": False,
        **overrides,
    }


def test_fact_valid_until_contract_and_direct_path_alias_match() -> None:
    today = date(2026, 7, 28)
    values = {
        "": True,
        "not-a-date": False,
        "2026-07-27": False,
        "2026-07-28": True,
        "2026-07-29T12:00:00+03:00": True,
    }

    for value, expected in values.items():
        assert fact_valid_until_ok(value, today=today) is expected
        assert _direct_path_valid_until_ok(value, today=today) is expected


def test_expired_fact_is_removed_from_snippet_context() -> None:
    snapshot = {
        "chunks": [
            _fact(valid_until="2000-01-01", text="Старая цена."),
            _fact(fact_id="fact:current", valid_until="2999-01-01", text="Текущая цена."),
        ]
    }

    chunks = _chunk_records(snapshot, active_brand="foton")

    assert [chunk["fact_id"] for chunk in chunks] == ["fact:current"]


def test_expired_fact_is_not_usable_as_precise_answer() -> None:
    assert _usable_for_precise_answer(
        _fact(valid_until="2000-01-01"),
        active_brand="foton",
    ) is False
    assert _usable_for_precise_answer(
        _fact(valid_until="2999-01-01"),
        active_brand="foton",
    ) is True


def test_fact_without_expiry_keeps_previous_behavior() -> None:
    assert _chunk_records(
        {"chunks": [_fact(text="Бессрочный факт.")]},
        active_brand="foton",
    )
    assert _usable_for_precise_answer(_fact(), active_brand="foton") is True


def test_built_chunk_mirrors_fact_validity_window() -> None:
    chunk = build_chunks(
        [
            _fact(
                valid_from="2026-07-01",
                valid_until="2026-07-31",
                fact_types=["price"],
            )
        ]
    )[0]

    assert chunk["valid_from"] == "2026-07-01"
    assert chunk["valid_until"] == "2026-07-31"
