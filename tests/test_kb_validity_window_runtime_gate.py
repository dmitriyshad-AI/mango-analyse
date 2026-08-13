from datetime import date

from mango_mvp.channels.subscription_llm_parts.support import (
    _direct_path_valid_until_ok,
)
from mango_mvp.channels.telegram_pilot_context_builder import (
    _chunk_records,
    _usable_for_precise_answer,
)
from mango_mvp.knowledge_base.fact_registry import (
    evaluate_fact_freshness_sla,
    fact_runtime_time_ok,
    fact_valid_until_ok,
    fact_validity_window_ok,
)
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
        "freshness_check_date": "2026-08-13",
        "fact_type": "price",
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


def test_fact_validity_window_blocks_future_and_malformed_facts() -> None:
    today = date(2026, 8, 13)
    cases = (
        ("", "", True),
        ("2026-08-13", "2026-08-13", True),
        ("2026-08-14", "2027-05-31", False),
        ("2026-08-12", "2026-08-12", False),
        ("not-a-date", "2027-05-31", False),
        ("2026-08-13", "not-a-date", False),
    )
    for valid_from, valid_until, expected in cases:
        assert fact_validity_window_ok(
            valid_from=valid_from,
            valid_until=valid_until,
            today=today,
        ) is expected


def test_runtime_time_contract_rejects_stale_and_future_verification_dates() -> None:
    today = date(2026, 8, 13)
    assert fact_runtime_time_ok(_fact(freshness_check_date="2026-08-13"), today=today) is True
    assert fact_runtime_time_ok(_fact(freshness_check_date="2026-08-05"), today=today) is False
    assert fact_runtime_time_ok(_fact(freshness_check_date="2026-08-14"), today=today) is False
    assert evaluate_fact_freshness_sla(
        _fact(freshness_check_date="2026-08-14"), today=today
    ).within_sla is False


def test_runtime_time_contract_keeps_legacy_inline_fact_without_check_date() -> None:
    assert fact_runtime_time_ok(
        _fact(freshness_check_date="", valid_from="2026-08-13", valid_until="2026-08-13"),
        today=date(2026, 8, 13),
    ) is True


def test_expired_fact_is_removed_from_snippet_context() -> None:
    snapshot = {
        "chunks": [
            _fact(valid_until="2000-01-01", text="Старая цена."),
            _fact(fact_id="fact:current", valid_until="2999-01-01", text="Текущая цена."),
        ]
    }

    chunks = _chunk_records(snapshot, active_brand="foton")

    assert [chunk["fact_id"] for chunk in chunks] == ["fact:current"]


def test_future_fact_is_removed_from_snippet_context() -> None:
    snapshot = {
        "chunks": [
            _fact(valid_from="2999-01-01", valid_until="2999-12-31", text="Будущая цена."),
            _fact(fact_id="fact:current", valid_from="2000-01-01", valid_until="2999-12-31", text="Текущая цена."),
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
    assert chunk["freshness_check_date"] == "2026-08-13"
