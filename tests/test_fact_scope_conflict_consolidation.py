from __future__ import annotations

import pytest

from mango_mvp.channels.fact_scope_spec import fact_scope_conflicts_with_query_text
from mango_mvp.channels.subscription_llm_parts.direct_path import (
    DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV,
    _direct_path_keyword_fact_pack_from_records,
    _direct_path_wide_fact_pack,
)
from mango_mvp.channels.subscription_llm_parts.reliable_answerer import build_answer_coverage_plan


CITY_QUERY = "Где проходит городской лагерь?"
LVSH_QUERY = "Где проходит выездной лагерь ЛВШ в Менделеево?"
CITY_TEXT = "Адрес городского лагеря: Москва, ул. Правды, 24. Без проживания."
LVSH_TEXT = "Адрес выездного лагеря ЛВШ: Менделеево, с проживанием."


def fact(key: str, text: str, fact_type: str, program_kind: str) -> dict[str, object]:
    return {
        "fact_id": key,
        "fact_key": key,
        "fact_type": fact_type,
        "fact_types": [fact_type],
        "program_kind": program_kind,
        "client_safe_text": text,
    }


CITY_FACT = fact("loc.city", CITY_TEXT, "camp_city", "camp_city")
LVSH_FACT = fact("loc.lvsh", LVSH_TEXT, "camp_lvsh", "camp_lvsh")


def keyword_pack(
    query: str,
    records: list[dict[str, object]],
    *,
    legacy: dict[str, str] | None = None,
    relevance_fallback: bool = False,
) -> dict[str, object]:
    context = {DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV: "1"} if relevance_fallback else None
    return dict(
        _direct_path_keyword_fact_pack_from_records(
            records,
            legacy=legacy or {},
            active_brand="foton",
            context=context,
            client_message=query,
            max_facts=10,
            max_chars=5000,
        )
    )


def coverage_plan(query: str, records: list[dict[str, object]]) -> dict[str, object]:
    facts = {str(item["fact_key"]): str(item["client_safe_text"]) for item in records}
    metadata = {
        str(item["fact_key"]): {
            "fact_type": item["fact_type"],
            "fact_types": item["fact_types"],
            "program_kind": item["program_kind"],
            "venue": "lvsh_mendeleevo" if item["program_kind"] == "camp_lvsh" else "moscow_regular",
        }
        for item in records
    }
    return dict(
        build_answer_coverage_plan(
            query,
            fact_pack={
                "facts": facts,
                "exact_keys": list(facts),
                "adjacent_keys": [],
                "fact_metadata": metadata,
            },
            context={"TELEGRAM_RELIABLE_ANSWERER_STEP1": "1"},
        )
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    ((CITY_QUERY, CITY_TEXT), (LVSH_QUERY, LVSH_TEXT)),
)
def test_fact_selectors_keep_only_the_requested_product(query: str, expected: str) -> None:
    pack = keyword_pack(query, [CITY_FACT, LVSH_FACT])
    assert list(dict(pack["facts"]).values()) == [expected]

    plan = coverage_plan(query, [CITY_FACT, LVSH_FACT])
    covered = {key for item in plan["covered_facets"] for key in item["fact_keys"]}
    assert covered == ({"loc.city"} if query == CITY_QUERY else {"loc.lvsh"})


@pytest.mark.parametrize("relevance_fallback", (False, True))
def test_wrong_only_fact_never_returns_through_fallback(relevance_fallback: bool) -> None:
    pack = keyword_pack(
        CITY_QUERY,
        [LVSH_FACT],
        legacy={"legacy.lvsh": LVSH_TEXT},
        relevance_fallback=relevance_fallback,
    )
    assert pack["facts"] == {}


def test_structured_product_type_blocks_neutral_wrong_fact() -> None:
    wrong = fact("loc.lvsh", "Адрес площадки: ул. Лесная, 1.", "camp_lvsh", "camp_lvsh")
    assert keyword_pack(CITY_QUERY, [wrong])["facts"] == {}
    assert coverage_plan(CITY_QUERY, [wrong])["covered_facets"] == []


@pytest.mark.parametrize("nested", (False, True))
def test_program_kind_blocks_neutral_wrong_fact_at_supported_metadata_levels(nested: bool) -> None:
    wrong = fact("loc.lvsh", "Адрес площадки: ул. Лесная, 1.", "address", "")
    if nested:
        wrong["metadata"] = {"program_kind": "camp_lvsh"}
    else:
        wrong["program_kind"] = "camp_lvsh"
    assert keyword_pack(CITY_QUERY, [wrong])["facts"] == {}


def test_legacy_key_and_empty_snapshot_cannot_bypass_scope_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from mango_mvp.channels.subscription_llm_parts import direct_path

    monkeypatch.setattr(direct_path, "_direct_path_load_snapshot", lambda _path: {})
    monkeypatch.setattr(
        direct_path,
        "_direct_path_legacy_context_fact_items",
        lambda _context, limit: {"camp_lvsh.address": "Адрес площадки: ул. Лесная, 1."},
    )
    pack = _direct_path_wide_fact_pack(None, client_message=CITY_QUERY)
    assert pack["facts"] == {}


def test_structured_legacy_metadata_survives_until_scope_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from mango_mvp.channels.subscription_llm_parts import direct_path

    monkeypatch.setattr(direct_path, "_direct_path_load_snapshot", lambda _path: {})
    context = {
        "active_brand": "foton",
        "conversation_intent_plan": {"fact_scope": "city_day_camp"},
        "confirmed_facts": {
            "locations.current": {
                "client_safe_text": "Адрес площадки: ул. Лесная, 1.",
                "allowed_for_client_answer": True,
                "brand": "foton",
                "metadata": {"program_kind": "camp_lvsh"},
            }
        },
    }
    pack = _direct_path_wide_fact_pack(context, client_message="А адрес какой?")
    assert pack["facts"] == {}


def test_short_followup_uses_held_fact_scope() -> None:
    context = {"conversation_intent_plan": {"fact_scope": "city_day_camp"}}
    pack = dict(
        _direct_path_keyword_fact_pack_from_records(
            [CITY_FACT, LVSH_FACT],
            legacy={},
            active_brand="foton",
            context=context,
            client_message="А адрес какой?",
            max_facts=10,
            max_chars=5000,
        )
    )
    assert list(dict(pack["facts"]).values()) == [CITY_TEXT]


def test_comparison_between_products_is_not_answered_by_one_side() -> None:
    query = "Чем городской лагерь отличается от выездного ЛВШ?"
    assert keyword_pack(query, [CITY_FACT, LVSH_FACT])["facts"] == {}


def test_unrelated_query_does_not_block_scoped_fact() -> None:
    assert fact_scope_conflicts_with_query_text(CITY_TEXT, "Сколько стоит обучение?") is False


def test_declared_compatible_neighbor_remains_allowed() -> None:
    fact_text = "Фрагмент занятия онлайн-формата доступен для пробы."
    query = "Хочу записаться на очное пробное занятие."
    assert fact_scope_conflicts_with_query_text(fact_text, query) is False
