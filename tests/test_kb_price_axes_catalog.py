from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.channels.fact_retrieval import select_confirmed_facts
from mango_mvp.knowledge_base.price_axes_catalog import (
    _extract_grade,
    build_price_axes_catalog,
    extract_price_query_axes,
    normalize_subject,
    select_price,
    select_price_fact_for_query,
    select_price_result_for_query,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"


def _facts() -> list[dict[str, object]]:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return list(snapshot.get("facts") or [])


def _catalog() -> dict[str, object]:
    return build_price_axes_catalog(_facts())


def _entries(catalog: dict[str, object]) -> list[dict[str, object]]:
    return list(catalog.get("entries") or [])


def test_catalog_derives_grade_axes_from_classes_not_fact_key_dates() -> None:
    catalog = _catalog()
    foton_online_year = [
        entry
        for entry in _entries(catalog)
        if entry.get("brand") == "foton"
        and entry.get("format") == "online"
        and entry.get("period") == "year"
        and entry.get("amount") == 47250
        and entry.get("product_code") == "regular_course"
    ]

    assert foton_online_year
    entry = foton_online_year[0]
    assert entry["grade_min"] == 5
    assert entry["grade_max"] == 11
    assert entry["grade_values"] == [5, 6, 7, 8, 9, 10, 11]
    assert 2026 not in entry["grade_values"]
    assert entry["subjects"] == ["math", "physics", "informatics", "russian", "ai"]


def test_price_selector_respects_current_russian_product_line() -> None:
    catalog = _catalog()

    assert normalize_subject("russian") == "russian"
    for grade in (3, 8):
        result = select_price(
            catalog,
            brand="foton",
            grade=grade,
            subject="russian",
            format="online",
            period="semester",
        )
        assert result["status"] == "not_found"
        assert result["reason"] == "subject_not_offered"

    allowed = select_price(
        catalog,
        brand="foton",
        grade=9,
        subject="russian",
        format="online",
        period="semester",
    )
    assert allowed["status"] == "exact"
    assert allowed["entry"]["amount"] == 29750

    unpk = select_price(
        catalog,
        brand="unpk",
        grade=9,
        subject="russian",
        format="online",
        period="semester",
        schedule="weekend",
    )
    assert unpk["status"] == "not_found"
    assert unpk["reason"] == "subject_not_offered"


def test_catalog_excludes_superseded_unpk_weekday_prices() -> None:
    catalog = _catalog()
    entries = _entries(catalog)
    unpk = [
        entry
        for entry in entries
        if entry.get("brand") == "unpk" and entry.get("source_kind") == "unpk_online_kc_source_price"
    ]

    assert len(unpk) == 2
    amounts = {(entry["classes"], entry["schedule"], entry["period"], entry["amount"]) for entry in unpk}
    assert ("5-11", "weekend", "semester", 37000) in amounts
    assert ("5-11", "weekend", "year", 59000) in amounts
    assert not any(entry["schedule"] == "weekday" for entry in unpk)
    assert all(entry.get("client_safe_text") for entry in unpk)
    assert all("Фотон" not in str(entry.get("client_safe_text")) for entry in unpk)


def test_catalog_splits_m9_m11_tariffs_into_atomic_positions() -> None:
    catalog = _catalog()
    tariffs = [entry for entry in _entries(catalog) if entry.get("source_kind") == "foton_m9_m11_tariff_price"]

    assert len(tariffs) == 8
    by_product_tariff = {(entry["product_code"], entry["tariff_id"]): entry for entry in tariffs}
    assert by_product_tariff[("m9", "base")]["amount"] == 18900
    assert by_product_tariff[("m9", "standard")]["amount"] == 47250
    assert by_product_tariff[("m9", "advanced")]["amount"] == 59900
    assert by_product_tariff[("m9", "full_immersion")]["amount"] == 94500
    assert by_product_tariff[("m11", "standard")]["structured_value"]["grade_values"] == [11]
    assert by_product_tariff[("m11", "full_immersion")]["tariff_includes"]


def test_catalog_marks_ranges_and_empty_client_safe_text_as_not_final_prices() -> None:
    catalog = _catalog()
    issues = list(catalog.get("issues") or [])

    assert any(issue.get("issue") == "range_not_final_price" and issue.get("amount_min") == 29900 for issue in issues)
    assert all(entry.get("client_safe_text") for entry in _entries(catalog))
    assert not [
        entry
        for entry in _entries(catalog)
        if entry.get("brand") == "foton"
        and entry.get("format") == "online"
        and entry.get("period") == "year"
        and entry.get("grade_min") == 3
        and entry.get("grade_max") == 4
    ]


def test_selector_returns_exact_regular_price_without_subject_dependency() -> None:
    result = select_price(_catalog(), brand="Фотон", grade=9, subject="математика", format="онлайн", period="год")

    assert result["status"] == "exact"
    assert result["entry"]["amount"] == 47250
    assert result["entry"]["subjects"] == ["math", "physics", "informatics", "russian", "ai"]


def test_selector_returns_only_current_unpk_online_price() -> None:
    result = select_price(_catalog(), brand="УНПК", grade=9, subject="математика", format="онлайн", period="год")

    assert result["status"] == "exact"
    assert result["entry"]["amount"] == 59000
    assert result["entry"]["schedule"] == "weekend"


def test_selector_can_pick_unpk_weekday_price_by_subject() -> None:
    result = select_price(_catalog(), brand="УНПК", grade=9, subject="информатика", format="онлайн", period="год")

    assert result["status"] == "not_found"


def test_selector_respects_explicit_unpk_weekday_schedule() -> None:
    result = select_price(
        _catalog(),
        brand="УНПК",
        grade=9,
        subject="математика",
        format="онлайн",
        period="год",
        schedule="будни",
    )

    assert result["status"] == "not_found"


def test_catalog_never_resurrects_disallowed_or_expired_fact(monkeypatch) -> None:
    monkeypatch.setenv("MANGO_EVALUATION_DATE", "2026-07-28")
    base = {
        "fact_id": "fact:old-price",
        "fact_key": "prices_regular_2026_27.offline_5_11.year",
        "fact_type": "price",
        "brand": "foton",
        "allowed_for_client_answer": True,
        "usable_for_precise_answer": True,
        "freshness_status": "document_verified",
        "valid_until": "2026-07-01",
        "client_safe_text": "Фотон: год — 74 500 ₽.",
        "structured_value": {
            "amount": 74500,
            "classes": "5-11",
            "format": "offline",
            "period": "year",
        },
    }
    assert build_price_axes_catalog([base])["entries"] == []
    assert build_price_axes_catalog([{**base, "valid_until": "2026-12-31", "allowed_for_client_answer": False}])["entries"] == []


def test_selector_does_not_reuse_weekend_price_for_missing_weekday_grade() -> None:
    result = select_price(
        _catalog(),
        brand="УНПК",
        grade=10,
        subject="",
        format="онлайн",
        period="семестр",
        schedule="будни",
    )

    assert result["status"] == "not_found"
    assert result["reason"] == "no_exact_price_for_axes"
    assert select_price_fact_for_query(
        _facts(),
        active_brand="unpk",
        query="10 класс онлайн по будням за семестр сколько стоит?",
    ) is None


def test_selector_can_pick_m9_tariff_only_when_product_and_tariff_are_explicit() -> None:
    regular = select_price(_catalog(), brand="Фотон", grade=9, subject="математика", format="онлайн", period="год")
    tariff = select_price(
        _catalog(),
        brand="Фотон",
        grade=9,
        subject="математика",
        format="онлайн",
        period="год",
        product_code="М9",
        tariff_id="стандарт",
    )

    assert regular["status"] == "exact"
    assert regular["entry"]["product_code"] == "regular_course"
    assert tariff["status"] == "exact"
    assert tariff["entry"]["product_code"] == "m9"
    assert tariff["entry"]["tariff_id"] == "standard"
    assert tariff["entry"]["amount"] == 47250


def test_fact_retrieval_price_axis_selector_is_flagged_and_brand_safe(monkeypatch) -> None:
    facts = _facts()
    query = "Сколько стоит математика для 9 класса онлайн за год?"

    monkeypatch.delenv("TELEGRAM_PRICE_AXES_SELECTOR", raising=False)
    monkeypatch.delenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", raising=False)
    off = select_confirmed_facts(facts, active_brand="foton", required_fact_keys=["prices.current"], query=query, k=3)
    assert not str((off[0].get("__fact") or off[0]).get("fact_id") or "").startswith("fact:v3:price_axes_selector")

    monkeypatch.setenv("TELEGRAM_PRICE_AXES_SELECTOR", "1")
    on = select_confirmed_facts(facts, active_brand="foton", required_fact_keys=["prices.current"], query=query, k=3)
    first = on[0].get("__fact")
    assert isinstance(first, dict)
    assert str(first.get("fact_id")).startswith("fact:v3:price_axes_selector")
    assert first["brand"] == "foton"
    assert "47 250" in first["client_safe_text"]
    assert "УНПК" not in first["client_safe_text"]


def test_fact_retrieval_price_axis_selector_is_enabled_by_pilot_profile(monkeypatch) -> None:
    facts = _facts()
    query = "Сколько стоит математика для 9 класса онлайн за год?"

    monkeypatch.delenv("TELEGRAM_PRICE_AXES_SELECTOR", raising=False)
    monkeypatch.setenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")
    on = select_confirmed_facts(facts, active_brand="foton", required_fact_keys=["prices.current"], query=query, k=3)
    first = on[0].get("__fact")

    assert isinstance(first, dict)
    assert str(first.get("fact_id")).startswith("fact:v3:price_axes_selector")
    assert "47 250" in first["client_safe_text"]


def test_fact_retrieval_price_axis_selector_explicit_zero_overrides_pilot_profile(monkeypatch) -> None:
    facts = _facts()
    query = "Сколько стоит математика для 9 класса онлайн за год?"

    monkeypatch.setenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")
    monkeypatch.setenv("TELEGRAM_PRICE_AXES_SELECTOR", "0")
    off = select_confirmed_facts(facts, active_brand="foton", required_fact_keys=["prices.current"], query=query, k=3)

    assert not str((off[0].get("__fact") or off[0]).get("fact_id") or "").startswith("fact:v3:price_axes_selector")


def test_fact_retrieval_clean_defer_drops_irrelevant_facts_on_dead_end(monkeypatch) -> None:
    facts = _facts()
    query = "10 класс онлайн по будням за семестр сколько стоит?"

    monkeypatch.setenv("TELEGRAM_PRICE_AXES_SELECTOR", "1")
    monkeypatch.delenv("TELEGRAM_PRICE_AXES_CLEAN_DEFER", raising=False)
    off = select_confirmed_facts(facts, active_brand="unpk", required_fact_keys=["prices.current"], query=query, k=5)
    assert off

    monkeypatch.setenv("TELEGRAM_PRICE_AXES_CLEAN_DEFER", "1")
    on = select_confirmed_facts(facts, active_brand="unpk", required_fact_keys=["prices.current"], query=query, k=5)
    assert on == []


def test_fact_retrieval_clean_defer_is_enabled_by_pilot_profile(monkeypatch) -> None:
    facts = _facts()
    query = "10 класс онлайн по будням за семестр сколько стоит?"

    monkeypatch.delenv("TELEGRAM_PRICE_AXES_SELECTOR", raising=False)
    monkeypatch.delenv("TELEGRAM_PRICE_AXES_CLEAN_DEFER", raising=False)
    monkeypatch.setenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")
    on = select_confirmed_facts(facts, active_brand="unpk", required_fact_keys=["prices.current"], query=query, k=5)

    assert on == []


def test_fact_retrieval_clean_defer_keeps_valid_price_facts(monkeypatch) -> None:
    facts = _facts()
    query = "УНПК онлайн, математика 5 класс по выходным, семестр сколько?"

    monkeypatch.setenv("TELEGRAM_PRICE_AXES_SELECTOR", "1")
    monkeypatch.setenv("TELEGRAM_PRICE_AXES_CLEAN_DEFER", "1")
    on = select_confirmed_facts(facts, active_brand="unpk", required_fact_keys=["prices.current"], query=query, k=5)
    first = on[0].get("__fact")

    assert isinstance(first, dict)
    assert str(first.get("fact_id")).startswith("fact:v3:price_axes_selector")
    assert first["brand"] == "unpk"
    assert "37 000" in first["client_safe_text"]


# --- БЛОК 7 (2026-07-25): _extract_grade() price-risk audit -----------------
#
# Verify-only per ТЗ (no regex->LLM migration). `_extract_grade` sits on the
# live runtime price path: fact_retrieval.select_confirmed_facts (called from
# telegram_pilot_context_builder._select_confirmed_facts, which is the real
# context builder used by scripts/run_amo_wappi_draft_loop.py, the live
# draft-loop) calls price_axes_catalog.select_price_result_for_query ->
# extract_price_query_axes -> _extract_grade whenever
# TELEGRAM_PRICE_AXES_SELECTOR is on (default-on under the accepted
# `pilot_gold_v1` profile). The crash fixed below was reproducible on that
# path before this fix (see git history of this file for the exact
# `"м" in match.group(0)` bug); the ambiguity case after it is a documented,
# intentionally NOT fixed limitation of the current regex approach.


def test_extract_grade_handles_latin_m9_m11_like_cyrillic() -> None:
    """Regression repro for a real crash: the bot's own client_safe_text
    renders "M9"/"M11" in the LATIN alphabet (`product_code.upper()` on the
    ascii "m9"/"m11" strings -- see `_entries_from_m9_m11_tariff_fact`), so a
    client echoing that exact text back triggered `_extract_grade` to raise
    `IndexError: no such group` (patterns[1]/[2] have no capturing group; the
    old code checked for a Cyrillic "м" to decide whether to use it, which is
    False for Latin "m" and fell through to the missing group). Before the
    БЛОК 7 fix, both `_extract_grade("сколько стоит m9 тариф стандарт?")` and
    `extract_price_query_axes(...)` raised IndexError instead of returning 9.
    """
    assert _extract_grade("сколько стоит m9 тариф стандарт?") == 9
    assert _extract_grade("сколько стоит m11 тариф стандарт?") == 11
    # Cyrillic must keep working exactly as before.
    assert _extract_grade("сколько стоит м9 тариф стандарт?") == 9
    assert _extract_grade("сколько стоит м11 тариф стандарт?") == 11


def test_extract_grade_latin_m9_selects_correct_tariff_price_end_to_end() -> None:
    """End-to-end proof that the fixed extraction reaches the right price."""
    axes = extract_price_query_axes("Сколько стоит M9 тариф Стандарт?", active_brand="foton")
    assert axes["grade"] == 9
    assert axes["product_code"] == "m9"

    result = select_price_result_for_query(_facts(), active_brand="foton", query="Сколько стоит M9 тариф Стандарт онлайн за год?")
    assert result["status"] == "exact"
    assert result["entry"]["amount"] == 47250
    assert result["entry"]["product_code"] == "m9"
    assert result["entry"]["tariff_id"] == "standard"


def test_extract_grade_first_mentioned_grade_wins_when_message_has_two_children() -> None:
    """Documented, NOT fixed limitation (ТЗ: verify only, no migration).

    `_extract_grade` returns the grade from whichever pattern matches
    earliest in the text; it cannot detect "this message names two
    different grades" and never asks for disambiguation. Here the client is
    unambiguously asking about "М9" (a grade-9-only product), but the
    earlier "11 классе" mention wins, so query_axes carries grade=11 next to
    product_code="m9" -- a pairing that can never resolve to a real M9 price.
    Fixing this is a semantic/understanding change (which grade did the
    client mean), not an output-scrub bugfix, so ADR003's regex-understanding
    moratorium reserves it for a SemanticFrame/LLM change, not a regex edit.
    """
    text = "Хочу М9 для сына, а дочь в 11 классе, сколько за М9?"

    axes = extract_price_query_axes(text, active_brand="foton")

    assert axes["product_code"] == "m9"
    assert axes["grade"] == 11  # NOT 9, despite the client asking about M9 twice.

    # Currently safe by accident: the mismatched pairing cannot match any
    # catalog entry (M9 entries only cover grade 9), so this dead-ends as
    # "needs_slot"/"not_found" rather than quoting a wrong price outright.
    # That safety net depends on today's catalog never having a grade-11
    # entry under product_code "m9"; it is not a property _extract_grade
    # guarantees.
    result = select_price_result_for_query(_facts(), active_brand="foton", query=text)
    assert result["status"] != "exact"
