from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.channels.fact_retrieval import select_confirmed_facts
from mango_mvp.knowledge_base.price_axes_catalog import build_price_axes_catalog, select_price


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


def test_catalog_adds_confirmed_unpk_online_atomic_prices() -> None:
    catalog = _catalog()
    entries = _entries(catalog)
    unpk = [
        entry
        for entry in entries
        if entry.get("brand") == "unpk" and entry.get("source_kind") == "unpk_online_kc_source_price"
    ]

    assert len(unpk) == 4
    amounts = {(entry["classes"], entry["schedule"], entry["period"], entry["amount"]) for entry in unpk}
    assert ("5-11", "weekend", "semester", 37000) in amounts
    assert ("5-11", "weekend", "year", 59000) in amounts
    assert ("9 и 11", "weekday", "semester", 41800) in amounts
    assert ("9 и 11", "weekday", "year", 69900) in amounts
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
    assert sum(1 for issue in issues if issue.get("issue") == "empty_client_safe_text_not_final_price") >= 2
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


def test_selector_requires_schedule_when_unpk_axes_have_two_prices() -> None:
    result = select_price(_catalog(), brand="УНПК", grade=9, subject="математика", format="онлайн", period="год")

    assert result["status"] == "needs_slot"
    assert "schedule" in result["missing_slots"]


def test_selector_can_pick_unpk_weekday_price_by_subject() -> None:
    result = select_price(_catalog(), brand="УНПК", grade=9, subject="информатика", format="онлайн", period="год")

    assert result["status"] == "exact"
    assert result["entry"]["amount"] == 69900
    assert result["entry"]["schedule"] == "weekday"


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
