from __future__ import annotations

import pytest

from mango_mvp.channels.output_verification_floor import verify_output


def _codes(text: str, brand: str, facts: dict[str, str] | None = None) -> set[str]:
    return {item.code for item in verify_output(text, facts=facts or {}, active_brand=brand)}


def test_foton_academic_mfti_phrase_requires_supporting_fact() -> None:
    text = "Преподаватели — из МФТИ, МГУ, ВШЭ"
    facts = {"teachers": "Преподаватели из МГУ и МФТИ, вожатые из топовых вузов."}
    assert "brand_leak" not in _codes(text, "foton", facts)
    assert "brand_leak" in _codes(text, "foton")


def test_foton_academic_mipt_phrase_requires_supporting_fact() -> None:
    text = "Преподаватели — из MIPT и МГУ"
    facts = {"teachers": "Преподаватели из МГУ и MIPT."}
    assert "brand_leak" not in _codes(text, "foton", facts)
    assert "brand_leak" in _codes("MIPT", "foton", facts)


@pytest.mark.parametrize(
    ("text", "brand", "blocked"),
    (
        ("Приходите в УНПК на kmipt.ru", "foton", True),
        ("Наш адрес — Сретенка", "foton", True),
        ("Фотон — это квант света", "unpk", False),
        ("условия Фотона выгоднее", "unpk", True),
        ("оплата Долями", "unpk", True),
        ("УНФТ", "foton", True),
        ("MIPT", "foton", True),
        ("ТБанк", "unpk", True),
    ),
)
def test_structured_brand_examples(text: str, brand: str, blocked: bool) -> None:
    assert ("brand_leak" in _codes(text, brand)) is blocked


def test_mfti_discount_is_valid_for_unpk_when_number_is_fact_backed() -> None:
    text = "Скидка 10% сотрудникам МФТИ"
    assert not _codes(text, "unpk", {"discount": text})


@pytest.mark.parametrize("brand", ("foton", "unpk"))
def test_shared_krasnoselskaya_address_is_valid_for_both_brands(brand: str) -> None:
    text = "Верхняя Красносельская, 30"
    assert not _codes(text, brand, {"address": text})


def test_shared_mendeleevo_price_is_decided_by_number_gate_not_brand_gate() -> None:
    valid = "Менделеево, 93 100 ₽"
    invalid = "Менделеево, 114 000 ₽"
    facts = {"price": "ЛВШ Менделеево стоит 93 100 ₽"}

    assert not _codes(valid, "foton", facts)
    invalid_codes = _codes(invalid, "foton", facts)
    assert "brand_leak" not in invalid_codes
    assert invalid_codes


@pytest.mark.parametrize(
    "text",
    (
        "В Фотоне рассказывают, что фотон — квант света",
        "В Фотоне есть курс про фотон как квант света",
    ),
)
def test_physical_photon_exception_does_not_hide_a_real_brand_mention(text: str) -> None:
    assert "brand_leak" in _codes(text, "unpk")


def test_physical_photon_context_remains_allowed() -> None:
    assert "brand_leak" not in _codes(
        "Фотон как частица света изучается в теме квантовой физики",
        "unpk",
    )


@pytest.mark.parametrize("text", ("T-Банк", "Т bank", "TБанк", "Тbank", "Тинькофф"))
def test_unpk_blocks_mixed_and_legacy_tbank_names(text: str) -> None:
    assert "brand_leak" in _codes(text, "unpk")
