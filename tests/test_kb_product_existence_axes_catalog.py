from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.knowledge_base.product_existence_axes_catalog import (
    build_product_existence_axes_catalog,
    verify_product_format_exists,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"


def _facts() -> list[dict[str, object]]:
    return list(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")).get("facts") or [])


def _catalog() -> dict[str, object]:
    return build_product_existence_axes_catalog(_facts())


def test_catalog_proves_unpk_online_olympiad_physics_grade_9_exists() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        grade=9,
        subject="физика",
        format="онлайн",
        program_kind="олимпиадная подготовка",
    )

    assert result["status"] == "exists"
    entry = result["entry"]
    assert entry["brand"] == "unpk"
    assert entry["format"] == "online"
    assert "physics" in entry["subjects"]
    assert 9 in entry["grade_values"]
    assert entry["existence_status"] == "exists"
    assert entry["source_fact_key"].startswith("schedule_2026_27.")


def test_catalog_proves_unpk_summer_school_for_grade_5_exists() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        grade=5,
        product_family="летняя школа",
    )

    assert result["status"] == "exists"
    assert result["entry"]["product_family"] == "camp"
    assert 5 in result["entry"]["grade_values"]


def test_catalog_returns_not_offered_only_for_explicit_negative_fact() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        subject="химия",
    )

    assert result["status"] == "not_offered"
    assert result["reason"] == "explicit_not_offered_fact"
    assert result["entry"]["existence_status"] == "not_offered"
    assert "Химии сейчас нет" in result["entry"]["client_safe_text"]


def test_catalog_does_not_cross_brands_for_negative_facts() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="Фотон",
        subject="химия",
    )

    assert result["status"] == "unknown"
    assert result["reason"] == "no_exact_product_existence_fact"


def test_catalog_requires_identifying_slot_beyond_brand_and_format() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        format="онлайн",
    )

    assert result["status"] == "needs_slot"
    assert "subject_or_product" in result["missing_slots"]


def test_catalog_unknown_is_not_not_offered_when_no_fact_matches() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        grade=3,
        subject="биология",
        format="онлайн",
    )

    assert result["status"] == "unknown"
    assert result["reason"] == "no_exact_product_existence_fact"


def test_cancelled_shift_does_not_make_all_unpk_camps_not_offered() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        product_family="летняя школа",
    )

    assert result["status"] == "exists"
    assert result["entry"]["existence_status"] == "exists"
    assert "cancelled" not in result["entry"]["source_fact_key"]


def test_raw_false_fact_is_not_converted_to_positive_exists() -> None:
    entries = [
        entry
        for entry in _catalog()["entries"]
        if entry["source_fact_key"] == "intensives_2026.ege_intensive.available_in_foton"
    ]

    assert entries
    assert all(entry["existence_status"] == "not_offered" for entry in entries)


def test_operational_payment_facts_do_not_prove_product_existence() -> None:
    forbidden_keys = {
        "tg_unpk_verified_2026_05_21.client_facts.payment_deadline_after_registration.client_safe_text",
        "tg_unpk_verified_2026_05_21.client_facts.payment_purpose.client_safe_text",
        "lvsh_mendeleevo_2026.booking_policy.client_safe_text",
        "intensives_2026.oge_foton.includes.7",
    }
    keys = {entry["source_fact_key"] for entry in _catalog()["entries"]}

    assert keys.isdisjoint(forbidden_keys)


def test_noisy_grade_string_does_not_broaden_to_grade_9() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        grade="кабинет 9",
        subject="физика",
        format="онлайн",
        program_kind="олимпиадная подготовка",
    )

    assert result["status"] == "needs_slot"
    assert result["reason"] == "invalid_axis"
    assert result["invalid_slots"] == ["grade"]


def test_ordinal_grade_phrase_is_accepted_for_product_existence() -> None:
    result = verify_product_format_exists(
        _catalog(),
        brand="УНПК",
        grade="закончил 5-й класс",
        product_family="летняя школа",
    )

    assert result["status"] == "exists"
    assert 5 in result["entry"]["grade_values"]


def test_arbitrary_word_before_class_is_not_parsed_as_grade() -> None:
    catalog = _catalog()

    for noisy_grade in ("5 дней класс", "9 кабинет класс", "5 смена класс"):
        result = verify_product_format_exists(
            catalog,
            brand="УНПК",
            grade=noisy_grade,
            product_family="летняя школа",
        )

        assert result["status"] == "needs_slot"
        assert result["reason"] == "invalid_axis"
        assert result["invalid_slots"] == ["grade"]


def test_enrollment_facts_do_not_prove_product_existence() -> None:
    facts = [
        {
            "brand": "unpk",
            "fact_key": "synthetic.registration.open",
            "fact_type": "deadline",
            "product": "summer_school",
            "client_safe_text": "Регистрация на летнюю школу для 5 класса открыта.",
            "allowed_for_client_answer": True,
            "forbidden_for_client": False,
            "internal_only": False,
            "valid_until": "2027-05-30",
        },
        {
            "brand": "unpk",
            "fact_key": "synthetic.application.open",
            "fact_type": "program",
            "product": "summer_school",
            "client_safe_text": "Можно оставить заявку на летнюю школу для 5 класса.",
            "allowed_for_client_answer": True,
            "forbidden_for_client": False,
            "internal_only": False,
            "valid_until": "2027-05-30",
        },
    ]
    catalog = build_product_existence_axes_catalog(facts)

    assert catalog["entries"] == []
