from __future__ import annotations

import subprocess

import pytest

from mango_mvp.question_catalog.classifier import (
    classify_question,
    load_valid_theme_and_service_ids,
    validate_against_taxonomy,
)
from mango_mvp.question_catalog.extractors import build_question_item


def test_canonical_question_function_removed() -> None:
    import mango_mvp.question_catalog.normalization as norm

    assert not hasattr(norm, "_canonical_question")
    result = subprocess.run(
        ["grep", "-r", "_canonical_question", "src/"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1


def test_refine_broad_fallback_subclass_removed() -> None:
    import mango_mvp.question_catalog.normalization as norm

    assert not hasattr(norm, "_refine_broad_fallback_subclass")
    result = subprocess.run(
        ["grep", "-r", "_refine_broad_fallback_subclass", "src/"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1


def test_classify_question_returns_valid_theme_or_service() -> None:
    valid_ids = load_valid_theme_and_service_ids()
    cases = {
        "Сколько стоит?": "theme:001_pricing",
        "Когда занятия?": "theme:013_schedule",
        "Не пришла ссылка": "theme:025_missing_links_access",
        "ыфвпролд": "service:S2_unclear",
    }

    for text, expected in cases.items():
        result = classify_question(text, source="test", metadata={})
        assert result.theme_id in valid_ids
        assert result.theme_id == expected


def test_classify_question_extracts_parameters() -> None:
    result = classify_question(
        "Сколько стоит ЕГЭ по математике для 11 класса онлайн?",
        source="test",
        metadata={},
    )

    assert result.theme_id == "theme:001_pricing"
    assert result.extracted_params["product"] == "регулярный_курс"
    assert result.extracted_params["subject"] == "математика"
    assert result.extracted_params["grade"] == "11_класс"
    assert result.extracted_params["format"] == "онлайн"


def test_validate_against_taxonomy_rejects_invalid_id() -> None:
    with pytest.raises(ValueError):
        validate_against_taxonomy("theme:999_fake")


def test_legacy_callers_still_work_through_theme_metadata() -> None:
    item = build_question_item(
        tenant_id="foton",
        source_channel="telegram",
        source_ref="telegram:test:stage-c",
        question_raw="Сколько стоит ЕГЭ по математике для 11 класса онлайн?",
    )

    assert item.intent == "theme:001_pricing"
    assert item.metadata["theme_id"] == "theme:001_pricing"
    assert item.metadata["theme_name"] == "Стоимость обучения"
    assert item.metadata["extracted_params"]["subject"] == "математика"
    assert "class_key" not in item.metadata
    assert item.question_class_id.startswith("question_class:")
