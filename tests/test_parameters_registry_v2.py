from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from mango_mvp.question_catalog.parameters_registry import (
    extract_parameters,
    load_parameters_registry,
    validate_parameters_registry,
)


ROOT = Path(__file__).resolve().parents[1]
QUESTION_CLASSES_PATH = ROOT / "product_data" / "question_catalog" / "customer_question_classes.csv"


def test_8_parameters_present() -> None:
    registry = load_parameters_registry()
    parameter_ids = [item["parameter_id"] for item in registry["parameters"]]

    assert parameter_ids == [
        "product",
        "subject",
        "grade",
        "format",
        "customer_status",
        "urgency",
        "sentiment",
        "document_type",
    ]


def test_all_values_closed_lists() -> None:
    registry = load_parameters_registry()
    errors = validate_parameters_registry(registry)

    assert errors == []
    for parameter in registry["parameters"]:
        assert isinstance(parameter["values"], list)
        assert "не_указано" in parameter["values"]
        assert parameter["fallback_value"] in parameter["values"]
        assert parameter["extraction_method"] in {"regex", "external_lookup"}


def test_extraction_recall_on_real_data() -> None:
    rows = _sample_rows_with_visible_parameters(limit=100)
    assert len(rows) >= 100

    score = _score_recall(rows)

    for parameter_id, recall in score.items():
        assert recall >= 0.80, f"{parameter_id} recall={recall:.2%}, score={score}"


def test_urgency_negation_does_not_trigger_critical() -> None:
    assert extract_parameters("Это не срочно, ответьте когда будет удобно")["urgency"] == "низкая"
    assert extract_parameters("Это не критично")["urgency"] == "низкая"
    assert extract_parameters("Срочно нужен ответ по оплате")["urgency"] == "критическая"


def test_sentiment_negation_does_not_trigger_positive() -> None:
    assert extract_parameters("Мы не довольны организацией занятий")["sentiment"] == "негативный"
    assert extract_parameters("Мы не рады такому изменению расписания")["sentiment"] == "негативный"
    assert extract_parameters("Ребенку не нравится преподаватель")["sentiment"] == "негативный"
    assert extract_parameters("Мы довольны преподавателем, спасибо")["sentiment"] == "позитивный"


def _sample_rows_with_visible_parameters(*, limit: int) -> list[dict[str, str]]:
    with QUESTION_CLASSES_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row for row in rows
        if _expected_product(row) or _expected_subject(row) or _expected_grade(row) or _visible_format(row)
    ]
    random.Random(20260514).shuffle(candidates)
    return candidates[:limit]


def _score_recall(rows: list[dict[str, str]]) -> dict[str, float]:
    hits = {key: 0 for key in ("product", "subject", "grade", "format")}
    totals = {key: 0 for key in hits}

    for row in rows:
        text = _row_text(row)
        extracted = extract_parameters(text)
        expected = {
            "product": _expected_product(row),
            "subject": _expected_subject(row),
            "grade": _expected_grade(row),
            "format": _visible_format(row),
        }
        for parameter_id, expected_value in expected.items():
            if not expected_value:
                continue
            totals[parameter_id] += 1
            if extracted[parameter_id] == expected_value:
                hits[parameter_id] += 1

    return {
        parameter_id: hits[parameter_id] / totals[parameter_id]
        for parameter_id in hits
        if totals[parameter_id]
    }


def _row_text(row: dict[str, str]) -> str:
    return " | ".join(
        value for value in (
            row.get("canonical_question", ""),
            row.get("narrow_scope", ""),
            row.get("examples_for_rop", ""),
            row.get("examples_redacted", ""),
        )
        if value
    )


def _expected_product(row: dict[str, str]) -> str | None:
    value = (row.get("products") or "").strip()
    text = _row_text(row).lower()
    if value == "летняя школа" and re.search(r"\b(?:лагер\w*|смен[ауы]|путевк\w*|кампус)\b", text):
        return "лагерь"
    if value == "регулярный курс":
        if re.search(r"\bинтенсив\w*\b", text):
            return "интенсив"
        if re.search(r"\b(?:индивидуальн\w+|репетитор\w*|один\s+на\s+один)\b", text):
            return "индивидуальные"
        if re.search(r"\b(?:пробн\w+\s+заняти\w*|пробн\w+\s+урок)\b", text):
            return "пробное"
    mapping = {
        "регулярный курс": "регулярный_курс",
        "ЕГЭ": "регулярный_курс",
        "ОГЭ": "регулярный_курс",
        "ЗВШ": "регулярный_курс",
        "зимняя школа": "регулярный_курс",
        "летняя школа": "летняя_школа",
        "олимпиады": "олимпиады",
        "пробное занятие": "пробное",
    }
    return mapping.get(value)


def _expected_subject(row: dict[str, str]) -> str | None:
    value = (row.get("subjects") or "").strip()
    mapping = {
        "математика": "математика",
        "физика": "физика",
        "химия": "химия",
        "биология": "биология",
        "информатика": "информатика",
        "русский язык": "русский",
        "английский язык": "английский",
        "обществознание": "обществознание",
        "литература": "литература",
    }
    return mapping.get(value)


def _expected_grade(row: dict[str, str]) -> str | None:
    value = (row.get("grades") or "").strip()
    if value in {"1 класс", "2 класс", "3 класс", "4 класс"}:
        return "1_4_класс"
    if value in {"5 класс", "6 класс", "7 класс", "8 класс"}:
        return "5_8_класс"
    if value in {"9 класс", "10 класс", "11 класс"}:
        return value.replace(" ", "_")
    return None


def _visible_format(row: dict[str, str]) -> str | None:
    text = _row_text(row).lower()
    if re.search(r"\b(?:онлайн|дистанц)", text) and re.search(r"\b(?:очн|офлайн)", text):
        return "гибрид"
    if re.search(r"\b(?:в\s+запис[ьи]|запись\s+(?:урока|заняти|вебинара))", text):
        return "запись"
    if re.search(r"\b(?:онлайн|дистанц|zoom|зум|вебинар)", text):
        return "онлайн"
    if re.search(r"\b(?:очно|очная\s+форма|офлайн|на\s+площадк\w+)", text):
        return "очно"
    return None
