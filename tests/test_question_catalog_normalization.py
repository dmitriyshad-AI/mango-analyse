from __future__ import annotations

from mango_mvp.question_catalog import (
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    classify_question,
    infer_question_metadata,
    is_question_like,
    split_candidate_questions,
)


def test_question_detection_handles_requests_without_question_mark() -> None:
    assert is_question_like("Пришлите, пожалуйста, курсы по математике для 6 класса")
    assert is_question_like("Сколько стоит подготовка к ЕГЭ?")
    assert not is_question_like("Письмо сгенерировано автоматически. Отписаться от рассылки.")
    assert not is_question_like("Будьте в курсе всех новостей и интересных событий!")
    assert not is_question_like("Счёт: 30101810400000000225 ИНН: 7700000000 КПП: 770001001")


def test_split_candidate_questions_keeps_narrow_parts() -> None:
    parts = split_candidate_questions("Здравствуйте. Сколько стоит математика? Где проходят занятия очно? Спасибо")

    assert parts == ["Сколько стоит математика?", "Где проходят занятия очно?"]


def test_infer_metadata_for_price_ege_math_grade() -> None:
    meta = infer_question_metadata("Стоимость подготовки к ЕГЭ по профильной математике для 11 класса")

    assert meta.intent == "price"
    assert meta.product == "ЕГЭ"
    assert meta.subject == "математика"
    assert meta.grade == "11 класс"
    assert meta.dynamic_fact_types == ("price",)
    assert meta.required_fact_keys == ("price.current",)
    assert meta.bot_permission == BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK
    assert "стоимость" in meta.canonical_question


def test_classify_question_uses_fallback_signal() -> None:
    payload = classify_question("Расскажите подробнее", fallback_signal="schedule_question")

    assert payload["intent"] == "schedule"
    assert payload["required_fact_keys"] == ("schedule.current",)
