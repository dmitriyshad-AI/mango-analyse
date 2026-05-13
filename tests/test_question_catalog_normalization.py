from __future__ import annotations

from mango_mvp.question_catalog import (
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    BOT_PERMISSION_MANAGER_ONLY,
    BOT_PERMISSION_NOT_ALLOWED,
    classify_question,
    detect_noise_reason,
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
    assert detect_noise_reason("Если у вас остались вопросы, вы можете задать их в Telegram")


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


def test_general_question_split_intents_are_narrow() -> None:
    cases = {
        "По какому поводу вы звонили?": "call_reason",
        "Мне ничего не пришло на почту": "message_not_received",
        "Можно ли вернуть оплату и получить чек?": "payment_service",
        "Как оформить письмо для налоговой за прошлый год?": "documents_letter",
        "Детей везут до лагеря на автобусах?": "transport_logistics",
        "Ребенок не поедет на летнюю смену": "camp_trip",
        "У нас слишком легкие задания, можно дать обратную связь?": "quality_feedback",
        "Пока неактуально, мы сами наберем": "no_interest",
        "Нужно обсудить, продолжать ли обучение в следующем году": "continuation_decision",
        "Ребята распределяются по уровню подготовки?": "age_or_level_fit",
        "Это ваш сайт или другой?": "site_confusion",
        "Подскажите пожалуйста, вы готовите детей к ВПР?": "program",
        "Что дальше делать?": "general_next_step",
        "Появилась ли информация?": "status_followup",
        "Как смотреть повторы?": "technical_access",
        "Нужно ли распечатывать справку в поликлинике и ставить печать?": "documents_letter",
        "А про это?": "incomplete_context",
        "Можно еще вопрос?": "general_consultation",
        "К кому обратиться по бытовому вопросу?": "camp_living_conditions",
        "Ребенка отметите заранее?": "attendance_absence",
    }

    for text, expected_intent in cases.items():
        assert infer_question_metadata(text).intent == expected_intent


def test_manager_only_and_noise_permissions() -> None:
    payment = infer_question_metadata("Можно ли вернуть оплату и получить чек?")
    assert payment.bot_permission == BOT_PERMISSION_MANAGER_ONLY

    noise = infer_question_metadata("С уважением, команда Фотон")
    assert noise.intent == "not_customer_question"
    assert noise.bot_permission == BOT_PERMISSION_NOT_ALLOWED
