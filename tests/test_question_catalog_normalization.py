from __future__ import annotations

from mango_mvp.question_catalog import (
    classify_question,
    detect_noise_reason,
    infer_question_metadata,
    is_question_like,
    split_candidate_questions,
)
from mango_mvp.question_catalog.normalization import is_outbound_system_text
from mango_mvp.question_catalog.safety import compress_mask_runs, score_example_readability


def test_question_detection_handles_requests_without_question_mark() -> None:
    assert is_question_like("Пришлите, пожалуйста, курсы по математике для 6 класса")
    assert is_question_like("Сколько стоит подготовка к ЕГЭ?")
    assert not is_question_like("Письмо сгенерировано автоматически. Отписаться от рассылки.")
    assert not is_question_like("Будьте в курсе всех новостей и интересных событий!")
    assert not is_question_like("Счёт: 30101810400000000225 ИНН: 7700000000 КПП: 770001001")
    assert detect_noise_reason("Если у вас остались вопросы, вы можете задать их в Telegram")
    assert detect_noise_reason("возражение/ограничение: время")
    assert detect_noise_reason("Содержательный запрос не сформулирован: клиент только сказал «алло»")
    assert is_outbound_system_text("Ваше расписание занятий в 2026 учебном году")
    assert not is_question_like("Менеджер пообещала направить лицензию письмом")
    assert detect_noise_reason("Re: Возврат д/с.")
    assert detect_noise_reason("Изменения в расписании: уведомление о закрытии группы по математике.")
    assert detect_noise_reason("-------- Пересылаемое сообщение -------- От кого: Учебный центр")
    assert detect_noise_reason("kmipt. ru: заполнена web-форма [1] Записаться на курсы.")


def test_readable_example_helpers_downrank_heavy_masks() -> None:
    noisy = (
        "действующие правила изменения или отмены услуги действующие правила изменения или отмены услуги "
        "актуальное окно записи актуальное окно записи"
    )

    assert compress_mask_runs(noisy).count("действующие правила изменения или отмены услуги") == 1
    assert score_example_readability(noisy) < score_example_readability("Сколько стоит летняя школа в июне?")


def test_split_candidate_questions_keeps_narrow_parts() -> None:
    parts = split_candidate_questions("Здравствуйте. Сколько стоит математика? Где проходят занятия очно? Спасибо")

    assert parts == ["Сколько стоит математика?", "Где проходят занятия очно?"]


def test_infer_metadata_returns_v2_contract() -> None:
    meta = infer_question_metadata("Стоимость подготовки к ЕГЭ по профильной математике для 11 класса онлайн")

    assert meta.theme_id == "theme:001_pricing"
    assert meta.extracted_params["product"] == "регулярный_курс"
    assert meta.extracted_params["subject"] == "математика"
    assert meta.extracted_params["grade"] == "11_класс"
    assert meta.extracted_params["format"] == "онлайн"
    assert 0 <= meta.confidence_hint <= 1
    assert meta.classification_method.startswith("rule_stub_")


def test_infer_metadata_no_longer_exposes_legacy_class_key_fields() -> None:
    meta = infer_question_metadata("Сколько стоит подготовка к ЕГЭ?")

    assert not hasattr(meta, "intent")
    assert not hasattr(meta, "subclass_key")
    assert not hasattr(meta, "class_key")
    assert not hasattr(meta, "canonical_question")


def test_classify_question_uses_fallback_signal_with_v2_result() -> None:
    result = classify_question("Расскажите подробнее", fallback_signal="schedule_question")

    assert result.theme_id == "theme:013_schedule"
    assert result.required_facts == ("schedule.current",)
    assert result.classification_method == "rule_stub_intent_override"


def test_obvious_questions_map_to_allowed_v2_themes_or_services() -> None:
    cases = {
        "Сколько стоит подготовка к ЕГЭ?": "theme:001_pricing",
        "Итого 71 250 руб.?": "theme:003_payment_status",
        "Оплата материнским капиталом возможна?": "theme:007_matkap_payment",
        "Если деньги придут к вам позже, вы нам вернете эти пятьдесят семь тысяч?": "theme:009_refund",
        "Где проходят занятия очно?": "theme:015_address",
        "Когда занятия?": "theme:013_schedule",
        "Не пришла ссылка на личный кабинет": "theme:025_missing_links_access",
        "Детей везут до лагеря на автобусах?": "theme:028_transport_logistics",
        "ыфвпролд": "service:S2_unclear",
    }

    for text, expected_theme in cases.items():
        assert classify_question(text).theme_id == expected_theme


def test_short_context_uses_service_bucket_not_legacy_fallback_bucket() -> None:
    result = classify_question("Это он и есть?")

    assert result.theme_id.startswith("service:")
    assert "manual_review" not in result.theme_id
    assert "mixed_context" not in result.theme_id
    assert "conversation_summary" not in result.theme_id
