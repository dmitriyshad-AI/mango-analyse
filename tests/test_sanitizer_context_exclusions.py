from __future__ import annotations

import mango_mvp.insights.sanitizers as sanitizer_module
from mango_mvp.insights.sanitizers import (
    has_any_safety_risk,
    has_brand_risk,
    has_money_or_terms_risk,
    has_personal_data_risk,
    sanitize_answer,
)


def test_sanitize_answer_keeps_non_money_counts() -> None:
    for text in ("5 000 человек", "за 5000 человек", "2 500 баллов"):
        result = sanitize_answer(text, mode="bot")

        assert "[CURRENT_PRICE]" not in result.text
        assert "актуальную стоимость" not in result.text
        assert "price_redacted" not in result.flags
        assert has_money_or_terms_risk(text) is False


def test_sanitize_answer_preserves_money_amount_regressions() -> None:
    cases = (
        "50к",
        "100 т.р.",
        "пятьдесят тысяч рублей",
        "стоимость 50000",
        "50 000 рублей",
        "7900 за 4 занятия",
    )

    for text in cases:
        result = sanitize_answer(text, mode="bot")

        assert ("[CURRENT_PRICE]" in result.text) or ("актуальную стоимость" in result.text)
        assert "price_redacted" in result.flags
        assert has_money_or_terms_risk(text) is True


def test_sanitize_answer_keeps_non_discount_percent_context() -> None:
    for text in ("100% результат", "100 процентов результат", "95% посещаемость", "98% сдача"):
        result = sanitize_answer(text, mode="bot")

        assert "[PAYMENT_OPTIONS]" not in result.text
        assert "актуальные варианты" not in result.text
        assert "percent_redacted" not in result.flags
        assert has_money_or_terms_risk(text) is False

    guarantee_result = sanitize_answer("100% гарантия результата", mode="bot")
    assert "[PAYMENT_OPTIONS]" not in guarantee_result.text
    assert "актуальные варианты" not in guarantee_result.text
    assert "percent_redacted" not in guarantee_result.flags


def test_sanitize_answer_preserves_discount_percent_forms() -> None:
    for text in ("10 процентов", "десять процентов", "15% скидка", "скидка 10%"):
        result = sanitize_answer(text, mode="bot")

        assert ("[PAYMENT_OPTIONS]" in result.text) or ("актуальные варианты" in result.text)
        assert has_money_or_terms_risk(text) is True


def test_sanitize_answer_removes_internal_fact_metadata() -> None:
    result = sanitize_answer(
        "Ответ клиенту. fact_id:abc fact:v3:foton:price source_id=fact:v3:price trace_id=run-1",
        mode="bot",
    )

    assert "fact_id" not in result.text
    assert "fact:v3" not in result.text
    assert "source_id" not in result.text
    assert "trace_id" not in result.text
    assert "internal_metadata_redacted" in result.flags
    assert has_any_safety_risk("Ответ fact_id:abc") is True


def test_sanitize_answer_flags_raw_json_leak() -> None:
    raw = '{"route":"draft_for_manager","draft_text":"Ответ","trace_id":"run-1"}'
    result = sanitize_answer(raw, mode="bot")

    assert "raw_json_redacted" in result.flags
    assert has_any_safety_risk(raw) is True


def test_sanitize_answer_normalizes_brand_money_terms_and_personal_data() -> None:
    raw = (
        "Ольга Михайловна, в НПК МФТИ стоимость 50 000 рублей, скидка 10%, "
        "рассрочка на 12 месяцев и возврат до 15 мая. Пишите на test@example.com или +7 900 123-45-67."
    )

    manager = sanitize_answer(raw, mode="manager")
    bot = sanitize_answer(raw, mode="bot")

    assert "Фотон" in manager.text
    assert "НПК" not in manager.text
    assert "Ольга Михайловна" not in manager.text
    assert "50 000" not in manager.text
    assert "10%" not in manager.text
    assert "test@example.com" not in manager.text
    assert "+7 900" not in manager.text
    assert "наш учебный центр" in bot.text
    assert "Точные условия менеджер подтвердит" in bot.text
    assert not has_brand_risk(bot.text)
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)
    assert "brand_normalized" in manager.flags
    assert "price_redacted" in manager.flags
    assert "person_name_redacted" in manager.flags


def test_sanitize_answer_catches_hidden_dates_spoken_percent_and_single_names() -> None:
    raw = (
        "Мария, бронь держим до пятницы, 10 апреля. Оплата 50к, скидка 10 процентов. "
        "Михаил занимается в субботу 10:00-12:00."
    )

    bot = sanitize_answer(raw, mode="bot")

    assert "Мария" not in bot.text
    assert "Михаил" not in bot.text
    assert "пятницы" not in bot.text.lower()
    assert "10 апреля" not in bot.text.lower()
    assert "50к" not in bot.text.lower()
    assert "10 процентов" not in bot.text.lower()
    assert "10:00" not in bot.text
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)
    assert "deadline_redacted" in bot.flags
    assert "price_redacted" in bot.flags
    assert "percent_redacted" in bot.flags
    assert "person_name_redacted" in bot.flags


def test_sanitize_answer_catches_stage15_adversarial_bot_export_risks() -> None:
    raw = "Максим получит ссылку @anna_photon, стоимость пятьдесят тысяч рублей или 50 т.р., звонить с 10 до 22."

    bot = sanitize_answer(raw, mode="bot")

    assert "Максим" not in bot.text
    assert "@anna_photon" not in bot.text
    assert "пятьдесят тысяч" not in bot.text.lower()
    assert "50 т" not in bot.text.lower()
    assert "с 10 до 22" not in bot.text.lower()
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)


def test_sanitize_answer_catches_claude_stage15_price_leak_patterns() -> None:
    raw = (
        "По оплате: физика 7900 за 4 занятия. "
        "Первый семестр за 88000, год целиком за 147000. "
        "При ранней оплате 78400."
    )

    bot = sanitize_answer(raw, mode="bot")

    assert "7900" not in bot.text
    assert "88000" not in bot.text
    assert "147000" not in bot.text
    assert "78400" not in bot.text
    assert "актуальную стоимость" in bot.text
    assert "price_redacted" in bot.flags
    assert not has_money_or_terms_risk(bot.text)


def test_sanitize_answer_catches_claude_stage15_location_teacher_deadline_and_promise_patterns() -> None:
    raw = (
        "Преподаватель Лукина ждет вас в Долгопрудном: проспект Пацаева, 7 корпус 1, "
        "4 этаж, кабинет 49, рядом со Скорняжным переулком и Чистыми прудами. До конца дня вернемся с подтверждением. "
        "Письмо от Альфа-банка придет на почту vidu@. . . в районе Сухаревки. "
        "Файл Word «Разбивка 1» отправим отдельно."
    )

    bot = sanitize_answer(raw, mode="bot")

    assert "Лукина" not in bot.text
    assert "Долгопруд" not in bot.text
    assert "Пацаева" not in bot.text
    assert "Скорняж" not in bot.text
    assert "Чист" not in bot.text
    assert "кабинет 49" not in bot.text
    assert "до конца дня" not in bot.text.lower()
    assert "Альфа" not in bot.text
    assert "vidu@" not in bot.text
    assert "Сухарев" not in bot.text
    assert "Разбивка 1" not in bot.text
    assert "адрес, который подтвердит менеджер" in bot.text
    assert "менеджер свяжется с вами после проверки" in bot.text
    assert "role_name_redacted" in bot.flags
    assert "location_redacted" in bot.flags
    assert "service_promise_redacted" in bot.flags
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)


def test_sanitize_answer_catches_claude_reaudit_orphan_names_dates_and_compensation() -> None:
    raw = (
        "По ученик Николаевне нет статистики, но физику будет вести Кондрашова. "
        "Преподаватель - ученик Гамзяков, очная группа ученик Еделькина. "
        "Будет ли Камаринцев вести информатику? По Камаринцеву уточним. "
        "Скажите фамилию Николаев, подойдите к вахте в КПМ на Майской, кабинет 324. "
        "Пусть Катерина восстановится, действует до 15 числа, тестирование до 17 числа, "
        "важно компенсировать занятие."
    )

    bot = sanitize_answer(raw, mode="bot")

    for leaked in (
        "Николаев",
        "Кондраш",
        "Гамзяк",
        "Еделькин",
        "Камаринц",
        "Майск",
        "КПМ",
        "кабинет 324",
        "Катерин",
        "15 числа",
        "17 числа",
        "компенсировать",
    ):
        assert leaked not in bot.text
    assert "актуальное окно записи" in bot.text
    assert "адрес, который подтвердит менеджер" in bot.text
    assert "менеджер свяжется с вами после проверки" in bot.text
    assert "person_name_redacted" in bot.flags
    assert "location_redacted" in bot.flags
    assert "deadline_redacted" in bot.flags
    assert "service_promise_redacted" in bot.flags
    assert not has_money_or_terms_risk(bot.text)
    assert not has_personal_data_risk(bot.text)


def test_sanitize_answer_is_strongly_idempotent_on_adversarial_bot_cases() -> None:
    cases = [
        (
            "Ольга Михайловна, в НПК МФТИ стоимость 50 000 рублей, скидка 10%, "
            "рассрочка до 15 мая, пишите на test@example.com."
        ),
        "Первый семестр за 88000, год целиком за 147000. Физика 7900 за 4 занятия.",
        (
            "Преподаватель Лукина ждет в Долгопрудном: проспект Пацаева 7 корпус 1, "
            "кабинет 49. До конца дня вернемся."
        ),
        (
            "По ученик Николаевне нет статистики, будет вести Кондрашова. "
            "Преподаватель - ученик Гамзяков, действует до 17 числа."
        ),
        "Менеджер свяжется после проверки. Точные условия менеджер подтвердит по актуальным правилам.",
    ]

    for raw in cases:
        first = sanitize_answer(raw, mode="bot")
        second = sanitize_answer(first.text, mode="bot")

        assert first.fixpoint_reached is True
        assert first.pass_count >= 1
        assert second.text == first.text
        assert second.fixpoint_reached is True


def test_sanitize_answer_blocks_when_fixpoint_is_not_reached(monkeypatch) -> None:
    def unstable_pass(text: object, *, mode: sanitizer_module.SanitizerMode = "bot") -> sanitizer_module.SanitizedText:
        return sanitizer_module.SanitizedText(
            f"{sanitizer_module.clean_text(text)} x",
            ("person_name_redacted",),
            "safe_with_placeholders",
            pass_count=1,
        )

    monkeypatch.setattr(sanitizer_module, "_sanitize_answer_pass", unstable_pass)

    result = sanitizer_module.sanitize_answer("циклический текст", mode="bot", max_passes=3)

    assert result.status == "fixpoint_not_reached"
    assert result.text == ""
    assert result.fixpoint_reached is False
    assert result.pass_count == 3


def test_sanitize_customer_text_blocks_when_fixpoint_is_not_reached(monkeypatch) -> None:
    def unstable_pass(text: object, *, mode: sanitizer_module.SanitizerMode = "customer") -> sanitizer_module.SanitizedText:
        return sanitizer_module.SanitizedText(
            f"{sanitizer_module.clean_text(text)} x",
            ("phone_redacted",),
            "safe_with_placeholders",
            pass_count=1,
        )

    monkeypatch.setattr(sanitizer_module, "_sanitize_answer_pass", unstable_pass)

    result = sanitizer_module.sanitize_answer("циклический клиентский текст", mode="customer", max_passes=3)

    assert result.status == "fixpoint_not_reached"
    assert result.text == ""
    assert result.fixpoint_reached is False
    assert result.pass_count == 3


def test_sanitize_manager_text_keeps_diagnostic_text_when_fixpoint_is_not_reached(monkeypatch) -> None:
    def unstable_pass(text: object, *, mode: sanitizer_module.SanitizerMode = "manager") -> sanitizer_module.SanitizedText:
        return sanitizer_module.SanitizedText(
            f"{sanitizer_module.clean_text(text)} x",
            ("price_redacted",),
            "safe_with_placeholders",
            pass_count=1,
        )

    monkeypatch.setattr(sanitizer_module, "_sanitize_answer_pass", unstable_pass)

    result = sanitizer_module.sanitize_answer("циклический текст", mode="manager", max_passes=3)

    assert result.status == "fixpoint_not_reached"
    assert result.text == "циклический текст x x x"
    assert result.fixpoint_reached is False
    assert result.pass_count == 3


def test_sanitize_answer_stable_text_keeps_existing_client_safe_parity() -> None:
    result = sanitize_answer("Менеджер подтвердит актуальную стоимость.", mode="bot")

    assert result.status == "safe_no_changes"
    assert result.text == "Менеджер подтвердит актуальную стоимость."
    assert result.fixpoint_reached is True
