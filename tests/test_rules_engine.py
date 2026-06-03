from __future__ import annotations

import re

from mango_mvp.channels.rules_engine import MIGRATED, apply_rule, load_rules_registry, select_rule


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d[\d\s]*(?:[.,]\d+)?(?:\s*(?:₽|%))?", text))


def test_rules_registry_loads_approved_migrated_rules() -> None:
    registry = load_rules_registry()

    assert len(registry) == 16
    assert set(registry) == set(MIGRATED)
    assert set(MIGRATED) == {
        "teacher",
        "recordings",
        "contact_address",
        "docs",
        "matkap",
        "tax",
        "olympiad",
        "platform_access",
        "installment",
        "discount",
        "price",
        "format_choice",
        "trial",
        "camp_lvsh",
        "enrollment_process",
        "schedule",
    }
    assert select_rule("teacher", registry).rule_id == "teacher"  # type: ignore[union-attr]
    assert select_rule("recording", registry).rule_id == "recordings"  # type: ignore[union-attr]
    assert select_rule("address", registry).rule_id == "contact_address"  # type: ignore[union-attr]
    assert select_rule("document", registry).rule_id == "docs"  # type: ignore[union-attr]
    assert select_rule("matkap", registry).rule_id == "matkap"  # type: ignore[union-attr]
    assert select_rule("tax", registry).rule_id == "tax"  # type: ignore[union-attr]
    assert select_rule("olympiad_online", registry).rule_id == "olympiad"  # type: ignore[union-attr]
    assert select_rule("platform_access", registry).rule_id == "platform_access"  # type: ignore[union-attr]
    assert select_rule("installment", registry).rule_id == "installment"  # type: ignore[union-attr]
    assert select_rule("payment_method", registry).rule_id == "installment"  # type: ignore[union-attr]
    assert select_rule("discount", registry).rule_id == "discount"  # type: ignore[union-attr]
    assert select_rule("pricing", registry).rule_id == "price"  # type: ignore[union-attr]
    assert select_rule("price_fix", registry).rule_id == "price"  # type: ignore[union-attr]
    assert select_rule("format", registry).rule_id == "format_choice"  # type: ignore[union-attr]
    assert select_rule("trial", registry).rule_id == "trial"  # type: ignore[union-attr]
    assert select_rule("camp", registry).rule_id == "camp_lvsh"  # type: ignore[union-attr]
    assert select_rule("live_availability", registry).rule_id == "camp_lvsh"  # type: ignore[union-attr]
    assert select_rule("enrollment_process", registry).rule_id == "enrollment_process"  # type: ignore[union-attr]
    assert select_rule("schedule", registry).rule_id == "schedule"  # type: ignore[union-attr]


def test_rules_engine_teacher_uses_fact_and_does_not_invent_specific_name() -> None:
    registry = load_rules_registry()
    rule = registry["teacher"]
    facts = {
        "bot_policy.approved_phrases.theme_17_teachers.foton": (
            "Преподаватели — из МФТИ, МГУ, ВШЭ, МГТУ им. Баумана, МИФИ. Эксперты ЕГЭ и члены жюри олимпиад."
        )
    }

    outcome = apply_rule(
        rule,
        plan={"primary_intent": "teacher", "direct_question": "как зовут преподавателя физики в Лобне?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "draft_for_manager"
    assert "Менеджер уточнит" in outcome.text
    assert "МГУ" in outcome.text
    assert "Иван" not in outcome.text


def test_rules_engine_contact_address_uses_registry_foton_spelling() -> None:
    registry = load_rules_registry()
    rule = registry["contact_address"]

    outcome = apply_rule(
        rule,
        plan={"primary_intent": "address", "direct_question": "где очные занятия, адрес?", "active_brand": "foton"},
        facts={},
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "bot_answer_self_for_pilot"
    assert "Верхняя Красносельская" in outcome.text
    assert "Скорняжный" not in outcome.text
    assert "УНПК" not in outcome.text


def test_rules_engine_contact_address_prefers_foton_kb_fact() -> None:
    registry = load_rules_registry()
    rule = registry["contact_address"]

    outcome = apply_rule(
        rule,
        plan={"primary_intent": "address", "direct_question": "какой адрес Фотона?", "active_brand": "foton"},
        facts={"contact.foton.address": "Фотон: адрес и место занятий — Верхняя Красносельская ул., 30."},
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "bot_answer_self_for_pilot"
    assert "Москва, Верхняя Красносельская ул., 30" in outcome.text
    assert "Скорняжный" not in outcome.text
    assert "contact.foton.address" in outcome.facts


def test_rules_engine_schedule_contact_hours_are_not_class_days() -> None:
    rule = load_rules_registry()["schedule"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "schedule", "direct_question": "в какие дни 7 класс физика?", "active_brand": "foton"},
        facts={
            "contacts_foton.contact_hours.client_safe_text": (
                "Связаться с Фотоном можно ежедневно с 10:00 до 18:00. Это часы связи, а не расписание занятий групп."
            )
        },
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "draft_for_manager"
    assert "rules_engine_schedule_manager_check" in outcome.flags
    assert "10:00-18:00 не считаю расписанием занятий" in outcome.text
    assert "Пн-Вс" not in outcome.text


def test_rules_engine_schedule_exact_group_days_only_from_group_fact() -> None:
    rule = load_rules_registry()["schedule"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "schedule", "direct_question": "когда математика 11 класс очно?", "active_brand": "foton"},
        facts={
            "schedule_2026_27.groups.group_start_date_c13_krasnoselskaya_sat_1000_1200_math_11_advanced.client_safe_text": (
                "Математика, 11 класс, продвинутая группа, очно, Верхняя Красносельская, 30: суббота 10:00-12:00, старт 12.09.2026. Точное расписание конкретной группы уточняется."
            )
        },
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "bot_answer_self_for_pilot"
    assert "rules_engine_schedule_group_fact" in outcome.flags
    assert "суббота 10:00-12:00" in outcome.text
    assert "Верхняя Красносельская" in outcome.text


def test_step4_phase1_warmed_rule_strings_preserve_verbatim_facts_and_numbers() -> None:
    registry = load_rules_registry()
    teacher_fact = "Преподаватели — из МФТИ, МГУ, ВШЭ. Эксперты ЕГЭ и члены жюри олимпиад."
    teacher = apply_rule(
        registry["teacher"],
        plan={"primary_intent": "teacher", "direct_question": "кто преподаёт?", "active_brand": "foton"},
        facts={"teacher.fact": teacher_fact},
        context={"active_brand": "foton"},
    )
    assert teacher is not None
    assert "Про преподавателей могу дать такой ориентир" in teacher.text
    assert "По преподавателям:" not in teacher.text
    assert "МГУ, ВШЭ" in teacher.text

    price_facts = {
        "prices_regular_2026_27.online_grade10.semester": "Фотон: цены на 2026/27 учебный год, 10 класс, онлайн, семестр — 29 750 ₽.",
        "prices_regular_2026_27.online_grade10.year": "Фотон: цены на 2026/27 учебный год, 10 класс, онлайн, год — 47 250 ₽.",
    }
    price = apply_rule(
        registry["price"],
        plan={"primary_intent": "pricing", "direct_question": "Сколько стоит онлайн для 10 класса?", "active_brand": "foton"},
        facts=price_facts,
        context={"active_brand": "foton"},
    )
    assert price is not None
    assert "подтверждена такая стоимость" in price.text
    assert "По подтверждённым ценам" not in price.text
    assert "29 750 ₽" in price.text
    assert "47 250 ₽" in price.text
    assert _number_tokens(price.text) <= _number_tokens(" ".join(price_facts.values()))

    schedule_fact = "Математика, 11 класс, продвинутая группа, очно, Верхняя Красносельская, 30: суббота 10:00-12:00, старт 12.09.2026. Точное расписание конкретной группы уточняется."
    schedule = apply_rule(
        registry["schedule"],
        plan={"primary_intent": "schedule", "direct_question": "когда математика 11 класс очно?", "active_brand": "foton"},
        facts={"schedule_2026_27.groups.group_start_date_c13_krasnoselskaya_sat_1000_1200_math_11_advanced.client_safe_text": schedule_fact},
        context={"active_brand": "foton"},
    )
    assert schedule is not None
    assert "Нашёл такую группу" in schedule.text
    assert "По найденной группе" not in schedule.text
    assert "суббота 10:00-12:00" in schedule.text
    assert "12.09.2026" in schedule.text
    assert "Верхняя Красносельская, 30" in schedule.text
    assert _number_tokens(schedule.text) <= _number_tokens(schedule_fact)


def test_rules_engine_schedule_unpublished_group_goes_to_manager_check() -> None:
    rule = load_rules_registry()["schedule"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "schedule", "direct_question": "по каким дням химия 7 класс очно?", "active_brand": "unpk"},
        facts={
            "tg_unpk_verified_2026_05_21.client_facts.regular_courses_schedule_publication.client_safe_text": (
                "По курсам 2026/27 есть опубликованные группы с днями и временем; конкретный вариант зависит от класса, предмета, формата и площадки."
            )
        },
        context={"active_brand": "unpk"},
    )

    assert outcome is not None
    assert outcome.route == "draft_for_manager"
    assert "менеджер проверит класс, предмет, формат и площадку" in outcome.text
    assert "вторник" not in outcome.text.casefold()


def test_rules_engine_schedule_weekend_is_soft_guidance() -> None:
    rule = load_rules_registry()["schedule"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "schedule", "direct_question": "бывают по выходным?", "active_brand": "unpk"},
        facts={"objection_responses.inconvenient_time.1": "УНПК: по расписанию есть разные слоты по выходным."},
        context={"active_brand": "unpk"},
    )

    assert outcome is not None
    assert outcome.route == "bot_answer_self_for_pilot"
    assert "выходным" in outcome.text.casefold()
    assert "Точный день" in outcome.text


def test_rules_engine_schedule_start_and_weekly_lessons_from_facts() -> None:
    rule = load_rules_registry()["schedule"]
    start = apply_rule(
        rule,
        plan={"primary_intent": "schedule", "direct_question": "когда старт занятий в УНПК?", "active_brand": "unpk"},
        facts={"academic_year_2026_27.start": "УНПК: учебный год 2026/27, старт занятий — 12-27 сентября 2026 в зависимости от площадки."},
        context={"active_brand": "unpk"},
    )
    weekly = apply_rule(
        rule,
        plan={"primary_intent": "schedule", "direct_question": "сколько раз в неделю занятия?", "active_brand": "foton"},
        facts={"academic_year_2026_27.weekly_lessons": "Фотон: в учебном году 2026/27 занятия проходят 1 раз в неделю."},
        context={"active_brand": "foton"},
    )

    assert start is not None
    assert "12-27 сентября 2026" in start.text
    assert weekly is not None
    assert "1 раз в неделю" in weekly.text
    assert "2026 раз" not in weekly.text


def test_rules_engine_schedule_brand_split_does_not_mix_foton_groups_into_unpk() -> None:
    rule = load_rules_registry()["schedule"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "schedule", "direct_question": "когда математика 11 класс очно в УНПК?", "active_brand": "unpk"},
        facts={
            "schedule_2026_27.groups.foton_math_11.client_safe_text": (
                "Фотон: Математика, 11 класс, очно, Верхняя Красносельская, 30: суббота 10:00-12:00, старт 12.09.2026."
            )
        },
        context={"active_brand": "unpk"},
    )

    assert outcome is None


def test_rules_engine_docs_license_never_exposes_number() -> None:
    rule = load_rules_registry()["docs"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "document", "direct_question": "дайте номер лицензии", "active_brand": "foton"},
        facts={"licenses.client_safe_summary": "Фотон: у учебного центра есть лицензия на образовательную деятельность."},
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert "есть лицензия" in outcome.text.casefold()
    assert "номер лицензии" in outcome.text.casefold()
    assert not any(char.isdigit() for char in outcome.text)


def test_rules_engine_matkap_does_not_promise_sfr_approval_and_rejects_regional() -> None:
    rule = load_rules_registry()["matkap"]
    federal_facts = {
        "matkap.client_safe_text.when_asked": "Да, оплата материнским капиталом возможна. Работаем с федеральным маткапиталом. Менеджер поможет с оформлением через СФР.",
        "matkap.timeline.sfr_review_days": "СФР рассматривает заявление на оплату материнским капиталом до 10 рабочих дней.",
    }

    approval = apply_rule(
        rule,
        plan={"primary_intent": "matkap", "direct_question": "точно одобрят маткапитал?", "active_brand": "unpk"},
        facts=federal_facts,
        context={"active_brand": "unpk"},
    )
    regional = apply_rule(
        rule,
        plan={"primary_intent": "matkap", "direct_question": "региональный маткапитал примете?", "active_brand": "unpk"},
        facts={"matkap.client_safe_text.when_regional": "К сожалению, региональный маткапитал не принимаем. Если у вас федеральный — менеджер подскажет порядок оформления."},
        context={"active_brand": "unpk"},
    )

    assert approval is not None
    assert "не можем обещать одобрение" in approval.text
    assert regional is not None
    assert "региональный маткапитал не принимаем" in regional.text


def test_rules_engine_tax_license_and_fns_decision_are_not_guarantees() -> None:
    rule = load_rules_registry()["tax"]
    facts = {
        "licenses.client_safe_summary": "Фотон: у учебного центра есть лицензия на образовательную деятельность.",
        "tax_deduction.client_safe_text.when_asked": "Налоговый вычет за обучение возможен — у нас есть лицензия. Менеджер подготовит справку; решение и сроки выплаты остаются на стороне ФНС.",
    }

    license_outcome = apply_rule(
        rule,
        plan={"primary_intent": "tax", "direct_question": "номер лицензии для вычета?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    fns_outcome = apply_rule(
        rule,
        plan={"primary_intent": "tax", "direct_question": "точно вернут 13%?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert license_outcome is not None
    assert "лицензия" in license_outcome.text.casefold()
    assert "1151158" not in license_outcome.text
    assert fns_outcome is not None
    assert "возврат мы не гарантируем" in fns_outcome.text


def test_rules_engine_olympiad_only_confirms_9_and_11() -> None:
    rule = load_rules_registry()["olympiad"]
    facts = {
        "prices_regular_2026_27.online_olympiad_phystech_classes.client_safe_text": "Олимпиадная подготовка Физтех онлайн — для 9 и 11 классов; по другим классам возможность группы уточнит менеджер."
    }

    allowed = apply_rule(
        rule,
        plan={"primary_intent": "olympiad_online", "direct_question": "олимпиадная Физтех онлайн для 9 класса?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    outside = apply_rule(
        rule,
        plan={"primary_intent": "olympiad_online", "direct_question": "олимпиадная для 7 класса?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    regular = apply_rule(
        rule,
        plan={"primary_intent": "olympiad_online", "direct_question": "не олимпиадный, обычный онлайн для 9 класса", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )

    assert allowed is not None
    assert allowed.route == "bot_answer_self_for_pilot"
    assert outside is not None
    assert outside.route == "draft_for_manager"
    assert "для другого класса менеджер" in outside.text.casefold()
    assert regular is None


def test_rules_engine_platform_access_uses_fact_but_not_identity_branch() -> None:
    rule = load_rules_registry()["platform_access"]
    facts = {
        "presentation_format_facts_2026_05_21.client_facts.student_account_access.client_safe_text": "У ученика есть личный кабинет на учебной платформе. Если пароль забыт, его восстанавливают через кнопку «Забыли пароль».",
    }

    outcome = apply_rule(
        rule,
        plan={"primary_intent": "platform_access", "direct_question": "как зайти в личный кабинет?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    identity = apply_rule(
        rule,
        plan={"primary_intent": "platform_access", "direct_question": "ты бот? как зайти в кабинет?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert "личный кабинет" in outcome.text.casefold()
    assert identity is None


def test_rules_engine_installment_is_brand_split_and_unpk_uses_verified_fallback_facts() -> None:
    rule = load_rules_registry()["installment"]
    unpk_facts = {
        "payment_options.bank_installment.absent.client_safe_text": "В УНПК отдельной банковской рассрочки нет.",
        "payment_options.client_safe_text.when_asked_about_installment": "У нас оплата возможна помесячно, за семестр или за год.",
        "discounts.semester_payment.pct": "УНПК: при оплате за семестр действует скидка 10%.",
        "discounts.year_payment.pct": "УНПК: при оплате за год действует скидка 14%.",
    }

    unpk = apply_rule(
        rule,
        plan={"primary_intent": "installment", "direct_question": "У вас есть рассрочка?", "active_brand": "unpk"},
        facts=unpk_facts,
        context={"active_brand": "unpk"},
    )
    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "installment", "direct_question": "А у Фотона рассрочка есть?", "active_brand": "unpk"},
        facts=unpk_facts,
        context={"active_brand": "unpk"},
    )
    invoice_transfer = apply_rule(
        rule,
        plan={"primary_intent": "payment_method", "direct_question": "Можно банковским переводом на счёт?", "active_brand": "foton"},
        facts={"installment.foton": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями."},
        context={"active_brand": "foton"},
    )

    assert unpk is not None
    assert unpk.route == "bot_answer_self_for_pilot"
    assert "рассрочки нет" in unpk.text.casefold()
    assert "10%" in unpk.text
    assert "14%" in unpk.text
    assert "Фотон" not in unpk.text
    assert "Т-Банк" not in unpk.text
    assert "Долями" not in unpk.text
    assert cross_brand is None
    assert invoice_transfer is None


def test_rules_engine_installment_foton_uses_own_fact_without_unpk() -> None:
    rule = load_rules_registry()["installment"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "installment", "direct_question": "В Фотоне есть Долями или рассрочка?", "active_brand": "foton"},
        facts={"installment.foton": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями."},
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "bot_answer_self_for_pilot"
    assert "6, 10 или 12" in outcome.text
    assert "Долями" in outcome.text
    assert "УНПК" not in outcome.text


def test_rules_engine_selling_price_objection_foton_installment_uses_fact_numbers() -> None:
    rule = load_rules_registry()["installment"]
    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "installment",
            "direct_question": "Дороговато, можно частями?",
            "active_brand": "foton",
            "selling": {"objection": "price", "exit_signal": False},
        },
        facts={"installment.foton": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями."},
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert "rules_engine_selling_price_objection" in outcome.flags
    assert "6, 10 или 12 месяцев" in outcome.text
    assert "Долями" in outcome.text
    assert "Подобрать удобный вариант" in outcome.text
    assert "24" not in outcome.text
    assert "УНПК" not in outcome.text


def test_rules_engine_selling_gen_accepts_grounded_composition() -> None:
    rule = load_rules_registry()["installment"]
    facts = {"installment.foton": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями."}

    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "installment",
            "direct_question": "Серьёзная сумма для семьи, можно частями?",
            "active_brand": "foton",
            "selling": {"objection": "price", "exit_signal": False},
        },
        facts=facts,
        context={
            "active_brand": "foton",
            "selling_mode": "gen",
            "selling_compose_fn": lambda _prompt: {
                "text": (
                    "Понимаю, важно распределить оплату. В Фотоне оплату можно разбить на 6, 10 или 12 месяцев; "
                    "доступен сервис Долями. Подобрать удобный вариант?"
                )
            },
        },
    )

    assert outcome is not None
    assert "rules_engine_selling_gen_applied" in outcome.flags
    assert "rules_engine_selling_gen_fallback" not in outcome.flags
    assert outcome.metadata["selling"]["mode"] == "gen"
    assert outcome.metadata["selling"]["gen_applied"] is True
    assert "6, 10 или 12 месяцев" in outcome.text
    assert "Долями" in outcome.text


def test_rules_engine_selling_gen_falls_back_on_unsupported_number_pressure_or_brand() -> None:
    rule = load_rules_registry()["installment"]
    facts = {"installment.foton": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями."}
    plan = {
        "primary_intent": "installment",
        "direct_question": "Дорого, можно частями?",
        "active_brand": "foton",
        "selling": {"objection": "price", "exit_signal": False},
    }

    bad_outputs = (
        "Понимаю, можно оформить на 24 месяца.",
        "Только сегодня успейте оформить оплату на 6, 10 или 12 месяцев.",
        "В УНПК можно оплатить частями на 6, 10 или 12 месяцев.",
        "Исправим ребёнку оценку на пятёрку, если оформите оплату на 6 месяцев.",
    )
    for bad_text in bad_outputs:
        outcome = apply_rule(
            rule,
            plan=plan,
            facts=facts,
            context={"active_brand": "foton", "selling_mode": "gen", "selling_compose_fn": lambda _prompt, text=bad_text: {"text": text}},
        )

        assert outcome is not None
        assert "rules_engine_selling_gen_fallback" in outcome.flags
        assert "24" not in outcome.text
        assert "Только сегодня" not in outcome.text
        assert "УНПК" not in outcome.text
        assert "6, 10 или 12 месяцев" in outcome.text


def test_rules_engine_selling_det_mode_ignores_gen_composer() -> None:
    rule = load_rules_registry()["installment"]

    def _must_not_call(_prompt: str):
        raise AssertionError("det mode must not call selling composer")

    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "installment",
            "direct_question": "Дороговато, можно частями?",
            "active_brand": "foton",
            "selling": {"objection": "price", "exit_signal": False},
        },
        facts={"installment.foton": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями."},
        context={"active_brand": "foton", "selling_mode": "det", "selling_compose_fn": _must_not_call},
    )

    assert outcome is not None
    assert "rules_engine_selling_price_objection" in outcome.flags
    assert "rules_engine_selling_gen_applied" not in outcome.flags
    assert "rules_engine_selling_gen_fallback" not in outcome.flags


def test_rules_engine_selling_price_objection_unpk_uses_only_unpk_terms() -> None:
    rule = load_rules_registry()["installment"]
    facts = {
        "payment_options.bank_installment.absent.client_safe_text": "В УНПК отдельной банковской рассрочки нет.",
        "payment_options.client_safe_text.when_asked_about_installment": "УНПК: оплата возможна помесячно, за семестр или за год.",
        "discounts.semester_payment.pct": "УНПК: при оплате за семестр действует скидка 10%.",
        "discounts.year_payment.pct": "УНПК: при оплате за год действует скидка 14%.",
    }
    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "installment",
            "direct_question": "Дорого, есть рассрочка?",
            "active_brand": "unpk",
            "selling": {"objection": "price", "exit_signal": False},
        },
        facts=facts,
        context={"active_brand": "unpk"},
    )

    assert outcome is not None
    assert "rules_engine_selling_price_objection" in outcome.flags
    assert "помесячно" in outcome.text
    assert "за семестр" in outcome.text
    assert "за год" in outcome.text
    assert "10%" in outcome.text
    assert "14%" in outcome.text
    assert "Фотон" not in outcome.text
    assert "Т-Банк" not in outcome.text
    assert "Долями" not in outcome.text


def test_rules_engine_discount_second_subject_multichild_stacking_and_promocode_are_safe() -> None:
    rule = load_rules_registry()["discount"]
    facts = {
        "discounts.second_subject.online.pct": "Фотон: на второй онлайн-предмет действует скидка 30%.",
        "discounts.second_subject.offline.pct": "Фотон: на второй очный предмет действует скидка 20%.",
        "discounts.multichild.pct": "Фотон: многодетная скидка 10% по удостоверению многодетной семьи.",
        "discounts.stacking.rule": "Фотон: скидки не суммируются; применяется наибольшая доступная скидка.",
    }

    second = apply_rule(
        rule,
        plan={"primary_intent": "discount", "direct_question": "Сколько скидка на второй онлайн-предмет?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    stacking = apply_rule(
        rule,
        plan={"primary_intent": "discount", "direct_question": "Многодетная семья и второй предмет, скидки сложите?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    multichild = apply_rule(
        rule,
        plan={"primary_intent": "discount", "direct_question": "У нас двое детей, многодетная скидка есть?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    promocode = apply_rule(
        rule,
        plan={"primary_intent": "discount", "direct_question": "Дайте промокод на скидку", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "discount", "direct_question": "В УНПК многодетным 20%, а у вас сколько?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert second is not None
    assert "30%" in second.text
    assert "УНПК" not in second.text
    assert stacking is not None
    assert "не суммируются" in stacking.text.casefold()
    assert "наибольшая" in stacking.text.casefold()
    assert "40%" not in stacking.text
    assert multichild is not None
    assert "статус многодетной семьи" in multichild.text.casefold()
    assert promocode is not None
    assert "LVSH" not in promocode.text
    assert "ABRAMOV" not in promocode.text
    assert cross_brand is None


def test_rules_engine_discount_unpk_second_subject_and_period_are_brand_safe() -> None:
    rule = load_rules_registry()["discount"]
    facts = {
        "discounts.second_subject.online.pct": "УНПК: скидка на второй предмет составляет 20%.",
        "discounts.semester_payment.pct": "УНПК: при оплате за семестр действует скидка 10%.",
        "discounts.year_payment.pct": "УНПК: при оплате за год действует скидка 14%.",
    }

    second = apply_rule(
        rule,
        plan={"primary_intent": "discount", "direct_question": "Есть скидка на второй предмет?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    period = apply_rule(
        rule,
        plan={"primary_intent": "discount", "direct_question": "Какая скидка за семестр и за год?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )

    assert second is not None
    assert "20%" in second.text
    assert "Фотон" not in second.text
    assert period is not None
    assert "10%" in period.text
    assert "14%" in period.text
    assert "Фотон" not in period.text


def test_rules_engine_price_uses_format_and_grade_scoped_facts_only() -> None:
    rule = load_rules_registry()["price"]
    facts = {
        "prices_regular_2026_27.offline_5_11.semester": "УНПК: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 49 000 ₽.",
        "prices_regular_2026_27.offline_5_11.year": "УНПК: цены на 2026/27 учебный год, 5-11 класс, очно, год — 82 000 ₽.",
        "prices_regular_2026_27.online_5_11.semester": "УНПК: онлайн-курсы для 5-11 классов, формат 2 раза в неделю по 90 минут, семестр — 41 800 ₽.",
        "prices_regular_2026_27.online_5_11.year": "УНПК: онлайн-курсы для 5-11 классов, формат 2 раза в неделю по 90 минут, год — 69 900 ₽.",
    }

    online = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Сколько стоит онлайн для 9 класса?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    offline = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Сколько стоит очно для 9 класса?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    no_format = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Сколько стоит для 9 класса?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    wrong_grade = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Сколько стоит онлайн для 4 класса?", "active_brand": "unpk"},
        facts={"prices_regular_2026_27.online_5_11.semester": facts["prices_regular_2026_27.online_5_11.semester"]},
        context={"active_brand": "unpk"},
    )

    assert online is not None
    assert "41 800 ₽" in online.text
    assert "69 900 ₽" in online.text
    assert "49 000" not in online.text
    assert offline is not None
    assert "49 000 ₽" in offline.text
    assert "82 000 ₽" in offline.text
    assert "41 800" not in offline.text
    assert no_format is None
    assert wrong_grade is None


def test_rules_engine_price_cross_brand_and_missing_fact_do_not_invent() -> None:
    rule = load_rules_registry()["price"]
    facts = {
        "prices_regular_2026_27.online_5_11.semester": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, семестр — 29 750 ₽.",
    }

    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "В УНПК онлайн сколько стоит для 9 класса?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    missing_price = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Сколько стоит очно для 9 класса?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert cross_brand is None
    assert missing_price is None


def test_rules_engine_price_uses_period_from_fact_key_when_text_is_generic() -> None:
    rule = load_rules_registry()["price"]
    facts = {
        "prices_regular_2026_27.early_booking_2026_27.online_5_11_semester": (
            "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 29 750 ₽."
        ),
        "prices_regular_2026_27.early_booking_2026_27.online_5_11_year": (
            "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 47 250 ₽."
        ),
    }

    outcome = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Сколько стоит онлайн для 10 класса?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert "семестр — 29 750 ₽" in outcome.text
    assert "год — 47 250 ₽" in outcome.text
    assert "цена —" not in outcome.text


def test_rules_engine_selling_price_objection_price_adds_payment_step_only_on_signal() -> None:
    rule = load_rules_registry()["price"]
    facts = {
        "prices_regular_2026_27.online_5_11.semester": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, семестр — 29 750 ₽.",
        "prices_regular_2026_27.online_5_11.year": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, год — 47 250 ₽.",
        "installment.foton": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями.",
    }

    objection = apply_rule(
        rule,
        plan={
            "primary_intent": "pricing",
            "direct_question": "Дорого, сколько стоит онлайн для 10 класса?",
            "active_brand": "foton",
            "selling": {"objection": "price", "exit_signal": False},
        },
        facts=facts,
        context={"active_brand": "foton"},
    )
    plain = apply_rule(
        rule,
        plan={
            "primary_intent": "pricing",
            "direct_question": "Сколько стоит онлайн для 10 класса?",
            "active_brand": "foton",
            "selling": {"objection": "none", "exit_signal": False, "anxiety": True, "unmet_need": "мягче", "readiness": "ready"},
        },
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert objection is not None
    assert "29 750 ₽" in objection.text
    assert "47 250 ₽" in objection.text
    assert "6, 10 или 12 месяцев" in objection.text
    assert "Подсказать удобный вариант" in objection.text
    assert "rules_engine_selling_price_objection" in objection.flags

    assert plain is not None
    assert "29 750 ₽" in plain.text
    assert "47 250 ₽" in plain.text
    assert "Подсказать удобный вариант" not in plain.text
    assert "rules_engine_selling_price_objection" not in plain.flags
    assert "rules_engine_selling_anxiety" not in plain.flags
    assert "rules_engine_selling_unmet_need" not in plain.flags
    assert "rules_engine_selling_readiness" not in plain.flags


def test_rules_engine_coverage_two_format_price_is_flagged_and_uses_only_available_facts() -> None:
    rule = load_rules_registry()["price"]
    facts = {
        "prices.unpk.online_9.semester": "УНПК: 9 класс, онлайн, семестр — 41 800 ₽.",
        "prices.unpk.online_9.year": "УНПК: 9 класс, онлайн, год — 69 900 ₽.",
        "prices.unpk.offline_9.semester": "УНПК: 9 класс, очно, семестр — 52 000 ₽.",
    }

    off = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Онлайн и очно для 9 класса сколько стоят?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    both = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Онлайн и очно для 9 класса сколько стоят?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk", "coverage_enabled": True},
    )
    partial = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Онлайн и очно для 9 класса сколько стоят?", "active_brand": "unpk"},
        facts={key: value for key, value in facts.items() if "offline" not in key},
        context={"active_brand": "unpk", "coverage_enabled": True},
    )
    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "pricing", "direct_question": "Онлайн и очно для 9 класса сколько стоят?", "active_brand": "unpk"},
        facts={"prices.foton.online_9.semester": "Фотон: 9 класс, онлайн, семестр — 29 750 ₽."},
        context={"active_brand": "unpk", "coverage_enabled": True},
    )

    assert off is None
    assert both is not None
    assert "rules_engine_coverage_price_two_formats" in both.flags
    assert "41 800 ₽" in both.text
    assert "69 900 ₽" in both.text
    assert "52 000 ₽" in both.text
    assert "Фотон" not in both.text
    assert partial is not None
    assert "41 800 ₽" in partial.text
    assert "точную стоимость менеджер сверит отдельно" in partial.text
    assert "52 000 ₽" not in partial.text
    assert cross_brand is None


def test_rules_engine_coverage_multi_subject_price_does_not_invent_total_sum() -> None:
    rule = load_rules_registry()["price"]
    facts = {
        "prices.foton.online_10.semester": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, семестр — 29 750 ₽.",
        "discounts.second_subject.online.pct": "Фотон: на второй онлайн-предмет действует скидка 30%.",
    }

    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "pricing",
            "direct_question": "Сколько будет стоить физика и математика вместе за полгода онлайн для 10 класса?",
            "active_brand": "foton",
        },
        facts=facts,
        context={"active_brand": "foton", "coverage_enabled": True},
    )
    missing_price = apply_rule(
        rule,
        plan={
            "primary_intent": "pricing",
            "direct_question": "Сколько будет стоить физика и математика вместе за полгода для 10 класса?",
            "active_brand": "foton",
        },
        facts={"discounts.second_subject.online.pct": "Фотон: на второй онлайн-предмет действует скидка 30%."},
        context={"active_brand": "foton", "coverage_enabled": True},
    )

    assert outcome is not None
    assert "rules_engine_coverage_price_multi_subjects" in outcome.flags
    assert "29 750 ₽" in outcome.text
    assert "30%" in outcome.text
    assert "Итоговую сумму" in outcome.text
    assert "59 500" not in outcome.text
    assert "50 575" not in outcome.text
    assert missing_price is not None
    assert "30%" in missing_price.text
    assert "₽" not in missing_price.text
    assert "Итоговую сумму" in missing_price.text


def test_rules_engine_selling_full_signals_are_flagged_grounded_and_non_diagnostic() -> None:
    rule = load_rules_registry()["format_choice"]
    facts = {
        "formats.foton.online": "Фотон: онлайн-курсы проходят дистанционно.",
        "documents.license.foton": "Фотон: лицензия на образовательную деятельность № Л035-01234-77/00000000.",
        "teachers.foton": "Фотон: преподаватели — эксперты ЕГЭ и члены жюри олимпиад.",
        "process.enrollment.steps": "Фотон: менеджер уточнит класс, предмет и формат, затем поможет оформить заявку.",
    }

    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "format",
            "direct_question": "Боюсь, что зря потратим деньги: ребёнку физика тяжело даётся, но мы готовы записаться. Онлайн есть?",
            "active_brand": "foton",
            "selling": {
                "objection": "none",
                "exit_signal": False,
                "anxiety": True,
                "unmet_need": "ребёнку физика тяжело даётся",
                "readiness": "ready",
            },
        },
        facts=facts,
        context={"active_brand": "foton", "selling_mode": "det", "selling_signals_full": True},
    )

    assert outcome is not None
    assert "rules_engine_selling_anxiety" in outcome.flags
    assert "rules_engine_selling_unmet_need" in outcome.flags
    assert "rules_engine_selling_readiness" in outcome.flags
    assert "лицензия на образовательную деятельность" in outcome.text
    assert "Л035" not in outcome.text
    assert "физика тяжело" not in outcome.text
    assert "исправим" not in outcome.text.casefold()
    assert "пят" not in outcome.text.casefold()
    assert "менеджер уточнит класс" in outcome.text.casefold()
    assert "УНПК" not in outcome.text


def test_rules_engine_selling_readiness_without_enrollment_fact_does_not_invent_payment_process() -> None:
    rule = load_rules_registry()["format_choice"]
    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "format",
            "direct_question": "Онлайн есть? Уже готовы записаться, куда платить?",
            "active_brand": "foton",
            "selling": {
                "objection": "none",
                "exit_signal": False,
                "anxiety": False,
                "unmet_need": "",
                "readiness": "ready",
            },
        },
        facts={"formats.foton.online": "Фотон: онлайн-курсы проходят дистанционно."},
        context={"active_brand": "foton", "selling_mode": "det", "selling_signals_full": True},
    )

    assert outcome is not None
    assert "rules_engine_selling_readiness" in outcome.flags
    assert "куда платить" not in outcome.text.casefold()
    assert "оплат" not in outcome.text.casefold()
    assert "менеджер подтвердит порядок записи" in outcome.text


def test_rules_engine_format_choice_presents_only_verified_formats() -> None:
    rule = load_rules_registry()["format_choice"]
    both_facts = {
        "formats.unpk.online": "УНПК: онлайн-курсы проходят дистанционно.",
        "formats.unpk.offline": "УНПК: есть очные курсы на площадке.",
        "schedule.weekend_slots": "УНПК: бывают разные слоты по выходным.",
    }
    offline_only = {"formats.unpk.offline": "УНПК: есть очные курсы на площадке."}

    both = apply_rule(
        rule,
        plan={"primary_intent": "format", "direct_question": "Онлайн или очно можно учиться?", "active_brand": "unpk"},
        facts=both_facts,
        context={"active_brand": "unpk"},
    )
    single = apply_rule(
        rule,
        plan={"primary_intent": "format", "direct_question": "Онлайн или очно можно учиться?", "active_brand": "unpk"},
        facts=offline_only,
        context={"active_brand": "unpk"},
    )
    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "format", "direct_question": "А у Фотона онлайн или очно?", "active_brand": "unpk"},
        facts=both_facts,
        context={"active_brand": "unpk"},
    )

    assert both is not None
    assert "онлайн-формат" in both.text
    assert "очный формат" in both.text
    assert "формат удобнее выбрать вам" in both.text
    assert single is not None
    assert "очный формат" in single.text
    assert "онлайн-формат" not in single.text
    assert cross_brand is None


def test_rules_engine_trial_guards_free_offline_and_manager_request() -> None:
    rule = load_rules_registry()["trial"]
    facts = {"trial.foton.online_fragment": "Фотон: по онлайн-формату можно прислать фрагмент занятия, оформление дистанционное."}

    offline_free = apply_rule(
        rule,
        plan={"primary_intent": "trial", "fact_scope": "trial_offline", "direct_question": "Бесплатное очное пробное можно?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton", "known_slots": {"format": "очно"}},
    )
    manager = apply_rule(
        rule,
        plan={"primary_intent": "trial", "direct_question": "Передайте менеджеру", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    negated_online = apply_rule(
        rule,
        plan={"primary_intent": "trial", "direct_question": "Пробное хочу, но не онлайн", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert offline_free is not None
    assert offline_free.route == "draft_for_manager"
    assert "бесплатное пробное по умолчанию не обещаю" in offline_free.text.casefold()
    assert "онлайн-фрагмент" in offline_free.text
    assert manager is not None
    assert manager.route == "draft_for_manager"
    assert "фрагмент занятия" not in manager.text.casefold()
    assert negated_online is None


def test_rules_engine_trial_uses_active_brand_fragment_fact() -> None:
    rule = load_rules_registry()["trial"]
    outcome = apply_rule(
        rule,
        plan={"primary_intent": "trial", "fact_scope": "trial_online_fragment", "direct_question": "Как получить пробный онлайн-фрагмент?", "active_brand": "unpk"},
        facts={"trial.unpk.online_fragment": "УНПК: по онлайн-формату можно прислать фрагмент занятия для знакомства с подачей и уровнем."},
        context={"active_brand": "unpk", "known_slots": {"format": "онлайн"}},
    )
    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "trial", "direct_question": "А у Фотона такой фрагмент есть?", "active_brand": "unpk"},
        facts={"trial.unpk.online_fragment": "УНПК: по онлайн-формату можно прислать фрагмент занятия."},
        context={"active_brand": "unpk"},
    )

    assert outcome is not None
    assert outcome.route == "bot_answer_self_for_pilot"
    assert "фрагмент" in outcome.text.casefold()
    assert "Фотон" not in outcome.text
    assert cross_brand is None


def test_rules_engine_selling_exit_signal_is_grounded_or_neutral() -> None:
    rule = load_rules_registry()["format_choice"]
    format_fact = {"formats.foton.online": "Фотон: онлайн-курсы проходят дистанционно."}
    with_trial = {
        **format_fact,
        "trial.foton.online_fragment": "Фотон: по онлайн-формату можно прислать фрагмент занятия для знакомства с подачей.",
    }

    neutral = apply_rule(
        rule,
        plan={
            "primary_intent": "format",
            "direct_question": "Спасибо, подумаю. Онлайн есть?",
            "active_brand": "foton",
            "selling": {"objection": "none", "exit_signal": True},
        },
        facts=format_fact,
        context={"active_brand": "foton"},
    )
    grounded = apply_rule(
        rule,
        plan={
            "primary_intent": "format",
            "direct_question": "Спасибо, подумаю. Онлайн есть?",
            "active_brand": "foton",
            "selling": {"objection": "none", "exit_signal": True},
        },
        facts=with_trial,
        context={"active_brand": "foton"},
    )
    cross_brand = apply_rule(
        rule,
        plan={
            "primary_intent": "format",
            "direct_question": "А у УНПК онлайн есть?",
            "active_brand": "foton",
            "selling": {"objection": "none", "exit_signal": True},
        },
        facts=with_trial,
        context={"active_brand": "foton"},
    )

    assert neutral is not None
    assert "rules_engine_selling_exit_signal" in neutral.flags
    assert "Спокойно подумайте" in neutral.text
    assert "пробн" not in neutral.text.casefold()

    assert grounded is not None
    assert "rules_engine_selling_exit_signal" in grounded.flags
    assert "фрагмент занятия" in grounded.text
    assert "Подсказать, как записаться" in grounded.text

    assert cross_brand is None


def test_rules_engine_camp_live_seats_and_refund_stay_manager_only() -> None:
    rule = load_rules_registry()["camp_lvsh"]
    facts = {"camp.unpk.lvsh.seats": "УНПК: по ЛВШ места уже почти распроданы, наличие и запись проверяет живой менеджер."}

    seats = apply_rule(
        rule,
        plan={"primary_intent": "live_availability", "product_family": "camp", "direct_question": "Есть места на ЛВШ 20 июня?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk", "known_slots": {"grade": "10"}},
    )
    refund = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Оплатили лагерь, занятий нет, верните деньги", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )

    assert seats is not None
    assert seats.route == "draft_for_manager"
    assert "не буду обещать" in seats.text.casefold()
    assert "проверил наличие" in seats.text.casefold()
    assert refund is not None
    assert refund.route == "manager_only"
    assert "high_risk_manager_only" in refund.flags


def test_rules_engine_camp_brand_split_zvsh_and_scarcity_fact_are_safe() -> None:
    rule = load_rules_registry()["camp_lvsh"]
    unpk_facts = {
        "camp.unpk.lvsh.seats": "УНПК: по ЛВШ места уже почти распроданы, наличие и запись проверяет живой менеджер.",
        "camp.unpk.zvsh.status": "УНПК: даты ЗВШ пока уточняются, можно записаться в лист ожидания.",
    }
    foton_facts = {
        "camp.foton.lvsh.dates": "Фотон: ЛВШ Менделеево проходит 20-28 июня и 18-26 июля.",
    }

    scarcity = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Правда места почти распроданы в ЛВШ?", "active_brand": "unpk"},
        facts=unpk_facts,
        context={"active_brand": "unpk"},
    )
    zvsh = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Какие даты ЗВШ 2026/27?", "active_brand": "unpk"},
        facts=unpk_facts,
        context={"active_brand": "unpk"},
    )
    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Сравните ЛВШ Фотона и УНПК", "active_brand": "foton"},
        facts=foton_facts,
        context={"active_brand": "foton"},
    )
    non_camp = apply_rule(
        rule,
        plan={"primary_intent": "general_consultation", "direct_question": "Мне не лагерь, нужен обычный курс", "active_brand": "foton"},
        facts=foton_facts,
        context={"active_brand": "foton"},
    )

    assert scarcity is not None
    assert scarcity.route == "bot_answer_self_for_pilot"
    assert "почти распроданы" in scarcity.text
    assert "последний шанс" not in scarcity.text.casefold()
    assert "успейте" not in scarcity.text.casefold()
    assert zvsh is not None
    assert "лист ожидания" in zvsh.text.casefold()
    assert cross_brand is None
    assert non_camp is None


def test_rules_engine_camp_included_transfer_and_price_are_scoped_to_question() -> None:
    rule = load_rules_registry()["camp_lvsh"]
    facts = {
        "lvsh_mendeleevo_2026.program.total_academic_hours": "Фотон: ЛВШ Менделеево — 72+.",
        "lvsh_mendeleevo_2026.pricing.client_safe_text_when_price_asked": (
            "Фотон: ЛВШ Менделеево — ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽. Полная стоимость — 98 000 ₽."
        ),
        "lvsh_mendeleevo_2026.accommodation.room_capacity": "Фотон: ЛВШ Менделеево — 2-3 человека.",
        "lvsh_mendeleevo_2026.accommodation.meals_per_day": "Фотон: в ЛВШ Менделеево предусмотрено 5 приёмов пищи в день.",
        "lvsh_mendeleevo_2026.transfer_from_moscow.client_safe_text": (
            "Трансфер до ЛВШ Фотона включён в стоимость; ориентир места сбора — метро Ховрино, точные детали отправляем перед сменой."
        ),
    }

    included = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Что входит в ЛВШ Менделеево?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    transfer = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Трансфер из Москвы есть?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    price = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Сколько стоит ЛВШ Менделеево?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert included is not None
    assert included.subvariant == "included_composition"
    assert "проживание" in included.text.casefold()
    assert "питание" in included.text.casefold()
    assert "трансфер" in included.text.casefold()
    assert "72+" not in included.text

    assert transfer is not None
    assert transfer.subvariant == "transfer"
    assert "Трансфер до ЛВШ Фотона включён в стоимость" in transfer.text
    assert "Ховрино" in transfer.text
    assert "93 100" not in transfer.text
    assert "98 000" not in transfer.text

    assert price is not None
    assert price.subvariant == "price"
    assert "ЛВШ Менделеево — ЛВШ Менделеево" not in price.text
    assert "Фотон:" not in price.text
    assert "ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽. Полная стоимость — 98 000 ₽." in price.text


def test_rules_engine_camp_scoped_fix_keeps_brand_and_p0_boundaries() -> None:
    rule = load_rules_registry()["camp_lvsh"]
    facts = {
        "lvsh_mendeleevo_2026.pricing.client_safe_text_when_price_asked": (
            "Фотон: ЛВШ Менделеево — ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽. Полная стоимость — 98 000 ₽."
        ),
        "tg_unpk.lvsh.transfer.client_safe_text": (
            "Трансфер на выездную школу включён в стоимость: сбор у метро Ховрино, отъезд в 11:00; обратно отъезд с базы в 15:30."
        ),
    }

    cross_brand = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "У Фотона что входит в ЛВШ?", "active_brand": "unpk"},
        facts=facts,
        context={"active_brand": "unpk"},
    )
    p0 = apply_rule(
        rule,
        plan={"primary_intent": "camp", "product_family": "camp", "direct_question": "Оплатили ЛВШ, недовольны, верните деньги", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert cross_brand is None
    assert p0 is not None
    assert p0.route == "manager_only"
    assert "high_risk_manager_only" in p0.flags


def test_rules_engine_enrollment_presale_refund_vs_real_p0_and_dolyami() -> None:
    rule = load_rules_registry()["enrollment_process"]
    facts = {
        "refund_post_payment.client_safe_text": "Фотон: возвращается остаток неистраченных средств.",
        "process.enrollment.steps": "Фотон: для записи менеджер уточнит класс, предмет, формат и подходящую группу, затем поможет оформить заявку.",
    }

    presale = apply_rule(
        rule,
        plan={"primary_intent": "refund_policy", "direct_question": "Если передумаю до оплаты, деньги вернёте?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    real_refund = apply_rule(
        rule,
        plan={"primary_intent": "enrollment_process", "direct_question": "Я оплатил, занятий нет, верните деньги", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    dolyami = apply_rule(
        rule,
        plan={"primary_intent": "enrollment_process", "direct_question": "Оформить можно через Долями?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )
    process = apply_rule(
        rule,
        plan={"primary_intent": "enrollment_process", "direct_question": "Как записаться на курс?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert presale is not None
    assert presale.route == "bot_answer_self_for_pilot"
    assert "остаток неистраченных средств" in presale.text
    assert "полный возврат" not in presale.text.casefold()
    assert real_refund is not None
    assert real_refund.route == "manager_only"
    assert dolyami is None
    assert process is not None
    assert "менеджер уточнит класс" in process.text.casefold()


def test_rules_engine_selling_does_not_override_real_refund_p0() -> None:
    rule = load_rules_registry()["enrollment_process"]
    outcome = apply_rule(
        rule,
        plan={
            "primary_intent": "enrollment_process",
            "direct_question": "Дорого, я оплатил, занятий нет, верните деньги",
            "active_brand": "foton",
            "selling": {
                "objection": "price",
                "exit_signal": True,
                "anxiety": True,
                "unmet_need": "ребёнку нужна поддержка",
                "readiness": "ready",
            },
        },
        facts={"process.enrollment.steps": "Фотон: менеджер помогает оформить заявку."},
        context={"active_brand": "foton", "selling_signals_full": True},
    )

    assert outcome is not None
    assert outcome.route == "manager_only"
    assert "high_risk_manager_only" in outcome.flags
    assert "rules_engine_selling_price_objection" not in outcome.flags
    assert "rules_engine_selling_exit_signal" not in outcome.flags
    assert "rules_engine_selling_anxiety" not in outcome.flags
    assert "rules_engine_selling_unmet_need" not in outcome.flags
    assert "rules_engine_selling_readiness" not in outcome.flags
