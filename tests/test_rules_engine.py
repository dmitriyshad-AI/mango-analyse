from __future__ import annotations

from mango_mvp.channels.rules_engine import MIGRATED, apply_rule, load_rules_registry, select_rule


def test_rules_registry_loads_approved_migrated_rules() -> None:
    registry = load_rules_registry()

    assert len(registry) == 16
    assert set(MIGRATED) == {
        "teacher",
        "recordings",
        "contact_address",
        "docs",
        "matkap",
        "tax",
        "olympiad",
        "platform_access",
    }
    assert select_rule("teacher", registry).rule_id == "teacher"  # type: ignore[union-attr]
    assert select_rule("recording", registry).rule_id == "recordings"  # type: ignore[union-attr]
    assert select_rule("address", registry).rule_id == "contact_address"  # type: ignore[union-attr]
    assert select_rule("document", registry).rule_id == "docs"  # type: ignore[union-attr]
    assert select_rule("matkap", registry).rule_id == "matkap"  # type: ignore[union-attr]
    assert select_rule("tax", registry).rule_id == "tax"  # type: ignore[union-attr]
    assert select_rule("olympiad_online", registry).rule_id == "olympiad"  # type: ignore[union-attr]
    assert select_rule("platform_access", registry).rule_id == "platform_access"  # type: ignore[union-attr]
    assert select_rule("pricing", registry) is None


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
    assert "Скорняжный" in outcome.text
    assert "УНПК" not in outcome.text


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
