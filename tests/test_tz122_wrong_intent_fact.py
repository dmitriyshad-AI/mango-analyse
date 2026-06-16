from __future__ import annotations

from mango_mvp.channels.dialogue_contract_pipeline import (
    AnswerContract,
    Subquestion,
    verify_output as verify_dialogue_contract_output,
)


FLAG = "TELEGRAM_WRONG_INTENT_FACT_CALIBRATION"


def _wrong_intent_details(
    draft: str,
    *,
    facts: dict[str, str],
    contract: AnswerContract,
    client_message: str,
    context: dict[str, object] | None = None,
) -> tuple[str, ...]:
    findings = verify_dialogue_contract_output(
        draft,
        facts=facts,
        active_brand=contract.active_brand,
        contract=contract,
        client_message=client_message,
        context=context,
    )
    return tuple(finding.detail for finding in findings if finding.code == "wrong_intent_fact")


def test_tz122_wrong_intent_flag_off_keeps_address_guard_behavior() -> None:
    facts = {
        "locations_foton.addresses.1.address": "Фотон: адрес очных занятий — Москва, Верхняя Красносельская ул., 30.",
    }
    details = _wrong_intent_details(
        "Занятия проходят по адресу: Москва, Верхняя Красносельская ул., 30.",
        facts=facts,
        contract=AnswerContract(active_brand="foton", answerability="answer_self"),
        client_message="где очные курсы для 7 класса?",
        context={
            "active_brand": "foton",
            "answer_contract": {
                "direct_question": "где очные курсы для 7 класса?",
                "primary_intent": "address",
                "required_fact_keys": ["locations"],
            },
        },
    )

    assert any("Адресный факт" in detail for detail in details)


def test_tz122_address_fact_allowed_when_address_core_is_requested() -> None:
    facts = {
        "locations_foton.addresses.1.address": "Фотон: адрес очных занятий — Москва, Верхняя Красносельская ул., 30.",
    }
    details = _wrong_intent_details(
        "Занятия проходят по адресу: Москва, Верхняя Красносельская ул., 30.",
        facts=facts,
        contract=AnswerContract(active_brand="foton", answerability="answer_self"),
        client_message="где очные курсы для 7 класса?",
        context={
            FLAG: True,
            "active_brand": "foton",
            "answer_contract": {
                "direct_question": "где очные курсы для 7 класса?",
                "primary_intent": "address",
                "required_fact_keys": ["locations"],
            },
        },
    )

    assert not any("Адресный факт" in detail for detail in details)


def test_tz122_address_fact_still_rejected_for_price_question() -> None:
    facts = {
        "locations_foton.addresses.1.address": "Фотон: адрес очных занятий — Москва, Верхняя Красносельская ул., 30.",
    }
    details = _wrong_intent_details(
        "Занятия проходят по адресу: Москва, Верхняя Красносельская ул., 30.",
        facts=facts,
        contract=AnswerContract(active_brand="foton", answerability="answer_self"),
        client_message="сколько стоит онлайн-курс по математике?",
        context={
            FLAG: True,
            "active_brand": "foton",
            "answer_contract": {
                "direct_question": "сколько стоит онлайн-курс по математике?",
                "primary_intent": "pricing",
                "required_fact_keys": ["pricing"],
            },
        },
    )

    assert any("Адресный факт" in detail for detail in details)


def test_tz122_broad_address_phrase_covers_where_courses_question() -> None:
    facts = {
        "locations_foton.addresses.1.address": "Фотон: адрес очных занятий — Москва, Верхняя Красносельская ул., 30.",
    }
    details = _wrong_intent_details(
        "Занятия проходят по адресу: Москва, Верхняя Красносельская ул., 30.",
        facts=facts,
        contract=AnswerContract(active_brand="foton", answerability="answer_self"),
        client_message="где курсы для 7 класса?",
        context={FLAG: True, "active_brand": "foton"},
    )

    assert not any("Адресный факт" in detail for detail in details)


def test_tz122_start_dates_are_not_demoted_by_location_scoped_fact() -> None:
    facts = {
        "academic_year_2026_27.start_by_location.moscow": (
            "Фотон: учебный год 2026/27, Москва — 12-13 сентября 2026."
        ),
        "academic_year_2026_27.start_by_location.online": (
            "Фотон: учебный год 2026/27, онлайн — 19-20 сентября 2026."
        ),
    }
    details = _wrong_intent_details(
        "Учебный год 2026/27 в Фотоне начинается в сентябре 2026 года: "
        "очные занятия в Москве стартуют 12-13 сентября 2026, онлайн-занятия — 19-20 сентября 2026.",
        facts=facts,
        contract=AnswerContract(
            active_brand="foton",
            current_question="Когда начинается учебный год? Какие даты старта?",
            answerability="answer_self",
            subquestions=(
                Subquestion(
                    text="Когда начинается учебный год? Какие даты старта?",
                    answerable="self",
                    needed_fact_keys=("academic_year_2026_27.start_by_location",),
                ),
            ),
        ),
        client_message="Когда начинается учебный год? Какие даты старта?",
        context={FLAG: True, "active_brand": "foton"},
    )

    assert not any("Адресный факт" in detail for detail in details)


def test_tz122_camp_shift_question_is_not_demoted_when_scope_does_not_conflict() -> None:
    facts = {
        "camp.foton.lvsh.price": "ЛВШ Менделеево: стоимость смены — 49 000 ₽.",
    }
    details = _wrong_intent_details(
        "По ЛВШ Менделеево стоимость смены — 49 000 ₽.",
        facts=facts,
        contract=AnswerContract(active_brand="foton", current_question="сколько стоит смена", answerability="answer_self"),
        client_message="сколько стоит смена?",
        context={FLAG: True, "active_brand": "foton"},
    )

    assert not any("Лагерный" in detail for detail in details)


def test_tz122_camp_scope_mismatch_still_demotes() -> None:
    facts = {
        "camp.foton.lvsh.price": "ЛВШ Менделеево: стоимость смены с проживанием — 49 000 ₽.",
    }
    details = _wrong_intent_details(
        "По ЛВШ Менделеево стоимость смены с проживанием — 49 000 ₽.",
        facts=facts,
        contract=AnswerContract(
            active_brand="foton",
            current_question="сколько стоит городской лагерь без проживания?",
            answerability="answer_self",
            subquestions=(
                Subquestion(
                    text="сколько стоит городской лагерь без проживания?",
                    answerable="self",
                    needed_fact_keys=("camp",),
                ),
            ),
        ),
        client_message="сколько стоит городской лагерь без проживания?",
        context={FLAG: True, "active_brand": "foton"},
    )

    assert any("Лагерный" in detail for detail in details)


def test_tz122_camp_scope_uses_fact_visible_in_draft_not_adjacent_fact_pack() -> None:
    facts = {
        "camp.foton.city_day.price": "Городской дневной лагерь без проживания: стоимость смены — 34 300 ₽.",
        "camp.foton.lvsh.price": "ЛВШ Менделеево: стоимость смены с проживанием — 49 000 ₽.",
    }
    details = _wrong_intent_details(
        "По городскому дневному лагерю без проживания стоимость смены — 34 300 ₽.",
        facts=facts,
        contract=AnswerContract(
            active_brand="foton",
            current_question="сколько стоит городской лагерь без проживания?",
            answerability="answer_self",
            subquestions=(
                Subquestion(
                    text="сколько стоит городской лагерь без проживания?",
                    answerable="self",
                    needed_fact_keys=("camp",),
                ),
            ),
        ),
        client_message="сколько стоит городской лагерь без проживания?",
        context={FLAG: True, "active_brand": "foton"},
    )

    assert not any("Лагерный" in detail for detail in details)


def test_tz122_contact_hours_as_class_schedule_still_demotes() -> None:
    facts = {
        "contacts.foton.schedule": "Фотон: офис на связи Пн-Вс с 10:00 до 18:00.",
    }
    details = _wrong_intent_details(
        "Расписание занятий: Пн-Вс с 10:00 до 18:00.",
        facts=facts,
        contract=AnswerContract(
            active_brand="foton",
            current_question="по каким дням занятия в группе?",
            answerability="answer_self",
            subquestions=(
                Subquestion(
                    text="по каким дням занятия в группе?",
                    answerable="self",
                    needed_fact_keys=("class_schedule",),
                ),
            ),
        ),
        client_message="по каким дням занятия в группе?",
        context={FLAG: True, "active_brand": "foton"},
    )

    assert any("Контактные часы" in detail for detail in details)
