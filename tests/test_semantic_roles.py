from __future__ import annotations

import pytest

from mango_mvp.channels.answer_plan import build_answer_plan
from mango_mvp.channels.semantic_roles import tag_message_roles


@pytest.mark.parametrize(
    ("message", "field", "expected"),
    (
        ("Подскажите по точной дате старта курса", "training_format", ""),
        ("Хочу заниматься очно в классе", "training_format", "ochno"),
        ("Можно онлайн дистанционно из дома?", "training_format", "online"),
        ("Как записаться на курс по физике?", "enrollment_vs_recording", "enroll"),
        ("Хочу оформиться к вам", "enrollment_vs_recording", "enroll"),
        ("Будет ли запись урока, если пропущу занятие?", "enrollment_vs_recording", "recording"),
        ("Я пропустил вебинар, можно пересмотреть?", "enrollment_vs_recording", "recording"),
        ("Переведите меня на менеджера, пожалуйста", "transfer_sense", "manager"),
        ("Можно оплатить банковским переводом на счёт?", "transfer_sense", "money"),
        ("Можно перевести ребёнка в другую группу?", "transfer_sense", "group"),
        ("А Долями можно оплатить?", "payment_method", "dolyami"),
        ("Есть рассрочка на год?", "payment_method", "rassrochka"),
        ("Можно оплатить материнским капиталом?", "payment_source", "matkap"),
        ("Дадите справку на налоговый вычет?", "payment_source", "tax_deduction"),
        ("А если передумаю до начала, деньги вернут?", "refund_frame", "presale_policy"),
        ("Перед оплатой хочу понять условия возврата", "refund_frame", "presale_policy"),
        ("Если ребёнку не понравится, можно вернуть деньги?", "refund_frame", "presale_policy"),
        ("Верните деньги за курс немедленно", "refund_frame", "dispute"),
        ("Я уже оплатил, хочу возврат", "refund_frame", "dispute"),
    ),
)
def test_semantic_roles_axes(message: str, field: str, expected: str) -> None:
    assert getattr(tag_message_roles(message), field) == expected


def test_semantic_roles_do_not_treat_multichild_as_schedule() -> None:
    roles = tag_message_roles("Есть ли скидка для многодетной семьи?")

    assert "discount" in roles.topics
    assert "schedule" not in roles.topics


@pytest.mark.parametrize(
    ("message", "topic"),
    (
        ("Сколько стоит курс?", "price"),
        ("Есть скидка?", "discount"),
        ("Можно пробное занятие?", "trial"),
        ("Лагерь в Менделеево есть?", "camp"),
        ("Какое расписание?", "schedule"),
        ("Какой адрес?", "address"),
        ("Нужна справка об обучении", "document"),
        ("Вы бот или человек?", "identity"),
        ("Чем айфон 17 лучше 13?", "off_topic"),
        ("Можно оплатить маткапиталом?", "matkap"),
        ("Как получить налоговый вычет?", "tax"),
    ),
)
def test_semantic_roles_topic_flags(message: str, topic: str) -> None:
    assert topic in tag_message_roles(message).topics


def test_semantic_roles_collect_multitopic_answer_topics() -> None:
    plan = build_answer_plan(tag_message_roles("Сколько стоит и можно ли в рассрочку?"))

    assert "price" in plan.answer_topics
    assert "installment" in plan.answer_topics


def test_semantic_roles_format_plus_schedule_multitopic() -> None:
    roles = tag_message_roles("Это онлайн? И по каким дням занятия?")
    plan = build_answer_plan(roles)

    assert roles.training_format == "online"
    assert "schedule" in plan.answer_topics


def test_semantic_roles_matkap_installment_forbidden_pair() -> None:
    plan = build_answer_plan(tag_message_roles("Можно маткапиталом и сразу в рассрочку?"))

    assert "matkap+installment" in plan.forbidden_pairs


def test_answer_plan_template_only_as_fallback() -> None:
    empty = build_answer_plan(tag_message_roles(""), substantive_answer_present=False)
    price = build_answer_plan(tag_message_roles("Сколько стоит курс?"), substantive_answer_present=True)
    presale = build_answer_plan(tag_message_roles("А если передумаю до начала, деньги вернут?"))
    dispute = build_answer_plan(tag_message_roles("Верните деньги, я уже оплатил"))
    external = build_answer_plan(tag_message_roles("Сколько стоит курс?"), external_p0=True)

    assert empty.template_allowed is True
    assert price.template_allowed is False
    assert "price" in price.answer_topics
    assert presale.p0_required is False
    assert presale.route == "bot_answer_self"
    assert dispute.p0_required is True
    assert dispute.route == "manager_only"
    assert external.p0_required is True
