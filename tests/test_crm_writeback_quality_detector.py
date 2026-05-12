from __future__ import annotations

from pathlib import Path

import pytest

from mango_mvp.quality.crm_writeback_quality_detector import detect_crm_writeback_quality_risks


@pytest.mark.parametrize(
    "text, expected_risk",
    [
        (
            "Представитель компании Деламачка, интегратора АМА CRM, звонил по вопросу установленного виджета в amoCRM.",
            "out_of_domain_b2b",
        ),
        (
            "Клиент позвонил по вакансии программиста и попросил соединить с HR.",
            "out_of_domain_b2b",
        ),
        (
            "Клиент уточнял, с кем можно связаться, чтобы предложить услуги по сайту.",
            "out_of_domain_b2b",
        ),
        (
            "Абонент позвонил не по адресу и уточнил про Тетрайдер, упаковочная компания.",
            "wrong_direction_or_no_content",
        ),
        (
            "Звонок от Пенсионного фонда Долгопрудный по вопросу корректировки по ЭЭДО.",
            "out_of_domain_b2b",
        ),
        (
            "В звонке сработало голосовое меню социального казначейства Москвы.",
            "technical_ivr_or_service_check",
        ),
        (
            "Звонок связан со служебной проверкой телефонии Mango Office, отдел колл-трекинга.",
            "technical_ivr_or_service_check",
        ),
        (
            "Содержательного диалога с клиентом не было: в записи только служебное приветствие контактного центра.",
            "no_content_or_no_edtech_intent",
        ),
        (
            "Номер относится к исходящей связи сервиса Яндекс.Справочник. Это техническая проверка, не заявка.",
            "technical_ivr_or_service_check",
        ),
        (
            "Клиент сообщил, что зашел не на тот сайт и ему нужен другой ресурс; заявка не относится к учебному центру.",
            "out_of_domain_b2b",
        ),
        (
            "Представитель образовательной платформы предлагает сотрудничество и расширяет пул партнеров; заявки на обучение нет.",
            "out_of_domain_b2b",
        ),
        (
            "Корпоративный номер относится к Ростелекому; собеседник не может определить, кто звонил, и просит удалить номер из базы.",
            "out_of_domain_b2b",
        ),
        (
            "Разговор не состоялся, клиент не поддержал диалог; тема обращения, продукт, ученик и следующий шаг не установлены.",
            "no_content_or_no_edtech_intent",
        ),
        (
            "Контакт не подтвердился: менеджер звонил по летней школе, но на линии была не та Светлана, произошла путаница с именем.",
            "wrong_direction_or_no_content",
        ),
        (
            "Обсуждение программы, интереса к продукту и следующих шагов не состоялось.",
            "no_content_or_no_edtech_intent",
        ),
    ],
)
def test_detector_blocks_known_crm_noise_classes(text: str, expected_risk: str) -> None:
    findings = detect_crm_writeback_quality_risks(text)

    assert {finding.risk_type for finding in findings} >= {expected_risk}


@pytest.mark.parametrize(
    "text",
    [
        "Клиент интересовался летним выездным лагерем для ребенка из 9 класса по физико-математическому направлению.",
        "Клиентка обратилась по записи на онлайн-группу по профильной математике ЕГЭ для 11 класса.",
        "Обратилась по заявке на онлайн-обучение для 9 класса, менеджер предложила подготовку к ОГЭ по математике.",
        "Клиент подтвердил оплату курса по информатике для ученика 10 класса и уточнил доступ к занятиям.",
        "Холодный B2B-звонок по вопросу сотрудничества в образовательных проектах для школьников.",
        "Клиент обсуждал онлайн-занятия для 10 класса через МТС Линк, уточнил расписание курса по физике.",
        "Клиент обсуждал курс по математике для ребенка, но сейчас ребенку не хватает времени на занятия.",
        "Клиент обсуждал курс, но понял, что это не та программа: нужен олимпиадный формат по математике.",
    ],
)
def test_detector_allows_edtech_sales_and_service_context(text: str) -> None:
    assert detect_crm_writeback_quality_risks(text) == []


def test_detector_is_independent_from_export_builder() -> None:
    source = Path("src/mango_mvp/quality/crm_writeback_quality_detector.py").read_text(encoding="utf-8")

    assert "build_post_backfill_amo_ready_export" not in source
    assert "scripts." not in source
