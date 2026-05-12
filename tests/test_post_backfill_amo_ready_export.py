from __future__ import annotations

from datetime import date

from scripts import build_post_backfill_amo_ready_export as builder


def test_low_value_wrong_number_summary_is_not_crm_contentful() -> None:
    assert builder._is_low_value_for_crm(
        {
            "history_summary": (
                "Содержательного диалога не состоялось: после приветствия получен только краткий ответ. "
                "Запрос, предмет обращения и следующий шаг не были озвучены."
            )
        }
    )


def test_low_value_ivr_out_of_domain_and_no_content_are_not_crm_contentful() -> None:
    bad_summaries = [
        "Вместо содержательного диалога воспроизвелось сообщение санитарно-гигиенической компании.",
        "Номер оказался ошибочным: после приветствия клиент ответил «Ростелеком».",
        "Номер из файла оказался телефоном отделения Социального фонда.",
        "Клиент заявил, что не оставлял заявку на обратный звонок.",
        "Звонок от компании Тетра Транс по поводу контейнерных перевозок.",
        "Содержание обращения не уточнено. Запрос, интерес и дальнейшие договоренности не определены.",
        "Содержательного запроса не прозвучало, ответ клиента не позволил определить тему обращения.",
        "Клиент сообщил, что не звонил в учебный центр.",
        "Содержательного обсуждения не было. Клиент не подтвердил интерес.",
        "Звонок был не по теме EdTech: клиент предлагал продажу и обслуживание трансформаторов и КТП.",
        "Запись содержит только обрывок фразы без понятного запроса.",
        "Клиент звонил по письму на имя директора по поводу поставок оборудования.",
        "Клиент сообщил, что с доступом к интернету сейчас все стабильно.",
        "Представилась партнером компании Яндекс, обсуждение Яндекс Бизнес.",
        "Клиент из команды С5, партнера интегратора МСРМ.",
        "Обсуждался запрос по закупке сувенирной продукции для лагеря: футболки и кепки.",
        "Представитель компании спецоператора обратился по поводу продления ФИСФРДО.",
        "Звонок оказался не по теме учебного центра: клиент уточнил вопрос по корпоративным подаркам.",
        "Номер указан как технический и не используется для обратной связи.",
        "Звонок не относится к целевому обращению: содержательного разговора о курсах не было.",
        "Передать обращение в отдел маркетинга по вопросу продления по номеру CT3608.",
    ]

    for summary in bad_summaries:
        assert builder._is_low_value_for_crm({"history_summary": summary}), summary


def test_crm_text_redacts_callback_phones_without_touching_short_codes() -> None:
    text = "Отправить в WhatsApp по номеру +7 996 417-04-23, номер бронирования 64-64-58 оставить."

    sanitized = builder._sanitize_crm_text(text)

    assert "+7 996 417-04-23" not in sanitized
    assert "[PHONE]" in sanitized
    assert "64-64-58" in sanitized


def test_crm_text_does_not_redact_dates_or_time_ranges_as_phones() -> None:
    text = "28.11.2025 17:35 менеджер договорился перезвонить 2026-05-10 17:00."

    sanitized = builder._sanitize_crm_text(text)

    assert sanitized == text


def test_contact_history_uses_contentful_calls_only() -> None:
    rows = [
        {
            "Дата и время звонка": "2026-05-02 10:00:00",
            "Телефон клиента": "+79990000000",
            "Менеджер": "Менеджер",
            "Свежий период": "Да",
            "Статус Analyze": "done",
            "Содержательный звонок": "Нет",
            "Нужна ручная проверка": "Нет",
            "Краткое резюме разговора": "Автоответчик, разговора не было.",
            "Тип звонка": "non_conversation",
        },
        {
            "Дата и время звонка": "2026-05-01 10:00:00",
            "Телефон клиента": "+79990000000",
            "Менеджер": "Менеджер",
            "Свежий период": "Да",
            "Статус Analyze": "done",
            "Содержательный звонок": "Да",
            "Нужна ручная проверка": "Нет",
            "Краткое резюме разговора": "Клиент интересуется годовым курсом по математике.",
            "Тип звонка": "sales_call",
            "Рекомендуемая дата следующего контакта": "2026-05-03",
            "Приоритет лида": "warm",
            "Вероятность продажи, %": "70",
            "Следующий шаг": "Перезвонить",
        },
    ]

    contacts, amo_rows, manual_review = builder._build_contact_rows(
        call_rows=rows,
        chains_by_phone={"+79990000000": {"objections_top": "ошибочный номер: 2", "amo_contact_ids": "123"}},
        analysis_date=date(2026, 5, 10),
    )

    assert len(contacts) == 1
    assert len(amo_rows) == 1
    assert manual_review == []
    assert "Автоответчик" not in contacts[0]["Краткая история общения"]
    assert "ошибочный номер" not in contacts[0]["Краткая история общения"]
    assert "ошибочный номер" not in contacts[0]["Возражения"]
    assert "Перезвонить" in contacts[0]["Краткая история общения"]
    assert contacts[0]["Готово к записи в AMO"] == "Да"


def test_truncated_crm_history_is_manual_review_not_amo_ready() -> None:
    rows = [
        {
            "Дата и время звонка": "2026-05-01 10:00:00",
            "Телефон клиента": "+79990000000",
            "Менеджер": "Менеджер",
            "Свежий период": "Да",
            "Статус Analyze": "done",
            "Содержательный звонок": "Да",
            "Нужна ручная проверка": "Нет",
            "Краткое резюме разговора": "Клиент интересуется летним лагерем, но резюме обрывается...",
            "Тип звонка": "sales_call",
            "Следующий шаг": "Перезвонить",
        }
    ]

    contacts, amo_rows, manual_review = builder._build_contact_rows(
        call_rows=rows,
        chains_by_phone={"+79990000000": {"amo_contact_ids": "123"}},
        analysis_date=date(2026, 5, 10),
    )

    assert contacts[0]["Готово к записи в AMO"] == "Нет"
    assert "многоточием" in contacts[0]["Причина статуса AMO"]
    assert amo_rows == []
    assert len(manual_review) == 1


def test_service_call_is_not_live_amo_ready_but_stays_in_contacts() -> None:
    rows = [
        {
            "Дата и время звонка": "2026-05-01 10:00:00",
            "Телефон клиента": "+79990000000",
            "Менеджер": "Менеджер",
            "Свежий период": "Да",
            "Статус Analyze": "done",
            "Содержательный звонок": "Да",
            "Нужна ручная проверка": "Нет",
            "Краткое резюме разговора": "Действующий клиент уточнил справку для налогового вычета.",
            "Тип звонка": "existing_client_progress",
            "Следующий шаг": "Отправить справку",
        }
    ]

    contacts, amo_rows, manual_review = builder._build_contact_rows(
        call_rows=rows,
        chains_by_phone={"+79990000000": {"amo_contact_ids": "123"}},
        analysis_date=date(2026, 5, 10),
    )

    assert len(contacts) == 1
    assert contacts[0]["Готово к записи в AMO"] == "Нет"
    assert "service/existing-client" in contacts[0]["Причина статуса AMO"]
    assert amo_rows == []
    assert len(manual_review) == 1


def test_orphan_without_single_amo_contact_id_is_not_live_amo_ready() -> None:
    rows = [
        {
            "Дата и время звонка": "2026-05-01 10:00:00",
            "Телефон клиента": "+79990000000",
            "Менеджер": "Менеджер",
            "Свежий период": "Да",
            "Статус Analyze": "done",
            "Содержательный звонок": "Да",
            "Нужна ручная проверка": "Нет",
            "Краткое резюме разговора": "Клиент интересуется курсом по математике.",
            "Тип звонка": "sales_call",
            "Следующий шаг": "Перезвонить",
        }
    ]

    contacts, amo_rows, manual_review = builder._build_contact_rows(
        call_rows=rows,
        chains_by_phone={},
        analysis_date=date(2026, 5, 10),
    )

    assert contacts[0]["Готово к записи в AMO"] == "Нет"
    assert "нет AMO contact ID" in contacts[0]["Причина статуса AMO"]
    assert amo_rows == []
    assert len(manual_review) == 1


def test_sales_call_with_single_amo_contact_id_stays_live_ready_even_if_tallanto_missing() -> None:
    rows = [
        {
            "Дата и время звонка": "2026-05-01 10:00:00",
            "Телефон клиента": "+79990000000",
            "Менеджер": "Менеджер",
            "Свежий период": "Да",
            "Статус Analyze": "done",
            "Содержательный звонок": "Да",
            "Нужна ручная проверка": "Нет",
            "Краткое резюме разговора": "Клиент интересуется летней школой по физике.",
            "Тип звонка": "sales_call",
            "Следующий шаг": "Перезвонить",
        }
    ]

    contacts, amo_rows, manual_review = builder._build_contact_rows(
        call_rows=rows,
        chains_by_phone={"+79990000000": {"amo_contact_ids": "123"}},
        analysis_date=date(2026, 5, 10),
    )

    assert contacts[0]["Готово к записи в AMO"] == "Да"
    assert contacts[0]["CRM writeback policy"] == "live_update_ready"
    assert len(amo_rows) == 1
    assert manual_review == []


def test_contact_summary_does_not_embed_full_latest_summary() -> None:
    rows = [
        {
            "Дата и время звонка": "2026-05-01 10:00:00",
            "Телефон клиента": "+79990000000",
            "Менеджер": "Менеджер",
            "Свежий период": "Да",
            "Статус Analyze": "done",
            "Содержательный звонок": "Да",
            "Нужна ручная проверка": "Нет",
            "Краткое резюме разговора": "Очень длинный подробный пересказ последнего разговора про летнюю школу и расписание.",
            "Тип звонка": "sales_call",
            "Рекомендуемый продукт": "Летняя школа",
            "Следующий шаг": "Перезвонить",
        }
    ]

    contacts, _, _ = builder._build_contact_rows(
        call_rows=rows,
        chains_by_phone={"+79990000000": {"amo_contact_ids": "123"}},
        analysis_date=date(2026, 5, 10),
    )

    assert "Последний содержательный контекст" not in contacts[0]["Краткая история общения"]
    assert "Очень длинный подробный пересказ" not in contacts[0]["Краткая история общения"]
    assert contacts[0]["Хронология общения (последние 5 касаний)"] == ""


def test_history_line_compacts_without_ellipsis() -> None:
    line = builder._history_line(
        {
            "Дата и время звонка": "2026-05-01 10:00:00",
            "Менеджер": "Менеджер",
            "Тип звонка": "sales_call",
            "Краткое резюме разговора": " ".join(["Клиент подробно рассказывал про интерес к летнему лагерю"] * 20),
        }
    )

    assert "..." not in line
    assert "…" not in line
    assert "[сжато]" in line


def test_unique_parts_with_counts_deduplicates_raw_and_counted_labels() -> None:
    value = builder._unique_parts_with_counts(["летний лагерь", "летний лагерь: 14", "математика", "математика: 3"])

    assert value == "летний лагерь (14 касаний) | математика (3 касаний)"


def test_contact_objections_split_current_and_historical_labels() -> None:
    value = builder._format_contact_objections(
        [
            {"Дата и время звонка": "2026-05-10 10:00:00", "Возражения": "цена | нужно согласовать даты"},
            {"Дата и время звонка": "2026-05-09 10:00:00", "Возражения": "неактуально"},
            {"Дата и время звонка": "2026-05-01 10:00:00", "Возражения": "доверие | нет перерасчета без проживания"},
        ]
    )

    assert "Актуальные: нужно согласовать даты" in value
    assert "цена" not in value
    assert "доверие" not in value
    assert "Исторические:" in value
    assert "неактуально" in value


def test_closure_next_step_downgrades_priority_and_clears_followup_date() -> None:
    follow_up, priority, probability = builder._adjust_operational_fields(
        analysis_date=date(2026, 5, 10),
        last_contentful_raw="2026-05-09 10:00:00",
        follow_up_raw="2026-05-10",
        priority_raw="warm",
        probability_raw="65",
        next_step_raw="Не беспокоить по этому предложению",
    )

    assert follow_up == ""
    assert priority == "cold"
    assert probability == "25"
