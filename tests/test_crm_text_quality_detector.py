from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.quality.crm_text_quality_detector import (
    detect_crm_text_quality_batch_risks,
    detect_crm_text_quality_risks,
    findings_to_risk_counts,
    has_blocking_crm_text_quality_risk,
)


FIXTURE = Path("tests/fixtures/crm_text_quality_cases.jsonl")


def _risk_types(row: object, *, analysis_date: str = "2026-05-10") -> set[str]:
    return {finding.risk_type for finding in detect_crm_text_quality_risks(row, analysis_date=analysis_date)}


def test_detects_internal_ellipsis_anywhere_in_target_ai_field() -> None:
    row = {"Авто история общения": "Клиент уточнил оплату. В хронологии есть Клиент... и потеря контекста."}

    findings = detect_crm_text_quality_risks(row)

    assert {finding.risk_type for finding in findings} == {"lossy_ellipsis_truncation"}
    assert findings[0].class_id == "Q1"
    assert findings[0].severity == "P1"


def test_detects_duplicate_raw_label_and_count_label() -> None:
    row = {"Авто история общения": "Продукты интереса: летний лагерь | летний лагерь: 14 | математика"}

    assert "duplicate_label_and_count" in _risk_types(row)


def test_weak_filler_objection_labels_are_warning_only() -> None:
    row = {"Возражения": "время | доверие | цена", "Следующий шаг": "Перезвонить 12.05.2026"}

    findings = detect_crm_text_quality_risks(row)

    assert findings_to_risk_counts(findings)["weak_filler_objection_label"] == 3
    assert has_blocking_crm_text_quality_risk(row) is False


def test_strong_negative_objection_conflicts_with_sales_next_step() -> None:
    row = {
        "Возражения": "неактуально",
        "Следующий шаг": "Отправить ссылку на оплату",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "65",
    }

    findings = detect_crm_text_quality_risks(row)

    assert "strong_negative_objection_conflict" in {finding.risk_type for finding in findings}
    assert has_blocking_crm_text_quality_risk(row) is True


def test_closure_next_step_requires_downgrade_and_manual_review() -> None:
    row = {"Следующий шаг": "Отменить запись и не беспокоить клиента", "Приоритет лида": "warm"}

    risks = _risk_types(row)

    assert "closure_next_step_requires_downgrade" in risks
    assert "priority_next_step_conflict" in risks


def test_closure_next_step_detects_remove_application_language() -> None:
    row = {"Следующий шаг": "Снять заявку и убрать из списков", "Приоритет лида": "cold"}

    assert "closure_next_step_requires_downgrade" in _risk_types(row)


def test_vague_next_step_without_concrete_date_is_reported() -> None:
    row = {"Следующий шаг": "Связаться позже, если клиент изменит решение"}

    assert "vague_next_step" in _risk_types(row)


def test_waiting_for_customer_choice_without_date_is_vague_next_step() -> None:
    row = {"Следующий шаг": "Ждать выбора дат и предложить подходящую смену на июнь или август"}

    assert "vague_next_step" in _risk_types(row)


def test_lost_lead_competitor_purchase_conflicts_with_active_next_step() -> None:
    row = {
        "Авто история общения": (
            "Клиент сообщила, что уже приобрели программу у другого образовательного лагеря. "
            "Дальнейшее продолжение сделки не требуется, так как интерес закрыт покупкой у конкурента."
        ),
        "Следующий шаг": "Перезвонить клиенту",
        "Приоритет лида": "cold",
        "Вероятность продажи, %": "40",
    }

    risks = _risk_types(row)

    assert "lost_lead_next_step_conflict" in risks
    assert has_blocking_crm_text_quality_risk(row) is True


def test_lost_lead_signal_without_sales_context_does_not_block() -> None:
    row = {
        "Авто история общения": "Клиент уже купил курс в другой школе, возврат в продажу не нужен.",
        "Следующий шаг": "",
        "Приоритет лида": "lost",
        "Вероятность продажи, %": "0",
    }

    assert "lost_lead_next_step_conflict" not in _risk_types(row)


def test_comparison_with_other_camps_does_not_trigger_lost_lead_conflict() -> None:
    row = {
        "Авто история общения": "Клиент сравнивает несколько лагерей и просит прислать программу.",
        "Следующий шаг": "Отправить материалы и перезвонить 15.05.2026",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "60",
    }

    assert detect_crm_text_quality_risks(row, analysis_date="2026-05-12") == []


def test_passive_customer_return_conflicts_with_active_next_step() -> None:
    row = {
        "Авто история общения": (
            "Клиент сказал, что пока решение не принято и свяжется сам, когда определятся. "
            "Контактный следующий шаг перенесен на повторное обращение клиента."
        ),
        "Возражения": "пока не готовы, просили не предлагать активно",
        "Следующий шаг": "Отправить материалы",
        "Приоритет лида": "cold",
        "Вероятность продажи, %": "40",
    }

    assert "passive_customer_next_step_conflict" in _risk_types(row)


def test_passive_customer_signal_without_active_sales_context_does_not_block() -> None:
    row = {
        "Авто история общения": "Клиент сказал, что свяжется сам, когда определится.",
        "Следующий шаг": "",
        "Приоритет лида": "cold",
        "Вероятность продажи, %": "10",
    }

    assert "passive_customer_next_step_conflict" not in _risk_types(row)


def test_explicit_no_next_step_conflicts_with_active_callback() -> None:
    row = {
        "Авто история общения": (
            "Клиент увидел различие по датам и ценам, после разъяснения отказался от дальнейшего интереса. "
            "Договоренности о следующем шаге не было."
        ),
        "Следующий шаг": "Перезвонить клиенту",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "65",
    }

    assert "explicit_no_next_step_conflict" in _risk_types(row)


def test_wrong_person_identity_mismatch_blocks_writeback() -> None:
    row = {
        "Авто история общения": (
            "Контакт не подтвердился: менеджер звонил по поводу летней школы, "
            "но в разговоре выяснилась путаница с именем и на линии была не та Светлана. "
            "Обсуждение программы не состоялось."
        ),
        "Следующий шаг": "Отправить материалы",
        "Приоритет лида": "warm",
    }

    assert "wrong_person_or_identity_mismatch" in _risk_types(row)
    assert has_blocking_crm_text_quality_risk(row) is True


def test_active_client_loss_reason_requires_entity_resolution() -> None:
    row = {
        "AMO причина отказа": "Действующий клиент",
        "AI-фактический статус сделки": "похоже на отказ или потерю интереса",
        "AI-рекомендованный следующий шаг": "Перезвонить клиенту и предложить годовой курс",
        "AI-приоритет сделки": "warm",
    }

    assert "active_client_loss_reason_requires_entity_resolution" in _risk_types(row)
    assert has_blocking_crm_text_quality_risk(row) is True


def test_regular_loss_reason_does_not_trigger_active_client_resolution() -> None:
    row = {
        "AMO причина отказа": "Дорого",
        "AI-рекомендованный следующий шаг": "Вернуться к клиенту при новой акции",
        "AI-приоритет сделки": "review",
    }

    assert "active_client_loss_reason_requires_entity_resolution" not in _risk_types(row)


def test_completed_payment_conflicts_with_collect_payment_next_step() -> None:
    row = {
        "Авто история общения": "Клиент уже прислал чек об оплате, сделка закрыта успешно.",
        "Следующий шаг": "Прислать оплату и подписанный договор по электронной почте",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "65",
    }

    assert "completed_payment_next_step_conflict" in _risk_types(row)


def test_completed_payment_receipt_amount_conflicts_with_active_materials_step() -> None:
    row = {
        "Авто история общения": "Менеджер проверил почту и подтвердил наличие чека и ответа по оплате на сумму 82 700.",
        "Следующий шаг": "Отправить материалы",
        "Приоритет лида": "warm",
    }

    assert "completed_payment_next_step_conflict" in _risk_types(row)


def test_paid_invoice_wording_conflicts_with_active_materials_step() -> None:
    row = {
        "Авто история общения": "Клиентка уточняла порядок оформления после оплаты: платежку оплатили 29-го числа.",
        "Следующий шаг": "Отправить материалы",
        "Приоритет лида": "warm",
    }

    assert "completed_payment_next_step_conflict" in _risk_types(row)


def test_pending_payment_does_not_trigger_completed_payment_conflict() -> None:
    row = {
        "Авто история общения": "Клиент подтвердил готовность оплатить после получения договора.",
        "Следующий шаг": "Отправить договор и дождаться оплаты",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "65",
    }

    assert "completed_payment_next_step_conflict" not in _risk_types(row)


def test_stale_followup_date_equal_to_analysis_date_is_reported() -> None:
    row = {
        "Следующий шаг": "Связаться в мае",
        "Рекомендуемая дата следующего контакта": "2026-05-10",
    }

    assert "stale_uniform_followup_date" in _risk_types(row, analysis_date="2026-05-10")


def test_relative_tomorrow_next_step_must_match_followup_date() -> None:
    row = {
        "Следующий шаг": "Перезвонить в первой половине дня завтра",
        "Рекомендуемая дата следующего контакта": "2026-05-12",
    }

    assert "relative_next_step_date_mismatch" in _risk_types(row, analysis_date="2026-05-12")


def test_relative_year_next_step_must_not_default_to_today() -> None:
    row = {
        "Следующий шаг": "Вернуться к контакту через год и уточнить готовность к обучению",
        "Рекомендуемая дата следующего контакта": "2026-05-12",
    }

    assert "relative_next_step_date_mismatch" in _risk_types(row, analysis_date="2026-05-12")


def test_old_source_call_blocks_fresh_materials_next_step() -> None:
    row = {
        "Дата последнего свежего звонка": "2026-01-30 10:49:13",
        "Следующий шаг": "Отправить материалы",
        "Рекомендуемая дата следующего контакта": "2026-05-12",
    }

    assert "stale_source_next_step" in _risk_types(row, analysis_date="2026-05-12")


def test_old_source_call_blocks_broad_active_next_step_family() -> None:
    row = {
        "Дата последнего свежего звонка": "2026-04-05 13:10:00",
        "Следующий шаг": "Перезвонить родителям",
        "Рекомендуемая дата следующего контакта": "2026-05-15",
    }

    assert "stale_source_next_step" in _risk_types(row, analysis_date="2026-05-12")


def test_very_old_source_call_blocks_any_non_empty_next_step() -> None:
    row = {
        "Дата последнего свежего звонка": "2025-08-26 13:10:00",
        "Следующий шаг": "Уточнить наличие очных занятий в Жуковском",
        "Рекомендуемая дата следующего контакта": "2026-05-12",
    }

    assert "stale_source_next_step" in _risk_types(row, analysis_date="2026-05-12")


def test_concrete_next_step_date_does_not_trigger_stale_followup() -> None:
    row = {
        "Следующий шаг": "Перезвонить 12.05.2026 и подтвердить расписание",
        "Рекомендуемая дата следующего контакта": "2026-05-12",
    }

    assert detect_crm_text_quality_risks(row, analysis_date="2026-05-10") == []


def test_verbose_manager_ux_warns_for_long_summary_plus_chronology() -> None:
    repeated = " Клиент сравнивает форматы, уточняет оплату и просит сохранить договоренности."
    row = {"Авто история общения": "Сводка клиента: интерес к курсу. Хронология общения:" + repeated * 20}

    risks = _risk_types(row)

    assert "verbose_manager_ux" in risks


def test_empty_auto_history_readback_is_blocking() -> None:
    row = {"Авто история общения": "", "Последняя AI-сводка": "Клиент интересуется курсом."}

    findings = detect_crm_text_quality_risks(row)

    assert {finding.risk_type for finding in findings} == {"empty_auto_history"}
    assert has_blocking_crm_text_quality_risk(row) is True


def test_cross_field_duplicate_information_is_blocking() -> None:
    repeated = (
        "Клиент интересуется летним лагерем, уточняет стоимость и просит отправить материалы "
        "для согласования с семьей."
    )
    row = {
        "Последняя AI-сводка": repeated,
        "Авто история общения": f"Сводка клиента: {repeated}",
        "AI-рекомендованный следующий шаг": "Отправить материалы и перезвонить 15.05.2026",
    }

    risks = _risk_types(row)

    assert "cross_field_duplicate_information" in risks
    assert has_blocking_crm_text_quality_risk(row) is True


def test_distinct_contact_fields_do_not_trigger_cross_field_duplication() -> None:
    row = {
        "AI-краткая сводка клиента": "Семья выбирает летний лагерь для ученика 8 класса, интерес теплый.",
        "AI-история общения": "27.04 обсудили даты. 29.04 клиент попросил отправить программу и условия оплаты.",
        "AI-рекомендованный следующий шаг": "Отправить программу и перезвонить 15.05.2026",
        "AI-учебный контекст Tallanto": "В Tallanto активных занятий и списаний по этому продукту пока нет.",
    }

    assert "cross_field_duplicate_information" not in _risk_types(row)


def test_allows_compact_actionable_crm_card() -> None:
    row = {
        "Авто история общения": (
            "Клиент интересуется годовым курсом по математике для 9 класса. "
            "Актуальный следующий шаг: перезвонить 12.05.2026 и подтвердить расписание."
        ),
        "Возражения": "нужен онлайн-формат",
        "Следующий шаг": "Перезвонить 12.05.2026",
        "Рекомендуемая дата следующего контакта": "2026-05-12",
        "Приоритет лида": "warm",
    }

    assert detect_crm_text_quality_risks(row, analysis_date="2026-05-10") == []


def test_batch_detector_reports_uniform_run_date_followups() -> None:
    rows = [
        {"Следующий шаг": "Перезвонить клиенту", "Рекомендуемая дата следующего контакта": "2026-05-10"},
        {"Следующий шаг": "Уточнить решение", "Рекомендуемая дата следующего контакта": "2026-05-10"},
        {"Следующий шаг": "Отправить письмо", "Рекомендуемая дата следующего контакта": "2026-05-10"},
    ]

    findings = detect_crm_text_quality_batch_risks(rows, analysis_date="2026-05-10")

    stale = [finding for finding in findings if finding.risk_type == "stale_uniform_followup_date"]
    assert {finding.row_index for finding in stale} == {1, 2, 3}


def test_fixture_cases_match_expected_risks() -> None:
    for raw_line in FIXTURE.read_text(encoding="utf-8").splitlines():
        case = json.loads(raw_line)
        findings = detect_crm_text_quality_risks(case["input"], analysis_date=case.get("analysis_date"))
        actual = {finding.risk_type for finding in findings}

        assert actual >= set(case["expected_risk_types"]), case["case_id"]
        if not case["expected_risk_types"]:
            assert actual == set(), case["case_id"]


def test_detector_is_independent_from_builder_and_writer() -> None:
    source = Path("src/mango_mvp/quality/crm_text_quality_detector.py").read_text(encoding="utf-8")

    assert "build_post_backfill_amo_ready_export" not in source
    assert "write_amo_ready_contacts" not in source
    assert "stable_runtime" not in source
