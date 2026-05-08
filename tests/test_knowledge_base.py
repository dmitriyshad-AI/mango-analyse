from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from mango_mvp.insights.knowledge_base import (
    KnowledgeBaseConfig,
    build_sales_insight_knowledge_base,
    classify_answer_pattern,
    commercial_usefulness,
    enrich_review_row,
    outcome_group,
    quality_band,
)


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "moment_id": "pilot-00001",
        "phone": "79000000000",
        "source_filename": "call.mp3",
        "started_at": "2026-04-01 10:00:00",
        "manager_name": "Менеджер",
        "llm_customer_signal_type": "price_question",
        "llm_hidden_sales_stage": "price_discussion",
        "final_outcome_label": "payment_pending",
        "outcome_confidence_tier": "strong",
        "customer_question": "Сколько стоит курс?",
        "manager_answer": "Менеджер объяснил стоимость, ценность программы и отправил ссылку на оплату.",
        "overall_quality_score": 82,
        "extraction_confidence": 0.88,
        "what_manager_did_well": "Ответил на цену",
        "what_manager_missed": "",
        "ideal_reaction": "Объяснить ценность и следующий шаг.",
        "ideal_answer_example": "Стоимость зависит от формата. Я пришлю расчет, а сейчас кратко объясню, что входит в курс.",
        "risk_flags": "",
        "avoid_using_when": "",
        "customer_quote": "Сколько стоит?",
        "manager_quote": "Отправлю ссылку",
        "history_summary": "Клиент спрашивал цену.",
        "rubric_factual_correctness": 80,
        "rubric_completeness": 80,
        "rubric_persuasiveness": 80,
        "rubric_personalization": 80,
        "rubric_objection_handling": 80,
        "rubric_next_step_clarity": 80,
        "rubric_empathy_tone": 80,
        "rubric_sales_discipline": 80,
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_quality_and_outcome_grouping() -> None:
    assert quality_band(80) == "high"
    assert quality_band(60) == "medium"
    assert quality_band(40) == "low"
    assert outcome_group("won_paid_or_active") == "paid_or_payment_path"
    assert outcome_group("reopen_or_follow_up_opportunity") == "follow_up_opportunity"
    assert outcome_group("lost_or_refused") == "lost_or_churn"


def test_classify_answer_pattern_for_core_cases() -> None:
    assert classify_answer_pattern(_row()) == "price_payment_handled_with_value_or_instruction"
    assert classify_answer_pattern(
        _row(
            llm_customer_signal_type="schedule_question",
            manager_answer="Перезвоним как только будет понятно.",
            risk_flags="Нет точной даты follow-up.",
        )
    ) == "vague_or_missing_next_step"
    assert classify_answer_pattern(
        _row(
            llm_customer_signal_type="technical_or_access_issue",
            llm_hidden_sales_stage="existing_client_service",
            manager_answer="Передам в поддержку, проверим доступ и продублируем ссылку.",
        )
    ) == "service_resolution_or_escalation"
    assert classify_answer_pattern(_row(manager_answer="Абонент недоступен, оставили голосовое сообщение.")) == "no_live_contact_or_voicemail"


def test_enrich_review_row_adds_business_fields() -> None:
    enriched = enrich_review_row(_row())

    assert enriched["quality_band"] == "high"
    assert enriched["outcome_group"] == "paid_or_payment_path"
    assert enriched["commercial_usefulness"] == "playbook_candidate"
    assert enriched["bot_seed_status"] == "ready_for_bot_draft"
    assert enriched["signal_ru"] == "Вопрос о цене"
    assert enriched["final_outcome_ru"] == "Есть путь к оплате"
    assert enriched["answer_pattern_ru"] == "Цена/оплата объяснены через ценность или инструкцию"
    assert "мессенджеры" in enriched["data_scope_note"]
    assert commercial_usefulness(_row(final_outcome_label="lost_or_refused"), 40, "lost_or_refused") == "revenue_leakage_risk"


def test_build_sales_insight_knowledge_base_outputs_workbook(tmp_path: Path) -> None:
    rows = [
        _row(moment_id="pilot-00001", manager_name="Анна", overall_quality_score=82),
        _row(
            moment_id="pilot-00002",
            manager_name="Олег",
            llm_customer_signal_type="schedule_question",
            final_outcome_label="lost_or_refused",
            manager_answer="Перезвоним позже.",
            risk_flags="Нет точной даты следующего контакта.",
            overall_quality_score=42,
        ),
        _row(
            moment_id="pilot-00003",
            manager_name="Олег",
            llm_customer_signal_type="technical_or_access_issue",
            final_outcome_label="service_or_existing_context",
            manager_answer="Передам в поддержку и продублирую ссылку.",
            overall_quality_score=73,
        ),
    ]
    reviews_csv = tmp_path / "reviews.csv"
    _write_csv(reviews_csv, rows)

    summary = build_sales_insight_knowledge_base(
        KnowledgeBaseConfig(
            project_root=tmp_path,
            reviews_csv=reviews_csv,
            out_root=tmp_path / "kb",
            min_group_count=1,
            top_examples=10,
        )
    )

    assert summary["totals"]["reviews"] == 3
    assert summary["quality"]["low_quality_count"] == 1
    workbook_path = tmp_path / "kb" / "sales_insight_knowledge_base.xlsx"
    assert workbook_path.exists()
    assert (tmp_path / "kb" / "bot_knowledge_seeds.csv").exists()
    brief = (tmp_path / "kb" / "signal_summary.csv").read_text(encoding="utf-8-sig")
    assert "Вопрос о цене" in brief
    assert "Только звонки" in brief
    with zipfile.ZipFile(workbook_path) as xlsx:
        workbook_xml = xlsx.read("xl/workbook.xml")
    root = ElementTree.fromstring(workbook_xml)
    sheet_names = [sheet.attrib["name"] for sheet in root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet")]
    assert "Сводка РОПа" in sheet_names
    assert "Лучшие ответы" in sheet_names
    assert "ROP brief" not in sheet_names
