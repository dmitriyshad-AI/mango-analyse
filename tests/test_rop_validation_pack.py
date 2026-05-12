from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from mango_mvp.insights.rop_validation_pack import (
    ROPValidationPackConfig,
    build_rop_validation_pack,
    is_reviewable_business_row,
    select_diverse_rows,
)


def _row(idx: int, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "moment_id": f"pilot-{idx:05d}",
        "phone": f"79000000{idx:03d}",
        "source_filename": f"call-{idx}.mp3",
        "started_at": "2026-04-01 10:00:00",
        "manager_name": "Менеджер",
        "llm_customer_signal_type": "price_question",
        "signal_ru": "Вопрос о цене",
        "stage_ru": "Обсуждение цены",
        "answer_pattern": "price_payment_handled_with_value_or_instruction",
        "answer_pattern_ru": "Цена/оплата объяснены через ценность или инструкцию",
        "final_outcome_ru": "Есть путь к оплате",
        "commercial_usefulness": "playbook_candidate",
        "commercial_usefulness_ru": "Кандидат в базу лучших ответов",
        "bot_seed_status": "ready_for_bot_draft",
        "bot_seed_status_ru": "Можно брать как черновик для бота",
        "overall_quality_score": 80,
        "extraction_confidence": 0.8,
        "customer_question": "Сколько стоит?",
        "manager_answer": "Менеджер объяснил стоимость.",
        "ideal_answer_example": "Стоимость зависит от формата; подберем вариант и зафиксируем следующий шаг.",
        "ideal_answer_manager_sanitized": "Стоимость зависит от формата; подберем вариант и зафиксируем следующий шаг.",
        "bot_safe_answer": "Актуальную стоимость подберет менеджер по выбранному формату.",
        "bot_safety_status": "safe_no_changes",
        "bot_safety_status_ru": "Безопасно без замен",
        "sanitizer_flags": "",
        "what_manager_did_well": "Ответил по цене",
        "what_manager_missed": "",
        "risk_flags": "",
        "rop_action": "Проверить пример для скрипта.",
        "avoid_using_when": "",
        "data_scope_note": "Оценка только по звонкам: мессенджеры и почта в этом слое не учтены.",
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_diverse_rows_respects_caps_then_fills() -> None:
    rows = [_row(idx, signal_ru="Одинаковый сигнал", overall_quality_score=100 - idx) for idx in range(5)]

    selected = select_diverse_rows(rows, limit=4, caps={"signal_ru": 2})

    assert len(selected) == 4
    assert selected[0]["moment_id"] == "pilot-00000"


def test_build_rop_validation_pack_outputs_workbook(tmp_path: Path) -> None:
    rows = [
        _row(1),
        _row(2, commercial_usefulness="revenue_leakage_risk", commercial_usefulness_ru="Риск потери выручки", overall_quality_score=35),
        _row(3, commercial_usefulness="process_fix_needed", commercial_usefulness_ru="Нужна правка процесса", answer_pattern_ru="Нет точного следующего шага", overall_quality_score=50),
        _row(4, bot_seed_status="needs_rop_validation", bot_seed_status_ru="Нужна проверка РОПом", overall_quality_score=70),
        _row(
            5,
            commercial_usefulness="revenue_leakage_risk",
            answer_pattern="no_live_contact_or_voicemail",
            answer_pattern_ru="Не было живого контакта / автоответчик",
            manager_answer="Абонент недоступен.",
            overall_quality_score=30,
        ),
        _row(
            6,
            review_trust_status="needs_live_llm_refresh",
            commercial_usefulness="playbook_candidate",
            overall_quality_score=95,
        ),
        _row(
            7,
            ideal_answer_example="В НПК МФТИ стоимость 50 000 рублей, скидка 10% до 15 мая.",
            ideal_answer_manager_sanitized="В Фотоне актуальную стоимость и условия оплаты нужно уточнить по текущей политике.",
            bot_safe_answer="Актуальную стоимость и условия оплаты менеджер подтвердит по текущим правилам.",
            sanitizer_flags="brand_normalized | price_redacted | discount_terms_redacted | deadline_redacted",
            bot_safety_status="safe_with_placeholders",
            bot_safety_status_ru="Безопасно после sanitization",
        ),
    ]
    kb_root = tmp_path / "kb"
    _write_csv(kb_root / "enriched_reviews.csv", rows)

    summary = build_rop_validation_pack(
        ROPValidationPackConfig(
            project_root=tmp_path,
            kb_root=kb_root,
            out_root=tmp_path / "rop_pack",
            top_answers_limit=10,
            bot_seeds_limit=10,
        )
    )

    assert summary["totals"]["source_reviews"] == 7
    assert summary["totals"]["excluded_from_validation"] == 2
    assert summary["totals"]["revenue_risks_for_validation"] == 1
    workbook_path = tmp_path / "rop_pack" / "ROP_validation_pack_v1.xlsx"
    assert workbook_path.exists()
    with zipfile.ZipFile(workbook_path) as xlsx:
        workbook_xml = xlsx.read("xl/workbook.xml")
    root = ElementTree.fromstring(workbook_xml)
    sheet_names = [sheet.attrib["name"] for sheet in root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet")]
    assert "Проверка РОПа" in sheet_names
    assert "Риски потери выручки" in sheet_names
    assert "Исключено из проверки" in sheet_names
    revenue_risks = (tmp_path / "rop_pack" / "revenue_leakage_risks.csv").read_text(encoding="utf-8-sig")
    assert "pilot-00005" not in revenue_risks
    bot_drafts = (tmp_path / "rop_pack" / "bot_knowledge_drafts.csv").read_text(encoding="utf-8-sig")
    assert "50 000" not in bot_drafts
    assert "НПК" not in bot_drafts
    assert "Безопасный ответ для бота" in bot_drafts
    assert "Актуальную стоимость" in bot_drafts
    excluded = (tmp_path / "rop_pack" / "excluded_from_validation.csv").read_text(encoding="utf-8-sig")
    assert "pilot-00005" in excluded
    assert "pilot-00006" in excluded


def test_is_reviewable_business_row_excludes_no_live_and_untrusted() -> None:
    assert is_reviewable_business_row(_row(1))
    assert not is_reviewable_business_row(
        _row(2, answer_pattern="no_live_contact_or_voicemail", manager_answer="Автоответчик")
    )
    assert not is_reviewable_business_row(_row(3, review_trust_status="needs_live_llm_refresh"))
    assert not is_reviewable_business_row(_row(4, manager_answer="Продолжение следует..."))
