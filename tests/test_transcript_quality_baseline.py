from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.quality.transcript_quality_baseline import BaselineConfig, build_transcript_quality_baseline


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_transcript_quality_baseline_counts_core_risks(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness"
    kb = tmp_path / "kb"
    rop = tmp_path / "rop"
    _write_json(readiness / "summary.json", {"totals": {"terminal_analyzed_calls": 2}})
    _write_json(kb / "summary.json", {"totals": {"source_reviews": 2}})
    _write_json(rop / "summary.json", {"totals": {"combined_unique_moments": 2}})
    _write_csv(
        readiness / "calls_terminal_analyzed.csv",
        [
            {
                "source_filename": "no_live.mp3",
                "source_db": "db.sqlite",
                "started_at": "2026-04-01 10:00:00",
                "month": "2026-04",
                "phone": "79000000000",
                "manager_name": "Анна",
                "call_type": "technical_call",
                "contentful": "True",
                "follow_up_score": "70",
                "next_step": "Перезвонить позднее",
                "history_summary": "Абонент недоступен, звонок перенаправлен на голосовую почту. Контакты: канал: email.",
            },
            {
                "source_filename": "live.mp3",
                "source_db": "db.sqlite",
                "started_at": "2026-04-01 11:00:00",
                "month": "2026-04",
                "phone": "79000000001",
                "manager_name": "Олег",
                "call_type": "sales_call",
                "contentful": "True",
                "follow_up_score": "70",
                "next_step": "Отправить материалы",
                "history_summary": "Клиент спросил стоимость курса и попросил отправить материалы.",
            },
        ],
    )
    _write_csv(
        readiness / "client_chains.csv",
        [
            {
                "touch_bucket": "1",
                "sample_stratum": "sales_call",
                "outcome_availability": "strong",
                "contentful_call_count": "1",
            }
        ],
    )
    _write_csv(
        kb / "enriched_reviews.csv",
        [
            {
                "answer_pattern": "no_live_contact_or_voicemail",
                "commercial_usefulness": "revenue_leakage_risk",
                "bot_seed_status": "exclude_no_dialogue",
                "quality_band": "low",
                "ideal_answer_example": "Клиент недоступен.",
            },
            {
                "answer_pattern": "price_payment_handled_with_value_or_instruction",
                "commercial_usefulness": "playbook_candidate",
                "bot_seed_status": "ready_for_bot_draft",
                "quality_band": "high",
                "ideal_answer_example": "Стоимость 50 000 рублей, скидка 10%.",
            },
        ],
    )
    _write_csv(
        rop / "rop_validation.csv",
        [
            {
                "Категория проверки": "Риск потери выручки",
                "Приоритет": "P0 риск выручки",
                "Идеальный ответ": "Абонент недоступен, оставьте сообщение после сигнала.",
            },
            {
                "Категория проверки": "Черновик для бота",
                "Приоритет": "P1 бот",
                "Идеальный ответ": "Стоимость 50 000 рублей, скидка 10%.",
            },
        ],
    )

    summary = build_transcript_quality_baseline(
        BaselineConfig(
            project_root=tmp_path,
            readiness_root=readiness,
            kb_root=kb,
            rop_root=rop,
            out_root=tmp_path / "out",
        )
    )

    assert summary["readiness_metrics"]["terminal_analyzed_calls"] == 2
    assert summary["readiness_metrics"]["suspicious_contentful_by_history"] == 1
    assert summary["readiness_metrics"]["suspicious_contentful_with_next_step"] == 1
    assert summary["readiness_metrics"]["false_email_from_voice_mail_candidates"] == 1
    assert summary["kb_metrics"]["no_live_revenue_risk"] == 1
    assert summary["kb_metrics"]["bot_ready_money_or_terms"] == 1
    assert summary["rop_metrics"]["p0_no_live_or_artifact"] == 1
    assert summary["rop_metrics"]["bot_candidate_money_or_terms"] == 1
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "BASELINE_REPORT.md").exists()
    assert (tmp_path / "out" / "suspicious_contentful_sample.csv").exists()


def test_baseline_uses_bot_safe_answer_for_stage13_safety_metrics(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness"
    kb = tmp_path / "kb"
    rop = tmp_path / "rop"
    _write_json(readiness / "summary.json", {"totals": {"terminal_analyzed_calls": 1}})
    _write_json(kb / "summary.json", {"totals": {"source_reviews": 1}})
    _write_json(rop / "summary.json", {"totals": {"combined_unique_moments": 1}})
    _write_csv(
        readiness / "calls_terminal_analyzed.csv",
        [
            {
                "source_filename": "live.mp3",
                "source_db": "db.sqlite",
                "started_at": "2026-04-01 11:00:00",
                "month": "2026-04",
                "phone": "79000000001",
                "manager_name": "Олег",
                "call_type": "sales_call",
                "contentful": "True",
                "follow_up_score": "70",
                "next_step": "Отправить материалы",
                "history_summary": "Клиент спросил стоимость курса.",
            },
        ],
    )
    _write_csv(
        readiness / "client_chains.csv",
        [{"touch_bucket": "1", "sample_stratum": "sales_call", "outcome_availability": "strong", "contentful_call_count": "1"}],
    )
    _write_csv(
        kb / "enriched_reviews.csv",
        [
            {
                "answer_pattern": "price_payment_handled_with_value_or_instruction",
                "commercial_usefulness": "playbook_candidate",
                "bot_seed_status": "ready_for_bot_draft",
                "quality_band": "high",
                "ideal_answer_example": "В НПК МФТИ стоимость 50 000 рублей, скидка 10% до 15 мая.",
                "ideal_answer_manager_sanitized": "В Фотоне актуальную стоимость и условия оплаты нужно уточнить по текущей политике.",
                "bot_safe_answer": "Актуальную стоимость и условия оплаты менеджер подтвердит по текущим правилам.",
                "bot_safety_status": "safe_with_placeholders",
            },
        ],
    )
    _write_csv(
        rop / "rop_validation.csv",
        [
            {
                "Категория проверки": "Черновик для бота",
                "Приоритет": "P1 бот",
                "Идеальный ответ": "В Фотоне актуальную стоимость и условия оплаты нужно уточнить по текущей политике.",
                "Идеальный ответ для менеджера": "В Фотоне актуальную стоимость и условия оплаты нужно уточнить по текущей политике.",
                "Безопасный ответ для бота": "Актуальную стоимость и условия оплаты менеджер подтвердит по текущим правилам.",
            },
        ],
    )

    summary = build_transcript_quality_baseline(
        BaselineConfig(project_root=tmp_path, readiness_root=readiness, kb_root=kb, rop_root=rop, out_root=tmp_path / "out")
    )

    assert summary["kb_metrics"]["raw_ideal_answer_brand_risk"] == 1
    assert summary["kb_metrics"]["raw_ideal_answer_money_or_terms"] == 1
    assert summary["kb_metrics"]["bot_ready_money_or_terms"] == 0
    assert summary["kb_metrics"]["bot_safe_answer_brand_risk"] == 0
    assert summary["kb_metrics"]["bot_safe_answer_personal_data_risk"] == 0
    assert summary["rop_metrics"]["bot_candidate_money_or_terms"] == 0
    assert summary["rop_metrics"]["bot_safe_answer_brand_risk"] == 0
