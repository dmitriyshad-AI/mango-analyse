from __future__ import annotations

import csv
from pathlib import Path

from mango_mvp.deal_aware.deal_quality_gate import (
    DealQualityGatePaths,
    evaluate_row,
    run_deal_quality_gate,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_payload() -> dict[str, str]:
    return {
        "AI-сводка по сделке": "Сделка рабочая. Содержание звонков вынесено в историю.",
        "AI-история по сделке": "05.05.2026 - Менеджер: клиент подтвердил интерес к летней очной школе.",
        "AI-рекомендованный следующий шаг": "Отправить материалы и зафиксировать результат в AMO.",
        "AI-дата следующего касания": "2026-05-15",
        "AI-фактический статус сделки": "Фактически: рабочая активная сделка.",
        "AI-приоритет сделки": "warm",
        "AI-актуальные возражения": "Актуальные возражения в релевантных звонках не выделены.",
        "AI-основание рекомендации": "Основание: последний содержательный звонок и активный статус сделки.",
        "AI-качество привязки к сделке": "Привязка: один выбранный AMO deal, один телефон.",
        "AI-предупреждение по сделке": "Критичных предупреждений Stage 3 не выявил.",
        "AI-Tallanto статус по сделке": "Tallanto: по телефону нет надежного точного сопоставления.",
        "AI-дата обновления сделки": "2026-05-13T00:00:00+00:00",
    }


def test_stage5_allows_clean_preview_row_for_stage6_dry_run() -> None:
    row = {
        **_safe_payload(),
        "review_id": "r1",
        "selected_deal_id": "100",
        "selected_status_name": "В работе",
        "selected_loss_reason": "",
        "crm_text_quality_passed": "Да",
        "quality_risk_types": "",
        "candidate_phone_count": "1",
        "stage3_risk_flags": "",
        "tallanto_context_status": "exact_phone_single",
    }

    hard, warnings = evaluate_row(row, _safe_payload(), row_index=1, analysis_date="2026-05-13")

    assert hard == []
    assert warnings == []


def test_stage5_blocks_payment_conflict_and_stage4_hard_risk() -> None:
    payload = _safe_payload() | {
        "AI-история по сделке": "Клиент уже прислал чек, оплата подтверждена.",
        "AI-рекомендованный следующий шаг": "Отправить ссылку на оплату.",
    }
    row = {
        **payload,
        "review_id": "r1",
        "selected_deal_id": "100",
        "selected_status_name": "В работе",
        "selected_loss_reason": "",
        "crm_text_quality_passed": "Нет",
        "quality_risk_types": "completed_payment_next_step_conflict",
        "candidate_phone_count": "1",
        "stage3_risk_flags": "",
        "tallanto_context_status": "exact_phone_single",
    }

    hard, _warnings = evaluate_row(row, payload, row_index=1, analysis_date="2026-05-13")
    gate_types = {finding["gate_type"] for finding in hard}

    assert "payment_next_step_consistency" in gate_types


def test_stage5_downgrades_stage4_payment_terms_false_positive_to_live_review() -> None:
    payload = _safe_payload() | {
        "AI-история по сделке": "Менеджер объяснил условия: при оплате до 1 апреля стоимость ниже.",
        "AI-рекомендованный следующий шаг": "Отправить материалы.",
    }
    row = {
        **payload,
        "review_id": "r1",
        "selected_deal_id": "100",
        "selected_status_name": "В работе",
        "selected_loss_reason": "",
        "crm_text_quality_passed": "Нет",
        "quality_risk_types": "completed_payment_next_step_conflict",
        "candidate_phone_count": "1",
        "stage3_risk_flags": "",
        "tallanto_context_status": "exact_phone_single",
    }

    hard, warnings = evaluate_row(row, payload, row_index=1, analysis_date="2026-05-13")

    assert hard == []
    assert {finding["gate_type"] for finding in warnings} == {"completed_payment_next_step_conflict"}


def test_stage5_builds_report_and_remains_fail_closed_for_live(tmp_path: Path) -> None:
    stage4 = tmp_path / "stage4"
    out = tmp_path / "out"
    clean = _safe_payload() | {
        "review_id": "r1",
        "selected_deal_id": "100",
        "selected_status_name": "В работе",
        "selected_loss_reason": "",
        "crm_text_quality_passed": "Да",
        "quality_risk_types": "",
        "candidate_phone_count": "1",
        "stage3_risk_flags": "",
        "tallanto_context_status": "exact_phone_single",
    }
    blocked = _safe_payload() | {
        "review_id": "r2",
        "selected_deal_id": "101",
        "selected_status_name": "Закрыто и не реализовано",
        "selected_loss_reason": "Дубль",
        "crm_text_quality_passed": "Нет",
        "quality_risk_types": "duplicate_loss_reason_requires_entity_resolution",
        "candidate_phone_count": "1",
        "stage3_risk_flags": "",
        "tallanto_context_status": "exact_phone_single",
    }
    _write_csv(stage4 / "deal_stage4_preview.csv", [clean, blocked])
    (stage4 / "deal_stage4_payloads.jsonl").write_text(
        "\n".join(
            [
                '{"review_id":"r1","payload":' + __import__("json").dumps(_safe_payload(), ensure_ascii=False) + "}",
                '{"review_id":"r2","payload":' + __import__("json").dumps(blocked, ensure_ascii=False) + "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (stage4 / "summary.json").write_text('{"schema_version": "stage4"}', encoding="utf-8")

    summary = run_deal_quality_gate(DealQualityGatePaths(stage4_preview_root=stage4, out_root=out))

    assert summary["coverage"]["input_rows"] == 2
    assert summary["coverage"]["stage6_dry_run_candidates"] == 1
    assert summary["coverage"]["blocked_rows"] == 1
    assert summary["readiness"]["passed_for_stage6_dry_run"] is True
    assert summary["readiness"]["passed_for_live_writeback"] is False
    assert summary["readiness"]["deal_aware_stage6_live_writeback_ready"] is False
    assert summary["input"]["stage4_preview_sha256"]
    assert (out / "deal_stage5_stage6_dry_run_candidates.csv").exists()
