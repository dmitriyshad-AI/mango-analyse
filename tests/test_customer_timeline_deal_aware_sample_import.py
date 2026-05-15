from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.context_provider import get_customer_context_for_phone
from mango_mvp.customer_timeline.deal_aware_sample_import import (
    DealAwareTimelineSampleConfig,
    audit_deal_aware_timeline_sample,
    build_deal_aware_timeline_sample,
)


NOW = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _config(tmp_path: Path, *, risk_classes: str = "") -> DealAwareTimelineSampleConfig:
    selected = tmp_path / "selected_phones.csv"
    all_candidates = tmp_path / "all_709.csv"
    calls = tmp_path / "master_calls_ru.csv"
    contacts = tmp_path / "master_contacts_ru.csv"
    _write_csv(
        selected,
        [
            {
                "primary_phone": "+79161234567",
                "normalized_phones": "+79161234567 | +79161234568",
                "selected_deal_id": "deal-1",
                "selected_deal_name": "Курс математики",
                "selected_pipeline_name": "Сделки B2C",
                "selected_status_name": "Перспектива",
                "candidate_call_count": "2",
                "candidate_phone_count": "2",
                "tallanto_context_status": "exact_phone_single",
                "AI-приоритет сделки": "warm",
                "AI-рекомендованный следующий шаг": "Связаться после проверки оплаты.",
                "AI-сводка по сделке": "Клиент выбирает курс математики.",
                "risk_classes": risk_classes,
                "selection_reason": "test",
            }
        ],
    )
    _write_csv(
        all_candidates,
        [
            {
                "selected_deal_id": "deal-1",
                "selected_deal_name": "Курс математики",
                "selected_status_name": "Перспектива",
                "selected_pipeline_name": "Сделки B2C",
                "AI-история по сделке": "Клиент дважды обсуждал курс и условия оплаты.",
                "AI-Tallanto статус по сделке": "Есть точная связь Tallanto.",
                "AI-сводка по сделке": "Клиент выбирает курс математики.",
                "AI-рекомендованный следующий шаг": "Связаться после проверки оплаты.",
                "risk_classes": risk_classes,
            }
        ],
    )
    _write_csv(
        calls,
        [
            {
                "ID звонка": "call-1",
                "Дата и время звонка": "2026-05-01 10:00:00",
                "Телефон клиента": "+79161234567",
                "Менеджер": "Настя",
                "Содержательный звонок": "Да",
                "Краткое резюме разговора": "Клиент спросил про курс математики и расписание.",
                "Тип звонка": "sales_call",
            },
            {
                "ID звонка": "call-2",
                "Дата и время звонка": "2026-05-02 11:00:00",
                "Телефон клиента": "+79161234568",
                "Менеджер": "Настя",
                "Содержательный звонок": "Да",
                "Краткое резюме разговора": "Клиент уточнил оплату и следующий шаг.",
                "Тип звонка": "sales_call",
            },
        ],
    )
    _write_csv(
        contacts,
        [
            {
                "Телефон клиента": "+79161234567",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Статус матчинга Tallanto": "exact_phone_single",
                "ID Tallanto": "student-1",
                "ФИО родителя Tallanto": "Иван Петров",
            },
            {
                "Телефон клиента": "+79161234568",
                "Первый звонок": "2026-05-01 10:00:00",
                "Последний звонок": "2026-05-02 11:00:00",
                "Статус матчинга Tallanto": "exact_phone_single",
                "ID Tallanto": "student-1",
                "ФИО родителя Tallanto": "Иван Петров",
            },
        ],
    )
    return DealAwareTimelineSampleConfig(
        selected_groups_csv=selected,
        all_candidates_csv=all_candidates,
        master_calls_csv=calls,
        master_contacts_csv=contacts,
        allowed_root=tmp_path / "product_data",
        out_root=tmp_path / "product_data" / "customer_timeline" / "sample",
        timeline_db=tmp_path / "product_data" / "customer_timeline" / "sample" / "customer_timeline.sqlite",
        generated_at=NOW,
    )


def test_build_local_timeline_from_deal_aware_sample_and_read_by_phone(tmp_path: Path) -> None:
    config = _config(tmp_path)

    report = build_deal_aware_timeline_sample(config)
    context = get_customer_context_for_phone("+79161234568", timeline_db=config.timeline_db)

    assert report["summary"]["selected_groups"] == 1
    assert report["summary"]["selected_unique_phones"] == 2
    assert report["summary"]["matched_call_rows"] == 2
    assert report["summary"]["store_counts"]["customer_identities"] == 1
    assert context["source"] == "customer_timeline"
    assert context["fallback_used"] is False
    assert context["timeline"]["event_types"]["mango_call"] == 2


def test_audit_marks_imported_clean_sample_ready_for_preview(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_deal_aware_timeline_sample(config)

    audit = audit_deal_aware_timeline_sample(config)

    assert audit["summary"]["timeline_matched_phone_groups"] == 1
    assert audit["summary"]["ready_for_preview"] == 1
    assert audit["summary"]["gate_decision"]["can_enable_timeline_preview_enabled"] is True


def test_audit_keeps_dealaware_conflicts_in_manual_review(tmp_path: Path) -> None:
    config = _config(tmp_path, risk_classes="amo_tallanto_mismatch|payment_stage")
    build_deal_aware_timeline_sample(config)

    audit = audit_deal_aware_timeline_sample(config)

    assert audit["summary"]["timeline_matched_phone_groups"] == 1
    assert audit["summary"]["needs_manual_review"] == 1
    assert audit["rows"][0]["verdict"] == "needs_manual_review"
    assert "dealaware_amo_tallanto_mismatch" in audit["rows"][0]["not_ready_reasons"]


def test_output_root_must_not_be_under_stable_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path)
    unsafe = DealAwareTimelineSampleConfig(
        selected_groups_csv=config.selected_groups_csv,
        all_candidates_csv=config.all_candidates_csv,
        master_calls_csv=config.master_calls_csv,
        master_contacts_csv=config.master_contacts_csv,
        allowed_root=tmp_path,
        out_root=tmp_path / "stable_runtime" / "customer_timeline",
        timeline_db=tmp_path / "stable_runtime" / "customer_timeline" / "customer_timeline.sqlite",
        generated_at=NOW,
    )

    with pytest.raises(ValueError, match="stable_runtime"):
        build_deal_aware_timeline_sample(unsafe)
