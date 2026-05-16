from __future__ import annotations

import csv
from pathlib import Path

from mango_mvp.deal_aware.deal_quality_gate import evaluate_row
from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS
from mango_mvp.question_catalog.extractors import extract_call_questions
from mango_mvp.question_catalog.source_index import build_source_index, load_source_index, write_source_index


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _payload(next_step: str = "Связаться с клиентом и уточнить актуальный интерес.") -> dict[str, str]:
    payload = {field: "Безопасный тестовый текст." for field in DEAL_AI_FIELDS}
    payload["AI-рекомендованный следующий шаг"] = next_step
    payload["AI-дата следующего касания"] = "2026-05-15"
    payload["AI-приоритет сделки"] = "warm"
    return payload


def _row(call_ids: str = "call-1") -> dict[str, str]:
    return {
        "review_id": "deal-stage5-00001",
        "selected_deal_id": "123",
        "selected_status_name": "Переговоры",
        "selected_loss_reason": "",
        "crm_text_quality_passed": "Да",
        "quality_risk_types": "",
        "candidate_phone_count": "1",
        "tallanto_context_status": "exact_phone_single",
        "stage3_risk_flags": "",
        "call_ids": call_ids,
    }


def test_call_question_extractor_keeps_raw_call_id_in_metadata(tmp_path: Path) -> None:
    source = tmp_path / "enriched_reviews.csv"
    _write_csv(
        source,
        [
            {
                "started_at": "2026-05-10T10:00:00+00:00",
                "call_id": "call-raw-1",
                "recording_id": "rec-1",
                "moment_id": "moment-1",
                "customer_question": "Сколько стоит курс?",
            }
        ],
    )

    items, _ = extract_call_questions(source, tenant_id="foton")

    assert items[0].metadata["call_id"] == "call-raw-1"
    assert items[0].metadata["recording_id"] == "rec-1"
    assert items[0].metadata["moment_id"] == "moment-1"
    assert items[0].metadata["source_kind"] == "call"
    assert items[0].metadata["source_table"] == "enriched_reviews.csv"


def test_source_index_maps_call_id_to_theme_id(tmp_path: Path) -> None:
    source = tmp_path / "enriched_reviews.csv"
    _write_csv(
        source,
        [
            {
                "started_at": "2026-05-10T10:00:00+00:00",
                "call_id": "call-raw-1",
                "customer_question": "Сколько стоит курс?",
            }
        ],
    )
    items, _ = extract_call_questions(source, tenant_id="foton")

    index = build_source_index(items)

    assert "call-raw-1" in index
    assert index["call-raw-1"]["theme_ids"] or index["call-raw-1"]["service_ids"]


def test_source_index_json_roundtrip_preserves_manager_only_mode(tmp_path: Path) -> None:
    rows = [
        {
            "call_id": "call-1",
            "theme_ids": "theme:009_refund",
            "service_ids": "",
            "policy_statuses": "manager_only",
            "bot_allowed_modes": "manager_only",
            "risk_flags": "manager_only",
        }
    ]

    output = write_source_index(tmp_path, rows)
    index = load_source_index(Path(output["json"]))

    assert index["call-1"]["theme_ids"] == ["theme:009_refund"]
    assert index["call-1"]["bot_allowed_modes"] == ["manager_only"]
    assert index["call-1"]["risk_flags"] == ["manager_only"]


def test_deal_quality_gate_without_catalog_index_is_backward_compatible() -> None:
    hard, warnings = evaluate_row(
        _row(),
        _payload(),
        row_index=1,
        analysis_date="2026-05-13",
        question_index=None,
    )

    assert hard == []
    assert [item["gate_type"] for item in warnings] == []


def test_deal_quality_gate_blocks_service_theme_with_sales_next_step() -> None:
    hard, _ = evaluate_row(
        _row(),
        _payload("Связаться с клиентом с предложением курса."),
        row_index=1,
        analysis_date="2026-05-13",
        question_index={"call-1": {"theme_ids": ["theme:013_schedule"], "service_ids": [], "policy_statuses": [], "bot_allowed_modes": [], "risk_flags": []}},
    )

    assert "catalog_service_theme_sales_next_step" in {item["gate_type"] for item in hard}


def test_deal_quality_gate_blocks_manager_only_theme_with_autonomous_action() -> None:
    hard, _ = evaluate_row(
        _row(),
        _payload("Отправить клиенту готовое решение автоматически."),
        row_index=1,
        analysis_date="2026-05-13",
        question_index={"call-1": {"theme_ids": ["theme:refund"], "service_ids": [], "policy_statuses": [], "bot_allowed_modes": ["manager_only"], "risk_flags": []}},
    )

    assert "catalog_manager_only_theme_autonomous_action" in {item["gate_type"] for item in hard}


def test_payment_theme_strengthens_payment_next_step_conflict() -> None:
    hard, _ = evaluate_row(
        _row(),
        _payload("Проверить оплату и отправить чек."),
        row_index=1,
        analysis_date="2026-05-13",
        question_index={"call-1": {"theme_ids": ["theme:payment_receipt"], "service_ids": [], "policy_statuses": [], "bot_allowed_modes": [], "risk_flags": []}},
    )

    assert "catalog_payment_theme_next_step_conflict" in {item["gate_type"] for item in hard}


def test_catalog_index_missing_call_id_is_warning_not_crash() -> None:
    hard, warnings = evaluate_row(
        _row("missing-call"),
        _payload(),
        row_index=1,
        analysis_date="2026-05-13",
        question_index={"other-call": {"theme_ids": ["theme:001_pricing"], "service_ids": [], "policy_statuses": [], "bot_allowed_modes": [], "risk_flags": []}},
    )

    assert hard == []
    assert "catalog_index_missing_call_id" in {item["gate_type"] for item in warnings}
