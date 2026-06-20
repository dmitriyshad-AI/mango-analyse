from __future__ import annotations

import json

import pytest

from mango_mvp.deal_aware.deal_text_builder import (
    DEAL_AI_REQUIRED_FIELDS,
    build_deal_payload,
    build_objections,
    build_structured_objections,
    history_call_summary,
    normalize_manager_text,
    normalize_objection,
    short_evidence,
    serialize_structured_objections,
)
from mango_mvp.deal_aware.deal_writeback import build_dry_run_row, validate_field_catalog


def _call(**kwargs: str) -> dict[str, str]:
    base = {
        "call_id": "call-1",
        "started_at": "2026-05-10 10:00:00",
        "manager_name": "Менеджер",
        "call_summary": "Клиент сомневается и спрашивает про стоимость.",
        "next_step": "Перезвонить",
        "objections": "цена",
    }
    base.update(kwargs)
    return base


def _required_row() -> dict[str, str]:
    row = {field: f"{field}: безопасный тестовый текст" for field in DEAL_AI_REQUIRED_FIELDS}
    row.update(
        {
            "review_id": "deal-stage5-00001",
            "selected_deal_id": "123",
            "stage5_decision": "allow_stage6_dry_run",
            "AI-дата следующего касания": "2026-05-15",
            "AI-дата обновления сделки": "2026-05-13T11:57:33+00:00",
            "AI-возражения структура": "[{}]",
        }
    )
    return row


def _field_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": 3000 + index,
            "name": field,
            "type": "date_time" if field == "AI-дата обновления сделки" else "textarea",
            "is_api_only": False,
        }
        for index, field in enumerate(DEAL_AI_REQUIRED_FIELDS)
    ]


def test_structured_objections_keep_source_call_id_and_date() -> None:
    rows = build_structured_objections(
        [_call(call_id="call-42", started_at="2026-05-10 12:30:00", manager_name="Анна")],
        {},
    )

    assert rows[0]["source_call_id"] == "call-42"
    assert rows[0]["source_started_at"] == "2026-05-10 12:30:00"
    assert rows[0]["source_manager"] == "Анна"
    assert rows[0]["category"] == "цена"


def test_human_objections_field_remains_backward_compatible() -> None:
    text = build_objections([_call(objections="цена | время")], {})

    assert text == "Актуальные: есть вопрос по стоимости; нужно согласовать удобное время или дату."


def test_structured_objections_are_deduped_without_losing_latest_source() -> None:
    rows = build_structured_objections(
        [
            _call(call_id="old", started_at="2026-05-01 10:00:00", objections="цена"),
            _call(call_id="new", started_at="2026-05-02 10:00:00", objections="цена"),
        ],
        {},
    )

    assert len(rows) == 1
    assert rows[0]["source_call_id"] == "new"
    assert rows[0]["is_latest"] == "Да"


def test_structured_objections_do_not_invent_category() -> None:
    rows = build_structured_objections([_call(objections="нужно обсудить дома")], {})

    assert rows[0]["category"] == "other"


def test_objections_json_truncation_is_reported() -> None:
    rows = [
        {
            "objection_id": str(index),
            "text": f"возражение {index}",
            "normalized_text": f"возражение {index}",
            "source_call_id": f"call-{index}",
            "source_started_at": "2026-05-10",
            "source_manager": "Менеджер",
            "source_rank": str(index),
            "is_latest": "Нет",
            "evidence": "x" * 200,
            "category": "other",
        }
        for index in range(10)
    ]
    payload, truncated = serialize_structured_objections(rows, max_items=3, max_chars=500)

    assert truncated is True
    assert len(json.loads(payload)) <= 3


def test_structured_objections_are_not_sent_to_amo_payload_by_default() -> None:
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "ЛВШ",
            "selected_status_name": "Перспектива",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "full_active",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-05-05 10:00:00",
        },
        [],
        [_call()],
        tallanto_context={"text": "Tallanto: нет точного сопоставления."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )
    catalog = _field_catalog()
    dry_row, _ = build_dry_run_row(
        _required_row(),
        row_index=1,
        field_catalog=catalog,
        field_guard=validate_field_catalog(catalog),
        analysis_date="2026-05-13",
    )
    preview_payload = json.loads(dry_row["preview_payload"])

    assert "AI-возражения структура" not in payload
    assert "AI-возражения структура" not in preview_payload


def test_empty_objections_keep_current_human_fallback_text() -> None:
    assert build_objections([], {}) == "Актуальные возражения в релевантных звонках не выделены."


def test_long_objection_is_compacted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRM_OBJECTION_COMPACT", raising=False)
    text = (
        "клиент подробно сомневается из-за стоимости программы и хочет сначала обсудить бюджет "
        "с родителями перед оплатой"
    )

    normalized = normalize_objection(text)

    assert normalized
    assert len(normalized) <= 90
    assert normalized.endswith("…")
    assert "Текст сокращен" not in normalized
    assert "роди…" not in normalized


def test_long_objection_keeps_old_drop_behavior_when_compaction_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRM_OBJECTION_COMPACT", "0")

    assert (
        normalize_objection(
            "клиент подробно сомневается из-за стоимости программы и хочет сначала обсудить бюджет "
            "с родителями перед оплатой"
        )
        == ""
    )


def test_long_objection_uses_explicit_compaction_marker_when_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRM_DEAL_OBJECTION_EXPLICIT_COMPACT", "1")
    text = (
        "клиент подробно сомневается из-за стоимости программы и хочет сначала обсудить бюджет "
        "с родителями перед оплатой"
    )

    normalized = normalize_objection(text)
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "ЛВШ",
            "selected_status_name": "Перспектива",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "full_active",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-05-05 10:00:00",
        },
        [],
        [_call(objections=text)],
        tallanto_context={"text": "Tallanto: нет точного сопоставления."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )

    assert len(normalized) <= 90
    assert normalized.endswith("[сжато]")
    assert "…" not in normalized
    assert "[сжато]" in payload["AI-актуальные возражения"]


def test_short_dictionary_objection_remains_backward_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRM_OBJECTION_COMPACT", raising=False)

    assert normalize_objection("цена") == "есть вопрос по стоимости"


def test_manager_text_drops_truncation_mark_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRM_KEEP_TRUNCATION_MARK", raising=False)

    assert normalize_manager_text("Клиент [сжато] обсуждает оплату") == "Клиент обсуждает оплату"


def test_manager_text_preserves_single_truncation_mark_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRM_KEEP_TRUNCATION_MARK", "1")

    normalized = normalize_manager_text("Клиент [truncated] подробно обсуждает оплату... [сжато]")

    assert normalized == "Клиент подробно обсуждает оплату [сжато]"
    assert normalized.count("[сжато]") == 1
    assert "..." not in normalized
    assert "…" not in normalized


def test_short_evidence_compacts_on_word_boundary() -> None:
    source = " ".join(["Клиент спокойно обсуждает стоимость программы"] * 12)

    evidence = short_evidence({"call_summary": source})

    assert len(evidence) <= 160
    assert evidence.endswith("[сжато]")
    assert "програм [сжато]" not in evidence


def test_history_call_summary_fallback_compacts_on_word_boundary() -> None:
    source = " ".join(["Клиент спокойно обсуждает стоимость программы"] * 10)

    summary = history_call_summary(source, max_sentences=1, max_chars=120)

    assert len(summary) <= 120
    assert summary.endswith(". Детали в полном звонке.")
    assert "програм. Детали" not in summary
