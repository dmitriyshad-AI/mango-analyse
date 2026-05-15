from __future__ import annotations

import json

from mango_mvp.deal_aware.deal_text_builder import (
    DEAL_AI_OPTIONAL_FIELDS,
    DEAL_AI_REQUIRED_FIELDS,
    build_commercial_payload,
    classify_budget_range,
)
from mango_mvp.deal_aware.deal_writeback import build_dry_run_row, validate_field_catalog


def _required_row() -> dict[str, str]:
    row = {field: f"{field}: безопасный тестовый текст" for field in DEAL_AI_REQUIRED_FIELDS}
    row.update(
        {
            "review_id": "deal-stage5-00001",
            "selected_deal_id": "123",
            "stage5_decision": "allow_stage6_dry_run",
            "AI-дата следующего касания": "2026-05-15",
            "AI-дата обновления сделки": "2026-05-13T11:57:33+00:00",
        }
    )
    return row


def _field_catalog(*, include_optional: bool = False) -> list[dict[str, object]]:
    fields = list(DEAL_AI_REQUIRED_FIELDS) + (list(DEAL_AI_OPTIONAL_FIELDS) if include_optional else [])
    return [
        {
            "id": 2000 + index,
            "name": field,
            "type": "date_time" if field == "AI-дата обновления сделки" else "textarea",
            "is_api_only": False,
        }
        for index, field in enumerate(fields)
    ]


def test_commercial_payload_uses_latest_supported_budget() -> None:
    payload = build_commercial_payload(
        [
            {"started_at": "2026-05-01", "budget": "до 25 000 рублей"},
            {"started_at": "2026-05-02", "budget": "готовы до 80 тыс рублей"},
        ],
        {},
    )

    assert payload["AI-бюджет диапазон"] == "50k_100k"
    assert "до 25" not in payload.get("AI-бюджет комментарий", "")


def test_budget_range_classifies_supported_ranges() -> None:
    assert classify_budget_range("до 25 000") == "under_30k"
    assert classify_budget_range("40 тыс рублей") == "30k_50k"
    assert classify_budget_range("80 000") == "50k_100k"
    assert classify_budget_range("120000 рублей") == "100k_150k"
    assert classify_budget_range("200 тыс") == "over_150k"
    assert classify_budget_range("маткапитал") == "matcapital_or_certificate"
    assert classify_budget_range("нужна рассрочка") == "installment_needed"


def test_budget_range_keeps_comment_for_complex_budget_signal() -> None:
    payload = build_commercial_payload(
        [{"budget": "есть маткапитал, сумму доплаты пока обсуждают с супругом"}],
        {},
    )

    assert payload["AI-бюджет диапазон"] == "matcapital_or_certificate"
    assert "маткапитал" in payload["AI-бюджет комментарий"]


def test_commercial_payload_does_not_treat_course_price_as_client_budget() -> None:
    assert classify_budget_range("курс стоит 50 000 рублей") == "not_applicable"
    payload = build_commercial_payload(
        [{"call_summary": "Клиент спросил, сколько стоит курс. Менеджер сказал, что курс стоит 50 000 рублей."}],
        {},
    )

    assert "AI-бюджет диапазон" not in payload


def test_discount_interest_requires_discount_or_installment_signal() -> None:
    price_only = build_commercial_payload(
        [{"call_summary": "Клиент спросил, сколько стоит курс."}],
        {},
    )
    discount = build_commercial_payload(
        [{"call_summary": "Клиент спросил, есть ли скидка или рассрочка."}],
        {},
    )

    assert price_only.get("AI-интерес к скидке") != "yes"
    assert discount["AI-интерес к скидке"] == "yes"


def test_optional_commercial_fields_do_not_block_when_missing_by_default() -> None:
    row = _required_row()
    row["AI-бюджет диапазон"] = "50k_100k"
    catalog = _field_catalog(include_optional=False)
    field_guard = validate_field_catalog(catalog)

    dry_row, findings = build_dry_run_row(
        row,
        row_index=1,
        field_catalog=catalog,
        field_guard=field_guard,
        analysis_date="2026-05-13",
    )
    payload = json.loads(dry_row["preview_payload"])

    assert dry_row["stage6_status"] == "dry_run"
    assert findings == []
    assert "AI-бюджет диапазон" not in payload
    assert dry_row["optional_commercial_fields_missing_in_catalog"] == "AI-бюджет диапазон"


def test_require_commercial_fields_blocks_when_catalog_missing_fields() -> None:
    catalog = _field_catalog(include_optional=False)
    field_guard = validate_field_catalog(catalog, require_commercial_fields=True)
    dry_row, findings = build_dry_run_row(
        _required_row(),
        row_index=1,
        field_catalog=catalog,
        field_guard=field_guard,
        analysis_date="2026-05-13",
        require_commercial_fields=True,
    )

    assert dry_row["stage6_status"] == "blocked"
    assert dry_row["stage6_reason"] == "missing_optional_commercial_fields"
    assert [finding["risk_type"] for finding in findings] == ["missing_optional_commercial_fields"]


def test_commercial_fields_are_included_when_catalog_supports_them() -> None:
    row = _required_row()
    row["AI-бюджет диапазон"] = "50k_100k"
    row["AI-чувствительность к цене"] = "high"
    row["AI-интерес к скидке"] = "yes"
    catalog = _field_catalog(include_optional=True)
    field_guard = validate_field_catalog(catalog)

    dry_row, findings = build_dry_run_row(
        row,
        row_index=1,
        field_catalog=catalog,
        field_guard=field_guard,
        analysis_date="2026-05-13",
    )
    payload = json.loads(dry_row["preview_payload"])

    assert findings == []
    assert payload["AI-бюджет диапазон"] == "50k_100k"
    assert payload["AI-чувствительность к цене"] == "high"
    assert payload["AI-интерес к скидке"] == "yes"


def test_discount_promise_without_policy_is_blocked() -> None:
    row = _required_row()
    row["AI-рекомендованный следующий шаг"] = "Предоставить клиенту скидку 10 процентов."
    row["AI-интерес к скидке"] = "yes"
    catalog = _field_catalog(include_optional=True)
    field_guard = validate_field_catalog(catalog)

    dry_row, findings = build_dry_run_row(
        row,
        row_index=1,
        field_catalog=catalog,
        field_guard=field_guard,
        analysis_date="2026-05-13",
    )

    assert dry_row["stage6_status"] == "blocked"
    assert "discount_promise_without_policy" in {finding["risk_type"] for finding in findings}
