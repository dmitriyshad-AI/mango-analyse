from __future__ import annotations

from scripts import readback_amo_contact_writeback as readback


def _contact(custom_fields: dict[str, str]) -> dict:
    return {
        "id": 123,
        "custom_fields_values": [
            {"field_name": name, "values": [{"value": value}]}
            for name, value in custom_fields.items()
        ],
    }


def _clean_fields() -> dict[str, str]:
    return {
        "Статус матчинга": "exact_phone_single",
        "AI-приоритет": "warm",
        "AI-рекомендованный следующий шаг": "Отправить ссылку на оплату",
        "Последняя AI-сводка": "Клиент готов получить ссылку на оплату.",
        "Авто история общения": "Сводка клиента: интерес к летнему лагерю. Следующий шаг: отправить ссылку на оплату.",
    }


def test_extract_custom_field_values_flattens_amo_values() -> None:
    values = readback.extract_custom_field_values(
        {
            "custom_fields_values": [
                {"field_name": "Авто история общения", "values": [{"value": "строка 1"}, {"value": "строка 2"}]},
                {"field_name": "AI-приоритет", "values": [{"value": "warm"}]},
            ]
        }
    )

    assert values["Авто история общения"] == "строка 1 | строка 2"
    assert values["AI-приоритет"] == "warm"


def test_readback_allows_clean_written_contact() -> None:
    rows = [{"row_index": "1", "status": "written", "phone": "+79000000000", "contact_id": "123"}]

    report = readback.evaluate_readback_rows(rows, fetch_contact=lambda _contact_id: _contact(_clean_fields()))

    assert report[0]["decision"] == "allow"
    assert report[0]["readback_status"] == "evaluated"
    assert report[0]["risk_types"] == ""


def test_readback_blocks_lossy_ellipsis_after_amo_storage() -> None:
    fields = _clean_fields()
    fields["Авто история общения"] = "Клиент уточнил оплату и текст оборвался..."
    rows = [{"row_index": "1", "status": "written", "phone": "+79000000000", "contact_id": "123"}]

    report = readback.evaluate_readback_rows(rows, fetch_contact=lambda _contact_id: _contact(fields))

    assert report[0]["decision"] == "block"
    assert "lossy_ellipsis_truncation" in report[0]["risk_types"]


def test_readback_blocks_missing_target_field() -> None:
    fields = _clean_fields()
    del fields["Авто история общения"]
    rows = [{"row_index": "1", "status": "written", "phone": "+79000000000", "contact_id": "123"}]

    report = readback.evaluate_readback_rows(rows, fetch_contact=lambda _contact_id: _contact(fields))

    assert report[0]["decision"] == "block"
    assert "missing_readback_target_field" in report[0]["risk_types"]
    assert "empty_auto_history" in report[0]["risk_types"]


def test_readback_blocks_value_that_differs_from_written_payload() -> None:
    expected = _clean_fields()
    actual = dict(expected)
    actual["AI-рекомендованный следующий шаг"] = "Позвонить завтра"
    rows = [
        {
            "row_index": "1",
            "status": "written",
            "phone": "+79000000000",
            "contact_id": "123",
            "preview_payload": __import__("json").dumps(expected, ensure_ascii=False),
        }
    ]

    report = readback.evaluate_readback_rows(rows, fetch_contact=lambda _contact_id: _contact(actual))

    assert report[0]["decision"] == "block"
    assert "readback_value_mismatch" in report[0]["risk_types"]


def test_readback_skips_non_written_source_rows_by_default() -> None:
    rows = [{"row_index": "1", "status": "dry_run", "phone": "+79000000000", "contact_id": "123"}]

    report = readback.evaluate_readback_rows(rows, fetch_contact=lambda _contact_id: _contact(_clean_fields()))

    assert report[0]["decision"] == "skip"
    assert report[0]["readback_status"] == "skipped"


def test_split_keeps_risk_type_names_with_pipe_separator() -> None:
    assert readback._split("duplicate_label_and_count | lossy_ellipsis_truncation") == [
        "duplicate_label_and_count",
        "lossy_ellipsis_truncation",
    ]


def test_split_keeps_whitespace_status_list_support() -> None:
    assert readback._split("written dry_run") == ["written", "dry_run"]
