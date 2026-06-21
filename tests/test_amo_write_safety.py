from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.amocrm_runtime.amo_integration import (
    CONTACT_WRITE_ALLOWED_FIELDS,
    CONTACT_WRITE_PROTECTED_FIELDS,
    LEAD_WRITE_ALLOWED_FIELDS,
    build_custom_fields_values,
)
from mango_mvp.deal_aware.amo_rollback import build_pre_write_snapshot_rows
from mango_mvp.deal_aware.amo_write_safety import (
    allowed_payload_after_pre_patch,
    append_write_journal_rows,
    journal_rows_from_decisions,
    load_last_written_sha,
    pre_patch_write_decisions,
)


def _entity_with_values(values: dict[str, str]) -> dict[str, object]:
    return {
        "id": 123,
        "custom_fields_values": [
            {"field_id": index, "field_name": field, "values": [{"value": value}]}
            for index, (field, value) in enumerate(values.items(), start=1000)
        ],
    }


def _field_catalog() -> list[dict[str, object]]:
    return [{"id": 1000, "name": "AI-сводка по сделке", "type": "textarea"}]


def _target_field_catalog() -> list[dict[str, object]]:
    return [
        {"id": 1001, "name": "AI-сводка по сделке", "type": "textarea"},
        {"id": 1002, "name": "Авто история общения", "type": "textarea"},
    ]


def _snapshot_rows_for_target_fields(tmp_path: Path) -> list[dict[str, object]]:
    field_catalog = _target_field_catalog()
    lead_rows = build_pre_write_snapshot_rows(
        batch_id="batch",
        input_csv=tmp_path / "input.csv",
        input_sha256="abc",
        row_index=1,
        review_id="lead-row",
        lead_id="123",
        payload={"AI-сводка по сделке": "ЧЕРНОВИК БОТА: новая сводка"},
        current_lead=_entity_with_values({"AI-сводка по сделке": "старая сводка"}),
        field_catalog=field_catalog,
        operator_approval_path=None,
        snapshot_taken_at="2026-06-21T10:00:00+00:00",
    )
    contact_rows = build_pre_write_snapshot_rows(
        batch_id="batch",
        input_csv=tmp_path / "input.csv",
        input_sha256="abc",
        row_index=1,
        review_id="contact-row",
        lead_id="123",
        entity_type="contact",
        entity_id="777",
        payload={"Авто история общения": "ЧЕРНОВИК БОТА: новая автоистория"},
        current_lead=_entity_with_values({"Авто история общения": "старая автоистория"}),
        field_catalog=field_catalog,
        operator_approval_path=None,
        snapshot_taken_at="2026-06-21T10:00:00+00:00",
    )
    return lead_rows + contact_rows


def test_pre_patch_write_decision_blocks_manager_clobber(tmp_path: Path) -> None:
    snapshot_rows = build_pre_write_snapshot_rows(
        batch_id="batch",
        input_csv=tmp_path / "input.csv",
        input_sha256="abc",
        row_index=1,
        review_id="review",
        lead_id="123",
        payload={"AI-сводка по сделке": "новая сводка"},
        current_lead=_entity_with_values({"AI-сводка по сделке": "старая сводка"}),
        field_catalog=_field_catalog(),
        operator_approval_path=None,
    )

    decisions = pre_patch_write_decisions(
        snapshot_rows=snapshot_rows,
        current_entity=_entity_with_values({"AI-сводка по сделке": "ручная правка"}),
    )

    assert decisions[0]["action"] == "clobber_protected"
    assert decisions[0]["reason"] == "current_value_changed_since_snapshot"
    assert allowed_payload_after_pre_patch({"AI-сводка по сделке": "новая сводка"}, decisions) == {}


def test_pre_patch_write_decision_blocks_contact_clobber(tmp_path: Path) -> None:
    snapshot_rows = build_pre_write_snapshot_rows(
        batch_id="batch",
        input_csv=tmp_path / "input.csv",
        input_sha256="abc",
        row_index=1,
        review_id="contact-row",
        lead_id="777",
        entity_type="contact",
        entity_id="777",
        payload={"Авто история общения": "новая история"},
        current_lead=_entity_with_values({"Авто история общения": "старая история"}),
        field_catalog=[{"id": 1000, "name": "Авто история общения", "type": "textarea"}],
        operator_approval_path=None,
    )

    decisions = pre_patch_write_decisions(
        snapshot_rows=snapshot_rows,
        current_entity=_entity_with_values({"Авто история общения": "ручная правка менеджера"}),
    )

    assert decisions[0]["entity_type"] == "contact"
    assert decisions[0]["entity_id"] == "777"
    assert decisions[0]["field_name"] == "Авто история общения"
    assert decisions[0]["action"] == "clobber_protected"
    assert decisions[0]["reason"] == "current_value_changed_since_snapshot"
    allowed_payload = allowed_payload_after_pre_patch({"Авто история общения": "новая история"}, decisions)
    assert allowed_payload == {}
    assert build_custom_fields_values(allowed_payload, _target_field_catalog()) == []


def test_pre_patch_snapshot_to_patch_blocks_changed_target_fields_and_allows_unchanged(tmp_path: Path) -> None:
    snapshot_rows = _snapshot_rows_for_target_fields(tmp_path)
    target_payload = {
        "AI-сводка по сделке": "ЧЕРНОВИК БОТА: новая сводка",
        "Авто история общения": "ЧЕРНОВИК БОТА: новая автоистория",
    }
    field_catalog = _target_field_catalog()

    blocked_decisions = pre_patch_write_decisions(
        snapshot_rows=snapshot_rows,
        current_entity=_entity_with_values(
            {
                "AI-сводка по сделке": "ручная правка сделки после снимка",
                "Авто история общения": "ручная правка контакта после снимка",
                "Телефон": "+79990000000",
                "ФИО": "Иван Иванов",
                "История общения": "ручная история",
            }
        ),
    )

    assert {
        (item["entity_type"], item["field_name"], item["action"], item["reason"]) for item in blocked_decisions
    } == {
        ("lead", "AI-сводка по сделке", "clobber_protected", "current_value_changed_since_snapshot"),
        ("contact", "Авто история общения", "clobber_protected", "current_value_changed_since_snapshot"),
    }
    blocked_payload = allowed_payload_after_pre_patch(target_payload, blocked_decisions)
    assert blocked_payload == {}
    assert build_custom_fields_values(blocked_payload, field_catalog) == []

    allowed_decisions = pre_patch_write_decisions(
        snapshot_rows=snapshot_rows,
        current_entity=_entity_with_values(
            {
                "AI-сводка по сделке": "старая сводка",
                "Авто история общения": "старая автоистория",
                "Телефон": "+79990000000",
                "ФИО": "Иван Иванов",
                "История общения": "ручная история",
            }
        ),
    )

    assert {
        (item["entity_type"], item["field_name"], item["action"], item["reason"]) for item in allowed_decisions
    } == {
        ("lead", "AI-сводка по сделке", "allowed", "current_matches_snapshot"),
        ("contact", "Авто история общения", "allowed", "current_matches_snapshot"),
    }
    allowed_payload = allowed_payload_after_pre_patch(target_payload, allowed_decisions)
    assert allowed_payload == target_payload
    patch_fields = build_custom_fields_values(allowed_payload, field_catalog)
    assert {item["field_id"] for item in patch_fields} == {1001, 1002}
    assert [item["values"][0]["value"] for item in patch_fields] == list(target_payload.values())


def test_write_allowed_lists_do_not_include_human_owned_identity_fields() -> None:
    protected_human_fields = {"Телефон", "Телефон клиента", "ФИО", "История общения"}

    assert protected_human_fields <= CONTACT_WRITE_PROTECTED_FIELDS
    assert protected_human_fields.isdisjoint(CONTACT_WRITE_ALLOWED_FIELDS)
    assert protected_human_fields.isdisjoint(LEAD_WRITE_ALLOWED_FIELDS)


def test_pre_patch_write_decision_skips_unchanged_repeat(tmp_path: Path) -> None:
    snapshot_rows = build_pre_write_snapshot_rows(
        batch_id="batch",
        input_csv=tmp_path / "input.csv",
        input_sha256="abc",
        row_index=1,
        review_id="review",
        lead_id="123",
        payload={"AI-сводка по сделке": "новая сводка"},
        current_lead=_entity_with_values({"AI-сводка по сделке": "новая сводка"}),
        field_catalog=_field_catalog(),
        operator_approval_path=None,
    )

    decisions = pre_patch_write_decisions(
        snapshot_rows=snapshot_rows,
        current_entity=_entity_with_values({"AI-сводка по сделке": "новая сводка"}),
    )

    assert decisions[0]["action"] == "skipped"
    assert decisions[0]["reason"] == "unchanged"


def test_dry_run_journal_does_not_become_last_written_sha(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    append_write_journal_rows(
        journal,
        [
            {
                "entity_type": "lead",
                "entity_id": "123",
                "field": "AI-сводка по сделке",
                "action": "written-dry",
                "after_sha": "dry-sha",
            }
        ],
    )

    assert load_last_written_sha(journal, entity_type="lead", entity_id="123") == {}

    append_write_journal_rows(
        journal,
        [
            {
                "entity_type": "lead",
                "entity_id": "123",
                "field": "AI-сводка по сделке",
                "action": "written",
                "after_sha": "real-sha",
            }
        ],
    )

    assert load_last_written_sha(journal, entity_type="lead", entity_id="123") == {"AI-сводка по сделке": "real-sha"}
    assert all(json.loads(line)["schema_version"] for line in journal.read_text(encoding="utf-8").splitlines())


def test_contact_snapshot_rows_keep_entity_identity(tmp_path: Path) -> None:
    rows = build_pre_write_snapshot_rows(
        batch_id="batch",
        input_csv=tmp_path / "input.csv",
        input_sha256="abc",
        row_index=1,
        review_id="contact-row",
        lead_id="777",
        entity_type="contact",
        entity_id="777",
        payload={"Авто история общения": "новая история"},
        current_lead=_entity_with_values({"Авто история общения": "старая история"}),
        field_catalog=[{"id": 1000, "name": "Авто история общения", "type": "textarea"}],
        operator_approval_path=None,
    )

    assert rows[0]["entity_type"] == "contact"
    assert rows[0]["entity_id"] == "777"
