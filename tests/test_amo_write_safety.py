from __future__ import annotations

import json
from pathlib import Path

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


def _catalog_for(field_name: str) -> list[dict[str, object]]:
    return [{"id": 1000, "name": field_name, "type": "textarea"}]


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
    assert allowed_payload_after_pre_patch({"AI-сводка по сделке": "новая сводка"}, decisions) == {}


def test_pre_patch_write_decision_blocks_contact_and_lead_clobber_with_positive_control(tmp_path: Path) -> None:
    cases = [
        {
            "entity_type": "lead",
            "entity_id": "123",
            "field_name": "AI-сводка по сделке",
            "old_value": "старая сводка сделки",
            "new_value": "новая сводка сделки",
            "changed_current": "ручная правка сделки",
        },
        {
            "entity_type": "contact",
            "entity_id": "777",
            "field_name": "Авто история общения",
            "old_value": "старая история контакта",
            "new_value": "новая история контакта",
            "changed_current": "ручная правка контакта",
        },
    ]
    for index, case in enumerate(cases, start=1):
        snapshot_rows = build_pre_write_snapshot_rows(
            batch_id="batch",
            input_csv=tmp_path / "input.csv",
            input_sha256="abc",
            row_index=index,
            review_id=f"review-{index}",
            lead_id=str(case["entity_id"]),
            entity_type=str(case["entity_type"]),
            entity_id=str(case["entity_id"]),
            payload={str(case["field_name"]): str(case["new_value"])},
            current_lead=_entity_with_values({str(case["field_name"]): str(case["old_value"])}),
            field_catalog=_catalog_for(str(case["field_name"])),
            operator_approval_path=None,
        )

        blocked = pre_patch_write_decisions(
            snapshot_rows=snapshot_rows,
            current_entity=_entity_with_values({str(case["field_name"]): str(case["changed_current"])}),
        )
        allowed = pre_patch_write_decisions(
            snapshot_rows=snapshot_rows,
            current_entity=_entity_with_values({str(case["field_name"]): str(case["old_value"])}),
        )

        assert blocked[0]["entity_type"] == case["entity_type"]
        assert blocked[0]["entity_id"] == case["entity_id"]
        assert blocked[0]["field_name"] == case["field_name"]
        assert blocked[0]["action"] == "clobber_protected"
        assert blocked[0]["reason"] == "current_value_changed_since_snapshot"
        assert allowed_payload_after_pre_patch({str(case["field_name"]): str(case["new_value"])}, blocked) == {}
        assert allowed[0]["action"] == "allowed"
        assert allowed_payload_after_pre_patch({str(case["field_name"]): str(case["new_value"])}, allowed) == {
            str(case["field_name"]): str(case["new_value"])
        }


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
