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
