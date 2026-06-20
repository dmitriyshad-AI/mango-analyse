from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.deal_aware.amo_rollback import extract_custom_field_values, sha256_text, utc_now_iso
from mango_mvp.deal_aware.stage1_snapshot import safe_text, stringify


AMO_WRITEBACK_JOURNAL_SCHEMA_VERSION = "amo_writeback_journal_v1"
DEFAULT_AMO_WRITEBACK_JOURNAL = Path.home() / ".mango_local" / "amo_writeback" / "journal.jsonl"


def snapshot_entity_type(snapshot_row: Mapping[str, Any]) -> str:
    entity_type = safe_text(snapshot_row.get("entity_type")) or "lead"
    return entity_type if entity_type in {"lead", "contact"} else "lead"


def snapshot_entity_id(snapshot_row: Mapping[str, Any]) -> str:
    return safe_text(snapshot_row.get("entity_id")) or safe_text(snapshot_row.get("lead_id"))


def load_last_written_sha(
    journal_path: Path | None,
    *,
    entity_type: str,
    entity_id: str | int,
) -> dict[str, str]:
    if journal_path is None or not journal_path.exists():
        return {}
    target_type = safe_text(entity_type)
    target_id = safe_text(entity_id)
    result: dict[str, str] = {}
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if safe_text(row.get("entity_type")) != target_type or safe_text(row.get("entity_id")) != target_id:
            continue
        if safe_text(row.get("action")) != "written":
            continue
        field_name = safe_text(row.get("field"))
        after_sha = safe_text(row.get("after_sha"))
        if field_name and after_sha:
            result[field_name] = after_sha
    return result


def pre_patch_write_decisions(
    *,
    snapshot_rows: Sequence[Mapping[str, Any]],
    current_entity: Mapping[str, Any],
    last_written_sha: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not snapshot_rows:
        return [
            {
                "entity_type": "",
                "entity_id": "",
                "field_name": "",
                "action": "skipped",
                "reason": "no_snapshot",
                "current_value": "",
                "before_sha": "",
                "after_sha": "",
            }
        ]
    current_values = extract_custom_field_values(dict(current_entity))
    last_sha = dict(last_written_sha or {})
    result: list[dict[str, Any]] = []
    for row in snapshot_rows:
        field_name = safe_text(row.get("field_name"))
        field_id = safe_text(row.get("field_id"))
        current_value = current_values.get(field_name) or current_values.get(field_id) or ""
        current_sha = sha256_text(current_value)
        new_sha = safe_text(row.get("new_value_sha256")) or sha256_text(row.get("new_value"))
        expected_sha = safe_text(last_sha.get(field_name)) or safe_text(row.get("old_value_sha256"))
        action = "allowed"
        reason = "current_matches_snapshot"
        if not expected_sha:
            action = "skipped"
            reason = "no_snapshot"
        elif current_sha != expected_sha:
            action = "clobber_protected"
            reason = "current_value_changed_since_snapshot"
        elif current_sha == new_sha:
            action = "skipped"
            reason = "unchanged"
        result.append(
            {
                "entity_type": snapshot_entity_type(row),
                "entity_id": snapshot_entity_id(row),
                "field_name": field_name,
                "field_id": field_id,
                "action": action,
                "reason": reason,
                "current_value": current_value,
                "before_sha": current_sha,
                "expected_before_sha": expected_sha,
                "after_sha": new_sha,
                "snapshot_taken_at": safe_text(row.get("snapshot_taken_at")),
            }
        )
    return result


def allowed_payload_after_pre_patch(
    payload: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allowed_fields = {safe_text(item.get("field_name")) for item in decisions if safe_text(item.get("action")) == "allowed"}
    return {field: value for field, value in payload.items() if field in allowed_fields}


def blocking_pre_patch_reasons(decisions: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for decision in decisions:
        action = safe_text(decision.get("action"))
        if action == "allowed":
            continue
        field = safe_text(decision.get("field_name")) or "<unknown>"
        reason = safe_text(decision.get("reason")) or action
        reasons.append(f"{action}:{field}:{reason}")
    return reasons


def append_write_journal_rows(
    journal_path: Path | None,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if journal_path is None:
        return
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            payload = {
                "schema_version": AMO_WRITEBACK_JOURNAL_SCHEMA_VERSION,
                "ts": safe_text(row.get("ts")) or utc_now_iso(),
                "entity_type": safe_text(row.get("entity_type")),
                "entity_id": safe_text(row.get("entity_id")),
                "contact_id": safe_text(row.get("contact_id")),
                "deal_id": safe_text(row.get("deal_id")),
                "field": safe_text(row.get("field")),
                "action": safe_text(row.get("action")),
                "reason": safe_text(row.get("reason")),
                "before_sha": safe_text(row.get("before_sha")),
                "after_sha": safe_text(row.get("after_sha")),
                "snapshot_path": safe_text(row.get("snapshot_path")),
            }
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def journal_rows_from_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    action_for_allowed: str,
    reason_for_allowed: str,
    snapshot_path: Path,
    contact_id: str | int = "",
    deal_id: str | int = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        allowed = safe_text(decision.get("action")) == "allowed"
        action = action_for_allowed if allowed else safe_text(decision.get("action"))
        reason = reason_for_allowed if allowed else safe_text(decision.get("reason"))
        rows.append(
            {
                "entity_type": safe_text(decision.get("entity_type")),
                "entity_id": safe_text(decision.get("entity_id")),
                "contact_id": safe_text(contact_id),
                "deal_id": safe_text(deal_id),
                "field": safe_text(decision.get("field_name")),
                "action": action,
                "reason": reason,
                "before_sha": safe_text(decision.get("before_sha")),
                "after_sha": safe_text(decision.get("after_sha")),
                "snapshot_path": stringify(snapshot_path),
            }
        )
    return rows


__all__ = [
    "AMO_WRITEBACK_JOURNAL_SCHEMA_VERSION",
    "DEFAULT_AMO_WRITEBACK_JOURNAL",
    "allowed_payload_after_pre_patch",
    "append_write_journal_rows",
    "blocking_pre_patch_reasons",
    "journal_rows_from_decisions",
    "load_last_written_sha",
    "pre_patch_write_decisions",
    "snapshot_entity_id",
    "snapshot_entity_type",
]
