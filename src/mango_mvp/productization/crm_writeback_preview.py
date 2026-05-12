from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.crm_entity_resolver import (
    RESOLVE_CRM_ENTITY,
    build_crm_entity_resolution_report,
)
from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


CRM_WRITEBACK_PREVIEW_SCHEMA_VERSION = "crm_writeback_preview_v1"
PREVIEW_READY = "PREVIEW_READY"
BLOCK_MISSING_CRM_ENTITY = "BLOCK_MISSING_CRM_ENTITY"
BLOCK_MISSING_INSIGHT = "BLOCK_MISSING_INSIGHT"
BLOCK_POLICY_FORBIDDEN = "BLOCK_POLICY_FORBIDDEN"
STAGE_LIMITS = {
    "batch_10": 10,
    "batch_50": 50,
    "batch_300": 300,
    "full": None,
}
ALLOWED_LOGICAL_FIELDS = (
    "close_verdict",
    "premature_close_risk",
    "close_reason_summary",
    "recommended_next_step",
    "follow_up_due_at",
    "deal_summary",
    "last_ai_summary",
    "ai_priority",
)
FORBIDDEN_ACTIONS = (
    "close_lead_automatically",
    "delete_contact",
    "merge_contact",
    "send_client_message",
    "overwrite_non_empty_field_without_safe_mode",
)


@dataclass(frozen=True)
class CrmWritebackPreviewSummary:
    schema_version: str
    product_db_path: str
    stage: str
    stage_limit: Optional[int]
    requested_limit: Optional[int]
    calls_seen: int
    selected_items: int
    preview_ready: int
    blocked_missing_crm_entity: int
    blocked_missing_insight: int
    blocked_policy_forbidden: int
    validation_ok: bool
    blocked: int
    warnings: int
    write_crm: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_crm_writeback_preview(
    product_db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
    *,
    stage: str = "batch_10",
    limit: Optional[int] = None,
    include_blocked: bool = True,
    crm_snapshot_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root, out_path = resolve_preview_paths(
        product_db_path=product_db_path,
        product_root=product_root,
        out_path=out_path,
    )
    stage = clean(stage) or "batch_10"
    if stage not in STAGE_LIMITS:
        raise ValueError(f"unknown writeback stage: {stage}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    stage_limit = STAGE_LIMITS[stage]
    effective_limit = min_non_null(stage_limit, limit)
    calls = read_product_calls_for_preview(product_db_path, limit=effective_limit)
    resolver_report = None
    if crm_snapshot_path is not None:
        crm_snapshot_path = crm_snapshot_path.resolve(strict=False)
        if "stable_runtime" in crm_snapshot_path.parts:
            raise ValueError("CRM snapshot must not be under stable_runtime")
        if not path_is_relative_to(crm_snapshot_path, product_root):
            raise ValueError(f"CRM snapshot must stay under product root: {product_root}")
        resolver_report = build_crm_entity_resolution_report(
            product_db_path=product_db_path,
            product_root=product_root,
            crm_snapshot_path=crm_snapshot_path,
            limit=effective_limit,
        )
        calls = apply_crm_resolution(calls, resolver_report)
    items = [build_preview_item(call) for call in calls]
    if not include_blocked:
        items = [item for item in items if clean(item.get("action")) == PREVIEW_READY]

    action_counts = Counter(clean(item.get("action")) for item in items)
    blocked = int(
        action_counts[BLOCK_MISSING_CRM_ENTITY]
        + action_counts[BLOCK_MISSING_INSIGHT]
        + action_counts[BLOCK_POLICY_FORBIDDEN]
    )
    warnings = int(action_counts[BLOCK_MISSING_INSIGHT])
    summary = CrmWritebackPreviewSummary(
        schema_version=CRM_WRITEBACK_PREVIEW_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        stage=stage,
        stage_limit=stage_limit,
        requested_limit=limit,
        calls_seen=len(calls),
        selected_items=len(items),
        preview_ready=int(action_counts[PREVIEW_READY]),
        blocked_missing_crm_entity=int(action_counts[BLOCK_MISSING_CRM_ENTITY]),
        blocked_missing_insight=int(action_counts[BLOCK_MISSING_INSIGHT]),
        blocked_policy_forbidden=int(action_counts[BLOCK_POLICY_FORBIDDEN]),
        validation_ok=True,
        blocked=blocked,
        warnings=warnings,
        write_crm=False,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": dict(sorted(action_counts.items())),
        "items": items,
        "field_mapping": field_mapping_contract(),
        "crm_resolution": compact_crm_resolution(resolver_report),
        "staged_queue": staged_queue_contract(stage=stage, stage_limit=stage_limit, selected=len(items)),
        "policy_gates": policy_gates(),
        "rollback_plan": rollback_plan(),
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def build_preview_item(call: Mapping[str, Any]) -> Mapping[str, Any]:
    blockers = []
    if not clean(call.get("crm_entity_id")):
        blockers.append("missing_crm_entity_id")
    if not has_writeback_insight(call):
        blockers.append("missing_writeback_insight")
    forbidden = forbidden_policy_hits(call)
    blockers.extend(forbidden)
    if "missing_crm_entity_id" in blockers:
        action = BLOCK_MISSING_CRM_ENTITY
        reason = "crm_target_entity_not_resolved"
    elif "missing_writeback_insight" in blockers:
        action = BLOCK_MISSING_INSIGHT
        reason = "analysis_payload_not_ready"
    elif forbidden:
        action = BLOCK_POLICY_FORBIDDEN
        reason = "policy_forbidden_action_requested"
    else:
        action = PREVIEW_READY
        reason = "safe_preview_ready"

    return {
        "schema_version": CRM_WRITEBACK_PREVIEW_SCHEMA_VERSION,
        "action": action,
        "reason": reason,
        "tenant_id": clean(call.get("tenant_id")),
        "crm_provider": "amocrm",
        "crm_entity_type": clean(call.get("crm_entity_type")) or "lead_or_contact_pending_resolution",
        "crm_entity_id": clean(call.get("crm_entity_id")) or None,
        "crm_resolution": call.get("crm_resolution") if isinstance(call.get("crm_resolution"), Mapping) else None,
        "provider": clean(call.get("telephony_provider")),
        "provider_call_id": clean(call.get("provider_call_id")),
        "event_key": clean(call.get("event_key")),
        "recording_id": clean(call.get("recording_id")) or None,
        "started_at": clean(call.get("started_at")) or None,
        "manager_extension": clean(call.get("manager_extension")) or None,
        "crm_owner_id": optional_int(call.get("crm_owner_id")),
        "crm_owner_name": clean(call.get("crm_owner_name")) or None,
        "confidence": confidence_for(call, blockers),
        "blockers": blockers,
        "diff": preview_diff(call),
        "source_metadata": {
            "source_filename": clean(call.get("source_filename")),
            "raw_payload_ref": clean(call.get("raw_payload_ref")) or None,
            "source_repository_ref": clean(call.get("source_repository_ref")) or None,
        },
        "policy": {
            "safe_mode": True,
            "write_crm": False,
            "requires_human_approval": True,
            "forbidden_actions": FORBIDDEN_ACTIONS,
        },
    }


def preview_diff(call: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    base_reason = "pending_insight_layer" if not has_writeback_insight(call) else "ready_from_insight_payload"
    return [
        {
            "logical_field": field,
            "current_value": "unknown_read_only_preview",
            "proposed_value": proposed_value_for(call, field),
            "write_policy": "preview_only",
            "reason": base_reason,
        }
        for field in ALLOWED_LOGICAL_FIELDS
    ]


def proposed_value_for(call: Mapping[str, Any], field: str) -> Optional[str]:
    if not has_writeback_insight(call):
        return None
    if field in {"last_ai_summary", "deal_summary"}:
        return clean(call.get("insight_summary")) or None
    if field == "recommended_next_step":
        return clean(call.get("recommended_next_step")) or None
    if field == "ai_priority":
        return clean(call.get("ai_priority")) or None
    return clean(call.get(field)) or None


def has_writeback_insight(call: Mapping[str, Any]) -> bool:
    return bool(clean(call.get("insight_summary")) or clean(call.get("recommended_next_step")) or clean(call.get("ai_priority")))


def forbidden_policy_hits(call: Mapping[str, Any]) -> list[str]:
    requested_actions = call.get("requested_actions")
    if isinstance(requested_actions, str):
        try:
            requested_actions = json.loads(requested_actions)
        except json.JSONDecodeError:
            requested_actions = []
    if not isinstance(requested_actions, Sequence) or isinstance(requested_actions, (str, bytes)):
        requested_actions = []
    return sorted({clean(action) for action in requested_actions if clean(action) in FORBIDDEN_ACTIONS})


def confidence_for(call: Mapping[str, Any], blockers: Sequence[str]) -> float:
    if blockers:
        return 0.0
    if has_writeback_insight(call):
        return 0.8
    return 0.0


def read_product_calls_for_preview(product_db_path: Path, limit: Optional[int]) -> list[Mapping[str, Any]]:
    limit_sql = "LIMIT ?" if limit is not None else ""
    params: tuple[Any, ...] = (int(limit),) if limit is not None else ()
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        ensure_table(con, "product_calls")
        rows = con.execute(
            f"""
            SELECT
              pc.tenant_id,
              pc.telephony_provider,
              pc.provider_call_id,
              pc.event_key,
              pc.recording_id,
              pc.source_filename,
              pc.started_at,
              pc.duration_sec,
              pc.manager_extension,
              pc.manager_display_name,
              pc.crm_owner_id,
              pc.crm_owner_name,
              pc.crm_match_status,
              pc.raw_payload_ref,
              pc.source_repository_ref,
              NULL AS crm_entity_id,
              NULL AS crm_entity_type,
              NULL AS insight_summary,
              NULL AS recommended_next_step,
              NULL AS ai_priority,
              NULL AS requested_actions
            FROM product_calls pc
            ORDER BY pc.started_at DESC, pc.provider_call_id
            {limit_sql}
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def apply_crm_resolution(
    calls: Sequence[Mapping[str, Any]],
    resolver_report: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    by_event = mapping_or_empty(mapping_or_empty(resolver_report.get("resolution_index")).get("by_event_key"))
    by_call = mapping_or_empty(mapping_or_empty(resolver_report.get("resolution_index")).get("by_provider_call_id"))
    items_by_event = {
        clean(item.get("event_key")): item
        for item in resolver_report.get("items", [])
        if isinstance(item, Mapping) and clean(item.get("event_key"))
    }
    items_by_call = {
        clean(item.get("provider_call_id")): item
        for item in resolver_report.get("items", [])
        if isinstance(item, Mapping) and clean(item.get("provider_call_id"))
    }
    resolved = []
    for call in calls:
        event_key = clean(call.get("event_key"))
        provider_call_id = clean(call.get("provider_call_id"))
        resolution = by_event.get(event_key) or by_call.get(provider_call_id)
        any_resolution = resolution or items_by_event.get(event_key) or items_by_call.get(provider_call_id)
        if isinstance(resolution, Mapping) and clean(resolution.get("action")) == RESOLVE_CRM_ENTITY:
            resolved.append(
                dict(call)
                | {
                    "crm_entity_id": clean(resolution.get("crm_entity_id")),
                    "crm_entity_type": clean(resolution.get("crm_entity_type")),
                    "crm_resolution": compact_resolution_item(resolution),
                }
            )
        else:
            resolved.append(dict(call) | {"crm_resolution": compact_resolution_item(any_resolution)})
    return resolved


def compact_crm_resolution(resolver_report: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not resolver_report:
        return {
            "enabled": False,
            "reason": "crm_snapshot_not_provided",
        }
    return {
        "enabled": True,
        "summary": resolver_report.get("summary"),
        "action_counts": resolver_report.get("action_counts"),
        "snapshot_contract": resolver_report.get("snapshot_contract"),
    }


def compact_resolution_item(item: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(item, Mapping):
        return None
    return {
        "action": clean(item.get("action")),
        "reason": clean(item.get("reason")) or None,
        "call_phone": clean(item.get("call_phone")) or None,
        "candidate_count": optional_int(item.get("candidate_count")),
        "crm_entity_type": clean(item.get("crm_entity_type")) or None,
        "crm_entity_id": clean(item.get("crm_entity_id")) or None,
        "confidence": item.get("confidence"),
    }


def field_mapping_contract() -> Mapping[str, Any]:
    return {
        "crm_provider": "amocrm",
        "write_mode": "preview_only",
        "allowed_logical_fields": list(ALLOWED_LOGICAL_FIELDS),
        "protected_fields": ["Id Tallanto", "Филиал Tallanto"],
        "safe_mode": {
            "overwrite_non_empty_fields": False,
            "close_lead_automatically": False,
            "delete_or_merge_contacts": False,
            "send_client_messages": False,
        },
    }


def staged_queue_contract(stage: str, stage_limit: Optional[int], selected: int) -> Mapping[str, Any]:
    return {
        "current_stage": stage,
        "stage_limit": stage_limit,
        "selected_items": selected,
        "rollout_order": ["batch_10", "batch_50", "batch_300", "full"],
        "full_rollout_requires": ["human_approval", "latest_preview_audit", "rollback_export"],
    }


def policy_gates() -> Mapping[str, Any]:
    return {
        "live_write_enabled": False,
        "required_live_confirmation": "WRITE_AMO_LIVE",
        "requires_human_approval": True,
        "forbidden_actions": list(FORBIDDEN_ACTIONS),
        "l3_l4_policy": "approval_required_or_forbidden",
    }


def rollback_plan() -> Mapping[str, Any]:
    return {
        "before_live_batch": [
            "export_current_target_fields",
            "save_preview_report_json",
            "save_selected_entity_ids",
        ],
        "during_live_batch": ["write_audit_row_per_entity", "stop_on_first_unexpected_error"],
        "after_live_batch": ["compare_updated_fields", "manual_restore_from_export_if_needed"],
    }


def resolve_preview_paths(
    product_db_path: Path,
    product_root: Path,
    out_path: Optional[Path],
) -> tuple[Path, Path, Optional[Path]]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if out_path:
        if "stable_runtime" in out_path.parts:
            raise ValueError("writeback preview output must not be under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"writeback preview output must stay under product root: {product_root}")
    return product_db_path, product_root, out_path


def ensure_table(con: sqlite3.Connection, name: str) -> None:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (clean(name),),
    ).fetchone()
    if row is None:
        raise ValueError(f"required table not found: {name}")


def min_non_null(left: Optional[int], right: Optional[int]) -> Optional[int]:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def optional_int(value: Any) -> Optional[int]:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safety_contract() -> Mapping[str, bool]:
    return {
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
