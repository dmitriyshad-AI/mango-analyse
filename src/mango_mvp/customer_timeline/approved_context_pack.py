from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.approval_decisions import (
    CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION,
    load_decision_jsonl_rows,
    load_workspace_json,
    validate_customer_timeline_approval_decisions,
    workspace_fingerprint,
)
from mango_mvp.customer_timeline.approval_workspace import CUSTOMER_TIMELINE_APPROVAL_WORKSPACE_SCHEMA_VERSION
from mango_mvp.customer_timeline.ids import stable_prefixed_id
from mango_mvp.customer_timeline.read_api import (
    CustomerTimelineReadApi,
    CustomerTimelineReadApiConfig,
    customer_timeline_read_api_safety_contract,
)
from mango_mvp.customer_timeline.safety import blocked_live_actions, guard_customer_timeline_output_path


CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION = "customer_timeline_approved_context_pack_v1"
APPROVED_CONTEXT_PACK_STATUS = "approved_read_only_context_pack"
BLOCKED_CONTEXT_PACK_STATUS = "blocked"
FORBIDDEN_PACK_MARKERS = (
    "raw_payload",
    "provider_raw_payload",
    "record_json",
    "audio_path",
    "transcript_path",
    "/not/read/",
    "/secret/",
)


@dataclass(frozen=True)
class CustomerTimelineApprovedContextPackConfig:
    timeline_db: Path
    allowed_root: Path
    workspace_json: Path
    decisions_jsonl: Path
    approval_report_json: Optional[Path] = None
    out_pack_json: Optional[Path] = None

    def __post_init__(self) -> None:
        read_config = CustomerTimelineReadApiConfig(timeline_db=self.timeline_db, allowed_root=self.allowed_root)
        workspace_json = guard_context_pack_input_path(self.workspace_json, read_config.allowed_root)
        decisions_jsonl = guard_context_pack_input_path(self.decisions_jsonl, read_config.allowed_root)
        approval_report_json = guard_context_pack_input_path(self.approval_report_json, read_config.allowed_root) if self.approval_report_json else None
        out_pack_json = guard_context_pack_output_path(self.out_pack_json, read_config, workspace_json, decisions_jsonl, approval_report_json)
        object.__setattr__(self, "timeline_db", read_config.timeline_db)
        object.__setattr__(self, "allowed_root", read_config.allowed_root)
        object.__setattr__(self, "workspace_json", workspace_json)
        object.__setattr__(self, "decisions_jsonl", decisions_jsonl)
        object.__setattr__(self, "approval_report_json", approval_report_json)
        object.__setattr__(self, "out_pack_json", out_pack_json)


def build_customer_timeline_approved_context_pack(
    *,
    config: CustomerTimelineApprovedContextPackConfig,
    limit: int = 25,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    workspace = load_workspace_json(config.workspace_json)
    decision_rows = load_decision_jsonl_rows(config.decisions_jsonl)
    approval_report = validate_customer_timeline_approval_decisions(workspace, decision_rows, generated_at=generated)
    cached_approval_report = load_json_object(config.approval_report_json, "approval report JSON") if config.approval_report_json else None
    assert_known_input_schemas(workspace, approval_report, cached_approval_report)
    tenant_id = str(workspace.get("tenant_id") or approval_report.get("tenant_id") or "").strip()
    summary = workspace.get("summary") if isinstance(workspace.get("summary"), Mapping) else {}
    customer_id = str(summary.get("selected_customer_id") or approval_report.get("customer_id") or "").strip()
    workspace_hash = file_sha256(config.workspace_json)
    decisions_hash = file_sha256(config.decisions_jsonl)
    cached_approval_report_hash = file_sha256(config.approval_report_json) if config.approval_report_json else None
    blockers = approval_context_pack_blockers(workspace, approval_report)
    blockers.extend(cached_report_blockers(approval_report, cached_approval_report))
    bot_context: Mapping[str, Any] = empty_bot_context(tenant_id, customer_id)
    customer: Mapping[str, Any] = {}
    current_conflicts: Mapping[str, Any] = empty_conflicts(tenant_id, customer_id)
    health: Mapping[str, Any] = {}

    if not tenant_id:
        blockers.append("tenant_id_required")
    if not customer_id:
        blockers.append("customer_id_required")

    if not blockers:
        with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=config.timeline_db, allowed_root=config.allowed_root)) as api:
            health = api.health()
            if not health.get("validation_ok"):
                blockers.append("timeline_db_health_not_valid")
            profile = api.customer_profile(tenant_id, customer_id, event_limit=1, bot_context_limit=limit, include_children=False)
            if not profile.get("found"):
                blockers.append("customer_not_found_in_current_db")
            else:
                customer = safe_customer_for_context(profile.get("customer") if isinstance(profile.get("customer"), Mapping) else {})
            current_conflicts = api.list_conflicts(tenant_id, customer_id=customer_id, status="open", limit=limit)
            if int((current_conflicts.get("summary") or {}).get("open_conflicts") or 0):
                blockers.append("current_db_open_conflicts")
            bot_context = api.bot_context(tenant_id, customer_id, allowed_only=True, limit=limit)
            if int((bot_context.get("summary") or {}).get("visible_chunks") or 0) == 0:
                blockers.append("no_bot_safe_context")
            if not bot_context_items_are_safe(bot_context.get("items") or ()):
                blockers.append("bot_context_items_not_safe")

    chunks = [] if blockers else [project_context_chunk(item) for item in bot_context.get("items") or ()]
    pack_id = stable_context_pack_id(
        tenant_id=tenant_id,
        customer_id=customer_id,
        workspace_fingerprint_value=workspace_fingerprint(workspace),
        decisions_hash=decisions_hash,
        chunks=chunks,
    )
    status = APPROVED_CONTEXT_PACK_STATUS if not blockers else BLOCKED_CONTEXT_PACK_STATUS
    pack = {
        "schema_version": CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION,
        "artifact": "customer_timeline_approved_context_pack",
        "generated_at": generated.isoformat(),
        "validation_ok": not blockers,
        "pack_id": pack_id,
        "status": status,
        "tenant_id": tenant_id,
        "customer_id": customer_id,
        "summary": {
            "validation_ok": not blockers,
            "status": status,
            "blocked_reasons": blockers,
            "context_chunks": len(chunks),
            "bot_context_visible_chunks": int((bot_context.get("summary") or {}).get("visible_chunks") or 0),
            "bot_context_total_chunks": int((bot_context.get("summary") or {}).get("total_chunks") or 0),
            "bot_context_review_required_chunks": int((bot_context.get("summary") or {}).get("review_required_chunks") or 0),
            "current_open_conflicts": int((current_conflicts.get("summary") or {}).get("open_conflicts") or 0),
            "live_actions_available": False,
        },
        "source_refs": {
            "workspace_sha256": workspace_hash,
            "workspace_schema_version": workspace.get("schema_version"),
            "workspace_fingerprint": workspace_fingerprint(workspace),
            "workspace_generated_at": workspace.get("generated_at"),
            "decisions_jsonl_sha256": decisions_hash,
            "approval_report_sha256": cached_approval_report_hash,
            "approval_report_schema_version": cached_approval_report.get("schema_version") if cached_approval_report else None,
            "approval_report_generated_at": cached_approval_report.get("generated_at") if cached_approval_report else None,
            "self_validation_generated_at": approval_report.get("generated_at"),
        },
        "approval": project_approval_summary(approval_report),
        "customer": customer if not blockers else {},
        "approved_context": {
            "scope": "bot_safe_customer_context",
            "items": chunks,
            "summary": bot_context.get("summary") if not blockers else empty_bot_context_summary(),
        },
        "channel_context": {
            "normalized_customer_id": customer_id if not blockers else None,
            "approved_context_pack_id": pack_id if not blockers else None,
            "safe_context_summary": summarize_chunks_for_channel(chunks) if not blockers else "",
            "can_build_draft": not blockers,
            "can_send": False,
            "requires_manager_approval_before_send": True,
        },
        "current_read_api_health": {
            "status": health.get("status"),
            "validation_ok": bool(health.get("validation_ok")) if health else None,
            "read_only": bool(health.get("read_only")) if health else None,
        },
        "safety": customer_timeline_approved_context_pack_safety_contract(),
    }
    leaked_markers = forbidden_pack_markers(pack)
    if leaked_markers:
        pack["validation_ok"] = False
        pack["status"] = BLOCKED_CONTEXT_PACK_STATUS
        pack["summary"]["validation_ok"] = False
        pack["summary"]["status"] = BLOCKED_CONTEXT_PACK_STATUS
        pack["summary"]["blocked_reasons"] = list(pack["summary"]["blocked_reasons"]) + [
            f"forbidden_marker:{marker}" for marker in leaked_markers
        ]
        pack["approved_context"]["items"] = []
        pack["approved_context"]["summary"] = empty_bot_context_summary()
        pack["channel_context"]["normalized_customer_id"] = None
        pack["channel_context"]["approved_context_pack_id"] = None
        pack["channel_context"]["safe_context_summary"] = ""
        pack["channel_context"]["can_build_draft"] = False
    if config.out_pack_json:
        config.out_pack_json.parent.mkdir(parents=True, exist_ok=True)
        config.out_pack_json.write_text(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pack


def approval_context_pack_blockers(workspace: Mapping[str, Any], approval_report: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    workspace_summary = workspace.get("summary") if isinstance(workspace.get("summary"), Mapping) else {}
    approval_summary = approval_report.get("summary") if isinstance(approval_report.get("summary"), Mapping) else {}
    if approval_report.get("workspace_fingerprint") != workspace_fingerprint(workspace):
        blockers.append("workspace_fingerprint_mismatch")
    if approval_report.get("tenant_id") != workspace.get("tenant_id"):
        blockers.append("tenant_id_mismatch")
    if approval_report.get("customer_id") != workspace_summary.get("selected_customer_id"):
        blockers.append("customer_id_mismatch")
    if workspace_summary.get("validation_ok") is not True:
        blockers.append("workspace_not_valid")
    if workspace_summary.get("status") != "ready_for_review":
        blockers.append("workspace_not_ready_for_review")
    if workspace_summary.get("selected_customer_found") is not True:
        blockers.append("workspace_customer_not_found")
    if int(workspace_summary.get("open_conflicts") or 0):
        blockers.append("workspace_open_conflicts")
    if int(workspace_summary.get("bot_allowed_chunks") or 0) <= 0:
        blockers.append("workspace_no_bot_allowed_context")
    if approval_report.get("validation_ok") is not True:
        blockers.append("approval_report_not_valid")
    if approval_summary.get("validation_ok") is not True:
        blockers.append("approval_summary_not_valid")
    if approval_summary.get("workflow_status") != "approved_for_next_dry_run":
        blockers.append(f"approval_workflow_not_approved:{approval_summary.get('workflow_status') or 'unknown'}")
    if int(approval_summary.get("approved") or 0) <= 0:
        blockers.append("approval_has_no_approved_rows")
    if int(approval_summary.get("rejected") or 0):
        blockers.append("approval_contains_rejected_rows")
    if int(approval_summary.get("needs_rework") or 0):
        blockers.append("approval_contains_needs_rework_rows")
    if int(approval_summary.get("invalid_rows") or 0):
        blockers.append("approval_contains_invalid_rows")
    if int(approval_summary.get("pending_rows") or 0):
        blockers.append("approval_contains_pending_rows")
    if approval_summary.get("ready_for_live") is not False:
        blockers.append("approval_report_must_not_enable_live")
    if approval_report.get("next_safe_step") != "prepare_read_only_dry_run_pack":
        blockers.append("approval_next_step_not_context_pack")
    blockers.extend(validate_pack_safety_snapshot(approval_report.get("safety")))
    return stable_unique(blockers)


def cached_report_blockers(
    self_validated_report: Mapping[str, Any],
    cached_report: Optional[Mapping[str, Any]],
) -> list[str]:
    if cached_report is None:
        return []
    blockers: list[str] = []
    cached_summary = cached_report.get("summary") if isinstance(cached_report.get("summary"), Mapping) else {}
    self_summary = self_validated_report.get("summary") if isinstance(self_validated_report.get("summary"), Mapping) else {}
    if cached_report.get("workspace_fingerprint") != self_validated_report.get("workspace_fingerprint"):
        blockers.append("cached_approval_report_workspace_fingerprint_mismatch")
    if cached_report.get("tenant_id") != self_validated_report.get("tenant_id"):
        blockers.append("cached_approval_report_tenant_id_mismatch")
    if cached_report.get("customer_id") != self_validated_report.get("customer_id"):
        blockers.append("cached_approval_report_customer_id_mismatch")
    for key in ("validation_ok", "workflow_status", "approved", "rejected", "needs_rework", "invalid_rows", "pending_rows"):
        if cached_summary.get(key) != self_summary.get(key):
            blockers.append(f"cached_approval_report_summary_{key}_mismatch")
    return blockers


def project_approval_summary(approval_report: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = approval_report.get("summary") if isinstance(approval_report.get("summary"), Mapping) else {}
    accepted_rows = approval_report.get("accepted_rows") if isinstance(approval_report.get("accepted_rows"), list) else []
    return {
        "workflow_status": summary.get("workflow_status"),
        "next_safe_step": approval_report.get("next_safe_step"),
        "approved": int(summary.get("approved") or 0),
        "rejected": int(summary.get("rejected") or 0),
        "needs_rework": int(summary.get("needs_rework") or 0),
        "accepted_rows": len(accepted_rows),
        "approved_decision_ids": [
            row.get("decision_id")
            for row in accepted_rows
            if isinstance(row, Mapping) and row.get("decision") == "approve"
        ],
        "reviewers": sorted(
            {
                str(row.get("reviewer") or "").strip()
                for row in accepted_rows
                if isinstance(row, Mapping) and str(row.get("reviewer") or "").strip()
            }
        ),
        "decided_at": [
            row.get("decided_at")
            for row in accepted_rows
            if isinstance(row, Mapping) and row.get("decision") == "approve"
        ],
        "live_actions_available": False,
    }


def project_context_chunk(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "chunk_id": item.get("chunk_id"),
        "source_system": item.get("source_system"),
        "chunk_type": item.get("chunk_type"),
        "summary": item.get("summary"),
        "text": item.get("text"),
        "event_at": item.get("event_at"),
        "freshness_score": item.get("freshness_score"),
        "relevance_tags": list(item.get("relevance_tags") or ()),
        "allowed_for_bot": bool(item.get("allowed_for_bot")),
        "requires_manager_review": bool(item.get("requires_manager_review")),
    }


def safe_customer_for_context(customer: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "customer_id": customer.get("customer_id"),
        "identity_status": customer.get("identity_status"),
        "display_name": customer.get("display_name"),
        "primary_phone": customer.get("primary_phone"),
        "primary_email": customer.get("primary_email"),
        "touch_count": customer.get("touch_count"),
        "last_seen_at": customer.get("last_seen_at"),
    }


def summarize_chunks_for_channel(chunks: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for item in chunks[:5]:
        text = str(item.get("summary") or item.get("text") or "").strip()
        if text:
            parts.append(text)
    return " | ".join(parts)[:1200]


def bot_context_items_are_safe(items: Sequence[Any]) -> bool:
    for item in items:
        if not isinstance(item, Mapping):
            return False
        if item.get("allowed_for_bot") is not True:
            return False
        if item.get("requires_manager_review") is not False:
            return False
        if item.get("customer_id") is not None or item.get("opportunity_id") is not None or item.get("event_id") is not None:
            return False
    return True


def forbidden_pack_markers(pack: Mapping[str, Any]) -> list[str]:
    text = json.dumps(pack, ensure_ascii=False, sort_keys=True).casefold()
    return [marker for marker in FORBIDDEN_PACK_MARKERS if marker.casefold() in text]


def validate_pack_safety_snapshot(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["approval_safety_required"]
    blockers: list[str] = []
    for action in blocked_live_actions():
        if value.get(action) is not False:
            blockers.append(f"approval_safety_{action}_must_be_false")
    for action in ("network_calls", "subprocess_calls", "write_product_timeline_db"):
        if value.get(action) is not False:
            blockers.append(f"approval_safety_{action}_must_be_false")
    return blockers


def customer_timeline_approved_context_pack_safety_contract() -> Mapping[str, Any]:
    base = dict(customer_timeline_read_api_safety_contract())
    base.update(
        {
            "schema_version": CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION,
            "read_only_context_for_channels": True,
            "approved_context_pack_only": True,
            "write_approved_context_pack_artifact": True,
            "write_product_timeline_db": False,
            "write_crm": False,
            "write_tallanto": False,
            "send_email": False,
            "send_messenger": False,
            "live_send": False,
            "run_asr": False,
            "run_ra": False,
            "write_runtime_db": False,
            "stable_runtime_writes": False,
            "network_calls": False,
            "subprocess_calls": False,
            "llm_calls": False,
            "rag_used": False,
            "local_artifact_paths_exposed": False,
        }
    )
    return base


def assert_known_input_schemas(
    workspace: Mapping[str, Any],
    approval_report: Mapping[str, Any],
    cached_approval_report: Optional[Mapping[str, Any]],
) -> None:
    if workspace.get("schema_version") != CUSTOMER_TIMELINE_APPROVAL_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("workspace JSON schema_version is not customer_timeline_approval_workspace_v1")
    if approval_report.get("schema_version") != CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION:
        raise ValueError("approval report JSON schema_version is not customer_timeline_approval_decisions_v1")
    if approval_report.get("artifact") != "approval_decision_validation":
        raise ValueError("approval report JSON must be an approval_decision_validation artifact")
    if cached_approval_report is not None:
        if cached_approval_report.get("schema_version") != CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION:
            raise ValueError("cached approval report JSON schema_version is not customer_timeline_approval_decisions_v1")
        if cached_approval_report.get("artifact") != "approval_decision_validation":
            raise ValueError("cached approval report JSON must be an approval_decision_validation artifact")


def load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be an object")
    return payload


def stable_context_pack_id(
    *,
    tenant_id: str,
    customer_id: str,
    workspace_fingerprint_value: str,
    decisions_hash: str,
    chunks: Sequence[Mapping[str, Any]],
) -> str:
    return stable_prefixed_id(
        "approved_context_pack",
        {
            "tenant_id": tenant_id,
            "customer_id": customer_id,
            "workspace_fingerprint": workspace_fingerprint_value,
            "decisions_hash": decisions_hash,
            "chunk_ids": [item.get("chunk_id") for item in chunks],
        },
        length=24,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_unique(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def empty_bot_context(tenant_id: str, customer_id: str) -> Mapping[str, Any]:
    return {
        "schema_version": "customer_timeline_read_api_v1",
        "endpoint": "GET /customer/bot-context",
        "tenant_id": tenant_id,
        "customer_id": customer_id,
        "allowed_only": True,
        "items": [],
        "summary": empty_bot_context_summary(),
    }


def empty_bot_context_summary() -> Mapping[str, int]:
    return {
        "visible_chunks": 0,
        "total_chunks": 0,
        "allowed_chunks": 0,
        "review_required_chunks": 0,
        "blocked_chunks": 0,
    }


def empty_conflicts(tenant_id: str, customer_id: str) -> Mapping[str, Any]:
    return {
        "tenant_id": tenant_id,
        "customer_id": customer_id,
        "items": [],
        "summary": {"total": 0, "open_conflicts": 0, "by_type": {}, "by_status": {}},
    }


def guard_context_pack_input_path(path: Path | str, allowed_root: Path) -> Path:
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    if not resolved.exists():
        raise ValueError(f"approved context pack input does not exist: {resolved}")
    if resolved.is_dir():
        raise ValueError(f"approved context pack input must be a file: {resolved}")
    return resolved


def guard_context_pack_output_path(
    path: Optional[Path],
    read_config: CustomerTimelineReadApiConfig,
    workspace_json: Path,
    decisions_jsonl: Path,
    approval_report_json: Optional[Path],
) -> Optional[Path]:
    if path is None:
        return None
    resolved = guard_customer_timeline_output_path(path, read_config.allowed_root)
    if resolved == read_config.timeline_db:
        raise ValueError("approved context pack output must not overwrite timeline DB")
    if resolved == workspace_json:
        raise ValueError("approved context pack output must not overwrite workspace JSON")
    if resolved == decisions_jsonl:
        raise ValueError("approved context pack output must not overwrite decisions JSONL")
    if approval_report_json and resolved == approval_report_json:
        raise ValueError("approved context pack output must not overwrite approval report JSON")
    return resolved


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = CustomerTimelineApprovedContextPackConfig(
            timeline_db=Path(args.timeline_db),
            allowed_root=Path(args.allowed_root),
            workspace_json=Path(args.workspace_json),
            decisions_jsonl=Path(args.decisions_jsonl),
            approval_report_json=Path(args.approval_report_json) if args.approval_report_json else None,
            out_pack_json=Path(args.out_pack_json) if args.out_pack_json else None,
        )
        pack = build_customer_timeline_approved_context_pack(config=config, limit=args.limit)
        if not args.out_pack_json:
            print(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if pack["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - CLI-facing compact error.
        print(f"customer timeline approved context pack failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only approved bot/context pack for Customer Timeline.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--workspace-json", required=True)
    parser.add_argument("--decisions-jsonl", required=True)
    parser.add_argument("--approval-report-json")
    parser.add_argument("--out-pack-json")
    parser.add_argument("--limit", type=int, default=25)
    return parser


__all__ = [
    "APPROVED_CONTEXT_PACK_STATUS",
    "BLOCKED_CONTEXT_PACK_STATUS",
    "CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION",
    "CustomerTimelineApprovedContextPackConfig",
    "build_customer_timeline_approved_context_pack",
    "customer_timeline_approved_context_pack_safety_contract",
    "main",
]
