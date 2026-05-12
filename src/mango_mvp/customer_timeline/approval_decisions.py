from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.approval_workspace import (
    CUSTOMER_TIMELINE_APPROVAL_WORKSPACE_SCHEMA_VERSION,
    CustomerTimelineApprovalWorkspaceConfig,
    build_customer_timeline_approval_workspace,
)
from mango_mvp.customer_timeline.ids import stable_prefixed_id
from mango_mvp.customer_timeline.read_api import customer_timeline_read_api_safety_contract
from mango_mvp.customer_timeline.safety import blocked_live_actions, guard_customer_timeline_output_path


CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION = "customer_timeline_approval_decisions_v1"
CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION = "customer_timeline_approval_decision_row_v1"
PENDING_DECISION = "pending"
FINAL_DECISIONS = ("approve", "reject", "needs_rework")
ALL_DECISIONS = (PENDING_DECISION, *FINAL_DECISIONS)
ALLOWED_REASON_CODES = (
    "reviewed_ok",
    "identity_conflict",
    "bot_context_issue",
    "missing_context",
    "data_quality_issue",
    "operator_rejected",
    "needs_rework",
    "other",
)
REQUIRED_ACKNOWLEDGEMENTS = (
    "reviewed_customer",
    "reviewed_timeline",
    "reviewed_bot_context",
    "reviewed_conflicts",
    "understands_no_live_writes",
)


@dataclass(frozen=True)
class CustomerTimelineApprovalDecisionConfig:
    allowed_root: Path
    workspace_json: Optional[Path] = None
    timeline_db: Optional[Path] = None
    out_template_jsonl: Optional[Path] = None
    out_report_json: Optional[Path] = None

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).resolve(strict=False)
        workspace_json = guard_decision_input_path(self.workspace_json, root) if self.workspace_json else None
        timeline_db = None
        if self.timeline_db:
            workspace_config = CustomerTimelineApprovalWorkspaceConfig(timeline_db=self.timeline_db, allowed_root=root)
            timeline_db = workspace_config.timeline_db
        out_template = guard_decision_output_path(self.out_template_jsonl, root, workspace_json) if self.out_template_jsonl else None
        out_report = guard_decision_output_path(self.out_report_json, root, workspace_json) if self.out_report_json else None
        if timeline_db and out_template == timeline_db:
            raise ValueError("approval decision output must not overwrite timeline DB")
        if timeline_db and out_report == timeline_db:
            raise ValueError("approval decision output must not overwrite timeline DB")
        if out_template and out_report and out_template == out_report:
            raise ValueError("approval decision outputs must be separate files")
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "workspace_json", workspace_json)
        object.__setattr__(self, "timeline_db", timeline_db)
        object.__setattr__(self, "out_template_jsonl", out_template)
        object.__setattr__(self, "out_report_json", out_report)


def build_customer_timeline_approval_decision_template(
    workspace: Mapping[str, Any],
    *,
    generated_at: Optional[datetime] = None,
    out_template_jsonl: Optional[Path] = None,
    out_report_json: Optional[Path] = None,
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    rows = build_decision_template_rows(workspace)
    report = {
        "schema_version": CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION,
        "artifact": "approval_decision_template",
        "generated_at": generated.isoformat(),
        "workspace_schema_version": workspace.get("schema_version"),
        "workspace_fingerprint": workspace_fingerprint(workspace),
        "tenant_id": workspace.get("tenant_id"),
        "customer_id": (workspace.get("summary") or {}).get("selected_customer_id"),
        "summary": {
            "validation_ok": True,
            "decision_rows": len(rows),
            "pending_rows": len(rows),
            "required_rows": len([row for row in rows if row.get("required")]),
            "live_actions_available": False,
            "status": "template_pending_operator_decisions",
        },
        "template_rows": rows,
        "reason_codes": list(ALLOWED_REASON_CODES),
        "safety": customer_timeline_approval_decisions_safety_contract(),
    }
    write_json_report(report, out_report_json)
    write_jsonl_rows(rows, out_template_jsonl)
    return report


def validate_customer_timeline_approval_decisions(
    workspace: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: Optional[datetime] = None,
    out_report_json: Optional[Path] = None,
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    expected_rows = build_decision_template_rows(workspace)
    expected_by_id = {str(row["decision_id"]): row for row in expected_rows}
    seen: set[str] = set()
    accepted_rows: list[Mapping[str, Any]] = []
    invalid_rows: list[Mapping[str, Any]] = []
    pending_rows = 0
    unknown_rows = 0
    duplicate_rows = 0
    approved = 0
    rejected = 0
    needs_rework = 0

    for index, row in enumerate(decision_rows, start=1):
        errors = validate_decision_row(row, expected_by_id, seen, index=index)
        decision_id = str(row.get("decision_id") or "")
        expected = expected_by_id.get(decision_id)
        decision = str(row.get("decision") or "").strip()
        if decision == PENDING_DECISION:
            pending_rows += 1
        if decision_id not in expected_by_id:
            unknown_rows += 1
        if decision_id in seen:
            duplicate_rows += 1
        if not errors and expected is not None:
            accepted = project_accepted_decision(row, expected)
            accepted_rows.append(accepted)
            approved += 1 if decision == "approve" else 0
            rejected += 1 if decision == "reject" else 0
            needs_rework += 1 if decision == "needs_rework" else 0
        else:
            invalid_rows.append(
                {
                    "line": index,
                    "decision_id": decision_id,
                    "source_action": row.get("source_action"),
                    "decision": row.get("decision"),
                    "errors": errors,
                }
            )
        if decision_id:
            seen.add(decision_id)

    missing_ids = sorted(set(expected_by_id) - seen)
    workflow_status = workflow_status_for_decisions(approved=approved, rejected=rejected, needs_rework=needs_rework)
    validation_ok = not invalid_rows and not pending_rows and not missing_ids and len(accepted_rows) == len(expected_rows)
    report = {
        "schema_version": CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION,
        "artifact": "approval_decision_validation",
        "generated_at": generated.isoformat(),
        "workspace_schema_version": workspace.get("schema_version"),
        "workspace_fingerprint": workspace_fingerprint(workspace),
        "tenant_id": workspace.get("tenant_id"),
        "customer_id": (workspace.get("summary") or {}).get("selected_customer_id"),
        "summary": {
            "validation_ok": validation_ok,
            "workflow_status": workflow_status,
            "expected_rows": len(expected_rows),
            "received_rows": len(decision_rows),
            "accepted_rows": len(accepted_rows),
            "invalid_rows": len(invalid_rows),
            "pending_rows": pending_rows,
            "missing_required_rows": len(missing_ids),
            "unknown_rows": unknown_rows,
            "duplicate_rows": duplicate_rows,
            "approved": approved,
            "rejected": rejected,
            "needs_rework": needs_rework,
            "live_actions_available": False,
            "ready_for_live": False,
        },
        "accepted_rows": accepted_rows,
        "invalid_rows": invalid_rows,
        "missing_required_decision_ids": missing_ids,
        "next_safe_step": next_safe_step(workflow_status),
        "safety": customer_timeline_approval_decisions_safety_contract(),
        "validation_ok": validation_ok,
    }
    write_json_report(report, out_report_json)
    return report


def run_customer_timeline_approval_decisions(
    *,
    config: CustomerTimelineApprovalDecisionConfig,
    mode: str,
    tenant_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 25,
    decisions_jsonl: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    workspace = load_workspace_for_decisions(
        config=config,
        tenant_id=tenant_id,
        customer_id=customer_id,
        query=query,
        limit=limit,
    )
    if mode == "template":
        return build_customer_timeline_approval_decision_template(
            workspace,
            generated_at=generated_at,
            out_template_jsonl=config.out_template_jsonl,
            out_report_json=config.out_report_json,
        )
    if mode == "validate":
        if decisions_jsonl is None:
            raise ValueError("validate mode requires decisions_jsonl")
        decisions_path = guard_decision_input_path(decisions_jsonl, config.allowed_root)
        if decisions_path == config.workspace_json:
            raise ValueError("decisions input must not be the same file as workspace JSON")
        if config.out_report_json and config.out_report_json == decisions_path:
            raise ValueError("validation report must not overwrite decisions JSONL")
        rows = load_decision_jsonl_rows(decisions_path)
        return validate_customer_timeline_approval_decisions(
            workspace,
            rows,
            generated_at=generated_at,
            out_report_json=config.out_report_json,
        )
    raise ValueError(f"unsupported approval decision mode: {mode}")


def build_decision_template_rows(workspace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    summary = workspace.get("summary") if isinstance(workspace.get("summary"), Mapping) else {}
    queue = workspace.get("review_queue") if isinstance(workspace.get("review_queue"), list) else []
    if not queue:
        queue = [
            {
                "action": "WORKSPACE_NOT_READY_REVIEW",
                "priority": "high",
                "label": "Approval workspace has no actionable customer selection",
                "live_write": False,
            }
        ]
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(queue, start=1):
        if not isinstance(item, Mapping):
            continue
        source_action = str(item.get("action") or "REVIEW_WORKSPACE").strip() or "REVIEW_WORKSPACE"
        priority = str(item.get("priority") or "normal").strip() or "normal"
        label = str(item.get("label") or source_action).strip() or source_action
        approval_allowed = approval_allowed_for_action(source_action, str(summary.get("status") or ""))
        allowed = list(FINAL_DECISIONS if approval_allowed else ("reject", "needs_rework"))
        queue_item = {
            "index": index,
            "action": source_action,
            "priority": priority,
            "label": label,
            "live_write": False,
        }
        decision_id = stable_prefixed_id(
            "approval_decision",
            {
                "tenant_id": workspace.get("tenant_id"),
                "customer_id": summary.get("selected_customer_id"),
                "workspace_status": summary.get("status"),
                "position": index,
                "source_action": source_action,
                "label": label,
            },
            length=24,
        )
        rows.append(
            {
                "schema_version": CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION,
                "record_type": "customer_timeline_approval_decision",
                "decision_id": decision_id,
                "tenant_id": workspace.get("tenant_id"),
                "customer_id": summary.get("selected_customer_id"),
                "selected_customer_found": bool(summary.get("selected_customer_found")),
                "workspace_fingerprint": workspace_fingerprint(workspace),
                "workspace_ref": workspace_ref(workspace),
                "workspace_summary_snapshot": workspace_summary_snapshot(summary),
                "workspace_status": summary.get("status"),
                "source_action": source_action,
                "priority": priority,
                "label": label,
                "target_type": "review_queue_item",
                "target_id": stable_prefixed_id(
                    "review_queue_item",
                    {"decision_id": decision_id, "source_action": source_action, "position": index},
                    length=24,
                ),
                "queue_item": queue_item,
                "required": True,
                "approval_allowed": approval_allowed,
                "decision": PENDING_DECISION,
                "allowed_decisions": allowed,
                "allowed_reason_codes": list(ALLOWED_REASON_CODES),
                "reviewer": "",
                "decided_by": "",
                "reason": "",
                "reason_codes": [],
                "comment": "",
                "decided_at": None,
                "acknowledgements": {key: False for key in REQUIRED_ACKNOWLEDGEMENTS},
                "rework_items": [],
                "supersedes_decision_id": None,
                "live_write": False,
                "safety": customer_timeline_approval_decisions_safety_contract(),
                "metadata": {
                    "workspace_schema_version": workspace.get("schema_version"),
                    "workspace_generated_at": workspace.get("generated_at"),
                    "source_live_write": bool(item.get("live_write")),
                },
            }
        )
    return rows


def validate_decision_row(
    row: Mapping[str, Any],
    expected_by_id: Mapping[str, Mapping[str, Any]],
    seen: set[str],
    *,
    index: int,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, Mapping):
        return [f"line_{index}_is_not_object"]
    if row.get("schema_version") != CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if row.get("record_type") != "customer_timeline_approval_decision":
        errors.append("record_type_mismatch")
    decision_id = str(row.get("decision_id") or "").strip()
    if not decision_id:
        errors.append("decision_id_required")
        return errors
    if decision_id in seen:
        errors.append("duplicate_decision_id")
    expected = expected_by_id.get(decision_id)
    if expected is None:
        errors.append("unknown_decision_id")
        return errors
    if row.get("workspace_fingerprint") != expected.get("workspace_fingerprint"):
        errors.append("workspace_fingerprint_mismatch")
    if row.get("workspace_ref") != expected.get("workspace_ref"):
        errors.append("workspace_ref_mismatch")
    if row.get("workspace_summary_snapshot") != expected.get("workspace_summary_snapshot"):
        errors.append("workspace_summary_snapshot_mismatch")
    if row.get("queue_item") != expected.get("queue_item"):
        errors.append("queue_item_mismatch")
    if row.get("source_action") != expected.get("source_action"):
        errors.append("source_action_mismatch")
    if row.get("tenant_id") != expected.get("tenant_id"):
        errors.append("tenant_id_mismatch")
    if row.get("customer_id") != expected.get("customer_id"):
        errors.append("customer_id_mismatch")
    if row.get("live_write") is not False:
        errors.append("live_write_must_be_false")
    errors.extend(validate_safety_snapshot(row.get("safety")))
    decision = str(row.get("decision") or "").strip()
    if decision not in ALL_DECISIONS:
        errors.append("decision_not_allowed")
    if decision == PENDING_DECISION:
        errors.append("decision_still_pending")
    if decision in FINAL_DECISIONS:
        if decision not in expected.get("allowed_decisions", ()):
            errors.append("decision_not_allowed_for_workspace_state")
        reviewer = str(row.get("reviewer") or "").strip()
        decided_by = str(row.get("decided_by") or "").strip()
        if reviewer and decided_by and reviewer != decided_by:
            errors.append("reviewer_decided_by_mismatch")
        if not (reviewer or decided_by):
            errors.append("reviewer_required")
        if not str(row.get("reason") or "").strip():
            errors.append("reason_required")
        reason_codes = row.get("reason_codes")
        if not isinstance(reason_codes, list) or not reason_codes:
            errors.append("reason_codes_required")
        elif any(code not in ALLOWED_REASON_CODES for code in reason_codes):
            errors.append("reason_code_not_allowed")
        acknowledgements = row.get("acknowledgements")
        if not isinstance(acknowledgements, Mapping):
            errors.append("acknowledgements_required")
        elif acknowledgements.get("understands_no_live_writes") is not True:
            errors.append("understands_no_live_writes_ack_required")
        elif decision == "approve" and any(acknowledgements.get(key) is not True for key in REQUIRED_ACKNOWLEDGEMENTS):
            errors.append("all_approval_acknowledgements_required")
        rework_items = row.get("rework_items")
        if decision == "needs_rework" and (not isinstance(rework_items, list) or not rework_items):
            errors.append("rework_items_required")
        if decision == "approve" and rework_items:
            errors.append("approve_rework_items_must_be_empty")
        decided_at = row.get("decided_at")
        if not str(decided_at or "").strip():
            errors.append("decided_at_required")
        else:
            try:
                parsed = datetime.fromisoformat(str(decided_at).replace("Z", "+00:00"))
            except ValueError:
                errors.append("decided_at_invalid_iso")
            else:
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    errors.append("decided_at_must_be_timezone_aware")
    return errors


def project_accepted_decision(row: Mapping[str, Any], expected: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "decision_id": row.get("decision_id"),
        "tenant_id": expected.get("tenant_id"),
        "customer_id": expected.get("customer_id"),
        "source_action": expected.get("source_action"),
        "priority": expected.get("priority"),
        "decision": row.get("decision"),
        "reviewer": str(row.get("reviewer") or row.get("decided_by") or "").strip(),
        "reason": str(row.get("reason") or "").strip(),
        "reason_codes": row.get("reason_codes") if isinstance(row.get("reason_codes"), list) else [],
        "comment": str(row.get("comment") or "").strip(),
        "rework_items": row.get("rework_items") if isinstance(row.get("rework_items"), list) else [],
        "decided_at": row.get("decided_at"),
        "live_write": False,
    }


def load_workspace_for_decisions(
    *,
    config: CustomerTimelineApprovalDecisionConfig,
    tenant_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 25,
) -> Mapping[str, Any]:
    if config.workspace_json:
        workspace = load_workspace_json(config.workspace_json)
    else:
        if not config.timeline_db:
            raise ValueError("workspace_json or timeline_db is required")
        if not tenant_id:
            raise ValueError("tenant_id is required when building workspace from timeline_db")
        workspace = build_customer_timeline_approval_workspace(
            config=CustomerTimelineApprovalWorkspaceConfig(timeline_db=config.timeline_db, allowed_root=config.allowed_root),
            tenant_id=tenant_id,
            customer_id=customer_id,
            query=query,
            limit=limit,
        )
    if workspace.get("schema_version") != CUSTOMER_TIMELINE_APPROVAL_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("workspace JSON schema_version is not customer_timeline_approval_workspace_v1")
    return workspace


def load_workspace_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("workspace JSON root must be an object")
    return {
        **payload,
        "_approval_workspace_source": {
            "path": str(path),
            "sha256": file_sha256(path),
        },
    }


def load_decision_jsonl_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"decision JSONL line {lineno} must be an object")
        rows.append(payload)
    return rows


def write_jsonl_rows(rows: Sequence[Mapping[str, Any]], path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def write_json_report(report: Mapping[str, Any], path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def guard_decision_input_path(path: Optional[Path], allowed_root: Path) -> Optional[Path]:
    if path is None:
        return None
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    if not resolved.exists():
        raise ValueError(f"approval decision input does not exist: {resolved}")
    if resolved.is_dir():
        raise ValueError(f"approval decision input must be a file: {resolved}")
    return resolved


def guard_decision_output_path(path: Optional[Path], allowed_root: Path, workspace_json: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    if workspace_json and resolved == workspace_json:
        raise ValueError("approval decision output must not overwrite workspace JSON")
    return resolved


def workspace_fingerprint(workspace: Mapping[str, Any]) -> str:
    summary = workspace.get("summary") if isinstance(workspace.get("summary"), Mapping) else {}
    queue = workspace.get("review_queue") if isinstance(workspace.get("review_queue"), list) else []
    payload = {
        "schema_version": workspace.get("schema_version"),
        "tenant_id": workspace.get("tenant_id"),
        "selected_customer_id": summary.get("selected_customer_id"),
        "status": summary.get("status"),
        "open_conflicts": summary.get("open_conflicts"),
        "bot_allowed_chunks": summary.get("bot_allowed_chunks"),
        "bot_review_required_chunks": summary.get("bot_review_required_chunks"),
        "review_queue": [
            {
                "action": item.get("action"),
                "priority": item.get("priority"),
                "label": item.get("label"),
                "live_write": bool(item.get("live_write")),
            }
            for item in queue
            if isinstance(item, Mapping)
        ],
    }
    return stable_prefixed_id("approval_workspace", payload, length=24)


def workspace_ref(workspace: Mapping[str, Any]) -> Mapping[str, Any]:
    source = workspace.get("_approval_workspace_source") if isinstance(workspace.get("_approval_workspace_source"), Mapping) else {}
    return {
        "path": source.get("path"),
        "sha256": source.get("sha256"),
        "schema_version": workspace.get("schema_version"),
        "generated_at": workspace.get("generated_at"),
        "read_api_schema_version": workspace.get("read_api_schema_version"),
    }


def workspace_summary_snapshot(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "validation_ok": bool(summary.get("validation_ok")),
        "status": summary.get("status"),
        "selected_customer_found": bool(summary.get("selected_customer_found")),
        "open_conflicts": int(summary.get("open_conflicts") or 0),
        "bot_allowed_chunks": int(summary.get("bot_allowed_chunks") or 0),
        "bot_review_required_chunks": int(summary.get("bot_review_required_chunks") or 0),
        "warnings": int(summary.get("warnings") or 0),
        "blocked": int(summary.get("blocked") or 0),
        "live_actions_available": bool(summary.get("live_actions_available")),
    }


def validate_safety_snapshot(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["safety_required"]
    errors: list[str] = []
    for action in blocked_live_actions():
        if value.get(action) is not False:
            errors.append(f"safety_{action}_must_be_false")
    for action in ("network_calls", "subprocess_calls", "write_product_timeline_db"):
        if value.get(action) is not False:
            errors.append(f"safety_{action}_must_be_false")
    return errors


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def approval_allowed_for_action(source_action: str, workspace_status: str) -> bool:
    if "blocked" in workspace_status or workspace_status == "needs_context":
        return False
    if source_action == "REVIEW_IDENTITY_CONFLICT":
        return False
    if source_action == "WORKSPACE_NOT_READY_REVIEW":
        return False
    return True


def workflow_status_for_decisions(*, approved: int, rejected: int, needs_rework: int) -> str:
    if needs_rework:
        return "needs_rework"
    if rejected:
        return "rejected"
    if approved:
        return "approved_for_next_dry_run"
    return "no_final_decisions"


def next_safe_step(workflow_status: str) -> str:
    if workflow_status == "approved_for_next_dry_run":
        return "prepare_read_only_dry_run_pack"
    if workflow_status == "rejected":
        return "stop_and_review_operator_reason"
    if workflow_status == "needs_rework":
        return "fix_workspace_blockers_and_regenerate_template"
    return "collect_operator_decisions"


def customer_timeline_approval_decisions_safety_contract() -> Mapping[str, Any]:
    base = dict(customer_timeline_read_api_safety_contract())
    base.update(
        {
            "schema_version": CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION,
            "decision_artifact_only": True,
            "write_decision_artifacts": True,
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
        }
    )
    return base


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = CustomerTimelineApprovalDecisionConfig(
            allowed_root=Path(args.allowed_root),
            workspace_json=Path(args.workspace_json) if args.workspace_json else None,
            timeline_db=Path(args.timeline_db) if args.timeline_db else None,
            out_template_jsonl=Path(args.out_template_jsonl) if getattr(args, "out_template_jsonl", None) else None,
            out_report_json=Path(args.out_report_json) if args.out_report_json else None,
        )
        report = run_customer_timeline_approval_decisions(
            config=config,
            mode=args.command,
            tenant_id=args.tenant_id,
            customer_id=args.customer_id,
            query=args.query,
            limit=args.limit,
            decisions_jsonl=Path(args.decisions_jsonl) if getattr(args, "decisions_jsonl", None) else None,
        )
        if not args.out_report_json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.get("validation_ok") or args.command == "template" else 1
    except Exception as exc:  # noqa: BLE001 - CLI-facing compact error.
        print(f"customer timeline approval decisions failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate read-only Customer Timeline approval decisions.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template", help="Create an operator decision JSONL template.")
    add_common_args(template)
    template.add_argument("--out-template-jsonl", required=True)
    validate = subparsers.add_parser("validate", help="Validate an operator decision JSONL artifact.")
    add_common_args(validate)
    validate.add_argument("--decisions-jsonl", required=True)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--workspace-json")
    parser.add_argument("--timeline-db")
    parser.add_argument("--tenant-id")
    parser.add_argument("--customer-id")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out-report-json")


__all__ = [
    "ALL_DECISIONS",
    "CUSTOMER_TIMELINE_APPROVAL_DECISIONS_SCHEMA_VERSION",
    "CUSTOMER_TIMELINE_APPROVAL_DECISION_ROW_SCHEMA_VERSION",
    "CustomerTimelineApprovalDecisionConfig",
    "FINAL_DECISIONS",
    "PENDING_DECISION",
    "ALLOWED_REASON_CODES",
    "REQUIRED_ACKNOWLEDGEMENTS",
    "build_customer_timeline_approval_decision_template",
    "build_decision_template_rows",
    "customer_timeline_approval_decisions_safety_contract",
    "load_decision_jsonl_rows",
    "main",
    "run_customer_timeline_approval_decisions",
    "validate_customer_timeline_approval_decisions",
    "workspace_fingerprint",
]
