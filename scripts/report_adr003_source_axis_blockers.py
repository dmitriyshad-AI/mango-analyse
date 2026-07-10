#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_current_handoff_fact_gap import _sha256, _source_rev


SCHEMA_VERSION = "adr003_source_axis_blockers_v1_2026_07_02"
HANDOFF_ROUTES = {"draft_for_manager", "manager_only"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize ADR-003 current handoff source/proof-axis blockers.")
    parser.add_argument("--queue-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(queue_report=args.queue_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_source_axis_blockers_report.json"
    md_path = args.out_dir / "adr003_source_axis_blockers_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, queue_report: Path) -> dict[str, Any]:
    queue = _read_json(queue_report)
    rows = [_case(row) for row in _current_handoff_rows(queue)]
    totals = _totals(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "queue_report": str(queue_report),
            "queue_report_sha256": _sha256(queue_report),
        },
        "totals": totals,
        "breakdowns": _breakdowns(rows),
        "cases": rows,
        "acceptance": _acceptance(totals),
        "notes": [
            "Report-only diagnostic: no route/text/runtime behavior changes.",
            "No client message or bot answer text is exported.",
            "Current handoff rows can become autonomy candidates only after proof/source-axis and text policy blockers are clear.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    lines = [
        "# ADR-003 Source-Axis Blockers",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Current handoff rows: `{totals.get('current_handoff_rows', 0)}`",
        f"- Route-only review candidates: `{totals.get('route_only_review_candidates', 0)}`",
        f"- Source-axis blocked rows: `{totals.get('source_axis_blocked', 0)}`",
        f"- Alignment review unclear rows: `{totals.get('alignment_review_unclear', 0)}`",
        f"- Danger-adjacent rows: `{totals.get('danger_adjacent', 0)}`",
        f"- Shadow renderer candidates: `{totals.get('shadow_text_renderer_candidates', 0)}`",
        "",
        "## Next workstream",
        "",
    ]
    for code, count in sorted((breakdowns.get("by_next_workstream") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code}`: `{count}`")
    lines.extend(["", "## Source alignment", ""])
    for code, count in sorted((breakdowns.get("by_source_alignment_status") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code or '<empty>'}`: `{count}`")
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        lines.append(
            f"- `{case.get('dialog_id')}#{case.get('turn')}` route=`{case.get('route')}` "
            f"action=`{case.get('requested_action')}` policy=`{case.get('policy_status')}`"
        )
        lines.append(
            f"  - next=`{case.get('next_autonomy_workstream')}` "
            f"source_alignment=`{case.get('source_alignment_status')}` "
            f"renderer=`{case.get('shadow_text_renderer_status')}`"
        )
        blockers = ", ".join(str(item) for item in (case.get("policy_blockers") or []))
        lines.append(f"  - blockers: `{blockers}`")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _current_handoff_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    real = report.get("real_lever_analysis") if isinstance(report.get("real_lever_analysis"), Mapping) else {}
    queue = real.get("current_handoff_queue") if isinstance(real.get("current_handoff_queue"), Mapping) else {}
    examples = queue.get("examples") if isinstance(queue.get("examples"), list) else []
    return [row for row in examples if isinstance(row, Mapping) and str(row.get("route") or "") in HANDOFF_ROUTES]


def _case(row: Mapping[str, Any]) -> dict[str, Any]:
    status, blockers = _policy_status(row)
    return {
        "dialog_id": str(row.get("dialog_id") or ""),
        "turn": _int(row.get("turn")),
        "route": str(row.get("route") or ""),
        "requested_action": str(row.get("requested_action") or ""),
        "next_autonomy_workstream": str(row.get("next_autonomy_workstream") or ""),
        "danger_adjacent": bool(row.get("danger_adjacent")),
        "policy_status": status,
        "policy_blockers": blockers,
        "active_behavior_allowed": False,
        "client_text_exported": False,
        "bot_text_exported": False,
        "proof_reconciliation_status": str(row.get("proof_reconciliation_status") or ""),
        "proof_reconciliation_reason": str(row.get("proof_reconciliation_reason") or ""),
        "proof_reconciliation_would_reconcile": bool(row.get("proof_reconciliation_would_reconcile")),
        "proof_reconciliation_exact_fact_key_count": len(row.get("proof_reconciliation_exact_fact_keys") or []),
        "text_readiness_status": str(row.get("text_readiness_status") or ""),
        "text_readiness_blockers": [str(item) for item in (row.get("text_readiness_blockers") or []) if str(item)],
        "source_alignment_status": str(row.get("source_alignment_status") or ""),
        "source_alignment_blockers": [str(item) for item in (row.get("source_alignment_blockers") or []) if str(item)],
        "source_alignment_missing_fact_categories": [
            str(item) for item in (row.get("source_alignment_missing_fact_categories") or []) if str(item)
        ],
        "source_alignment_fact_categories": [
            str(item) for item in (row.get("source_alignment_fact_categories") or []) if str(item)
        ],
        "source_alignment_uncovered_categories": [
            str(item) for item in (row.get("source_alignment_uncovered_categories") or []) if str(item)
        ],
        "shadow_text_renderer_status": str(row.get("shadow_text_renderer_status") or ""),
        "shadow_text_renderer_blockers": [
            str(item) for item in (row.get("shadow_text_renderer_blockers") or []) if str(item)
        ],
        "shadow_text_candidate_length": _int(row.get("shadow_text_candidate_length")),
        "shadow_text_candidate_hash": str(row.get("shadow_text_candidate_hash") or ""),
        "shadow_text_candidate_exported": bool(row.get("shadow_text_candidate_exported")),
        "active_blockers": [str(item) for item in (row.get("active_blockers") or []) if str(item)],
    }


def _policy_status(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = ["report_only", "no_text_generated", "active_behavior_forbidden"]
    next_workstream = str(row.get("next_autonomy_workstream") or "")
    route = str(row.get("route") or "")
    source_status = str(row.get("source_alignment_status") or "")
    renderer_status = str(row.get("shadow_text_renderer_status") or "")
    if bool(row.get("danger_adjacent")) or next_workstream == "danger_adjacent_do_not_lower":
        blockers.append("danger_adjacent_do_not_lower")
        return "blocked_danger_adjacent", blockers
    if route == "manager_only":
        blockers.append("manager_only_route")
        return "blocked_manager_only_route", blockers
    if next_workstream == "route_only_ack_status_candidate_review":
        blockers.extend(["requires_owner_review", "requires_semantic_review"])
        return "route_only_review_candidate", blockers
    if source_status.startswith("blocked_source_axis"):
        blockers.append("source_axis_mismatch")
        return "blocked_source_axis_mismatch", blockers
    if source_status == "alignment_review_unclear":
        blockers.append("source_axis_review_unclear")
        return "blocked_source_axis_review_unclear", blockers
    if renderer_status.startswith("blocked_"):
        blockers.append(renderer_status)
        return "blocked_shadow_renderer", blockers
    blockers.append("no_active_candidate")
    return "blocked_no_active_candidate", blockers


def _totals(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "current_handoff_rows": len(cases),
        "route_only_review_candidates": sum(1 for case in cases if case.get("policy_status") == "route_only_review_candidate"),
        "source_axis_blocked": sum(
            1 for case in cases if str(case.get("source_alignment_status") or "").startswith("blocked_source_axis")
        ),
        "source_axis_policy_primary": sum(
            1 for case in cases if case.get("policy_status") == "blocked_source_axis_mismatch"
        ),
        "alignment_review_unclear": sum(
            1 for case in cases if case.get("source_alignment_status") == "alignment_review_unclear"
        ),
        "danger_adjacent": sum(1 for case in cases if case.get("policy_status") == "blocked_danger_adjacent"),
        "manager_only_route": sum(1 for case in cases if case.get("policy_status") == "blocked_manager_only_route"),
        "proof_reconciliation_would_reconcile": sum(1 for case in cases if case.get("proof_reconciliation_would_reconcile")),
        "shadow_text_renderer_candidates": sum(
            1 for case in cases if case.get("shadow_text_renderer_status") == "candidate_rendered"
        ),
        "shadow_text_exported": sum(1 for case in cases if case.get("shadow_text_candidate_exported")),
    }


def _breakdowns(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_policy_status": dict(Counter(str(case.get("policy_status") or "") for case in cases)),
        "by_next_workstream": dict(Counter(str(case.get("next_autonomy_workstream") or "") for case in cases)),
        "by_route": dict(Counter(str(case.get("route") or "") for case in cases)),
        "by_requested_action": dict(Counter(str(case.get("requested_action") or "") for case in cases)),
        "by_source_alignment_status": dict(Counter(str(case.get("source_alignment_status") or "") for case in cases)),
        "by_shadow_text_renderer_status": dict(Counter(str(case.get("shadow_text_renderer_status") or "") for case in cases)),
    }


def _acceptance(totals: Mapping[str, Any]) -> dict[str, Any]:
    candidates = int(totals.get("route_only_review_candidates") or 0)
    notes = [
        "Active autonomy remains NO-GO: this report emits no route or text changes.",
        "Source/proof-axis blockers must clear before any text or route policy can be discussed.",
    ]
    if candidates:
        notes.append("Route-only review candidates still need owner and semantic review before implementation.")
    else:
        notes.append("No route-only current handoff candidate is available in this queue.")
    return {
        "status": "pass_report_only",
        "active_readiness": "no_go",
        "route_only_review_candidate_count": candidates,
        "notes": notes,
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
