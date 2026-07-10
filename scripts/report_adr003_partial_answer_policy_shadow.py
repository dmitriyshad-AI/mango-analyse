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


SCHEMA_VERSION = "adr003_partial_answer_policy_shadow_v1_2026_07_02"
PARTIAL_CANDIDATE_STATUS = "draft_partial_shadow_candidate"
DANGER_WORKSTREAM = "danger_adjacent_do_not_lower"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join ADR-003 partial-answer candidates with calibration queue blockers."
    )
    parser.add_argument("--partial-report", type=Path, required=True)
    parser.add_argument("--queue-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(partial_report=args.partial_report, queue_report=args.queue_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_partial_answer_policy_shadow_report.json"
    md_path = args.out_dir / "adr003_partial_answer_policy_shadow_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, partial_report: Path, queue_report: Path) -> dict[str, Any]:
    partial = _read_json(partial_report)
    queue = _read_json(queue_report)
    queue_index = _queue_index(queue)
    partial_cases = [
        case
        for case in (partial.get("partial_cases") or [])
        if isinstance(case, Mapping)
        and isinstance(case.get("partial_answer_shadow"), Mapping)
        and case["partial_answer_shadow"].get("status") == PARTIAL_CANDIDATE_STATUS
    ]
    cases = [_case_from_partial(case, queue_index=queue_index) for case in partial_cases]
    totals = _totals(partial, cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "partial_report": str(partial_report),
            "partial_report_sha256": _sha256(partial_report),
            "queue_report": str(queue_report),
            "queue_report_sha256": _sha256(queue_report),
        },
        "totals": totals,
        "breakdowns": _breakdowns(cases),
        "cases": cases,
        "acceptance": _acceptance(totals),
        "notes": [
            "Report-only diagnostic: no route/text/runtime behavior changes.",
            "No customer-facing text is generated or exported.",
            "Partial-answer policy can only advance after cross-report blockers are clear.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    lines = [
        "# ADR-003 F2ac Partial Answer Policy Shadow",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Partial draft candidates in input: `{totals.get('partial_draft_candidates_input', 0)}`",
        f"- Joined with calibration queue: `{totals.get('joined_with_queue', 0)}`",
        f"- Policy shadow candidates: `{totals.get('policy_shadow_candidates', 0)}`",
        f"- Blocked by danger adjacency: `{totals.get('blocked_danger_adjacent', 0)}`",
        f"- Blocked by source-axis mismatch: `{totals.get('blocked_source_axis_mismatch', 0)}`",
        "",
        "## Status breakdown",
        "",
    ]
    for code, count in sorted((breakdowns.get("by_policy_status") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code}`: `{count}`")
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        lines.append(
            f"- `{case.get('dialog_id')}#{case.get('turn')}` route=`{case.get('route')}` "
            f"partial=`{case.get('partial_status')}` policy=`{case.get('policy_status')}`"
        )
        lines.append(
            f"  - queue: workstream=`{case.get('queue_next_workstream', '')}` "
            f"danger=`{case.get('queue_danger_adjacent', False)}` "
            f"source_alignment=`{case.get('queue_source_alignment_status', '')}`"
        )
        blockers = ", ".join(str(item) for item in (case.get("policy_blockers") or []))
        lines.append(f"  - blockers: `{blockers}`")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _case_from_partial(case: Mapping[str, Any], *, queue_index: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    dialog_id = str(case.get("dialog_id") or "")
    turn = _int(case.get("turn"))
    queue = queue_index.get((dialog_id, turn), {})
    policy = _policy_shadow(case, queue)
    partial = case.get("partial_answer_shadow") if isinstance(case.get("partial_answer_shadow"), Mapping) else {}
    support = case.get("kb_support") if isinstance(case.get("kb_support"), Mapping) else {}
    return {
        "dialog_id": dialog_id,
        "turn": turn,
        "route": str(case.get("route") or ""),
        "requested_action": str(case.get("requested_action") or ""),
        "risk_class": str(case.get("risk_class") or ""),
        "answerability": str(case.get("answerability") or ""),
        "must_handoff": case.get("must_handoff"),
        "confidence": case.get("confidence"),
        "partial_status": str(partial.get("status") or ""),
        "partial_proven_parts": [str(item) for item in (support.get("proven_parts") or []) if str(item)],
        "partial_missing_slots": [str(item) for item in (support.get("missing_slots") or []) if str(item)],
        "partial_uncovered_categories": [str(item) for item in (support.get("uncovered_categories") or []) if str(item)],
        "queue_joined": bool(queue),
        "queue_next_workstream": str(queue.get("next_autonomy_workstream") or ""),
        "queue_danger_adjacent": bool(queue.get("danger_adjacent")),
        "queue_active_blockers": [str(item) for item in (queue.get("active_blockers") or []) if str(item)],
        "queue_source_alignment_status": str(queue.get("source_alignment_status") or ""),
        "queue_source_alignment_blockers": [str(item) for item in (queue.get("source_alignment_blockers") or []) if str(item)],
        "queue_text_policy_readiness_status": str(queue.get("text_policy_readiness_status") or ""),
        "queue_shadow_text_renderer_status": str(queue.get("shadow_text_renderer_status") or ""),
        "queue_text_readiness_status": str(queue.get("text_readiness_status") or ""),
        **policy,
    }


def _policy_shadow(case: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = ["report_only", "no_text_generated", "active_behavior_forbidden"]
    status = "blocked_unknown_queue_status"
    if not queue:
        blockers.append("missing_queue_join")
        return _policy(status, blockers)
    if bool(queue.get("danger_adjacent")) or str(queue.get("next_autonomy_workstream") or "") == DANGER_WORKSTREAM:
        blockers.append("danger_adjacent_do_not_lower")
        return _policy("blocked_danger_adjacent", blockers)
    if str(case.get("route") or "") != "draft_for_manager":
        blockers.append("route_not_draft_for_manager")
        return _policy("blocked_route_not_draft_for_manager", blockers)
    if str(case.get("requested_action") or "") != "answer_question":
        blockers.append("requested_action_not_answer_question")
        return _policy("blocked_requested_action", blockers)
    if str(case.get("risk_class") or "") in {"p0", "manager_action"}:
        blockers.append("risk_class_not_safe_reference")
        return _policy("blocked_risk_class", blockers)
    if _has_blocker(queue, "source_axis_mismatch"):
        blockers.append("source_axis_mismatch")
        return _policy("blocked_source_axis_mismatch", blockers)
    if str(queue.get("text_readiness_status") or "").startswith("blocked_"):
        blockers.append(str(queue.get("text_readiness_status") or "text_readiness_blocked"))
        return _policy("blocked_current_text_not_reusable", blockers)
    if str(queue.get("shadow_text_renderer_status") or "").startswith("blocked_"):
        blockers.append(str(queue.get("shadow_text_renderer_status") or "shadow_renderer_blocked"))
        return _policy("blocked_shadow_renderer", blockers)
    if bool(queue.get("direct_quote_forbidden")) and str(queue.get("template_registry_status") or "") != "found":
        blockers.append("direct_quote_forbidden_without_template")
        return _policy("blocked_no_template_renderer", blockers)
    blockers.append("requires_owner_partial_answer_policy")
    blockers.append("requires_semantic_review")
    return _policy("policy_shadow_candidate_requires_owner_review", blockers)


def _policy(status: str, blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "policy_status": status,
        "policy_blockers": list(dict.fromkeys(str(item) for item in blockers if str(item))),
        "active_behavior_allowed": False,
        "generated_text_exported": False,
        "customer_text_exported": False,
    }


def _has_blocker(queue: Mapping[str, Any], marker: str) -> bool:
    values = [
        queue.get("source_alignment_status"),
        queue.get("text_policy_readiness_status"),
        queue.get("shadow_text_renderer_status"),
        *(queue.get("source_alignment_blockers") or ()),
        *(queue.get("text_policy_blockers") or ()),
        *(queue.get("shadow_text_renderer_blockers") or ()),
    ]
    return any(marker in str(value or "") for value in values)


def _queue_index(queue_report: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    real = queue_report.get("real_lever_analysis") if isinstance(queue_report.get("real_lever_analysis"), Mapping) else {}
    rows: list[Mapping[str, Any]] = []
    for source in (real.get("examples"), (real.get("current_handoff_queue") or {}).get("examples")):
        if isinstance(source, list):
            rows.extend(item for item in source if isinstance(item, Mapping))
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dialog_id") or ""), _int(row.get("turn")))
        if key[0] and key not in index:
            index[key] = row
    return index


def _totals(partial_report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_total = sum(
        1
        for case in (partial_report.get("partial_cases") or [])
        if isinstance(case, Mapping)
        and isinstance(case.get("partial_answer_shadow"), Mapping)
        and case["partial_answer_shadow"].get("status") == PARTIAL_CANDIDATE_STATUS
    )
    return {
        "partial_draft_candidates_input": input_total,
        "joined_with_queue": sum(1 for case in cases if case.get("queue_joined")),
        "policy_shadow_candidates": sum(
            1 for case in cases if case.get("policy_status") == "policy_shadow_candidate_requires_owner_review"
        ),
        "blocked_danger_adjacent": sum(1 for case in cases if case.get("policy_status") == "blocked_danger_adjacent"),
        "blocked_source_axis_mismatch": sum(1 for case in cases if case.get("policy_status") == "blocked_source_axis_mismatch"),
        "blocked_other": sum(
            1
            for case in cases
            if str(case.get("policy_status") or "").startswith("blocked_")
            and case.get("policy_status") not in {"blocked_danger_adjacent", "blocked_source_axis_mismatch"}
        ),
    }


def _breakdowns(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_policy_status": dict(Counter(str(case.get("policy_status") or "") for case in cases)),
        "by_queue_workstream": dict(Counter(str(case.get("queue_next_workstream") or "") for case in cases)),
        "by_source_alignment": dict(Counter(str(case.get("queue_source_alignment_status") or "") for case in cases)),
    }


def _acceptance(totals: Mapping[str, Any]) -> dict[str, Any]:
    candidates = int(totals.get("policy_shadow_candidates") or 0)
    notes = [
        "Active autonomy remains NO-GO: this report emits no route or text changes.",
        "Partial-answer candidates must survive cross-report queue blockers before any text policy discussion.",
    ]
    if candidates:
        notes.append("Policy candidates still require Дмитрий/Claude owner review and semantic review before any text work.")
    else:
        notes.append("No partial-answer policy candidate survives cross-report blockers.")
    return {
        "status": "pass_report_only",
        "active_readiness": "no_go",
        "policy_shadow_candidate_count": candidates,
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
