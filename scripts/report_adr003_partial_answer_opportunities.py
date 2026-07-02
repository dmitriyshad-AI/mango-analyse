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

from mango_mvp.knowledge_base.product_existence_axes_catalog import build_product_existence_axes_catalog

from scripts.report_adr003_current_handoff_fact_gap import (
    HANDOFF_ROUTES,
    _case_missing_categories,
    _kb_support,
    _listish,
    _load_kb_facts,
    _redacted_excerpt,
    _sha256,
    _source_rev,
)


SCHEMA_VERSION = "adr003_partial_answer_opportunities_v1_2026_07_02"
SAFE_ACTIONS = {"answer_question"}
P0_OR_MONEY_ACTIONS = {"refund_or_cancel", "send_payment_link", "send_document", "handoff_manager", "enroll"}
P0_OR_MONEY_PAYMENTS = {"ready_to_pay", "paid", "dispute"}
P0_OR_MONEY_STAGES = {"post_payment", "support"}
HARD_BLOCKING_MISSING_CATEGORIES = {
    "live_availability",
    "payment_access",
}
COMPLEX_MISSING_CATEGORIES = {
    "boarding_food",
    "dates_schedule",
    "location_address",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report ADR-003 partial-answer opportunities from enriched transcripts.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(transcripts=args.transcripts, kb_snapshot=args.kb_snapshot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_partial_answer_opportunities_report.json"
    md_path = args.out_dir / "adr003_partial_answer_opportunities_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, transcripts: Path, kb_snapshot: Path) -> dict[str, Any]:
    facts = _load_kb_facts(kb_snapshot)
    catalog = build_product_existence_axes_catalog(facts)
    turns = _load_turn_rows(transcripts)
    cases = [_case_from_turn(turn, facts=facts, catalog=catalog) for turn in turns]
    handoff_cases = [case for case in cases if case["route"] in HANDOFF_ROUTES]
    partial_cases = [case for case in handoff_cases if case["partial_answer_shadow"]["has_partial_support"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "transcripts": str(transcripts),
            "transcripts_sha256": _sha256(transcripts),
            "kb_snapshot": str(kb_snapshot),
            "kb_snapshot_sha256": _sha256(kb_snapshot),
        },
        "totals": _totals(cases, handoff_cases, partial_cases),
        "breakdowns": _breakdowns(handoff_cases, partial_cases),
        "partial_cases": partial_cases,
        "acceptance": _acceptance(partial_cases),
        "notes": [
            "Report-only diagnostic: no route/text/runtime behavior changes.",
            "No customer-facing partial answer is generated or exported.",
            "A future active step requires owner-approved text policy and semantic review.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    lines = [
        "# ADR-003 F2z Partial Answer Opportunities",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Total turns: `{totals.get('total_turns', 0)}`",
        f"- Handoff turns: `{totals.get('handoff_turns', 0)}`",
        f"- Partial-support handoff turns: `{totals.get('partial_support_handoff_turns', 0)}`",
        f"- Draft partial shadow candidates: `{totals.get('draft_partial_shadow_candidates', 0)}`",
        f"- Manager-only partial policy blocked: `{totals.get('manager_only_partial_policy_blocked', 0)}`",
        f"- Action/danger excluded partial rows: `{totals.get('action_or_danger_excluded_partial_rows', 0)}`",
        "",
        "## Partial Root Causes",
        "",
    ]
    for code, count in sorted((breakdowns.get("partial_by_status") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code}`: `{count}`")
    lines.extend(["", "## Partial Cases", ""])
    for case in report.get("partial_cases") or []:
        if not isinstance(case, Mapping):
            continue
        partial = case.get("partial_answer_shadow") if isinstance(case.get("partial_answer_shadow"), Mapping) else {}
        support = case.get("kb_support") if isinstance(case.get("kb_support"), Mapping) else {}
        lines.append(
            f"- `{case.get('dialog_id')}#{case.get('turn')}` route=`{case.get('route')}` "
            f"action=`{case.get('requested_action')}` status=`{partial.get('status')}`"
        )
        lines.append(
            f"  - proven: `{', '.join(support.get('proven_parts') or [])}`; "
            f"missing slots: `{', '.join(support.get('missing_slots') or [])}`; "
            f"uncovered: `{', '.join(support.get('uncovered_categories') or [])}`"
        )
        lines.append(f"  - why not active: `{', '.join(partial.get('why_not_active') or [])}`")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _case_from_turn(
    turn: Mapping[str, Any],
    *,
    facts: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    requested = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    proof = (
        turn.get("bot_semantic_frame_proof_reconciliation_shadow")
        if isinstance(turn.get("bot_semantic_frame_proof_reconciliation_shadow"), Mapping)
        else {}
    )
    missing_text = " ".join(str(item or "") for item in _listish(proof.get("result_missing_facts") or turn.get("bot_missing_facts")))
    missing_categories = _case_missing_categories(missing_text)
    support = _kb_support(facts, catalog, turn=turn, requested=requested, missing_categories=missing_categories)
    partial = _partial_answer_shadow(turn, frame=frame, support=support, missing_categories=missing_categories)
    return {
        "dialog_id": str(turn.get("__dialog_id") or ""),
        "turn": int(turn.get("__turn_no") or 0),
        "route": str(turn.get("bot_route") or ""),
        "requested_action": str(frame.get("requested_action") or ""),
        "risk_class": str(frame.get("risk_class") or ""),
        "answerability": str(frame.get("answerability") or ""),
        "must_handoff": frame.get("must_handoff"),
        "confidence": frame.get("confidence"),
        "client_excerpt": _redacted_excerpt(turn.get("client_message"), limit=180),
        "missing_fact_categories": missing_categories,
        "missing_fact_count": len(_listish(proof.get("result_missing_facts") or turn.get("bot_missing_facts"))),
        "kb_support": support,
        "partial_answer_shadow": partial,
    }


def _partial_answer_shadow(
    turn: Mapping[str, Any],
    *,
    frame: Mapping[str, Any],
    support: Mapping[str, Any],
    missing_categories: Sequence[str],
) -> dict[str, Any]:
    route = str(turn.get("bot_route") or "")
    requested_action = str(frame.get("requested_action") or "").strip().casefold()
    risk_class = str(frame.get("risk_class") or "").strip().casefold()
    payment_readiness = str(frame.get("payment_readiness") or "").strip().casefold()
    deal_stage = str(frame.get("deal_stage") or "").strip().casefold()
    proven_parts = [str(item) for item in (support.get("proven_parts") or []) if str(item)]
    missing_slots = [str(item) for item in (support.get("missing_slots") or []) if str(item)]
    uncovered = [str(item) for item in (support.get("uncovered_categories") or []) if str(item)]
    has_partial_support = bool(proven_parts) and bool(missing_slots or uncovered)
    reasons = ["report_only", "no_text_generated"]
    status = "no_partial_support"
    if not has_partial_support:
        return {
            "has_partial_support": False,
            "status": status,
            "why_not_active": reasons,
            "active_behavior_allowed": False,
        }
    if route not in HANDOFF_ROUTES:
        status = "already_not_handoff"
        reasons.append("route_not_handoff")
    elif requested_action not in SAFE_ACTIONS:
        status = "action_or_danger_excluded"
        reasons.append(f"requested_action:{requested_action or 'unknown'}")
    elif risk_class == "p0" or requested_action in P0_OR_MONEY_ACTIONS:
        status = "action_or_danger_excluded"
        reasons.append("p0_or_money_action")
    elif payment_readiness in P0_OR_MONEY_PAYMENTS or deal_stage in P0_OR_MONEY_STAGES:
        status = "action_or_danger_excluded"
        reasons.append("payment_or_support_context")
    elif set(uncovered).intersection(HARD_BLOCKING_MISSING_CATEGORIES):
        status = "hard_missing_axis_blocked"
        reasons.append("hard_missing_axis")
    elif route == "manager_only":
        status = "manager_only_policy_blocked"
        reasons.append("route_manager_only")
    elif len(set(uncovered).intersection(COMPLEX_MISSING_CATEGORIES)) >= 2:
        status = "broad_missing_axes_blocked"
        reasons.append("broad_missing_axes")
    else:
        status = "draft_partial_shadow_candidate"
        reasons.append("requires_owner_partial_answer_policy")
    return {
        "has_partial_support": True,
        "status": status,
        "why_not_active": list(dict.fromkeys(reasons)),
        "active_behavior_allowed": False,
        "would_require_text_policy": status == "draft_partial_shadow_candidate",
        "generated_text_exported": False,
    }


def _load_turn_rows(transcripts: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with transcripts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            dialog = json.loads(line)
            if not isinstance(dialog, Mapping):
                continue
            dialog_id = str(dialog.get("dialog_id") or "")
            dialog_brand = str(dialog.get("brand") or "")
            for index, turn in enumerate(dialog.get("turns") or [], 1):
                if not isinstance(turn, Mapping):
                    continue
                row = dict(turn)
                row["__dialog_id"] = dialog_id
                row["__turn_no"] = int(turn.get("turn") or index)
                row["__dialog_brand"] = dialog_brand or str(turn.get("brand") or "")
                rows.append(row)
    return rows


def _totals(
    cases: Sequence[Mapping[str, Any]],
    handoff_cases: Sequence[Mapping[str, Any]],
    partial_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "total_turns": len(cases),
        "handoff_turns": len(handoff_cases),
        "partial_support_handoff_turns": len(partial_cases),
        "draft_partial_shadow_candidates": sum(
            1 for case in partial_cases if (case.get("partial_answer_shadow") or {}).get("status") == "draft_partial_shadow_candidate"
        ),
        "manager_only_partial_policy_blocked": sum(
            1 for case in partial_cases if (case.get("partial_answer_shadow") or {}).get("status") == "manager_only_policy_blocked"
        ),
        "action_or_danger_excluded_partial_rows": sum(
            1 for case in partial_cases if (case.get("partial_answer_shadow") or {}).get("status") == "action_or_danger_excluded"
        ),
        "hard_missing_axis_blocked": sum(
            1 for case in partial_cases if (case.get("partial_answer_shadow") or {}).get("status") == "hard_missing_axis_blocked"
        ),
        "broad_missing_axes_blocked": sum(
            1 for case in partial_cases if (case.get("partial_answer_shadow") or {}).get("status") == "broad_missing_axes_blocked"
        ),
    }


def _breakdowns(
    handoff_cases: Sequence[Mapping[str, Any]],
    partial_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "handoff_by_route": dict(Counter(str(case.get("route") or "") for case in handoff_cases)),
        "handoff_by_requested_action": dict(Counter(str(case.get("requested_action") or "") for case in handoff_cases)),
        "partial_by_status": dict(
            Counter(str((case.get("partial_answer_shadow") or {}).get("status") or "") for case in partial_cases)
        ),
        "partial_by_route": dict(Counter(str(case.get("route") or "") for case in partial_cases)),
    }


def _acceptance(partial_cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    draft_candidates = [
        case
        for case in partial_cases
        if (case.get("partial_answer_shadow") or {}).get("status") == "draft_partial_shadow_candidate"
    ]
    notes = [
        "Active autonomy remains NO-GO: this report emits no route or text changes.",
        "Partial-answer candidates require owner-approved text policy and semantic review.",
        "manager_only rows stay blocked even with partial support.",
    ]
    if draft_candidates:
        notes.append("There are draft-route partial shadow candidates worth owner review; this is not approval to send text.")
    return {
        "status": "pass_report_only",
        "active_readiness": "no_go",
        "draft_partial_shadow_candidate_count": len(draft_candidates),
        "notes": notes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
