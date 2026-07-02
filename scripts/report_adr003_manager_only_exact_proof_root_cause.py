#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_fact_gated_self_answer_readiness import build_report as build_readiness_report


SCHEMA_VERSION = "adr003_manager_only_exact_proof_root_cause_v1_2026_07_02"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report root causes for manager_only rows with exact product proof.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    args = parser.parse_args(argv)

    report = build_report(
        transcripts=args.transcripts,
        gold=args.gold,
        kb_snapshot=args.kb_snapshot,
        confidence_threshold=args.confidence_threshold,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_manager_only_exact_proof_root_cause_report.json"
    md_path = args.out_dir / "adr003_manager_only_exact_proof_root_cause_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(
    *,
    transcripts: Path,
    gold: Path,
    kb_snapshot: Path,
    confidence_threshold: float = 0.90,
) -> dict[str, Any]:
    readiness = build_readiness_report(
        transcripts=transcripts,
        gold=gold,
        kb_snapshot=kb_snapshot,
        confidence_threshold=confidence_threshold,
    )
    turns = _load_turn_index(transcripts)
    rows = _manager_only_exact_proof_rows(readiness)
    cases = [_case_from_row(row, turns.get((str(row.get("dialog_id") or ""), int(row.get("turn") or 0)))) for row in rows]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "transcripts": str(transcripts),
            "gold": str(gold),
            "kb_snapshot": str(kb_snapshot),
            "confidence_threshold": confidence_threshold,
        },
        "totals": _totals(cases, readiness),
        "breakdowns": _breakdowns(cases),
        "cases": cases,
        "acceptance": _acceptance(cases),
        "notes": [
            "Report-only scorer: route/text/runtime behavior is unchanged.",
            "This report diagnoses manager_only exact-proof rows; it does not authorize demoting manager_only.",
            "Exact product proof comes from the F2f readiness report; runtime retrieval evidence is read from transcripts.",
        ],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    lines = [
        "# ADR-003 F2g Manager-Only Exact-Proof Root Cause",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Manager-only exact-proof rows: `{totals.get('manager_only_exact_proof_rows', 0)}`",
        f"- Runtime exact proof missing: `{totals.get('runtime_exact_proof_missing', 0)}`",
        f"- Conversation plan lacks product scope: `{totals.get('conversation_plan_lacks_product_scope', 0)}`",
        f"- Frame says manager action: `{totals.get('frame_manager_action', 0)}`",
        f"- Low confidence: `{totals.get('low_confidence', 0)}`",
        "",
        "## Root Cause Codes",
        "",
    ]
    for code, count in sorted((breakdowns.get("by_root_cause") or {}).items()):
        lines.append(f"- `{code}`: `{count}`")
    lines.extend(["", "## Cases", ""])
    for item in report.get("cases") or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"- `{item.get('dialog_id')}#{item.get('turn')}` route=`{item.get('route')}` "
            f"frame=`{item.get('frame_risk_class')}/{item.get('frame_answerability')}` "
            f"action=`{item.get('requested_action')}` confidence=`{item.get('frame_confidence')}`"
        )
        lines.append(f"  - exact fact: `{item.get('source_fact_key')}` status=`{item.get('existence_status')}`")
        lines.append(
            "  - runtime retrieval: "
            f"candidate_count=`{item.get('runtime_candidate_count')}`, "
            f"selected_exact_ids=`{item.get('runtime_selected_exact_ids')}`, "
            f"selected_adjacent_ids=`{item.get('runtime_selected_adjacent_ids')}`"
        )
        lines.append(
            "  - plan: "
            f"primary_intent=`{item.get('plan_primary_intent')}`, "
            f"topic_id=`{item.get('plan_topic_id')}`, "
            f"product_scope=`{item.get('plan_product_scope')}`, "
            f"required_fact_keys=`{item.get('plan_required_fact_keys')}`"
        )
        lines.append(f"  - root causes: `{', '.join(item.get('root_cause_codes') or [])}`")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _load_turn_index(transcripts: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for line in transcripts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        dialog = json.loads(line)
        if not isinstance(dialog, Mapping):
            continue
        dialog_id = str(dialog.get("dialog_id") or "")
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            try:
                turn_no = int(turn.get("turn") or 0)
            except (TypeError, ValueError):
                continue
            result[(dialog_id, turn_no)] = turn
    return result


def _manager_only_exact_proof_rows(readiness: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    groups = readiness.get("groups") if isinstance(readiness.get("groups"), Mapping) else {}
    group = groups.get("manager_only_exact_proof_needs_policy")
    examples = group.get("examples") if isinstance(group, Mapping) else []
    return [row for row in examples or [] if isinstance(row, Mapping)]


def _case_from_row(row: Mapping[str, Any], turn: Mapping[str, Any] | None) -> dict[str, Any]:
    turn = turn if isinstance(turn, Mapping) else {}
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    plan = turn.get("bot_conversation_intent_plan") if isinstance(turn.get("bot_conversation_intent_plan"), Mapping) else {}
    contract = turn.get("bot_answer_contract") if isinstance(turn.get("bot_answer_contract"), Mapping) else {}
    retrieval = turn.get("bot_fact_retrieval_trace") if isinstance(turn.get("bot_fact_retrieval_trace"), Mapping) else {}
    self_shadow = (
        turn.get("bot_semantic_frame_self_answer_shadow")
        if isinstance(turn.get("bot_semantic_frame_self_answer_shadow"), Mapping)
        else {}
    )
    best = row.get("best_kb_match") if isinstance(row.get("best_kb_match"), Mapping) else {}
    product_check = (
        row.get("product_existence_check") if isinstance(row.get("product_existence_check"), Mapping) else {}
    )
    direct_path = turn.get("bot_direct_path") if isinstance(turn.get("bot_direct_path"), Mapping) else {}
    selected_exact = _strings(retrieval.get("selected_exact_ids") or direct_path.get("wide_fact_exact_keys"))
    selected_adjacent = _strings(retrieval.get("selected_adjacent_ids") or direct_path.get("wide_fact_adjacent_keys"))
    runtime_candidate_count = _int_or_zero(retrieval.get("candidate_count"))
    source_fact_key = str(best.get("fact_key") or (product_check.get("entry") or {}).get("source_fact_key") or "")
    frame_confidence = _float_or_none(row.get("frame_confidence") if row.get("frame_confidence") is not None else frame.get("confidence"))
    missing_facts = _strings(turn.get("bot_missing_facts"))
    root_causes = _root_causes(
        source_fact_key=source_fact_key,
        selected_exact=selected_exact,
        runtime_candidate_count=runtime_candidate_count,
        plan=plan,
        contract=contract,
        row=row,
        frame=frame,
        confidence=frame_confidence,
        missing_facts=missing_facts,
        self_shadow=self_shadow,
    )
    return {
        "dialog_id": str(row.get("dialog_id") or ""),
        "turn": _int_or_zero(row.get("turn")),
        "route": str(row.get("route") or turn.get("bot_route") or ""),
        "reason_class": str(turn.get("bot_reason_class") or ""),
        "message_type": str(turn.get("bot_message_type") or ""),
        "safety_flags": _safe_flags(turn.get("bot_safety_flags")),
        "source_fact_key": source_fact_key,
        "existence_status": str(product_check.get("status") or best.get("existence_status") or ""),
        "requested_axes": _strings(row.get("requested_axes")),
        "requested_action": str(row.get("requested_action") or frame.get("requested_action") or ""),
        "frame_risk_class": str(row.get("frame_risk_class") or frame.get("risk_class") or ""),
        "frame_answerability": str(row.get("frame_answerability") or frame.get("answerability") or ""),
        "frame_must_handoff": bool(row.get("frame_must_handoff") or frame.get("must_handoff") is True),
        "frame_confidence": frame_confidence,
        "blocked_reasons": _strings(row.get("blocked_reasons")),
        "runtime_candidate_count": runtime_candidate_count,
        "runtime_selected_exact_ids": selected_exact,
        "runtime_selected_adjacent_ids": selected_adjacent,
        "runtime_model_needed_facts": _strings(retrieval.get("model_needed_facts")),
        "runtime_retrieval_mode": str(retrieval.get("mode") or ""),
        "plan_primary_intent": str(plan.get("primary_intent") or ""),
        "plan_topic_id": str(plan.get("topic_id") or ""),
        "plan_route_bias": str(plan.get("route_bias") or ""),
        "plan_product_scope": str(plan.get("product_scope") or ""),
        "plan_fact_scope": str(plan.get("fact_scope") or ""),
        "plan_direct_question_present": bool(str(plan.get("direct_question") or "").strip()),
        "plan_required_fact_keys": _strings(plan.get("required_fact_keys")),
        "plan_known_slot_keys": sorted(str(key) for key in (plan.get("known_slots") or {}).keys())
        if isinstance(plan.get("known_slots"), Mapping)
        else [],
        "contract_route": str(contract.get("route") or ""),
        "contract_route_bias": str(contract.get("route_bias") or ""),
        "contract_route_reason": str(contract.get("route_reason") or ""),
        "contract_required_fact_keys": _strings(contract.get("required_fact_keys")),
        "self_shadow_reason": str(self_shadow.get("reason") or ""),
        "self_shadow_freshness_reason": str((self_shadow.get("guards") or {}).get("freshness", {}).get("reason") or "")
        if isinstance(self_shadow.get("guards"), Mapping)
        else "",
        "missing_fact_count": len(missing_facts),
        "root_cause_codes": root_causes,
    }


def _root_causes(
    *,
    source_fact_key: str,
    selected_exact: Sequence[str],
    runtime_candidate_count: int,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    row: Mapping[str, Any],
    frame: Mapping[str, Any],
    confidence: float | None,
    missing_facts: Sequence[str],
    self_shadow: Mapping[str, Any],
) -> list[str]:
    causes: list[str] = ["route_locked_manager_only"]
    if source_fact_key and source_fact_key not in set(selected_exact):
        causes.append("runtime_retrieval_missed_exact_fact")
    if runtime_candidate_count <= 0 and source_fact_key:
        causes.append("runtime_retrieval_zero_candidates")
    if not _strings(plan.get("required_fact_keys")) and not str(plan.get("product_scope") or "").strip() and not str(
        plan.get("direct_question") or ""
    ).strip():
        causes.append("conversation_plan_no_product_scope")
    if not _strings(contract.get("required_fact_keys")):
        causes.append("answer_contract_no_required_fact_keys")
    risk = str(row.get("frame_risk_class") or frame.get("risk_class") or "").strip().casefold()
    answerability = str(row.get("frame_answerability") or frame.get("answerability") or "").strip().casefold()
    action = str(row.get("requested_action") or frame.get("requested_action") or "").strip().casefold()
    if risk == "manager_action" or answerability == "manager_only" or frame.get("must_handoff") is True:
        causes.append("frame_marks_manager_action")
    if action in {"check_availability", "enroll", "capture_lead", "send_payment_link"}:
        causes.append("frame_action_not_safe_reference")
    if confidence is None or confidence < 0.90:
        causes.append("frame_confidence_below_threshold")
    if missing_facts:
        causes.append("runtime_missing_facts_present")
    freshness = self_shadow.get("guards", {}).get("freshness") if isinstance(self_shadow.get("guards"), Mapping) else {}
    if isinstance(freshness, Mapping) and str(freshness.get("reason") or "") == "no_exact_fact_keys":
        causes.append("self_shadow_has_no_runtime_exact_fact_keys")
    return list(dict.fromkeys(causes))


def _totals(cases: Sequence[Mapping[str, Any]], readiness: Mapping[str, Any]) -> dict[str, Any]:
    cause_counts = Counter(code for case in cases for code in case.get("root_cause_codes") or [])
    readiness_totals = readiness.get("totals") if isinstance(readiness.get("totals"), Mapping) else {}
    return {
        "readiness_strict_f3_draft_candidates": readiness_totals.get("strict_f3_draft_candidates", 0),
        "manager_only_exact_proof_rows": len(cases),
        "runtime_exact_proof_missing": cause_counts.get("runtime_retrieval_missed_exact_fact", 0),
        "conversation_plan_lacks_product_scope": cause_counts.get("conversation_plan_no_product_scope", 0),
        "frame_manager_action": cause_counts.get("frame_marks_manager_action", 0),
        "low_confidence": cause_counts.get("frame_confidence_below_threshold", 0),
    }


def _breakdowns(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_root_cause": dict(Counter(code for case in cases for code in case.get("root_cause_codes") or [])),
        "by_requested_action": dict(Counter(str(case.get("requested_action") or "") for case in cases)),
        "by_reason_class": dict(Counter(str(case.get("reason_class") or "") for case in cases)),
        "by_message_type": dict(Counter(str(case.get("message_type") or "") for case in cases)),
    }


def _acceptance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        status = "pass_no_manager_only_exact_proof_rows"
        notes = ["No manager_only exact-proof rows remain in this input."]
    else:
        status = "pass_diagnosed"
        notes = [
            "Active F3 remains NO-GO: this report diagnoses manager_only rows only.",
            "If runtime retrieval missed exact proof, the next safe step is shadow evidence injection or retrieval diagnostics, not manager_only demotion.",
            "If frame marks manager_action, frame calibration is required before any active route change.",
        ]
    notes.append("Report-only: no route/text/runtime changes.")
    return {"status": status, "active_readiness": "no_go", "notes": notes}


def _safe_flags(value: object) -> list[str]:
    flags = _strings(value)
    return [flag for flag in flags if flag not in {"manager_approval_required", "no_auto_send"}]


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return []


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _source_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
