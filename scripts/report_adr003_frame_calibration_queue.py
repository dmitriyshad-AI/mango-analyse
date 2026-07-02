#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_exact_proof_injection_shadow import build_report as build_injection_report
from scripts.report_adr003_frame_gold_calibration import build_report as build_gold_report
from scripts.report_adr003_manager_only_exact_proof_root_cause import build_report as build_root_cause_report
from scripts.report_adr003_overhandoff_levers import build_report as build_overhandoff_report


SCHEMA_VERSION = "adr003_frame_calibration_queue_v1_2026_07_02"
HANDOFF_ROUTES = {"manager_only", "draft_for_manager"}
SAFE_ACTIONS = {"answer_question", "acknowledge", "acknowledge_status", "acknowledge_pause"}
OPERATIONAL_ACTIONS = {"check_availability", "enroll", "book", "reserve", "handoff_manager", "send_payment_link"}
FRAME_MANAGER_RISKS = {"manager_action", "missing_facts"}
WORKSTREAMS = (
    "semanticframe_existence_vs_availability",
    "semanticframe_safe_reference_missing_facts",
    "semanticframe_low_confidence",
    "retrieval_delivery_runtime_missing_exact_proof",
    "conversation_plan_scope_missing",
    "policy_manager_only_exact_proof",
    "policy_context_update_exact_proof",
    "danger_adjacent_do_not_lower",
    "already_self_no_active_leverage",
    "measurement_review_unclear",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ADR-003 SemanticFrame calibration/work queue from gold + shadow reports.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    args = parser.parse_args(argv)

    report = build_report(
        transcripts=args.transcripts,
        gold=args.gold,
        kb_snapshot=args.kb_snapshot,
        confidence_threshold=args.confidence_threshold,
        as_of_date=_parse_date(args.as_of_date),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_frame_calibration_queue_report.json"
    md_path = args.out_dir / "adr003_frame_calibration_queue_report.md"
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
    as_of_date: date | None = None,
) -> dict[str, Any]:
    gold_report = build_gold_report(transcripts=transcripts, gold=gold)
    overhandoff = build_overhandoff_report(transcripts=transcripts, gold=gold)
    root_cause = build_root_cause_report(
        transcripts=transcripts,
        gold=gold,
        kb_snapshot=kb_snapshot,
        confidence_threshold=confidence_threshold,
    )
    injection = build_injection_report(
        transcripts=transcripts,
        gold=gold,
        kb_snapshot=kb_snapshot,
        confidence_threshold=confidence_threshold,
        as_of_date=as_of_date or date.today(),
    )

    gold_rows = [row for row in gold_report.get("rows") or [] if isinstance(row, Mapping)]
    work_items = _build_work_items(
        gold_rows=gold_rows,
        overhandoff=overhandoff,
        root_cause=root_cause,
        injection=injection,
        confidence_threshold=confidence_threshold,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "transcripts": str(transcripts),
            "gold": str(gold),
            "kb_snapshot": str(kb_snapshot),
            "confidence_threshold": confidence_threshold,
            "as_of_date": (as_of_date or date.today()).isoformat(),
        },
        "totals": _totals(gold_report=gold_report, overhandoff=overhandoff, injection=injection, work_items=work_items),
        "workstreams": _workstreams(work_items),
        "field_error_breakdowns": _field_error_breakdowns(gold_rows),
        "source_report_summaries": {
            "gold_calibration": gold_report.get("summary") if isinstance(gold_report.get("summary"), Mapping) else {},
            "overhandoff": overhandoff.get("totals") if isinstance(overhandoff.get("totals"), Mapping) else {},
            "manager_only_root_cause": root_cause.get("totals") if isinstance(root_cause.get("totals"), Mapping) else {},
            "exact_proof_injection": injection.get("totals") if isinstance(injection.get("totals"), Mapping) else {},
        },
        "acceptance": _acceptance(work_items, gold_report=gold_report, overhandoff=overhandoff, injection=injection),
        "notes": [
            "Report-only scorer: route/text/runtime behavior is unchanged.",
            "This report separates manual too-cautious labels from true SemanticFrame field errors.",
            "Active demotion remains NO-GO until SemanticFrame, retrieval delivery, and policy blockers are resolved in shadow.",
        ],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    workstreams = report.get("workstreams") if isinstance(report.get("workstreams"), Mapping) else {}
    lines = [
        "# ADR-003 F2i SemanticFrame Calibration Queue",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Gold compared rows: `{totals.get('gold_compared_rows', 0)}`",
        f"- Safe/self gold rows: `{totals.get('safe_self_rows', 0)}`",
        f"- Manual too-cautious labels: `{totals.get('manual_too_cautious_labels', 0)}`",
        f"- True frame must_handoff too-cautious: `{totals.get('true_frame_too_cautious', 0)}`",
        f"- True frame too-confident: `{totals.get('true_frame_too_confident', 0)}`",
        f"- Current safe over-handoff candidates: `{totals.get('current_safe_over_handoff', 0)}`",
        f"- Strict active candidates now: `{totals.get('strict_active_candidates_now', 0)}`",
        f"- Manager-only exact-proof rows: `{totals.get('manager_only_exact_proof_rows', 0)}`",
        "",
        "## Workstreams",
        "",
    ]
    for name in WORKSTREAMS:
        value = workstreams.get(name) if isinstance(workstreams.get(name), Mapping) else {}
        lines.append(f"- `{name}`: `{value.get('count', 0)}`")
    lines.extend(["", "## Priority Examples", ""])
    for name in (
        "semanticframe_existence_vs_availability",
        "retrieval_delivery_runtime_missing_exact_proof",
        "conversation_plan_scope_missing",
        "policy_manager_only_exact_proof",
        "policy_context_update_exact_proof",
        "semanticframe_low_confidence",
    ):
        value = workstreams.get(name) if isinstance(workstreams.get(name), Mapping) else {}
        examples = value.get("examples") if isinstance(value.get("examples"), list) else []
        if not examples:
            continue
        lines.append(f"### `{name}`")
        for item in examples[:10]:
            lines.append(
                f"- `{item.get('dialog_id')}#{item.get('turn')}` "
                f"route=`{item.get('route')}` action=`{item.get('requested_action')}` "
                f"confidence=`{item.get('frame_confidence')}`"
            )
            if item.get("reasons"):
                lines.append(f"  - reasons: `{', '.join(item.get('reasons') or [])}`")
        lines.append("")
    lines.extend(["## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _build_work_items(
    *,
    gold_rows: Sequence[Mapping[str, Any]],
    overhandoff: Mapping[str, Any],
    root_cause: Mapping[str, Any],
    injection: Mapping[str, Any],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in gold_rows:
        if not _safe_self(row):
            if _true_frame_too_confident(row):
                items.append(_item(row, "measurement_review_unclear", ["frame_too_confident_on_handoff_gold"]))
            continue
        frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
        current_route = str(row.get("current_route") or "")
        notes = str(row.get("notes") or "").casefold()
        action = str(frame.get("requested_action") or "").casefold()
        risk = str(frame.get("risk_class") or "").casefold()
        answerability = str(frame.get("answerability") or "").casefold()
        if _existence_or_format_notes(notes) and (action in OPERATIONAL_ACTIONS or risk in FRAME_MANAGER_RISKS or answerability == "manager_only"):
            reasons = ["safe_existence_or_format_labeled_as_manager_or_operational"]
            if action in OPERATIONAL_ACTIONS:
                reasons.append(f"frame_action:{action}")
            if risk in FRAME_MANAGER_RISKS:
                reasons.append(f"frame_risk:{risk}")
            if answerability == "manager_only":
                reasons.append("frame_answerability:manager_only")
            items.append(_item(row, "semanticframe_existence_vs_availability", reasons))
        if risk == "missing_facts" or (answerability == "manager_only" and action in SAFE_ACTIONS):
            items.append(_item(row, "semanticframe_safe_reference_missing_facts", ["safe_reference_promoted_to_missing_facts_or_manager_only"]))
        confidence = _float_or_none(frame.get("confidence"))
        if confidence is None or confidence < confidence_threshold:
            items.append(_item(row, "semanticframe_low_confidence", [f"confidence_below:{confidence_threshold}"]))
        if current_route in HANDOFF_ROUTES and _fields_all_correct(row):
            items.append(_item(row, "measurement_review_unclear", ["runtime_handoff_but_frame_fields_are_correct"]))
        if current_route not in HANDOFF_ROUTES:
            items.append(_item(row, "already_self_no_active_leverage", ["current_route_already_self_or_not_handoff"]))

    over_groups = overhandoff.get("groups") if isinstance(overhandoff.get("groups"), Mapping) else {}
    danger = over_groups.get("danger_adjacent_blocked") if isinstance(over_groups.get("danger_adjacent_blocked"), Mapping) else {}
    for row in danger.get("examples") or []:
        if isinstance(row, Mapping):
            items.append(_item_from_mapping(row, "danger_adjacent_do_not_lower", ["danger_adjacent_dialog"]))

    for case in root_cause.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        root_codes = set(str(code) for code in (case.get("root_cause_codes") or []))
        if "conversation_plan_no_product_scope" in root_codes:
            items.append(_item_from_mapping(case, "conversation_plan_scope_missing", ["conversation_plan_no_product_scope"]))

    for case in injection.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        if case.get("runtime_exact_fact_was_missing"):
            items.append(_item_from_mapping(case, "retrieval_delivery_runtime_missing_exact_proof", ["runtime_did_not_receive_exact_kb_proof"]))
        blockers = set(str(code) for code in (case.get("residual_blockers") or []))
        if "route_is_manager_only" in blockers and case.get("fresh_client_safe_exact_proof"):
            items.append(_item_from_mapping(case, "policy_manager_only_exact_proof", ["manager_only_policy_blocks_even_with_fresh_exact_proof"]))
        if "message_type_context_update" in blockers and case.get("fresh_client_safe_exact_proof"):
            items.append(_item_from_mapping(case, "policy_context_update_exact_proof", ["context_update_policy_blocks_even_with_fresh_exact_proof"]))
    return items


def _item(row: Mapping[str, Any], workstream: str, reasons: Sequence[str]) -> dict[str, Any]:
    frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
    return {
        "workstream": workstream,
        "active_allowed": False,
        "active_block_reason": _active_block_reason(workstream),
        "calibration_target": _calibration_target(workstream, row),
        "recommended_review_question": _review_question(workstream),
        "dialog_id": str(row.get("dialog_id") or ""),
        "turn": _int_or_zero(row.get("turn")),
        "route": str(row.get("current_route") or row.get("route") or ""),
        "requested_action": str(frame.get("requested_action") or row.get("requested_action") or ""),
        "frame_risk_class": str(frame.get("risk_class") or row.get("frame_risk_class") or ""),
        "frame_answerability": str(frame.get("answerability") or row.get("frame_answerability") or ""),
        "frame_must_handoff": frame.get("must_handoff") if "must_handoff" in frame else row.get("frame_must_handoff"),
        "frame_confidence": frame.get("confidence") if "confidence" in frame else row.get("frame_confidence"),
        "reasons": list(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    }


def _item_from_mapping(row: Mapping[str, Any], workstream: str, reasons: Sequence[str]) -> dict[str, Any]:
    frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
    return {
        "workstream": workstream,
        "active_allowed": False,
        "active_block_reason": _active_block_reason(workstream),
        "calibration_target": _calibration_target(workstream, row),
        "recommended_review_question": _review_question(workstream),
        "dialog_id": str(row.get("dialog_id") or ""),
        "turn": _int_or_zero(row.get("turn")),
        "route": str(row.get("route") or ""),
        "requested_action": str(row.get("requested_action") or frame.get("requested_action") or ""),
        "frame_risk_class": str(row.get("frame_risk_class") or frame.get("risk_class") or ""),
        "frame_answerability": str(row.get("frame_answerability") or frame.get("answerability") or ""),
        "frame_must_handoff": row.get("frame_must_handoff") if row.get("frame_must_handoff") is not None else frame.get("must_handoff"),
        "frame_confidence": row.get("frame_confidence") if row.get("frame_confidence") is not None else frame.get("confidence"),
        "reasons": list(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    }


def _active_block_reason(workstream: str) -> str:
    reasons = {
        "semanticframe_existence_vs_availability": "frame_confuses_safe_reference_with_live_availability_or_enrollment",
        "semanticframe_safe_reference_missing_facts": "frame_or_prompt_marks_safe_reference_as_missing_facts",
        "semanticframe_low_confidence": "semantic_frame_confidence_below_shadow_threshold",
        "retrieval_delivery_runtime_missing_exact_proof": "runtime_retrieval_did_not_deliver_exact_client_safe_fact",
        "conversation_plan_scope_missing": "conversation_plan_does_not_carry_product_scope_or_required_fact_keys",
        "policy_manager_only_exact_proof": "manager_only_route_requires_separate_owner_policy_decision",
        "policy_context_update_exact_proof": "context_update_route_requires_separate_owner_policy_decision",
        "danger_adjacent_do_not_lower": "danger_p0_money_or_fabrication_adjacent",
        "already_self_no_active_leverage": "current_route_already_self_no_route_leverage",
        "measurement_review_unclear": "measurement_or_gold_review_required",
    }
    return reasons.get(workstream, "no_active_authorization")


def _calibration_target(workstream: str, row: Mapping[str, Any]) -> list[str]:
    if workstream == "semanticframe_existence_vs_availability":
        return ["risk_class", "answerability", "requested_action", "must_handoff"]
    if workstream == "semanticframe_safe_reference_missing_facts":
        return ["risk_class", "answerability", "must_handoff"]
    if workstream == "semanticframe_low_confidence":
        return ["confidence"]
    if workstream == "retrieval_delivery_runtime_missing_exact_proof":
        return ["retrieval_trace", "required_fact_keys", "product_scope"]
    if workstream == "conversation_plan_scope_missing":
        return ["conversation_intent_plan", "answer_contract", "required_fact_keys"]
    if workstream.startswith("policy_"):
        return ["DecisionPolicy", "route_policy"]
    if workstream == "danger_adjacent_do_not_lower":
        return ["manual_safety_review"]
    if workstream == "measurement_review_unclear":
        field_results = row.get("field_results") if isinstance(row.get("field_results"), Mapping) else {}
        wrong = [str(field) for field, result in field_results.items() if result == "wrong"]
        return wrong or ["gold_label", "measurement"]
    return []


def _review_question(workstream: str) -> str:
    questions = {
        "semanticframe_existence_vs_availability": "Is the user asking for a stable existence/format fact, not live seats, booking, payment, or enrollment?",
        "semanticframe_safe_reference_missing_facts": "Can the frame distinguish missing live facts from enough stable facts for a safe reference answer?",
        "semanticframe_low_confidence": "Why is a safe/self reference below the confidence threshold on the production stack?",
        "retrieval_delivery_runtime_missing_exact_proof": "Why did runtime retrieval miss the exact KB fact found by the offline verifier?",
        "conversation_plan_scope_missing": "Where should product scope and required fact keys be carried before route policy?",
        "policy_manager_only_exact_proof": "Should a future policy ever allow manager_only to become self-answer, and under what owner-approved limits?",
        "policy_context_update_exact_proof": "Should context_update be eligible for self-answer only after explicit user question detection is improved?",
        "danger_adjacent_do_not_lower": "Keep out of active candidates unless a human review proves the neighboring risk is irrelevant.",
        "already_self_no_active_leverage": "No action for autonomy; use only to calibrate frame quality.",
        "measurement_review_unclear": "Is this a gold-label issue, a frame issue, or a runtime route issue?",
    }
    return questions.get(workstream, "Review before any active behavior change.")


def _totals(
    *,
    gold_report: Mapping[str, Any],
    overhandoff: Mapping[str, Any],
    injection: Mapping[str, Any],
    work_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold_summary = gold_report.get("summary") if isinstance(gold_report.get("summary"), Mapping) else {}
    over_totals = overhandoff.get("totals") if isinstance(overhandoff.get("totals"), Mapping) else {}
    injection_totals = injection.get("totals") if isinstance(injection.get("totals"), Mapping) else {}
    return {
        "gold_compared_rows": gold_summary.get("compared_rows", 0),
        "safe_self_rows": gold_summary.get("safe_self_candidates", 0),
        "manual_too_cautious_labels": (gold_summary.get("review_labels") or {}).get("frame_too_cautious", 0)
        if isinstance(gold_summary.get("review_labels"), Mapping)
        else 0,
        "true_frame_too_cautious": gold_summary.get("too_cautious", 0),
        "true_frame_too_confident": gold_summary.get("too_confident", 0),
        "current_safe_over_handoff": over_totals.get("safe_handoff_total", gold_summary.get("current_over_handoff_candidates", 0)),
        "strict_active_candidates_now": injection_totals.get("readiness_strict_f3_draft_candidates", 0),
        "manager_only_exact_proof_rows": injection_totals.get("manager_only_exact_proof_rows", 0),
        "work_items_total": len(work_items),
    }


def _workstreams(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in WORKSTREAMS:
        rows = [item for item in items if item.get("workstream") == name]
        result[name] = {
            "count": len(rows),
            "by_requested_action": dict(Counter(str(row.get("requested_action") or "") for row in rows)),
            "examples": rows[:50],
        }
    return result


def _field_error_breakdowns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    for row in rows:
        field_results = row.get("field_results") if isinstance(row.get("field_results"), Mapping) else {}
        for field, result in field_results.items():
            if result == "wrong":
                errors[str(field)] += 1
    return {
        "wrong_fields": dict(errors),
        "too_cautious_by_action": dict(
            Counter(str((row.get("frame") or {}).get("requested_action") or "") for row in rows if _true_frame_too_cautious(row))
        ),
    }


def _acceptance(
    work_items: Sequence[Mapping[str, Any]],
    *,
    gold_report: Mapping[str, Any],
    overhandoff: Mapping[str, Any],
    injection: Mapping[str, Any],
) -> dict[str, Any]:
    notes = [
        "Active autonomy remains NO-GO: this is a calibration queue, not a behavior change.",
        "Existence/format vs live availability must be fixed in SemanticFrame/policy and validated in shadow first.",
        "manager_only/context_update rows need an explicit policy decision before any active demotion discussion.",
    ]
    gold_summary = gold_report.get("summary") if isinstance(gold_report.get("summary"), Mapping) else {}
    if gold_summary.get("too_confident", 0):
        notes.append("Frame has too-confident rows; do not activate any demotion until reviewed.")
    injection_totals = injection.get("totals") if isinstance(injection.get("totals"), Mapping) else {}
    if injection_totals.get("evidence_only_sufficient_rows", 0) == 0:
        notes.append("Fresh exact evidence alone is insufficient on current runtime telemetry.")
    over_totals = overhandoff.get("totals") if isinstance(overhandoff.get("totals"), Mapping) else {}
    if over_totals.get("draft_candidates_for_future_active", 0) == 0:
        notes.append("No draft_for_manager route-only active candidate is ready.")
    return {"status": "pass_report_only", "active_readiness": "no_go", "notes": notes}


def _safe_self(row: Mapping[str, Any]) -> bool:
    expected = row.get("expected") if isinstance(row.get("expected"), Mapping) else {}
    return (
        expected.get("must_handoff") is False
        and str(expected.get("risk_class") or "").casefold() == "safe"
        and str(expected.get("answerability") or "").casefold() == "answer_self"
    )


def _true_frame_too_cautious(row: Mapping[str, Any]) -> bool:
    expected = row.get("expected") if isinstance(row.get("expected"), Mapping) else {}
    frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
    return expected.get("must_handoff") is False and frame.get("must_handoff") is True


def _true_frame_too_confident(row: Mapping[str, Any]) -> bool:
    expected = row.get("expected") if isinstance(row.get("expected"), Mapping) else {}
    frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
    return expected.get("must_handoff") is True and frame.get("must_handoff") is False


def _fields_all_correct(row: Mapping[str, Any]) -> bool:
    field_results = row.get("field_results") if isinstance(row.get("field_results"), Mapping) else {}
    return bool(field_results) and all(value in {"correct", "not_labeled"} for value in field_results.values())


def _existence_or_format_notes(value: str) -> bool:
    return any(
        marker in value
        for marker in (
            "existence",
            "format",
            "course",
            "camp",
            "grade",
            "class",
            "существ",
            "формат",
            "курс",
            "лагер",
            "класс",
        )
    )


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return date.min


def _source_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
