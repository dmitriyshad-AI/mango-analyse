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
FACT_ASSERTION_MARKERS = (
    "safe reference",
    "existence",
    "format",
    "course",
    "camp",
    "grade",
    "class",
    "age suitability",
    "platform",
    "price",
    "schedule",
    "address",
    "trial",
    "direction suitability",
    "существ",
    "формат",
    "курс",
    "лагер",
    "класс",
    "возраст",
    "платформ",
    "цена",
    "стоим",
    "распис",
    "адрес",
    "пробн",
)
FACTLESS_ACK_STATUS_MARKERS = (
    "ack",
    "ack/status",
    "harmless status",
    "thanks",
    "thank",
    "pause",
    "no fact",
    "без фак",
    "спасибо",
    "понятно",
)
DANGER_ADJACENT_MARKERS = (
    "p0",
    "payment",
    "paid",
    "refund",
    "dispute",
    "legal",
    "complaint",
    "fabrication",
    "оплат",
    "возврат",
    "жалоб",
    "договор",
)
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
        "real_lever_analysis": _real_lever_analysis(gold_rows),
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
        "## Real Lever Analysis",
        "",
    ]
    real_lever = report.get("real_lever_analysis") if isinstance(report.get("real_lever_analysis"), Mapping) else {}
    real_totals = real_lever.get("totals") if isinstance(real_lever.get("totals"), Mapping) else {}
    lines.extend(
        [
            f"- True too-cautious rows: `{real_totals.get('too_cautious_total', 0)}`",
            f"- Current handoff among too-cautious: `{real_totals.get('current_handoff_total', 0)}`",
            f"- Fact assertion required: `{real_totals.get('fact_assertion_required', 0)}`",
            f"- Factless ack/status: `{real_totals.get('factless_ack_status', 0)}`",
            f"- Danger-adjacent: `{real_totals.get('danger_adjacent', 0)}`",
            f"- Clean route-only discussion rows: `{real_totals.get('clean_route_only_discussion', 0)}`",
            f"- Stable existence misread as check_availability: `{real_totals.get('stable_existence_as_check_availability', 0)}`",
            f"- Stable existence misread as enroll: `{real_totals.get('stable_existence_as_enroll', 0)}`",
            f"- True live availability negative controls: `{real_totals.get('true_live_availability_negative_controls', 0)}`",
            f"- True enroll/booking negative controls: `{real_totals.get('true_enroll_booking_negative_controls', 0)}`",
            "",
            "### Too-cautious by frame requested_action",
            "",
        ]
    )
    by_action = real_lever.get("by_frame_requested_action") if isinstance(real_lever.get("by_frame_requested_action"), Mapping) else {}
    for action, count in sorted(by_action.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{action or 'missing'}`: `{count}`")
    lines.extend(
        [
            "",
            "### Too-cautious classes",
            "",
        ]
    )
    by_class = real_lever.get("by_lever_class") if isinstance(real_lever.get("by_lever_class"), Mapping) else {}
    for name, count in sorted(by_class.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{name}`: `{count}`")
    scope_confusion = real_lever.get("scope_confusion") if isinstance(real_lever.get("scope_confusion"), Mapping) else {}
    if scope_confusion:
        lines.extend(["", "### Scope confusion", ""])
        lines.append(f"- total: `{scope_confusion.get('count', 0)}`")
        by_scope_action = scope_confusion.get("by_requested_action") if isinstance(scope_confusion.get("by_requested_action"), Mapping) else {}
        for action, count in sorted(by_scope_action.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{action or 'missing'}`: `{count}`")
    negative_controls = real_lever.get("negative_controls") if isinstance(real_lever.get("negative_controls"), list) else []
    if negative_controls:
        lines.extend(["", "### Negative controls", ""])
        lines.append(f"- rows: `{len(negative_controls)}`")
        for item in negative_controls[:8]:
            lines.append(
                f"- `{item.get('dialog_id')}#{item.get('turn')}` "
                f"expected=`{item.get('expected_action')}` frame_scope=`{item.get('frame_scope')}` "
                f"frame_action=`{item.get('frame_requested_action')}`"
            )
    examples = real_lever.get("examples") if isinstance(real_lever.get("examples"), list) else []
    if examples:
        lines.extend(["", "### Real-lever examples", ""])
        for item in examples[:12]:
            lines.append(
                f"- `{item.get('dialog_id')}#{item.get('turn')}` "
                f"route=`{item.get('route')}` action=`{item.get('requested_action')}` "
                f"class=`{item.get('lever_class')}` confidence=`{item.get('frame_confidence')}`"
            )
            if item.get("active_blockers"):
                lines.append(f"  - blockers: `{', '.join(item.get('active_blockers') or [])}`")
            if item.get("review_question"):
                lines.append(f"  - review: {item.get('review_question')}")
    lines.extend(
        [
            "",
        "## Workstreams",
        "",
        ]
    )
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


def _real_lever_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    too_cautious = [row for row in rows if _true_frame_too_cautious(row) and _safe_self(row)]
    classified = [_real_lever_row(row) for row in too_cautious]
    negative_controls = _negative_control_rows(rows)
    return {
        "totals": {
            "too_cautious_total": len(classified),
            "current_handoff_total": sum(1 for row in classified if row.get("current_handoff")),
            "current_draft_for_manager": sum(1 for row in classified if row.get("route") == "draft_for_manager"),
            "current_manager_only": sum(1 for row in classified if row.get("route") == "manager_only"),
            "already_self_or_no_route_leverage": sum(1 for row in classified if row.get("already_self_or_no_route_leverage")),
            "fact_assertion_required": sum(1 for row in classified if row.get("requires_fact_assertion")),
            "factless_ack_status": sum(1 for row in classified if row.get("factless_ack_status")),
            "danger_adjacent": sum(1 for row in classified if row.get("danger_adjacent")),
            "clean_route_only_discussion": sum(1 for row in classified if row.get("clean_route_only_discussion")),
            "stable_existence_as_check_availability": sum(
                1
                for row in classified
                if row.get("user_scope") == "stable_existence_format"
                and row.get("frame_scope") == "live_availability_or_enroll"
                and row.get("requested_action") == "check_availability"
            ),
            "stable_existence_as_enroll": sum(
                1
                for row in classified
                if row.get("user_scope") == "stable_existence_format"
                and row.get("frame_scope") == "live_availability_or_enroll"
                and row.get("requested_action") == "enroll"
            ),
            "true_live_availability_negative_controls": sum(
                1 for row in negative_controls if row.get("expected_scope") == "live_availability_or_enroll"
            ),
            "true_enroll_booking_negative_controls": sum(
                1 for row in negative_controls if row.get("expected_action") in {"enroll", "book", "reserve"}
            ),
        },
        "by_frame_requested_action": dict(Counter(str(row.get("requested_action") or "") for row in classified)),
        "by_frame_risk_class": dict(Counter(str(row.get("frame_risk_class") or "") for row in classified)),
        "by_frame_answerability": dict(Counter(str(row.get("frame_answerability") or "") for row in classified)),
        "by_route": dict(Counter(str(row.get("route") or "") for row in classified)),
        "by_lever_class": dict(Counter(str(row.get("lever_class") or "") for row in classified)),
        "scope_confusion": _scope_confusion_summary(classified),
        "negative_controls": negative_controls[:50],
        "examples": classified[:50],
        "notes": [
            "This is diagnostic only: no route/text change is authorized by this report.",
            "Fact-assertion rows need a verified client-safe fact path before any self-answer policy.",
            "Clean route-only discussion rows, if any, still require Claude/Dmitry review before active work.",
        ],
    }


def _real_lever_row(row: Mapping[str, Any]) -> dict[str, Any]:
    frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
    route = str(row.get("current_route") or row.get("route") or "")
    notes = str(row.get("notes") or "").casefold()
    dialog_id = str(row.get("dialog_id") or "")
    requested_action = str(frame.get("requested_action") or "")
    risk_class = str(frame.get("risk_class") or "")
    answerability = str(frame.get("answerability") or "")
    current_handoff = route in HANDOFF_ROUTES
    user_scope = _user_scope(row)
    frame_scope = _frame_scope(frame)
    fact_assertion = _contains_any(notes, FACT_ASSERTION_MARKERS)
    factless_ack = _contains_any(notes, FACTLESS_ACK_STATUS_MARKERS) and not fact_assertion
    danger = _contains_any(f"{dialog_id} {notes}", DANGER_ADJACENT_MARKERS)
    operational_frame = requested_action.casefold() in OPERATIONAL_ACTIONS or risk_class.casefold() in FRAME_MANAGER_RISKS
    already_self = route not in HANDOFF_ROUTES
    blockers: list[str] = []
    if already_self:
        blockers.append("already_self_or_no_route_leverage")
    if route == "manager_only":
        blockers.append("manager_only_policy")
    if fact_assertion:
        blockers.append("requires_verified_fact_assertion")
    if danger:
        blockers.append("danger_adjacent")
    if operational_frame:
        blockers.append("frame_marks_operational_or_manager_risk")
    if requested_action.casefold() not in SAFE_ACTIONS:
        blockers.append("requested_action_not_safe_reference")
    clean_route_only = current_handoff and route != "manager_only" and factless_ack and not danger and not operational_frame
    if clean_route_only:
        lever_class = "clean_factless_ack_status_discussion"
    elif danger:
        lever_class = "danger_adjacent_do_not_lower"
    elif fact_assertion:
        lever_class = "fact_assertion_required"
    elif already_self:
        lever_class = "already_self_no_route_leverage"
    elif route == "manager_only":
        lever_class = "manager_only_policy_required"
    else:
        lever_class = "measurement_review_required"
    return {
        "dialog_id": dialog_id,
        "turn": _int_or_zero(row.get("turn")),
        "route": route,
        "requested_action": requested_action,
        "frame_risk_class": risk_class,
        "frame_answerability": answerability,
        "frame_must_handoff": frame.get("must_handoff"),
        "frame_confidence": frame.get("confidence"),
        "requires_fact_assertion": fact_assertion,
        "factless_ack_status": factless_ack,
        "user_scope": user_scope,
        "frame_scope": frame_scope,
        "scope_confusion": user_scope == "stable_existence_format" and frame_scope == "live_availability_or_enroll",
        "stable_existence_format": user_scope == "stable_existence_format",
        "live_availability_or_enroll": frame_scope == "live_availability_or_enroll",
        "danger_adjacent": danger,
        "current_handoff": current_handoff,
        "already_self_or_no_route_leverage": already_self,
        "clean_route_only_discussion": clean_route_only,
        "lever_class": lever_class,
        "active_blockers": blockers,
        "review_question": _real_lever_review_question(lever_class),
        "notes": str(row.get("notes") or ""),
    }


def _negative_control_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected") if isinstance(row.get("expected"), Mapping) else {}
        expected_action = str(expected.get("requested_action") or "").casefold()
        if expected_action not in OPERATIONAL_ACTIONS:
            continue
        frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
        result.append(
            {
                "dialog_id": str(row.get("dialog_id") or ""),
                "turn": _int_or_zero(row.get("turn")),
                "route": str(row.get("current_route") or row.get("route") or ""),
                "expected_action": expected_action,
                "expected_scope": "live_availability_or_enroll",
                "frame_scope": _frame_scope(frame),
                "frame_requested_action": str(frame.get("requested_action") or ""),
                "frame_must_handoff": frame.get("must_handoff"),
                "frame_confidence": frame.get("confidence"),
                "active_block_reason": "negative_control_true_live_or_operational_request",
            }
        )
    return result


def _scope_confusion_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confused = [row for row in rows if row.get("scope_confusion")]
    return {
        "count": len(confused),
        "by_requested_action": dict(Counter(str(row.get("requested_action") or "") for row in confused)),
        "by_route": dict(Counter(str(row.get("route") or "") for row in confused)),
        "examples": confused[:20],
    }


def _user_scope(row: Mapping[str, Any]) -> str:
    notes = str(row.get("notes") or "").casefold()
    expected = row.get("expected") if isinstance(row.get("expected"), Mapping) else {}
    expected_action = str(expected.get("requested_action") or "").casefold()
    if expected_action in OPERATIONAL_ACTIONS:
        return "live_availability_or_enroll"
    if _contains_any(notes, FACT_ASSERTION_MARKERS):
        return "stable_existence_format"
    if _contains_any(notes, FACTLESS_ACK_STATUS_MARKERS):
        return "factless_ack_status"
    return "other_safe_reference"


def _frame_scope(frame: Mapping[str, Any]) -> str:
    action = str(frame.get("requested_action") or "").casefold()
    risk = str(frame.get("risk_class") or "").casefold()
    answerability = str(frame.get("answerability") or "").casefold()
    if action in OPERATIONAL_ACTIONS:
        return "live_availability_or_enroll"
    if risk in FRAME_MANAGER_RISKS or answerability == "manager_only":
        return "manager_or_missing"
    if action in SAFE_ACTIONS:
        return "stable_reference_or_ack"
    return "unknown"


def _real_lever_review_question(lever_class: str) -> str:
    questions = {
        "clean_factless_ack_status_discussion": "Is this truly a factless acknowledgement/status turn that can be route-only demoted?",
        "danger_adjacent_do_not_lower": "Keep out of active candidates because the dialog is close to P0/money/fabrication.",
        "fact_assertion_required": "What exact fresh client-safe fact would justify a self-answer?",
        "already_self_no_route_leverage": "No autonomy gain: runtime already answered self.",
        "manager_only_policy_required": "Would owner policy ever allow this manager_only row to become self-answer?",
        "measurement_review_required": "Classify whether this is measurement noise, frame calibration, or policy work.",
    }
    return questions.get(lever_class, "Review before any active behavior change.")


def _contains_any(value: str, markers: Sequence[str]) -> bool:
    value_cf = value.casefold()
    return any(marker.casefold() in value_cf for marker in markers)


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
