#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from scripts.report_adr003_fact_gated_self_answer_readiness import build_report as build_fact_gated_report
from scripts.report_adr003_frame_gold_calibration import build_report as build_gold_report
from scripts.report_adr003_manager_only_exact_proof_root_cause import build_report as build_root_cause_report
from scripts.report_adr003_overhandoff_levers import build_report as build_overhandoff_report


SCHEMA_VERSION = "adr003_frame_calibration_queue_v2_2026_07_02"
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
    "semanticframe_proof_reconciliation_candidate",
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
    parser.add_argument("--bot-template-registry", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    args = parser.parse_args(argv)

    report = build_report(
        transcripts=args.transcripts,
        gold=args.gold,
        kb_snapshot=args.kb_snapshot,
        bot_template_registry=args.bot_template_registry,
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
    bot_template_registry: Path | None = None,
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
    fact_gated = build_fact_gated_report(
        transcripts=transcripts,
        gold=gold,
        kb_snapshot=kb_snapshot,
        confidence_threshold=confidence_threshold,
    )
    turns_by_key = _load_turns_by_turn(transcripts)
    proof_reconciliation = _load_proof_reconciliation_by_turn(transcripts)
    fact_index = _load_fact_index(kb_snapshot)
    template_registry_path = bot_template_registry or _default_bot_template_registry_path(kb_snapshot)
    template_index = _load_template_index(template_registry_path)
    proof_text_readiness = _proof_text_readiness_summary(
        proof_reconciliation=proof_reconciliation,
        turns_by_key=turns_by_key,
        fact_index=fact_index,
        template_index=template_index,
        as_of_date=as_of_date or date.today(),
    )

    gold_rows = [row for row in gold_report.get("rows") or [] if isinstance(row, Mapping)]
    work_items = _build_work_items(
        gold_rows=gold_rows,
        overhandoff=overhandoff,
        root_cause=root_cause,
        injection=injection,
        proof_reconciliation=proof_reconciliation,
        proof_text_readiness=proof_text_readiness.get("by_turn", {})
        if isinstance(proof_text_readiness.get("by_turn"), Mapping)
        else {},
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
            "bot_template_registry": str(template_registry_path) if template_registry_path else "",
            "confidence_threshold": confidence_threshold,
            "as_of_date": (as_of_date or date.today()).isoformat(),
        },
        "totals": _totals(
            gold_report=gold_report,
            overhandoff=overhandoff,
            injection=injection,
            fact_gated=fact_gated,
            work_items=work_items,
            proof_reconciliation=proof_reconciliation,
            proof_text_readiness=proof_text_readiness,
        ),
        "workstreams": _workstreams(work_items),
        "field_error_breakdowns": _field_error_breakdowns(gold_rows),
        "real_lever_analysis": _real_lever_analysis(
            gold_rows,
            proof_reconciliation=proof_reconciliation,
            proof_text_readiness=proof_text_readiness.get("by_turn", {})
            if isinstance(proof_text_readiness.get("by_turn"), Mapping)
            else {},
        ),
        "proof_reconciliation": _proof_reconciliation_summary(proof_reconciliation),
        "proof_reconciliation_text_readiness": proof_text_readiness,
        "source_report_summaries": {
            "gold_calibration": gold_report.get("summary") if isinstance(gold_report.get("summary"), Mapping) else {},
            "overhandoff": overhandoff.get("totals") if isinstance(overhandoff.get("totals"), Mapping) else {},
            "fact_gated_self_answer_readiness": fact_gated.get("totals")
            if isinstance(fact_gated.get("totals"), Mapping)
            else {},
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
    proof_reconciliation = report.get("proof_reconciliation") if isinstance(report.get("proof_reconciliation"), Mapping) else {}
    proof_text_readiness = (
        report.get("proof_reconciliation_text_readiness")
        if isinstance(report.get("proof_reconciliation_text_readiness"), Mapping)
        else {}
    )
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
        f"- Fact-gated strict draft candidates: `{totals.get('fact_gated_strict_f3_draft_candidates', 0)}`",
        f"- Fact-gated manager-only exact-proof rows: `{totals.get('fact_gated_manager_only_exact_proof_needs_policy', 0)}`",
        f"- Fact-gated already-self exact-proof rows: `{totals.get('fact_gated_already_self_exact_proof', 0)}`",
        f"- Fact-gated blocked no-proof rows: `{totals.get('fact_gated_blocked_no_exact_proof', 0)}`",
        f"- Manager-only exact-proof rows: `{totals.get('manager_only_exact_proof_rows', 0)}`",
        f"- Proof reconciliation would-fix-frame rows: `{totals.get('proof_reconciliation_would_reconcile', 0)}`",
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
            f"- Proof reconciliation would-fix-frame rows: `{real_totals.get('proof_reconciliation_would_reconcile', 0)}`",
            f"- Proof reconciliation send-as-is review candidates: `{real_totals.get('proof_reconciliation_send_as_is_review_candidates', 0)}`",
            f"- Proof reconciliation text blocked: `{real_totals.get('proof_reconciliation_text_blocked', 0)}`",
            f"- Proof source fact text ready: `{real_totals.get('proof_text_source_fact_ready', 0)}`",
            f"- Proof source fact text present: `{real_totals.get('proof_text_source_fact_present', 0)}`",
            f"- Proof source fact PII signal: `{real_totals.get('proof_text_source_fact_pii_signal', 0)}`",
            f"- Proof template registry found: `{real_totals.get('proof_text_template_registry_found', 0)}`",
            f"- Proof direct quote forbidden: `{real_totals.get('proof_text_direct_quote_forbidden', 0)}`",
            f"- Proof shadow renderer candidates: `{real_totals.get('proof_text_shadow_renderer_candidates', 0)}`",
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
    current_queue = (
        real_lever.get("current_handoff_queue") if isinstance(real_lever.get("current_handoff_queue"), Mapping) else {}
    )
    if current_queue:
        lines.extend(["", "### Current Handoff Queue", ""])
        lines.append(f"- total: `{current_queue.get('total', 0)}`")
        lines.append(f"- fact assertion required: `{current_queue.get('fact_assertion_required', 0)}`")
        lines.append(f"- factless ack/status: `{current_queue.get('factless_ack_status', 0)}`")
        lines.append(f"- clean route-only discussion: `{current_queue.get('clean_route_only_discussion', 0)}`")
        lines.append(f"- proof reconciliation would fix frame: `{current_queue.get('proof_reconciliation_would_reconcile', 0)}`")
        lines.append(f"- shadow renderer candidates: `{current_queue.get('shadow_text_renderer_candidates', 0)}`")
        by_next = (
            current_queue.get("by_next_autonomy_workstream")
            if isinstance(current_queue.get("by_next_autonomy_workstream"), Mapping)
            else {}
        )
        if by_next:
            lines.extend(["", "#### Current handoff by next workstream", ""])
            for name, count in sorted(by_next.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{name or 'missing'}`: `{count}`")
        current_examples = current_queue.get("examples") if isinstance(current_queue.get("examples"), list) else []
        if current_examples:
            lines.extend(["", "#### Current handoff examples", ""])
            for item in current_examples[:10]:
                lines.append(
                    f"- `{item.get('dialog_id')}#{item.get('turn')}` "
                    f"route=`{item.get('route')}` action=`{item.get('requested_action')}` "
                    f"next=`{item.get('next_autonomy_workstream')}`"
                )
                if item.get("active_unlock_reason"):
                    lines.append(f"  - reason: `{item.get('active_unlock_reason')}`")
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
                f"class=`{item.get('lever_class')}` next=`{item.get('next_autonomy_workstream')}` "
                f"confidence=`{item.get('frame_confidence')}`"
            )
            if item.get("active_blockers"):
                lines.append(f"  - blockers: `{', '.join(item.get('active_blockers') or [])}`")
            if item.get("review_question"):
                lines.append(f"  - review: {item.get('review_question')}")
    fact_gated = (
        report.get("source_report_summaries", {}).get("fact_gated_self_answer_readiness")
        if isinstance(report.get("source_report_summaries"), Mapping)
        and isinstance(report.get("source_report_summaries", {}).get("fact_gated_self_answer_readiness"), Mapping)
        else {}
    )
    if fact_gated:
        lines.extend(["", "## Fact-Gated Readiness Summary", ""])
        lines.append(f"- strict draft candidates: `{fact_gated.get('strict_f3_draft_candidates', 0)}`")
        lines.append(f"- manager-only exact-proof rows: `{fact_gated.get('manager_only_exact_proof_needs_policy', 0)}`")
        lines.append(f"- already-self exact-proof rows: `{fact_gated.get('already_self_exact_proof', 0)}`")
        lines.append(f"- blocked no exact proof: `{fact_gated.get('blocked_no_exact_proof', 0)}`")
        lines.append(f"- excluded danger/money/P0: `{fact_gated.get('excluded_danger_money_p0', 0)}`")
    if proof_reconciliation:
        lines.extend(["", "## Proof Reconciliation Shadow", ""])
        lines.append(f"- traced rows: `{proof_reconciliation.get('total', 0)}`")
        lines.append(f"- would reconcile: `{proof_reconciliation.get('would_reconcile', 0)}`")
        lines.append(f"- already aligned: `{proof_reconciliation.get('already_aligned', 0)}`")
        by_reason = proof_reconciliation.get("by_reason") if isinstance(proof_reconciliation.get("by_reason"), Mapping) else {}
        for reason, count in sorted(by_reason.items(), key=lambda item: (-item[1], item[0]))[:12]:
            lines.append(f"- `{reason or 'missing'}`: `{count}`")
    if proof_text_readiness:
        lines.extend(["", "## Proof Reconciliation Text Readiness", ""])
        lines.append(f"- traced would-reconcile rows: `{proof_text_readiness.get('total', 0)}`")
        lines.append(f"- send-as-is review candidates: `{proof_text_readiness.get('send_as_is_review_candidates', 0)}`")
        lines.append(f"- active behavior allowed: `{proof_text_readiness.get('active_behavior_allowed', False)}`")
        lines.append(f"- source fact client_safe_text present: `{proof_text_readiness.get('source_fact_client_safe_text_present', 0)}`")
        lines.append(f"- source fact client_safe_text PII signal: `{proof_text_readiness.get('source_fact_client_safe_text_pii_signal', 0)}`")
        lines.append(f"- bot_template_required: `{proof_text_readiness.get('bot_template_required', 0)}`")
        lines.append(f"- template registry found: `{proof_text_readiness.get('template_registry_found', 0)}`")
        lines.append(f"- direct quote forbidden: `{proof_text_readiness.get('direct_quote_forbidden', 0)}`")
        lines.append(f"- structured_value available: `{proof_text_readiness.get('structured_value_available', 0)}`")
        lines.append(f"- source fact brand index collisions: `{proof_text_readiness.get('source_fact_brand_index_collisions', 0)}`")
        lines.append(f"- shadow renderer candidates: `{proof_text_readiness.get('shadow_text_renderer_candidates', 0)}`")
        by_status = (
            proof_text_readiness.get("by_status") if isinstance(proof_text_readiness.get("by_status"), Mapping) else {}
        )
        if by_status:
            lines.extend(["", "### Text readiness by status", ""])
            for status, count in sorted(by_status.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{status or 'missing'}`: `{count}`")
        by_blocker = (
            proof_text_readiness.get("by_blocker") if isinstance(proof_text_readiness.get("by_blocker"), Mapping) else {}
        )
        if by_blocker:
            lines.extend(["", "### Text readiness blockers", ""])
            for blocker, count in sorted(by_blocker.items(), key=lambda item: (-item[1], item[0]))[:16]:
                lines.append(f"- `{blocker or 'missing'}`: `{count}`")
        source_status = (
            proof_text_readiness.get("source_fact_lookup_by_status")
            if isinstance(proof_text_readiness.get("source_fact_lookup_by_status"), Mapping)
            else {}
        )
        if source_status:
            lines.extend(["", "### Source fact lookup", ""])
            for status, count in sorted(source_status.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{status or 'missing'}`: `{count}`")
        candidate_status = (
            proof_text_readiness.get("text_candidate_readiness_by_status")
            if isinstance(proof_text_readiness.get("text_candidate_readiness_by_status"), Mapping)
            else {}
        )
        if candidate_status:
            lines.extend(["", "### Source text candidate readiness", ""])
            for status, count in sorted(candidate_status.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{status or 'missing'}`: `{count}`")
        policy_status = (
            proof_text_readiness.get("text_policy_readiness_by_status")
            if isinstance(proof_text_readiness.get("text_policy_readiness_by_status"), Mapping)
            else {}
        )
        if policy_status:
            lines.extend(["", "### Text policy readiness", ""])
            for status, count in sorted(policy_status.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{status or 'missing'}`: `{count}`")
        template_status = (
            proof_text_readiness.get("template_registry_by_status")
            if isinstance(proof_text_readiness.get("template_registry_by_status"), Mapping)
            else {}
        )
        if template_status:
            lines.extend(["", "### Template registry", ""])
            for status, count in sorted(template_status.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{status or 'missing'}`: `{count}`")
        renderer_status = (
            proof_text_readiness.get("shadow_text_renderer_by_status")
            if isinstance(proof_text_readiness.get("shadow_text_renderer_by_status"), Mapping)
            else {}
        )
        if renderer_status:
            lines.extend(["", "### Shadow text renderer", ""])
            for status, count in sorted(renderer_status.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{status or 'missing'}`: `{count}`")
        alignment_status = (
            proof_text_readiness.get("source_alignment_by_status")
            if isinstance(proof_text_readiness.get("source_alignment_by_status"), Mapping)
            else {}
        )
        if alignment_status:
            lines.extend(["", "### Source fact alignment", ""])
            for status, count in sorted(alignment_status.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- `{status or 'missing'}`: `{count}`")
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
    proof_reconciliation: Mapping[str, Mapping[str, Any]],
    proof_text_readiness: Mapping[str, Mapping[str, Any]],
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
        turn_key = _turn_key(row.get("dialog_id"), row.get("turn"))
        reconciliation = proof_reconciliation.get(turn_key)
        if isinstance(reconciliation, Mapping) and reconciliation.get("status") == "would_reconcile_to_safe_reference":
            text_readiness = proof_text_readiness.get(turn_key)
            reasons = ["fresh_exact_proof_would_reconcile_missing_facts_frame"]
            if isinstance(text_readiness, Mapping):
                reasons.append(f"text_readiness:{text_readiness.get('status') or 'unknown'}")
                reasons.extend(f"text_blocker:{blocker}" for blocker in text_readiness.get("blockers") or [])
            items.append(
                _item(
                    row,
                    "semanticframe_proof_reconciliation_candidate",
                    reasons,
                )
            )

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
        "semanticframe_proof_reconciliation_candidate": "fresh_exact_proof_would_reconcile_missing_facts_frame_but_active_not_authorized",
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
    if workstream == "semanticframe_proof_reconciliation_candidate":
        return ["proof_reconciliation_shadow", "frame_contract", "text_readiness_policy"]
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
        "semanticframe_proof_reconciliation_candidate": "Does fresh exact proof justify a future frame correction, and is the existing text safe enough without regeneration?",
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
    fact_gated: Mapping[str, Any],
    work_items: Sequence[Mapping[str, Any]],
    proof_reconciliation: Mapping[str, Mapping[str, Any]],
    proof_text_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    gold_summary = gold_report.get("summary") if isinstance(gold_report.get("summary"), Mapping) else {}
    over_totals = overhandoff.get("totals") if isinstance(overhandoff.get("totals"), Mapping) else {}
    injection_totals = injection.get("totals") if isinstance(injection.get("totals"), Mapping) else {}
    fact_gated_totals = fact_gated.get("totals") if isinstance(fact_gated.get("totals"), Mapping) else {}
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
        "fact_gated_strict_f3_draft_candidates": fact_gated_totals.get("strict_f3_draft_candidates", 0),
        "fact_gated_manager_only_exact_proof_needs_policy": fact_gated_totals.get(
            "manager_only_exact_proof_needs_policy", 0
        ),
        "fact_gated_already_self_exact_proof": fact_gated_totals.get("already_self_exact_proof", 0),
        "fact_gated_blocked_no_exact_proof": fact_gated_totals.get("blocked_no_exact_proof", 0),
        "fact_gated_excluded_danger_money_p0": fact_gated_totals.get("excluded_danger_money_p0", 0),
        "fact_gated_current_handoff_rows": fact_gated_totals.get("current_handoff_rows", 0),
        "manager_only_exact_proof_rows": injection_totals.get("manager_only_exact_proof_rows", 0),
        "proof_reconciliation_would_reconcile": sum(
            1 for trace in proof_reconciliation.values() if trace.get("status") == "would_reconcile_to_safe_reference"
        ),
        "proof_reconciliation_text_readiness_rows": proof_text_readiness.get("total", 0),
        "proof_reconciliation_send_as_is_review_candidates": proof_text_readiness.get("send_as_is_review_candidates", 0),
        "proof_text_source_fact_client_safe_text_present": proof_text_readiness.get("source_fact_client_safe_text_present", 0),
        "proof_text_source_fact_pii_signal": proof_text_readiness.get("source_fact_client_safe_text_pii_signal", 0),
        "proof_text_bot_template_required": proof_text_readiness.get("bot_template_required", 0),
        "proof_text_template_registry_found": proof_text_readiness.get("template_registry_found", 0),
        "proof_text_direct_quote_forbidden": proof_text_readiness.get("direct_quote_forbidden", 0),
        "proof_text_shadow_renderer_candidates": proof_text_readiness.get("shadow_text_renderer_candidates", 0),
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


def _turn_key(dialog_id: Any, turn: Any) -> str:
    return f"{str(dialog_id or '')}#{_int_or_zero(turn)}"


def _load_proof_reconciliation_by_turn(transcripts: Path) -> dict[str, Mapping[str, Any]]:
    traces: dict[str, Mapping[str, Any]] = {}
    if not transcripts.exists():
        return traces
    with transcripts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                dialog = json.loads(line)
            except json.JSONDecodeError:
                continue
            dialog_id = str(dialog.get("dialog_id") or "")
            turns = dialog.get("turns") if isinstance(dialog.get("turns"), list) else []
            for turn in turns:
                if not isinstance(turn, Mapping):
                    continue
                trace = turn.get("bot_semantic_frame_proof_reconciliation_shadow")
                if not isinstance(trace, Mapping):
                    direct = turn.get("bot_direct_path") if isinstance(turn.get("bot_direct_path"), Mapping) else {}
                    trace = direct.get("semantic_frame_proof_reconciliation_shadow") if isinstance(direct, Mapping) else {}
                if isinstance(trace, Mapping) and trace:
                    traces[_turn_key(dialog_id, turn.get("turn"))] = trace
    return traces


def _load_turns_by_turn(transcripts: Path) -> dict[str, Mapping[str, Any]]:
    turns_by_key: dict[str, Mapping[str, Any]] = {}
    if not transcripts.exists():
        return turns_by_key
    with transcripts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                dialog = json.loads(line)
            except json.JSONDecodeError:
                continue
            dialog_id = str(dialog.get("dialog_id") or "")
            turns = dialog.get("turns") if isinstance(dialog.get("turns"), list) else []
            for turn in turns:
                if isinstance(turn, Mapping):
                    turns_by_key[_turn_key(dialog_id, turn.get("turn"))] = turn
    return turns_by_key


def _load_fact_index(kb_snapshot: Path) -> dict[str, list[Mapping[str, Any]]]:
    facts: list[Mapping[str, Any]] = []
    if not kb_snapshot.exists():
        return {}
    try:
        if kb_snapshot.suffix == ".jsonl":
            with kb_snapshot.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, Mapping):
                        facts.append(item)
        else:
            payload = json.loads(kb_snapshot.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                raw_facts = payload.get("facts") or payload.get("facts_registry") or payload.get("client_safe_facts")
                if isinstance(raw_facts, list):
                    facts.extend(item for item in raw_facts if isinstance(item, Mapping))
            elif isinstance(payload, list):
                facts.extend(item for item in payload if isinstance(item, Mapping))
    except Exception:
        return {}
    index: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        for key_name in ("fact_key", "source_fact_key", "fact_id", "id", "key"):
            key = str(fact.get(key_name) or "").strip()
            if key:
                index.setdefault(key, []).append(fact)
    return index


def _default_bot_template_registry_path(kb_snapshot: Path) -> Path | None:
    release_dir = kb_snapshot.parent
    candidates = [
        release_dir / "bot_template_registry.json",
        release_dir.parent / f"{release_dir.name}_bot_pack" / "bot_template_registry.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_template_index(path: Path | None) -> dict[str, Mapping[str, Any]]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    templates = payload.get("templates") if isinstance(payload, Mapping) else payload
    if not isinstance(templates, list):
        return {}
    index: dict[str, Mapping[str, Any]] = {}
    for template in templates:
        if not isinstance(template, Mapping):
            continue
        for key_name in ("fact_key", "fact_id"):
            key = str(template.get(key_name) or "").strip()
            if key:
                index.setdefault(key, template)
    return index


def _proof_reconciliation_summary(traces: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(trace.get("status") or "") for trace in traces.values())
    reasons = Counter(str(trace.get("reason") or "") for trace in traces.values())
    would = [
        {
            "turn_key": key,
            "route_before": trace.get("route_before"),
            "reason": trace.get("reason"),
            "exact_fact_keys": list(trace.get("exact_fact_keys") or [])[:5]
            if isinstance(trace.get("exact_fact_keys"), list)
            else [],
            "frame_before": trace.get("frame_before") if isinstance(trace.get("frame_before"), Mapping) else {},
        }
        for key, trace in traces.items()
        if trace.get("status") == "would_reconcile_to_safe_reference"
    ]
    return {
        "total": len(traces),
        "would_reconcile": statuses.get("would_reconcile_to_safe_reference", 0),
        "already_aligned": statuses.get("already_aligned", 0),
        "by_status": dict(statuses),
        "by_reason": dict(reasons),
        "examples": would[:50],
    }


def _proof_text_readiness_summary(
    *,
    proof_reconciliation: Mapping[str, Mapping[str, Any]],
    turns_by_key: Mapping[str, Mapping[str, Any]],
    fact_index: Mapping[str, Sequence[Mapping[str, Any]]],
    template_index: Mapping[str, Mapping[str, Any]],
    as_of_date: date,
) -> dict[str, Any]:
    by_turn: dict[str, Mapping[str, Any]] = {}
    for key, trace in proof_reconciliation.items():
        if trace.get("status") != "would_reconcile_to_safe_reference":
            continue
        by_turn[key] = _proof_text_readiness_for_turn(
            turn_key=key,
            trace=trace,
            turn=turns_by_key.get(key),
            fact_index=fact_index,
            template_index=template_index,
            as_of_date=as_of_date,
        )
    statuses = Counter(str(row.get("status") or "") for row in by_turn.values())
    blockers = Counter(
        str(blocker)
        for row in by_turn.values()
        for blocker in (row.get("blockers") or [])
    )
    fact_statuses = Counter(str(row.get("source_fact_lookup_status") or "") for row in by_turn.values())
    text_statuses = Counter(str(row.get("text_candidate_readiness_status") or "") for row in by_turn.values())
    policy_statuses = Counter(str(row.get("text_policy_readiness_status") or "") for row in by_turn.values())
    template_statuses = Counter(str(row.get("template_registry_status") or "") for row in by_turn.values())
    renderer_statuses = Counter(str(row.get("shadow_text_renderer_status") or "") for row in by_turn.values())
    alignment_statuses = Counter(str(row.get("source_alignment_status") or "") for row in by_turn.values())
    alignment_blockers = Counter(
        str(blocker)
        for row in by_turn.values()
        for blocker in (row.get("source_alignment_blockers") or [])
    )
    candidates = [row for row in by_turn.values() if row.get("send_as_is_review_candidate")]
    renderer_candidates = [row for row in by_turn.values() if row.get("shadow_text_renderer_status") == "candidate_rendered"]
    return {
        "total": len(by_turn),
        "send_as_is_review_candidates": len(candidates),
        "shadow_text_renderer_candidates": len(renderer_candidates),
        "active_behavior_allowed": False,
        "active_readiness": "no_go",
        "by_status": dict(statuses),
        "by_blocker": dict(blockers),
        "source_fact_lookup_by_status": dict(fact_statuses),
        "text_candidate_readiness_by_status": dict(text_statuses),
        "text_policy_readiness_by_status": dict(policy_statuses),
        "template_registry_by_status": dict(template_statuses),
        "shadow_text_renderer_by_status": dict(renderer_statuses),
        "source_alignment_by_status": dict(alignment_statuses),
        "source_alignment_by_blocker": dict(alignment_blockers),
        "source_fact_client_safe_text_present": sum(
            1 for row in by_turn.values() if row.get("source_fact_client_safe_text_present")
        ),
        "source_fact_client_safe_text_pii_signal": sum(
            1 for row in by_turn.values() if row.get("source_fact_client_safe_text_pii_signal")
        ),
        "bot_template_required": sum(1 for row in by_turn.values() if row.get("bot_template_required")),
        "template_registry_found": sum(1 for row in by_turn.values() if row.get("template_registry_status") == "found"),
        "direct_quote_forbidden": sum(1 for row in by_turn.values() if row.get("direct_quote_forbidden")),
        "structured_value_available": sum(1 for row in by_turn.values() if row.get("structured_value_available")),
        "source_fact_brand_index_collisions": sum(
            1 for row in by_turn.values() if row.get("source_fact_brand_index_collision")
        ),
        "by_turn": by_turn,
        "examples": list(by_turn.values())[:50],
        "notes": [
            "Report-only text readiness: no route/text/runtime behavior is changed.",
            "A fresh exact proof is not enough to send text as-is; existing text gates must also be clean.",
            "Fact-text candidate readiness uses only raw client_safe_text from the KB snapshot; fallback fact_text is not sendable proof.",
            "Full client_safe_text is not exported; report stores only length/hash and safety status.",
            "A found template registry entry is not a generated answer; it only proves a future renderer has a policy source.",
            "Shadow text renderer candidates are measurement-only and exported as hash/length, not as sendable client text.",
            "semantic_output_verifier unavailable is treated as a blocker, not as permission.",
        ],
    }


def _proof_text_readiness_for_turn(
    *,
    turn_key: str,
    trace: Mapping[str, Any],
    turn: Mapping[str, Any] | None,
    fact_index: Mapping[str, Sequence[Mapping[str, Any]]],
    template_index: Mapping[str, Mapping[str, Any]],
    as_of_date: date,
) -> dict[str, Any]:
    if not isinstance(turn, Mapping):
        return {
            "turn_key": turn_key,
            "status": "blocked_missing_turn_telemetry",
            "blockers": ["missing_turn_telemetry"],
            "send_as_is_review_candidate": False,
            "active_behavior_allowed": False,
        }
    route = str(turn.get("bot_route") or trace.get("route_before") or "")
    direct_path = turn.get("bot_direct_path") if isinstance(turn.get("bot_direct_path"), Mapping) else {}
    fact_readiness = _source_fact_text_readiness(
        trace,
        turn=turn,
        fact_index=fact_index,
        template_index=template_index,
        as_of_date=as_of_date,
    )
    blockers: list[str] = []
    if route == "manager_only":
        blockers.append("current_route_manager_only")
    elif route != "draft_for_manager":
        blockers.append("no_route_leverage_currently_self_or_other")
    if bool(turn.get("bot_is_manager_deferral")) or bool(direct_path.get("is_manager_deferral")) or bool(
        direct_path.get("deferral_text_in_self")
    ):
        blockers.append("deferral_or_manager_text_signal")
    missing_facts = _listish(turn.get("bot_missing_facts")) or _listish(trace.get("result_missing_facts"))
    if missing_facts:
        blockers.append("missing_facts_present")
    answer_quality_findings = _listish(turn.get("bot_answer_quality_findings"))
    if answer_quality_findings:
        blockers.append("answer_quality_findings_present")
    safety_flags = _listish(turn.get("bot_safety_flags"))
    p0_or_brand_flags = [
        flag
        for flag in safety_flags
        if str(flag).startswith(("p0_", "brand_")) or str(flag) in {"p0_required", "cross_brand_blocked"}
    ]
    if p0_or_brand_flags:
        blockers.append("brand_or_p0_blocker")
    sanitizer = turn.get("bot_output_sanitizer")
    if not isinstance(sanitizer, Mapping):
        metadata = turn.get("bot_metadata") if isinstance(turn.get("bot_metadata"), Mapping) else {}
        sanitizer = metadata.get("output_sanitizer") if isinstance(metadata.get("output_sanitizer"), Mapping) else {}
    if _output_sanitizer_applied(sanitizer, safety_flags=safety_flags):
        blockers.append("output_sanitizer_applied")
    if _forbidden_promises_present(turn, direct_path):
        blockers.append("forbidden_promises_present")
    auth_gate = turn.get("bot_authoritative_output_gate")
    if isinstance(auth_gate, Mapping) and str(auth_gate.get("action") or "pass") not in {"", "pass"}:
        blockers.append("authoritative_gate_not_pass")
    verifier = turn.get("bot_semantic_output_verifier")
    if not isinstance(verifier, Mapping) or not verifier:
        blockers.append("semantic_verifier_unavailable")
    elif _semantic_verifier_blocks(verifier):
        blockers.append("semantic_verifier_finding")
    status = "send_as_is_review_candidate" if not blockers else "blocked_" + blockers[0]
    return {
        "turn_key": turn_key,
        "dialog_id": turn_key.rsplit("#", 1)[0],
        "turn": _int_or_zero(turn.get("turn")),
        "route": route,
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "send_as_is_review_candidate": not blockers,
        "active_behavior_allowed": False,
        "proof_reason": trace.get("reason"),
        "exact_fact_keys": list(trace.get("exact_fact_keys") or [])[:5]
        if isinstance(trace.get("exact_fact_keys"), list)
        else [],
        **fact_readiness,
        "missing_facts_count": len(missing_facts),
        "answer_quality_findings_count": len(answer_quality_findings),
        "semantic_verifier_action": verifier.get("action") if isinstance(verifier, Mapping) else "",
        "authoritative_gate_action": auth_gate.get("action") if isinstance(auth_gate, Mapping) else "",
    }


def _source_fact_text_readiness(
    trace: Mapping[str, Any],
    *,
    turn: Mapping[str, Any],
    fact_index: Mapping[str, Mapping[str, Any]],
    template_index: Mapping[str, Mapping[str, Any]],
    as_of_date: date,
) -> dict[str, Any]:
    fact_key = _source_fact_key(trace)
    active_brand = _turn_active_brand(turn, trace)
    base = {
        "source_fact_key": fact_key,
        "source_fact_lookup_status": "",
        "source_fact_candidate_count": 0,
        "source_fact_candidate_brands": [],
        "source_fact_selected_brand": "",
        "source_fact_brand_index_collision": False,
        "source_fact_client_safe_text_present": False,
        "source_fact_client_safe_text_length": 0,
        "source_fact_client_safe_text_hash": "",
        "source_fact_client_safe_text_pii_signal": False,
        "structured_value_available": False,
        "structured_value_keys": [],
        "structured_value_field_count": 0,
        "bot_template_required": False,
        "template_registry_status": "",
        "template_id": "",
        "template_text_length": 0,
        "template_text_hash": "",
        "direct_quote_forbidden": False,
        "text_candidate_readiness_status": "blocked",
        "text_policy_readiness_status": "blocked",
        "text_candidate_blockers": [],
        "text_policy_blockers": [],
        "source_alignment_status": "",
        "source_alignment_blockers": [],
        "source_alignment_missing_fact_categories": [],
        "source_alignment_fact_categories": [],
        "shadow_text_renderer_status": "blocked",
        "shadow_text_renderer_blockers": [],
        "shadow_text_renderer_source": "",
        "shadow_text_candidate_length": 0,
        "shadow_text_candidate_hash": "",
        "shadow_text_candidate_exported": False,
        "raw_text_exported": False,
        "active_behavior_allowed": False,
    }
    blockers: list[str] = ["shadow_only_text_candidate", "requires_template_or_text_policy"]
    policy_blockers: list[str] = ["shadow_only_text_policy", "requires_text_policy_renderer"]
    if not fact_key:
        return {
            **base,
            "source_fact_lookup_status": "missing_source_fact_key",
            "text_candidate_readiness_status": "blocked_missing_source_fact_key",
            "text_candidate_blockers": [*blockers, "missing_source_fact_key"],
            "text_policy_readiness_status": "blocked_missing_source_fact_key",
            "text_policy_blockers": [*policy_blockers, "missing_source_fact_key"],
        }
    fact_candidates = list(fact_index.get(fact_key) or [])
    fact = _select_fact_candidate(fact_candidates, active_brand=active_brand)
    if not isinstance(fact, Mapping):
        return {
            **base,
            "source_fact_candidate_count": len(fact_candidates),
            "source_fact_candidate_brands": _candidate_brands(fact_candidates),
            "source_fact_lookup_status": "missing",
            "text_candidate_readiness_status": "blocked_missing_source_fact",
            "text_candidate_blockers": [*blockers, "missing_source_fact"],
            "text_policy_readiness_status": "blocked_missing_source_fact",
            "text_policy_blockers": [*policy_blockers, "missing_source_fact"],
        }
    status, fact_blockers = _source_fact_status(fact, active_brand=active_brand, as_of_date=as_of_date)
    raw_text = str(fact.get("client_safe_text") or "").strip()
    structured_value = fact.get("structured_value") if isinstance(fact.get("structured_value"), Mapping) else {}
    structured_keys = sorted(str(key) for key in structured_value.keys())
    pii_signal = _pii_signal(raw_text)
    if pii_signal:
        fact_blockers.append("client_safe_text_pii_signal")
    alignment_status, alignment_blockers, alignment_details = _source_alignment_status(trace, fact=fact)
    if alignment_blockers:
        fact_blockers.extend(alignment_blockers)
    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else ""
    text_ready = status == "found" and not fact_blockers and bool(raw_text)
    template = _template_for_fact(fact, template_index=template_index)
    template_status = _template_status(fact, template)
    template_text = str(template.get("template_text") or "").strip() if isinstance(template, Mapping) else ""
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest() if template_text else ""
    direct_quote_forbidden = bool(raw_text)
    policy_status, policy_specific_blockers = _text_policy_status(
        source_status=status,
        source_blockers=fact_blockers,
        raw_text_present=bool(raw_text),
        template_status=template_status,
    )
    renderer = _shadow_text_renderer_status(
        fact,
        active_brand=active_brand,
        source_status=status,
        source_blockers=fact_blockers,
        template_status=template_status,
    )
    return {
        **base,
        "source_fact_candidate_count": len(fact_candidates),
        "source_fact_candidate_brands": _candidate_brands(fact_candidates),
        "source_fact_selected_brand": str(fact.get("brand") or fact.get("active_brand") or "").strip().casefold(),
        "source_fact_brand_index_collision": len(set(_candidate_brands(fact_candidates))) > 1,
        "source_fact_lookup_status": status,
        "source_fact_client_safe_text_present": bool(raw_text),
        "source_fact_client_safe_text_length": len(raw_text),
        "source_fact_client_safe_text_hash": text_hash,
        "source_fact_client_safe_text_pii_signal": pii_signal,
        "structured_value_available": bool(structured_value),
        "structured_value_keys": structured_keys,
        "structured_value_field_count": len(structured_keys),
        "bot_template_required": bool(fact.get("bot_template_required")),
        "template_registry_status": template_status,
        "template_id": str(template.get("template_id") or "") if isinstance(template, Mapping) else "",
        "template_text_length": len(template_text),
        "template_text_hash": template_hash,
        "direct_quote_forbidden": direct_quote_forbidden,
        "text_candidate_readiness_status": "source_text_ready" if text_ready else f"blocked_{fact_blockers[0] if fact_blockers else status}",
        "text_candidate_blockers": [*blockers, *fact_blockers],
        "text_policy_readiness_status": policy_status,
        "text_policy_blockers": [
            *policy_blockers,
            *policy_specific_blockers,
            *(["direct_quote_forbidden"] if direct_quote_forbidden else []),
        ],
        "source_alignment_status": alignment_status,
        "source_alignment_blockers": alignment_blockers,
        **alignment_details,
        **renderer,
    }


def _select_fact_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    active_brand: str,
) -> Mapping[str, Any]:
    if active_brand:
        for fact in candidates:
            brand = str(fact.get("brand") or fact.get("active_brand") or "").strip().casefold()
            if brand == active_brand:
                return fact
    return candidates[0] if candidates else {}


def _candidate_brands(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    brands = {
        str(fact.get("brand") or fact.get("active_brand") or "").strip().casefold()
        for fact in candidates
        if str(fact.get("brand") or fact.get("active_brand") or "").strip()
    }
    return sorted(brands)


def _source_fact_key(trace: Mapping[str, Any]) -> str:
    key = str(trace.get("source_fact_key") or "").strip()
    if key:
        return key
    exact = trace.get("exact_fact_keys")
    if isinstance(exact, list):
        return next((str(item or "").strip() for item in exact if str(item or "").strip()), "")
    return ""


def _turn_active_brand(turn: Mapping[str, Any], trace: Mapping[str, Any]) -> str:
    for value in (
        turn.get("brand"),
        turn.get("active_brand"),
        turn.get("bot_active_brand"),
        trace.get("active_brand"),
    ):
        brand = str(value or "").strip().casefold()
        if brand in {"foton", "unpk"}:
            return brand
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    product = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    brand = str(product.get("brand") or "").strip().casefold()
    return brand if brand in {"foton", "unpk"} else ""


def _source_fact_status(fact: Mapping[str, Any], *, active_brand: str, as_of_date: date) -> tuple[str, list[str]]:
    blockers: list[str] = []
    brand = str(fact.get("brand") or fact.get("active_brand") or "").strip().casefold()
    if active_brand and brand and brand != active_brand:
        return "wrong_brand", ["wrong_brand"]
    if fact.get("internal_only") is True:
        return "internal_only", ["internal_only"]
    if fact.get("forbidden_for_client") is True:
        return "forbidden_for_client", ["forbidden_for_client"]
    if fact.get("allowed_for_client_answer") is not True:
        return "not_client_safe", ["not_client_safe"]
    if not _valid_until_ok(fact.get("valid_until"), as_of_date=as_of_date):
        return "stale", ["stale"]
    raw_text = str(fact.get("client_safe_text") or "").strip()
    if not raw_text:
        return "empty_client_safe_text", ["empty_client_safe_text"]
    if fact.get("bot_template_required") is True:
        blockers.append("bot_template_required")
    return "found", blockers


def _source_alignment_status(
    trace: Mapping[str, Any],
    *,
    fact: Mapping[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    missing_facts = " ".join(str(item or "") for item in _listish(trace.get("result_missing_facts"))).casefold()
    if not missing_facts:
        return (
            "aligned_no_missing_facts_trace",
            [],
            {
                "source_alignment_missing_fact_categories": [],
                "source_alignment_fact_categories": _fact_categories(fact),
            },
        )
    missing_categories = _missing_fact_categories(missing_facts)
    fact_categories = _fact_categories(fact)
    if not missing_categories:
        return (
            "alignment_review_unclear",
            ["source_axis_review_unclear"],
            {
                "source_alignment_missing_fact_categories": [],
                "source_alignment_fact_categories": fact_categories,
            },
        )
    uncovered = [category for category in missing_categories if category not in fact_categories]
    if uncovered:
        return (
            "blocked_source_axis_mismatch",
            ["source_axis_mismatch"],
            {
                "source_alignment_missing_fact_categories": missing_categories,
                "source_alignment_fact_categories": fact_categories,
                "source_alignment_uncovered_categories": uncovered,
            },
        )
    return (
        "aligned_covers_missing_fact_axis",
        [],
        {
            "source_alignment_missing_fact_categories": missing_categories,
            "source_alignment_fact_categories": fact_categories,
            "source_alignment_uncovered_categories": [],
        },
    )


def _missing_fact_categories(value: str) -> list[str]:
    categories: list[str] = []
    marker_map = (
        ("live_availability", ("наличие мест", "свободн", "места", "актуальные смены", "смены")),
        ("boarding_food", ("прожив", "питан", "дневн", "выездн", "условия")),
        ("medical", ("медсестр", "медицин")),
        ("security", ("охран", "камер", "видеонаблюд")),
        ("payment_access", ("после оплаты", "доступ", "оплат")),
        ("dates_schedule", ("даты", "распис", "время", "старт")),
        ("location_address", ("локац", "адрес", "площадк", "где")),
        ("price_cost", ("стоим", "цена", "прайс")),
        ("class_grade", ("класс", "возраст", "уровень")),
        ("program_direction", ("программа", "направлен", "физмат", "ит-направ", "ит направление")),
    )
    for category, markers in marker_map:
        if any(marker in value for marker in markers):
            categories.append(category)
    return categories


def _fact_categories(fact: Mapping[str, Any]) -> list[str]:
    fact_type = str(fact.get("fact_type") or "").casefold()
    fact_key = str(fact.get("fact_key") or fact.get("source_fact_key") or "").casefold()
    program_kind = str(fact.get("program_kind") or "").casefold()
    product = str(fact.get("product") or "").casefold()
    structured = fact.get("structured_value") if isinstance(fact.get("structured_value"), Mapping) else {}
    structured_keys = {str(key).casefold() for key in structured.keys()}
    text = f"{fact_type} {fact_key} {program_kind} {product} {' '.join(structured_keys)} {fact.get('client_safe_text') or ''}".casefold()
    categories: set[str] = set()
    if {"classes", "classes_raw"} & structured_keys or "classes" in fact_key or "класс" in text:
        categories.add("class_grade")
    if fact_type in {"deadline", "schedule"} or {"number"} & structured_keys or any(
        marker in text for marker in ("распис", "воскресенье", "старт", "10:00", "12:00", "даты")
    ):
        categories.add("dates_schedule")
    if any(marker in text for marker in ("адрес", "красносельск", "площадк", "локац", "где")):
        categories.add("location_address")
    if any(marker in text for marker in ("стоим", "цена", "₽", "руб")):
        categories.add("price_cost")
    if any(marker in text for marker in ("прожив", "интернат", "номере", "дневные")):
        categories.add("boarding_food")
    if any(marker in text for marker in ("медсестр", "медицин")):
        categories.add("medical")
    if any(marker in text for marker in ("охран", "камер", "видеонаблюд")):
        categories.add("security")
    if any(marker in text for marker in ("доступ", "после оплаты")):
        categories.add("payment_access")
    if any(marker in text for marker in ("направлен", "физико", "физмат", "ит-направ", "ит направление", "программа")):
        categories.add("program_direction")
    if any(marker in text for marker in ("наличие мест", "свободн")):
        categories.add("live_availability")
    return sorted(categories)


def _template_for_fact(
    fact: Mapping[str, Any],
    *,
    template_index: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    for key in (fact.get("fact_key"), fact.get("fact_id")):
        template = template_index.get(str(key or "").strip())
        if isinstance(template, Mapping):
            return template
    return {}


def _template_status(fact: Mapping[str, Any], template: Mapping[str, Any]) -> str:
    if fact.get("bot_template_required") is not True:
        return "not_required"
    if not isinstance(template, Mapping) or not template:
        return "missing"
    if not str(template.get("template_id") or "").strip():
        return "missing_template_id"
    if not str(template.get("template_text") or "").strip():
        return "empty_template_text"
    return "found"


def _text_policy_status(
    *,
    source_status: str,
    source_blockers: Sequence[str],
    raw_text_present: bool,
    template_status: str,
) -> tuple[str, list[str]]:
    if source_status != "found":
        return f"blocked_{source_status}", [source_status]
    if source_blockers:
        non_template_blockers = [item for item in source_blockers if item != "bot_template_required"]
        if non_template_blockers:
            return f"blocked_{non_template_blockers[0]}", non_template_blockers
    if not raw_text_present:
        return "blocked_empty_client_safe_text", ["empty_client_safe_text"]
    if template_status == "found":
        return "template_registry_found_requires_renderer", ["requires_template_renderer"]
    if template_status in {"missing", "missing_template_id", "empty_template_text"}:
        return f"blocked_{template_status}_bot_template", [f"{template_status}_bot_template"]
    return "source_text_ready_requires_nonquote_policy", ["requires_nonquote_text_policy"]


def _shadow_text_renderer_status(
    fact: Mapping[str, Any],
    *,
    active_brand: str,
    source_status: str,
    source_blockers: Sequence[str],
    template_status: str,
) -> dict[str, Any]:
    base = {
        "shadow_text_renderer_status": "blocked",
        "shadow_text_renderer_blockers": [],
        "shadow_text_renderer_source": "",
        "shadow_text_candidate_length": 0,
        "shadow_text_candidate_hash": "",
        "shadow_text_candidate_exported": False,
    }
    if source_status != "found":
        return {**base, "shadow_text_renderer_status": f"blocked_{source_status}", "shadow_text_renderer_blockers": [source_status]}
    hard_blockers = [item for item in source_blockers if item != "bot_template_required"]
    if hard_blockers:
        return {
            **base,
            "shadow_text_renderer_status": f"blocked_{hard_blockers[0]}",
            "shadow_text_renderer_blockers": hard_blockers,
        }
    if fact.get("bot_template_required") is True:
        if template_status == "found":
            return {
                **base,
                "shadow_text_renderer_status": "blocked_template_renderer_not_implemented",
                "shadow_text_renderer_blockers": ["requires_template_renderer"],
            }
        return {
            **base,
            "shadow_text_renderer_status": f"blocked_{template_status}_bot_template",
            "shadow_text_renderer_blockers": [f"{template_status}_bot_template"],
        }

    structured_value = fact.get("structured_value") if isinstance(fact.get("structured_value"), Mapping) else {}
    fact_type = str(fact.get("fact_type") or "").strip()
    brand_label = "УНПК" if active_brand == "unpk" else "Фотон" if active_brand == "foton" else ""
    candidate = ""
    source = ""
    if fact_type == "course_parameter":
        classes = _shadow_renderer_clean_atomic_value(structured_value.get("classes_raw") or structured_value.get("classes"))
        if classes and brand_label:
            candidate = f"Для этого направления {brand_label} указан диапазон классов: {classes}."
            source = "structured_value:classes"
    if not candidate:
        reason = "unsupported_structured_value" if structured_value else "missing_structured_value"
        return {**base, "shadow_text_renderer_status": f"blocked_{reason}", "shadow_text_renderer_blockers": [reason]}
    return {
        **base,
        "shadow_text_renderer_status": "candidate_rendered",
        "shadow_text_renderer_blockers": ["shadow_only_renderer", "requires_semantic_review", "requires_runtime_text_policy"],
        "shadow_text_renderer_source": source,
        "shadow_text_candidate_length": len(candidate),
        "shadow_text_candidate_hash": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    }


def _shadow_renderer_clean_atomic_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > 40:
        return ""
    if re.search(r"[\n\r:;.!?]", text):
        return ""
    if _pii_signal(text):
        return ""
    return text


def _valid_until_ok(value: Any, *, as_of_date: date) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    try:
        return date.fromisoformat(raw[:10]) >= as_of_date
    except ValueError:
        return False


def _pii_signal(value: str) -> bool:
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", value, flags=re.IGNORECASE):
        return True
    if re.search(r"(?:\+7|8)[\s()\\-]*\d{3}[\s()\\-]*\d{3}[\s\\-]*\d{2}[\s\\-]*\d{2}", value):
        return True
    if re.search(r"\b(?:amo|entity|contact|lead|deal)[_-]?id\b", value, flags=re.IGNORECASE):
        return True
    return False


def _real_lever_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    proof_reconciliation: Mapping[str, Mapping[str, Any]],
    proof_text_readiness: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    too_cautious = [row for row in rows if _true_frame_too_cautious(row) and _safe_self(row)]
    classified = [
        _real_lever_row(row, proof_reconciliation=proof_reconciliation, proof_text_readiness=proof_text_readiness)
        for row in too_cautious
    ]
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
            "proof_reconciliation_would_reconcile": sum(1 for row in classified if row.get("proof_reconciliation_would_reconcile")),
            "proof_reconciliation_current_handoff": sum(
                1 for row in classified if row.get("proof_reconciliation_would_reconcile") and row.get("current_handoff")
            ),
            "proof_reconciliation_send_as_is_review_candidates": sum(
                1
                for row in classified
                if row.get("proof_reconciliation_would_reconcile") and row.get("send_as_is_review_candidate")
            ),
            "proof_reconciliation_text_blocked": sum(
                1
                for row in classified
                if row.get("proof_reconciliation_would_reconcile")
                and row.get("text_readiness_status")
                and not row.get("send_as_is_review_candidate")
            ),
            "proof_text_source_fact_ready": sum(
                1 for row in classified if row.get("text_candidate_readiness_status") == "source_text_ready"
            ),
            "proof_text_source_fact_present": sum(
                1 for row in classified if row.get("source_fact_client_safe_text_present")
            ),
            "proof_text_source_fact_pii_signal": sum(
                1 for row in classified if row.get("source_fact_client_safe_text_pii_signal")
            ),
            "proof_text_template_registry_found": sum(
                1 for row in classified if row.get("template_registry_status") == "found"
            ),
            "proof_text_direct_quote_forbidden": sum(1 for row in classified if row.get("direct_quote_forbidden")),
            "proof_text_shadow_renderer_candidates": sum(
                1 for row in classified if row.get("shadow_text_renderer_status") == "candidate_rendered"
            ),
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
        "by_proof_reconciliation_status": dict(Counter(str(row.get("proof_reconciliation_status") or "") for row in classified)),
        "by_text_readiness_status": dict(Counter(str(row.get("text_readiness_status") or "") for row in classified)),
        "by_next_autonomy_workstream": dict(Counter(str(row.get("next_autonomy_workstream") or "") for row in classified)),
        "current_handoff_by_frame_requested_action": dict(
            Counter(str(row.get("requested_action") or "") for row in classified if row.get("current_handoff"))
        ),
        "current_handoff_by_next_autonomy_workstream": dict(
            Counter(str(row.get("next_autonomy_workstream") or "") for row in classified if row.get("current_handoff"))
        ),
        "scope_confusion": _scope_confusion_summary(classified),
        "current_handoff_queue": _current_handoff_queue(classified),
        "negative_controls": negative_controls[:50],
        "examples": classified[:50],
        "notes": [
            "This is diagnostic only: no route/text change is authorized by this report.",
            "Fact-assertion rows need a verified client-safe fact path before any self-answer policy.",
            "Clean route-only discussion rows, if any, still require Claude/Dmitry review before active work.",
        ],
    }


def _real_lever_row(
    row: Mapping[str, Any],
    *,
    proof_reconciliation: Mapping[str, Mapping[str, Any]],
    proof_text_readiness: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
    route = str(row.get("current_route") or row.get("route") or "")
    notes = str(row.get("notes") or "").casefold()
    dialog_id = str(row.get("dialog_id") or "")
    requested_action = str(frame.get("requested_action") or "")
    risk_class = str(frame.get("risk_class") or "")
    answerability = str(frame.get("answerability") or "")
    current_handoff = route in HANDOFF_ROUTES
    reconciliation = proof_reconciliation.get(_turn_key(dialog_id, row.get("turn")))
    if not isinstance(reconciliation, Mapping):
        reconciliation = {}
    text_readiness = proof_text_readiness.get(_turn_key(dialog_id, row.get("turn")))
    if not isinstance(text_readiness, Mapping):
        text_readiness = {}
    reconciliation_status = str(reconciliation.get("status") or "")
    reconciliation_would_fix = reconciliation_status == "would_reconcile_to_safe_reference"
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
    result = {
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
        "proof_reconciliation_status": reconciliation_status,
        "proof_reconciliation_reason": str(reconciliation.get("reason") or ""),
        "proof_reconciliation_would_reconcile": reconciliation_would_fix,
        "proof_reconciliation_exact_fact_keys": list(reconciliation.get("exact_fact_keys") or [])
        if isinstance(reconciliation.get("exact_fact_keys"), list)
        else [],
        "text_readiness_status": str(text_readiness.get("status") or ""),
        "text_readiness_blockers": list(text_readiness.get("blockers") or [])
        if isinstance(text_readiness.get("blockers"), list)
        else [],
        "send_as_is_review_candidate": bool(text_readiness.get("send_as_is_review_candidate")),
        "source_fact_lookup_status": str(text_readiness.get("source_fact_lookup_status") or ""),
        "source_fact_candidate_count": text_readiness.get("source_fact_candidate_count", 0),
        "source_fact_candidate_brands": list(text_readiness.get("source_fact_candidate_brands") or [])
        if isinstance(text_readiness.get("source_fact_candidate_brands"), list)
        else [],
        "source_fact_selected_brand": str(text_readiness.get("source_fact_selected_brand") or ""),
        "source_fact_brand_index_collision": bool(text_readiness.get("source_fact_brand_index_collision")),
        "source_fact_client_safe_text_present": bool(text_readiness.get("source_fact_client_safe_text_present")),
        "source_fact_client_safe_text_length": text_readiness.get("source_fact_client_safe_text_length", 0),
        "source_fact_client_safe_text_hash": str(text_readiness.get("source_fact_client_safe_text_hash") or ""),
        "source_fact_client_safe_text_pii_signal": bool(text_readiness.get("source_fact_client_safe_text_pii_signal")),
        "structured_value_available": bool(text_readiness.get("structured_value_available")),
        "structured_value_keys": list(text_readiness.get("structured_value_keys") or [])
        if isinstance(text_readiness.get("structured_value_keys"), list)
        else [],
        "structured_value_field_count": text_readiness.get("structured_value_field_count", 0),
        "bot_template_required": bool(text_readiness.get("bot_template_required")),
        "template_registry_status": str(text_readiness.get("template_registry_status") or ""),
        "template_id": str(text_readiness.get("template_id") or ""),
        "template_text_length": text_readiness.get("template_text_length", 0),
        "template_text_hash": str(text_readiness.get("template_text_hash") or ""),
        "direct_quote_forbidden": bool(text_readiness.get("direct_quote_forbidden")),
        "text_candidate_readiness_status": str(text_readiness.get("text_candidate_readiness_status") or ""),
        "text_candidate_blockers": list(text_readiness.get("text_candidate_blockers") or [])
        if isinstance(text_readiness.get("text_candidate_blockers"), list)
        else [],
        "text_policy_readiness_status": str(text_readiness.get("text_policy_readiness_status") or ""),
        "text_policy_blockers": list(text_readiness.get("text_policy_blockers") or [])
        if isinstance(text_readiness.get("text_policy_blockers"), list)
        else [],
        "source_alignment_status": str(text_readiness.get("source_alignment_status") or ""),
        "source_alignment_blockers": list(text_readiness.get("source_alignment_blockers") or [])
        if isinstance(text_readiness.get("source_alignment_blockers"), list)
        else [],
        "source_alignment_missing_fact_categories": list(text_readiness.get("source_alignment_missing_fact_categories") or [])
        if isinstance(text_readiness.get("source_alignment_missing_fact_categories"), list)
        else [],
        "source_alignment_fact_categories": list(text_readiness.get("source_alignment_fact_categories") or [])
        if isinstance(text_readiness.get("source_alignment_fact_categories"), list)
        else [],
        "source_alignment_uncovered_categories": list(text_readiness.get("source_alignment_uncovered_categories") or [])
        if isinstance(text_readiness.get("source_alignment_uncovered_categories"), list)
        else [],
        "shadow_text_renderer_status": str(text_readiness.get("shadow_text_renderer_status") or ""),
        "shadow_text_renderer_blockers": list(text_readiness.get("shadow_text_renderer_blockers") or [])
        if isinstance(text_readiness.get("shadow_text_renderer_blockers"), list)
        else [],
        "shadow_text_renderer_source": str(text_readiness.get("shadow_text_renderer_source") or ""),
        "shadow_text_candidate_length": text_readiness.get("shadow_text_candidate_length", 0),
        "shadow_text_candidate_hash": str(text_readiness.get("shadow_text_candidate_hash") or ""),
        "shadow_text_candidate_exported": bool(text_readiness.get("shadow_text_candidate_exported")),
        "lever_class": lever_class,
        "active_blockers": blockers,
        "review_question": _real_lever_review_question(lever_class),
        "notes": str(row.get("notes") or ""),
    }
    result["next_autonomy_workstream"] = _next_autonomy_workstream(result)
    result["active_unlock_reason"] = _active_unlock_reason(result)
    return result


def _current_handoff_queue(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current = [row for row in rows if row.get("current_handoff")]
    return {
        "total": len(current),
        "by_route": dict(Counter(str(row.get("route") or "") for row in current)),
        "by_frame_requested_action": dict(Counter(str(row.get("requested_action") or "") for row in current)),
        "by_lever_class": dict(Counter(str(row.get("lever_class") or "") for row in current)),
        "by_next_autonomy_workstream": dict(Counter(str(row.get("next_autonomy_workstream") or "") for row in current)),
        "fact_assertion_required": sum(1 for row in current if row.get("requires_fact_assertion")),
        "factless_ack_status": sum(1 for row in current if row.get("factless_ack_status")),
        "danger_adjacent": sum(1 for row in current if row.get("danger_adjacent")),
        "clean_route_only_discussion": sum(1 for row in current if row.get("clean_route_only_discussion")),
        "proof_reconciliation_would_reconcile": sum(1 for row in current if row.get("proof_reconciliation_would_reconcile")),
        "shadow_text_renderer_candidates": sum(
            1 for row in current if row.get("shadow_text_renderer_status") == "candidate_rendered"
        ),
        "examples": current[:50],
        "notes": [
            "Only current handoff rows can improve autonomy by changing routing.",
            "Rows that already answer self are kept for frame calibration, but are not active autonomy leverage.",
            "Fact-assertion rows need a verified fact/text path before any route change.",
        ],
    }


def _next_autonomy_workstream(row: Mapping[str, Any]) -> str:
    if not row.get("current_handoff"):
        return "no_current_route_leverage"
    if row.get("danger_adjacent"):
        return "danger_adjacent_do_not_lower"
    if row.get("clean_route_only_discussion"):
        return "route_only_ack_status_candidate_review"
    if row.get("requires_fact_assertion"):
        if row.get("proof_reconciliation_would_reconcile"):
            renderer_status = str(row.get("shadow_text_renderer_status") or "")
            source_status = str(row.get("source_fact_lookup_status") or "")
            alignment_status = str(row.get("source_alignment_status") or "")
            if alignment_status.startswith("blocked_source_axis") or alignment_status == "alignment_review_unclear":
                return "fix_proof_axis_alignment"
            if renderer_status == "candidate_rendered":
                return "text_policy_semantic_review_needed"
            if source_status == "wrong_brand" or renderer_status == "blocked_wrong_brand":
                return "fix_wrong_brand_proof_alignment"
            if renderer_status == "blocked_template_renderer_not_implemented":
                return "template_renderer_needed"
            if renderer_status == "blocked_unsupported_structured_value":
                return "renderer_or_structured_fact_needed"
            return "proof_text_policy_needed"
        return "fact_verification_or_retrieval_needed"
    if str(row.get("route") or "") == "manager_only":
        return "manager_only_policy_review"
    return "measurement_review_required"


def _active_unlock_reason(row: Mapping[str, Any]) -> str:
    workstream = str(row.get("next_autonomy_workstream") or "")
    reasons = {
        "no_current_route_leverage": "runtime_already_self_answered",
        "danger_adjacent_do_not_lower": "near_p0_money_fabrication_or_safety_case",
        "route_only_ack_status_candidate_review": "possible_route_only_but_requires_owner_semantic_review",
        "fact_verification_or_retrieval_needed": "needs_verified_client_safe_fact_before_self_answer",
        "text_policy_semantic_review_needed": "shadow_renderer_candidate_exists_but_active_text_policy_not_approved",
        "fix_wrong_brand_proof_alignment": "proof_points_to_other_brand_or_wrong_active_brand",
        "fix_proof_axis_alignment": "proof_fact_does_not_cover_requested_missing_fact_axis",
        "template_renderer_needed": "fact_requires_template_renderer_before_self_answer",
        "renderer_or_structured_fact_needed": "fresh_proof_exists_but_no_safe_renderer_for_structured_value",
        "proof_text_policy_needed": "fresh_proof_exists_but_text_policy_is_not_sendable",
        "manager_only_policy_review": "manager_only_route_needs_owner_policy_before_any_demotion",
        "measurement_review_required": "needs_measurement_or_gold_review_before_implementation",
    }
    return reasons.get(workstream, "review_required")


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


def _listish(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, "", False):
        return []
    return [value]


def _output_sanitizer_applied(value: Mapping[str, Any], *, safety_flags: Sequence[Any]) -> bool:
    if any(str(flag).startswith("output_sanitizer") for flag in safety_flags):
        return True
    if not isinstance(value, Mapping) or not value:
        return False
    if bool(value.get("applied")) or bool(value.get("changed")) or bool(value.get("fallback")):
        return True
    reasons = value.get("reasons")
    return bool(reasons)


def _forbidden_promises_present(turn: Mapping[str, Any], direct_path: Mapping[str, Any]) -> bool:
    for key in ("bot_forbidden_promises_detected", "forbidden_promises_detected"):
        if _listish(turn.get(key)):
            return True
    if _listish(direct_path.get("forbidden_promises_detected")):
        return True
    return False


def _semantic_verifier_blocks(value: Mapping[str, Any]) -> bool:
    action = str(value.get("action") or "").casefold()
    if action and action not in {"pass", "keep", "none", "ok"}:
        return True
    if value.get("unavailable_reason"):
        return True
    findings = value.get("findings")
    if isinstance(findings, list) and findings:
        return True
    finding_codes = value.get("finding_codes")
    if isinstance(finding_codes, list) and finding_codes:
        return True
    return False


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
