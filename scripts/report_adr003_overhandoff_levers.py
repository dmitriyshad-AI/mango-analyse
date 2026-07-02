#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_semantic_frame_eval import _load_transcripts, _strict_bool, _turns


SCHEMA_VERSION = "adr003_overhandoff_levers_v1_2026_07_02"
SAFE_HANDOFF_GROUPS = (
    "existence_format_needs_fact_verification_blocked",
    "danger_adjacent_blocked",
    "harmless_context_ack_status_candidate",
    "safe_reference_without_exact_facts",
    "low_confidence_or_missing_facts_blocked",
    "p0_or_money_or_operational_blocked",
    "unclear_review_required",
    "already_self",
)
HANDOFF_ROUTES = {"manager_only", "draft_for_manager"}
SELF_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot", "self_answer"}
CONTEXT_MESSAGE_TYPES = {"context_update", "non_question", "wait_for_more"}
SAFE_REQUESTED_ACTIONS = {"answer_question", "acknowledge_status", "acknowledge"}
OPERATIONAL_REQUESTED_ACTIONS = {
    "check_availability",
    "enroll",
    "book",
    "reserve",
    "send_payment_link",
    "handoff_manager",
    "send_document",
}
OPERATIONAL_DEAL_STAGES = {"closing", "post_payment", "support"}
MONEY_PAYMENT_READINESS = {"ready_to_pay", "paid", "dispute"}
DANGER_DIALOG_MARKERS = (
    "p0",
    "fabrication",
    "paid_transfer",
    "paid_no_access",
    "payment",
    "refund",
    "legal",
    "complaint",
)
EXISTENCE_REFERENCE_MARKERS = (
    "safe reference",
    "existence",
    "format",
    "course",
    "camp",
    "существ",
    "формат",
    "курс",
    "лагер",
    "групп",
    "класс",
    "пригод",
)
RISK_MARKERS = (
    "p0",
    "refund",
    "legal",
    "complaint",
    "payment_dispute",
    "paid_operation_context",
    "high_risk",
    "money",
    "payment",
    "оплат",
    "возврат",
    "жалоб",
    "договор",
)
OUTPUT_BLOCK_MARKERS = (
    "менеджер свяж",
    "менеджер напиш",
    "заброни",
    "мест",
    "запис",
    "стоим",
    "цена",
    "распис",
    "дата",
    "codex",
    "claude",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report ADR-003 safe over-handoff levers from SemanticFrame telemetry.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(transcripts=args.transcripts, gold=args.gold)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_overhandoff_levers_report.json"
    md_path = args.out_dir / "adr003_overhandoff_levers_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, transcripts: Path, gold: Path) -> dict[str, Any]:
    dialogs = _load_transcripts(transcripts)
    turn_map = _build_turn_map(dialogs)
    gold_rows = _load_gold_rows(gold)
    compared_rows: list[dict[str, Any]] = []
    missing_gold_turns: list[dict[str, Any]] = []
    for gold_row in gold_rows:
        key = (str(gold_row.get("dialog_id") or ""), int(gold_row.get("turn") or 0))
        turn = turn_map.get(key)
        if turn is None:
            missing_gold_turns.append({"dialog_id": key[0], "turn": key[1]})
            continue
        compared_rows.append(_classify_gold_turn(gold_row, turn))

    totals = _totals(dialogs=dialogs, rows=compared_rows, missing_gold_turns=missing_gold_turns)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "transcripts": str(transcripts),
            "transcripts_sha256": _sha256(transcripts),
            "gold": str(gold),
            "gold_sha256": _sha256(gold),
        },
        "totals": totals,
        "breakdowns": _breakdowns(compared_rows),
        "frame_too_cautious": _frame_too_cautious_summary(compared_rows),
        "groups": _groups(compared_rows),
        "acceptance": _acceptance(totals, compared_rows),
        "notes": [
            "Report-only scorer: route/text/runtime behavior is not changed.",
            "Runtime actual_p0 uses model/flag/shadow evidence and does not treat route=manager_only as P0 by itself.",
            "manager_only candidates are diagnostic only and require a separate policy decision before any future active demotion.",
        ],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    frame_too_cautious = (
        report.get("frame_too_cautious") if isinstance(report.get("frame_too_cautious"), Mapping) else {}
    )
    lines = [
        "# ADR-003 Ф2b Over-Handoff Levers",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Total turns: `{totals.get('turns_total', 0)}`",
        f"- Gold compared rows: `{totals.get('gold_compared_rows', 0)}`",
        f"- Gold safe/self rows: `{totals.get('gold_safe_self_rows', 0)}`",
        f"- Safe already self: `{totals.get('safe_already_self', 0)}`",
        f"- Safe handoff total: `{totals.get('safe_handoff_total', 0)}`",
        f"- Safe manager_only: `{totals.get('safe_manager_only', 0)}`",
        f"- Safe draft_for_manager: `{totals.get('safe_draft_for_manager', 0)}`",
        f"- Existence/format blocked before fact verification: `{totals.get('existence_format_needs_fact_verification_blocked', 0)}`",
        f"- Danger-adjacent blocked: `{totals.get('danger_adjacent_blocked', 0)}`",
        f"- Frame too cautious safe/self rows: `{frame_too_cautious.get('count', 0)}`",
        f"- Frame too cautious existence/format rows: `{frame_too_cautious.get('existence_format_count', 0)}`",
        f"- Harmless context/status candidates: `{totals.get('harmless_context_ack_status_candidates', 0)}`",
        f"- Draft candidates for possible future route-only active: `{totals.get('draft_candidates_for_future_active', 0)}`",
        f"- Manager-only candidates needing policy decision: `{totals.get('manager_only_candidates_need_policy_decision', 0)}`",
        "",
        "## Группы",
        "",
    ]
    for group in SAFE_HANDOFF_GROUPS:
        value = groups.get(group) if isinstance(groups.get(group), Mapping) else {}
        lines.append(f"- `{group}`: `{value.get('count', 0)}`")
    lines.extend(["", "## Доминанта по requested_action", ""])
    by_action = report.get("breakdowns", {}).get("safe_by_frame_requested_action", {}) if isinstance(report.get("breakdowns"), Mapping) else {}
    for action, count in sorted(by_action.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{action or 'missing'}`: `{count}`")
    lines.extend(["", "## Frame Too Cautious по requested_action", ""])
    by_action_frame = frame_too_cautious.get("by_requested_action") if isinstance(frame_too_cautious.get("by_requested_action"), Mapping) else {}
    for action, count in sorted(by_action_frame.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{action or 'missing'}`: `{count}`")
    lines.extend(["", "## Кандидаты", ""])
    candidates = groups.get("harmless_context_ack_status_candidate") if isinstance(groups.get("harmless_context_ack_status_candidate"), Mapping) else {}
    for item in candidates.get("examples", [])[:20]:
        lines.append(
            "- "
            f"`{item.get('dialog_id')}#{item.get('turn')}` "
            f"route=`{item.get('route')}` status=`{item.get('candidate_status')}` "
            f"confidence=`{item.get('frame', {}).get('confidence')}` "
            f"message_type=`{item.get('message_type')}`"
        )
        if item.get("client_excerpt"):
            lines.append(f"  - client: {item.get('client_excerpt')}")
        if item.get("bot_excerpt"):
            lines.append(f"  - bot: {item.get('bot_excerpt')}")
        if item.get("blocked_reasons"):
            lines.append(f"  - reasons: `{', '.join(item.get('blocked_reasons') or [])}`")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _load_gold_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and isinstance(data.get("rows"), list):
            return [row for row in data["rows"] if isinstance(row, Mapping)]
        return []
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, Mapping):
                rows.append(item)
    return rows


def _build_turn_map(dialogs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for index, turn in enumerate(_turns(dialog), 1):
            turn_no = int(turn.get("turn") or index)
            merged = dict(turn)
            merged.setdefault("__dialog_brand", dialog.get("brand"))
            result[(dialog_id, turn_no)] = merged
    return result


def _classify_gold_turn(gold_row: Mapping[str, Any], turn: Mapping[str, Any]) -> dict[str, Any]:
    expected = gold_row.get("expected") if isinstance(gold_row.get("expected"), Mapping) else gold_row
    route = str(turn.get("bot_route") or "")
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    contract = turn.get("bot_answer_contract") if isinstance(turn.get("bot_answer_contract"), Mapping) else {}
    self_shadow = (
        turn.get("bot_semantic_frame_self_answer_shadow")
        if isinstance(turn.get("bot_semantic_frame_self_answer_shadow"), Mapping)
        else {}
    )
    guards = self_shadow.get("guards") if isinstance(self_shadow.get("guards"), Mapping) else {}
    message_type = str(turn.get("bot_message_type") or "")
    expected_must = _strict_bool(expected.get("must_handoff"))
    expected_risk = _clean(expected.get("risk_class"))
    expected_answerability = _clean(expected.get("answerability"))
    is_gold_safe_self = expected_must is False and expected_risk == "safe" and expected_answerability == "answer_self"
    row = _evidence_row(gold_row=gold_row, turn=turn, route=route, frame=frame, contract=contract, self_shadow=self_shadow)
    row["is_gold_safe_self"] = is_gold_safe_self
    row["current_route_handoff"] = route in HANDOFF_ROUTES
    row["runtime_actual_p0"] = _runtime_actual_p0(turn, guards)
    row["danger_adjacent"] = _danger_adjacent(row)
    row["requires_existence_fact_verification"] = _requires_existence_fact_verification(row)
    row["facts_verified_for_existence"] = _facts_verified_for_existence(guards)
    row["frame_too_cautious"] = is_gold_safe_self and row["frame"]["must_handoff"] is True

    if not is_gold_safe_self:
        row["group"] = "not_gold_safe_self"
        row["blocked_reasons"] = ["gold_expected_handoff_or_not_safe_self"]
        return row
    if route in SELF_ROUTES:
        row["group"] = "already_self"
        row["candidate_status"] = "already_self"
        row["blocked_reasons"] = []
        return row
    if route not in HANDOFF_ROUTES:
        row["group"] = "unclear_review_required"
        row["blocked_reasons"] = ["unexpected_route"]
        return row

    blockers = _candidate_blockers(turn=turn, frame=frame, contract=contract, guards=guards, route=route)
    if row["requires_existence_fact_verification"] and not row["facts_verified_for_existence"]:
        blockers.append("requires_existence_fact_verification")
    if row["danger_adjacent"]:
        blockers.append("danger_adjacent_dialog")
    row["blocked_reasons"] = blockers
    if not blockers:
        row["group"] = "harmless_context_ack_status_candidate"
        row["candidate_status"] = (
            "would_need_manager_only_policy_decision" if route == "manager_only" else "would_allow_self_context_ack"
        )
    elif row["runtime_actual_p0"]:
        row["group"] = "p0_or_money_or_operational_blocked"
    elif row["danger_adjacent"]:
        row["group"] = "danger_adjacent_blocked"
    elif row["requires_existence_fact_verification"] and not row["facts_verified_for_existence"]:
        row["group"] = "existence_format_needs_fact_verification_blocked"
    elif _has_risk_or_operational_blocker(blockers):
        row["group"] = "p0_or_money_or_operational_blocked"
    elif _has_fact_blocker(blockers):
        row["group"] = "safe_reference_without_exact_facts"
    elif any(reason in blockers for reason in ("low_confidence", "frame_too_cautious", "message_type_not_context")):
        row["group"] = "low_confidence_or_missing_facts_blocked"
    else:
        row["group"] = "unclear_review_required"
    return row


def _candidate_blockers(
    *,
    turn: Mapping[str, Any],
    frame: Mapping[str, Any],
    contract: Mapping[str, Any],
    guards: Mapping[str, Any],
    route: str,
) -> list[str]:
    reasons: list[str] = []
    message_type = str(turn.get("bot_message_type") or "")
    if message_type not in CONTEXT_MESSAGE_TYPES:
        reasons.append("message_type_not_context")
    if str(contract.get("direct_question") or "").strip():
        reasons.append("direct_question_present")
    if route not in HANDOFF_ROUTES:
        reasons.append("route_not_handoff")
    if _runtime_actual_p0(turn, guards):
        reasons.append("runtime_actual_p0")
    if _strict_bool(frame.get("must_handoff")) is not False:
        reasons.append("frame_too_cautious")
    if _clean(frame.get("risk_class")) != "safe":
        reasons.append("frame_risk_not_safe")
    if _clean(frame.get("answerability")) != "answer_self":
        reasons.append("frame_answerability_not_self")
    if _clean(frame.get("requested_action")) not in SAFE_REQUESTED_ACTIONS:
        reasons.append("requested_action_not_safe_ack")
    if _float_or_none(frame.get("confidence")) is None or (_float_or_none(frame.get("confidence")) or 0.0) < 0.90:
        reasons.append("low_confidence")
    if _money_or_operational_signal(turn, frame, contract):
        reasons.append("money_or_operational_signal")
    if _has_missing_facts(turn, guards):
        reasons.append("missing_facts")
    if _partial_or_unknown_freshness(guards):
        reasons.append("facts_not_fresh_client_safe")
    if _brand_scope_unclear_or_mixed(turn, frame):
        reasons.append("brand_scope_unclear_or_mixed")
    if _unsafe_output_text(turn, guards):
        reasons.append("unsafe_output_text")
    if _has_blocking_flags(turn):
        reasons.append("blocking_safety_flags")
    return _dedupe(reasons)


def _evidence_row(
    *,
    gold_row: Mapping[str, Any],
    turn: Mapping[str, Any],
    route: str,
    frame: Mapping[str, Any],
    contract: Mapping[str, Any],
    self_shadow: Mapping[str, Any],
) -> dict[str, Any]:
    guards = self_shadow.get("guards") if isinstance(self_shadow.get("guards"), Mapping) else {}
    return {
        "dialog_id": str(gold_row.get("dialog_id") or ""),
        "turn": int(gold_row.get("turn") or 0),
        "brand": str(turn.get("brand") or gold_row.get("brand") or ""),
        "route": route,
        "message_type": str(turn.get("bot_message_type") or ""),
        "topic_id": str(turn.get("bot_topic_id") or ""),
        "safety_flags": [str(flag) for flag in (turn.get("bot_safety_flags") or [])],
        "missing_facts_count": len(turn.get("bot_missing_facts") or []),
        "client_excerpt": _redacted_excerpt(turn.get("client_message")),
        "bot_excerpt": _redacted_excerpt(turn.get("bot_text")),
        "frame": {
            "risk_class": _clean(frame.get("risk_class")),
            "answerability": _clean(frame.get("answerability")),
            "requested_action": _clean(frame.get("requested_action")),
            "deal_stage": _clean(frame.get("deal_stage")),
            "payment_readiness": _clean(frame.get("payment_readiness")),
            "must_handoff": _strict_bool(frame.get("must_handoff")),
            "confidence": _float_or_none(frame.get("confidence")),
            "requested_product_brand": _requested_product_brand(frame),
        },
        "contract": {
            "route": str(contract.get("route") or ""),
            "route_reason": str(contract.get("route_reason") or ""),
            "primary_intent": str(contract.get("primary_intent") or ""),
            "answer_policy": str(contract.get("answer_policy") or ""),
            "direct_question_present": bool(str(contract.get("direct_question") or "").strip()),
            "p0_required": _strict_bool(contract.get("p0_required")),
        },
        "self_answer_shadow": {
            "status": str(self_shadow.get("status") or ""),
            "reason": str(self_shadow.get("reason") or ""),
            "self_class": str(self_shadow.get("self_class") or ""),
            "guards": {
                "actual_p0": _strict_bool(guards.get("actual_p0")),
                "has_missing_facts": _strict_bool(guards.get("has_missing_facts")),
                "blocking_flags": [str(flag) for flag in (guards.get("blocking_flags") or [])],
                "freshness": guards.get("freshness") if isinstance(guards.get("freshness"), Mapping) else {},
            },
        },
        "gold": {
            "expected": gold_row.get("expected") if isinstance(gold_row.get("expected"), Mapping) else {},
            "review_label": str(gold_row.get("review_label") or ""),
            "notes": str(gold_row.get("notes") or "")[:180],
        },
    }


def _totals(
    *,
    dialogs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    missing_gold_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    safe_rows = [row for row in rows if row.get("is_gold_safe_self")]
    safe_handoff = [row for row in safe_rows if row.get("current_route_handoff")]
    harmless = [row for row in safe_rows if row.get("group") == "harmless_context_ack_status_candidate"]
    return {
        "dialogs_total": len(dialogs),
        "turns_total": sum(len(_turns(dialog)) for dialog in dialogs),
        "gold_compared_rows": len(rows),
        "missing_gold_turns": len(missing_gold_turns),
        "missing_gold_turn_examples": list(missing_gold_turns)[:10],
        "gold_safe_self_rows": len(safe_rows),
        "safe_already_self": sum(1 for row in safe_rows if row.get("group") == "already_self"),
        "safe_handoff_total": len(safe_handoff),
        "safe_manager_only": sum(1 for row in safe_handoff if row.get("route") == "manager_only"),
        "safe_draft_for_manager": sum(1 for row in safe_handoff if row.get("route") == "draft_for_manager"),
        "existence_format_needs_fact_verification_blocked": sum(
            1 for row in safe_rows if row.get("group") == "existence_format_needs_fact_verification_blocked"
        ),
        "danger_adjacent_blocked": sum(1 for row in safe_rows if row.get("group") == "danger_adjacent_blocked"),
        "harmless_context_ack_status_candidates": len(harmless),
        "draft_candidates_for_future_active": sum(
            1 for row in harmless if row.get("candidate_status") == "would_allow_self_context_ack"
        ),
        "manager_only_candidates_need_policy_decision": sum(
            1 for row in harmless if row.get("candidate_status") == "would_need_manager_only_policy_decision"
        ),
    }


def _breakdowns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe_rows = [row for row in rows if row.get("is_gold_safe_self")]
    return {
        "safe_by_group": _counter(safe_rows, "group"),
        "safe_by_route": _counter(safe_rows, "route"),
        "safe_by_message_type": _counter(safe_rows, "message_type"),
        "safe_by_route_reason": _nested_counter(safe_rows, ("contract", "route_reason")),
        "safe_by_primary_intent": _nested_counter(safe_rows, ("contract", "primary_intent")),
        "safe_by_frame_risk_class": _nested_counter(safe_rows, ("frame", "risk_class")),
        "safe_by_frame_answerability": _nested_counter(safe_rows, ("frame", "answerability")),
        "safe_by_frame_requested_action": _nested_counter(safe_rows, ("frame", "requested_action")),
        "safe_handoff_by_frame_requested_action": _nested_counter(
            [row for row in safe_rows if row.get("current_route_handoff")],
            ("frame", "requested_action"),
        ),
        "safe_by_self_shadow_reason": _nested_counter(safe_rows, ("self_answer_shadow", "reason")),
        "safe_by_self_shadow_class": _nested_counter(safe_rows, ("self_answer_shadow", "self_class")),
        "safe_safety_flags": _flag_counter(safe_rows),
        "blocked_reasons": _blocked_reason_counter(safe_rows),
    }


def _frame_too_cautious_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cautious = [row for row in rows if row.get("frame_too_cautious")]
    return {
        "count": len(cautious),
        "by_requested_action": _nested_counter(cautious, ("frame", "requested_action")),
        "by_route": _counter(cautious, "route"),
        "existence_format_count": sum(1 for row in cautious if row.get("requires_existence_fact_verification")),
        "danger_adjacent_count": sum(1 for row in cautious if row.get("danger_adjacent")),
        "examples": cautious[:50],
    }


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in SAFE_HANDOFF_GROUPS:
        grouped = [row for row in rows if row.get("group") == group]
        result[group] = {"count": len(grouped), "examples": grouped[:50]}
    return result


def _acceptance(totals: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    notes: list[str] = []
    status = "pass"
    candidate_rows = [row for row in rows if row.get("group") == "harmless_context_ack_status_candidate"]
    unsafe_candidates = [
        row
        for row in candidate_rows
        if row.get("runtime_actual_p0")
        or _has_risk_or_operational_blocker(row.get("blocked_reasons") or [])
        or row.get("frame", {}).get("requested_action") not in SAFE_REQUESTED_ACTIONS
    ]
    if totals.get("missing_gold_turns"):
        status = "needs_review"
        notes.append("Some gold rows are absent from transcripts.")
    if unsafe_candidates:
        status = "blocked"
        notes.append("At least one candidate has P0/money/operational evidence.")
    if not candidate_rows:
        notes.append("No harmless context/status candidates found; this may still be a valid negative result.")
    frame_existence = sum(
        1
        for row in rows
        if row.get("frame_too_cautious") and row.get("requires_existence_fact_verification")
    )
    if frame_existence:
        notes.append("Frame too-cautious existence/format rows need fact verification, not route-only demotion.")
    if totals.get("danger_adjacent_blocked", 0):
        notes.append("Some safe-label handoffs are danger-adjacent and must stay out of clean active candidates.")
    if totals.get("draft_candidates_for_future_active", 0) == 0:
        notes.append("No draft_for_manager candidates are ready even for a future route-only active discussion.")
    if totals.get("manager_only_candidates_need_policy_decision", 0):
        notes.append("manager_only candidates are diagnostics only; active use requires a separate owner decision.")
    return {"status": status, "notes": notes}


def _runtime_actual_p0(turn: Mapping[str, Any], guards: Mapping[str, Any] | None = None) -> bool:
    if guards and _strict_bool(guards.get("actual_p0")) is True:
        return True
    flags = " ".join(str(flag) for flag in (turn.get("bot_safety_flags") or [])).casefold()
    if any(marker in flags for marker in RISK_MARKERS):
        return True
    model_p0 = turn.get("bot_direct_path_model_p0") if isinstance(turn.get("bot_direct_path_model_p0"), Mapping) else {}
    if _strict_bool(model_p0.get("is_p0")) is True:
        return True
    direct_path = turn.get("bot_direct_path") if isinstance(turn.get("bot_direct_path"), Mapping) else {}
    nested_model = direct_path.get("direct_path_model_p0") if isinstance(direct_path.get("direct_path_model_p0"), Mapping) else {}
    if _strict_bool(nested_model.get("is_p0")) is True:
        return True
    plan = turn.get("bot_conversation_intent_plan") if isinstance(turn.get("bot_conversation_intent_plan"), Mapping) else {}
    risk_codes = " ".join(str(code) for code in (plan.get("risk_codes") or [])).casefold()
    return any(marker in risk_codes for marker in RISK_MARKERS)


def _money_or_operational_signal(turn: Mapping[str, Any], frame: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    payment_readiness = _clean(frame.get("payment_readiness"))
    requested_action = _clean(frame.get("requested_action"))
    deal_stage = _clean(frame.get("deal_stage"))
    contract_text = " ".join(
        str(value)
        for value in (
            contract.get("primary_intent"),
            contract.get("route_reason"),
            contract.get("answer_policy"),
            turn.get("bot_reason_class"),
        )
    ).casefold()
    return (
        payment_readiness in MONEY_PAYMENT_READINESS
        or requested_action in OPERATIONAL_REQUESTED_ACTIONS
        or deal_stage in OPERATIONAL_DEAL_STAGES
        or any(marker in contract_text for marker in RISK_MARKERS)
    )


def _has_missing_facts(turn: Mapping[str, Any], guards: Mapping[str, Any]) -> bool:
    if _strict_bool(guards.get("has_missing_facts")) is True:
        return True
    missing_facts = turn.get("bot_missing_facts")
    return isinstance(missing_facts, list) and bool(missing_facts)


def _partial_or_unknown_freshness(guards: Mapping[str, Any]) -> bool:
    freshness = guards.get("freshness") if isinstance(guards.get("freshness"), Mapping) else {}
    exact = _safe_int(freshness.get("exact_fact_count"))
    fresh = _safe_int(freshness.get("fresh_client_safe_count"))
    if exact <= 0:
        return False
    return not bool(freshness.get("ok")) or fresh < exact or freshness.get("all_exact_facts_fresh_client_safe") is False


def _brand_scope_unclear_or_mixed(turn: Mapping[str, Any], frame: Mapping[str, Any]) -> bool:
    active_brand = _clean(turn.get("brand") or turn.get("__dialog_brand") or turn.get("active_brand"))
    if active_brand not in {"foton", "unpk"}:
        return True
    product_brand = _requested_product_brand(frame)
    return bool(product_brand and product_brand in {"foton", "unpk"} and product_brand != active_brand)


def _requested_product_brand(frame: Mapping[str, Any]) -> str:
    requested_product = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    return _clean(requested_product.get("brand"))


def _unsafe_output_text(turn: Mapping[str, Any], guards: Mapping[str, Any]) -> bool:
    text = str(turn.get("bot_text") or "").casefold()
    if not text:
        return False
    if re.search(r"(^|\\W)(бот|ии|codex|claude)(\\W|$)", text, flags=re.IGNORECASE):
        return True
    freshness = guards.get("freshness") if isinstance(guards.get("freshness"), Mapping) else {}
    facts_verified = _safe_int(freshness.get("exact_fact_count")) > 0 and bool(freshness.get("ok"))
    return not facts_verified and any(marker in text for marker in OUTPUT_BLOCK_MARKERS)


def _has_blocking_flags(turn: Mapping[str, Any]) -> bool:
    flags = {str(flag) for flag in (turn.get("bot_safety_flags") or [])}
    blocking = {
        "autonomy_default_cautious_unverified_fact",
        "low_confidence_manager_only",
        "p0_required",
        "payment_required",
        "legal_required",
        "complaint_required",
    }
    return bool(flags & blocking)


def _danger_adjacent(row: Mapping[str, Any]) -> bool:
    dialog_id = str(row.get("dialog_id") or "").casefold()
    notes = str(row.get("gold", {}).get("notes") or "").casefold()
    flags = " ".join(str(flag) for flag in (row.get("safety_flags") or [])).casefold()
    return bool(row.get("runtime_actual_p0")) or any(
        marker in dialog_id or marker in notes or marker in flags for marker in DANGER_DIALOG_MARKERS
    )


def _requires_existence_fact_verification(row: Mapping[str, Any]) -> bool:
    frame = row.get("frame") if isinstance(row.get("frame"), Mapping) else {}
    action = str(frame.get("requested_action") or "").casefold()
    notes = str(row.get("gold", {}).get("notes") or "").casefold()
    self_class = str(row.get("self_answer_shadow", {}).get("self_class") or "").casefold()
    return (
        action in {"check_availability", "enroll", "handoff_manager"}
        and (
            "safe_reference" in self_class
            or any(marker in notes for marker in EXISTENCE_REFERENCE_MARKERS)
        )
    )


def _facts_verified_for_existence(guards: Mapping[str, Any]) -> bool:
    freshness = guards.get("freshness") if isinstance(guards.get("freshness"), Mapping) else {}
    exact = _safe_int(freshness.get("exact_fact_count"))
    fresh = _safe_int(freshness.get("fresh_client_safe_count"))
    return exact > 0 and bool(freshness.get("ok")) and fresh == exact and freshness.get("all_exact_facts_fresh_client_safe") is True


def _has_risk_or_operational_blocker(reasons: Sequence[str]) -> bool:
    return any(
        reason
        in {
            "runtime_actual_p0",
            "frame_risk_not_safe",
            "requested_action_not_safe_ack",
            "money_or_operational_signal",
            "brand_scope_unclear_or_mixed",
            "unsafe_output_text",
            "blocking_safety_flags",
        }
        for reason in reasons
    )


def _has_fact_blocker(reasons: Sequence[str]) -> bool:
    return any(
        reason in {"missing_facts", "facts_not_fresh_client_safe", "requires_existence_fact_verification"}
        for reason in reasons
    )


def _counter(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows))


def _nested_counter(rows: Sequence[Mapping[str, Any]], path: tuple[str, str]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        result[str(value or "")] += 1
    return dict(result)


def _flag_counter(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        result.update(str(flag) for flag in (row.get("safety_flags") or []))
    return dict(result)


def _blocked_reason_counter(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        result.update(str(reason) for reason in (row.get("blocked_reasons") or []))
    return dict(result)


def _redacted_excerpt(value: Any, *, limit: int = 180) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}", "[email]", text)
    text = re.sub(r"\b(id|amo|entity_id)\s*[:#-]?\s*\d{5,}\b", r"\1 [id]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:\+?\d[\s()\-]*){7,}", "[phone]", text)
    text = re.sub(r"\b\d{6,}\b", "[id]", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _source_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_or_none(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean(value: Any) -> str:
    return str(value or "").strip().casefold()


def _dedupe(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
