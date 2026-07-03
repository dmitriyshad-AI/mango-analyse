#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    OFF_TOPIC_INPUT_RE,
    _asks_explicit_live_availability_question,
    _selling_slots_from_text,
)
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    SemanticReading,
    off_topic_reading_decision,
    sense_seats_reading_decision,
    slots_reading_candidates,
)


SCHEMA_VERSION = "adr003_semantic_frame_eval_report_v1_1_2026_07_04"
FRAME_EMISSION_THRESHOLD = 0.97
DIRECT_PATH_P0_PREBLOCK_REASONS = {"p0_pre_gate", "direct_path_preblocked_p0"}
REQUIRED_FRAME_FIELDS = (
    "intent",
    "risk_class",
    "deal_stage",
    "payment_readiness",
    "requested_product",
    "requested_action",
    "answerability",
    "must_handoff",
    "evidence",
    "confidence",
)
P0_FLAG_MARKERS = (
    "p0",
    "refund",
    "legal",
    "complaint",
    "payment_dispute",
    "paid_operation_context",
    "high_risk",
)
MONEY_PAYMENT_READINESS_MARKERS = {"ready_to_pay", "paid", "dispute"}
MONEY_REQUESTED_ACTION_MARKERS = {"send_payment_link", "refund_or_cancel"}
OPERATIONAL_REQUESTED_ACTION_MARKERS = {"check_availability", "enroll", "send_document", "handoff_manager"}
OPERATIONAL_DEAL_STAGE_MARKERS = {"closing", "post_payment", "support"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ADR-003 SemanticFrame OFF/ON eval report from dynamic sim outputs.")
    parser.add_argument("--on-transcripts", type=Path, required=True)
    parser.add_argument("--on-summary", type=Path, default=None)
    parser.add_argument("--off-transcripts", type=Path, default=None)
    parser.add_argument("--off-summary", type=Path, default=None)
    parser.add_argument("--posthoc-transcripts", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(
        on_transcripts=args.on_transcripts,
        on_summary=args.on_summary,
        off_transcripts=args.off_transcripts,
        off_summary=args.off_summary,
        posthoc_transcripts=args.posthoc_transcripts,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_semantic_frame_eval_report.json"
    md_path = args.out_dir / "adr003_semantic_frame_eval_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(
    *,
    on_transcripts: Path,
    on_summary: Path | None = None,
    off_transcripts: Path | None = None,
    off_summary: Path | None = None,
    posthoc_transcripts: Path | None = None,
) -> dict[str, Any]:
    on_dialogs = _load_transcripts(on_transcripts)
    off_dialogs = _load_transcripts(off_transcripts) if off_transcripts else []
    posthoc_dialogs = _load_transcripts(posthoc_transcripts) if posthoc_transcripts else []
    on_summary_data = _load_json(on_summary)
    off_summary_data = _load_json(off_summary)
    paired_dialogs = _paired_dialog_metrics(off_dialogs, on_dialogs) if off_dialogs else {"status": "not_provided"}

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "on_transcripts": str(on_transcripts),
            "on_summary": str(on_summary or ""),
            "off_transcripts": str(off_transcripts or ""),
            "off_summary": str(off_summary or ""),
            "posthoc_transcripts": str(posthoc_transcripts or ""),
        },
        "paired_dialogs": paired_dialogs,
        "totals": _dialog_totals(on_dialogs),
        "off_on_diff": _compare_off_on(off_dialogs, on_dialogs) if off_dialogs else {"status": "not_provided"},
        "baseline_vs_inline_text_health": _baseline_vs_inline_text_health(off_dialogs, on_dialogs)
        if off_dialogs
        else {"status": "not_provided"},
        "inline_text_health_gate": _inline_text_health_gate(off_dialogs, on_dialogs)
        if off_dialogs
        else {"status": "not_provided"},
        "inline_vs_posthoc_agreement": _inline_vs_posthoc_agreement(on_dialogs, posthoc_dialogs)
        if posthoc_dialogs
        else {"status": "not_provided"},
        "reader_agreement": _reader_agreement_metrics(on_dialogs),
        "llm_calls": _llm_call_delta(off_summary_data, on_summary_data),
        "semantic_frame": _semantic_frame_metrics(on_dialogs),
        "frame_decision_shadow": _frame_decision_shadow_metrics(on_dialogs),
        "semantic_frame_proof_reconciliation_shadow": _semantic_frame_proof_reconciliation_shadow_metrics(on_dialogs),
        "semantic_frame_self_answer_shadow": _semantic_frame_self_answer_shadow_metrics(on_dialogs),
        "hard_gate_failures": {
            "on": len(on_summary_data.get("hard_gate_failure_dialogs") or []),
            "off": len(off_summary_data.get("hard_gate_failure_dialogs") or []) if off_summary_data else None,
        },
    }
    report["acceptance"] = _acceptance(report)
    report["decision_readiness"] = _decision_readiness(report)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    frame = report.get("semantic_frame") if isinstance(report.get("semantic_frame"), Mapping) else {}
    diff = report.get("off_on_diff") if isinstance(report.get("off_on_diff"), Mapping) else {}
    llm = report.get("llm_calls") if isinstance(report.get("llm_calls"), Mapping) else {}
    shadow = report.get("frame_decision_shadow") if isinstance(report.get("frame_decision_shadow"), Mapping) else {}
    reconciliation_shadow = (
        report.get("semantic_frame_proof_reconciliation_shadow")
        if isinstance(report.get("semantic_frame_proof_reconciliation_shadow"), Mapping)
        else {}
    )
    self_shadow = (
        report.get("semantic_frame_self_answer_shadow")
        if isinstance(report.get("semantic_frame_self_answer_shadow"), Mapping)
        else {}
    )
    inline_posthoc = (
        report.get("inline_vs_posthoc_agreement") if isinstance(report.get("inline_vs_posthoc_agreement"), Mapping) else {}
    )
    text_health = (
        report.get("baseline_vs_inline_text_health")
        if isinstance(report.get("baseline_vs_inline_text_health"), Mapping)
        else {}
    )
    inline_gate = (
        report.get("inline_text_health_gate")
        if isinstance(report.get("inline_text_health_gate"), Mapping)
        else {}
    )
    paired = report.get("paired_dialogs") if isinstance(report.get("paired_dialogs"), Mapping) else {}
    lines = [
        "# ADR-003 SemanticFrame Eval Report",
        "",
        f"- Acceptance: `{acceptance.get('status', 'unknown')}`",
        f"- Technical shadow status: `{(report.get('decision_readiness') or {}).get('technical_shadow_status', 'unknown')}`",
        f"- Semantic decision status: `{(report.get('decision_readiness') or {}).get('semantic_decision_status', 'unknown')}`",
        f"- Active behavior allowed: `{(report.get('decision_readiness') or {}).get('active_behavior_allowed', False)}`",
        f"- ON turns: `{frame.get('turns_total', 0)}`",
        f"- Paired dialogs common/baseline-only/inline-only: `{paired.get('common_count', 'n/a')}` / `{paired.get('baseline_only_count', 'n/a')}` / `{paired.get('inline_only_count', 'n/a')}`",
        f"- Frame present: `{frame.get('present_count', 0)}` / `{frame.get('turns_total', 0)}`",
        f"- Direct-path model-not-called turns: `{frame.get('model_not_called_count', 0)}`",
        f"- Frame eligible model-called turns: `{frame.get('eligible_model_called_turns', 0)}`",
        f"- Frame eligible emission: `{frame.get('eligible_frame_count', 0)}` / `{frame.get('eligible_model_called_turns', 0)}`",
        f"- Frame eligible emission rate: `{frame.get('eligible_frame_rate', 'n/a')}`",
        f"- Direct-path P0 preblocked turns: `{frame.get('preblocked_p0_count', 0)}`",
        f"- Provider timeouts: `{frame.get('provider_timeout_count', 0)}`",
        f"- Infra timeout present: `{frame.get('infra_timeout_present', False)}`",
        f"- Frame schema complete: `{frame.get('complete_required_count', 0)}` / `{frame.get('present_count', 0)}`",
        f"- OFF/ON route-text diffs: `{diff.get('route_text_diff_count', 'n/a')}`",
        f"- OFF/ON input diffs: `{diff.get('input_diff_count', 'n/a')}`",
        f"- Baseline vs inline dangerous flips: `{text_health.get('dangerous_flip_count', 'n/a')}`",
        f"- Inline text health gate: `{inline_gate.get('status', 'n/a')}`",
        f"- P0 route lost: `{inline_gate.get('p0_route_lost_count', 'n/a')}`",
        f"- P0 hygiene flag diffs: `{inline_gate.get('p0_hygiene_flag_diff_count', 'n/a')}`",
        f"- P0 hygiene lost/added: `{inline_gate.get('p0_hygiene_lost_count', 'n/a')}` / `{inline_gate.get('p0_hygiene_added_count', 'n/a')}`",
        f"- Unverified new-number turns: `{inline_gate.get('new_number_unverified_count', 'n/a')}`",
        f"- Adjacent-fact new-number warnings: `{inline_gate.get('new_number_adjacent_warning_count', 'n/a')}`",
        f"- Dangerous manager-to-self flips: `{inline_gate.get('route_flip_dangerous_count', 'n/a')}`",
        f"- Inline vs posthoc compared turns: `{inline_posthoc.get('compared_turns', 'n/a')}`",
        f"- Inline vs posthoc mismatch count: `{inline_posthoc.get('mismatch_count', 'n/a')}`",
        f"- Reader agreement compared turns: `{(report.get('reader_agreement') or {}).get('compared_turns', 'n/a')}`",
        f"- Reader agreement mismatch count: `{(report.get('reader_agreement') or {}).get('mismatch_count', 'n/a')}`",
        f"- LLM call mode: `{llm.get('mode', 'unknown')}`",
        f"- LLM raw total delta: `{llm.get('raw_total_delta', 'n/a')}`",
        f"- LLM expected extra calls: `{llm.get('extra_total', 'n/a')}`",
        f"- LLM non-frame ON calls: `{llm.get('on_non_frame_total', 'n/a')}`",
        f"- Frame decision shadow turns: `{shadow.get('turn_count', 0)}`",
        f"- Proof reconciliation shadow turns: `{reconciliation_shadow.get('turn_count', 0)}`",
        f"- Proof reconciliation would-fix-frame rows: `{reconciliation_shadow.get('would_reconcile_count', 0)}`",
        f"- Proof reconciliation active allowed rows: `{reconciliation_shadow.get('active_allowed_count', 0)}`",
        f"- Self-answer shadow turns: `{self_shadow.get('turn_count', 0)}`",
        f"- Self-answer candidates: `{self_shadow.get('would_demote_count', 0)}`",
        f"- Self-answer P0-lowered candidates: `{self_shadow.get('p0_lowered_count', 0)}`",
        f"- Self-answer money-lowered candidates: `{self_shadow.get('money_lowered_count', 0)}`",
        f"- Self-answer operational-lowered candidates: `{self_shadow.get('operational_lowered_count', 0)}`",
        f"- Self-answer freshness-unknown candidates: `{self_shadow.get('freshness_unknown_self_candidates', 0)}`",
        f"- Self-answer partial-freshness candidates: `{self_shadow.get('partial_freshness_self_candidates', 0)}`",
        "",
        "## Acceptance Flags",
        "",
    ]
    for key, value in sorted((acceptance.get("flags") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _load_transcripts(path: Path | None) -> list[Mapping[str, Any]]:
    if path is None:
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


def _load_json(path: Path | None) -> Mapping[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else {}


def _dialog_totals(dialogs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    turns = sum(len(_turns(dialog)) for dialog in dialogs)
    return {"dialogs": len(dialogs), "turns": turns}


def _paired_dialog_metrics(
    baseline_dialogs: Sequence[Mapping[str, Any]],
    inline_dialogs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_ids = _dialog_id_set(baseline_dialogs)
    inline_ids = _dialog_id_set(inline_dialogs)
    common = sorted(baseline_ids & inline_ids)
    baseline_only = sorted(baseline_ids - inline_ids)
    inline_only = sorted(inline_ids - baseline_ids)
    status = "matched" if not baseline_only and not inline_only else "mismatch"
    return {
        "schema_version": "paired_dialogs_v1_2026_07_04",
        "status": status,
        "common_count": len(common),
        "baseline_only_count": len(baseline_only),
        "inline_only_count": len(inline_only),
        "baseline_only_dialog_ids": baseline_only[:50],
        "inline_only_dialog_ids": inline_only[:50],
    }


def _turns(dialog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = dialog.get("turns")
    return [turn for turn in raw if isinstance(turn, Mapping)] if isinstance(raw, list) else []


def _turn_map(dialogs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for index, turn in enumerate(_turns(dialog), 1):
            turn_no = int(turn.get("turn") or index)
            result[(dialog_id, turn_no)] = turn
    return result


def _compare_off_on(off_dialogs: Sequence[Mapping[str, Any]], on_dialogs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    off_map = _turn_map(off_dialogs)
    on_map = _turn_map(on_dialogs)
    common = sorted(set(off_map) & set(on_map))
    missing_off = sorted(set(on_map) - set(off_map))
    missing_on = sorted(set(off_map) - set(on_map))
    diffs: list[dict[str, Any]] = []
    input_diffs: list[dict[str, Any]] = []
    for key in common:
        off_turn = off_map[key]
        on_turn = on_map[key]
        input_changed: dict[str, dict[str, Any]] = {}
        for field in ("client_message", "client_stop"):
            if off_turn.get(field) != on_turn.get(field):
                input_changed[field] = {"off": off_turn.get(field), "on": on_turn.get(field)}
        if input_changed:
            input_diffs.append({"dialog_id": key[0], "turn": key[1], "changed": input_changed})
        changed: dict[str, dict[str, Any]] = {}
        for field in ("bot_route", "bot_text", "bot_safety_flags", "bot_manager_checklist"):
            if off_turn.get(field) != on_turn.get(field):
                changed[field] = {"off": off_turn.get(field), "on": on_turn.get(field)}
        if changed:
            diffs.append({"dialog_id": key[0], "turn": key[1], "changed": changed})
    return {
        "status": "compared",
        "compared_turns": len(common),
        "missing_off_turns": len(missing_off),
        "missing_on_turns": len(missing_on),
        "input_diff_count": len(input_diffs),
        "input_diff_examples": input_diffs[:25],
        "route_text_diff_count": len(diffs),
        "diff_examples": diffs[:25],
    }


def _frame_from_turn(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    frame = turn.get("bot_semantic_frame")
    if isinstance(frame, Mapping):
        return frame
    direct = turn.get("bot_direct_path") if isinstance(turn.get("bot_direct_path"), Mapping) else {}
    frame = direct.get("semantic_frame") if isinstance(direct.get("semantic_frame"), Mapping) else {}
    return frame if isinstance(frame, Mapping) else {}


def _direct_path_from_turn(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = turn.get("bot_direct_path")
    return direct if isinstance(direct, Mapping) else {}


def _answerability_trace_from_turn(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = turn.get("bot_answerability_trace")
    return trace if isinstance(trace, Mapping) else {}


def _provider_error_value(turn: Mapping[str, Any]) -> str:
    candidates: list[Any] = [
        turn.get("bot_provider_error"),
        turn.get("provider_error"),
    ]
    reason = turn.get("reason_evidence")
    if isinstance(reason, Mapping):
        candidates.append(reason.get("provider_error"))
    direct = _direct_path_from_turn(turn)
    direct_reason = direct.get("reason_evidence")
    if isinstance(direct_reason, Mapping):
        candidates.append(direct_reason.get("provider_error"))
    answerability = _answerability_trace_from_turn(turn)
    answerability_direct = answerability.get("direct_path")
    if isinstance(answerability_direct, Mapping):
        answerability_reason = answerability_direct.get("reason_evidence")
        if isinstance(answerability_reason, Mapping):
            candidates.append(answerability_reason.get("provider_error"))
    for candidate in candidates:
        value = str(candidate or "").strip().casefold()
        if value:
            return value
    return ""


def _is_provider_timeout_turn(turn: Mapping[str, Any]) -> bool:
    return _provider_error_value(turn) == "timeout"


def _is_direct_path_preblocked_p0_turn(turn: Mapping[str, Any]) -> bool:
    direct = _direct_path_from_turn(turn)
    return (
        direct.get("model_called") is False
        and direct.get("preblocked") is True
        and str(direct.get("preblock_reason") or "").strip() in DIRECT_PATH_P0_PREBLOCK_REASONS
    )


def _is_model_called_turn(turn: Mapping[str, Any]) -> bool:
    return _direct_path_from_turn(turn).get("model_called") is True


def _is_frame_emission_eligible_turn(turn: Mapping[str, Any]) -> bool:
    return _is_model_called_turn(turn) and not _is_provider_timeout_turn(turn)


def _dialog_id_set(dialogs: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(dialog.get("dialog_id") or "") for dialog in dialogs if str(dialog.get("dialog_id") or "")}


def _model_intent_from_turn(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _direct_path_from_turn(turn)
    candidates = (
        turn.get("bot_direct_path_model_intent"),
        turn.get("bot_model_intent"),
        direct.get("model_intent"),
        direct.get("direct_path_model_intent"),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _requested_product(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    product = frame.get("requested_product")
    return product if isinstance(product, Mapping) else {}


def _float01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _semantic_reading_from_turn(turn: Mapping[str, Any]) -> SemanticReading | None:
    frame = _frame_from_turn(turn)
    model_intent = _model_intent_from_turn(turn)
    if not frame and not model_intent:
        return None
    product = _requested_product(frame)
    return SemanticReading(
        source="inline",
        primary_intent=str(model_intent.get("primary_intent") or "").strip(),
        sense=str(model_intent.get("sense") or "").strip(),
        scope=str(model_intent.get("scope") or "").strip(),
        intent_confidence=_float01(model_intent.get("confidence")),
        requested_action=str(frame.get("requested_action") or "").strip(),
        product_grade=str(product.get("grade") or "").strip(),
        product_subject=str(product.get("subject") or "").strip(),
        product_format=str(product.get("format") or "").strip(),
        product_raw_text=str(product.get("raw_text") or "").strip(),
        frame_confidence=_float01(frame.get("confidence")),
    )


def _semantic_value(turn: Mapping[str, Any], field: str) -> str:
    frame = _frame_from_turn(turn)
    model_intent = _model_intent_from_turn(turn)
    product = _requested_product(frame)
    if field == "model_intent.primary_intent":
        return str(model_intent.get("primary_intent") or "").strip().casefold()
    if field == "model_intent.sense":
        return str(model_intent.get("sense") or "").strip().casefold()
    if field == "frame.intent":
        return str(frame.get("intent") or "").strip().casefold()
    if field == "frame.requested_action":
        return str(frame.get("requested_action") or "").strip().casefold()
    if field == "frame.answerability":
        return str(frame.get("answerability") or "").strip().casefold()
    if field == "frame.product_grade":
        return str(product.get("grade") or "").strip().casefold()
    if field == "frame.product_subject":
        return str(product.get("subject") or "").strip().casefold()
    if field == "frame.product_format":
        return str(product.get("format") or "").strip().casefold()
    return ""


def _inline_vs_posthoc_agreement(
    inline_dialogs: Sequence[Mapping[str, Any]],
    posthoc_dialogs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    inline_map = _turn_map(inline_dialogs)
    posthoc_map = _turn_map(posthoc_dialogs)
    common = sorted(set(inline_map) & set(posthoc_map))
    fields = (
        "model_intent.primary_intent",
        "model_intent.sense",
        "frame.intent",
        "frame.requested_action",
        "frame.answerability",
        "frame.product_grade",
        "frame.product_subject",
        "frame.product_format",
    )
    per_field = {
        field: {"match": 0, "mismatch": 0, "missing_inline": 0, "missing_posthoc": 0}
        for field in fields
    }
    examples: list[dict[str, Any]] = []
    mismatch_count = 0
    for key in common:
        inline_turn = inline_map[key]
        posthoc_turn = posthoc_map[key]
        changed: dict[str, dict[str, str]] = {}
        for field in fields:
            inline_value = _semantic_value(inline_turn, field)
            posthoc_value = _semantic_value(posthoc_turn, field)
            if not inline_value and posthoc_value:
                per_field[field]["missing_inline"] += 1
            elif inline_value and not posthoc_value:
                per_field[field]["missing_posthoc"] += 1
            elif inline_value == posthoc_value:
                per_field[field]["match"] += 1
            else:
                per_field[field]["mismatch"] += 1
                changed[field] = {"inline": inline_value, "posthoc": posthoc_value}
        if changed:
            mismatch_count += 1
            if len(examples) < 50:
                examples.append({"dialog_id": key[0], "turn": key[1], "changed": changed})
    return {
        "schema_version": "inline_vs_posthoc_agreement_v1_2026_07_03",
        "status": "compared",
        "compared_turns": len(common),
        "missing_inline_turns": len(set(posthoc_map) - set(inline_map)),
        "missing_posthoc_turns": len(set(inline_map) - set(posthoc_map)),
        "per_field": per_field,
        "mismatch_count": mismatch_count,
        "mismatch_examples": examples,
    }


def _reader_agreement_metrics(dialogs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    compared = 0
    mismatch_count = 0
    per_reader = {
        "sense_seats": {"match": 0, "mismatch": 0, "missing_reading": 0},
        "off_topic": {"match": 0, "mismatch": 0, "missing_reading": 0},
        "slot_grade": {"match": 0, "mismatch": 0, "missing_reading": 0, "legacy_only": 0, "reading_only": 0},
        "slot_subject": {"match": 0, "mismatch": 0, "missing_reading": 0, "legacy_only": 0, "reading_only": 0},
        "slot_format": {"match": 0, "mismatch": 0, "missing_reading": 0, "legacy_only": 0, "reading_only": 0},
    }
    examples: list[dict[str, Any]] = []
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        client_history: list[str] = []
        for index, turn in enumerate(_turns(dialog), 1):
            client_message = str(turn.get("client_message") or "")
            if client_message:
                client_history.append(f"Клиент: {client_message}")
            reading = _semantic_reading_from_turn(turn)
            if reading is None:
                continue
            compared += 1
            changed: dict[str, Any] = {}

            legacy_seats = "seats" if _asks_explicit_live_availability_question(client_message) else "not_seats"
            reading_seats = sense_seats_reading_decision(reading, client_message)
            if not reading_seats:
                per_reader["sense_seats"]["missing_reading"] += 1
            elif reading_seats == legacy_seats:
                per_reader["sense_seats"]["match"] += 1
            else:
                per_reader["sense_seats"]["mismatch"] += 1
                changed["sense_seats"] = {"legacy": legacy_seats, "reading": reading_seats}

            legacy_off_topic = "off_topic" if OFF_TOPIC_INPUT_RE.search(client_message) else "not_off_topic"
            reading_off_topic = off_topic_reading_decision(reading)
            if not reading_off_topic:
                per_reader["off_topic"]["missing_reading"] += 1
            elif reading_off_topic == legacy_off_topic:
                per_reader["off_topic"]["match"] += 1
            else:
                per_reader["off_topic"]["mismatch"] += 1
                changed["off_topic"] = {"legacy": legacy_off_topic, "reading": reading_off_topic}

            legacy_slots = dict(_selling_slots_from_text(client_message))
            reading_slots = {
                key: str(value.get("value") or "")
                for key, value in slots_reading_candidates(reading, tuple(client_history)).items()
                if isinstance(value, Mapping)
            }
            for slot_name in ("grade", "subject", "format"):
                key = f"slot_{slot_name}"
                legacy_value = str(legacy_slots.get(slot_name) or "")
                reading_value = str(reading_slots.get(slot_name) or "")
                if legacy_value and reading_value and legacy_value == reading_value:
                    per_reader[key]["match"] += 1
                elif legacy_value and reading_value and legacy_value != reading_value:
                    per_reader[key]["mismatch"] += 1
                    changed[key] = {"legacy": legacy_value, "reading": reading_value}
                elif legacy_value and not reading_value:
                    per_reader[key]["legacy_only"] += 1
                elif reading_value and not legacy_value:
                    per_reader[key]["reading_only"] += 1
                else:
                    per_reader[key]["match"] += 1

            if changed:
                mismatch_count += 1
                if len(examples) < 50:
                    examples.append(
                        {
                            "dialog_id": dialog_id,
                            "turn": int(turn.get("turn") or index),
                            "client_message": client_message,
                            "changed": changed,
                        }
                    )
    return {
        "schema_version": "semantic_reading_reader_agreement_v1_2026_07_03",
        "status": "compared" if compared else "no_frames",
        "compared_turns": compared,
        "per_reader": per_reader,
        "mismatch_count": mismatch_count,
        "mismatch_examples": examples,
    }


_NUMBER_RE = re.compile(r"(?<!\w)(?:\d[\d\s\u00a0.,]*\d|\d)(?:\s*(?:₽|руб|%|класс|кл\.?))?", re.I)


def _numbers(text: Any) -> set[str]:
    return {" ".join(item.group(0).split()).casefold() for item in _NUMBER_RE.finditer(str(text or ""))}


def _baseline_vs_inline_text_health(
    baseline_dialogs: Sequence[Mapping[str, Any]],
    inline_dialogs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_map = _turn_map(baseline_dialogs)
    inline_map = _turn_map(inline_dialogs)
    common = sorted(set(baseline_map) & set(inline_map))
    dangerous_examples: list[dict[str, Any]] = []
    changed_lengths: list[int] = []
    baseline_lengths: list[int] = []
    p0_flip_count = 0
    route_text_diff_count = 0
    new_number_count = 0
    for key in common:
        baseline_turn = baseline_map[key]
        inline_turn = inline_map[key]
        base_text = str(baseline_turn.get("bot_text") or "")
        inline_text = str(inline_turn.get("bot_text") or "")
        if baseline_turn.get("bot_route") != inline_turn.get("bot_route") or base_text != inline_text:
            route_text_diff_count += 1
            baseline_lengths.append(len(base_text))
            changed_lengths.append(len(inline_text))
        reasons: list[str] = []
        if _actual_p0_signal(baseline_turn) and not _actual_p0_signal(inline_turn):
            p0_flip_count += 1
            reasons.append("p0_signal_lost")
        new_numbers = sorted(_numbers(inline_text) - _numbers(base_text))
        if new_numbers:
            new_number_count += 1
            reasons.append("new_number")
        if reasons and len(dangerous_examples) < 50:
            dangerous_examples.append(
                {
                    "dialog_id": key[0],
                    "turn": key[1],
                    "reasons": reasons,
                    "baseline_route": baseline_turn.get("bot_route"),
                    "inline_route": inline_turn.get("bot_route"),
                    "new_numbers": new_numbers[:8],
                }
            )
    median_delta_pct: float | None = None
    if baseline_lengths and changed_lengths:
        base_median = median(baseline_lengths)
        if base_median:
            median_delta_pct = round((median(changed_lengths) - base_median) / base_median, 4)
    return {
        "schema_version": "baseline_vs_inline_text_health_v1_2026_07_03",
        "status": "compared",
        "compared_turns": len(common),
        "missing_baseline_turns": len(set(inline_map) - set(baseline_map)),
        "missing_inline_turns": len(set(baseline_map) - set(inline_map)),
        "route_text_diff_count": route_text_diff_count,
        "p0_signal_lost_count": p0_flip_count,
        "new_number_diff_count": new_number_count,
        "dangerous_flip_count": p0_flip_count + new_number_count,
        "changed_text_median_delta_pct": median_delta_pct,
        "dangerous_flip_examples": dangerous_examples,
    }


def _inline_text_health_gate(
    baseline_dialogs: Sequence[Mapping[str, Any]],
    inline_dialogs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_map = _turn_map(baseline_dialogs)
    inline_map = _turn_map(inline_dialogs)
    client_history_numbers = _client_history_number_map(inline_dialogs)
    timeout_dialog_ids = _timeout_dialog_ids(baseline_dialogs) | _timeout_dialog_ids(inline_dialogs)
    common = sorted(set(baseline_map) & set(inline_map))
    missing_baseline = sorted(set(inline_map) - set(baseline_map))
    missing_inline = sorted(set(baseline_map) - set(inline_map))
    missing_baseline_explained, missing_baseline_unexplained = _split_missing_turns_by_timeout(
        missing_baseline,
        timeout_dialog_ids,
    )
    missing_inline_explained, missing_inline_unexplained = _split_missing_turns_by_timeout(
        missing_inline,
        timeout_dialog_ids,
    )
    p0_route_lost: list[dict[str, Any]] = []
    p0_hygiene_lost: list[dict[str, Any]] = []
    p0_hygiene_added: list[dict[str, Any]] = []
    new_number_unverified: list[dict[str, Any]] = []
    new_number_adjacent_warning: list[dict[str, Any]] = []
    route_flip_dangerous: list[dict[str, Any]] = []
    new_number_verified = 0
    number_source_counts: Counter[str] = Counter()

    for key in common:
        baseline_turn = baseline_map[key]
        inline_turn = inline_map[key]
        baseline_route = str(baseline_turn.get("bot_route") or "")
        inline_route = str(inline_turn.get("bot_route") or "")
        baseline_handoff = _actual_route_handoff(baseline_turn)
        inline_handoff = _actual_route_handoff(inline_turn)
        baseline_p0 = _actual_p0_signal(baseline_turn)
        inline_p0 = _actual_p0_signal(inline_turn)
        example_base = {
            "dialog_id": key[0],
            "turn": key[1],
            "baseline_route": baseline_route,
            "inline_route": inline_route,
        }
        if baseline_handoff and not inline_handoff:
            route_flip_dangerous.append(dict(example_base))
            if baseline_p0:
                p0_route_lost.append(dict(example_base))
        elif baseline_p0 != inline_p0:
            example = {
                **example_base,
                "baseline_p0_signal": baseline_p0,
                "inline_p0_signal": inline_p0,
            }
            if baseline_p0 and not inline_p0:
                p0_hygiene_lost.append(example)
            else:
                p0_hygiene_added.append(example)

        baseline_numbers = _numbers(baseline_turn.get("bot_text"))
        inline_numbers = _numbers(inline_turn.get("bot_text"))
        classified = _classify_new_numbers(
            inline_turn,
            inline_numbers - baseline_numbers,
            client_history_numbers.get(key, set()),
        )
        for source, numbers in classified.items():
            number_source_counts[source] += len(numbers)
        verified = sorted(
            number
            for source, numbers in classified.items()
            if source in {"audit", "exact_fact", "client_current", "client_history"}
            for number in numbers
        )
        adjacent = sorted(classified.get("adjacent_fact_warning", ()))
        unverified = sorted(classified.get("unverified", ()))
        if verified:
            new_number_verified += 1
        if adjacent:
            new_number_adjacent_warning.append(
                {
                    **example_base,
                    "new_numbers": adjacent[:8],
                    "verified_new_numbers": verified[:8],
                }
            )
        if unverified:
            new_number_unverified.append(
                {
                    **example_base,
                    "new_numbers": unverified[:8],
                    "verified_new_numbers": verified[:8],
                }
            )

    fail = bool(p0_route_lost or new_number_unverified or route_flip_dangerous)
    review = bool(
        missing_baseline_unexplained
        or missing_inline_unexplained
        or p0_hygiene_lost
        or p0_hygiene_added
        or new_number_adjacent_warning
    )
    status = "fail" if fail else ("needs_review" if review else "pass")
    p0_hygiene_flag_diff = p0_hygiene_lost + p0_hygiene_added
    return {
        "schema_version": "inline_text_health_gate_v1_1_2026_07_03",
        "status": status,
        "compared_turns": len(common),
        "missing_baseline_turns": len(missing_baseline),
        "missing_baseline_explained_count": len(missing_baseline_explained),
        "missing_baseline_unexplained_count": len(missing_baseline_unexplained),
        "missing_baseline_explained_examples": missing_baseline_explained[:25],
        "missing_baseline_unexplained_examples": missing_baseline_unexplained[:25],
        "missing_inline_turns": len(missing_inline),
        "missing_inline_explained_count": len(missing_inline_explained),
        "missing_inline_unexplained_count": len(missing_inline_unexplained),
        "missing_inline_explained_examples": missing_inline_explained[:25],
        "missing_inline_unexplained_examples": missing_inline_unexplained[:25],
        "p0_route_lost_count": len(p0_route_lost),
        "p0_route_lost_examples": p0_route_lost[:25],
        "p0_hygiene_flag_diff_count": len(p0_hygiene_flag_diff),
        "p0_hygiene_flag_diff_examples": p0_hygiene_flag_diff[:25],
        "p0_hygiene_lost_count": len(p0_hygiene_lost),
        "p0_hygiene_lost_examples": p0_hygiene_lost[:25],
        "p0_hygiene_added_count": len(p0_hygiene_added),
        "p0_hygiene_added_examples": p0_hygiene_added[:25],
        "new_number_verified_turn_count": new_number_verified,
        "number_verified_by_audit_count": number_source_counts["audit"],
        "number_verified_by_exact_fact_count": number_source_counts["exact_fact"],
        "number_verified_by_client_current_count": number_source_counts["client_current"],
        "number_verified_by_client_history_count": number_source_counts["client_history"],
        "number_adjacent_warning_count": number_source_counts["adjacent_fact_warning"],
        "number_unverified_claim_count": number_source_counts["unverified"],
        "new_number_unverified_count": len(new_number_unverified),
        "new_number_unverified_examples": new_number_unverified[:25],
        "new_number_adjacent_warning_count": len(new_number_adjacent_warning),
        "new_number_adjacent_warning_examples": new_number_adjacent_warning[:25],
        "route_flip_dangerous_count": len(route_flip_dangerous),
        "route_flip_dangerous_examples": route_flip_dangerous[:25],
    }


def _verified_number_claims_from_audit(turn: Mapping[str, Any]) -> set[str]:
    audit = turn.get("number_audit") if isinstance(turn.get("number_audit"), Mapping) else {}
    verified_levels = {"client_echo", "retrieved_match", "same_brand_global_match"}
    result: set[str] = set()
    for item in audit.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("level") or "").strip() not in verified_levels:
            continue
        result.update(_numbers(item.get("claim_text")))
    return result


def _classify_new_numbers(
    turn: Mapping[str, Any],
    numbers: set[str],
    client_history_numbers: set[str],
) -> dict[str, set[str]]:
    classified: dict[str, set[str]] = {
        "audit": set(),
        "exact_fact": set(),
        "client_current": set(),
        "client_history": set(),
        "adjacent_fact_warning": set(),
        "unverified": set(),
    }
    if not numbers:
        return classified
    audit_numbers = _verified_number_claims_from_audit(turn)
    exact_fact_numbers = _selected_fact_numbers(turn, "selected_exact_ids")
    adjacent_fact_numbers = _selected_fact_numbers(turn, "selected_adjacent_ids")
    current_client_numbers = _numbers(turn.get("client_message"))
    for number in numbers:
        if _number_in(number, audit_numbers):
            classified["audit"].add(number)
        elif _number_in(number, exact_fact_numbers):
            classified["exact_fact"].add(number)
        elif _number_in(number, current_client_numbers):
            classified["client_current"].add(number)
        elif _number_in(number, client_history_numbers):
            classified["client_history"].add(number)
        elif _number_in(number, adjacent_fact_numbers):
            classified["adjacent_fact_warning"].add(number)
        else:
            classified["unverified"].add(number)
    return classified


def _number_in(number: str, candidates: set[str]) -> bool:
    return number in candidates or any(_number_equivalent(number, candidate) for candidate in candidates)


def _number_equivalent(left: str, right: str) -> bool:
    left_norm = str(left or "").casefold().replace("\u00a0", " ").strip()
    right_norm = str(right or "").casefold().replace("\u00a0", " ").strip()
    if left_norm == right_norm:
        return True
    left_digits = re.sub(r"\D+", "", left_norm)
    right_digits = re.sub(r"\D+", "", right_norm)
    if not left_digits or left_digits != right_digits:
        return False
    left_is_class = "класс" in left_norm or "кл" in left_norm
    right_is_class = "класс" in right_norm or "кл" in right_norm
    left_is_bare = bool(re.fullmatch(r"\d{1,2}", left_digits)) and not any(ch in left_norm for ch in "₽%.,")
    right_is_bare = bool(re.fullmatch(r"\d{1,2}", right_digits)) and not any(ch in right_norm for ch in "₽%.,")
    return (left_is_class and right_is_bare) or (right_is_class and left_is_bare)


def _selected_fact_numbers(turn: Mapping[str, Any], selection_key: str) -> set[str]:
    trace = turn.get("bot_fact_retrieval_trace") if isinstance(turn.get("bot_fact_retrieval_trace"), Mapping) else {}
    selected_ids = {str(item) for item in trace.get(selection_key) or [] if str(item or "").strip()}
    if selection_key == "selected_exact_ids":
        for field in ("context_used", "fact_refs", "bot_context_used", "bot_fact_refs"):
            value = turn.get(field)
            if isinstance(value, list):
                selected_ids.update(str(item) for item in value if str(item or "").strip())
    if not selected_ids:
        return set()
    direct = _direct_path_from_turn(turn)
    retrieved = direct.get("retrieved_facts") if isinstance(direct.get("retrieved_facts"), Mapping) else {}
    texts: list[Any] = []
    for fact_id in selected_ids:
        if fact_id in retrieved:
            texts.append(retrieved[fact_id])
    for item in turn.get("bot_confirmed_facts") or []:
        if not isinstance(item, str):
            continue
        prefix = item.split(":", 1)[0].strip()
        if item.startswith("fact:v3:"):
            prefix = ":".join(item.split(":", 4)[:4])
        if any(item.startswith(f"{fact_id}:") or prefix == fact_id for fact_id in selected_ids):
            texts.append(item)
    result = _numbers(" ".join(str(text) for text in texts))
    if selection_key == "selected_exact_ids":
        result.update(_academic_year_numbers_from_fact_ids(selected_ids))
    return result


def _academic_year_numbers_from_fact_ids(fact_ids: set[str]) -> set[str]:
    result: set[str] = set()
    for fact_id in fact_ids:
        for match in re.finditer(r"(?<!\d)(20\d{2})[_/-](\d{2})(?!\d)", str(fact_id or "")):
            result.add(match.group(1))
            result.add(match.group(2))
            result.add(f"{match.group(1)}/{match.group(2)}")
    return result


def _client_history_number_map(dialogs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], set[str]]:
    result: dict[tuple[str, int], set[str]] = {}
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        seen: set[str] = set()
        for index, turn in enumerate(_turns(dialog), 1):
            turn_no = int(turn.get("turn") or index)
            result[(dialog_id, turn_no)] = set(seen)
            seen.update(_numbers(turn.get("client_message")))
    return result


def _timeout_dialog_ids(dialogs: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for dialog in dialogs:
        if str(dialog.get("run_status") or "").strip().casefold() == "timeout" or any(
            _is_provider_timeout_turn(turn) for turn in _turns(dialog)
        ):
            result.add(str(dialog.get("dialog_id") or ""))
    return result


def _split_missing_turns_by_timeout(
    missing: Sequence[tuple[str, int]],
    timeout_dialog_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    explained: list[dict[str, Any]] = []
    unexplained: list[dict[str, Any]] = []
    for dialog_id, turn_no in missing:
        item = {"dialog_id": dialog_id, "turn": turn_no}
        if dialog_id in timeout_dialog_ids:
            explained.append({**item, "reason": "provider_timeout_dialog"})
        else:
            unexplained.append(item)
    return explained, unexplained


def _llm_call_delta(off_summary: Mapping[str, Any], on_summary: Mapping[str, Any]) -> dict[str, Any]:
    off_calls = off_summary.get("llm_calls") if isinstance(off_summary.get("llm_calls"), Mapping) else {}
    on_calls = on_summary.get("llm_calls") if isinstance(on_summary.get("llm_calls"), Mapping) else {}
    off_total = _int_value(off_calls.get("total")) if off_calls else None
    on_total = _int_value(on_calls.get("total")) if on_calls else None
    on_frame = _int_value(on_calls.get("bot_semantic_frame_shadow")) if on_calls else 0
    off_frame = _int_value(off_calls.get("bot_semantic_frame_shadow")) if off_calls else 0
    raw_total_delta = (on_total - off_total) if off_total is not None and on_total is not None else None
    enrichment = on_summary.get("semantic_frame_enrichment") if isinstance(on_summary.get("semantic_frame_enrichment"), Mapping) else {}
    enrichment_status = str(enrichment.get("status") or ("all" if on_summary.get("semantic_frame_enriched") else "none"))
    on_non_frame_total = max((on_total or 0) - on_frame, 0)
    if enrichment_status == "all":
        extra_total = on_total
        extra_frame = on_frame
        mode = "semantic_frame_enrichment"
    elif enrichment_status == "partial":
        extra_total = on_total
        extra_frame = on_frame
        mode = "semantic_frame_enrichment_partial"
    else:
        extra_total = raw_total_delta
        extra_frame = (on_frame - off_frame) if off_calls and on_calls else None
        mode = "paired_full_run"
    return {
        "mode": mode,
        "enrichment_status": enrichment_status,
        "off_total": off_total,
        "on_total": on_total,
        "on_non_frame_total": on_non_frame_total,
        "raw_total_delta": raw_total_delta,
        "extra_total": extra_total,
        "extra_semantic_frame_shadow": extra_frame,
        "off": dict(off_calls),
        "on": dict(on_calls),
    }


def _semantic_frame_metrics(dialogs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    turns_total = 0
    present = 0
    eligible_turns = 0
    eligible_present = 0
    preblocked_p0 = 0
    timeouts = 0
    model_not_called = 0
    complete_required = 0
    missing_required: Counter[str] = Counter()
    must_handoff = Counter()
    route_alignment = Counter()
    p0_alignment = Counter()
    confidence_values: list[float] = []
    mismatches: list[dict[str, Any]] = []
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for turn in _turns(dialog):
            turns_total += 1
            eligible = _is_frame_emission_eligible_turn(turn)
            if _is_direct_path_preblocked_p0_turn(turn):
                preblocked_p0 += 1
            if _is_provider_timeout_turn(turn):
                timeouts += 1
            if not _is_model_called_turn(turn):
                model_not_called += 1
            if eligible:
                eligible_turns += 1
            frame = turn.get("bot_semantic_frame")
            if not isinstance(frame, Mapping) or not frame:
                continue
            present += 1
            if eligible:
                eligible_present += 1
            missing = [field for field in REQUIRED_FRAME_FIELDS if field not in frame]
            frame_must = _strict_bool(frame.get("must_handoff"))
            if frame_must is None:
                missing.append("must_handoff:invalid_bool")
            if missing:
                missing_required.update(missing)
            else:
                complete_required += 1
            if frame_must is None:
                must_handoff["invalid"] += 1
            else:
                must_handoff["true" if frame_must else "false"] += 1
            route_handoff = _actual_route_handoff(turn)
            p0_signal = _actual_p0_signal(turn)
            route_key = "invalid_frame" if frame_must is None else ("match" if frame_must == route_handoff else "mismatch")
            p0_key = "invalid_frame" if frame_must is None else ("match" if frame_must == p0_signal else "mismatch")
            route_alignment[route_key] += 1
            p0_alignment[p0_key] += 1
            if route_key == "mismatch" or p0_key == "mismatch":
                mismatches.append(
                    {
                        "dialog_id": dialog_id,
                        "turn": turn.get("turn"),
                        "bot_route": turn.get("bot_route"),
                        "frame_must_handoff": frame_must,
                        "actual_route_handoff": route_handoff,
                        "actual_p0_signal": p0_signal,
                        "risk_class": frame.get("risk_class"),
                        "answerability": frame.get("answerability"),
                        "intent": frame.get("intent"),
                    }
                )
            confidence = _float_value(frame.get("confidence"))
            if confidence is not None:
                confidence_values.append(confidence)
    return {
        "turns_total": turns_total,
        "present_count": present,
        "missing_count": turns_total - present,
        "present_rate": _ratio(present, turns_total),
        "preblocked_p0_count": preblocked_p0,
        "provider_timeout_count": timeouts,
        "infra_timeout_present": timeouts > 0,
        "model_not_called_count": model_not_called,
        "eligible_model_called_turns": eligible_turns,
        "eligible_frame_count": eligible_present,
        "eligible_missing_count": eligible_turns - eligible_present,
        "eligible_frame_rate": _ratio(eligible_present, eligible_turns),
        "eligible_frame_threshold": FRAME_EMISSION_THRESHOLD,
        "complete_required_count": complete_required,
        "complete_required_rate": _ratio(complete_required, present),
        "missing_required_fields": dict(missing_required),
        "must_handoff": dict(must_handoff),
        "must_handoff_vs_route": dict(route_alignment),
        "must_handoff_vs_p0_signal": dict(p0_alignment),
        "confidence": {
            "count": len(confidence_values),
            "avg": round(mean(confidence_values), 4) if confidence_values else None,
            "min": min(confidence_values) if confidence_values else None,
            "max": max(confidence_values) if confidence_values else None,
        },
        "mismatch_examples": mismatches[:50],
    }


def _frame_decision_shadow_metrics(dialogs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    handoff_alignment: Counter[str] = Counter()
    p0_alignment: Counter[str] = Counter()
    action_alignment: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    turns = 0
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for turn in _turns(dialog):
            shadow = turn.get("bot_frame_decision_shadow")
            if not isinstance(shadow, Mapping) or not shadow:
                continue
            turns += 1
            status_counts[str(shadow.get("status") or "unknown")] += 1
            comparisons = shadow.get("comparisons") if isinstance(shadow.get("comparisons"), Mapping) else {}
            handoff = str(comparisons.get("must_handoff_vs_route") or "unknown")
            p0 = str(comparisons.get("p0_vs_actual") or "unknown")
            handoff_alignment[handoff] += 1
            p0_alignment[p0] += 1
            action = comparisons.get("action") if isinstance(comparisons.get("action"), Mapping) else {}
            action_alignment[str(action.get("status") or "unknown")] += 1
            if handoff == "mismatch" or p0 == "mismatch":
                examples.append(
                    {
                        "dialog_id": dialog_id,
                        "turn": turn.get("turn"),
                        "status": shadow.get("status"),
                        "comparisons": comparisons,
                    }
                )
    return {
        "turn_count": turns,
        "status": dict(status_counts),
        "must_handoff_vs_route": dict(handoff_alignment),
        "p0_vs_actual": dict(p0_alignment),
        "action_status": dict(action_alignment),
        "mismatch_examples": examples[:50],
    }


def _semantic_frame_proof_reconciliation_shadow_metrics(dialogs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    proof_status_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    turns = 0
    would_reconcile = 0
    active_allowed = 0
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for turn in _turns(dialog):
            shadow = turn.get("bot_semantic_frame_proof_reconciliation_shadow")
            if not isinstance(shadow, Mapping) or not shadow:
                direct = turn.get("bot_direct_path") if isinstance(turn.get("bot_direct_path"), Mapping) else {}
                shadow = direct.get("semantic_frame_proof_reconciliation_shadow") if isinstance(direct, Mapping) else {}
            if not isinstance(shadow, Mapping) or not shadow:
                continue
            turns += 1
            status = str(shadow.get("status") or "unknown")
            reason = str(shadow.get("reason") or "unknown")
            status_counts[status] += 1
            reason_counts[reason] += 1
            proof_status_counts[str(shadow.get("proof_status") or "unknown")] += 1
            if bool(shadow.get("active_behavior_allowed")):
                active_allowed += 1
            if status == "would_reconcile_to_safe_reference":
                would_reconcile += 1
                if len(examples) < 50:
                    examples.append(
                        {
                            "dialog_id": dialog_id,
                            "turn": turn.get("turn"),
                            "bot_route": turn.get("bot_route"),
                            "reason": reason,
                            "proof_status": shadow.get("proof_status"),
                            "exact_fact_keys": list(shadow.get("exact_fact_keys") or [])[:5]
                            if isinstance(shadow.get("exact_fact_keys"), list)
                            else [],
                            "active_blockers": list(shadow.get("active_blockers") or [])[:8]
                            if isinstance(shadow.get("active_blockers"), list)
                            else [],
                        }
                    )
    return {
        "schema_version": "semantic_frame_proof_reconciliation_shadow_metrics_v1_2026_07_02",
        "turn_count": turns,
        "status": dict(status_counts),
        "reasons": dict(reason_counts),
        "proof_status": dict(proof_status_counts),
        "would_reconcile_count": would_reconcile,
        "active_allowed_count": active_allowed,
        "candidate_examples": examples,
    }


def _semantic_frame_self_answer_shadow_metrics(dialogs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    candidate_by_class: Counter[str] = Counter()
    unsafe_examples: list[dict[str, Any]] = []
    candidate_examples: list[dict[str, Any]] = []
    turns = 0
    would_demote = 0
    p0_lowered = 0
    manager_only_lowered = 0
    money_lowered = 0
    operational_lowered = 0
    freshness_unknown = 0
    partial_freshness = 0
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for turn in _turns(dialog):
            shadow = turn.get("bot_semantic_frame_self_answer_shadow")
            if not isinstance(shadow, Mapping) or not shadow:
                continue
            turns += 1
            status = str(shadow.get("status") or "unknown")
            reason = str(shadow.get("reason") or "unknown")
            self_class = str(shadow.get("self_class") or "unknown")
            status_counts[status] += 1
            reason_counts[reason] += 1
            class_counts[self_class] += 1
            if status != "would_demote_to_self":
                continue
            would_demote += 1
            candidate_by_class[self_class] += 1
            guards = shadow.get("guards") if isinstance(shadow.get("guards"), Mapping) else {}
            freshness = guards.get("freshness") if isinstance(guards.get("freshness"), Mapping) else {}
            row = {
                "dialog_id": dialog_id,
                "turn": turn.get("turn"),
                "brand": dialog.get("brand"),
                "bot_route": turn.get("bot_route"),
                "self_class": self_class,
                "reason": reason,
                "route_after_if_active": shadow.get("route_after_if_active"),
            }
            if _actual_p0_signal(turn):
                p0_lowered += 1
                unsafe_examples.append({**row, "unsafe_reason": "actual_p0_signal"})
            if str(turn.get("bot_route") or "") == "manager_only":
                manager_only_lowered += 1
                unsafe_examples.append({**row, "unsafe_reason": "manager_only_route"})
            if _money_signal(shadow):
                money_lowered += 1
                unsafe_examples.append({**row, "unsafe_reason": "money_signal"})
            if _operational_signal(shadow):
                operational_lowered += 1
                unsafe_examples.append({**row, "unsafe_reason": "operational_signal"})
            if not bool(freshness.get("ok")):
                freshness_unknown += 1
                unsafe_examples.append({**row, "unsafe_reason": "freshness_unknown"})
            exact_fact_count = _safe_int(freshness.get("exact_fact_count"))
            fresh_client_safe_count = _safe_int(freshness.get("fresh_client_safe_count"))
            if bool(freshness.get("ok")) and exact_fact_count > fresh_client_safe_count:
                partial_freshness += 1
                unsafe_examples.append(
                    {
                        **row,
                        "unsafe_reason": "partial_freshness",
                        "exact_fact_count": exact_fact_count,
                        "fresh_client_safe_count": fresh_client_safe_count,
                    }
                )
            if len(candidate_examples) < 50:
                candidate_examples.append(row)
    return {
        "schema_version": "semantic_frame_self_answer_shadow_metrics_v1_2026_07_02",
        "turn_count": turns,
        "status": dict(status_counts),
        "reasons": dict(reason_counts),
        "self_classes": dict(class_counts),
        "would_demote_count": would_demote,
        "would_demote_by_class": dict(candidate_by_class),
        "p0_lowered_count": p0_lowered,
        "manager_only_lowered_count": manager_only_lowered,
        "money_lowered_count": money_lowered,
        "operational_lowered_count": operational_lowered,
        "freshness_unknown_self_candidates": freshness_unknown,
        "partial_freshness_self_candidates": partial_freshness,
        "unsafe_candidate_examples": unsafe_examples[:50],
        "candidate_examples": candidate_examples,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _money_signal(shadow: Mapping[str, Any]) -> bool:
    frame = shadow.get("frame") if isinstance(shadow.get("frame"), Mapping) else {}
    payment_readiness = str(frame.get("payment_readiness") or "").strip().casefold()
    requested_action = str(frame.get("requested_action") or "").strip().casefold()
    return payment_readiness in MONEY_PAYMENT_READINESS_MARKERS or requested_action in MONEY_REQUESTED_ACTION_MARKERS


def _operational_signal(shadow: Mapping[str, Any]) -> bool:
    frame = shadow.get("frame") if isinstance(shadow.get("frame"), Mapping) else {}
    requested_action = str(frame.get("requested_action") or "").strip().casefold()
    deal_stage = str(frame.get("deal_stage") or "").strip().casefold()
    return requested_action in OPERATIONAL_REQUESTED_ACTION_MARKERS or deal_stage in OPERATIONAL_DEAL_STAGE_MARKERS


def _actual_route_handoff(turn: Mapping[str, Any]) -> bool:
    route = str(turn.get("bot_route") or "")
    return route in {"manager_only", "draft_for_manager"}


def _actual_p0_signal(turn: Mapping[str, Any]) -> bool:
    route = str(turn.get("bot_route") or "")
    flags = " ".join(str(flag) for flag in (turn.get("bot_safety_flags") or [])).casefold()
    direct_path = turn.get("bot_direct_path") if isinstance(turn.get("bot_direct_path"), Mapping) else {}
    model_p0 = turn.get("bot_direct_path_model_p0") if isinstance(turn.get("bot_direct_path_model_p0"), Mapping) else {}
    plan = turn.get("bot_conversation_intent_plan") if isinstance(turn.get("bot_conversation_intent_plan"), Mapping) else {}
    risk_codes = " ".join(str(code) for code in (plan.get("risk_codes") or [])).casefold()
    direct_p0 = direct_path.get("direct_path_model_p0") if isinstance(direct_path.get("direct_path_model_p0"), Mapping) else {}
    return (
        route == "manager_only"
        or any(marker in flags for marker in P0_FLAG_MARKERS)
        or any(marker in risk_codes for marker in P0_FLAG_MARKERS)
        or _strict_bool(model_p0.get("is_p0")) is True
        or _strict_bool(direct_p0.get("is_p0")) is True
    )


def _strict_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _acceptance(report: Mapping[str, Any]) -> dict[str, Any]:
    diff = report.get("off_on_diff") if isinstance(report.get("off_on_diff"), Mapping) else {}
    llm = report.get("llm_calls") if isinstance(report.get("llm_calls"), Mapping) else {}
    frame = report.get("semantic_frame") if isinstance(report.get("semantic_frame"), Mapping) else {}
    paired = report.get("paired_dialogs") if isinstance(report.get("paired_dialogs"), Mapping) else {}
    inline_gate = (
        report.get("inline_text_health_gate")
        if isinstance(report.get("inline_text_health_gate"), Mapping)
        else {}
    )
    self_shadow = (
        report.get("semantic_frame_self_answer_shadow")
        if isinstance(report.get("semantic_frame_self_answer_shadow"), Mapping)
        else {}
    )
    hard = report.get("hard_gate_failures") if isinstance(report.get("hard_gate_failures"), Mapping) else {}
    extra_total = llm.get("extra_total")
    extra_frame = llm.get("extra_semantic_frame_shadow")
    frame_present = int(frame.get("present_count") or 0)
    if llm.get("mode") == "semantic_frame_enrichment":
        expected_call_delta = (
            extra_total == extra_frame == frame_present
            and frame_present > 0
            and llm.get("on_total") == extra_total
            and llm.get("on_non_frame_total") == 0
        )
    elif llm.get("mode") == "semantic_frame_enrichment_partial":
        expected_call_delta = False
    else:
        expected_call_delta = extra_total == 0 or (extra_total == extra_frame == frame_present and frame_present > 0)
    flags = {
        "paired_dialogs_match": paired.get("status") in (None, "matched", "not_provided"),
        "inline_text_health_gate_ok": inline_gate.get("status") == "pass",
        "extra_model_calls_expected": expected_call_delta,
        "semantic_frame_eligible_rate_ok": (
            bool(frame.get("eligible_model_called_turns"))
            and float(frame.get("eligible_frame_rate") or 0.0) >= FRAME_EMISSION_THRESHOLD
        ),
        "semantic_frame_required_fields_complete": frame.get("present_count") == frame.get("complete_required_count"),
        "self_answer_partial_freshness_zero": self_shadow.get("partial_freshness_self_candidates", 0) == 0,
        "hard_gate_failures_zero": hard.get("on") in (None, 0),
    }
    notes: list[str] = []
    if paired.get("status") == "mismatch":
        notes.append(
            "Baseline and inline dialog sets differ; route/text comparisons are limited to common dialog IDs "
            f"(common={paired.get('common_count', 0)}, baseline_only={paired.get('baseline_only_count', 0)}, "
            f"inline_only={paired.get('inline_only_count', 0)})."
        )
    if diff.get("status") != "compared":
        notes.append("OFF transcripts were not provided; route/text no-op cannot be proven by this report.")
    elif inline_gate.get("status") == "fail":
        notes.append("Inline text health gate failed; inspect p0_route_lost/new_number/route_flip examples.")
    elif inline_gate.get("status") == "needs_review":
        notes.append("Inline text health gate needs review; inspect hygiene, adjacent-number, or missing-turn examples.")
    if llm.get("extra_total") is None:
        notes.append("OFF/ON summary pair was not provided; extra model call delta cannot be proven by this report.")
    elif llm.get("mode") == "semantic_frame_enrichment":
        if expected_call_delta:
            notes.append("ON run is paired SemanticFrame enrichment; model calls are only post-hoc frame metadata calls.")
        else:
            notes.append("SemanticFrame enrichment run made non-frame model calls or did not cover every ON turn.")
    elif llm.get("mode") == "semantic_frame_enrichment_partial":
        notes.append("SemanticFrame enrichment is partial; every ON turn must be enriched for strict no-op acceptance.")
    elif extra_total not in (0, extra_frame):
        notes.append("Extra model calls are not fully explained by post-hoc SemanticFrame shadow calls.")
    elif extra_total == extra_frame and extra_total:
        notes.append("Extra model calls are expected post-hoc SemanticFrame shadow calls.")
    if not flags["semantic_frame_eligible_rate_ok"]:
        notes.append(
            "SemanticFrame eligible emission is below threshold on model-called turns "
            f"({frame.get('eligible_frame_count', 0)}/{frame.get('eligible_model_called_turns', 0)})."
        )
    if frame.get("infra_timeout_present"):
        notes.append("Provider timeout is present; this is an infra marker, not a frame-quality miss.")
    if not flags["self_answer_partial_freshness_zero"]:
        notes.append("At least one self-answer shadow candidate has only partial freshness/client-safe fact coverage.")
    status = "pass" if all(flags.values()) else "needs_review"
    return {"status": status, "flags": flags, "notes": notes}


def _decision_readiness(report: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    return {
        "technical_shadow_status": "pass" if acceptance.get("status") == "pass" else "needs_review",
        "semantic_decision_status": "not_pass",
        "active_behavior_allowed": False,
        "reason": "SemanticFrame has no filled expected-frame gold labels in this report.",
    }


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


if __name__ == "__main__":
    raise SystemExit(main())
