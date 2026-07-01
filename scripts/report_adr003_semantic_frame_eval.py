#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "adr003_semantic_frame_eval_report_v1_2026_07_01"
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ADR-003 SemanticFrame OFF/ON eval report from dynamic sim outputs.")
    parser.add_argument("--on-transcripts", type=Path, required=True)
    parser.add_argument("--on-summary", type=Path, default=None)
    parser.add_argument("--off-transcripts", type=Path, default=None)
    parser.add_argument("--off-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(
        on_transcripts=args.on_transcripts,
        on_summary=args.on_summary,
        off_transcripts=args.off_transcripts,
        off_summary=args.off_summary,
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
) -> dict[str, Any]:
    on_dialogs = _load_transcripts(on_transcripts)
    off_dialogs = _load_transcripts(off_transcripts) if off_transcripts else []
    on_summary_data = _load_json(on_summary)
    off_summary_data = _load_json(off_summary)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "on_transcripts": str(on_transcripts),
            "on_summary": str(on_summary or ""),
            "off_transcripts": str(off_transcripts or ""),
            "off_summary": str(off_summary or ""),
        },
        "totals": _dialog_totals(on_dialogs),
        "off_on_diff": _compare_off_on(off_dialogs, on_dialogs) if off_dialogs else {"status": "not_provided"},
        "llm_calls": _llm_call_delta(off_summary_data, on_summary_data),
        "semantic_frame": _semantic_frame_metrics(on_dialogs),
        "frame_decision_shadow": _frame_decision_shadow_metrics(on_dialogs),
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
    lines = [
        "# ADR-003 SemanticFrame Eval Report",
        "",
        f"- Acceptance: `{acceptance.get('status', 'unknown')}`",
        f"- Technical shadow status: `{(report.get('decision_readiness') or {}).get('technical_shadow_status', 'unknown')}`",
        f"- Semantic decision status: `{(report.get('decision_readiness') or {}).get('semantic_decision_status', 'unknown')}`",
        f"- Active behavior allowed: `{(report.get('decision_readiness') or {}).get('active_behavior_allowed', False)}`",
        f"- ON turns: `{frame.get('turns_total', 0)}`",
        f"- Frame present: `{frame.get('present_count', 0)}` / `{frame.get('turns_total', 0)}`",
        f"- Frame schema complete: `{frame.get('complete_required_count', 0)}` / `{frame.get('present_count', 0)}`",
        f"- OFF/ON route-text diffs: `{diff.get('route_text_diff_count', 'n/a')}`",
        f"- OFF/ON input diffs: `{diff.get('input_diff_count', 'n/a')}`",
        f"- LLM call mode: `{llm.get('mode', 'unknown')}`",
        f"- LLM raw total delta: `{llm.get('raw_total_delta', 'n/a')}`",
        f"- LLM expected extra calls: `{llm.get('extra_total', 'n/a')}`",
        f"- LLM non-frame ON calls: `{llm.get('on_non_frame_total', 'n/a')}`",
        f"- Frame decision shadow turns: `{shadow.get('turn_count', 0)}`",
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
            frame = turn.get("bot_semantic_frame")
            if not isinstance(frame, Mapping) or not frame:
                continue
            present += 1
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
        "route_text_diff_zero": (
            diff.get("status") == "compared"
            and diff.get("route_text_diff_count") == 0
            and diff.get("missing_off_turns") == 0
            and diff.get("missing_on_turns") == 0
        ),
        "input_turns_match": (
            diff.get("status") == "compared"
            and diff.get("input_diff_count") == 0
            and diff.get("missing_off_turns") == 0
            and diff.get("missing_on_turns") == 0
        ),
        "extra_model_calls_expected": expected_call_delta,
        "semantic_frame_present_on_all_turns": bool(frame.get("turns_total")) and frame.get("missing_count") == 0,
        "semantic_frame_required_fields_complete": frame.get("present_count") == frame.get("complete_required_count"),
        "hard_gate_failures_zero": hard.get("on") in (None, 0),
    }
    notes: list[str] = []
    if diff.get("status") != "compared":
        notes.append("OFF transcripts were not provided; route/text no-op cannot be proven by this report.")
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
    if not flags["semantic_frame_present_on_all_turns"]:
        notes.append("SemanticFrame is missing on at least one ON turn or ON turn count is zero.")
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
