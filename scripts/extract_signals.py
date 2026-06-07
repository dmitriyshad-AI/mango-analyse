#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SELF_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot", "answer_self"}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _transcript_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "dynamic_dialog_transcripts.jsonl"
    return candidate.resolve(strict=False)


def _read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    resolved = _transcript_path(path)
    dialogs: list[Mapping[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, Mapping):
                dialogs.append(item)
    return dialogs


def _lookup(obj: Mapping[str, Any], *path: str) -> Any:
    current: Any = obj
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_present(turn: Mapping[str, Any], paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        value = _lookup(turn, *path)
        if value not in (None, "", [], {}):
            return value
    return None


def _turn_route(turn: Mapping[str, Any]) -> str:
    value = _first_present(
        turn,
        (
            ("bot_route",),
            ("route",),
            ("bot_answer_contract", "route"),
            ("bot_dialogue_contract_pipeline", "route"),
            ("bot_dialogue_contract_pipeline", "contract", "route"),
            ("bot_dialogue_contract_pipeline", "contract", "decision"),
        ),
    )
    return str(value or "unknown")


def _fallback_reason(turn: Mapping[str, Any]) -> str:
    value = _first_present(
        turn,
        (
            ("fallback_reason",),
            ("bot_fallback_reason",),
            ("bot_dialogue_contract_pipeline", "fallback_reason"),
            ("bot_dialogue_contract_pipeline", "reason_class"),
            ("bot_answer_contract", "fallback_reason"),
            ("bot_answer_contract", "reason_class"),
            ("bot_answer_contract", "metadata", "reason_class"),
        ),
    )
    return str(value or "")


def _handoff_trace(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _first_present(
        turn,
        (
            ("handoff_trace",),
            ("bot_handoff_trace",),
            ("bot_dialogue_contract_pipeline", "handoff_trace"),
            ("bot_dialogue_contract_pipeline", "metadata", "handoff_trace"),
            ("bot_answer_contract", "handoff_trace"),
            ("bot_answer_contract", "metadata", "handoff_trace"),
        ),
    )
    return _as_mapping(value)


def _count_facts(value: Any) -> int:
    if isinstance(value, Mapping):
        return len([key for key, item in value.items() if key and item])
    if isinstance(value, list):
        return len([item for item in value if item])
    return 0


def _fact_count(turn: Mapping[str, Any]) -> int:
    counts = [
        _count_facts(_lookup(turn, "bot_dialogue_contract_pipeline", "retrieved_facts")),
        _count_facts(_lookup(turn, "bot_dialogue_contract_pipeline", "retrieved_fact_keys")),
        _count_facts(turn.get("bot_confirmed_facts")),
        _count_facts(turn.get("bot_knowledge_snippets")),
        _count_facts(turn.get("retrieved_facts")),
        _count_facts(turn.get("retrieved_fact_keys")),
    ]
    return max(counts)


def _tone_score(turn: Mapping[str, Any]) -> float | None:
    value = _first_present(
        turn,
        (
            ("tone_score",),
            ("tone_metric", "tone_score"),
            ("bot_tone_score",),
            ("bot_tone_metric", "tone_score"),
            ("bot_answer_quality", "tone_score"),
        ),
    )
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _violated_gates(dialog: Mapping[str, Any]) -> list[str]:
    judge = _as_mapping(dialog.get("judge_result"))
    gates = judge.get("violated_gates")
    return [str(item) for item in _as_list(gates)]


def _route_is_handoff(route: str) -> bool:
    return route not in SELF_ROUTES and route != "unknown"


def summarize_transcripts(path: str | Path) -> dict[str, Any]:
    dialogs = _read_jsonl(path)
    routes: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    trace_layers: Counter[str] = Counter()
    trace_reasons: Counter[str] = Counter()
    gates: Counter[str] = Counter()
    by_brand: Counter[str] = Counter()
    over_handoff_rows: list[dict[str, Any]] = []
    tone_values: list[float] = []
    turns_total = 0

    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        brand = str(dialog.get("brand") or "unknown")
        by_brand[brand] += 1
        gates.update(_violated_gates(dialog))
        for turn in _as_list(dialog.get("turns")):
            if not isinstance(turn, Mapping):
                continue
            turns_total += 1
            route = _turn_route(turn)
            routes[route] += 1
            reason = _fallback_reason(turn)
            fallback_reasons[reason or "<empty>"] += 1
            trace = _handoff_trace(turn)
            if trace:
                trace_layers[str(trace.get("layer") or "<unknown>")] += 1
                trace_reasons[str(trace.get("fallback_reason") or trace.get("reason") or "<empty>")] += 1
            tone = _tone_score(turn)
            if tone is not None:
                tone_values.append(tone)
            fact_count = _fact_count(turn)
            if _route_is_handoff(route) and fact_count > 0:
                over_handoff_rows.append(
                    {
                        "dialog_id": dialog_id,
                        "turn": turn.get("turn"),
                        "brand": brand,
                        "route": route,
                        "fallback_reason": reason,
                        "fact_count": fact_count,
                        "trace_layer": trace.get("layer") if trace else None,
                        "trace_reason": trace.get("fallback_reason") or trace.get("reason") if trace else None,
                        "client_message": str(turn.get("client_message") or "")[:240],
                        "bot_text": str(turn.get("bot_text") or "")[:240],
                    }
                )

    hard_gate_failures = sum(1 for dialog in dialogs if _as_mapping(dialog.get("judge_result")).get("hard_gates_passed") is False)
    brand_gate_failures = sum(count for gate, count in gates.items() if "brand" in gate)
    p0_gate_failures = sum(count for gate, count in gates.items() if "p0" in gate or "payment" in gate or "refund" in gate)
    tone_avg = round(sum(tone_values) / len(tone_values), 2) if tone_values else None

    return {
        "path": str(_transcript_path(path)),
        "dialogs": len(dialogs),
        "turns": turns_total,
        "brands": dict(by_brand),
        "routes": dict(routes),
        "fallback_reasons": dict(fallback_reasons),
        "handoff_trace_layers": dict(trace_layers),
        "handoff_trace_reasons": dict(trace_reasons),
        "handoff_with_facts": len(over_handoff_rows),
        "handoff_with_facts_rows": over_handoff_rows,
        "tone_score_avg": tone_avg,
        "tone_score_count": len(tone_values),
        "violated_gates": dict(gates),
        "hard_gate_failures": hard_gate_failures,
        "brand_gate_failures": brand_gate_failures,
        "p0_gate_failures": p0_gate_failures,
    }


def compare_summaries(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def counter_delta(key: str) -> dict[str, int]:
        left_counter = Counter({str(k): int(v) for k, v in _as_mapping(left.get(key)).items()})
        right_counter = Counter({str(k): int(v) for k, v in _as_mapping(right.get(key)).items()})
        keys = sorted(set(left_counter) | set(right_counter))
        return {key: right_counter[key] - left_counter[key] for key in keys if right_counter[key] - left_counter[key]}

    left_tone = left.get("tone_score_avg")
    right_tone = right.get("tone_score_avg")
    tone_delta = None
    if isinstance(left_tone, (int, float)) and isinstance(right_tone, (int, float)):
        tone_delta = round(float(right_tone) - float(left_tone), 2)
    return {
        "left": left.get("path"),
        "right": right.get("path"),
        "dialogs_delta": int(right.get("dialogs") or 0) - int(left.get("dialogs") or 0),
        "turns_delta": int(right.get("turns") or 0) - int(left.get("turns") or 0),
        "routes_delta": counter_delta("routes"),
        "fallback_reasons_delta": counter_delta("fallback_reasons"),
        "handoff_with_facts_delta": int(right.get("handoff_with_facts") or 0) - int(left.get("handoff_with_facts") or 0),
        "tone_score_avg_delta": tone_delta,
        "hard_gate_failures_delta": int(right.get("hard_gate_failures") or 0) - int(left.get("hard_gate_failures") or 0),
        "brand_gate_failures_delta": int(right.get("brand_gate_failures") or 0) - int(left.get("brand_gate_failures") or 0),
        "p0_gate_failures_delta": int(right.get("p0_gate_failures") or 0) - int(left.get("p0_gate_failures") or 0),
    }


def _print_counter(title: str, values: Mapping[str, Any], *, limit: int = 30) -> None:
    print(title)
    for key, value in Counter({str(k): int(v) for k, v in values.items()}).most_common(limit):
        print(f"  {key}: {value}")


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"path: {summary['path']}")
    print(f"dialogs: {summary['dialogs']}")
    print(f"turns: {summary['turns']}")
    print(f"handoff_with_facts: {summary['handoff_with_facts']}")
    print(f"tone_score_avg: {summary['tone_score_avg']} ({summary['tone_score_count']} turns)")
    print(f"hard_gate_failures: {summary['hard_gate_failures']}")
    print(f"brand_gate_failures: {summary['brand_gate_failures']}")
    print(f"p0_gate_failures: {summary['p0_gate_failures']}")
    _print_counter("routes:", _as_mapping(summary.get("routes")))
    _print_counter("fallback_reasons:", _as_mapping(summary.get("fallback_reasons")))
    _print_counter("handoff_trace_layers:", _as_mapping(summary.get("handoff_trace_layers")))
    rows = _as_list(summary.get("handoff_with_facts_rows"))
    if rows:
        print("handoff_with_facts_examples:")
        for row in rows[:20]:
            print(
                f"  {row.get('dialog_id')} t{row.get('turn')}: "
                f"{row.get('route')} / {row.get('fallback_reason') or '<empty>'} / "
                f"{row.get('trace_layer') or '<no-trace>'} :: {row.get('client_message')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract route, handoff, gate and tone signals from dynamic bot transcripts.")
    parser.add_argument("run", help="Run directory or dynamic_dialog_transcripts.jsonl")
    parser.add_argument("--compare", help="Second run directory or transcript file for delta output")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = summarize_transcripts(args.run)
    if args.compare:
        other = summarize_transcripts(args.compare)
        payload = {"left": summary, "right": other, "compare": compare_summaries(summary, other)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("LEFT")
            print_summary(summary)
            print("\nRIGHT")
            print_summary(other)
            print("\nDELTA")
            print(json.dumps(payload["compare"], ensure_ascii=False, indent=2))
        return 0

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
