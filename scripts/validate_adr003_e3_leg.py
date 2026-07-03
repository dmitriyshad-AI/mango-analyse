#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

INFRA_PROVIDER_ERRORS = {
    "timeout",
    "binary_not_found",
    "codex_binary_not_found",
    "invalid_json",
    "codex_error",
}
GATE_BLOCKED_PROVIDER_ERRORS = {
    "authoritative_output_gate_blocked",
    "hard_verification_failed",
    "identity_disclosure_guarded",
    "output_sanitizer_fallback",
    "semantic_verifier_downgrade",
}
GATE_BLOCKED_FLAGS = {
    "authoritative_output_gate_blocked",
    "identity_disclosure_guarded",
    "output_sanitizer_fallback",
}
P0_PREBLOCK_REASONS = {"p0_pre_gate", "direct_path_preblocked_p0"}
FRAME_EMISSION_THRESHOLD_NUMERATOR = 97
FRAME_EMISSION_THRESHOLD_DENOMINATOR = 100
TIMEOUT_TOLERANCE_MIN = 3
TIMEOUT_TOLERANCE_RATIO = 0.02


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one ADR-003 E3 dynamic-sim leg.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--leg", required=True)
    parser.add_argument("--expect-trace", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    result = validate_leg(
        summary_path=args.summary,
        transcripts_path=args.transcripts,
        leg=args.leg,
        expect_trace=args.expect_trace,
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_valid_line(result))
    return 0


def validate_leg(
    *,
    summary_path: Path,
    transcripts_path: Path,
    leg: str,
    expect_trace: bool,
) -> dict[str, Any]:
    if not summary_path.is_file():
        _fail(leg, f"missing summary: {summary_path}")
    if not transcripts_path.is_file():
        _fail(leg, f"missing transcripts: {transcripts_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = (
        summary.get("run_config", {})
        .get("key_flags", {})
        .get("profile", {})
    )
    if profile.get("env") != "pilot_gold_v1" or profile.get("effective") is not True:
        _fail(leg, f"pilot profile not active: {profile!r}")

    llm_calls = summary.get("llm_calls") or {}
    if int(llm_calls.get("bot_direct_draft") or 0) <= 0:
        _fail(leg, f"bot_direct_draft is not positive: {llm_calls!r}")

    dialogs = [
        json.loads(line)
        for line in transcripts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    turns = [turn for dialog in dialogs for turn in (dialog.get("turns") or [])]
    if not turns:
        _fail(leg, "no turns in transcripts")

    provider_error_values = [_provider_error(turn) for turn in turns]
    infra_provider_errors = [
        value for value in provider_error_values if _is_infra_provider_error(value) and value != "timeout"
    ]
    unknown_provider_errors = [
        value
        for value in provider_error_values
        if value and not _is_infra_provider_error(value) and value not in GATE_BLOCKED_PROVIDER_ERRORS
    ]
    if infra_provider_errors:
        _fail(leg, f"infra provider errors: {infra_provider_errors[:3]!r}")
    if unknown_provider_errors:
        _fail(leg, f"unknown provider errors: {unknown_provider_errors[:3]!r}")

    direct_turns = [
        turn
        for turn in turns
        if isinstance(turn.get("bot_direct_path"), dict) and turn.get("bot_direct_path")
    ]
    if len(direct_turns) != len(turns):
        _fail(leg, f"direct path metadata on {len(direct_turns)}/{len(turns)} turns")

    preblocked_p0 = [turn for turn in turns if _is_p0_preblocked(turn)]
    timeouts = [turn for turn in turns if _provider_error(turn) == "timeout"]
    timeout_dialogs = [dialog for dialog in dialogs if _is_timeout_dialog(dialog)]
    timeout_total = len(timeouts) + len(timeout_dialogs)
    timeout_budget = max(float(TIMEOUT_TOLERANCE_MIN), len(turns) * TIMEOUT_TOLERANCE_RATIO)
    if timeout_total > timeout_budget:
        _fail(
            leg,
            "timeout noise above tolerance "
            f"{timeout_total}/{timeout_budget:.2f} "
            f"(timeout_turns={len(timeouts)} timeout_dialogs={len(timeout_dialogs)})",
        )
    model_not_called = [turn for turn in turns if not _is_model_called(turn)]
    eligible_turns = [turn for turn in turns if _is_model_called(turn) and not _is_timeout(turn)]
    eligible_frame_turns = [turn for turn in eligible_turns if _has_frame(turn)]
    if not eligible_turns:
        _fail(leg, "no eligible model-called turns for semantic frame validation")
    eligible_frame_rate = len(eligible_frame_turns) / len(eligible_turns)
    if len(eligible_frame_turns) * FRAME_EMISSION_THRESHOLD_DENOMINATOR < (
        len(eligible_turns) * FRAME_EMISSION_THRESHOLD_NUMERATOR
    ):
        _fail(
            leg,
            "semantic frame eligible emission "
            f"{len(eligible_frame_turns)}/{len(eligible_turns)} "
            f"(preblocked_p0={len(preblocked_p0)} timeouts={len(timeouts)})",
        )

    trace_turns = [
        turn
        for turn in turns
        if isinstance(turn.get("bot_semantic_reading_trace"), list) and turn.get("bot_semantic_reading_trace")
    ]
    if expect_trace and not trace_turns:
        _fail(leg, "ON leg has no semantic_reading_trace records")
    if not expect_trace and trace_turns:
        _fail(leg, f"B leg unexpectedly has semantic_reading_trace on {len(trace_turns)} turns")

    gate_blocked_turns = [turn for turn in turns if _is_gate_blocked_turn(turn)]
    return {
        "schema_version": "adr003_e3_leg_validation_v2_2026_07_03",
        "leg": leg,
        "status": "valid",
        "dialogs": len(dialogs),
        "turns": len(turns),
        "preblocked_p0": len(preblocked_p0),
        "model_not_called": len(model_not_called),
        "timeouts": timeout_total,
        "timeout_turns": len(timeouts),
        "timeout_dialogs": len(timeout_dialogs),
        "timeout_budget": round(timeout_budget, 3),
        "timeout_tolerated": timeout_total > 0,
        "model_called_eligible": len(eligible_turns),
        "frames": len(eligible_frame_turns),
        "eligible_frame_rate": round(eligible_frame_rate, 6),
        "bot_direct_draft": llm_calls.get("bot_direct_draft"),
        "trace_turns": len(trace_turns),
        "gate_blocked_turns": len(gate_blocked_turns),
    }


def _valid_line(result: Mapping[str, Any]) -> str:
    timeout_suffix = "(tolerated)" if result.get("timeout_tolerated") else ""
    return (
        f"VALID_E3_{result['leg']}: dialogs={result['dialogs']} turns={result['turns']} "
        f"preblocked_p0={result['preblocked_p0']} timeouts={result['timeouts']}{timeout_suffix} "
        f"timeout_turns={result.get('timeout_turns', 0)} timeout_dialogs={result.get('timeout_dialogs', 0)} "
        f"model_not_called={result.get('model_not_called', 0)} "
        f"model_called_eligible={result['model_called_eligible']} frames={result['frames']} "
        f"eligible_frame_rate={float(result['eligible_frame_rate']):.4f} "
        f"bot_direct_draft={result['bot_direct_draft']} trace_turns={result['trace_turns']} "
        f"gate_blocked_turns={result['gate_blocked_turns']}"
    )


def _fail(leg: str, message: str) -> None:
    print(f"INVALID_{leg}: {message}", flush=True)
    raise SystemExit(1)


def _direct_path(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    value = turn.get("bot_direct_path")
    return value if isinstance(value, Mapping) else {}


def _reason_evidence(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = []
    reason = turn.get("reason_evidence")
    if isinstance(reason, Mapping):
        candidates.append(reason)
    direct_reason = _direct_path(turn).get("reason_evidence")
    if isinstance(direct_reason, Mapping):
        candidates.append(direct_reason)
    trace = turn.get("bot_answerability_trace")
    if isinstance(trace, Mapping):
        trace_direct = trace.get("direct_path")
        if isinstance(trace_direct, Mapping):
            trace_reason = trace_direct.get("reason_evidence")
            if isinstance(trace_reason, Mapping):
                candidates.append(trace_reason)
    for candidate in candidates:
        if candidate:
            return candidate
    return {}


def _provider_error(turn: Mapping[str, Any]) -> str:
    candidates = [
        turn.get("bot_provider_error"),
        turn.get("provider_error"),
        _reason_evidence(turn).get("provider_error"),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip().casefold()
        if value:
            return value
    return ""


def _is_infra_provider_error(value: str) -> bool:
    value = str(value or "").strip().casefold()
    return value in INFRA_PROVIDER_ERRORS or value.startswith("codex_error") or value.startswith("invalid_json")


def _is_timeout(turn: Mapping[str, Any]) -> bool:
    return _provider_error(turn) == "timeout"


def _is_timeout_dialog(dialog: Mapping[str, Any]) -> bool:
    return str(dialog.get("run_status") or "").strip().casefold() == "timeout"


def _is_p0_preblocked(turn: Mapping[str, Any]) -> bool:
    direct = _direct_path(turn)
    return (
        direct.get("model_called") is False
        and direct.get("preblocked") is True
        and str(direct.get("preblock_reason") or "").strip() in P0_PREBLOCK_REASONS
    )


def _is_model_called(turn: Mapping[str, Any]) -> bool:
    return _direct_path(turn).get("model_called") is True


def _has_frame(turn: Mapping[str, Any]) -> bool:
    frame = turn.get("bot_semantic_frame")
    return isinstance(frame, Mapping) and bool(frame)


def _is_gate_blocked_turn(turn: Mapping[str, Any]) -> bool:
    provider_error = _provider_error(turn)
    if provider_error in GATE_BLOCKED_PROVIDER_ERRORS:
        return True
    flags = {str(flag or "").strip().casefold() for flag in (turn.get("bot_safety_flags") or [])}
    return bool(flags & GATE_BLOCKED_FLAGS)


if __name__ == "__main__":
    raise SystemExit(main())
