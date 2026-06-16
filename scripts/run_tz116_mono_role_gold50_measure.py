#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.config import get_settings
from mango_mvp.services.transcribe import TranscribeService


MODES = {"off", "shadow", "primary"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)
    if mode == "primary":
        raise SystemExit("primary is blocked for TZ-116 D; use shadow and wait for gold + regrede")

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_in = read_csv(input_path)
    service = build_service(
        mode,
        min_confidence=float(args.min_confidence),
        llm_threshold=float(args.llm_threshold),
        low_info_filter_mode=str(args.low_info_filter_mode or ""),
        segment_guard_mode=str(args.segment_guard_mode or ""),
        model=str(args.model or ""),
        reasoning_effort=str(args.reasoning_effort or ""),
        timeout_sec=int(args.timeout_sec),
    )
    llm_attempts = {"count": 0}
    if mode == "shadow":
        wrap_codex_counter(service, llm_attempts)

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows_in, start=1):
        result = evaluate_row(source, index=index, service=service, mode=mode, args=args)
        rows.append(result["row"])
        results.append(result["jsonl"])
        if result["needs_review"]:
            review_rows.append(result["review_row"])

    summary = build_summary(
        rows,
        mode=mode,
        input_path=input_path,
        llm_calls_total=llm_attempts["count"],
        min_confidence=float(args.min_confidence),
        llm_threshold=float(args.llm_threshold),
        low_info_filter_mode=str(args.low_info_filter_mode or ""),
        segment_guard_mode=str(args.segment_guard_mode or ""),
    )
    write_csv(out_dir / "mono_role_gold50_measure_rows.csv", rows)
    write_csv(out_dir / "mono_role_gold50_manual_review_queue.csv", review_rows)
    write_jsonl(out_dir / "mono_role_gold50_measure_results.jsonl", results)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def normalize_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MODES else "off"


def build_service(
    mode: str,
    *,
    min_confidence: float,
    llm_threshold: float,
    low_info_filter_mode: str,
    segment_guard_mode: str,
    model: str,
    reasoning_effort: str,
    timeout_sec: int,
) -> TranscribeService:
    settings = get_settings()
    cache_dir = ".codex_local/tz116_mono_role_gold50_measure/cache_disabled"
    return TranscribeService(
        replace(
            settings,
            mono_role_assignment_mode="off" if mode == "off" else "codex_selective",
            mono_role_low_info_filter_mode=(
                "off" if mode == "off" else (low_info_filter_mode.strip().lower() or "off")
            ),
            mono_role_segment_guard_mode=(
                "off" if mode == "off" else (segment_guard_mode.strip().lower() or "off")
            ),
            mono_role_assignment_min_confidence=min_confidence,
            mono_role_assignment_llm_threshold=llm_threshold,
            codex_transcribe_model=model.strip() or settings.codex_transcribe_model,
            codex_reasoning_effort=reasoning_effort.strip().lower() or settings.codex_reasoning_effort,
            codex_cli_timeout_sec=max(30, timeout_sec),
            llm_cache_enabled=False,
            llm_cache_dir=cache_dir,
        )
    )


def wrap_codex_counter(service: TranscribeService, counter: dict[str, int]) -> None:
    original = service._assign_roles_with_codex  # noqa: SLF001

    def counted(turns: list[dict[str, Any]], manager_name: str) -> dict[str, Any]:
        counter["count"] += 1
        return original(turns, manager_name)

    service._assign_roles_with_codex = counted  # type: ignore[method-assign] # noqa: SLF001


def evaluate_row(
    source: Mapping[str, Any],
    *,
    index: int,
    service: TranscribeService,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    call_id = str(source.get("canonical_call_id") or source.get("call_id") or f"row:{index}")
    turns = parse_turns(source.get("turns_json"))
    manager_name = str(source.get("manager_name") or "")
    gold_roles = parse_roles(source.get("gold_roles"))

    rule_result = service._assign_roles_rule_based(turns, manager_name) if turns else None  # noqa: SLF001
    rule_meta = rule_result.get("meta", {}) if rule_result else {}
    rule_roles = [str(item) for item in rule_meta.get("roles") or []]
    rule_confidence = as_float(rule_meta.get("confidence"))
    rule_has_both_roles = bool(rule_meta.get("has_both_roles"))
    rule_low_confidence = not rule_has_both_roles or rule_confidence < float(args.llm_threshold)

    warnings: list[str] = []
    selected = None
    if mode == "shadow" and turns:
        selected = service._assign_roles_for_mono(turns, manager_name, warnings)  # noqa: SLF001
    selected_meta = selected.get("meta", {}) if selected else {}
    selected_provider = str(selected_meta.get("provider") or ("off" if mode == "off" else "none"))
    selected_roles = [str(item) for item in selected_meta.get("roles") or []]
    selected_confidence = as_float(selected_meta.get("confidence"))
    model_roles = selected_roles if selected_provider == "codex_cli" else []
    raw_model_roles = [
        str(item)
        for item in (selected_meta.get("raw_model_roles_before_guard") or model_roles)
    ] if selected_provider == "codex_cli" else []
    model_confidence = selected_confidence if selected_provider == "codex_cli" else 0.0

    rule_gold = compare_roles(rule_roles, gold_roles)
    selected_gold = compare_roles(selected_roles, gold_roles)
    model_gold = compare_roles(model_roles, gold_roles)
    raw_model_gold = compare_roles(raw_model_roles, gold_roles)
    model_rule = compare_roles(model_roles, rule_roles)
    selected_rule = compare_roles(selected_roles, rule_roles)
    guard_indexes = parse_index_list(selected_meta.get("segment_guard_turn_indexes"))
    guard_changed_indexes = parse_index_list(selected_meta.get("segment_guard_changed_indexes"))
    guard_effect = guard_effect_counts(
        raw_roles=raw_model_roles,
        post_roles=selected_roles,
        gold_roles=gold_roles,
        guarded_indexes=guard_indexes,
    )
    raw_error = turn_error_counts(raw_model_roles, gold_roles)
    post_error = turn_error_counts(selected_roles, gold_roles)
    guarded_error = turn_error_counts_for_indexes(selected_roles, gold_roles, guard_indexes)
    changed_error = turn_error_counts_for_indexes(selected_roles, gold_roles, guard_changed_indexes)
    raw_segments = role_error_segment_stats(raw_model_roles, gold_roles)
    post_segments = role_error_segment_stats(selected_roles, gold_roles)

    codex_called = selected_provider == "codex_cli"
    needs_review = (
        bool(args.review_low_confidence_all and rule_low_confidence)
        or codex_called
        or (bool(model_roles) and not model_rule["exact_match"])
        or not gold_roles
    )

    row = {
        "canonical_call_id": call_id,
        "source_filename": source.get("source_filename", ""),
        "started_at": source.get("started_at", ""),
        "manager_name": manager_name,
        "duration_sec": source.get("duration_sec", ""),
        "turn_count": len(turns),
        "mode": mode,
        "rule_confidence": f"{rule_confidence:.6f}" if rule_result else "",
        "rule_low_confidence": "Да" if rule_low_confidence else "Нет",
        "rule_has_both_roles": "Да" if rule_has_both_roles else "Нет",
        "selected_provider": selected_provider,
        "selected_confidence": f"{selected_confidence:.6f}" if selected else "",
        "codex_called": "Да" if codex_called else "Нет",
        "gold_labeled": "Да" if gold_roles else "Нет",
        "rule_exact_vs_gold": truthy_cell(rule_gold["exact_match"], gold_roles),
        "selected_exact_vs_gold": truthy_cell(selected_gold["exact_match"], gold_roles),
        "model_exact_vs_gold": truthy_cell(model_gold["exact_match"], gold_roles and model_roles),
        "raw_model_exact_vs_gold": truthy_cell(raw_model_gold["exact_match"], gold_roles and raw_model_roles),
        "model_exact_vs_rule": truthy_cell(model_rule["exact_match"], model_roles and rule_roles),
        "selected_exact_vs_rule": truthy_cell(selected_rule["exact_match"], selected_roles and rule_roles),
        "rule_per_turn_accuracy_vs_gold": ratio_cell(rule_gold),
        "selected_per_turn_accuracy_vs_gold": ratio_cell(selected_gold),
        "model_per_turn_accuracy_vs_gold": ratio_cell(model_gold),
        "raw_model_per_turn_accuracy_vs_gold": ratio_cell(raw_model_gold),
        "model_per_turn_accuracy_vs_rule": ratio_cell(model_rule),
        "warnings": " | ".join(warnings),
        "rule_roles_json": json.dumps(rule_roles, ensure_ascii=False),
        "raw_model_roles_json": json.dumps(raw_model_roles, ensure_ascii=False),
        "post_guard_roles_json": json.dumps(selected_roles, ensure_ascii=False),
        "selected_roles_json": json.dumps(selected_roles, ensure_ascii=False),
        "model_roles_json": json.dumps(model_roles, ensure_ascii=False),
        "gold_roles_json": json.dumps(gold_roles, ensure_ascii=False),
        "codex_notes": str(selected_meta.get("notes") or "") if selected_provider == "codex_cli" else "",
        "codex_rationale": str(selected_meta.get("rationale") or "") if selected_provider == "codex_cli" else "",
        "low_info_filter_mode": str(selected_meta.get("low_info_filter_mode") or ""),
        "low_info_filter_applied": "Да" if selected_meta.get("low_info_filter_applied") else "Нет",
        "low_info_turn_count": int(selected_meta.get("low_info_turn_count") or 0),
        "low_info_changed_count": int(selected_meta.get("low_info_changed_count") or 0),
        "low_info_turn_indexes_json": json.dumps(selected_meta.get("low_info_turn_indexes") or [], ensure_ascii=False),
        "low_info_changed_indexes_json": json.dumps(selected_meta.get("low_info_changed_indexes") or [], ensure_ascii=False),
        "segment_guard_mode": str(selected_meta.get("segment_guard_mode") or ""),
        "segment_guard_applied": "Да" if selected_meta.get("segment_guard_applied") else "Нет",
        "segment_guard_segment_count": int(selected_meta.get("segment_guard_segment_count") or 0),
        "segment_guard_turn_count": int(selected_meta.get("segment_guard_turn_count") or 0),
        "segment_guard_changed_count": int(selected_meta.get("segment_guard_changed_count") or 0),
        "segment_guard_manager_anchor_count": len(parse_index_list(selected_meta.get("segment_guard_manager_anchor_indexes"))),
        "segment_guard_low_info_count": len(parse_index_list(selected_meta.get("segment_guard_low_info_indexes"))),
        "segment_guard_turn_indexes_json": json.dumps(selected_meta.get("segment_guard_turn_indexes") or [], ensure_ascii=False),
        "segment_guard_changed_indexes_json": json.dumps(selected_meta.get("segment_guard_changed_indexes") or [], ensure_ascii=False),
        "raw_model_turn_total": raw_error["total"],
        "raw_model_turn_error_count": raw_error["errors"],
        "raw_model_turn_error_rate": f"{raw_error['error_rate']:.6f}" if raw_error["total"] else "",
        "post_guard_turn_total": post_error["total"],
        "post_guard_turn_error_count": post_error["errors"],
        "post_guard_turn_error_rate": f"{post_error['error_rate']:.6f}" if post_error["total"] else "",
        "guarded_turn_count": guarded_error["total"],
        "guarded_turn_error_count_post": guarded_error["errors"],
        "guarded_turn_error_rate_post": f"{guarded_error['error_rate']:.6f}" if guarded_error["total"] else "",
        "changed_turn_count": changed_error["total"],
        "changed_turn_error_count_post": changed_error["errors"],
        "changed_turn_error_rate_post": f"{changed_error['error_rate']:.6f}" if changed_error["total"] else "",
        "guard_fixed_count": guard_effect["fixed"],
        "guard_broke_count": guard_effect["broke"],
        "guard_neutral_correct_count": guard_effect["neutral_correct"],
        "guard_neutral_wrong_count": guard_effect["neutral_wrong"],
        "guard_net_delta": guard_effect["fixed"] - guard_effect["broke"],
        "raw_error_segments_count": raw_segments["segments"],
        "raw_error_segment_turns": raw_segments["turns"],
        "raw_error_segments_len_ge3_count": raw_segments["segments_len_ge3"],
        "post_guard_error_segments_count": post_segments["segments"],
        "post_guard_error_segment_turns": post_segments["turns"],
        "post_guard_error_segments_len_ge3_count": post_segments["segments_len_ge3"],
    }
    review_row = {
        "canonical_call_id": call_id,
        "source_filename": source.get("source_filename", ""),
        "started_at": source.get("started_at", ""),
        "manager_name": manager_name,
        "duration_sec": source.get("duration_sec", ""),
        "turn_count": len(turns),
        "review_reason": review_reason(row, gold_roles=gold_roles),
        "rule_confidence": row["rule_confidence"],
        "selected_provider": selected_provider,
        "rule_roles_json": row["rule_roles_json"],
        "selected_roles_json": row["selected_roles_json"],
        "model_roles_json": row["model_roles_json"],
        "gold_roles": source.get("gold_roles", ""),
        "notes_for_reviewer": source.get("notes_for_reviewer", ""),
        "turns_json": json.dumps(serializable_turns(turns), ensure_ascii=False),
    }
    return {
        "row": row,
        "review_row": review_row,
        "jsonl": {
            "canonical_call_id": call_id,
            "mode": mode,
            "turns": serializable_turns(turns),
            "rule": {"roles": rule_roles, "confidence": rule_confidence, "meta": rule_meta},
            "selected": {"roles": selected_roles, "confidence": selected_confidence, "meta": selected_meta},
            "gold_roles": gold_roles,
            "warnings": warnings,
        },
        "needs_review": needs_review,
    }


def parse_turns(value: Any) -> list[dict[str, Any]]:
    try:
        payload = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    turns: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        turns.append(
            {
                "start": as_float(item.get("start")),
                "approximate": bool(item.get("approximate", True)),
                "text": text,
            }
        )
    return turns


def parse_roles(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not isinstance(parsed, list):
        return []
    roles: list[str] = []
    for item in parsed:
        role = str(item).strip().lower()
        if role in {"manager", "client"}:
            roles.append(role)
    return roles


def compare_roles(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    if not left or not right:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "exact_match": False, "length_mismatch": bool(left or right)}
    total = min(len(left), len(right))
    correct = sum(1 for a, b in zip(left, right) if a == b)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "exact_match": len(left) == len(right) and correct == total,
        "length_mismatch": len(left) != len(right),
    }


def parse_index_list(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    indexes: set[int] = set()
    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index > 0:
            indexes.add(index)
    return indexes


def turn_error_counts(roles: Sequence[str], gold_roles: Sequence[str]) -> dict[str, Any]:
    total = min(len(roles), len(gold_roles))
    if total <= 0:
        return {"total": 0, "errors": 0, "error_rate": 0.0}
    errors = sum(1 for role, gold in zip(roles, gold_roles) if role != gold)
    return {"total": total, "errors": errors, "error_rate": errors / total}


def turn_error_counts_for_indexes(
    roles: Sequence[str],
    gold_roles: Sequence[str],
    indexes_1based: set[int],
) -> dict[str, Any]:
    total = 0
    errors = 0
    for index in sorted(indexes_1based):
        zero_index = index - 1
        if zero_index < 0 or zero_index >= len(roles) or zero_index >= len(gold_roles):
            continue
        total += 1
        if roles[zero_index] != gold_roles[zero_index]:
            errors += 1
    return {"total": total, "errors": errors, "error_rate": errors / total if total else 0.0}


def guard_effect_counts(
    *,
    raw_roles: Sequence[str],
    post_roles: Sequence[str],
    gold_roles: Sequence[str],
    guarded_indexes: set[int],
) -> dict[str, int]:
    counts = {
        "fixed": 0,
        "broke": 0,
        "neutral_correct": 0,
        "neutral_wrong": 0,
        "excluded": 0,
    }
    for index in sorted(guarded_indexes):
        zero_index = index - 1
        if (
            zero_index < 0
            or zero_index >= len(raw_roles)
            or zero_index >= len(post_roles)
            or zero_index >= len(gold_roles)
        ):
            counts["excluded"] += 1
            continue
        raw_correct = raw_roles[zero_index] == gold_roles[zero_index]
        post_correct = post_roles[zero_index] == gold_roles[zero_index]
        if not raw_correct and post_correct:
            counts["fixed"] += 1
        elif raw_correct and not post_correct:
            counts["broke"] += 1
        elif raw_correct and post_correct:
            counts["neutral_correct"] += 1
        else:
            counts["neutral_wrong"] += 1
    return counts


def role_error_segment_stats(roles: Sequence[str], gold_roles: Sequence[str]) -> dict[str, int]:
    segments = 0
    turns = 0
    segments_len_ge3 = 0
    turns_len_ge3 = 0
    current_len = 0
    for role, gold in zip(roles, gold_roles):
        if role != gold:
            current_len += 1
            continue
        if current_len:
            segments += 1
            turns += current_len
            if current_len >= 3:
                segments_len_ge3 += 1
                turns_len_ge3 += current_len
            current_len = 0
    if current_len:
        segments += 1
        turns += current_len
        if current_len >= 3:
            segments_len_ge3 += 1
            turns_len_ge3 += current_len
    return {
        "segments": segments,
        "turns": turns,
        "segments_len_ge3": segments_len_ge3,
        "turns_len_ge3": turns_len_ge3,
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    input_path: Path,
    llm_calls_total: int,
    min_confidence: float,
    llm_threshold: float,
    low_info_filter_mode: str,
    segment_guard_mode: str,
) -> dict[str, Any]:
    provider_counts = Counter(str(row.get("selected_provider") or "") for row in rows)
    low_conf = [row for row in rows if row.get("rule_low_confidence") == "Да"]
    gold_rows = [row for row in rows if row.get("gold_labeled") == "Да"]
    model_rows = [row for row in rows if row.get("codex_called") == "Да"]
    low_info_rows = [row for row in rows if row.get("low_info_filter_applied") == "Да"]
    segment_guard_rows = [row for row in rows if row.get("segment_guard_applied") == "Да"]
    raw_turn_total = sum(int(row.get("raw_model_turn_total") or 0) for row in rows)
    raw_turn_errors = sum(int(row.get("raw_model_turn_error_count") or 0) for row in rows)
    post_turn_total = sum(int(row.get("post_guard_turn_total") or 0) for row in rows)
    post_turn_errors = sum(int(row.get("post_guard_turn_error_count") or 0) for row in rows)
    guarded_turn_total = sum(int(row.get("guarded_turn_count") or 0) for row in rows)
    guarded_turn_errors_post = sum(int(row.get("guarded_turn_error_count_post") or 0) for row in rows)
    changed_turn_total = sum(int(row.get("changed_turn_count") or 0) for row in rows)
    changed_turn_errors_post = sum(int(row.get("changed_turn_error_count_post") or 0) for row in rows)
    guard_fixed_total = sum(int(row.get("guard_fixed_count") or 0) for row in rows)
    guard_broke_total = sum(int(row.get("guard_broke_count") or 0) for row in rows)
    raw_error_segments_total = sum(int(row.get("raw_error_segments_count") or 0) for row in rows)
    raw_error_segment_turns = sum(int(row.get("raw_error_segment_turns") or 0) for row in rows)
    post_error_segments_total = sum(int(row.get("post_guard_error_segments_count") or 0) for row in rows)
    post_error_segment_turns = sum(int(row.get("post_guard_error_segment_turns") or 0) for row in rows)
    raw_model_missing = any(
        row.get("codex_called") == "Да" and not str(row.get("raw_model_roles_json") or "[]").strip("[]")
        for row in rows
    )
    length_mismatch_or_gold_gaps = any(
        row.get("gold_labeled") == "Да"
        and (
            int(row.get("turn_count") or 0) != int(row.get("raw_model_turn_total") or 0)
            or int(row.get("turn_count") or 0) != int(row.get("post_guard_turn_total") or 0)
        )
        for row in rows
    )
    stop_conditions = {
        "raw_model_roles_missing": raw_model_missing,
        "post_guard_worse_than_raw": bool(post_turn_total and raw_turn_total and post_turn_errors > raw_turn_errors),
        "guard_net_delta_negative": guard_fixed_total - guard_broke_total < 0,
        "guard_broke_total_positive": guard_broke_total > 0,
        "length_mismatch_or_gold_gaps": length_mismatch_or_gold_gaps,
    }
    stop_conditions["stop_recommended_for_primary"] = any(stop_conditions.values())
    return {
        "schema_version": "tz116_mono_role_gold50_measure_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input": str(input_path),
        "calls_total": len(rows),
        "turns_total": sum(int(row.get("turn_count") or 0) for row in rows),
        "rule_low_confidence_calls": len(low_conf),
        "rule_high_confidence_calls": len(rows) - len(low_conf),
        "gold_labeled_calls": len(gold_rows),
        "codex_called_calls": len(model_rows),
        "llm_calls_total": llm_calls_total,
        "selected_provider_counts": dict(provider_counts),
        "low_info_filter": {
            "calls_with_filter": len(low_info_rows),
            "turns_filtered_total": sum(int(row.get("low_info_turn_count") or 0) for row in rows),
            "turns_changed_total": sum(int(row.get("low_info_changed_count") or 0) for row in rows),
        },
        "segment_guard": {
            "mode": "off" if mode == "off" else (segment_guard_mode.strip().lower() or "off"),
            "calls_with_guard": len(segment_guard_rows),
            "segments_total": sum(int(row.get("segment_guard_segment_count") or 0) for row in rows),
            "turns_guarded_total": guarded_turn_total,
            "turns_changed_total": changed_turn_total,
            "manager_anchor_turns_total": sum(int(row.get("segment_guard_manager_anchor_count") or 0) for row in rows),
            "low_info_turns_total": sum(int(row.get("segment_guard_low_info_count") or 0) for row in rows),
            "guard_fixed_total": guard_fixed_total,
            "guard_broke_total": guard_broke_total,
            "guard_neutral_correct_total": sum(int(row.get("guard_neutral_correct_count") or 0) for row in rows),
            "guard_neutral_wrong_total": sum(int(row.get("guard_neutral_wrong_count") or 0) for row in rows),
            "guard_net_delta": guard_fixed_total - guard_broke_total,
            "guarded_turn_error_rate_post": guarded_turn_errors_post / guarded_turn_total if guarded_turn_total else 0.0,
            "changed_turn_error_rate_post": changed_turn_errors_post / changed_turn_total if changed_turn_total else 0.0,
        },
        "per_turn_micro": {
            "raw_model_total": raw_turn_total,
            "raw_model_errors": raw_turn_errors,
            "raw_model_error_rate": raw_turn_errors / raw_turn_total if raw_turn_total else 0.0,
            "post_guard_total": post_turn_total,
            "post_guard_errors": post_turn_errors,
            "post_guard_error_rate": post_turn_errors / post_turn_total if post_turn_total else 0.0,
        },
        "error_segments": {
            "raw_model_segments_total": raw_error_segments_total,
            "raw_model_segment_turns_total": raw_error_segment_turns,
            "raw_model_segment_turn_ratio": raw_error_segment_turns / raw_turn_total if raw_turn_total else 0.0,
            "post_guard_segments_total": post_error_segments_total,
            "post_guard_segment_turns_total": post_error_segment_turns,
            "post_guard_segment_turn_ratio": post_error_segment_turns / post_turn_total if post_turn_total else 0.0,
            "raw_model_segments_len_ge3_total": sum(int(row.get("raw_error_segments_len_ge3_count") or 0) for row in rows),
            "post_guard_segments_len_ge3_total": sum(int(row.get("post_guard_error_segments_len_ge3_count") or 0) for row in rows),
        },
        "rule_vs_gold": aggregate_exact(rows, "rule_exact_vs_gold", "rule_per_turn_accuracy_vs_gold"),
        "selected_vs_gold": aggregate_exact(rows, "selected_exact_vs_gold", "selected_per_turn_accuracy_vs_gold"),
        "model_vs_gold": aggregate_exact(rows, "model_exact_vs_gold", "model_per_turn_accuracy_vs_gold"),
        "raw_model_vs_gold": aggregate_exact(rows, "raw_model_exact_vs_gold", "raw_model_per_turn_accuracy_vs_gold"),
        "model_vs_rule": aggregate_exact(rows, "model_exact_vs_rule", "model_per_turn_accuracy_vs_rule"),
        "thresholds": {
            "min_confidence": min_confidence,
            "llm_threshold": llm_threshold,
        },
        "low_info_filter_mode": "off" if mode == "off" else (low_info_filter_mode.strip().lower() or "off"),
        "segment_guard_mode": "off" if mode == "off" else (segment_guard_mode.strip().lower() or "off"),
        "stop_conditions": stop_conditions,
        "safety": {
            "model_transport": "codex_cli" if mode == "shadow" else "none",
            "uses_openai_api_key": False,
            "reads_audio": False,
            "runs_asr": False,
            "writes_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "primary_blocked": True,
        },
    }


def aggregate_exact(rows: list[Mapping[str, Any]], exact_field: str, accuracy_field: str) -> dict[str, Any]:
    labeled = [row for row in rows if str(row.get(exact_field) or "")]
    correct = sum(1 for row in labeled if row.get(exact_field) == "Да")
    per_turn_values = [float(row.get(accuracy_field) or 0.0) for row in labeled if str(row.get(accuracy_field) or "")]
    return {
        "total": len(labeled),
        "exact_correct": correct,
        "exact_accuracy": correct / len(labeled) if labeled else 0.0,
        "mean_per_turn_accuracy": sum(per_turn_values) / len(per_turn_values) if per_turn_values else 0.0,
    }


def truthy_cell(value: Any, enabled: Any) -> str:
    if not enabled:
        return ""
    return "Да" if bool(value) else "Нет"


def ratio_cell(result: Mapping[str, Any]) -> str:
    return f"{float(result.get('accuracy') or 0.0):.6f}" if int(result.get("total") or 0) else ""


def review_reason(row: Mapping[str, Any], *, gold_roles: Sequence[str]) -> str:
    reasons: list[str] = []
    if row.get("rule_low_confidence") == "Да":
        reasons.append("rule_low_confidence")
    if row.get("codex_called") == "Да":
        reasons.append("codex_called")
    if row.get("model_exact_vs_rule") == "Нет":
        reasons.append("model_rule_disagreement")
    if not gold_roles:
        reasons.append("gold_missing")
    return " | ".join(reasons)


def serializable_turns(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"i": index, "start": turn.get("start"), "approximate": turn.get("approximate"), "text": turn.get("text", "")}
        for index, turn in enumerate(turns, start=1)
    ]


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TZ-116 D Mono Role Gold-50 Measurement",
            "",
            f"- Mode: `{summary['mode']}`",
            f"- Calls total: `{summary['calls_total']}`",
            f"- Rule low-confidence calls: `{summary['rule_low_confidence_calls']}`",
            f"- Codex-called calls: `{summary['codex_called_calls']}`",
            f"- Gold-labeled calls: `{summary['gold_labeled_calls']}`",
            f"- LLM calls total: `{summary['llm_calls_total']}`",
            f"- Selected providers: `{json.dumps(summary['selected_provider_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Low-info filter mode: `{summary.get('low_info_filter_mode', '')}`",
            f"- Low-info filter: `{json.dumps(summary.get('low_info_filter', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- Segment guard mode: `{summary.get('segment_guard_mode', '')}`",
            f"- Segment guard: `{json.dumps(summary.get('segment_guard', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- Per-turn micro: `{json.dumps(summary.get('per_turn_micro', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- Error segments: `{json.dumps(summary.get('error_segments', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- Stop conditions: `{json.dumps(summary.get('stop_conditions', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- Rule vs gold: `{json.dumps(summary['rule_vs_gold'], ensure_ascii=False, sort_keys=True)}`",
            f"- Raw model vs gold: `{json.dumps(summary.get('raw_model_vs_gold', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- Model vs gold: `{json.dumps(summary['model_vs_gold'], ensure_ascii=False, sort_keys=True)}`",
            f"- Model vs rule: `{json.dumps(summary['model_vs_rule'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "Safety: read-only CSV input, no audio, no ASR, no DB writes, Codex CLI only, primary blocked.",
        ]
    ) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-116 D: measure codex_selective role assignment on the fixed 50-call gold set.")
    parser.add_argument("--input", required=True, help="CSV from mono_role_gold_review_sample.csv with turns_json and optional gold_roles.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz116_mono_role_gold50_measure")
    parser.add_argument("--mode", choices=sorted(MODES), default="off")
    parser.add_argument("--min-confidence", type=float, default=0.62)
    parser.add_argument("--llm-threshold", type=float, default=0.72)
    parser.add_argument("--model", default="")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--low-info-filter-mode", choices=["off", "mark", "filter"], default="off")
    parser.add_argument("--segment-guard-mode", choices=["off", "repair"], default="off")
    parser.add_argument("--review-low-confidence-all", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
