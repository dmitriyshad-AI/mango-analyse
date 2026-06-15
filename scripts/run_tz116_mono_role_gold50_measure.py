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
    model_confidence = selected_confidence if selected_provider == "codex_cli" else 0.0

    rule_gold = compare_roles(rule_roles, gold_roles)
    selected_gold = compare_roles(selected_roles, gold_roles)
    model_gold = compare_roles(model_roles, gold_roles)
    model_rule = compare_roles(model_roles, rule_roles)
    selected_rule = compare_roles(selected_roles, rule_roles)

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
        "model_exact_vs_rule": truthy_cell(model_rule["exact_match"], model_roles and rule_roles),
        "selected_exact_vs_rule": truthy_cell(selected_rule["exact_match"], selected_roles and rule_roles),
        "rule_per_turn_accuracy_vs_gold": ratio_cell(rule_gold),
        "selected_per_turn_accuracy_vs_gold": ratio_cell(selected_gold),
        "model_per_turn_accuracy_vs_gold": ratio_cell(model_gold),
        "model_per_turn_accuracy_vs_rule": ratio_cell(model_rule),
        "warnings": " | ".join(warnings),
        "rule_roles_json": json.dumps(rule_roles, ensure_ascii=False),
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


def build_summary(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    input_path: Path,
    llm_calls_total: int,
    min_confidence: float,
    llm_threshold: float,
    low_info_filter_mode: str,
) -> dict[str, Any]:
    provider_counts = Counter(str(row.get("selected_provider") or "") for row in rows)
    low_conf = [row for row in rows if row.get("rule_low_confidence") == "Да"]
    gold_rows = [row for row in rows if row.get("gold_labeled") == "Да"]
    model_rows = [row for row in rows if row.get("codex_called") == "Да"]
    low_info_rows = [row for row in rows if row.get("low_info_filter_applied") == "Да"]
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
        "rule_vs_gold": aggregate_exact(rows, "rule_exact_vs_gold", "rule_per_turn_accuracy_vs_gold"),
        "selected_vs_gold": aggregate_exact(rows, "selected_exact_vs_gold", "selected_per_turn_accuracy_vs_gold"),
        "model_vs_gold": aggregate_exact(rows, "model_exact_vs_gold", "model_per_turn_accuracy_vs_gold"),
        "model_vs_rule": aggregate_exact(rows, "model_exact_vs_rule", "model_per_turn_accuracy_vs_rule"),
        "thresholds": {
            "min_confidence": min_confidence,
            "llm_threshold": llm_threshold,
        },
        "low_info_filter_mode": "off" if mode == "off" else (low_info_filter_mode.strip().lower() or "off"),
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
            f"- Rule vs gold: `{json.dumps(summary['rule_vs_gold'], ensure_ascii=False, sort_keys=True)}`",
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
    parser.add_argument("--review-low-confidence-all", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
