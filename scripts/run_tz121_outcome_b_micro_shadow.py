#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.insights.outcome_linker import classify_tallanto_rows


DEFAULT_INPUT = "tests/fixtures/tz121_outcome_b_micro_gold.csv"
ALLOWED_PRIMARY_FLIP = "won_paid_or_active->known_student_or_lead"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = str(args.mode or "shadow").strip().lower()
    if mode != "shadow":
        raise SystemExit("TZ-121 B micro runner is shadow-only until Claude/Dmitry regrede.")

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_in = read_csv(input_path)
    trace_rows = [build_trace_row(row, index=index) for index, row in enumerate(rows_in, start=1)]
    summary = build_summary(trace_rows, input_path=input_path, out_dir=out_dir)

    write_csv(out_dir / "tz121_b_outcome_trace.csv", trace_rows)
    write_jsonl(out_dir / "tz121_b_outcome_trace.jsonl", trace_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_trace_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    case_id = str(row.get("case_id") or f"row:{index}")
    gold = str(row.get("gold_label") or "").strip()
    tallanto_row = {
        "tallanto_id": case_id,
        "student_type": row.get("student_type", ""),
        "history_raw": row.get("history_raw", ""),
    }
    legacy = classify_tallanto_rows([tallanto_row], outcome_model_mode="off")
    shadow_holder = classify_tallanto_rows([tallanto_row], outcome_model_mode="shadow")
    shadow = shadow_holder.metadata.get("outcome_model_shadow") or {}
    model = str(shadow.get("semantic_label") or legacy.label)
    confidence = float(shadow.get("semantic_confidence_score") or shadow_holder.confidence_score or 0.0)
    flip = f"{legacy.label}->{model}"
    primary_allowed = flip == ALLOWED_PRIMARY_FLIP
    matched_gold = model == gold if gold else False
    error_type = classify_error_type(gold=gold, rule=legacy.label, model=model)
    return {
        "id": case_id,
        "input_fragment": str(row.get("history_raw") or "")[:240],
        "student_type": row.get("student_type", ""),
        "gold": gold,
        "rule": legacy.label,
        "model": model,
        "confidence": f"{confidence:.6f}",
        "rationale": rationale_for(shadow=shadow, flip=flip, primary_allowed=primary_allowed),
        "matched_gold": "Да" if matched_gold else ("Нет" if gold else ""),
        "error_type": error_type,
        "flip": flip,
        "primary_allowed": "Да" if primary_allowed else "Нет",
        "primary_policy": "only_won_paid_or_active_to_known_student_or_lead",
        "case_note": row.get("case_note", ""),
    }


def rationale_for(*, shadow: Mapping[str, Any], flip: str, primary_allowed: bool) -> str:
    if primary_allowed:
        return "Отрицание сняло ложные признаки оплаты/записи; flip входит в allowlist primary."
    if flip == "won_paid_or_active->payment_pending":
        return "Отрицание сняло ложную оплату, но payment_pending-flip запрещен для primary."
    if not bool(shadow.get("label_changed")):
        return "Кандидат совпал с legacy; primary ничего не меняет."
    return "Кандидат отличается от legacy, но flip не входит в allowlist primary."


def classify_error_type(*, gold: str, rule: str, model: str) -> str:
    if not gold:
        return "no_gold"
    rule_ok = rule == gold
    model_ok = model == gold
    if model_ok and not rule_ok:
        return "model_fix"
    if rule_ok and not model_ok:
        return "model_break"
    if model_ok and rule_ok:
        return "both_correct"
    if rule == model:
        return "both_wrong_same"
    return "both_wrong"


def build_summary(rows: list[Mapping[str, Any]], *, input_path: Path, out_dir: Path) -> dict[str, Any]:
    error_counts = Counter(str(row.get("error_type") or "") for row in rows)
    flips = Counter(str(row.get("flip") or "") for row in rows)
    allowed_rows = [row for row in rows if row.get("primary_allowed") == "Да"]
    payment_pending_flips = [row for row in rows if row.get("flip") == "won_paid_or_active->payment_pending"]
    model_breaks = [row for row in rows if row.get("error_type") == "model_break"]
    return {
        "schema_version": "tz121_b_outcome_micro_shadow_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow",
        "input": str(input_path),
        "out_dir": str(out_dir),
        "rows_total": len(rows),
        "gold_present_total": sum(1 for row in rows if row.get("gold")),
        "allowed_primary_flip": ALLOWED_PRIMARY_FLIP,
        "allowed_flip_rows": len(allowed_rows),
        "allowed_flip_correct": sum(1 for row in allowed_rows if row.get("matched_gold") == "Да"),
        "allowed_flip_wrong": sum(1 for row in allowed_rows if row.get("matched_gold") == "Нет"),
        "payment_pending_flip_rows": len(payment_pending_flips),
        "payment_pending_flip_primary_blocked": len(payment_pending_flips),
        "model_break_rows": len(model_breaks),
        "error_type_counts": dict(error_counts.most_common()),
        "flip_counts": dict(flips.most_common()),
        "llm_calls_total": 0,
        "primary_run": False,
        "stop_for_regrede": True,
        "safety": {
            "calls_model": False,
            "uses_openai_api_key": False,
            "writes_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "reads_stable_runtime": False,
            "runs_full_set": False,
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TZ-121 B Outcome Micro Shadow",
            "",
            f"- Mode: `{summary['mode']}`",
            f"- Rows: `{summary['rows_total']}`",
            f"- Allowed flip: `{summary['allowed_primary_flip']}`",
            f"- Allowed flip rows: `{summary['allowed_flip_rows']}`",
            f"- Allowed flip correct: `{summary['allowed_flip_correct']}`",
            f"- Allowed flip wrong: `{summary['allowed_flip_wrong']}`",
            f"- Payment-pending flips blocked for primary: `{summary['payment_pending_flip_primary_blocked']}`",
            f"- Model breaks: `{summary['model_break_rows']}`",
            f"- Error types: `{json.dumps(summary['error_type_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Flips: `{json.dumps(summary['flip_counts'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "Safety: synthetic micro-set only, no model calls, no full set, no DB/CRM/Tallanto writes.",
            "",
            "Stop: wait for Claude/Dmitry regrede before enabling B primary or moving to E.",
        ]
    ) + "\n"


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-121 B: shadow micro-measurement for selective outcome primary.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="audits/_inbox/tz121_b_outcome_micro_shadow")
    parser.add_argument("--mode", default="shadow")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
