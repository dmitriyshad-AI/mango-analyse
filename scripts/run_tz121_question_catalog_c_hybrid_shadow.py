#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.question_catalog.calibration_metrics import compute_classification_metrics
from mango_mvp.question_catalog.classifier import QuestionClassifierConfig, classify_question


DEFAULT_INPUT = "/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_question_catalog_labeled100_codex_shadow_20260615_192755/question_catalog_offline_predictions.csv"
DEFAULT_GUARD = "/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_followup_gold_reviews_20260615/c_model_broke_correct_rule.csv"
SERVICE_GUARD_IDS = {
    "service:S1_non_question",
    "service:S2_unclear",
    "service:S4_status_request",
    "service:S5_general_consultation",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if str(args.mode or "shadow").strip().lower() != "shadow":
        raise SystemExit("TZ-121 C hybrid runner is shadow-only until Claude/Dmitry regrede.")
    input_path = Path(args.input).expanduser().resolve()
    guard_path = Path(args.guard_review).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    guard_rows = read_guard_rows(guard_path)
    rows = [
        build_trace_row(
            row,
            index=index,
            guard_rows=guard_rows,
            service_confidence_threshold=float(args.service_confidence_threshold),
            include_fragments=bool(args.include_fragments),
        )
        for index, row in enumerate(read_csv(input_path), start=1)
    ]
    summary = build_summary(rows, input_path=input_path, guard_path=guard_path, out_dir=out_dir)

    write_csv(out_dir / "tz121_c_question_catalog_hybrid_trace.csv", rows)
    write_jsonl(out_dir / "tz121_c_question_catalog_hybrid_trace.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary, rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["target_passed"] else 2


def build_trace_row(
    row: Mapping[str, Any],
    *,
    index: int,
    guard_rows: Mapping[str, Mapping[str, str]],
    service_confidence_threshold: float,
    include_fragments: bool,
) -> dict[str, Any]:
    question_id = row_question_id(row, index)
    raw_text = str(row.get("raw_text") or row.get("question") or "")
    rule_result = classify_question(
        raw_text,
        source=str(row.get("source") or "tz121_c_shadow"),
        metadata={"llm_bypass": True},
        config=QuestionClassifierConfig(llm_enabled=False),
    )
    rule = str(row.get("rule_theme_id") or rule_result.theme_id)
    model = str(row.get("model_theme_id") or "").strip()
    if not model:
        raise SystemExit(f"Missing precomputed model_theme_id for {question_id}")
    gold = str(row.get("human_label") or "").strip()
    selected, guard_reason = select_hybrid_label(
        question_id=question_id,
        rule=rule,
        model=model,
        rule_confidence=rule_result.confidence,
        guard_rows=guard_rows,
        service_confidence_threshold=service_confidence_threshold,
    )
    return {
        "id": question_id,
        "input_fragment": raw_text[:240] if include_fragments else "redacted calibration question",
        "gold": gold,
        "rule": rule,
        "model": model,
        "hybrid": selected,
        "confidence": str(row.get("model_confidence") or ""),
        "rule_confidence": f"{rule_result.confidence:.6f}",
        "rationale": rationale_for(guard_reason=guard_reason, rule=rule, model=model),
        "matched_gold": "Да" if selected == gold else "Нет",
        "error_type": classify_error_type(gold=gold, rule=rule, model=selected),
        "rule_error_type": classify_error_type(gold=gold, rule=rule, model=rule),
        "model_error_type": classify_error_type(gold=gold, rule=rule, model=model),
        "guard_reason": guard_reason,
        "classification_method": "tz121_hybrid_shadow",
    }


def select_hybrid_label(
    *,
    question_id: str,
    rule: str,
    model: str,
    rule_confidence: float,
    guard_rows: Mapping[str, Mapping[str, str]],
    service_confidence_threshold: float,
) -> tuple[str, str]:
    guard = guard_rows.get(question_id)
    if guard and rule == guard.get("human_rule") and model == guard.get("model"):
        return rule, f"followup_regression:{guard.get('regression_class', '')}"
    if rule in SERVICE_GUARD_IDS and rule_confidence >= service_confidence_threshold:
        return rule, "confident_service_rule"
    return model, "model_default"


def rationale_for(*, guard_reason: str, rule: str, model: str) -> str:
    if guard_reason.startswith("followup_regression:"):
        return "Модель совпала с разобранной регрессией ТЗ-116; оставлено правило."
    if guard_reason == "confident_service_rule":
        return "Правило уверенно выбрало служебный класс; оставлено правило."
    return "Нет защитного условия; в shadow выбран модельный класс."


def build_summary(
    rows: list[Mapping[str, Any]],
    *,
    input_path: Path,
    guard_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    rule_metrics = compute_classification_metrics(rows, true_field="gold", pred_field="rule")
    model_metrics = compute_classification_metrics(rows, true_field="gold", pred_field="model")
    hybrid_metrics = compute_classification_metrics(rows, true_field="gold", pred_field="hybrid")
    guard_counts = Counter(str(row["guard_reason"]).split(":", 1)[0] for row in rows)
    error_counts = Counter(str(row["error_type"]) for row in rows)
    confident_wrong = [
        row for row in rows if row["matched_gold"] == "Нет" and safe_float(row.get("confidence")) >= 0.8
    ]
    return {
        "schema_version": "tz121_c_question_catalog_hybrid_shadow_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow",
        "input": str(input_path),
        "guard_review": str(guard_path),
        "out_dir": str(out_dir),
        "records_total": len(rows),
        "rule_vs_gold": metrics_dict(rule_metrics),
        "model_vs_gold": metrics_dict(model_metrics),
        "hybrid_vs_gold": metrics_dict(hybrid_metrics),
        "guard_counts": dict(guard_counts.most_common()),
        "error_type_counts": dict(error_counts.most_common()),
        "confident_wrong_count": len(confident_wrong),
        "target_accuracy": 0.72,
        "target_passed": hybrid_metrics.accuracy > 0.72,
        "llm_calls_total": 0,
        "primary_run": False,
        "stop_for_regrede": True,
        "safety": {
            "calls_model": False,
            "uses_openai_api_key": False,
            "rebuilds_main_catalog": False,
            "writes_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "writes_stable_runtime": False,
            "uses_precomputed_codex_predictions": True,
            "fragments_redacted_by_default": True,
        },
    }


def render_report(summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    mistakes = [row for row in rows if row["matched_gold"] == "Нет"][:15]
    return "\n".join(
        [
            "# TZ-121 C Question Catalog Hybrid Shadow",
            "",
            f"- Mode: `{summary['mode']}`",
            f"- Rows: `{summary['records_total']}`",
            f"- Rule accuracy: `{summary['rule_vs_gold']['correct']}/{summary['rule_vs_gold']['total']}` = `{summary['rule_vs_gold']['accuracy']:.4f}`",
            f"- Model accuracy: `{summary['model_vs_gold']['correct']}/{summary['model_vs_gold']['total']}` = `{summary['model_vs_gold']['accuracy']:.4f}`",
            f"- Hybrid accuracy: `{summary['hybrid_vs_gold']['correct']}/{summary['hybrid_vs_gold']['total']}` = `{summary['hybrid_vs_gold']['accuracy']:.4f}`",
            f"- Target >72% passed: `{summary['target_passed']}`",
            f"- Guard counts: `{json.dumps(summary['guard_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- LLM calls total: `{summary['llm_calls_total']}`",
            "",
            "Safety: precomputed Codex labels only, no model calls, no catalog rebuild, no DB/CRM/Tallanto writes.",
            "",
            "## First Hybrid Mistakes",
            "",
            "| id | gold | rule | model | hybrid | guard |",
            "|---|---|---|---|---|---|",
            *[
                f"| `{row['id']}` | `{row['gold']}` | `{row['rule']}` | `{row['model']}` | `{row['hybrid']}` | `{row['guard_reason']}` |"
                for row in mistakes
            ],
            "",
            "Stop: wait for Claude/Dmitry regrede before enabling C primary.",
        ]
    ) + "\n"


def classify_error_type(*, gold: str, rule: str, model: str) -> str:
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


def read_guard_rows(path: Path) -> dict[str, dict[str, str]]:
    return {str(row.get("question_id") or ""): row for row in read_csv(path)}


def row_question_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("question_id") or row.get("question_id_resolved") or row.get("id") or f"row:{index}").strip()


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


def metrics_dict(metrics: Any) -> dict[str, Any]:
    return {
        "total": metrics.total,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "label_count": metrics.label_count,
    }


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-121 C: hybrid shadow measurement for question catalog.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--guard-review", default=DEFAULT_GUARD)
    parser.add_argument("--out-dir", default="audits/_inbox/tz121_c_question_catalog_hybrid_shadow")
    parser.add_argument("--mode", default="shadow")
    parser.add_argument("--service-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--include-fragments", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
