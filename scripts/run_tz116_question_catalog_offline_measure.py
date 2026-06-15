#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.question_catalog.calibration_metrics import compute_classification_metrics
from mango_mvp.question_catalog.classifier import QuestionClassifierConfig, classify_question


MODES = {"off", "shadow", "primary"}
MODEL_LABEL_FIELDS = ("model_theme_id", "llm_theme_id", "model_label", "llm_label", "predicted_theme_id_model")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_csv(input_path)
    predictions = [build_prediction(row, mode=mode) for row in source_rows]
    metrics = compute_classification_metrics(predictions)
    method_counts = Counter(str(row.get("classification_method") or "") for row in predictions)
    comparison_counts = Counter(str(row.get("model_comparison") or "") for row in predictions if row.get("model_comparison"))
    missing_model = sum(1 for row in predictions if row.get("model_required_missing") == "1")
    if mode == "primary" and missing_model:
        raise SystemExit(f"primary mode requires precomputed model labels; missing={missing_model}")

    summary = {
        "schema_version": "tz116_question_catalog_offline_measure_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input": str(input_path),
        "total": metrics.total,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "label_count": metrics.label_count,
        "classification_method_counts": dict(method_counts),
        "model_comparison_counts": dict(comparison_counts),
        "missing_model_labels": missing_model,
        "llm_calls_total": 0,
        "safety": {
            "calls_live_llm": False,
            "rebuilds_main_catalog": False,
            "writes_stable_runtime": False,
        },
    }
    write_csv(out_dir / "question_catalog_offline_predictions.csv", predictions)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary, predictions), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_prediction(row: Mapping[str, Any], *, mode: str) -> dict[str, str]:
    raw_text = str(row.get("raw_text") or row.get("question") or "")
    params = parse_params(row.get("extracted_params"))
    rule_result = classify_question(
        raw_text,
        source=str(row.get("source") or "tz116_offline"),
        metadata={"llm_bypass": True},
        config=QuestionClassifierConfig(llm_enabled=False),
    )
    model_label = first_text(row, MODEL_LABEL_FIELDS)
    final_label = rule_result.theme_id
    method = "rule_off"
    comparison = ""
    missing_model = "0"
    if mode == "shadow":
        method = "rule_shadow"
        if model_label:
            comparison = "agree" if model_label == rule_result.theme_id else "disagree"
    elif mode == "primary":
        if model_label:
            final_label = model_label
            method = "precomputed_model_primary"
            comparison = "agree" if model_label == rule_result.theme_id else "disagree"
        else:
            method = "missing_model_primary_blocked"
            missing_model = "1"
    return {
        **{str(key): str(value) for key, value in row.items()},
        "rule_theme_id": rule_result.theme_id,
        "model_theme_id": model_label,
        "predicted_theme_id": final_label,
        "classification_method": method,
        "model_comparison": comparison,
        "model_required_missing": missing_model,
        "params_used": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }


def normalize_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MODES else "off"


def parse_params(value: Any) -> dict[str, str]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(val) for key, val in parsed.items()}


def first_text(row: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, Any], predictions: list[Mapping[str, Any]]) -> str:
    mismatches = [row for row in predictions if str(row.get("human_label") or "") != str(row.get("predicted_theme_id") or "")][:20]
    lines = [
        "# TZ-116 C Question Catalog Offline Measurement",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Total: `{summary['total']}`",
        f"- Accuracy: `{summary['accuracy']:.4f}`",
        f"- Macro-F1: `{summary['macro_f1']:.4f}`",
        f"- LLM calls total: `{summary['llm_calls_total']}`",
        "",
        "Safety: no live LLM calls and no main catalog rebuild.",
        "",
        "## First Mismatches",
        "",
        "| id | human | predicted | rule | model |",
        "|---|---|---|---|---|",
    ]
    for row in mismatches:
        lines.append(
            f"| `{row.get('question_id', '')}` | `{row.get('human_label', '')}` | `{row.get('predicted_theme_id', '')}` | "
            f"`{row.get('rule_theme_id', '')}` | `{row.get('model_theme_id', '')}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-116 C: offline question catalog measurement without live LLM calls.")
    parser.add_argument("--input", required=True, help="CSV with question_id, raw_text, human_label and optional precomputed model label.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz116_question_catalog_offline_measure")
    parser.add_argument("--mode", choices=sorted(MODES), default="off")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
