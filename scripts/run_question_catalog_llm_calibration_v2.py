#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.config import get_settings
from mango_mvp.question_catalog.calibration_metrics import compute_classification_metrics, validate_labeled_rows
from mango_mvp.question_catalog.classifier import QuestionClassifierConfig, classify_question, load_valid_theme_and_service_ids
from mango_mvp.question_catalog.theme_assigner_llm import ThemeAssignerConfig


DEFAULT_INPUT = Path("product_data/question_catalog/stratified_calibration_sample_v2_labeled.csv")
DEFAULT_OUT_DIR = Path("product_data/question_catalog/llm_calibration_v2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Question Catalog v2 LLM calibration on labeled 100-row sample.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mode", choices=("rule", "llm"), default="rule", help="rule is a no-cost baseline; llm is the D.2 acceptance run.")
    parser.add_argument("--use-llm", action="store_true", help="Call OpenAI LLM. Without this flag, runs rule-only baseline.")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--fail-below-threshold", action="store_true")
    args = parser.parse_args()
    use_llm = args.use_llm or args.mode == "llm"

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    rows = read_csv(input_path)
    if args.max_rows:
        rows = rows[: args.max_rows]

    valid_labels = load_valid_theme_and_service_ids()
    validation_errors = validate_labeled_rows(rows, valid_labels)
    if validation_errors:
        raise SystemExit("Invalid calibration labels:\n" + "\n".join(validation_errors[:20]))

    if use_llm and not openai_key_available():
        raise SystemExit(
            "OPENAI_API_KEY is required for LLM calibration. "
            "Add it to the shell environment or project .env, then rerun with --mode llm."
        )

    threshold = ThemeAssignerConfig.from_env().macro_f1_threshold
    classifier_config = QuestionClassifierConfig(
        llm_enabled=bool(use_llm),
        llm_config=ThemeAssignerConfig.from_env(),
    )

    predictions: list[dict[str, str]] = []
    for row in rows:
        params = parse_params(row.get("extracted_params"))
        result = classify_question(
            row.get("raw_text", ""),
            source=row.get("source") or "calibration",
            metadata={"llm_bypass": not use_llm},
            config=classifier_config,
        )
        predictions.append(
            {
                **row,
                "predicted_theme_id": result.theme_id,
                "predicted_confidence": f"{result.confidence:.6f}",
                "classification_method": result.classification_method,
                "llm_model": result.llm_model,
                "llm_reasoning": result.reasoning,
                "params_used": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "is_correct": "1" if result.theme_id == row.get("human_label") else "0",
            }
        )

    metrics = compute_classification_metrics(predictions)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "calibration_predictions_v2.csv"
    metrics_path = out_dir / "calibration_metrics_v2.json"
    report_path = out_dir / "SELECTOR_D_CALIBRATION_REPORT.md"
    write_csv(predictions_path, predictions)
    metrics_payload = metrics_to_json(metrics, method_counts=Counter(row["classification_method"] for row in predictions), threshold=threshold)
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        build_report(metrics_payload, predictions=predictions, predictions_path=predictions_path, input_path=input_path, use_llm=use_llm),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {"report": str(report_path), "macro_f1": metrics.macro_f1, "passed": metrics.macro_f1 >= threshold, "mode": "llm" if use_llm else "rule"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.fail_below_threshold and metrics.macro_f1 < threshold:
        raise SystemExit(2)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def openai_key_available() -> bool:
    try:
        get_settings()
    except Exception:  # noqa: BLE001 - key detection should not fail on an unrelated malformed setting.
        pass
    return bool(os.getenv("OPENAI_API_KEY"))


def parse_params(value: Any) -> dict[str, str]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(val) for key, val in parsed.items()}


def metrics_to_json(metrics, *, method_counts: Counter[str], threshold: float) -> dict[str, Any]:
    return {
        "schema_version": "question_catalog_v2_calibration_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {"macro_f1": threshold},
        "total": metrics.total,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "label_count": metrics.label_count,
        "passed": metrics.macro_f1 >= threshold,
        "classification_method_counts": dict(method_counts),
        "per_theme": [
            {
                "theme_id": item.label,
                "support": item.support,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
                "tp": item.true_positive,
                "fp": item.false_positive,
                "fn": item.false_negative,
            }
            for item in metrics.per_label
        ],
        "worst_recall": [
            {"theme_id": item.label, "support": item.support, "recall": item.recall, "f1": item.f1}
            for item in metrics.worst_recall()
        ],
    }


def build_report(
    payload: Mapping[str, Any],
    *,
    predictions: list[Mapping[str, Any]],
    predictions_path: Path,
    input_path: Path,
    use_llm: bool,
) -> str:
    mismatches = [row for row in predictions if str(row.get("is_correct")) != "1"][:20]
    lines = [
        "# SELECTOR D Calibration Report",
        "",
        f"Input: `{input_path}`",
        f"Predictions: `{predictions_path}`",
        f"Mode: `{'llm' if use_llm else 'rule_only_baseline'}`",
    ]
    if not use_llm:
        lines.extend(
            [
                "",
                "> This is a no-cost rule-only baseline. It smoke-tests the labeled sample and reporting pipeline, but it is not the D.2 acceptance run. The acceptance run requires `--mode llm` and `OPENAI_API_KEY`.",
            ]
        )
    lines.extend(
        [
        "",
        "## Summary",
        "",
        f"- Total: {payload['total']}",
        f"- Correct: {payload['correct']}",
        f"- Accuracy: {payload['accuracy']:.4f}",
        f"- Macro-F1: {payload['macro_f1']:.4f}",
        f"- Label count: {payload['label_count']}",
        f"- Threshold: {payload['thresholds']['macro_f1']:.4f}",
        f"- Passed: {payload['passed']}",
        f"- Classification methods: `{json.dumps(payload['classification_method_counts'], ensure_ascii=False)}`",
        "",
        "## Worst Recall",
        "",
        "| Theme | Support | Recall | F1 |",
        "|---|---:|---:|---:|",
        ]
    )
    for item in payload["worst_recall"]:
        lines.append(f"| `{item['theme_id']}` | {item['support']} | {item['recall']:.4f} | {item['f1']:.4f} |")
    lines.extend(["", "## Per Theme", "", "| Theme | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"])
    for item in payload["per_theme"]:
        lines.append(
            f"| `{item['theme_id']}` | {item['support']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )
    lines.extend(["", "## First Mismatches", "", "| Row | Human label | Predicted label | Text |", "|---:|---|---|---|"])
    for row in mismatches:
        text = str(row.get("raw_text") or "").replace("|", "\\|").replace("\n", " ")[:220]
        lines.append(
            f"| {row.get('question_id', '')} | `{row.get('human_label', '')}` | `{row.get('predicted_theme_id', '')}` | {text} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
