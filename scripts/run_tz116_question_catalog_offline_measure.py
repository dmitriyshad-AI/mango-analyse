#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.subscription_llm import build_codex_exec_command, build_codex_exec_env
from mango_mvp.question_catalog.calibration_metrics import compute_classification_metrics
from mango_mvp.question_catalog.classifier import QuestionClassifierConfig, classify_question
from mango_mvp.question_catalog.codex_full_run import (
    DEFAULT_TAXONOMY_PATH,
    BatchSpec,
    CodexFullRunCandidate,
    build_full_classification_prompt,
    extract_json_object,
    sha256_file,
    sha256_text,
    validate_and_normalize_predictions,
)


MODES = {"off", "shadow", "primary"}
LLM_SOURCES = {"precomputed", "codex"}
MODEL_LABEL_FIELDS = ("model_theme_id", "llm_theme_id", "model_label", "llm_label", "predicted_theme_id_model")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_csv(input_path)
    llm_source = normalize_llm_source(args.llm_source)
    codex_predictions: dict[str, dict[str, Any]] = {}
    llm_calls_total = 0
    if mode in {"shadow", "primary"} and llm_source == "codex":
        codex_predictions, llm_calls_total = run_codex_shadow(source_rows, args=args, out_dir=out_dir)
    predictions = [
        build_prediction(
            row,
            mode=mode,
            llm_source=llm_source,
            row_index=index,
            codex_prediction=codex_predictions.get(row_question_id(row, index)),
        )
        for index, row in enumerate(source_rows, start=1)
    ]
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
        "llm_source": llm_source,
        "input": str(input_path),
        "records_total": len(predictions),
        "gold_labeled_total": metrics.total,
        "total": metrics.total,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "label_count": metrics.label_count,
        "classification_method_counts": dict(method_counts),
        "model_comparison_counts": dict(comparison_counts),
        "missing_model_labels": missing_model,
        "llm_calls_total": llm_calls_total,
        "safety": {
            "calls_live_llm": bool(llm_source == "codex" and mode in {"shadow", "primary"}),
            "model_transport": "codex_cli" if llm_source == "codex" and mode in {"shadow", "primary"} else "none",
            "uses_openai_api_key": False,
            "rebuilds_main_catalog": False,
            "writes_stable_runtime": False,
        },
    }
    write_csv(out_dir / "question_catalog_offline_predictions.csv", predictions)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary, predictions), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_prediction(
    row: Mapping[str, Any],
    *,
    mode: str,
    llm_source: str = "precomputed",
    row_index: int = 1,
    codex_prediction: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    raw_text = str(row.get("raw_text") or row.get("question") or "")
    params = parse_params(row.get("extracted_params"))
    rule_result = classify_question(
        raw_text,
        source=str(row.get("source") or "tz116_offline"),
        metadata={"llm_bypass": True},
        config=QuestionClassifierConfig(llm_enabled=False),
    )
    model_label = str((codex_prediction or {}).get("predicted_theme_id") or "").strip() or first_text(row, MODEL_LABEL_FIELDS)
    final_label = rule_result.theme_id
    method = "rule_off"
    comparison = ""
    missing_model = "0"
    if mode == "shadow":
        method = "rule_shadow_codex" if llm_source == "codex" else "rule_shadow"
        if model_label:
            comparison = "agree" if model_label == rule_result.theme_id else "disagree"
    elif mode == "primary":
        if model_label:
            final_label = model_label
            method = "codex_model_primary_blocked" if llm_source == "codex" else "precomputed_model_primary"
            comparison = "agree" if model_label == rule_result.theme_id else "disagree"
        else:
            method = "missing_model_primary_blocked"
            missing_model = "1"
    return {
        **{str(key): str(value) for key, value in row.items()},
        "question_id_resolved": row_question_id(row, row_index),
        "rule_theme_id": rule_result.theme_id,
        "model_theme_id": model_label,
        "model_confidence": str((codex_prediction or {}).get("confidence") or ""),
        "predicted_theme_id": final_label,
        "classification_method": method,
        "model_comparison": comparison,
        "model_required_missing": missing_model,
        "params_used": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }


def normalize_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MODES else "off"


def normalize_llm_source(value: Any) -> str:
    source = str(value or "precomputed").strip().lower()
    return source if source in LLM_SOURCES else "precomputed"


def row_question_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("question_id") or row.get("question_item_id") or row.get("id") or f"row:{index}").strip()


def run_codex_shadow(
    rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    candidate = CodexFullRunCandidate.parse(args.candidate)
    taxonomy_path = Path(args.taxonomy).expanduser().resolve()
    taxonomy_sha = sha256_file(taxonomy_path)
    batch_size = max(1, int(args.batch_size))
    prepared = [
        {
            "question_item_id": row_question_id(row, index),
            "tenant_id": str(row.get("tenant_id") or "foton"),
            "source_channel": str(row.get("source") or row.get("source_channel") or "tz116_offline"),
            "source_ref": str(row.get("source_ref") or row_question_id(row, index)),
            "question_class_id": "",
            "customer_text_redacted": str(row.get("raw_text") or row.get("question") or ""),
            "input_text_sha256": sha256_text(str(row.get("raw_text") or row.get("question") or "")),
            "metadata": {"extracted_params": parse_params(row.get("extracted_params"))},
        }
        for index, row in enumerate(rows, start=1)
    ]
    predictions: dict[str, dict[str, Any]] = {}
    calls = 0
    prediction_rows: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(prepared), batch_size), start=1):
        batch = BatchSpec(
            batch_id=f"tz116_batch_{batch_index:06d}",
            index=batch_index,
            rows=tuple(prepared[start : start + batch_size]),
        )
        prompt = build_full_classification_prompt(batch, taxonomy_path=taxonomy_path)
        raw = call_codex_batch(
            prompt,
            codex_bin=str(args.codex_bin or "codex"),
            candidate=candidate,
            timeout_sec=max(30, int(args.timeout_sec)),
        )
        calls += 1
        payload = extract_json_object(raw)
        normalized = validate_and_normalize_predictions(
            payload,
            batch,
            candidate=candidate,
            taxonomy_sha256=taxonomy_sha,
            prompt_sha256=sha256_text(prompt),
            response_sha256=sha256_text(raw),
        )
        for item in normalized:
            predictions[str(item["question_item_id"])] = item
            prediction_rows.append(item)
    write_jsonl(out_dir / "question_catalog_codex_shadow_predictions.jsonl", prediction_rows)
    return predictions, calls


def call_codex_batch(
    prompt: str,
    *,
    codex_bin: str,
    candidate: CodexFullRunCandidate,
    timeout_sec: int,
) -> str:
    with tempfile.NamedTemporaryFile(prefix="mango_tz116_qcatalog_codex_", suffix=".json") as out_file:
        output_path = Path(out_file.name)
        cmd = build_codex_exec_command(
            output_path=output_path,
            codex_bin=codex_bin,
            model=candidate.model,
            reasoning_effort=candidate.reasoning_effort,
            isolated=True,
        )
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=build_codex_exec_env(),
        )
        raw = output_path.read_text(encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-40:])
        raise RuntimeError(f"codex exec failed rc={proc.returncode}: {stderr_tail}")
    return raw or proc.stdout or proc.stderr


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def render_report(summary: dict[str, Any], predictions: list[Mapping[str, Any]]) -> str:
    mismatches = [row for row in predictions if str(row.get("human_label") or "") != str(row.get("predicted_theme_id") or "")][:20]
    lines = [
        "# TZ-116 C Question Catalog Offline Measurement",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- LLM source: `{summary['llm_source']}`",
        f"- Records: `{summary['records_total']}`",
        f"- Gold-labeled: `{summary['gold_labeled_total']}`",
        f"- Accuracy: `{summary['accuracy']:.4f}`",
        f"- Macro-F1: `{summary['macro_f1']:.4f}`",
        f"- LLM calls total: `{summary['llm_calls_total']}`",
        "",
        "Safety: Codex CLI only when llm_source=codex, no OpenAI API key, no main catalog rebuild.",
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
    parser = argparse.ArgumentParser(description="TZ-116 C: offline question catalog measurement.")
    parser.add_argument("--input", required=True, help="CSV with question_id, raw_text, human_label and optional precomputed model label.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz116_question_catalog_offline_measure")
    parser.add_argument("--mode", choices=sorted(MODES), default="off")
    parser.add_argument("--llm-source", choices=sorted(LLM_SOURCES), default="precomputed")
    parser.add_argument("--candidate", default="gpt-5.4-mini:medium")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
