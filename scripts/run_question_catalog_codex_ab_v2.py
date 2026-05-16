#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from mango_mvp.question_catalog.calibration_metrics import compute_classification_metrics, validate_labeled_rows
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids


DEFAULT_INPUT = Path("product_data/question_catalog/stratified_calibration_sample_v2_labeled.csv")
DEFAULT_OUT_DIR = Path("product_data/question_catalog/codex_ab_v2")
TAXONOMY_PATH = Path("src/mango_mvp/question_catalog/themes_taxonomy.yaml")
PROMPT_VERSION = "question_catalog_codex_ab_v2_prompt_v1"


@dataclass(frozen=True)
class Candidate:
    label: str
    model: str
    reasoning_effort: str

    @classmethod
    def parse(cls, raw: str) -> "Candidate":
        parts = raw.split(":")
        if len(parts) == 1:
            model = parts[0].strip()
            return cls(label=f"{model}_medium", model=model, reasoning_effort="medium")
        if len(parts) == 2:
            model, reasoning = (part.strip() for part in parts)
            return cls(label=f"{model}_{reasoning}", model=model, reasoning_effort=reasoning)
        if len(parts) == 3:
            label, model, reasoning = (part.strip() for part in parts)
            return cls(label=label, model=model, reasoning_effort=reasoning)
        raise ValueError(f"Bad candidate format: {raw!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test Question Catalog v2 classification through Codex CLI.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--taxonomy", default=str(TAXONOMY_PATH))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Ignore cached Codex CLI batch responses.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate as model:reasoning or label:model:reasoning. Repeatable.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    taxonomy_path = Path(args.taxonomy)
    rows = read_csv(input_path)
    if args.max_rows:
        rows = rows[: args.max_rows]
    valid_labels = load_valid_theme_and_service_ids()
    errors = validate_labeled_rows(rows, valid_labels)
    if errors:
        raise SystemExit("Invalid calibration labels:\n" + "\n".join(errors[:20]))

    candidates = [Candidate.parse(raw) for raw in args.candidate] or [
        Candidate("gpt-5.4-mini_medium", "gpt-5.4-mini", "medium"),
        Candidate("gpt-5.5_medium", "gpt-5.5", "medium"),
        Candidate("gpt-5.5_xhigh", "gpt-5.5", "xhigh"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"[{candidate.label}] start rows={len(rows)} batch_size={args.batch_size}", flush=True)
        result = run_candidate(
            candidate,
            rows,
            taxonomy_path=taxonomy_path,
            out_dir=out_dir,
            batch_size=max(1, args.batch_size),
            timeout_sec=max(30, args.timeout_sec),
            force=args.force,
        )
        all_summaries.append(result)
        print(
            json.dumps(
                {
                    "candidate": candidate.label,
                    "macro_f1": result["macro_f1"],
                    "accuracy": result["accuracy"],
                    "tokens_used": result["tokens_used"],
                    "passed": result["passed"],
                    "report": result["report_path"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_combined_report(out_dir / "CODEX_AB_SUMMARY.md", all_summaries)
    (out_dir / "codex_ab_summary.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "CODEX_AB_SUMMARY.md"), "candidates": len(all_summaries)}, ensure_ascii=False, indent=2))


def run_candidate(
    candidate: Candidate,
    rows: list[dict[str, str]],
    *,
    taxonomy_path: Path,
    out_dir: Path,
    batch_size: int,
    timeout_sec: int,
    force: bool,
) -> dict[str, Any]:
    candidate_dir = out_dir / safe_name(candidate.label)
    cache_dir = candidate_dir / "cache"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, str]] = []
    total_tokens = 0
    total_elapsed = 0.0
    failed_batches = 0
    cache_hits = 0
    for index, batch in enumerate(chunks(rows, batch_size), start=1):
        prompt = build_prompt(batch, taxonomy_path=taxonomy_path)
        batch_result = run_codex_batch(
            prompt,
            candidate=candidate,
            cache_dir=cache_dir,
            timeout_sec=timeout_sec,
            force=force,
        )
        total_tokens += batch_result["tokens_used"]
        total_elapsed += batch_result["elapsed_sec"]
        cache_hits += 1 if batch_result["cache_hit"] else 0
        parsed_items = batch_result["items"]
        by_id = {str(item.get("question_id")): item for item in parsed_items}
        if len(by_id) != len(batch):
            failed_batches += 1
        for row in batch:
            item = by_id.get(row["question_id"], {})
            predicted = str(item.get("theme_id") or "service:S2_unclear").strip()
            confidence = clamp_float(item.get("confidence"), default=0.0)
            predictions.append(
                {
                    **row,
                    "predicted_theme_id": predicted,
                    "predicted_confidence": f"{confidence:.6f}",
                    "classification_method": "codex_cli",
                    "codex_model": candidate.model,
                    "codex_reasoning_effort": candidate.reasoning_effort,
                    "codex_reasoning": str(item.get("reasoning") or ""),
                    "batch_index": str(index),
                    "batch_cache_hit": "1" if batch_result["cache_hit"] else "0",
                    "is_correct": "1" if predicted == row["human_label"] else "0",
                }
            )
        print(f"[{candidate.label}] batch {index} rows={len(batch)} cache={batch_result['cache_hit']} tokens={batch_result['tokens_used']}", flush=True)

    metrics = compute_classification_metrics(predictions)
    method_counts = Counter(row["classification_method"] for row in predictions)
    predictions_path = candidate_dir / "predictions.csv"
    metrics_path = candidate_dir / "metrics.json"
    report_path = candidate_dir / "REPORT.md"
    write_csv(predictions_path, predictions)
    payload = metrics_to_json(
        metrics,
        candidate=candidate,
        method_counts=method_counts,
        tokens_used=total_tokens,
        elapsed_sec=total_elapsed,
        failed_batches=failed_batches,
        cache_hits=cache_hits,
        batch_count=(len(rows) + batch_size - 1) // batch_size,
        predictions_path=predictions_path,
        report_path=report_path,
    )
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_candidate_report(payload, predictions), encoding="utf-8")
    return payload


def run_codex_batch(
    prompt: str,
    *,
    candidate: Candidate,
    cache_dir: Path,
    timeout_sec: int,
    force: bool,
) -> dict[str, Any]:
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": candidate.model,
                "reasoning_effort": candidate.reasoning_effort,
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists() and not force:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
        return payload

    with tempfile.NamedTemporaryFile(prefix="mango_qc_codex_", suffix=".txt") as out_file:
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--model",
            candidate.model,
            "-c",
            f'model_reasoning_effort="{candidate.reasoning_effort}"',
            "--output-last-message",
            out_file.name,
            "-",
        ]
        started = time.time()
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, check=False, timeout=timeout_sec)
        elapsed = time.time() - started
        raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise RuntimeError(f"codex exec failed for {candidate.label}: rc={proc.returncode}\n{stderr_tail}")
    items = parse_items(raw or proc.stdout or proc.stderr)
    payload = {
        "items": items,
        "raw": raw,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-60:]),
        "tokens_used": parse_tokens_used(proc.stderr or ""),
        "elapsed_sec": elapsed,
        "cache_hit": False,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_prompt(rows: Sequence[Mapping[str, str]], *, taxonomy_path: Path) -> str:
    taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    topics = []
    for item in taxonomy["themes"]:
        topics.append(
            {
                "id": item["theme_id"],
                "name": item["theme_name"],
                "description": item.get("short_description", ""),
                "examples": item.get("example_phrasings", [])[:3],
            }
        )
    services = []
    for item in taxonomy["service_categories"]:
        services.append(
            {
                "id": item["service_id"],
                "name": item["service_name"],
                "description": item.get("short_description", ""),
                "examples": item.get("example_phrasings", [])[:3],
            }
        )
    batch = [
        {
            "question_id": row["question_id"],
            "raw_text": row["raw_text"],
            "source": row.get("source") or "",
            "extracted_params": parse_json_object(row.get("extracted_params")),
        }
        for row in rows
    ]
    return (
        "Ты классификатор клиентских вопросов для образовательной компании Фотон / УНПК МФТИ.\n"
        "Нужно выбрать ровно одну тему или служебную категорию для каждого вопроса.\n"
        "Не добавляй новые темы. Не используй старые fallback-классы. Если вопрос непонятен, выбирай service:S2_unclear.\n"
        "Верни только JSON без Markdown в формате: "
        "{\"items\":[{\"question_id\":\"...\",\"theme_id\":\"theme:001_pricing\",\"confidence\":0.95,\"reasoning\":\"коротко\"}]}.\n"
        "Все question_id из входа должны быть в ответе ровно один раз.\n\n"
        "Темы:\n"
        f"{json.dumps(topics, ensure_ascii=False, indent=2)}\n\n"
        "Служебные категории:\n"
        f"{json.dumps(services, ensure_ascii=False, indent=2)}\n\n"
        "Особые правила:\n"
        "- Возврат денег — theme:009_refund.\n"
        "- Налоговый вычет / лицензия для вычета — theme:008_tax_deduction.\n"
        "- Материнский капитал — theme:007_matkap_payment.\n"
        "- Статус уже сделанной оплаты, чек, квитанция, счёт на оплату — theme:003_payment_status, если клиент просит подтвердить/получить платежный документ.\n"
        "- Способ оплаты — theme:002_payment_method, если клиент спрашивает как оплатить.\n"
        "- Перенос, заморозка, изменение условий — theme:010_change_terms.\n"
        "- Вопрос родителя про прогресс, посещаемость или отметки ученика — theme:032_student_progress_inquiry.\n"
        "- Позитивная обратная связь — theme:019a_positive_feedback; жалоба или претензия — theme:019b_negative_feedback.\n\n"
        "Вопросы для классификации:\n"
        f"{json.dumps(batch, ensure_ascii=False, indent=2)}\n"
    )


def parse_items(text: str) -> list[dict[str, Any]]:
    payload = extract_json_object(text)
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Codex response does not contain items list")
    return [item for item in items if isinstance(item, dict)]


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise RuntimeError("Codex response JSON root must be an object")
    return payload


def parse_tokens_used(stderr: str) -> int:
    match = re.search(r"tokens used\s*([0-9\s\u00a0]+)", stderr or "", flags=re.I)
    if not match:
        return 0
    digits = "".join(ch for ch in match.group(1) if ch.isdigit())
    return int(digits) if digits else 0


def metrics_to_json(
    metrics,
    *,
    candidate: Candidate,
    method_counts: Counter[str],
    tokens_used: int,
    elapsed_sec: float,
    failed_batches: int,
    cache_hits: int,
    batch_count: int,
    predictions_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "question_catalog_codex_ab_v2_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate.label,
        "model": candidate.model,
        "reasoning_effort": candidate.reasoning_effort,
        "total": metrics.total,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "label_count": metrics.label_count,
        "classification_method_counts": dict(method_counts),
        "tokens_used": tokens_used,
        "estimated_dollar_cost": None,
        "cost_note": "Codex CLI subscription run exposes tokens used, not dollar cost.",
        "elapsed_sec": elapsed_sec,
        "batch_count": batch_count,
        "cache_hits": cache_hits,
        "failed_batches": failed_batches,
        "passed": metrics.macro_f1 >= 0.85,
        "predictions_path": str(predictions_path),
        "report_path": str(report_path),
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


def build_candidate_report(payload: Mapping[str, Any], predictions: list[Mapping[str, Any]]) -> str:
    mismatches = [row for row in predictions if str(row.get("is_correct")) != "1"][:25]
    lines = [
        f"# Codex A/B Report: {payload['candidate']}",
        "",
        f"- Model: `{payload['model']}`",
        f"- Reasoning effort: `{payload['reasoning_effort']}`",
        f"- Total: {payload['total']}",
        f"- Correct: {payload['correct']}",
        f"- Accuracy: {payload['accuracy']:.4f}",
        f"- Macro-F1: {payload['macro_f1']:.4f}",
        f"- Tokens used: {payload['tokens_used']}",
        f"- Dollar cost: not exposed by Codex CLI subscription",
        f"- Elapsed seconds: {payload['elapsed_sec']:.1f}",
        f"- Cache hits: {payload['cache_hits']}/{payload['batch_count']}",
        "",
        "## Worst Recall",
        "",
        "| Theme | Support | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for item in payload["worst_recall"]:
        lines.append(f"| `{item['theme_id']}` | {item['support']} | {item['recall']:.4f} | {item['f1']:.4f} |")
    lines.extend(["", "## Per Theme", "", "| Theme | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"])
    for item in payload["per_theme"]:
        lines.append(
            f"| `{item['theme_id']}` | {item['support']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )
    lines.extend(["", "## First Mismatches", "", "| Row | Human label | Predicted label | Text |", "|---:|---|---|---|"])
    for row in mismatches:
        text = str(row.get("raw_text") or "").replace("|", "\\|").replace("\n", " ")[:240]
        lines.append(
            f"| {row.get('question_id', '')} | `{row.get('human_label', '')}` | `{row.get('predicted_theme_id', '')}` | {text} |"
        )
    return "\n".join(lines) + "\n"


def write_combined_report(path: Path, summaries: list[Mapping[str, Any]]) -> None:
    ranked = sorted(summaries, key=lambda item: (-float(item["macro_f1"]), int(item["tokens_used"])))
    lines = [
        "# Question Catalog v2 Codex A/B Summary",
        "",
        "Dollar cost is not exposed by Codex CLI subscription runs, so cost is reported as tokens used plus elapsed time.",
        "",
        "| Rank | Candidate | Model | Effort | Macro-F1 | Accuracy | Correct | Tokens | Seconds | Report |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | `{item['candidate']}` | `{item['model']}` | `{item['reasoning_effort']}` | "
            f"{item['macro_f1']:.4f} | {item['accuracy']:.4f} | {item['correct']}/{item['total']} | "
            f"{item['tokens_used']} | {item['elapsed_sec']:.1f} | `{item['report_path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chunks(rows: Sequence[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for index in range(0, len(rows), size):
        yield list(rows[index : index + size])


def parse_json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def clamp_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, parsed))


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "candidate"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
