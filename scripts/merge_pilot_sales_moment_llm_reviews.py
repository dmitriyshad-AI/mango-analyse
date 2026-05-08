#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mango_mvp.insights.llm_review import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    LLMReviewConfig,
    build_review_summary,
    read_jsonl,
    sorted_reviews,
    write_review_outputs,
)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    input_jsonl = (project_root / args.input_jsonl).resolve()
    out_root = (project_root / args.out_root).resolve()
    review_roots = [(project_root / root).resolve() for root in args.review_root]
    summary = merge_review_roots(
        project_root=project_root,
        input_jsonl=input_jsonl,
        out_root=out_root,
        review_roots=review_roots,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded pilot sales moment LLM review outputs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input-jsonl", default="stable_runtime/pilot_sales_moments_20260507/llm_sales_moment_input.jsonl")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--review-root", action="append", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    return parser.parse_args()


def merge_review_roots(
    *,
    project_root: Path,
    input_jsonl: Path,
    out_root: Path,
    review_roots: list[Path],
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    all_items = read_jsonl(input_jsonl)
    all_ids = [str(item.get("id") or "") for item in all_items if item.get("id")]

    merged_by_id: dict[str, dict[str, Any]] = {}
    duplicate_same = 0
    duplicate_conflicts: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    errors: list[dict[str, Any]] = []

    for root in review_roots:
        reviews_path = root / "reviews.jsonl"
        if not reviews_path.exists():
            source_counts[str(root)] = 0
            continue
        source_rows = read_jsonl(reviews_path)
        source_counts[str(root)] = len(source_rows)
        for row in source_rows:
            moment_id = str(row.get("moment_id") or "").strip()
            if not moment_id:
                continue
            previous = merged_by_id.get(moment_id)
            if previous is None:
                merged_by_id[moment_id] = row
                continue
            if canonical_json(previous) == canonical_json(row):
                duplicate_same += 1
                continue
            duplicate_conflicts.append({"moment_id": moment_id, "kept_from": str(root), "reason": "conflicting_duplicate"})

        errors.extend(read_errors_csv(root / "errors.csv", source_root=root))

    reviews = sorted_reviews(list(merged_by_id.values()))
    missing_ids = [moment_id for moment_id in all_ids if moment_id not in merged_by_id]
    extra_ids = [moment_id for moment_id in merged_by_id if moment_id not in set(all_ids)]
    if missing_ids:
        (out_root / "missing_moment_ids.txt").write_text("\n".join(missing_ids) + "\n", encoding="utf-8")
    if duplicate_conflicts:
        write_csv(out_root / "duplicate_conflicts.csv", duplicate_conflicts)
    if extra_ids:
        write_csv(out_root / "extra_moment_ids.csv", [{"moment_id": moment_id} for moment_id in sorted(extra_ids)])

    config = LLMReviewConfig(
        project_root=project_root,
        input_jsonl=input_jsonl,
        out_root=out_root,
        provider="codex_cli",
        model=model,
        reasoning_effort=reasoning_effort,
        limit=0,
        offset=0,
        sample_strategy="first",
        dry_run=False,
        force=False,
        cache_enabled=False,
    )
    provider_stats = {
        "single_provider_calls": 0,
        "codex_batch_provider_calls": 0,
        "codex_batch_fallback_single_calls": 0,
    }
    summary = build_review_summary(
        config,
        all_items,
        all_items,
        reviews,
        errors,
        skipped_existing=0,
        cache_hits=0,
        dry_run_count=0,
        provider_stats=provider_stats,
    )
    summary["merge"] = {
        "source_review_roots": [str(root) for root in review_roots],
        "source_counts": source_counts,
        "duplicate_same": duplicate_same,
        "duplicate_conflicts": len(duplicate_conflicts),
        "missing_moment_ids": len(missing_ids),
        "extra_moment_ids": len(extra_ids),
        "complete": len(missing_ids) == 0 and len(extra_ids) == 0 and not duplicate_conflicts,
    }
    outputs = write_review_outputs(out_root, summary, reviews, errors)
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def read_errors_csv(path: Path, *, source_root: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [{**row, "source_root": str(source_root)} for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def canonical_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
