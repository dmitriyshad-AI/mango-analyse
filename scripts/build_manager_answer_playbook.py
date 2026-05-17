#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.knowledge_base.manager_answer_playbook import (
    DEFAULT_CATALOG_ROOT,
    DEFAULT_MIN_SAMPLE_SIZE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PATTERN_LIMIT,
    DEFAULT_SAMPLE_SIZE,
    analyze_manager_answers,
    build_manager_answer_playbook,
    load_question_items,
    write_playbook_outputs,
)


DEFAULT_INPUT = Path("product_data/question_catalog/customer_question_items.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manager answer playbook from redacted question catalog.")
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--input", type=Path, default=None, help="Legacy mode: read only customer_question_items.jsonl.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--min-sample-size", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=None, help="Legacy alias for --sample-size.")
    parser.add_argument("--minimum-size", type=int, default=None, help="Legacy alias for --min-sample-size.")
    parser.add_argument("--pattern-limit", type=int, default=DEFAULT_PATTERN_LIMIT)
    parser.add_argument("--skip-xlsx", action="store_true", help="Write CSV/JSONL/MD only.")
    args = parser.parse_args()

    if "stable_runtime" in args.out_dir.resolve().parts:
        raise ValueError("Refusing to write manager playbook under stable_runtime")

    sample_size = args.sample_size or args.target_size or DEFAULT_SAMPLE_SIZE
    min_sample_size = args.min_sample_size or args.minimum_size or DEFAULT_MIN_SAMPLE_SIZE
    if args.input is not None:
        items = load_question_items(args.input)
        records, patterns = analyze_manager_answers(
            items,
            target_size=sample_size,
            minimum_size=min_sample_size,
            pattern_limit=args.pattern_limit,
        )
        result = write_playbook_outputs(
            records,
            patterns,
            out_dir=args.out_dir,
            write_xlsx=not args.skip_xlsx,
        )
    else:
        result = build_manager_answer_playbook(
            args.catalog_root,
            args.out_dir,
            sample_size=sample_size,
            min_sample_size=min_sample_size,
            pattern_limit=args.pattern_limit,
            write_xlsx=not args.skip_xlsx,
        )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "mode": result["mode"],
                "summary": result["summary"],
                "safety": result["safety"],
                "outputs": result.get("outputs", {}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
