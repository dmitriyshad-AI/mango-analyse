#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.question_catalog.answer_review_pack import (
    DEFAULT_BUILD_DATE,
    DEFAULT_CATALOG_ROOT,
    DEFAULT_ROW_LIMIT,
    build_pack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 100+ question-answer quality review pack for ROP/Claude.")
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--date", default=DEFAULT_BUILD_DATE)
    parser.add_argument("--iteration", default="iter1")
    parser.add_argument("--row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_csv = args.catalog_root / f"question_answer_quality_review_{args.date}_{args.iteration}.csv"
    output_summary = args.catalog_root / f"question_answer_quality_review_{args.date}_{args.iteration}.summary.json"
    result = build_pack(
        args.catalog_root,
        output_csv,
        output_summary,
        row_limit=args.row_limit,
        iteration=args.iteration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
