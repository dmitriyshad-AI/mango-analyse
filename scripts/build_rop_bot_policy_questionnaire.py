#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.question_catalog.rop_questionnaire import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    DEFAULT_SUMMARY,
    build_questionnaire,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Russian ROP questionnaire for bot answer policy decisions.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_questionnaire(args.source, args.output_csv, args.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
