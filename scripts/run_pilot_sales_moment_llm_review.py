#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.insights.llm_review import config_from_args, parse_args, run_pilot_sales_moment_llm_review


def main() -> int:
    args = parse_args()
    summary = run_pilot_sales_moment_llm_review(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
