#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.insights.knowledge_base import build_sales_insight_knowledge_base, config_from_args, parse_args


def main() -> int:
    args = parse_args()
    summary = build_sales_insight_knowledge_base(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
