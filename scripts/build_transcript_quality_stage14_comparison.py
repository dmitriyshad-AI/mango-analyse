#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.stage14_quality_comparison import build_stage14_quality_comparison, config_from_args, parse_args


def main() -> int:
    args = parse_args()
    summary = build_stage14_quality_comparison(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
