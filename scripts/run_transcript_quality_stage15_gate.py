#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.stage15_export_quality_gate import build_stage15_export_quality_gate, config_from_args, parse_args


def main() -> int:
    args = parse_args()
    summary = build_stage15_export_quality_gate(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
