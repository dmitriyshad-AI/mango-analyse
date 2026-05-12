#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.transcript_quality_hard_gate_backfill_dry_run import (
    build_hard_gate_backfill_dry_run,
    config_from_args,
    parse_args,
)


def main() -> int:
    args = parse_args()
    summary = build_hard_gate_backfill_dry_run(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
