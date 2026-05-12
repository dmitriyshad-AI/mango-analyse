#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.transcript_quality_backfill import (
    config_from_args,
    parse_args,
    run_transcript_quality_backfill,
)


def main() -> int:
    args = parse_args()
    summary = run_transcript_quality_backfill(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
