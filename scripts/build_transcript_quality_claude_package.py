#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.transcript_quality_claude_package import (
    build_transcript_quality_claude_package,
    config_from_args,
    parse_args,
)


def main() -> int:
    args = parse_args()
    summary = build_transcript_quality_claude_package(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
