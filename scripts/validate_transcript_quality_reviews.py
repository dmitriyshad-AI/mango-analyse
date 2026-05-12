#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.transcript_quality_review_validator import (
    config_from_args,
    parse_args,
    validate_transcript_quality_reviews,
)


def main() -> int:
    args = parse_args()
    summary = validate_transcript_quality_reviews(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
