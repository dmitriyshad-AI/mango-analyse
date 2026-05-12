#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.hard_gate_gpt_review import (
    config_from_args,
    parse_args,
    run_hard_gate_gpt_review,
)


def main() -> int:
    summary = run_hard_gate_gpt_review(config_from_args(parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
