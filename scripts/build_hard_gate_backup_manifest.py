#!/usr/bin/env python3
from __future__ import annotations

import json

from mango_mvp.quality.hard_gate_backup_manifest import (
    build_hard_gate_backup_manifest,
    config_from_args,
    parse_args,
)


def main() -> int:
    summary = build_hard_gate_backup_manifest(config_from_args(parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
