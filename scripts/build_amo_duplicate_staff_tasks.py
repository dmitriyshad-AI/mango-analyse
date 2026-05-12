#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.amo_duplicate_staff_tasks import (  # noqa: E402
    DEFAULT_DUPLICATE_PACK_ROOT,
    DEFAULT_OUT_ROOT,
    build_amo_duplicate_staff_tasks,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = build_amo_duplicate_staff_tasks(
        duplicate_pack_root=Path(args.duplicate_pack_root),
        out_root=Path(args.out_root),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only staff tasks for AMO duplicate cleanup.")
    parser.add_argument("--duplicate-pack-root", default=str(DEFAULT_DUPLICATE_PACK_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
