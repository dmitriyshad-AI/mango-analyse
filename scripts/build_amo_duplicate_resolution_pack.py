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

from mango_mvp.productization.amo_duplicate_resolution import (  # noqa: E402
    DEFAULT_AMO_BASE_URL,
    DEFAULT_CURRENT_RUNTIME_PATH,
    DEFAULT_MANUAL_PACK_ROOT,
    DEFAULT_OUT_ROOT,
    build_amo_duplicate_resolution_pack,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = build_amo_duplicate_resolution_pack(
        manual_pack_root=Path(args.manual_pack_root),
        out_root=Path(args.out_root),
        current_runtime_path=Path(args.current_runtime),
        amo_base_url=args.amo_base_url,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only AMO duplicate/contact-mismatch resolution pack.")
    parser.add_argument("--manual-pack-root", default=str(DEFAULT_MANUAL_PACK_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--current-runtime", default=str(DEFAULT_CURRENT_RUNTIME_PATH))
    parser.add_argument("--amo-base-url", default=DEFAULT_AMO_BASE_URL)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
