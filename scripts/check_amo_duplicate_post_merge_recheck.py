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

from mango_mvp.productization.amo_duplicate_recheck import (  # noqa: E402
    DEFAULT_DUPLICATE_PACK_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPORTS_ROOT,
    build_amo_duplicate_post_merge_recheck,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_amo_duplicate_post_merge_recheck(
        duplicate_pack_root=Path(args.duplicate_pack_root),
        report_dir=Path(args.report_dir) if args.report_dir else None,
        reports_root=Path(args.reports_root),
        out_root=Path(args.out_root),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("passed") else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AMO duplicate post-merge dry-run recheck results.")
    parser.add_argument("--duplicate-pack-root", default=str(DEFAULT_DUPLICATE_PACK_ROOT))
    parser.add_argument("--report-dir", default="", help="Contact writeback dry-run report directory. If omitted, a matching report is searched by input path.")
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
