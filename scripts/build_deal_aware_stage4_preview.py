#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.deal_aware.deal_text_builder import DealTextPaths, build_deal_text_preview  # noqa: E402


DEFAULT_STAGE1_ROOT = ROOT / "stable_runtime" / "deal_aware_stage1_snapshot_20260513_v2"
DEFAULT_STAGE3_ROOT = ROOT / "stable_runtime" / "deal_aware_stage3_deal_state_20260513_v1"
DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "deal_aware_stage4_preview_20260513_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deal-aware Stage 4 deal text preview.")
    parser.add_argument("--stage1-root", default=str(DEFAULT_STAGE1_ROOT))
    parser.add_argument("--stage3-root", default=str(DEFAULT_STAGE3_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--analysis-date", default="2026-05-13")
    parser.add_argument("--customer-timeline-db", default=None)
    parser.add_argument("--enable-customer-timeline-context", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_deal_text_preview(
        DealTextPaths(
            stage1_snapshot_root=Path(args.stage1_root).expanduser().resolve(),
            stage3_deal_state_root=Path(args.stage3_root).expanduser().resolve(),
            out_root=Path(args.out_root).expanduser().resolve(),
            analysis_date=args.analysis_date,
        ),
        timeline_db=Path(args.customer_timeline_db).expanduser().resolve() if args.customer_timeline_db else None,
        include_timeline_context=bool(args.enable_customer_timeline_context),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
