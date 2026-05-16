#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mango_mvp.deal_aware.deal_attribution import AttributionPaths, build_deal_attribution_dry_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_ROOT = PROJECT_ROOT / "stable_runtime" / "deal_aware_stage1_snapshot_20260513_v1"
DEFAULT_AMO_LIVE_ROOT = PROJECT_ROOT / "stable_runtime" / "deal_aware_amo_live_snapshot_20260513_v1"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "stable_runtime" / "deal_aware_stage2_attribution_20260513_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deal-aware Stage 2 call-to-deal attribution dry-run.")
    parser.add_argument("--stage1-root", default=str(DEFAULT_STAGE1_ROOT))
    parser.add_argument("--amo-live-root", default=str(DEFAULT_AMO_LIVE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_deal_attribution_dry_run(
        AttributionPaths(
            stage1_snapshot_root=Path(args.stage1_root).expanduser().resolve(),
            amo_live_snapshot_root=Path(args.amo_live_root).expanduser().resolve(),
            out_root=Path(args.out_root).expanduser().resolve(),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
