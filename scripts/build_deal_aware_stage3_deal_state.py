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

from mango_mvp.deal_aware.deal_state_classifier import (  # noqa: E402
    DealStatePaths,
    build_deal_state_classifier,
)


DEFAULT_STAGE2_ROOT = ROOT / "stable_runtime" / "deal_aware_stage2_attribution_20260513_v2"
DEFAULT_AMO_LIVE_ROOT = ROOT / "stable_runtime" / "deal_aware_amo_live_snapshot_20260513_v2"
DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "deal_aware_stage3_deal_state_20260513_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deal-aware Stage 3 deal-state classifier.")
    parser.add_argument("--stage2-root", default=str(DEFAULT_STAGE2_ROOT))
    parser.add_argument("--amo-live-root", default=str(DEFAULT_AMO_LIVE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_deal_state_classifier(
        DealStatePaths(
            stage2_attribution_root=Path(args.stage2_root).expanduser().resolve(),
            amo_live_snapshot_root=Path(args.amo_live_root).expanduser().resolve(),
            out_root=Path(args.out_root).expanduser().resolve(),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
