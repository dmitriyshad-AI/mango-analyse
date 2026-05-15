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

from mango_mvp.deal_aware.deal_quality_gate import DealQualityGatePaths, run_deal_quality_gate  # noqa: E402


DEFAULT_STAGE4_ROOT = ROOT / "stable_runtime" / "deal_aware_stage4_preview_20260513_v1"
DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "deal_aware_stage5_quality_gate_20260513_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deal-aware Stage 5 quality gate.")
    parser.add_argument("--stage4-root", default=str(DEFAULT_STAGE4_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--analysis-date", default="2026-05-13")
    parser.add_argument("--question-catalog-source-index", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_deal_quality_gate(
        DealQualityGatePaths(
            stage4_preview_root=Path(args.stage4_root).expanduser().resolve(),
            out_root=Path(args.out_root).expanduser().resolve(),
            analysis_date=args.analysis_date,
            question_catalog_source_index_json=Path(args.question_catalog_source_index).expanduser().resolve()
            if args.question_catalog_source_index
            else None,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["readiness"]["passed_for_stage6_dry_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
