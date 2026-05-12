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

from mango_mvp.productization.amo_duplicate_after_staff_done import (  # noqa: E402
    DEFAULT_CURRENT_RUNTIME_PATH,
    DEFAULT_DUPLICATE_PACK_ROOT,
    DEFAULT_FROZEN_CORPUS,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPORTS_ROOT,
    build_amo_duplicate_after_staff_done_pipeline,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_amo_duplicate_after_staff_done_pipeline(
        project_root=Path(args.project_root),
        duplicate_pack_root=Path(args.duplicate_pack_root),
        report_dir=Path(args.report_dir) if args.report_dir else None,
        reports_root=Path(args.reports_root),
        out_root=Path(args.out_root),
        current_runtime_path=Path(args.current_runtime),
        frozen_corpus_jsonl=Path(args.frozen_corpus_jsonl),
        analysis_date=args.analysis_date or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("status") in {"ready_for_quality_gate", "partial_ready_for_quality_gate", "waiting_for_staff_done_and_recheck"} else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simplified AMO duplicate after-staff-done pipeline.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--duplicate-pack-root", default=str(DEFAULT_DUPLICATE_PACK_ROOT))
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--current-runtime", default=str(DEFAULT_CURRENT_RUNTIME_PATH))
    parser.add_argument("--frozen-corpus-jsonl", default=str(DEFAULT_FROZEN_CORPUS))
    parser.add_argument("--analysis-date", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
