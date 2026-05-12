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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mango_mvp.productization.amo_waiting_autonomous_work import (  # noqa: E402
    DEFAULT_CONTACT_WRITEBACK_REPORTS_ROOT,
    DEFAULT_CURRENT_RUNTIME_PATH,
    DEFAULT_FROZEN_CORPUS,
    DEFAULT_OUT_ROOT,
    DEFAULT_QUEUE_ROOT,
    DEFAULT_STAGE15_SUMMARY,
    build_amo_waiting_autonomous_work,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_amo_waiting_autonomous_work(
        project_root=Path(args.project_root),
        queue_root=Path(args.queue_root),
        out_root=Path(args.out_root),
        current_runtime_path=Path(args.current_runtime),
        contact_writeback_reports_root=Path(args.contact_writeback_reports_root),
        stage15_summary=Path(args.stage15_summary),
        frozen_corpus_jsonl=Path(args.frozen_corpus_jsonl),
        analysis_date=args.analysis_date or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare safe autonomous AMO work while duplicate cleanup is pending.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--queue-root", default=str(DEFAULT_QUEUE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--current-runtime", default=str(DEFAULT_CURRENT_RUNTIME_PATH))
    parser.add_argument("--contact-writeback-reports-root", default=str(DEFAULT_CONTACT_WRITEBACK_REPORTS_ROOT))
    parser.add_argument("--stage15-summary", default=str(DEFAULT_STAGE15_SUMMARY))
    parser.add_argument("--frozen-corpus-jsonl", default=str(DEFAULT_FROZEN_CORPUS))
    parser.add_argument("--analysis-date", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
