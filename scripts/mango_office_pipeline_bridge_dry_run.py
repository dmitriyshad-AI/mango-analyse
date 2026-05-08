#!/usr/bin/env python3
"""Build a read-only import plan from Mango capture manifest to legacy pipeline.

This script does not copy audio into the working folder, does not write DBs,
does not start ASR/R+A and does not write to CRM.
"""

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

from mango_mvp.productization.pipeline_bridge import (  # noqa: E402
    build_pipeline_bridge_plan,
    write_bridge_plan_csv,
)


DEFAULT_MANIFEST = "_local_archive_mango_api_downloads_20260507/capture_manifest.jsonl"
DEFAULT_SOURCE_DIR = "2026-03-09--26"
DEFAULT_DB = "mango_mvp.db"
DEFAULT_OUT = "_local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.json"
DEFAULT_CSV_OUT = "_local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.csv"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db_paths = tuple(Path(db_path) for db_path in (args.db or [DEFAULT_DB]))
    plan = build_pipeline_bridge_plan(
        manifest_path=Path(args.manifest),
        source_dir=Path(args.source_dir),
        db_paths=db_paths,
        tolerance_sec=args.tolerance_sec,
        verify_checksum=not args.no_verify_checksum,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_out:
        write_bridge_plan_csv(plan, Path(args.csv_out))
    print(json.dumps({"out": str(out_path), "summary": plan["summary"], "audit": plan["audit"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if plan["audit"].get("blocked", 0) == 0 else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only Mango capture -> legacy pipeline import plan.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--db", action="append", default=None, help="SQLite DB path to inspect read-only. Repeatable.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--tolerance-sec", type=int, default=120)
    parser.add_argument("--no-verify-checksum", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
