#!/usr/bin/env python3
"""Build a dry-run quarantine import package plan from a bridge plan.

This script writes only planning artifacts: JSON plan and metadata.csv. It does
not copy/link audio into the working folder, does not write DBs and does not
start ASR/R+A.
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

from mango_mvp.productization.quarantine_import import build_quarantine_import_plan  # noqa: E402


DEFAULT_BRIDGE_PLAN = "_local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.json"
DEFAULT_OUT_ROOT = "_local_archive_mango_api_downloads_20260507/quarantine_import"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    quarantine_dir = Path(args.quarantine_dir) if args.quarantine_dir else out_root / "audio"
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else out_root / "metadata.csv"
    out_plan = Path(args.out) if args.out else out_root / "quarantine_import_plan.json"

    plan = build_quarantine_import_plan(
        bridge_plan_path=Path(args.bridge_plan),
        quarantine_dir=quarantine_dir,
        metadata_csv_path=metadata_csv,
        verify_checksum=not args.no_verify_checksum,
    )
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_plan), "summary": plan["summary"], "audit": plan["audit"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if plan["audit"].get("blocked", 0) == 0 else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dry-run quarantine import package plan.")
    parser.add_argument("--bridge-plan", default=DEFAULT_BRIDGE_PLAN)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--quarantine-dir")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--out")
    parser.add_argument("--no-verify-checksum", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
