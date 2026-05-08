#!/usr/bin/env python3
"""Materialize a quarantine audio package from a validated import plan.

This script writes only inside the quarantine package directory declared by the
plan. It does not write runtime DBs, does not touch stable_runtime, and does not
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

from mango_mvp.productization.quarantine_import import (  # noqa: E402
    MATERIALIZE_MODES,
    materialize_quarantine_package,
)


DEFAULT_PLAN = "_local_archive_mango_api_downloads_20260507/quarantine_import/quarantine_import_plan.json"
DEFAULT_OUT = "_local_archive_mango_api_downloads_20260507/quarantine_import/materialization_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = materialize_quarantine_package(
        plan_path=Path(args.plan),
        mode=args.mode,
        verify_checksum=not args.no_verify_checksum,
        overwrite=args.overwrite,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary": report["summary"], "audit": report["audit"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("blocked", 0) == 0 else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a validated Mango quarantine package.")
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=MATERIALIZE_MODES, default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify-checksum", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
