#!/usr/bin/env python3
"""Build a dry-run bridge plan from controlled Mango recording downloads.

This command converts the Stage 7 download manifest to a capture-manifest view
and builds a read-only bridge/import plan. It does not copy audio to legacy
folders, write DBs, run ASR/R+A, or write CRM.
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

from mango_mvp.productization.recording_download_bridge import (  # noqa: E402
    build_recording_download_bridge_dry_run,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_DOWNLOAD_MANIFEST = f"{DEFAULT_PRODUCT_ROOT}/recording_capture_downloads/recording_download_manifest_stage7.jsonl"
DEFAULT_STAGE8_DIR = f"{DEFAULT_PRODUCT_ROOT}/recording_bridge_stage8"
DEFAULT_CAPTURE_MANIFEST = f"{DEFAULT_STAGE8_DIR}/capture_manifest_from_downloads_stage8.jsonl"
DEFAULT_BRIDGE_PLAN = f"{DEFAULT_STAGE8_DIR}/recording_bridge_plan_stage8.json"
DEFAULT_CSV = f"{DEFAULT_STAGE8_DIR}/recording_bridge_plan_stage8.csv"
DEFAULT_SOURCE_DIR = f"{DEFAULT_STAGE8_DIR}/empty_legacy_source_index"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_recording_download_bridge_dry_run(
            download_manifest_path=Path(args.download_manifest),
            product_root=Path(args.product_root),
            capture_manifest_path=Path(args.capture_manifest),
            bridge_plan_path=Path(args.out),
            source_dir=Path(args.source_dir),
            csv_path=Path(args.csv_out) if args.csv_out else None,
            db_paths=tuple(Path(path) for path in (args.db or ())),
            tolerance_sec=args.tolerance_sec,
            verify_checksum=not args.no_verify_checksum,
        )
    except Exception as exc:
        print(f"recording bridge dry-run failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"out": str(Path(args.out).resolve(strict=False)), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok", True) else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dry-run bridge plan from Stage 7 recording downloads.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--download-manifest", default=DEFAULT_DOWNLOAD_MANIFEST)
    parser.add_argument("--capture-manifest", default=DEFAULT_CAPTURE_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_BRIDGE_PLAN)
    parser.add_argument("--csv-out", default=DEFAULT_CSV)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--db", action="append", default=None, help="Optional SQLite DB to inspect read-only. Repeatable.")
    parser.add_argument("--tolerance-sec", type=int, default=120)
    parser.add_argument("--no-verify-checksum", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
