#!/usr/bin/env python3
"""Build a read-only processing lifecycle report for product capture items.

The report explains which captured calls are ready for a dry-run ASR handoff,
which are waiting for recording assets, and which are blocked. It never runs
ASR/R+A, writes runtime DBs, or writes CRM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.processing_lifecycle import build_processing_lifecycle_report  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_ASSET_DB = f"{DEFAULT_PRODUCT_ROOT}/recording_quarantine_stage11/recording_asset_ingest_stage11.sqlite"
DEFAULT_HANDOFF_MANIFEST = f"{DEFAULT_PRODUCT_ROOT}/processing_handoff_stage12/asr_handoff_manifest_stage12.jsonl"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/processing_lifecycle_stage5/processing_lifecycle_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    asset_db = None if args.no_asset_db else Path(args.asset_db)
    handoff_manifest = None if args.no_handoff_manifest else Path(args.handoff_manifest)
    try:
        report = build_processing_lifecycle_report(
            product_db_path=Path(args.product_db),
            product_root=Path(args.product_root),
            asset_db_path=asset_db,
            handoff_manifest_path=handoff_manifest,
            out_path=Path(args.out),
            limit=args.limit,
        )
    except Exception as exc:
        print(f"processing lifecycle report failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", False) else 1


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "downloads_audio": safety.get("downloads_audio"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_crm": safety.get("write_crm"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only processing lifecycle report.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--asset-db", default=DEFAULT_ASSET_DB)
    parser.add_argument("--handoff-manifest", default=DEFAULT_HANDOFF_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-asset-db", action="store_true")
    parser.add_argument("--no-handoff-manifest", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
