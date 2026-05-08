#!/usr/bin/env python3
"""Build a dry-run ASR processing handoff manifest from isolated asset DB.

This command reads the Stage 11 asset DB read-only and writes only JSON/JSONL
under product_appliance. It does not run ASR/R+A, does not write runtime DBs,
and does not write CRM.
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

from mango_mvp.productization.processing_handoff import build_processing_handoff_dry_run  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_ASSET_DB = f"{DEFAULT_PRODUCT_ROOT}/recording_quarantine_stage11/recording_asset_ingest_stage11.sqlite"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/processing_handoff_stage12"
DEFAULT_MANIFEST = f"{DEFAULT_OUT_DIR}/asr_handoff_manifest_stage12.jsonl"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_handoff_stage12_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_handoff_stage12_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_processing_handoff_dry_run(
            asset_db_path=Path(args.asset_db),
            product_root=Path(args.product_root),
            out_dir=Path(args.out_dir),
            manifest_path=Path(args.manifest),
            out_path=Path(args.out),
            package_ref=args.package_ref,
            limit=args.limit,
            verify_checksum=not args.no_verify_checksum,
        )
    except Exception as exc:
        print(f"processing handoff failed: {exc}", file=sys.stderr)
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
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dry-run ASR processing handoff manifest.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--asset-db", default=DEFAULT_ASSET_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--package-ref", default="recording_quarantine_stage11")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 12 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
