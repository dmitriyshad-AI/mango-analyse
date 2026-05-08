#!/usr/bin/env python3
"""Run isolated recording asset ingest for a Mango quarantine package.

This script writes only to the requested productization SQLite DB and JSON
audit under product_appliance. It does not write runtime DB, does not touch
stable_runtime, does not run ASR/R+A, and does not write CRM.
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

from mango_mvp.productization.recording_asset_ingest import run_recording_asset_ingest  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PACKAGE_ROOT = f"{DEFAULT_PRODUCT_ROOT}/recording_quarantine_stage9"
DEFAULT_AUDIO_DIR = f"{DEFAULT_PACKAGE_ROOT}/audio"
DEFAULT_METADATA_CSV = f"{DEFAULT_PACKAGE_ROOT}/metadata.csv"
DEFAULT_STAGE10_ROOT = f"{DEFAULT_PRODUCT_ROOT}/recording_quarantine_stage10"
DEFAULT_DB = f"{DEFAULT_STAGE10_ROOT}/recording_asset_ingest_stage10.sqlite"
DEFAULT_OUT = f"{DEFAULT_STAGE10_ROOT}/recording_asset_ingest_stage10_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_STAGE10_ROOT}/recording_asset_ingest_stage10_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_recording_asset_ingest(
        package_root=Path(args.package_root),
        audio_dir=Path(args.audio_dir),
        metadata_csv_path=Path(args.metadata_csv),
        db_path=Path(args.db),
        product_root=Path(args.product_root),
        out_path=Path(args.out),
        package_ref=args.package_ref,
        replace_existing_db=args.replace_db,
        allow_existing_db=args.allow_existing_db,
        dry_run=args.dry_run,
        verify_checksum=not args.no_verify_checksum,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "out": str(Path(args.out)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "db_audit": compact_db_audit(report["db_audit"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok") else 1


def compact_db_audit(audit: Mapping[str, Any]) -> dict:
    return {
        "schema_migrations": audit.get("schema_migrations"),
        "import_packages": audit.get("import_packages"),
        "ingest_runs": audit.get("ingest_runs"),
        "assets_total": audit.get("assets_total"),
        "assets_for_package": audit.get("assets_for_package"),
        "status_counts": audit.get("status_counts"),
        "manager_counts": audit.get("manager_counts"),
        "blocked": audit.get("blocked"),
        "blocked_reasons": audit.get("blocked_reasons"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated Mango recording asset ingest.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--package-root", default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--package-ref", default="recording_quarantine_stage9")
    parser.add_argument("--replace-db", action="store_true")
    parser.add_argument("--allow-existing-db", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 10 idempotency audit path",
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    raise SystemExit(main())
