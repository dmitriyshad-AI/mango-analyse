#!/usr/bin/env python3
"""Build/materialize a guarded product-appliance quarantine package.

This command writes only under the product appliance root. It does not write
runtime DBs, does not touch stable_runtime, does not run ASR/R+A, and does not
write CRM.
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

from mango_mvp.productization.recording_quarantine_package import (  # noqa: E402
    build_recording_quarantine_plan,
    materialize_recording_quarantine_package,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_BRIDGE_PLAN = f"{DEFAULT_PRODUCT_ROOT}/recording_bridge_stage8/recording_bridge_plan_stage8_legacy_source_check.json"
DEFAULT_PACKAGE_ROOT = f"{DEFAULT_PRODUCT_ROOT}/recording_quarantine_stage9"
DEFAULT_AUDIO_DIR = f"{DEFAULT_PACKAGE_ROOT}/audio"
DEFAULT_METADATA_CSV = f"{DEFAULT_PACKAGE_ROOT}/metadata.csv"
DEFAULT_PLAN = f"{DEFAULT_PACKAGE_ROOT}/recording_quarantine_plan_stage9.json"
DEFAULT_NORMALIZED_BRIDGE = f"{DEFAULT_PACKAGE_ROOT}/normalized_bridge_plan_stage9.json"
DEFAULT_MATERIALIZATION_AUDIT = f"{DEFAULT_PACKAGE_ROOT}/recording_quarantine_materialization_stage9_audit.json"
DEFAULT_IDEMPOTENCY_AUDIT = f"{DEFAULT_PACKAGE_ROOT}/recording_quarantine_materialization_stage9_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            report = build_recording_quarantine_plan(
                source_bridge_plan_path=Path(args.bridge_plan),
                product_root=Path(args.product_root),
                package_root=Path(args.package_root),
                quarantine_dir=Path(args.quarantine_dir),
                metadata_csv_path=Path(args.metadata_csv),
                plan_path=Path(args.out),
                normalized_bridge_plan_path=Path(args.normalized_bridge_plan),
                verify_checksum=not args.no_verify_checksum,
            )
            out = args.out
        elif args.command == "materialize":
            report = materialize_recording_quarantine_package(
                plan_path=Path(args.plan),
                product_root=Path(args.product_root),
                out_path=Path(args.out),
                mode=args.mode,
                verify_checksum=not args.no_verify_checksum,
                overwrite=args.overwrite,
            )
            out = args.out
        else:
            raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"recording quarantine package failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(Path(out).resolve(strict=False)), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok", True) else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/materialize a guarded recording quarantine package.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--bridge-plan", default=DEFAULT_BRIDGE_PLAN)
    plan.add_argument("--package-root", default=DEFAULT_PACKAGE_ROOT)
    plan.add_argument("--quarantine-dir", default=DEFAULT_AUDIO_DIR)
    plan.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    plan.add_argument("--normalized-bridge-plan", default=DEFAULT_NORMALIZED_BRIDGE)
    plan.add_argument("--out", default=DEFAULT_PLAN)
    plan.add_argument("--no-verify-checksum", action="store_true")

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--plan", default=DEFAULT_PLAN)
    materialize.add_argument("--out", default=DEFAULT_MATERIALIZATION_AUDIT)
    materialize.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    materialize.add_argument("--overwrite", action="store_true")
    materialize.add_argument("--no-verify-checksum", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
