#!/usr/bin/env python3
"""Build/audit recording capture dry-run plans from the product capture inbox.

This command does not download audio, run ASR/R+A, write CRM, or touch runtime
DBs. It only reads the product appliance DB and writes JSON/JSONL reports under
the product appliance root.
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

from mango_mvp.productization.recording_capture_plan import (  # noqa: E402
    audit_recording_capture_plan,
    build_recording_capture_plan,
)
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_PLAN_DIR = f"{DEFAULT_PRODUCT_ROOT}/recording_capture_dry_run"
DEFAULT_RECORDINGS_DIR = f"{DEFAULT_PLAN_DIR}/recordings"
DEFAULT_MANIFEST = f"{DEFAULT_PLAN_DIR}/recording_capture_plan.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root).resolve(strict=False)
    product_db = Path(args.product_db).resolve(strict=False)
    guard_under_root(product_root, "product DB", product_db)

    if args.command == "build":
        recordings_dir = Path(args.recordings_dir).resolve(strict=False)
        manifest = Path(args.manifest).resolve(strict=False)
        out = Path(args.out).resolve(strict=False)
        guard_under_root(product_root, "recordings dir", recordings_dir)
        guard_under_root(product_root, "manifest", manifest)
        guard_under_root(product_root, "audit output", out)
        report = build_recording_capture_plan(
            product_db_path=product_db,
            product_root=product_root,
            recordings_dir=recordings_dir,
            manifest_path=manifest,
            out_path=out,
            limit=args.limit,
            manager_ref=args.manager_ref,
        )
    elif args.command == "audit":
        manifest = Path(args.manifest).resolve(strict=False)
        out = Path(args.out).resolve(strict=False)
        guard_under_root(product_root, "manifest", manifest)
        guard_under_root(product_root, "audit output", out)
        report = audit_recording_capture_plan(
            manifest_path=manifest,
            product_root=product_root,
            out_path=out,
        )
    else:
        raise ValueError(f"unknown command: {args.command}")

    print(json.dumps({"out": str(out), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok", True) else 1


def guard_under_root(product_root: Path, label: str, path: Path) -> None:
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/audit recording capture dry-run plans.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    build.add_argument("--manifest", default=DEFAULT_MANIFEST)
    build.add_argument("--out", default=f"{DEFAULT_PLAN_DIR}/recording_capture_plan_audit.json")
    build.add_argument("--limit", type=int)
    build.add_argument("--manager-ref")

    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", default=DEFAULT_MANIFEST)
    audit.add_argument("--out", default=f"{DEFAULT_PLAN_DIR}/recording_capture_plan_verify_audit.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
