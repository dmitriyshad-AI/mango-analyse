#!/usr/bin/env python3
"""Durable product capture inbox for Mango shadow poll decisions.

This command only writes the isolated product appliance DB. It does not
download audio, run ASR/R+A, write CRM, or touch runtime DBs.
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

from mango_mvp.productization.capture_inbox import (  # noqa: E402
    apply_shadow_poll_report_to_capture_inbox,
    audit_capture_inbox,
)
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root).resolve(strict=False)
    product_db = Path(args.product_db).resolve(strict=False)
    guard_under_root(product_root, "product DB", product_db)

    if args.command == "apply-report":
        report_path = Path(args.report).resolve(strict=False)
        out = Path(args.out).resolve(strict=False)
        guard_under_root(product_root, "shadow poll report", report_path)
        guard_under_root(product_root, "audit output", out)
        report = apply_shadow_poll_report_to_capture_inbox(
            product_db_path=product_db,
            product_root=product_root,
            report_path=report_path,
            out_path=out,
        )
    elif args.command == "audit":
        out = Path(args.out).resolve(strict=False)
        guard_under_root(product_root, "audit output", out)
        report = audit_capture_inbox(
            product_db_path=product_db,
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
    parser = argparse.ArgumentParser(description="Apply/audit product capture inbox from shadow poll reports.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    apply_report = sub.add_parser("apply-report")
    apply_report.add_argument("--report", required=True)
    apply_report.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/capture_inbox_apply_audit.json")

    audit = sub.add_parser("audit")
    audit.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/capture_inbox_audit.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
