#!/usr/bin/env python3
"""Healthcheck, backup, verify, and restore dry-run for product appliance."""

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

from mango_mvp.productization.product_ops import (  # noqa: E402
    build_product_ops_diagnostics_bundle,
    build_product_ops_healthcheck,
    build_restore_dry_run,
    run_product_db_backup,
    verify_product_db_backup,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root)
    product_db = Path(args.product_db)
    try:
        if args.command == "healthcheck":
            report = build_product_ops_healthcheck(
                product_root=product_root,
                product_db_path=product_db,
                out_path=Path(args.out),
                backup_dir=Path(args.backup_dir) if args.backup_dir else None,
            )
        elif args.command == "backup":
            report = run_product_db_backup(
                product_root=product_root,
                product_db_path=product_db,
                backup_path=Path(args.backup),
                out_path=Path(args.out),
            )
        elif args.command == "verify-backup":
            report = verify_product_db_backup(
                product_root=product_root,
                product_db_path=product_db,
                backup_path=Path(args.backup),
                out_path=Path(args.out),
            )
        elif args.command == "restore-dry-run":
            report = build_restore_dry_run(
                product_root=product_root,
                product_db_path=product_db,
                backup_path=Path(args.backup),
                out_path=Path(args.out),
            )
        elif args.command == "diagnostics":
            report = build_product_ops_diagnostics_bundle(
                product_root=product_root,
                product_db_path=product_db,
                out_dir=Path(args.out_dir),
                backup_dir=Path(args.backup_dir) if args.backup_dir else None,
            )
        else:
            raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"Product ops failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(getattr(args, "out", getattr(args, "out_dir", ""))).resolve(strict=False)),
                "summary": report["summary"],
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
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
        "restore_executed": safety.get("restore_executed"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product appliance ops.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("healthcheck")
    health.add_argument("--backup-dir")
    health.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/ops/healthcheck.json")

    backup = sub.add_parser("backup")
    backup.add_argument("--backup", default=f"{DEFAULT_PRODUCT_ROOT}/backups/mango_product_appliance_backup.sqlite")
    backup.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/ops/backup.json")

    verify = sub.add_parser("verify-backup")
    verify.add_argument("--backup", required=True)
    verify.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/ops/verify_backup.json")

    restore = sub.add_parser("restore-dry-run")
    restore.add_argument("--backup", required=True)
    restore.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/ops/restore_dry_run.json")

    diagnostics = sub.add_parser("diagnostics")
    diagnostics.add_argument("--backup-dir")
    diagnostics.add_argument("--out-dir", default=f"{DEFAULT_PRODUCT_ROOT}/ops/diagnostics_bundle")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
