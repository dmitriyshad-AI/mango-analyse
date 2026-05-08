#!/usr/bin/env python3
"""Admin operations for the isolated Mango product appliance DB.

Operations are constrained to the product root. No command writes runtime DBs,
ASR/R+A artifacts, AMO, or Tallanto.
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

from mango_mvp.productization.product_db import (  # noqa: E402
    audit_product_db,
    audit_product_retention,
    backup_product_db,
    restore_product_db_from_backup,
    snapshot_tenant_config,
    upgrade_product_db,
)
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_TENANT_CONFIG = f"{DEFAULT_PRODUCT_ROOT}/config/tenant_owner_mapping_foton_mango.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root).resolve(strict=False)
    product_db = Path(args.product_db).resolve(strict=False)
    guard_under_root(product_root, "product DB", product_db)

    if args.command == "integrity":
        report = audit_product_db(product_db, product_root)
        out = Path(args.out).resolve(strict=False)
    elif args.command == "upgrade":
        out = Path(args.out).resolve(strict=False)
        report = upgrade_product_db(product_db, product_root, out_path=out)
    elif args.command == "backup":
        backup_path = Path(args.backup).resolve(strict=False)
        guard_under_root(product_root, "backup", backup_path)
        report = {"summary": backup_product_db(product_db, backup_path, product_root)}
        out = Path(args.out).resolve(strict=False)
    elif args.command == "restore":
        backup_path = Path(args.backup).resolve(strict=False)
        pre_restore = Path(args.pre_restore_backup).resolve(strict=False) if args.pre_restore_backup else None
        out = Path(args.out).resolve(strict=False)
        guard_under_root(product_root, "backup", backup_path)
        if pre_restore:
            guard_under_root(product_root, "pre-restore backup", pre_restore)
        report = restore_product_db_from_backup(
            backup_path=backup_path,
            product_db_path=product_db,
            out_allowed_root=product_root,
            replace_existing=args.replace,
            pre_restore_backup_path=pre_restore,
            out_path=out,
        )
    elif args.command == "retention-audit":
        out = Path(args.out).resolve(strict=False)
        report = audit_product_retention(product_db, product_root, out_path=out)
    elif args.command == "snapshot-config":
        config_path = Path(args.config).resolve(strict=False)
        out = Path(args.out).resolve(strict=False)
        guard_under_root(product_root, "tenant config", config_path)
        report = snapshot_tenant_config(
            product_db_path=product_db,
            config_path=config_path,
            out_allowed_root=product_root,
            snapshot_reason=args.reason,
            out_path=out,
        )
    else:
        raise ValueError(f"unknown command: {args.command}")

    guard_under_root(product_root, "audit output", out)
    if args.command in {"integrity", "backup"}:
        write_json(out, report)
    print(json.dumps({"out": str(out), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok", True) else 1


def guard_under_root(product_root: Path, label: str, path: Path) -> None:
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Admin operations for isolated Mango product DB.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    integrity = sub.add_parser("integrity")
    integrity.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/product_db_integrity_audit.json")

    upgrade = sub.add_parser("upgrade")
    upgrade.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/product_db_upgrade_audit.json")

    backup = sub.add_parser("backup")
    backup.add_argument("--backup", default=f"{DEFAULT_PRODUCT_ROOT}/backups/mango_product_appliance_manual.sqlite")
    backup.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/product_db_backup_audit.json")

    restore = sub.add_parser("restore")
    restore.add_argument("--backup", required=True)
    restore.add_argument("--pre-restore-backup")
    restore.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/product_db_restore_audit.json")
    restore.add_argument("--replace", action="store_true")

    retention = sub.add_parser("retention-audit")
    retention.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/product_db_retention_audit.json")

    snapshot = sub.add_parser("snapshot-config")
    snapshot.add_argument("--config", default=DEFAULT_TENANT_CONFIG)
    snapshot.add_argument("--reason", default="manual_admin_snapshot")
    snapshot.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/tenant_config_snapshot_audit.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
