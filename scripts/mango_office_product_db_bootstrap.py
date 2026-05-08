#!/usr/bin/env python3
"""Bootstrap the isolated SaaS product appliance SQLite DB.

This command reads the disposable quarantine repository and writes only inside
the product appliance root. It does not write runtime DBs, run ASR/R+A, or write
CRM systems.
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
    backup_product_db,
    bootstrap_product_db_from_repository,
)
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402


DEFAULT_SOURCE_ROOT = "_local_archive_mango_api_downloads_20260507/quarantine_import"
DEFAULT_SOURCE_DB = f"{DEFAULT_SOURCE_ROOT}/test_ingest/quarantine_test_ingest.sqlite"
DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_TENANT_CONFIG = f"{DEFAULT_PRODUCT_ROOT}/config/tenant_owner_mapping_foton_mango.json"
DEFAULT_AUDIT_OUT = f"{DEFAULT_PRODUCT_ROOT}/product_db_bootstrap_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root).resolve(strict=False)
    product_db = Path(args.product_db).resolve(strict=False)
    tenant_config = Path(args.tenant_config).resolve(strict=False)
    audit_out = Path(args.out).resolve(strict=False)
    guard_script_paths(product_root=product_root, product_db=product_db, tenant_config=tenant_config, audit_out=audit_out)

    report = bootstrap_product_db_from_repository(
        source_db_path=Path(args.source_db),
        source_allowed_root=Path(args.source_root),
        product_db_path=product_db,
        product_root=product_root,
        tenant_owner_config_path=tenant_config,
        replace_existing=args.replace,
        audit_out=audit_out,
    )
    backup_report = None
    if args.backup:
        backup_report = backup_product_db(
            db_path=product_db,
            backup_path=Path(args.backup).resolve(strict=False),
            out_allowed_root=product_root,
        )
        report = dict(report)
        report["backup"] = backup_report
        audit_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Re-open the DB after all writes, so the printed summary reflects the final artifact.
    integrity = audit_product_db(product_db, product_root)
    compact = {
        "out": str(audit_out),
        "summary": report["summary"],
        "integrity": integrity["summary"],
        "backup": backup_report,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"]["validation_ok"] and integrity["summary"]["validation_ok"] else 1


def guard_script_paths(product_root: Path, product_db: Path, tenant_config: Path, audit_out: Path) -> None:
    for label, path in (
        ("product DB", product_db),
        ("tenant config", tenant_config),
        ("audit output", audit_out),
    ):
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap isolated Mango product appliance DB.")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--tenant-config", default=DEFAULT_TENANT_CONFIG)
    parser.add_argument("--out", default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--backup")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
