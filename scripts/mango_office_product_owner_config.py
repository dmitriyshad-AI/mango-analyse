#!/usr/bin/env python3
"""Validate or apply tenant owner config to the product appliance DB.

Default mode is dry-run. Applying writes only to the isolated product DB and
does not write CRM, ASR/R+A, or runtime DBs.
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
    apply_tenant_owner_config_to_product_db,
    apply_tenant_owner_config_to_product_db_dry_run,
)
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_TENANT_CONFIG = f"{DEFAULT_PRODUCT_ROOT}/config/tenant_owner_mapping_foton_mango.json"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/tenant_owner_config_dry_run_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root).resolve(strict=False)
    product_db = Path(args.product_db).resolve(strict=False)
    config_path = Path(args.config).resolve(strict=False)
    out_path = Path(args.out).resolve(strict=False)
    guard_script_paths(product_root, product_db, config_path, out_path)

    fn = apply_tenant_owner_config_to_product_db if args.apply else apply_tenant_owner_config_to_product_db_dry_run
    report = fn(
        product_db_path=product_db,
        config_path=config_path,
        out_allowed_root=product_root,
        out_path=out_path,
    )
    print(
        json.dumps(
            {
                "out": str(out_path),
                "summary": report["summary"],
                "blocked_actions": [action for action in report["actions"] if str(action["action"]).startswith("BLOCK_")],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"]["validation_ok"] else 1


def guard_script_paths(product_root: Path, product_db: Path, config_path: Path, out_path: Path) -> None:
    for label, path in (
        ("product DB", product_db),
        ("tenant owner config", config_path),
        ("audit output", out_path),
    ):
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or apply tenant owner config to product DB.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--config", default=DEFAULT_TENANT_CONFIG)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--apply", action="store_true", help="Apply config to the isolated product DB.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
