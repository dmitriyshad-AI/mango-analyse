#!/usr/bin/env python3
"""Run product appliance configuration checks."""

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

from mango_mvp.productization.appliance_config_wizard import build_appliance_config_wizard_report  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/appliance_config_wizard/appliance_config_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_appliance_config_wizard_report(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            out_path=Path(args.out),
            crm_snapshot_path=Path(args.crm_snapshot) if args.crm_snapshot else None,
            backup_dir=Path(args.backup_dir) if args.backup_dir else None,
            require_mango_credentials=args.require_mango_credentials,
            write_templates=args.write_templates,
        )
    except Exception as exc:
        print(f"Appliance config wizard failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "next_actions": report["next_actions"],
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
        "live_crm_reads": safety.get("live_crm_reads"),
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run product appliance configuration checks.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--crm-snapshot")
    parser.add_argument("--backup-dir")
    parser.add_argument("--require-mango-credentials", action="store_true")
    parser.add_argument("--write-templates", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
