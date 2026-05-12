#!/usr/bin/env python3
"""Build a sanitized real-data demo product root from an existing product DB."""

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

from mango_mvp.productization.sanitized_real_demo import build_sanitized_real_demo_root  # noqa: E402


DEFAULT_SOURCE_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_SOURCE_PRODUCT_DB = f"{DEFAULT_SOURCE_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_DEMO_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/sanitized_real_demo_appliance"
DEFAULT_OUT = f"{DEFAULT_DEMO_PRODUCT_ROOT}/sanitized_real_demo_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_sanitized_real_demo_root(
            source_product_root=Path(args.source_product_root),
            source_product_db_path=Path(args.source_product_db),
            demo_product_root=Path(args.demo_product_root),
            out_path=Path(args.out),
            replace_existing=args.replace,
            salt=args.salt,
            row_limit=args.row_limit,
        )
    except Exception as exc:
        print(f"Sanitized real demo failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "sanitizer": report["sanitizer"],
                "demo_commands": report["demo_commands"],
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
        "reads_source_product_db": safety.get("reads_source_product_db"),
        "reads_runtime_db": safety.get("reads_runtime_db"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "contains_real_personal_data": safety.get("contains_real_personal_data"),
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sanitized real-data demo appliance root.")
    parser.add_argument("--source-product-root", default=DEFAULT_SOURCE_PRODUCT_ROOT)
    parser.add_argument("--source-product-db", default=DEFAULT_SOURCE_PRODUCT_DB)
    parser.add_argument("--demo-product-root", default=DEFAULT_DEMO_PRODUCT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--salt", default="mango-sanitized-real-demo-v1")
    parser.add_argument("--row-limit", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
