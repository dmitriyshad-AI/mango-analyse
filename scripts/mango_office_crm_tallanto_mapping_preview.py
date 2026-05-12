#!/usr/bin/env python3
"""Build read-only AMO/Tallanto mapping preview from local snapshots."""

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

from mango_mvp.productization.crm_tallanto_mapping_preview import build_crm_tallanto_mapping_preview  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/crm_mapping_preview/crm_tallanto_mapping_preview.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_crm_tallanto_mapping_preview(
            product_db_path=Path(args.product_db),
            product_root=Path(args.product_root),
            out_path=Path(args.out),
            amo_snapshot_path=Path(args.amo_snapshot) if args.amo_snapshot else None,
            tallanto_snapshot_path=Path(args.tallanto_snapshot) if args.tallanto_snapshot else None,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"CRM/Tallanto mapping preview failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "snapshot_paths": report["snapshot_paths"],
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
        "live_crm_reads": safety.get("live_crm_reads"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "run_asr": safety.get("run_asr"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only AMO/Tallanto mapping preview.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--amo-snapshot")
    parser.add_argument("--tallanto-snapshot")
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
