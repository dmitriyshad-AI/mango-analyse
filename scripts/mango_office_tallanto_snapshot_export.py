#!/usr/bin/env python3
"""Export read-only Tallanto snapshot under product root."""

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

from mango_mvp.productization.tallanto_snapshot_exporter import export_tallanto_snapshot  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/crm_snapshots/tallanto_entities.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = export_tallanto_snapshot(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            output_path=Path(args.out),
            env_path=Path(args.env) if args.env else None,
            phone_limit=args.phone_limit,
            max_contacts_per_phone=args.max_contacts_per_phone,
        )
    except Exception as exc:
        print(f"Tallanto snapshot export failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
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
        "network_read_only": safety.get("network_read_only"),
        "write_tallanto": safety.get("write_tallanto"),
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export read-only Tallanto snapshot under product root.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--env")
    parser.add_argument("--phone-limit", type=int, default=250)
    parser.add_argument("--max-contacts-per-phone", type=int, default=5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
