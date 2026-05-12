#!/usr/bin/env python3
"""Generate client-hosted service templates without installing or starting them."""

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

from mango_mvp.productization.appliance_service_pack import build_appliance_service_pack  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/service_pack"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_appliance_service_pack(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            out_dir=Path(args.out_dir),
            host=args.host,
            port=args.port,
            python_bin=args.python_bin,
        )
    except Exception as exc:
        print(f"Appliance service pack failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out_dir": str(Path(args.out_dir).resolve(strict=False)),
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
        "templates_only": safety.get("templates_only"),
        "installs_services": safety.get("installs_services"),
        "starts_services": safety.get("starts_services"),
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate service templates for client-hosted appliance.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--python-bin", default="python3")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
