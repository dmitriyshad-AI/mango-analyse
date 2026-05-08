#!/usr/bin/env python3
"""Read-only local Product API HTTP layer utilities."""

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

from mango_mvp.productization.product_api_http import (  # noqa: E402
    build_product_api_http_readiness_report,
    run_product_api_http_server,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/product_api_http_20260508/product_api_http_readiness_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "serve":
        run_product_api_http_server(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            host=args.host,
            port=args.port,
            workspace_root=ROOT,
        )
        return 0
    try:
        report = build_product_api_http_readiness_report(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            out_path=Path(args.out),
            workspace_root=ROOT,
        )
    except Exception as exc:
        print(f"Product API HTTP readiness failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "routes": report["routes"],
                "blocked_mutation_check": report["blocked_mutation_check"]["status"],
                "safety": report["safety"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok") else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Product API HTTP layer utilities.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    readiness = sub.add_parser("readiness")
    readiness.add_argument("--out", default=DEFAULT_OUT)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
