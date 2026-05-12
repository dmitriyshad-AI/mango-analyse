#!/usr/bin/env python3
"""Build read-only acceptance gates before connecting processing."""

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

from mango_mvp.productization.processing_acceptance_gates import build_processing_acceptance_gates_report  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/processing_acceptance_gates/processing_acceptance_gates.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_processing_acceptance_gates_report(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            out_path=Path(args.out),
            processing_quality_ready=args.processing_quality_ready,
            processing_quality_report_path=Path(args.processing_quality_report) if args.processing_quality_report else None,
        )
    except Exception as exc:
        print(f"Processing acceptance gates failed: {exc}", file=sys.stderr)
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
        "read_only": safety.get("read_only"),
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processing acceptance gates.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--processing-quality-ready", action="store_true")
    parser.add_argument("--processing-quality-report")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
