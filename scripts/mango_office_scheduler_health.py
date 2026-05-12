#!/usr/bin/env python3
"""Build scheduler health/readiness report for product appliance."""

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

from mango_mvp.productization.scheduler_health import build_scheduler_health_report  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/scheduler_health/scheduler_health_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_scheduler_health_report(
            product_db_path=Path(args.product_db),
            product_root=Path(args.product_root),
            out_path=Path(args.out),
            stale_after_minutes=args.stale_after_minutes,
            recent_limit=args.recent_limit,
        )
    except Exception as exc:
        print(f"Scheduler health failed: {exc}", file=sys.stderr)
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
        "product_db_writes": safety.get("product_db_writes"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "run_asr": safety.get("run_asr"),
        "write_crm": safety.get("write_crm"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scheduler health/readiness report.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--stale-after-minutes", type=int, default=30)
    parser.add_argument("--recent-limit", type=int, default=50)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
