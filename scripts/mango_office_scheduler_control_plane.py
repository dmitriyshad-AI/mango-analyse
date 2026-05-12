#!/usr/bin/env python3
"""Build scheduler/supervisor control-plane readiness report."""

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

from mango_mvp.productization.scheduler_control_plane import build_scheduler_control_plane_report  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/scheduler_control_plane/scheduler_control_plane_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_scheduler_control_plane_report(
            product_db_path=Path(args.product_db),
            product_root=Path(args.product_root),
            out_path=Path(args.out),
            worker_id=args.worker_id,
            tick_limit=args.tick_limit,
            stale_after_minutes=args.stale_after_minutes,
        )
    except Exception as exc:
        print(f"Scheduler control-plane failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "recommended_actions": report["recommended_actions"],
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
        "executes_jobs": safety.get("executes_jobs"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "run_asr": safety.get("run_asr"),
        "write_crm": safety.get("write_crm"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scheduler control-plane readiness report.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--worker-id", default="appliance-supervisor")
    parser.add_argument("--tick-limit", type=int, default=1)
    parser.add_argument("--stale-after-minutes", type=int, default=30)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
