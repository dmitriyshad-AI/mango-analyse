#!/usr/bin/env python3
"""Build a controlled CRM/AMO writeback preview report.

This command only creates a preview diff and rollout policy report. It never
writes AMO, Tallanto, runtime DBs, or product DB rows.
"""

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

from mango_mvp.productization.crm_writeback_preview import build_crm_writeback_preview  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/crm_writeback_preview_stage6/crm_writeback_preview_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_crm_writeback_preview(
            product_db_path=Path(args.product_db),
            product_root=Path(args.product_root),
            out_path=Path(args.out),
            stage=args.stage,
            limit=args.limit,
            include_blocked=not args.only_ready,
            crm_snapshot_path=Path(args.crm_snapshot) if args.crm_snapshot else None,
        )
    except Exception as exc:
        print(f"CRM writeback preview failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "policy_gates": report["policy_gates"],
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
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a controlled CRM/AMO writeback preview report.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=["batch_10", "batch_50", "batch_300", "full"], default="batch_10")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--crm-snapshot")
    parser.add_argument("--only-ready", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
