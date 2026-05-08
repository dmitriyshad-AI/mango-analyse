#!/usr/bin/env python3
"""Build a dry-run ASR execution plan from an approved scheduler plan.

This command expands the approved Stage 17 scheduler dry-run plan into a
per-item execution plan. It does not dispatch ASR, run R+A, write runtime DBs,
or write CRM/Tallanto.
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

from mango_mvp.productization.asr_execution_plan import build_asr_execution_plan  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_SCHEDULER_PLAN = f"{DEFAULT_PRODUCT_ROOT}/asr_scheduler_approved_dry_run_stage17/asr_scheduler_approved_dry_run_plan_stage17.json"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_execution_plan_stage18"
DEFAULT_EXECUTION_PLAN = f"{DEFAULT_OUT_DIR}/asr_execution_plan_stage18.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_execution_plan_stage18_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_execution_plan_stage18_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_execution_plan(
            product_root=Path(args.product_root),
            scheduler_plan_path=Path(args.scheduler_plan),
            out_dir=Path(args.out_dir),
            execution_plan_path=Path(args.execution_plan),
            out_path=Path(args.out),
            verify_checksum=not args.skip_checksum,
        )
    except Exception as exc:
        print(f"ASR execution plan failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "plan": compact_plan(report["execution_plan"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": plan.get("status"),
        "approval_ref": plan.get("approval_ref"),
        "run_asr": plan.get("run_asr"),
        "scheduler_dispatch": plan.get("scheduler_dispatch"),
        "execution_allowed": plan.get("execution_allowed"),
        "workload": plan.get("workload"),
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "writes_execution_plan": safety.get("writes_execution_plan"),
        "product_db_writes": safety.get("product_db_writes"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
        "scheduler_dispatch": safety.get("scheduler_dispatch"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dry-run ASR execution plan.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--scheduler-plan", default=DEFAULT_SCHEDULER_PLAN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--execution-plan", default=DEFAULT_EXECUTION_PLAN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 18 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
