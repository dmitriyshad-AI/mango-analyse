#!/usr/bin/env python3
"""Build a dry-run ASR scheduler approval view.

This command reads the Stage 15 ASR execution approval job plan and an optional
operator approval record. It writes only JSON dry-run artifacts under the
product appliance root. It does not dispatch ASR, write runtime DBs, or write
CRM/Tallanto.
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

from mango_mvp.productization.asr_scheduler_dry_run import build_asr_scheduler_dry_run  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_JOB_PLAN = f"{DEFAULT_PRODUCT_ROOT}/asr_execution_approval_stage15/asr_execution_job_plan_stage15.json"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_scheduler_dry_run_stage16"
DEFAULT_SCHEDULER_PLAN = f"{DEFAULT_OUT_DIR}/asr_scheduler_dry_run_plan_stage16.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_scheduler_dry_run_stage16_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_scheduler_dry_run_stage16_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_scheduler_dry_run(
            product_root=Path(args.product_root),
            job_plan_path=Path(args.job_plan),
            out_dir=Path(args.out_dir),
            scheduler_plan_path=Path(args.scheduler_plan),
            out_path=Path(args.out),
            approval_path=Path(args.approval) if args.approval else None,
        )
    except Exception as exc:
        print(f"ASR scheduler dry-run failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "scheduler": compact_scheduler(report["scheduler_plan"]),
                "approval": compact_approval(report["approval"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_scheduler(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": plan.get("status"),
        "scheduler_may_dispatch": plan.get("scheduler_may_dispatch"),
        "execution_allowed": plan.get("execution_allowed"),
        "approval_ref": plan.get("approval_ref"),
    }


def compact_approval(approval: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "approval_present": approval.get("approval_present"),
        "approval_valid": approval.get("approval_valid"),
        "approval_ref": approval.get("approval_ref"),
        "reasons": approval.get("reasons") or [],
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
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
    parser = argparse.ArgumentParser(description="Build a dry-run ASR scheduler approval view.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--job-plan", default=DEFAULT_JOB_PLAN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--scheduler-plan", default=DEFAULT_SCHEDULER_PLAN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--approval", help="Optional ASR approval record JSON; dry-run validation only.")
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 16 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
