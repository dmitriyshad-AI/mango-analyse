#!/usr/bin/env python3
"""Build dry-run ASR worker command envelopes from an execution plan.

This command reads the Stage 18 execution plan and writes only JSON worker
dry-run artifacts. It does not dispatch a worker, run ASR/R+A, write
transcripts, write runtime DBs, or write CRM/Tallanto.
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

from mango_mvp.productization.asr_worker_execution_dry_run import build_asr_worker_execution_dry_run  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_EXECUTION_PLAN = f"{DEFAULT_PRODUCT_ROOT}/asr_execution_plan_stage18/asr_execution_plan_stage18.json"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_dry_run_stage19"
DEFAULT_WORKER_PLAN = f"{DEFAULT_OUT_DIR}/asr_worker_dry_run_plan_stage19.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_dry_run_stage19_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_dry_run_stage19_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_worker_execution_dry_run(
            product_root=Path(args.product_root),
            execution_plan_path=Path(args.execution_plan),
            out_dir=Path(args.out_dir),
            worker_plan_path=Path(args.worker_plan),
            out_path=Path(args.out),
        )
    except Exception as exc:
        print(f"ASR worker dry-run failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "worker_plan": compact_worker_plan(report["worker_plan"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_worker_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": plan.get("status"),
        "approval_ref": plan.get("approval_ref"),
        "dispatch_allowed": plan.get("dispatch_allowed"),
        "run_asr": plan.get("run_asr"),
        "execution_allowed": plan.get("execution_allowed"),
        "write_outputs": plan.get("write_outputs"),
        "workload": plan.get("workload"),
        "resource_estimate": plan.get("resource_estimate"),
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "writes_worker_plan": safety.get("writes_worker_plan"),
        "product_db_writes": safety.get("product_db_writes"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "dispatch_worker": safety.get("dispatch_worker"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_transcripts": safety.get("write_transcripts"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dry-run ASR worker command envelopes.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--execution-plan", default=DEFAULT_EXECUTION_PLAN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--worker-plan", default=DEFAULT_WORKER_PLAN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 19 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
