#!/usr/bin/env python3
"""Build an ASR worker sandbox execution contract.

This command reads the Stage 20 readiness report and Stage 19 worker plan,
selects a ready ASR engine for a future sandbox execution, and writes only JSON
contract/audit artifacts. It does not dispatch workers, run ASR/R+A, create
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

from mango_mvp.productization.asr_worker_sandbox_execution_contract import build_asr_worker_sandbox_execution_contract  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_READINESS_REPORT = (
    f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_readiness_stage20/asr_worker_sandbox_readiness_report_stage20.json"
)
DEFAULT_WORKER_PLAN = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_dry_run_stage19/asr_worker_dry_run_plan_stage19.json"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_contract_stage21"
DEFAULT_CONTRACT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_execution_contract_stage21.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_contract_stage21_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_contract_stage21_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    worker_plan = Path(args.worker_plan) if args.worker_plan else None
    try:
        report = build_asr_worker_sandbox_execution_contract(
            product_root=Path(args.product_root),
            readiness_report_path=Path(args.readiness_report),
            worker_plan_path=worker_plan,
            out_dir=Path(args.out_dir),
            contract_path=Path(args.contract),
            out_path=Path(args.out),
            preferred_engine=args.engine,
            sandbox_output_root=Path(args.sandbox_output_root) if args.sandbox_output_root else None,
            sandbox_tmp_root=Path(args.sandbox_tmp_root) if args.sandbox_tmp_root else None,
        )
    except Exception as exc:
        print(f"ASR worker sandbox contract failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "contract": compact_contract(report["contract"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": contract.get("status"),
        "approval_ref": contract.get("approval_ref"),
        "selected_engine": contract.get("selected_engine"),
        "dispatch_allowed": contract.get("dispatch_allowed"),
        "run_asr": contract.get("run_asr"),
        "execution_allowed": contract.get("execution_allowed"),
        "write_outputs": contract.get("write_outputs"),
        "write_transcripts": contract.get("write_transcripts"),
        "workload": contract.get("workload"),
        "batch_resource_limits": contract.get("batch_resource_limits"),
        "sandbox_roots": contract.get("sandbox_roots"),
        "next_stage_contract": contract.get("next_stage_contract"),
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "writes_execution_contract": safety.get("writes_execution_contract"),
        "creates_sandbox_output_dirs": safety.get("creates_sandbox_output_dirs"),
        "creates_sandbox_tmp_dirs": safety.get("creates_sandbox_tmp_dirs"),
        "imports_asr_modules": safety.get("imports_asr_modules"),
        "loads_models": safety.get("loads_models"),
        "dispatch_worker": safety.get("dispatch_worker"),
        "run_asr": safety.get("run_asr"),
        "write_transcripts": safety.get("write_transcripts"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ASR worker sandbox execution contract.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--readiness-report", default=DEFAULT_READINESS_REPORT)
    parser.add_argument("--worker-plan", default=DEFAULT_WORKER_PLAN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--engine", default="auto", choices=("auto", "mlx", "gigaam", "openai"))
    parser.add_argument("--sandbox-output-root", default=None)
    parser.add_argument("--sandbox-tmp-root", default=None)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 21 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
