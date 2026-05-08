#!/usr/bin/env python3
"""Build a dry-run ASR execution approval gate job plan.

This command reads the Stage 14 ASR worker pack verify audit and writes only a
JSON job plan/audit under product_appliance. It does not run ASR/R+A, write
runtime DBs, or write CRM.
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

from mango_mvp.productization.asr_execution_approval_gate import build_asr_execution_approval_gate  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_VERIFY_AUDIT = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_pack_stage13/asr_worker_pack_verify_stage14_audit.json"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_execution_approval_stage15"
DEFAULT_JOB_PLAN = f"{DEFAULT_OUT_DIR}/asr_execution_job_plan_stage15.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_execution_approval_stage15_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_execution_approval_stage15_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_execution_approval_gate(
            product_root=Path(args.product_root),
            verify_audit_path=Path(args.verify_audit),
            out_dir=Path(args.out_dir),
            job_plan_path=Path(args.job_plan),
            out_path=Path(args.out),
            approval_ref=args.approval_ref,
        )
    except Exception as exc:
        print(f"ASR execution approval gate failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "approval_gate": compact_gate(report["approval_gate"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_gate(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "execution_allowed": gate.get("execution_allowed"),
        "approval_required": gate.get("approval_required"),
        "approval_present": gate.get("approval_present"),
        "next_allowed_step": gate.get("next_allowed_step"),
    }


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
    parser = argparse.ArgumentParser(description="Build a dry-run ASR execution approval gate job plan.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--verify-audit", default=DEFAULT_VERIFY_AUDIT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--job-plan", default=DEFAULT_JOB_PLAN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--approval-ref", help="Optional approval reference to record in the dry-run plan; does not run ASR.")
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 15 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
