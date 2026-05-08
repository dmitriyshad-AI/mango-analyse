#!/usr/bin/env python3
"""Write or validate a dry-run ASR approval record.

This command creates/validates only JSON approval artifacts under the isolated
product appliance root. It does not run ASR/R+A, dispatch scheduler jobs, write
runtime DBs, or write CRM/Tallanto.
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

from mango_mvp.productization.asr_approval_record import (  # noqa: E402
    validate_asr_approval_record,
    write_asr_approval_record,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_JOB_PLAN = f"{DEFAULT_PRODUCT_ROOT}/asr_execution_approval_stage15/asr_execution_job_plan_stage15.json"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_approval_record_stage17"
DEFAULT_APPROVAL = f"{DEFAULT_OUT_DIR}/asr_execution_approval_record_stage17.json"
DEFAULT_WRITE_OUT = f"{DEFAULT_OUT_DIR}/asr_approval_record_stage17_audit.json"
DEFAULT_VALIDATE_OUT = f"{DEFAULT_OUT_DIR}/asr_approval_record_stage17_validation_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "write":
            report = write_asr_approval_record(
                product_root=Path(args.product_root),
                job_plan_path=Path(args.job_plan),
                approval_path=Path(args.approval),
                out_path=Path(args.out),
                approval_ref=args.approval_ref,
                approved_by=args.approved_by,
                approved_at=args.approved_at,
                reason=args.reason,
                replace_existing=args.replace,
            )
        elif args.command == "validate":
            report = validate_asr_approval_record(
                product_root=Path(args.product_root),
                job_plan_path=Path(args.job_plan),
                approval_path=Path(args.approval),
                out_path=Path(args.out),
            )
        else:
            raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"ASR approval record {args.command} failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "approval": compact_approval(report["approval"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_approval(approval: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "approval_present": approval.get("approval_present"),
        "approval_valid": approval.get("approval_valid"),
        "approval_ref": approval.get("approval_ref"),
        "reasons": approval.get("reasons") or [],
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "writes_approval_record": safety.get("writes_approval_record"),
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
    parser = argparse.ArgumentParser(description="Write or validate dry-run ASR approval records.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--job-plan", default=DEFAULT_JOB_PLAN)
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write")
    write.add_argument("--approval", default=DEFAULT_APPROVAL)
    write.add_argument("--out", default=DEFAULT_WRITE_OUT)
    write.add_argument("--approval-ref", required=True)
    write.add_argument("--approved-by", required=True)
    write.add_argument("--approved-at")
    write.add_argument("--reason", default="stage17_scheduler_dry_run_approval")
    write.add_argument("--replace", action="store_true")

    validate = sub.add_parser("validate")
    validate.add_argument("--approval", default=DEFAULT_APPROVAL)
    validate.add_argument("--out", default=DEFAULT_VALIDATE_OUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
