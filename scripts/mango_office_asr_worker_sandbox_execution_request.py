#!/usr/bin/env python3
"""Build a dry-run ASR sandbox execution request.

This script is intentionally request-only. It validates the Stage 23 approval
packet, the Stage 21 sandbox contract, and the optional Stage 24 human approval
record, then writes a Stage 25 JSON request/audit. It never dispatches a worker,
runs ASR/R+A, creates sandbox output directories, writes transcripts, writes
runtime DBs, or writes CRM/Tallanto.
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

from mango_mvp.productization.asr_worker_sandbox_execution_request import (  # noqa: E402
    build_asr_worker_sandbox_execution_request,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_APPROVAL_PACKET = (
    f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_approval_stage23/asr_worker_sandbox_execution_approval_packet_stage23.json"
)
DEFAULT_APPROVAL_RECORD = (
    f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_human_approval_stage24/asr_worker_sandbox_human_approval_record_stage24.json"
)
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_execution_request_stage25"
DEFAULT_REQUEST = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_execution_request_stage25.json"
DEFAULT_AUDIT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_execution_request_stage25_audit.json"
DEFAULT_IDEMPOTENCY_AUDIT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_execution_request_stage25_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_worker_sandbox_execution_request(
            product_root=Path(args.product_root),
            approval_packet_path=Path(args.approval_packet),
            approval_record_path=Path(args.approval_record) if args.approval_record else None,
            contract_path=Path(args.contract) if args.contract else None,
            out_dir=Path(args.out_dir),
            request_path=Path(args.request),
            out_path=Path(args.out),
        )
    except Exception as exc:
        print(f"ASR sandbox execution request dry-run failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "request": str(Path(args.request).resolve(strict=False)),
                "summary": report["summary"],
                "approval": compact_approval(report["approval"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok") else 1


def compact_approval(approval: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "approval_packet_valid": approval.get("approval_packet_valid"),
        "approval_record_present": approval.get("approval_record_present"),
        "approval_record_valid": approval.get("approval_record_valid"),
        "approval_packet_ref": approval.get("approval_packet_ref"),
        "execution_approved_by_human_record": approval.get("execution_approved_by_human_record"),
        "execution_request_ready": approval.get("execution_request_ready"),
        "dispatch_allowed": approval.get("dispatch_allowed"),
        "run_asr": approval.get("run_asr"),
        "missing_or_invalid_reasons": approval.get("missing_or_invalid_reasons") or [],
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "reads_approval_packet": safety.get("reads_approval_packet"),
        "reads_human_approval_record_if_present": safety.get("reads_human_approval_record_if_present"),
        "reads_contract": safety.get("reads_contract"),
        "writes_execution_request": safety.get("writes_execution_request"),
        "writes_audit_json": safety.get("writes_audit_json"),
        "reads_audio": safety.get("reads_audio"),
        "creates_sandbox_output_dirs": safety.get("creates_sandbox_output_dirs"),
        "creates_sandbox_tmp_dirs": safety.get("creates_sandbox_tmp_dirs"),
        "dispatch_worker": safety.get("dispatch_worker"),
        "run_asr": safety.get("run_asr"),
        "write_transcripts": safety.get("write_transcripts"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a request-only dry-run ASR sandbox execution request.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--approval-packet", default=DEFAULT_APPROVAL_PACKET)
    parser.add_argument("--approval-record", default=DEFAULT_APPROVAL_RECORD)
    parser.add_argument("--contract", help="optional explicit Stage 21 sandbox execution contract path")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--request", default=DEFAULT_REQUEST)
    parser.add_argument("--out", default=DEFAULT_AUDIT)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_AUDIT,
        dest="out",
        help="write to the default Stage 25 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
