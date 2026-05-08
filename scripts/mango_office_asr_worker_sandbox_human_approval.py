#!/usr/bin/env python3
"""Report, write, or validate ASR sandbox human approval records.

The default safe operation is `requirements`: it reads the Stage 23 approval
packet and reports what a human must approve. `write` creates an approval
record only when the exact approval phrase and all acknowledgements are
provided. None of the operations dispatch workers, run ASR/R+A, create sandbox
directories, write transcripts, write runtime DBs, or write CRM/Tallanto.
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

from mango_mvp.productization.asr_worker_sandbox_approval_packet import REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS  # noqa: E402
from mango_mvp.productization.asr_worker_sandbox_human_approval_record import (  # noqa: E402
    build_asr_worker_sandbox_human_approval_requirements,
    validate_asr_worker_sandbox_human_approval_record,
    write_asr_worker_sandbox_human_approval_record,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_APPROVAL_PACKET = (
    f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_approval_stage23/asr_worker_sandbox_execution_approval_packet_stage23.json"
)
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_human_approval_stage24"
DEFAULT_APPROVAL_RECORD = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_human_approval_record_stage24.json"
DEFAULT_REQUIREMENTS_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_human_approval_requirements_stage24_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_human_approval_requirements_stage24_idempotency_audit.json"
DEFAULT_WRITE_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_human_approval_write_stage24_audit.json"
DEFAULT_VALIDATE_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_human_approval_validate_stage24_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "requirements":
            report = build_asr_worker_sandbox_human_approval_requirements(
                product_root=Path(args.product_root),
                approval_packet_path=Path(args.approval_packet),
                approval_record_path=Path(args.approval_record) if args.approval_record else None,
                out_path=Path(args.out),
            )
        elif args.command == "write":
            report = write_asr_worker_sandbox_human_approval_record(
                product_root=Path(args.product_root),
                approval_packet_path=Path(args.approval_packet),
                approval_record_path=Path(args.approval_record),
                out_path=Path(args.out),
                approved_by=args.approved_by,
                approval_phrase=args.approval_phrase,
                acknowledgements=acknowledgements_from_args(args),
                approved_at=args.approved_at,
                reason=args.reason,
                replace_existing=args.replace,
            )
        elif args.command == "validate":
            report = validate_asr_worker_sandbox_human_approval_record(
                product_root=Path(args.product_root),
                approval_packet_path=Path(args.approval_packet),
                approval_record_path=Path(args.approval_record),
                out_path=Path(args.out),
            )
        else:
            raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"ASR sandbox human approval {args.command} failed: {exc}", file=sys.stderr)
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


def acknowledgements_from_args(args: argparse.Namespace) -> Mapping[str, bool]:
    if getattr(args, "acknowledge_all", False):
        return {key: True for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS}
    selected = set(args.ack or [])
    return {key: key in selected for key in REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS}


def compact_approval(approval: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "approval_packet_present": approval.get("approval_packet_present"),
        "approval_packet_valid": approval.get("approval_packet_valid"),
        "approval_record_present": approval.get("approval_record_present"),
        "approval_record_valid": approval.get("approval_record_valid"),
        "approval_packet_ref": approval.get("approval_packet_ref"),
        "required_approval_phrase": approval.get("required_approval_phrase"),
        "required_acknowledgements": approval.get("required_acknowledgements"),
        "execution_approved": approval.get("execution_approved"),
        "dispatch_allowed": approval.get("dispatch_allowed"),
        "run_asr": approval.get("run_asr"),
        "missing_or_invalid_reasons": approval.get("missing_or_invalid_reasons") or [],
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "reads_approval_packet": safety.get("reads_approval_packet"),
        "writes_approval_record": safety.get("writes_approval_record"),
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
    parser = argparse.ArgumentParser(description="Report, write, or validate ASR sandbox human approval records.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--approval-packet", default=DEFAULT_APPROVAL_PACKET)
    sub = parser.add_subparsers(dest="command", required=True)

    requirements = sub.add_parser("requirements")
    requirements.add_argument("--approval-record", default=DEFAULT_APPROVAL_RECORD)
    requirements.add_argument("--out", default=DEFAULT_REQUIREMENTS_OUT)
    requirements.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 24 idempotency audit path",
    )

    write = sub.add_parser("write")
    write.add_argument("--approval-record", default=DEFAULT_APPROVAL_RECORD)
    write.add_argument("--out", default=DEFAULT_WRITE_OUT)
    write.add_argument("--approved-by", required=True)
    write.add_argument("--approved-at")
    write.add_argument("--approval-phrase", required=True)
    write.add_argument("--reason", default="stage24_human_sandbox_execution_approval")
    write.add_argument("--ack", action="append", choices=REQUIRED_SANDBOX_EXECUTION_ACKNOWLEDGEMENTS)
    write.add_argument("--acknowledge-all", action="store_true")
    write.add_argument("--replace", action="store_true")

    validate = sub.add_parser("validate")
    validate.add_argument("--approval-record", default=DEFAULT_APPROVAL_RECORD)
    validate.add_argument("--out", default=DEFAULT_VALIDATE_OUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
