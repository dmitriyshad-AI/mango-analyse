#!/usr/bin/env python3
"""Build a pending human approval packet for ASR sandbox execution.

This command reads the Stage 22 final preflight report and Stage 21 contract,
then writes a deterministic approval packet for a human operator. It does not
grant approval, dispatch workers, run ASR/R+A, create sandbox directories,
write transcripts, write runtime DBs, or write CRM/Tallanto.
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

from mango_mvp.productization.asr_worker_sandbox_approval_packet import build_asr_worker_sandbox_approval_packet  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PREFLIGHT_REPORT = (
    f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_preflight_stage22/asr_worker_sandbox_preflight_report_stage22.json"
)
DEFAULT_CONTRACT = (
    f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_contract_stage21/asr_worker_sandbox_execution_contract_stage21.json"
)
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_approval_stage23"
DEFAULT_APPROVAL_PACKET = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_execution_approval_packet_stage23.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_approval_stage23_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_approval_stage23_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    contract = Path(args.contract) if args.contract else None
    try:
        report = build_asr_worker_sandbox_approval_packet(
            product_root=Path(args.product_root),
            preflight_report_path=Path(args.preflight_report),
            contract_path=contract,
            out_dir=Path(args.out_dir),
            approval_packet_path=Path(args.approval_packet),
            out_path=Path(args.out),
        )
    except Exception as exc:
        print(f"ASR worker sandbox approval packet failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "approval_packet": compact_packet(report["approval_packet"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_packet(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": packet.get("status"),
        "approval_packet_ref": packet.get("approval_packet_ref"),
        "approval_status": packet.get("approval_status"),
        "approval_required": packet.get("approval_required"),
        "execution_approved": packet.get("execution_approved"),
        "selected_engine": packet.get("selected_engine"),
        "dispatch_allowed": packet.get("dispatch_allowed"),
        "run_asr": packet.get("run_asr"),
        "write_transcripts": packet.get("write_transcripts"),
        "workload": packet.get("workload"),
        "preflight_summary": packet.get("preflight_summary"),
        "required_approval_phrase": packet.get("required_approval_phrase"),
        "required_acknowledgements": packet.get("required_acknowledgements"),
        "next_stage_contract": packet.get("next_stage_contract"),
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "reads_preflight_report": safety.get("reads_preflight_report"),
        "reads_contract": safety.get("reads_contract"),
        "writes_approval_packet": safety.get("writes_approval_packet"),
        "reads_audio": safety.get("reads_audio"),
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
    parser = argparse.ArgumentParser(description="Build a pending ASR sandbox execution approval packet.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--preflight-report", default=DEFAULT_PREFLIGHT_REPORT)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--approval-packet", default=DEFAULT_APPROVAL_PACKET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 23 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
