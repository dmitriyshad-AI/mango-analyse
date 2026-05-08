#!/usr/bin/env python3
"""Run a final dry-run preflight for an ASR worker sandbox contract.

This command reads the Stage 21 sandbox execution contract and verifies audio
checksums, output collisions, disk space, directory feasibility, and selected
engine capability without dispatching workers, running ASR/R+A, creating
sandbox output/tmp directories, writing transcripts, writing runtime DBs, or
writing CRM/Tallanto.
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

from mango_mvp.productization.asr_worker_sandbox_preflight import build_asr_worker_sandbox_preflight  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_CONTRACT = (
    f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_contract_stage21/asr_worker_sandbox_execution_contract_stage21.json"
)
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_preflight_stage22"
DEFAULT_PREFLIGHT_REPORT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_preflight_report_stage22.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_preflight_stage22_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_preflight_stage22_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_worker_sandbox_preflight(
            product_root=Path(args.product_root),
            contract_path=Path(args.contract),
            out_dir=Path(args.out_dir),
            preflight_report_path=Path(args.preflight_report),
            out_path=Path(args.out),
            disk_safety_margin_bytes=args.disk_safety_margin_bytes,
        )
    except Exception as exc:
        print(f"ASR worker sandbox preflight failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "preflight": compact_preflight(report["preflight_report"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_preflight(preflight: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": preflight.get("status"),
        "selected_engine": preflight.get("selected_engine"),
        "preflight_ready": preflight.get("preflight_ready"),
        "dispatch_allowed": preflight.get("dispatch_allowed"),
        "run_asr": preflight.get("run_asr"),
        "write_transcripts": preflight.get("write_transcripts"),
        "workload": preflight.get("workload"),
        "engine_preflight": preflight.get("engine_preflight"),
        "disk_preflight": preflight.get("disk_preflight"),
        "directory_preflight": preflight.get("directory_preflight"),
        "next_stage_contract": preflight.get("next_stage_contract"),
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "reads_audio_for_sha256": safety.get("reads_audio_for_sha256"),
        "writes_preflight_report": safety.get("writes_preflight_report"),
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
    parser = argparse.ArgumentParser(description="Run a final dry-run ASR sandbox preflight.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--preflight-report", default=DEFAULT_PREFLIGHT_REPORT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--disk-safety-margin-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 22 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
