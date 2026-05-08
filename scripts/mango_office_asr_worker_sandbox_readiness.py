#!/usr/bin/env python3
"""Build an ASR worker sandbox readiness report.

This command reads the Stage 19 worker dry-run plan and checks ASR capability
indicators without importing ASR modules, loading models, dispatching workers,
running ASR/R+A, writing transcripts, or writing runtime DBs.
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

from mango_mvp.productization.asr_worker_sandbox_readiness import build_asr_worker_sandbox_readiness  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_WORKER_PLAN = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_dry_run_stage19/asr_worker_dry_run_plan_stage19.json"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_sandbox_readiness_stage20"
DEFAULT_READINESS_REPORT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_readiness_report_stage20.json"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_readiness_stage20_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_OUT_DIR}/asr_worker_sandbox_readiness_stage20_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_worker_sandbox_readiness(
            product_root=Path(args.product_root),
            worker_plan_path=Path(args.worker_plan),
            out_dir=Path(args.out_dir),
            readiness_report_path=Path(args.readiness_report),
            out_path=Path(args.out),
        )
    except Exception as exc:
        print(f"ASR worker sandbox readiness failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "readiness": compact_readiness(report["readiness_report"]),
                "capabilities": compact_capabilities(report["capability_report"]),
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_readiness(readiness: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": readiness.get("status"),
        "worker_sandbox_ready": readiness.get("worker_sandbox_ready"),
        "asr_engine_ready": readiness.get("asr_engine_ready"),
        "dispatch_allowed": readiness.get("dispatch_allowed"),
        "run_asr": readiness.get("run_asr"),
        "write_transcripts": readiness.get("write_transcripts"),
    }


def compact_capabilities(capabilities: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "asr_engine_ready": capabilities.get("asr_engine_ready"),
        "ready_real_engines": capabilities.get("ready_real_engines"),
        "engine_candidates": [
            {
                "engine": engine.get("engine"),
                "ready": engine.get("ready"),
                "counts_as_real_asr": engine.get("counts_as_real_asr"),
                "missing": engine.get("missing") or [],
            }
            for engine in capabilities.get("engine_candidates", [])
            if isinstance(engine, Mapping)
        ],
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
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
    load_env_file()
    parser = argparse.ArgumentParser(description="Build ASR worker sandbox readiness report.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--worker-plan", default=DEFAULT_WORKER_PLAN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--readiness-report", default=DEFAULT_READINESS_REPORT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 20 idempotency audit path",
    )
    return parser.parse_args(argv)


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
