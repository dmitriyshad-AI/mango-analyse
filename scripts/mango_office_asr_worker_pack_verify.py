#!/usr/bin/env python3
"""Verify a portable ASR worker input pack read-only.

This command reads only the pack manifest/audio files and writes an audit JSON
under product_appliance. It does not run ASR/R+A, write DBs, copy audio, or
write CRM.
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

from mango_mvp.productization.asr_worker_pack_verifier import verify_asr_worker_pack  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PACK_ROOT = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_pack_stage13"
DEFAULT_PACK_MANIFEST = f"{DEFAULT_PACK_ROOT}/asr_worker_input_manifest_stage13.jsonl"
DEFAULT_OUT = f"{DEFAULT_PACK_ROOT}/asr_worker_pack_verify_stage14_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_PACK_ROOT}/asr_worker_pack_verify_stage14_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_asr_worker_pack(
            product_root=Path(args.product_root),
            pack_root=Path(args.pack_root),
            pack_manifest_path=Path(args.pack_manifest),
            out_path=Path(args.out),
            verify_checksum=not args.no_verify_checksum,
        )
    except Exception as exc:
        print(f"ASR worker pack verify failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "readiness_gate": compact_readiness(report["readiness_gate"]),
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
        "ready_for_worker": readiness.get("ready_for_worker"),
        "worker_may_run_asr": readiness.get("worker_may_run_asr"),
        "requires_explicit_runtime_target_approval": readiness.get("requires_explicit_runtime_target_approval"),
    }


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "read_only": safety.get("read_only"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "copies_audio": safety.get("copies_audio"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_crm": safety.get("write_crm"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a portable ASR worker input pack read-only.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--pack-root", default=DEFAULT_PACK_ROOT)
    parser.add_argument("--pack-manifest", default=DEFAULT_PACK_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 14 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
