#!/usr/bin/env python3
"""Build a portable dry-run ASR worker input pack.

This command reads the Stage 12 handoff manifest and writes only under
product_appliance. It may copy or hardlink audio into the pack, but it does not
run ASR/R+A, does not write runtime DBs, and does not write CRM.
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

from mango_mvp.productization.asr_worker_pack import build_asr_worker_pack  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_SOURCE_MANIFEST = f"{DEFAULT_PRODUCT_ROOT}/processing_handoff_stage12/asr_handoff_manifest_stage12.jsonl"
DEFAULT_PACK_ROOT = f"{DEFAULT_PRODUCT_ROOT}/asr_worker_pack_stage13"
DEFAULT_PACK_MANIFEST = f"{DEFAULT_PACK_ROOT}/asr_worker_input_manifest_stage13.jsonl"
DEFAULT_OUT = f"{DEFAULT_PACK_ROOT}/asr_worker_pack_stage13_audit.json"
DEFAULT_IDEMPOTENCY_OUT = f"{DEFAULT_PACK_ROOT}/asr_worker_pack_stage13_idempotency_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_asr_worker_pack(
            source_manifest_path=Path(args.source_manifest),
            product_root=Path(args.product_root),
            pack_root=Path(args.pack_root),
            pack_manifest_path=Path(args.pack_manifest),
            out_path=Path(args.out),
            dry_run=args.dry_run,
            mode=args.mode,
            overwrite=args.overwrite,
            verify_checksum=not args.no_verify_checksum,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"ASR worker pack failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", True) else 1


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "copies_audio": safety.get("copies_audio"),
        "hardlinks_audio": safety.get("hardlinks_audio"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_crm": safety.get("write_crm"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable dry-run ASR worker input pack.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--pack-root", default=DEFAULT_PACK_ROOT)
    parser.add_argument("--pack-manifest", default=DEFAULT_PACK_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="write manifest/audit only; do not copy or hardlink audio")
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--idempotency-out",
        action="store_const",
        const=DEFAULT_IDEMPOTENCY_OUT,
        dest="out",
        help="write to the default Stage 13 idempotency audit path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
