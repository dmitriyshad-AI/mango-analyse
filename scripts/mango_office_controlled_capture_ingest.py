#!/usr/bin/env python3
"""Plan/apply controlled Mango capture ingest from shadow poll reports.

Default mode is read-only. The apply mode writes only product DB capture inbox
rows for enqueue decisions. It never downloads audio, runs ASR/R+A, writes
runtime DBs, or writes CRM.
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

from mango_mvp.productization.controlled_capture_ingest import build_controlled_capture_ingest_report  # noqa: E402
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/controlled_capture_ingest_stage4"
DEFAULT_OUT = f"{DEFAULT_OUT_DIR}/controlled_capture_ingest_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root).resolve(strict=False)
    product_db = Path(args.product_db).resolve(strict=False)
    report_path = Path(args.report).resolve(strict=False)
    out = Path(args.out).resolve(strict=False)
    guard_under_root(product_root, "product DB", product_db)
    guard_under_root(product_root, "shadow poll report", report_path)
    guard_under_root(product_root, "controlled ingest output", out)

    try:
        report = build_controlled_capture_ingest_report(
            product_db_path=product_db,
            product_root=product_root,
            report_path=report_path,
            out_path=out,
            apply=args.command == "apply",
            delayed_recording_grace_hours=args.delayed_recording_grace_hours,
        )
    except Exception as exc:
        print(f"controlled capture ingest failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "out": str(out),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", False) else 1


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "product_db_writes": safety.get("product_db_writes"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "downloads_audio": safety.get("downloads_audio"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
    }


def guard_under_root(product_root: Path, label: str, path: Path) -> None:
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan/apply controlled Mango capture ingest.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--report", required=True, help="Shadow poll JSON report under product root.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--delayed-recording-grace-hours", type=int, default=24)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Read-only controlled ingest plan.")
    sub.add_parser("apply", help="Apply enqueue decisions to product DB capture inbox only.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
