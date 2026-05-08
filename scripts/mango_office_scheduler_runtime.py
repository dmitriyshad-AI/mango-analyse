#!/usr/bin/env python3
"""Scheduler runtime for the isolated Mango product appliance DB.

Only `shadow_poll` dry-run/live-read-only jobs are executable here. The
scheduler does not download audio, run ASR/R+A, write CRM, or touch runtime DBs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.scheduler_runtime import (  # noqa: E402
    audit_scheduler_runtime,
    run_scheduler_tick,
    schedule_live_shadow_poll_job,
    schedule_shadow_poll_job,
)
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_RAW_PAYLOAD_NAME = "raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl"
DEFAULT_LIVE_RAW_PAYLOAD_DIR_NAME = "raw_payload_archive/live_shadow_poll"
DEFAULT_OUTPUT_DIR_NAME = "scheduler_outputs"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    product_root = Path(args.product_root).resolve(strict=False)
    product_db = Path(args.product_db).resolve(strict=False)
    guard_under_root(product_root, "product DB", product_db)

    if args.command == "plan-shadow-poll":
        out = resolve_arg_path(args.out, product_root / "scheduler_plan_shadow_poll_audit.json")
        raw_payload = resolve_arg_path(args.raw_payload, product_root / DEFAULT_RAW_PAYLOAD_NAME)
        output_dir = resolve_arg_path(args.output_dir, product_root / DEFAULT_OUTPUT_DIR_NAME)
        guard_under_root(product_root, "raw payload", raw_payload)
        guard_under_root(product_root, "output dir", output_dir)
        guard_under_root(product_root, "audit output", out)
        report = schedule_shadow_poll_job(
            product_db_path=product_db,
            product_root=product_root,
            tenant_id=args.tenant,
            raw_payload_path=raw_payload,
            output_dir=output_dir,
            window_hours=args.window_hours,
            max_attempts=args.max_attempts,
            out_path=out,
        )
    elif args.command == "plan-live-shadow-poll":
        out = resolve_arg_path(args.out, product_root / "scheduler_plan_live_shadow_poll_audit.json")
        raw_payload_dir = resolve_arg_path(args.raw_payload_dir, product_root / DEFAULT_LIVE_RAW_PAYLOAD_DIR_NAME)
        output_dir = resolve_arg_path(args.output_dir, product_root / DEFAULT_OUTPUT_DIR_NAME)
        guard_under_root(product_root, "raw payload dir", raw_payload_dir)
        guard_under_root(product_root, "output dir", output_dir)
        guard_under_root(product_root, "audit output", out)
        report = schedule_live_shadow_poll_job(
            product_db_path=product_db,
            product_root=product_root,
            tenant_id=args.tenant,
            raw_payload_dir=raw_payload_dir,
            output_dir=output_dir,
            window_hours=args.window_hours,
            max_attempts=args.max_attempts,
            base_url=args.base_url,
            allow_metadata_only=args.allow_metadata_only,
            out_path=out,
        )
    elif args.command == "tick":
        out = resolve_arg_path(args.out, product_root / "scheduler_tick_audit.json")
        guard_under_root(product_root, "audit output", out)
        report = run_scheduler_tick(
            product_db_path=product_db,
            product_root=product_root,
            worker_id=args.worker_id,
            limit=args.limit,
            lock_seconds=args.lock_seconds,
            out_path=out,
        )
    elif args.command == "audit":
        out = resolve_arg_path(args.out, product_root / "scheduler_runtime_audit.json")
        guard_under_root(product_root, "audit output", out)
        report = audit_scheduler_runtime(
            product_db_path=product_db,
            product_root=product_root,
            out_path=out,
        )
    else:
        raise ValueError(f"unknown command: {args.command}")

    print(json.dumps({"out": str(out), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok", True) else 1


def guard_under_root(product_root: Path, label: str, path: Path) -> None:
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def resolve_arg_path(value: Optional[str], default: Path) -> Path:
    return Path(value).resolve(strict=False) if value else default.resolve(strict=False)


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="Run product scheduler runtime operations.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan-shadow-poll")
    plan.add_argument("--tenant", default="foton")
    plan.add_argument("--raw-payload")
    plan.add_argument("--output-dir")
    plan.add_argument("--window-hours", type=float, default=2.0)
    plan.add_argument("--max-attempts", type=int, default=3)
    plan.add_argument("--out")

    live = sub.add_parser("plan-live-shadow-poll")
    live.add_argument("--tenant", default="foton")
    live.add_argument("--raw-payload-dir")
    live.add_argument("--output-dir")
    live.add_argument("--window-hours", type=float, default=2.0)
    live.add_argument("--max-attempts", type=int, default=3)
    live.add_argument("--base-url", default=os.getenv("MANGO_OFFICE_BASE_URL", "https://app.mango-office.ru"))
    live.add_argument("--allow-metadata-only", action="store_true")
    live.add_argument("--out")

    tick = sub.add_parser("tick")
    tick.add_argument("--worker-id", default="codex-stage3-worker")
    tick.add_argument("--limit", type=int, default=1)
    tick.add_argument("--lock-seconds", type=int, default=300)
    tick.add_argument("--out")

    audit = sub.add_parser("audit")
    audit.add_argument("--out")
    return parser.parse_args(argv)


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
