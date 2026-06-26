#!/usr/bin/env python3
"""Run AMO read-only incremental import on a customer_timeline test copy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.amo_incremental import (  # noqa: E402
    DEFAULT_SOURCE_DB,
    AmoIncrementalConfig,
    run_amo_incremental,
)
from mango_mvp.existing_clients.amo_step1_snapshot import DEFAULT_ENV_PATH  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMO-only incremental Customer Timeline test-copy runner.")
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB), help="Absolute source customer_timeline.sqlite.")
    parser.add_argument("--out-root", required=True, help="Output folder for the test copy and reports.")
    parser.add_argument("--mcp-env", default=str(DEFAULT_ENV_PATH), help="Read-only AMO connector env file.")
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--safety-overlap-seconds", type=int, default=300)
    parser.add_argument("--page-limit", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=1.05)
    parser.add_argument("--since", help="ISO datetime lower bound for first run. Defaults to now-24h.")
    parser.add_argument("--use-existing-copy", action="store_true", help="Do not copy source DB if target already exists.")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_amo_incremental(
        AmoIncrementalConfig(
            source_db=Path(args.source_db),
            out_root=Path(args.out_root),
            mcp_env=Path(args.mcp_env),
            tenant_id=args.tenant_id,
            safety_overlap_seconds=args.safety_overlap_seconds,
            page_limit=args.page_limit,
            max_pages=args.max_pages,
            sleep_sec=args.sleep_sec,
            since=parse_datetime(args.since) if args.since else None,
            copy_db=not args.use_existing_copy,
        )
    )
    if args.summary_only:
        report = {
            "schema_version": report.get("schema_version"),
            "timeline_db": report.get("timeline_db"),
            "cursor_before": report.get("cursor_before"),
            "cursor_after": report.get("cursor_after"),
            "fetch": report.get("fetch"),
            "first_run": {
                "affected_customer_count": report.get("first_run", {}).get("affected_customer_count"),
                "changed_customer_count": report.get("first_run", {}).get("changed_customer_count"),
            },
            "second_run": {
                "affected_customer_count": report.get("second_run", {}).get("affected_customer_count"),
                "changed_customer_count": report.get("second_run", {}).get("changed_customer_count"),
            },
            "event_body_status": report.get("event_body_status"),
            "safety": report.get("safety"),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
