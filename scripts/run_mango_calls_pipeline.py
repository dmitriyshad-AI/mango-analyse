#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.calls_two_processes import (  # noqa: E402
    CallsTwoProcessesConfig,
    run_capture,
    run_cycle,
    run_local_watchdog,
    run_pipeline,
    run_process_a,
    run_process_b,
    pipeline_freshness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated two-process Mango calls pipeline.")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="Mango API -> immutable audio -> capture manifest only.")
    capture.add_argument("--since")
    capture.add_argument("--until")
    sub.add_parser("pipeline", help="Drain a frozen capture snapshot through sequential stages.")
    sub.add_parser("watchdog", help="Read local heartbeats and provenance without processing calls.")
    process_a = sub.add_parser("process-a", help="Capture, ASR, Resolve+Analyze, publish ready drop DB.")
    process_a.add_argument("--since")
    process_a.add_argument("--until")
    process_a.add_argument("--skip-capture", action="store_true")
    process_a.add_argument("--skip-workers", action="store_true")
    sub.add_parser("process-b", help="Build mango_processed_summary and import it into timeline staging.")
    sub.add_parser("status", help="Check data watermarks for both scheduled processes.")
    cycle = sub.add_parser("cycle", help="Run process A, then process B from one scheduler trigger.")
    cycle.add_argument("--since")
    cycle.add_argument("--until")
    cycle.add_argument("--skip-capture", action="store_true")
    cycle.add_argument("--skip-workers", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = CallsTwoProcessesConfig.from_json(Path(args.config))
        if args.command == "capture":
            report = run_capture(config, since=args.since, until=args.until)
        elif args.command == "pipeline":
            report = run_pipeline(config)
        elif args.command == "watchdog":
            report = run_local_watchdog(config)
        elif args.command == "process-a":
            report = run_process_a(
                config,
                since=args.since,
                until=args.until,
                skip_capture=args.skip_capture,
                skip_workers=args.skip_workers,
            )
        elif args.command == "process-b":
            report = run_process_b(config)
        elif args.command == "cycle":
            report = run_cycle(
                config,
                since=args.since,
                until=args.until,
                skip_capture=args.skip_capture,
                skip_workers=args.skip_workers,
            )
        else:
            report = pipeline_freshness(config)
    except Exception as exc:
        report = {
            "schema_version": "mango_calls_two_processes_v1",
            "process": args.command,
            "status": "failed",
            "stop_reason": f"cli_exception:{type(exc).__name__}",
            "counters": {},
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"ok", "idle", "locked", "deferred", "fresh"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
