#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.full_memory_ingest import (  # noqa: E402
    DEFAULT_FRESH_IDENTITY_DB,
    DEFAULT_PRODUCTION_DB,
    DEFAULT_STAGE2_CORPUS_RELINK_DECISIONS,
    DEFAULT_STAGE2_CORPUS_EVENTS,
    DEFAULT_STAGE2_DELTA_RELINK_DECISIONS,
    DEFAULT_STAGE2_DELTA_EVENTS,
    FullMemoryIngestConfig,
    parse_generated_at,
    run_full_memory_production_apply,
    run_full_memory_test_procedure,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the safe full-memory customer_timeline ingest procedure. "
            "Production apply requires an explicit command and creates a full DB backup first."
        )
    )
    parser.add_argument("command", choices=("test-copy", "apply-production"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--production-db", type=Path, default=DEFAULT_PRODUCTION_DB)
    parser.add_argument(
        "--test-out-root",
        type=Path,
        default=ROOT
        / "product_data"
        / "customer_timeline"
        / f"customer_timeline_prod_20260621_testcopy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--identity-db", type=Path, default=DEFAULT_FRESH_IDENTITY_DB)
    parser.add_argument("--event-jsonl", type=Path, action="append")
    parser.add_argument("--relink-decision-csv", type=Path, action="append")
    parser.add_argument("--generated-at", default="2026-06-21T00:00:00+00:00")
    parser.add_argument("--email-limit", type=int)
    parser.add_argument("--max-call-events-per-contact", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    event_paths = tuple(args.event_jsonl or (DEFAULT_STAGE2_CORPUS_EVENTS, DEFAULT_STAGE2_DELTA_EVENTS))
    decision_paths = tuple(
        args.relink_decision_csv
        or (DEFAULT_STAGE2_CORPUS_RELINK_DECISIONS, DEFAULT_STAGE2_DELTA_RELINK_DECISIONS)
    )
    config = FullMemoryIngestConfig(
        project_root=args.project_root,
        production_db=args.production_db,
        test_out_root=args.test_out_root,
        tenant_id=args.tenant_id,
        identity_db=args.identity_db,
        event_jsonl_paths=event_paths,
        relink_decision_paths=decision_paths,
        generated_at=parse_generated_at(args.generated_at),
        email_limit=args.email_limit,
        max_call_events_per_contact=args.max_call_events_per_contact,
    )
    if args.command == "test-copy":
        report = run_full_memory_test_procedure(config)
    else:
        report = run_full_memory_production_apply(config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    validation = report.get("validation", {})
    if args.command == "test-copy":
        return 0 if validation.get("production_apply_not_performed") else 1
    return 0 if validation.get("production_apply_performed") and not validation.get("restore_performed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
