#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mango_mvp.customer_timeline.stage5_money_ingest import Stage5MoneyIngestConfig, run_stage5_money_ingest
from mango_mvp.customer_timeline.store import json_dumps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest primary money facts into staging customer_timeline.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--source", required=True, help="Safe-projection JSON from read-only AMO/Tallanto collection")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--apply", action="store_true", help="Write only to the staging DB. Default is dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_stage5_money_ingest(
        Stage5MoneyIngestConfig(
            timeline_db_path=Path(args.timeline_db),
            allowed_root=Path(args.allowed_root),
            source_path=Path(args.source),
            out_dir=Path(args.out_dir),
            tenant_id=args.tenant_id,
            apply=bool(args.apply),
        )
    )
    print(json_dumps(report))
    return 0 if report["final_checks"]["quick_check"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
