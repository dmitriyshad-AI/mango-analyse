#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mango_mvp.customer_timeline.stage4b_bot_opening import Stage4BBotOpeningConfig, run_stage4b_bot_opening
from mango_mvp.customer_timeline.store import json_dumps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open vetted mail stage2 chunks to bot on staging DB.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--apply", action="store_true", help="Write to the staging DB. Default is dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_stage4b_bot_opening(
        Stage4BBotOpeningConfig(
            timeline_db_path=Path(args.timeline_db),
            allowed_root=Path(args.allowed_root),
            out_dir=Path(args.out_dir),
            tenant_id=args.tenant_id,
            apply=bool(args.apply),
        )
    )
    print(json_dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
