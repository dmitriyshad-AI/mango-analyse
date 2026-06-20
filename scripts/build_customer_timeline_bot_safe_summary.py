#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.customer_timeline.bot_safe_summary import BotSafeSummaryBuildConfig, build_bot_safe_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bot-safe customer_timeline summaries from structural fields only.")
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_bot_safe_summaries(
        BotSafeSummaryBuildConfig(
            timeline_db=args.timeline_db,
            allowed_root=args.allowed_root,
            tenant_id=args.tenant_id,
            apply=args.apply,
            limit=args.limit,
        )
    ).to_json_dict()
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
