#!/usr/bin/env python3
"""Offline, read-only conclusive-reason breakdown of personal Wappi chats without an
AMO contact (BLOK C1). Reads only the local `wappi_amo_links` cache produced by
`collect_wappi_widget_links`; makes zero network calls itself."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.customer_timeline.wappi_unmatched_breakdown import (
    build_wappi_unmatched_breakdown,
    write_wappi_unmatched_breakdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widget-link-db", type=Path, required=True, help="Path to the local wappi_amo_links sqlite cache.")
    parser.add_argument("--exclusions", type=Path, default=None, help="Optional JSON stoplist of employee/test/system chat ids.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional .codex_local output dir for the per-chat CSV + summary.")
    args = parser.parse_args()

    report = build_wappi_unmatched_breakdown(widget_link_db=args.widget_link_db, exclusions_path=args.exclusions)
    files = {}
    if args.out_dir is not None:
        files = write_wappi_unmatched_breakdown(args.out_dir, report)
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps({**summary, "files": files}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
