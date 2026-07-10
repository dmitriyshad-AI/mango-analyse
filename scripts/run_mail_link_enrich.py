#!/usr/bin/env python3
"""Run staging-only mail link enrich for pending mail_archive_stage2 events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.mail_link_enrich import MailLinkEnrichConfig, run_mail_link_enrich


DEFAULT_ALLOWED_ROOT = ROOT / ".codex_local" / "staging"
DEFAULT_TIMELINE_DB = DEFAULT_ALLOWED_ROOT / "customer_timeline_staging.sqlite"
DEFAULT_OUT_DIR = DEFAULT_ALLOWED_ROOT / "mail_link_enrich"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich pending mail links in a staging Customer Timeline DB.")
    parser.add_argument("--timeline-db", default=str(DEFAULT_TIMELINE_DB))
    parser.add_argument("--allowed-root", default=str(DEFAULT_ALLOWED_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--max-events", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Write changes to the staging DB.")
    mode.add_argument("--dry-run", action="store_true", help="Plan only. This is the default.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MailLinkEnrichConfig(
        timeline_db=Path(args.timeline_db),
        allowed_root=Path(args.allowed_root),
        out_dir=Path(args.out_dir),
        tenant_id=args.tenant_id,
        apply=bool(args.apply),
        max_events=args.max_events,
    )
    report = run_mail_link_enrich(config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    if report.get("safety", {}).get("allowed_for_bot_changed"):
        return 2
    if report.get("safety", {}).get("mail_stage2_allowed_for_bot_changed"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
