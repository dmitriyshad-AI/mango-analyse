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
    parser.add_argument(
        "--reconsider-pending",
        action="store_true",
        help="Recheck prior unmatched mail after identity sources were refreshed; brand still gates context, not identity.",
    )
    parser.add_argument(
        "--revalidate-existing-strong",
        action="store_true",
        help="Separately revalidate existing strong mail links; only explicit conflicts may lower them.",
    )
    parser.add_argument(
        "--archive-db",
        action="append",
        default=[],
        help="Read-only canonical mail archive fallback; may be provided more than once.",
    )
    parser.add_argument(
        "--tallanto-identity-db",
        action="append",
        default=[],
        help="Read-only historical Tallanto identity map; may be provided more than once.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Write only aggregate counts; do not persist row-level identifiers.",
    )
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
        reconsider_pending=bool(args.reconsider_pending),
        revalidate_existing_strong=bool(args.revalidate_existing_strong),
        fallback_archive_dbs=tuple(Path(path) for path in args.archive_db),
        tallanto_identity_dbs=tuple(Path(path) for path in args.tallanto_identity_db),
        aggregate_only=bool(args.aggregate_only),
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
