#!/usr/bin/env python3
"""Build the canonical read-only customer_timeline from local snapshots."""

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

from mango_mvp.customer_timeline.canonical_readonly_import import (  # noqa: E402
    CANONICAL_READONLY_NORMALIZER_VERSION,
    DEFAULT_OUT_ROOT,
    CanonicalReadonlyTimelineConfig,
    build_canonical_readonly_customer_timeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local canonical read-only customer_timeline.sqlite from "
            "Mango, AMO, Tallanto and mail handoff snapshots."
        )
    )
    parser.add_argument("--project-root", default=str(ROOT), help="Project root. Defaults to this repository.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root under project root.")
    parser.add_argument("--timeline-db", help="Output SQLite path. Defaults to <out-root>/customer_timeline.sqlite.")
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--current-runtime-json")
    parser.add_argument("--master-contacts-csv")
    parser.add_argument("--master-calls-csv")
    parser.add_argument("--canonical-calls-db")
    parser.add_argument("--amo-contacts-csv")
    parser.add_argument("--amo-deals-csv")
    parser.add_argument("--mail-handoff-db")
    parser.add_argument("--mail-bridge-db")
    parser.add_argument("--max-call-events-per-contact", type=int, default=0)
    parser.add_argument("--source-cache-dir", help="Directory for normalized source-batch cache.")
    parser.add_argument(
        "--normalizer-version",
        default=CANONICAL_READONLY_NORMALIZER_VERSION,
        help="Cache normalizer version. Changing it invalidates all source-batch cache entries.",
    )
    parser.add_argument("--disable-source-cache", action="store_true", help="Parse all sources even when cache entries exist.")
    parser.add_argument(
        "--generated-at",
        help="UTC ISO timestamp for deterministic tests/rebuilds. Defaults to current UTC time.",
    )
    return parser.parse_args(argv)


def parse_generated_at(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = CanonicalReadonlyTimelineConfig(
        project_root=Path(args.project_root).expanduser(),
        out_root=Path(args.out_root).expanduser(),
        timeline_db=optional_path(args.timeline_db),
        tenant_id=args.tenant_id,
        current_runtime_json=optional_path(args.current_runtime_json),
        master_contacts_csv=optional_path(args.master_contacts_csv),
        master_calls_csv=optional_path(args.master_calls_csv),
        canonical_calls_db=optional_path(args.canonical_calls_db),
        amo_contacts_csv=optional_path(args.amo_contacts_csv),
        amo_deals_csv=optional_path(args.amo_deals_csv),
        mail_handoff_db=optional_path(args.mail_handoff_db),
        mail_bridge_db=optional_path(args.mail_bridge_db),
        generated_at=parse_generated_at(args.generated_at),
        max_call_events_per_contact=max(0, int(args.max_call_events_per_contact)),
        source_cache_dir=optional_path(args.source_cache_dir),
        normalizer_version=args.normalizer_version,
        disable_source_cache=bool(args.disable_source_cache),
    )
    report = build_canonical_readonly_customer_timeline(config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
