#!/usr/bin/env python3
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

from mango_mvp.customer_timeline.contact_control_sample_import import (  # noqa: E402
    ContactControlTimelineSampleConfig,
    DEFAULT_HARD_TARGET_BUCKET_COUNTS,
    DEFAULT_TARGET_BUCKET_COUNTS,
    audit_contact_control_timeline_sample,
    build_contact_control_timeline_sample,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local customer_timeline.sqlite from 100 ordinary contact-control rows and audit it."
    )
    parser.add_argument("--master-contacts-csv", required=True)
    parser.add_argument("--master-calls-csv", required=True)
    parser.add_argument("--exclude-phones-csv")
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--sample-profile", choices=("ordinary", "hard"), default="ordinary")
    parser.add_argument("--max-call-events-per-contact", type=int, default=50)
    parser.add_argument("--generated-at")
    parser.add_argument("--allow-non-russian-phones", action="store_true")
    args = parser.parse_args(argv)

    generated_at = datetime.fromisoformat(args.generated_at) if args.generated_at else datetime.now(timezone.utc).replace(microsecond=0)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    target_bucket_counts = DEFAULT_HARD_TARGET_BUCKET_COUNTS if args.sample_profile == "hard" else DEFAULT_TARGET_BUCKET_COUNTS
    config = ContactControlTimelineSampleConfig(
        master_contacts_csv=Path(args.master_contacts_csv),
        master_calls_csv=Path(args.master_calls_csv),
        exclude_phones_csv=Path(args.exclude_phones_csv) if args.exclude_phones_csv else None,
        allowed_root=Path(args.allowed_root),
        out_root=Path(args.out_root),
        timeline_db=Path(args.timeline_db),
        tenant_id=args.tenant_id,
        sample_profile=args.sample_profile,
        target_bucket_counts=target_bucket_counts,
        max_call_events_per_contact=args.max_call_events_per_contact,
        generated_at=generated_at,
        require_russian_phone=not args.allow_non_russian_phones,
    )
    import_report = build_contact_control_timeline_sample(config)
    audit_report = audit_contact_control_timeline_sample(config)
    payload = {
        "import_summary": import_report["summary"],
        "audit_summary": audit_report["summary"],
        "outputs": {
            **import_report["outputs"],
            "coverage_summary_json": str(config.out_root / "coverage_summary.json"),
            "coverage_report_csv": str(config.out_root / "timeline_coverage_report.csv"),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
