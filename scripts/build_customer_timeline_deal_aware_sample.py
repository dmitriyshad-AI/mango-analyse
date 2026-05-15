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

from mango_mvp.customer_timeline.deal_aware_sample_import import (  # noqa: E402
    DealAwareTimelineSampleConfig,
    audit_deal_aware_timeline_sample,
    build_deal_aware_timeline_sample,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local customer_timeline.sqlite from a fixed deal-aware sample and audit it."
    )
    parser.add_argument("--selected-groups-csv", required=True)
    parser.add_argument("--all-candidates-csv", required=True)
    parser.add_argument("--master-calls-csv", required=True)
    parser.add_argument("--master-contacts-csv", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--max-call-events-per-group", type=int, default=50)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)

    generated_at = datetime.fromisoformat(args.generated_at) if args.generated_at else datetime.now(timezone.utc).replace(microsecond=0)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    config = DealAwareTimelineSampleConfig(
        selected_groups_csv=Path(args.selected_groups_csv),
        all_candidates_csv=Path(args.all_candidates_csv),
        master_calls_csv=Path(args.master_calls_csv),
        master_contacts_csv=Path(args.master_contacts_csv),
        allowed_root=Path(args.allowed_root),
        out_root=Path(args.out_root),
        timeline_db=Path(args.timeline_db),
        tenant_id=args.tenant_id,
        max_call_events_per_group=args.max_call_events_per_group,
        generated_at=generated_at,
    )
    import_report = build_deal_aware_timeline_sample(config)
    audit_report = audit_deal_aware_timeline_sample(config)
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
