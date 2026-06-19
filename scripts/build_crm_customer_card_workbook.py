#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.crm_card_workbook import CrmCardWorkbookConfig, build_crm_card_workbook  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = CrmCardWorkbookConfig(
        timeline_db=Path(args.timeline_db),
        allowed_root=Path(args.allowed_root),
        out_xlsx=Path(args.out_xlsx),
        tenant_id=args.tenant_id,
        sample_size=args.sample_size,
        manager_facts_csv=Path(args.manager_facts_csv) if args.manager_facts_csv else None,
        amo_base_url=args.amo_base_url,
    )
    summary = build_crm_card_workbook(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only CRM customer card review workbook from customer_timeline.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--out-xlsx", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--manager-facts-csv",
        default="",
        help="Optional old analyze fallback facts keyed by customer_id/new_customer_id/old_customer_id. No phone join.",
    )
    parser.add_argument("--amo-base-url", default="https://educent.amocrm.ru")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
