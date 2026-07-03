#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.family_gold import FamilyGoldCheckConfig, check_family_gold  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check family graph output against family_gold_v1 JSONL.")
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--gold-jsonl", required=True, type=Path)
    parser.add_argument("--allowed-root", default=ROOT, type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on unflagged count mismatches or false high.")
    args = parser.parse_args(argv)

    summary = check_family_gold(
        FamilyGoldCheckConfig(
            timeline_db=args.timeline_db,
            gold_jsonl=args.gold_jsonl,
            allowed_root=args.allowed_root,
            out_json=args.out_json,
            tenant_id=args.tenant_id,
        )
    )
    public_summary = {key: value for key, value in summary.items() if key != "cases"}
    print(json.dumps(public_summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and not summary["strict_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
