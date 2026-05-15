#!/usr/bin/env python3
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

from mango_mvp.customer_timeline.context_provider import (  # noqa: E402
    CustomerTimelineCoveragePaths,
    audit_customer_timeline_coverage,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit customer timeline coverage for deal-aware candidate phones.")
    parser.add_argument("--deal-aware-candidates", required=True)
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--tenant-id", default="foton")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_customer_timeline_coverage(
        CustomerTimelineCoveragePaths(
            deal_aware_candidates_csv=Path(args.deal_aware_candidates).expanduser().resolve(),
            timeline_db=Path(args.timeline_db).expanduser().resolve(),
            out_root=Path(args.out_root).expanduser().resolve(),
            tenant_id=args.tenant_id,
        )
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
