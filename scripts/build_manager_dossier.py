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

from mango_mvp.customer_timeline.manager_dossier import build_manager_dossier_workbook  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build manager dossier workbook from staging Customer Timeline.")
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--allowed-root", default=ROOT, type=Path)
    parser.add_argument("--out-xlsx", required=True, type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--customer-id", action="append", default=[])
    parser.add_argument("--customer-ids-file", type=Path)
    parser.add_argument("--canonical-calls-db", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    customer_ids = list(args.customer_id or [])
    if args.customer_ids_file:
        customer_ids.extend(
            line.strip()
            for line in args.customer_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    summary = build_manager_dossier_workbook(
        timeline_db=args.timeline_db,
        allowed_root=args.allowed_root,
        out_xlsx=args.out_xlsx,
        tenant_id=args.tenant_id,
        customer_ids=tuple(dict.fromkeys(customer_ids)),
        canonical_calls_db=args.canonical_calls_db,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
