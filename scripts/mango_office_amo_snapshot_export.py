#!/usr/bin/env python3
"""Export a read-only amoCRM entity snapshot for productization matching.

Reads amoCRM contacts/leads and writes `crm_snapshots/amocrm_entities.json`
under product root. It never writes amoCRM/Tallanto/runtime DBs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.amo_snapshot_exporter import export_amo_snapshot  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/crm_snapshots/amocrm_entities.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = export_amo_snapshot(
            product_root=Path(args.product_root),
            output_path=Path(args.out),
            base_url=args.base_url,
            access_token=args.access_token,
            contacts_limit=args.contacts_limit,
            leads_limit=args.leads_limit,
            timeout_seconds=args.timeout_seconds,
            page_limit=args.page_limit,
            sleep_sec=args.sleep_sec,
        )
    except Exception as exc:
        print(f"AMO snapshot export failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", False) else 1


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "live_crm_reads": safety.get("live_crm_reads"),
        "write_crm": safety.get("write_crm"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export read-only amoCRM snapshot under product root.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--base-url", help="Defaults to CRM_AMO_BASE_URL or AMOCRM_BASE_URL.")
    parser.add_argument("--access-token", help="Defaults to CRM_AMO_API_TOKEN or AMOCRM_ACCESS_TOKEN.")
    parser.add_argument("--contacts-limit", type=int, default=500)
    parser.add_argument("--leads-limit", type=int, default=500)
    parser.add_argument("--page-limit", type=int, default=250)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
