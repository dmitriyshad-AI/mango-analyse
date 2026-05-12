#!/usr/bin/env python3
"""Resolve product calls to CRM entity candidates from a local snapshot.

This is read-only: it reads product DB + snapshot file and writes only a JSON
report under product root. It does not call live CRM and does not write CRM.
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

from mango_mvp.productization.crm_entity_resolver import build_crm_entity_resolution_report  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_CRM_SNAPSHOT = f"{DEFAULT_PRODUCT_ROOT}/crm_snapshots/amocrm_entities.json"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/crm_entity_resolver_stage6/crm_entity_resolution_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_crm_entity_resolution_report(
            product_db_path=Path(args.product_db),
            product_root=Path(args.product_root),
            crm_snapshot_path=Path(args.crm_snapshot),
            out_path=Path(args.out),
            limit=args.limit,
        )
    except Exception as exc:
        print(f"CRM entity resolver failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "action_counts": report["action_counts"],
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
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "live_crm_reads": safety.get("live_crm_reads"),
        "write_crm": safety.get("write_crm"),
        "write_tallanto": safety.get("write_tallanto"),
        "run_asr": safety.get("run_asr"),
        "run_ra": safety.get("run_ra"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve product calls to CRM entity candidates.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--crm-snapshot", default=DEFAULT_CRM_SNAPSHOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
