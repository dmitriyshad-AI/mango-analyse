#!/usr/bin/env python3
"""Install Mango manager identity mapping into the disposable test ingest DB.

This writes only the manager identity sidecar table and view in the disposable
SQLite DB under the quarantine root. It does not write runtime DBs, does not
run ASR/R+A, and does not write CRM.
"""

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

from mango_mvp.productization.manager_identity import install_manager_identity_map  # noqa: E402


DEFAULT_ROOT = "_local_archive_mango_api_downloads_20260507/quarantine_import"
DEFAULT_OUT_ROOT = DEFAULT_ROOT
DEFAULT_DB = f"{DEFAULT_ROOT}/test_ingest/quarantine_test_ingest.sqlite"
DEFAULT_MANGO_USERS = f"{DEFAULT_ROOT}/raw_payload_archive/mango_users_config_20260507.json"
DEFAULT_AMO_USERS = "prod_runtime_transfer/data_handoff/live_export/users.json"
DEFAULT_OUT = f"{DEFAULT_ROOT}/test_ingest/manager_identity_audit.json"
DEFAULT_CSV_OUT = f"{DEFAULT_ROOT}/test_ingest/manager_identity_map.csv"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    amo_users = Path(args.amo_users) if args.amo_users else None
    csv_out = Path(args.csv_out) if args.csv_out else None
    report = install_manager_identity_map(
        db_path=Path(args.db),
        mango_users_path=Path(args.mango_users),
        amo_users_path=amo_users,
        out_allowed_root=Path(args.out_root),
        replace_existing=args.replace,
        csv_out=csv_out,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out_path), "summary": report["summary"], "audit": compact_audit(report["audit"])},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok") else 1


def compact_audit(audit: dict) -> dict:
    return {
        "table_name": audit.get("table_name"),
        "view_name": audit.get("view_name"),
        "manager_extensions": audit.get("manager_extensions"),
        "sidecar_rows": audit.get("sidecar_rows"),
        "view_rows": audit.get("view_rows"),
        "mapped_mango_users": audit.get("mapped_mango_users"),
        "missing_mango_users": audit.get("missing_mango_users"),
        "crm_owner_matched": audit.get("crm_owner_matched"),
        "crm_owner_unmatched": audit.get("crm_owner_unmatched"),
        "calls_with_mango_user": audit.get("calls_with_mango_user"),
        "calls_with_crm_owner": audit.get("calls_with_crm_owner"),
        "crm_owner_unmatched_call_count": audit.get("crm_owner_unmatched_call_count"),
        "blocked": audit.get("blocked"),
        "blocked_reasons": audit.get("blocked_reasons"),
        "warnings": audit.get("warnings"),
        "warning_reasons": audit.get("warning_reasons"),
        "mapping_status_counts": audit.get("mapping_status_counts"),
        "crm_match_status_counts": audit.get("crm_match_status_counts"),
        "manager_call_counts": audit.get("manager_call_counts"),
        "manual_review_items": audit.get("manual_review_items"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Mango manager identity map into disposable DB.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--mango-users", default=DEFAULT_MANGO_USERS)
    parser.add_argument("--amo-users", default=DEFAULT_AMO_USERS)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
