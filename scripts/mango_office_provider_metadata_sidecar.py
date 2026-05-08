#!/usr/bin/env python3
"""Install Mango provider metadata sidecar into the disposable test ingest DB.

This writes only the sidecar table in the disposable SQLite DB under the allowed
test-ingest root. It does not write runtime DBs and does not run ASR/R+A.
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

from mango_mvp.productization.provider_metadata import install_provider_metadata_sidecar  # noqa: E402


DEFAULT_ROOT = "_local_archive_mango_api_downloads_20260507/quarantine_import"
DEFAULT_METADATA_CSV = f"{DEFAULT_ROOT}/metadata.csv"
DEFAULT_OUT_ROOT = f"{DEFAULT_ROOT}/test_ingest"
DEFAULT_DB = f"{DEFAULT_OUT_ROOT}/quarantine_test_ingest.sqlite"
DEFAULT_OUT = f"{DEFAULT_OUT_ROOT}/provider_metadata_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = install_provider_metadata_sidecar(
        db_path=Path(args.db),
        metadata_csv_path=Path(args.metadata_csv),
        out_allowed_root=Path(args.out_root),
        replace_existing=args.replace,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary": report["summary"], "audit": compact_audit(report["audit"])}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok") else 1


def compact_audit(audit: dict) -> dict:
    return {
        "table_name": audit.get("table_name"),
        "metadata_rows": audit.get("metadata_rows"),
        "call_records": audit.get("call_records"),
        "sidecar_rows": audit.get("sidecar_rows"),
        "blocked": audit.get("blocked"),
        "blocked_reasons": audit.get("blocked_reasons"),
        "warnings": audit.get("warnings"),
        "warning_reasons": audit.get("warning_reasons"),
        "tenant_provider_counts": audit.get("tenant_provider_counts"),
        "manager_extension_counts": audit.get("manager_extension_counts"),
        "known_gaps": audit.get("known_gaps"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install provider metadata sidecar into disposable DB.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
