#!/usr/bin/env python3
"""Archive Mango payload rows and fill raw_payload_ref in disposable sidecar DB.

This writes only under the quarantine output root and updates only the
provider_call_metadata table in the disposable SQLite DB.
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

from mango_mvp.productization.payload_archive import archive_mango_payloads_and_update_sidecar  # noqa: E402


DEFAULT_ROOT = "_local_archive_mango_api_downloads_20260507/quarantine_import"
DEFAULT_METADATA_CSV = f"{DEFAULT_ROOT}/metadata.csv"
DEFAULT_OUT_ROOT = DEFAULT_ROOT
DEFAULT_DB = f"{DEFAULT_ROOT}/test_ingest/quarantine_test_ingest.sqlite"
DEFAULT_SOURCE_PAYLOAD = f"{DEFAULT_ROOT}/raw_payload_archive/shadow_poll_raw_rows.jsonl"
DEFAULT_ARCHIVE_ROOT = f"{DEFAULT_ROOT}/raw_payload_archive/by_call"
DEFAULT_OUT = f"{DEFAULT_ROOT}/test_ingest/payload_archive_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = archive_mango_payloads_and_update_sidecar(
        db_path=Path(args.db),
        metadata_csv_path=Path(args.metadata_csv),
        source_payload_path=Path(args.source_payload),
        archive_root=Path(args.archive_root),
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
        "archive_root": audit.get("archive_root"),
        "archived_entries": audit.get("archived_entries"),
        "archive_files": audit.get("archive_files"),
        "archive_file_rows": audit.get("archive_file_rows"),
        "sidecar_rows": audit.get("sidecar_rows"),
        "sidecar_refs_present": audit.get("sidecar_refs_present"),
        "blocked": audit.get("blocked"),
        "blocked_reasons": audit.get("blocked_reasons"),
        "warnings": audit.get("warnings"),
        "warning_reasons": audit.get("warning_reasons"),
        "source_kind_counts": audit.get("source_kind_counts"),
        "tenant_provider_counts": audit.get("tenant_provider_counts"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Mango payload rows and update raw_payload_ref.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--source-payload", default=DEFAULT_SOURCE_PAYLOAD)
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
