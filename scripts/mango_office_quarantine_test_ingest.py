#!/usr/bin/env python3
"""Run a disposable SQLite ingest test for a materialized Mango package.

The script writes only to the requested disposable DB and JSON report under the
allowed output root. It does not write the runtime DB, does not touch
stable_runtime, and does not run ASR/R+A.
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

from mango_mvp.productization.test_ingest import run_quarantine_test_ingest  # noqa: E402


DEFAULT_ROOT = "_local_archive_mango_api_downloads_20260507/quarantine_import"
DEFAULT_AUDIO_DIR = f"{DEFAULT_ROOT}/audio"
DEFAULT_METADATA_CSV = f"{DEFAULT_ROOT}/metadata.csv"
DEFAULT_OUT_ROOT = f"{DEFAULT_ROOT}/test_ingest"
DEFAULT_DB = f"{DEFAULT_OUT_ROOT}/quarantine_test_ingest.sqlite"
DEFAULT_OUT = f"{DEFAULT_OUT_ROOT}/test_ingest_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    report = run_quarantine_test_ingest(
        audio_dir=Path(args.audio_dir),
        metadata_csv_path=Path(args.metadata_csv),
        db_path=Path(args.db),
        out_allowed_root=Path(args.out_root),
        replace_existing=args.replace,
        allow_existing=args.allow_existing,
        limit=args.limit,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary": report["summary"], "audit": compact_audit(report["audit"])}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok") else 1


def compact_audit(audit: dict) -> dict:
    return {
        "blocked": audit.get("blocked"),
        "blocked_reasons": audit.get("blocked_reasons"),
        "warnings": audit.get("warnings"),
        "warning_reasons": audit.get("warning_reasons"),
        "status_counts": audit.get("status_counts"),
        "direction_counts": audit.get("direction_counts"),
        "db_call_records": audit.get("db_call_records"),
        "metadata_rows": audit.get("metadata_rows"),
        "audio_files": audit.get("audio_files"),
        "current_call_records_model_gaps": audit.get("current_call_records_model_gaps"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a disposable quarantine ingest test.")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
