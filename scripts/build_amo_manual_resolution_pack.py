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

from mango_mvp.productization.amo_manual_resolution import build_amo_manual_resolution_pack  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    defaults = _runtime_defaults()
    queue_root = Path(args.queue_root or defaults["queue_root"])
    source_csv = Path(args.source_csv or defaults["source_csv"])
    result = build_amo_manual_resolution_pack(
        queue_root=queue_root,
        source_csv=source_csv,
        out_root=Path(args.out_root),
        decisions_csv=Path(args.decisions_csv) if args.decisions_csv else None,
    )
    print(json.dumps({"out_root": str(Path(args.out_root).resolve(strict=False)), "summary": result["summary"], "next_actions": result["next_actions"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["summary"].get("validation_ok") else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed AMO manual-resolution pack from AMO queue buckets.")
    parser.add_argument("--queue-root", default="", help="Defaults to CURRENT_RUNTIME.json paths.amo_queue_summary parent.")
    parser.add_argument("--source-csv", default="", help="Defaults to CURRENT_RUNTIME.json paths.amo_export_ready_csv.")
    parser.add_argument("--out-root", default="stable_runtime/amo_manual_resolution_20260511_v1")
    parser.add_argument("--decisions-csv", default="")
    return parser.parse_args(argv)


def _runtime_defaults() -> dict[str, str]:
    runtime_path = ROOT / "stable_runtime" / "CURRENT_RUNTIME.json"
    if not runtime_path.exists():
        return {
            "queue_root": "stable_runtime/amo_writeback_queue_20260510_v2_production",
            "source_csv": "stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_ru.csv",
        }
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    paths = payload.get("paths") if isinstance(payload, dict) else {}
    if not isinstance(paths, dict):
        paths = {}
    queue_summary = str(paths.get("amo_queue_summary") or "")
    source_csv = str(paths.get("amo_export_ready_csv") or "")
    return {
        "queue_root": str(Path(queue_summary).parent) if queue_summary else "stable_runtime/amo_writeback_queue_20260510_v2_production",
        "source_csv": source_csv or "stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_ru.csv",
    }


if __name__ == "__main__":
    raise SystemExit(main())
