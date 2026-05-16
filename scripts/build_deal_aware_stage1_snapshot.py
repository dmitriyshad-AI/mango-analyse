#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mango_mvp.deal_aware.stage1_snapshot import (
    Stage1Paths,
    build_stage1_snapshot,
    loss_reason_dirs_default,
    stage_dirs_default,
    writeoff_xlsx_default,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_CONTACTS = (
    PROJECT_ROOT / "stable_runtime" / "sales_master_export_20260513_human_history_v8_normalized" / "master_contacts_ru.csv"
)
DEFAULT_MASTER_CALLS = (
    PROJECT_ROOT / "stable_runtime" / "sales_master_export_20260513_human_history_v8_normalized" / "master_calls_ru.csv"
)
DEFAULT_AMO_READY = (
    PROJECT_ROOT / "stable_runtime" / "sales_master_export_20260513_human_history_v8_normalized" / "amo_export_ready_ru.csv"
)
DEFAULT_CALLS = (
    PROJECT_ROOT / "stable_runtime" / "insight_readiness_report_after_quality_backfill_20260510_v1" / "calls_terminal_analyzed.csv"
)
DEFAULT_CURRENT_RUNTIME = PROJECT_ROOT / "stable_runtime" / "CURRENT_RUNTIME.json"
DEFAULT_CANONICAL_EXPORT_POINTER = PROJECT_ROOT / "stable_runtime" / "CANONICAL_EXPORT.txt"
DEFAULT_AMO_LIVE_SNAPSHOT_DIR = PROJECT_ROOT / "stable_runtime" / "deal_aware_amo_live_snapshot_20260513_v1"
DEFAULT_TALLANTO_STUDENTS = PROJECT_ROOT / "_external_handoffs" / "tallanto_students_export_2026-05-12" / "Ученики.csv"
DEFAULT_TALLANTO_WRITEOFF_COMBINED = (
    PROJECT_ROOT / "stable_runtime" / "tallanto_write_off_visits_history_20260512" / "write_off_visits_combined_unique.csv"
)
DEFAULT_TALLANTO_WRITEOFF_SUMMARY = (
    PROJECT_ROOT / "stable_runtime" / "tallanto_write_off_visits_history_20260512" / "write_off_visits_by_student.csv"
)
DEFAULT_TALLANTO_SCHEMA = PROJECT_ROOT / "stable_runtime" / "tallanto_schema_extended_20260512" / "tallanto_fields_extended.json"
DEFAULT_QUALITY_SUMMARIES = (
    PROJECT_ROOT / "stable_runtime" / "sales_master_export_20260513_human_history_v8_normalized" / "summary.json",
    PROJECT_ROOT / "stable_runtime" / "tenant_text_normalizer_gate_20260513_v2_after_rebuild" / "summary.json",
    PROJECT_ROOT / "stable_runtime" / "transcript_quality_stage15_export_gate_20260510_v11_frozen_gate" / "summary.json",
    PROJECT_ROOT / "stable_runtime" / "crm_writeback_quality_gate_20260513_human_history_v4_normalized" / "summary.json",
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "stable_runtime" / "deal_aware_stage1_snapshot_20260513_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only Stage 1 snapshot for deal-aware layer.")
    parser.add_argument("--master-contacts", default=str(DEFAULT_MASTER_CONTACTS))
    parser.add_argument("--master-calls", default=str(DEFAULT_MASTER_CALLS))
    parser.add_argument("--amo-ready", default=str(DEFAULT_AMO_READY))
    parser.add_argument("--calls", default=str(DEFAULT_CALLS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--current-runtime", default=str(DEFAULT_CURRENT_RUNTIME))
    parser.add_argument("--canonical-export-pointer", default=str(DEFAULT_CANONICAL_EXPORT_POINTER))
    parser.add_argument("--amo-live-snapshot-dir", default=str(DEFAULT_AMO_LIVE_SNAPSHOT_DIR))
    parser.add_argument("--tallanto-students", default=str(DEFAULT_TALLANTO_STUDENTS))
    parser.add_argument("--tallanto-writeoff-combined", default=str(DEFAULT_TALLANTO_WRITEOFF_COMBINED))
    parser.add_argument("--tallanto-writeoff-summary", default=str(DEFAULT_TALLANTO_WRITEOFF_SUMMARY))
    parser.add_argument("--tallanto-schema", default=str(DEFAULT_TALLANTO_SCHEMA))
    parser.add_argument("--quality-summary", action="append", default=[])
    parser.add_argument("--amo-stage-dir", action="append", default=[])
    parser.add_argument("--amo-loss-reason-dir", action="append", default=[])
    parser.add_argument("--tallanto-writeoff-xlsx", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    amo_stage_dirs = tuple(Path(path).expanduser().resolve() for path in args.amo_stage_dir) or stage_dirs_default(
        PROJECT_ROOT
    )
    loss_reason_dirs = tuple(Path(path).expanduser().resolve() for path in args.amo_loss_reason_dir) or loss_reason_dirs_default(
        PROJECT_ROOT
    )
    writeoff_xlsx = tuple(Path(path).expanduser().resolve() for path in args.tallanto_writeoff_xlsx) or writeoff_xlsx_default(
        PROJECT_ROOT
    )
    quality_summaries = tuple(Path(path).expanduser().resolve() for path in args.quality_summary) or tuple(
        path for path in DEFAULT_QUALITY_SUMMARIES if path.exists()
    )
    summary = build_stage1_snapshot(
        Stage1Paths(
            master_contacts_csv=Path(args.master_contacts).expanduser().resolve(),
            master_calls_csv=Path(args.master_calls).expanduser().resolve(),
            amo_ready_csv=Path(args.amo_ready).expanduser().resolve(),
            calls_csv=Path(args.calls).expanduser().resolve(),
            out_root=Path(args.out_root).expanduser().resolve(),
            current_runtime_json=Path(args.current_runtime).expanduser().resolve(),
            canonical_export_pointer=Path(args.canonical_export_pointer).expanduser().resolve(),
            amo_live_snapshot_dir=Path(args.amo_live_snapshot_dir).expanduser().resolve(),
            tallanto_students_csv=Path(args.tallanto_students).expanduser().resolve(),
            tallanto_writeoff_combined_csv=Path(args.tallanto_writeoff_combined).expanduser().resolve(),
            tallanto_writeoff_summary_csv=Path(args.tallanto_writeoff_summary).expanduser().resolve(),
            tallanto_schema_json=Path(args.tallanto_schema).expanduser().resolve(),
            quality_summary_paths=quality_summaries,
            amo_stage_dirs=amo_stage_dirs,
            amo_loss_reason_dirs=loss_reason_dirs,
            tallanto_writeoff_xlsx=writeoff_xlsx,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
