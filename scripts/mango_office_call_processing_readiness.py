#!/usr/bin/env python3
"""Build a read-only readiness report for the industrial call-processing layer."""

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

from mango_mvp.productization.call_processing_readiness import build_call_processing_readiness_report  # noqa: E402


DEFAULT_OUT = "stable_runtime/call_processing_readiness_20260510_stage2/report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_call_processing_readiness_report(
            project_root=Path(args.project_root),
            out_path=Path(args.out),
            canonical_export_pointer=Path(args.canonical_export_pointer) if args.canonical_export_pointer else None,
            canonical_summary_path=Path(args.canonical_summary) if args.canonical_summary else None,
            stage15_summary_path=Path(args.stage15_summary) if args.stage15_summary else None,
            export_summary_path=Path(args.export_summary) if args.export_summary else None,
            crm_quality_summary_path=Path(args.crm_quality_summary) if args.crm_quality_summary else None,
            amo_queue_summary_path=Path(args.amo_queue_summary) if args.amo_queue_summary else None,
            quarantine_manifest_path=Path(args.quarantine_manifest) if args.quarantine_manifest else None,
        )
    except Exception as exc:
        print(f"call processing readiness failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(Path(args.out).resolve(strict=False)), "summary": report["summary"], "next_actions": report["next_actions"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok") else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only call-processing readiness report.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--canonical-export-pointer")
    parser.add_argument("--canonical-summary")
    parser.add_argument("--stage15-summary")
    parser.add_argument("--export-summary")
    parser.add_argument("--crm-quality-summary")
    parser.add_argument("--amo-queue-summary")
    parser.add_argument("--quarantine-manifest")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
