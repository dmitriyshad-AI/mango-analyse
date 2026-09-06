#!/usr/bin/env python3
"""Run customer_timeline nightly service on staging.

The service does not install launchd and does not write prod/CRM/Tallanto.
It executes only supported local source steps from the JSON config.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.nightly_service import (  # noqa: E402
    run_nightly_service,
    service_config_from_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staging customer_timeline nightly service.")
    parser.add_argument("--config", required=True, help="Nightly service JSON config.")
    parser.add_argument("--summary-only", action="store_true", help="Print compact service summary.")
    parser.add_argument("--approve-code-release", nargs=2, metavar=("PREVIOUS_SHA", "NEW_HEAD"))
    parser.add_argument("--review-receipt", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.approve_code_release) != bool(args.review_receipt):
        parser.error("--approve-code-release and --review-receipt must be provided together")
    if args.approve_code_release:
        from mango_mvp.customer_timeline.nightly_service import approve_writer_code_release

        try:
            report = approve_writer_code_release(Path(args.config), *args.approve_code_release, args.review_receipt)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(json.dumps({"status": "stopped", "reason": str(exc)}))
            return 75 if isinstance(exc, TimeoutError) else 1
        print(json.dumps(report, sort_keys=True))
        return 0
    report = run_nightly_service(service_config_from_json(Path(args.config)))
    exit_code = 0 if report.get("overall_status") == "ok" and report.get("data_quality_status") == "pass" else 1
    if args.summary_only:
        report = {
            "schema_version": report.get("schema_version"),
            "run_id": report.get("run_id"),
            "started_at": report.get("started_at"),
            "finished_at": report.get("finished_at"),
            "overall_status": report.get("overall_status"),
            "data_quality_status": report.get("data_quality_status"),
            "partial_failure": report.get("partial_failure"),
            "failed_required_steps": report.get("failed_required_steps"),
            "required_sources_check": report.get("required_sources_check"),
            "degraded_steps": report.get("degraded_steps"),
            "degraded_sources": report.get("degraded_sources"),
            "duration_seconds": report.get("duration_seconds"),
            "steps": [
                {
                    "name": step.get("name"),
                    "kind": step.get("kind"),
                    "status": step.get("status"),
                    "summary": step.get("summary"),
                }
                for step in report.get("steps", ())
            ],
            "snapshot_manifest": report.get("snapshot_manifest"),
            "safety": report.get("safety"),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
