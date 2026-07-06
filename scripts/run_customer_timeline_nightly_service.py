#!/usr/bin/env python3
"""Run customer_timeline nightly service on staging.

The service does not install launchd and does not write prod/CRM/Tallanto.
It executes only supported local source steps from the JSON config.
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

from mango_mvp.customer_timeline.nightly_service import (  # noqa: E402
    run_nightly_service,
    service_config_from_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staging customer_timeline nightly service.")
    parser.add_argument("--config", required=True, help="Nightly service JSON config.")
    parser.add_argument("--summary-only", action="store_true", help="Print compact service summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_nightly_service(service_config_from_json(Path(args.config)))
    if args.summary_only:
        report = {
            "schema_version": report.get("schema_version"),
            "run_id": report.get("run_id"),
            "overall_status": report.get("overall_status"),
            "partial_failure": report.get("partial_failure"),
            "failed_required_steps": report.get("failed_required_steps"),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
