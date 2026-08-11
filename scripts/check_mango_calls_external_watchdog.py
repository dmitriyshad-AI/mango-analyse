#!/usr/bin/env python3
"""Validate a read-only Mango Calls observation collected outside both Macs."""
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

from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    EXTERNAL_WATCHDOG_VERDICT_SCHEMA,
    load_owner_only_json,
    validate_external_watchdog_observation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate safe external Mango Calls watchdog evidence."
    )
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--expected-active-host-id", required=True)
    parser.add_argument("--expected-previous-host-id", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-cutover-manifest-sha256", required=True)
    parser.add_argument("--expected-previous-host-snapshot-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        observation = load_owner_only_json(
            args.observation, label="external_watchdog_observation"
        )
        report = validate_external_watchdog_observation(
            observation,
            expected_active_host_id=args.expected_active_host_id,
            expected_previous_host_id=args.expected_previous_host_id,
            expected_code_sha=args.expected_code_sha,
            expected_cutover_manifest_sha256=(
                args.expected_cutover_manifest_sha256
            ),
            expected_previous_host_snapshot_sha256=(
                args.expected_previous_host_snapshot_sha256
            ),
        )
    except Exception as exc:  # noqa: BLE001 - safe CLI boundary
        report = {
            "schema_version": EXTERNAL_WATCHDOG_VERDICT_SCHEMA,
            "status": "alert",
            "ok": False,
            "errors": [f"watchdog_input_error:{type(exc).__name__}"],
            "safety": {
                "read_only_observation": True,
                "runs_asr": False,
                "runs_resolve_analyze": False,
                "writes_external_systems": False,
            },
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
