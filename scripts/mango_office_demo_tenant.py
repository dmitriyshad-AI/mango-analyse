#!/usr/bin/env python3
"""Build an anonymized demo product appliance root.

Creates fake product DB rows, CRM snapshot, readiness reports, and dashboard
inputs under a separate product root. It never touches runtime DB/audio,
stable_runtime, live CRM, or ASR/R+A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/demo_product_appliance"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_demo_tenant_product_root(
            product_root=Path(args.product_root),
            out_path=Path(args.out),
            replace_existing=args.replace,
        )
    except Exception as exc:
        print(f"Demo tenant build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "demo_commands": report["demo_commands"],
                "safety": compact_safety(report["safety"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok", False) else 1


def compact_safety(safety: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "fake_data_only": safety.get("fake_data_only"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
        "stable_runtime_writes": safety.get("stable_runtime_writes"),
        "live_crm_reads": safety.get("live_crm_reads"),
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anonymized demo product appliance root.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--out", default=f"{DEFAULT_PRODUCT_ROOT}/demo_tenant_report.json")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
