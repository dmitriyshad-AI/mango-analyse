#!/usr/bin/env python3
"""Build the 9-stage SaaS/productization gate report.

The report is read-only over existing productization state and writes only its
own JSON audit under the product appliance root. It never downloads audio, runs
ASR/R+A, writes runtime DBs, or writes CRM/Tallanto.
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

from mango_mvp.productization.saas_stage_gates import build_saas_stage_gates_report  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT = f"{DEFAULT_PRODUCT_ROOT}/saas_stage_gates_20260508/saas_stage_gates_report.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_saas_stage_gates_report(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            out_path=Path(args.out),
            workspace_root=ROOT,
        )
    except Exception as exc:
        print(f"SaaS stage gates report failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve(strict=False)),
                "summary": report["summary"],
                "status_counts": report["status_counts"],
                "stage_blockers": {
                    stage["key"]: stage["blockers"]
                    for stage in report["stages"]
                    if stage["blockers"]
                },
                "safety": report["safety"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"].get("validation_ok") else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 9-stage SaaS/productization gate report.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
