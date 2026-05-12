#!/usr/bin/env python3
"""Build a demo/pilot playbook for the product appliance."""

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

from mango_mvp.productization.demo_pilot_playbook import build_demo_pilot_playbook  # noqa: E402


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_PRODUCT_DB = f"{DEFAULT_PRODUCT_ROOT}/mango_product_appliance.sqlite"
DEFAULT_OUT_DIR = f"{DEFAULT_PRODUCT_ROOT}/demo_pilot_playbook"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_demo_pilot_playbook(
            product_root=Path(args.product_root),
            product_db_path=Path(args.product_db),
            out_dir=Path(args.out_dir),
        )
    except Exception as exc:
        print(f"Demo pilot playbook failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "out_dir": str(Path(args.out_dir).resolve(strict=False)),
                "summary": report["summary"],
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
        "read_only": safety.get("read_only"),
        "write_crm": safety.get("write_crm"),
        "run_asr": safety.get("run_asr"),
        "runtime_db_writes": safety.get("runtime_db_writes"),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build demo/pilot playbook.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--product-db", default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
