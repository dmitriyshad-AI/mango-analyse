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

from mango_mvp.productization.amo_resolution_pipeline import (  # noqa: E402
    DEFAULT_AUDIT_INBOX,
    DEFAULT_OUT_ROOT,
    DEFAULT_PACK_ROOT,
    DEFAULT_WORKBOOK,
    AmoResolutionAfterXlsxConfig,
    run_amo_resolution_after_xlsx_pipeline,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_amo_resolution_after_xlsx_pipeline(
        AmoResolutionAfterXlsxConfig(
            project_root=ROOT,
            pack_root=Path(args.pack_root),
            workbook_path=Path(args.workbook),
            out_root=Path(args.out_root),
            audit_pack_root=Path(args.audit_pack_root),
            run_quality_gate=not args.skip_quality_gate,
            update_operator_status=not args.skip_operator_status,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["manual_resolution_summary"].get("validation_ok") else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fail-closed AMO manual-resolution pipeline after XLSX review.")
    parser.add_argument("--pack-root", default=str(DEFAULT_PACK_ROOT))
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--audit-pack-root", default=str(DEFAULT_AUDIT_INBOX))
    parser.add_argument("--skip-quality-gate", action="store_true", help="Do not run CRM quality gate even if candidates exist.")
    parser.add_argument("--skip-operator-status", action="store_true", help="Do not refresh operator-status artifacts.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
