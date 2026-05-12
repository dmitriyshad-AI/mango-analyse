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

from mango_mvp.productization.amo_resolution_workbook import (  # noqa: E402
    DEFAULT_AMO_BASE_URL,
    build_amo_resolution_review_html,
    build_amo_resolution_review_workbook,
    export_decisions_from_amo_resolution_workbook,
)


DEFAULT_PACK_ROOT = "stable_runtime/amo_manual_resolution_20260511_v1"
DEFAULT_XLSX = f"{DEFAULT_PACK_ROOT}/resolution_decisions_manual_template.xlsx"
DEFAULT_CSV = f"{DEFAULT_PACK_ROOT}/resolution_decisions_from_xlsx.csv"
DEFAULT_HTML = f"{DEFAULT_PACK_ROOT}/resolution_review_operator.html"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        result = build_amo_resolution_review_workbook(
            pack_root=Path(args.pack_root),
            out_xlsx=Path(args.out_xlsx),
            amo_base_url=args.amo_base_url,
        )
    elif args.command == "html":
        result = build_amo_resolution_review_html(
            pack_root=Path(args.pack_root),
            out_html=Path(args.out_html),
            amo_base_url=args.amo_base_url,
        )
    elif args.command == "build-all":
        workbook = build_amo_resolution_review_workbook(
            pack_root=Path(args.pack_root),
            out_xlsx=Path(args.out_xlsx),
            amo_base_url=args.amo_base_url,
        )
        html = build_amo_resolution_review_html(
            pack_root=Path(args.pack_root),
            out_html=Path(args.out_html),
            amo_base_url=args.amo_base_url,
        )
        result = {"workbook": workbook, "html": html}
    else:
        result = export_decisions_from_amo_resolution_workbook(
            workbook_path=Path(args.workbook),
            out_csv=Path(args.out_csv),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/convert AMO manual-resolution review workbook.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build operator-friendly XLSX from manual-resolution pack.")
    build.add_argument("--pack-root", default=DEFAULT_PACK_ROOT)
    build.add_argument("--out-xlsx", default=DEFAULT_XLSX)
    build.add_argument("--amo-base-url", default=DEFAULT_AMO_BASE_URL)

    html = sub.add_parser("html", help="Build read-only HTML review page from manual-resolution pack.")
    html.add_argument("--pack-root", default=DEFAULT_PACK_ROOT)
    html.add_argument("--out-html", default=DEFAULT_HTML)
    html.add_argument("--amo-base-url", default=DEFAULT_AMO_BASE_URL)

    build_all = sub.add_parser("build-all", help="Build XLSX and read-only HTML review assets.")
    build_all.add_argument("--pack-root", default=DEFAULT_PACK_ROOT)
    build_all.add_argument("--out-xlsx", default=DEFAULT_XLSX)
    build_all.add_argument("--out-html", default=DEFAULT_HTML)
    build_all.add_argument("--amo-base-url", default=DEFAULT_AMO_BASE_URL)

    convert = sub.add_parser("convert", help="Convert filled XLSX back to decisions CSV.")
    convert.add_argument("--workbook", default=DEFAULT_XLSX)
    convert.add_argument("--out-csv", default=DEFAULT_CSV)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
