#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from mango_mvp.customer_timeline.preview_quality_audit import (
    build_preview_quality_audit,
    render_preview_quality_audit_markdown,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_preview_quality_audit(
            project_root=Path(args.project_root),
            telegram_export_dir=Path(args.telegram_export_dir) if args.telegram_export_dir else None,
            real_pair_limit=args.real_pair_limit,
        )
        if args.out_json:
            out_json = Path(args.out_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.out_md:
            out_md = Path(args.out_md)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(render_preview_quality_audit_markdown(report), encoding="utf-8")
        if not args.out_json and not args.out_md:
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - compact CLI-facing error.
        print(f"customer timeline preview quality audit failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit for Customer Timeline channel draft previews against synthetic and Telegram messages."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--telegram-export-dir")
    parser.add_argument("--real-pair-limit", type=int, default=100)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
