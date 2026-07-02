#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.customer_timeline.mail_stage2_visibility import (
    assert_mail_stage2_visibility_gate,
    harden_mail_stage2_bot_visibility,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Close mail_archive_stage2 chunks to manager-only in staging.")
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = harden_mail_stage2_bot_visibility(
        args.timeline_db,
        allowed_root=args.allowed_root,
        apply=args.apply,
    )
    if args.apply:
        assert_mail_stage2_visibility_gate(
            args.timeline_db,
            allowed_root=args.allowed_root,
        )
    out_path = guard_customer_timeline_output_path(args.out.expanduser(), args.allowed_root.expanduser())
    if "customer_timeline_prod_" in str(out_path):
        raise ValueError(f"refusing to write hardening report to prod-like path: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
