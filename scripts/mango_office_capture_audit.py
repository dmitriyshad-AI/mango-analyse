#!/usr/bin/env python3
"""Audit a Mango capture staging manifest without touching runtime DBs."""

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

from mango_mvp.productization.capture_staging import audit_capture_manifest  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    audit = audit_capture_manifest(
        manifest_path=Path(args.manifest),
        recordings_dir=Path(args.recordings_dir) if args.recordings_dir else None,
    )
    text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    has_errors = any(
        audit.get(key, 0)
        for key in ("missing_files", "zero_size_files", "checksum_missing", "duration_missing")
    )
    return 1 if has_errors else 0


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a Mango capture staging manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--recordings-dir")
    parser.add_argument("--out")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
