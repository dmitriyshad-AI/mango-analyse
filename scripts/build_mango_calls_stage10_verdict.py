#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.capture_staging import CaptureManifestStore  # noqa: E402
from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    build_stage10_verdict,
    load_ready_rows,
    write_private_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the closed-balance verdict for one Moscow Mango calls day."
    )
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--enumeration", type=Path, required=True)
    parser.add_argument("--ready-db", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    enumeration = json.loads(args.enumeration.read_text(encoding="utf-8"))
    if not isinstance(enumeration, dict):
        raise RuntimeError("Mango enumeration evidence must be a JSON object")
    entries = CaptureManifestStore(args.capture_manifest).read_entries()
    report = build_stage10_verdict(
        day=args.day,
        enumeration=enumeration,
        capture_entries=entries,
        ready_rows=load_ready_rows(args.ready_db),
    )
    if args.out:
        write_private_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["consistency_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
