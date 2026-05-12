#!/usr/bin/env python3
"""Build or inspect the current runtime contract."""

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

from mango_mvp.productization.current_runtime import DEFAULT_CURRENT_RUNTIME_PATH, build_current_runtime_contract  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        contract = build_current_runtime_contract(
            project_root=Path(args.project_root),
            out_path=Path(args.out),
        )
    except Exception as exc:
        print(f"current runtime contract failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(Path(args.out).resolve(strict=False)), "summary": contract["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if contract["summary"].get("validation_ok") else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the read-only current runtime contract.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_CURRENT_RUNTIME_PATH))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
