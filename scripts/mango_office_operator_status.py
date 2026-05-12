#!/usr/bin/env python3
"""Build the read-only operator status pack for the current runtime."""

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

from mango_mvp.productization.operator_status import DEFAULT_OPERATOR_STATUS_ROOT, build_operator_status  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        status = build_operator_status(
            project_root=Path(args.project_root),
            runtime_contract_path=Path(args.runtime_contract) if args.runtime_contract else None,
            out_root=Path(args.out_root),
        )
    except Exception as exc:
        print(f"operator status failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out_root": str(Path(args.out_root).resolve(strict=False)), "summary": status["summary"], "next_actions": status["amo_production_loop"]["next_operator_actions"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status["summary"].get("runtime_validation_ok") else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only operator status, CRM queue and dashboard artifacts.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runtime-contract")
    parser.add_argument("--out-root", default=str(DEFAULT_OPERATOR_STATUS_ROOT))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
