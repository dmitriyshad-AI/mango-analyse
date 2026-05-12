#!/usr/bin/env python3
"""Build a read-only index of stable_runtime artifacts."""

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

from mango_mvp.productization.runtime_artifact_index import (  # noqa: E402
    DEFAULT_RUNTIME_ARTIFACT_INDEX_ROOT,
    build_runtime_artifact_index,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_runtime_artifact_index(
        project_root=Path(args.project_root),
        out_root=Path(args.out_root),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only stable_runtime artifact index.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-root", default=str(DEFAULT_RUNTIME_ARTIFACT_INDEX_ROOT))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
