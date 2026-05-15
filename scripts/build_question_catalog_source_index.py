#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.question_catalog.source_index import build_source_index_rows, write_source_index  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build call_id -> question catalog theme/policy index.")
    parser.add_argument("--items-jsonl", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    items = read_jsonl(Path(args.items_jsonl).expanduser().resolve())
    rows = build_source_index_rows(items)
    summary = write_source_index(Path(args.out_root).expanduser().resolve(), rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
