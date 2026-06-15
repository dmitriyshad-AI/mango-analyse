#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mango_mvp.channels.action_decision_judge import evaluate_action_gold_rows


DEFAULT_GOLD = Path("product_data/telegram_dynamic_test_sets/action_decision_judge_gold_20260614.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate deterministic action-decision judge on manual gold rows.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args(argv)
    rows = load_gold_rows(args.gold)
    report = evaluate_action_gold_rows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("accepted") else 2


def load_gold_rows(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError("Gold JSON must be a list")
        return [item for item in payload if isinstance(item, Mapping)]
    rows: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Gold row {line_no} must be an object")
        rows.append(payload)
    return rows


if __name__ == "__main__":
    sys.exit(main())
