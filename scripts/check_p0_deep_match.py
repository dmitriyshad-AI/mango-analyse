#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from mango_mvp.channels import p0_recall_spec
from mango_mvp.channels.p0_recall_spec import codes_from_text


DEEP_MATCH_ENV = "TELEGRAM_P0_DEEP_MATCH"
DEFAULT_SCENARIOS = Path("product_data/telegram_dynamic_test_sets/p0_deep_match_tz147_20260618.jsonl")


def _truthy_codes(codes: Sequence[str]) -> str:
    return ",".join(codes) if codes else "-"


def _load_rows(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        data = json.loads(line)
        if data.get("type") != "persona":
            continue
        rows.append(data)
    return rows


def _scenario_text(row: Mapping[str, object]) -> str:
    behaviors = row.get("behaviors")
    if isinstance(behaviors, list):
        return " ".join(str(item).strip() for item in behaviors if str(item).strip())
    return str(row.get("text") or row.get("message") or "")


def _is_positive(row: Mapping[str, object]) -> bool:
    target = str(row.get("flag_target") or "").casefold()
    if "tz147_p0_deep_pos" in target:
        return True
    if "tz147_p0_deep_neg" in target:
        return False
    return bool(row.get("injected_p0"))


def _codes_with_env(text: str, *, enabled: bool) -> tuple[str, ...]:
    previous = os.environ.get(DEEP_MATCH_ENV)
    try:
        if enabled:
            os.environ[DEEP_MATCH_ENV] = "1"
        else:
            os.environ.pop(DEEP_MATCH_ENV, None)
        return tuple(codes_from_text(text))
    finally:
        if previous is None:
            os.environ.pop(DEEP_MATCH_ENV, None)
        else:
            os.environ[DEEP_MATCH_ENV] = previous


def _shorten(text: str, limit: int = 72) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _print_table(rows: Iterable[Mapping[str, object]]) -> tuple[int, int, int, int]:
    pos_total = pos_off = pos_on = neg_total = neg_fp_off = neg_fp_on = 0
    print("| # | label | dialog_id | OFF codes | ON codes | text |")
    print("|---:|---|---|---|---|---|")
    for idx, row in enumerate(rows, start=1):
        text = _scenario_text(row)
        off_codes = _codes_with_env(text, enabled=False)
        on_codes = _codes_with_env(text, enabled=True)
        is_pos = _is_positive(row)
        label = "POS" if is_pos else "NEG"
        if is_pos:
            pos_total += 1
            if "payment_dispute" in off_codes:
                pos_off += 1
            if "payment_dispute" in on_codes:
                pos_on += 1
        else:
            neg_total += 1
            if off_codes:
                neg_fp_off += 1
            if on_codes:
                neg_fp_on += 1
        print(
            f"| {idx} | {label} | {row.get('dialog_id', '')} | "
            f"{_truthy_codes(off_codes)} | {_truthy_codes(on_codes)} | {_shorten(text)} |"
        )
    print()
    print(f"POS recall: OFF={pos_off}/{pos_total}, ON={pos_on}/{pos_total}, delta={pos_on - pos_off}")
    print(f"NEG false positives: OFF={neg_fp_off}/{neg_total}, ON={neg_fp_on}/{neg_total}, delta={neg_fp_on - neg_fp_off}")
    return pos_total, pos_on, neg_fp_off, neg_fp_on


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic OFF/ON check for tz147 P0 deep-match.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--strict", action="store_true", help="Fail if ON misses POS rows or adds NEG false positives.")
    args = parser.parse_args()

    if not args.scenarios.exists():
        raise SystemExit(f"Scenario file not found: {args.scenarios}")

    rows = _load_rows(args.scenarios)
    if not rows:
        raise SystemExit(f"No persona rows found in {args.scenarios}")

    deep_flag_supported = hasattr(p0_recall_spec, "P0_DEEP_MATCH_ENV")
    if not deep_flag_supported:
        print(
            "WARNING: current p0_recall_spec has no P0_DEEP_MATCH_ENV; "
            "OFF/ON may be identical until tz147 code is ported."
        )
        print()

    pos_total, pos_on, neg_fp_off, neg_fp_on = _print_table(rows)
    if args.strict and (pos_on < pos_total or neg_fp_on > neg_fp_off):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
