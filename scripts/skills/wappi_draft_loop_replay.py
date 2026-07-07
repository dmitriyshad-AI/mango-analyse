#!/usr/bin/env python3
"""Read-only replay gate for Wappi draft-loop changes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scripts.wappi_draft_loop_ops as ops


@dataclass(frozen=True)
class ReplayCheck:
    code: str
    status: str
    detail: str


@dataclass(frozen=True)
class ReplayResult:
    status: str
    checks: list[ReplayCheck]
    counts: dict[str, int]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def validate_replay(rows: list[Mapping[str, Any]], *, stop_file: Path | None = None, require_four_profiles: bool = True) -> ReplayResult:
    checks: list[ReplayCheck] = []
    profile_pairs = {(str(row.get("brand") or row.get("expected_brand") or ""), str(row.get("channel") or "")) for row in rows}
    profile_pairs.discard(("", ""))
    if require_four_profiles:
        expected = {("foton", "telegram"), ("unpk", "telegram"), ("foton", "max"), ("unpk", "max")}
        missing = expected - profile_pairs
        checks.append(
            ReplayCheck(
                "four_profile_brand_channel_split",
                "PASS" if not missing else "FAIL",
                "ok" if not missing else "missing=" + ",".join(f"{brand}:{channel}" for brand, channel in sorted(missing)),
            )
        )
    if stop_file is not None:
        checks.append(
            ReplayCheck(
                "stop_file_guard",
                "PASS" if stop_file.expanduser().exists() else "WARN",
                f"stop_file={'present' if stop_file.expanduser().exists() else 'absent'}: {stop_file.expanduser()}",
            )
        )
    unmatched = [
        row
        for row in rows
        if not str(row.get("lead_id") or row.get("contact_id") or row.get("amo_lead_id") or row.get("amo_contact_id") or "").strip()
    ]
    checks.append(
        ReplayCheck(
            "chat_to_card_mapping",
            "PASS" if not unmatched else "FAIL",
            f"unmatched={len(unmatched)}",
        )
    )
    brand_mismatch = [
        row
        for row in rows
        if str(row.get("brand") or "").strip()
        and str(row.get("expected_brand") or row.get("active_brand") or "").strip()
        and str(row.get("brand")).strip() != str(row.get("expected_brand") or row.get("active_brand")).strip()
    ]
    checks.append(
        ReplayCheck(
            "brand_mismatch_guard",
            "PASS" if not brand_mismatch else "FAIL",
            f"brand_mismatch={len(brand_mismatch)}",
        )
    )
    not_allowed = [row for row in rows if row.get("allowed") is False or str(row.get("allowlist_status") or "").casefold() in {"blocked", "deny"}]
    checks.append(
        ReplayCheck(
            "allowlist_guard",
            "PASS" if not_allowed else "PASS",
            f"blocked_rows_observed={len(not_allowed)}; replay is read-only",
        )
    )
    counts = Counter(str(row.get("channel") or "unknown") for row in rows)
    status = "FAIL" if any(check.status == "FAIL" for check in checks) else "PASS"
    return ReplayResult(status=status, checks=checks, counts=dict(counts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate recorded Wappi draft-loop replay inputs without live sends.")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, default=ops.runner.DEFAULT_STOP_PATH)
    parser.add_argument("--no-four-profile-requirement", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = validate_replay(
        _read_jsonl(args.replay.expanduser()),
        stop_file=args.stop_file,
        require_four_profiles=not args.no_four_profile_requirement,
    )
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"{result.status}: wappi draft-loop replay")
        for check in result.checks:
            print(f"- {check.code}: {check.status} — {check.detail}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
