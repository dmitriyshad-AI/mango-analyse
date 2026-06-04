#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "kb_release_v3_snapshot.json",
    "client_safe_facts_foton.jsonl",
    "client_safe_facts_unpk.jsonl",
    "manager_only_or_internal_facts.jsonl",
    "quality_report.json",
    "semantic_review.json",
)


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _truthy_gate(data: dict, key: str) -> bool:
    if key in data:
        return bool(data.get(key))
    gates = data.get("gates")
    if isinstance(gates, dict) and key in gates:
        return bool(gates.get(key))
    checks = data.get("checks")
    if isinstance(checks, dict) and key in checks:
        return bool(checks.get(key))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Mango KB v6 release gates.")
    parser.add_argument("release_dir")
    args = parser.parse_args()

    release = Path(args.release_dir).resolve()
    missing = [name for name in REQUIRED_FILES if not (release / name).exists()]
    quality = _load_json(release / "quality_report.json")
    semantic = _load_json(release / "semantic_review.json")

    summary = {
        "release_dir": str(release),
        "missing_files": missing,
        "quality_passed": bool(quality.get("quality_passed")),
        "semantic_pass": bool(semantic.get("semantic_pass")),
        "text_number_grounded": _truthy_gate(quality, "text_number_grounded"),
        "field_ranges_ok": _truthy_gate(quality, "field_ranges_ok"),
        "weekly_frequency_is_plausible": _truthy_gate(quality, "weekly_frequency_is_plausible"),
        "control_numbers_present": _truthy_gate(quality, "control_numbers_present"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if missing or not summary["quality_passed"] or not summary["semantic_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
