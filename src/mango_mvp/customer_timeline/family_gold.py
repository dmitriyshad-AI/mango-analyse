from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path


FAMILY_GOLD_CHECK_SCHEMA_VERSION = "family_gold_check_v1"


@dataclass(frozen=True)
class FamilyGoldCheckConfig:
    timeline_db: Path
    gold_jsonl: Path
    allowed_root: Path
    out_json: Path | None = None
    tenant_id: str = "foton"


def check_family_gold(config: FamilyGoldCheckConfig) -> Mapping[str, Any]:
    gold_rows = _read_gold_rows(config.gold_jsonl)
    with _connect_ro(config.timeline_db) as con:
        cases = [_check_case(con, row, tenant_id=config.tenant_id) for row in gold_rows]
        quick_check = con.execute("PRAGMA quick_check").fetchone()[0]
    count_mismatches = [case for case in cases if not case["count_ok"]]
    unflagged_mismatches = [case for case in count_mismatches if not case["flags"]]
    false_high_cases = [case for case in cases if case["false_high"]]
    flagged_cases = [case for case in cases if case["flags"]]
    summary = {
        "schema_version": FAMILY_GOLD_CHECK_SCHEMA_VERSION,
        "tenant_id": config.tenant_id,
        "gold_rows": len(gold_rows),
        "exact_count_ok": sum(1 for case in cases if case["count_ok"]),
        "count_mismatches": len(count_mismatches),
        "unflagged_count_mismatches": len(unflagged_mismatches),
        "flagged_count_mismatches": len(count_mismatches) - len(unflagged_mismatches),
        "false_high": len(false_high_cases),
        "flagged_cases": len(flagged_cases),
        "quick_check": quick_check,
        "strict_pass": not unflagged_mismatches and not false_high_cases,
        "architect_review_required": bool(unflagged_mismatches),
        "flag_keys_seen": sorted({key for case in cases for key in case["flag_keys"]}),
        "parent_name_among_children_rows": sum(
            1 for case in cases if "parent_name_among_children" in set(case["flag_keys"])
        ),
        "cases": cases,
    }
    if config.out_json:
        out = guard_customer_timeline_output_path(config.out_json, config.allowed_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _check_case(con: sqlite3.Connection, gold: Mapping[str, Any], *, tenant_id: str) -> Mapping[str, Any]:
    customer_id = str(gold["customer_id"])
    flags = tuple(str(item) for item in gold.get("flags") or ())
    flag_keys = tuple(_flag_key(item) for item in flags)
    expected_count = int(gold["expected_children_count"])
    expected_confidence = str(gold.get("expected_link_confidence") or "")
    actual_rows = con.execute(
        """
        SELECT canonical_name, status, confidence, name_variants_json, record_json
        FROM family_links_v1
        WHERE tenant_id = ? AND customer_id = ?
        ORDER BY canonical_name
        """,
        (tenant_id, customer_id),
    ).fetchall()
    actual = [_actual_child(row) for row in actual_rows]
    kept = [item for item in actual if item["status"] != "excluded"]
    false_high = any(item["confidence"] == "high" for item in kept) and expected_confidence != "high"
    return {
        "customer_id": customer_id,
        "expected_children_count": expected_count,
        "actual_children_count": len(kept),
        "count_ok": len(kept) == expected_count,
        "expected_link_confidence": expected_confidence,
        "actual_confidence_counts": _counts(item["confidence"] for item in kept),
        "false_high": false_high,
        "flags": list(flags),
        "flag_keys": list(flag_keys),
        "needs_architect_review": len(kept) != expected_count and not flags,
        "actual_children": actual,
    }


def _actual_child(row: sqlite3.Row) -> Mapping[str, Any]:
    payload = _safe_json(row["record_json"])
    suspicious = payload.get("suspicious_reasons") if isinstance(payload, Mapping) else []
    return {
        "canonical_name": str(row["canonical_name"] or ""),
        "status": str(row["status"] or ""),
        "confidence": str(row["confidence"] or ""),
        "name_variants": _safe_json(row["name_variants_json"], default=[]),
        "suspicious_reasons": suspicious if isinstance(suspicious, list) else [],
    }


def _read_gold_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"gold row {line_no} must be a JSON object")
        if "customer_id" not in payload or "expected_children_count" not in payload:
            raise ValueError(f"gold row {line_no} misses required fields")
        rows.append(payload)
    return rows


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"{Path(path).resolve(strict=False).as_uri()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def _safe_json(value: str | None, default: Any | None = None) -> Any:
    fallback = {} if default is None else default
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _flag_key(value: str) -> str:
    return value.split(":", 1)[0].strip()


def _counts(values: Sequence[str]) -> Mapping[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))
