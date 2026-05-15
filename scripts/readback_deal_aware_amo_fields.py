#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ENV_FILES = (
    ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    ROOT / "prod_runtime_transfer" / ".env.private",
)


def _load_env_files_early() -> None:
    import os

    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{(ROOT / 'stable_runtime' / 'amocrm_runtime' / 'amo_runtime.db').resolve()}")


_load_env_files_early()

from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS  # noqa: E402
from mango_mvp.deal_aware.stage1_snapshot import safe_text, write_csv  # noqa: E402
from mango_mvp.quality.crm_text_quality_detector import (  # noqa: E402
    CrmTextQualityFinding,
    detect_crm_text_quality_risks,
    has_blocking_crm_text_findings,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read back deal-aware AMO lead AI fields and verify values.")
    parser.add_argument("--writeback-report", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--statuses", default="written")
    parser.add_argument("--expected-evaluated", type=int, default=None)
    parser.add_argument("--min-severity", default="P2")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.writeback_report).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = read_report(report_path)
    statuses = set(split_tokens(args.statuses)) or {"written"}

    from scripts.write_amo_ready_contacts import _load_env_files, _preflight_runtime_db  # noqa: PLC0415

    _load_env_files()
    from mango_mvp.amocrm_runtime.amo_integration import fetch_lead  # noqa: PLC0415
    from mango_mvp.amocrm_runtime.db import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        ok, error = _preflight_runtime_db(session)
        if not ok:
            summary = build_summary(report_path, out_root, rows, [], statuses, False, True, error, args.expected_evaluated)
            write_outputs(out_root, [], summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2

        def fetcher(lead_id: int) -> dict[str, Any]:
            return fetch_lead(session, lead_id=lead_id)

        result_rows = evaluate_rows(rows, fetch_lead=fetcher, statuses=statuses, min_severity=args.min_severity)
    finally:
        session.close()

    summary = build_summary(report_path, out_root, rows, result_rows, statuses, True, False, "", args.expected_evaluated)
    write_outputs(out_root, result_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    fetch_lead,
    statuses: set[str],
    min_severity: str,
) -> list[dict[str, Any]]:
    result = []
    for source in rows:
        row_index = safe_text(source.get("row_index"))
        status = safe_text(source.get("status"))
        lead_id_raw = safe_text(source.get("lead_id"))
        report_row: dict[str, Any] = {
            "row_index": row_index,
            "source_status": status,
            "lead_id": lead_id_raw,
            "review_id": safe_text(source.get("review_id")),
            "readback_status": "",
            "decision": "",
            "risk_types": "",
            "finding_count": 0,
            "findings_json": "[]",
        }
        if status not in statuses:
            report_row["readback_status"] = "skipped"
            report_row["decision"] = "skip"
            report_row["risk_types"] = "source_status_not_selected"
            result.append(report_row)
            continue
        try:
            lead_id = int(lead_id_raw)
        except ValueError:
            report_row["readback_status"] = "failed"
            report_row["decision"] = "block"
            report_row["risk_types"] = "missing_lead_id"
            result.append(report_row)
            continue
        try:
            lead = fetch_lead(lead_id)
            values = extract_custom_field_values(lead)
            payload = {field: values.get(field, "") for field in DEAL_AI_FIELDS}
            expected = expected_payload(source)
            findings = readback_findings(payload, expected, min_severity=min_severity)
            report_row["readback_status"] = "evaluated"
            report_row["decision"] = "block" if has_blocking_crm_text_findings(findings) else "allow"
            report_row["risk_types"] = " | ".join(sorted({item.risk_type for item in findings}))
            report_row["finding_count"] = len(findings)
            report_row["findings_json"] = json.dumps([finding_to_dict(item) for item in findings], ensure_ascii=False)
            for field in DEAL_AI_FIELDS:
                report_row[f"field::{field}"] = payload.get(field, "")
        except Exception as exc:  # noqa: BLE001
            report_row["readback_status"] = "failed"
            report_row["decision"] = "block"
            report_row["risk_types"] = "readback_fetch_failed"
            report_row["finding_count"] = 1
            report_row["findings_json"] = json.dumps([{"risk_type": "readback_fetch_failed", "severity": "P1", "reason": str(exc)}], ensure_ascii=False)
        result.append(report_row)
    return result


def readback_findings(payload: dict[str, str], expected: dict[str, str], *, min_severity: str) -> list[CrmTextQualityFinding]:
    findings = detect_crm_text_quality_risks(payload, min_severity=min_severity)
    for field in DEAL_AI_FIELDS:
        actual = safe_text(payload.get(field))
        expected_value = safe_text(expected.get(field))
        if expected_value and not values_equivalent(actual, expected_value):
            findings.append(
                CrmTextQualityFinding(
                    class_id="Q6",
                    risk_type="readback_value_mismatch",
                    severity="P1",
                    field=field,
                    matched_text=actual[:160],
                    reason="amoCRM lead readback differs from expected deal-aware payload",
                )
            )
        if not actual:
            findings.append(
                CrmTextQualityFinding(
                    class_id="Q6",
                    risk_type="missing_readback_target_field",
                    severity="P1",
                    field=field,
                    matched_text="",
                    reason="amoCRM readback did not return an expected deal-aware AI field",
                )
            )
    return findings


def values_equivalent(actual: str, expected: str) -> bool:
    actual_text = safe_text(actual)
    expected_text = safe_text(expected)
    if actual_text == expected_text:
        return True
    actual_ts = parse_timestamp_like(actual_text)
    expected_ts = parse_timestamp_like(expected_text)
    if actual_ts is not None and expected_ts is not None:
        return actual_ts == expected_ts
    return False


def parse_timestamp_like(value: str) -> int | None:
    text = safe_text(value)
    if not text:
        return None
    if text.isdigit():
        return int(text)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None  # type: ignore[assignment]
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp())


def expected_payload(source: dict[str, Any]) -> dict[str, str]:
    raw = source.get("preview_payload") or ""
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(safe_text(raw) or "{}")
        except json.JSONDecodeError:
            payload = {}
    return {field: safe_text(payload.get(field)) for field in DEAL_AI_FIELDS if safe_text(payload.get(field))}


def extract_custom_field_values(entity: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in entity.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        field_name = safe_text(item.get("field_name"))
        values = []
        for value_item in item.get("values") or []:
            if isinstance(value_item, dict) and safe_text(value_item.get("value")):
                values.append(safe_text(value_item.get("value")))
        if field_name:
            result[field_name] = " | ".join(values)
    return result


def build_summary(
    input_path: Path,
    out_root: Path,
    source_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    statuses: set[str],
    preflight_ok: bool,
    preflight_failed: bool,
    preflight_error: str,
    expected_evaluated: int | None,
) -> dict[str, Any]:
    selected = sum(1 for row in result_rows if row.get("source_status") in statuses)
    evaluated = sum(1 for row in result_rows if row.get("readback_status") == "evaluated")
    blocking = sum(1 for row in result_rows if row.get("decision") == "block")
    failed = sum(1 for row in result_rows if row.get("readback_status") == "failed")
    mismatch = expected_evaluated is not None and evaluated != expected_evaluated
    passed = preflight_ok and selected > 0 and evaluated == selected and blocking == 0 and failed == 0 and not mismatch
    risks = Counter(risk for row in result_rows for risk in split_tokens(row.get("risk_types")))
    return {
        "schema_version": "deal_aware_stage6_readback_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(input_path),
        "passed": passed,
        "total_source_rows": len(source_rows),
        "selected_source_rows": selected,
        "evaluated_rows": evaluated,
        "blocking_rows": blocking,
        "failed_rows": failed,
        "expected_evaluated": expected_evaluated,
        "expected_count_mismatch": mismatch,
        "preflight_failed": preflight_failed,
        "preflight_error": preflight_error,
        "risk_counts": dict(risks.most_common()),
        "target_fields": list(DEAL_AI_FIELDS),
        "outputs": {
            "report_json": str(out_root / "readback_report.json"),
            "report_csv": str(out_root / "readback_report.csv"),
            "summary_json": str(out_root / "summary.json"),
        },
    }


def write_outputs(out_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_csv(out_root / "readback_report.csv", rows)
    (out_root / "readback_report.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def read_report(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else []
        return [row for row in rows if isinstance(row, dict)]
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def split_tokens(value: Any) -> list[str]:
    import re

    text = safe_text(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"\s*(?:\||,|;)\s*", text) if part.strip()]


def finding_to_dict(finding: CrmTextQualityFinding) -> dict[str, Any]:
    return {
        "class_id": finding.class_id,
        "risk_type": finding.risk_type,
        "severity": finding.severity,
        "field": finding.field,
        "matched_text": finding.matched_text,
        "reason": finding.reason,
    }


if __name__ == "__main__":
    raise SystemExit(main())
