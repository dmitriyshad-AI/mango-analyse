#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mango_mvp.quality.crm_text_quality_detector import (
    CrmTextQualityFinding,
    detect_crm_text_quality_risks,
    has_blocking_crm_text_findings,
)
from scripts.write_amo_ready_contacts import (
    REPORT_ROOT,
    TARGET_CONTACT_FIELDS,
    _load_env_files,
    _preflight_runtime_db,
    _safe_text,
    _write_report_csv,
    _write_report_xlsx,
)


ReadbackFetcher = Callable[[int], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read back written amoCRM contact AI fields and run CRM text quality gate."
    )
    parser.add_argument("--writeback-report", required=True, help="contact_writeback_report.csv/json from live writeback.")
    parser.add_argument("--out-root", default="", help="Output folder. Defaults to contact_writebacks/readbacks/<run_id>.")
    parser.add_argument("--min-severity", default="P2")
    parser.add_argument(
        "--statuses",
        default="written",
        help="Pipe/comma/space-separated source report statuses to read back. Default: written.",
    )
    parser.add_argument("--expected-evaluated", type=int, default=None, help="Fail if evaluated row count differs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = Path(args.writeback_report).expanduser().resolve()
    rows = _read_writeback_report(report_path)
    statuses = set(_split(args.statuses)) or {"written"}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else REPORT_ROOT / "readbacks" / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    _load_env_files()
    from mango_mvp.amocrm_runtime.amo_integration import fetch_contact
    from mango_mvp.amocrm_runtime.db import SessionLocal

    session = SessionLocal()
    try:
        ok, error = _preflight_runtime_db(session)
        if not ok:
            summary = _summary_payload(
                input_path=report_path,
                out_root=out_root,
                total_source_rows=len(rows),
                selected_source_rows=0,
                evaluated_rows=0,
                blocking_rows=0,
                failed_rows=0,
                skipped_rows=0,
                passed=False,
                preflight_failed=True,
                preflight_error=error,
                risk_counts={},
            )
            _write_outputs(out_root, [], summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2

        def fetcher(contact_id: int) -> dict[str, Any]:
            return fetch_contact(session, contact_id=contact_id)

        report_rows = evaluate_readback_rows(
            rows,
            fetch_contact=fetcher,
            statuses=statuses,
            min_severity=args.min_severity,
        )
    finally:
        session.close()

    risk_counts = Counter(
        risk
        for row in report_rows
        for risk in _split(row.get("risk_types", ""))
    )
    selected_source_rows = sum(1 for row in report_rows if row["source_status"] in statuses)
    evaluated_rows = sum(1 for row in report_rows if row["readback_status"] == "evaluated")
    blocking_rows = sum(1 for row in report_rows if row["decision"] == "block")
    failed_rows = sum(1 for row in report_rows if row["readback_status"] == "failed")
    expected_mismatch = args.expected_evaluated is not None and evaluated_rows != args.expected_evaluated
    passed = (
        selected_source_rows > 0
        and evaluated_rows == selected_source_rows
        and blocking_rows == 0
        and failed_rows == 0
        and not expected_mismatch
    )
    summary = _summary_payload(
        input_path=report_path,
        out_root=out_root,
        total_source_rows=len(rows),
        selected_source_rows=selected_source_rows,
        evaluated_rows=evaluated_rows,
        blocking_rows=blocking_rows,
        failed_rows=failed_rows,
        skipped_rows=sum(1 for row in report_rows if row["readback_status"] == "skipped"),
        passed=passed,
        preflight_failed=False,
        preflight_error="",
        expected_evaluated=args.expected_evaluated,
        expected_count_mismatch=expected_mismatch,
        risk_counts=dict(risk_counts.most_common()),
    )
    _write_outputs(out_root, report_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def evaluate_readback_rows(
    rows: Iterable[dict[str, Any]],
    *,
    fetch_contact: ReadbackFetcher,
    statuses: set[str] | None = None,
    target_fields: Iterable[str] = TARGET_CONTACT_FIELDS,
    min_severity: str = "P2",
) -> list[dict[str, Any]]:
    statuses = statuses or {"written"}
    target_fields = tuple(target_fields)
    result: list[dict[str, Any]] = []
    for source in rows:
        row_index = _safe_text(source.get("row_index"))
        source_status = _safe_text(source.get("status"))
        contact_id_raw = _safe_text(source.get("contact_id"))
        report_row: dict[str, Any] = {
            "row_index": row_index,
            "source_status": source_status,
            "phone": _safe_text(source.get("phone")),
            "contact_id": contact_id_raw,
            "contact_name": _safe_text(source.get("contact_name")),
            "readback_status": "",
            "decision": "",
            "risk_types": "",
            "finding_count": 0,
            "findings_json": "[]",
        }
        if source_status not in statuses:
            report_row["readback_status"] = "skipped"
            report_row["decision"] = "skip"
            report_row["risk_types"] = "source_status_not_selected"
            result.append(report_row)
            continue
        try:
            contact_id = int(contact_id_raw)
        except ValueError:
            report_row["readback_status"] = "failed"
            report_row["decision"] = "block"
            report_row["risk_types"] = "missing_contact_id"
            report_row["findings_json"] = json.dumps(
                [{"risk_type": "missing_contact_id", "severity": "P1", "field": "contact_id"}],
                ensure_ascii=False,
            )
            report_row["finding_count"] = 1
            result.append(report_row)
            continue

        try:
            contact = fetch_contact(contact_id)
            values = extract_custom_field_values(contact)
            payload = {field: values.get(field, "") for field in target_fields}
            expected_payload = _expected_payload_from_source(source, target_fields)
            findings = _readback_findings(
                payload,
                values,
                target_fields,
                min_severity=min_severity,
                expected_payload=expected_payload,
            )
            report_row["readback_status"] = "evaluated"
            report_row["decision"] = "block" if has_blocking_crm_text_findings(findings) else "allow"
            report_row["risk_types"] = " | ".join(sorted({finding.risk_type for finding in findings}))
            report_row["finding_count"] = len(findings)
            report_row["findings_json"] = json.dumps([_finding_to_dict(finding) for finding in findings], ensure_ascii=False)
            for field in target_fields:
                report_row[f"field::{field}"] = payload.get(field, "")
        except Exception as exc:
            report_row["readback_status"] = "failed"
            report_row["decision"] = "block"
            report_row["risk_types"] = "readback_fetch_failed"
            report_row["finding_count"] = 1
            report_row["findings_json"] = json.dumps(
                [{"risk_type": "readback_fetch_failed", "severity": "P1", "reason": str(exc)}],
                ensure_ascii=False,
            )
        result.append(report_row)
    return result


def extract_custom_field_values(contact: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in contact.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        field_name = _safe_text(item.get("field_name"))
        if not field_name:
            continue
        values: list[str] = []
        for value_item in item.get("values") or []:
            if not isinstance(value_item, dict):
                continue
            value = _safe_text(value_item.get("value"))
            if value:
                values.append(value)
        result[field_name] = " | ".join(values)
    return result


def _readback_findings(
    payload: dict[str, str],
    all_values: dict[str, str],
    target_fields: Iterable[str],
    *,
    min_severity: str,
    expected_payload: dict[str, str] | None = None,
) -> list[CrmTextQualityFinding]:
    findings = detect_crm_text_quality_risks(payload, min_severity=min_severity)
    expected_payload = expected_payload or {}
    for field, expected_value in expected_payload.items():
        actual_value = _safe_text(payload.get(field))
        if expected_value and actual_value != expected_value:
            findings.append(
                CrmTextQualityFinding(
                    class_id="Q6",
                    risk_type="readback_value_mismatch",
                    severity="P1",
                    field=field,
                    matched_text=actual_value[:160],
                    reason="amoCRM readback value differs from the payload recorded in the live writeback report",
                )
            )
    for field in target_fields:
        if field not in all_values:
            findings.append(
                CrmTextQualityFinding(
                    class_id="Q6",
                    risk_type="missing_readback_target_field",
                    severity="P1",
                    field=field,
                    matched_text="",
                    reason="amoCRM readback did not return an expected AI target field",
                )
            )
    return findings


def _expected_payload_from_source(source: dict[str, Any], target_fields: Iterable[str]) -> dict[str, str]:
    raw = source.get("preview_payload") or source.get("expected_payload") or ""
    if isinstance(raw, dict):
        payload = raw
    else:
        text = _safe_text(raw)
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        payload = parsed
    target_set = set(target_fields)
    return {field: _safe_text(value) for field, value in payload.items() if field in target_set}


def _read_writeback_report(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"JSON writeback report has no rows list: {path}")
        return [dict(row) for row in rows if isinstance(row, dict)]
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_outputs(out_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "readback_report.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report_csv(out_root / "readback_report.csv", rows)
    _write_report_xlsx(out_root / "readback_report.xlsx", rows)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_payload(
    *,
    input_path: Path,
    out_root: Path,
    total_source_rows: int,
    selected_source_rows: int,
    evaluated_rows: int,
    blocking_rows: int,
    failed_rows: int,
    skipped_rows: int,
    passed: bool,
    preflight_failed: bool,
    preflight_error: str,
    risk_counts: dict[str, int],
    expected_evaluated: int | None = None,
    expected_count_mismatch: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "amo_contact_readback_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "passed": bool(passed),
        "total_source_rows": total_source_rows,
        "selected_source_rows": selected_source_rows,
        "evaluated_rows": evaluated_rows,
        "blocking_rows": blocking_rows,
        "failed_rows": failed_rows,
        "skipped_rows": skipped_rows,
        "expected_evaluated": expected_evaluated,
        "expected_count_mismatch": expected_count_mismatch,
        "preflight_failed": preflight_failed,
        "preflight_error": preflight_error,
        "risk_counts": risk_counts,
        "target_fields": list(TARGET_CONTACT_FIELDS),
        "outputs": {
            "report_json": str(out_root / "readback_report.json"),
            "report_csv": str(out_root / "readback_report.csv"),
            "report_xlsx": str(out_root / "readback_report.xlsx"),
            "summary_json": str(out_root / "summary.json"),
        },
    }


def _finding_to_dict(finding: CrmTextQualityFinding) -> dict[str, Any]:
    return {
        "class_id": finding.class_id,
        "risk_type": finding.risk_type,
        "severity": finding.severity,
        "field": finding.field,
        "matched_text": finding.matched_text,
        "reason": finding.reason,
        "row_index": finding.row_index,
    }


def _split(value: str) -> list[str]:
    import re

    text = _safe_text(value).strip()
    if not text:
        return []
    if any(separator in text for separator in "|,;"):
        return [part.strip() for part in re.split(r"\s*(?:\||,|;)\s*", text) if part.strip()]
    return [part for part in re.split(r"\s+", text) if part]


if __name__ == "__main__":
    raise SystemExit(main())
