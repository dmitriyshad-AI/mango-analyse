from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.amocrm_runtime.amo_integration import build_custom_fields_values
from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS
from mango_mvp.deal_aware.stage1_snapshot import quote_ident, read_csv, safe_text, stringify, write_csv
from mango_mvp.quality.crm_text_quality_detector import detect_crm_text_quality_risks


SCHEMA_VERSION = "deal_aware_stage6_writeback_preflight_v1"
LIVE_CONFIRMATION = "WRITE_AMO_DEAL_AWARE_LIVE"
PROTECTED_LEAD_FIELDS = {
    "Id Tallanto",
    "Филиал Tallanto",
    "Телефон",
    "Телефон клиента",
    "ФИО",
    "Email",
}


@dataclass(frozen=True)
class DealAwareStage6Paths:
    input_csv: Path
    stage5_summary_json: Path
    field_catalog_cache_json: Path
    out_root: Path
    analysis_date: str = "2026-05-13"
    stage20_size: int = 20


def run_deal_aware_stage6_preflight(paths: DealAwareStage6Paths) -> dict[str, Any]:
    paths.out_root.mkdir(parents=True, exist_ok=True)
    input_rows = read_csv(paths.input_csv)
    stage5_summary = load_json(paths.stage5_summary_json)
    field_catalog_payload = load_json(paths.field_catalog_cache_json)
    field_catalog = field_catalog_payload.get("fields") if isinstance(field_catalog_payload.get("fields"), list) else []

    field_guard = validate_field_catalog(field_catalog)
    stage5_guard = validate_stage5_summary(stage5_summary, paths.input_csv)

    report_rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for index, row in enumerate(input_rows, start=1):
        report_row, row_findings = build_dry_run_row(
            row,
            row_index=index,
            field_catalog=field_catalog,
            field_guard=field_guard,
            analysis_date=paths.analysis_date,
        )
        report_rows.append(report_row)
        findings.extend(row_findings)

    stage20_candidates = choose_stage20_candidates(report_rows, limit=paths.stage20_size)
    outputs = {
        "dry_run_report_csv": paths.out_root / "deal_stage6_dry_run_report.csv",
        "stage20_candidates_csv": paths.out_root / "deal_stage6_stage20_candidates.csv",
        "findings_csv": paths.out_root / "deal_stage6_findings.csv",
        "sqlite": paths.out_root / "deal_aware_stage6_writeback_preflight.sqlite",
        "summary_json": paths.out_root / "summary.json",
        "readme": paths.out_root / "README.md",
        "approve_stage20_sh": paths.out_root / "approve_stage20_live_write.sh",
        "next_live_stage20_sh": paths.out_root / "next_live_stage20_then_readback.sh",
    }
    write_csv(outputs["dry_run_report_csv"], report_rows)
    write_csv(outputs["stage20_candidates_csv"], stage20_candidates)
    write_csv(outputs["findings_csv"], findings)
    write_sqlite(
        outputs["sqlite"],
        {
            "dry_run_report": report_rows,
            "stage20_candidates": stage20_candidates,
            "findings": findings,
        },
    )
    summary = build_summary(
        paths=paths,
        input_rows=input_rows,
        report_rows=report_rows,
        stage20_candidates=stage20_candidates,
        findings=findings,
        field_guard=field_guard,
        stage5_guard=stage5_guard,
        field_catalog_payload=field_catalog_payload,
        outputs=outputs,
    )
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["readme"].write_text(render_readme(summary), encoding="utf-8")
    write_stage20_wrappers(paths.out_root, summary)
    return summary


def build_dry_run_row(
    row: dict[str, Any],
    *,
    row_index: int,
    field_catalog: list[dict[str, Any]],
    field_guard: dict[str, Any],
    analysis_date: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    review_id = safe_text(row.get("review_id"))
    lead_id = safe_text(row.get("selected_deal_id"))
    payload = {field: safe_text(row.get(field)) for field in DEAL_AI_FIELDS if safe_text(row.get(field))}
    findings: list[dict[str, Any]] = []
    status = "dry_run"
    reason = "live_write_not_confirmed"
    custom_fields_values: list[dict[str, Any]] = []

    if safe_text(row.get("stage5_decision")) != "allow_stage6_dry_run":
        status = "blocked"
        reason = "stage5_decision_is_not_allow_stage6_dry_run"
        findings.append(finding(row_index, review_id, lead_id, "stage5_decision_not_allowed", "P1", "stage5_decision", safe_text(row.get("stage5_decision"))))
    if not lead_id.isdigit():
        status = "blocked"
        reason = "missing_selected_deal_id"
        findings.append(finding(row_index, review_id, lead_id, "missing_selected_deal_id", "P1", "selected_deal_id", lead_id))
    missing_payload_fields = [field for field in DEAL_AI_FIELDS if not safe_text(row.get(field))]
    if missing_payload_fields:
        status = "blocked"
        reason = "missing_deal_ai_fields"
        findings.append(finding(row_index, review_id, lead_id, "missing_deal_ai_fields", "P1", "payload", " | ".join(missing_payload_fields)))
    protected_fields = sorted(set(payload) & PROTECTED_LEAD_FIELDS)
    if protected_fields:
        status = "blocked"
        reason = "protected_field_in_payload"
        findings.append(finding(row_index, review_id, lead_id, "protected_field_in_payload", "P1", "payload", " | ".join(protected_fields)))
    if field_guard["missing_fields"]:
        status = "blocked"
        reason = "missing_amo_lead_fields"
        findings.append(finding(row_index, review_id, lead_id, "missing_amo_lead_fields", "P1", "field_catalog", " | ".join(field_guard["missing_fields"])))
    if field_guard["api_only_fields"]:
        status = "blocked"
        reason = "api_only_amo_lead_fields"
        findings.append(finding(row_index, review_id, lead_id, "api_only_amo_lead_fields", "P1", "field_catalog", " | ".join(field_guard["api_only_fields"])))

    text_payload = {field: payload.get(field, "") for field in DEAL_AI_FIELDS}
    crm_findings = detect_crm_text_quality_risks(text_payload, analysis_date=analysis_date, min_severity="P2")
    for crm_finding in crm_findings:
        status = "blocked"
        reason = "crm_text_quality_blocker"
        findings.append(
            finding(
                row_index,
                review_id,
                lead_id,
                crm_finding.risk_type,
                crm_finding.severity,
                crm_finding.field,
                crm_finding.matched_text,
                crm_finding.reason,
            )
        )

    if status == "dry_run":
        try:
            custom_fields_values = build_custom_fields_values(payload, field_catalog)
        except Exception as exc:  # noqa: BLE001 - report as gate finding.
            status = "blocked"
            reason = "custom_field_payload_build_failed"
            findings.append(finding(row_index, review_id, lead_id, "custom_field_payload_build_failed", "P1", "payload", str(exc)))

    report_row = {
        **row,
        "stage6_row_index": row_index,
        "stage6_mode": "dry_run",
        "stage6_status": status,
        "stage6_reason": reason,
        "lead_id": lead_id,
        "updated_fields": " | ".join(DEAL_AI_FIELDS),
        "preview_payload": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "custom_fields_values_preview": json.dumps(custom_fields_values, ensure_ascii=False, sort_keys=True),
        "stage6_finding_types": " | ".join(sorted({safe_text(item.get("risk_type")) for item in findings if safe_text(item.get("risk_type"))})),
        "stage6_finding_count": len(findings),
        "stage6_live_write_allowed_now": "Нет",
    }
    return report_row, dedupe_findings(findings)


def choose_stage20_candidates(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if safe_text(row.get("stage6_status")) == "dry_run"]

    def score(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
        priority_penalty = 10 if safe_text(row.get("AI-приоритет сделки")).casefold() == "review" else 0
        phone_penalty = 5 if int_or_zero(row.get("candidate_phone_count")) > 1 else 0
        tallanto_penalty = 2 if safe_text(row.get("tallanto_context_status")) != "exact_phone_single" else 0
        warning_count = int_or_zero(row.get("stage5_warning_gate_count"))
        return (priority_penalty + phone_penalty + tallanto_penalty + warning_count, warning_count, phone_penalty, tallanto_penalty, safe_text(row.get("review_id")))

    selected = sorted(eligible, key=score)[:limit]
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        result.append({**row, "stage6_batch": "stage20", "stage6_batch_rank": rank})
    return result


def validate_field_catalog(field_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {safe_text(item.get("name")): item for item in field_catalog}
    missing = [field for field in DEAL_AI_FIELDS if field not in by_name]
    api_only = [field for field in DEAL_AI_FIELDS if bool((by_name.get(field) or {}).get("is_api_only"))]
    field_types = {field: safe_text((by_name.get(field) or {}).get("type")) for field in DEAL_AI_FIELDS if field in by_name}
    return {
        "required_fields": list(DEAL_AI_FIELDS),
        "missing_fields": missing,
        "api_only_fields": api_only,
        "field_types": field_types,
        "passed": not missing and not api_only,
    }


def validate_stage5_summary(summary: dict[str, Any], input_csv: Path) -> dict[str, Any]:
    readiness = summary.get("readiness") if isinstance(summary.get("readiness"), dict) else {}
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
    expected_input = Path(safe_text(outputs.get("dry_run_candidates_csv"))).resolve() if safe_text(outputs.get("dry_run_candidates_csv")) else None
    actual_input = input_csv.resolve()
    return {
        "schema_version": safe_text(summary.get("schema_version")),
        "passed_for_stage6_dry_run": bool(readiness.get("passed_for_stage6_dry_run")),
        "passed_for_live_writeback": bool(readiness.get("passed_for_live_writeback")),
        "input_matches_stage5_output": expected_input == actual_input if expected_input else False,
        "expected_input": str(expected_input) if expected_input else "",
        "actual_input": str(actual_input),
        "passed": bool(readiness.get("passed_for_stage6_dry_run")) and not bool(readiness.get("passed_for_live_writeback")) and expected_input == actual_input,
    }


def build_summary(
    *,
    paths: DealAwareStage6Paths,
    input_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    stage20_candidates: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    field_guard: dict[str, Any],
    stage5_guard: dict[str, Any],
    field_catalog_payload: dict[str, Any],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    status_counts = dict(Counter(safe_text(row.get("stage6_status")) for row in report_rows).most_common())
    risk_counts = dict(Counter(safe_text(row.get("risk_type")) for row in findings).most_common())
    dry_run_rows = status_counts.get("dry_run", 0)
    blocked_rows = len(report_rows) - dry_run_rows
    systemic_passed = bool(field_guard.get("passed")) and bool(stage5_guard.get("passed"))
    passed_for_stage20_preflight = (
        systemic_passed
        and dry_run_rows > 0
        and len(stage20_candidates) == min(paths.stage20_size, dry_run_rows)
    )
    passed_for_full_batch_preflight = systemic_passed and dry_run_rows == len(input_rows) and blocked_rows == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_date": paths.analysis_date,
        "input": {
            "stage5_candidates_csv": str(paths.input_csv),
            "stage5_candidates_sha256": sha256_file(paths.input_csv),
            "stage5_summary_json": str(paths.stage5_summary_json),
            "stage5_summary_sha256": sha256_file(paths.stage5_summary_json),
            "field_catalog_cache_json": str(paths.field_catalog_cache_json),
            "field_catalog_cache_sha256": sha256_file(paths.field_catalog_cache_json),
        },
        "safety": {
            "read_only": True,
            "dry_run_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
        "coverage": {
            "input_rows": len(input_rows),
            "dry_run_report_rows": len(report_rows),
            "dry_run_rows": dry_run_rows,
            "blocked_rows": blocked_rows,
            "failed_rows": status_counts.get("failed", 0),
            "stage20_candidate_rows": len(stage20_candidates),
            "findings": len(findings),
        },
        "status_counts": status_counts,
        "risk_counts": risk_counts,
        "field_catalog_guard": {
            **field_guard,
            "synced_at": safe_text(field_catalog_payload.get("synced_at")),
        },
        "stage5_guard": stage5_guard,
        "target_fields": list(DEAL_AI_FIELDS),
        "readiness": {
            "stage6_dry_run_built": True,
            "passed_for_stage20_preflight": passed_for_stage20_preflight,
            "passed_for_full_batch_preflight": passed_for_full_batch_preflight,
            "passed_for_live_writeback": False,
            "safe_to_write_deal_fields": False,
            "live_write_blocker": "Stage 6 produced dry-run and stage20 candidates only. Live write requires Claude audit, operator approval, explicit env confirmation and post-write readback.",
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def render_readme(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    return "\n".join(
        [
            "# Deal-Aware Stage 6 Writeback Preflight",
            "",
            "Dry-run/preflight for writing deal-aware AI fields to AMO deals. No live write was executed.",
            "",
            "## Coverage",
            "",
            f"- input rows: {coverage['input_rows']}",
            f"- dry-run rows: {coverage['dry_run_rows']}",
            f"- blocked rows: {coverage['blocked_rows']}",
            f"- stage20 candidates: {coverage['stage20_candidate_rows']}",
            "",
            "## Readiness",
            "",
            f"- passed for Stage20 preflight: {summary['readiness']['passed_for_stage20_preflight']}",
            f"- passed for live writeback: {summary['readiness']['passed_for_live_writeback']}",
            "- Live write still requires Claude audit, explicit operator approval and readback.",
            "",
            "## Outputs",
            "",
            *[f"- `{key}`: `{path}`" for key, path in summary["outputs"].items()],
            "",
        ]
    )


def write_stage20_wrappers(out_root: Path, summary: dict[str, Any]) -> None:
    approve = out_root / "approve_stage20_live_write.sh"
    next_live = out_root / "next_live_stage20_then_readback.sh"
    approval_json = out_root / "operator_approval_stage20.json"
    candidates = summary["outputs"]["stage20_candidates_csv"]
    summary_json = summary["outputs"]["summary_json"]
    report_dir = out_root / "stage20_live_report"
    readback_dir = out_root / "readback_after_stage20"
    token = "WRITE_AMO_DEAL_AWARE_STAGE20_20260513"
    approve.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cat > {shell_quote(str(approval_json))} <<'JSON'
{{
  "schema_version": "deal_aware_stage20_operator_approval_v1",
  "stage_id": "deal_aware_stage20_20260513",
  "expected_written": 20,
  "input": "{candidates}",
  "approved_at": "manual",
  "operator": "Dmitry"
}}
JSON
echo "Wrote operator approval: {approval_json}"
""",
        encoding="utf-8",
    )
    next_live.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd {shell_quote(str(Path.cwd()))}
if [[ "${{CONFIRM_DEAL_AWARE_STAGE20_LIVE_WRITE:-}}" != "{token}" ]]; then
  echo "BLOCKED: set CONFIRM_DEAL_AWARE_STAGE20_LIVE_WRITE={token} to run live Stage20." >&2
  exit 2
fi
if [[ ! -f {shell_quote(str(approval_json))} ]]; then
  echo "BLOCKED: operator approval file is missing. Run approve_stage20_live_write.sh first." >&2
  exit 2
fi
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/private/tmp/uv-cache uv run --with pandas --with openpyxl --with sqlalchemy --with requests --with 'psycopg[binary]' \\
  python scripts/write_deal_aware_amo_fields.py \\
  --input {shell_quote(candidates)} \\
  --stage5-summary {shell_quote(summary['input']['stage5_summary_json'])} \\
  --field-catalog-cache {shell_quote(summary['input']['field_catalog_cache_json'])} \\
  --out-root {shell_quote(str(report_dir))} \\
  --execute-live-write \\
  --live-confirmation {LIVE_CONFIRMATION} \\
  --expected-written 20 \\
  --operator-approval {shell_quote(str(approval_json))}
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/private/tmp/uv-cache uv run --with pandas --with openpyxl --with sqlalchemy --with requests --with 'psycopg[binary]' \\
  python scripts/readback_deal_aware_amo_fields.py \\
  --writeback-report {shell_quote(str(report_dir / 'deal_stage6_writeback_report.csv'))} \\
  --out-root {shell_quote(str(readback_dir))} \\
  --expected-evaluated 20
""",
        encoding="utf-8",
    )
    approve.chmod(0o755)
    next_live.chmod(0o755)


def finding(
    row_index: int,
    review_id: str,
    lead_id: str,
    risk_type: str,
    severity: str,
    field: str,
    matched_text: str,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "review_id": review_id,
        "lead_id": lead_id,
        "risk_type": risk_type,
        "severity": severity,
        "field": field,
        "matched_text": matched_text,
        "reason": reason,
    }


def dedupe_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for row in rows:
        key = (row.get("review_id"), row.get("risk_type"), row.get("severity"), row.get("field"), row.get("matched_text"))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value).replace(",", ".")))
    except ValueError:
        return 0


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_sqlite(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        for table, rows in tables.items():
            if not rows:
                con.execute(f'CREATE TABLE "{table}" (empty TEXT)')
                continue
            columns = sorted({key for row in rows for key in row.keys()})
            con.execute(f'CREATE TABLE "{table}" ({", ".join(f"{quote_ident(col)} TEXT" for col in columns)})')
            placeholders = ", ".join("?" for _ in columns)
            con.executemany(
                f'INSERT INTO "{table}" ({", ".join(quote_ident(col) for col in columns)}) VALUES ({placeholders})',
                [[stringify(row.get(col)) for col in columns] for row in rows],
            )
        con.commit()
    finally:
        con.close()
