#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.productization.tenant_config import load_tenant_config, tenant_config_summary
from mango_mvp.quality.crm_writeback_frozen_corpus import (
    CrmWritebackCorpusValidationConfig,
    validate_crm_writeback_frozen_corpus,
)
from mango_mvp.quality.crm_writeback_quality_detector import (
    CrmWritebackFinding,
    detect_crm_writeback_quality_risks,
    findings_to_risk_counts,
)
from mango_mvp.quality.crm_writeback_population_recall import (
    scan_crm_writeback_population_recall,
    write_population_recall_outputs,
)
from mango_mvp.quality.crm_text_quality_detector import (
    CrmTextQualityFinding,
    detect_crm_text_quality_batch_risks,
    detect_crm_text_quality_risks,
    has_blocking_crm_text_findings,
)


CRM_TEXT_FIELDS = (
    "Краткое резюме последнего свежего звонка",
    "Краткая история общения",
    "Хронология общения (последние 5 касаний)",
    "Возражения",
    "Следующий шаг",
    "История общения Tallanto",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRM writeback quality gate on AMO-ready export.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--frozen-corpus-jsonl", default="")
    parser.add_argument("--detector-min-severity", default="P2")
    parser.add_argument("--fail-on-blocking-rows", action="store_true", default=True)
    parser.add_argument("--tenant-config", default="")
    parser.add_argument("--population-recall-mode", choices=("observe", "fail-live"), default="observe")
    parser.add_argument("--population-high-precision-uncovered-max", type=int, default=0)
    parser.add_argument("--analysis-date", default=date.today().isoformat())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser().resolve()
    tenant_config_result = load_tenant_config(args.tenant_config) if args.tenant_config else None

    rows = _read_csv(input_path)
    report_rows: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []
    crm_text_blocking_rows = 0
    crm_text_warning_rows = 0
    crm_text_risk_counter: Counter[str] = Counter()
    crm_text_warning_counter: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        text = _crm_text(row)
        findings = detect_crm_writeback_quality_risks(text, min_severity=args.detector_min_severity)
        findings.extend(_field_level_findings(row))
        findings.extend(_metadata_findings(row))
        crm_text_findings = detect_crm_text_quality_risks(
            row,
            min_severity="P3",
            analysis_date=args.analysis_date,
        )
        crm_text_blocking = [
            finding for finding in crm_text_findings if has_blocking_crm_text_findings([finding])
        ]
        crm_text_warnings = [
            finding for finding in crm_text_findings if not has_blocking_crm_text_findings([finding])
        ]
        if crm_text_blocking:
            crm_text_blocking_rows += 1
        if crm_text_warnings:
            crm_text_warning_rows += 1
        crm_text_risk_counter.update(finding.risk_type for finding in crm_text_blocking)
        crm_text_warning_counter.update(finding.risk_type for finding in crm_text_warnings)
        findings.extend(_crm_text_to_writeback_findings(crm_text_blocking))
        risk_counts = findings_to_risk_counts(findings)
        report_row = {
            "row_index": index,
            "phone": row.get("Телефон клиента", ""),
            "decision": "block" if findings else "allow",
            "risk_types": " | ".join(sorted(risk_counts)),
            "crm_text_warning_types": " | ".join(sorted({finding.risk_type for finding in crm_text_warnings})),
            "detector_matches": " | ".join(f"{finding.risk_type}:{finding.matched_text}" for finding in findings[:10]),
            "crm_text_warning_matches": " | ".join(
                f"{finding.risk_type}:{finding.matched_text}" for finding in crm_text_warnings[:10]
            ),
            "text_preview": text[:1000],
        }
        report_rows.append(report_row)
        if findings:
            blocking.append(report_row)

    batch_crm_text_findings = [
        finding
        for finding in detect_crm_text_quality_batch_risks(
            rows,
            min_severity="P2",
            analysis_date=args.analysis_date,
        )
        if has_blocking_crm_text_findings([finding])
    ]
    if batch_crm_text_findings:
        blocking_by_index = {row["row_index"]: row for row in blocking}
        for finding in batch_crm_text_findings:
            if finding.row_index is None or finding.row_index < 1 or finding.row_index > len(report_rows):
                continue
            report_row = report_rows[finding.row_index - 1]
            report_row["decision"] = "block"
            report_row["risk_types"] = _merge_pipe_values(report_row["risk_types"], [finding.risk_type])
            report_row["detector_matches"] = _merge_pipe_values(
                report_row["detector_matches"],
                [f"{finding.risk_type}:{finding.field}: {finding.matched_text}"],
            )
            if finding.row_index not in blocking_by_index:
                blocking.append(report_row)
                blocking_by_index[finding.row_index] = report_row
        crm_text_risk_counter.update(finding.risk_type for finding in batch_crm_text_findings)
        crm_text_blocking_rows = len({row["row_index"] for row in blocking if "crm_text" in row["risk_types"] or any(
            risk in row["risk_types"] for risk in (
                "lossy_ellipsis_truncation",
                "duplicate_label_and_count",
                "empty_auto_history",
                "strong_negative_objection_conflict",
                "strong_negative_objection_label",
                "closure_next_step_requires_downgrade",
                "priority_next_step_conflict",
                "vague_next_step",
                "stale_uniform_followup_date",
                "lost_lead_next_step_conflict",
                "passive_customer_next_step_conflict",
                "explicit_no_next_step_conflict",
                "wrong_person_or_identity_mismatch",
                "active_client_loss_reason_requires_entity_resolution",
                "completed_payment_next_step_conflict",
                "relative_next_step_date_mismatch",
                "stale_source_next_step",
                "cross_field_duplicate_information",
            )
        )})

    corpus_summary: dict[str, Any] | None = None
    if args.frozen_corpus_jsonl:
        corpus_out = out_root / "frozen_corpus_validation"
        corpus_summary = validate_crm_writeback_frozen_corpus(
            CrmWritebackCorpusValidationConfig(
                corpus_jsonl=Path(args.frozen_corpus_jsonl),
                out_root=corpus_out,
                detector_min_severity=args.detector_min_severity,
            )
        )

    population_result = scan_crm_writeback_population_recall(
        rows,
        text_fields=CRM_TEXT_FIELDS,
        detector_min_severity=args.detector_min_severity,
        high_precision_uncovered_max=args.population_high_precision_uncovered_max,
    )
    population_outputs = write_population_recall_outputs(out_root, population_result)
    c12_summary = _c12_overlap_summary(rows)

    outputs = {
        "report_csv": out_root / "crm_writeback_quality_report.csv",
        "blocking_rows_csv": out_root / "crm_writeback_quality_blocking_rows.csv",
        "summary_json": out_root / "summary.json",
    }
    _write_csv(outputs["report_csv"], report_rows)
    _write_csv(outputs["blocking_rows_csv"], blocking, fieldnames=list(report_rows[0].keys()) if report_rows else [])

    population_passes_mode = (
        args.population_recall_mode == "observe"
        or bool((population_result.get("summary") or {}).get("passed_for_live"))
    )
    passed = (
        len(blocking) == 0
        and (corpus_summary is None or bool(corpus_summary.get("passed")))
        and population_passes_mode
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "tenant_config": tenant_config_summary(tenant_config_result),
        "detector_min_severity": args.detector_min_severity,
        "population_recall_mode": args.population_recall_mode,
        "rows": len(rows),
        "passed": passed,
        "blocking_rows": len(blocking),
        "decision_counts": dict(Counter(row["decision"] for row in report_rows)),
        "risk_counts": dict(Counter(risk for row in report_rows for risk in _split(row["risk_types"])).most_common()),
        "frozen_corpus": corpus_summary,
        "population_recall": {
            **dict(population_result.get("summary") or {}),
            "outputs": population_outputs,
        },
        "crm_text_quality": {
            "schema_version": "crm_text_quality_gate_v1",
            "passed_for_live": crm_text_blocking_rows == 0,
            "blocking_rows": crm_text_blocking_rows,
            "warning_rows": crm_text_warning_rows,
            "blocking_risk_counts": dict(crm_text_risk_counter.most_common()),
            "warning_risk_counts": dict(crm_text_warning_counter.most_common()),
            "fail_live_risk_types": [
                "lossy_ellipsis_truncation",
                "duplicate_label_and_count",
                "empty_auto_history",
                "strong_negative_objection_conflict",
                "strong_negative_objection_label",
                "closure_next_step_requires_downgrade",
                "priority_next_step_conflict",
                "vague_next_step",
                "stale_uniform_followup_date",
                "lost_lead_next_step_conflict",
                "passive_customer_next_step_conflict",
                "explicit_no_next_step_conflict",
                "wrong_person_or_identity_mismatch",
                "active_client_loss_reason_requires_entity_resolution",
                "completed_payment_next_step_conflict",
                "relative_next_step_date_mismatch",
                "stale_source_next_step",
                "cross_field_duplicate_information",
            ],
        },
        "c12_history_duplication": c12_summary,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _crm_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(field, "") or "") for field in CRM_TEXT_FIELDS).strip()


def _field_level_findings(row: dict[str, Any]) -> list[CrmWritebackFinding]:
    findings: list[CrmWritebackFinding] = []
    history = str(row.get("Краткая история общения", "") or "").strip()
    if not history:
        findings.append(
            CrmWritebackFinding(
                risk_type="empty_crm_history",
                severity="P2",
                matched_text="",
                reason="AMO-ready row has empty short history",
            )
        )
    for field in CRM_TEXT_FIELDS:
        value = str(row.get(field, "") or "").rstrip()
        if "..." in value or "…" in value:
            findings.append(
                CrmWritebackFinding(
                    risk_type="truncated_crm_text",
                    severity="P1",
                    matched_text=f"{field}: {value[-120:]}",
                    reason="CRM writeback field appears truncated by ellipsis",
                )
            )
    return findings


def _crm_text_to_writeback_findings(findings: list[CrmTextQualityFinding]) -> list[CrmWritebackFinding]:
    return [
        CrmWritebackFinding(
            risk_type=finding.risk_type,
            severity=finding.severity,
            matched_text=f"{finding.field}: {finding.matched_text}",
            reason=finding.reason,
        )
        for finding in findings
    ]


SERVICE_CONTEXT_CALL_TYPES = {"service_call", "existing_client_progress", "technical_call"}


def _metadata_findings(row: dict[str, Any]) -> list[CrmWritebackFinding]:
    findings: list[CrmWritebackFinding] = []
    call_type = str(row.get("Тип последнего свежего звонка", "") or "").strip().casefold()
    if call_type in SERVICE_CONTEXT_CALL_TYPES:
        findings.append(
            CrmWritebackFinding(
                risk_type="service_or_existing_client_live_writeback",
                severity="P2",
                matched_text=call_type,
                reason="Service/existing-client context must not be promoted as a new sales live writeback row",
            )
        )
    amo_ids = _split_ids(row.get("AMO contact IDs"))
    if len(amo_ids) != 1:
        findings.append(
            CrmWritebackFinding(
                risk_type="amo_orphan_or_ambiguous_contact",
                severity="P2",
                matched_text=str(row.get("AMO contact IDs", "") or ""),
                reason="Live contact writeback requires exactly one known AMO contact id",
            )
        )
    policy = str(row.get("CRM writeback policy", "") or "").strip()
    if policy and policy != "live_update_ready":
        findings.append(
            CrmWritebackFinding(
                risk_type="crm_writeback_policy_not_live_ready",
                severity="P2",
                matched_text=policy,
                reason="Builder policy does not allow live update",
            )
        )
    return findings


def _split_ids(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    import re

    return [part for part in re.split(r"[|,;\s]+", text) if part.strip()]


def _c12_overlap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_history_overlap_rows = 0
    history_chronology_overlap_rows = 0
    for row in rows:
        latest = str(row.get("Краткое резюме последнего свежего звонка", "") or "")
        history = str(row.get("Краткая история общения", "") or "")
        chronology = str(row.get("Хронология общения (последние 5 касаний)", "") or "")
        if _overlap_ratio(latest, history) >= 0.8 and latest and history:
            summary_history_overlap_rows += 1
        if _overlap_ratio(history, chronology) >= 0.8 and history and chronology:
            history_chronology_overlap_rows += 1
    return {
        "schema_version": "crm_writeback_c12_overlap_v1",
        "rows_scanned": len(rows),
        "summary_history_overlap_rows": summary_history_overlap_rows,
        "history_chronology_overlap_rows": history_chronology_overlap_rows,
        "blocking": False,
        "policy": "soft_counter_only",
    }


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    smaller, larger = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    return len(smaller & larger) / max(len(smaller), 1)


def _token_set(value: str) -> set[str]:
    import re

    return {token for token in re.findall(r"[а-яa-z0-9]{4,}", value.casefold())}


def _split(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _merge_pipe_values(current: str, values: list[str]) -> str:
    parts = _split(current)
    for value in values:
        text = str(value or "").strip()
        if text and text not in parts:
            parts.append(text)
    return " | ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
