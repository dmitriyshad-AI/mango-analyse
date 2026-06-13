from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS, resolve_analysis_date
from mango_mvp.deal_aware.stage1_snapshot import quote_ident, read_csv, safe_text, stringify, write_csv
from mango_mvp.question_catalog.source_index import load_source_index, split_tokens
from mango_mvp.quality.crm_text_quality_detector import detect_crm_text_quality_risks


SCHEMA_VERSION = "deal_aware_stage5_quality_gate_v1"

DRY_RUN_BLOCKING_STAGE4_RISK_TYPES = {
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
    "duplicate_loss_reason_requires_entity_resolution",
    "no_application_loss_reason_blocks_sales_writeback",
    "no_contact_archive_loss_reason_requires_no_action",
    "invalid_or_test_loss_reason_blocks_writeback",
    "company_side_loss_reason_requires_review",
    "refund_or_postsale_loss_reason_requires_service_review",
    "graduate_loss_reason_requires_alumni_policy",
    "not_qualified_or_out_of_scope_loss_reason_requires_review",
    "ambiguous_loss_reason_requires_manual_review",
    "terminal_lost_without_loss_reason_requires_manual_review",
    "relative_next_step_date_mismatch",
    "stale_source_next_step",
}

LIVE_REVIEW_STAGE4_RISK_TYPES = {
    "completed_payment_next_step_conflict",
    "future_prospect_loss_reason_requires_reactivation_policy",
    "terminal_lost_reason_blocks_active_sales_writeback",
    "cross_field_duplicate_information",
    "verbose_manager_ux",
}

PAYMENT_WORD_RE = re.compile(r"\b(?:оплат\w*|плат[её]ж\w*|чек\w*|квитанц\w*|счет\w*|сч[её]т\w*)\b", re.I)
COMPLETED_PAYMENT_RE = re.compile(
    r"\b(?:оплат\w+|плат[её]ж\w+|чек\w+|квитанц\w+)\s+(?:уже\s+)?(?:получен\w*|поступил\w*|прислан\w*|подтвержден\w*|внес[её]н\w*)|"
    r"\bсделк\w+\s+(?:закрыт\w+|успешн\w+\s+закрыт\w+|оплачен\w+)",
    re.I,
)
TERMINAL_STATUS_RE = re.compile(r"\b(?:закрыт\w+\s+и\s+не\s+реализован\w+|closed\s+lost)\b", re.I)
DUPLICATE_OR_EXISTING_RE = re.compile(r"\b(?:дубл\w+|действующ\w+\s+клиент\w+|выпускник\w+)\b", re.I)
SAFE_HOLD_NEXT_STEP_RE = re.compile(
    r"\b(?:лист\s+ожидан|уведомить\s+при|при\s+открытии|ручн\w+\s+контрол|не\s+делать\s+активн|не\s+дожим|без\s+дожим|вернуться\s+к\s+вопросу)\b",
    re.I,
)
BAD_TENANT_TERM_RE = re.compile(r"(?<!У)(?:МПК|НПК)\s+МФТИ|\bночн\w+\s+школ", re.I)
ELLIPSIS_RE = re.compile(r"\.\.\.|…")
COUNT_TAG_SOUP_RE = re.compile(
    r"(?:^|\|)\s*[А-ЯЁA-Zа-яёa-z][^|:]{1,60}:\s*\d{1,4}\s*(?:касан\w*|звон\w*|раз[а-я]*|$)",
    re.I,
)
RAW_PHONE_RE = re.compile(r"(?:\+7|8)\s*[\d ()-]{9,}\d")


@dataclass(frozen=True)
class DealQualityGatePaths:
    stage4_preview_root: Path
    out_root: Path
    analysis_date: str | None = None
    question_catalog_source_index_json: Path | None = None


def run_deal_quality_gate(paths: DealQualityGatePaths) -> dict[str, Any]:
    paths.out_root.mkdir(parents=True, exist_ok=True)
    analysis_date = resolve_analysis_date(paths.analysis_date)
    preview_rows = read_csv(paths.stage4_preview_root / "deal_stage4_preview.csv")
    payload_rows = read_payload_jsonl(paths.stage4_preview_root / "deal_stage4_payloads.jsonl")
    stage4_findings_by_review_id = read_stage4_findings_by_review_id(paths.stage4_preview_root / "deal_stage4_quality_findings.csv")
    stage4_summary = load_json(paths.stage4_preview_root / "summary.json")
    question_index = load_source_index(paths.question_catalog_source_index_json) if paths.question_catalog_source_index_json else {}

    payload_by_review_id = {safe_text(row.get("review_id")): row for row in payload_rows}
    report_rows: list[dict[str, Any]] = []
    dry_run_candidates: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    findings_rows: list[dict[str, Any]] = []

    for index, row in enumerate(preview_rows, start=1):
        payload_obj = payload_by_review_id.get(safe_text(row.get("review_id")), {})
        payload = payload_obj.get("payload") if isinstance(payload_obj.get("payload"), dict) else {}
        hard_findings, warning_findings = evaluate_row(
            row,
            payload,
            row_index=index,
            analysis_date=analysis_date,
            stage4_findings=stage4_findings_by_review_id.get(safe_text(row.get("review_id")), []),
            question_index=question_index,
        )
        decision = "allow_stage6_dry_run" if not hard_findings else "block_stage6"
        report_row = {
            **row,
            "row_index": index,
            "stage5_decision": decision,
            "stage5_hard_gate_types": " | ".join(sorted({finding["gate_type"] for finding in hard_findings})),
            "stage5_warning_gate_types": " | ".join(sorted({finding["gate_type"] for finding in warning_findings})),
            "stage5_hard_gate_count": len(hard_findings),
            "stage5_warning_gate_count": len(warning_findings),
            "stage5_live_write_allowed_now": "Нет",
            "stage5_live_write_blocker": "Stage 5 only authorizes Stage 6 dry-run; live write requires audit, dry-run report, operator approval and readback gate.",
        }
        report_rows.append(report_row)
        findings_rows.extend(hard_findings)
        findings_rows.extend(warning_findings)
        if hard_findings:
            blocked_rows.append(report_row)
        else:
            dry_run_candidates.append(report_row)
        if warning_findings:
            warning_rows.append(report_row)

    outputs = {
        "gate_report_csv": paths.out_root / "deal_stage5_quality_gate_report.csv",
        "dry_run_candidates_csv": paths.out_root / "deal_stage5_stage6_dry_run_candidates.csv",
        "blocked_rows_csv": paths.out_root / "deal_stage5_blocked_rows.csv",
        "warning_rows_csv": paths.out_root / "deal_stage5_warning_rows.csv",
        "findings_csv": paths.out_root / "deal_stage5_findings.csv",
        "sqlite": paths.out_root / "deal_aware_stage5_quality_gate.sqlite",
        "summary_json": paths.out_root / "summary.json",
        "readme": paths.out_root / "README.md",
    }
    write_csv(outputs["gate_report_csv"], report_rows)
    write_csv(outputs["dry_run_candidates_csv"], dry_run_candidates)
    write_csv(outputs["blocked_rows_csv"], blocked_rows)
    write_csv(outputs["warning_rows_csv"], warning_rows)
    write_csv(outputs["findings_csv"], findings_rows)
    write_sqlite(
        outputs["sqlite"],
        {
            "gate_report": report_rows,
            "stage6_dry_run_candidates": dry_run_candidates,
            "blocked_rows": blocked_rows,
            "warning_rows": warning_rows,
            "findings": findings_rows,
        },
    )

    summary = build_summary(
        paths=paths,
        preview_rows=preview_rows,
        report_rows=report_rows,
        dry_run_candidates=dry_run_candidates,
        blocked_rows=blocked_rows,
        warning_rows=warning_rows,
        findings_rows=findings_rows,
        stage4_summary=stage4_summary,
        outputs=outputs,
        analysis_date=analysis_date,
    )
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["readme"].write_text(render_readme(summary), encoding="utf-8")
    return summary


def evaluate_row(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    row_index: int,
    analysis_date: str,
    stage4_findings: list[dict[str, Any]] | None = None,
    question_index: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    review_id = safe_text(row.get("review_id"))
    deal_id = safe_text(row.get("selected_deal_id"))
    full_text = " ".join(safe_text(payload.get(field) or row.get(field)) for field in DEAL_AI_FIELDS)
    next_step = safe_text(payload.get("AI-рекомендованный следующий шаг") or row.get("AI-рекомендованный следующий шаг"))
    status = safe_text(row.get("selected_status_name"))
    loss_reason = safe_text(row.get("selected_loss_reason"))
    stage4_findings = stage4_findings or []
    stage4_risks = set(split_pipe(row.get("quality_risk_types")))
    stage4_risks.update(safe_text(f.get("risk_type")) for f in stage4_findings if safe_text(f.get("risk_type")))

    for risk in sorted(stage4_risks & DRY_RUN_BLOCKING_STAGE4_RISK_TYPES):
        hard.append(finding(row_index, review_id, deal_id, risk, "P1", "quality_risk_types", risk, "Stage 4 emitted a hard quality risk type."))
    for risk in sorted(stage4_risks & LIVE_REVIEW_STAGE4_RISK_TYPES):
        if risk == "terminal_lost_reason_blocks_active_sales_writeback" and not SAFE_HOLD_NEXT_STEP_RE.search(next_step):
            hard.append(finding(row_index, review_id, deal_id, risk, "P1", "quality_risk_types", risk, "Terminal lost deal still appears to have an active sales next step."))
            continue
        warnings.append(finding(row_index, review_id, deal_id, risk, "P3", "quality_risk_types", risk, "Stage 4 emitted a live-review risk. Dry-run is allowed, live writeback requires audit or policy override."))
    if safe_text(row.get("crm_text_quality_passed")) != "Да" and not (stage4_risks & (DRY_RUN_BLOCKING_STAGE4_RISK_TYPES | LIVE_REVIEW_STAGE4_RISK_TYPES)):
        hard.append(finding(row_index, review_id, deal_id, "stage4_blocking_quality_unknown", "P1", "crm_text_quality_passed", safe_text(row.get("quality_risk_types")), "Stage 4 row did not pass CRM text quality, but no known risk type was attached."))

    if ELLIPSIS_RE.search(full_text):
        hard.append(finding(row_index, review_id, deal_id, "lossy_ellipsis_truncation", "P1", "payload", "...", "Deal payload contains lossy ellipsis."))
    if BAD_TENANT_TERM_RE.search(full_text):
        hard.append(finding(row_index, review_id, deal_id, "tenant_term_normalization_failure", "P1", "payload", BAD_TENANT_TERM_RE.search(full_text).group(0), "Known tenant/ASR term is not normalized."))
    if COUNT_TAG_SOUP_RE.search(full_text):
        hard.append(finding(row_index, review_id, deal_id, "tag_count_artifact", "P2", "payload", COUNT_TAG_SOUP_RE.search(full_text).group(0), "Deal field contains raw tag-count artifact."))

    if TERMINAL_STATUS_RE.search(status) and not SAFE_HOLD_NEXT_STEP_RE.search(next_step):
        hard.append(finding(row_index, review_id, deal_id, "closed_deal_conflict", "P1", "selected_status_name", status, "Closed/lost deal must not receive normal deal-aware sales writeback."))
    elif TERMINAL_STATUS_RE.search(status):
        warnings.append(finding(row_index, review_id, deal_id, "closed_deal_safe_hold_requires_policy", "P3", "selected_status_name", status, "Closed/lost status has a safe hold/manual-control next step. Dry-run allowed; live needs policy."))
    if DUPLICATE_OR_EXISTING_RE.search(loss_reason):
        hard.append(finding(row_index, review_id, deal_id, "duplicate_existing_client_requires_entity_resolution", "P1", "selected_loss_reason", loss_reason, "Duplicate/existing-client deals require entity resolution."))
    if COMPLETED_PAYMENT_RE.search(full_text) and PAYMENT_WORD_RE.search(next_step):
        hard.append(finding(row_index, review_id, deal_id, "payment_next_step_consistency", "P1", "AI-рекомендованный следующий шаг", next_step, "Payment evidence conflicts with payment-oriented next step."))

    catalog_hard, catalog_warnings = evaluate_question_catalog_conflicts(
        row,
        payload,
        question_index=question_index or {},
        row_index=row_index,
        review_id=review_id,
        deal_id=deal_id,
    )
    hard.extend(catalog_hard)
    warnings.extend(catalog_warnings)

    for field in DEAL_AI_FIELDS:
        value = safe_text(payload.get(field) or row.get(field))
        if not value:
            hard.append(finding(row_index, review_id, deal_id, "missing_deal_ai_field", "P1", field, "", "Required deal AI field is empty."))

    crm_findings = detect_crm_text_quality_risks(
        {
            **{field: safe_text(payload.get(field) or row.get(field)) for field in DEAL_AI_FIELDS},
            "AMO статус сделки": status,
            "AMO причина отказа": loss_reason,
            "priority": safe_text(payload.get("AI-приоритет сделки") or row.get("AI-приоритет сделки")),
            "Рекомендуемая дата следующего контакта": safe_text(payload.get("AI-дата следующего касания") or row.get("AI-дата следующего касания")),
        },
        analysis_date=analysis_date,
        min_severity="P2",
        compact_max_chars=1800,
        verbose_max_chars=5000,
    )
    for crm_finding in crm_findings:
        target = warnings if crm_finding.risk_type in LIVE_REVIEW_STAGE4_RISK_TYPES else hard
        if crm_finding.risk_type == "terminal_lost_reason_blocks_active_sales_writeback" and not SAFE_HOLD_NEXT_STEP_RE.search(next_step):
            target = hard
        severity = "P3" if target is warnings else crm_finding.severity
        target.append(
            finding(
                row_index,
                review_id,
                deal_id,
                crm_finding.risk_type,
                severity,
                crm_finding.field,
                crm_finding.matched_text,
                crm_finding.reason,
            )
        )

    if int_or_zero(row.get("candidate_phone_count")) > 1:
        warnings.append(finding(row_index, review_id, deal_id, "multi_phone_deal_requires_audit", "P3", "candidate_phone_count", safe_text(row.get("candidate_phone_count")), "Deal is linked to multiple phones; dry-run is allowed, live requires audit sampling."))
    if "stage2_confidence_low" in split_pipe(row.get("stage3_risk_flags")):
        warnings.append(finding(row_index, review_id, deal_id, "stage2_confidence_low", "P3", "stage3_risk_flags", "stage2_confidence_low", "Attribution was accepted by Stage 3 but confidence score is in the low bucket."))
    if safe_text(row.get("tallanto_context_status")) != "exact_phone_single":
        warnings.append(finding(row_index, review_id, deal_id, "tallanto_context_not_exact_single", "P3", "tallanto_context_status", safe_text(row.get("tallanto_context_status")), "Tallanto context is not a single exact phone match; do not overclaim payments/groups."))
    if safe_text(row.get("AI-приоритет сделки")).casefold() == "review":
        warnings.append(finding(row_index, review_id, deal_id, "review_priority_requires_manager_attention", "P3", "AI-приоритет сделки", "review", "Payload intentionally routes the deal to manager review."))
    if RAW_PHONE_RE.search(full_text):
        warnings.append(finding(row_index, review_id, deal_id, "phone_visible_in_internal_deal_payload", "P3", "payload", RAW_PHONE_RE.search(full_text).group(0), "Phone may be acceptable for internal CRM but must be tenant-policy controlled for SaaS."))

    return dedupe_findings(hard), dedupe_findings(warnings)


SERVICE_THEME_RE = re.compile(r"(schedule|document|docs|tax|matcap|refund|return|personal|access|lk|service|receipt|certificate)", re.I)
PAYMENT_THEME_RE = re.compile(r"(payment|receipt|check|invoice|pricing|price|оплат|чек|счет|счёт)", re.I)
RISKY_DECISION_THEME_RE = re.compile(r"(refund|return|matcap|tax|recalc|перерасч|возврат|налог|маткап)", re.I)
SALES_ACTION_RE = re.compile(r"\b(?:дожим|предлож\w*|продаж\w*|купить|запис\w*|оплат\w*|связаться\s+с\s+предложением)\b", re.I)
AUTONOMOUS_ACTION_RE = re.compile(r"\b(?:автоматическ\w*|отправить\s+клиент\w*|предложить|записать|подтвердить)\b", re.I)
PROMISE_OR_DECISION_RE = re.compile(r"\b(?:обеща\w*|гарантир\w*|точно|одобр\w*|верн[её]м|сделаем\s+возврат|можно\s+использовать)\b", re.I)


def evaluate_question_catalog_conflicts(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    question_index: dict[str, dict[str, list[str]]],
    row_index: int,
    review_id: str,
    deal_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not question_index:
        return [], []
    hard: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    next_step = safe_text(payload.get("AI-рекомендованный следующий шаг") or row.get("AI-рекомендованный следующий шаг"))
    call_ids = extract_call_ids(row, payload)
    if not call_ids:
        return [], []
    matched = []
    missing = []
    for call_id in call_ids:
        entry = question_index.get(call_id)
        if entry:
            matched.append((call_id, entry))
        else:
            missing.append(call_id)
    for call_id in missing:
        warnings.append(
            finding(
                row_index,
                review_id,
                deal_id,
                "catalog_index_missing_call_id",
                "P3",
                "question_catalog_source_index",
                call_id,
                "Question catalog source index has no entry for a deal-aware call id.",
            )
        )
    for call_id, entry in matched:
        themes = entry.get("theme_ids", []) + entry.get("service_ids", [])
        modes = entry.get("bot_allowed_modes", [])
        statuses = entry.get("policy_statuses", [])
        theme_text = " | ".join(themes)
        if any(PAYMENT_THEME_RE.search(theme) for theme in themes) and PAYMENT_WORD_RE.search(next_step):
            hard.append(
                finding(
                    row_index,
                    review_id,
                    deal_id,
                    "catalog_payment_theme_next_step_conflict",
                    "P1",
                    "AI-рекомендованный следующий шаг",
                    next_step,
                    f"Question catalog payment theme for call {call_id} strengthens payment next-step conflict.",
                )
            )
        if any(SERVICE_THEME_RE.search(theme) for theme in themes) and SALES_ACTION_RE.search(next_step):
            hard.append(
                finding(
                    row_index,
                    review_id,
                    deal_id,
                    "catalog_service_theme_sales_next_step",
                    "P1",
                    "AI-рекомендованный следующий шаг",
                    next_step,
                    f"Question catalog service theme conflicts with sales next step: {theme_text}.",
                )
            )
        if ("manager_only" in modes or "manager_only" in statuses) and AUTONOMOUS_ACTION_RE.search(next_step):
            hard.append(
                finding(
                    row_index,
                    review_id,
                    deal_id,
                    "catalog_manager_only_theme_autonomous_action",
                    "P1",
                    "AI-рекомендованный следующий шаг",
                    next_step,
                    f"Question catalog marks call {call_id} as manager-only, but payload proposes an autonomous action.",
                )
            )
        if any(RISKY_DECISION_THEME_RE.search(theme) for theme in themes) and PROMISE_OR_DECISION_RE.search(next_step):
            hard.append(
                finding(
                    row_index,
                    review_id,
                    deal_id,
                    "catalog_sensitive_theme_promise_conflict",
                    "P1",
                    "AI-рекомендованный следующий шаг",
                    next_step,
                    f"Sensitive catalog theme requires manual check before promises or concrete decisions: {theme_text}.",
                )
            )
    return dedupe_findings(hard), dedupe_findings(warnings)


def extract_call_ids(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    values = []
    for key in ("call_id", "source_call_id", "call_ids", "source_call_ids", "relevant_call_ids"):
        values.extend(split_tokens(row.get(key)))
        values.extend(split_tokens(payload.get(key)))
    structured_raw = row.get("structured_objections_json")
    try:
        structured = json.loads(safe_text(structured_raw) or "[]")
    except json.JSONDecodeError:
        structured = []
    if isinstance(structured, list):
        for item in structured:
            if isinstance(item, dict):
                values.extend(split_tokens(item.get("source_call_id")))
    return list(dict.fromkeys(value for value in values if value))


def finding(
    row_index: int,
    review_id: str,
    deal_id: str,
    gate_type: str,
    severity: str,
    field: str,
    matched_text: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "review_id": review_id,
        "selected_deal_id": deal_id,
        "gate_type": gate_type,
        "severity": severity,
        "field": field,
        "matched_text": matched_text,
        "reason": reason,
    }


def dedupe_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for row in rows:
        key = (
            row.get("review_id"),
            row.get("gate_type"),
            row.get("severity"),
            row.get("field"),
            row.get("matched_text"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def build_summary(
    *,
    paths: DealQualityGatePaths,
    preview_rows: list[dict[str, str]],
    report_rows: list[dict[str, Any]],
    dry_run_candidates: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, Any]],
    findings_rows: list[dict[str, Any]],
    stage4_summary: dict[str, Any],
    outputs: dict[str, Path],
    analysis_date: str,
) -> dict[str, Any]:
    hard_findings = [row for row in findings_rows if safe_text(row.get("severity")) in {"P0", "P1", "P2"}]
    warning_findings = [row for row in findings_rows if safe_text(row.get("severity")) == "P3"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_date": analysis_date,
        "input": {
            "stage4_preview_csv": str(paths.stage4_preview_root / "deal_stage4_preview.csv"),
            "stage4_preview_sha256": sha256_file(paths.stage4_preview_root / "deal_stage4_preview.csv"),
            "stage4_payloads_jsonl": str(paths.stage4_preview_root / "deal_stage4_payloads.jsonl"),
            "stage4_payloads_sha256": sha256_file(paths.stage4_preview_root / "deal_stage4_payloads.jsonl"),
        },
        "sources": {
            "stage4_preview_root": str(paths.stage4_preview_root),
            "stage4_schema_version": safe_text(stage4_summary.get("schema_version")),
        },
        "safety": {
            "read_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
        "coverage": {
            "input_rows": len(preview_rows),
            "gate_report_rows": len(report_rows),
            "stage6_dry_run_candidates": len(dry_run_candidates),
            "blocked_rows": len(blocked_rows),
            "warning_rows": len(warning_rows),
            "findings": len(findings_rows),
            "hard_findings": len(hard_findings),
            "warning_findings": len(warning_findings),
        },
        "decision_counts": dict(Counter(safe_text(row.get("stage5_decision")) for row in report_rows).most_common()),
        "hard_gate_counts": dict(Counter(safe_text(row.get("gate_type")) for row in hard_findings).most_common()),
        "warning_gate_counts": dict(Counter(safe_text(row.get("gate_type")) for row in warning_findings).most_common()),
        "readiness": {
            "stage5_quality_gate_built": True,
            "passed_for_stage6_dry_run": len(dry_run_candidates) > 0 and len(report_rows) == len(preview_rows),
            "passed_for_live_writeback": False,
            "deal_aware_stage6_live_writeback_ready": False,
            "crm_quality_writeback_ready": False,
            "safe_to_write_deal_fields": False,
            "live_write_blocker": "Stage 5 does not authorize live writeback. Stage 6 must run dry-run, audit, operator approval and readback.",
        },
        "expected_rows": {
            "total": len(report_rows),
            "stage6_dry_run_candidates": len(dry_run_candidates),
            "blocked": len(blocked_rows),
            "warnings": len(warning_rows),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def render_readme(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    return "\n".join(
        [
            "# Deal-Aware Stage 5 Quality Gate",
            "",
            "Read-only quality gate for deal-aware Stage 4 output. No AMO/Tallanto writes.",
            "",
            "## Coverage",
            "",
            f"- input rows: {coverage['input_rows']}",
            f"- Stage 6 dry-run candidates: {coverage['stage6_dry_run_candidates']}",
            f"- blocked rows: {coverage['blocked_rows']}",
            f"- warning rows: {coverage['warning_rows']}",
            f"- hard findings: {coverage['hard_findings']}",
            f"- warning findings: {coverage['warning_findings']}",
            "",
            "## Readiness",
            "",
            f"- passed for Stage 6 dry-run: {summary['readiness']['passed_for_stage6_dry_run']}",
            f"- passed for live writeback: {summary['readiness']['passed_for_live_writeback']}",
            "- Live writeback still requires Stage 6 dry-run, external audit, explicit operator approval and post-writeback readback.",
            "",
            "## Outputs",
            "",
            *[f"- `{key}`: `{path}`" for key, path in summary["outputs"].items()],
            "",
        ]
    )


def read_payload_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def read_stage4_findings_by_review_id(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = {key.lstrip("\ufeff"): value for key, value in raw.items()}
            review_id = safe_text(row.get("review_id"))
            risk_type = safe_text(row.get("risk_type"))
            if not review_id or not risk_type:
                continue
            key = (
                review_id,
                risk_type,
                safe_text(row.get("severity")),
                safe_text(row.get("field")),
                safe_text(row.get("matched_text")),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    by_review_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_review_id.setdefault(safe_text(row.get("review_id")), []).append(row)
    return by_review_id


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_pipe(value: Any) -> list[str]:
    text = safe_text(value)
    if not text:
        return []
    return [part.strip(" .;,") for part in re.split(r"\s+\|\s+|\n|;", text) if part.strip(" .;,")]


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value).replace(",", ".")))
    except ValueError:
        return 0


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
