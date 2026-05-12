from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.insights.sanitizers import has_brand_risk, has_money_or_terms_risk, has_personal_data_risk, sanitize_answer


REQUIRED_STAGE14_CHECKS = (
    "required_kb_columns_present",
    "required_rop_columns_present",
    "bot_seed_safe_columns_present",
    "no_residual_bot_safe_risks",
    "kb_no_live_revenue_risk_zero",
    "rop_p0_no_live_or_artifact_zero",
    "rop_revenue_no_live_or_artifact_zero",
    "kb_bot_ready_money_or_terms_zero",
    "rop_bot_candidate_money_or_terms_zero",
    "bot_ready_rows_have_safe_answer",
    "audit_sample_built",
)
REQUIRED_ZERO_BASELINE_RISKS = (
    "kb_no_live_revenue_risk",
    "kb_bot_ready_money_or_terms",
    "kb_ideal_answer_brand_risk",
    "kb_bot_safe_answer_brand_risk",
    "kb_bot_safe_answer_personal_data_risk",
    "rop_p0_no_live_or_artifact",
    "rop_revenue_risk_no_live_or_artifact",
    "rop_bot_candidate_money_or_terms",
    "rop_bot_safe_answer_brand_risk",
    "rop_bot_safe_answer_personal_data_risk",
)
REQUIRED_FILES = (
    "stage14_summary",
    "stage14_audit_sample",
    "stage14_residual_risk_sample",
    "stage14_over_sanitization_candidates",
    "baseline_summary",
    "kb_summary",
    "kb_bot_seeds",
    "kb_enriched_reviews",
    "rop_summary",
    "rop_validation",
    "rop_bot_drafts",
)
BOT_EXPORT_COLUMNS = (
    "source_export",
    "moment_id",
    "signal_code",
    "signal",
    "stage",
    "answer_pattern_code",
    "answer_pattern",
    "customer_question_example",
    "bot_safe_answer",
    "when_not_to_use",
    "sanitizer_status",
    "sanitizer_flags",
    "safety_brand_risk",
    "safety_money_or_terms_risk",
    "safety_personal_data_risk",
    "safety_additional_export_risk",
    "safety_additional_risk_types",
    "data_limitations",
    "quality_score",
    "outcome_code",
    "outcome",
)
BOT_EXPORT_TEXT_COLUMNS = (
    "customer_question_example",
    "bot_safe_answer",
    "when_not_to_use",
    "data_limitations",
)
FORBIDDEN_BOT_EXPORT_COLUMN_MARKERS = (
    "phone",
    "телефон",
    "manager",
    "менеджер",
    "source_filename",
    "файл",
    "raw",
    "сыр",
    "answer_manager",
    "ответ менеджера",
    "идеальный ответ",
    "ideal_answer",
    "client_name",
    "email",
)
BOT_SOURCE_SAFE_ANSWER_COLUMNS = ("Безопасный ответ для бота", "bot_safe_answer")

SPOKEN_MONEY_RE = re.compile(
    r"\b(?:пятьдесят|сорок|тридцать|двадцать|десять|пятнадцать|шестьдесят|семьдесят|восемьдесят|девяносто|сто)\s+"
    r"(?:тысяч\w*(?:\s+рубл\w*)?|рубл\w*)\b|"
    r"\b\d{1,4}\s*(?:т\.\s*р\.|тыс\.?\s*руб\.?|тыс\.?\s*р\.?)(?!\w)|"
    r"\b(?:цена|стоимость|стоит|оплата|плат[её]ж|абонемент)\D{0,24}\b\d{4,6}\b",
    re.I,
)
SPOKEN_PERCENT_RE = re.compile(
    r"\b(?:пять|десять|пятнадцать|двадцать|тридцать|сорок|пятьдесят)\s+процент(?:а|ов)?\b",
    re.I,
)
TELEGRAM_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{4,}(?!\w)")
ANY_PLACEHOLDER_RE = re.compile(r"\[(?:CURRENT_PRICE|CURRENT_DEADLINE|PAYMENT_OPTIONS|REFUND_POLICY|CLIENT_NAME|PHONE|EMAIL|COMPANY_NAME|[^\]]*(?:ЦЕН|СКИД|РАССРОЧ|ТЕЛЕФОН|ПОЧТ|ИМЯ|КЛИЕНТ)[^\]]*)\]", re.I)
BRAND_VARIANT_RE = re.compile(r"\b(?:МФТ|МФТЫ|МФТИШ?К?И?|НФК|УНФК)\b|черн[ыи]й?\s*центр|черныйцентр|чебенцентр", re.I)
LIKELY_SINGLE_NAME_RE = re.compile(
    r"\b(?:Павел|Павла|Павлу|Максим|Максима|Максиму|Роман|Романа|Роману|Сергей|Сергея|Сергею|"
    r"Елена|Елены|Елене|Наталья|Натальи|Наталье|Татьяна|Татьяны|Татьяне|Светлана|Светланы|Светлане|"
    r"Карина|Карины|Карине|Дарина|Дарины|Дарине|Екатерина|Екатерины|Екатерине|Анастасия|Анастасии|Анастасию|"
    r"Алиса|Алису|Алисы|Александр|Александра|Александру|Никита|Никиту|Никиты|Никите|Анна|Анне|Анну|"
    r"Ева|Еву|Евы|Евой|Амир|Амира|Амиру|Георгий|Георгия|Георгию|Глеб|Глеба|Глебу|Сава|Савы|Саву|"
    r"Арсений|Арсения|Арсению|Евгений|Евгения|Евгению|Паша|Пашу|Паше|Платон|Платона|Платону|"
    r"Таисия|Таисии|Таисию|Валерия|Валерии|Валерию|Лиза|Лизы|Лизе|Лизу|Вова|Вову|Вовы|"
    r"Лука|Луку|Луки|Луке|Илья|Ильи|Илье|Илью|Демид|Демида|Демиду|Антон|Антона|Антону|Федор|Федора|Федору|Фёдор|Фёдора|Фёдору|Рома|Рому|Ромы|Злата|Злату|Златы|Береву|Беревы|Маша|Маше|Машу|Елизавета|Елизаветы|Елизавете|Владислав|Владислава|Владиславу|"
    r"Агния|Агнии|Агнию|Ярослав|Ярослава|Ярославу|Константин|Константина|Константину|Тимур|Тимура|Тимуру|Аслан|Аслана|Аслану|Надежда|Надежду|Надежды|"
    r"Василий|Василия|Василию|Настя|Насте|Настю|Святослав|Святослава|Святославу|Антоний|Антония|Антонию|Виктория|Виктории|Викторию|"
    r"Иван|Ивана|Иваном|Ивану|Влад|Влада|Владу|Амир|Амира|Амиру|Амиром|Миша|Мишей|Мише|Мишу|Вячеслав|Вячеслава|Вячеславу|Ибрагимов|Ибрагимова|Ибрагимову)\b"
)


@dataclass(frozen=True)
class Stage15ExportGateConfig:
    project_root: Path
    stage14_root: Path
    kb_root: Path
    rop_root: Path
    baseline_root: Path
    out_root: Path
    min_audit_sample_rows: int = 100
    expected_audit_sample_rows: int = 200
    block_bot_production_on_over_sanitization_queue: bool = True


def build_stage15_export_quality_gate(config: Stage15ExportGateConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    paths = input_paths(config)
    missing_files = [key for key, path in paths.items() if not path.exists()]
    stage14_summary = _load_json(paths["stage14_summary"]) if paths["stage14_summary"].exists() else {}
    baseline_summary = _load_json(paths["baseline_summary"]) if paths["baseline_summary"].exists() else {}
    kb_summary = _load_json(paths["kb_summary"]) if paths["kb_summary"].exists() else {}
    rop_summary = _load_json(paths["rop_summary"]) if paths["rop_summary"].exists() else {}

    audit_rows = _read_csv(paths["stage14_audit_sample"])
    residual_rows = _read_csv(paths["stage14_residual_risk_sample"])
    over_sanitization_rows = _read_csv(paths["stage14_over_sanitization_candidates"])
    kb_bot_seed_rows = _read_csv(paths["kb_bot_seeds"])
    rop_bot_draft_rows = _read_csv(paths["rop_bot_drafts"])

    bot_export_rows, blocked_bot_rows = build_bot_export_allowlist(kb_bot_seed_rows, rop_bot_draft_rows)
    bot_export_risks = risk_counts_for_export_rows(bot_export_rows)
    source_bot_risks = {
        "kb_bot_knowledge_seeds": risk_counts_for_rows(kb_bot_seed_rows, "Безопасный ответ для бота"),
        "rop_bot_knowledge_drafts": risk_counts_for_rows(rop_bot_draft_rows, "Безопасный ответ для бота"),
    }
    audit_sample_report = audit_sample_quality(audit_rows, config)
    stage14_report = stage14_acceptance_report(stage14_summary)
    baseline_report = baseline_zero_report(baseline_summary)
    source_path_report = source_path_consistency_report(stage14_summary, config)
    bot_column_report = bot_export_column_report(bot_export_rows)
    row_count_report = {
        "kb_bot_seed_rows": len(kb_bot_seed_rows),
        "rop_bot_draft_rows": len(rop_bot_draft_rows),
        "bot_export_allowlist_rows": len(bot_export_rows),
        "blocked_bot_export_rows": len(blocked_bot_rows),
        "stage14_audit_sample_rows": len(audit_rows),
        "stage14_residual_risk_rows": len(residual_rows),
        "stage14_over_sanitization_rows": len(over_sanitization_rows),
    }

    checks = {
        "required_files_exist": not missing_files,
        "stage14_acceptance_passed": stage14_report["passed"],
        "stage14_required_checks_passed": stage14_report["required_checks_passed"],
        "stage14_residual_risk_rows_zero": len(residual_rows) == 0 and int(stage14_summary.get("residual_risk_samples", {}).get("rows") or 0) == 0,
        "baseline_required_risks_zero": baseline_report["passed"],
        "stage14_inputs_match_current_roots": source_path_report["passed"],
        "audit_sample_sufficient_and_unique": audit_sample_report["passed"],
        "source_bot_safe_answers_have_zero_risks": all(_all_zero(value) for value in source_bot_risks.values()),
        "bot_export_allowlist_non_empty": len(bot_export_rows) > 0,
        "bot_export_allowlist_has_only_safe_columns": bot_column_report["passed"],
        "bot_export_allowlist_has_zero_risks": _all_zero(bot_export_risks) and not blocked_bot_rows,
    }
    hard_gate_passed = all(checks.values())
    bot_production_ready = bool(hard_gate_passed and (not config.block_bot_production_on_over_sanitization_queue or len(over_sanitization_rows) == 0))

    outputs = {
        "summary_json": out_root / "summary.json",
        "export_gate_report_md": out_root / "STAGE15_EXPORT_GATE_REPORT.md",
        "bot_export_allowlist_csv": out_root / "bot_export_allowlist.csv",
        "bot_export_allowlist_schema_json": out_root / "bot_export_allowlist.schema.json",
        "blocked_bot_export_rows_csv": out_root / "blocked_bot_export_rows.csv",
        "export_gate_runbook_md": out_root / "EXPORT_GATE_RUNBOOK.md",
    }
    _write_csv(outputs["bot_export_allowlist_csv"], bot_export_rows, fieldnames=list(BOT_EXPORT_COLUMNS))
    _write_csv(outputs["blocked_bot_export_rows_csv"], blocked_bot_rows)
    outputs["bot_export_allowlist_schema_json"].write_text(
        json.dumps(bot_export_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "gate_version": "transcript_quality_stage15_export_gate_v2_hardened",
        "inputs": {key: str(path) for key, path in paths.items()},
        "input_fingerprints": [_fingerprint(path, project_root) for path in paths.values() if path.exists()],
        "checks": checks,
        "passed": hard_gate_passed,
        "readiness": {
            "rop_internal_export_ready": hard_gate_passed,
            "crm_quality_writeback_ready": hard_gate_passed,
            "bot_allowlist_export_ready": hard_gate_passed,
            "bot_autonomous_production_ready": bot_production_ready,
            "bot_autonomous_production_blockers": bot_production_blockers(
                hard_gate_passed=hard_gate_passed,
                over_sanitization_rows=len(over_sanitization_rows),
                block_on_over_sanitization=config.block_bot_production_on_over_sanitization_queue,
            ),
            "interpretation": (
                "Quality gate проверяет качество и безопасность экспортных данных. "
                "CRM live writeback всё равно требует отдельный staged/dry-run/live confirmation guard."
            ),
        },
        "missing_files": missing_files,
        "stage14": stage14_report,
        "baseline": baseline_report,
        "source_path_consistency": source_path_report,
        "audit_sample": audit_sample_report,
        "row_counts": row_count_report,
        "bot_export": {
            "columns": list(BOT_EXPORT_COLUMNS),
            "forbidden_column_markers": list(FORBIDDEN_BOT_EXPORT_COLUMN_MARKERS),
            "column_report": bot_column_report,
            "risk_counts": bot_export_risks,
            "source_risk_counts": source_bot_risks,
            "duplicate_policy": "Deduplicate by moment_id, otherwise by normalized question+answer. KB rows are preferred over ROP draft duplicates.",
        },
        "over_sanitization_queue": {
            "rows": len(over_sanitization_rows),
            "blocks_autonomous_bot_production": bool(config.block_bot_production_on_over_sanitization_queue and over_sanitization_rows),
            "interpretation": "Очередь проверки полезности: не ошибка качества, но до автономного бота РОП должен проверить, что sanitizer не сделал ответы слишком общими.",
        },
        "kb_summary_key_metrics": summary_key_metrics(kb_summary),
        "rop_summary_key_metrics": summary_key_metrics(rop_summary),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }

    outputs["export_gate_report_md"].write_text(markdown_report(summary), encoding="utf-8")
    outputs["export_gate_runbook_md"].write_text(runbook(summary), encoding="utf-8")
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def input_paths(config: Stage15ExportGateConfig) -> dict[str, Path]:
    return {
        "stage14_summary": config.stage14_root / "summary.json",
        "stage14_audit_sample": config.stage14_root / "audit_sample.csv",
        "stage14_residual_risk_sample": config.stage14_root / "residual_risk_sample.csv",
        "stage14_over_sanitization_candidates": config.stage14_root / "over_sanitization_candidates.csv",
        "baseline_summary": config.baseline_root / "summary.json",
        "kb_summary": config.kb_root / "summary.json",
        "kb_bot_seeds": config.kb_root / "bot_knowledge_seeds.csv",
        "kb_enriched_reviews": config.kb_root / "enriched_reviews.csv",
        "rop_summary": config.rop_root / "summary.json",
        "rop_validation": config.rop_root / "rop_validation.csv",
        "rop_bot_drafts": config.rop_root / "bot_knowledge_drafts.csv",
    }


def build_bot_export_allowlist(
    kb_bot_seed_rows: list[dict[str, str]],
    rop_bot_draft_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, source_rows in (
        ("kb_bot_knowledge_seeds", kb_bot_seed_rows),
        ("rop_bot_knowledge_drafts", rop_bot_draft_rows),
    ):
        for row in source_rows:
            source_answer = clean(first_present(row, BOT_SOURCE_SAFE_ANSWER_COLUMNS))
            source_answer_risks = export_safety_risks(source_answer)
            if any(source_answer_risks.values()):
                blocked.append(blocked_bot_row(source, row, "unsafe_source_bot_safe_answer", risks=source_answer_risks, text=source_answer))
                continue
            exported = bot_export_row(source, row)
            answer = clean(exported["bot_safe_answer"])
            if not answer:
                blocked.append(blocked_bot_row(source, row, "missing_bot_safe_answer"))
                continue
            risks = export_row_safety_risks(exported)
            if any(risks.values()):
                blocked.append(blocked_bot_row(source, row, "unsafe_bot_safe_answer", risks=risks, text=answer))
                continue
            key = bot_export_dedup_key(exported)
            if key in seen:
                continue
            seen.add(key)
            rows.append(exported)
    return rows, blocked


def bot_export_row(source: str, row: dict[str, str]) -> dict[str, Any]:
    answer = safe_bot_export_text(first_present(row, ("Безопасный ответ для бота", "bot_safe_answer")))
    question = safe_bot_export_text(
        first_present(row, ("customer_question_sanitized", "Пример вопроса клиента", "Вопрос клиента", "customer_question"))
    )
    when_not_to_use = safe_bot_export_text(first_present(row, ("Когда не использовать",)))
    return {
        "source_export": source,
        "moment_id": clean(first_present(row, ("ID момента", "moment_id"))),
        "signal_code": clean(first_present(row, ("Код сигнала", "signal"))),
        "signal": clean(first_present(row, ("Сигнал клиента", "signal_ru"))),
        "stage": clean(first_present(row, ("Стадия", "stage_ru"))),
        "answer_pattern_code": clean(first_present(row, ("Код паттерна", "answer_pattern"))),
        "answer_pattern": clean(first_present(row, ("Паттерн ответа", "answer_pattern_ru"))),
        "customer_question_example": question,
        "bot_safe_answer": answer,
        "when_not_to_use": when_not_to_use,
        "sanitizer_status": clean(first_present(row, ("Статус sanitizer", "bot_safety_status_ru", "bot_safety_status"))),
        "sanitizer_flags": clean(first_present(row, ("Флаги sanitizer", "sanitizer_flags"))),
        "safety_brand_risk": "Да" if has_brand_risk(answer) else "Нет",
        "safety_money_or_terms_risk": "Да" if has_money_or_terms_risk(answer) else "Нет",
        "safety_personal_data_risk": "Да" if has_personal_data_risk(answer) else "Нет",
        "safety_additional_export_risk": "Да" if additional_export_risk_types(answer) else "Нет",
        "safety_additional_risk_types": " | ".join(additional_export_risk_types(answer)),
        "data_limitations": clean(first_present(row, ("Ограничение данных", "data_limitations"))),
        "quality_score": clean(first_present(row, ("Оценка", "overall_quality_score"))),
        "outcome_code": clean(first_present(row, ("Код итога", "final_outcome"))),
        "outcome": clean(first_present(row, ("Итог сделки", "final_outcome_ru"))),
    }


def bot_export_dedup_key(row: dict[str, Any]) -> str:
    moment_id = clean(row.get("moment_id"))
    if moment_id:
        return f"moment:{moment_id}"
    return "qa:" + re.sub(r"\s+", " ", f"{row.get('customer_question_example', '')}\n{row.get('bot_safe_answer', '')}".lower()).strip()


def blocked_bot_row(
    source: str,
    row: dict[str, str],
    reason: str,
    *,
    risks: dict[str, bool] | None = None,
    text: str = "",
) -> dict[str, Any]:
    risks = risks or {}
    return {
        "source_export": source,
        "reason": reason,
        "moment_id": clean(first_present(row, ("ID момента", "moment_id"))),
        "risk_brand": str(bool(risks.get("brand"))),
        "risk_money_or_terms": str(bool(risks.get("money_or_terms"))),
        "risk_personal_data": str(bool(risks.get("personal_data"))),
        "risk_additional_export": str(any(bool(value) for key, value in risks.items() if key not in {"brand", "money_or_terms", "personal_data"})),
        "additional_risk_types": " | ".join(key for key, value in risks.items() if key not in {"brand", "money_or_terms", "personal_data"} and value),
        "text": text,
    }


def stage14_acceptance_report(stage14_summary: dict[str, Any]) -> dict[str, Any]:
    acceptance = stage14_summary.get("acceptance") or {}
    checks = acceptance.get("checks") or {}
    required = {key: bool(checks.get(key)) for key in REQUIRED_STAGE14_CHECKS}
    missing = [key for key in REQUIRED_STAGE14_CHECKS if key not in checks]
    failed = [key for key, value in required.items() if not value]
    return {
        "passed": bool(acceptance.get("passed")) and not failed and not missing,
        "acceptance_passed_flag": bool(acceptance.get("passed")),
        "required_checks_passed": not failed and not missing,
        "required_checks": required,
        "missing_required_checks": missing,
        "failed_required_checks": failed,
        "warnings": acceptance.get("warnings") or [],
    }


def baseline_zero_report(baseline_summary: dict[str, Any]) -> dict[str, Any]:
    risks = baseline_summary.get("baseline_risks") or {}
    values = {key: int(_to_number(risks.get(key)) or 0) for key in REQUIRED_ZERO_BASELINE_RISKS}
    non_zero = {key: value for key, value in values.items() if value != 0}
    missing = [key for key in REQUIRED_ZERO_BASELINE_RISKS if key not in risks]
    return {
        "passed": not non_zero and not missing,
        "required_zero_values": values,
        "non_zero_required_risks": non_zero,
        "missing_required_risks": missing,
        "raw_source_risks_not_blocking": {
            "kb_raw_ideal_answer_brand_risk": int(_to_number(risks.get("kb_raw_ideal_answer_brand_risk")) or 0),
            "kb_raw_ideal_answer_money_or_terms": int(_to_number(risks.get("kb_raw_ideal_answer_money_or_terms")) or 0),
            "interpretation": "Raw/source поля могут содержать риск и сохраняются для аудита. В bot/CRM export они не должны попадать.",
        },
    }


def source_path_consistency_report(stage14_summary: dict[str, Any], config: Stage15ExportGateConfig) -> dict[str, Any]:
    inputs = stage14_summary.get("inputs") or {}
    expected = {
        "after_kb_summary": (config.kb_root / "summary.json").resolve(),
        "after_kb_enriched": (config.kb_root / "enriched_reviews.csv").resolve(),
        "after_kb_bot_seeds": (config.kb_root / "bot_knowledge_seeds.csv").resolve(),
        "after_rop_summary": (config.rop_root / "summary.json").resolve(),
        "after_rop_validation": (config.rop_root / "rop_validation.csv").resolve(),
        "after_rop_bot_drafts": (config.rop_root / "bot_knowledge_drafts.csv").resolve(),
        "after_baseline_summary": (config.baseline_root / "summary.json").resolve(),
    }
    mismatches: dict[str, dict[str, str]] = {}
    for key, expected_path in expected.items():
        actual = clean(inputs.get(key))
        if not actual:
            mismatches[key] = {"actual": "", "expected": str(expected_path)}
            continue
        actual_path = Path(actual).expanduser().resolve()
        if actual_path != expected_path:
            mismatches[key] = {"actual": str(actual_path), "expected": str(expected_path)}
    return {"passed": not mismatches, "mismatches": mismatches}


def audit_sample_quality(rows: list[dict[str, str]], config: Stage15ExportGateConfig) -> dict[str, Any]:
    moment_ids = [clean(row.get("moment_id") or row.get("ID момента")) for row in rows]
    non_empty = [value for value in moment_ids if value]
    duplicates = sorted(key for key, count in Counter(non_empty).items() if count > 1)
    buckets = Counter(clean(row.get("audit_bucket")) or "unknown" for row in rows)
    sufficient = len(rows) >= config.min_audit_sample_rows
    return {
        "passed": sufficient and not duplicates,
        "rows": len(rows),
        "min_required_rows": config.min_audit_sample_rows,
        "expected_rows": config.expected_audit_sample_rows,
        "unique_moment_ids": len(set(non_empty)),
        "duplicate_moment_ids": duplicates[:50],
        "by_bucket": dict(buckets.most_common()),
    }


def bot_export_column_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = set(BOT_EXPORT_COLUMNS)
    forbidden = [column for column in columns if column_has_forbidden_marker(column)]
    row_extra_columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                row_extra_columns.append(key)
            elif column_has_forbidden_marker(key):
                forbidden.append(key)
    return {
        "passed": not forbidden and not row_extra_columns,
        "allowed_columns": list(BOT_EXPORT_COLUMNS),
        "forbidden_columns_found": sorted(set(forbidden)),
        "extra_columns_found": sorted(set(row_extra_columns)),
    }


def column_has_forbidden_marker(column: str) -> bool:
    normalized = column.lower().replace("_", " ").strip()
    return any(marker in normalized for marker in FORBIDDEN_BOT_EXPORT_COLUMN_MARKERS)


def risk_counts_for_rows(rows: list[dict[str, str]] | list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = {
        "brand": 0,
        "money_or_terms": 0,
        "personal_data": 0,
        "spoken_money_or_terms": 0,
        "messenger_handle": 0,
        "unsafe_placeholder": 0,
        "brand_variant": 0,
        "likely_single_name": 0,
        "fixpoint_not_reached": 0,
        "missing": 0,
    }
    for row in rows:
        text = clean(row.get(field))
        if not text:
            counts["missing"] += 1
            continue
        for key, value in export_safety_risks(text).items():
            counts[key] += int(value)
    return counts


def risk_counts_for_export_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = empty_risk_counts()
    for row in rows:
        risks = export_row_safety_risks(row)
        for key, value in risks.items():
            counts[key] += int(value)
    return counts


def empty_risk_counts() -> dict[str, int]:
    return {
        "brand": 0,
        "money_or_terms": 0,
        "personal_data": 0,
        "spoken_money_or_terms": 0,
        "messenger_handle": 0,
        "unsafe_placeholder": 0,
        "brand_variant": 0,
        "likely_single_name": 0,
        "fixpoint_not_reached": 0,
        "missing": 0,
    }


def export_row_safety_risks(row: dict[str, Any]) -> dict[str, bool]:
    combined = " ".join(clean(row.get(column)) for column in BOT_EXPORT_TEXT_COLUMNS)
    risks = export_safety_risks(combined)
    risks["missing"] = not clean(row.get("bot_safe_answer"))
    return risks


def export_safety_risks(text: object) -> dict[str, bool]:
    value = clean(text)
    sanitized = sanitize_answer(value, mode="bot") if value else None
    return {
        "brand": has_brand_risk(value),
        "money_or_terms": has_money_or_terms_risk(value),
        "personal_data": has_personal_data_risk(value),
        "spoken_money_or_terms": bool(SPOKEN_MONEY_RE.search(value) or SPOKEN_PERCENT_RE.search(value)),
        "messenger_handle": bool(TELEGRAM_HANDLE_RE.search(value)),
        "unsafe_placeholder": bool(ANY_PLACEHOLDER_RE.search(value)),
        "brand_variant": bool(BRAND_VARIANT_RE.search(value)),
        "likely_single_name": bool(LIKELY_SINGLE_NAME_RE.search(value)),
        "fixpoint_not_reached": bool(sanitized and sanitized.status == "fixpoint_not_reached"),
    }


def safe_bot_export_text(text: object) -> str:
    value = clean(text)
    if not value:
        return ""
    sanitized = sanitize_answer(value, mode="bot")
    if sanitized.status == "fixpoint_not_reached":
        return ""
    suffix = "Точные условия менеджер подтвердит по актуальным правилам."
    return clean(sanitized.text.replace(suffix, ""))


def additional_export_risk_types(text: object) -> tuple[str, ...]:
    risks = export_safety_risks(text)
    return tuple(
        key
        for key in ("spoken_money_or_terms", "messenger_handle", "unsafe_placeholder", "brand_variant", "likely_single_name", "fixpoint_not_reached")
        if risks.get(key)
    )


def bot_production_blockers(*, hard_gate_passed: bool, over_sanitization_rows: int, block_on_over_sanitization: bool) -> list[str]:
    blockers: list[str] = []
    if not hard_gate_passed:
        blockers.append("stage15_hard_gate_failed")
    if block_on_over_sanitization and over_sanitization_rows:
        blockers.append("over_sanitization_queue_requires_rop_review_before_autonomous_bot")
    return blockers


def bot_export_schema() -> dict[str, Any]:
    return {
        "version": "bot_export_allowlist_v1",
        "description": "Production-safe Telegram bot seed export. Use only these columns for external/autonomous bot ingestion.",
        "columns": [
            {
                "name": column,
                "required": column in {"source_export", "bot_safe_answer"},
                "description": bot_export_column_description(column),
            }
            for column in BOT_EXPORT_COLUMNS
        ],
        "forbidden_source_columns": list(FORBIDDEN_BOT_EXPORT_COLUMN_MARKERS),
        "answer_source_policy": "Only bot_safe_answer is allowed. Raw ideal answers, manager answers, phone, manager name and source files are intentionally excluded.",
    }


def bot_export_column_description(column: str) -> str:
    descriptions = {
        "source_export": "Internal artifact source: KB seeds or ROP bot drafts.",
        "moment_id": "Internal moment id for traceability, not a client identifier.",
        "signal_code": "Stable client signal taxonomy code if available.",
        "signal": "Human-readable client signal.",
        "stage": "Approximate sales/service stage.",
        "answer_pattern_code": "Stable response pattern code if available.",
        "answer_pattern": "Human-readable response pattern.",
        "customer_question_example": "Sanitized example question from client context.",
        "bot_safe_answer": "Only answer text allowed for bot ingestion.",
        "when_not_to_use": "Safety/fit limitation for bot/RAG retrieval.",
        "sanitizer_status": "Sanitizer status after Stage 13.",
        "sanitizer_flags": "Redaction/normalization flags.",
        "safety_brand_risk": "Computed Stage 15 brand/ASR artifact risk flag.",
        "safety_money_or_terms_risk": "Computed Stage 15 price/terms/deadline risk flag.",
        "safety_personal_data_risk": "Computed Stage 15 personal data risk flag.",
        "safety_additional_export_risk": "Computed Stage 15 independent detector risk flag.",
        "safety_additional_risk_types": "Specific independent detector risk types.",
        "data_limitations": "Known data limitations from call-only context.",
        "quality_score": "Original answer quality score for ranking, not shown to clients.",
        "outcome_code": "Outcome code if available, for offline ranking.",
        "outcome": "Outcome label if available, for offline ranking.",
    }
    return descriptions.get(column, "")


def summary_key_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": summary.get("generated_at", ""),
        "totals": summary.get("totals", {}),
        "quality": summary.get("quality", {}),
        "sanitizer": summary.get("sanitizer", {}),
    }


def markdown_report(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    readiness = summary["readiness"]
    row_counts = summary["row_counts"]
    baseline = summary["baseline"]
    stage14 = summary["stage14"]
    lines = [
        "# Stage 15 Export Quality Gate Report",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Gate version: `{summary['gate_version']}`",
        "",
        "## Verdict",
        "",
        f"- Hard gate passed: `{summary['passed']}`",
        f"- ROP/internal export ready: `{readiness['rop_internal_export_ready']}`",
        f"- CRM quality-writeback ready: `{readiness['crm_quality_writeback_ready']}`",
        f"- Bot allowlist export ready: `{readiness['bot_allowlist_export_ready']}`",
        f"- Autonomous bot production ready: `{readiness['bot_autonomous_production_ready']}`",
        f"- Autonomous bot blockers: `{', '.join(readiness['bot_autonomous_production_blockers']) or 'none'}`",
        "",
        "## Hard Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Key Counts",
            "",
            f"- KB bot seed rows: `{row_counts['kb_bot_seed_rows']}`",
            f"- ROP bot draft rows: `{row_counts['rop_bot_draft_rows']}`",
            f"- Bot export allowlist rows: `{row_counts['bot_export_allowlist_rows']}`",
            f"- Blocked bot export rows: `{row_counts['blocked_bot_export_rows']}`",
            f"- Stage 14 audit sample rows: `{row_counts['stage14_audit_sample_rows']}`",
            f"- Stage 14 residual risk rows: `{row_counts['stage14_residual_risk_rows']}`",
            f"- Over-sanitization queue rows: `{row_counts['stage14_over_sanitization_rows']}`",
            "",
            "## Stage 14",
            "",
            f"- Acceptance flag: `{stage14['acceptance_passed_flag']}`",
            f"- Failed required checks: `{', '.join(stage14['failed_required_checks']) or 'none'}`",
            f"- Missing required checks: `{', '.join(stage14['missing_required_checks']) or 'none'}`",
            "",
            "## Baseline Required Zero Risks",
            "",
        ]
    )
    for key, value in baseline["required_zero_values"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Output Policy",
            "",
            "Use `bot_export_allowlist.csv` for bot/RAG ingestion. Do not ingest raw `bot_knowledge_drafts.csv`, `rop_validation.csv`, `enriched_reviews.csv`, or raw ideal/manager answer columns into an autonomous bot.",
            "",
            "CRM writeback is quality-ready only if this gate passes. Live CRM writeback still requires the separate staged preview/live confirmation policy.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def runbook(summary: dict[str, Any]) -> str:
    return f"""# Stage 15 Export Gate Runbook

## What This Gate Protects

This gate prevents unsafe transcript-derived data from reaching ROP production packs, CRM writeback previews, or Telegram bot/RAG exports.

It checks:

1. Stage 14 acceptance is still green.
2. Residual bot-safe risks are zero.
3. Baseline required risk counters are zero.
4. Stage 14 input roots match the current KB/ROP/baseline roots.
5. Audit sample is large enough and has unique moment ids.
6. Bot export uses an allowlist schema and only `bot_safe_answer`, not raw/manager answer columns.

## Current Verdict

- Hard gate passed: `{summary['passed']}`
- Bot allowlist export ready: `{summary['readiness']['bot_allowlist_export_ready']}`
- Autonomous bot production ready: `{summary['readiness']['bot_autonomous_production_ready']}`

## Safe Files To Use

- `bot_export_allowlist.csv`: safe bot/RAG seed export.
- `STAGE15_EXPORT_GATE_REPORT.md`: human-readable gate report.
- `summary.json`: machine-readable gate report.

## Files Not Safe For Direct Bot Ingestion

- `bot_knowledge_drafts.csv`: internal ROP artifact; contains manager/phone/raw context.
- `rop_validation.csv`: internal ROP artifact; contains coaching and review columns.
- `enriched_reviews.csv`: internal analytics artifact; contains raw and manager answer columns.
- Any column named like raw answer, manager answer, phone, manager, file, or source filename.

## Future Production Export Order

1. Rebuild downstream layers after new R+A/backfill.
2. Rebuild transcript-quality baseline.
3. Rebuild Stage 14 comparison.
4. Run this Stage 15 gate.
5. If the gate fails, do not export to CRM/bot. Fix source generators and rerun.
6. If the gate passes, use only allowlisted bot export columns for Telegram bot/RAG.
7. For CRM, still run the existing CRM dry-run/staged preview/live confirmation flow.
8. CLI live AMO writeback must include `--quality-gate-summary <Stage15 summary.json>` in addition to `--execute-live-write --live-confirmation WRITE_AMO_LIVE`.

## Command Template

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_transcript_quality_stage15_gate.py \\
  --project-root . \\
  --stage14-root stable_runtime/transcript_quality_stage14_comparison_20260510_v1 \\
  --kb-root stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v3_stage13_sanitized \\
  --rop-root stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v3_stage13_sanitized \\
  --baseline-root stable_runtime/transcript_quality_baseline_after_quality_backfill_20260510_v3_stage13_sanitized \\
  --out-root stable_runtime/transcript_quality_stage15_export_gate_20260510_v1
```

## Human Review Still Needed

`over_sanitization_candidates.csv` is not a safety failure. It is a usefulness queue: ROP should review whether answers became too generic before autonomous bot production.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except csv.Error:
        return []


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path: Path, project_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        relative = str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        relative = str(path.resolve())
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest}


def first_present(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = clean(row.get(key))
        if value:
            return value
    return ""


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _to_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _all_zero(values: dict[str, int]) -> bool:
    return all(int(value or 0) == 0 for key, value in values.items() if key != "missing")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 15 permanent export quality gate.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage14-root", type=Path, required=True)
    parser.add_argument("--kb-root", type=Path, required=True)
    parser.add_argument("--rop-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--min-audit-sample-rows", type=int, default=100)
    parser.add_argument("--expected-audit-sample-rows", type=int, default=200)
    parser.add_argument(
        "--allow-autonomous-bot-with-over-sanitization-queue",
        action="store_true",
        help="Do not block autonomous bot readiness on non-empty over-sanitization usefulness queue.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Stage15ExportGateConfig:
    project_root = args.project_root.expanduser().resolve()

    def resolve(path: Path) -> Path:
        return (project_root / path).resolve() if not path.is_absolute() else path.expanduser().resolve()

    return Stage15ExportGateConfig(
        project_root=project_root,
        stage14_root=resolve(args.stage14_root),
        kb_root=resolve(args.kb_root),
        rop_root=resolve(args.rop_root),
        baseline_root=resolve(args.baseline_root),
        out_root=resolve(args.out_root),
        min_audit_sample_rows=args.min_audit_sample_rows,
        expected_audit_sample_rows=args.expected_audit_sample_rows,
        block_bot_production_on_over_sanitization_queue=not args.allow_autonomous_bot_with_over_sanitization_queue,
    )


__all__ = [
    "BOT_EXPORT_COLUMNS",
    "REQUIRED_STAGE14_CHECKS",
    "REQUIRED_ZERO_BASELINE_RISKS",
    "Stage15ExportGateConfig",
    "build_bot_export_allowlist",
    "build_stage15_export_quality_gate",
    "config_from_args",
    "parse_args",
]
