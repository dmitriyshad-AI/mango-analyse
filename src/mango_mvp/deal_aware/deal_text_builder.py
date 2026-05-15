from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.deal_aware.stage1_snapshot import quote_ident, read_csv, safe_text, stringify, write_csv
from mango_mvp.quality.crm_text_quality_detector import (
    CrmTextQualityFinding,
    detect_crm_text_quality_batch_risks,
    detect_crm_text_quality_risks,
)


SCHEMA_VERSION = "deal_aware_stage4_preview_v1"

DEAL_AI_REQUIRED_FIELDS = (
    "AI-сводка по сделке",
    "AI-история по сделке",
    "AI-рекомендованный следующий шаг",
    "AI-дата следующего касания",
    "AI-фактический статус сделки",
    "AI-приоритет сделки",
    "AI-актуальные возражения",
    "AI-основание рекомендации",
    "AI-качество привязки к сделке",
    "AI-предупреждение по сделке",
    "AI-Tallanto статус по сделке",
    "AI-дата обновления сделки",
)
DEAL_AI_OPTIONAL_FIELDS = (
    "AI-бюджет диапазон",
    "AI-бюджет комментарий",
    "AI-чувствительность к цене",
    "AI-интерес к скидке",
)
DEAL_AI_FIELDS = DEAL_AI_REQUIRED_FIELDS

BUDGET_RANGES = (
    "unknown",
    "under_30k",
    "30k_50k",
    "50k_100k",
    "100k_150k",
    "over_150k",
    "matcapital_or_certificate",
    "installment_needed",
    "not_applicable",
)

BLOCKING_QUALITY_SEVERITIES = {"P0", "P1", "P2"}

TENANT_TERM_REPLACEMENTS = (
    (re.compile(r"\b(?:МПК|НПК|УМПК)\s+МФТИ\b", re.I), "УНПК МФТИ"),
    (re.compile(r"\bУНП\s+К\s+МФТИ\b", re.I), "УНПК МФТИ"),
    (re.compile(r"\bлетн(?:ая|ие|ых|ым|ую)?\s+ночн(?:ая|ые|ых|ым|ую)?\s+школ", re.I), "летняя очная школ"),
)
RELATIVE_TIME_RE = re.compile(r"\b(?:сегодня|завтра|послезавтра|до\s+конца\s+дня|после\s+18:00)\b", re.I)
PAYMENT_NEXT_STEP_RE = re.compile(r"\b(?:оплат\w*|плат[её]ж\w*|чек\w*|квитанц\w*|договор\w*)\b", re.I)
COMPLETED_PAYMENT_EVIDENCE_RE = re.compile(
    r"\b(?:чек\w*|квитанц\w*|подтверждени\w+)\s+(?:уже\s+)?(?:прислан\w*|получен\w*|отправлен\w*)|"
    r"\b(?:оплат\w+|плат[её]ж\w+)\s+(?:уже\s+)?(?:получен\w*|поступил\w*|внес[её]н\w*|подтвержден\w*)|"
    r"\b(?:оплатил\w*|оплатил[аи]?|оплатили|оплачен\w*|оплачена|оплачено|заплатил\w*)\b|"
    r"\bсделк\w+\s+(?:закрыт\w+|успешн\w+\s+закрыт\w+|оплачен\w+)",
    re.I,
)
SERVICE_FEEDBACK_RE = re.compile(
    r"\b(?:жалоб\w*|претензи\w*|обратн\w+\s+связ|замечани\w*|качество\s+обучени\w*|"
    r"\bдз\b|провер\w+\s+(?:дз|домашн\w*)|"
    r"успеваемост\w*)\b",
    re.I,
)
SERVICE_CONTEXT_WORD_RE = re.compile(r"\b(?:преподавател\w*|куратор\w*|урок\w*|заняти\w*|обучени\w*)\b", re.I)
SERVICE_PROBLEM_WORD_RE = re.compile(r"\b(?:не\s+провер\w*|нет\s+провер\w*|плохо|замечани\w*|проблем\w*)\b", re.I)
PAYMENT_AND_RECEIPT_RE = re.compile(r"\b(?:плат[её]жк\w*|сч[её]т\w*|реквизит\w*)\b[^.]{0,80}\bчек\w*\s+после\s+оплат", re.I)
CUSTOMER_SIDE_WAIT_RE = re.compile(
    r"\bклиент\w*\s+(?:передаст|обсудит|посмотрит|подумает|перезвонит|свяжется|верн[её]тся)\b|"
    r"\b(?:мама|папа|родител\w*)\s+(?:перезвонит|свяжется|обсудит)\b",
    re.I,
)
PAYMENT_DEADLINE_RE = re.compile(
    r"\b(?:до|к)\s+\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b|"
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b",
    re.I,
)
DISCOUNT_INTEREST_RE = re.compile(r"\b(?:скидк\w*|рассрочк\w*|акци\w*|промокод\w*|льгот\w*)\b", re.I)
DISCOUNT_PROMISE_RE = re.compile(r"\b(?:обеща\w*|гарантир\w*|точно|дадим|дать|предостав\w*)\b.{0,40}\bскидк\w*", re.I)
COURSE_PRICE_CONTEXT_RE = re.compile(
    r"\b(?:стоимость|цена|стоит)\s+(?:курс\w*|программ\w*|заняти\w*|обучени\w*)|"
    r"\b(?:курс\w*|программ\w*|заняти\w*|обучени\w*)\s+(?:стоит|стоимость|цена)",
    re.I,
)
VAGUE_NEXT_STEP_RE = re.compile(
    r"\b(?:перезвонить|связаться|созвониться|вернуться)\s*(?:позже|потом|в\s+мае|летом)?\b",
    re.I,
)
CLOSURE_OR_PASSIVE_NEXT_STEP_RE = re.compile(
    r"\b(?:дождаться|ожидать\s+решени\w+|не\s+беспокоить|не\s+звонить|ждать\s+обращени\w+|"
    r"клиент\w*\s+сам\w*\s+(?:свяж\w+|перезвон\w+|обрат\w+)|сам\w*\s+(?:свяж\w+|перезвон\w+|обрат\w+))\b",
    re.I,
)
CONCRETE_DATE_RE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|"
    r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    re.I,
)
CALL_BOILERPLATE_RE = re.compile(
    r"^\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}\s+менеджер\s+[^.]{1,120}?\s+общал[а-я]+\s+с\s+клиентом\.\s*",
    re.I,
)
DATE_PREFIX_RE = re.compile(r"^(\d{2}\.\d{2}\.\d{4})")


@dataclass(frozen=True)
class DealTextPaths:
    stage1_snapshot_root: Path
    stage3_deal_state_root: Path
    out_root: Path
    analysis_date: str = "2026-05-13"


def build_deal_text_preview(paths: DealTextPaths) -> dict[str, Any]:
    paths.out_root.mkdir(parents=True, exist_ok=True)

    candidates = read_csv(paths.stage3_deal_state_root / "deal_stage4_deal_candidates.csv")
    policy_rows = read_csv(paths.stage3_deal_state_root / "deal_call_writeback_policy.csv")
    calls = read_csv(paths.stage1_snapshot_root / "call_snapshot.csv")
    phone_rollup = read_csv(paths.stage1_snapshot_root / "phone_rollup.csv")
    tallanto_students = read_csv(paths.stage1_snapshot_root / "tallanto_students_snapshot.csv")
    writeoff_summary = read_csv(paths.stage1_snapshot_root / "tallanto_writeoff_summary_by_student.csv")
    stage1_summary = load_json(paths.stage1_snapshot_root / "summary.json")
    stage3_summary = load_json(paths.stage3_deal_state_root / "summary.json")

    call_by_id = {safe_text(row.get("call_id")): row for row in calls if safe_text(row.get("call_id"))}
    policy_by_deal = group_policy_rows(policy_rows)
    phone_rollup_by_phone = {safe_text(row.get("phone")): row for row in phone_rollup if safe_text(row.get("phone"))}
    student_by_tallanto_id = {
        safe_text(row.get("tallanto_id")): row for row in tallanto_students if safe_text(row.get("tallanto_id"))
    }
    writeoff_by_barcode = {
        safe_text(row.get("barcode")): row for row in writeoff_summary if safe_text(row.get("barcode"))
    }

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preview_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    quality_findings: list[dict[str, Any]] = []
    blocked_quality: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        deal_id = safe_text(candidate.get("selected_deal_id"))
        deal_policy_rows = policy_by_deal.get(deal_id, [])
        full_calls = hydrate_policy_calls(deal_policy_rows, call_by_id)
        tallanto_context = build_tallanto_context(
            candidate,
            phone_rollup_by_phone=phone_rollup_by_phone,
            student_by_tallanto_id=student_by_tallanto_id,
            writeoff_by_barcode=writeoff_by_barcode,
        )
        payload = build_deal_payload(
            candidate,
            deal_policy_rows,
            full_calls,
            tallanto_context=tallanto_context,
            generated_at=generated_at,
            analysis_date=paths.analysis_date,
        )
        row_findings = detect_crm_text_quality_risks(
            quality_payload(candidate, payload),
            analysis_date=paths.analysis_date,
            min_severity="P3",
            compact_max_chars=1800,
            verbose_max_chars=3500,
        )
        blocking = [finding for finding in row_findings if finding.severity in BLOCKING_QUALITY_SEVERITIES]
        quality_passed = not blocking
        risk_types = sorted({finding.risk_type for finding in row_findings})
        base_preview = build_preview_row(
            index=index,
            candidate=candidate,
            payload=payload,
            tallanto_context=tallanto_context,
            row_findings=row_findings,
            quality_passed=quality_passed,
        )
        preview_rows.append(base_preview)
        payload_rows.append(
            {
                "review_id": base_preview["review_id"],
                "selected_deal_id": deal_id,
                "selected_deal_name": safe_text(candidate.get("selected_deal_name")),
                "deal_writeback_mode": safe_text(candidate.get("deal_writeback_mode")),
                "crm_text_quality_passed": "Да" if quality_passed else "Нет",
                "quality_risk_types": " | ".join(risk_types),
                "payload": payload,
            }
        )
        quality_findings.extend(findings_to_rows(base_preview["review_id"], deal_id, row_findings))
        if blocking:
            blocked_quality.append(base_preview)

    batch_findings = detect_crm_text_quality_batch_risks(
        [quality_payload(row, {field: row.get(field, "") for field in DEAL_AI_REQUIRED_FIELDS + DEAL_AI_OPTIONAL_FIELDS}) for row in preview_rows],
        analysis_date=paths.analysis_date,
        min_severity="P3",
    )
    quality_findings.extend(findings_to_rows("batch", "", batch_findings))
    rop_review_sample = build_rop_review_sample(preview_rows)

    outputs = {
        "preview_csv": paths.out_root / "deal_stage4_preview.csv",
        "ready_for_audit_csv": paths.out_root / "deal_stage4_ready_for_audit.csv",
        "blocked_quality_csv": paths.out_root / "deal_stage4_blocked_quality.csv",
        "quality_findings_csv": paths.out_root / "deal_stage4_quality_findings.csv",
        "rop_review_50_csv": paths.out_root / "deal_stage4_rop_review_50.csv",
        "payloads_jsonl": paths.out_root / "deal_stage4_payloads.jsonl",
        "sqlite": paths.out_root / "deal_aware_stage4_preview.sqlite",
        "summary_json": paths.out_root / "summary.json",
        "readme": paths.out_root / "README.md",
    }

    ready_for_audit = [
        row for row in preview_rows if row.get("crm_text_quality_passed") == "Да"
    ]
    write_csv(outputs["preview_csv"], preview_rows)
    write_csv(outputs["ready_for_audit_csv"], ready_for_audit)
    write_csv(outputs["blocked_quality_csv"], blocked_quality)
    write_csv(outputs["quality_findings_csv"], quality_findings)
    write_csv(outputs["rop_review_50_csv"], rop_review_sample)
    write_jsonl(outputs["payloads_jsonl"], payload_rows)
    write_sqlite(
        outputs["sqlite"],
        {
            "deal_stage4_preview": preview_rows,
            "deal_stage4_ready_for_audit": ready_for_audit,
            "deal_stage4_blocked_quality": blocked_quality,
            "deal_stage4_quality_findings": quality_findings,
            "deal_stage4_rop_review_50": rop_review_sample,
        },
    )

    summary = build_summary(
        paths=paths,
        candidates=candidates,
        preview_rows=preview_rows,
        ready_for_audit=ready_for_audit,
        blocked_quality=blocked_quality,
        quality_findings=quality_findings,
        rop_review_sample=rop_review_sample,
        stage1_summary=stage1_summary,
        stage3_summary=stage3_summary,
        outputs=outputs,
    )
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["readme"].write_text(render_readme(summary), encoding="utf-8")
    return summary


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = safe_text(row.get(key))
        if value:
            return value
    return ""


def classify_budget_range(raw_budget: Any, context: str = "") -> str:
    text = safe_text(raw_budget)
    combined = " ".join([text, safe_text(context)])
    if not text:
        return ""
    if COURSE_PRICE_CONTEXT_RE.search(combined):
        return "not_applicable"
    lowered = text.casefold()
    if re.search(r"\b(?:маткапитал|материнск\w+\s+капитал|сертификат)\b", lowered):
        return "matcapital_or_certificate"
    if "рассроч" in lowered:
        return "installment_needed"
    numbers = extract_money_numbers(text)
    if not numbers:
        return "unknown"
    amount = max(numbers)
    if amount < 30000:
        return "under_30k"
    if amount <= 50000:
        return "30k_50k"
    if amount <= 100000:
        return "50k_100k"
    if amount <= 150000:
        return "100k_150k"
    return "over_150k"


def extract_money_numbers(text: str) -> list[int]:
    values = []
    for match in re.finditer(r"(?<!\d)(\d[\d\s]{1,8})(?:[,.]\d+)?\s*(к|тыс\.?|т\.?р\.?|руб(?:\.|лей|ля|ль)?)?", text, re.I):
        raw = re.sub(r"\s+", "", match.group(1))
        if not raw.isdigit():
            continue
        value = int(raw)
        suffix = safe_text(match.group(2)).casefold()
        if suffix.startswith(("к", "тыс", "т")) and value < 1000:
            value *= 1000
        if value >= 1000:
            values.append(value)
    return values


def normalize_price_sensitivity(value: Any, context: str = "") -> str:
    text = " ".join([safe_text(value), safe_text(context)]).casefold()
    explicit = safe_text(value).casefold()
    if explicit in {"high", "medium", "low", "unknown"}:
        return explicit
    if not text:
        return ""
    if re.search(r"\b(?:дорого|цена\s+высокая|не\s+потян\w*|нет\s+денег|дороговат\w*)\b", text):
        return "high"
    if re.search(r"\b(?:сравнива\w+\s+цен|важна\s+цена|зависит\s+от\s+цен)\b", text):
        return "medium"
    if re.search(r"\b(?:цена\s+устраивает|по\s+цене\s+ок|бюджет\s+есть)\b", text):
        return "low"
    return "unknown" if explicit else ""


def normalize_discount_interest(value: Any, context: str = "") -> str:
    text = " ".join([safe_text(value), safe_text(context)]).casefold()
    explicit = safe_text(value).casefold()
    if explicit in {"yes", "да", "true", "1", "интересуется", "нужна"}:
        return "yes"
    if explicit in {"no", "нет", "false", "0"}:
        return "no"
    if DISCOUNT_INTEREST_RE.search(text):
        return "yes"
    return "unknown" if explicit else ""


def build_commercial_payload(calls_sorted: list[dict[str, str]], candidate: dict[str, str]) -> dict[str, str]:
    rows = list(reversed(calls_sorted)) + [candidate]
    latest_budget = ""
    budget_context = ""
    latest_price_sensitivity = ""
    latest_discount_interest = ""
    seen_budget_ranges: set[str] = set()
    for row in rows:
        context = " ".join(
            [
                safe_text(row.get("call_summary") or row.get("latest_call_summary")),
                safe_text(row.get("next_step") or row.get("latest_call_next_step")),
                safe_text(row.get("objections") or row.get("latest_call_objections")),
            ]
        )
        budget = first_nonempty(
            row,
            (
                "budget",
                "client_budget",
                "budget_text",
                "commercial_budget",
                "Бюджет",
                "Бюджет клиента",
                "Коммерческий бюджет",
            ),
        )
        if budget:
            budget_range = classify_budget_range(budget, context)
            if budget_range:
                seen_budget_ranges.add(budget_range)
            if not latest_budget:
                latest_budget = budget
                budget_context = context
        if not latest_price_sensitivity:
            latest_price_sensitivity = first_nonempty(
                row,
                (
                    "price_sensitivity",
                    "price_objection",
                    "Чувствительность к цене",
                    "Ценовая чувствительность",
                ),
            )
        if not latest_discount_interest:
            latest_discount_interest = first_nonempty(
                row,
                (
                    "discount_interest",
                    "installment_interest",
                    "Интерес к скидке",
                    "Интерес к рассрочке",
                ),
            )
        if latest_budget and latest_price_sensitivity and latest_discount_interest:
            break
    conflict = len({item for item in seen_budget_ranges if item not in {"unknown", "not_applicable"}}) > 1
    payload: dict[str, str] = {}
    if latest_budget:
        budget_range = classify_budget_range(latest_budget, budget_context)
        payload["AI-бюджет диапазон"] = budget_range or "unknown"
        if budget_range in {"unknown", "matcapital_or_certificate", "installment_needed"} or conflict:
            comment = latest_budget
            if conflict:
                comment = f"{comment}; есть разные бюджетные сигналы в истории"
            payload["AI-бюджет комментарий"] = comment
    price_sensitivity = normalize_price_sensitivity(latest_price_sensitivity)
    if price_sensitivity:
        payload["AI-чувствительность к цене"] = price_sensitivity
    discount_interest = normalize_discount_interest(latest_discount_interest, " ".join(safe_text(row.get("call_summary")) for row in rows))
    if discount_interest:
        payload["AI-интерес к скидке"] = discount_interest
    return payload


def build_deal_payload(
    candidate: dict[str, str],
    policy_rows: list[dict[str, str]],
    calls: list[dict[str, str]],
    *,
    tallanto_context: dict[str, Any],
    generated_at: str,
    analysis_date: str,
) -> dict[str, str]:
    mode = safe_text(candidate.get("deal_writeback_mode"))
    status = safe_text(candidate.get("selected_status_name"))
    pipeline = safe_text(candidate.get("selected_pipeline_name"))
    deal_name = safe_text(candidate.get("selected_deal_name"))
    calls_sorted = sorted(calls, key=lambda row: safe_text(row.get("started_at")))
    latest = calls_sorted[-1] if calls_sorted else candidate
    call_count = len(calls_sorted) or int_or_zero(candidate.get("candidate_call_count"))
    phone_count = int_or_zero(candidate.get("candidate_phone_count"))
    last_call_at = safe_text(candidate.get("last_call_at") or latest.get("started_at"))
    latest_summary = normalize_manager_text(safe_text(latest.get("call_summary") or candidate.get("latest_call_summary")))
    latest_next_step = normalize_manager_text(safe_text(latest.get("next_step") or candidate.get("latest_call_next_step")))
    all_call_text = " ".join(
        " ".join([safe_text(row.get("call_summary")), safe_text(row.get("next_step"))]) for row in calls_sorted
    )
    commercial_payload = build_commercial_payload(calls_sorted, candidate)
    has_completed_payment_evidence = bool(
        COMPLETED_PAYMENT_EVIDENCE_RE.search(all_call_text)
    )
    latest_service_feedback = is_service_feedback_context(latest_summary)

    payload = {
        "AI-сводка по сделке": build_deal_summary(
            deal_name=deal_name,
            pipeline=pipeline,
            status=status,
            mode=mode,
            call_count=call_count,
            last_call_at=last_call_at,
            latest_summary=latest_summary,
            latest_next_step=latest_next_step,
            has_completed_payment_evidence=has_completed_payment_evidence,
            latest_service_feedback=latest_service_feedback,
        ),
        "AI-история по сделке": build_deal_history(calls_sorted),
        "AI-рекомендованный следующий шаг": build_next_step(
            candidate,
            latest,
            calls_sorted,
            mode=mode,
            status=status,
            has_completed_payment_evidence=has_completed_payment_evidence,
            latest_service_feedback=latest_service_feedback,
        ),
        "AI-дата следующего касания": build_followup_hint(candidate, mode=mode, status=status, analysis_date=analysis_date),
        "AI-фактический статус сделки": build_actual_status(candidate, mode=mode),
        "AI-приоритет сделки": build_priority(
            candidate,
            latest,
            mode=mode,
            status=status,
            has_completed_payment_evidence=has_completed_payment_evidence,
        ),
        "AI-актуальные возражения": build_objections(calls_sorted, candidate),
        "AI-основание рекомендации": build_recommendation_reason(
            candidate,
            latest,
            mode=mode,
            call_count=call_count,
            has_completed_payment_evidence=has_completed_payment_evidence,
            latest_service_feedback=latest_service_feedback,
        ),
        "AI-качество привязки к сделке": build_binding_quality(candidate, policy_rows, phone_count=phone_count),
        "AI-предупреждение по сделке": build_warnings(
            candidate,
            mode=mode,
            phone_count=phone_count,
            tallanto_context=tallanto_context,
            status=status,
        ),
        "AI-Tallanto статус по сделке": safe_text(tallanto_context.get("text")),
        "AI-дата обновления сделки": generated_at,
    }
    payload.update(commercial_payload)
    return {field: fit_text(normalize_manager_text(value), field_limit(field)) for field, value in payload.items()}


def build_deal_summary(
    *,
    deal_name: str,
    pipeline: str,
    status: str,
    mode: str,
    call_count: int,
    last_call_at: str,
    latest_summary: str,
    latest_next_step: str = "",
    has_completed_payment_evidence: bool = False,
    latest_service_feedback: bool = False,
) -> str:
    topic = infer_deal_topic(" ".join([deal_name, latest_summary]))
    parts = [
        f"Сделка: {normalize_manager_text(deal_name) or 'без названия'}.",
        f"Суть: {topic}.",
        f"Последний релевантный контакт: {format_date(last_call_at) or 'дата не указана'}.",
    ]
    if has_completed_payment_evidence:
        parts.append("Есть признаки, что оплата или чек уже были; перед дожимом нужно сверить AMO/Tallanto и документы.")
    elif latest_service_feedback and mode == "full_active":
        parts.append("Последний звонок похож на сервисную обратную связь по обучению, поэтому коммерческий статус сделки нужно сверить вручную.")
    if mode != "full_active":
        parts.append("Режим только контекст: без повторного коммерческого дожима.")
    if call_count > 1:
        parts.append(f"В истории сделки {call_count} релевантных звонков; детали ниже в хронологии.")
    return " ".join(part for part in parts if part)


def build_deal_history(calls: list[dict[str, str]], *, max_calls: int = 12) -> str:
    if not calls:
        return "Релевантные содержательные звонки для этой сделки не найдены."
    selected = calls[-max_calls:]
    prefix = ""
    if len(calls) > max_calls:
        prefix = (
            f"Показаны последние {max_calls} из {len(calls)} релевантных звонков; "
            "более ранние звонки оставлены в источнике, чтобы не перегружать карточку. "
        )
    lines = []
    for row in selected:
        date_text = format_date(safe_text(row.get("started_at"))) or "дата не указана"
        manager = safe_text(row.get("manager_name")) or "менеджер не указан"
        summary = history_call_summary(safe_text(row.get("call_summary")), max_sentences=2, max_chars=320)
        next_step = safe_text(row.get("next_step"))
        if next_step and not RELATIVE_TIME_RE.search(next_step):
            summary = f"{summary} Следующий шаг: {clean_next_step_text(next_step)}"
        lines.append(f"{date_text} - {manager}: {summary}".strip())
    return prefix + "\n".join(lines)


def build_next_step(
    candidate: dict[str, str],
    latest: dict[str, str],
    calls: list[dict[str, str]],
    *,
    mode: str,
    status: str,
    has_completed_payment_evidence: bool | None = None,
    latest_service_feedback: bool | None = None,
) -> str:
    status_cf = status.casefold()
    if mode == "context_only_paid_or_success":
        return "Проверить сервисный статус, документы и учебный маршрут; не делать коммерческий дожим без ручной сверки."

    raw = safe_text(latest.get("next_step") or candidate.get("latest_call_next_step"))
    raw_cf = raw.casefold()
    all_call_text = " ".join(safe_text(row.get("call_summary")) for row in calls)
    if has_completed_payment_evidence is None:
        has_completed_payment_evidence = bool(COMPLETED_PAYMENT_EVIDENCE_RE.search(all_call_text))
    if latest_service_feedback is None:
        latest_service_feedback = is_service_feedback_context(safe_text(latest.get("call_summary")))
    if latest_service_feedback and ("ожидание оплаты" in status_cf or "заключение договора" in status_cf):
        return "Передать обратную связь куратору или ответственному за обучение, зафиксировать ответ родителю и отдельно проверить актуальность сделки."
    if has_completed_payment_evidence and PAYMENT_NEXT_STEP_RE.search(raw):
        return "Сверить текущий финансовый статус в AMO/Tallanto и после этого согласовать следующий сервисный шаг."
    if has_completed_payment_evidence:
        topic = infer_deal_topic(" ".join([safe_text(candidate.get("selected_deal_name")), safe_text(latest.get("call_summary"))]))
        return f"Проверить поступление оплаты и документы по направлению «{topic}» в AMO/Tallanto; если всё подтверждено, обновить статус сделки."
    if CLOSURE_OR_PASSIVE_NEXT_STEP_RE.search(raw):
        return "Поставить сделку на ручной контроль и не делать активный дожим без нового сигнала клиента."
    if PAYMENT_AND_RECEIPT_RE.search(raw):
        return "Выслать клиенту реквизиты или счет для оплаты; после поступления оплаты проконтролировать отправку чека."
    if manager_action := manager_action_from_next_step(raw):
        return manager_action
    if "ожидание оплаты" in status_cf:
        if raw and not RELATIVE_TIME_RE.search(raw):
            if PAYMENT_NEXT_STEP_RE.search(raw) or PAYMENT_DEADLINE_RE.search(raw) or len(raw) >= 25:
                return clean_next_step_text(raw)
        return "Проверить, поступила ли оплата; если нет — связаться с клиентом и зафиксировать конкретную дату оплаты."
    if "заключение договора" in status_cf:
        if raw and not RELATIVE_TIME_RE.search(raw):
            return clean_next_step_text(raw)
        return "Проверить готовность документов и подтвердить клиенту ближайшее действие по оформлению."
    if not raw or RELATIVE_TIME_RE.search(raw):
        return "Связаться с клиентом и уточнить актуальный интерес, выбранную программу и ближайшее решение."
    if VAGUE_NEXT_STEP_RE.search(raw) and not CONCRETE_DATE_RE.search(raw):
        return "Поставить ручной контроль с конкретной датой в задаче AMO; не делать автоматический дожим без новой информации."
    if raw.strip().casefold() in {"отправить материалы", "выслать материалы", "прислать материалы"}:
        return "Отправить клиенту материалы по программе в канал текущей переписки и зафиксировать отправку в AMO."
    if "связаться с клиентом с предложением" in raw_cf:
        return "Связаться с клиентом с конкретным предложением по актуальной программе и зафиксировать результат."
    return clean_next_step_text(raw)


def build_followup_hint(candidate: dict[str, str], *, mode: str, status: str, analysis_date: str) -> str:
    if mode == "context_only_paid_or_success":
        return "Без автоматического коммерческого касания; только при сервисном поводе."
    base = parse_date_only(analysis_date) or datetime(2026, 5, 13, tzinfo=timezone.utc)
    deal_id = safe_text(candidate.get("selected_deal_id"))
    spread = stable_mod(deal_id, 5)
    status_cf = status.casefold()
    if "ожидание оплаты" in status_cf or "заключение договора" in status_cf:
        due = base + timedelta(days=1 + stable_mod(deal_id, 2))
    elif "принимают решение" in status_cf or "переговор" in status_cf:
        due = base + timedelta(days=2 + spread)
    else:
        due = base + timedelta(days=4 + spread)
    return due.date().isoformat()


def build_actual_status(candidate: dict[str, str], *, mode: str) -> str:
    status = safe_text(candidate.get("selected_status_name")) or "статус не указан"
    pipeline = safe_text(candidate.get("selected_pipeline_name")) or "воронка не указана"
    loss_reason = safe_text(candidate.get("selected_loss_reason"))
    mode_text = "рабочая активная сделка" if mode == "full_active" else "оплаченная или успешная сделка, только контекст"
    parts = [f"Фактически: {mode_text}.", f"AMO-статус: {pipeline} / {status}."]
    if loss_reason:
        parts.append(f"Причина отказа в AMO: {loss_reason}.")
    return " ".join(parts)


def build_priority(
    candidate: dict[str, str],
    latest: dict[str, str],
    *,
    mode: str,
    status: str,
    has_completed_payment_evidence: bool = False,
) -> str:
    if mode == "context_only_paid_or_success":
        return "service-paid"
    raw_next_step = safe_text(latest.get("next_step") or candidate.get("latest_call_next_step"))
    status_cf = status.casefold()
    if is_service_feedback_context(safe_text(latest.get("call_summary"))) and (
        "ожидание оплаты" in status_cf or "заключение договора" in status_cf
    ):
        return "review"
    if has_completed_payment_evidence:
        return "review"
    if CLOSURE_OR_PASSIVE_NEXT_STEP_RE.search(raw_next_step):
        return "review"
    if not raw_next_step or RELATIVE_TIME_RE.search(raw_next_step):
        return "review"
    if VAGUE_NEXT_STEP_RE.search(raw_next_step) and not CONCRETE_DATE_RE.search(raw_next_step):
        return "review"
    status_cf = status.casefold()
    if "ожидание оплаты" in status_cf or "заключение договора" in status_cf:
        return "hot"
    next_step = safe_text(latest.get("next_step") or candidate.get("latest_call_next_step")).casefold()
    if PAYMENT_NEXT_STEP_RE.search(next_step):
        return "hot"
    if "принимают решение" in status_cf or "переговор" in status_cf:
        return "warm"
    return "warm"


def build_objections(calls: list[dict[str, str]], candidate: dict[str, str]) -> str:
    items: list[str] = []
    for row in calls:
        for part in split_pipe(safe_text(row.get("objections"))):
            normalized = normalize_objection(part)
            if normalized:
                items.append(normalized)
    if not items:
        for part in split_pipe(safe_text(candidate.get("latest_call_summary"))):
            normalized = normalize_objection(part)
            if normalized:
                items.append(normalized)
    deduped = dedupe_preserve_order(items)
    if not deduped:
        return "Актуальные возражения в релевантных звонках не выделены."
    return "Актуальные: " + "; ".join(deduped[:8]) + "."


def build_recommendation_reason(
    candidate: dict[str, str],
    latest: dict[str, str],
    *,
    mode: str,
    call_count: int,
    has_completed_payment_evidence: bool = False,
    latest_service_feedback: bool = False,
) -> str:
    last_call = format_date(safe_text(candidate.get("last_call_at") or latest.get("started_at"))) or "последний звонок без даты"
    if mode == "context_only_paid_or_success":
        return (
            f"Основание: {call_count} связанных звонков, последний содержательный контакт {last_call}. "
            "Чтобы не запрашивать уже завершенное действие повторно, рекомендация ограничена сервисной сверкой."
        )
    if has_completed_payment_evidence:
        return (
            f"Основание: в связанных звонках есть признаки уже совершенной оплаты или отправленного чека; "
            f"последний содержательный контакт {last_call}. Сначала нужна финансовая сверка, а не новый дожим."
        )
    if latest_service_feedback:
        return (
            f"Основание: последний содержательный контакт {last_call} похож на сервисную обратную связь по обучению. "
            "Перед коммерческим действием нужно сверить, относится ли этот разговор к текущей стадии сделки."
        )
    return (
        f"Основание: {call_count} связанных звонков, последний содержательный контакт {last_call}; "
        "Stage 3 разрешил режим рабочей сделки и связал эти звонки с выбранной сделкой."
    )


def build_binding_quality(candidate: dict[str, str], policy_rows: list[dict[str, str]], *, phone_count: int) -> str:
    confidence_counts = Counter(safe_text(row.get("confidence_bucket")) or "unknown" for row in policy_rows)
    candidate_count = safe_text(candidate.get("candidate_call_count")) or str(len(policy_rows))
    mode = safe_text(candidate.get("deal_writeback_mode")) or "none"
    confidence = ", ".join(f"{key}: {value}" for key, value in confidence_counts.most_common()) or "нет данных"
    return (
        f"Привязка к сделке: один выбранный AMO deal, режим {mode}, релевантных звонков {candidate_count}, "
        f"телефонов в связке {phone_count}. Confidence по звонкам: {confidence}. Источник: свежий AMO snapshot + Stage 2/3 policy."
    )


def build_warnings(
    candidate: dict[str, str],
    *,
    mode: str,
    phone_count: int,
    tallanto_context: dict[str, Any] | None = None,
    status: str = "",
) -> str:
    flags = set(split_pipe(safe_text(candidate.get("stage3_risk_flags"))))
    warnings = []
    if "paid_deal_has_payment_next_step_in_call" in flags:
        warnings.append("В звонке был платежный следующий шаг, но сделка уже оплачена/успешна; не дожимать клиента без сверки.")
    if "deal_has_overdue_open_tasks" in flags:
        warnings.append("В AMO есть просроченные открытые задачи; сверить с задачами менеджера.")
    if phone_count > 1:
        warnings.append("У сделки несколько телефонов; возможна семейная связка или дубль, проверять перед спорными действиями.")
    if mode == "context_only_paid_or_success":
        warnings.append("Режим только контекст: запрещено превращать исторический звонок в новый коммерческий следующий шаг.")
    tallanto_context = tallanto_context or {}
    if bool(tallanto_context.get("active_learning")) and re.search(r"\b(?:ожидание оплаты|заключение договора|перспектива)\b", status, re.I):
        visits = safe_text(tallanto_context.get("visit_count"))
        last_lesson = format_date(tallanto_context.get("last_lesson_at")) or safe_text(tallanto_context.get("last_lesson_at"))
        warnings.append(
            "AMO-статус нужно сверить: Tallanto показывает активного ученика"
            + (f", посещений/списаний {visits}" if visits else "")
            + (f", последнее занятие {last_lesson}" if last_lesson else "")
            + "."
        )
    if not warnings:
        return "Критичных предупреждений Stage 3 не выявил."
    return " ".join(dedupe_preserve_order(warnings))


def build_tallanto_context(
    candidate: dict[str, str],
    *,
    phone_rollup_by_phone: dict[str, dict[str, str]],
    student_by_tallanto_id: dict[str, dict[str, str]],
    writeoff_by_barcode: dict[str, dict[str, str]],
) -> dict[str, Any]:
    phones = split_pipe(safe_text(candidate.get("phones")))
    exact_ids = []
    statuses = []
    for phone in phones:
        rollup = phone_rollup_by_phone.get(phone)
        if not rollup:
            continue
        statuses.append(safe_text(rollup.get("tallanto_match_status")))
        if safe_text(rollup.get("tallanto_match_status")) == "exact_phone_single":
            exact_ids.append(safe_text(rollup.get("tallanto_id")))
    exact_ids = dedupe_preserve_order([value for value in exact_ids if value])
    if not exact_ids:
        return {
            "match_status": "no_reliable_tallanto_match",
            "active_learning": False,
            "text": "Tallanto: по телефонам сделки нет надежного точного сопоставления. Использовать только AMO и звонки.",
        }
    if len(exact_ids) > 1:
        candidates = []
        for tallanto_id in exact_ids[:3]:
            student = student_by_tallanto_id.get(tallanto_id, {})
            barcode = safe_text(student.get("barcode"))
            writeoff = writeoff_by_barcode.get(barcode, {}) if barcode else {}
            name = safe_text(student.get("full_name")) or "ФИО не указано"
            student_type = safe_text(student.get("student_type")) or "тип не указан"
            last_lesson = safe_text(writeoff.get("last_lesson_at"))
            suffix = f", последнее занятие {format_date(last_lesson) or last_lesson}" if last_lesson else ""
            candidates.append(f"{name} ({student_type}{suffix})")
        return {
            "match_status": "multiple_tallanto_matches",
            "tallanto_ids": " | ".join(exact_ids),
            "active_learning": False,
            "text": "Tallanto: найдено несколько точных сопоставлений по телефонам сделки: "
            + "; ".join(candidates)
            + ". Выбрать правильного ученика вручную перед действием.",
        }

    tallanto_id = exact_ids[0]
    student = student_by_tallanto_id.get(tallanto_id, {})
    barcode = safe_text(student.get("barcode"))
    writeoff = writeoff_by_barcode.get(barcode, {}) if barcode else {}
    name = safe_text(student.get("full_name")) or "ФИО не указано"
    student_type = safe_text(student.get("student_type")) or "тип не указан"
    branch = safe_text(student.get("branch")) or "филиал не указан"
    balance = safe_text(student.get("balance"))
    topup = safe_text(student.get("money_topup"))
    spent = safe_text(student.get("money_spent"))
    visits = safe_text(writeoff.get("visit_count"))
    last_lesson = safe_text(writeoff.get("last_lesson_at"))
    lessons = summarize_tallanto_lessons(safe_text(writeoff.get("top_lessons") or writeoff.get("recent_lessons")))
    active_learning = int_or_zero(visits) >= 3 and bool(re.match(r"2026-(?:04|05)", last_lesson))
    parts = [
        f"Tallanto: точное сопоставление — ученик {name}; тип {student_type}; филиал {branch}.",
    ]
    finance = "; ".join(
        part
        for part in [
            f"баланс {format_money(balance)}" if balance else "",
            f"пополнено {format_money(topup)}" if topup else "",
            f"списано {format_money(spent)}" if spent else "",
        ]
        if part
    )
    if finance:
        parts.append(f"Финансы: {finance}.")
    if visits or last_lesson or lessons:
        parts.append(
            "Обучение: "
            + "; ".join(
                part
                for part in [
                    f"посещений/списаний {visits}" if visits else "",
                    f"последнее занятие {format_date(last_lesson) or last_lesson}" if last_lesson else "",
                    f"основные группы: {lessons}" if lessons else "",
                ]
                if part
            )
            + "."
        )
    return {
        "match_status": "exact_phone_single",
        "tallanto_id": tallanto_id,
        "visit_count": visits,
        "last_lesson_at": last_lesson,
        "active_learning": active_learning,
        "text": " ".join(parts),
    }


def build_preview_row(
    *,
    index: int,
    candidate: dict[str, str],
    payload: dict[str, str],
    tallanto_context: dict[str, Any],
    row_findings: list[CrmTextQualityFinding],
    quality_passed: bool,
) -> dict[str, Any]:
    risk_types = sorted({finding.risk_type for finding in row_findings})
    severities = sorted({finding.severity for finding in row_findings})
    return {
        "review_id": f"deal-stage4-{index:05d}",
        "selected_deal_id": safe_text(candidate.get("selected_deal_id")),
        "selected_deal_name": safe_text(candidate.get("selected_deal_name")),
        "selected_pipeline_name": safe_text(candidate.get("selected_pipeline_name")),
        "selected_status_name": safe_text(candidate.get("selected_status_name")),
        "selected_loss_reason": safe_text(candidate.get("selected_loss_reason")),
        "deal_writeback_mode": safe_text(candidate.get("deal_writeback_mode")),
        "candidate_call_count": safe_text(candidate.get("candidate_call_count")),
        "candidate_phone_count": safe_text(candidate.get("candidate_phone_count")),
        "phones": safe_text(candidate.get("phones")),
        "managers": safe_text(candidate.get("managers")),
        "first_call_at": safe_text(candidate.get("first_call_at")),
        "last_call_at": safe_text(candidate.get("last_call_at")),
        "stage3_risk_flags": safe_text(candidate.get("stage3_risk_flags")),
        "tallanto_context_status": safe_text(tallanto_context.get("match_status")),
        "crm_text_quality_passed": "Да" if quality_passed else "Нет",
        "quality_severities": " | ".join(severities),
        "quality_risk_types": " | ".join(risk_types),
        "safe_for_deal_live_writeback_now": "Нет",
        "writeback_blocker": "Stage 4 is preview-only; requires audit pack and live preflight.",
        **payload,
    }


def quality_payload(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["AMO статус сделки"] = safe_text(candidate.get("selected_status_name"))
    result["AMO причина отказа"] = safe_text(candidate.get("selected_loss_reason"))
    result["priority"] = safe_text(payload.get("AI-приоритет сделки"))
    result["Рекомендуемая дата следующего контакта"] = safe_text(payload.get("AI-дата следующего касания"))
    return result


def group_policy_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if safe_text(row.get("safe_for_stage4_generation")) != "Да":
            continue
        deal_id = safe_text(row.get("selected_deal_id"))
        if deal_id:
            grouped[deal_id].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: safe_text(row.get("started_at")))
    return grouped


def hydrate_policy_calls(policy_rows: list[dict[str, str]], call_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for row in policy_rows:
        call_id = safe_text(row.get("call_id"))
        source = call_by_id.get(call_id, {})
        hydrated = dict(row)
        for key in (
            "call_summary",
            "started_at",
            "manager_name",
            "next_step",
            "products",
            "subjects",
            "objections",
            "call_type",
        ):
            if safe_text(source.get(key)):
                hydrated[key] = safe_text(source.get(key))
        if not safe_text(hydrated.get("next_step")):
            hydrated["next_step"] = safe_text(row.get("call_next_step"))
        result.append(hydrated)
    return result


def normalize_manager_text(value: Any) -> str:
    text = safe_text(value).replace("…", " ").replace("...", " ")
    text = text.replace("[сжато]", "").replace("[truncated]", "")
    for pattern, replacement in TENANT_TERM_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\bунпк\b", "УНПК", text, flags=re.I)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_service_feedback_context(value: Any) -> bool:
    text = normalize_manager_text(value)
    if not text:
        return False
    if SERVICE_FEEDBACK_RE.search(text):
        return True
    return bool(SERVICE_CONTEXT_WORD_RE.search(text) and SERVICE_PROBLEM_WORD_RE.search(text))


def infer_deal_topic(value: Any) -> str:
    text = normalize_manager_text(value)
    cf = text.casefold()
    products: list[str] = []
    if re.search(r"\b(?:лвш|выездн\w+\s+школ|с\s+проживани\w+|лагер\w*)\b", cf):
        products.append("летняя выездная школа/лагерь")
    if re.search(r"\b(?:очн\w+\s+школ|дневн\w+\s+формат|без\s+проживани\w+|мск|москв|долгопрудн)\b", cf):
        products.append("летняя очная школа без проживания")
    if re.search(r"\b(?:онлайн|дистанцион)\b", cf):
        products.append("онлайн-формат")
    if re.search(r"\b(?:26[/\\-]?27|2026[/\\-]?27|следующ\w+\s+учебн\w+|курсы?\s+на)\b", cf):
        products.append("курсы на следующий учебный год")
    if re.search(r"\b(?:договор|документ|оформлени)\b", cf):
        products.append("оформление документов")
    if re.search(r"\b(?:оплат|плат[её]ж|чек|реквизит)\b", cf):
        products.append("оплата")
    if not products:
        products.append("активный запрос клиента")

    subjects: list[str] = []
    for pattern, label in [
        (r"\bфиз(?:ика|мат|ико-математ)", "физика/физмат"),
        (r"\bматемат", "математика"),
        (r"\bинформат|программирован|кибер|искусственн\w+\s+интеллект|ии\b", "информатика/ИИ"),
        (r"\bолимпиад", "олимпиадная подготовка"),
    ]:
        if re.search(pattern, cf):
            subjects.append(label)

    product_text = " + ".join(dedupe_preserve_order(products[:3]))
    subject_text = ", ".join(dedupe_preserve_order(subjects[:4]))
    if subject_text:
        return f"{product_text}; направление: {subject_text}"
    return product_text


def format_money(value: Any) -> str:
    text = normalize_manager_text(value)
    if not text:
        return ""
    cleaned = text.replace("руб.", "").replace("руб", "").replace(" ", "").replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError:
        return text
    amount = int(round(number))
    return f"{amount:,}".replace(",", " ") + " руб."


def summarize_tallanto_lessons(value: Any, *, max_items: int = 3) -> str:
    raw_items = split_pipe(value)
    cleaned_items: list[str] = []
    for item in raw_items:
        text = normalize_manager_text(item)
        text = re.sub(r"^\d+\.\s*", "", text)
        text = re.sub(r":\s*\d+\s*$", "", text)
        text = re.sub(r"\s*\(Закрыта\)", "", text, flags=re.I)
        text = re.sub(r"\s+до\s+\d+\s+чел\b", "", text, flags=re.I)
        text = re.sub(r"\s+\d{3,4}\s+ГК\b", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip(" .;")
        if not text:
            continue
        cleaned_items.append(fit_text(text, 120))
    deduped = dedupe_preserve_order(cleaned_items)
    if not deduped:
        return ""
    suffix = ""
    if len(deduped) > max_items:
        suffix = f"; ещё {len(deduped) - max_items} групп(ы) в Tallanto"
    return "; ".join(deduped[:max_items]) + suffix


def manager_action_from_next_step(value: Any) -> str:
    text = normalize_manager_text(value).strip(" .;")
    if not text:
        return ""
    cf = text.casefold()
    deadline = extract_deadline(text)
    if re.match(r"^(?:ждать|подождать|дождаться|ожидать)\b", cf):
        if PAYMENT_NEXT_STEP_RE.search(text):
            deadline_text = f" до {deadline}" if deadline else ""
            fallback = f"если {deadline or 'к контрольной дате'} оплаты нет"
            return (
                f"Контролировать поступление оплаты{deadline_text}; "
                f"{fallback} — связаться с клиентом и зафиксировать новую дату."
            )
        return "Поставить контрольный срок ожидания ответа клиента; если ответа нет — связаться и зафиксировать решение в AMO."
    if re.match(r"^(?:оплатить|внести\s+оплату|произвести\s+оплату|согласовать\b.*\bоплат)", cf):
        deadline_text = f" до {deadline}" if deadline else ""
        return (
            f"Проконтролировать оплату выбранной программы{deadline_text}; "
            "если оплата не поступит к контрольной дате — связаться с клиентом и зафиксировать решение."
        )
    if re.match(r"^(?:прислать|отправить|выслать|передать)\b", cf) and re.search(
        r"\b(?:скан\w*|документ\w*|договор\w*|анкет\w*)\b", cf
    ):
        return (
            "Проконтролировать получение документов от клиента; "
            "если документы не поступят к контрольной дате — напомнить и зафиксировать статус в AMO."
        )
    if CUSTOMER_SIDE_WAIT_RE.search(text):
        return "Поставить контрольный срок ожидания ответа клиента; если клиент не вернется сам, перезвонить и зафиксировать решение."
    return ""


def extract_deadline(value: Any) -> str:
    text = normalize_manager_text(value)
    match = PAYMENT_DEADLINE_RE.search(text)
    if not match:
        return ""
    return re.sub(r"^(?:до|к)\s+", "", match.group(0), flags=re.I)


def sentence_summary(value: Any, *, max_sentences: int, max_chars: int) -> str:
    text = normalize_manager_text(value)
    text = CALL_BOILERPLATE_RE.sub("", text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = " ".join(sentence.strip() for sentence in sentences[:max_sentences] if sentence.strip())
    if not selected:
        selected = text
    return fit_text(selected, max_chars)


def history_call_summary(value: Any, *, max_sentences: int, max_chars: int) -> str:
    text = normalize_manager_text(value)
    text = CALL_BOILERPLATE_RE.sub("", text)
    if not text:
        return ""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    selected = " ".join(sentences[:max_sentences]) if sentences else text
    if len(selected) <= max_chars:
        return selected
    clauses = re.split(r"(?<=[,;:])\s+", selected)
    chunk = ""
    for clause in clauses:
        candidate = f"{chunk} {clause}".strip()
        if len(candidate) > max_chars - 26:
            break
        chunk = candidate
    if len(chunk) < 80:
        chunk = selected[: max_chars - 26].rstrip(" ,;:.")
    return f"{chunk.rstrip(' ,;:.')}. Детали в полном звонке."


def fit_text(value: Any, max_chars: int = 1200) -> str:
    text = normalize_manager_text(value)
    if len(text) <= max_chars:
        return text
    suffix = " Текст сокращен до лимита поля."
    limit = max(0, max_chars - len(suffix))
    chunk = text[:limit].rstrip()
    last_stop = max(chunk.rfind(". "), chunk.rfind("; "), chunk.rfind("! "), chunk.rfind("? "))
    if last_stop >= 160:
        chunk = chunk[: last_stop + 1].rstrip()
    else:
        chunk = chunk.rstrip(" ,;:.")
    separator = "" if chunk.endswith((".", "!", "?")) else "."
    return f"{chunk}{separator}{suffix}".strip()


def field_limit(field: str) -> int:
    if field == "AI-история по сделке":
        return 4500
    if field in {"AI-сводка по сделке", "AI-Tallanto статус по сделке"}:
        return 1600
    if field in {"AI-рекомендованный следующий шаг", "AI-дата следующего касания", "AI-приоритет сделки"}:
        return 350
    return 1300


def clean_next_step_text(value: Any) -> str:
    text = normalize_manager_text(value).strip(" .;")
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def normalize_objection(value: Any) -> str:
    text = normalize_manager_text(value).strip(" .;:").casefold()
    if not text:
        return ""
    if text in {"цена", "дорого"}:
        return "есть вопрос по стоимости"
    if text in {"время", "неудобно"}:
        return "нужно согласовать удобное время или дату"
    if text == "доверие":
        return "нужно подтвердить условия и ценность"
    if text in {"неактуально", "не актуально"}:
        return "ранее звучала неактуальность, нужна проверка текущего статуса"
    if len(text) > 90:
        return ""
    if len(text) <= 2 or text.isdigit():
        return ""
    return text


def split_pipe(value: Any) -> list[str]:
    text = safe_text(value)
    if not text:
        return []
    return [part.strip(" .;,") for part in re.split(r"\s+\|\s+|\n|;", text) if part.strip(" .;,")]


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = safe_text(value).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(safe_text(value))
    return result


def format_date(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text.replace("Z", "+0000"), fmt)
            return dt.strftime("%d.%m.%Y")
        except ValueError:
            continue
    if match := DATE_PREFIX_RE.match(text):
        return match.group(1)
    if len(text) >= 10 and re.match(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").strftime("%d.%m.%Y")
        except ValueError:
            return text[:10]
    return text[:10]


def parse_date_only(value: Any) -> datetime | None:
    text = safe_text(value)
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def stable_mod(value: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return sum(ord(char) for char in safe_text(value)) % modulo


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value).replace(",", ".")))
    except ValueError:
        return 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def findings_to_rows(review_id: str, deal_id: str, findings: Iterable[CrmTextQualityFinding]) -> list[dict[str, Any]]:
    return [
        {
            "review_id": review_id,
            "selected_deal_id": deal_id,
            "class_id": finding.class_id,
            "risk_type": finding.risk_type,
            "severity": finding.severity,
            "field": finding.field,
            "matched_text": finding.matched_text,
            "reason": finding.reason,
            "row_index": finding.row_index or "",
        }
        for finding in findings
    ]


def build_rop_review_sample(preview_rows: list[dict[str, Any]], *, size: int = 50) -> list[dict[str, Any]]:
    ready = [row for row in preview_rows if row.get("crm_text_quality_passed") == "Да"]
    blocked = [row for row in preview_rows if row.get("crm_text_quality_passed") != "Да"]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(pool: list[dict[str, Any]], target_size: int, key_func: Any) -> None:
        buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            buckets[key_func(row)].append(row)
        while len(selected) < target_size and any(buckets.values()):
            for bucket in sorted(list(buckets), key=stringify):
                if len(selected) >= target_size:
                    break
                if not buckets[bucket]:
                    continue
                row = buckets[bucket].pop(0)
                review_id = safe_text(row.get("review_id"))
                if review_id in seen:
                    continue
                seen.add(review_id)
                selected.append(row)

    take(
        [row for row in ready if row.get("deal_writeback_mode") == "full_active"],
        34,
        lambda row: (safe_text(row.get("selected_status_name")), safe_text(row.get("tallanto_context_status"))),
    )
    take(
        [row for row in ready if row.get("deal_writeback_mode") == "context_only_paid_or_success"],
        42,
        lambda row: safe_text(row.get("tallanto_context_status")),
    )
    take(
        blocked,
        size,
        lambda row: split_pipe(safe_text(row.get("quality_risk_types")))[0] if row.get("quality_risk_types") else "",
    )

    result = []
    for index, row in enumerate(selected[:size], start=1):
        result.append(
            {
                "rop_review_id": f"rop-stage4-{index:03d}",
                "review_id": safe_text(row.get("review_id")),
                "crm_text_quality_passed": safe_text(row.get("crm_text_quality_passed")),
                "quality_risk_types": safe_text(row.get("quality_risk_types")),
                "deal_writeback_mode": safe_text(row.get("deal_writeback_mode")),
                "selected_deal_id": safe_text(row.get("selected_deal_id")),
                "selected_deal_name": safe_text(row.get("selected_deal_name")),
                "selected_pipeline_name": safe_text(row.get("selected_pipeline_name")),
                "selected_status_name": safe_text(row.get("selected_status_name")),
                "candidate_call_count": safe_text(row.get("candidate_call_count")),
                "candidate_phone_count": safe_text(row.get("candidate_phone_count")),
                "tallanto_context_status": safe_text(row.get("tallanto_context_status")),
                "AI-сводка по сделке": safe_text(row.get("AI-сводка по сделке")),
                "AI-история по сделке": safe_text(row.get("AI-история по сделке")),
                "AI-рекомендованный следующий шаг": safe_text(row.get("AI-рекомендованный следующий шаг")),
                "AI-фактический статус сделки": safe_text(row.get("AI-фактический статус сделки")),
                "AI-приоритет сделки": safe_text(row.get("AI-приоритет сделки")),
                "AI-актуальные возражения": safe_text(row.get("AI-актуальные возражения")),
                "AI-основание рекомендации": safe_text(row.get("AI-основание рекомендации")),
                "AI-качество привязки к сделке": safe_text(row.get("AI-качество привязки к сделке")),
                "AI-предупреждение по сделке": safe_text(row.get("AI-предупреждение по сделке")),
                "AI-Tallanto статус по сделке": safe_text(row.get("AI-Tallanto статус по сделке")),
                "rop_verdict": "",
                "rop_right_deal": "",
                "rop_fields_useful": "",
                "rop_comment": "",
            }
        )
    return result


def build_summary(
    *,
    paths: DealTextPaths,
    candidates: list[dict[str, str]],
    preview_rows: list[dict[str, Any]],
    ready_for_audit: list[dict[str, Any]],
    blocked_quality: list[dict[str, Any]],
    quality_findings: list[dict[str, Any]],
    rop_review_sample: list[dict[str, Any]],
    stage1_summary: dict[str, Any],
    stage3_summary: dict[str, Any],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    mode_counts = Counter(safe_text(row.get("deal_writeback_mode")) for row in preview_rows)
    quality_counts = Counter(safe_text(row.get("crm_text_quality_passed")) for row in preview_rows)
    risk_counts = Counter(safe_text(row.get("risk_type")) for row in quality_findings if row.get("risk_type"))
    severity_counts = Counter(safe_text(row.get("severity")) for row in quality_findings if row.get("severity"))
    tallanto_counts = Counter(safe_text(row.get("tallanto_context_status")) for row in preview_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_date": paths.analysis_date,
        "sources": {
            "stage1_snapshot_root": str(paths.stage1_snapshot_root),
            "stage3_deal_state_root": str(paths.stage3_deal_state_root),
            "stage1_schema_version": safe_text(stage1_summary.get("schema_version")),
            "stage3_schema_version": safe_text(stage3_summary.get("schema_version")),
        },
        "safety": {
            "read_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
        "coverage": {
            "stage3_deal_candidates": len(candidates),
            "preview_rows": len(preview_rows),
            "ready_for_audit_rows": len(ready_for_audit),
            "blocked_by_crm_text_quality_rows": len(blocked_quality),
            "quality_findings": len(quality_findings),
            "rop_review_sample_rows": len(rop_review_sample),
        },
        "mode_counts": dict(mode_counts.most_common()),
        "quality_counts": dict(quality_counts.most_common()),
        "quality_risk_counts": dict(risk_counts.most_common()),
        "quality_severity_counts": dict(severity_counts.most_common()),
        "tallanto_context_counts": dict(tallanto_counts.most_common()),
        "readiness": {
            "stage4_preview_built": True,
            "safe_to_write_deal_fields": False,
            "requires_claude_or_rop_audit_before_live_deal_writeback": True,
            "ready_for_audit_rows": len(ready_for_audit),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def render_readme(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    return "\n".join(
        [
            "# Deal-Aware Stage 4 Preview",
            "",
            "Read-only deal text builder. This artifact does not write to AMO or Tallanto.",
            "",
            "## Coverage",
            "",
            f"- Stage 3 deal candidates: {coverage['stage3_deal_candidates']}",
            f"- preview rows: {coverage['preview_rows']}",
            f"- ready for audit rows: {coverage['ready_for_audit_rows']}",
            f"- blocked by CRM text quality rows: {coverage['blocked_by_crm_text_quality_rows']}",
            f"- quality findings: {coverage['quality_findings']}",
            "",
            "## Readiness",
            "",
            f"- Stage 4 preview built: {summary['readiness']['stage4_preview_built']}",
            f"- safe to write deal fields now: {summary['readiness']['safe_to_write_deal_fields']}",
            "- live writeback requires a separate audit pack, dry-run, operator approval and post-writeback readback gate.",
            "",
            "## Outputs",
            "",
            *[f"- `{key}`: `{path}`" for key, path in summary["outputs"].items()],
            "",
        ]
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
