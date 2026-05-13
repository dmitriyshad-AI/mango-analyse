#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import urllib.parse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from mango_mvp.productization.tenant_config import load_tenant_config, tenant_config_summary
from mango_mvp.quality.crm_text_quality_detector import (
    detect_crm_text_quality_risks,
    has_blocking_crm_text_findings,
)
from mango_mvp.quality.crm_writeback_quality_detector import detect_crm_writeback_quality_risks
from mango_mvp.quality.tenant_text_normalizer import (
    format_product_list,
    normalize_manager_text,
    normalize_objection_label,
    normalize_product_label,
    objection_key,
)
from mango_mvp.services.export_excel import call_to_row
from mango_mvp.utils.phone import normalize_phone

try:
    import xlsxwriter
except ImportError:  # pragma: no cover - optional dependency
    xlsxwriter = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_DB = (
    PROJECT_ROOT
    / "stable_runtime"
    / "canonical_master_20260510_after_quality_backfill_v1"
    / "canonical_calls_master.db"
)
DEFAULT_CLIENT_CHAINS = (
    PROJECT_ROOT
    / "stable_runtime"
    / "insight_readiness_report_after_quality_backfill_20260510_v1"
    / "client_chains.csv"
)
DEFAULT_STAGE15_SUMMARY = (
    PROJECT_ROOT
    / "stable_runtime"
    / "transcript_quality_stage15_export_gate_20260510_v11_frozen_gate"
    / "summary.json"
)
DEFAULT_OUT_ROOT_PARENT = PROJECT_ROOT / "stable_runtime"
DEFAULT_OUT_ROOT_PREFIX = "sales_master_export_post_backfill"
DEFAULT_TENANT_CONFIG = (
    PROJECT_ROOT
    / "_local_archive_mango_api_downloads_20260507"
    / "product_appliance"
    / "tenants"
    / "foton"
    / "config"
    / "tenant_config_v1.json"
)

MASTER_CALLS_HEADERS = [
    "ID звонка",
    "Дата и время звонка",
    "Телефон клиента",
    "Менеджер",
    "Направление звонка",
    "Длительность, сек",
    "Имя исходного файла",
    "Путь к записи",
    "Свежий период",
    "Основной ASR готов",
    "Второй ASR (GigaAM) готов",
    "Статус Resolve",
    "Статус Analyze",
    "Полная цепочка выполнена",
    "Содержательный звонок",
    "Нужна ручная проверка",
    "Причины ручной проверки",
    "Краткое резюме разговора",
    "Тип звонка",
    "ФИО родителя",
    "ФИО ребенка",
    "Email",
    "Класс",
    "Школа",
    "Продукты интереса",
    "Формат обучения",
    "Предметы интереса",
    "Целевые экзамены",
    "Рекомендуемый продукт",
    "Коммерческие ограничения",
    "Возражения",
    "Следующий шаг",
    "Срок следующего шага",
    "Приоритет лида",
    "Вероятность продажи, %",
    "Причина оценки вероятности",
    "Рекомендуемая дата следующего контакта",
    "Причина рекомендуемой даты",
    "Источник лучшего статуса",
]

MASTER_CONTACTS_HEADERS = [
    "Телефон клиента",
    "Всего звонков в истории",
    "Содержательных звонков в истории",
    "Несодержательных звонков в истории",
    "Звонков с полным анализом",
    "Незакрытых звонков в истории",
    "Полная история проанализирована",
    "Первый звонок",
    "Последний звонок",
    "Свежих звонков за период",
    "Последний свежий звонок",
    "Последний свежий звонок проанализирован",
    "Менеджер последнего свежего звонка",
    "Краткое резюме последнего свежего звонка",
    "Тип последнего свежего звонка",
    "Краткая история общения",
    "Хронология общения (последние 5 касаний)",
    "ФИО родителя",
    "ФИО ребенка",
    "Email",
    "Продукты интереса",
    "Рекомендуемый продукт",
    "Возражения",
    "Следующий шаг",
    "Рекомендуемая дата следующего контакта",
    "Приоритет лида",
    "Вероятность продажи, %",
    "Нужна ручная проверка",
    "Статус матчинга Tallanto",
    "Количество кандидатов Tallanto",
    "ID Tallanto",
    "ФИО родителя Tallanto",
    "Контакт Tallanto",
    "Ответственный Tallanto",
    "Тип ученика Tallanto",
    "Филиал Tallanto",
    "AMO contact IDs",
    "AMO lead IDs",
    "Outcome source",
    "Utility score",
    "CRM writeback policy",
    "CRM writeback blockers",
    "AMO entity policy",
    "Готово к записи в AMO",
    "Причина статуса AMO",
]

AMO_EXPORT_HEADERS = [
    "Телефон клиента",
    "ID Tallanto",
    "Статус матчинга Tallanto",
    "ФИО родителя",
    "ФИО ребенка",
    "Email",
    "Ответственный Tallanto",
    "Тип ученика Tallanto",
    "Филиал Tallanto",
    "Дата последнего свежего звонка",
    "Менеджер последнего свежего звонка",
    "Краткое резюме последнего свежего звонка",
    "Тип последнего свежего звонка",
    "Краткая история общения",
    "Хронология общения (последние 5 касаний)",
    "Продукты интереса",
    "Рекомендуемый продукт",
    "Возражения",
    "Следующий шаг",
    "Рекомендуемая дата следующего контакта",
    "Приоритет лида",
    "Вероятность продажи, %",
    "История общения Tallanto",
    "AMO contact IDs",
    "AMO lead IDs",
    "Outcome source",
    "Utility score",
    "CRM writeback policy",
    "CRM writeback blockers",
    "AMO entity policy",
    "Готово к записи в AMO",
    "Причина статуса AMO",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AMO-ready master export from post-backfill canonical DB and phone-chain layer."
    )
    parser.add_argument("--canonical-db", default=str(DEFAULT_CANONICAL_DB))
    parser.add_argument("--client-chains-csv", default=str(DEFAULT_CLIENT_CHAINS))
    parser.add_argument("--stage15-summary", default=str(DEFAULT_STAGE15_SUMMARY))
    parser.add_argument(
        "--out-root",
        default="",
        help="Output folder. Defaults to a new timestamped stable_runtime/sales_master_export_post_backfill_* folder.",
    )
    parser.add_argument("--fresh-from", default="2025-01-01")
    parser.add_argument("--fresh-to", default="2026-05-31")
    parser.add_argument("--analysis-date", default=date.today().isoformat())
    parser.add_argument("--tenant-config", default=str(DEFAULT_TENANT_CONFIG) if DEFAULT_TENANT_CONFIG.exists() else "")
    return parser.parse_args()


def _default_out_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return DEFAULT_OUT_ROOT_PARENT / f"{DEFAULT_OUT_ROOT_PREFIX}_{stamp}"


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


PHONE_IN_CRM_TEXT_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")


CRM_TEXT_FIELDS_TO_SANITIZE = {
    "Краткое резюме разговора",
    "Коммерческие ограничения",
    "Возражения",
    "Следующий шаг",
    "Причина оценки вероятности",
    "Причина рекомендуемой даты",
    "Краткое резюме последнего свежего звонка",
    "Краткая история общения",
    "Хронология общения (последние 5 касаний)",
    "Продукты интереса",
    "Рекомендуемый продукт",
    "История общения Tallanto",
}


def _redact_phone_match(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if ":" in raw or re.search(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b", raw):
        return raw
    if len(digits) >= 10:
        return "[PHONE]"
    return raw


def _sanitize_crm_text(value: Any) -> str:
    """Keep CRM context useful, but remove raw callback phones from AI text."""
    text = _safe_text(value)
    if not text:
        return ""
    return normalize_manager_text(PHONE_IN_CRM_TEXT_RE.sub(_redact_phone_match, text))


def _sanitize_crm_text_fields(row: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    cleaned = dict(row)
    for field in fields:
        if field in cleaned:
            cleaned[field] = _sanitize_crm_text(cleaned.get(field))
    return cleaned


def _safe_bool_text(value: bool) -> str:
    return "Да" if value else "Нет"


def _parse_dt(value: Any) -> datetime | None:
    text = _safe_text(value).replace("T", " ")
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalize_phone(value: Any) -> str:
    return normalize_phone(_safe_text(value))


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = "file:" + urllib.parse.quote(str(path)) + "?immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _write_xlsx(path: Path, sheets: list[tuple[str, list[str], list[dict[str, Any]]]]) -> None:
    if xlsxwriter is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path), {"constant_memory": True})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    header_format = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top"})
    for sheet_name, headers, rows in sheets:
        worksheet = workbook.add_worksheet(sheet_name[:31])
        for col_idx, header in enumerate(headers):
            worksheet.write(0, col_idx, header, header_format)
            worksheet.set_column(col_idx, col_idx, min(max(len(header) + 2, 12), 80))
        for row_idx, row in enumerate(rows, start=1):
            for col_idx, header in enumerate(headers):
                worksheet.write(row_idx, col_idx, _safe_text(row.get(header, "")), wrap)
    workbook.close()


def _load_client_chains(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        phone = _normalize_phone(row.get("phone"))
        if phone:
            result[phone] = row
    return result


def _json_loads(raw: Any, fallback: Any) -> Any:
    text = _safe_text(raw)
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _unique_parts(values: Iterable[Any], *, limit: int | None = None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in _safe_text(raw).replace("\n", " | ").split("|"):
            value = part.strip(" ,;")
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
            if limit is not None and len(result) >= limit:
                return " | ".join(result)
    return " | ".join(result)


COUNTED_LABEL_RE = re.compile(r"^(?P<label>.+?)\s*:\s*(?P<count>\d+)$")


def _unique_parts_with_counts(values: Iterable[Any], *, limit: int | None = None) -> str:
    result: list[str] = []
    index_by_label: dict[str, int] = {}
    for raw in values:
        for part in _safe_text(raw).replace("\n", " | ").split("|"):
            value = part.strip(" ,;")
            if not value:
                continue
            label, count = _split_counted_label(value)
            key = _label_key(label)
            if not key:
                continue
            display = f"{label} ({count} касаний)" if count else label
            if key in index_by_label:
                if count and "(" not in result[index_by_label[key]]:
                    result[index_by_label[key]] = display
                continue
            index_by_label[key] = len(result)
            result.append(display)
            if limit is not None and len(result) >= limit:
                return " | ".join(result)
    return " | ".join(result)


def _unique_plain_labels(values: Iterable[Any], *, limit: int | None = None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in _safe_text(raw).replace("\n", " | ").split("|"):
            value = part.strip(" ,;")
            if not value:
                continue
            label, _count = _split_counted_label(value)
            key = _label_key(label)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(label)
            if limit is not None and len(result) >= limit:
                return " | ".join(result)
    return " | ".join(result)


def _unique_product_labels(values: Iterable[Any], *, limit: int | None = None) -> str:
    return format_product_list(values, max_items=limit)


def _unique_subject_labels(values: Iterable[Any], *, limit: int | None = None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in _safe_text(raw).replace("\n", " | ").split("|"):
            value = normalize_manager_text(part.strip(" ,;"))
            if not value:
                continue
            label, _count = _split_counted_label(value)
            key = _label_key(label)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(label)
            if limit is not None and len(result) >= limit:
                return " | ".join(result)
    return " | ".join(result)


def _split_counted_label(value: str) -> tuple[str, str]:
    match = COUNTED_LABEL_RE.match(value.strip())
    if not match:
        return (value.strip(), "")
    return (match.group("label").strip(), match.group("count").strip())


def _label_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip(" ,;")


def _compact_without_ellipsis(text: Any, *, limit: int = 220) -> str:
    value = _safe_text(text)
    if len(value) <= limit:
        return value
    suffix = " [сжато]"
    budget = max(20, limit - len(suffix))
    candidate = value[:budget].rstrip()
    # Prefer cutting at a word boundary so the marker means structural compaction, not silent word truncation.
    word_boundary = max(candidate.rfind(" "), candidate.rfind(","), candidate.rfind(";"), candidate.rfind("."))
    if word_boundary >= int(budget * 0.55):
        candidate = candidate[:word_boundary].rstrip(" ,;.")
    return f"{candidate}{suffix}"


def _is_contentful(call_type: str) -> bool:
    return _safe_text(call_type).casefold() != "non_conversation"


def _is_low_value_for_crm(flat: dict[str, Any]) -> bool:
    text = " ".join(
        _safe_text(flat.get(key))
        for key in (
            "history_summary",
            "next_step_action",
            "objections",
            "recommended_product",
            "interests_products",
            "interests_subjects",
        )
    )
    return bool(detect_crm_writeback_quality_risks(text))


def _call_gist(row: dict[str, Any], *, limit: int = 420) -> str:
    summary = _safe_text(row.get("Краткое резюме разговора"))
    next_step = _safe_text(row.get("Следующий шаг"))
    objections = _safe_text(row.get("Возражения"))
    if not summary and next_step:
        summary = f"Следующий шаг: {next_step}"
    if not summary and objections:
        summary = f"Возражение: {objections}"
    if next_step and next_step.casefold() not in summary.casefold():
        summary = f"{summary} Следующий шаг: {next_step}".strip()
    return _compact_without_ellipsis(_sanitize_crm_text(summary), limit=limit)


def _history_line(row: dict[str, Any]) -> str:
    dt = _parse_dt(row.get("Дата и время звонка"))
    date_text = dt.strftime("%d.%m.%Y") if dt else _safe_text(row.get("Дата и время звонка"))[:10]
    manager = _safe_text(row.get("Менеджер"))
    call_type = _safe_text(row.get("Тип звонка"))
    prefix = date_text
    if manager:
        prefix += f" — {manager}"
    if call_type:
        prefix += f" ({call_type})"
    gist = _call_gist(row, limit=520)
    return f"{prefix}: {gist}" if gist else prefix


def _excel_safe_text(text: Any, *, limit: int = 32000) -> str:
    value = _safe_text(text)
    if len(value) <= limit:
        return value
    suffix = " [обрезано по лимиту Excel]"
    return value[: max(limit - len(suffix), 0)].rstrip() + suffix


def _has_truncated_crm_text(row: dict[str, Any]) -> bool:
    for field in (
        "Краткое резюме последнего свежего звонка",
        "Краткая история общения",
        "Хронология общения (последние 5 касаний)",
        "Возражения",
        "Следующий шаг",
        "История общения Tallanto",
    ):
        value = _safe_text(row.get(field)).rstrip()
        if "..." in value or "…" in value:
            return True
    return False


SERVICE_CONTEXT_CALL_TYPES = {"service_call", "existing_client_progress", "technical_call"}
WEAK_OBJECTION_LABELS = {"время", "доверие", "цена", "неудобно"}
STRONG_NEGATIVE_LABELS = {"неактуально", "отказ", "не интересно", "неинтересно", "не беспокоить", "отменить", "отмена"}
CLOSURE_OR_PASSIVE_NEXT_STEP_RE = re.compile(
    r"не\s+беспокоить|не\s+продолжать|отменить|отмена\s+запис|снять\s+заявк|убрать\s+из\s+списк|"
    r"оставить\s+клиент\w+\s+без|ждать\s+обращени|ожидать\s+обновлени|"
    r"ожидать\s+решени|ожидать,\s*пока\s+клиент\w+|дождаться|"
    r"при\s+изменени\w+\s+решени|при\s+необходимости|если\s+понадобится|если\s+будет\s+актуально|"
    r"вернуться\s+к\s+(?:предложени\w+|запис\w+|подбор\w+)\s+при|"
    r"\b(?:подумает|подумают|сам\s+перезвонит|сама\s+перезвонит|сами\s+перезвонят|сама\s+свяжется|сам\s+свяжется)\b|"
    r"\b(?:быть|остаться)\s+на\s+связи\b|"
    r"\b(?:перезвонить|связаться|созвониться|вернуться)\s+(?:позже|потом|летом|осенью|зимой|весной)\b|"
    r"\b(?:перезвонить|связаться|созвониться|вернуться)\s+(?:в|во|к|ближе\s+к)\s+"
    r"(?:январ[юе]|феврал[юе]|март[уе]|апрел[юе]|ма[юе]|июн[юе]|июл[юе]|август[уе]|"
    r"сентябр[юе]|октябр[юе]|ноябр[юе]|декабр[юе])\b|"
    r"\b(?:через\s+пару\s+недель|в\s+следующем\s+году)\b",
    re.IGNORECASE,
)


def _split_ids(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    parts = re.split(r"[|,;\s]+", text)
    return [part for part in (item.strip() for item in parts) if part]


def _single_id_status(value: Any) -> str:
    ids = _split_ids(value)
    if len(ids) == 1:
        return "single"
    if len(ids) > 1:
        return "multiple"
    return "empty"


def _is_service_context_call_type(call_type: Any) -> bool:
    return _safe_text(call_type).casefold() in SERVICE_CONTEXT_CALL_TYPES


def _crm_writeback_policy_for_contact(
    *,
    contact: dict[str, Any],
    latest_contentful: dict[str, Any],
    tenant_config: dict[str, Any] | None,
) -> tuple[str, str, str, list[str]]:
    crm_config = (tenant_config or {}).get("crm") if isinstance(tenant_config, dict) else {}
    if not isinstance(crm_config, dict):
        crm_config = {}
    require_single_amo = bool(crm_config.get("require_single_amo_contact_id_for_live", True))
    service_policy = _safe_text(crm_config.get("service_call_live_writeback_policy")) or "manual_review"
    orphan_policy = _safe_text(crm_config.get("orphan_contact_policy")) or "manual_review"

    blockers: list[str] = []
    call_type = _safe_text(latest_contentful.get("Тип звонка") or contact.get("Тип последнего свежего звонка"))
    if _is_service_context_call_type(call_type) and service_policy != "allow_live_sales_writeback":
        blockers.append(
            f"последний содержательный звонок {call_type}: service/existing-client context не является новым sales lead"
        )

    amo_status = _single_id_status(contact.get("AMO contact IDs"))
    if require_single_amo and amo_status != "single":
        if amo_status == "empty":
            blockers.append("нет AMO contact ID в post-backfill phone-chain; требуется live lookup/create policy")
        else:
            blockers.append("несколько AMO contact IDs; требуется ручной выбор контакта")

    if blockers:
        entity_policy = "manual_review_orphan" if amo_status != "single" and orphan_policy == "manual_review" else "manual_review"
        if _is_service_context_call_type(call_type):
            entity_policy = "service_context_manual_review"
        return ("blocked_for_live_writeback", " | ".join(blockers), entity_policy, blockers)
    return ("live_update_ready", "", "update_existing_single_amo_contact", [])


def _build_contact_summary(contentful_desc: list[dict[str, Any]], chain: dict[str, str]) -> str:
    if not contentful_desc:
        return ""
    latest = contentful_desc[0]
    earliest = contentful_desc[-1]
    latest_dt = _parse_dt(latest.get("Дата и время звонка"))
    earliest_dt = _parse_dt(earliest.get("Дата и время звонка"))
    period_start = earliest_dt.strftime("%d.%m.%Y") if earliest_dt else _safe_text(earliest.get("Дата и время звонка"))[:10]
    period_end = latest_dt.strftime("%d.%m.%Y") if latest_dt else _safe_text(latest.get("Дата и время звонка"))[:10]
    managers = _unique_parts((row.get("Менеджер") for row in contentful_desc), limit=4)
    products = _unique_product_labels(
        list(row.get("Рекомендуемый продукт") for row in contentful_desc)
        + list(row.get("Продукты интереса") for row in contentful_desc)
        + [_safe_text(chain.get("products_top"))],
        limit=5,
    )
    subjects = _unique_subject_labels(
        list(row.get("Предметы интереса") for row in contentful_desc) + [_safe_text(chain.get("subjects_top"))],
        limit=5,
    )
    objections = _format_contact_objections(contentful_desc, limit_current=5, limit_historical=3)
    next_step = next((_safe_text(row.get("Следующий шаг")) for row in contentful_desc if _safe_text(row.get("Следующий шаг"))), "")
    parts = [
        f"Клиент в истории с {period_start} по {period_end}; содержательных звонков: {len(contentful_desc)}.",
    ]
    if products:
        parts.append(f"Основной интерес: {products}.")
    if subjects:
        parts.append(f"Предметы/направления: {subjects}.")
    latest_status_bits = []
    latest_product = normalize_product_label(latest.get("Рекомендуемый продукт")) or normalize_product_label(
        latest.get("Продукты интереса")
    )
    if latest_product:
        latest_status_bits.append(f"обсуждается {latest_product}")
    if latest_status_bits:
        parts.append(f"Текущая ситуация: {'; '.join(latest_status_bits)}.")
    if objections:
        parts.append("Есть актуальные ограничения/возражения; подробности вынесены в отдельный блок.")
    if next_step:
        parts.append(f"Следующий шаг: {next_step}.")
    if managers:
        parts.append(f"С клиентом работали: {managers}.")
    return _excel_safe_text(_sanitize_crm_text(" ".join(part for part in parts if part)))


def _format_contact_objections(
    contentful_desc: list[dict[str, Any]],
    *,
    limit_current: int = 6,
    limit_historical: int = 4,
) -> str:
    current: list[str] = []
    historical: list[str] = []
    current_seen: set[str] = set()
    historical_seen: set[tuple[str, str]] = set()
    recent_rows = contentful_desc[:3]
    for row in recent_rows:
        for objection in _split_parts(row.get("Возражения")):
            objection = normalize_objection_label(objection)
            key = objection_key(objection)
            if not key or key in WEAK_OBJECTION_LABELS:
                continue
            if key in STRONG_NEGATIVE_LABELS or any(label in key for label in STRONG_NEGATIVE_LABELS):
                date_text = (_parse_dt(row.get("Дата и время звонка")) or datetime.min).strftime("%d.%m.%Y")
                item = f"{date_text}: {objection}"
                hist_key = (date_text, key)
                if hist_key not in historical_seen:
                    historical_seen.add(hist_key)
                    historical.append(item)
                continue
            if key not in current_seen:
                current_seen.add(key)
                current.append(objection)
    for row in contentful_desc[3:]:
        for objection in _split_parts(row.get("Возражения")):
            objection = normalize_objection_label(objection)
            key = objection_key(objection)
            if not key or key in WEAK_OBJECTION_LABELS:
                continue
            date_text = (_parse_dt(row.get("Дата и время звонка")) or datetime.min).strftime("%d.%m.%Y")
            hist_key = (date_text, key)
            if hist_key not in historical_seen:
                historical_seen.add(hist_key)
                historical.append(f"{date_text}: {objection}")
            if len(historical) >= limit_historical:
                break
        if len(historical) >= limit_historical:
            break
    parts: list[str] = []
    if current:
        parts.append("Актуальные: " + " | ".join(current[:limit_current]))
    if historical:
        parts.append("Исторические: " + " | ".join(historical[:limit_historical]))
    return " | ".join(parts)


def _split_parts(value: Any) -> list[str]:
    return [part.strip(" ,;") for part in _safe_text(value).replace("\n", " | ").split("|") if part.strip(" ,;")]


def _build_contact_history(calls_desc: list[dict[str, Any]], chain: dict[str, str]) -> tuple[str, str]:
    contentful_desc = [row for row in calls_desc if row.get("Содержательный звонок") == "Да"]
    if not contentful_desc:
        return ("", "")
    all_chronological = list(reversed(contentful_desc))
    chronology = "\n".join(_history_line(row) for row in all_chronological)
    return (_build_contact_summary(contentful_desc, chain), _excel_safe_text(_sanitize_crm_text(chronology)))


def _parse_probability(value: Any) -> int | None:
    text = _safe_text(value).replace("%", "")
    if not text:
        return None
    try:
        return max(0, min(100, int(round(float(text)))))
    except Exception:
        return None


def _priority_rank(value: Any) -> int:
    return {"cold": 1, "warm": 2, "hot": 3}.get(_safe_text(value).casefold(), 0)


def _rank_to_priority(rank: int, fallback: Any = "") -> str:
    if rank <= 0:
        return _safe_text(fallback)
    return {1: "cold", 2: "warm", 3: "hot"}.get(rank, _safe_text(fallback))


def _infer_priority(probability: int | None) -> str:
    if probability is None:
        return ""
    if probability >= 75:
        return "hot"
    if probability >= 45:
        return "warm"
    return "cold"


def _adjust_operational_fields(
    *,
    analysis_date: date,
    last_contentful_raw: Any,
    follow_up_raw: Any,
    priority_raw: Any,
    probability_raw: Any,
    next_step_raw: Any = "",
) -> tuple[str, str, str]:
    last_contentful_dt = _parse_dt(last_contentful_raw)
    if last_contentful_dt is None:
        return (_safe_text(follow_up_raw), _safe_text(priority_raw), _safe_text(probability_raw))
    next_step = _safe_text(next_step_raw)
    days_since = max(0, (analysis_date - last_contentful_dt.date()).days)
    probability = _parse_probability(probability_raw)
    if probability is None:
        probability = {"hot": 80, "warm": 60, "cold": 30}.get(_safe_text(priority_raw).casefold())
    if _is_closure_or_passive_next_step(next_step):
        probability = min(probability if probability is not None else 25, 25)
        return ("", "cold", str(probability))
    if probability is not None:
        decay = 0
        if days_since > 7:
            decay += 5
        if days_since > 14:
            decay += 10
        if days_since > 30:
            decay += 15
        if days_since > 60:
            decay += 15
        probability = max(0, probability - decay)
    priority_rank = _priority_rank(priority_raw) or _priority_rank(_infer_priority(probability))
    if days_since > 14 and priority_rank > 1:
        priority_rank -= 1
    if days_since > 45 and priority_rank > 1:
        priority_rank -= 1
    priority = _rank_to_priority(priority_rank, _safe_text(priority_raw) or _infer_priority(probability))
    follow_up_dt = _parse_dt(follow_up_raw)
    if follow_up_dt is not None and follow_up_dt.date() > analysis_date:
        follow_up = follow_up_dt.date()
    else:
        follow_up = _fresh_followup_date(
            analysis_date=analysis_date,
            next_step=next_step,
            priority=priority,
            probability=probability,
            days_since_last_contact=days_since,
        )
    return (
        follow_up.isoformat() if follow_up else "",
        priority,
        str(probability) if probability is not None else _safe_text(probability_raw),
    )


def _fresh_followup_date(
    *,
    analysis_date: date,
    next_step: str,
    priority: str,
    probability: int | None,
    days_since_last_contact: int,
) -> date | None:
    """Assign a new actionable date instead of mass-promoting stale dates to the run date."""
    text = _safe_text(next_step).casefold()
    if not text:
        return None
    if _is_closure_or_passive_next_step(text):
        return None
    if re.search(r"\b(?:оплат|плат[её]ж|счет|сч[её]т|договор|брон|запис)\w*", text):
        return analysis_date + timedelta(days=1)
    if re.search(r"\b(?:отправ|прислать|направить|выслать|выслат)\w*", text):
        return analysis_date + timedelta(days=2)
    if re.search(r"\b(?:уточнить|сообщить|подтвердить|передать|соединить|подать)\w*", text):
        return analysis_date + timedelta(days=2)
    if re.search(r"\b(?:перезвон|созвон|позвон|связаться)\w*", text):
        if priority == "hot" or (probability is not None and probability >= 75):
            return analysis_date + timedelta(days=1)
        if priority == "warm" or (probability is not None and probability >= 45):
            return analysis_date + timedelta(days=2)
        return analysis_date + timedelta(days=5 if days_since_last_contact > 30 else 3)
    if priority == "hot" or (probability is not None and probability >= 75):
        return analysis_date + timedelta(days=1)
    if priority == "warm" or (probability is not None and probability >= 45):
        return analysis_date + timedelta(days=3)
    return None


def _is_closure_or_passive_next_step(value: Any) -> bool:
    return bool(CLOSURE_OR_PASSIVE_NEXT_STEP_RE.search(_safe_text(value)))


def _has_variant_b(raw: Any) -> bool:
    payload = _json_loads(raw, {})
    if not isinstance(payload, dict):
        return False
    for section in ("manager", "client", "full"):
        part = payload.get(section)
        if isinstance(part, dict) and _safe_text(part.get("variant_b")):
            return True
    return False


def _load_call_rows(canonical_db: Path, fresh_from: str, fresh_to: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conn = _connect_ro(canonical_db)
    query = """
        SELECT
            c.*,
            q.call_type AS quality_call_type,
            q.needs_review AS quality_needs_review,
            q.review_reasons_json AS quality_review_reasons_json
        FROM canonical_calls c
        LEFT JOIN call_quality_current q ON q.canonical_call_id = c.canonical_call_id
        WHERE c.is_actionable = 1
          AND c.analysis_status = 'done'
          AND COALESCE(c.analysis_json, '') != ''
        ORDER BY c.started_at, c.source_filename
    """
    for item in conn.execute(query):
        row = dict(item)
        analysis = _json_loads(row.get("analysis_json"), {})
        if not isinstance(analysis, dict):
            continue
        call_obj = SimpleNamespace(
            id=row.get("canonical_call_id"),
            started_at=_parse_dt(row.get("started_at")),
            phone=row.get("phone"),
            manager_name=row.get("manager_name"),
            duration_sec=row.get("duration_sec"),
            source_filename=row.get("source_filename"),
            source_file=row.get("source_file"),
            direction=row.get("direction"),
            transcript_variants_json=row.get("transcript_variants_json"),
        )
        flat = call_to_row(call_obj, analysis)
        quality_call_type = _safe_text(row.get("quality_call_type"))
        if quality_call_type:
            flat["call_type"] = quality_call_type
        quality_needs_review = row.get("quality_needs_review")
        if quality_needs_review is not None:
            flat["needs_review"] = bool(quality_needs_review)
        quality_review_reasons = _json_loads(row.get("quality_review_reasons_json"), [])
        if isinstance(quality_review_reasons, list) and quality_review_reasons:
            flat["review_reasons"] = _unique_parts(quality_review_reasons)
        call_type = _safe_text(flat.get("call_type"))
        started = _safe_text(row.get("started_at"))
        is_fresh = fresh_from <= started[:10] <= fresh_to
        contentful = _is_contentful(call_type) and not _is_low_value_for_crm(flat)
        rows.append(
            {
                "ID звонка": row.get("canonical_call_id"),
                "Дата и время звонка": started,
                "Телефон клиента": _normalize_phone(row.get("phone")),
                "Менеджер": _safe_text(flat.get("manager_name")),
                "Направление звонка": _safe_text(row.get("direction")),
                "Длительность, сек": _safe_text(row.get("duration_sec")),
                "Имя исходного файла": _safe_text(row.get("source_filename")),
                "Путь к записи": _safe_text(row.get("source_file")),
                "Свежий период": _safe_bool_text(is_fresh),
                "Основной ASR готов": _safe_bool_text(_safe_text(row.get("transcription_status")) == "done"),
                "Второй ASR (GigaAM) готов": _safe_bool_text(_has_variant_b(row.get("transcript_variants_json"))),
                "Статус Resolve": _safe_text(row.get("resolve_status")),
                "Статус Analyze": _safe_text(row.get("analysis_status")),
                "Полная цепочка выполнена": "Да",
                "Содержательный звонок": _safe_bool_text(contentful),
                "Нужна ручная проверка": _safe_bool_text(bool(flat.get("needs_review"))),
                "Причины ручной проверки": _safe_text(flat.get("review_reasons")),
                "Краткое резюме разговора": _sanitize_crm_text(flat.get("history_summary")),
                "Тип звонка": call_type,
                "ФИО родителя": _safe_text(flat.get("parent_fio")),
                "ФИО ребенка": _safe_text(flat.get("child_fio")),
                "Email": _safe_text(flat.get("email")),
                "Класс": _safe_text(flat.get("grade_current")),
                "Школа": _safe_text(flat.get("school")),
                "Продукты интереса": _safe_text(flat.get("interests_products")),
                "Формат обучения": _safe_text(flat.get("interests_format")),
                "Предметы интереса": _safe_text(flat.get("interests_subjects")),
                "Целевые экзамены": _safe_text(flat.get("exam_targets")),
                "Рекомендуемый продукт": _safe_text(flat.get("recommended_product")),
                "Коммерческие ограничения": _sanitize_crm_text(
                    _unique_parts(
                        [
                            flat.get("price_sensitivity"),
                            flat.get("budget"),
                            flat.get("discount_interest"),
                        ]
                    )
                ),
                "Возражения": _sanitize_crm_text(flat.get("objections")),
                "Следующий шаг": _sanitize_crm_text(flat.get("next_step_action")),
                "Срок следующего шага": _safe_text(flat.get("next_step_due_raw")),
                "Приоритет лида": _safe_text(flat.get("lead_priority")),
                "Вероятность продажи, %": _safe_text(flat.get("sale_probability_pct")),
                "Причина оценки вероятности": _sanitize_crm_text(flat.get("sale_probability_reason")),
                "Рекомендуемая дата следующего контакта": _safe_text(flat.get("recommended_followup_date")),
                "Причина рекомендуемой даты": _sanitize_crm_text(flat.get("recommended_followup_reason")),
                "Источник лучшего статуса": _safe_text(row.get("selected_source_db")),
            }
        )
    conn.close()
    return rows


def _chain_match_status(chain: dict[str, str]) -> tuple[str, str, str, str]:
    if _safe_text(chain.get("has_tallanto_match")).casefold() not in {"true", "1", "yes", "да"}:
        return ("no_exact_phone_match", "0", "", "")
    count = _safe_text(chain.get("tallanto_ids_count")) or "1"
    status = "exact_phone_single" if count == "1" else "exact_phone_multiple"
    return (
        status,
        count,
        _safe_text(chain.get("tallanto_ids")),
        _safe_text(chain.get("tallanto_branches")),
    )


def _build_contact_rows(
    *,
    call_rows: list[dict[str, Any]],
    chains_by_phone: dict[str, dict[str, str]],
    analysis_date: date,
    tenant_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in call_rows:
        phone = _normalize_phone(row.get("Телефон клиента"))
        if phone:
            by_phone[phone].append(row)

    contacts: list[dict[str, Any]] = []
    amo_rows: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []

    for phone, rows in sorted(by_phone.items()):
        rows_desc = sorted(
            rows,
            key=lambda row: (_parse_dt(row.get("Дата и время звонка")) or datetime.min, _safe_text(row.get("Имя исходного файла"))),
            reverse=True,
        )
        chain = chains_by_phone.get(phone, {})
        contentful_desc = [row for row in rows_desc if row.get("Содержательный звонок") == "Да"]
        non_conversation_count = len(rows_desc) - len(contentful_desc)
        latest_any = rows_desc[0] if rows_desc else {}
        latest_contentful = contentful_desc[0] if contentful_desc else {}
        short_history, chronology = _build_contact_history(rows_desc, chain)
        needs_review = any(row.get("Нужна ручная проверка") == "Да" for row in contentful_desc)
        match_status, match_count, tallanto_ids, tallanto_branches = _chain_match_status(chain)
        adjusted_follow_up, adjusted_priority, adjusted_probability = _adjust_operational_fields(
            analysis_date=analysis_date,
            last_contentful_raw=latest_contentful.get("Дата и время звонка"),
            follow_up_raw=latest_contentful.get("Рекомендуемая дата следующего контакта"),
            priority_raw=latest_contentful.get("Приоритет лида"),
            probability_raw=latest_contentful.get("Вероятность продажи, %"),
            next_step_raw=latest_contentful.get("Следующий шаг"),
        )
        if not contentful_desc:
            amo_ready, amo_reason = ("Нет", "нет содержательных звонков после transcript-quality backfill")
        elif needs_review:
            amo_ready, amo_reason = ("Нет", "есть содержательные звонки с ручной проверкой")
        elif not short_history:
            amo_ready, amo_reason = ("Нет", "пустая история общения")
        else:
            amo_ready, amo_reason = ("Да", "готово к AMO dry-run из post-backfill слоя")

        contact = {
            "Телефон клиента": phone,
            "Всего звонков в истории": len(rows_desc),
            "Содержательных звонков в истории": len(contentful_desc),
            "Несодержательных звонков в истории": non_conversation_count,
            "Звонков с полным анализом": len(rows_desc),
            "Незакрытых звонков в истории": 0,
            "Полная история проанализирована": "Да",
            "Первый звонок": min((_safe_text(row.get("Дата и время звонка")) for row in rows_desc if row.get("Дата и время звонка")), default=""),
            "Последний звонок": max((_safe_text(row.get("Дата и время звонка")) for row in rows_desc if row.get("Дата и время звонка")), default=""),
            "Свежих звонков за период": sum(1 for row in rows_desc if row.get("Свежий период") == "Да"),
            "Последний свежий звонок": _safe_text(latest_contentful.get("Дата и время звонка")),
            "Последний свежий звонок проанализирован": _safe_bool_text(bool(latest_contentful)),
            "Менеджер последнего свежего звонка": _safe_text(latest_contentful.get("Менеджер")),
            "Краткое резюме последнего свежего звонка": _safe_text(latest_contentful.get("Краткое резюме разговора")),
            "Тип последнего свежего звонка": _safe_text(latest_contentful.get("Тип звонка")),
            "Краткая история общения": short_history,
            "Хронология общения (последние 5 касаний)": chronology,
            "ФИО родителя": _safe_text(latest_contentful.get("ФИО родителя")),
            "ФИО ребенка": _safe_text(latest_contentful.get("ФИО ребенка")),
            "Email": _safe_text(latest_contentful.get("Email")),
            "Продукты интереса": _unique_product_labels(
                [row.get("Продукты интереса") for row in contentful_desc] + [_safe_text(chain.get("products_top"))],
                limit=8,
            ),
            "Рекомендуемый продукт": normalize_product_label(latest_contentful.get("Рекомендуемый продукт")),
            "Возражения": _format_contact_objections(contentful_desc, limit_current=6, limit_historical=4),
            "Следующий шаг": _safe_text(latest_contentful.get("Следующий шаг")),
            "Рекомендуемая дата следующего контакта": adjusted_follow_up,
            "Приоритет лида": adjusted_priority,
            "Вероятность продажи, %": adjusted_probability,
            "Нужна ручная проверка": _safe_bool_text(needs_review),
            "Статус матчинга Tallanto": match_status,
            "Количество кандидатов Tallanto": match_count,
            "ID Tallanto": tallanto_ids,
            "ФИО родителя Tallanto": "",
            "Контакт Tallanto": "",
            "Ответственный Tallanto": "",
            "Тип ученика Tallanto": _safe_text(chain.get("tallanto_student_types")),
            "Филиал Tallanto": tallanto_branches,
            "AMO contact IDs": _safe_text(chain.get("amo_contact_ids")),
            "AMO lead IDs": _safe_text(chain.get("amo_lead_ids")),
            "Outcome source": _safe_text(chain.get("outcome_source")),
            "Utility score": _safe_text(chain.get("utility_score")),
            "CRM writeback policy": "",
            "CRM writeback blockers": "",
            "AMO entity policy": "",
            "Готово к записи в AMO": amo_ready,
            "Причина статуса AMO": amo_reason,
        }
        contact = _sanitize_crm_text_fields(contact, CRM_TEXT_FIELDS_TO_SANITIZE)
        if amo_ready == "Да":
            policy, blockers_text, entity_policy, policy_blockers = _crm_writeback_policy_for_contact(
                contact=contact,
                latest_contentful=latest_contentful,
                tenant_config=tenant_config,
            )
            contact["CRM writeback policy"] = policy
            contact["CRM writeback blockers"] = blockers_text
            contact["AMO entity policy"] = entity_policy
            if policy_blockers:
                amo_ready = "Нет"
                amo_reason = blockers_text
                needs_review = True
                contact["Готово к записи в AMO"] = amo_ready
                contact["Причина статуса AMO"] = amo_reason
                contact["Нужна ручная проверка"] = "Да"
        if amo_ready == "Да" and _has_truncated_crm_text(contact):
            amo_ready = "Нет"
            amo_reason = "текст для CRM заканчивается многоточием / вероятно обрезан"
            needs_review = True
            contact["Готово к записи в AMO"] = amo_ready
            contact["Причина статуса AMO"] = amo_reason
            contact["Нужна ручная проверка"] = "Да"
        if amo_ready == "Да" and _is_closure_or_passive_next_step(contact.get("Следующий шаг")):
            amo_ready = "Нет"
            amo_reason = "пассивный или закрывающий следующий шаг; требуется ручная проверка перед AMO writeback"
            needs_review = True
            contact["Готово к записи в AMO"] = amo_ready
            contact["Причина статуса AMO"] = amo_reason
            contact["Нужна ручная проверка"] = "Да"
        if amo_ready == "Да":
            crm_text_findings = detect_crm_text_quality_risks(
                contact,
                min_severity="P2",
                analysis_date=analysis_date,
            )
            if has_blocking_crm_text_findings(crm_text_findings):
                amo_ready = "Нет"
                risk_types = _unique_parts(finding.risk_type for finding in crm_text_findings)
                amo_reason = f"crm text quality gate: {risk_types}"
                needs_review = True
                contact["Готово к записи в AMO"] = amo_ready
                contact["Причина статуса AMO"] = amo_reason
                contact["Нужна ручная проверка"] = "Да"
        contacts.append(contact)
        if amo_ready == "Да":
            amo_rows.append(
                _sanitize_crm_text_fields(
                    {
                        "Телефон клиента": phone,
                        "ID Tallanto": contact["ID Tallanto"],
                        "Статус матчинга Tallanto": contact["Статус матчинга Tallanto"],
                        "ФИО родителя": contact["ФИО родителя"],
                        "ФИО ребенка": contact["ФИО ребенка"],
                        "Email": contact["Email"],
                        "Ответственный Tallanto": contact["Ответственный Tallanto"],
                        "Тип ученика Tallanto": contact["Тип ученика Tallanto"],
                        "Филиал Tallanto": contact["Филиал Tallanto"],
                        "Дата последнего свежего звонка": contact["Последний свежий звонок"],
                        "Менеджер последнего свежего звонка": contact["Менеджер последнего свежего звонка"],
                        "Краткое резюме последнего свежего звонка": contact["Краткое резюме последнего свежего звонка"],
                        "Тип последнего свежего звонка": contact["Тип последнего свежего звонка"],
                        "Краткая история общения": contact["Краткая история общения"],
                        "Хронология общения (последние 5 касаний)": contact["Хронология общения (последние 5 касаний)"],
                        "Продукты интереса": contact["Продукты интереса"],
                        "Рекомендуемый продукт": contact["Рекомендуемый продукт"],
                        "Возражения": contact["Возражения"],
                        "Следующий шаг": contact["Следующий шаг"],
                        "Рекомендуемая дата следующего контакта": contact["Рекомендуемая дата следующего контакта"],
                        "Приоритет лида": contact["Приоритет лида"],
                        "Вероятность продажи, %": contact["Вероятность продажи, %"],
                        "История общения Tallanto": _safe_text(chain.get("tallanto_history_terms")),
                        "AMO contact IDs": contact["AMO contact IDs"],
                        "AMO lead IDs": contact["AMO lead IDs"],
                        "Outcome source": contact["Outcome source"],
                        "Utility score": contact["Utility score"],
                        "CRM writeback policy": contact["CRM writeback policy"],
                        "CRM writeback blockers": contact["CRM writeback blockers"],
                        "AMO entity policy": contact["AMO entity policy"],
                        "Готово к записи в AMO": amo_ready,
                        "Причина статуса AMO": amo_reason,
                    },
                    CRM_TEXT_FIELDS_TO_SANITIZE,
                )
            )
        elif needs_review:
            manual_review.append(contact)

    return contacts, amo_rows, manual_review


def main() -> int:
    args = _parse_args()
    canonical_db = Path(args.canonical_db).expanduser().resolve()
    client_chains_csv = Path(args.client_chains_csv).expanduser().resolve()
    stage15_summary = Path(args.stage15_summary).expanduser().resolve()
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else _default_out_root().resolve()
    )
    analysis_date = (_parse_dt(args.analysis_date) or datetime.strptime(args.analysis_date, "%Y-%m-%d")).date()
    tenant_config_result = load_tenant_config(args.tenant_config) if args.tenant_config else None
    out_root.mkdir(parents=True, exist_ok=True)
    review_root = out_root / "review_queues"
    review_root.mkdir(parents=True, exist_ok=True)

    chains_by_phone = _load_client_chains(client_chains_csv)
    call_rows = _load_call_rows(canonical_db, args.fresh_from, args.fresh_to)
    contacts, amo_rows, manual_review = _build_contact_rows(
        call_rows=call_rows,
        chains_by_phone=chains_by_phone,
        analysis_date=analysis_date,
        tenant_config=dict(tenant_config_result.config) if tenant_config_result else None,
    )

    _write_csv(out_root / "master_calls_ru.csv", MASTER_CALLS_HEADERS, call_rows)
    _write_csv(out_root / "master_contacts_ru.csv", MASTER_CONTACTS_HEADERS, contacts)
    _write_csv(out_root / "amo_export_ready_ru.csv", AMO_EXPORT_HEADERS, amo_rows)
    _write_csv(review_root / "manual_review_contacts_current.csv", MASTER_CONTACTS_HEADERS, manual_review)

    _write_xlsx(out_root / "master_contacts_ru.xlsx", [("Контакты", MASTER_CONTACTS_HEADERS, contacts)])
    _write_xlsx(out_root / "amo_export_ready_ru.xlsx", [("AMO_Export", AMO_EXPORT_HEADERS, amo_rows)])
    _write_xlsx(review_root / "manual_review_contacts_current.xlsx", [("Manual_Review", MASTER_CONTACTS_HEADERS, manual_review)])
    _write_xlsx(
        out_root / "master_export_pack_ru.xlsx",
        [
            ("Контакты", MASTER_CONTACTS_HEADERS, contacts),
            ("AMO_Export", AMO_EXPORT_HEADERS, amo_rows),
            ("Manual_Review", MASTER_CONTACTS_HEADERS, manual_review),
        ],
    )

    stage15_payload: dict[str, Any] = {}
    if stage15_summary.exists():
        stage15_payload = _json_loads(stage15_summary.read_text(encoding="utf-8"), {})

    summary = {
        "schema_version": "post_backfill_amo_ready_export_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "canonical_db": str(canonical_db),
        "client_chains_csv": str(client_chains_csv),
        "stage15_summary": str(stage15_summary),
        "tenant_config": tenant_config_summary(tenant_config_result),
        "stage15_passed": bool(stage15_payload.get("passed")) if isinstance(stage15_payload, dict) else False,
        "crm_quality_writeback_ready": bool(
            ((stage15_payload.get("readiness") or {}).get("crm_quality_writeback_ready"))
            if isinstance(stage15_payload, dict)
            else False
        ),
        "fresh_from": args.fresh_from,
        "fresh_to": args.fresh_to,
        "analysis_date": analysis_date.isoformat(),
        "master_calls_rows": len(call_rows),
        "master_contacts_rows": len(contacts),
        "amo_export_ready_rows": len(amo_rows),
        "manual_review_rows": len(manual_review),
        "contentful_calls": sum(1 for row in call_rows if row["Содержательный звонок"] == "Да"),
        "non_conversation_calls": sum(1 for row in call_rows if row["Содержательный звонок"] != "Да"),
        "call_type_counts": dict(Counter(row["Тип звонка"] for row in call_rows)),
        "amo_ready_reason_counts": dict(Counter(row["Причина статуса AMO"] for row in contacts)),
        "tallanto_match_status_counts": dict(Counter(row["Статус матчинга Tallanto"] for row in contacts)),
        "crm_writeback_policy_counts": dict(Counter(row.get("CRM writeback policy", "") for row in contacts)),
        "amo_entity_policy_counts": dict(Counter(row.get("AMO entity policy", "") for row in contacts)),
        "amo_ready_by_latest_call_type": dict(Counter(row.get("Тип последнего свежего звонка", "") for row in amo_rows)),
        "blocked_service_or_existing_client_rows": sum(
            1
            for row in contacts
            if row.get("CRM writeback policy") == "blocked_for_live_writeback"
            and row.get("AMO entity policy") == "service_context_manual_review"
        ),
        "blocked_orphan_or_ambiguous_amo_rows": sum(
            1
            for row in contacts
            if row.get("CRM writeback policy") == "blocked_for_live_writeback"
            and row.get("AMO entity policy") == "manual_review_orphan"
        ),
        "output_files": {
            "master_calls_csv": str(out_root / "master_calls_ru.csv"),
            "master_contacts_csv": str(out_root / "master_contacts_ru.csv"),
            "amo_export_ready_csv": str(out_root / "amo_export_ready_ru.csv"),
            "master_contacts_xlsx": str(out_root / "master_contacts_ru.xlsx"),
            "amo_export_ready_xlsx": str(out_root / "amo_export_ready_ru.xlsx"),
            "workbook_xlsx": str(out_root / "master_export_pack_ru.xlsx"),
            "manual_review_csv": str(review_root / "manual_review_contacts_current.csv"),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
