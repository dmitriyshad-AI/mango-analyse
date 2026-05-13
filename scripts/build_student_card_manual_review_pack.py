#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.tenant_text_normalizer import (
    format_objection_list,
    format_product_list,
    normalize_manager_text,
    normalize_objection_label,
    normalize_product_label,
    objection_key,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional runtime dependency
    pd = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "stable_runtime" / "amo_live_stage200_batch3_20260512_v1" / "live_stage200_batch3_candidates_ru.csv",
    ROOT / "stable_runtime" / "amo_live_stage100_batch2_20260512_v1" / "live_stage100_batch2_candidates_ru.csv",
    ROOT / "stable_runtime" / "amo_live_stage100_20260512_v1" / "live_stage100_candidates_ru.csv",
)
DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "student_card_manual_review_next50_20260513_v1"
DEFAULT_CALLS = (
    ROOT
    / "stable_runtime"
    / "insight_readiness_report_after_quality_backfill_20260510_v1"
    / "calls_terminal_analyzed.csv"
)
PREVIOUS_ROP_CHECKED_PHONES = {
    "79057362984",
    "79152124287",
    "79167853078",
    "79265401099",
    "79641490133",
    "79004322255",
    "79096080330",
    "79213095117",
    "79037458510",
    "79261098559",
}
PAYMENT_RE = re.compile(r"\b(оплат|чек|квитанц|счет|счёт|договор|ссылк[аи]\s+на\s+оплат|платеж|платёж)\w*", re.I)
WEAK_NEXT_STEP_RE = re.compile(r"\b(отправить материалы|перезвонить|связаться|уточнить)\b", re.I)
OUT_OF_DOMAIN_RE = re.compile(r"\b(мфти|другая организация|не относится|нецелев|не оставлял|ошибочн)\w*", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a next manual ROP review pack from already written contact-card stages.")
    parser.add_argument("--input", action="append", default=[], help="CSV candidate file. Can be passed multiple times.")
    parser.add_argument("--calls", default=str(DEFAULT_CALLS), help="Post-backfill per-call source for human-readable full chronology.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--sample-size", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = [Path(path).expanduser().resolve() for path in args.input] or [path.resolve() for path in DEFAULT_INPUTS]
    calls_path = Path(args.calls).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    calls_by_phone = _load_calls_by_phone(calls_path)

    rows = []
    for path in input_paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            row["_source_file"] = str(path)
            rows.append(row)

    enriched = [_review_candidate(row, calls_by_phone=calls_by_phone) for row in rows if _eligible(row)]
    selected = _select(enriched, sample_size=max(1, args.sample_size))
    for idx, row in enumerate(selected, start=1):
        row["review_id"] = f"student-card-{idx:03d}"

    preview_csv = out_root / "student_card_manual_review_next50_for_rop.csv"
    preview_xlsx = out_root / "student_card_manual_review_next50_for_rop.xlsx"
    summary_json = out_root / "summary.json"
    guide_md = out_root / "ROP_REVIEW_GUIDE.md"

    _write_csv(preview_csv, selected)
    if pd is not None:
        with pd.ExcelWriter(preview_xlsx, engine="xlsxwriter") as writer:
            pd.DataFrame(selected).to_excel(writer, sheet_name="Manual review 50", index=False)
            workbook = writer.book
            wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
            header = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top", "bg_color": "#D9EAF7"})
            ws = writer.sheets["Manual review 50"]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(1, len(selected)), max(1, len(selected[0]) - 1 if selected else 1))
            ws.set_row(0, None, header)
            ws.set_column(0, 8, 18, wrap)
            ws.set_column(9, 24, 30, wrap)
            ws.set_column(25, 44, 46, wrap)

    bucket_counts = Counter(row["Тип проверки"] for row in selected)
    risk_counts = Counter(row["Почему проверяем"] for row in selected)
    summary = {
        "schema_version": "student_card_manual_review_pack_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "manual ROP quality review of already written contact-card AI fields",
        "input_files": [str(path) for path in input_paths],
        "calls_source": str(calls_path),
        "source_rows": len(rows),
        "eligible_rows": len(enriched),
        "selected_rows": len(selected),
        "bucket_counts": dict(bucket_counts.most_common()),
        "risk_counts": dict(risk_counts.most_common()),
        "outputs": {
            "preview_csv": str(preview_csv),
            "preview_xlsx": str(preview_xlsx),
            "rop_guide": str(guide_md),
        },
        "limitation": "This fallback pack is contact-card quality review, not fresh active-deal selection from live AMO.",
        "format_change": "Technical risk columns were replaced with Russian manager-readable columns; contact history is rebuilt from per-call R+A summaries.",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    guide_md.write_text(_guide(), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(selected) >= args.sample_size else 1


def _eligible(row: dict[str, str]) -> bool:
    phone = _digits(row.get("Телефон клиента"))
    if not phone or phone in PREVIOUS_ROP_CHECKED_PHONES:
        return False
    if _safe_text(row.get("Готово к записи в AMO")).casefold() not in {"да", "yes", "true", "1"}:
        return False
    if _safe_text(row.get("CRM writeback policy")) != "live_update_ready":
        return False
    if not _safe_text(row.get("AMO contact IDs")):
        return False
    return True


def _load_calls_by_phone(path: Path) -> dict[str, list[dict[str, str]]]:
    calls_by_phone: dict[str, list[dict[str, str]]] = {}
    if not path.exists():
        return calls_by_phone
    for row in _read_csv(path):
        if _safe_text(row.get("contentful")).casefold() not in {"true", "1", "yes", "да"}:
            continue
        phone = _digits(row.get("phone"))
        if not phone:
            continue
        calls_by_phone.setdefault(phone, []).append(row)
    for calls in calls_by_phone.values():
        calls.sort(key=lambda item: _safe_text(item.get("started_at")))
    return calls_by_phone


def _review_candidate(row: dict[str, str], *, calls_by_phone: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    text = " ".join(
        [
            _safe_text(row.get("Краткое резюме последнего свежего звонка")),
            _safe_text(row.get("Краткая история общения")),
            _safe_text(row.get("Хронология общения (последние 5 касаний)")),
            _safe_text(row.get("Следующий шаг")),
            _safe_text(row.get("Возражения")),
        ]
    )
    risk_flags = []
    if PAYMENT_RE.search(text):
        risk_flags.append("payment_or_contract_signal")
    if PAYMENT_RE.search(text) and WEAK_NEXT_STEP_RE.search(_safe_text(row.get("Следующий шаг"))):
        risk_flags.append("payment_next_step_conflict_risk")
    if _safe_text(row.get("Статус матчинга Tallanto")) != "exact_phone_single":
        risk_flags.append("non_exact_tallanto_match")
    if _safe_int(row.get("Содержательных звонков в истории")) >= 5:
        risk_flags.append("long_history")
    if OUT_OF_DOMAIN_RE.search(text):
        risk_flags.append("out_of_domain_or_wrong_direction_risk")
    if _safe_int(row.get("Несодержательных звонков в истории")) > _safe_int(row.get("Содержательных звонков в истории")):
        risk_flags.append("many_non_conversation_calls")

    sample_bucket = "baseline"
    if "payment_or_contract_signal" in risk_flags:
        sample_bucket = "payment_boundary"
    elif "out_of_domain_or_wrong_direction_risk" in risk_flags:
        sample_bucket = "wrong_direction_boundary"
    elif "non_exact_tallanto_match" in risk_flags:
        sample_bucket = "tallanto_match_boundary"
    elif "long_history" in risk_flags:
        sample_bucket = "long_history"

    phone_digits = _digits(row.get("Телефон клиента"))
    contact_id = _first_id(row.get("AMO contact IDs"))
    call_rows = calls_by_phone.get(phone_digits, [])
    full_chronology = _full_call_chronology(call_rows)
    manager_summary = _manager_contact_summary(row=row, call_rows=call_rows)
    return {
        "review_id": "",
        "_sample_bucket": sample_bucket,
        "Тип проверки": _sample_bucket_ru(sample_bucket),
        "Почему проверяем": _risk_flags_ru(risk_flags),
        "Телефон": _format_phone(row.get("Телефон клиента")),
        "Контакт AMO": f"https://educent.amocrm.ru/contacts/detail/{contact_id}" if contact_id else "",
        "AMO contact IDs": _safe_text(row.get("AMO contact IDs")),
        "AMO lead IDs": _safe_text(row.get("AMO lead IDs")),
        "ФИО родителя": _safe_text(row.get("ФИО родителя")),
        "ФИО ребенка": _safe_text(row.get("ФИО ребенка")),
        "Содержательных звонков": _safe_int(row.get("Содержательных звонков в истории")),
        "Несодержательных звонков": _safe_int(row.get("Несодержательных звонков в истории")),
        "Последний звонок": _safe_text(row.get("Последний свежий звонок")),
        "Менеджер последнего звонка": _safe_text(row.get("Менеджер последнего свежего звонка")),
        "Рекомендуемый продукт": normalize_product_label(row.get("Рекомендуемый продукт")),
        "Продукты интереса": format_product_list(row.get("Продукты интереса"), max_items=8),
        "Возражения": _format_objections_for_display(row.get("Возражения")),
        "Следующий шаг": normalize_manager_text(row.get("Следующий шаг")),
        "Приоритет": _safe_text(row.get("Приоритет лида")),
        "Вероятность продажи": _safe_text(row.get("Вероятность продажи, %")),
        "Статус Tallanto": _safe_text(row.get("Статус матчинга Tallanto")),
        "ID Tallanto": _safe_text(row.get("ID Tallanto")),
        "Тип ученика Tallanto": _safe_text(row.get("Тип ученика Tallanto")),
        "Филиал Tallanto": _safe_text(row.get("Филиал Tallanto")),
        "Краткое резюме последнего звонка": normalize_manager_text(row.get("Краткое резюме последнего свежего звонка")),
        "Авто история общения": manager_summary,
        "Хронология общения": full_chronology,
        "Что проверить": _what_to_check(risk_flags),
        "rop_ok": "",
        "rop_wrong_or_stale": "",
        "rop_comment": "",
        "_score": _score(row, risk_flags),
    }


def _sample_bucket_ru(bucket: str) -> str:
    mapping = {
        "payment_boundary": "Платёжный/договорный пограничный случай",
        "wrong_direction_boundary": "Риск нецелевого или чужого запроса",
        "tallanto_match_boundary": "Нет точного сопоставления с Tallanto",
        "long_history": "Длинная история клиента",
        "baseline": "Обычная контрольная проверка",
    }
    return mapping.get(bucket, "Контрольная проверка")


def _risk_flags_ru(risks: list[str]) -> str:
    if not risks:
        return "Специальных рисков не найдено; строка нужна для базового контроля качества."
    mapping = {
        "payment_or_contract_signal": "есть сигнал оплаты, договора, счёта или чека",
        "payment_next_step_conflict_risk": "следующий шаг может спорить с оплатой или договором",
        "non_exact_tallanto_match": "нет точного сопоставления с Tallanto",
        "long_history": "длинная история, возможны устаревшие возражения",
        "out_of_domain_or_wrong_direction_risk": "есть риск нецелевого, чужого или ошибочного запроса",
        "many_non_conversation_calls": "много несодержательных дозвонов относительно содержательных разговоров",
    }
    return "; ".join(mapping.get(risk, risk) for risk in risks) + "."


def _manager_contact_summary(*, row: dict[str, str], call_rows: list[dict[str, str]]) -> str:
    contentful_count = len(call_rows) or _safe_int(row.get("Содержательных звонков в истории"))
    first_seen = _safe_text(row.get("Первый звонок"))
    last_seen = _safe_text(row.get("Последний звонок"))
    products = format_product_list(row.get("Продукты интереса"), max_items=3)
    subjects = _compact_items_from_pipe(row.get("Краткая история общения"), marker="Предметы:", max_items=4)
    current_objections = _extract_current_objections(row.get("Возражения")) or _extract_current_objections(row.get("Краткая история общения"))
    next_step = normalize_manager_text(row.get("Следующий шаг"))
    latest = normalize_manager_text(row.get("Краткое резюме последнего свежего звонка"))
    current_status = _short_current_status(latest=latest, next_step=next_step)
    parts = [
        f"Клиент в истории с {first_seen} по {last_seen}; содержательных звонков: {contentful_count}." if first_seen or last_seen else f"Содержательных звонков: {contentful_count}.",
        f"Основной интерес: {products}." if products else "",
        f"Предметы/направления: {subjects}." if subjects else "",
        f"Текущая ситуация: {current_status}." if current_status else "",
        f"Актуальные ограничения: {current_objections}." if current_objections else "",
        f"Следующий шаг: {next_step}." if next_step else "",
    ]
    return _join_sentences(parts, max_chars=1200)


def _full_call_chronology(call_rows: list[dict[str, str]]) -> str:
    if not call_rows:
        return "Нет детальной хронологии из post-backfill per-call слоя."
    lines = [_call_timeline_line(row) for row in call_rows]
    text = "\n".join(line for line in lines if line)
    if len(text) <= 30000:
        return text
    compact_lines = [_call_timeline_line(row, max_summary_chars=280) for row in call_rows]
    text = "\n".join(line for line in compact_lines if line)
    if len(text) <= 30000:
        return text
    return text[:29900].rstrip() + "\n[Хронология слишком длинная для одной Excel-ячейки; нужна отдельная карточка клиента.]"


def _call_timeline_line(row: dict[str, str], *, max_summary_chars: int = 520) -> str:
    started = _safe_text(row.get("started_at"))
    date = _format_display_datetime(started)
    manager = _safe_text(row.get("manager_name"))
    call_type = _call_type_ru(row.get("call_type"))
    summary = _clean_call_summary(row.get("history_summary"), manager=manager)
    summary = _first_sentences(summary, max_sentences=2, max_chars=max_summary_chars)
    next_step = _safe_text(row.get("next_step"))
    tail = f" Следующий шаг: {next_step}." if next_step and next_step.casefold() not in summary.casefold() else ""
    return f"{date} — {manager} ({call_type}): {summary}{tail}".strip()


def _clean_call_summary(value: Any, *, manager: str) -> str:
    text = normalize_manager_text(value)
    if not text:
        return ""
    manager_pattern = re.escape(manager) if manager else r"[^.]{1,80}"
    text = re.sub(
        rf"^\d{{2}}\.\d{{2}}\.\d{{4}}\s+\d{{2}}:\d{{2}}\s+менеджер\s+{manager_pattern}\s+общался\s+с\s+клиентом\.\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*Итог:\s*", " Итог: ", text)
    return text.strip()


def _first_sentences(text: str, *, max_sentences: int, max_chars: int) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return _clip_text(text, max_chars=max_chars)
    result = " ".join(sentences[:max_sentences])
    return _clip_text(result, max_chars=max_chars)


def _clip_text(text: str, *, max_chars: int) -> str:
    text = _safe_text(text)
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars + 1]
    boundary = max(clipped.rfind(". "), clipped.rfind("; "))
    if boundary >= int(max_chars * 0.6):
        return clipped[: boundary + 1].strip()
    hard = clipped[:max_chars].rstrip(" ,.;:-")
    return hard + " [сокращено]"


def _format_display_datetime(value: str) -> str:
    value = _safe_text(value)
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", value)
    if not match:
        return value
    year, month, day, hour, minute = match.groups()
    return f"{day}.{month}.{year} {hour}:{minute}"


def _call_type_ru(value: Any) -> str:
    mapping = {
        "sales_call": "продажа",
        "service_call": "сервис",
        "existing_client_progress": "действующий клиент",
        "technical_call": "технический",
        "non_conversation": "несодержательный",
    }
    return mapping.get(_safe_text(value), _safe_text(value))


def _short_current_status(*, latest: str, next_step: str) -> str:
    text = _safe_text(latest)
    if not text:
        return ""
    text = _clean_call_summary(text, manager="")
    sentence = _first_sentences(text, max_sentences=1, max_chars=260)
    if next_step:
        return f"{sentence} Нужно: {next_step}".strip()
    return sentence


def _extract_current_objections(value: Any) -> str:
    text = normalize_manager_text(value)
    if not text:
        return ""
    match = re.search(r"Актуальные:\s*(.*?)(?:\|\s*Исторические:|$)", text, flags=re.I)
    source = match.group(1) if match else text
    source = re.sub(r"\bИсторические:.*$", "", source, flags=re.I).strip()
    return format_objection_list(source, max_items=5)


def _format_objections_for_display(value: Any) -> str:
    text = normalize_manager_text(value)
    if not text:
        return ""
    current = ""
    historical = ""
    current_match = re.search(r"Актуальные:\s*(.*?)(?:\|\s*Исторические:|$)", text, flags=re.I)
    historical_match = re.search(r"Исторические:\s*(.*)$", text, flags=re.I)
    if current_match:
        current = format_objection_list(current_match.group(1), max_items=8)
    if historical_match:
        historical = _format_historical_objections(historical_match.group(1), max_items=8)
    if current or historical:
        parts = []
        if current:
            parts.append(f"Актуальные: {current}")
        if historical:
            parts.append(f"Исторические: {historical}")
        return " | ".join(parts)
    return format_objection_list(text, max_items=10)


def _format_historical_objections(value: Any, *, max_items: int) -> str:
    result: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in re.split(r"\s*\|\s*|;\s*", normalize_manager_text(value)):
        raw = item.strip(" .;,")
        if not raw:
            continue
        date_match = re.match(r"(?P<date>\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)\s*:\s*(?P<label>.*)", raw)
        date_text = date_match.group("date") if date_match else ""
        label_raw = date_match.group("label") if date_match else raw
        label = normalize_objection_label(label_raw)
        key = (date_text, objection_key(label))
        if not label or key in seen:
            continue
        seen.add(key)
        result.append(f"{date_text}: {label}" if date_text else label)
        if len(result) >= max_items:
            break
    return " | ".join(result)


def _compact_items(value: Any, *, max_items: int) -> str:
    text = normalize_manager_text(value)
    if not text:
        return ""
    raw_items = [part.strip() for part in re.split(r"\s*\|\s*|;\s*", text) if part.strip()]
    result = []
    seen = set()
    for item in raw_items:
        item = re.sub(r"\s*\(\d+\s+касани[йя]\)", "", item, flags=re.I)
        item = re.sub(r":\s*\d+\b", "", item).strip(" .;")
        if not item or item.casefold() in seen:
            continue
        seen.add(item.casefold())
        result.append(item)
        if len(result) >= max_items:
            break
    return " | ".join(result)


def _compact_items_from_pipe(value: Any, *, marker: str, max_items: int) -> str:
    text = _safe_text(value)
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    for stop in ("Ограничения/возражения:", "Последний содержательный контакт:", "Текущий следующий шаг:"):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
    return _compact_items(tail, max_items=max_items)


def _join_sentences(parts: list[str], *, max_chars: int) -> str:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    return _clip_text(text, max_chars=max_chars)


def _select(rows: list[dict[str, Any]], *, sample_size: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: row["_score"], reverse=True)
    bucket_limits = {
        "payment_boundary": max(10, sample_size // 4),
        "wrong_direction_boundary": max(5, sample_size // 10),
        "tallanto_match_boundary": max(10, sample_size // 4),
        "long_history": max(10, sample_size // 4),
        "baseline": sample_size,
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket, limit in bucket_limits.items():
        added = 0
        for row in rows:
            phone = _digits(row.get("Телефон"))
            if row["_sample_bucket"] != bucket or phone in seen:
                continue
            selected.append({key: value for key, value in row.items() if not key.startswith("_")})
            seen.add(phone)
            added += 1
            if len(selected) >= sample_size or added >= limit:
                break
        if len(selected) >= sample_size:
            break
    for row in rows:
        if len(selected) >= sample_size:
            break
        phone = _digits(row.get("Телефон"))
        if phone in seen:
            continue
        selected.append({key: value for key, value in row.items() if not key.startswith("_")})
        seen.add(phone)
    return selected


def _score(row: dict[str, str], risk_flags: list[str]) -> int:
    score = 0
    score += min(40, _safe_int(row.get("Содержательных звонков в истории")) * 3)
    score += min(25, _safe_int(row.get("Utility score")) // 5)
    score += 25 if "payment_or_contract_signal" in risk_flags else 0
    score += 20 if "payment_next_step_conflict_risk" in risk_flags else 0
    score += 15 if "non_exact_tallanto_match" in risk_flags else 0
    score += 15 if "out_of_domain_or_wrong_direction_risk" in risk_flags else 0
    return score


def _what_to_check(risks: list[str]) -> str:
    checks = ["Оценить, полезна ли сводка в карточке за 30-60 секунд."]
    if "payment_or_contract_signal" in risks:
        checks.append("Проверить, не надо ли сверить оплату/договор вместо обычного перезвона.")
    if "payment_next_step_conflict_risk" in risks:
        checks.append("Проверить конфликт: есть платежный сигнал, но следующий шаг слишком общий.")
    if "non_exact_tallanto_match" in risks:
        checks.append("Проверить, хватает ли данных без точного Tallanto-сопоставления.")
    if "out_of_domain_or_wrong_direction_risk" in risks:
        checks.append("Проверить, не попал ли нерелевантный/чужой запрос.")
    if "long_history" in risks:
        checks.append("Проверить, не слишком ли шумная история и нет ли устаревших возражений.")
    return " ".join(checks)


def _guide() -> str:
    return """# Инструкция РОП для проверки 50 карточек

Проверяйте не “идеальность текста”, а практическую пользу для менеджера:

1. За 30-60 секунд понятно, что происходит с клиентом?
2. Следующий шаг не противоречит оплате, договору, отказу или текущей ситуации?
3. В истории нет явного мусора, чужих организаций, автоответчиков или нецелевых звонков?
4. Возражения выглядят актуальными, а не старыми словами вроде “время/доверие/неактуально” без контекста?
5. Если нет точного Tallanto-сопоставления, это действительно безопасно для карточки?
6. Если есть ошибка, укажите класс проблемы, а не только конкретную строку.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _first_id(value: Any) -> str:
    parts = [part for part in re.split(r"[|,;\s]+", _safe_text(value)) if part]
    return parts[0] if parts else ""


def _digits(value: Any) -> str:
    return re.sub(r"\D+", "", _safe_text(value))


def _format_phone(value: Any) -> str:
    digits = _digits(value)
    if not digits:
        return _safe_text(value)
    return "+" + digits


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    raise SystemExit(main())
