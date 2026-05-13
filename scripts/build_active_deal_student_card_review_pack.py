#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.quality.crm_text_quality_detector import (  # noqa: E402
    detect_crm_text_quality_risks,
    has_blocking_crm_text_findings,
)
from mango_mvp.utils.phone import normalize_phone  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional runtime dependency
    pd = None


ENV_FILES = (
    ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    ROOT / "prod_runtime_transfer" / ".env.private",
)
DEFAULT_CHAINS = (
    ROOT
    / "stable_runtime"
    / "insight_readiness_report_after_quality_backfill_20260510_v1"
    / "client_chains.csv"
)
DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "student_card_active_deal_review_50_20260513_v1"
DEFAULT_AUDIT_INBOX = ROOT / "audits" / "_inbox" / "student_card_active_deal_review_50_20260513_v1"
TERMINAL_STATUS_IDS = {142, 143}
PAYMENT_MARKERS_RE = re.compile(r"\b(оплат|чек|квитанц|счет|счёт|договор|ссылк[аи]\s+на\s+оплат)\w*", re.I)
ACTIVE_STATUS_PRIORITY = {
    "ожидание оплаты": 110,
    "оплата получена": 108,
    "заключение договора": 105,
    "запись в группу": 100,
    "принимают решение": 95,
    "переговоры": 90,
    "в работе": 80,
    "перспектива": 70,
    "недозвон": 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only ROP review pack from live active AMO deals matched to the "
            "post-backfill phone-chain layer."
        )
    )
    parser.add_argument("--chains", default=str(DEFAULT_CHAINS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--audit-inbox", default=str(DEFAULT_AUDIT_INBOX))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--candidate-pool-size", type=int, default=240)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means fetch all available AMO lead pages.")
    parser.add_argument("--page-limit", type=int, default=250)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--analysis-date", default=datetime.now(timezone.utc).date().isoformat())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_env_files()

    from mango_mvp.amocrm_runtime.amo_integration import (  # noqa: PLC0415
        fetch_contact,
        fetch_pipelines_with_statuses,
        fetch_users,
        get_amo_connection_status,
    )
    from mango_mvp.amocrm_runtime.db import SessionLocal  # noqa: PLC0415

    chains_path = Path(args.chains).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    audit_inbox = Path(args.audit_inbox).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    chains_by_phone = _load_client_chains(chains_path)

    session = SessionLocal()
    contact_cache: dict[int, dict[str, Any]] = {}
    try:
        connection = get_amo_connection_status(session)
        pipelines = fetch_pipelines_with_statuses(session)
        users = fetch_users(session)
        pipeline_map, status_map = _pipeline_status_maps(pipelines)
        user_map = {_safe_int(user.get("id")): _safe_text(user.get("name")) for user in users}
        leads, pages = _fetch_all_leads(
            session,
            page_limit=max(1, min(args.page_limit, 250)),
            max_pages=max(0, args.max_pages),
            sleep_sec=max(0.0, args.sleep_sec),
        )

        lead_scan_counts: Counter[str] = Counter()
        candidates: list[dict[str, Any]] = []
        for lead in _lead_scan_order(leads):
            if len(candidates) >= max(args.candidate_pool_size, args.sample_size):
                break
            candidate = _candidate_from_lead(
                session=session,
                lead=lead,
                chains_by_phone=chains_by_phone,
                contact_cache=contact_cache,
                pipeline_map=pipeline_map,
                status_map=status_map,
                user_map=user_map,
                scan_counts=lead_scan_counts,
                analysis_date=args.analysis_date,
            )
            if candidate is not None:
                candidates.append(candidate)
                if len(candidates) % 25 == 0:
                    print(f"matched {len(candidates)} candidate active deals", flush=True)
    finally:
        session.close()

    selected = _select_review_rows(candidates, sample_size=max(1, args.sample_size))
    outputs = _write_outputs(
        out_root=out_root,
        audit_inbox=audit_inbox,
        chains_path=chains_path,
        selected=selected,
        candidates=candidates,
        lead_scan_counts=lead_scan_counts,
        connection=connection,
        pages=pages,
        contact_cache_size=len(contact_cache),
        analysis_date=args.analysis_date,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0 if len(selected) >= args.sample_size else 1


def _load_env_files() -> None:
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _load_client_chains(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        phone = normalize_phone(row.get("phone") or row.get("Телефон") or "")
        if not phone:
            continue
        result[phone] = row
        if len(phone) >= 10:
            result[phone[-10:]] = row
    return result


def _fetch_all_leads(
    session: Any,
    *,
    page_limit: int,
    max_pages: int,
    sleep_sec: float,
) -> tuple[list[dict[str, Any]], int]:
    from mango_mvp.amocrm_runtime.amo_integration import amo_api_request  # noqa: PLC0415

    params = {"limit": page_limit, "with": "contacts"}
    next_url: str | None = "/api/v4/leads"
    items: list[dict[str, Any]] = []
    pages = 0
    while next_url:
        payload = amo_api_request(session, method="GET", path_or_url=next_url, params=params if pages == 0 else None)
        pages += 1
        embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
        leads = embedded.get("leads") if isinstance(embedded, dict) else []
        if isinstance(leads, list):
            items.extend(lead for lead in leads if isinstance(lead, dict))
        if max_pages and pages >= max_pages:
            break
        next_link = None
        links = payload.get("_links") if isinstance(payload, dict) else {}
        if isinstance(links, dict):
            next_meta = links.get("next")
            if isinstance(next_meta, dict):
                next_link = next_meta.get("href")
        next_url = next_link
        params = None
        if next_url and sleep_sec:
            time.sleep(sleep_sec)
    return items, pages


def _lead_scan_order(leads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [lead for lead in leads if _safe_int(lead.get("status_id")) not in TERMINAL_STATUS_IDS]
    return sorted(active, key=lambda lead: (_status_priority_text(lead), _safe_int(lead.get("updated_at"))), reverse=True)


def _candidate_from_lead(
    *,
    session: Any,
    lead: dict[str, Any],
    chains_by_phone: dict[str, dict[str, str]],
    contact_cache: dict[int, dict[str, Any]],
    pipeline_map: dict[int, dict[str, Any]],
    status_map: dict[tuple[int, int], dict[str, Any]],
    user_map: dict[int, str],
    scan_counts: Counter[str],
    analysis_date: str,
) -> dict[str, Any] | None:
    lead_id = _safe_int(lead.get("id"))
    status_id = _safe_int(lead.get("status_id"))
    pipeline_id = _safe_int(lead.get("pipeline_id"))
    if not lead_id or status_id in TERMINAL_STATUS_IDS:
        scan_counts["terminal_or_no_id"] += 1
        return None

    contact_ids = _lead_contact_ids(lead)
    if len(contact_ids) != 1:
        scan_counts["not_single_contact"] += 1
        return None
    contact_id = contact_ids[0]
    contact = contact_cache.get(contact_id)
    if contact is None:
        from mango_mvp.amocrm_runtime.amo_integration import AmoIntegrationError  # noqa: PLC0415

        try:
            contact = fetch_contact_cached(session, contact_id=contact_id)
        except AmoIntegrationError:
            scan_counts["contact_fetch_error"] += 1
            return None
        contact_cache[contact_id] = contact

    phones = _contact_phones(contact)
    if not phones:
        scan_counts["contact_without_phone"] += 1
        return None
    chain = _first_matching_chain(phones, chains_by_phone)
    if chain is None:
        scan_counts["no_phone_chain_match"] += 1
        return None
    if _safe_int(chain.get("contentful_call_count")) <= 0:
        scan_counts["no_contentful_calls"] += 1
        return None
    if _safe_int(chain.get("sales_call_count")) <= 0:
        scan_counts["no_sales_calls"] += 1
        return None

    pipeline_name = _safe_text((pipeline_map.get(pipeline_id) or {}).get("name"))
    status_name = _safe_text((status_map.get((pipeline_id, status_id)) or {}).get("name"))
    lead_name = _safe_text(lead.get("name"))
    contact_name = _safe_text(contact.get("name"))
    responsible = user_map.get(_safe_int(lead.get("responsible_user_id")), str(_safe_int(lead.get("responsible_user_id")) or ""))
    phone = _best_phone_for_chain(phones, chain)

    sample_bucket = _sample_bucket(lead=lead, chain=chain, status_name=status_name)
    contact_history = _contact_history_text(chain)
    contact_summary = _contact_summary_text(chain)
    deal_summary = _deal_summary_text(lead_name=lead_name, pipeline=pipeline_name, status=status_name, chain=chain)
    deal_next_step = _deal_next_step_text(chain=chain, status=status_name)
    deal_warning = _deal_warning_text(lead=lead, chain=chain, status=status_name)

    quality_payload = {
        "Контакт: Авто история общения": contact_history,
        "Контакт: Последняя AI-сводка": contact_summary,
        "Сделка: AI-сводка по сделке": deal_summary,
        "Сделка: AI-рекомендованный следующий шаг": deal_next_step,
        "Сделка: AI-предупреждение": deal_warning,
        "AMO статус сделки": status_name,
        "AMO status_id": str(status_id),
    }
    findings = detect_crm_text_quality_risks(quality_payload, analysis_date=analysis_date, min_severity="P3")
    blocking = [finding for finding in findings if has_blocking_crm_text_findings([finding])]
    live_eligible = "нет" if blocking else "да"
    quality_decision = "needs_review" if blocking else "allow"

    review_row = {
        "review_id": "",
        "sample_bucket": sample_bucket,
        "live_eligible_preview": live_eligible,
        "control_reason": _control_reason(sample_bucket),
        "Телефон": _format_phone(phone),
        "Контакт AMO": f"https://educent.amocrm.ru/contacts/detail/{contact_id}",
        "Сделка AMO": f"https://educent.amocrm.ru/leads/detail/{lead_id}",
        "ID контакта AMO": contact_id,
        "ID сделки AMO": lead_id,
        "Контакт AMO ФИО": contact_name,
        "Название сделки": lead_name,
        "Воронка": pipeline_name,
        "Статус сделки": status_name,
        "status_id": status_id,
        "Ответственный AMO": responsible,
        "Дата создания сделки": _format_ts(lead.get("created_at")),
        "Дата обновления сделки": _format_ts(lead.get("updated_at")),
        "Содержательных звонков": _safe_int(chain.get("contentful_call_count")),
        "Sales-звонков": _safe_int(chain.get("sales_call_count")),
        "Последний звонок": _safe_text(chain.get("last_seen_at")),
        "Доминантный тип звонков": _safe_text(chain.get("dominant_call_type")),
        "Менеджеры по звонкам": _safe_text(chain.get("managers")),
        "Продукты по истории": _safe_text(chain.get("products_top")),
        "Предметы по истории": _safe_text(chain.get("subjects_top")),
        "Возражения по истории": _safe_text(chain.get("objections_top")),
        "Tallanto match": _safe_text(chain.get("has_tallanto_match")),
        "Tallanto IDs": _safe_text(chain.get("tallanto_ids")),
        "Tallanto типы": _safe_text(chain.get("tallanto_student_types")),
        "Tallanto филиалы": _safe_text(chain.get("tallanto_branches")),
        "AMO связи из истории": _safe_text(chain.get("amo_lead_ids")),
        "Контакт: Авто история общения": contact_history,
        "Контакт: Последняя AI-сводка": contact_summary,
        "Сделка: AI-сводка по сделке": deal_summary,
        "Сделка: AI-рекомендованный следующий шаг": deal_next_step,
        "Сделка: AI-предупреждение": deal_warning,
        "Quality decision": quality_decision,
        "Quality blockers": " | ".join(sorted({finding.risk_type for finding in blocking})),
        "Quality blocker evidence": " | ".join(f"{finding.risk_type}: {finding.matched_text}" for finding in blocking[:5]),
        "risk_types": " | ".join(sorted({finding.risk_type for finding in findings})),
        "Что проверить РОП": (
            "1) Это правильная активная сделка? 2) Следующий шаг не спорит со статусом/оплатой? "
            "3) Контакт и сделка не дублируют один и тот же текст? 4) Нужны ли данные из Tallanto/мессенджеров?"
        ),
        "rop_right_deal": "",
        "rop_next_step_ok": "",
        "rop_duplicate_text": "",
        "rop_payment_conflict": "",
        "rop_tallanto_missing": "",
        "rop_verdict": "",
        "rop_comment": "",
    }
    score = _candidate_score(lead=lead, chain=chain, status_name=status_name, sample_bucket=sample_bucket, blocking=bool(blocking))
    return {
        "review_row": review_row,
        "score": score,
        "sample_bucket": sample_bucket,
        "lead": _slim_lead(lead, pipeline_name=pipeline_name, status_name=status_name),
        "contact": {
            "id": contact_id,
            "name": contact_name,
            "phones": phones,
        },
        "chain": chain,
        "quality_findings": [
            {
                "risk_type": finding.risk_type,
                "severity": finding.severity,
                "field": finding.field,
                "matched_text": finding.matched_text,
                "reason": finding.reason,
            }
            for finding in findings
        ],
    }


def fetch_contact_cached(session: Any, *, contact_id: int) -> dict[str, Any]:
    from mango_mvp.amocrm_runtime.amo_integration import fetch_contact  # noqa: PLC0415

    return fetch_contact(session, contact_id=contact_id, with_fields="leads")


def _select_review_rows(candidates: list[dict[str, Any]], *, sample_size: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    bucket_limits = {
        "payment_boundary": max(6, sample_size // 5),
        "active_current": max(26, int(sample_size * 0.7)),
        "tallanto_missing": max(5, sample_size // 8),
        "needs_review_quality": max(5, sample_size // 8),
    }
    selected: list[dict[str, Any]] = []
    seen_phones: set[str] = set()
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_bucket[candidate["sample_bucket"]].append(candidate)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=lambda item: item["score"], reverse=True)

    for bucket, limit in bucket_limits.items():
        for candidate in by_bucket.get(bucket, [])[: limit * 3]:
            if len(selected) >= sample_size:
                break
            phone = normalize_phone(candidate["review_row"].get("Телефон") or "")
            if phone in seen_phones:
                continue
            selected.append(candidate)
            seen_phones.add(phone)
            if sum(1 for item in selected if item["sample_bucket"] == bucket) >= limit:
                break

    remaining = sorted(candidates, key=lambda item: item["score"], reverse=True)
    for candidate in remaining:
        if len(selected) >= sample_size:
            break
        phone = normalize_phone(candidate["review_row"].get("Телефон") or "")
        if phone in seen_phones:
            continue
        selected.append(candidate)
        seen_phones.add(phone)

    for idx, candidate in enumerate(selected, start=1):
        candidate["review_row"]["review_id"] = f"active-deal-{idx:03d}"
    return selected


def _candidate_score(
    *,
    lead: dict[str, Any],
    chain: dict[str, str],
    status_name: str,
    sample_bucket: str,
    blocking: bool,
) -> float:
    score = _status_priority(status_name)
    score += min(30, _safe_int(chain.get("contentful_call_count")))
    score += min(20, _safe_int(chain.get("sales_call_count")))
    score += min(25, _safe_int(chain.get("utility_score")) / 10)
    score += min(15, max(0, (_safe_int(lead.get("updated_at")) - 1_700_000_000) / 20_000_000))
    if sample_bucket == "payment_boundary":
        score += 20
    if sample_bucket == "tallanto_missing":
        score += 10
    if blocking:
        score -= 40
    return round(score, 2)


def _sample_bucket(*, lead: dict[str, Any], chain: dict[str, str], status_name: str) -> str:
    text = " ".join(
        [
            _safe_text(lead.get("name")),
            status_name,
            _safe_text(chain.get("example_latest_summary")),
            _safe_text(chain.get("products_top")),
            _safe_text(chain.get("objections_top")),
        ]
    )
    if PAYMENT_MARKERS_RE.search(text):
        return "payment_boundary"
    if _safe_text(chain.get("has_tallanto_match")).casefold() not in {"true", "1", "yes", "да"}:
        return "tallanto_missing"
    return "active_current"


def _control_reason(bucket: str) -> str:
    if bucket == "payment_boundary":
        return "Проверить, не конфликтует ли рекомендация с оплатой, чеком, договором или счетом."
    if bucket == "tallanto_missing":
        return "Проверить, хватает ли данных без точного Tallanto-контекста."
    if bucket == "needs_review_quality":
        return "Строка специально включена как пограничная по качеству текста."
    return "Проверить базовый сценарий активной сделки."


def _contact_history_text(chain: dict[str, str]) -> str:
    parts = [
        f"История звонков: с {_safe_text(chain.get('first_seen_at'))} по {_safe_text(chain.get('last_seen_at'))}.",
        f"Содержательных звонков: {_safe_int(chain.get('contentful_call_count'))}; sales-звонков: {_safe_int(chain.get('sales_call_count'))}.",
        f"Продукты: {_compact_list(chain.get('products_top'))}." if _safe_text(chain.get("products_top")) else "",
        f"Предметы: {_compact_list(chain.get('subjects_top'))}." if _safe_text(chain.get("subjects_top")) else "",
        f"Менеджеры: {_safe_text(chain.get('managers'))}." if _safe_text(chain.get("managers")) else "",
    ]
    return _join(parts, max_chars=900)


def _contact_summary_text(chain: dict[str, str]) -> str:
    latest = _compact_sentence(chain.get("example_latest_summary"), max_chars=560)
    objections = _compact_list(chain.get("objections_top"))
    if objections:
        return _join([latest, f"Исторические ограничения/возражения: {objections}."], max_chars=900)
    return latest


def _deal_summary_text(*, lead_name: str, pipeline: str, status: str, chain: dict[str, str]) -> str:
    product = _compact_list(chain.get("products_top"), max_items=3)
    parts = [
        f"Активная сделка: {lead_name}.",
        f"Воронка/статус: {pipeline} / {status}.",
        f"Интерес по звонкам: {product}." if product else "",
    ]
    return _join(parts, max_chars=520)


def _deal_next_step_text(*, chain: dict[str, str], status: str) -> str:
    latest = _safe_text(chain.get("example_latest_summary"))
    status_l = status.casefold()
    if "оплат" in status_l:
        return "Проверить оплату и документы в AMO/Tallanto, затем обновить сделку."
    if re.search(r"\b(оплат|чек|квитанц|договор)\w*", latest, re.I):
        return "Сверить оплату/договор с AMO и Tallanto, затем определить следующий контакт."
    if "недозвон" in status_l:
        return "Проверить последнюю задачу и повторить контакт по регламенту."
    if "принимают решение" in status_l:
        return "Связаться с клиентом и уточнить, что мешает принять решение."
    return "Проверить актуальную задачу по сделке и продолжить контакт по статусу."


def _deal_warning_text(*, lead: dict[str, Any], chain: dict[str, str], status: str) -> str:
    warnings = []
    if _safe_text(chain.get("has_tallanto_match")).casefold() not in {"true", "1", "yes", "да"}:
        warnings.append("Нет точного Tallanto-сопоставления: оплату, группы и занятия нужно проверять отдельно.")
    if PAYMENT_MARKERS_RE.search(" ".join([status, _safe_text(chain.get("example_latest_summary"))])):
        warnings.append("Есть платежный/договорный сигнал: нельзя предлагать обычный follow-up без сверки оплаты.")
    if _safe_int(chain.get("amo_lead_ids_count")) > 1:
        warnings.append("У клиента несколько AMO-сделок в истории: РОП должен подтвердить выбранную сделку.")
    return " ".join(warnings) if warnings else "Критичных предупреждений для ручной проверки нет."


def _write_outputs(
    *,
    out_root: Path,
    audit_inbox: Path,
    chains_path: Path,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    lead_scan_counts: Counter[str],
    connection: dict[str, Any],
    pages: int,
    contact_cache_size: int,
    analysis_date: str,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    preview_rows = [item["review_row"] for item in selected]
    quality_rows = [
        {
            "review_id": item["review_row"]["review_id"],
            "phone": item["review_row"]["Телефон"],
            "lead_id": item["review_row"]["ID сделки AMO"],
            "decision": item["review_row"]["Quality decision"],
            "risk_types": item["review_row"]["risk_types"],
            "blocking": item["review_row"]["Quality blockers"],
            "evidence": item["review_row"]["Quality blocker evidence"],
        }
        for item in selected
    ]
    payload_rows = [
        {
            "review_id": item["review_row"]["review_id"],
            "lead": item["lead"],
            "contact": item["contact"],
            "chain": item["chain"],
            "quality_findings": item["quality_findings"],
        }
        for item in selected
    ]

    preview_csv = out_root / "student_card_active_deal_review_50_for_rop.csv"
    preview_xlsx = out_root / "student_card_active_deal_review_50_for_rop.xlsx"
    quality_csv = out_root / "quality_findings.csv"
    payload_jsonl = out_root / "active_deal_review_payloads.jsonl"
    summary_json = out_root / "summary.json"
    readme = out_root / "README.md"
    rop_guide = out_root / "ROP_REVIEW_GUIDE.md"
    audit_scope = out_root / "CLAUDE_AUDIT_SCOPE.md"

    _write_csv(preview_csv, preview_rows)
    _write_csv(quality_csv, quality_rows)
    payload_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in payload_rows) + "\n", encoding="utf-8")
    if pd is not None:
        with pd.ExcelWriter(preview_xlsx, engine="xlsxwriter") as writer:
            pd.DataFrame(preview_rows).to_excel(writer, sheet_name="ROP active deals", index=False)
            pd.DataFrame(quality_rows).to_excel(writer, sheet_name="quality", index=False)
            workbook = writer.book
            wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
            header = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top", "bg_color": "#D9EAF7"})
            for sheet_name in ("ROP active deals", "quality"):
                ws = writer.sheets[sheet_name]
                ws.freeze_panes(1, 0)
                ws.autofilter(0, 0, max(1, len(preview_rows)), max(1, len(preview_rows[0]) - 1 if preview_rows else 1))
                ws.set_row(0, None, header)
                ws.set_column(0, 7, 18, wrap)
                ws.set_column(8, 22, 24, wrap)
                ws.set_column(23, 40, 46, wrap)

    bucket_counts = Counter(item["sample_bucket"] for item in selected)
    quality_counts = Counter(row["Quality decision"] for row in preview_rows)
    summary = {
        "schema_version": "student_card_active_deal_review_pack_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_date": analysis_date,
        "purpose": "manual ROP review of contact-card history against active AMO deal selection",
        "chains_input": str(chains_path),
        "rows_selected": len(selected),
        "candidate_pool": len(candidates),
        "bucket_counts": dict(bucket_counts.most_common()),
        "quality_decision_counts": dict(quality_counts.most_common()),
        "lead_fetch_pages": pages,
        "contact_fetches": contact_cache_size,
        "scan_counts": dict(lead_scan_counts.most_common()),
        "connection": {
            "status": connection.get("status"),
            "connected": connection.get("connected"),
            "account_base_url": connection.get("account_base_url"),
        },
        "outputs": {
            "preview_csv": str(preview_csv),
            "preview_xlsx": str(preview_xlsx),
            "quality_csv": str(quality_csv),
            "payload_jsonl": str(payload_jsonl),
            "readme": str(readme),
            "rop_guide": str(rop_guide),
            "audit_scope": str(audit_scope),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme.write_text(_readme(summary), encoding="utf-8")
    rop_guide.write_text(_rop_guide(), encoding="utf-8")
    audit_scope.write_text(_audit_scope(), encoding="utf-8")

    if audit_inbox.exists():
        shutil.rmtree(audit_inbox)
    audit_inbox.mkdir(parents=True, exist_ok=True)
    for path in (preview_csv, preview_xlsx, quality_csv, payload_jsonl, summary_json, readme, rop_guide, audit_scope):
        if path.exists():
            shutil.copy2(path, audit_inbox / path.name)
    return {**summary, "audit_inbox": str(audit_inbox)}


def _readme(summary: dict[str, Any]) -> str:
    return f"""# Student card active deal review 50

This is a read-only manual review pack. It is not a live-write package.

Goal: verify whether our cleaned phone-chain history can be safely attached to the right active AMO deal.

Rows selected: {summary['rows_selected']}.
Candidate pool: {summary['candidate_pool']}.
Buckets: {json.dumps(summary['bucket_counts'], ensure_ascii=False)}.

Use `student_card_active_deal_review_50_for_rop.xlsx` as the main file.
"""


def _rop_guide() -> str:
    return """# ROP review guide

For each row, check:

1. Is the AMO deal really the right active deal for this client?
2. Does the recommended next step fit the current deal status?
3. Is there a payment/check/contract signal that changes the next step?
4. Does the contact field contain overall client history, while the deal field contains only deal-specific information?
5. Is important Tallanto/payment/group context missing?
6. Would this text help a manager in 30-60 seconds, or is it too noisy?

Fill the `rop_*` columns if possible:

- `rop_right_deal`: yes/no/unclear.
- `rop_next_step_ok`: yes/no/unclear.
- `rop_duplicate_text`: yes/no.
- `rop_payment_conflict`: yes/no.
- `rop_tallanto_missing`: yes/no.
- `rop_verdict`: ok / needs_fix / block.
- `rop_comment`: short reason.
"""


def _audit_scope() -> str:
    return """# Claude audit scope

Audit this read-only ROP review pack.

Do not write outside `audits/_results`.
Do not run live AMO writes.
Do not run ASR/R+A.

Check:

1. The pack uses active non-terminal AMO deals, not closed/lost deals.
2. Each row has one AMO contact and one selected AMO deal.
3. Contact summary and deal summary have distinct roles and do not duplicate the same content.
4. Payment/contract signals are not converted into a naive sales follow-up.
5. Missing Tallanto context is clearly flagged.
6. ROP can understand what to check without reading code.

Return PASS / PASS_WITH_LIMITATIONS / FAIL plus findings.csv and row_decisions.csv.
"""


def _pipeline_status_maps(
    pipelines: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    pipeline_map: dict[int, dict[str, Any]] = {}
    status_map: dict[tuple[int, int], dict[str, Any]] = {}
    for pipeline in pipelines:
        pipeline_id = _safe_int(pipeline.get("id"))
        if not pipeline_id:
            continue
        pipeline_map[pipeline_id] = pipeline
        for status in (pipeline.get("_embedded") or {}).get("statuses") or []:
            status_id = _safe_int(status.get("id"))
            if status_id:
                status_map[(pipeline_id, status_id)] = status
    return pipeline_map, status_map


def _lead_contact_ids(lead: dict[str, Any]) -> list[int]:
    contacts = (lead.get("_embedded") or {}).get("contacts") or []
    ids: list[int] = []
    for contact in contacts:
        contact_id = _safe_int(contact.get("id") if isinstance(contact, dict) else contact)
        if contact_id and contact_id not in ids:
            ids.append(contact_id)
    return ids


def _contact_phones(contact: dict[str, Any]) -> list[str]:
    phones: list[str] = []
    for field in contact.get("custom_fields_values") or []:
        field_name = _safe_text(field.get("field_name")).casefold()
        field_code = _safe_text(field.get("field_code")).casefold()
        if "phone" not in field_code and "телефон" not in field_name:
            continue
        for item in field.get("values") or []:
            phone = normalize_phone(item.get("value") or "")
            if phone and phone not in phones:
                phones.append(phone)
    return phones


def _first_matching_chain(phones: list[str], chains_by_phone: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for phone in phones:
        for key in (phone, phone[-10:]):
            if key in chains_by_phone:
                return chains_by_phone[key]
    return None


def _best_phone_for_chain(phones: list[str], chain: dict[str, str]) -> str:
    chain_phone = normalize_phone(chain.get("phone") or "")
    for phone in phones:
        if phone == chain_phone or phone[-10:] == chain_phone[-10:]:
            return phone
    return phones[0] if phones else chain_phone


def _slim_lead(lead: dict[str, Any], *, pipeline_name: str, status_name: str) -> dict[str, Any]:
    return {
        "id": lead.get("id"),
        "name": lead.get("name"),
        "pipeline_id": lead.get("pipeline_id"),
        "pipeline_name": pipeline_name,
        "status_id": lead.get("status_id"),
        "status_name": status_name,
        "responsible_user_id": lead.get("responsible_user_id"),
        "created_at": lead.get("created_at"),
        "updated_at": lead.get("updated_at"),
        "closed_at": lead.get("closed_at"),
    }


def _status_priority_text(lead: dict[str, Any]) -> int:
    # Used before status names are resolved. Prefer recently updated active deals.
    return 50 if _safe_int(lead.get("status_id")) not in TERMINAL_STATUS_IDS else 0


def _status_priority(status_name: str) -> int:
    text = status_name.casefold()
    for marker, value in ACTIVE_STATUS_PRIORITY.items():
        if marker in text:
            return value
    return 60


def _compact_list(value: Any, *, max_items: int = 5) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    items = [re.sub(r":\s*\d+\b", "", part).strip() for part in re.split(r"\s*\|\s*", text) if part.strip()]
    deduped = []
    seen = set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return " | ".join(deduped[:max_items])


def _compact_sentence(value: Any, *, max_chars: int) -> str:
    text = _safe_text(value)
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars + 1]
    boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(" | "), clipped.rfind(", "))
    if boundary >= int(max_chars * 0.55):
        return clipped[: boundary + 1].strip()
    return clipped[:max_chars].rstrip(" ,.;:-")


def _join(parts: list[str], *, max_chars: int) -> str:
    return _compact_sentence(" ".join(part.strip() for part in parts if part and part.strip()), max_chars=max_chars)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _format_phone(value: Any) -> str:
    phone = normalize_phone(_safe_text(value))
    if not phone:
        return _safe_text(value)
    return "+" + phone if not phone.startswith("+") else phone


def _format_ts(value: Any) -> str:
    ts = _safe_int(value)
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
    except (OSError, ValueError):
        return ""


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


if __name__ == "__main__":
    raise SystemExit(main())
