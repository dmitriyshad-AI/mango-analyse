#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.utils.phone import normalize_phone  # noqa: E402


DEFAULT_CHAINS = ROOT / "stable_runtime" / "insight_readiness_report_after_quality_backfill_20260510_v1" / "client_chains.csv"
DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "active_deal_manual_review_50_20260513_v1"
ENV_FILES = (
    ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    ROOT / "prod_runtime_transfer" / ".env.private",
)
TERMINAL_STATUS_IDS = {142, 143}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only manual review pack for active AMO deals matched to call phone chains.")
    parser.add_argument("--client-chains", default=str(DEFAULT_CHAINS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--max-leads", type=int, default=2500)
    parser.add_argument("--page-limit", type=int, default=250)
    parser.add_argument("--sleep-sec", type=float, default=0.03)
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

    chains_path = Path(args.client_chains).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    chains = _load_chains(chains_path)

    session = SessionLocal()
    skipped: Counter[str] = Counter()
    contact_cache: dict[int, dict[str, Any]] = {}
    try:
        connection_before = get_amo_connection_status(session)
        pipelines = fetch_pipelines_with_statuses(session)
        status_by_id = _status_by_id(pipelines)
        users = {str(user.get("id")): _safe_text(user.get("name")) for user in fetch_users(session)}
        leads, pages = _fetch_all_leads(
            session,
            limit=max(1, args.max_leads),
            page_limit=max(1, min(args.page_limit, 250)),
            sleep_sec=max(0.0, args.sleep_sec),
        )
        rows: list[dict[str, Any]] = []
        payloads: list[dict[str, Any]] = []
        seen_leads: set[int] = set()
        seen_phones: set[str] = set()
        for lead in _sort_active_leads(leads, status_by_id):
            if len(rows) >= max(1, args.sample_size):
                break
            lead_id = _safe_int(lead.get("id"))
            if not lead_id or lead_id in seen_leads:
                skipped["duplicate_lead"] += 1
                continue
            status_id = _safe_int(lead.get("status_id"))
            if status_id in TERMINAL_STATUS_IDS:
                skipped["terminal_status"] += 1
                continue
            contact_id = _first_contact_id(lead)
            if not contact_id:
                skipped["no_contact"] += 1
                continue
            contact = contact_cache.get(contact_id)
            if contact is None:
                try:
                    contact = fetch_contact(session, contact_id=contact_id, with_fields="leads")
                except Exception as exc:  # pragma: no cover - live API guard
                    skipped[f"contact_fetch_error:{type(exc).__name__}"] += 1
                    continue
                contact_cache[contact_id] = contact
                if args.sleep_sec:
                    time.sleep(max(0.0, args.sleep_sec))
            phones = _contact_phones(contact)
            chain = None
            phone = ""
            for candidate_phone in phones:
                chain = chains.get(candidate_phone)
                if chain:
                    phone = candidate_phone
                    break
            if not chain:
                skipped["no_phone_chain_match"] += 1
                continue
            if phone in seen_phones:
                skipped["duplicate_phone_in_pack"] += 1
                continue
            if _safe_int(chain.get("contentful_call_count")) <= 0:
                skipped["no_contentful_calls"] += 1
                continue
            status_meta = status_by_id.get(status_id, {})
            pipeline_name = _safe_text(status_meta.get("pipeline_name"))
            status_name = _safe_text(status_meta.get("status_name"))
            responsible_name = users.get(str(lead.get("responsible_user_id")), "")
            review_id = f"active-deal-{len(rows) + 1:05d}"
            row = _review_row(
                review_id=review_id,
                phone=phone,
                contact_id=contact_id,
                lead=lead,
                pipeline_name=pipeline_name,
                status_name=status_name,
                responsible_name=responsible_name,
                chain=chain,
            )
            rows.append(row)
            payloads.append(
                {
                    "review_id": review_id,
                    "phone": phone,
                    "contact": _slim_contact(contact),
                    "lead": _slim_lead(lead, pipeline_name=pipeline_name, status_name=status_name),
                    "client_chain": chain,
                }
            )
            seen_leads.add(lead_id)
            seen_phones.add(phone)
        connection_after = get_amo_connection_status(session)
    finally:
        session.close()

    summary = {
        "schema_version": "active_deal_manual_review_pack_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "client_chains": str(chains_path),
        "connection": {
            "before_read": _connection_summary(connection_before),
            "after_read": _connection_summary(connection_after),
            "api_read_succeeded": True,
        },
        "fetch": {
            "leads_seen": len(leads),
            "pages_fetched": pages,
            "contact_fetches": len(contact_cache),
            "sample_size_requested": args.sample_size,
            "rows_selected": len(rows),
        },
        "status_counts": dict(Counter(row["Статус сделки"] for row in rows).most_common()),
        "skipped": dict(skipped.most_common()),
        "outputs": {
            "review_csv": str(out_root / "active_deal_manual_review_50.csv"),
            "review_xlsx": str(out_root / "active_deal_manual_review_50.xlsx"),
            "payload_jsonl": str(out_root / "active_deal_manual_review_payloads.jsonl"),
            "summary_json": str(out_root / "summary.json"),
            "guide": str(out_root / "ROP_REVIEW_GUIDE.md"),
        },
    }
    _write_csv(out_root / "active_deal_manual_review_50.csv", rows)
    _write_xlsx(out_root / "active_deal_manual_review_50.xlsx", rows)
    (out_root / "active_deal_manual_review_payloads.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + ("\n" if payloads else ""),
        encoding="utf-8",
    )
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "ROP_REVIEW_GUIDE.md").write_text(_guide_text(), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if rows else 1


def _fetch_all_leads(session: Any, *, limit: int, page_limit: int, sleep_sec: float) -> tuple[list[dict[str, Any]], int]:
    from mango_mvp.amocrm_runtime.amo_integration import amo_api_request  # noqa: PLC0415

    params = {"limit": page_limit, "with": "contacts"}
    next_url: str | None = "/api/v4/leads"
    rows: list[dict[str, Any]] = []
    pages = 0
    while next_url and len(rows) < limit:
        payload = amo_api_request(session, method="GET", path_or_url=next_url, params=params if pages == 0 else None)
        pages += 1
        embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
        leads = embedded.get("leads") if isinstance(embedded, dict) else []
        if isinstance(leads, list):
            for lead in leads:
                if isinstance(lead, dict):
                    rows.append(lead)
                    if len(rows) >= limit:
                        break
        links = payload.get("_links") if isinstance(payload, dict) else {}
        next_meta = links.get("next") if isinstance(links, dict) else None
        next_url = next_meta.get("href") if isinstance(next_meta, dict) else None
        params = None
        if sleep_sec and next_url:
            time.sleep(sleep_sec)
    return rows, pages


def _sort_active_leads(leads: list[dict[str, Any]], status_by_id: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    def score(lead: dict[str, Any]) -> tuple[int, int]:
        status_id = _safe_int(lead.get("status_id"))
        status_name = _safe_text(status_by_id.get(status_id, {}).get("status_name")).casefold()
        priority = 0
        if status_id in TERMINAL_STATUS_IDS:
            priority -= 1000
        if any(marker in status_name for marker in ("ожидание оплат", "заключение", "запись", "принимают решение")):
            priority += 100
        elif any(marker in status_name for marker in ("в работе", "переговор")):
            priority += 70
        elif "недозвон" in status_name:
            priority += 20
        elif "оплата получена" in status_name:
            priority += 10
        return (priority, _safe_int(lead.get("updated_at")))

    return sorted(leads, key=score, reverse=True)


def _review_row(
    *,
    review_id: str,
    phone: str,
    contact_id: int,
    lead: dict[str, Any],
    pipeline_name: str,
    status_name: str,
    responsible_name: str,
    chain: dict[str, str],
) -> dict[str, Any]:
    lead_id = _safe_int(lead.get("id"))
    latest = _safe_text(chain.get("example_latest_summary"))
    products = _safe_text(chain.get("products_top"))
    subjects = _safe_text(chain.get("subjects_top"))
    objections = _safe_text(chain.get("objections_top"))
    tallanto = _join_nonempty(
        [
            f"Tallanto: {chain.get('tallanto_ids_count')} ID" if _safe_int(chain.get("tallanto_ids_count")) else "",
            _safe_text(chain.get("tallanto_student_types")),
            _safe_text(chain.get("tallanto_branches")),
            _safe_text(chain.get("tallanto_history_terms")),
        ]
    )
    return {
        "review_id": review_id,
        "Телефон": f"+{phone}" if phone and not phone.startswith("+") else phone,
        "Контакт AMO": f"https://educent.amocrm.ru/contacts/detail/{contact_id}",
        "Сделка AMO": f"https://educent.amocrm.ru/leads/detail/{lead_id}",
        "ID контакта AMO": contact_id,
        "ID сделки AMO": lead_id,
        "Название сделки": _safe_text(lead.get("name")),
        "Воронка": pipeline_name,
        "Статус сделки": status_name,
        "Ответственный AMO": responsible_name,
        "Обновлена AMO": _format_ts(lead.get("updated_at")),
        "Содержательных звонков": chain.get("contentful_call_count", ""),
        "Последний звонок": chain.get("last_seen_at", ""),
        "Менеджеры в звонках": chain.get("managers", ""),
        "Продукты из звонков": products,
        "Предметы из звонков": subjects,
        "Возражения из звонков": objections,
        "Последняя AI-сводка из звонков": latest,
        "Tallanto контекст": tallanto,
        "AMO связи из phone-chain": f"contacts={chain.get('amo_contact_ids')} | leads={chain.get('amo_lead_ids')} | statuses={chain.get('amo_statuses')}",
        "Что проверить РОП": (
            "1) Эта ли активная сделка соответствует истории звонков? "
            "2) Не устарел ли следующий шаг? "
            "3) Нужно ли писать AI-сводку в сделку, контакт или обе карточки? "
            "4) Есть ли оплата/занятия/TG/email, которых нет в звонках?"
        ),
        "Решение РОП": "",
        "Комментарий РОП": "",
    }


def _load_chains(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            phone = normalize_phone(row.get("phone"))
            if phone:
                result[phone] = row
    return result


def _contact_phones(contact: dict[str, Any]) -> list[str]:
    phones: list[str] = []
    for field in contact.get("custom_fields_values") or []:
        if not isinstance(field, dict):
            continue
        field_code = _safe_text(field.get("field_code")).upper()
        field_name = _safe_text(field.get("field_name")).casefold()
        if field_code != "PHONE" and "тел" not in field_name and "phone" not in field_name:
            continue
        for item in field.get("values") or []:
            raw = item.get("value") if isinstance(item, dict) else item
            phone = normalize_phone(raw)
            if phone and phone not in phones:
                phones.append(phone)
    return phones


def _first_contact_id(lead: dict[str, Any]) -> int:
    contacts = (lead.get("_embedded") or {}).get("contacts") or []
    if not isinstance(contacts, list):
        return 0
    for item in contacts:
        if isinstance(item, dict):
            contact_id = _safe_int(item.get("id"))
            if contact_id:
                return contact_id
    return 0


def _status_by_id(pipelines: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for pipeline in pipelines:
        pipeline_name = _safe_text(pipeline.get("name"))
        statuses = (pipeline.get("_embedded") or {}).get("statuses") or []
        if not isinstance(statuses, list):
            continue
        for status in statuses:
            if not isinstance(status, dict):
                continue
            status_id = _safe_int(status.get("id"))
            if status_id:
                result[status_id] = {
                    "pipeline_name": pipeline_name,
                    "status_name": _safe_text(status.get("name")),
                }
    return result


def _slim_contact(contact: dict[str, Any]) -> dict[str, Any]:
    return {"id": contact.get("id"), "name": contact.get("name"), "phones": _contact_phones(contact)}


def _slim_lead(lead: dict[str, Any], *, pipeline_name: str, status_name: str) -> dict[str, Any]:
    return {
        "id": lead.get("id"),
        "name": lead.get("name"),
        "pipeline_id": lead.get("pipeline_id"),
        "pipeline_name": pipeline_name,
        "status_id": lead.get("status_id"),
        "status_name": status_name,
        "responsible_user_id": lead.get("responsible_user_id"),
        "updated_at": lead.get("updated_at"),
    }


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


def _write_xlsx(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="manual_review", index=False)
        workbook = writer.book
        wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
        header = workbook.add_format({"bold": True, "bg_color": "#EAF2F8", "border": 1})
        ws = writer.sheets["manual_review"]
        ws.freeze_panes(1, 0)
        for idx, column in enumerate(rows[0].keys() if rows else []):
            ws.write(0, idx, column, header)
            width = 18
            if column in {"Последняя AI-сводка из звонков", "Что проверить РОП", "AMO связи из phone-chain"}:
                width = 56
            elif column in {"Менеджеры в звонках", "Продукты из звонков", "Предметы из звонков", "Возражения из звонков", "Tallanto контекст"}:
                width = 42
            ws.set_column(idx, idx, width, wrap)


def _guide_text() -> str:
    return """# ROP Manual Review Guide

This pack is read-only. It is not a writeback pack.

Check each row:

1. Is the selected active AMO deal the right deal for this phone-chain history?
2. Should AI context be written to contact, deal, or both?
3. Does the call history miss important Telegram/email/Tallanto/payment context?
4. Is the current AMO status consistent with the latest call summary?
5. Is there a duplicate/current-client issue that should be routed to entity resolution instead?

Use `Решение РОП`:

- `ok`;
- `wrong_deal`;
- `write_contact_only`;
- `write_deal_only`;
- `write_both`;
- `needs_entity_resolution`;
- `skip_no_value`;
- `needs_more_context`.
"""


def _connection_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "connected": payload.get("connected"),
        "account_base_url": payload.get("account_base_url"),
        "token_source": payload.get("token_source"),
    }


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


def _join_nonempty(parts: list[str]) -> str:
    return " | ".join(part for part in (_safe_text(item) for item in parts) if part)


def _safe_text(value: Any) -> str:
    return str(value or "").replace("\u2028", " ").strip()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _format_ts(value: Any) -> str:
    ts = _safe_int(value)
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (OSError, ValueError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
