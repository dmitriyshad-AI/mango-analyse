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
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "deal_aware_amo_live_snapshot_20260513_v1"
ENV_FILES = (
    ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    ROOT / "prod_runtime_transfer" / ".env.private",
)
AI_FIELD_PREFIXES = ("AI", "Авто", "Статус матчинга", "Последняя AI")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only AMO snapshot for deal-aware Stage 1.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--contacts-limit", type=int, default=0, help="0 means all available pages.")
    parser.add_argument("--leads-limit", type=int, default=0, help="0 means all available pages.")
    parser.add_argument("--tasks-limit", type=int, default=10000, help="0 means all available pages.")
    parser.add_argument("--page-limit", type=int, default=250)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no page cap.")
    parser.add_argument("--sleep-sec", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _load_env_files()
    from mango_mvp.amocrm_runtime.amo_integration import (  # noqa: PLC0415
        amo_api_request,
        fetch_contact_field_catalog,
        fetch_lead_field_catalog,
        fetch_pipelines_with_statuses,
        fetch_users,
        get_amo_connection_status,
    )
    from mango_mvp.amocrm_runtime.db import SessionLocal  # noqa: PLC0415

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        session = SessionLocal()
        try:
            connection_before = get_amo_connection_status(session)
            pipelines = fetch_pipelines_with_statuses(session)
            users = fetch_users(session)
            contact_fields = fetch_contact_field_catalog(session)
            lead_fields = fetch_lead_field_catalog(session)
            contacts, contact_pages = fetch_collection(
                amo_api_request,
                session,
                path="/api/v4/contacts",
                embedded_key="contacts",
                params={"with": "leads"},
                limit=max(0, args.contacts_limit),
                page_limit=max(1, min(args.page_limit, 250)),
                max_pages=max(0, args.max_pages),
                sleep_sec=max(0.0, args.sleep_sec),
            )
            leads, lead_pages = fetch_collection(
                amo_api_request,
                session,
                path="/api/v4/leads",
                embedded_key="leads",
                params={"with": "contacts"},
                limit=max(0, args.leads_limit),
                page_limit=max(1, min(args.page_limit, 250)),
                max_pages=max(0, args.max_pages),
                sleep_sec=max(0.0, args.sleep_sec),
            )
            tasks, task_pages = fetch_collection(
                amo_api_request,
                session,
                path="/api/v4/tasks",
                embedded_key="tasks",
                params={"filter[entity_type]": "leads"},
                limit=max(0, args.tasks_limit),
                page_limit=max(1, min(args.page_limit, 250)),
                max_pages=max(0, args.max_pages),
                sleep_sec=max(0.0, args.sleep_sec),
            )
            connection_after = get_amo_connection_status(session)
        finally:
            session.close()
    except Exception as exc:
        summary = failed_summary(out_root, args=args, exc=exc)
        for filename in (
            "amo_contacts_snapshot.csv",
            "amo_deals_snapshot.csv",
            "amo_tasks_snapshot.csv",
            "amo_status_catalog.csv",
            "amo_user_catalog.csv",
            "amo_field_catalog.csv",
        ):
            write_csv(out_root / filename, [])
        (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_root / "README.md").write_text(render_readme(summary), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    status_by_id = status_catalog_rows(pipelines)
    status_meta = {int(row["status_id"]): row for row in status_by_id if str(row.get("status_id", "")).isdigit()}
    user_meta = {str(user.get("id")): safe_text(user.get("name")) for user in users if safe_text(user.get("id"))}
    contact_rows = [contact_snapshot_row(contact, user_meta=user_meta) for contact in contacts]
    lead_rows = [lead_snapshot_row(lead, status_meta=status_meta, user_meta=user_meta) for lead in leads]
    task_rows = [task_snapshot_row(task, user_meta=user_meta) for task in tasks]
    field_rows = field_catalog_rows(contact_fields, entity_type="contact") + field_catalog_rows(lead_fields, entity_type="lead")

    write_csv(out_root / "amo_contacts_snapshot.csv", contact_rows)
    write_csv(out_root / "amo_deals_snapshot.csv", lead_rows)
    write_csv(out_root / "amo_tasks_snapshot.csv", task_rows)
    write_csv(out_root / "amo_status_catalog.csv", status_by_id)
    write_csv(out_root / "amo_user_catalog.csv", user_catalog_rows(users))
    write_csv(out_root / "amo_field_catalog.csv", field_rows)
    (out_root / "raw_counts.json").write_text(
        json.dumps(
            {
                "contacts": len(contacts),
                "leads": len(leads),
                "tasks": len(tasks),
                "pipelines": len(pipelines),
                "users": len(users),
                "contact_fields": len(contact_fields),
                "lead_fields": len(lead_fields),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "deal_aware_amo_live_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "live_read_only",
        "safety": {
            "read_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
        "connection": {
            "before": compact_connection(connection_before),
            "after": compact_connection(connection_after),
        },
        "fetch": {
            "contacts_seen": len(contacts),
            "contact_pages": contact_pages,
            "leads_seen": len(leads),
            "lead_pages": lead_pages,
            "tasks_seen": len(tasks),
            "task_pages": task_pages,
            "contacts_limit": max(0, args.contacts_limit),
            "leads_limit": max(0, args.leads_limit),
            "tasks_limit": max(0, args.tasks_limit),
            "page_limit": max(1, min(args.page_limit, 250)),
            "max_pages": max(0, args.max_pages),
        },
        "coverage": {
            "contacts_with_phone": sum(1 for row in contact_rows if row.get("phones")),
            "leads_with_linked_contacts": sum(1 for row in lead_rows if row.get("linked_contact_ids")),
            "leads_terminal_lost": sum(1 for row in lead_rows if row.get("status_name") == "Закрыто и не реализовано"),
            "leads_with_ai_fields": sum(1 for row in lead_rows if row.get("ai_field_values")),
            "contacts_with_ai_fields": sum(1 for row in contact_rows if row.get("ai_field_values")),
            "tasks_by_status": dict(Counter(row.get("is_completed", "") for row in task_rows).most_common()),
        },
        "outputs": {
            "contacts_csv": str(out_root / "amo_contacts_snapshot.csv"),
            "deals_csv": str(out_root / "amo_deals_snapshot.csv"),
            "tasks_csv": str(out_root / "amo_tasks_snapshot.csv"),
            "status_catalog_csv": str(out_root / "amo_status_catalog.csv"),
            "user_catalog_csv": str(out_root / "amo_user_catalog.csv"),
            "field_catalog_csv": str(out_root / "amo_field_catalog.csv"),
            "summary_json": str(out_root / "summary.json"),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def fetch_collection(
    request_func: Any,
    session: Any,
    *,
    path: str,
    embedded_key: str,
    params: dict[str, Any],
    limit: int,
    page_limit: int,
    max_pages: int,
    sleep_sec: float,
) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    pages = 0
    next_url: str | None = path
    request_params: dict[str, Any] | None = dict(params) | {"limit": page_limit}
    while next_url:
        payload = request_func(session, method="GET", path_or_url=next_url, params=request_params)
        pages += 1
        embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
        page_items = embedded.get(embedded_key) if isinstance(embedded, dict) else []
        if isinstance(page_items, list):
            for item in page_items:
                if isinstance(item, dict):
                    items.append(item)
                    if limit and len(items) >= limit:
                        return items, pages
        if max_pages and pages >= max_pages:
            break
        next_url = next_href(payload)
        request_params = None
        if next_url and sleep_sec:
            time.sleep(sleep_sec)
    return items, pages


def next_href(payload: dict[str, Any]) -> str | None:
    links = payload.get("_links") if isinstance(payload, dict) else {}
    next_meta = links.get("next") if isinstance(links, dict) else None
    if isinstance(next_meta, dict):
        value = next_meta.get("href")
        return str(value) if value else None
    return None


def contact_snapshot_row(contact: dict[str, Any], *, user_meta: dict[str, str]) -> dict[str, Any]:
    custom = custom_field_values(contact)
    linked_leads = (contact.get("_embedded") or {}).get("leads") or []
    return {
        "contact_id": safe_text(contact.get("id")),
        "contact_name": safe_text(contact.get("name")),
        "responsible_user_id": safe_text(contact.get("responsible_user_id")),
        "responsible_user_name": user_meta.get(safe_text(contact.get("responsible_user_id")), ""),
        "created_at": format_ts(contact.get("created_at")),
        "updated_at": format_ts(contact.get("updated_at")),
        "phones": " | ".join(extract_phones(contact)),
        "emails": " | ".join(extract_emails(contact)),
        "linked_lead_ids": " | ".join(safe_text(item.get("id")) for item in linked_leads if isinstance(item, dict)),
        "ai_field_values": json.dumps({k: v for k, v in custom.items() if is_ai_field(k)}, ensure_ascii=False, sort_keys=True),
        "custom_field_values_json": json.dumps(custom, ensure_ascii=False, sort_keys=True),
    }


def lead_snapshot_row(
    lead: dict[str, Any],
    *,
    status_meta: dict[int, dict[str, Any]],
    user_meta: dict[str, str],
) -> dict[str, Any]:
    custom = custom_field_values(lead)
    status_id = safe_int(lead.get("status_id"))
    status = status_meta.get(status_id, {})
    linked_contacts = (lead.get("_embedded") or {}).get("contacts") or []
    return {
        "lead_id": safe_text(lead.get("id")),
        "lead_name": safe_text(lead.get("name")),
        "pipeline_id": safe_text(lead.get("pipeline_id")),
        "pipeline_name": safe_text(status.get("pipeline_name")),
        "status_id": safe_text(lead.get("status_id")),
        "status_name": safe_text(status.get("status_name")),
        "responsible_user_id": safe_text(lead.get("responsible_user_id")),
        "responsible_user_name": user_meta.get(safe_text(lead.get("responsible_user_id")), ""),
        "price": safe_text(lead.get("price")),
        "created_at": format_ts(lead.get("created_at")),
        "updated_at": format_ts(lead.get("updated_at")),
        "closed_at": format_ts(lead.get("closed_at")),
        "linked_contact_ids": " | ".join(safe_text(item.get("id")) for item in linked_contacts if isinstance(item, dict)),
        "loss_reason": extract_loss_reason(lead, custom),
        "ai_field_values": json.dumps({k: v for k, v in custom.items() if is_ai_field(k)}, ensure_ascii=False, sort_keys=True),
        "custom_field_values_json": json.dumps(custom, ensure_ascii=False, sort_keys=True),
    }


def task_snapshot_row(task: dict[str, Any], *, user_meta: dict[str, str]) -> dict[str, Any]:
    return {
        "task_id": safe_text(task.get("id")),
        "entity_id": safe_text(task.get("entity_id")),
        "entity_type": safe_text(task.get("entity_type")),
        "text": safe_text(task.get("text")),
        "task_type_id": safe_text(task.get("task_type_id")),
        "responsible_user_id": safe_text(task.get("responsible_user_id")),
        "responsible_user_name": user_meta.get(safe_text(task.get("responsible_user_id")), ""),
        "complete_till": format_ts(task.get("complete_till")),
        "created_at": format_ts(task.get("created_at")),
        "updated_at": format_ts(task.get("updated_at")),
        "is_completed": safe_text(task.get("is_completed")),
        "result": safe_text(task.get("result", {}).get("text") if isinstance(task.get("result"), dict) else ""),
    }


def status_catalog_rows(pipelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pipeline in pipelines:
        statuses = (pipeline.get("_embedded") or {}).get("statuses") or []
        for status in statuses if isinstance(statuses, list) else []:
            if not isinstance(status, dict):
                continue
            rows.append(
                {
                    "pipeline_id": safe_text(pipeline.get("id")),
                    "pipeline_name": safe_text(pipeline.get("name")),
                    "status_id": safe_text(status.get("id")),
                    "status_name": safe_text(status.get("name")),
                    "sort": safe_text(status.get("sort")),
                    "is_editable": safe_text(status.get("is_editable")),
                    "type": safe_text(status.get("type")),
                }
            )
    return rows


def user_catalog_rows(users: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "user_id": safe_text(user.get("id")),
            "name": safe_text(user.get("name")),
            "email": safe_text(user.get("email")),
            "is_active": safe_text(user.get("is_active")),
        }
        for user in users
    ]


def field_catalog_rows(fields: list[dict[str, Any]], *, entity_type: str) -> list[dict[str, Any]]:
    rows = []
    for field in fields:
        enums = field.get("enums") if isinstance(field.get("enums"), list) else []
        enum_values = []
        for enum in enums:
            if isinstance(enum, dict):
                enum_values.append(safe_text(enum.get("value")))
        rows.append(
            {
                "entity_type": entity_type,
                "field_id": safe_text(field.get("id")),
                "name": safe_text(field.get("name")),
                "code": safe_text(field.get("code")),
                "type": safe_text(field.get("type")),
                "is_ai_field": safe_text(is_ai_field(safe_text(field.get("name")))),
                "enum_count": len([item for item in enum_values if item]),
                "enum_values": " | ".join(item for item in enum_values if item),
            }
        )
    return rows


def custom_field_values(entity: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in entity.get("custom_fields_values") or []:
        if not isinstance(field, dict):
            continue
        name = safe_text(field.get("field_name") or field.get("name") or field.get("field_code"))
        if not name:
            continue
        values = []
        for value in field.get("values") or []:
            if isinstance(value, dict):
                values.append(safe_text(value.get("value")))
            else:
                values.append(safe_text(value))
        result[name] = " | ".join(item for item in values if item)
    return result


def extract_phones(entity: dict[str, Any]) -> list[str]:
    phones = []
    for field in entity.get("custom_fields_values") or []:
        if not isinstance(field, dict):
            continue
        if safe_text(field.get("field_code")).upper() != "PHONE" and safe_text(field.get("field_name")).casefold() != "телефон":
            continue
        for value in field.get("values") or []:
            raw = value.get("value") if isinstance(value, dict) else value
            phone = normalize_phone_safe(raw)
            if phone:
                phones.append(phone)
    return list(dict.fromkeys(phones))


def extract_emails(entity: dict[str, Any]) -> list[str]:
    emails = []
    for field in entity.get("custom_fields_values") or []:
        if not isinstance(field, dict):
            continue
        if safe_text(field.get("field_code")).upper() != "EMAIL":
            continue
        for value in field.get("values") or []:
            raw = safe_text(value.get("value") if isinstance(value, dict) else value)
            if raw:
                emails.append(raw)
    return list(dict.fromkeys(emails))


def normalize_phone_safe(value: Any) -> str:
    from mango_mvp.utils.phone import normalize_phone  # noqa: PLC0415

    return normalize_phone(value)


def extract_loss_reason(lead: dict[str, Any], custom: dict[str, str]) -> str:
    embedded = lead.get("_embedded") if isinstance(lead.get("_embedded"), dict) else {}
    embedded_reason = embedded.get("loss_reason")
    if isinstance(embedded_reason, dict) and safe_text(embedded_reason.get("name")):
        return safe_text(embedded_reason.get("name"))
    for key, value in custom.items():
        if "причина отказ" in key.casefold() and value:
            return value
    return safe_text(lead.get("loss_reason"))


def is_ai_field(name: str) -> bool:
    clean = safe_text(name)
    return any(clean.startswith(prefix) for prefix in AI_FIELD_PREFIXES)


def compact_connection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "connected": payload.get("connected"),
        "account_base_url": payload.get("account_base_url"),
        "token_source": payload.get("token_source"),
        "last_error": payload.get("last_error"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def render_readme(summary: dict[str, Any]) -> str:
    fetch = summary["fetch"]
    return f"""# Deal-Aware AMO Live Snapshot

Read-only AMO snapshot for deal-aware Stage 1. No AMO writes were performed.

- Contacts fetched: {fetch['contacts_seen']}
- Leads fetched: {fetch['leads_seen']}
- Tasks fetched: {fetch['tasks_seen']}
- Contact pages: {fetch['contact_pages']}
- Lead pages: {fetch['lead_pages']}
- Task pages: {fetch['task_pages']}

Outputs are CSV files plus `summary.json`.
"""


def failed_summary(out_root: Path, *, args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": "deal_aware_amo_live_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "live_read_only",
        "safety": {
            "read_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
        "connection": {"before": {}, "after": {}},
        "fetch": {
            "contacts_seen": 0,
            "contact_pages": 0,
            "leads_seen": 0,
            "lead_pages": 0,
            "tasks_seen": 0,
            "task_pages": 0,
            "contacts_limit": max(0, args.contacts_limit),
            "leads_limit": max(0, args.leads_limit),
            "tasks_limit": max(0, args.tasks_limit),
            "page_limit": max(1, min(args.page_limit, 250)),
            "max_pages": max(0, args.max_pages),
        },
        "coverage": {
            "contacts_with_phone": 0,
            "leads_with_linked_contacts": 0,
            "leads_terminal_lost": 0,
            "leads_with_ai_fields": 0,
            "contacts_with_ai_fields": 0,
            "tasks_by_status": {},
        },
        "api_read_succeeded": False,
        "preflight_error": f"{type(exc).__name__}: {exc}",
        "outputs": {
            "contacts_csv": str(out_root / "amo_contacts_snapshot.csv"),
            "deals_csv": str(out_root / "amo_deals_snapshot.csv"),
            "tasks_csv": str(out_root / "amo_tasks_snapshot.csv"),
            "status_catalog_csv": str(out_root / "amo_status_catalog.csv"),
            "user_catalog_csv": str(out_root / "amo_user_catalog.csv"),
            "field_catalog_csv": str(out_root / "amo_field_catalog.csv"),
            "summary_json": str(out_root / "summary.json"),
        },
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


def safe_text(value: Any) -> str:
    return str(value or "").replace("\u2028", " ").strip()


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def format_ts(value: Any) -> str:
    ts = safe_int(value)
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
