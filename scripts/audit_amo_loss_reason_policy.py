#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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

from mango_mvp.quality.amo_loss_reason_policy import classify_amo_loss_reason  # noqa: E402


LOSS_REASON_FIELD_NAMES = {"Причина отказа (лид)", "Причина отказа (B2C)", "AMO причина отказа", "loss_reason"}
TERMINAL_LOST_STATUS_ID = 143
DEFAULT_OUT_ROOT = ROOT / "stable_runtime" / "amo_loss_reason_policy_audit_20260513_v1"
ENV_FILES = (
    ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    ROOT / "prod_runtime_transfer" / ".env.private",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit of AMO loss reasons against the policy taxonomy.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--limit", type=int, default=0, help="0 means fetch all pages available from AMO.")
    parser.add_argument("--page-limit", type=int, default=250)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no page cap.")
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_env_files()
    from mango_mvp.amocrm_runtime.amo_integration import (  # noqa: PLC0415
        fetch_pipelines_with_statuses,
        get_amo_connection_status,
    )
    from mango_mvp.amocrm_runtime.db import SessionLocal  # noqa: PLC0415

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    session = SessionLocal()
    try:
        connection_before = get_amo_connection_status(session)
        pipelines = fetch_pipelines_with_statuses(session)
        status_by_id = _status_by_id(pipelines)
        lead_fields = _fetch_lead_custom_fields(session)
        reason_field_catalog = [field for field in lead_fields if _is_loss_reason_field(field)]
        leads, pages = _fetch_all_leads(
            session,
            limit=max(0, args.limit),
            page_limit=max(1, min(args.page_limit, 250)),
            max_pages=max(0, args.max_pages),
            sleep_sec=max(0.0, args.sleep_sec),
        )
        connection_after = get_amo_connection_status(session)
    finally:
        session.close()

    rows: list[dict[str, Any]] = []
    value_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    samples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_reasons: set[str] = set()
    terminal_lost_without_reason = 0

    for lead in leads:
        lead_id = _safe_int(lead.get("id"))
        status_id = _safe_int(lead.get("status_id"))
        pipeline_id = _safe_int(lead.get("pipeline_id"))
        status_meta = status_by_id.get(status_id, {})
        status_name = str(status_meta.get("status_name") or lead.get("status_name") or "").strip()
        pipeline_name = str(status_meta.get("pipeline_name") or "").strip()
        status_key = f"{status_id}:{pipeline_name}/{status_name}".strip("/")
        status_counter[status_key] += 1
        reasons = _extract_loss_reasons(lead)
        if status_id == TERMINAL_LOST_STATUS_ID and not reasons:
            terminal_lost_without_reason += 1
        for reason in reasons:
            policies = classify_amo_loss_reason(reason)
            if not policies:
                unknown_reasons.add(reason)
            categories = " | ".join(policy.category for policy in policies) or "unknown"
            risks = " | ".join(policy.risk_type for policy in policies) or "unknown_loss_reason_policy"
            value_counter[reason] += 1
            for policy in policies:
                category_counter[policy.category] += 1
                risk_counter[policy.risk_type] += 1
            row = {
                "lead_id": lead_id,
                "lead_link": f"https://educent.amocrm.ru/leads/detail/{lead_id}" if lead_id else "",
                "lead_name": _safe_text(lead.get("name")),
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline_name,
                "status_id": status_id,
                "status_name": status_name,
                "closed_at": _format_ts(lead.get("closed_at")),
                "updated_at": _format_ts(lead.get("updated_at")),
                "loss_reason": reason,
                "policy_categories": categories,
                "policy_risk_types": risks,
                "policy_status": "known" if policies else "unknown",
            }
            rows.append(row)
            if len(samples_by_reason[reason]) < 20:
                samples_by_reason[reason].append(row)

    reason_rows = []
    for reason, count in value_counter.most_common():
        policies = classify_amo_loss_reason(reason)
        reason_rows.append(
            {
                "loss_reason": reason,
                "count": count,
                "policy_status": "known" if policies else "unknown",
                "policy_categories": " | ".join(policy.category for policy in policies) or "unknown",
                "policy_risk_types": " | ".join(policy.risk_type for policy in policies) or "unknown_loss_reason_policy",
                "sample_lead_links": " | ".join(row["lead_link"] for row in samples_by_reason[reason][:5]),
            }
        )

    catalog_reason_values = _catalog_reason_values(reason_field_catalog)
    catalog_unknown_values = sorted(value for value in catalog_reason_values if not classify_amo_loss_reason(value))
    summary = {
        "schema_version": "amo_loss_reason_policy_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "connection": {
            "before_read": {
                "status": connection_before.get("status"),
                "connected": connection_before.get("connected"),
                "account_base_url": connection_before.get("account_base_url"),
                "token_source": connection_before.get("token_source"),
            },
            "after_read": {
                "status": connection_after.get("status"),
                "connected": connection_after.get("connected"),
                "account_base_url": connection_after.get("account_base_url"),
                "token_source": connection_after.get("token_source"),
            },
            "api_read_succeeded": True,
        },
        "fetch": {
            "leads_seen": len(leads),
            "pages_fetched": pages,
            "limit": max(0, args.limit),
            "page_limit": max(1, min(args.page_limit, 250)),
            "max_pages": max(0, args.max_pages),
        },
        "lead_status_counts_top": dict(status_counter.most_common(30)),
        "loss_reason": {
            "lead_rows_with_reason": len(rows),
            "unique_values": len(value_counter),
            "known_unique_values": len(value_counter) - len(unknown_reasons),
            "unknown_unique_values": len(unknown_reasons),
            "unknown_values": sorted(unknown_reasons),
            "terminal_lost_without_reason": terminal_lost_without_reason,
            "value_counts": dict(value_counter.most_common()),
            "category_counts": dict(category_counter.most_common()),
            "risk_counts": dict(risk_counter.most_common()),
        },
        "lead_reason_fields": _field_catalog_summary(reason_field_catalog),
        "lead_reason_field_catalog_policy": {
            "unique_enum_values": len(catalog_reason_values),
            "known_enum_values": len(catalog_reason_values) - len(catalog_unknown_values),
            "unknown_enum_values": len(catalog_unknown_values),
            "unknown_values": catalog_unknown_values,
        },
        "outputs": {
            "summary_json": str(out_root / "summary.json"),
            "loss_reason_values_csv": str(out_root / "amo_loss_reason_values.csv"),
            "lead_samples_csv": str(out_root / "amo_loss_reason_lead_samples.csv"),
            "lead_field_catalog_json": str(out_root / "amo_loss_reason_field_catalog.json"),
            "readme": str(out_root / "README.md"),
        },
    }

    _write_csv(out_root / "amo_loss_reason_values.csv", reason_rows)
    _write_csv(out_root / "amo_loss_reason_lead_samples.csv", rows)
    (out_root / "amo_loss_reason_field_catalog.json").write_text(
        json.dumps(reason_field_catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not unknown_reasons else 1


def _fetch_all_leads(
    session: Any,
    *,
    limit: int,
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
            for lead in leads:
                if isinstance(lead, dict):
                    items.append(lead)
                    if limit and len(items) >= limit:
                        return items, pages
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


def _fetch_lead_custom_fields(session: Any) -> list[dict[str, Any]]:
    from mango_mvp.amocrm_runtime.amo_integration import amo_api_request  # noqa: PLC0415

    fields: list[dict[str, Any]] = []
    next_url: str | None = "/api/v4/leads/custom_fields?limit=50"
    while next_url:
        payload = amo_api_request(session, method="GET", path_or_url=next_url)
        embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
        items = embedded.get("custom_fields") if isinstance(embedded, dict) else []
        if isinstance(items, list):
            fields.extend(item for item in items if isinstance(item, dict))
        next_link = None
        links = payload.get("_links") if isinstance(payload, dict) else {}
        if isinstance(links, dict):
            next_meta = links.get("next")
            if isinstance(next_meta, dict):
                next_link = next_meta.get("href")
        next_url = next_link
    return fields


def _extract_loss_reasons(lead: dict[str, Any]) -> list[str]:
    values: list[str] = []
    embedded = lead.get("_embedded") if isinstance(lead.get("_embedded"), dict) else {}
    embedded_reason = embedded.get("loss_reason")
    if isinstance(embedded_reason, dict):
        _append_clean(values, embedded_reason.get("name"))
    elif isinstance(embedded_reason, list):
        for item in embedded_reason:
            if isinstance(item, dict):
                _append_clean(values, item.get("name"))
    _append_clean(values, lead.get("loss_reason"))
    for field in lead.get("custom_fields_values") or []:
        if not isinstance(field, dict):
            continue
        if _safe_text(field.get("field_name")) not in LOSS_REASON_FIELD_NAMES:
            continue
        for item in field.get("values") or []:
            if isinstance(item, dict):
                _append_clean(values, item.get("value"))
            else:
                _append_clean(values, item)
    return list(dict.fromkeys(values))


def _is_loss_reason_field(field: dict[str, Any]) -> bool:
    name = _safe_text(field.get("name"))
    code = _safe_text(field.get("code"))
    return name in LOSS_REASON_FIELD_NAMES or code in LOSS_REASON_FIELD_NAMES or "причина отказ" in name.casefold()


def _field_catalog_summary(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in fields:
        enums = field.get("enums") if isinstance(field.get("enums"), list) else []
        enum_values = []
        for enum in enums:
            if isinstance(enum, dict):
                value = _safe_text(enum.get("value"))
                if value:
                    enum_values.append(value)
        result.append(
            {
                "id": field.get("id"),
                "name": field.get("name"),
                "code": field.get("code"),
                "type": field.get("type"),
                "enum_count": len(enum_values),
                "enum_values": " | ".join(enum_values),
            }
        )
    return result


def _catalog_reason_values(fields: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for field in fields:
        enums = field.get("enums") if isinstance(field.get("enums"), list) else []
        for enum in enums:
            if isinstance(enum, dict):
                _append_clean(values, enum.get("value"))
    return list(dict.fromkeys(values))


def _status_by_id(pipelines: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for pipeline in pipelines:
        pipeline_id = _safe_int(pipeline.get("id"))
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
                    "pipeline_id": str(pipeline_id),
                    "pipeline_name": pipeline_name,
                    "status_name": _safe_text(status.get("name")),
                }
    return result


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


def _readme(summary: dict[str, Any]) -> str:
    loss = summary["loss_reason"]
    return f"""# AMO Loss Reason Policy Audit

Read-only audit. No AMO writes were performed.

- Leads fetched: {summary['fetch']['leads_seen']}
- Pages fetched: {summary['fetch']['pages_fetched']}
- Lead rows with loss reason: {loss['lead_rows_with_reason']}
- Unique loss reason values: {loss['unique_values']}
- Unknown unique values: {loss['unknown_unique_values']}
- Unknown AMO enum values: {summary['lead_reason_field_catalog_policy']['unknown_enum_values']}
- Terminal lost leads without extracted reason: {loss['terminal_lost_without_reason']}

Outputs:

- `summary.json`
- `amo_loss_reason_values.csv`
- `amo_loss_reason_lead_samples.csv`
- `amo_loss_reason_field_catalog.json`
"""


def _append_clean(values: list[str], value: Any) -> None:
    text = _safe_text(value)
    if text:
        values.append(text)


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
