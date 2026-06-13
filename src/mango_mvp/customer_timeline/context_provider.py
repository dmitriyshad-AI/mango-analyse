from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.ids import normalize_identity_value
from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig
from mango_mvp.customer_timeline.safety import (
    assert_customer_timeline_safety_contract,
    customer_timeline_safety_contract,
    guard_customer_timeline_output_path,
)
from mango_mvp.utils.phone import normalize_phone as canonical_normalize_phone


CUSTOMER_TIMELINE_CONTEXT_PROVIDER_SCHEMA_VERSION = "customer_timeline_context_provider_v1"

TIMELINE_PROMOTION_STAGES = (
    "timeline_available",
    "timeline_coverage_verified",
    "timeline_preview_enabled",
    "timeline_primary_read_enabled",
    "timeline_live_write_context_allowed",
)

PHONE_KEYS = (
    "phone",
    "phones",
    "Телефон",
    "Телефон клиента",
    "primary_phone",
    "client_phone",
)
CALL_EVENT_TYPES = {"mango_call", "call", "phone_call"}
AMO_SOURCE_SYSTEMS = {"amocrm", "amocrm_snapshot", "amo", "amo_snapshot"}
TALLANTO_SOURCE_SYSTEMS = {"tallanto", "tallanto_snapshot"}
TELEGRAM_EMAIL_SOURCE_SYSTEMS = {"telegram", "telegram_export", "email", "mail", "mail_archive"}


@dataclass(frozen=True)
class CustomerTimelineCoveragePaths:
    deal_aware_candidates_csv: Path
    timeline_db: Path
    out_root: Path
    tenant_id: str = "foton"


def get_customer_context_for_phone(
    phone: str,
    *,
    timeline_db: Path | str | None = None,
    fallback_rows: Sequence[Mapping[str, Any]] | None = None,
    tenant_id: str = "foton",
    limit: int = 25,
) -> dict[str, Any]:
    """Return read-only customer context by phone with fallback.

    The function never writes to AMO, Tallanto, customer channels, runtime DBs,
    or stable_runtime. If timeline is unavailable or incomplete, callers get a
    warning plus a deterministic fallback context.
    """

    assert_customer_timeline_safety_contract(customer_timeline_safety_contract())
    normalized_phone = normalize_phone_for_match(phone)
    warnings: list[str] = []
    timeline_context: dict[str, Any] | None = None
    if timeline_db is not None:
        try:
            timeline_context = _read_context_from_timeline(
                phone=phone,
                timeline_db=Path(timeline_db),
                tenant_id=tenant_id,
                limit=limit,
            )
            warnings.extend(timeline_context.get("warnings", []))
        except Exception as exc:  # noqa: BLE001 - provider must fail closed to fallback.
            warnings.append(f"timeline_unavailable: {exc}")

    if timeline_context and timeline_context.get("found"):
        return {
            **timeline_context,
            "schema_version": CUSTOMER_TIMELINE_CONTEXT_PROVIDER_SCHEMA_VERSION,
            "phone": normalized_phone,
            "source": "customer_timeline",
            "fallback_used": False,
            "warnings": warnings,
            "safety": context_provider_safety_contract(),
        }

    fallback = build_fallback_context_for_phone(phone, fallback_rows or ())
    fallback_warnings = list(warnings)
    if timeline_db is None:
        fallback_warnings.append("timeline_db_not_provided")
    elif not timeline_context or not timeline_context.get("found"):
        fallback_warnings.append("timeline_customer_not_found")
    return {
        "schema_version": CUSTOMER_TIMELINE_CONTEXT_PROVIDER_SCHEMA_VERSION,
        "phone": normalized_phone,
        "source": "fallback_rows" if fallback["items"] else "empty_fallback",
        "found": bool(fallback["items"]),
        "fallback_used": True,
        "customer_id": "",
        "customer": {},
        "summary": fallback["summary"],
        "timeline": {"items": fallback["items"], "source_systems": fallback["source_systems"], "event_types": fallback["event_types"]},
        "bot_context": {"items": [], "summary": {"visible_chunks": 0, "allowed_chunks": 0, "review_required_chunks": 0}},
        "readiness": {
            "safe_for_automatic_bot": False,
            "requires_manager_review": True,
            "events": len(fallback["items"]),
            "bot_allowed_chunks": 0,
            "open_conflicts": 0,
        },
        "warnings": fallback_warnings,
        "safety": context_provider_safety_contract(),
    }


def build_fallback_context_for_phone(phone: str, fallback_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized_phone = normalize_phone_for_match(phone)
    items: list[dict[str, Any]] = []
    source_systems: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    for row in fallback_rows:
        row_phones = extract_phones_from_row(row)
        if normalized_phone and normalized_phone not in row_phones:
            continue
        source_system = safe_text(row.get("source_system") or row.get("source") or "deal_aware_fallback")
        event_type = safe_text(row.get("event_type") or row.get("type") or "deal_aware_row")
        summary = safe_text(
            row.get("call_summary")
            or row.get("latest_call_summary")
            or row.get("AI-сводка по сделке")
            or row.get("summary")
            or row.get("text")
        )
        next_step = safe_text(row.get("next_step") or row.get("latest_call_next_step") or row.get("AI-рекомендованный следующий шаг"))
        if not summary and not next_step:
            continue
        items.append(
            {
                "event_at": safe_text(row.get("started_at") or row.get("last_call_at") or row.get("created_at")),
                "source_system": source_system,
                "event_type": event_type,
                "summary": summary,
                "next_step": next_step,
            }
        )
        source_systems[source_system] += 1
        event_types[event_type] += 1
    items.sort(key=lambda item: safe_text(item.get("event_at")), reverse=True)
    return {
        "items": items[:25],
        "source_systems": dict(source_systems),
        "event_types": dict(event_types),
        "summary": fallback_summary(items),
    }


def audit_customer_timeline_coverage(paths: CustomerTimelineCoveragePaths) -> dict[str, Any]:
    out_root = guard_customer_timeline_output_path(paths.out_root, paths.out_root.parent)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(paths.deal_aware_candidates_csv)
    phones = sorted({phone for row in rows for phone in extract_phones_from_row(row) if phone})
    report_rows: list[dict[str, Any]] = []
    source_totals: Counter[str] = Counter()
    event_totals: Counter[str] = Counter()

    for phone in phones:
        context = get_customer_context_for_phone(phone, timeline_db=paths.timeline_db, tenant_id=paths.tenant_id)
        timeline_items = context.get("timeline", {}).get("items", [])
        source_counts = Counter(safe_text(item.get("source_system")) for item in timeline_items if safe_text(item.get("source_system")))
        event_counts = Counter(safe_text(item.get("event_type")) for item in timeline_items if safe_text(item.get("event_type")))
        source_totals.update(source_counts)
        event_totals.update(event_counts)
        report_rows.append(
            {
                "phone": phone,
                "timeline_found": "Да" if context.get("source") == "customer_timeline" and context.get("found") else "Нет",
                "event_count": str(len(timeline_items)),
                "has_calls": "Да" if any(value in CALL_EVENT_TYPES for value in event_counts) else "Нет",
                "has_amo": "Да" if any(value in AMO_SOURCE_SYSTEMS for value in source_counts) else "Нет",
                "has_tallanto": "Да" if any(value in TALLANTO_SOURCE_SYSTEMS for value in source_counts) else "Нет",
                "has_telegram_or_email": "Да" if any(value in TELEGRAM_EMAIL_SOURCE_SYSTEMS for value in source_counts) else "Нет",
                "warnings": " | ".join(context.get("warnings", [])),
            }
        )

    matched = sum(1 for row in report_rows if row["timeline_found"] == "Да")
    total = len(phones)
    missing = total - matched
    summary = {
        "schema_version": CUSTOMER_TIMELINE_CONTEXT_PROVIDER_SCHEMA_VERSION,
        "report_kind": "customer_timeline_coverage",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tenant_id": paths.tenant_id,
        "deal_aware_candidate_rows": len(rows),
        "deal_aware_unique_phones": total,
        "timeline_matched_phones": matched,
        "timeline_missing_phones": missing,
        "coverage_ratio": round(matched / total, 6) if total else 1.0,
        "source_system_counts": dict(source_totals),
        "event_type_counts": dict(event_totals),
        "timeline_available": Path(paths.timeline_db).exists(),
        "safety": context_provider_safety_contract(),
    }
    write_csv_rows(out_root / "timeline_coverage_report.csv", report_rows)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "summary": summary,
        "rows": report_rows,
        "outputs": {
            "coverage_csv": str(out_root / "timeline_coverage_report.csv"),
            "summary_json": str(out_root / "summary.json"),
        },
    }


def evaluate_timeline_promotion(
    coverage_summary: Mapping[str, Any],
    *,
    preview_enabled: bool = False,
    primary_read_enabled: bool = False,
    live_write_context_requested: bool = False,
    fallback_required: bool = True,
    min_coverage_ratio: float = 0.95,
) -> dict[str, Any]:
    total = int_or_zero(coverage_summary.get("deal_aware_unique_phones"))
    missing = int_or_zero(coverage_summary.get("timeline_missing_phones"))
    ratio = float_or_zero(coverage_summary.get("coverage_ratio"))
    timeline_available = bool(coverage_summary.get("timeline_available"))
    coverage_verified = timeline_available and (total == 0 or (missing == 0 and ratio >= min_coverage_ratio))
    stages = {
        "timeline_available": timeline_available,
        "timeline_coverage_verified": coverage_verified,
        "timeline_preview_enabled": coverage_verified and preview_enabled,
        "timeline_primary_read_enabled": coverage_verified and preview_enabled and primary_read_enabled,
        "timeline_live_write_context_allowed": (
            coverage_verified
            and preview_enabled
            and primary_read_enabled
            and live_write_context_requested
            and fallback_required
        ),
    }
    blocked = [stage for stage, allowed in stages.items() if not allowed]
    return {
        "schema_version": CUSTOMER_TIMELINE_CONTEXT_PROVIDER_SCHEMA_VERSION,
        "stages": stages,
        "blocked_stages": blocked,
        "coverage_verified": coverage_verified,
        "live_write_never_requires_timeline": True,
        "fallback_required_for_live_write_context": fallback_required,
        "min_coverage_ratio": min_coverage_ratio,
    }


def assert_timeline_stage_allowed(stage: str, promotion: Mapping[str, Any]) -> None:
    if stage not in TIMELINE_PROMOTION_STAGES:
        raise ValueError(f"unknown timeline promotion stage: {stage}")
    stages = promotion.get("stages", {})
    if not bool(stages.get(stage)):
        raise ValueError(f"timeline promotion stage is not allowed yet: {stage}")


def context_provider_safety_contract() -> dict[str, Any]:
    contract = dict(customer_timeline_safety_contract())
    contract.update(
        {
            "schema_version": CUSTOMER_TIMELINE_CONTEXT_PROVIDER_SCHEMA_VERSION,
            "read_customer_timeline_db": True,
            "write_customer_timeline_db": False,
            "network_calls": False,
            "subprocess_calls": False,
            "live_writeback_required": False,
        }
    )
    return contract


def _read_context_from_timeline(*, phone: str, timeline_db: Path, tenant_id: str, limit: int) -> dict[str, Any]:
    config = CustomerTimelineReadApiConfig(timeline_db=timeline_db, allowed_root=timeline_db.parent)
    with CustomerTimelineReadApi.open(config) as api:
        health = api.health()
        if not health.get("validation_ok"):
            return {"found": False, "warnings": ["timeline_health_not_valid"]}
        customer = _find_customer_by_phone(api, tenant_id, phone)
        if not customer:
            return {"found": False, "warnings": ["timeline_customer_not_found"]}
        customer_id = safe_text(customer.get("customer_id"))
        profile = api.customer_profile(tenant_id, customer_id, event_limit=limit, bot_context_limit=limit)
    if not profile.get("found"):
        return {"found": False, "warnings": ["timeline_customer_profile_not_found"]}
    timeline_items = list(profile.get("timeline", {}).get("items", []))
    bot_context = profile.get("bot_context", {})
    readiness = dict(profile.get("readiness", {}))
    return {
        "found": True,
        "warnings": [],
        "customer_id": customer_id,
        "customer": profile.get("customer", {}),
        "summary": timeline_summary(timeline_items, bot_context),
        "timeline": {
            "items": timeline_items,
            "source_systems": count_values(timeline_items, "source_system"),
            "event_types": count_values(timeline_items, "event_type"),
        },
        "bot_context": bot_context,
        "readiness": readiness,
        "conflicts": profile.get("conflicts", {}),
        "redaction": profile.get("redaction", {}),
    }


def _find_customer_by_phone(api: CustomerTimelineReadApi, tenant_id: str, phone: str) -> Mapping[str, Any] | None:
    try:
        normalized = normalize_identity_value("phone", phone)
    except ValueError:
        normalized = normalize_phone_for_match(phone)
    links = api.store.list_identity_links(tenant_id, link_type="phone", link_value=normalized, limit=10)
    customer_ids = sorted({safe_text(item.get("customer_id")) for item in links if safe_text(item.get("customer_id"))})
    if len(customer_ids) == 1:
        customer = api.customer_profile(tenant_id, customer_ids[0], event_limit=1, bot_context_limit=1).get("customer")
        if isinstance(customer, Mapping):
            return customer
    if len(customer_ids) > 1:
        return None
    seen: set[str] = set()
    for term in phone_search_terms(phone):
        result = api.list_customers(tenant_id, q=term, limit=10)
        for item in result.get("items", []):
            customer_id = safe_text(item.get("customer_id"))
            if customer_id and customer_id not in seen:
                return item
            seen.add(customer_id)
    return None


def phone_search_terms(phone: str) -> list[str]:
    raw = safe_text(phone)
    digits = re.sub(r"\D+", "", raw)
    terms = [raw, digits]
    if len(digits) >= 10:
        terms.append(digits[-10:])
        terms.append("+7" + digits[-10:])
    return [term for term in dedupe_preserve_order(terms) if term]


def normalize_phone_for_match(value: Any) -> str:
    return canonical_normalize_phone(value) or ""


def extract_phones_from_row(row: Mapping[str, Any]) -> set[str]:
    phones: set[str] = set()
    for key in PHONE_KEYS:
        raw = safe_text(row.get(key))
        if not raw:
            continue
        for part in re.split(r"[|,; ]+", raw):
            normalized = normalize_phone_for_match(part)
            if normalized:
                phones.add(normalized)
    return phones


def timeline_summary(timeline_items: Sequence[Mapping[str, Any]], bot_context: Mapping[str, Any]) -> str:
    parts = []
    if timeline_items:
        latest = timeline_items[0]
        latest_at = safe_text(latest.get("event_at"))
        latest_summary = safe_text(latest.get("summary") or latest.get("text_preview") or latest.get("subject"))
        parts.append(f"Timeline: найдено событий {len(timeline_items)}.")
        if latest_summary:
            parts.append(f"Последнее: {latest_at} {latest_summary}".strip())
    visible_chunks = int_or_zero(bot_context.get("summary", {}).get("visible_chunks"))
    if visible_chunks:
        parts.append(f"Bot-context фрагментов: {visible_chunks}.")
    return " ".join(parts) if parts else "Timeline: данных по клиенту не найдено."


def fallback_summary(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "Fallback: данных по клиенту не найдено."
    latest = items[0]
    summary = safe_text(latest.get("summary"))
    return f"Fallback: найдено строк {len(items)}." + (f" Последнее: {summary}" if summary else "")


def count_values(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(safe_text(item.get(key)) for item in items if safe_text(item.get(key))))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def safe_text(value: Any) -> str:
    return str(value or "").strip()
