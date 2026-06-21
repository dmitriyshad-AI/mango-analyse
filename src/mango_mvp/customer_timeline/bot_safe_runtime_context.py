from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig


BOT_SAFE_CRM_CONTEXT_ENV = "TELEGRAM_BOT_SAFE_CRM_CONTEXT"
BOT_SAFE_CRM_CONTEXT_DB_ENV = "TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB"
BOT_SAFE_CRM_CONTEXT_TENANT_ENV = "TELEGRAM_BOT_SAFE_CRM_CONTEXT_TENANT"
BOT_SAFE_CRM_CONTEXT_SCHEMA_VERSION = "bot_safe_crm_context_v1_2026_06_21"
BOT_SAFE_TIMELINE_CONTEXT_SOURCE = "customer_timeline_bot_context"
BOT_SAFE_CHUNK_TYPE = "bot_safe_summary"
DEFAULT_BOT_SAFE_TENANT_ID = "foton"

_TRUTHY_VALUES = {"1", "true", "yes", "on", "да", "y"}
_KNOWN_BRANDS = {"foton", "unpk"}
_PHONE_RE = re.compile(r"(?:\+7|8|7)?[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.I)
_SERVICE_ID_RE = re.compile(
    r"\b(?:customer:[a-f0-9]{16,}|timeline_event:[a-f0-9]{16,}|bot_context_chunk:[a-f0-9]{16,}|botsafe:[^\s,;]+)\b",
    re.I,
)


@dataclass(frozen=True)
class BotSafeLookup:
    tenant_id: str = DEFAULT_BOT_SAFE_TENANT_ID
    customer_id: str = ""
    amo_lead_id: str = ""
    amo_contact_id: str = ""


def bot_safe_crm_context_enabled(value: object = None) -> bool:
    if value is None:
        value = os.getenv(BOT_SAFE_CRM_CONTEXT_ENV)
    return str(value or "").strip().casefold() in _TRUTHY_VALUES


def bot_safe_timeline_db_from_env() -> Optional[Path]:
    raw = str(os.getenv(BOT_SAFE_CRM_CONTEXT_DB_ENV) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def bot_safe_tenant_from_env(default: str = DEFAULT_BOT_SAFE_TENANT_ID) -> str:
    return _clean_text(os.getenv(BOT_SAFE_CRM_CONTEXT_TENANT_ENV) or default) or default


def build_bot_safe_crm_context(
    *,
    timeline_db: Path | str | None,
    allowed_root: Path | str | None = None,
    active_brand: str,
    lookup: BotSafeLookup,
    limit: int = 3,
) -> Mapping[str, Any]:
    """Build the only CRM context allowed for the bot draft prompt.

    This function reads only CustomerTimelineReadApi.bot_context(..., allowed_only=True).
    It never calls customer_profile(), never exposes raw ids to the prompt, and returns
    an empty mapping on any unsafe or ambiguous condition.
    """

    brand = _normalize_brand(active_brand)
    if brand not in _KNOWN_BRANDS:
        return _empty_context("active_brand_not_supported", active_brand=brand)
    if timeline_db is None:
        return _empty_context("timeline_db_not_configured", active_brand=brand)
    db_path = Path(timeline_db).expanduser()
    if not db_path.exists():
        return _empty_context("timeline_db_missing", active_brand=brand)
    root = Path(allowed_root).expanduser() if allowed_root is not None else db_path.parent
    tenant_id = _clean_text(lookup.tenant_id) or DEFAULT_BOT_SAFE_TENANT_ID

    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=root)) as api:
        customer_id, warnings = _resolve_customer_id(api, lookup)
        if not customer_id:
            return _empty_context(*(warnings or ("customer_not_resolved",)), active_brand=brand)
        bot_context = api.bot_context(tenant_id, customer_id, allowed_only=True, limit=max(1, min(int(limit or 3) * 4, 50)))

    items = _safe_items_for_brand(bot_context.get("items") or (), active_brand=brand, limit=limit)
    if not items:
        return _empty_context("no_brand_scoped_bot_safe_context", active_brand=brand, customer_resolved=True)
    summary = _render_summary(items)
    pii_findings = scan_bot_safe_context_pii(summary)
    if pii_findings:
        return _empty_context("bot_safe_context_pii_blocked", active_brand=brand, customer_resolved=True, pii_findings=pii_findings)
    return {
        "schema_version": BOT_SAFE_CRM_CONTEXT_SCHEMA_VERSION,
        "source": BOT_SAFE_TIMELINE_CONTEXT_SOURCE,
        "found": True,
        "allowed_only": True,
        "active_brand": brand,
        "summary": summary,
        "timeline_context": {
            "schema_version": BOT_SAFE_CRM_CONTEXT_SCHEMA_VERSION,
            "source": BOT_SAFE_TIMELINE_CONTEXT_SOURCE,
            "found": True,
            "allowed_only": True,
            "active_brand": brand,
            "summary": summary,
            "bot_context": {
                "allowed_only": True,
                "brand_scoped": True,
                "items": items,
            },
            "warnings": list(warnings),
            "safety": {
                "source_api": "bot_context",
                "customer_profile_included": False,
                "raw_timeline_events_included": False,
                "raw_ids_included": False,
                "pii_scan_passed": True,
            },
        },
    }


def scan_bot_safe_context_pii(text: object) -> tuple[str, ...]:
    value = str(text or "")
    findings: list[str] = []
    if _PHONE_RE.search(value):
        findings.append("phone")
    if _EMAIL_RE.search(value):
        findings.append("email")
    if _SERVICE_ID_RE.search(value):
        findings.append("service_id")
    return tuple(findings)


def _resolve_customer_id(api: CustomerTimelineReadApi, lookup: BotSafeLookup) -> tuple[str, tuple[str, ...]]:
    candidates: dict[str, set[str]] = {}
    tenant_id = _clean_text(lookup.tenant_id) or DEFAULT_BOT_SAFE_TENANT_ID
    explicit_customer_id = _clean_text(lookup.customer_id)
    if explicit_customer_id:
        customer = api.store.get_customer(tenant_id, explicit_customer_id)
        if customer is None:
            return "", ("customer_not_found",)
        return explicit_customer_id, ()
    for link_type, raw_value in (
        ("amo_contact_id", lookup.amo_contact_id),
        ("amo_lead_id", lookup.amo_lead_id),
    ):
        value = _clean_text(raw_value)
        if not value:
            continue
        for link in api.store.list_identity_links(tenant_id, link_type=link_type, link_value=value, limit=10):
            customer_id = _clean_text(link.get("customer_id"))
            if not customer_id:
                continue
            if str(link.get("match_class") or "").strip().casefold() in {"duplicate", "ambiguous", "unmatched"}:
                continue
            candidates.setdefault(customer_id, set()).add(link_type)
    if not candidates:
        return "", ("customer_not_resolved",)
    if len(candidates) > 1:
        return "", ("ambiguous_identity",)
    return next(iter(candidates)), ()


def _safe_items_for_brand(items: Sequence[Any], *, active_brand: str, limit: int) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        tags = tuple(_normalize_tag(tag) for tag in item.get("relevance_tags") or ())
        if not _item_visible_for_active_brand(tags, active_brand=active_brand):
            continue
        if item.get("allowed_for_bot") is not True or item.get("requires_manager_review") is True:
            continue
        if str(item.get("chunk_type") or "").strip().casefold() != BOT_SAFE_CHUNK_TYPE:
            continue
        text = _clean_text(item.get("summary")) or _clean_text(item.get("text"))
        if not text or scan_bot_safe_context_pii(text):
            continue
        result.append(
            {
                "chunk_type": BOT_SAFE_CHUNK_TYPE,
                "text": _truncate(text, 700),
                "event_at": _clean_text(item.get("event_at")),
                "next_step_status": _next_step_status(item),
                "freshness_score": item.get("freshness_score"),
                "relevance_tags": [tag for tag in tags if tag in {"bot_safe", "structured", active_brand, "unknown"}],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            }
        )
        if len(result) >= max(1, int(limit or 3)):
            break
    return tuple(result)


def _next_step_status(item: Mapping[str, Any]) -> str:
    status = _clean_text(item.get("next_step_status")).casefold()
    if not status:
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            next_step = metadata.get("next_step")
            if isinstance(next_step, Mapping):
                status = _clean_text(next_step.get("status")).casefold()
    return status if status in {"active", "needs_manager_review", "empty"} else ""


def _item_visible_for_active_brand(tags: Sequence[str], *, active_brand: str) -> bool:
    tag_set = set(tags)
    if "bot_safe" not in tag_set:
        return False
    known_brand_tags = tag_set & _KNOWN_BRANDS
    if known_brand_tags - {active_brand}:
        return False
    return active_brand in tag_set or "unknown" in tag_set


def _render_summary(items: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for item in items:
        text = _clean_text(item.get("text"))
        if text:
            lines.append(text)
    return _truncate("\n".join(lines), 1800)


def _empty_context(
    *warnings: str,
    active_brand: str = "",
    customer_resolved: bool = False,
    pii_findings: Sequence[str] = (),
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": BOT_SAFE_CRM_CONTEXT_SCHEMA_VERSION,
        "source": BOT_SAFE_TIMELINE_CONTEXT_SOURCE,
        "found": False,
        "allowed_only": True,
        "active_brand": _normalize_brand(active_brand),
        "warnings": [warning for warning in warnings if warning],
        "customer_resolved": bool(customer_resolved),
    }
    if pii_findings:
        payload["pii_findings"] = list(pii_findings)
    return payload


def _normalize_brand(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text in {"foton", "фотон", "cdpo", "цдпо"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти", "mipt"}:
        return "unpk"
    return text


def _normalize_tag(value: object) -> str:
    return str(value or "").strip().casefold().replace("ё", "е")


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _truncate(value: str, limit: int) -> str:
    text = _clean_text(value)
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"
