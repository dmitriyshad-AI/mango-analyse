from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib.parse import urljoin, urlparse

import requests

from mango_mvp.productization.crm_entity_resolver import normalize_snapshot_phones
from mango_mvp.productization.test_ingest import clean, path_is_relative_to
from mango_mvp.utils.phone import normalize_phone


AMO_SNAPSHOT_EXPORTER_SCHEMA_VERSION = "amo_snapshot_exporter_v1"


class HttpSession(Protocol):
    def get(self, url: str, *, headers: Mapping[str, str], params: Optional[Mapping[str, Any]], timeout: int) -> Any:
        ...


@dataclass(frozen=True)
class AmoSnapshotExportSummary:
    schema_version: str
    output_path: str
    base_url: str
    mode: str
    contacts_seen: int
    leads_seen: int
    entities_exported: int
    phones_indexed: int
    pages_fetched: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def export_amo_snapshot(
    product_root: Path,
    output_path: Path,
    *,
    base_url: Optional[str] = None,
    access_token: Optional[str] = None,
    session: Optional[HttpSession] = None,
    contacts_limit: int = 500,
    leads_limit: int = 500,
    timeout_seconds: int = 20,
    page_limit: int = 250,
    sleep_sec: float = 0.0,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    output_path = output_path.resolve(strict=False)
    guard_snapshot_output(product_root, output_path)
    if contacts_limit < 0 or leads_limit < 0:
        raise ValueError("contacts_limit and leads_limit must not be negative")
    if page_limit < 1 or page_limit > 250:
        raise ValueError("page_limit must be between 1 and 250")
    if sleep_sec < 0:
        raise ValueError("sleep_sec must not be negative")
    base = normalize_base_url(base_url or env_first("CRM_AMO_BASE_URL", "AMOCRM_BASE_URL"))
    token = clean(access_token) or env_first("CRM_AMO_API_TOKEN", "AMOCRM_ACCESS_TOKEN")
    if not base:
        raise ValueError("AMO base URL is required")
    if not token:
        raise ValueError("AMO access token is required")
    http = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    contacts, contact_pages = fetch_amo_collection(
        http,
        base_url=base,
        path="/api/v4/contacts",
        embedded_key="contacts",
        headers=headers,
        item_limit=contacts_limit,
        page_limit=page_limit,
        timeout_seconds=timeout_seconds,
        params={"with": "leads"},
        sleep_sec=sleep_sec,
    )
    leads, lead_pages = fetch_amo_collection(
        http,
        base_url=base,
        path="/api/v4/leads",
        embedded_key="leads",
        headers=headers,
        item_limit=leads_limit,
        page_limit=page_limit,
        timeout_seconds=timeout_seconds,
        params={"with": "contacts"},
        sleep_sec=sleep_sec,
    )
    entities = build_snapshot_entities(contacts=contacts, leads=leads)
    phones = sorted({phone for entity in entities for phone in entity.get("phones", [])})
    payload = {
        "schema_version": AMO_SNAPSHOT_EXPORTER_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "provider": "amocrm",
            "base_url": base,
            "mode": "live_read_only",
            "contacts_limit": contacts_limit,
            "leads_limit": leads_limit,
        },
        "entities": entities,
        "summary": AmoSnapshotExportSummary(
            schema_version=AMO_SNAPSHOT_EXPORTER_SCHEMA_VERSION,
            output_path=str(output_path),
            base_url=base,
            mode="live_read_only",
            contacts_seen=len(contacts),
            leads_seen=len(leads),
            entities_exported=len(entities),
            phones_indexed=len(phones),
            pages_fetched=contact_pages + lead_pages,
            validation_ok=True,
            blocked=0,
            warnings=entity_warnings(entities),
        ).to_json_dict(),
        "safety": safety_contract(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def fetch_amo_collection(
    session: HttpSession,
    *,
    base_url: str,
    path: str,
    embedded_key: str,
    headers: Mapping[str, str],
    item_limit: int,
    page_limit: int,
    timeout_seconds: int,
    params: Optional[Mapping[str, Any]] = None,
    sleep_sec: float = 0.0,
) -> tuple[list[Mapping[str, Any]], int]:
    if item_limit == 0:
        return [], 0
    items: list[Mapping[str, Any]] = []
    pages = 0
    next_url: Optional[str] = build_url(base_url, path)
    request_params = dict(params or {}) | {"limit": min(page_limit, item_limit)}
    while next_url and len(items) < item_limit:
        response = session.get(next_url, headers=headers, params=request_params, timeout=timeout_seconds)
        pages += 1
        payload = response_json(response)
        embedded = payload.get("_embedded") if isinstance(payload, Mapping) else {}
        page_items = embedded.get(embedded_key) if isinstance(embedded, Mapping) else []
        if isinstance(page_items, list):
            for item in page_items:
                if isinstance(item, Mapping):
                    items.append(dict(item))
                    if len(items) >= item_limit:
                        break
        next_url = next_href(payload)
        request_params = None
        if sleep_sec > 0 and next_url and len(items) < item_limit:
            time.sleep(sleep_sec)
    return items, pages


def build_snapshot_entities(
    *,
    contacts: Sequence[Mapping[str, Any]],
    leads: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    contact_by_id = {clean(contact.get("id")): contact for contact in contacts if clean(contact.get("id"))}
    entities: list[Mapping[str, Any]] = []
    for contact in contacts:
        phones = extract_contact_phones(contact)
        if not phones:
            continue
        linked_leads = embedded_items(contact, "leads")
        if not linked_leads:
            entities.append(
                {
                    "crm_provider": "amocrm",
                    "entity_type": "contact",
                    "entity_id": clean(contact.get("id")),
                    "entity_name": clean(contact.get("name")) or None,
                    "phones": phones,
                    "owner_id": clean(contact.get("responsible_user_id")) or None,
                    "owner_name": None,
                    "status": None,
                    "source_ref": f"amocrm:contact:{clean(contact.get('id'))}",
                }
            )
        for lead_ref in linked_leads:
            lead_id = clean(lead_ref.get("id"))
            if lead_id:
                entities.append(
                    {
                        "crm_provider": "amocrm",
                        "entity_type": "lead",
                        "entity_id": lead_id,
                        "entity_name": clean(lead_ref.get("name")) or f"Lead {lead_id}",
                        "phones": phones,
                        "owner_id": clean(lead_ref.get("responsible_user_id")) or clean(contact.get("responsible_user_id")) or None,
                        "owner_name": None,
                        "status": clean(lead_ref.get("status_id")) or None,
                        "source_ref": f"amocrm:contact:{clean(contact.get('id'))}:lead:{lead_id}",
                    }
                )
    for lead in leads:
        lead_id = clean(lead.get("id"))
        if not lead_id:
            continue
        lead_contacts = embedded_items(lead, "contacts")
        phones = []
        for lead_contact in lead_contacts:
            contact = contact_by_id.get(clean(lead_contact.get("id")))
            for phone in extract_contact_phones(contact or lead_contact):
                if phone not in phones:
                    phones.append(phone)
        if not phones:
            continue
        entities.append(
            {
                "crm_provider": "amocrm",
                "entity_type": "lead",
                "entity_id": lead_id,
                "entity_name": clean(lead.get("name")) or None,
                "phones": tuple(phones),
                "owner_id": clean(lead.get("responsible_user_id")) or None,
                "owner_name": None,
                "status": clean(lead.get("status_id")) or None,
                "source_ref": f"amocrm:lead:{lead_id}",
            }
        )
    return dedupe_entities(entities)


def extract_contact_phones(contact: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not contact:
        return ()
    phones = list(normalize_snapshot_phones(contact))
    for field in contact.get("custom_fields_values") or ():
        if not isinstance(field, Mapping):
            continue
        field_code = clean(field.get("field_code")).upper()
        field_name = clean(field.get("field_name")).casefold()
        if field_code != "PHONE" and "тел" not in field_name and "phone" not in field_name:
            continue
        for value in field.get("values") or ():
            raw = value.get("value") if isinstance(value, Mapping) else value
            phone = normalize_phone(str(raw))
            if phone and phone not in phones:
                phones.append(phone)
    return tuple(phones)


def embedded_items(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    embedded = payload.get("_embedded") if isinstance(payload, Mapping) else {}
    items = embedded.get(key) if isinstance(embedded, Mapping) else []
    return [item for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []


def dedupe_entities(entities: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    seen = set()
    for entity in entities:
        key = (
            clean(entity.get("crm_provider")),
            clean(entity.get("entity_type")),
            clean(entity.get("entity_id")),
            tuple(entity.get("phones") or ()),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(entity)
        normalized["phones"] = list(entity.get("phones") or ())
        result.append(normalized)
    return result


def entity_warnings(entities: Sequence[Mapping[str, Any]]) -> int:
    counts = Counter(phone for entity in entities for phone in entity.get("phones", ()))
    return sum(1 for count in counts.values() if count > 1)


def response_json(response: Any) -> Mapping[str, Any]:
    status_code = int(getattr(response, "status_code", 200))
    if status_code >= 300:
        text = getattr(response, "text", "")
        raise RuntimeError(f"amoCRM read-only export failed: HTTP {status_code} {text}")
    value = response.json()
    if not isinstance(value, Mapping):
        raise ValueError("amoCRM response must be a JSON object")
    return value


def next_href(payload: Mapping[str, Any]) -> Optional[str]:
    links = payload.get("_links") if isinstance(payload, Mapping) else {}
    next_meta = links.get("next") if isinstance(links, Mapping) else None
    href = next_meta.get("href") if isinstance(next_meta, Mapping) else None
    return clean(href) or None


def build_url(base_url: str, path: str) -> str:
    return urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))


def normalize_base_url(value: Optional[str]) -> str:
    text = clean(value)
    if not text:
        return ""
    if "://" not in text:
        text = f"https://{text}"
    parsed = urlparse(text)
    if not parsed.netloc:
        raise ValueError("AMO base URL must include a hostname")
    return f"{parsed.scheme}://{parsed.netloc}"


def env_first(*names: str) -> str:
    for name in names:
        value = clean(os.getenv(name))
        if value:
            return value
    return ""


def guard_snapshot_output(product_root: Path, output_path: Path) -> None:
    if "stable_runtime" in output_path.parts:
        raise ValueError("AMO snapshot output must not be under stable_runtime")
    if not path_is_relative_to(output_path, product_root):
        raise ValueError(f"AMO snapshot output must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "live_crm_reads": True,
        "write_crm": False,
        "write_tallanto": False,
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
    }
