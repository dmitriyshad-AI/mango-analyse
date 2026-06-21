from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from mango_mvp.existing_clients.amo_step1_snapshot import AmoMcpClient, AmoMcpConfig, read_mcp_env
from mango_mvp.integrations.draft_loop import DraftLoopKey, DraftLoopProfile, WappiHistoryMessage
from mango_mvp.utils.phone import normalize_phone


DEFAULT_AMO_MCP_ENV_PATH = Path("~/.mango_secrets/foton_crm_readonly_mcp_connector.env").expanduser()
DEFAULT_STOPLIST_PATH = Path.home() / ".mango_secrets" / "shared_phones_stoplist.json"
LEGACY_STOPLIST_PATH = Path.home() / ".mango_secrets" / "shared_phone_stoplist.json"
DRAFT_LOOP_AUTO_RESOLVER_ENV = "DRAFT_LOOP_AUTO_RESOLVER"
CLOSED_STATUS_IDS = {"142", "143"}
ORG_BRAND_KEYWORDS = {
    "foton": ("фотон", "cdpo", "цдпо"),
    "unpk": ("унпк", "мфти", "mipt"),
}


class AmoReadClient(Protocol):
    calls: int

    def amo_api_get(
        self,
        *,
        path: str,
        params: Mapping[str, Any] | None = None,
        limit: int = 50,
    ) -> Mapping[str, Any]:
        ...


def embedded_items(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    embedded = payload.get("_embedded")
    if isinstance(embedded, Mapping):
        raw = embedded.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
    raw = payload.get(key)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def custom_field_values(entity: Mapping[str, Any], *needles: str) -> list[str]:
    wanted = tuple(str(item).casefold() for item in needles if str(item).strip())
    values: list[str] = []
    for field in entity.get("custom_fields_values") or ():
        if not isinstance(field, Mapping):
            continue
        name = str(field.get("field_name") or field.get("name") or "").casefold()
        code = str(field.get("field_code") or "").casefold()
        if wanted and not any(needle in name or needle in code for needle in wanted):
            continue
        for item in field.get("values") or ():
            raw = item.get("value") if isinstance(item, Mapping) else item
            if str(raw or "").strip():
                values.append(str(raw).strip())
    return values


def contact_telegram_ids(contact: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for value in custom_field_values(contact, "telegram", "телеграм"):
        cleaned = re.sub(r"\D+", "", value)
        if cleaned:
            result.add(cleaned)
    return result


def contact_phones(contact: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for value in custom_field_values(contact, "phone", "телефон", "tel"):
        phone = normalize_phone(value)
        if phone:
            result.add(phone)
    return result


def lead_ids_from_contact(contact: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in embedded_items(contact, "leads"):
        lead_id = str(item.get("id") or "").strip()
        if lead_id and lead_id not in ids:
            ids.append(lead_id)
    return ids


def is_active_lead(lead: Mapping[str, Any]) -> bool:
    if bool(lead.get("is_deleted") or lead.get("deleted")):
        return False
    status_id = str(lead.get("status_id") or "").strip()
    closed_at = str(lead.get("closed_at") or "").strip()
    return status_id not in CLOSED_STATUS_IDS and not closed_at


def lead_org_values(lead: Mapping[str, Any]) -> list[str]:
    return custom_field_values(lead, "организация", "organization")


def lead_org_brand(lead: Mapping[str, Any]) -> str:
    values = lead_org_values(lead)
    text = " ".join(values).casefold()
    if not text:
        return ""
    matched: list[str] = []
    for brand, markers in ORG_BRAND_KEYWORDS.items():
        if any(marker in text for marker in markers):
            matched.append(brand)
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        return "mixed"
    return ""


def load_phone_stoplist(path: Path = DEFAULT_STOPLIST_PATH) -> tuple[set[str], str]:
    target = path.expanduser()
    if not target.exists() and target == DEFAULT_STOPLIST_PATH and LEGACY_STOPLIST_PATH.exists():
        target = LEGACY_STOPLIST_PATH
    if not target.exists():
        return set(), "shared_phone_stoplist_unavailable"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set(), "shared_phone_stoplist_invalid"
    raw: Any = payload.get("phones") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return set(), "shared_phone_stoplist_invalid"
    phones = {normalize_phone(item) for item in raw}
    phones.discard("")
    if not phones:
        return set(), "shared_phone_stoplist_unavailable"
    return phones, ""


def max_dialog_phone(dialog: Mapping[str, Any]) -> tuple[str, str]:
    direct = normalize_phone(dialog.get("phone") or dialog.get("number") or "")
    if direct:
        return direct, "max_phone_field"
    participants = dialog.get("participants")
    phones: set[str] = set()
    if isinstance(participants, Sequence) and not isinstance(participants, (str, bytes, bytearray)):
        for item in participants:
            if not isinstance(item, Mapping):
                continue
            phone = normalize_phone(item.get("phone") or item.get("number") or "")
            if phone:
                phones.add(phone)
    if len(phones) == 1:
        return next(iter(phones)), "max_participant_phone"
    if len(phones) > 1:
        return "", "max_multi_phone"
    return "", "max_phone_missing"


@dataclass
class AmoAutoResolver:
    client: AmoReadClient
    shared_phone_stoplist: set[str]
    stoplist_error: str = ""
    require_known_brand: bool = False

    def __post_init__(self) -> None:
        self.shared_phone_stoplist = {phone for phone in (normalize_phone(item) for item in self.shared_phone_stoplist) if phone}

    @property
    def calls(self) -> int:
        raw = getattr(self.client, "calls", 0)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return len(raw)
        return int(raw or 0)

    def __call__(
        self,
        *,
        key: DraftLoopKey,
        profile: DraftLoopProfile,
        dialog: Mapping[str, Any],
        messages: Sequence[WappiHistoryMessage],
        message: WappiHistoryMessage,
    ) -> Mapping[str, Any]:
        del messages, message
        if profile.channel == "telegram":
            return self._resolve_telegram(key=key, profile=profile)
        if profile.channel == "max":
            return self._resolve_max(key=key, profile=profile, dialog=dialog)
        return {"status": "rejected", "reason": "unsupported_channel"}

    def _resolve_telegram(self, *, key: DraftLoopKey, profile: DraftLoopProfile) -> Mapping[str, Any]:
        if not key.chat_id.isdigit():
            return {"status": "rejected", "reason": "username_only", "channel": "telegram"}
        contacts = self._search_contacts_exact_telegram_id(key.chat_id)
        if len(contacts) != 1:
            return {"status": "rejected", "reason": "multi_contact" if contacts else "telegram_id_no_contact", "channel": "telegram"}
        return self._resolve_contact(profile=profile, contact=contacts[0], match_key="Telegram ID", match_value=key.chat_id)

    def _resolve_max(self, *, key: DraftLoopKey, profile: DraftLoopProfile, dialog: Mapping[str, Any]) -> Mapping[str, Any]:
        del key
        phone, source = max_dialog_phone(dialog)
        if not phone:
            return {"status": "rejected", "reason": source, "channel": "max"}
        if self.stoplist_error:
            return {"status": "rejected", "reason": self.stoplist_error, "channel": "max"}
        if phone in self.shared_phone_stoplist:
            return {"status": "rejected", "reason": "shared_phone", "channel": "max", "match_key": source}
        contacts = self._search_contacts_exact_phone(phone)
        if len(contacts) != 1:
            return {"status": "rejected", "reason": "multi_contact" if contacts else "no_contact", "channel": "max", "match_key": source}
        return self._resolve_contact(profile=profile, contact=contacts[0], match_key=source, match_value=phone)

    def _search_contacts_exact_telegram_id(self, telegram_id: str) -> list[Mapping[str, Any]]:
        payload = self.client.amo_api_get(path="contacts", params={"query": telegram_id, "with": "leads"}, limit=50)
        contacts = embedded_items(payload, "contacts")
        return [contact for contact in contacts if telegram_id in contact_telegram_ids(contact)]

    def _search_contacts_exact_phone(self, phone: str) -> list[Mapping[str, Any]]:
        payload = self.client.amo_api_get(path="contacts", params={"query": phone, "with": "leads"}, limit=50)
        contacts = embedded_items(payload, "contacts")
        return [contact for contact in contacts if phone in contact_phones(contact)]

    def _resolve_contact(
        self,
        *,
        profile: DraftLoopProfile,
        contact: Mapping[str, Any],
        match_key: str,
        match_value: str,
    ) -> Mapping[str, Any]:
        contact_id = str(contact.get("id") or "").strip()
        lead_ids = lead_ids_from_contact(contact)
        if not lead_ids and contact_id:
            contact_payload = self.client.amo_api_get(path=f"contacts/{int(contact_id)}", params={"with": "leads"}, limit=1)
            lead_ids = lead_ids_from_contact(contact_payload)
        leads: list[Mapping[str, Any]] = []
        deleted_seen = False
        for lead_id in lead_ids:
            lead = self.client.amo_api_get(path=f"leads/{int(lead_id)}", params={"with": "contacts"}, limit=1)
            if bool(lead.get("is_deleted") or lead.get("deleted")):
                deleted_seen = True
            leads.append(lead)
        active = [lead for lead in leads if is_active_lead(lead)]
        if not active:
            reason = "deleted_lead" if deleted_seen else "closed_lead" if leads else "no_active_lead"
            return {"status": "rejected", "reason": reason, "contact_id": contact_id, "match_key": match_key}
        if len(active) != 1:
            return {"status": "rejected", "reason": "multi_active_lead", "contact_id": contact_id, "match_key": match_key}
        lead = active[0]
        org_brand = lead_org_brand(lead)
        org_values = lead_org_values(lead)
        if self.require_known_brand and not org_brand:
            return {
                "status": "rejected",
                "reason": "brand_unknown",
                "contact_id": contact_id,
                "lead_id": str(lead.get("id") or ""),
                "organization_values": org_values,
            }
        if org_brand and org_brand != profile.brand:
            return {
                "status": "rejected",
                "reason": "brand_mismatch",
                "contact_id": contact_id,
                "lead_id": str(lead.get("id") or ""),
                "organization_brand": org_brand,
                "organization_values": org_values,
            }
        return {
            "status": "matched",
            "lead_id": str(lead.get("id") or ""),
            "contact_id": contact_id,
            "match_key": match_key,
            "match_value": match_value,
            "lead_snapshot": {
                "status_id": str(lead.get("status_id") or ""),
                "closed_at": str(lead.get("closed_at") or ""),
                "pipeline_id": str(lead.get("pipeline_id") or ""),
                "organization_brand": org_brand,
                "organization_values": org_values,
            },
        }


def build_amo_auto_resolver(
    *,
    amo_mcp_env_file: Path = DEFAULT_AMO_MCP_ENV_PATH,
    shared_phone_stoplist: Path = DEFAULT_STOPLIST_PATH,
    user_agent: str = "mango-wappi-auto-resolver/1.0",
    require_known_brand: bool = False,
) -> AmoAutoResolver:
    stoplist, stoplist_error = load_phone_stoplist(shared_phone_stoplist)
    config = read_mcp_env(amo_mcp_env_file)
    if config.transport != "curl" or config.user_agent != user_agent:
        config = AmoMcpConfig(
            connector_url=config.connector_url,
            bearer_token=config.bearer_token,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            user_agent=user_agent,
            transport="curl",
        )
    return AmoAutoResolver(
        client=AmoMcpClient(config),
        shared_phone_stoplist=stoplist,
        stoplist_error=stoplist_error,
        require_known_brand=require_known_brand,
    )
