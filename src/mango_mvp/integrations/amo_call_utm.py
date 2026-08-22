from __future__ import annotations

import re
from typing import Any, Mapping

from mango_mvp.integrations.amo_wappi_auto_resolver import (
    embedded_items,
    lead_ids_from_contact,
)
from mango_mvp.utils.phone import normalize_phone


UTM_FIELDS = (
    ("UTM_SOURCE", "utm_source"),
    ("UTM_MEDIUM", "utm_medium"),
    ("UTM_CAMPAIGN", "utm_campaign"),
    ("UTM_CONTENT", "utm_content"),
    ("UTM_TERM", "utm_term"),
    ("UTM_REFERRER", "utm_referrer"),
)
MAX_ATTRIBUTION_AGE_SECONDS = 30 * 24 * 60 * 60
MAX_API_PAGES = 20
MAX_LINKED_LEADS = 500
MAX_DIRECT_CONTACTS = 50
MAX_UTM_TEXT_CHARS = 50_000
URL_LINE_RE = re.compile(r"^\s*url\s*:\s*(https?://\S+)\s*$", re.IGNORECASE)


class AmoCallUtmError(RuntimeError):
    pass


def _field_values(entity: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for field in entity.get("custom_fields_values") or ():
        if not isinstance(field, Mapping):
            continue
        code = str(field.get("field_code") or "").upper()
        values = [
            str(item.get("value") if isinstance(item, Mapping) else item).strip()
            for item in field.get("values") or ()
        ]
        values = [value for value in values if value]
        if code and values:
            result.setdefault(code, []).extend(values)
    return result


def _exact_phones(contact: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for field in contact.get("custom_fields_values") or ():
        if not isinstance(field, Mapping):
            continue
        if str(field.get("field_code") or "").upper() != "PHONE":
            continue
        for item in field.get("values") or ():
            phone = normalize_phone(item.get("value") if isinstance(item, Mapping) else item)
            if phone:
                result.add(phone)
    return result


def _url_from_notes(notes: list[Mapping[str, Any]], started_epoch: int) -> str:
    urls: set[str] = set()
    for note in notes:
        created_at = int(note.get("created_at") or 0)
        if (
            str(note.get("note_type") or "") != "common"
            or not 0 < created_at <= started_epoch
        ):
            continue
        for line in str((note.get("params") or {}).get("text") or "").splitlines():
            if match := URL_LINE_RE.fullmatch(line):
                urls.add(match.group(1))
    if len(urls) > 1:
        raise AmoCallUtmError("AMO lead has multiple historical landing URLs")
    return next(iter(urls), "")


def is_resolved_attribution(value: str) -> bool:
    return str(value).startswith("Связь с заявкой:")


def is_probable_attribution(value: str) -> bool:
    return str(value).startswith("Предполагаемая связь с заявкой:")


def _age_label(seconds: int) -> str:
    if seconds < 60:
        return "менее минуты"
    if seconds < 24 * 60 * 60:
        return f"{seconds // 60} мин" if seconds < 60 * 60 else f"{seconds // 3600} ч"
    return f"{seconds // (24 * 60 * 60)} дн"


class AmoCallUtmResolver:
    """Fail-closed attribution for a bounded publisher batch."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.phone_cache: dict[str, tuple[str, list[Mapping[str, Any]]]] = {}
        self.note_cache: dict[int, list[Mapping[str, Any]]] = {}

    def _phone_snapshot(self, phone: str) -> tuple[str, list[Mapping[str, Any]]]:
        contacts_by_id: dict[int, list[Mapping[str, Any]]] = {}
        exact_ids: set[int] = set()
        for page in range(1, MAX_API_PAGES + 1):
            payload = self.client.amo_api_get(
                path="contacts", params={"query": phone, "with": "leads", "page": page}, limit=50
            )
            page_items = embedded_items(payload, "contacts")
            for item in page_items:
                contact_id = int(item.get("id") or 0)
                if contact_id <= 0:
                    continue
                contacts_by_id.setdefault(contact_id, []).append(item)
                if phone in _exact_phones(item):
                    exact_ids.add(contact_id)
            has_next = bool((payload.get("_links") or {}).get("next"))
            if not page_items and has_next:
                raise AmoCallUtmError("AMO returned an empty contact page with next link")
            if not has_next:
                break
        else:
            raise AmoCallUtmError("AMO contact search exceeded its page limit")
        if not exact_ids:
            return "UTM не определены: точный контакт в AMO не найден", []
        if len(exact_ids) != 1:
            return "UTM не определены: номер связан с несколькими контактами", []
        contact_id = next(iter(exact_ids))
        lead_ids = sorted({
            int(lead_id)
            for contact in contacts_by_id[contact_id]
            for lead_id in lead_ids_from_contact(contact)
        })
        if not lead_ids:
            contact = self.client.amo_api_get(
                path=f"contacts/{contact_id}", params={"with": "leads"}, limit=1
            )
            lead_ids = lead_ids_from_contact(contact)
        if len(lead_ids) > MAX_LINKED_LEADS:
            raise AmoCallUtmError("AMO contact has too many linked leads")
        leads: list[Mapping[str, Any]] = []
        returned_ids: set[int] = set()
        for start in range(0, len(lead_ids), 50):
            chunk = [str(int(item)) for item in lead_ids[start : start + 50]]
            payload = self.client.amo_api_get(
                path="leads",
                params={"filter[id][]": chunk, "with": "contacts"},
                limit=len(chunk),
            )
            for lead in embedded_items(payload, "leads"):
                returned_ids.add(int(lead.get("id") or 0))
                embedded = lead.get("_embedded")
                raw_contacts = (
                    embedded.get("contacts") if isinstance(embedded, Mapping) else None
                )
                if not isinstance(raw_contacts, list):
                    raise AmoCallUtmError("AMO returned an incomplete linked lead snapshot")
                linked = {
                    str(item.get("id")) for item in raw_contacts
                    if isinstance(item, Mapping)
                }
                if str(contact_id) not in linked or bool(lead.get("is_deleted")):
                    continue
                if int(lead.get("created_at") or 0) <= 0:
                    raise AmoCallUtmError("AMO returned an incomplete linked lead snapshot")
                leads.append(lead)
        if returned_ids != {int(item) for item in lead_ids}:
            raise AmoCallUtmError("AMO returned an incomplete linked lead snapshot")
        return "", leads

    def _lead_url(self, lead_id: int, started_epoch: int) -> str:
        if lead_id in self.note_cache:
            return _url_from_notes(self.note_cache[lead_id], started_epoch)
        notes: list[Mapping[str, Any]] = []
        for page in range(1, MAX_API_PAGES + 1):
            payload = self.client.amo_api_get(
                path=f"leads/{lead_id}/notes", params={"page": page}, limit=50
            )
            page_items = embedded_items(payload, "notes")
            notes.extend(page_items)
            has_next = bool((payload.get("_links") or {}).get("next"))
            if not page_items and has_next:
                raise AmoCallUtmError("AMO returned an empty note page with next link")
            if not has_next:
                break
        else:
            raise AmoCallUtmError("AMO lead notes exceeded their page limit")
        self.note_cache[lead_id] = notes
        return _url_from_notes(notes, started_epoch)

    def _direct_phone_status(self, lead: Mapping[str, Any], phone: str) -> str:
        contact_ids = {
            int(item.get("id") or 0) for item in embedded_items(lead, "contacts")
            if int(item.get("id") or 0) > 0
        }
        if not contact_ids:
            return "UTM не определены: прямая заявка не связана с контактом"
        if len(contact_ids) > MAX_DIRECT_CONTACTS:
            raise AmoCallUtmError("AMO direct lead has too many linked contacts")
        payload = self.client.amo_api_get(
            path="contacts",
            params={"filter[id][]": [str(item) for item in sorted(contact_ids)]},
            limit=len(contact_ids),
        )
        contacts = embedded_items(payload, "contacts")
        if {int(item.get("id") or 0) for item in contacts} != contact_ids:
            raise AmoCallUtmError("AMO returned an incomplete direct contact snapshot")
        matches = [item for item in contacts if phone in _exact_phones(item)]
        if len(matches) != 1:
            return "UTM не определены: телефон не подтверждает прямую связь с заявкой"
        return ""

    def resolve(self, phone: Any, started_epoch: int, lead_id: Any = None) -> str:
        normalized = normalize_phone(phone)
        if not normalized:
            return "UTM не определены: телефон звонка некорректен"
        direct = lead_id not in (None, "")
        if direct:
            direct_id = int(lead_id)
            if direct_id <= 0:
                return "UTM не определены: прямая связь с заявкой некорректна"
            lead = self.client.amo_api_get(
                path=f"leads/{direct_id}", params={"with": "contacts"}, limit=1
            )
            created_at = int(lead.get("created_at") or 0)
            if int(lead.get("id") or 0) != direct_id or created_at <= 0:
                raise AmoCallUtmError("AMO returned an incomplete direct lead snapshot")
            if bool(lead.get("is_deleted")):
                return "UTM не определены: прямо связанная заявка удалена"
            if created_at > started_epoch:
                return "UTM не определены: прямо связанная заявка создана после звонка"
            if status := self._direct_phone_status(lead, normalized):
                return status
            relation = "Связь с заявкой: прямая из записи звонка"
        else:
            if normalized not in self.phone_cache:
                self.phone_cache[normalized] = self._phone_snapshot(normalized)
            status, leads = self.phone_cache[normalized]
            if status:
                return status
            past = [
                lead for lead in leads
                if 0 < int(lead.get("created_at") or 0) <= started_epoch
            ]
            if not past:
                return "UTM не определены: до звонка подходящей заявки не было"
            created = max(int(lead["created_at"]) for lead in past)
            nearest = [lead for lead in past if int(lead["created_at"]) == created]
            if len(nearest) != 1:
                return "UTM не определены: несколько заявок подходят к этому звонку"
            if started_epoch - created > MAX_ATTRIBUTION_AGE_SECONDS:
                return "UTM не определены: подходящая заявка старше 30 дней"
            lead = nearest[0]
            relation = (
                "Предполагаемая связь с заявкой: точный телефон; "
                "ближайшая заявка создана за "
                f"{_age_label(started_epoch - created)} до звонка"
            )
        values = _field_values(lead)
        utm_lines = [
            f"{label}: {value}"
            for code, label in UTM_FIELDS
            for value in values.get(code, ())
        ]
        lines = [
            relation,
            "UTM ниже — снимок карточки AMO при первом сопоставлении, а не "
            "исторические данные на момент звонка",
            *utm_lines,
        ]
        if not utm_lines:
            lines.append("UTM отсутствуют в связанной заявке")
        url = self._lead_url(int(lead["id"]), started_epoch)
        lines.append(f"url: {url}" if url else "Страница заявки не сохранена в AMO")
        result = "\n".join(lines)
        if len(result) > MAX_UTM_TEXT_CHARS:
            raise AmoCallUtmError("AMO UTM block exceeds the Google cell limit")
        return result
