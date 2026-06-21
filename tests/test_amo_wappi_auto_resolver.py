from __future__ import annotations

import json
from typing import Any

from mango_mvp.integrations.amo_wappi_auto_resolver import AmoAutoResolver
from mango_mvp.integrations.draft_loop import DraftLoopKey, DraftLoopProfile


class FakeMcp:
    def __init__(self, contacts=None, leads=None) -> None:
        self.contacts = contacts or []
        self.leads = {str(item["id"]): item for item in (leads or [])}
        self.calls: list[dict[str, Any]] = []

    def amo_api_get(self, *, path, params=None, limit=50):
        self.calls.append({"path": path, "params": params or {}, "limit": limit})
        if path == "contacts":
            query = str((params or {}).get("query") or "")
            return {"_embedded": {"contacts": [item for item in self.contacts if query in json.dumps(item, ensure_ascii=False)]}}
        if path.startswith("contacts/"):
            contact_id = path.split("/", 1)[1]
            return next((item for item in self.contacts if str(item.get("id")) == contact_id), {})
        if path.startswith("leads/"):
            return self.leads.get(path.split("/", 1)[1], {})
        raise AssertionError(path)


def test_telegram_resolver_requires_exact_telegram_id_not_phone() -> None:
    resolver = AmoAutoResolver(
        client=FakeMcp(contacts=[_contact("2002", phone="+7 999 123-45-67", leads=("1001",))], leads=[_lead("1001", org="Фотон")]),
        shared_phone_stoplist=set(),
    )

    result = resolver(
        key=DraftLoopKey("p-tg", "9991234567"),
        profile=DraftLoopProfile("p-tg", "foton", "telegram"),
        dialog={"phone": "+7 999 123-45-67"},
        messages=[],
        message=None,  # type: ignore[arg-type]
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "telegram_id_no_contact"


def test_strict_brand_resolver_fails_closed_on_unknown_organization_brand() -> None:
    resolver = AmoAutoResolver(
        client=FakeMcp(contacts=[_contact("2002", telegram_id="123456", leads=("1001",))], leads=[_lead("1001", org="")]),
        shared_phone_stoplist=set(),
        require_known_brand=True,
    )

    result = resolver(
        key=DraftLoopKey("p-tg", "123456"),
        profile=DraftLoopProfile("p-tg", "foton", "telegram"),
        dialog={},
        messages=[],
        message=None,  # type: ignore[arg-type]
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "brand_unknown"


def test_resolver_fails_closed_on_mixed_brand_organization() -> None:
    resolver = AmoAutoResolver(
        client=FakeMcp(contacts=[_contact("2002", telegram_id="123456", leads=("1001",))], leads=[_lead("1001", org="Фотон + УНПК МФТИ")]),
        shared_phone_stoplist=set(),
        require_known_brand=True,
    )

    result = resolver(
        key=DraftLoopKey("p-tg", "123456"),
        profile=DraftLoopProfile("p-tg", "foton", "telegram"),
        dialog={},
        messages=[],
        message=None,  # type: ignore[arg-type]
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "brand_mismatch"
    assert result["organization_brand"] == "mixed"


def _contact(contact_id="111", *, telegram_id="", phone="", leads=("49762441",)):
    fields = []
    if telegram_id:
        fields.append({"field_name": "Telegram ID", "values": [{"value": telegram_id}]})
    if phone:
        fields.append({"field_code": "PHONE", "field_name": "Телефон", "values": [{"value": phone}]})
    return {
        "id": contact_id,
        "custom_fields_values": fields,
        "_embedded": {"leads": [{"id": int(item)} for item in leads]},
    }


def _lead(lead_id="49762441", *, status_id=123, closed_at=None, deleted=False, org=""):
    fields = []
    if org:
        fields.append({"field_name": "Организация", "values": [{"value": org}]})
    return {
        "id": int(lead_id),
        "status_id": status_id,
        "closed_at": closed_at,
        "is_deleted": deleted,
        "pipeline_id": 999,
        "custom_fields_values": fields,
    }
