from __future__ import annotations

from types import SimpleNamespace
from urllib import parse as url_parse
from urllib import error as url_error
import io

import pytest

from mango_mvp.amocrm_runtime.tallanto_api import (
    TallantoApiClient,
    TallantoApiConfig,
    TallantoApiError,
)
from mango_mvp.amocrm_runtime.tallanto_context import build_live_tallanto_context, build_tallanto_live_card
import mango_mvp.amocrm_runtime.tallanto_context as tallanto_context_module
import mango_mvp.amocrm_runtime.tallanto_api as tallanto_api_module


def test_tallanto_batch_fetch_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TALLANTO_BATCH_FETCH", raising=False)

    assert not tallanto_api_module._batch_fetch_enabled()


def test_tallanto_client_requests_and_paginates(monkeypatch):
    captured: list[dict] = []

    def fake_http_json_request(**kwargs):
        captured.append(kwargs)
        parsed = url_parse.urlparse(kwargs["url"])
        query = url_parse.parse_qs(parsed.query)
        method_name = query["method"][0]
        if method_name == "list_possible_modules":
            return {"Ученики": {"module": "Contact"}}
        if method_name == "get_entry_list":
            form = dict(url_parse.parse_qsl(url_parse.urlencode(kwargs["form_items"], doseq=True)))
            offset = int(form["offset"])
            if offset == 0:
                return {"entry_list": [{"id": "1"}, {"id": "2"}], "next_offset": 2}
            return {"entry_list": [{"id": "3"}], "next_offset": None}
        raise AssertionError(kwargs)

    monkeypatch.setattr(tallanto_api_module, "_http_json_request", fake_http_json_request)

    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))
    modules = client.list_possible_modules()
    assert modules["Ученики"]["module"] == "Contact"

    entries = client.iter_entry_list(module="Contact", select_fields=["first_name", "phone_mobile"])
    assert [entry["id"] for entry in entries] == ["1", "2", "3"]
    assert captured[0]["headers"]["X-Auth-Token"] == "token"
    assert "select_fields%5B%5D=first_name" in captured[1]["url"]


def test_tallanto_client_handles_invalid_base_url():
    with pytest.raises(TallantoApiError):
        TallantoApiClient(TallantoApiConfig(base_url="", api_token="token")).healthcheck()


def test_build_contact_context_by_contact_id(monkeypatch):
    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))

    monkeypatch.setattr(client, "contact_by_id", lambda contact_id, select_fields=None: {"id": contact_id, "first_name": "Иван"})
    monkeypatch.setattr(client, "opportunities_by_contact", lambda contact_id, max_records=100: [{"id": "O-1"}])
    monkeypatch.setattr(client, "requests_by_contact", lambda contact_id, max_records=100: [{"id": "R-1"}])
    monkeypatch.setattr(client, "finances_by_contact", lambda contact_id, max_records=100: [{"id": "F-1"}])
    monkeypatch.setattr(client, "course_relations_by_contact", lambda contact_id, max_records=100: [{"id": "C-1"}])
    monkeypatch.setattr(client, "class_relations_by_contact", lambda contact_id, max_records=100: [{"id": "CL-1"}])
    monkeypatch.setattr(client, "abonements_by_contact", lambda contact_id, max_records=100: [{"id": "A-1"}])
    monkeypatch.setattr(client, "classes_by_ids", lambda class_ids, max_records=100: [{"id": "MC-1"}])

    payload = client.build_contact_context_by_contact_id("123")
    assert payload["contacts_found"] == 1
    assert payload["contexts"][0]["contact"]["id"] == "123"
    assert payload["contexts"][0]["opportunities"][0]["id"] == "O-1"


def test_search_contacts_by_phone_skips_not_found_errors(monkeypatch):
    def fake_get_entry_by_fields(*, module, field_values, select_fields=None):
        if "phone_mobile" in field_values:
            return {"id": "123", "phone_mobile": list(field_values.values())[0]}
        raise TallantoApiError('HTTP 400 from Tallanto: {"name":"Not find by id","description":"Entry does not exist"}')

    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))
    monkeypatch.setattr(client, "get_entry_by_fields", fake_get_entry_by_fields)

    records = client.search_contacts_by_phone("+79000000000")
    assert len(records) == 1
    assert records[0]["id"] == "123"


def test_search_contacts_by_phone_keeps_full_scan_when_batch_fetch_disabled(monkeypatch):
    monkeypatch.setenv("TALLANTO_BATCH_FETCH", "0")
    calls: list[dict[str, str]] = []

    def fake_get_entry_by_fields(*, module, field_values, select_fields=None):
        calls.append(field_values)
        return {"entry_list": [{"id": "123", "phone_mobile": list(field_values.values())[0]}]}

    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))
    monkeypatch.setattr(client, "get_entry_by_fields", fake_get_entry_by_fields)

    records = client.search_contacts_by_phone("+79000000000", max_records=50)

    assert records == [{"id": "123", "phone_mobile": "+79000000000"}]
    assert len(calls) == len(client.CONTACT_PHONE_FIELDS) * 3


def test_search_contacts_by_phone_returns_after_first_hit_when_batch_fetch_enabled(monkeypatch):
    monkeypatch.setenv("TALLANTO_BATCH_FETCH", "1")
    calls: list[dict[str, str]] = []

    def fake_get_entry_by_fields(*, module, field_values, select_fields=None):
        calls.append(field_values)
        return {"entry_list": [{"id": "123", "phone_mobile": list(field_values.values())[0]}]}

    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))
    monkeypatch.setattr(client, "get_entry_by_fields", fake_get_entry_by_fields)

    records = client.search_contacts_by_phone("+79000000000", max_records=50)

    assert records == [{"id": "123", "phone_mobile": "+79000000000"}]
    assert len(calls) == 1


def test_build_contact_context_live_card_only_skips_unused_blocks_without_changing_card(monkeypatch):
    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))

    def build_payload(*, batch_enabled: bool | None) -> tuple[dict, dict[str, int]]:
        if batch_enabled is None:
            monkeypatch.delenv("TALLANTO_BATCH_FETCH", raising=False)
        else:
            monkeypatch.setenv("TALLANTO_BATCH_FETCH", "1" if batch_enabled else "0")
        calls = {
            "opportunities": 0,
            "requests": 0,
            "finances": 0,
            "course_relations": 0,
            "class_relations": 0,
            "abonements": 0,
            "classes": 0,
        }
        monkeypatch.setattr(
            client,
            "search_contacts_by_phone",
            lambda phone, max_records=20, select_fields=None: [{"id": "123", "filial": "Онлайн"}],
        )
        monkeypatch.setattr(
            client,
            "opportunities_by_contact",
            lambda contact_id, max_records=100: calls.__setitem__("opportunities", calls["opportunities"] + 1) or [{"id": "O-1"}],
        )
        monkeypatch.setattr(
            client,
            "requests_by_contact",
            lambda contact_id, max_records=100: calls.__setitem__("requests", calls["requests"] + 1) or [{"id": "R-1"}],
        )
        monkeypatch.setattr(
            client,
            "finances_by_contact",
            lambda contact_id, max_records=100: calls.__setitem__("finances", calls["finances"] + 1) or [{"id": "F-1"}],
        )
        monkeypatch.setattr(
            client,
            "course_relations_by_contact",
            lambda contact_id, max_records=100: calls.__setitem__("course_relations", calls["course_relations"] + 1)
            or [{"id": "CR-1"}],
        )
        monkeypatch.setattr(
            client,
            "class_relations_by_contact",
            lambda contact_id, max_records=100: calls.__setitem__("class_relations", calls["class_relations"] + 1)
            or [{"id": "REL-1", "class_id": "CL-1"}],
        )
        monkeypatch.setattr(
            client,
            "abonements_by_contact",
            lambda contact_id, max_records=100: calls.__setitem__("abonements", calls["abonements"] + 1) or [{"id": "A-1"}],
        )
        monkeypatch.setattr(
            client,
            "classes_by_ids",
            lambda class_ids, max_records=100: calls.__setitem__("classes", calls["classes"] + 1) or [{"id": "CL-1"}],
        )
        return client.build_contact_context("+79000000000", live_card_only=True), calls

    full_payload, full_calls = build_payload(batch_enabled=False)
    slim_payload, slim_calls = build_payload(batch_enabled=True)
    default_payload, default_calls = build_payload(batch_enabled=None)

    full_card = build_tallanto_live_card([full_payload["contexts"][0]], active_brand="foton")
    slim_card = build_tallanto_live_card([slim_payload["contexts"][0]], active_brand="foton")
    default_card = build_tallanto_live_card([default_payload["contexts"][0]], active_brand="foton")

    assert full_card == slim_card
    assert default_card == full_card
    assert full_calls == {
        "opportunities": 1,
        "requests": 1,
        "finances": 1,
        "course_relations": 1,
        "class_relations": 1,
        "abonements": 1,
        "classes": 1,
    }
    assert slim_calls == {
        "opportunities": 0,
        "requests": 0,
        "finances": 1,
        "course_relations": 0,
        "class_relations": 1,
        "abonements": 1,
        "classes": 1,
    }
    assert default_calls == full_calls


def test_tallanto_batch_default_off_preserves_compact_counts_for_phone_and_contact_id(monkeypatch):
    monkeypatch.delenv("TALLANTO_BATCH_FETCH", raising=False)
    monkeypatch.setattr(tallanto_context_module, "settings", SimpleNamespace(crm_tallanto_mode="http"))
    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))
    calls: list[tuple[str, bool]] = []

    def payload(*, contact_id: str) -> dict:
        return {
            "generated_at": "2026-06-18T00:00:00+00:00",
            "base_url": "https://kmipt.tallanto.com",
            "contacts_found": 1,
            "contexts": [
                {
                    "contact": {"id": contact_id, "filial": "Фотон"},
                    "opportunities": [{"id": "O-1", "sales_stage": "В работе"}],
                    "requests": [{"id": "R-1"}],
                    "finances": [{"id": "F-1"}],
                    "course_relations": [{"id": "CR-1", "course_id": "C-1", "contact_id": contact_id}],
                    "class_relations": [{"id": "REL-1", "class_id": "CL-1", "contact_id": contact_id}],
                    "abonements": [{"id": "A-1"}],
                    "classes": [{"id": "CL-1", "status": "active"}],
                }
            ],
        }

    def fake_build_contact_context(phone, *, max_contacts=5, max_related_records=40, live_card_only=False):
        calls.append(("phone", live_card_only))
        return payload(contact_id="123")

    def fake_build_contact_context_by_contact_id(contact_id, *, max_related_records=40, live_card_only=False):
        calls.append(("id", live_card_only))
        return payload(contact_id=str(contact_id))

    monkeypatch.setattr(client, "build_contact_context", fake_build_contact_context)
    monkeypatch.setattr(client, "build_contact_context_by_contact_id", fake_build_contact_context_by_contact_id)
    monkeypatch.setattr(
        tallanto_context_module,
        "build_tallanto_api_config",
        lambda: TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"),
    )
    monkeypatch.setattr(tallanto_context_module, "TallantoApiClient", lambda _config: client)

    by_phone = build_live_tallanto_context(phone="+79000000000", active_brand="foton")
    by_id = build_live_tallanto_context(
        phone="+79000000000",
        tallanto_id="123",
        tallanto_match_status="exact_phone_single",
        active_brand="foton",
    )

    assert calls == [("phone", False), ("id", False)]
    for result in (by_phone, by_id):
        assert result["status"] == "ok"
        compact = result["contexts"][0]
        assert compact["opportunity_count"] > 0
        assert compact["course_relation_count"] > 0


def test_classes_by_ids_skips_not_found_errors(monkeypatch):
    client = TallantoApiClient(TallantoApiConfig(base_url="https://kmipt.tallanto.com", api_token="token"))

    def fake_get_entry_by_id(*, module, entry_id, select_fields=None):
        if entry_id == "missing":
            raise TallantoApiError('HTTP 400 from Tallanto: {"name":"Not find by id","description":"Entry does not exist"}')
        return {"entry_list": [{"id": entry_id, "status": "active"}]}

    monkeypatch.setattr(client, "get_entry_by_id", fake_get_entry_by_id)

    assert client.classes_by_ids(["missing", "ok"]) == [{"id": "ok", "status": "active"}]


def test_tallanto_http_json_request_retries_rate_limit(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    calls = {"count": 0}

    def fake_urlopen(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise url_error.HTTPError(
                url="https://kmipt.tallanto.com/service/api/rest.php",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"rate_limited"}'),
            )
        return _Response()

    monkeypatch.setattr(tallanto_api_module.url_request, "urlopen", fake_urlopen)
    monkeypatch.setattr(tallanto_api_module.time, "sleep", lambda *_args, **_kwargs: None)

    payload = tallanto_api_module._http_json_request(
        method="GET",
        url="https://kmipt.tallanto.com/service/api/rest.php?method=list_possible_modules",
        headers={"X-Auth-Token": "token"},
    )
    assert payload == {"ok": True}
    assert calls["count"] == 3
