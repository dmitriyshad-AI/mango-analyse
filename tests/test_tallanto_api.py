from __future__ import annotations

from urllib import parse as url_parse
from urllib import error as url_error
import io

import pytest

from mango_mvp.amocrm_runtime.tallanto_api import (
    TallantoApiClient,
    TallantoApiConfig,
    TallantoApiError,
)
import mango_mvp.amocrm_runtime.tallanto_api as tallanto_api_module


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
