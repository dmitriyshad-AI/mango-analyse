from __future__ import annotations

import pytest

from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy, TransportDenied


def test_transport_allows_only_declared_read_and_note_paths() -> None:
    calls = []

    def inner(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    transport = DefaultDenyTransport(inner)

    assert transport(method="GET", url="https://wappi.pro/tapi/sync/messages/get?profile_id=p&chat_id=c&mark_all=false") == {
        "ok": True
    }
    assert transport(method="GET", url="https://wappi.pro/maxapi/sync/chats/get?profile_id=p") == {"ok": True}
    assert transport(method="GET", url="https://wappi.pro/maxapi/sync/chats/get?profile_id=p&show_all=true") == {"ok": True}
    assert transport(method="GET", url="https://wappi.pro/maxapi/sync/messages/get?profile_id=p&chat_id=c&mark_all=false") == {
        "ok": True
    }
    assert transport(method="GET", url="https://educent.amocrm.ru/api/v4/contacts?query=test") == {"ok": True}
    assert transport(method="GET", url="https://educent.amocrm.ru/api/v4/contacts/2002/notes") == {"ok": True}
    assert transport(method="POST", url="https://wappi.pro/amocrm/contact/find") == {"ok": True}
    assert transport(method="POST", url="https://api.fotonai.online/api/integrations/amocrm/leads/49832125/notes") == {"ok": True}
    assert transport(method="POST", url="https://api.fotonai.online/api/integrations/amocrm/contacts/2002/notes") == {"ok": True}
    assert len(calls) == 9


def test_transport_denies_unknown_get_and_side_effect_wappi_params() -> None:
    transport = DefaultDenyTransport(lambda **kwargs: {"should": "not happen"})

    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://wappi.pro/tapi/profile/queue/purge?profile_id=p")
    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://wappi.pro/tapi/sync/messages/get?profile_id=p&chat_id=c&mark_all=true")
    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://wappi.pro/maxapi/sync/messages/get?profile_id=p&chat_id=c&mark_all=true")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://wappi.pro/maxapi/sync/chats/get?profile_id=p")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://wappi.pro/amocrm/contact/find?chat_id=1")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="http://wappi.pro/amocrm/contact/find")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://wappi.pro/amocrm/contact/other")
    with pytest.raises(TransportDenied):
        transport(method="PUT", url="https://wappi.pro/amocrm/contact/find")
    with pytest.raises(TransportDenied):
        transport(method="DELETE", url="https://educent.amocrm.ru/api/v4/leads/49832125")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://educent.amocrm.ru/api/v4/contacts")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://educent.amocrm.ru/api/v4/leads/49832125/notes")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://api.fotonai.online/api/v4/leads/49832125/notes")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://api.fotonai.online/api/integrations/amocrm/leads/49832125/notes?api_key=x")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="http://api.fotonai.online/api/integrations/amocrm/leads/49832125/notes")


def test_transport_denies_unknown_host() -> None:
    transport = DefaultDenyTransport(lambda **kwargs: {"should": "not happen"})

    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://example.com/api/v4/contacts")


def test_wappi_read_only_policy_denies_amo_and_ai_office_even_when_general_policy_allows_notes() -> None:
    transport = DefaultDenyTransport(lambda **kwargs: {"should": "not happen"}, policy=SafeTransportPolicy.wappi_read_only())

    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://educent.amocrm.ru/api/v4/contacts?query=test")
    with pytest.raises(TransportDenied):
        transport(method="POST", url="https://api.fotonai.online/api/integrations/amocrm/leads/49832125/notes")
