from __future__ import annotations

import pytest

from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, TransportDenied


def test_transport_allows_only_declared_read_and_note_paths() -> None:
    calls = []

    def inner(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    transport = DefaultDenyTransport(inner)

    assert transport(method="GET", url="https://wappi.pro/tapi/sync/messages/get?profile_id=p&chat_id=c&mark_all=false") == {
        "ok": True
    }
    assert transport(method="GET", url="https://educent.amocrm.ru/api/v4/contacts?query=test") == {"ok": True}
    assert transport(method="POST", url="https://api.fotonai.online/api/integrations/amocrm/leads/49832125/notes") == {"ok": True}
    assert len(calls) == 3


def test_transport_denies_unknown_get_and_side_effect_wappi_params() -> None:
    transport = DefaultDenyTransport(lambda **kwargs: {"should": "not happen"})

    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://wappi.pro/tapi/profile/queue/purge?profile_id=p")
    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://wappi.pro/tapi/sync/messages/get?profile_id=p&chat_id=c&mark_all=true")
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


def test_transport_denies_unknown_host() -> None:
    transport = DefaultDenyTransport(lambda **kwargs: {"should": "not happen"})

    with pytest.raises(TransportDenied):
        transport(method="GET", url="https://example.com/api/v4/contacts")
