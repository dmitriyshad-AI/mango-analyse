from __future__ import annotations

from argparse import Namespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mango_mvp.amocrm_runtime.auth import DEFAULT_DEV_CONTEXT, require_api_key
from mango_mvp.amocrm_runtime.db import get_db
from mango_mvp.amocrm_runtime.routers import deals as deals_router_module
from mango_mvp.amocrm_runtime.routers.deals import LIVE_WRITE_CONFIRMATION, router
from scripts import write_amo_ready_contacts, write_recent_actionable_deals


class FakeSession:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def _client(fake_session: FakeSession) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")

    def override_db():
        yield fake_session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[require_api_key] = lambda: DEFAULT_DEV_CONTEXT
    return TestClient(app)


def test_contact_writeback_script_defaults_to_dry_run() -> None:
    args = Namespace(execute_live_write=False, live_confirmation="")

    assert write_amo_ready_contacts._live_write_enabled(args) is False


def test_contact_writeback_script_requires_live_confirmation() -> None:
    args = Namespace(execute_live_write=True, live_confirmation="")

    with pytest.raises(ValueError, match="Live amoCRM writeback requires"):
        write_amo_ready_contacts._live_write_enabled(args)


def test_contact_writeback_script_accepts_explicit_live_confirmation() -> None:
    args = Namespace(execute_live_write=True, live_confirmation=write_amo_ready_contacts.LIVE_WRITE_CONFIRMATION)

    assert write_amo_ready_contacts._live_write_enabled(args) is True


def test_deal_writeback_script_defaults_to_dry_run() -> None:
    args = Namespace(execute_live_write=False, live_confirmation="")

    assert write_recent_actionable_deals._live_write_enabled(args) is False


def test_deal_writeback_script_requires_live_confirmation() -> None:
    args = Namespace(execute_live_write=True, live_confirmation="")

    with pytest.raises(ValueError, match="Live amoCRM writeback requires"):
        write_recent_actionable_deals._live_write_enabled(args)


def test_deal_writeback_endpoint_refuses_live_write_without_confirmation() -> None:
    fake_session = FakeSession()
    client = _client(fake_session)

    response = client.post(
        "/api/integrations/amocrm/deals/writeback",
        json={"analysis": {"matched_lead_id": 123}},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "live_write_confirmation_required"
    assert fake_session.committed is False
    assert fake_session.rolled_back is False


def test_deal_writeback_endpoint_allows_explicit_live_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = FakeSession()
    client = _client(fake_session)
    calls: list[dict] = []

    def fake_write_analysis_to_lead(db, *, analysis):
        calls.append({"db": db, "analysis": analysis})
        return {"status": "written", "updated_fields": ["AI-сводка по сделке"]}

    monkeypatch.setattr(deals_router_module, "write_analysis_to_lead", fake_write_analysis_to_lead)

    response = client.post(
        "/api/integrations/amocrm/deals/writeback",
        json={
            "analysis": {"matched_lead_id": 123},
            "execute_live_write": True,
            "live_confirmation": LIVE_WRITE_CONFIRMATION,
        },
    )

    assert response.status_code == 200
    assert response.json()["result"]["status"] == "written"
    assert calls == [{"db": fake_session, "analysis": {"matched_lead_id": 123}}]
    assert fake_session.committed is True


def test_queue_build_apply_writeback_refuses_without_live_confirmation() -> None:
    fake_session = FakeSession()
    client = _client(fake_session)

    response = client.post(
        "/api/integrations/amocrm/deals/queue/build",
        json={"apply_writeback": True, "days_back": 7},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["action"] == "deals/queue/build:apply_writeback"
    assert fake_session.committed is False
