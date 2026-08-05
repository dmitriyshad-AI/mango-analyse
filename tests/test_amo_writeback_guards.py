from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mango_mvp.amocrm_runtime import amo_integration
from mango_mvp.amocrm_runtime.auth import DEFAULT_DEV_CONTEXT, require_api_key
from mango_mvp.amocrm_runtime.db import get_db
from mango_mvp.amocrm_runtime.routers import deals as deals_router_module
from mango_mvp.amocrm_runtime.routers.deals import LIVE_WRITE_CONFIRMATION, router
from scripts import write_recent_actionable_deals


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


def _amo_context() -> amo_integration.AmoAccessContext:
    return amo_integration.AmoAccessContext(
        account_base_url="https://example.amocrm.ru",
        access_token="token",
        token_source="test",
        connection=None,
    )


def test_contact_update_ai_allowlist_env_off_still_uses_strict_payload_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRM_CONTACT_WRITEBACK_AI_ALLOWLIST", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(amo_integration, "resolve_amo_access_context", lambda _session: _amo_context())
    monkeypatch.setattr(
        amo_integration,
        "fetch_contact_field_catalog",
        lambda _session: [
            {"id": 1, "name": "AI-рекомендованный следующий шаг", "type": "text"},
            {"id": 2, "name": "Email", "type": "text"},
            {"id": 3, "name": "Внешнее поле", "type": "text"},
            {"id": 4, "name": "Id Tallanto", "type": "text"},
        ],
    )
    monkeypatch.setattr(amo_integration, "_amo_http_request", lambda **kwargs: calls.append(kwargs) or {"ok": True})

    result = amo_integration.send_contact_custom_field_update(
        FakeSession(),
        contact_id=123,
        field_payload={
            "AI-рекомендованный следующий шаг": "Позвонить",
            "Email": "parent@example.com",
            "Внешнее поле": "старое поведение",
            "Id Tallanto": "protected",
        },
    )

    assert result["updated_fields"] == ["AI-рекомендованный следующий шаг"]
    assert calls[0]["body"]["custom_fields_values"] == [{"field_id": 1, "values": [{"value": "Позвонить"}]}]


def test_contact_update_ai_allowlist_blocks_manual_and_identity_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRM_CONTACT_WRITEBACK_AI_ALLOWLIST", "1")
    calls: list[dict] = []
    monkeypatch.setattr(amo_integration, "resolve_amo_access_context", lambda _session: _amo_context())
    monkeypatch.setattr(
        amo_integration,
        "fetch_contact_field_catalog",
        lambda _session: [
            {"id": 1, "name": "AI-рекомендованный следующий шаг", "type": "text"},
            {"id": 2, "name": "Email", "type": "text"},
            {"id": 3, "name": "История общения", "type": "textarea"},
            {"id": 4, "name": "Внешнее поле", "type": "text"},
        ],
    )
    monkeypatch.setattr(amo_integration, "_amo_http_request", lambda **kwargs: calls.append(kwargs) or {"ok": True})

    result = amo_integration.send_contact_custom_field_update(
        FakeSession(),
        contact_id=123,
        field_payload={
            "AI-рекомендованный следующий шаг": "Позвонить",
            "Email": "parent@example.com",
            "История общения": "ручное поле",
            "Внешнее поле": "лишнее",
        },
    )

    assert result["updated_fields"] == ["AI-рекомендованный следующий шаг"]
    assert calls[0]["body"]["custom_fields_values"] == [{"field_id": 1, "values": [{"value": "Позвонить"}]}]


def test_contact_write_payload_allowlist_blocks_identity_and_manual_fields() -> None:
    payload = amo_integration.sanitize_contact_write_payload(
        {
            "AI-рекомендованный следующий шаг": "Позвонить",
            "Последняя AI-сводка": "Сводка",
            "Авто история общения": "История",
            "Email": "parent@example.com",
            "ФИО": "Иванов",
            "История общения": "ручное поле",
            "Статус матчинга": "exact",
        }
    )

    assert payload == {
        "AI-рекомендованный следующий шаг": "Позвонить",
        "Последняя AI-сводка": "Сводка",
        "Авто история общения": "История",
    }


def test_lead_write_payload_allowlist_blocks_status_and_responsible_fields() -> None:
    payload = amo_integration.sanitize_lead_write_payload(
        {
            "AI-сводка по сделке": "Сводка",
            "AI-рекомендованный следующий шаг": "Позвонить",
            "status_id": "123",
            "pipeline_id": "456",
            "Ответственный": "manager",
            "Email": "parent@example.com",
        }
    )

    assert payload == {
        "AI-сводка по сделке": "Сводка",
        "AI-рекомендованный следующий шаг": "Позвонить",
    }


def test_deal_writeback_script_defaults_to_dry_run() -> None:
    args = Namespace(execute_live_write=False, live_confirmation="")

    assert write_recent_actionable_deals._live_write_enabled(args) is False


def test_deal_writeback_script_requires_live_confirmation() -> None:
    args = Namespace(execute_live_write=True, live_confirmation="")

    with pytest.raises(ValueError, match="Live amoCRM writeback requires"):
        write_recent_actionable_deals._live_write_enabled(args)


def test_deal_writeback_script_accepts_explicit_live_confirmation_and_quality_gate(tmp_path: Path) -> None:
    args = Namespace(
        execute_live_write=True,
        live_confirmation=write_recent_actionable_deals.LIVE_WRITE_CONFIRMATION,
        quality_gate_summary=_quality_gate_summary_fixture(tmp_path),
    )

    assert write_recent_actionable_deals._live_write_enabled(args) is True


def test_deal_writeback_script_rejects_failed_quality_gate_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"passed": False, "readiness": {"crm_quality_writeback_ready": True}}), encoding="utf-8")
    args = Namespace(
        execute_live_write=True,
        live_confirmation=write_recent_actionable_deals.LIVE_WRITE_CONFIRMATION,
        quality_gate_summary=str(summary),
    )

    with pytest.raises(ValueError, match="not passed"):
        write_recent_actionable_deals._live_write_enabled(args)


def test_deal_writeback_endpoint_refuses_live_write_without_confirmation() -> None:
    fake_session = FakeSession()
    response = _client(fake_session).post(
        "/api/integrations/amocrm/deals/writeback",
        json={"analysis": {"matched_lead_id": 123}},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "live_write_confirmation_required"
    assert fake_session.committed is False
    assert fake_session.rolled_back is False


def test_deal_writeback_endpoint_allows_explicit_live_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = FakeSession()
    calls: list[dict] = []

    def fake_write_analysis_to_lead(db, *, analysis):
        calls.append({"db": db, "analysis": analysis})
        return {"status": "written", "updated_fields": ["AI-сводка по сделке"]}

    monkeypatch.setattr(deals_router_module, "write_analysis_to_lead", fake_write_analysis_to_lead)
    response = _client(fake_session).post(
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
    response = _client(fake_session).post(
        "/api/integrations/amocrm/deals/queue/build",
        json={"apply_writeback": True, "days_back": 7},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["action"] == "deals/queue/build:apply_writeback"
    assert fake_session.committed is False


def _quality_gate_summary_fixture(root: Path) -> str:
    path = root / "summary.json"
    path.write_text(
        json.dumps({"passed": True, "readiness": {"crm_quality_writeback_ready": True}}),
        encoding="utf-8",
    )
    return str(path)
