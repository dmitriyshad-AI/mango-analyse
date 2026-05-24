from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

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


class BrokenPreflightSession:
    def execute(self, _statement):
        raise RuntimeError("db tunnel down")


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


def test_contact_writeback_default_input_uses_active_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    active_root = tmp_path / "stable_runtime" / "sales_master_export_current"
    active_root.mkdir(parents=True)
    active_csv = active_root / "amo_export_ready_ru.csv"
    active_csv.write_text("Телефон клиента\n+79990000000\n", encoding="utf-8")
    pointer = tmp_path / "stable_runtime" / "CANONICAL_EXPORT.txt"
    pointer.write_text("sales_master_export_current\n", encoding="utf-8")
    legacy = tmp_path / "АКТУАЛЬНО_AMO_ready.xlsx"

    monkeypatch.setattr(write_amo_ready_contacts, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(write_amo_ready_contacts, "CANONICAL_EXPORT_POINTER", pointer)
    monkeypatch.setattr(write_amo_ready_contacts, "LEGACY_ROOT_AMO_READY_XLSX", legacy)

    assert write_amo_ready_contacts._default_amo_ready_input() == active_csv


def test_contact_writeback_script_reads_csv_without_pandas(tmp_path: Path) -> None:
    source = tmp_path / "amo_ready.csv"
    source.write_text(
        "Телефон клиента,Краткая история общения,Следующий шаг\n"
        "+79990000000,История клиента,Перезвонить\n",
        encoding="utf-8",
    )

    rows = write_amo_ready_contacts._read_rows(source)

    assert rows == [
        {
            "Телефон клиента": "+79990000000",
            "Краткая история общения": "История клиента",
            "Следующий шаг": "Перезвонить",
        }
    ]


def test_contact_writeback_script_requires_live_confirmation() -> None:
    args = Namespace(execute_live_write=True, live_confirmation="")

    with pytest.raises(ValueError, match="Live amoCRM writeback requires"):
        write_amo_ready_contacts._live_write_enabled(args)


def test_contact_writeback_script_accepts_explicit_live_confirmation(tmp_path: Path) -> None:
    args = Namespace(
        execute_live_write=True,
        live_confirmation=write_amo_ready_contacts.LIVE_WRITE_CONFIRMATION,
        quality_gate_summary=_quality_gate_summary_fixture(tmp_path),
        crm_writeback_quality_summary=_crm_writeback_quality_summary_fixture(tmp_path),
    )

    assert write_amo_ready_contacts._live_write_enabled(args) is True


def test_contact_writeback_script_rejects_missing_crm_text_quality_gate(tmp_path: Path) -> None:
    summary = tmp_path / "crm_writeback_summary.json"
    summary.write_text(
        json.dumps({"passed": True, "population_recall": {"passed_for_live": True}}),
        encoding="utf-8",
    )
    args = Namespace(
        execute_live_write=True,
        live_confirmation=write_amo_ready_contacts.LIVE_WRITE_CONFIRMATION,
        quality_gate_summary=_quality_gate_summary_fixture(tmp_path),
        crm_writeback_quality_summary=str(summary),
    )

    with pytest.raises(ValueError, match="crm_text_quality"):
        write_amo_ready_contacts._live_write_enabled(args)


def test_contact_writeback_script_rejects_failed_crm_text_quality_gate(tmp_path: Path) -> None:
    summary = tmp_path / "crm_writeback_summary.json"
    summary.write_text(
        json.dumps(
            {
                "passed": True,
                "population_recall": {"passed_for_live": True},
                "crm_text_quality": {"passed_for_live": False, "blocking_rows": 1},
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        execute_live_write=True,
        live_confirmation=write_amo_ready_contacts.LIVE_WRITE_CONFIRMATION,
        quality_gate_summary=_quality_gate_summary_fixture(tmp_path),
        crm_writeback_quality_summary=str(summary),
    )

    with pytest.raises(ValueError, match="CRM text quality gate"):
        write_amo_ready_contacts._live_write_enabled(args)


def test_contact_writeback_script_rejects_crm_quality_summary_for_other_input(tmp_path: Path) -> None:
    expected_input = tmp_path / "expected.csv"
    other_input = tmp_path / "other.csv"
    expected_input.write_text("Телефон клиента\n", encoding="utf-8")
    other_input.write_text("Телефон клиента\n", encoding="utf-8")
    summary = tmp_path / "crm_writeback_summary.json"
    summary.write_text(
        json.dumps(
            {
                "passed": True,
                "input": str(other_input),
                "population_recall": {"passed_for_live": True},
                "crm_text_quality": {"passed_for_live": True, "blocking_rows": 0},
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        input=str(expected_input),
        execute_live_write=True,
        live_confirmation=write_amo_ready_contacts.LIVE_WRITE_CONFIRMATION,
        quality_gate_summary=_quality_gate_summary_fixture(tmp_path),
        crm_writeback_quality_summary=str(summary),
    )

    with pytest.raises(ValueError, match="summary input does not match"):
        write_amo_ready_contacts._live_write_enabled(args)


def test_contact_writeback_script_requires_quality_gate_summary_for_live_write() -> None:
    args = Namespace(
        execute_live_write=True,
        live_confirmation=write_amo_ready_contacts.LIVE_WRITE_CONFIRMATION,
        quality_gate_summary="",
        crm_writeback_quality_summary="",
    )

    with pytest.raises(ValueError, match="quality-gate-summary"):
        write_amo_ready_contacts._live_write_enabled(args)


def test_deal_writeback_script_defaults_to_dry_run() -> None:
    args = Namespace(execute_live_write=False, live_confirmation="")

    assert write_recent_actionable_deals._live_write_enabled(args) is False


def test_contact_writeback_row_guard_blocks_service_or_orphan_rows_for_live() -> None:
    service_reasons = write_amo_ready_contacts._contact_row_guard_reasons(
        {
            "Готово к записи в AMO": "Да",
            "Тип последнего свежего звонка": "existing_client_progress",
            "AMO contact IDs": "123",
        }
    )
    orphan_reasons = write_amo_ready_contacts._contact_row_guard_reasons(
        {
            "Готово к записи в AMO": "Да",
            "Тип последнего свежего звонка": "sales_call",
            "AMO contact IDs": "",
        }
    )

    assert any(reason.startswith("service_or_existing_client_context") for reason in service_reasons)
    assert "missing_amo_contact_id" in orphan_reasons


def test_contact_auto_history_skips_redundant_chronology_block() -> None:
    payload = write_amo_ready_contacts._compose_auto_history(
        {
            "Краткая история общения": "Клиент интересуется летней школой по физике и просит перезвонить.",
            "Хронология общения (последние 5 касаний)": "Клиент интересуется летней школой по физике и просит перезвонить.",
        }
    )

    assert payload.count("летней школой") == 1
    assert "Хронология общения" not in payload


def test_contact_last_summary_compacts_without_ellipsis() -> None:
    payload = write_amo_ready_contacts._compose_last_summary(
        {"Краткая история общения": " ".join(["Клиент интересуется летним лагерем"] * 80)}
    )

    assert "..." not in payload
    assert "…" not in payload
    assert payload.endswith("[сжато]")
    assert len(payload) <= write_amo_ready_contacts.MAX_LAST_SUMMARY_CHARS


def test_contact_last_summary_compacts_fresh_summary_to_amo_text_capacity() -> None:
    payload = write_amo_ready_contacts._compose_last_summary(
        {"Краткое резюме последнего свежего звонка": " ".join(["Клиент подробно обсуждает оплату"] * 90)}
    )

    assert "..." not in payload
    assert "…" not in payload
    assert payload.endswith("[сжато]")
    assert len(payload) <= write_amo_ready_contacts.MAX_LAST_SUMMARY_CHARS


def test_contact_payload_compacts_short_text_fields_for_amo_readback() -> None:
    payload = write_amo_ready_contacts._build_contact_payload(
        {
            "Статус матчинга Tallanto": "exact_phone_single",
            "Приоритет лида": "warm",
            "Следующий шаг": " ".join(["Отправить подробные материалы и согласовать дату следующего звонка"] * 20),
            "Краткое резюме последнего свежего звонка": " ".join(["Клиент подробно обсуждает летний лагерь"] * 30),
            "Краткая история общения": "Полная история должна оставаться в длинном textarea-поле. " * 20,
        }
    )

    assert len(payload["AI-рекомендованный следующий шаг"]) <= write_amo_ready_contacts.MAX_NEXT_STEP_CHARS
    assert len(payload["AI-рекомендованный следующий шаг"]) > write_amo_ready_contacts.MAX_AMO_TEXT_FIELD_CHARS
    assert len(payload["Последняя AI-сводка"]) <= write_amo_ready_contacts.MAX_LAST_SUMMARY_CHARS
    assert len(payload["Последняя AI-сводка"]) > write_amo_ready_contacts.MAX_AMO_TEXT_FIELD_CHARS
    assert len(payload["Авто история общения"]) > write_amo_ready_contacts.MAX_AMO_TEXT_FIELD_CHARS


def test_contact_auto_history_uses_compact_chronology_marker_without_ellipsis() -> None:
    payload = write_amo_ready_contacts._compose_auto_history(
        {
            "Краткая история общения": "Клиент интересуется летним лагерем. " * 20,
            "Хронология общения (последние 5 касаний)": "01.05.2026: подробная строка хронологии. " * 40,
            "Следующий шаг": "Отправить материалы",
        }
    )

    assert "..." not in payload
    assert "Хронология: есть в полной рабочей таблице" in payload


def test_contact_writeback_runtime_db_preflight_reports_error_before_row_loop() -> None:
    ok, error = write_amo_ready_contacts._preflight_runtime_db(BrokenPreflightSession())

    assert ok is False
    assert "db tunnel down" in error


def test_contact_writeback_blocks_found_contact_id_that_differs_from_source() -> None:
    row = {"AMO contact IDs": "76005692"}

    assert (
        write_amo_ready_contacts._contact_id_mismatch_reason(row, 76005708)
        == "contact_id_mismatch_with_source_amo_contact_ids"
    )


def test_contact_field_catalog_guard_blocks_api_only_textarea_targets() -> None:
    reasons = write_amo_ready_contacts._contact_field_catalog_guard_reasons(
        [
            {"id": 1, "name": "Статус матчинга", "type": "text", "group_id": "contacts", "is_api_only": False},
            {"id": 2, "name": "AI-приоритет", "type": "text", "group_id": "contacts", "is_api_only": False},
            {
                "id": 3,
                "name": "AI-рекомендованный следующий шаг",
                "type": "textarea",
                "group_id": None,
                "is_api_only": True,
            },
            {
                "id": 4,
                "name": "Последняя AI-сводка",
                "type": "textarea",
                "group_id": "contacts",
                "is_api_only": True,
            },
            {"id": 5, "name": "Авто история общения", "type": "textarea", "group_id": "contacts", "is_api_only": False},
        ]
    )

    assert "contact_field_api_only_not_supported:AI-рекомендованный следующий шаг" in reasons
    assert "contact_field_missing_group:AI-рекомендованный следующий шаг" in reasons
    assert "contact_field_api_only_not_supported:Последняя AI-сводка" in reasons


def test_contact_field_catalog_guard_allows_regular_textarea_targets() -> None:
    reasons = write_amo_ready_contacts._contact_field_catalog_guard_reasons(
        [
            {"id": 1, "name": "Статус матчинга", "type": "text", "group_id": "contacts", "is_api_only": False},
            {"id": 2, "name": "AI-приоритет", "type": "text", "group_id": "contacts", "is_api_only": False},
            {
                "id": 3,
                "name": "AI-рекомендованный следующий шаг",
                "type": "textarea",
                "group_id": "contacts",
                "is_api_only": False,
            },
            {
                "id": 4,
                "name": "Последняя AI-сводка",
                "type": "textarea",
                "group_id": "contacts",
                "is_api_only": False,
            },
            {"id": 5, "name": "Авто история общения", "type": "textarea", "group_id": "contacts", "is_api_only": False},
        ]
    )

    assert reasons == []


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


def _quality_gate_summary_fixture(root: Path) -> str:
    path = root / "summary.json"
    path.write_text(
        json.dumps({"passed": True, "readiness": {"crm_quality_writeback_ready": True}}),
        encoding="utf-8",
    )
    return str(path)


def _crm_writeback_quality_summary_fixture(root: Path) -> str:
    path = root / "crm_writeback_summary.json"
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "population_recall": {"passed_for_live": True},
                "crm_text_quality": {"passed_for_live": True, "blocking_rows": 0},
            }
        ),
        encoding="utf-8",
    )
    return str(path)
