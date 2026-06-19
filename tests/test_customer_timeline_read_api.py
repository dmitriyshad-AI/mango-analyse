from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    CustomerTimelineReadApi,
    CustomerTimelineReadApiConfig,
    CustomerTimelineSQLiteStore,
    DerivedSignal,
    EventArtifact,
    IdentityLink,
    TimelineEvent,
    build_customer_timeline_read_report,
    route_customer_timeline_request,
)
from mango_mvp.customer_timeline.read_api import main


NOW = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
SHA = "d" * 64


def test_read_api_profile_projects_safe_customer_timeline(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)

    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        health = api.health()
        profile = api.customer_profile("foton", customer_id, event_limit=10)
        raw_text = json.dumps(profile, ensure_ascii=False)

        assert api.store.read_only is True
        assert api.store._con.execute("PRAGMA query_only").fetchone()[0] == 1
        assert health["read_only"] is True
        assert profile["found"] is True
        assert profile["snapshot_as_of"] == (NOW + timedelta(minutes=1)).isoformat()
        assert profile["last_event_at"] == (NOW + timedelta(minutes=1)).isoformat()
        assert profile["customer"]["primary_phone"] == "+***4567"
        assert profile["customer"]["primary_email"] == "p***@example.com"
        assert profile["customer_id_mappings"] == [
            {
                "mapping_id": profile["customer_id_mappings"][0]["mapping_id"],
                "tenant_id": "foton",
                "old_customer_id": "customer:legacy-phone",
                "new_customer_id": customer_id,
                "mapping_kind": "alias",
                "resolution_status": "active",
                "reason": "family_phone_ambiguous",
                "source_refs": ["fixture:legacy"],
                "created_at": profile["customer_id_mappings"][0]["created_at"],
                "updated_at": profile["customer_id_mappings"][0]["updated_at"],
            }
        ]
        assert profile["timeline"]["items"][0]["allowed_for_bot"] is False
        assert profile["timeline"]["items"][0]["requires_manager_review"] is True
        assert profile["timeline"]["items"][0]["artifacts"][0]["has_path"] is True
        assert profile["timeline"]["items"][0]["signals"][0]["allowed_for_bot"] is False
        assert profile["timeline"]["items"][0]["signals"][0]["requires_manager_review"] is True
        assert profile["signals"][0]["allowed_for_bot"] is False
        assert profile["signals"][0]["requires_manager_review"] is True
        assert {item["allowed_for_bot"] for item in profile["bot_context"]["items"]} == {False, True}
        assert {item["requires_manager_review"] for item in profile["bot_context"]["items"]} == {False, True}
        assert "path" not in profile["timeline"]["items"][0]["artifacts"][0]
        assert profile["readiness"]["bot_allowed_chunks"] == 1
        assert profile["readiness"]["bot_review_required_chunks"] == 1
        assert profile["readiness"]["open_conflicts"] == 1
        assert profile["readiness"]["safe_for_automatic_bot"] is False
        assert "provider_raw_payload" not in raw_text
        assert "record_json" not in raw_text
        assert "/not/read/transcript.json" not in raw_text
        assert "hidden" not in raw_text


def test_read_api_lists_customers_paginates_searches_and_filters_bot_context(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        filtered = api.list_customers("foton", q="Иванова", limit=1)
        first_page = api.list_customers("foton", limit=1)
        second_page = api.list_customers("foton", limit=1, cursor=first_page["next_cursor"])
        allowed_context = api.bot_context("foton", customer_id, allowed_only=True)
        all_context = api.bot_context("foton", customer_id, allowed_only=False)
        search = api.search("foton", "стоимость", customer_id=customer_id, scopes=("events", "bot_context", "signals"))
        blocked_context_search = api.search(
            "foton",
            "проверки",
            customer_id=customer_id,
            scopes=("bot_context",),
            allowed_for_bot=False,
        )
        bot_safe_context_search = api.search(
            "foton",
            "проверки",
            customer_id=customer_id,
            scopes=("bot_context",),
            allowed_for_bot=True,
        )
        timeline = api.customer_timeline("foton", customer_id, event_types=("mango_call",), source_systems=("mango",), limit=5)

    assert filtered["items"][0]["customer_id"] == customer_id
    assert first_page["next_cursor"] == "1"
    assert second_page["items"]
    assert allowed_context["summary"]["visible_chunks"] == 1
    assert allowed_context["summary"]["review_required_chunks"] == 1
    assert allowed_context["items"][0]["customer_id"] is None
    assert all_context["summary"]["visible_chunks"] == 2
    assert search["result"]["items"]
    assert len(blocked_context_search["result"]["items"]) == 1
    assert blocked_context_search["result"]["items"][0]["record"]["allowed_for_bot"] is False
    assert blocked_context_search["result"]["items"][0]["record"]["requires_manager_review"] is True
    assert bot_safe_context_search["result"]["items"] == []
    assert search["result"]["items"][0]["record"]
    assert "raw_payload" not in json.dumps(search, ensure_ascii=False)
    assert timeline["items"][0]["event_type"] == "mango_call"


def test_read_api_routes_are_get_only_and_report_is_deterministic(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    config = CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    fixed_time = datetime(2026, 5, 12, 15, 0, tzinfo=timezone.utc)
    out = tmp_path / "reports" / "timeline_read.json"

    first = build_customer_timeline_read_report(
        config=config,
        tenant_id="foton",
        customer_id=customer_id,
        query="стоимость",
        limit=10,
        out_path=out,
        generated_at=fixed_time,
    )
    second = build_customer_timeline_read_report(
        config=config,
        tenant_id="foton",
        customer_id=customer_id,
        query="стоимость",
        limit=10,
        generated_at=fixed_time,
    )
    with CustomerTimelineReadApi.open(config) as api:
        route_status, route_payload = route_customer_timeline_request(api, "GET", f"/customer?tenant_id=foton&customer_id={customer_id}")
        blocked_status, blocked_payload = route_customer_timeline_request(api, "POST", "/customer")
        not_found_status, _ = route_customer_timeline_request(api, "GET", "/unknown")

    assert first == second
    assert json.loads(out.read_text(encoding="utf-8")) == first
    assert first["validation_ok"] is True
    assert route_status == 200
    assert route_payload["found"] is True
    assert blocked_status == 405
    assert blocked_payload["read_only"] is True
    assert not_found_status == 404


def test_read_api_path_guards_and_missing_db_do_not_create_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "customer_timeline.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=missing, allowed_root=tmp_path))
    assert not missing.exists()
    assert not missing.parent.exists()

    stable = tmp_path / "stable_runtime" / "customer_timeline.sqlite"
    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineReadApiConfig(timeline_db=stable, allowed_root=tmp_path)

    with pytest.raises(ValueError, match="runtime-looking"):
        CustomerTimelineReadApiConfig(timeline_db=tmp_path / "mango_product_appliance.sqlite", allowed_root=tmp_path)

    with pytest.raises(ValueError, match="allowed root"):
        CustomerTimelineReadApiConfig(timeline_db=tmp_path.parent / "outside_customer_timeline.sqlite", allowed_root=tmp_path)


def test_read_api_cli_and_no_network_or_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read API must not use network/subprocess")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(os, "system", fail)
    monkeypatch.setattr(socket, "socket", fail)
    db_path, customer_id = seed_timeline_db(tmp_path)
    out = tmp_path / "read_report.json"

    rc = main(
        [
            "--tenant-id",
            "foton",
            "--timeline-db",
            str(db_path),
            "--allowed-root",
            str(tmp_path),
            "--customer-id",
            customer_id,
            "--query",
            "стоимость",
            "--out",
            str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["validation_ok"] is True
    assert report["safety"]["network_calls"] is False
    assert report["safety"]["subprocess_calls"] is False
    assert report["safety"]["write_product_timeline_db"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["run_ra"] is False


def seed_timeline_db(tmp_path: Path) -> tuple[Path, str]:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    customer = CustomerIdentity(
        tenant_id="foton",
        identity_status="strong",
        display_name="Иванова Мария",
        primary_phone="+79161234567",
        primary_email="parent@example.com",
        first_seen_at=NOW,
        last_seen_at=NOW + timedelta(minutes=2),
        touch_count=2,
        summary={"source_system": "test_fixture"},
        metadata={"raw_payload": {"hidden": True}},
        created_at=NOW,
        updated_at=NOW + timedelta(minutes=2),
    )
    store.upsert_customer(customer)
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            identity_status="partial",
            display_name="Сидоров Петр",
            primary_phone="+79160000000",
            first_seen_at=NOW - timedelta(days=1),
            last_seen_at=NOW - timedelta(days=1),
            touch_count=1,
            created_at=NOW - timedelta(days=1),
            updated_at=NOW - timedelta(days=1),
        )
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer.customer_id,
            link_type="phone",
            link_value="+79161234567",
            source_system="tallanto_snapshot",
            source_ref="students.csv#1",
            match_class="strong_unique",
            confidence=0.95,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )
    store.record_customer_id_mapping(
        "foton",
        old_customer_id="customer:legacy-phone",
        new_customer_id=customer.customer_id,
        reason="family_phone_ambiguous",
        source_refs=("fixture:legacy",),
        actor="test",
    )
    opportunity = CustomerOpportunity(
        tenant_id="foton",
        customer_id=customer.customer_id,
        opportunity_type="amo_deal",
        source_system="amocrm_snapshot",
        source_id="lead-1",
        title="ЕГЭ математика",
        status="open",
        opened_at=NOW,
        confidence=0.8,
        evidence={"raw_payload": {"hidden": True}},
    )
    store.upsert_opportunity(opportunity)
    event = TimelineEvent(
        tenant_id="foton",
        customer_id=customer.customer_id,
        opportunity_id=opportunity.opportunity_id,
        event_type="mango_call",
        event_at=NOW + timedelta(minutes=1),
        source_system="mango",
        source_id="call-1",
        direction="inbound",
        actor_name="Клиент",
        actor_ref="client-phone-1234567",
        subject="Вопрос про стоимость",
        text_preview="Сколько стоит подготовка к ЕГЭ?",
        summary="Клиент спросил стоимость курса и попросил перезвонить.",
        importance=3,
        match_status="strong_unique",
        confidence=0.9,
        record={"raw_payload": {"hidden": True}, "audio_path": "/secret/audio.mp3"},
        metadata={"provider_raw_payload": {"hidden": True}},
        created_at=NOW + timedelta(minutes=1),
    )
    store.upsert_event(event)
    store.upsert_artifact(
        EventArtifact(
            tenant_id="foton",
            event_id=event.event_id,
            artifact_type="call_transcript_json",
            path="/not/read/transcript.json",
            sha256=SHA,
            size_bytes=128,
            mime_type="application/json",
            source_system="processing_export",
            source_ref="call-1",
            extraction_status="extracted",
            created_at=NOW + timedelta(minutes=1),
        )
    )
    store.upsert_signal(
        DerivedSignal(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_id=event.event_id,
            source_event_ids=(event.event_id,),
            signal_type="price_interest",
            severity="high",
            evidence_text="Клиент явно спросил стоимость.",
            confidence=0.88,
            recommended_action="Перезвонить",
            requires_manager_review=True,
            metadata={"attachment_bytes": "hidden"},
            created_at=NOW + timedelta(minutes=1),
        )
    )
    store.upsert_bot_context_chunk(
        BotContextChunk(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_id=event.event_id,
            source_system="mango",
            source_ref="call-1",
            chunk_type="sales_context",
            text="Клиент спрашивал стоимость и ждет звонок менеджера.",
            summary="Интерес к цене",
            event_at=NOW + timedelta(minutes=1),
            freshness_score=0.9,
            relevance_tags=("sales", "price"),
            allowed_for_bot=True,
            requires_manager_review=False,
            metadata={"raw_file": "hidden"},
            created_at=NOW + timedelta(minutes=1),
        )
    )
    store.upsert_bot_context_chunk(
        BotContextChunk(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_id=event.event_id,
            source_system="mango",
            source_ref="call-1-review",
            chunk_type="manager_review_context",
            text="Этот фрагмент требует проверки менеджера.",
            summary="Нужна проверка",
            event_at=NOW,
            freshness_score=0.2,
            relevance_tags=("review",),
            allowed_for_bot=False,
            requires_manager_review=True,
            created_at=NOW,
        )
    )
    store.record_conflict(
        "foton",
        conflict_type="ambiguous_identity",
        entity_refs=("phone:+79161234567", customer.customer_id, "customer:other"),
        actor="test",
    )
    store.close()
    return db_path, customer.customer_id
