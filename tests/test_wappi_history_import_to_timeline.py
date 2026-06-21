from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pytest

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    OpportunityType,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.wappi_history_import import (
    WappiFetchLimits,
    WappiHistoryImportConfig,
    WappiHistoryTimelineNormalizer,
    WappiPairCustomerResolver,
    assert_readonly_wappi_client,
    run_wappi_history_import,
)
from mango_mvp.integrations.amo_wappi_phase1 import WappiClientConfig, WappiPhase1Client
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport
from mango_mvp.integrations.draft_loop import DraftLoopKey


def test_wappi_history_import_resolves_by_amo_pair_and_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    phase1 = write_phase1_config(tmp_path)
    pairs = write_pairs(tmp_path, lead_id="1001", contact_id="2002")
    client = FakeWappiClient(
        {
            "p-tg": [{"id": "chat-1", "type": "user"}],
            "p-max": [],
        },
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Здравствуйте, нужен курс", "time": 1_753_000_000},
            ]
        },
    )

    config = WappiHistoryImportConfig(
        timeline_db=db_path,
        allowed_root=tmp_path,
        phase1_config=phase1,
        pairs_file=pairs,
        auto_pairs_file=None,
        apply=True,
        limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
    )
    first = run_wappi_history_import(config, client=client)
    second = run_wappi_history_import(config, client=client)

    assert first["validation_ok"] is True
    assert first["summary"]["linked_by_pair"] == 1
    assert first["summary"]["pending_attribution"] == 0
    assert first["profiles"]["p-tg"]["brand"] == "foton"
    assert first["profiles"]["p-tg"]["source_system"] == "wappi_telegram"
    assert second["writes"]["status_counts"]["duplicate"] >= first["writes"]["status_counts"]["created"]

    event = fetch_one_json(db_path, "timeline_events")
    chunk = fetch_one_json(db_path, "bot_context_chunks")
    link = fetch_one_json(db_path, "identity_links", "source_system = 'wappi_telegram'")
    assert event["customer_id"] == customer_id
    assert event["source_system"] == "wappi_telegram"
    assert event["event_type"] == "telegram_message"
    assert event["record"]["message"]["allowed_for_bot"] is False
    assert event["metadata"]["brand"] == "foton"
    assert chunk["allowed_for_bot"] is False
    assert chunk["requires_manager_review"] is True
    assert link["link_type"] == "channel_session_id"
    assert link["link_value"] == "wappi_telegram:p-tg:chat-1"


def test_wappi_history_import_pending_does_not_create_customer_or_event(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    pairs = write_pairs(tmp_path, lead_id="missing-lead", contact_id="")
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Цена?", "time": 1_753_000_000},
            ]
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=pairs,
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
        ),
        client=client,
    )

    assert report["validation_ok"] is True
    assert report["summary"]["linked_by_pair"] == 0
    assert report["summary"]["pending_attribution"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM timeline_conflicts WHERE conflict_type='pending_attribution'").fetchone()[0] == 1


def test_wappi_resolver_fails_closed_on_lead_contact_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, customer_id="customer:lead", lead_id="1001", contact_id="")
    seed_customer_with_amo(db_path, tmp_path, customer_id="customer:contact", lead_id="", contact_id="2002")
    pairs = write_pairs(tmp_path, lead_id="1001", contact_id="2002")
    from mango_mvp.integrations.draft_loop import load_pairs_file

    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs=load_pairs_file(pairs))
    resolution = resolver.resolve(profile=profile("p-tg", "foton", "telegram"), chat_id="chat-1")

    assert resolution.resolved is False
    assert resolution.status == "pending_attribution"
    assert resolution.reason == "pair_matches_multiple_or_conflicting_customers"


def test_wappi_normalizer_rejects_allowed_for_bot_true() -> None:
    with pytest.raises(ValueError, match="allowed_for_bot=False"):
        WappiHistoryTimelineNormalizer(tenant_id="foton", source_system="wappi_telegram").normalize(
            source_record(
                {
                    "source_system": "wappi_telegram",
                    "source_ref": "wappi_telegram:p-tg:chat-1:m-1",
                    "channel": "telegram",
                    "brand": "foton",
                    "profile_id": "p-tg",
                    "chat_id": "chat-1",
                    "message_id": "m-1",
                    "message_sha256": "a" * 64,
                    "timeline_source_id": "p-tg:chat-1:m-1",
                    "event_at": "2026-06-21T10:00:00+00:00",
                    "text": "Здравствуйте",
                    "allowed_for_bot": True,
                    "resolved_customer_id": "customer:known",
                    "resolution_status": "resolved",
                }
            )
        )


def test_wappi_history_requires_default_deny_transport() -> None:
    unsafe = WappiPhase1Client(WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"), transport=None)
    with pytest.raises(RuntimeError, match="DefaultDenyTransport"):
        assert_readonly_wappi_client(unsafe)

    safe = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=DefaultDenyTransport(lambda **_kwargs: {"ok": True}),
    )
    assert_readonly_wappi_client(safe)


def test_wappi_history_pages_with_limit_100_and_mark_all_false(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    phase1 = write_phase1_config(tmp_path)
    pairs = write_pairs(tmp_path, lead_id="1001", contact_id="2002")
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1", "type": "user"}], "p-max": []},
        {("telegram", "p-tg", "chat-1"): [{"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Текст", "time": 1_753_000_000}]},
    )

    run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=pairs,
            auto_pairs_file=None,
            apply=False,
            limits=WappiFetchLimits(chat_limit_per_profile=250, messages_per_chat=250, page_size=250, message_limit_total=5, sleep_seconds=0),
        ),
        client=client,
    )

    assert client.calls
    assert all(call["method"] == "GET" for call in client.calls)
    assert all(1 <= call["limit"] <= 100 for call in client.calls)
    message_calls = [call for call in client.calls if call["kind"] == "messages"]
    assert message_calls
    assert all(call["mark_all"] is False for call in message_calls)


class FakeWappiClient:
    def __init__(self, chats: Mapping[str, list[Mapping[str, Any]]], messages: Mapping[tuple[str, str, str], list[Mapping[str, Any]]]) -> None:
        self.transport = DefaultDenyTransport(lambda **_kwargs: {"ok": True})
        self.chats = {key: list(value) for key, value in chats.items()}
        self.messages = {key: list(value) for key, value in messages.items()}
        self.calls: list[dict[str, Any]] = []

    def list_chats(self, *, channel: str, profile_id: str, limit: int = 50, offset: int = 0, order: str = "desc", show_all: bool = False) -> Mapping[str, Any]:
        self.calls.append({"kind": "chats", "method": "GET", "channel": channel, "profile_id": profile_id, "limit": limit, "offset": offset, "show_all": show_all})
        items = self.chats.get(profile_id, [])
        return {"dialogs": items[offset : offset + limit]}

    def get_chat_messages(
        self,
        *,
        channel: str,
        profile_id: str,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        mark_all: bool = False,
    ) -> Mapping[str, Any]:
        self.calls.append({"kind": "messages", "method": "GET", "channel": channel, "profile_id": profile_id, "chat_id": chat_id, "limit": limit, "offset": offset, "mark_all": mark_all})
        items = self.messages.get((channel, profile_id, chat_id), [])
        return {"messages": items[offset : offset + limit]}


def write_phase1_config(tmp_path: Path) -> Path:
    path = tmp_path / "amo_wappi_phase1.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "p-tg": {"brand": "foton", "channel": "telegram", "label": "Foton Telegram"},
                    "p-max": {"brand": "unpk", "channel": "max", "label": "UNPK Max"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def write_pairs(tmp_path: Path, *, lead_id: str, contact_id: str) -> Path:
    path = tmp_path / "draft_loop_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "profile_id": "p-tg",
                    "chat_id": "chat-1",
                    "lead_id": lead_id,
                    "contact_id": contact_id,
                    "expected_brand": "foton",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def seed_customer_with_amo(
    db_path: Path,
    allowed_root: Path,
    *,
    customer_id: str = "customer:known",
    lead_id: str = "1001",
    contact_id: str = "2002",
) -> str:
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root)
    customer = CustomerIdentity(
        tenant_id="foton",
        customer_id=customer_id,
        identity_status=IdentityStatus.STRONG,
        source_ref=f"synthetic:{customer_id}",
    )
    store.upsert_customer(customer, actor="test")
    if lead_id:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_lead_id",
                link_value=lead_id,
                source_system="amocrm_snapshot",
                source_ref=f"amocrm:lead:{lead_id}",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
            ),
            actor="test",
        )
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id=customer.customer_id,
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amocrm_snapshot",
                source_id=lead_id,
                title="Synthetic deal",
                status="open",
                confidence=1.0,
            ),
            actor="test",
        )
    if contact_id:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_contact_id",
                link_value=contact_id,
                source_system="amocrm_snapshot",
                source_ref=f"amocrm:contact:{contact_id}",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
            ),
            actor="test",
        )
    store.close()
    return customer.customer_id


def source_record(payload: Mapping[str, Any]):
    from mango_mvp.customer_timeline.ingestion import TimelineSourceRecord

    return TimelineSourceRecord(
        source_system=str(payload["source_system"]),
        source_ref=str(payload["source_ref"]),
        payload=payload,
    )


def profile(profile_id: str, brand: str, channel: str):
    from mango_mvp.customer_timeline.wappi_history_import import WappiProfileSpec

    return WappiProfileSpec(profile_id=profile_id, brand=brand, channel=channel)


def fetch_one_json(db_path: Path, table: str, where: str = "1=1") -> dict[str, Any]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(f"SELECT record_json FROM {table} WHERE {where} LIMIT 1").fetchone()
    assert row is not None
    return json.loads(row["record_json"])
