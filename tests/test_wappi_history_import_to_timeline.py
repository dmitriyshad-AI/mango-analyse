from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
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
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.wappi_history_import import (
    WappiFetchLimits,
    WappiChatResolution,
    WappiHistoryImportConfig,
    WappiHistoryTimelineNormalizer,
    WappiPairCustomerResolver,
    WappiProfileSpec,
    _is_exact_authority_override,
    assert_readonly_wappi_client,
    _build_safe_amo_talk_client,
    collect_wappi_widget_links,
    close_resolved_wappi_pending_conflicts,
    confirm_wappi_widget_candidates_from_amo_talks,
    enrich_wappi_widget_links_from_timeline_amo_events,
    hydrate_wappi_widget_contacts,
    is_personal_wappi_dialog,
    load_existing_wappi_event_customers,
    load_existing_unmatched_wappi_records,
    load_wappi_widget_links,
    open_readonly_sqlite,
    remove_orphaned_provisional_customers,
    run_wappi_history_import,
    safe_wappi_exception,
    sanitize_wappi_import_error,
    timeline_db_identity,
    wappi_dialog_identity_keys,
    wappi_message_to_record,
    write_json_report,
)
from mango_mvp.integrations.amo_wappi_phase1 import AmoWappiHttpError
from mango_mvp.integrations.amo_wappi_auto_resolver import AmoAutoResolver
from mango_mvp.integrations.amo_wappi_phase1 import WappiClientConfig, WappiPhase1Client
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy
from mango_mvp.integrations.draft_loop import DraftLoopKey, DraftLoopPair, WappiHistoryMessage


@pytest.mark.parametrize("flag", ("IsBot", "IsDeleted", "IsFake", "IsSelf", "IsSupport"))
def test_wappi_telegram_system_accounts_are_not_personal(flag: str) -> None:
    profile = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")

    assert is_personal_wappi_dialog(profile, {"id": "1", "type": "user", "user": {flag: True}}) is False
    assert is_personal_wappi_dialog(profile, {"id": "1", "type": "user", "user": {flag: False}}) is True


def test_wappi_max_requires_one_non_bot_peer_when_participants_are_present() -> None:
    profile = WappiProfileSpec(profile_id="p-max", brand="foton", channel="max")

    assert is_personal_wappi_dialog(
        profile,
        {"id": "room", "type": "DIALOG", "participants": [{"is_me": True}, {"is_me": False}]},
    ) is True
    assert is_personal_wappi_dialog(
        profile,
        {"id": "room", "type": "DIALOG", "participants": [{"is_me": True}, {"is_me": False, "is_bot": True}]},
    ) is False
    assert is_personal_wappi_dialog(
        profile,
        {"id": "room", "type": "DIALOG", "participants": [{"is_me": True}]},
    ) is False


@pytest.mark.parametrize("authority", ("wappi_amo_widget", "amo_talk_authoritative"))
def test_exact_amo_authorities_override_older_non_exact_owner(authority: str) -> None:
    assert _is_exact_authority_override("customer:old", "timeline_identity", "customer:real", authority) is True
    assert _is_exact_authority_override("customer:old", "wappi_amo_widget", "customer:real", authority) is False


def test_wappi_history_import_resolves_by_widget_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    phase1 = write_phase1_config(tmp_path)
    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-id")
    client = FakeWidgetWappiClient(
        {
            "p-tg": [{"id": "123456", "type": "user"}],
            "p-max": [],
        },
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "m-1", "chat_id": "123456", "type": "text", "body": "Здравствуйте, нужен курс", "time": 1_753_000_000},
                {"id": "m-2", "chat_id": "123456", "type": "text", "body": "Подскажите расписание", "time": 1_753_000_100},
            ]
        },
        {("telegram", "123456"): {"contact": {"id": 2002}, "leads": [{"id": 1001}]}},
    )
    hydrate_calls: list[Mapping[str, Any]] = []

    def fake_hydrate(**kwargs: Any) -> Mapping[str, Any]:
        hydrate_calls.append(kwargs)
        if len(hydrate_calls) == 1:
            with sqlite3.connect(kwargs["timeline_db"]) as con:
                con.execute("CREATE TABLE internal_hydrate_marker (id INTEGER PRIMARY KEY)")
        return {"requested": 1 if len(hydrate_calls) == 1 else 0, "fetched": 1, "fetch_errors": 0}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.wappi_history_import.hydrate_wappi_widget_contacts",
        fake_hydrate,
    )

    config = WappiHistoryImportConfig(
        timeline_db=db_path,
        allowed_root=tmp_path,
        phase1_config=phase1,
        pairs_file=None,
        auto_pairs_file=None,
        apply=True,
        widget_link_db=tmp_path / "wappi_amo_links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
    )
    first = run_wappi_history_import(config, client=client)
    second = run_wappi_history_import(config, client=client)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.wappi_history_import.collect_wappi_widget_links",
        lambda **_kwargs: pytest.fail("completed widget map must be reused without collection"),
    )
    reused = run_wappi_history_import(replace(config, refresh_widget_links=False), client=client)

    assert first["validation_ok"] is True
    assert first["summary"]["linked_by_amo_widget"] == 2
    assert first["summary"]["pending_attribution"] == 0
    assert first["summary"]["amo_widget_contact_hydrate"]["requested"] == 1
    assert second["summary"]["amo_widget_contact_hydrate"]["requested"] == 0
    assert reused["summary"]["amo_widget_link_map"]["reused"] is True
    assert len(hydrate_calls) == 3
    assert first["profiles"]["p-tg"]["widget_resolved_chats"] == 1
    assert first["profiles"]["p-tg"]["brand"] == "foton"
    assert first["profiles"]["p-tg"]["source_system"] == "wappi_telegram"
    assert first["provenance"]["input_hashes"]["phase1_config"]
    assert "tracked_diff_sha256" in first["provenance"]["worktree"]
    assert second["writes"]["status_counts"]["duplicate"] >= first["writes"]["status_counts"]["created"]
    assert second["writes"]["status_counts"].get("updated", 0) == 0
    assert reused["writes"]["status_counts"].get("updated", 0) == 0
    assert first["writes"]["import_groups_single_transaction"] is True
    assert first["writes"]["post_import_cleanup_same_transaction"] is False
    assert first["writes"]["all_db_mutations_single_transaction"] is False

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
    assert link["link_value"] == "wappi_telegram:p-tg:123456"


@pytest.mark.parametrize("timestamp", (0, -1, float("inf"), -(10**100)))
def test_wappi_invalid_timestamp_is_deterministic_epoch(timestamp: object) -> None:
    profile = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    message = WappiHistoryMessage(
        profile_id="p-tg",
        chat_id="chat-1",
        message_id="message-1",
        text="Тест",
        message_type="text",
        timestamp=timestamp,  # type: ignore[arg-type]
        from_me=False,
    )
    resolution = WappiChatResolution(status="resolved", customer_id="customer:1")

    first = wappi_message_to_record(profile=profile, message=message, resolution=resolution)
    second = wappi_message_to_record(profile=profile, message=message, resolution=resolution)
    batch = WappiHistoryTimelineNormalizer(tenant_id="foton", source_system="wappi_telegram").normalize(first)

    assert first.to_json_dict() == second.to_json_dict()
    assert first.payload["event_at"] == "1970-01-01T00:00:00+00:00"
    assert first.payload["event_time_status"] == "invalid_epoch"
    assert batch.events[0].metadata["event_time_status"] == "invalid_epoch"
    assert batch.identity_links[0].first_seen_at is None
    assert batch.identity_links[0].last_seen_at is None


def test_wappi_valid_timestamp_is_preserved() -> None:
    profile = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    message = WappiHistoryMessage(
        profile_id="p-tg",
        chat_id="chat-1",
        message_id="message-1",
        text="Тест",
        message_type="text",
        timestamp=1_753_000_000,
        from_me=False,
    )
    record = wappi_message_to_record(
        profile=profile,
        message=message,
        resolution=WappiChatResolution(status="pending_attribution"),
    )
    assert record.payload["event_at"] == "2025-07-20T08:26:40+00:00"
    assert record.payload["event_time_status"] == "source_valid"


def test_close_resolved_wappi_conflicts_scans_once_and_leaves_neighbor_open(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    base = {
        "tenant_id": "foton",
        "conflict_type": "pending_attribution",
        "severity": "low",
        "status": "open",
        "created_at": "2026-07-24T00:00:00+00:00",
    }
    with sqlite3.connect(db_path) as con:
        for message_id in ("message-1", "message-2"):
            payload = {
                **base,
                "conflict_id": f"conflict:{message_id}",
                "metadata": {
                    "source_system": "wappi_telegram",
                    "profile_id": "p-tg",
                    "chat_id": "chat-1",
                    "message_id": message_id,
                },
            }
            con.execute(
                """
                INSERT INTO timeline_conflicts
                (conflict_id, tenant_id, conflict_type, severity, status, created_at, resolved_at, record_hash, record_json)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    payload["conflict_id"],
                    payload["tenant_id"],
                    payload["conflict_type"],
                    payload["severity"],
                    payload["status"],
                    payload["created_at"],
                    "test-hash",
                    json.dumps(payload),
                ),
            )
    record = wappi_message_to_record(
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        message=WappiHistoryMessage(
            profile_id="p-tg",
            chat_id="chat-1",
            message_id="message-1",
            text="Тест",
            message_type="text",
            timestamp=1_753_000_000,
            from_me=False,
        ),
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:1",
            resolution_source="timeline_identity",
        ),
    )

    report = close_resolved_wappi_pending_conflicts(db_path, tenant_id="foton", records=(record,))

    assert report == {"resolved_pending_conflicts_closed": 1}
    with sqlite3.connect(db_path) as con:
        statuses = dict(con.execute("SELECT conflict_id, status FROM timeline_conflicts"))
    assert statuses == {"conflict:message-1": "resolved", "conflict:message-2": "open"}


def test_wappi_timeline_identity_preserves_evidence_and_is_not_manual() -> None:
    profile_spec = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    message = WappiHistoryMessage(
        profile_id="p-tg",
        chat_id="123456",
        message_id="message-1",
        text="Тест",
        message_type="text",
        timestamp=1_753_000_000,
        from_me=False,
    )
    record = wappi_message_to_record(
        profile=profile_spec,
        message=message,
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:1",
            resolution_source="timeline_identity",
            evidence={"brand_context_authorized": False, "customer_brand": "unpk"},
        ),
    )
    batch = WappiHistoryTimelineNormalizer(tenant_id="foton", source_system="wappi_telegram").normalize(record)

    assert batch.events[0].match_status == IdentityMatchClass.STRONG_UNIQUE
    assert batch.events[0].metadata["brand_context_authorized"] is False
    assert batch.identity_links[0].match_class == IdentityMatchClass.STRONG_UNIQUE
    assert batch.identity_links[0].evidence["brand_context_authorized"] is False
    assert batch.bot_context_chunks[0].metadata["brand_context_authorized"] is False


@pytest.mark.parametrize(
    ("resolution_status", "identity_authority", "error"),
    (
        ("pending_attribution", "timeline_identity", "resolution_status=resolved"),
        ("resolved", "", "requires identity_authority"),
        ("resolved", "unknown_resolver", "unsupported resolved Wappi identity_authority"),
    ),
)
def test_wappi_normalizer_rejects_untrusted_resolved_identity(
    resolution_status: str,
    identity_authority: str,
    error: str,
) -> None:
    record = wappi_message_to_record(
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        message=WappiHistoryMessage(
            profile_id="p-tg",
            chat_id="123456",
            message_id="message-1",
            text="Тест",
            message_type="text",
            timestamp=1_753_000_000,
            from_me=False,
        ),
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:1",
            resolution_source="timeline_identity",
        ),
    )
    payload = dict(record.payload)
    payload["resolution_status"] = resolution_status
    payload["identity_authority"] = identity_authority

    with pytest.raises(ValueError, match=error):
        WappiHistoryTimelineNormalizer(tenant_id="foton", source_system="wappi_telegram").normalize(
            type(record)(
                source_system=record.source_system,
                source_ref=record.source_ref,
                payload=payload,
                observed_at=record.observed_at,
            )
        )


def test_max_username_is_not_used_as_telegram_identity() -> None:
    strong, weak, reason = wappi_dialog_identity_keys(
        WappiProfileSpec(profile_id="p-max", brand="foton", channel="max"),
        "max-chat",
        {
            "phone": "+7 999 000-00-01",
            "participants": [
                {"phone": "+7 999 000-00-01", "user_id": "max-1", "username": "same_name"}
            ],
        },
    )

    assert reason == ""
    assert ("max_user_id", "max-1") in strong
    assert weak == ()


def test_max_unique_nonself_peer_id_is_strong_without_phone() -> None:
    strong, weak, reason = wappi_dialog_identity_keys(
        WappiProfileSpec(profile_id="p-max", brand="foton", channel="max"),
        "max-chat",
        {
            "participants": [
                {"user_id": "manager-1", "is_me": True, "phone": ""},
                {"user_id": "client-1", "is_me": False, "phone": ""},
            ],
        },
    )

    assert reason == ""
    assert strong == (("max_user_id", "client-1"),)
    assert weak == ()


def test_wappi_require_nonempty_profile_blocks_all_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-id")
    client = FakeWidgetWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {("telegram", "p-tg", "123456"): [{"id": "m-1", "chat_id": "123456", "type": "text", "body": "Тест", "time": 1_753_000_000}]},
        {("telegram", "123456"): {"contact": {"id": 2002}, "leads": [{"id": 1001}]}},
    )
    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=None,
            auto_pairs_file=None,
            apply=True,
            require_nonempty_profiles=True,
            limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
        ),
        client=client,
    )
    assert report["mode"] == "apply_blocked"
    assert report["summary"]["empty_profiles"] == ["p-max"]
    assert report["limit_hits"] == ["p-max:empty_profile"]
    assert report["writes"]["all_db_mutations_single_transaction"] is None
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events WHERE source_system LIKE 'wappi_%'").fetchone()[0] == 0


def test_wappi_error_serialization_removes_source_data() -> None:
    secret = "secret-chat-and-message"
    safe_error = sanitize_wappi_import_error(
        {"source_ref": secret, "error_type": "ValueError", "message": f"bad {secret}"}
    )
    safe_exception = safe_wappi_exception(RuntimeError(f"failed {secret}"))
    serialized = json.dumps({"error": safe_error, "exception": safe_exception}, ensure_ascii=False)
    assert secret not in serialized
    assert safe_error["redacted"] is True
    assert safe_exception["redacted"] is True


def test_wappi_db_identity_ignores_wal_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:wal-checkpoint",
                identity_status=IdentityStatus.STRONG,
            )
        )
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA wal_autocheckpoint=0")
        con.execute("UPDATE customer_identities SET updated_at=updated_at")
        con.commit()
        before = timeline_db_identity(db_path)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        after = timeline_db_identity(db_path)
    assert before["identity_digest"] == after["identity_digest"]


def test_wappi_db_identity_tracks_ingestion_cursor_update(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_ingestion_cursor(
            "foton",
            "wappi_telegram",
            last_cursor_ts=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
    before = timeline_db_identity(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE ingestion_cursors SET updated_at = ? WHERE tenant_id = ? AND source_system = ?",
            ("2026-07-22T12:00:00+00:00", "foton", "wappi_telegram"),
        )
        con.commit()
    after = timeline_db_identity(db_path)

    assert before["audit_seq"] == after["audit_seq"]
    assert before["cursor_updated_at"] != after["cursor_updated_at"]
    assert before["identity_digest"] != after["identity_digest"]


def test_wappi_history_apply_rolls_back_both_channels_on_second_source_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:tg",
        lead_id="1001",
        contact_id="2002",
        brand="foton",
    )
    seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:max",
        lead_id="3003",
        contact_id="4004",
        brand="unpk",
    )
    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-id")
    client = FakeWidgetWappiClient(
        {
            "p-tg": [{"id": "123456", "type": "user"}],
            "p-max": [
                {
                    "id": "max-room",
                    "type": "DIALOG",
                    "participants": [
                        {"user_id": "self", "is_me": True},
                        {"user_id": "max-person", "is_me": False},
                    ],
                }
            ],
        },
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "tg-1", "chat_id": "123456", "type": "text", "body": "Telegram", "time": 1_753_000_001}
            ],
            ("max", "p-max", "max-room"): [
                {"id": "max-1", "chat_id": "max-room", "type": "text", "body": "Max", "time": 1_753_000_002}
            ],
        },
        {
            ("telegram", "123456"): {"contact": {"id": 2002}, "leads": [{"id": 1001}]},
            ("max", "max-room"): {"contact": {"id": 4004}, "leads": [{"id": 3003}]},
        },
    )
    original_upsert_event = CustomerTimelineSQLiteStore.upsert_event

    def fail_on_max(self, event, *args, **kwargs):
        if event.source_system == "wappi_max":
            raise RuntimeError("synthetic max write failure")
        return original_upsert_event(self, event, *args, **kwargs)

    monkeypatch.setattr(CustomerTimelineSQLiteStore, "upsert_event", fail_on_max)

    with pytest.raises(RuntimeError, match="synthetic max write failure"):
        run_wappi_history_import(
            WappiHistoryImportConfig(
                timeline_db=db_path,
                allowed_root=tmp_path,
                phase1_config=write_phase1_config(tmp_path),
                pairs_file=None,
                auto_pairs_file=None,
                apply=True,
                limits=WappiFetchLimits(
                    chat_limit_per_profile=5,
                    messages_per_chat=5,
                    message_limit_total=20,
                    request_limit_total=30,
                    sleep_seconds=0,
                ),
            ),
            client=client,
        )

    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM timeline_events WHERE source_system IN ('wappi_telegram', 'wappi_max')"
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT COUNT(*) FROM identity_links WHERE source_system IN ('wappi_telegram', 'wappi_max')"
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT COUNT(*) FROM ingestion_runs WHERE source_system IN ('wappi_telegram', 'wappi_max')"
        ).fetchone()[0] == 0


def test_wappi_apply_keeps_unknown_widget_contact_pending_without_blocking_other_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-id")
    client = FakeWidgetWappiClient(
        {"p-tg": [{"id": "chat-1", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Цена?", "time": 1_753_000_000},
            ]
        },
        {("telegram", "chat-1"): {"contact": {"id": 9999}, "leads": [{"id": 8888}]}},
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
        ),
        client=client,
    )

    assert report["validation_ok"] is True
    assert report["mode"] == "apply"
    assert report["summary"]["pending_attribution"] == 1
    assert report["summary"]["amo_widget_missing_personal_chats"] == 1
    assert report["limit_hits"] == []
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM timeline_events WHERE customer_id IS NULL").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM timeline_conflicts").fetchone()[0] == 1


def test_wappi_apply_does_not_turn_optional_widget_gate_on_implicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    monkeypatch.delenv("AMO_WAPPI_CRM_ID", raising=False)
    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=write_pairs(
                tmp_path,
                lead_id="1001",
                contact_id="2002",
                chat_id="123456",
            ),
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=5,
                message_limit_total=20,
                sleep_seconds=0,
            ),
        ),
        client=FakeWappiClient(
            {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
            {
                ("telegram", "p-tg", "123456"): [
                    {"id": "m-1", "body": "Цена?", "time": 1_753_000_000}
                ]
            },
        ),
    )

    assert report["mode"] == "apply"
    assert report["provenance"]["limits"]["require_widget_linkage"] is False
    assert report["limit_hits"] == []
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 1


def test_wappi_old_unmatched_event_relinks_without_network_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    phase1 = write_phase1_config(tmp_path)
    pending_record = wappi_message_to_record(
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        message=WappiHistoryMessage(
            profile_id="p-tg",
            chat_id="123456",
            message_id="m-old",
            text="Цена?",
            message_type="text",
            timestamp=1_753_000_000,
            from_me=False,
        ),
        resolution=WappiChatResolution(status="pending_attribution"),
    )
    pending_batch = WappiHistoryTimelineNormalizer(
        tenant_id="foton",
        source_system="wappi_telegram",
    ).normalize(pending_record)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_event(pending_batch.events[0], actor="test")

    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-id")
    client = FakeWidgetWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {},
        {("telegram", "123456"): {"contact": {"id": 2002}, "leads": [{"id": 1001}]}},
    )
    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=5,
                message_limit_total=20,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["mode"] == "apply"
    assert report["summary"]["local_unmatched_relink_records"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 1
        assert con.execute("SELECT customer_id FROM timeline_events").fetchone()[0] == customer_id
        assert con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0] == 1


def test_wappi_apply_rejects_production_timeline_path(tmp_path: Path) -> None:
    prod_dir = tmp_path / "customer_timeline_prod_20260722"
    prod_dir.mkdir()
    with pytest.raises(ValueError, match="must not target a production"):
        WappiHistoryImportConfig(
            timeline_db=prod_dir / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            apply=True,
        )


def test_wappi_readonly_connection_sees_uncheckpointed_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "timeline.sqlite"
    writer = sqlite3.connect(db_path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample(value TEXT)")
        writer.commit()
        writer.execute("INSERT INTO sample VALUES ('visible')")
        writer.commit()
        assert Path(f"{db_path}-wal").stat().st_size > 0
        with open_readonly_sqlite(db_path) as reader:
            assert reader.execute("SELECT value FROM sample").fetchone()[0] == "visible"
    finally:
        writer.close()


def test_wappi_apply_blocks_widget_conflict_with_existing_chat_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, customer_id="customer:first", lead_id="1001", contact_id="2002")
    seed_customer_with_amo(db_path, tmp_path, customer_id="customer:second", lead_id="9001", contact_id="9002")
    phase1 = write_phase1_config(tmp_path)
    first_record = wappi_message_to_record(
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        message=WappiHistoryMessage(
            profile_id="p-tg",
            chat_id="123456",
            message_id="m-1",
            text="Здравствуйте",
            message_type="text",
            timestamp=1_753_000_000,
            from_me=False,
        ),
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:first",
            contact_id="2002",
            lead_id="1001",
            lead_ids=("1001",),
            resolution_source="wappi_amo_widget",
            evidence={"brand_context_authorized": True},
        ),
    )
    first_batch = WappiHistoryTimelineNormalizer(
        tenant_id="foton",
        source_system="wappi_telegram",
    ).normalize(first_record)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_event(first_batch.events[0], actor="test")
        for link in first_batch.identity_links:
            store.upsert_identity_link(link, actor="test")

    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-id")
    client = FakeWidgetWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "m-1", "chat_id": "123456", "type": "text", "body": "Здравствуйте", "time": 1_753_000_000},
            ]
        },
        {("telegram", "123456"): {"contact": {"id": 9002}, "leads": [{"id": 9001}]}},
    )
    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=5,
                message_limit_total=20,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["mode"] == "apply_blocked"
    assert report["summary"]["blocked_chat_relink_conflicts"] == 1
    assert "wappi_amo_widget:existing_customer_conflict" in report["limit_hits"]
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT record_json FROM timeline_events WHERE source_system='wappi_telegram'").fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0]["record_json"])["customer_id"] == "customer:first"


def test_wappi_superseded_exact_event_does_not_block_new_widget_owner(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:old",
        lead_id="1001",
        contact_id="2002",
    )
    new_customer = seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:new",
        lead_id="9001",
        contact_id="9002",
    )
    old_record = wappi_message_to_record(
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        message=WappiHistoryMessage(
            profile_id="p-tg",
            chat_id="123456",
            message_id="old-message",
            text="Старая привязка",
            message_type="text",
            timestamp=1_753_000_000,
            from_me=False,
        ),
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:old",
            contact_id="2002",
            lead_id="1001",
            lead_ids=("1001",),
            resolution_source="wappi_amo_widget",
        ),
    )
    batch = WappiHistoryTimelineNormalizer(
        tenant_id="foton",
        source_system="wappi_telegram",
    ).normalize(old_record)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_event(batch.events[0], actor="test")
        for link in batch.identity_links:
            store.upsert_identity_link(link, actor="test")
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE timeline_events SET superseded_by = 'replacement' WHERE event_id = ?",
            (batch.events[0].event_id,),
        )
        con.commit()

    assert load_existing_wappi_event_customers(
        db_path,
        tenant_id="foton",
        source_systems={"wappi_telegram"},
        source_ids=(batch.events[0].source_id,),
    ) == {}
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "resolved",
                "contact_id": "9002",
                "lead_ids": ("9001",),
                "resolution_source": "wappi_amo_widget",
            }
        },
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == new_customer
    assert resolution.resolution_source == "wappi_amo_widget"


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


def test_wappi_pair_resolution_requires_numeric_personal_chat_and_exact_brand(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:unpk",
        lead_id="1001",
        contact_id="2002",
        brand="unpk",
    )
    from mango_mvp.integrations.draft_loop import load_pairs_file

    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs=load_pairs_file(write_pairs(tmp_path, lead_id="1001", contact_id="2002", chat_id="123456")),
    )

    cross_brand = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )
    non_personal = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "group"},
        messages=(),
    )

    assert cross_brand.resolved is False
    assert cross_brand.reason == "draft_loop_pair_brand_mismatch"
    assert non_personal.resolved is False
    assert non_personal.reason == "draft_loop_pair_non_personal_chat"


@pytest.mark.parametrize(
    ("profile_id", "channel", "dialog", "link_type", "link_value", "brand"),
    (
        ("p-tg", "telegram", {"id": "123456", "type": "user"}, "telegram_user_id", "123456", "foton"),
        ("p-max", "max", {"id": "max-1", "phone": "+7 999 000-00-01"}, "phone", "+79990000001", "unpk"),
    ),
)
def test_wappi_pair_does_not_override_other_identity_owner(
    tmp_path: Path,
    profile_id: str,
    channel: str,
    dialog: Mapping[str, Any],
    link_type: str,
    link_value: str,
    brand: str,
) -> None:
    profile_spec = profile(profile_id, brand, channel)
    db_path = tmp_path / "customer_timeline.sqlite"
    pair_customer = seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:pair",
        lead_id="1001",
        contact_id="2002",
        brand=brand,
    )
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:identity-owner",
        link_type=link_type,
        link_value=link_value,
        brand=brand,
    )
    key = DraftLoopKey(profile_spec.profile_id, str(dialog["id"]))
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={key: DraftLoopPair(key=key, lead_id="1001", contact_id="2002", expected_brand=brand)},
    )

    resolution = resolver.resolve_chat(profile=profile_spec, dialog=dialog, messages=())

    assert resolution.resolved is False
    assert resolution.reason == "draft_loop_pair_identity_customer_conflict"
    assert pair_customer in resolution.candidate_customer_ids


def test_wappi_pair_requires_unique_support_value(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    pair_customer = seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:pair",
        lead_id="1001",
        contact_id="",
        brand="foton",
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    for customer_id in (pair_customer, "customer:other"):
        if customer_id == "customer:other":
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status=IdentityStatus.STRONG,
                    source_ref="synthetic:other",
                ),
                actor="test",
            )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type="tallanto_student_id",
                link_value="shared-student",
                source_system="tallanto_snapshot",
                source_ref=f"tallanto:{customer_id}",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
            ),
            actor="test",
        )
    store.close()
    key = DraftLoopKey("p-tg", "123456")
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={key: DraftLoopPair(key=key, lead_id="1001", expected_brand="foton")},
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "draft_loop_pair_support_missing"


def test_wappi_pair_respects_ambiguous_identity_and_phone_stoplist(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:pair",
        lead_id="1001",
        contact_id="2002",
        brand="unpk",
    )
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:pair",
        link_type="phone",
        link_value="+79990000001",
        brand="unpk",
    )
    key = DraftLoopKey("p-max", "max-1")
    pair = DraftLoopPair(key=key, lead_id="1001", contact_id="2002", expected_brand="unpk")
    stoplisted = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={key: pair},
        shared_phone_stoplist=("+79990000001",),
    ).resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={"id": "max-1", "phone": "+7 999 000-00-01"},
        messages=(),
    )

    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id="customer:second-owner",
            identity_status=IdentityStatus.STRONG,
            source_ref="synthetic:second-owner",
        ),
        actor="test",
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="customer:second-owner",
            link_type="phone",
            link_value="+79990000001",
            source_system="synthetic",
            source_ref="synthetic:second-owner:phone",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
        ),
        actor="test",
    )
    store.close()
    ambiguous = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={key: pair},
    ).resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={"id": "max-1", "phone": "+7 999 000-00-01"},
        messages=(),
    )

    assert stoplisted.reason == "shared_phone"
    assert ambiguous.reason == "draft_loop_pair_identity_ambiguous"


def test_wappi_resolver_uses_unique_timeline_identity_with_matching_brand(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:telegram",
        link_type="telegram_user_id",
        link_value="123456",
        brand="foton",
    )
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == "customer:telegram"
    assert resolution.reason == "timeline_identity_unique_brand_match"
    assert resolution.resolution_source == "timeline_identity"
    assert resolver.chat_resolutions[("wappi_telegram", "p-tg", "123456")].customer_id == "customer:telegram"


def test_wappi_resolver_rejects_noncanonical_telegram_id(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:telegram",
        link_type="telegram_user_id",
        link_value="1234",
        brand="foton",
    )
    resolution = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={}).resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "user-12x34", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "draft_loop_pair_missing"


def test_wappi_resolver_uses_max_id_only_for_matching_phone_participant(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:max",
        link_type="max_user_id",
        link_value="max-user-1",
        brand="unpk",
    )
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    unresolved = resolver.resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={"id": "chat", "phone": "+7 999 000-00-01", "participants": [{"user_id": "max-user-1"}]},
        messages=(),
    )
    resolved = resolver.resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={
            "id": "chat",
            "phone": "+7 999 000-00-01",
            "participants": [{"user_id": "max-user-1", "phone": "+7 999 000-00-01"}],
        },
        messages=(),
    )

    assert unresolved.resolved is False
    assert resolved.customer_id == "customer:max"


def test_wappi_resolver_uses_exact_telegram_user_phone(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:phone",
        link_type="phone",
        link_value="+79990000001",
        brand="foton",
    )
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={
            "id": "123456",
            "type": "user",
            "user": {"ID": 123456, "Phone": "+7 999 000-00-01"},
        },
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == "customer:phone"
    assert resolution.match_key == "phone"


def test_wappi_resolver_does_not_link_by_username_alone(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:username",
        link_type="telegram_username",
        link_value="parent_name",
        brand="foton",
    )
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user", "user": {"Username": "@Parent_Name"}},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "draft_loop_pair_missing"


def test_wappi_resolver_blocks_conflicting_exact_dialog_signals(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:telegram",
        link_type="telegram_user_id",
        link_value="123456",
        brand="foton",
    )
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:phone",
        link_type="phone",
        link_value="+79990000001",
        brand="foton",
    )
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={
            "id": "123456",
            "type": "user",
            "user": {"ID": 123456, "Phone": "+7 999 000-00-01"},
        },
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "timeline_identity_signal_conflict"


def test_wappi_resolver_vetoes_ambiguous_link_even_with_one_strong_owner(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:strong",
        link_type="phone",
        link_value="+79990000001",
        brand="unpk",
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id="customer:ambiguous",
            identity_status=IdentityStatus.PARTIAL,
            source_ref="synthetic:ambiguous",
        ),
        actor="test",
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="customer:ambiguous",
            link_type="phone",
            link_value="+79990000001",
            source_system="tallanto_snapshot",
            source_ref="tallanto:ambiguous-phone",
            match_class=IdentityMatchClass.AMBIGUOUS,
            confidence=0.5,
        ),
        actor="test",
    )
    store.close()

    resolution = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={}).resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={"id": "max-1", "phone": "+7 999 000-00-01"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "timeline_identity_ambiguous_value"


def test_wappi_resolver_keeps_shared_phone_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    for customer_id, link_type in (
        ("customer:first", "phone"),
        ("customer:second", "whatsapp_phone"),
    ):
        seed_local_identity(
            db_path,
            tmp_path,
            customer_id=customer_id,
            link_type=link_type,
            link_value="+79990000001",
            brand="unpk",
        )
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={"id": "max-1", "phone": "+7 999 000-00-01"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "timeline_identity_ambiguous_value"


def test_wappi_resolver_accepts_unique_phone_for_family_with_two_students(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:family",
        link_type="phone",
        link_value="+79990000001",
        brand="unpk",
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    for student_id in ("student-1", "student-2"):
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:family",
                link_type="tallanto_student_id",
                link_value=student_id,
                source_system="tallanto_snapshot",
                source_ref=f"tallanto:{student_id}",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
            ),
            actor="test",
        )
    store.close()

    resolution = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={}).resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={"id": "max-1", "phone": "+7 999 000-00-01"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == "customer:family"


def test_wappi_resolver_keeps_person_identity_but_not_brand_authority(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:unpk",
        link_type="phone",
        link_value="+79990000001",
        brand="unpk",
    )
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-max", "foton", "max"),
        dialog={"id": "max-1", "phone": "+7 999 000-00-01"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == "customer:unpk"
    assert resolution.reason == "timeline_identity_unique_cross_brand_person_match"
    assert resolution.evidence["brand_context_authorized"] is False


def test_wappi_resolver_uses_all_non_wappi_brand_history(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:mixed-history",
        link_type="telegram_user_id",
        link_value="123456",
        brand="foton",
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_opportunity(
        CustomerOpportunity(
            tenant_id="foton",
            customer_id="customer:mixed-history",
            opportunity_type=OpportunityType.AMO_DEAL,
            source_system="amocrm_snapshot",
            source_id="old-unpk",
            title="Old UNPK evidence",
            product_context={"brand": "unpk"},
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            confidence=1.0,
        ),
        actor="test",
    )
    store.close()

    resolution = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={}).resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == "customer:mixed-history"
    assert resolution.reason == "timeline_identity_unique_brand_unverified"
    assert resolution.evidence["brand_context_authorized"] is False


def test_wappi_resolver_ignores_missing_stored_brand_row(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:unknown",
        link_type="telegram_user_id",
        link_value="123456",
        brand="foton",
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_opportunity(
        CustomerOpportunity(
            tenant_id="foton",
            customer_id="customer:unknown",
            opportunity_type=OpportunityType.AMO_DEAL,
            source_system="amocrm_snapshot",
            source_id="brand-missing",
            title="Missing brand evidence",
            opened_at=datetime.now(timezone.utc),
            confidence=1.0,
        ),
        actor="test",
    )
    store.close()
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.reason == "timeline_identity_unique_brand_match"


def test_wappi_resolver_keeps_stoplisted_phone_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:family",
        link_type="phone",
        link_value="+79990000001",
        brand="unpk",
    )
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        shared_phone_stoplist=("+79990000001",),
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-max", "unpk", "max"),
        dialog={"id": "max-1", "phone": "+7 999 000-00-01"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "shared_phone"


def test_wappi_resolver_blocks_existing_chat_customer_change(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:new",
        link_type="telegram_user_id",
        link_value="123456",
        brand="foton",
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id="customer:old",
            identity_status=IdentityStatus.STRONG,
            source_ref="synthetic:old",
        ),
        actor="test",
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="customer:old",
            link_type="channel_session_id",
            link_value="wappi_telegram:p-tg:123456",
            source_system="wappi_telegram",
            source_ref="wappi_telegram:chat:p-tg:123456",
            match_class=IdentityMatchClass.MANUAL,
            confidence=1.0,
        ),
        actor="test",
    )
    store.close()
    resolver = WappiPairCustomerResolver.from_store(db_path, tenant_id="foton", pairs={})

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "existing_wappi_chat_customer_conflict"


def test_wappi_report_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "wappi_report.json"
    write_json_report(path, {"ok": True})

    assert path.stat().st_mode & 0o777 == 0o600


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
        transport=DefaultDenyTransport(
            lambda **_kwargs: {"ok": True},
            policy=SafeTransportPolicy.wappi_read_only(),
        ),
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


def test_wappi_history_limit_hit_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1"}, {"id": "chat-2"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Текст", "time": 1_753_000_000}
            ]
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            apply=False,
            limits=WappiFetchLimits(
                chat_limit_per_profile=1,
                messages_per_chat=5,
                message_limit_total=20,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["validation_ok"] is False
    assert "p-tg:chat_limit_hit" in report["limit_hits"]


def test_wappi_history_unions_reordered_chat_snapshots(tmp_path: Path) -> None:
    class DriftingWappiClient(FakeWappiClient):
        def __init__(self, chats, messages):
            super().__init__(chats, messages)
            self.second_page_calls = 0

        def list_chats(self, **kwargs):
            if kwargs.get("offset") == 1:
                self.second_page_calls += 1
                if self.second_page_calls >= 2:
                    self.chats[kwargs["profile_id"]][1] = {"id": "chat-new"}
            return super().list_chats(**kwargs)

    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    client = DriftingWappiClient(
        {"p-tg": [{"id": "chat-1"}, {"id": "chat-2"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [],
            ("telegram", "p-tg", "chat-2"): [],
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=None,
            auto_pairs_file=None,
            limits=WappiFetchLimits(
                chat_limit_per_profile=4,
                messages_per_chat=1,
                message_limit_total=10,
                request_limit_total=20,
                page_size=1,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["profiles"]["p-tg"]["pagination_drift_detected"] is False
    assert report["profiles"]["p-tg"]["chat_snapshot_drift_detected"] is True
    assert any(call.get("kind") == "messages" for call in client.calls)
    assert "p-tg:pagination_drift_detected" not in report["limit_hits"]
    assert report["validation_ok"] is True


def test_wappi_history_detects_message_pagination_drift(tmp_path: Path) -> None:
    class DriftingMessagesClient(FakeWappiClient):
        def __init__(self, chats, messages):
            super().__init__(chats, messages)
            self.second_page_calls = 0

        def get_chat_messages(self, **kwargs):
            if kwargs.get("offset") == 1:
                self.second_page_calls += 1
                if self.second_page_calls >= 2:
                    self.messages[(kwargs["channel"], kwargs["profile_id"], kwargs["chat_id"])][1] = {
                        "id": "m-new",
                        "chat_id": kwargs["chat_id"],
                        "type": "text",
                        "body": "Новое",
                        "time": 1_753_000_003,
                    }
            return super().get_chat_messages(**kwargs)

    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    client = DriftingMessagesClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "m-1", "chat_id": "123456", "type": "text", "body": "Один", "time": 1_753_000_001},
                {"id": "m-2", "chat_id": "123456", "type": "text", "body": "Два", "time": 1_753_000_002},
            ]
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=None,
            auto_pairs_file=None,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=3,
                message_limit_total=20,
                request_limit_total=20,
                page_size=1,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["profiles"]["p-tg"]["pagination_drift_detected"] is True
    assert report["profiles"]["p-tg"]["message_page_drift_detected"] is True
    assert "p-tg:pagination_drift_detected" in report["limit_hits"]
    assert report["validation_ok"] is False


def test_wappi_history_stable_multi_page_has_no_pagination_drift(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1", "type": "user"}, {"id": "chat-2", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Один", "time": 1_753_000_001},
                {"id": "m-2", "chat_id": "chat-1", "type": "text", "body": "Два", "time": 1_753_000_002},
            ],
            ("telegram", "p-tg", "chat-2"): [],
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=None,
            auto_pairs_file=None,
            limits=WappiFetchLimits(
                chat_limit_per_profile=3,
                messages_per_chat=3,
                message_limit_total=20,
                request_limit_total=30,
                page_size=1,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["profiles"]["p-tg"]["pagination_drift_detected"] is False
    assert report["validation_ok"] is True
    kinds = [call.get("kind") for call in client.calls]
    assert kinds.index("messages") > max(index for index, kind in enumerate(kinds) if kind == "chats")


def test_wappi_history_allows_append_only_growth_during_verification(tmp_path: Path) -> None:
    class GrowingWappiClient(FakeWappiClient):
        chat_second_page_calls = 0
        message_first_page_calls = 0

        def list_chats(self, **kwargs):
            if kwargs.get("offset") == 1:
                self.chat_second_page_calls += 1
                if self.chat_second_page_calls == 2:
                    self.chats[kwargs["profile_id"]].append({"id": "chat-3", "type": "user"})
            return super().list_chats(**kwargs)

        def get_chat_messages(self, **kwargs):
            key = (kwargs["channel"], kwargs["profile_id"], kwargs["chat_id"])
            if kwargs.get("offset") == 0 and kwargs["chat_id"] == "chat-1":
                self.message_first_page_calls += 1
                if self.message_first_page_calls == 2:
                    self.messages[key].append(
                        {"id": "m-2", "chat_id": kwargs["chat_id"], "type": "text", "body": "Позже", "time": 1_753_000_002}
                    )
            return super().get_chat_messages(**kwargs)

    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    client = GrowingWappiClient(
        {"p-tg": [{"id": "chat-1", "type": "user"}, {"id": "chat-2", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Сначала", "time": 1_753_000_001}
            ],
            ("telegram", "p-tg", "chat-2"): [],
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=None,
            auto_pairs_file=None,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=5,
                message_limit_total=20,
                request_limit_total=30,
                page_size=1,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["profiles"]["p-tg"]["pagination_drift_detected"] is False
    assert report["validation_ok"] is True
    assert all(call.get("order") == "asc" for call in client.calls)


def test_wappi_history_duplicate_chat_id_between_pages_is_deduplicated(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1", "type": "user"}, {"id": "chat-1", "type": "user"}], "p-max": []},
        {("telegram", "p-tg", "chat-1"): []},
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=None,
            auto_pairs_file=None,
            limits=WappiFetchLimits(
                chat_limit_per_profile=2,
                messages_per_chat=1,
                message_limit_total=10,
                request_limit_total=20,
                page_size=1,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    stats = report["profiles"]["p-tg"]
    assert stats["duplicate_chat_ids"] >= 1
    assert stats["chat_snapshot_drift_detected"] is False
    assert report["validation_ok"] is True
    assert len([call for call in client.calls if call.get("kind") == "messages"]) == 1


def test_wappi_history_request_budget_caps_inner_pagination(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    client = FakeWappiClient(
        {"p-tg": [], "p-max": [{"id": "chat-1", "type": "user"}]},
        {
            ("max", "p-max", "chat-1"): [
                {"unexpected": index}
                for index in range(350)
            ]
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=300,
                page_size=50,
                message_limit_total=500,
                request_limit_total=7,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert len(client.calls) <= 7
    message_calls = [call for call in client.calls if call["kind"] == "messages"]
    assert 0 < len(message_calls) < 6
    assert report["profiles"]["p-max"]["request_limit_hit"] is True
    assert "p-max:request_limit_hit" in report["limit_hits"]
    assert report["validation_ok"] is False
    assert report["mode"] == "apply_blocked"
    assert report["writes"]["applied"] is False
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0


def test_wappi_history_apply_limit_hit_writes_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": f"m-{index}", "chat_id": "chat-1", "type": "text", "body": "Текст", "time": 1_753_000_000 + index}
                for index in range(101)
            ]
        },
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=100,
                page_size=100,
                message_limit_total=500,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["mode"] == "apply_blocked"
    assert report["writes"]["applied"] is False
    assert "p-tg:message_limit_hit" in report["limit_hits"]
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0


def test_wappi_complete_history_reads_past_per_chat_and_total_limits(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    messages = [
        {
            "id": f"m-{index}",
            "chat_id": "chat-1",
            "type": "text",
            "body": f"Текст {index}",
            "time": 1_753_000_000 + index,
        }
        for index in range(205)
    ]
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1", "type": "user"}], "p-max": []},
        {("telegram", "p-tg", "chat-1"): messages},
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            apply=False,
            limits=WappiFetchLimits(
                chat_limit_per_profile=1,
                messages_per_chat=100,
                message_limit_total=100,
                request_limit_total=30,
                page_size=100,
                sleep_seconds=0,
                complete_message_history=True,
            ),
        ),
        client=client,
    )

    assert report["validation_ok"] is True
    assert report["summary"]["records_built"] == 205
    assert report["profiles"]["p-tg"]["message_limit_hit"] is False
    offsets = [call["offset"] for call in client.calls if call["kind"] == "messages"]
    assert 200 in offsets


def test_wappi_history_apply_provenance_drift_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mango_mvp.customer_timeline.wappi_history_import as module

    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    client = FakeWappiClient(
        {"p-tg": [{"id": "chat-1"}], "p-max": []},
        {
            ("telegram", "p-tg", "chat-1"): [
                {"id": "m-1", "chat_id": "chat-1", "type": "text", "body": "Текст", "time": 1_753_000_000}
            ]
        },
    )
    identities = iter(
        (
            {"identity_digest": "before"},
            {"identity_digest": "changed"},
            {"identity_digest": "changed"},
        )
    )
    monkeypatch.setattr(module, "timeline_db_identity", lambda _path: next(identities))

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=None,
            auto_pairs_file=None,
            apply=True,
            limits=WappiFetchLimits(
                chat_limit_per_profile=5,
                messages_per_chat=5,
                message_limit_total=20,
                request_limit_total=20,
                sleep_seconds=0,
            ),
        ),
        client=client,
    )

    assert report["mode"] == "apply_blocked"
    assert report["writes"]["applied"] is False
    assert "provenance_drift" in report["limit_hits"]
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0


def test_wappi_readonly_transport_retries_only_transient_get_errors(monkeypatch) -> None:
    import mango_mvp.customer_timeline.wappi_history_import as module

    calls = []

    def transient_then_ok(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise AmoWappiHttpError("Request failed: timed out")
        return {"ok": True}

    monkeypatch.setattr(module, "_json_http_request", transient_then_ok)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    assert module._readonly_wappi_request_with_backoff(method="GET", url="https://wappi.pro/test") == {"ok": True}
    assert len(calls) == 3

    calls.clear()
    with pytest.raises(module.WappiPhysicalRequestBudgetExceeded):
        module._readonly_wappi_request_with_backoff(
            method="GET",
            url="https://wappi.pro/test",
            request_budget=module.WappiPhysicalRequestBudget(2),
        )
    assert len(calls) == 2

    calls.clear()

    def unauthorized(**kwargs):
        calls.append(kwargs)
        raise AmoWappiHttpError("HTTP 401: unauthorized")

    monkeypatch.setattr(module, "_json_http_request", unauthorized)
    with pytest.raises(AmoWappiHttpError, match="401"):
        module._readonly_wappi_request_with_backoff(method="GET", url="https://wappi.pro/test")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 400: Команда fetchMessages сохранена для повторной отправки. TaskID: test",
        'HTTP 400: {"detail":"Driver not ready","status":"error"}',
        'HTTP 400: {"detail":"повторите запрос чуть позже","error":"TRY_AGAIN_LATER"}',
    ],
)
def test_wappi_readonly_transport_retries_deferred_fetch(monkeypatch, message: str) -> None:
    import mango_mvp.customer_timeline.wappi_history_import as module

    calls = []

    def deferred_then_ok(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AmoWappiHttpError(message)
        return {"ok": True}

    monkeypatch.setattr(module, "_json_http_request", deferred_then_ok)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    assert module._readonly_wappi_request_with_backoff(method="GET", url="https://wappi.pro/test") == {"ok": True}
    assert len(calls) == 2


class FakeWappiClient:
    def __init__(self, chats: Mapping[str, list[Mapping[str, Any]]], messages: Mapping[tuple[str, str, str], list[Mapping[str, Any]]]) -> None:
        self.transport = DefaultDenyTransport(
            lambda **_kwargs: {"ok": True},
            policy=SafeTransportPolicy.wappi_read_only(),
        )
        self.chats = {key: list(value) for key, value in chats.items()}
        self.messages = {key: list(value) for key, value in messages.items()}
        self.calls: list[dict[str, Any]] = []

    def list_chats(self, *, channel: str, profile_id: str, limit: int = 50, offset: int = 0, order: str = "desc", show_all: bool = False) -> Mapping[str, Any]:
        self.calls.append({"kind": "chats", "method": "GET", "channel": channel, "profile_id": profile_id, "limit": limit, "offset": offset, "order": order, "show_all": show_all})
        items = self.chats.get(profile_id, [])
        return {"dialogs": items[offset : offset + limit], "total_count": len(items)}

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
        self.calls.append({"kind": "messages", "method": "GET", "channel": channel, "profile_id": profile_id, "chat_id": chat_id, "limit": limit, "offset": offset, "order": order, "mark_all": mark_all})
        items = self.messages.get((channel, profile_id, chat_id), [])
        return {"messages": items[offset : offset + limit]}


class FakeWidgetWappiClient(FakeWappiClient, WappiPhase1Client):
    def __init__(
        self,
        chats: Mapping[str, list[Mapping[str, Any]]],
        messages: Mapping[tuple[str, str, str], list[Mapping[str, Any]]],
        widget_results: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> None:
        FakeWappiClient.__init__(self, chats, messages)
        self.widget_results = dict(widget_results)

    def list_all_profiles(self) -> list[Mapping[str, Any]]:
        return [
            {
                "profile_id": profile_id,
                "uuid": profile_id,
                "platform": "tg" if profile_id == "p-tg" else "max",
            }
            for profile_id in ("p-tg", "p-max")
        ]

    def find_amocrm_contact(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.widget_results.get(
            (str(kwargs["channel"]), str(kwargs["chat_id"])),
            {"contact": None, "leads": []},
        )


class FakeMcp:
    def __init__(self, contacts=None, leads=None) -> None:
        self.contacts = contacts or []
        self.leads = {str(item["id"]): item for item in (leads or [])}
        self.calls: list[dict[str, Any]] = []

    def amo_api_get(self, *, path, params=None, limit=50):
        self.calls.append({"path": path, "params": params or {}, "limit": limit})
        if path == "contacts":
            query = str((params or {}).get("query") or "")
            contacts = []
            for contact in self.contacts:
                haystack = json.dumps(contact, ensure_ascii=False)
                if query in haystack:
                    contacts.append(contact)
            return {"_embedded": {"contacts": contacts}}
        if path.startswith("contacts/"):
            contact_id = path.split("/", 1)[1]
            return next((item for item in self.contacts if str(item.get("id")) == contact_id), {})
        if path.startswith("leads/"):
            lead_id = path.split("/", 1)[1]
            return self.leads.get(lead_id, {})
        raise AssertionError(path)


def amo_contact(contact_id="111", *, telegram_id="", phone="", leads=("49762441",)):
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


def amo_lead(lead_id="49762441", *, status_id=123, closed_at=None, deleted=False, org=""):
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


def write_pairs(tmp_path: Path, *, lead_id: str, contact_id: str, chat_id: str = "chat-1") -> Path:
    path = tmp_path / "draft_loop_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "profile_id": "p-tg",
                    "chat_id": chat_id,
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


def test_wappi_widget_resolver_uses_contact_and_keeps_multiple_leads(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer_id,
            link_type="amo_lead_id",
            link_value="1002",
            source_system="amocrm_snapshot",
            source_ref="amocrm:lead:1002",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
        ),
        actor="test",
    )
    store.close()
    calls: list[Mapping[str, Any]] = []
    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=lambda **kwargs: calls.append(kwargs) or {
            "contact": {"id": 2002},
            "leads": [{"id": 1001}, {"id": 1002}],
        },
    )
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_client=client,
        widget_crm_id="crm-id",
        widget_profiles={"p-tg": {"profile_id": "p-tg", "uuid": "p-tg", "platform": "tg"}},
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user", "user": {"Username": "parent"}},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == customer_id
    assert resolution.contact_id == "2002"
    assert resolution.lead_ids == ("1001", "1002")
    assert resolution.resolution_source == "wappi_amo_widget"
    assert resolver.widget_calls == 1
    assert calls[0]["json_body"]["chat_id"] == "123456"

    normalized = WappiHistoryTimelineNormalizer(
        tenant_id="foton",
        source_system="wappi_telegram",
    ).normalize(
        wappi_message_to_record(
            profile=profile("p-tg", "foton", "telegram"),
            message=WappiHistoryMessage(
                profile_id="p-tg",
                chat_id="123456",
                message_id="m-widget",
                text="Здравствуйте",
                message_type="text",
                timestamp=1_753_000_000,
                from_me=False,
            ),
            resolution=resolution,
        )
    )
    assert normalized.events[0].metadata["lead_ids"] == ("1001", "1002")
    assert normalized.identity_links[0].evidence["lead_ids"] == ("1001", "1002")


def test_wappi_widget_resolver_hard_blocks_known_brand_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(
        db_path,
        tmp_path,
        lead_id="1001",
        contact_id="2002",
        brand="unpk",
    )
    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=lambda **kwargs: {"contact": {"id": 2002}, "leads": [{"id": 1001}]},
    )
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_client=client,
        widget_crm_id="crm-id",
        widget_profiles={"p-tg": {"profile_id": "p-tg", "uuid": "p-tg", "platform": "tg"}},
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user", "user": {"Username": "parent"}},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.status == "pending_attribution"
    assert resolution.reason == "wappi_widget_brand_mismatch"
    assert resolution.candidate_customer_ids == (customer_id,)
    assert resolution.evidence["brand_context_authorized"] is False


def test_wappi_widget_resolver_uses_unique_lead_when_contact_is_not_loaded(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="")
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "resolved",
                "contact_id": "2002",
                "lead_ids": ("1001",),
                "resolution_source": "wappi_amo_widget",
            }
        },
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == customer_id
    assert resolution.contact_id == "2002"
    assert resolution.match_key == "wappi_widget_lead"


def test_wappi_history_auto_resolver_accepts_exact_contact_without_active_deal_for_identity(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(
        db_path,
        tmp_path,
        lead_id="",
        contact_id="2002",
    )

    class ExactIdentityOnlyResolver:
        calls = 0
        stoplist_error = ""

        def __call__(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            return {
                "status": "rejected",
                "reason": "no_active_lead",
                "contact_id": "2002",
                "match_key": "Telegram ID",
            }

    auto_resolver = ExactIdentityOnlyResolver()
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        amo_auto_resolver=auto_resolver,  # type: ignore[arg-type]
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id == customer_id
    assert resolution.reason == "amo_auto_exact_identity_without_opportunity"
    assert resolution.evidence["single_active_lead"] is False
    assert resolution.evidence["identity_only"] is True
    assert auto_resolver.calls == 1


def test_wappi_history_auto_resolver_error_keeps_only_that_chat_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()

    class FailingResolver:
        calls = 0
        stoplist_error = ""

        def __call__(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            raise RuntimeError("sensitive upstream error")

    auto_resolver = FailingResolver()
    resolution = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        amo_auto_resolver=auto_resolver,  # type: ignore[arg-type]
    ).resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "amo_auto_lookup_error"
    assert auto_resolver.calls == 1


def test_wappi_widget_link_map_is_private_resumable_and_channel_scoped(tmp_path: Path) -> None:
    calls: list[Mapping[str, Any]] = []

    def transport(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs)
        url = str(kwargs["url"])
        if "/tapi/sync/chats/get" in url:
            return {"dialogs": [{"id": "tg-chat", "type": "user"}], "total_count": 1}
        if "/maxapi/sync/chats/get" in url:
            return {
                "dialogs": [
                    {
                        "id": "max-chat",
                        "type": "DIALOG",
                        "participants": [{"user_id": "max-peer", "is_me": False}],
                    }
                ],
                "total_count": 1,
            }
        body = kwargs["json_body"]
        if body["platform"] == "tg":
            return {
                "contact": {"id": 2002, "_embedded": {"leads": [{"id": 1002}]}},
                "leads": [{"id": 1001}, {"id": 1002}],
            }
        return {"contact": {"id": 3003}, "leads": [{"id": 3001}]}

    client = WappiPhase1Client(
        WappiClientConfig(
            base_url="https://wappi.pro",
            telegram_token="telegram-secret",
            max_token="max-secret",
        ),
        transport=transport,
    )
    profiles = (
        WappiProfileSpec(profile_id="same-profile", brand="foton", channel="telegram"),
        WappiProfileSpec(profile_id="same-profile", brand="foton", channel="max"),
    )
    runtime_profiles = {
        ("telegram", "same-profile"): {"uuid": "tg-uuid", "platform": "tg"},
        ("max", "same-profile"): {"uuid": "max-uuid", "platform": "max"},
    }
    link_db = tmp_path / "wappi_amo_links.sqlite"
    limits = WappiFetchLimits(
        chat_limit_per_profile=5,
        messages_per_chat=0,
        message_limit_total=10,
        request_limit_total=20,
        sleep_seconds=0,
    )

    first = collect_wappi_widget_links(
        client=client,
        profiles=profiles,
        runtime_profiles=runtime_profiles,
        crm_id="crm-id",
        db_path=link_db,
        limits=limits,
    )
    direct_calls_after_first = sum("/amocrm/contact/find" in str(item["url"]) for item in calls)
    second = collect_wappi_widget_links(
        client=client,
        profiles=profiles,
        runtime_profiles=runtime_profiles,
        crm_id="crm-id",
        db_path=link_db,
        limits=limits,
    )
    links = load_wappi_widget_links(link_db)

    assert first["complete"] is True
    assert second["complete"] is True
    assert direct_calls_after_first == 2
    assert sum("/amocrm/contact/find" in str(item["url"]) for item in calls) == 2
    assert links[("telegram", "same-profile", "tg-chat")]["lead_ids"] == ("1001", "1002")
    assert links[("max", "same-profile", "max-chat")]["contact_id"] == "3003"
    assert link_db.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(link_db) as con:
        columns = {str(row[1]) for row in con.execute("PRAGMA table_info(wappi_amo_links)")}
    assert not columns.intersection({"token", "phone", "username", "raw_response"})
    raw_map = link_db.read_bytes()
    assert b"telegram-secret" not in raw_map
    assert b"max-secret" not in raw_map
    assert not any("messages/get" in str(item["url"]) for item in calls)


def test_wappi_widget_catalog_unions_moving_pages_until_total_count(tmp_path: Path) -> None:
    class MovingCatalogClient(FakeWidgetWappiClient):
        def __init__(self) -> None:
            super().__init__({}, {}, {})
            self.catalog_pass = 0
            self.widget_calls: list[str] = []

        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            if int(kwargs["offset"]) == 0:
                self.catalog_pass += 1
            first = self.catalog_pass == 1
            pages = {
                0: [{"id": "a", "type": "user"}, {"id": "b", "type": "user"}],
                2: ([{"id": "b", "type": "user"}, {"id": "c", "type": "user"}] if first else
                    [{"id": "c", "type": "user"}, {"id": "d", "type": "user"}]),
            }
            return {"dialogs": pages.get(int(kwargs["offset"]), []), "total_count": 4}

        def find_amocrm_contact(self, **kwargs: Any) -> Mapping[str, Any]:
            chat_id = str(kwargs["chat_id"])
            self.widget_calls.append(chat_id)
            return {"contact": {"id": int(ord(chat_id) - 96)}, "leads": [{"id": 1000 + ord(chat_id)}]}

    client = MovingCatalogClient()
    report = collect_wappi_widget_links(
        client=client,
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=0, request_limit_total=20, page_size=2),
    )

    assert report["profiles"]["telegram:p-tg"]["catalog_passes"] == 2
    assert report["profiles"]["telegram:p-tg"]["unique_catalogued"] == 4
    assert report["accounting_complete"] is True
    assert report["linkage_complete"] is True
    assert sorted(client.widget_calls) == ["a", "b", "c", "d"]


def test_wappi_widget_coverage_reports_detailed_lookup_statuses(tmp_path: Path) -> None:
    chats = tuple(
        {"id": str(index), "type": "user"}
        for index in range(1, 10)
    )

    class StatusClient(FakeWidgetWappiClient):
        def __init__(self) -> None:
            super().__init__({"p-tg": list(chats)}, {}, {})

        def find_amocrm_contact(self, **kwargs: Any) -> Mapping[str, Any]:
            chat_id = str(kwargs["chat_id"])
            if chat_id == "1":
                return {"contact": {"id": 1}, "leads": []}
            if chat_id == "2":
                return {"contact": {"id": 2}, "leads": [{"id": 20}]}
            if chat_id == "3":
                return {"contact": {"id": 3}, "leads": [{"id": 30}, {"id": 31}]}
            if chat_id == "4":
                return {"contact": None, "leads": []}
            if chat_id == "5":
                raise AmoWappiHttpError("HTTP 401: unauthorized")
            if chat_id == "6":
                raise AmoWappiHttpError("HTTP 429: rate limit")
            if chat_id == "7":
                raise TimeoutError("timed out")
            if chat_id == "8":
                raise AmoWappiHttpError("HTTP 503: unavailable")
            return {"contact": {"id": "invalid"}, "leads": []}

    report = collect_wappi_widget_links(
        client=StatusClient(),
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=0, request_limit_total=20),
    )

    assert report["accounting_complete"] is True
    assert report["linkage_complete"] is False
    assert report["counts"] == {
        "auth_error": 1,
        "http_5xx": 1,
        "invalid_response": 1,
        "rate_limit": 1,
        "resolved_contact_only": 1,
        "resolved_multiple_leads": 1,
        "resolved_one_lead": 1,
        "timeout": 1,
        "widget_no_contact": 1,
    }


def test_wappi_widget_coverage_reports_catalog_transport_error(tmp_path: Path) -> None:
    class BrokenCatalogClient(FakeWidgetWappiClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            del kwargs
            raise AmoWappiHttpError("HTTP 429: rate limit")

    report = collect_wappi_widget_links(
        client=BrokenCatalogClient({}, {}, {}),
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=0, request_limit_total=20),
    )

    assert report["accounting_complete"] is False
    assert report["counts"]["rate_limit"] == 1
    assert report["profiles"]["telegram:p-tg"]["catalog_error"] == "rate_limit"


@pytest.mark.parametrize("bad_total", [None, "", "bad"])
def test_wappi_widget_coverage_rejects_invalid_total_count(tmp_path: Path, bad_total: Any) -> None:
    class BadTotalClient(FakeWidgetWappiClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            del kwargs
            return {"dialogs": [], "total_count": bad_total}

    report = collect_wappi_widget_links(
        client=BadTotalClient({}, {}, {}),
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=0, request_limit_total=20),
    )

    assert report["accounting_complete"] is False
    assert report["counts"]["total_count_missing"] == 1


def test_wappi_widget_coverage_rejects_later_malformed_total_count(tmp_path: Path) -> None:
    class LaterBadTotalClient(FakeWidgetWappiClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            if int(kwargs["offset"]) == 0:
                return {
                    "dialogs": [{"id": "1", "type": "user"}, {"id": "2", "type": "user"}],
                    "total_count": 3,
                }
            return {"dialogs": [{"id": "3", "type": "user"}], "total_count": "bad"}

    report = collect_wappi_widget_links(
        client=LaterBadTotalClient({}, {}, {}),
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(
            chat_limit_per_profile=10,
            messages_per_chat=0,
            request_limit_total=20,
            page_size=2,
        ),
    )

    assert report["accounting_complete"] is False
    assert report["profiles"]["telegram:p-tg"]["catalog_error"] == "invalid_response"


def test_wappi_widget_coverage_rejects_invalid_lead_multiplicity(tmp_path: Path) -> None:
    client = FakeWidgetWappiClient(
        {"p-tg": [{"id": "1", "type": "user"}]},
        {},
        {("telegram", "1"): {"contact": {"id": 10}, "leads": [{"id": "bad"}]}},
    )

    report = collect_wappi_widget_links(
        client=client,
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=0, request_limit_total=20),
    )

    assert report["accounting_complete"] is True
    assert report["linkage_complete"] is False
    assert report["counts"]["invalid_response"] == 1


def test_wappi_widget_coverage_exact_request_budget_is_not_a_limit_hit(tmp_path: Path) -> None:
    client = FakeWidgetWappiClient(
        {"p-tg": [{"id": "1", "type": "user"}]},
        {},
        {("telegram", "1"): {"contact": {"id": 10}, "leads": [{"id": 20}]}},
    )

    report = collect_wappi_widget_links(
        client=client,
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=0, request_limit_total=2),
    )

    assert report["requests"] == 2
    assert report["request_limit_hit"] is False
    assert report["complete"] is True


def test_wappi_widget_coverage_accounts_for_profiles_after_budget_exhaustion(tmp_path: Path) -> None:
    client = FakeWidgetWappiClient(
        {"p-one": [{"id": "1", "type": "user"}], "p-two": [{"id": "2", "type": "user"}]},
        {},
        {
            ("telegram", "1"): {"contact": {"id": 10}, "leads": [{"id": 20}]},
            ("telegram", "2"): {"contact": {"id": 11}, "leads": [{"id": 21}]},
        },
    )
    profiles = tuple(
        WappiProfileSpec(profile_id=profile_id, brand="foton", channel="telegram")
        for profile_id in ("p-one", "p-two")
    )

    report = collect_wappi_widget_links(
        client=client,
        profiles=profiles,
        runtime_profiles={
            ("telegram", profile.profile_id): {"uuid": profile.profile_id, "platform": "tg"}
            for profile in profiles
        },
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=0, request_limit_total=2),
    )

    assert set(report["profiles"]) == {"telegram:p-one", "telegram:p-two"}
    assert report["accounting_complete"] is False
    assert report["request_limit_hit"] is True
    assert report["profiles"]["telegram:p-two"]["catalog_error"] == "request_limit"


def test_wappi_widget_coverage_only_cli_flag(tmp_path: Path) -> None:
    from scripts.import_wappi_history_to_timeline import build_parser, config_from_args

    args = build_parser().parse_args(
        [
            "--timeline-db",
            str(tmp_path / "timeline.sqlite"),
            "--allowed-root",
            str(tmp_path),
            "--widget-link-db",
            str(tmp_path / "links.sqlite"),
            "--widget-coverage-only",
        ]
    )

    assert config_from_args(args).widget_coverage_only is True


def test_wappi_widget_coverage_only_reads_no_messages_and_writes_no_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path):
        pass
    before = db_path.read_bytes()
    phase1 = write_phase1_config(tmp_path)
    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-id")
    class CountingWidgetClient(FakeWidgetWappiClient):
        widget_call_count = 0

        def find_amocrm_contact(self, **kwargs: Any) -> Mapping[str, Any]:
            self.widget_call_count += 1
            return super().find_amocrm_contact(**kwargs)

    client = CountingWidgetClient(
        {"p-tg": [{"id": "1", "type": "user"}], "p-max": []},
        {},
        {("telegram", "1"): {"contact": {"id": 10}, "leads": [{"id": 20}]}},
    )
    config = WappiHistoryImportConfig(
        timeline_db=db_path,
        allowed_root=tmp_path,
        phase1_config=phase1,
        pairs_file=None,
        auto_pairs_file=None,
        widget_link_db=tmp_path / "links.sqlite",
        widget_coverage_only=True,
        limits=WappiFetchLimits(chat_limit_per_profile=10, messages_per_chat=10, request_limit_total=20),
    )

    report = run_wappi_history_import(config, client=client)
    repeated = run_wappi_history_import(config, client=client)

    assert report["mode"] == "widget_coverage_only"
    assert report["validation_ok"] is True
    assert report["summary"]["messages_read"] == 0
    assert report["summary"]["writes_applied"] == 0
    assert repeated["validation_ok"] is True
    assert client.widget_call_count == 2
    assert not any(call["kind"] == "messages" for call in client.calls)
    assert db_path.read_bytes() == before


def test_wappi_widget_contact_hydrate_reuses_known_lead_family(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(
        db_path,
        tmp_path,
        lead_id="42",
        contact_id="99",
    )

    class ContactClient:
        calls: list[tuple[str, Mapping[str, Any], int]] = []

        def amo_api_get(self, *, path: str, params: Mapping[str, Any], limit: int) -> Mapping[str, Any]:
            self.calls.append((path, params, limit))
            assert path == "contacts"
            assert params == {"filter[id][]": ["30"], "with": "leads"}
            assert limit == 1
            return {
                "_embedded": {
                    "contacts": [
                        {
                            "id": 30,
                            "name": "Parent",
                            "created_at": 1_753_000_000,
                            "updated_at": 1_753_000_001,
                            "_embedded": {"leads": [{"id": 42}]},
                        }
                    ]
                }
            }

    links = {
        ("telegram", "p-tg", "chat"): {
            "status": "resolved",
            "contact_id": "30",
            "lead_ids": ("42",),
        }
    }
    client = ContactClient()
    first = hydrate_wappi_widget_contacts(
        timeline_db=db_path,
        allowed_root=tmp_path,
        widget_links=links,
        amo_mcp_env_file=None,
        amo_client=client,
    )
    second = hydrate_wappi_widget_contacts(
        timeline_db=db_path,
        allowed_root=tmp_path,
        widget_links=links,
        amo_mcp_env_file=None,
        amo_client=client,
    )

    assert first["requested"] == 1
    assert first["batches"] == 1
    assert first["fetch_errors"] == 0
    assert second["requested"] == 0
    assert len(client.calls) == 1
    with sqlite3.connect(db_path) as con:
        owner = con.execute(
            "SELECT customer_id FROM identity_links WHERE link_type = 'amo_contact_id' AND link_value = '30'"
        ).fetchone()[0]
    assert owner == customer_id


def test_wappi_widget_contact_hydrate_batches_exact_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="42", contact_id="99")

    class ContactClient:
        calls: list[tuple[str, Mapping[str, Any], int]] = []

        def amo_api_get(self, *, path: str, params: Mapping[str, Any], limit: int) -> Mapping[str, Any]:
            ids = tuple(str(value) for value in params["filter[id][]"])
            self.calls.append((path, params, limit))
            return {
                "_embedded": {
                    "contacts": [
                        {
                            "id": int(contact_id),
                            "name": "Parent",
                            "created_at": 1_753_000_000,
                            "updated_at": 1_753_000_001,
                            "_embedded": {"leads": [{"id": 42}]},
                        }
                        for contact_id in ids
                    ]
                }
            }

    client = ContactClient()
    links = {
        ("telegram", "p-tg", f"chat-{contact_id}"): {
            "status": "resolved",
            "contact_id": str(contact_id),
            "lead_ids": ("42",),
        }
        for contact_id in range(1000, 1101)
    }
    report = hydrate_wappi_widget_contacts(
        timeline_db=db_path,
        allowed_root=tmp_path,
        widget_links=links,
        amo_mcp_env_file=None,
        amo_client=client,
    )

    assert report["requested"] == 101
    assert report["batches"] == 3
    assert report["fetched"] == 101
    assert len(client.calls) == 3
    assert all(path == "contacts" and limit <= 50 for path, _params, limit in client.calls)


def test_wappi_widget_contact_hydrate_falls_back_to_exact_contact_endpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="42", contact_id="99")

    class ContactClient:
        calls: list[str] = []

        def amo_api_get(self, *, path: str, params: Mapping[str, Any], limit: int) -> Mapping[str, Any]:
            self.calls.append(path)
            if path == "contacts":
                return {"_embedded": {"contacts": []}}
            assert path == "contacts/30"
            assert params == {"with": "leads"}
            assert limit == 1
            return {
                "id": 30,
                "name": "Parent",
                "created_at": 1_753_000_000,
                "updated_at": 1_753_000_001,
                "_embedded": {"leads": [{"id": 42}]},
            }

    client = ContactClient()
    report = hydrate_wappi_widget_contacts(
        timeline_db=db_path,
        allowed_root=tmp_path,
        widget_links={
            ("telegram", "p-tg", "chat"): {
                "status": "resolved",
                "contact_id": "30",
                "lead_ids": ("42",),
            }
        },
        amo_mcp_env_file=None,
        amo_client=client,
    )

    assert report["fallback_requested"] == 1
    assert report["fallback_fetched"] == 1
    assert report["fetch_errors"] == 0
    assert client.calls == ["contacts", "contacts/30"]
    with sqlite3.connect(db_path) as con:
        owner = con.execute(
            "SELECT customer_id FROM identity_links WHERE link_type = 'amo_contact_id' AND link_value = '30'"
        ).fetchone()[0]
    assert owner == customer_id


def test_wappi_widget_missing_contact_falls_back_to_static_pair_when_optional(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=lambda **kwargs: {"contact": None, "leads": []},
    )
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={
            DraftLoopKey("p-tg", "123456"): DraftLoopPair(
                key=DraftLoopKey("p-tg", "123456"),
                lead_id="1001",
                expected_brand="foton",
                contact_id="2002",
            )
        },
        widget_client=client,
        widget_crm_id="crm-id",
        widget_profiles={"p-tg": {"profile_id": "p-tg", "uuid": "p-tg", "platform": "tg"}},
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is True
    assert resolution.customer_id
    assert resolution.resolution_source == "draft_loop_pair"


@pytest.mark.parametrize("widget_status", ("conflict", "http_5xx", "timeout"))
def test_wappi_widget_exact_failure_does_not_fall_back_when_optional(
    tmp_path: Path,
    widget_status: str,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, customer_id="customer:static")
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={
            DraftLoopKey("p-tg", "123456"): DraftLoopPair(
                key=DraftLoopKey("p-tg", "123456"),
                lead_id="1001",
                expected_brand="foton",
                contact_id="2002",
            )
        },
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": widget_status,
                "contact_id": "",
                "lead_ids": (),
            }
        },
        widget_required=False,
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.customer_id is None
    assert resolution.reason == f"wappi_widget_{widget_status}"
    assert resolution.resolution_source == "wappi_amo_widget"


def test_wappi_widget_missing_contact_stays_pending_when_required(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "missing",
                "contact_id": "",
                "lead_ids": (),
            }
        },
        widget_required=True,
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "wappi_widget_contact_missing"


def test_wappi_widget_candidate_contact_never_becomes_identity_truth(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "candidate",
                "contact_id": "2002",
                "lead_ids": ("1001",),
                "resolution_source": "amo_event_sequence_candidate",
            }
        },
        widget_required=True,
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "wappi_widget_contact_unconfirmed"
    assert not resolution.customer_id


def test_wappi_widget_map_resolves_from_two_local_amo_event_matches(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    link_db = tmp_path / "wappi_amo_links.sqlite"
    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=lambda **kwargs: (
            {"dialogs": [{"id": "123456", "type": "user", "last_timestamp": 1_753_000_060}]}
            if "/chats/get" in str(kwargs["url"])
            else {"contact": None, "leads": []}
        ),
    )
    collect_wappi_widget_links(
        client=client,
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=link_db,
        limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=0, request_limit_total=10),
    )

    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    for index, (timestamp, direction) in enumerate(
        ((1_753_000_000, TimelineDirection.INBOUND), (1_753_000_060, TimelineDirection.OUTBOUND)),
        start=1,
    ):
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                event_type=TimelineEventType.TELEGRAM_MESSAGE,
                event_at=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                source_system="wappi_telegram",
                source_id=f"wappi-{index}",
                direction=direction,
                metadata={"profile_id": "p-tg", "chat_id": "123456"},
            ),
            actor="test",
        )
        amo_type = "incoming_chat_message" if direction == TimelineDirection.INBOUND else "outgoing_chat_message"
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type=TimelineEventType.AMO_NOTE,
                event_at=datetime.fromtimestamp(timestamp + 3, tz=timezone.utc),
                source_system="amocrm_event",
                source_id=f"amo-{index}",
                direction=direction,
                record={
                    "payload": {
                        "id": f"amo-{index}",
                        "type": amo_type,
                        "entity_type": "lead",
                        "created_at": timestamp + 3,
                        "entity_id": 1001,
                        "_embedded": {"entity": {"linked_talk_contact_id": 2002}},
                        "value_after": [{"message": {"origin": "pro.wappi.tg", "talk_id": 3003}}],
                    }
                },
            ),
            actor="test",
        )
        for tenant_id, source_id, talk_id in (
            ("other", f"amo-other-{index}", 9001),
            ("foton", f"amo-superseded-{index}", 9002),
        ):
            store.upsert_event(
                TimelineEvent(
                    tenant_id=tenant_id,
                    event_type=TimelineEventType.AMO_NOTE,
                    event_at=datetime.fromtimestamp(timestamp + 1, tz=timezone.utc),
                    source_system="amocrm_event",
                    source_id=source_id,
                    direction=direction,
                    record={
                        "payload": {
                            "id": source_id,
                            "type": amo_type,
                            "entity_type": "lead",
                            "created_at": timestamp + 1,
                            "entity_id": 9000,
                            "_embedded": {"entity": {"linked_talk_contact_id": 9000}},
                            "value_after": [{"message": {"origin": "pro.wappi.tg", "talk_id": talk_id}}],
                        }
                    },
                ),
                actor="test",
            )
    store.close()
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE timeline_events SET superseded_by = 'test-superseded' WHERE source_id LIKE 'amo-superseded-%'"
        )
        con.commit()

    report = enrich_wappi_widget_links_from_timeline_amo_events(
        timeline_db=db_path,
        widget_link_db=link_db,
    )
    links = load_wappi_widget_links(link_db)

    assert report == {
        "missing_before": 1,
        "candidates": 1,
        "ambiguous": 0,
        "cross_chat_ambiguous": 0,
        "insufficient": 0,
    }
    linked = links[("telegram", "p-tg", "123456")]
    assert linked["contact_id"] == "2002"
    assert linked["lead_ids"] == ("1001",)
    assert linked["status"] == "candidate"
    assert linked["resolution_source"] == "amo_event_sequence_candidate"
    assert linked["matched_points"] == 2
    with sqlite3.connect(link_db) as con:
        assert con.execute(
            "SELECT amo_talk_id, amo_chat_id FROM wappi_amo_links WHERE chat_id='123456'"
        ).fetchone() == ("3003", "")


def test_wappi_widget_event_sequence_rejects_one_talk_for_two_chats(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    link_db = tmp_path / "wappi_amo_links.sqlite"
    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=lambda **kwargs: (
            {
                "dialogs": [
                    {"id": "111111", "type": "user", "last_timestamp": 1_753_000_060},
                    {"id": "222222", "type": "user", "last_timestamp": 1_753_000_060},
                ]
            }
            if "/chats/get" in str(kwargs["url"])
            else {"contact": None, "leads": []}
        ),
    )
    collect_wappi_widget_links(
        client=client,
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=link_db,
        limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=0, request_limit_total=10),
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    for chat_id in ("111111", "222222"):
        for index, (timestamp, direction) in enumerate(
            ((1_753_000_000, TimelineDirection.INBOUND), (1_753_000_060, TimelineDirection.OUTBOUND)),
            start=1,
        ):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    event_type=TimelineEventType.TELEGRAM_MESSAGE,
                    event_at=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                    source_system="wappi_telegram",
                    source_id=f"wappi-{chat_id}-{index}",
                    direction=direction,
                    metadata={"profile_id": "p-tg", "chat_id": chat_id, "message_id": f"{chat_id}-{index}"},
                ),
                actor="test",
            )
    for index, (timestamp, direction) in enumerate(
        ((1_753_000_003, TimelineDirection.INBOUND), (1_753_000_063, TimelineDirection.OUTBOUND)),
        start=1,
    ):
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type=TimelineEventType.AMO_NOTE,
                event_at=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                source_system="amocrm_event",
                source_id=f"amo-shared-{index}",
                direction=direction,
                record={
                    "payload": {
                        "id": f"amo-shared-{index}",
                        "type": "incoming_chat_message" if direction == TimelineDirection.INBOUND else "outgoing_chat_message",
                        "entity_type": "lead",
                        "created_at": timestamp,
                        "entity_id": 1001,
                        "_embedded": {"entity": {"linked_talk_contact_id": 2002}},
                        "value_after": [{"message": {"origin": "pro.wappi.tg", "talk_id": 3003}}],
                    }
                },
            ),
            actor="test",
        )
    store.close()

    report = enrich_wappi_widget_links_from_timeline_amo_events(
        timeline_db=db_path,
        widget_link_db=link_db,
    )

    assert report["candidates"] == 0
    assert report["cross_chat_ambiguous"] == 2
    assert all(item["status"] == "missing" for item in load_wappi_widget_links(link_db).values())


def seed_amo_talk_candidate(
    link_db: Path,
    *,
    chat_id: str = "123456",
    talk_id: str = "3003",
    contact_id: str = "2002",
    lead_id: str = "1001",
) -> None:
    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=lambda **kwargs: (
            {"dialogs": [{"id": chat_id, "type": "user", "last_timestamp": 1}]}
            if "/chats/get" in str(kwargs["url"])
            else {"contact": None, "leads": []}
        ),
    )
    collect_wappi_widget_links(
        client=client,
        profiles=(WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        runtime_profiles={("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        crm_id="crm-id",
        db_path=link_db,
        limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=0, request_limit_total=10),
    )
    with sqlite3.connect(link_db) as con:
        con.execute(
            """
            UPDATE wappi_amo_links
            SET status='candidate', contact_id=?, lead_ids_json=?,
                resolution_source='amo_event_sequence_candidate', amo_talk_id=?
            WHERE chat_id=?
            """,
            (contact_id, json.dumps((lead_id,)), talk_id, chat_id),
        )


class FakeAmoTalkClient:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.paths: list[str] = []

    def amo_api_get(self, *, path: str, params: Mapping[str, Any], limit: int) -> Mapping[str, Any]:
        assert params == {}
        assert limit == 1
        self.paths.append(path)
        return self.payload


def amo_talk_payload(*, talk_id: str = "3003", contact_id: str = "2002", lead_id: str = "1001") -> Mapping[str, Any]:
    return {
        "talk_id": int(talk_id),
        "chat_id": "88278e98-2b8d-4ae2-a5f0-bfab511cd621",
        "contact_id": int(contact_id),
        "entity_id": int(lead_id),
        "entity_type": "lead",
        "_embedded": {"contacts": [{"id": int(contact_id)}], "leads": [{"id": int(lead_id)}]},
    }


def test_amo_talk_exact_confirmation_is_idempotent(tmp_path: Path) -> None:
    link_db = tmp_path / "wappi_amo_links.sqlite"
    timeline_db = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(timeline_db, tmp_path, lead_id="1001", contact_id="2002")
    seed_amo_talk_candidate(link_db)
    client = FakeAmoTalkClient(amo_talk_payload())

    first = confirm_wappi_widget_candidates_from_amo_talks(widget_link_db=link_db, amo_client=client)
    second = confirm_wappi_widget_candidates_from_amo_talks(widget_link_db=link_db, amo_client=client)

    assert first == {
        "candidates": 1,
        "resolved": 1,
        "identity_conflict": 0,
        "cross_chat_conflict": 0,
        "invalid_response": 0,
        "lookup_error": 0,
    }
    assert second["candidates"] == 0
    assert client.paths == ["/api/v4/talks/3003"]
    link = load_wappi_widget_links(link_db)[("telegram", "p-tg", "123456")]
    assert link["status"] == "resolved"
    assert link["resolution_source"] == "amo_talk_authoritative"
    with sqlite3.connect(link_db) as con:
        assert con.execute("SELECT amo_talk_id, amo_chat_id FROM wappi_amo_links").fetchone() == (
            "3003",
            "88278e98-2b8d-4ae2-a5f0-bfab511cd621",
        )
    resolution = WappiPairCustomerResolver.from_store(
        timeline_db,
        tenant_id="foton",
        pairs={},
        widget_links=load_wappi_widget_links(link_db),
    ).resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )
    assert resolution.customer_id == customer_id
    assert resolution.resolution_source == "amo_talk_authoritative"


def test_amo_talk_identity_conflict_stays_candidate(tmp_path: Path) -> None:
    link_db = tmp_path / "wappi_amo_links.sqlite"
    seed_amo_talk_candidate(link_db)

    report = confirm_wappi_widget_candidates_from_amo_talks(
        widget_link_db=link_db,
        amo_client=FakeAmoTalkClient(amo_talk_payload(contact_id="9009")),
    )

    assert report["identity_conflict"] == 1
    assert load_wappi_widget_links(link_db)[("telegram", "p-tg", "123456")]["status"] == "conflict"


def test_amo_talk_null_or_malformed_response_stays_candidate(tmp_path: Path) -> None:
    link_db = tmp_path / "wappi_amo_links.sqlite"
    seed_amo_talk_candidate(link_db)

    report = confirm_wappi_widget_candidates_from_amo_talks(
        widget_link_db=link_db,
        amo_client=FakeAmoTalkClient({}),
    )

    assert report["invalid_response"] == 1
    assert load_wappi_widget_links(link_db)[("telegram", "p-tg", "123456")]["status"] == "candidate"


def test_amo_talk_reused_by_two_wappi_chats_fails_closed(tmp_path: Path) -> None:
    link_db = tmp_path / "wappi_amo_links.sqlite"
    seed_amo_talk_candidate(link_db, chat_id="111111")
    seed_amo_talk_candidate(link_db, chat_id="222222")
    client = FakeAmoTalkClient(amo_talk_payload())

    report = confirm_wappi_widget_candidates_from_amo_talks(widget_link_db=link_db, amo_client=client)

    assert report["cross_chat_conflict"] == 2
    assert client.paths == []
    assert all(item["status"] == "conflict" for item in load_wappi_widget_links(link_db).values())


def test_amo_talk_client_requires_https_ai_office_proxy(tmp_path: Path) -> None:
    good = tmp_path / "good.env"
    good.write_text("CONNECTOR_URL=https://api.fotonai.online/api/mcp/foton-crm-readonly\nBEARER_TOKEN=x\n")
    assert _build_safe_amo_talk_client(good).config.connector_url.startswith("https://api.fotonai.online/")

    bad = tmp_path / "bad.env"
    bad.write_text("CONNECTOR_URL=https://educent.amocrm.ru/api/mcp\nBEARER_TOKEN=x\n")
    with pytest.raises(ValueError, match="HTTPS api.fotonai.online"):
        _build_safe_amo_talk_client(bad)


def test_wappi_widget_map_rechecks_activity_and_quarantines_changed_contact(tmp_path: Path) -> None:
    state = {"timestamp": 100, "contact_id": ""}
    calls = {"find": 0}

    def transport(**kwargs: Any) -> Mapping[str, Any]:
        if "/chats/get" in str(kwargs["url"]):
            return {
                "dialogs": [{"id": "123456", "type": "user", "last_timestamp": state["timestamp"]}],
                "total_count": 1,
            }
        calls["find"] += 1
        contact_id = str(state["contact_id"])
        return {"contact": {"id": contact_id} if contact_id else None, "leads": [{"id": 1001}] if contact_id else []}

    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="token"),
        transport=transport,
    )
    link_db = tmp_path / "wappi_amo_links.sqlite"
    kwargs = {
        "client": client,
        "profiles": (WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),),
        "runtime_profiles": {("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}},
        "crm_id": "crm-id",
        "db_path": link_db,
        "limits": WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=0, request_limit_total=10),
    }
    collect_wappi_widget_links(**kwargs)
    state.update(timestamp=200, contact_id="2002")
    collect_wappi_widget_links(**kwargs)
    assert load_wappi_widget_links(link_db)[("telegram", "p-tg", "123456")]["contact_id"] == "2002"

    state.update(timestamp=300, contact_id="3003")
    report = collect_wappi_widget_links(**kwargs)
    link = load_wappi_widget_links(link_db)[("telegram", "p-tg", "123456")]

    assert calls["find"] == 3
    assert report["counts"]["relation_conflict"] == 1
    assert link["status"] == "conflict"
    assert link["contact_id"] == "2002"


def test_wappi_widget_map_primes_old_unmatched_history(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_event(
        TimelineEvent(
            tenant_id="foton",
            event_type=TimelineEventType.TELEGRAM_MESSAGE,
            event_at=datetime.fromtimestamp(1_753_000_000, tz=timezone.utc),
            source_system="wappi_telegram",
            source_id="old-message",
            source_ref="wappi:old-message",
            direction=TimelineDirection.INBOUND,
            text_preview="Старое сообщение",
            record={
                "message": {
                    "channel": "telegram",
                    "brand": "foton",
                    "message_id": "old-message",
                    "message_type": "text",
                    "text": "Старое сообщение",
                }
            },
            metadata={"profile_id": "p-tg", "chat_id": "123456", "message_id": "old-message", "brand": "foton"},
        ),
        actor="test",
    )
    store.close()
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "resolved",
                "contact_id": "2002",
                "lead_ids": ("1001",),
                "resolution_source": "wappi_amo_widget",
            }
        },
    )
    prime = resolver.prime_widget_chat_resolutions(
        (WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),)
    )
    records = load_existing_unmatched_wappi_records(
        db_path,
        tenant_id="foton",
        chat_resolutions=resolver.widget_chat_resolutions,
    )

    assert prime["resolved"] == 1
    assert len(records) == 1
    assert records[0].payload["resolved_customer_id"] == customer_id


def test_wappi_inbound_email_is_only_identity_candidate_not_sender_truth(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:email",
        link_type="email",
        link_value="parent@example.com",
        brand="foton",
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_event(
        TimelineEvent(
            tenant_id="foton",
            event_type=TimelineEventType.TELEGRAM_MESSAGE,
            event_at=datetime.fromtimestamp(1_753_000_000, tz=timezone.utc),
            source_system="wappi_telegram",
            source_id="email-message",
            source_ref="wappi:email-message",
            direction=TimelineDirection.INBOUND,
            text_preview="Моя почта parent@example.com",
            record={
                "message": {
                    "channel": "telegram",
                    "brand": "foton",
                    "message_id": "email-message",
                    "message_type": "text",
                    "text": "Моя почта parent@example.com",
                }
            },
            metadata={"profile_id": "p-tg", "chat_id": "123456", "message_id": "email-message", "brand": "foton"},
        ),
        actor="test",
    )
    store.close()
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "missing",
                "contact_id": "",
                "lead_ids": (),
            }
        },
    )

    report = resolver.prime_existing_message_identity_resolutions(
        (WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),)
    )
    records = load_existing_unmatched_wappi_records(
        db_path,
        tenant_id="foton",
        chat_resolutions=resolver.chat_resolutions,
    )

    assert report["candidate_unique"] == 1
    assert records == ()


def test_wappi_missing_personal_chat_becomes_bot_blocked_provisional_family() -> None:
    profile_spec = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    message = WappiHistoryMessage(
        profile_id="p-tg",
        chat_id="123456",
        message_id="message-1",
        text="Здравствуйте",
        message_type="text",
        timestamp=1_753_000_000,
        from_me=False,
        contact_name="Родитель",
    )
    record = wappi_message_to_record(
        profile=profile_spec,
        message=message,
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:provisional",
            reason="provisional_wappi_family",
            resolution_source="wappi_provisional",
            evidence={"provisional_wappi_family": True, "brand_context_authorized": False},
        ),
    )

    batch = WappiHistoryTimelineNormalizer(
        tenant_id="foton",
        source_system="wappi_telegram",
    ).normalize(record)

    assert batch.customers[0].identity_status == IdentityStatus.PARTIAL
    assert batch.customers[0].metadata["provisional_wappi_family"] is True
    assert batch.identity_links[0].match_class == IdentityMatchClass.INFERRED
    assert batch.events[0].match_status == IdentityMatchClass.INFERRED
    assert batch.events[0].metadata["pending_attribution"] is True
    assert batch.bot_context_chunks == ()
    assert batch.conflicts[0]["metadata"]["resolution_status"] == "provisional_wappi_family"


def test_wappi_missing_widget_row_gets_stable_provisional_resolution(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path):
        pass
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "missing",
                "contact_id": "",
                "lead_ids": (),
            }
        },
    )
    profile_spec = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")

    first = resolver.prime_provisional_chat_resolutions((profile_spec,))
    first_resolution = resolver.chat_resolutions[("wappi_telegram", "p-tg", "123456")]
    second = resolver.prime_provisional_chat_resolutions((profile_spec,))
    second_resolution = resolver.chat_resolutions[("wappi_telegram", "p-tg", "123456")]

    assert first == {"created": 1}
    assert second == {"already_resolved": 1}
    assert first_resolution.customer_id == second_resolution.customer_id
    assert first_resolution.resolution_source == "wappi_provisional"
    assert first_resolution.evidence["brand_context_authorized"] is False


def test_wappi_exact_widget_link_can_upgrade_existing_provisional_family(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    profile_spec = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    message = WappiHistoryMessage(
        profile_id="p-tg",
        chat_id="123456",
        message_id="message-1",
        text="Здравствуйте",
        message_type="text",
        timestamp=1_753_000_000,
        from_me=False,
    )
    provisional_record = wappi_message_to_record(
        profile=profile_spec,
        message=message,
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:provisional",
            reason="provisional_wappi_family",
            resolution_source="wappi_provisional",
            evidence={"provisional_wappi_family": True, "brand_context_authorized": False},
        ),
    )
    batch = WappiHistoryTimelineNormalizer(
        tenant_id="foton",
        source_system="wappi_telegram",
    ).normalize(provisional_record)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(batch.customers[0], actor="test")
        store.upsert_identity_link(batch.identity_links[0], actor="test")
        store.upsert_event(batch.events[0], actor="test")
    real_customer = seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:real",
        lead_id="1001",
        contact_id="2002",
    )
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "resolved",
                "contact_id": "2002",
                "lead_ids": ("1001",),
                "resolution_source": "wappi_amo_widget",
            }
        },
    )

    prime = resolver.prime_widget_chat_resolutions((profile_spec,))
    records = load_existing_unmatched_wappi_records(
        db_path,
        tenant_id="foton",
        chat_resolutions=resolver.chat_resolutions,
    )

    assert prime["resolved"] == 1
    assert len(records) == 1
    assert records[0].payload["resolved_customer_id"] == real_customer
    assert records[0].payload["identity_authority"] == "wappi_amo_widget"


def test_exact_owner_is_not_replaced_when_its_customer_shell_is_provisional(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    profile_spec = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    provisional_record = wappi_message_to_record(
        profile=profile_spec,
        message=WappiHistoryMessage(
            profile_id="p-tg",
            chat_id="123456",
            message_id="message-1",
            text="Здравствуйте",
            message_type="text",
            timestamp=1_753_000_000,
            from_me=False,
        ),
        resolution=WappiChatResolution(
            status="resolved",
            customer_id="customer:provisional",
            reason="provisional_wappi_family",
            resolution_source="wappi_provisional",
            evidence={"provisional_wappi_family": True, "brand_context_authorized": False},
        ),
    )
    batch = WappiHistoryTimelineNormalizer(tenant_id="foton", source_system="wappi_telegram").normalize(
        provisional_record
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(batch.customers[0], actor="test")
        store.upsert_identity_link(batch.identity_links[0], actor="test")
        store.upsert_event(batch.events[0], actor="test")
    with sqlite3.connect(db_path) as con:
        raw = con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (batch.events[0].source_id,)).fetchone()[0]
        payload = json.loads(raw)
        payload["metadata"]["identity_authority"] = "wappi_amo_widget"
        con.execute(
            "UPDATE timeline_events SET record_json=? WHERE source_id=?",
            (json.dumps(payload, ensure_ascii=False), batch.events[0].source_id),
        )
    seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:real",
        lead_id="1001",
        contact_id="2002",
    )
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "resolved",
                "contact_id": "2002",
                "lead_ids": ("1001",),
                "resolution_source": "amo_talk_authoritative",
            }
        },
    )

    resolver.prime_widget_chat_resolutions((profile_spec,))
    resolution = resolver.chat_resolutions[("wappi_telegram", "p-tg", "123456")]

    assert resolution.status == "pending_attribution"
    assert resolution.reason == "existing_wappi_chat_customer_conflict"
    assert load_existing_unmatched_wappi_records(
        db_path,
        tenant_id="foton",
        chat_resolutions=resolver.chat_resolutions,
    ) == ()


def test_wappi_exact_widget_link_overrides_older_non_widget_assignment(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    old_customer = seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:old",
        lead_id="",
        contact_id="",
    )
    profile_spec = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    record = wappi_message_to_record(
        profile=profile_spec,
        message=WappiHistoryMessage(
            profile_id="p-tg",
            chat_id="123456",
            message_id="message-1",
            text="Здравствуйте",
            message_type="text",
            timestamp=1_753_000_000,
            from_me=False,
        ),
        resolution=WappiChatResolution(
            status="resolved",
            customer_id=old_customer,
            resolution_source="timeline_identity",
            evidence={"brand_context_authorized": True},
        ),
    )
    batch = WappiHistoryTimelineNormalizer(
        tenant_id="foton",
        source_system="wappi_telegram",
    ).normalize(record)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_identity_link(batch.identity_links[0], actor="test")
        store.upsert_event(batch.events[0], actor="test")
    real_customer = seed_customer_with_amo(
        db_path,
        tmp_path,
        customer_id="customer:real",
        lead_id="1001",
        contact_id="2002",
    )
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={},
        widget_links={
            ("telegram", "p-tg", "123456"): {
                "status": "resolved",
                "contact_id": "2002",
                "lead_ids": ("1001",),
                "resolution_source": "wappi_amo_widget",
            }
        },
    )

    prime = resolver.prime_widget_chat_resolutions((profile_spec,))
    records = load_existing_unmatched_wappi_records(
        db_path,
        tenant_id="foton",
        chat_resolutions=resolver.chat_resolutions,
    )

    assert prime["resolved"] == 1
    assert len(records) == 1
    assert records[0].payload["resolved_customer_id"] == real_customer


def test_wappi_provisional_cleanup_removes_only_orphan_shell(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for customer_id in ("customer:orphan", "customer:referenced"):
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status=IdentityStatus.PARTIAL,
                    source_ref=f"wappi_provisional:{customer_id}",
                    metadata={"provisional_wappi_family": True},
                ),
                actor="test",
            )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:referenced",
                event_type=TimelineEventType.TELEGRAM_MESSAGE,
                event_at=datetime.fromtimestamp(1_753_000_000, tz=timezone.utc),
                source_system="wappi_telegram",
                source_id="referenced-message",
                direction=TimelineDirection.INBOUND,
            ),
            actor="test",
        )

    report = remove_orphaned_provisional_customers(
        db_path,
        tenant_id="foton",
        customer_ids=("customer:orphan", "customer:referenced"),
    )

    assert report == {"candidates": 2, "removed": 1, "retained_with_references": 1}
    with sqlite3.connect(db_path) as con:
        ids = {str(row[0]) for row in con.execute("SELECT customer_id FROM customer_identities")}
    assert "customer:orphan" not in ids
    assert "customer:referenced" in ids


def test_wappi_required_widget_missing_profile_does_not_fall_back(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    resolver = WappiPairCustomerResolver.from_store(
        db_path,
        tenant_id="foton",
        pairs={
            DraftLoopKey("p-tg", "123456"): DraftLoopPair(
                key=DraftLoopKey("p-tg", "123456"),
                lead_id="1001",
                expected_brand="foton",
                contact_id="2002",
            )
        },
        widget_required=True,
    )

    resolution = resolver.resolve_chat(
        profile=profile("p-tg", "foton", "telegram"),
        dialog={"id": "123456", "type": "user"},
        messages=(),
    )

    assert resolution.resolved is False
    assert resolution.reason == "wappi_widget_unavailable"
    assert resolution.resolution_source == "wappi_amo_widget"


def seed_customer_with_amo(
    db_path: Path,
    allowed_root: Path,
    *,
    customer_id: str = "customer:known",
    lead_id: str = "1001",
    contact_id: str = "2002",
    brand: str = "foton",
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
    store.upsert_opportunity(
        CustomerOpportunity(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_type=OpportunityType.AMO_DEAL,
            source_system="amocrm_snapshot",
            source_id=f"brand:{customer_id}",
            title="Brand evidence",
            product_context={"brand": brand},
            opened_at=datetime.now(timezone.utc),
            confidence=1.0,
        ),
        actor="test",
    )
    store.close()
    return customer.customer_id


def seed_local_identity(
    db_path: Path,
    allowed_root: Path,
    *,
    customer_id: str,
    link_type: str,
    link_value: str,
    brand: str,
) -> None:
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root)
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id=customer_id,
            identity_status=IdentityStatus.STRONG,
            source_ref=f"synthetic:{customer_id}",
        ),
        actor="test",
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer_id,
            link_type=link_type,
            link_value=link_value,
            source_system="synthetic",
            source_ref=f"synthetic:{link_type}:{customer_id}",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
        ),
        actor="test",
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer_id,
            link_type="amo_contact_id",
            link_value=customer_id,
            source_system="amocrm_snapshot",
            source_ref=f"amocrm:contact:{customer_id}",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
        ),
        actor="test",
    )
    store.upsert_opportunity(
        CustomerOpportunity(
            tenant_id="foton",
            customer_id=customer_id,
            opportunity_type=OpportunityType.AMO_DEAL,
            source_system="amocrm_snapshot",
            source_id=f"brand:{customer_id}",
            title="Brand evidence",
            product_context={"brand": brand},
            opened_at=datetime.now(timezone.utc),
            confidence=1.0,
        ),
        actor="test",
    )
    store.close()


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
