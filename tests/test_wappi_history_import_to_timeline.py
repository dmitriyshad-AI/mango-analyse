from __future__ import annotations

import json
import sqlite3
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
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.wappi_history_import import (
    WappiFetchLimits,
    WappiChatResolution,
    WappiHistoryImportConfig,
    WappiHistoryTimelineNormalizer,
    WappiPairCustomerResolver,
    WappiProfileSpec,
    assert_readonly_wappi_client,
    open_readonly_sqlite,
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


def test_wappi_history_import_resolves_by_amo_pair_and_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    phase1 = write_phase1_config(tmp_path)
    pairs = write_pairs(tmp_path, lead_id="1001", contact_id="2002", chat_id="123456")
    client = FakeWappiClient(
        {
            "p-tg": [{"id": "123456", "type": "user"}],
            "p-max": [],
        },
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "m-1", "chat_id": "123456", "type": "text", "body": "Здравствуйте, нужен курс", "time": 1_753_000_000},
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
    assert first["provenance"]["input_hashes"]["phase1_config"]
    assert first["provenance"]["input_hashes"]["pairs_file"]
    assert "tracked_diff_sha256" in first["provenance"]["worktree"]
    assert second["writes"]["status_counts"]["duplicate"] >= first["writes"]["status_counts"]["created"]
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


def test_wappi_history_timeline_identity_apply_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_local_identity(
        db_path,
        tmp_path,
        customer_id="customer:timeline",
        link_type="telegram_user_id",
        link_value="123456",
        brand="foton",
    )
    client = FakeWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "123456"): [
                {
                    "id": "timeline-message-1",
                    "chat_id": "123456",
                    "type": "text",
                    "body": "Здравствуйте",
                    "time": 1_753_000_000,
                }
            ]
        },
    )
    config = WappiHistoryImportConfig(
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
    )

    first = run_wappi_history_import(config, client=client)
    second = run_wappi_history_import(config, client=client)

    assert first["validation_ok"] is True
    assert first["summary"]["linked_by_timeline"] == 1
    assert first["writes"]["status_counts"]["created"] > 0
    assert second["writes"]["status_counts"].get("created", 0) == 0
    assert second["writes"]["status_counts"].get("updated", 0) == 0
    assert second["summary"]["blocked_customer_relink_conflicts"] == 0


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


def test_wappi_require_nonempty_profile_blocks_all_apply(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    client = FakeWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {("telegram", "p-tg", "123456"): [{"id": "m-1", "chat_id": "123456", "type": "text", "body": "Тест", "time": 1_753_000_000}]},
    )
    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=write_phase1_config(tmp_path),
            pairs_file=write_pairs(tmp_path, lead_id="1001", contact_id="2002", chat_id="123456"),
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
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            [
                {"profile_id": "p-tg", "chat_id": "123456", "lead_id": "1001", "contact_id": "2002", "expected_brand": "foton"},
                {"profile_id": "p-max", "chat_id": "max-1", "lead_id": "3003", "contact_id": "4004", "expected_brand": "unpk"},
            ]
        ),
        encoding="utf-8",
    )
    client = FakeWappiClient(
        {
            "p-tg": [{"id": "123456", "type": "user"}],
            "p-max": [{"id": "max-1", "phone": "+7 999 000-00-01"}],
        },
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "tg-1", "chat_id": "123456", "type": "text", "body": "Telegram", "time": 1_753_000_001}
            ],
            ("max", "p-max", "max-1"): [
                {"id": "max-1", "chat_id": "max-1", "type": "text", "body": "Max", "time": 1_753_000_002}
            ],
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
                pairs_file=pairs,
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
        event = con.execute(
            "SELECT customer_id, match_status, record_json FROM timeline_events"
        ).fetchone()
        assert event[0] is None
        assert event[1] == "unmatched"
        assert json.loads(event[2])["metadata"]["pending_attribution"] is True
        assert con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM timeline_conflicts WHERE conflict_type='pending_attribution'").fetchone()[0] == 1


def test_wappi_history_import_resolves_missing_pair_through_amo_auto_tg(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    phase1 = write_phase1_config(tmp_path)
    client = FakeWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "m-1", "chat_id": "123456", "type": "text", "body": "Здравствуйте", "time": 1_753_000_000},
            ]
        },
    )
    auto = AmoAutoResolver(
        client=FakeMcp(contacts=[amo_contact("2002", telegram_id="123456", leads=("1001",))], leads=[amo_lead("1001", org="Фотон")]),
        shared_phone_stoplist={"+79990000000"},
        require_known_brand=True,
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            amo_auto_resolver_enabled=True,
            apply=True,
            limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
        ),
        client=client,
        amo_auto_resolver=auto,
    )

    assert report["validation_ok"] is True
    assert report["summary"]["linked_by_amo_auto"] == 1
    assert report["summary"]["pending_attribution"] == 0
    assert report["profiles"]["p-tg"]["coverage_counts"]["tg_chat_id_digit"] == 1
    event = fetch_one_json(db_path, "timeline_events")
    assert event["customer_id"] == customer_id
    assert event["metadata"]["identity_authority"] == "amo_auto_resolver"
    assert event["record"]["message"]["allowed_for_bot"] is False


def test_wappi_history_import_closes_stale_pending_after_auto_resolve(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    client = FakeWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "m-1", "chat_id": "123456", "type": "text", "body": "Здравствуйте", "time": 1_753_000_000},
            ]
        },
    )

    pending = run_wappi_history_import(
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
    assert pending["summary"]["pending_attribution"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_conflicts WHERE status='open'").fetchone()[0] == 1

    seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    auto = AmoAutoResolver(
        client=FakeMcp(contacts=[amo_contact("2002", telegram_id="123456", leads=("1001",))], leads=[amo_lead("1001", org="Фотон")]),
        shared_phone_stoplist={"+79990000000"},
        require_known_brand=True,
    )

    resolved = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            amo_auto_resolver_enabled=True,
            apply=True,
            limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
        ),
        client=client,
        amo_auto_resolver=auto,
    )

    assert resolved["summary"]["linked_by_amo_auto"] == 1
    assert resolved["stale_conflict_cleanup"]["resolved_pending_conflicts_closed"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM timeline_conflicts WHERE status='open'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM timeline_conflicts WHERE status='resolved'").fetchone()[0] == 1


def test_wappi_history_import_fails_closed_for_max_without_stoplist(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    phase1 = write_phase1_config(tmp_path)
    client = FakeWappiClient(
        {"p-tg": [], "p-max": [{"id": "max-chat-1", "phone": "+7 999 000-00-01"}]},
        {
            ("max", "p-max", "max-chat-1"): [
                {"id": "m-1", "chat_id": "max-chat-1", "type": "text", "body": "Добрый день", "time": 1_753_000_000},
            ]
        },
    )
    auto = AmoAutoResolver(
        client=FakeMcp(contacts=[amo_contact("2002", phone="+79990000001", leads=("1001",))], leads=[amo_lead("1001", org="УНПК")]),
        shared_phone_stoplist=set(),
        stoplist_error="shared_phone_stoplist_unavailable",
        require_known_brand=True,
    )

    report = run_wappi_history_import(
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            pairs_file=None,
            auto_pairs_file=None,
            amo_auto_resolver_enabled=True,
            apply=True,
            limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
        ),
        client=client,
        amo_auto_resolver=auto,
    )

    assert report["validation_ok"] is True
    assert report["summary"]["linked_by_amo_auto"] == 0
    assert report["summary"]["pending_attribution"] == 1
    assert report["profiles"]["p-max"]["coverage_counts"]["shared_phone_stoplist_unavailable"] == 1
    assert report["records"]["by_resolution_reason"]["shared_phone_stoplist_unavailable"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM timeline_events WHERE customer_id IS NULL AND match_status='unmatched'"
        ).fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM timeline_conflicts WHERE conflict_type='pending_attribution'").fetchone()[0] == 1


def test_wappi_pending_event_relinks_without_duplicate(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer_id = seed_customer_with_amo(db_path, tmp_path, lead_id="1001", contact_id="2002")
    phase1 = write_phase1_config(tmp_path)
    empty_pairs = tmp_path / "empty_pairs.json"
    empty_pairs.write_text('{"pairs": []}\n', encoding="utf-8")
    client = FakeWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {("telegram", "p-tg", "123456"): [{"id": "m-1", "body": "Цена?", "time": 1_753_000_000}]},
    )
    base = dict(
        timeline_db=db_path,
        allowed_root=tmp_path,
        phase1_config=phase1,
        auto_pairs_file=None,
        apply=True,
        limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
    )

    first = run_wappi_history_import(WappiHistoryImportConfig(pairs_file=empty_pairs, **base), client=client)
    pairs = write_pairs(tmp_path, lead_id="1001", contact_id="2002", chat_id="123456")
    second = run_wappi_history_import(WappiHistoryImportConfig(pairs_file=pairs, **base), client=client)

    assert first["summary"]["pending_attribution"] == 1
    assert second["summary"]["linked_by_pair"] == 1
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


def test_wappi_history_import_blocks_relinking_existing_source_to_other_customer(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_amo(db_path, tmp_path, customer_id="customer:first", lead_id="1001", contact_id="2002")
    seed_customer_with_amo(db_path, tmp_path, customer_id="customer:second", lead_id="9001", contact_id="9002")
    phase1 = write_phase1_config(tmp_path)
    client = FakeWappiClient(
        {"p-tg": [{"id": "123456", "type": "user"}], "p-max": []},
        {
            ("telegram", "p-tg", "123456"): [
                {"id": "m-1", "chat_id": "123456", "type": "text", "body": "Здравствуйте", "time": 1_753_000_000},
            ]
        },
    )

    base_config = WappiHistoryImportConfig(
        timeline_db=db_path,
        allowed_root=tmp_path,
        phase1_config=phase1,
        pairs_file=None,
        auto_pairs_file=None,
        amo_auto_resolver_enabled=True,
        apply=True,
        limits=WappiFetchLimits(chat_limit_per_profile=5, messages_per_chat=5, message_limit_total=20, sleep_seconds=0),
    )
    first = run_wappi_history_import(
        base_config,
        client=client,
        amo_auto_resolver=AmoAutoResolver(
            client=FakeMcp(contacts=[amo_contact("2002", telegram_id="123456", leads=("1001",))], leads=[amo_lead("1001", org="Фотон")]),
            shared_phone_stoplist={"+79990000000"},
            require_known_brand=True,
        ),
    )
    second = run_wappi_history_import(
        base_config,
        client=client,
        amo_auto_resolver=AmoAutoResolver(
            client=FakeMcp(contacts=[amo_contact("9002", telegram_id="123456", leads=("9001",))], leads=[amo_lead("9001", org="Фотон")]),
            shared_phone_stoplist={"+79990000000"},
            require_known_brand=True,
        ),
    )

    assert first["summary"]["linked_by_amo_auto"] == 1
    assert second["summary"]["blocked_chat_relink_conflicts"] == 1
    assert second["profiles"]["p-tg"]["linked_by_amo_auto"] == 0
    assert second["profiles"]["p-tg"]["pending_attribution"] == 1
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT record_json FROM timeline_events WHERE source_system='wappi_telegram'").fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0]["record_json"])["customer_id"] == "customer:first"


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
        self.calls.append({"kind": "messages", "method": "GET", "channel": channel, "profile_id": profile_id, "chat_id": chat_id, "limit": limit, "offset": offset, "order": order, "mark_all": mark_all})
        items = self.messages.get((channel, profile_id, chat_id), [])
        return {"messages": items[offset : offset + limit]}


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
