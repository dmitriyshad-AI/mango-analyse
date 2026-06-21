from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    AmoSnapshotNormalizer,
    ChannelMessageNormalizer,
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    MailMessageNormalizer,
    MangoCallSummaryNormalizer,
    TallantoSnapshotNormalizer,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineSourceRecord,
    file_sha256,
    load_local_source_records,
    load_sqlite_source_records,
    rows_from_csv,
    timeline_ingestion_safety_contract,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
SHA = "b" * 64


class FixedClock:
    def __call__(self) -> datetime:
        return NOW


def test_tallanto_csv_import_is_idempotent_preserves_source_and_records_conflict(tmp_path: Path) -> None:
    source = tmp_path / "students.csv"
    source.write_text(
        "entity_id\tname\temail\tphone\tcourse\tupdated_at\n"
        "s1\tИван Петров\tparent@example.com\t+7 916 111-22-33\tЕГЭ математика\t2026-05-01T10:00:00+00:00\n"
        "s2\tМария Петрова\tparent@example.com\t+7 916 111-22-33\tЕГЭ русский\t2026-05-01T10:05:00+00:00\n",
        encoding="cp1251",
    )
    before = source_snapshot(source)
    records = load_local_source_records(
        source,
        allowed_root=tmp_path,
        source_system="tallanto_snapshot",
        csv_encoding="cp1251",
        observed_at=NOW,
    )
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)

    first = service.import_records(
        records,
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="students.csv",
        idempotency_key="students-v1",
        actor="test",
    )
    second = service.import_records(
        records,
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="students.csv",
        idempotency_key="students-v1",
        actor="test",
    )
    after = source_snapshot(source)
    summary = store.summary()
    email_links = store.list_identity_links("foton", link_type="email", link_value="parent@example.com")
    conflicts = store.list_audit_log("foton", entity_type="timeline_conflict")["items"]

    assert before == after
    assert first.validation_ok is True
    assert first.source_unchanged is True
    assert second.source_unchanged is True
    assert summary["counts"]["customer_identities"] == 2
    assert summary["counts"]["identity_links"] == 6
    assert summary["counts"]["timeline_conflicts"] == 1
    assert summary["counts"]["customer_id_mappings"] == 2
    assert summary["counts"]["ingestion_runs"] == 1
    assert len({item["customer_id"] for item in email_links}) == 2
    assert {item["reason"] for item in store.list_customer_id_mappings("foton")} == {"family_phone_ambiguous"}
    assert conflicts[0]["action"] == "timeline_conflict_created"
    assert second.write_status_counts["duplicate"] >= first.write_status_counts["created"]
    store.close()


def test_phone_identity_union_writes_complete_mapping_and_keeps_brand_history(tmp_path: Path) -> None:
    records = (
        TimelineSourceRecord(
            source_system="brand_test",
            source_ref="amo#1",
            payload={"source_id": "amo-1", "phone": "+7 916 222-33-44", "brand": "foton", "name": "Иван"},
            observed_at=NOW,
        ),
        TimelineSourceRecord(
            source_system="brand_test",
            source_ref="mango#1",
            payload={"source_id": "call-1", "phone": "+79162223344", "brand": "unpk", "name": "Иван"},
            observed_at=NOW,
        ),
    )
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    report = TimelineImportService(store).import_records(
        records,
        normalizer=BrandHistoryNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="brand-history",
        idempotency_key="brand-history-v1",
        actor="test",
    )
    customers = store.list_customers("foton", limit=10)["items"]
    mappings = store.list_customer_id_mappings("foton")
    links = store.list_identity_links("foton", link_type="phone", link_value="+79162223344")

    assert report.validation_ok is True
    assert report.normalized_counts["customer_id_mappings"] == 2
    assert store.summary()["counts"]["customer_identities"] == 1
    assert store.summary()["counts"]["timeline_conflicts"] == 0
    assert customers[0]["summary"]["brands"] == ["foton", "unpk"]
    assert {item["reason"] for item in mappings} == {"phone_identity_union"}
    assert {item["old_customer_id"] for item in mappings}
    assert {item["new_customer_id"] for item in mappings} == {customers[0]["customer_id"]}
    assert len({item["customer_id"] for item in links}) == 1
    store.close()


def test_phone_identity_union_uses_existing_store_customer_across_import_runs(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)
    amo = TimelineSourceRecord(
        source_system="amocrm_snapshot",
        source_ref="lead#1",
        payload={
            "entity_id": "lead-1",
            "entity_type": "lead",
            "name": "Сделка ЕГЭ",
            "phone": "+7 916 333-44-55",
            "updated_at": "2026-05-04T11:00:00+00:00",
        },
    )
    mango = TimelineSourceRecord(
        source_system="mango_processed_summary",
        source_ref="call#1",
        payload={
            "call_id": "call-1",
            "client_phone": "+79163334455",
            "call_at": "2026-05-04T12:00:00+00:00",
            "summary": "Клиент уточнил стоимость.",
        },
    )

    service.import_records(
        (amo,),
        normalizer=AmoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="amo-run",
        idempotency_key="amo-run",
        actor="test",
    )
    service.import_records(
        (mango,),
        normalizer=MangoCallSummaryNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="mango-run",
        idempotency_key="mango-run",
        actor="test",
    )

    customers = store.list_customers("foton", limit=10)["items"]
    customer_id = customers[0]["customer_id"]
    events = store.list_events_by_customer("foton", customer_id, limit=10)["items"]
    mappings = store.list_customer_id_mappings("foton")

    assert store.summary()["counts"]["customer_identities"] == 1
    assert store.summary()["counts"]["timeline_conflicts"] == 0
    assert {item["event_type"] for item in events} == {"amo_deal_stage", "mango_call"}
    assert {item["new_customer_id"] for item in mappings} == {customer_id}
    assert {item["reason"] for item in mappings} >= {"phone_identity_union", "unchanged"}
    store.close()


def test_dry_run_preview_is_deterministic_and_does_not_mutate_store(tmp_path: Path) -> None:
    source = tmp_path / "amocrm_entities.json"
    source.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "501",
                        "entity_type": "lead",
                        "name": "ЕГЭ математика",
                        "phone": "+79990000000",
                        "updated_at": "2026-05-02T10:00:00+00:00",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    records = load_local_source_records(source, allowed_root=tmp_path, source_system="amocrm_snapshot")
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    before_summary = store.summary()
    service = TimelineImportService(store)

    first = service.import_records(
        records,
        normalizer=AmoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="amo-dry-run",
        dry_run=True,
    )
    second = service.import_records(
        records,
        normalizer=AmoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="amo-dry-run",
        dry_run=True,
    )
    after_summary = store.summary()

    assert first.to_json_dict() == second.to_json_dict()
    assert first.dry_run is True
    assert first.run_id is None
    assert first.write_status_counts == {}
    assert first.normalized_counts["customers"] == 1
    assert first.normalized_counts["opportunities"] == 1
    assert before_summary["counts"] == after_summary["counts"]
    store.close()


def test_local_sqlite_source_loader_is_read_only_and_mail_import_uses_metadata_only(tmp_path: Path) -> None:
    source_db = tmp_path / "mail_archive.sqlite"
    with sqlite3.connect(source_db) as con:
        con.execute(
            """
            CREATE TABLE messages (
              message_id TEXT,
              message_date_iso TEXT,
              subject TEXT,
              from_email TEXT,
              to_email TEXT,
              text_preview TEXT,
              raw_eml_path TEXT,
              sha256 TEXT,
              raw_size_bytes INTEGER,
              resolved_customer_id TEXT,
              resolved_tallanto_id TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "m-1",
                "2026-05-03T09:00:00+00:00",
                "Стоимость курса",
                "client@example.com",
                "edu@kmipt.ru",
                "Подскажите стоимость курса",
                "/archive/raw/m-1.eml",
                SHA,
                2048,
                "customer:fresh-relink-1",
                "student-1",
            ),
        )
    before = source_snapshot(source_db)

    records = load_sqlite_source_records(
        source_db,
        allowed_root=tmp_path,
        source_system="mail_archive",
        table_name="messages",
        source_ref_column="message_id",
    )
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    report = TimelineImportService(store).import_records(
        records,
        normalizer=MailMessageNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="mail-sqlite",
        actor="test",
    )
    after = source_snapshot(source_db)
    event = store.search_timeline("foton", "стоимость")["items"][0]["record"]

    assert before == after
    assert records[0].source_ref == "m-1"
    assert report.validation_ok is True
    assert report.source_unchanged is True
    assert store.summary()["counts"]["event_artifacts"] == 1
    assert event["event_type"] == "email_message"
    assert "raw_eml_path" in str(event)
    with pytest.raises(ValueError, match="read-only"):
        load_sqlite_source_records(source_db, allowed_root=tmp_path, source_system="mail_archive", table_name="messages", where_sql="delete from messages")
    store.close()


def test_mail_normalizer_uses_fresh_relink_customer_id_and_ignores_inline_customer_id() -> None:
    batch = MailMessageNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mail_archive",
            source_ref="mail#1",
            payload={
                "message_sha256": SHA,
                "customer_id": "interim-inline-id-must-not-be-used",
                "resolved_customer_id": "customer:fresh-relink-42",
                "resolved_tallanto_id": "student-42",
                "from_email": "client@example.com",
                "to_email": "edu@kmipt.ru",
                "subject": "Стоимость курса",
                "text_preview": "Подскажите стоимость курса",
                "date_last": "2026-05-03T09:00:00+00:00",
                "allowed_for_bot": False,
            },
        )
    )

    assert batch.customers[0].customer_id == "customer:fresh-relink-42"
    assert batch.events[0].customer_id == "customer:fresh-relink-42"
    assert batch.events[0].source_id == SHA
    assert {link.link_type.value for link in batch.identity_links} == {"email", "tallanto_student_id"}
    assert "interim-inline-id-must-not-be-used" != batch.customers[0].customer_id


def test_mail_normalizer_does_not_overwrite_existing_seed_customer_identity() -> None:
    batch = MailMessageNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mail_archive",
            source_ref="mail#existing",
            payload={
                "message_sha256": SHA,
                "resolved_customer_id": "customer:from-seed-timeline",
                "resolved_customer_exists": True,
                "resolved_tallanto_id": "student-from-seed",
                "subject": "Стоимость курса",
                "date_last": "2026-05-03T09:00:00+00:00",
                "allowed_for_bot": False,
            },
        )
    )

    assert batch.customers == ()
    assert batch.events[0].customer_id == "customer:from-seed-timeline"
    assert batch.opportunities[0].customer_id == "customer:from-seed-timeline"
    assert {link.link_type.value for link in batch.identity_links} == {"tallanto_student_id"}


def test_mail_normalizer_without_fresh_relink_goes_to_pending_attribution_only() -> None:
    batch = MailMessageNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mail_archive",
            source_ref="mail#pending",
            payload={
                "message_sha256": SHA,
                "customer_id": "interim-inline-id-must-not-be-used",
                "from_email": "client@example.com",
                "to_email": "edu@kmipt.ru",
                "date_last": "2026-05-03T09:00:00+00:00",
                "relink_decision": "unmatched",
                "relink_reason": "duplicate_identity_value",
                "allowed_for_bot": False,
            },
        )
    )

    assert batch.customers == ()
    assert batch.identity_links == ()
    assert batch.opportunities == ()
    assert batch.events == ()
    assert batch.conflicts[0]["conflict_type"] == "pending_attribution"
    assert batch.conflicts[0]["metadata"]["relink_decision"] == "unmatched"


def test_mail_thread_opportunity_source_id_includes_customer_to_avoid_cross_customer_collision() -> None:
    normalizer = MailMessageNormalizer(tenant_id="foton")

    first = normalizer.normalize(
        TimelineSourceRecord(
            source_system="mail_archive",
            source_ref="mail#1",
            payload={
                "message_sha256": "a" * 64,
                "thread_id": "shared-thread",
                "resolved_customer_id": "customer:fresh-relink-a",
                "resolved_tallanto_id": "student-a",
                "allowed_for_bot": False,
            },
        )
    )
    second = normalizer.normalize(
        TimelineSourceRecord(
            source_system="mail_archive",
            source_ref="mail#2",
            payload={
                "message_sha256": "b" * 64,
                "thread_id": "shared-thread",
                "resolved_customer_id": "customer:fresh-relink-b",
                "resolved_tallanto_id": "student-b",
                "allowed_for_bot": False,
            },
        )
    )

    assert first.opportunities[0].source_id == "shared-thread:customer:fresh-relink-a"
    assert second.opportunities[0].source_id == "shared-thread:customer:fresh-relink-b"
    assert first.opportunities[0].source_id != second.opportunities[0].source_id


def test_mail_and_channel_sources_reject_allowed_for_bot_true() -> None:
    with pytest.raises(ValueError, match="allowed_for_bot=False"):
        MailMessageNormalizer(tenant_id="foton").normalize(
            TimelineSourceRecord(
                source_system="mail_archive",
                source_ref="mail#unsafe",
                payload={
                    "message_sha256": SHA,
                    "resolved_customer_id": "customer:fresh-relink-unsafe",
                    "resolved_tallanto_id": "student-unsafe",
                    "from_email": "client@example.com",
                    "to_email": "edu@kmipt.ru",
                    "date_last": "2026-05-03T09:00:00+00:00",
                    "allowed_for_bot": True,
                },
            )
        )

    with pytest.raises(ValueError, match="allowed_for_bot=False"):
        ChannelMessageNormalizer(tenant_id="foton").normalize(
            TimelineSourceRecord(
                source_system="channel_snapshot",
                source_ref="telegram#unsafe",
                payload={
                    "channel": "telegram",
                    "channel_thread_id": "thread-1",
                    "channel_message_id": "msg-1",
                    "channel_user_id": "tg-100",
                    "text": "Здравствуйте",
                    "allowed_for_bot": True,
                },
            )
        )

    with pytest.raises(ValueError, match="allowed_for_bot=False"):
        ChannelMessageNormalizer(tenant_id="foton").normalize(
            TimelineSourceRecord(
                source_system="telegram_history",
                source_ref="telegram_history#unsafe",
                payload={
                    "channel": "telegram",
                    "channel_thread_id": "thread-2",
                    "channel_message_id": "msg-2",
                    "channel_user_id": "tg-200",
                    "text": "Здравствуйте",
                    "allowed_for_bot": True,
                },
            )
        )


def test_channel_mango_and_amo_normalizers_create_expected_timeline_contracts() -> None:
    channel_batch = ChannelMessageNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="channel_snapshot",
            source_ref="telegram#1",
            payload={
                "channel": "telegram",
                "channel_thread_id": "thread-1",
                "channel_message_id": "msg-1",
                "channel_user_id": "tg-100",
                "direction": "inbound",
                "text": "Хочу узнать стоимость",
                "received_at": "2026-05-04T09:00:00+00:00",
            },
        )
    )
    max_batch = ChannelMessageNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="channel_snapshot",
            source_ref="max#1",
            payload={
                "channel": "max",
                "channel_thread_id": "max-thread-1",
                "channel_message_id": "max-msg-1",
                "channel_user_id": "max-user-1",
                "direction": "inbound",
                "text": "Нужна консультация по оплате",
                "received_at": "2026-05-04T09:05:00+00:00",
            },
        )
    )
    mango_batch = MangoCallSummaryNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mango_processed_summary",
            source_ref="call#1",
            payload={
                "call_id": "call-1",
                "client_phone": "+79991112233",
                "call_at": "2026-05-04T10:00:00+00:00",
                "summary": "Клиент интересуется оплатой.",
                "recommended_action": "Перезвонить завтра",
                "audio_path": "/audio/call-1.mp3",
                "audio_path_sha256": SHA,
            },
        )
    )
    amo_batch = AmoSnapshotNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="amocrm_snapshot",
            source_ref="lead#1",
            payload={
                "entity_id": "lead-1",
                "entity_type": "lead",
                "name": "Сделка ЕГЭ",
                "phone": "+79991112233",
                "status": "new",
                "updated_at": "2026-05-04T11:00:00+00:00",
            },
        )
    )

    assert channel_batch.events[0].event_type.value == "telegram_message"
    assert channel_batch.bot_context_chunks[0].allowed_for_bot is False
    assert channel_batch.bot_context_chunks[0].requires_manager_review is True
    assert max_batch.events[0].event_type.value == "max_message"
    assert max_batch.bot_context_chunks[0].allowed_for_bot is False
    assert max_batch.bot_context_chunks[0].requires_manager_review is True
    assert mango_batch.events[0].event_type.value == "mango_call"
    assert mango_batch.artifacts[0].artifact_type.value == "call_audio"
    assert mango_batch.signals[0].signal_type == "sales_next_step"
    assert amo_batch.opportunities[0].opportunity_type.value == "amo_deal"
    assert amo_batch.events[0].event_type.value == "amo_deal_stage"


def test_channel_message_normalizer_uses_whatsapp_contract_types() -> None:
    batch = ChannelMessageNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="channel_snapshot",
            source_ref="whatsapp#1",
            payload={
                "channel": "whatsapp",
                "channel_thread_id": "+7 999 111-22-33",
                "channel_message_id": "msg-1",
                "channel_user_id": "+7 999 111-22-33",
                "direction": "inbound",
                "text": "Здравствуйте",
                "received_at": "2026-05-04T09:00:00+00:00",
            },
        )
    )

    assert batch.events[0].event_type == TimelineEventType.WHATSAPP_MESSAGE
    assert {link.link_type.value for link in batch.identity_links} == {"whatsapp_user_id", "channel_session_id"}
    assert batch.bot_context_chunks[0].allowed_for_bot is False
    assert batch.bot_context_chunks[0].requires_manager_review is True


def test_importer_safety_contract_and_no_network_or_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network/subprocess must not be used")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(os, "system", fail)
    monkeypatch.setattr(socket, "socket", fail)
    source = tmp_path / "messages.jsonl"
    source.write_text(
        json.dumps(
            {
                "channel": "site_chat",
                "channel_thread_id": "thread-1",
                "channel_message_id": "msg-1",
                "channel_user_id": "user-1",
                "text": "Здравствуйте",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_local_source_records(source, allowed_root=tmp_path, source_system="channel_snapshot")
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    report = TimelineImportService(store).import_records(
        records,
        normalizer=ChannelMessageNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="channel-no-live",
    )
    safety = timeline_ingestion_safety_contract()

    assert report.validation_ok is True
    assert safety["network_calls"] is False
    assert safety["subprocess_calls"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["send_messenger"] is False
    assert safety["run_asr"] is False
    assert safety["run_ra"] is False
    assert safety["write_runtime_db"] is False
    assert safety["stable_runtime_writes"] is False
    assert safety["source_sqlite_mode"] == "mode=ro"
    assert safety["source_sqlite_query_only"] is True
    assert safety["source_db_attached_to_writer"] is False
    assert safety["identity_conflicts_auto_merge"] is False
    store.close()


def test_source_loader_rejects_stable_runtime_and_outside_paths(tmp_path: Path) -> None:
    source = tmp_path / "rows.json"
    source.write_text("[]", encoding="utf-8")
    outside = tmp_path.parent / "outside_timeline_source.json"
    outside.write_text("[]", encoding="utf-8")
    stable = tmp_path / "stable_runtime"
    stable.mkdir()
    stable_source = stable / "rows.json"
    stable_source.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="allowed root"):
        load_local_source_records(outside, allowed_root=tmp_path, source_system="amocrm_snapshot")
    with pytest.raises(ValueError, match="stable_runtime"):
        load_local_source_records(stable_source, allowed_root=tmp_path, source_system="amocrm_snapshot")


def test_rows_from_csv_detects_tab_cp1251_exports(tmp_path: Path) -> None:
    source = tmp_path / "students.csv"
    source.write_text("ID\tИмя\tE-mail\n1\tИван\tivan@example.com\n", encoding="cp1251")

    rows = rows_from_csv(source, encoding="utf-8-sig")

    assert rows == ({"ID": "1", "Имя": "Иван", "E-mail": "ivan@example.com"},)


def source_snapshot(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }


class BrandHistoryNormalizer:
    source_system = "brand_test"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = record.payload
        phone = str(payload["phone"])
        source_id = str(payload["source_id"])
        brand = str(payload["brand"])
        customer = CustomerIdentity(
            tenant_id=self.tenant_id,
            identity_status=IdentityStatus.STRONG,
            display_name=str(payload["name"]),
            primary_phone=phone,
            source_ref=record.source_ref,
            first_seen_at=NOW,
            last_seen_at=NOW,
            touch_count=1,
            summary={"source_system": self.source_system, "brand": brand},
            metadata={"brand": brand},
            created_at=NOW,
            updated_at=NOW,
        )
        link = IdentityLink(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            link_type="phone",
            link_value=phone,
            source_system=self.source_system,
            source_ref=record.source_ref,
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.95,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            event_type=TimelineEventType.AMO_CONTACT_SNAPSHOT,
            event_at=NOW,
            source_system=self.source_system,
            source_id=source_id,
            source_ref=record.source_ref,
            direction=TimelineDirection.SYSTEM,
            subject=f"{brand} snapshot",
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.9,
            record={"brand": brand},
            metadata={"brand": brand},
            created_at=NOW,
        )
        return TimelineNormalizedBatch(
            source_record=record,
            customers=(customer,),
            identity_links=(link,),
            events=(event,),
        )
