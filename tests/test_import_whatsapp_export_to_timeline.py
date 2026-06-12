from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mango_mvp.customer_timeline.contracts import CustomerIdentity, IdentityStatus
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from scripts.import_whatsapp_export_to_timeline import (
    WhatsAppImportConfig,
    main,
    parse_whatsapp_export_text,
    run_whatsapp_import,
)


def synthetic_whatsapp_export() -> str:
    return """===== CHAT: +7 999 111-22-33 =====

Whatsapp - +7 999 111-22-33
Chat history with +7 999 111-22-33
2025-04-30
09:20
You
First outbound line
second outbound line
09:21
Client One
Not supported WhatsApp internal message
09:22
Client One
Inbound same minute one
09:22
You
Outbound same minute two
2025-05-01
10:00
Client One
Next day inbound
===== CHAT: Maria Client =====

Whatsapp - Maria Client
Chat history with Maria Client
2025-06-05
14:35
Maria Client
Hello, I need schedule info
14:36
You
Sure, I will check
14:37
Maria Client
Multi line question
line two
14:38
Maria Client
Not supported WhatsApp internal message
14:39
Maria Client
Phone is hidden in this chat
14:40
You
Final outbound
"""


def write_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "all_whatsapp_chats.txt"
    source.write_text(synthetic_whatsapp_export(), encoding="utf-8")
    return source


def test_parser_handles_multiline_outbound_service_skip_two_chats_and_phone_links(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)

    records, stats = parse_whatsapp_export_text(source.read_text(encoding="utf-8"), source_path=source, brand="foton")

    payloads = [record.payload for record in records]
    assert stats.to_json_dict() == {
        "chats_seen": 2,
        "unique_chats": 2,
        "messages_seen": 11,
        "records_built": 9,
        "skipped_service": 2,
        "skipped_empty": 0,
        "skipped_malformed": 0,
        "linked_by_phone": 4,
        "session_only": 5,
        "chats_linked_by_phone": 1,
        "chats_session_only": 1,
    }
    assert len(records) == 9
    assert payloads[0]["direction"] == "outbound"
    assert payloads[0]["text"] == "First outbound line\nsecond outbound line"
    assert payloads[0]["chat_phone"] == "+79991112233"
    assert payloads[0]["channel_thread_id"] == "+79991112233"
    assert records[0].source_ref == "whatsapp:+79991112233:2025-04-30T09:20:1"
    assert "Not supported WhatsApp internal message" not in "\n".join(str(item["text"]) for item in payloads)
    same_minute = [record.source_ref for record in records if "2025-04-30T09:22" in record.source_ref]
    assert same_minute == [
        "whatsapp:+79991112233:2025-04-30T09:22:1",
        "whatsapp:+79991112233:2025-04-30T09:22:2",
    ]
    assert payloads[-1]["direction"] == "outbound"
    assert payloads[-1]["chat_phone"] is None


def test_parser_default_brand_is_unpk_and_channel_shared(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)

    records, _stats = parse_whatsapp_export_text(source.read_text(encoding="utf-8"), source_path=source)

    payload = records[0].payload
    assert payload["brand_hint"] == "unpk"
    assert payload["channel_shared"] is True


def test_cli_defaults_to_dry_run_and_does_not_create_timeline_db(tmp_path: Path, capsys) -> None:
    source = write_fixture(tmp_path)
    timeline_db = tmp_path / "customer_timeline.sqlite"

    rc = main(
        [
            "--source",
            str(source),
            "--timeline-db",
            str(timeline_db),
            "--allowed-root",
            str(tmp_path),
            "--tenant-id",
            "foton",
            "--brand",
            "unpk",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert timeline_db.exists() is False
    assert report["mode"] == "dry_run_preview"
    assert report["dry_run"] is True
    assert report["summary"]["records_loaded"] == 9
    assert report["summary"]["write_applied"] is False
    assert report["summary"]["linked_by_phone"] == 4
    assert report["summary"]["session_only"] == 5
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["write_tallanto"] is False
    assert report["safety"]["send_messenger"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["run_ra"] is False
    assert report["safety"]["write_product_timeline_db"] is False


def test_malformed_whatsapp_fragment_is_reported_as_warning_not_failed_batch(tmp_path: Path) -> None:
    source = tmp_path / "all_whatsapp_chats.txt"
    source.write_text(
        """===== CHAT: +7 999 111-22-33 =====
2025-04-30
09:20
===== CHAT: +7 999 111-22-33 =====
2025-04-30
09:21
Client One
Valid inbound after malformed fragment
""",
        encoding="utf-8",
    )

    report = run_whatsapp_import(
        WhatsAppImportConfig(
            source=source,
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            brand="unknown",
            apply=False,
        )
    )

    assert report["validation_ok"] is True
    assert report["summary"]["status"] == "completed_with_warnings"
    assert report["summary"]["records_loaded"] == 1
    assert report["summary"]["skipped_malformed"] == 1


def test_apply_import_is_idempotent_and_adds_phone_identity_link(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    timeline_db = tmp_path / "customer_timeline.sqlite"
    config = WhatsAppImportConfig(
        source=source,
        timeline_db=timeline_db,
        allowed_root=tmp_path,
        tenant_id="foton",
        brand="unknown",
        apply=True,
    )

    first = run_whatsapp_import(config)
    second = run_whatsapp_import(config)

    store = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    try:
        summary = store.summary()
        phone_links = store.list_identity_links("foton", link_type="phone", link_value="+79991112233")
    finally:
        store.close()
    with sqlite3.connect(timeline_db) as con:
        chunk_payload = json.loads(con.execute("SELECT record_json FROM bot_context_chunks LIMIT 1").fetchone()[0])
    assert first["validation_ok"] is True
    assert second["validation_ok"] is True
    assert summary["counts"]["timeline_events"] == 9
    assert summary["counts"]["bot_context_chunks"] == 9
    assert summary["counts"]["ingestion_runs"] == 1
    assert len(phone_links) == 4
    assert second["import_report"]["write_status_counts"]["duplicate"] >= 9
    assert "brand:unknown" in chunk_payload["relevance_tags"]
    assert "unknown" not in chunk_payload["relevance_tags"]
    assert "channel_shared:true" in chunk_payload["relevance_tags"]


def test_apply_import_links_phone_chat_to_existing_customer_without_duplicate_profile(tmp_path: Path) -> None:
    source = tmp_path / "all_whatsapp_chats.txt"
    source.write_text(
        """===== CHAT: +7 999 111-22-33 =====
2025-04-30
09:20
Client One
Inbound question
""",
        encoding="utf-8",
    )
    timeline_db = tmp_path / "customer_timeline.sqlite"
    existing = CustomerIdentity(
        tenant_id="foton",
        customer_id="existing-customer",
        identity_status=IdentityStatus.STRONG,
        display_name="Existing",
        primary_phone="+79991112233",
        source_ref="seed",
    )
    store = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    try:
        store.upsert_customer(existing)
    finally:
        store.close()

    report = run_whatsapp_import(
        WhatsAppImportConfig(
            source=source,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            brand="unknown",
            apply=True,
        )
    )

    with sqlite3.connect(timeline_db) as con:
        customer_count = con.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0]
        event_customer_id = con.execute("SELECT customer_id FROM timeline_events").fetchone()[0]

    assert report["links"]["unique_existing_phone_matches"] == 1
    assert customer_count == 1
    assert event_customer_id == "existing-customer"


def test_ambiguous_phone_match_is_counted_without_first_match_merge(tmp_path: Path) -> None:
    source = tmp_path / "all_whatsapp_chats.txt"
    source.write_text(
        """===== CHAT: +7 999 111-22-33 =====
2025-04-30
09:20
Client One
Inbound question
""",
        encoding="utf-8",
    )
    timeline_db = tmp_path / "customer_timeline.sqlite"
    existing_ids = seed_duplicate_phone_customers(timeline_db, tmp_path, phone="+79991112233")

    report = run_whatsapp_import(
        WhatsAppImportConfig(
            source=source,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            brand="unknown",
            apply=True,
        )
    )

    with sqlite3.connect(timeline_db) as con:
        event_customer_id = con.execute("SELECT customer_id FROM timeline_events").fetchone()[0]

    assert report["links"]["unique_existing_phone_matches"] == 0
    assert report["links"]["ambiguous_phone_matches"] == 1
    assert event_customer_id not in existing_ids


def test_phone_and_non_phone_chats_do_not_crash_and_persist_expected_source_ids(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    timeline_db = tmp_path / "customer_timeline.sqlite"

    report = run_whatsapp_import(
        WhatsAppImportConfig(
            source=source,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            brand="foton",
            apply=True,
        )
    )

    with sqlite3.connect(timeline_db) as con:
        rows = con.execute(
            "SELECT source_id, direction, record_json FROM timeline_events ORDER BY event_at, source_id"
        ).fetchall()
    source_ids = [row[0] for row in rows]
    assert report["parser"]["chats_session_only"] == 1
    assert "whatsapp:+79991112233:2025-04-30T09:20:1" in source_ids
    assert "whatsapp:Maria Client:2025-06-05T14:40:1" in source_ids
    maria_record = next(json.loads(row[2]) for row in rows if row[0] == "whatsapp:Maria Client:2025-06-05T14:40:1")
    assert maria_record["record"]["message"]["chat_phone"] is None
    assert maria_record["record"]["message"]["brand_hint"] == "foton"


def seed_duplicate_phone_customers(db_path: Path, allowed_root: Path, *, phone: str) -> set[str]:
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root)
    customers = [
        CustomerIdentity(
            tenant_id="foton",
            customer_id="existing-customer-1",
            identity_status=IdentityStatus.STRONG,
            display_name="Existing 1",
            primary_phone=phone,
            source_ref="amocrm:contact:1",
        ),
        CustomerIdentity(
            tenant_id="foton",
            customer_id="existing-customer-2",
            identity_status=IdentityStatus.STRONG,
            display_name="Existing 2",
            primary_phone=phone,
            source_ref="amocrm:contact:2",
        ),
    ]
    try:
        for customer in customers:
            store.upsert_customer(customer, actor="test")
    finally:
        store.close()
    return {customer.customer_id for customer in customers}
