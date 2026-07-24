from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

import mango_mvp.customer_timeline.amo_incremental as amo_incremental_module
from mango_mvp.customer_timeline.amo_incremental import (
    AmoIncrementalConfig,
    event_summary,
    fetch_cards_source,
    fetch_collection,
    fetch_events_source,
    load_amo_link_index,
    run_amo_incremental,
)
from mango_mvp.customer_timeline.contracts import CustomerIdentity, IdentityLink, IdentityStatus
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.existing_clients.amo_step1_snapshot import AmoMcpError
from mango_mvp.customer_timeline.ingestion import TimelineSourceRecord
from mango_mvp.customer_timeline.nightly_incremental import (
    AmoEventNormalizer,
    IncrementalSourceConfig,
    normalizer_for_source,
)


NOW = datetime(2026, 6, 24, 8, 0, tzinfo=timezone.utc)


def test_run_amo_incremental_refuses_to_copy_over_explicit_timeline_db(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "staging.sqlite"
    sqlite3.connect(source).close()
    sqlite3.connect(target).close()

    with pytest.raises(ValueError, match="explicit timeline_db requires copy_db=False"):
        run_amo_incremental(
            AmoIncrementalConfig(
                source_db=source,
                out_root=tmp_path / "out",
                mcp_env=tmp_path / "missing.env",
                timeline_db=target,
            )
        )


def test_run_amo_incremental_rejects_prod_target_before_network_or_output(tmp_path, monkeypatch) -> None:
    out_root = tmp_path / "out"
    prod = tmp_path / "customer_timeline_prod_20260722" / "customer_timeline.sqlite"
    monkeypatch.setattr(
        amo_incremental_module,
        "read_mcp_env",
        lambda _path: pytest.fail("prod guard must run before reading MCP config"),
    )

    with pytest.raises(ValueError, match="snapshot-only"):
        run_amo_incremental(
            AmoIncrementalConfig(
                source_db=tmp_path / "source.sqlite",
                out_root=out_root,
                mcp_env=tmp_path / "amo.env",
                timeline_db=prod,
                allowed_root=tmp_path,
                copy_db=False,
            )
        )

    assert not out_root.exists()


@pytest.mark.parametrize("cap_source", ["leads", "contacts", "events"])
def test_run_amo_incremental_page_cap_writes_nothing_and_keeps_cursors(
    tmp_path, monkeypatch, cap_source
) -> None:
    db_path = tmp_path / "staging.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO ingestion_cursors (tenant_id, source_system, last_cursor_ts, updated_at, metadata_json)
            VALUES ('foton', 'amo_leads_updated_at', '2026-07-01T00:00:00+00:00',
                    '2026-07-01T00:00:00+00:00', '{}')
            """
        )
        con.commit()

    monkeypatch.setattr(amo_incremental_module, "read_mcp_env", lambda _path: object())
    monkeypatch.setattr(amo_incremental_module, "AmoMcpClient", lambda _config: object())

    def fake_collection(_client, **kwargs):
        return [], 1, cap_source == kwargs["path"]

    monkeypatch.setattr(amo_incremental_module, "fetch_collection", fake_collection)
    monkeypatch.setattr(
        amo_incremental_module,
        "fetch_events_collection",
        lambda *_args, **_kwargs: ([], 1, cap_source == "events"),
    )
    monkeypatch.setattr(
        amo_incremental_module,
        "run_nightly_incremental",
        lambda _config: pytest.fail("page-cap preflight must block before DB import"),
    )

    report = run_amo_incremental(
        AmoIncrementalConfig(
            source_db=db_path,
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_root=tmp_path / "out",
            mcp_env=tmp_path / "amo.env",
            copy_db=False,
        )
    )

    assert report["validation_ok"] is False
    assert report["apply_blocked"] is True
    assert report["cursor_after"] == report["cursor_before"]
    assert report["safety"]["staging_db_write"] is False
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0
        assert con.execute(
            "SELECT last_cursor_ts FROM ingestion_cursors WHERE source_system='amo_leads_updated_at'"
        ).fetchone()[0] == "2026-07-01T00:00:00+00:00"


def test_run_amo_incremental_imports_new_contact_before_linked_lead(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "staging.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    original_lead_cursor = "2026-06-23T20:00:00+00:00"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO ingestion_cursors (tenant_id, source_system, last_cursor_ts, updated_at, metadata_json)
            VALUES ('foton', 'amo_leads_updated_at', ?, ?, '{}')
            """,
            (original_lead_cursor, original_lead_cursor),
        )
        con.commit()
    payloads = {
        "contacts": {
            "_embedded": {
                "contacts": [
                    {
                        "id": 30,
                        "name": "New parent",
                        "created_at": 1782250000,
                        "updated_at": 1782250001,
                        "custom_fields_values": [
                            {"field_code": "PHONE", "values": [{"value": "8 (916) 123-45-67"}]},
                        ],
                    }
                ]
            }
        },
        "leads": {
            "_embedded": {
                "leads": [
                    {
                        "id": 42,
                        "name": "New linked lead",
                        "created_at": 1782250000,
                        "updated_at": 1782250002,
                        "_embedded": {"contacts": [{"id": 30}]},
                    },
                    {
                        "id": 43,
                        "name": "Still unresolved lead",
                        "created_at": 1782250000,
                        "updated_at": 1782250003,
                        "_embedded": {"contacts": []},
                    },
                ]
            }
        },
        "events": {"_embedded": {"events": []}},
    }

    class MultiPathAmoClient:
        def amo_api_get(self, *, path, params=None, limit=50):
            return payloads[path]

    monkeypatch.setattr(amo_incremental_module, "read_mcp_env", lambda _path: object())
    monkeypatch.setattr(amo_incremental_module, "AmoMcpClient", lambda _config: MultiPathAmoClient())

    report = run_amo_incremental(
        AmoIncrementalConfig(
            source_db=db_path,
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_root=tmp_path / "out",
            mcp_env=tmp_path / "amo.env",
            copy_db=False,
            max_pages=1,
            sleep_sec=0.0,
            since=NOW,
        )
    )

    assert report["fetch"]["amo_leads_updated_at"]["normalized"] == 1
    assert report["fetch"]["amo_leads_updated_at"]["skipped"]["unmatched"] == 1
    assert report["cursor_after"]["amo_leads_updated_at"] == original_lead_cursor
    with sqlite3.connect(db_path) as con:
        contact_owner = con.execute(
            "SELECT customer_id FROM identity_links WHERE link_type='amo_contact_id' AND link_value='30'"
        ).fetchone()[0]
        assert con.execute(
            "SELECT customer_id FROM identity_links WHERE link_type='amo_lead_id' AND link_value='42'"
        ).fetchone()[0] == contact_owner
        assert con.execute(
            "SELECT customer_id FROM customer_opportunities WHERE source_system='amocrm_snapshot' AND source_id='42'"
        ).fetchone()[0] == contact_owner


def test_load_amo_link_index_groups_all_phone_aliases(tmp_path) -> None:
    db_path = tmp_path / "staging.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    try:
        for customer_id, link_type in (
            ("customer:first", "phone"),
            ("customer:second", "mango_client_phone"),
            ("customer:third", "whatsapp_phone"),
        ):
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status=IdentityStatus.STRONG,
                )
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type=link_type,
                    link_value="+79161234567",
                    source_system="test",
                    source_ref=f"test:{customer_id}",
                )
            )
    finally:
        store.close()

    index = load_amo_link_index(db_path, tenant_id="foton")

    assert index[("phone", "+79161234567")] == (
        "customer:first",
        "customer:second",
        "customer:third",
    )
    assert ("mango_client_phone", "+79161234567") not in index
    assert ("whatsapp_phone", "+79161234567") not in index


class FakeAmoClient:
    def __init__(self, payload, *, expected_path="events"):
        self.payload = payload
        self.expected_path = expected_path

    def amo_api_get(self, *, path, params=None, limit=50):
        assert path == self.expected_path
        if path == "events":
            assert "filter[created_at][from]" in (params or {})
        else:
            assert "filter[updated_at][from]" in (params or {})
        return self.payload


class FlakyAmoClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def amo_api_get(self, *, path, params=None, limit=50):
        self.calls += 1
        if self.calls == 1:
            raise AmoMcpError('MCP tool error: {"error": "Tool call timed out."}')
        return self.payload


def test_amo_event_normalizer_creates_manager_review_raw_chunk() -> None:
    normalizer = AmoEventNormalizer(tenant_id="foton")

    batch = normalizer.normalize(
        TimelineSourceRecord(
            source_system="amo_events_created_at",
            source_ref="amocrm:event:evt-1",
            observed_at=NOW,
            payload={
                "event_id": "evt-1",
                "customer_id": "customer:test",
                "entity_type": "lead",
                "entity_id": "lead-1",
                "amo_event_type": "common_note_added",
                "created_at": NOW.isoformat(),
                "source_body_status": "note_body_missing",
                "summary": "AMO common_note_added for lead; body missing",
            },
        )
    )

    assert len(batch.events) == 1
    assert batch.events[0].event_type.value == "amo_note"
    assert batch.events[0].record["source_body_status"] == "note_body_missing"
    assert len(batch.bot_context_chunks) == 1
    assert batch.bot_context_chunks[0].allowed_for_bot is False
    assert batch.bot_context_chunks[0].requires_manager_review is True


def test_fetch_collection_retries_transient_mcp_timeout() -> None:
    payload = {"_embedded": {"leads": [{"id": 42}]}}
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, pages, page_cap_hit = fetch_collection(
        FlakyAmoClient(payload),
        path="leads",
        embedded_key="leads",
        params={"filter[updated_at][from]": 1},
        config=config,
    )

    assert pages == 1
    assert page_cap_hit is False
    assert rows == [{"id": 42}]


def test_amo_event_normalizer_requires_customer_id() -> None:
    normalizer = AmoEventNormalizer(tenant_id="foton")

    batch = normalizer.normalize(
        TimelineSourceRecord(
            source_system="amo_events_created_at",
            source_ref="amocrm:event:evt-2",
            observed_at=NOW,
            payload={
                "event_id": "evt-2",
                "entity_type": "lead",
                "entity_id": "lead-2",
                "amo_event_type": "incoming_chat_message",
                "created_at": NOW.isoformat(),
            },
        )
    )

    assert batch.events == ()
    assert batch.bot_context_chunks == ()


def test_normalizer_dispatch_supports_amo_snapshot_and_amo_event() -> None:
    snapshot = normalizer_for_source(
        IncrementalSourceConfig(
            name="lead_cards",
            source_system="amo_leads_updated_at",
            path="dummy.jsonl",
            normalizer="amo_snapshot",
        )
    )
    event = normalizer_for_source(
        IncrementalSourceConfig(
            name="events",
            source_system="amo_events_created_at",
            path="dummy.jsonl",
            normalizer="amo_event",
        )
    )

    assert snapshot.source_system == "amocrm_snapshot"
    assert event.source_system == "amocrm_event"


def test_fetch_events_source_marks_unmatched_and_ambiguous() -> None:
    payload = {
        "_embedded": {
            "events": [
                {"id": 1, "type": "incoming_chat_message", "entity_type": "lead", "entity_id": 10, "created_at": 1782250000},
                {"id": 2, "type": "common_note_added", "entity_type": "lead", "entity_id": 20, "created_at": 1782250001},
                {"id": 3, "type": "incoming_mail", "entity_type": "contact", "entity_id": 30, "created_at": 1782250002},
                {"id": 4, "type": "entity_linked", "entity_type": "lead", "entity_id": 10, "created_at": 1782250003},
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_events_source(
        FakeAmoClient(payload),
        from_ts=NOW,
        link_index={
            ("amo_lead_id", "10"): ("customer:lead-10",),
            ("amo_lead_id", "20"): ("customer:a", "customer:b"),
        },
        config=config,
    )

    assert len(rows) == 1
    assert rows[0]["customer_id"] == "customer:lead-10"
    assert rows[0]["source_body_status"] == "event_only"
    assert stats["skipped"]["ambiguous"] == 1
    assert stats["skipped"]["unmatched"] == 1
    assert stats["skipped"]["unsupported_type"] == 1
    assert event_summary({"type": "common_note_added", "entity_type": "lead"}, body_status="note_body_missing").endswith("body missing")


def test_fetch_cards_source_maps_lead_via_embedded_contact_identity() -> None:
    payload = {
        "_embedded": {
            "leads": [
                {
                    "id": 42,
                    "name": "Lead with known contact",
                    "created_at": 1782250000,
                    "updated_at": 1782250001,
                    "_embedded": {"contacts": [{"id": 30}]},
                }
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_cards_source(
        FakeAmoClient(payload, expected_path="leads"),
        path="leads",
        embedded_key="leads",
        entity_type="lead",
        cursor_name="amo_leads_updated_at",
        from_ts=NOW,
        link_index={("amo_contact_id", "30"): ("customer:known-contact",)},
        config=config,
    )

    assert len(rows) == 1
    assert rows[0]["customer_id"] == "customer:known-contact"
    assert stats["resolution_counts"]["embedded_contact_identity_link"] == 1
    assert stats["page_cap_hit"] is False


def test_fetch_cards_source_maps_contact_via_embedded_lead_identity() -> None:
    payload = {
        "_embedded": {
            "contacts": [
                {
                    "id": 30,
                    "name": "Known family contact",
                    "created_at": 1782250000,
                    "updated_at": 1782250001,
                    "_embedded": {"leads": [{"id": 42}, {"id": 43}]},
                }
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_cards_source(
        FakeAmoClient(payload, expected_path="contacts"),
        path="contacts",
        embedded_key="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_updated_at",
        from_ts=NOW,
        link_index={
            ("amo_lead_id", "42"): ("customer:family",),
            ("amo_lead_id", "43"): ("customer:family",),
        },
        config=config,
    )

    assert len(rows) == 1
    assert rows[0]["customer_id"] == "customer:family"
    assert stats["resolution_counts"]["embedded_lead_identity_link"] == 1


def test_fetch_cards_source_reports_page_cap_hit() -> None:
    payload = {
        "_embedded": {"leads": [{"id": 42, "updated_at": 1782250001, "_embedded": {"contacts": [{"id": 30}]}}]},
        "_links": {"next": {"href": "/api/v4/leads?page=2"}},
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    _rows, stats = fetch_cards_source(
        FakeAmoClient(payload, expected_path="leads"),
        path="leads",
        embedded_key="leads",
        entity_type="lead",
        cursor_name="amo_leads_updated_at",
        from_ts=NOW,
        link_index={("amo_contact_id", "30"): ("customer:known-contact",)},
        config=config,
    )

    assert stats["pages"] == 1
    assert stats["max_pages"] == 1
    assert stats["page_cap_hit"] is True


def test_fetch_cards_source_extracts_unique_contact_email_and_phone() -> None:
    payload = {
        "_embedded": {
            "contacts": [
                {
                    "id": 30,
                    "updated_at": 1782250001,
                    "custom_fields_values": [
                        {"field_code": "PHONE", "values": [{"value": "8 (916) 123-45-67"}]},
                        {"field_code": "EMAIL", "values": [{"value": " Parent@Example.COM "}]},
                    ],
                }
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_cards_source(
        FakeAmoClient(payload, expected_path="contacts"),
        path="contacts",
        embedded_key="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_updated_at",
        from_ts=NOW,
        link_index={},
        config=config,
    )

    assert rows[0]["phone"] == "+79161234567"
    assert rows[0]["email"] == "parent@example.com"
    assert stats["contact_identity_diagnostics"] == {"phone_selected": 1, "email_selected": 1}


def test_fetch_cards_source_blocks_contact_values_shared_by_different_customers() -> None:
    shared_fields = [
        {"field_code": "PHONE", "values": [{"value": "8 (916) 123-45-67"}]},
        {"field_code": "EMAIL", "values": [{"value": "parent@example.com"}]},
    ]
    payload = {
        "_embedded": {
            "contacts": [
                {"id": 30, "updated_at": 1782250001, "custom_fields_values": shared_fields},
                {"id": 31, "updated_at": 1782250002, "custom_fields_values": shared_fields},
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_cards_source(
        FakeAmoClient(payload, expected_path="contacts"),
        path="contacts",
        embedded_key="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_updated_at",
        from_ts=NOW,
        link_index={
            ("amo_contact_id", "30"): ("customer:first",),
            ("amo_contact_id", "31"): ("customer:second",),
        },
        config=config,
    )

    assert len(rows) == 2
    assert all("phone" not in row and "email" not in row for row in rows)
    assert stats["contact_identity_diagnostics"]["phone_cross_customer_ambiguous"] == 2
    assert stats["contact_identity_diagnostics"]["email_cross_customer_ambiguous"] == 2


def test_fetch_cards_source_does_not_select_ambiguous_contact_email() -> None:
    payload = {
        "_embedded": {
            "contacts": [
                {
                    "id": 31,
                    "updated_at": 1782250001,
                    "custom_fields_values": [
                        {
                            "field_code": "EMAIL",
                            "values": [{"value": "first@example.com"}, {"value": "second@example.com"}],
                        }
                    ],
                }
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_cards_source(
        FakeAmoClient(payload, expected_path="contacts"),
        path="contacts",
        embedded_key="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_updated_at",
        from_ts=NOW,
        link_index={},
        config=config,
    )

    assert "email" not in rows[0]
    assert stats["contact_identity_diagnostics"]["email_ambiguous_contacts"] == 1


def test_fetch_cards_source_contact_report_contains_no_raw_identity_values() -> None:
    raw_phone = "8 (916) 000-11-22"
    raw_email = "not-an-email-secret"
    payload = {
        "_embedded": {
            "contacts": [
                {
                    "id": 32,
                    "updated_at": 1782250001,
                    "custom_fields_values": [
                        {"field_code": "PHONE", "values": [{"value": raw_phone}]},
                        {"field_code": "EMAIL", "values": [{"value": raw_email}]},
                    ],
                }
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    _rows, stats = fetch_cards_source(
        FakeAmoClient(payload, expected_path="contacts"),
        path="contacts",
        embedded_key="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_updated_at",
        from_ts=NOW,
        link_index={},
        config=config,
    )

    report_text = json.dumps(stats, sort_keys=True)
    assert raw_phone not in report_text
    assert raw_email not in report_text
    assert stats["contact_identity_diagnostics"]["email_invalid_values_skipped"] == 1


def test_fetch_events_source_marks_mapping_after_card_import() -> None:
    payload = {
        "_embedded": {
            "events": [
                {"id": 10, "type": "incoming_mail", "entity_type": "contact", "entity_id": 30, "created_at": 1782250000},
                {"id": 11, "type": "common_note_added", "entity_type": "contact", "entity_id": 30, "created_at": 1782250001},
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_events_source(
        FakeAmoClient(payload),
        from_ts=NOW,
        link_index={("amo_contact_id", "30"): ("customer:after-card",)},
        diagnostic_link_index_before={},
        fetched_entity_ids={"contact": {"30"}},
        config=config,
    )

    assert len(rows) == 2
    assert {row["customer_id"] for row in rows} == {"customer:after-card"}
    assert stats["mapping_diagnostics_counts"]["mapped_after_card_import"] == 2
    assert stats["common_note_added_mapping_diagnostics"]["mapped_after_card_import"] == 1
    assert stats["source_body_status_counts"]["note_body_missing"] == 1


def test_fetch_events_source_sets_opportunity_for_lead_events_only() -> None:
    payload = {
        "_embedded": {
            "events": [
                {"id": 21, "type": "incoming_chat_message", "entity_type": "lead", "entity_id": 501, "created_at": 1782250000},
                {"id": 22, "type": "incoming_mail", "entity_type": "contact", "entity_id": 30, "created_at": 1782250001},
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, _stats = fetch_events_source(
        FakeAmoClient(payload),
        from_ts=NOW,
        link_index={
            ("amo_lead_id", "501"): ("customer:lead",),
            ("amo_contact_id", "30"): ("customer:contact",),
        },
        opportunity_index={
            "501": (
                {
                    "customer_id": "customer:lead",
                    "opportunity_id": "opportunity:lead-501",
                },
            )
        },
        diagnostic_link_index_before={},
        fetched_entity_ids={"lead": {"501"}, "contact": {"30"}},
        config=config,
    )

    by_id = {row["event_id"]: row for row in rows}
    assert by_id["21"]["opportunity_id"] == "opportunity:lead-501"
    assert by_id["22"]["opportunity_id"] is None
