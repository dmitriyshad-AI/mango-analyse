from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import mango_mvp.customer_timeline.amo_incremental as amo_incremental_module
from mango_mvp.customer_timeline.amo_incremental import (
    AmoIncrementalConfig,
    event_summary,
    fetch_cards_source,
    fetch_collection,
    fetch_endpoint_checkpointed,
    fetch_events_source,
    load_amo_link_index,
    load_amo_opportunity_index,
    run_amo_incremental,
)
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityStatus,
    OpportunityType,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.safe_copy import file_sha256
from mango_mvp.existing_clients.amo_step1_snapshot import AmoMcpError
from mango_mvp.customer_timeline.ingestion import TimelineSourceRecord
from mango_mvp.customer_timeline.nightly_incremental import (
    AmoEventNormalizer,
    IncrementalSourceConfig,
    JsonlTimelineNormalizer,
    normalizer_for_source,
)


NOW = datetime(2026, 6, 24, 8, 0, tzinfo=timezone.utc)


def test_amo_indexes_ignore_sql_and_textual_null_customer_ids(tmp_path) -> None:
    db_path = tmp_path / "staging.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        valid = CustomerIdentity(
            tenant_id="foton",
            customer_id="customer:valid",
            identity_status="strong",
            created_at=NOW,
            updated_at=NOW,
        )
        discarded = CustomerIdentity(
            tenant_id="foton",
            customer_id="customer:discarded",
            identity_status="strong",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(valid)
        store.upsert_customer(discarded)
        links = []
        opportunities = []
        for suffix, customer in (("valid", valid), ("null", discarded), ("text", discarded)):
            link = IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_contact_id",
                link_value=f"contact-{suffix}",
                source_system="amocrm_snapshot",
                source_ref=f"contact:{suffix}",
            )
            opportunity = CustomerOpportunity(
                tenant_id="foton",
                customer_id=customer.customer_id,
                opportunity_type="amo_deal",
                source_system="amocrm_snapshot",
                source_id=f"lead-{suffix}",
            )
            store.upsert_identity_link(link)
            store.upsert_opportunity(opportunity)
            links.append(link)
            opportunities.append(opportunity)
        store._con.execute(  # noqa: SLF001 - historical corruption fixture.
            "UPDATE identity_links SET customer_id=NULL WHERE link_id=?", (links[1].link_id,)
        )
        store._con.execute(  # noqa: SLF001
            "UPDATE identity_links SET customer_id='None' WHERE link_id=?", (links[2].link_id,)
        )
        store._con.execute(  # noqa: SLF001
            "UPDATE customer_opportunities SET customer_id='' WHERE opportunity_id=?",
            (opportunities[1].opportunity_id,),
        )
        store._con.execute(  # noqa: SLF001
            "UPDATE customer_opportunities SET customer_id='None' WHERE opportunity_id=?",
            (opportunities[2].opportunity_id,),
        )
        store._con.commit()  # noqa: SLF001

    assert load_amo_link_index(db_path, tenant_id="foton") == {
        ("amo_contact_id", "contact-valid"): ("customer:valid",)
    }
    assert load_amo_opportunity_index(db_path, tenant_id="foton") == {
        "lead-valid": (
            {
                "opportunity_id": opportunities[0].opportunity_id,
                "customer_id": "customer:valid",
            },
        )
    }


def _write_verified_task_snapshot(tmp_path, rows=()):
    rows = tuple(rows)
    root = tmp_path / "amo_tasks"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "amo_tasks_snapshot.csv"
    fieldnames = (
        "task_id", "entity_id", "entity_type", "text", "task_type_id",
        "responsible_user_id", "responsible_user_name", "complete_till",
        "created_at", "updated_at", "is_completed", "result",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "m1_timeline_amo_tasks_snapshot_v1",
        "generated_at_utc": "2026-06-24T07:00:00+00:00",
        "normalized_at_utc": "2026-06-24T07:05:00+00:00",
        "checkpoint": {"tasks_complete": True},
        "scope": {
            "entity_type": "leads",
            "includes_all_open_and_overdue": True,
            "is_completed": False,
        },
        "tasks": {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": file_sha256(path),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _task_row(**overrides):
    row = {
        "task_id": "task-1",
        "entity_id": "lead-1",
        "entity_type": "leads",
        "text": "Позвонить и согласовать расписание",
        "task_type_id": "1",
        "responsible_user_id": "7",
        "responsible_user_name": "Менеджер",
        "complete_till": "2026-06-25T12:00:00+00:00",
        "created_at": "2026-06-24T06:00:00+00:00",
        "updated_at": "2026-06-24T07:00:00+00:00",
        "is_completed": False,
        "result": "",
    }
    row.update(overrides)
    return row


def _seed_exact_amo_lead(db_path, allowed_root):
    customer = CustomerIdentity(
        tenant_id="foton",
        customer_id="customer:task-owner",
        identity_status=IdentityStatus.STRONG,
    )
    opportunity = CustomerOpportunity(
        tenant_id="foton",
        customer_id=customer.customer_id,
        opportunity_type=OpportunityType.AMO_DEAL,
        source_system="amocrm_snapshot",
        source_id="lead-1",
        title="Учебный курс",
        status="open",
        opened_at=NOW,
        confidence=1.0,
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        store.upsert_customer(customer)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_lead_id",
                link_value="lead-1",
                source_system="amocrm_snapshot",
                source_ref="amocrm:lead:lead-1",
            )
        )
        store.upsert_opportunity(opportunity)
    return customer, opportunity


def test_amo_checkpoint_with_truncated_utf8_is_ignored(tmp_path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / "amo_incremental_checkpoint.json").write_bytes(b'{"schema_version":"\xff')

    assert amo_incremental_module.load_amo_incremental_checkpoint(out_root) == {}


def test_amo_checkpoint_with_unknown_schema_is_ignored(tmp_path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / "amo_incremental_checkpoint.json").write_text(
        json.dumps({"schema_version": "future", "endpoints": {"leads": {"next_page": 99}}}),
        encoding="utf-8",
    )

    assert amo_incremental_module.load_amo_incremental_checkpoint(out_root) == {}


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


@pytest.mark.parametrize("cap_source", ["leads", "contacts", "events", "tasks"])
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
            tasks_snapshot=_write_verified_task_snapshot(tmp_path),
        )
    )

    assert report["validation_ok"] is False
    assert report["apply_blocked"] is True
    assert report["cursor_after"] == report["cursor_before"]
    assert report["safety"]["staging_db_write"] is False
    checkpoint = tmp_path / "out" / "amo_incremental_checkpoint.json"
    assert checkpoint.stat().st_mode & 0o777 == 0o600
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
        "tasks": {"_embedded": {"tasks": []}},
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
            tasks_snapshot=_write_verified_task_snapshot(tmp_path),
        )
    )

    assert report["completed_import_sources"] == ["amocrm_event", "amocrm_snapshot"]
    assert report["fetch"]["amo_leads_updated_at"]["normalized"] == 1
    assert report["fetch"]["amo_leads_updated_at"]["skipped"]["unmatched"] == 1
    assert report["cursor_after"]["amo_leads_updated_at"] > original_lead_cursor
    assert report["checkpoint"]["pending_lead_retries"] == 1
    assert report["identity_resolution"] == {
        "complete": False,
        "pending_lead_retries": 1,
        "pending_task_lead_gaps": 0,
        "pending_state": "private_checkpoint",
    }
    checkpoint = json.loads((tmp_path / "out" / "amo_incremental_checkpoint.json").read_text())
    assert [item["id"] for item in checkpoint["endpoints"]["amo_leads_pending"]["items"]] == [43]
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


# --- D1: page_cap_hit must never advance the cursor / count as ok, and must
# be resolved by continuing via bounded checkpoint windows across multiple
# runs (not aborting forever, not silently truncating). ---


class BigBacklogAmoClient:
    """Serves a large, purely in-memory paginated 'leads' backlog (page size
    == the caller's page_limit, matching real AMO's page-size semantics) and
    empty contacts/events collections. No real API/network calls anywhere.
    """

    def __init__(self, *, total_leads: int, linked_lead_id: int | None = None, linked_contact_id: str | None = None):
        self.total_leads = total_leads
        self.linked_lead_id = linked_lead_id
        self.linked_contact_id = linked_contact_id
        self.calls: list[tuple[str, int]] = []

    def amo_api_get(self, *, path, params=None, limit=50):
        page = int((params or {}).get("page") or 1)
        self.calls.append((path, page))
        if path != "leads":
            return {"_embedded": {path: []}}
        per_page = max(1, int(limit))
        start = (page - 1) * per_page
        end = min(start + per_page, self.total_leads)
        items = []
        for index in range(start, end):
            lead_id = index + 1
            item = {
                "id": lead_id,
                "name": f"Lead {lead_id}",
                "created_at": 1782250000 + index,
                "updated_at": 1782250000 + index,
            }
            if self.linked_lead_id is not None and lead_id == self.linked_lead_id:
                item["_embedded"] = {"contacts": [{"id": self.linked_contact_id}]}
            items.append(item)
        payload = {"_embedded": {"leads": items}}
        if end < self.total_leads:
            payload["_links"] = {"next": {"href": f"/api/v4/leads?page={page + 1}"}}
        return payload


def test_run_amo_incremental_checkpoint_completes_large_backlog_across_bounded_runs(tmp_path, monkeypatch) -> None:
    """D1: a backlog far larger than one run's page budget (>20 pages / 1000
    deals) must be walked via bounded checkpoint windows across multiple
    runs -- writing nothing and never advancing the cursor until the whole
    fixed universe has been read -- instead of aborting forever or silently
    truncating. Uses only an in-memory fake client; no real API calls.
    """
    db_path = tmp_path / "staging.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:seed",
                identity_status=IdentityStatus.STRONG,
                display_name="Seed parent",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:seed",
                link_type="amo_contact_id",
                link_value="30",
                source_system="test",
                source_ref="test",
            )
        )

    client = BigBacklogAmoClient(total_leads=1000, linked_lead_id=1, linked_contact_id="30")
    monkeypatch.setattr(amo_incremental_module, "read_mcp_env", lambda _path: object())
    monkeypatch.setattr(amo_incremental_module, "AmoMcpClient", lambda _config: client)

    config = AmoIncrementalConfig(
        source_db=db_path,
        timeline_db=db_path,
        allowed_root=tmp_path,
        out_root=tmp_path / "out",
        mcp_env=tmp_path / "amo.env",
        copy_db=False,
        max_pages=5,
        page_limit=40,
        sleep_sec=0.0,
        since=NOW,
        tasks_snapshot=_write_verified_task_snapshot(tmp_path),
    )

    reports = []
    for _ in range(8):  # 1000/40=25 pages, 5 pages/run -> exactly 5 runs expected; generous safety bound
        report = run_amo_incremental(config)
        reports.append(report)
        if "apply_blocked" in report:
            with sqlite3.connect(db_path) as con:
                assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0
        else:
            break
    else:
        pytest.fail("checkpoint cycle did not complete within the safety bound of 8 runs")

    assert len(reports) == 5, "1000 leads at 40/page (25 pages) over 5 pages/run must take exactly 5 runs"
    for blocked in reports[:-1]:
        assert blocked["apply_blocked"] is True
        assert blocked["validation_ok"] is False
        assert blocked["cursor_after"] == blocked["cursor_before"]
        assert blocked["checkpoint"]["pending_endpoints"] == ["amo_leads_updated_at"]

    final = reports[-1]
    assert "apply_blocked" not in final
    assert final["fetch"]["amo_leads_updated_at"]["fetched"] == 1000
    assert final["fetch"]["amo_leads_updated_at"]["pages"] == 25
    assert final["checkpoint"]["cleared"] is False
    assert final["checkpoint"]["pending_lead_retries"] == 999
    assert final["identity_resolution"]["complete"] is False
    # The 999 unresolved leads remain in the private checkpoint, while the
    # network cursor advances; future runs retry only this local subset.
    with sqlite3.connect(db_path) as con:
        # The one lead wired to a known contact (id=1, fetched on page 1
        # during run 1, carried through the checkpoint over 4 more runs)
        # is only imported once the whole universe is confirmed read.
        assert con.execute(
            "SELECT COUNT(*) FROM timeline_events WHERE source_system='amocrm_snapshot'"
        ).fetchone()[0] == 1

    lead_page_calls = [page for path, page in client.calls if path == "leads"]
    expected_calls = []
    for first in range(1, 26, 5):
        anchor = [first - 1] if first > 1 else []
        batch = list(range(first, first + 5))
        expected_calls.extend(anchor + batch + batch + anchor)
    assert lead_page_calls == expected_calls
    assert sum(path == "contacts" for path, _page in client.calls) == 2
    assert sum(path == "events" for path, _page in client.calls) == 2
    for source_path in (tmp_path / "out" / "amo_incremental_sources").glob("*.jsonl"):
        assert source_path.stat().st_mode & 0o777 == 0o600
    checkpoint = json.loads((tmp_path / "out" / "amo_incremental_checkpoint.json").read_text())
    assert len(checkpoint["endpoints"]["amo_leads_pending"]["items"]) == 999


def test_fetch_endpoint_checkpointed_collapses_identical_duplicate_ids(tmp_path) -> None:
    class DuplicateClient:
        def amo_api_get(self, *, path, params=None, limit=50):
            return {"_embedded": {"leads": [{"id": "same"}, {"id": "same"}]}}

    config = AmoIncrementalConfig(
        source_db=tmp_path / "source.sqlite", out_root=tmp_path / "out",
        mcp_env=tmp_path / "amo.env", max_pages=1, sleep_sec=0.0,
    )
    next_checkpoint: dict = {}

    items, stats = fetch_endpoint_checkpointed(
        DuplicateClient(), key="amo_leads_updated_at", path="leads", embedded_key="leads",
        params={"order[id]": "asc"}, lower_bound=NOW, config=config,
        checkpoint={}, next_checkpoint=next_checkpoint,
    )

    assert items == [{"id": "same"}]
    assert stats["complete"] is True
    assert stats["pagination_drift_detected"] is False
    assert stats["identical_duplicates_collapsed"] == 1


def test_fetch_endpoint_checkpointed_blocks_conflicting_duplicate_ids(tmp_path) -> None:
    class DuplicateClient:
        def amo_api_get(self, *, path, params=None, limit=50):
            return {"_embedded": {"leads": [{"id": "same", "name": "A"}, {"id": "same", "name": "B"}]}}

    config = AmoIncrementalConfig(
        source_db=tmp_path / "source.sqlite", out_root=tmp_path / "out",
        mcp_env=tmp_path / "amo.env", max_pages=1, sleep_sec=0.0,
    )
    next_checkpoint: dict = {}

    _items, stats = fetch_endpoint_checkpointed(
        DuplicateClient(), key="amo_leads_updated_at", path="leads", embedded_key="leads",
        params={"order[id]": "asc"}, lower_bound=NOW, config=config,
        checkpoint={}, next_checkpoint=next_checkpoint,
    )

    assert stats["complete"] is False
    assert stats["pagination_drift_detected"] is True
    assert stats["conflicting_duplicates"] == 1
    assert next_checkpoint["amo_leads_updated_at"]["items"] == []
    assert next_checkpoint["amo_leads_updated_at"]["unverified_items"] == [
        {"id": "same", "name": "A"}, {"id": "same", "name": "B"},
    ]


def test_fetch_endpoint_checkpointed_resumes_on_match_and_restarts_on_fingerprint_change(tmp_path) -> None:
    """D1: a saved checkpoint must only be resumed when the universe
    fingerprint (endpoint + lower_bound) still matches. A lower_bound change
    (the cursor moved to a different window) must discard the old,
    incomplete checkpoint and start that endpoint over at page 1 -- an
    incomplete checkpoint from a stale window must never be silently
    continued into a different one.
    """

    class TwoItemPagedClient:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def amo_api_get(self, *, path, params=None, limit=50):
            page = int((params or {}).get("page") or 1)
            self.calls.append(page)
            items = [{"id": f"{page}-{i}", "updated_at": 1782250000} for i in range(2)]
            payload = {"_embedded": {"leads": items}}
            if page < 3:
                payload["_links"] = {"next": {"href": "/api/v4/leads?page=next"}}
            return payload

    config = AmoIncrementalConfig(
        source_db=tmp_path / "source.sqlite",
        out_root=tmp_path / "out",
        mcp_env=tmp_path / "amo.env",
        max_pages=1,
        sleep_sec=0.0,
    )
    client = TwoItemPagedClient()
    old_lower_bound = datetime(2026, 6, 1, tzinfo=timezone.utc)

    first_checkpoint: dict = {}
    items1, stats1 = fetch_endpoint_checkpointed(
        client,
        key="amo_leads_updated_at",
        path="leads",
        embedded_key="leads",
        params={},
        lower_bound=old_lower_bound,
        config=config,
        checkpoint={},
        next_checkpoint=first_checkpoint,
    )
    assert stats1["complete"] is False
    assert client.calls == [1, 1]
    saved_checkpoint = {"endpoints": first_checkpoint}

    # A DIFFERENT lower_bound (the universe moved) must not resume the
    # checkpoint above: it must restart at page 1, not the stale next_page.
    new_lower_bound = datetime(2026, 7, 1, tzinfo=timezone.utc)
    mismatched_next: dict = {}
    items2, stats2 = fetch_endpoint_checkpointed(
        client,
        key="amo_leads_updated_at",
        path="leads",
        embedded_key="leads",
        params={},
        lower_bound=new_lower_bound,
        config=config,
        checkpoint=saved_checkpoint,
        next_checkpoint=mismatched_next,
    )
    assert stats2["start_page_this_run"] == 1
    assert stats2["carried_over_from_checkpoint"] == 0
    assert client.calls == [1, 1, 1, 1]

    # The SAME lower_bound as the original call must resume at the saved
    # next_page, carrying the previously-fetched items forward.
    matching_next: dict = {}
    items3, stats3 = fetch_endpoint_checkpointed(
        client,
        key="amo_leads_updated_at",
        path="leads",
        embedded_key="leads",
        params={},
        lower_bound=old_lower_bound,
        config=config,
        checkpoint=saved_checkpoint,
        next_checkpoint=matching_next,
    )
    assert stats3["start_page_this_run"] == 2
    assert stats3["carried_over_from_checkpoint"] == 2
    assert len(items3) == 4
    assert client.calls == [1, 1, 1, 1, 1, 2, 2, 1]


def test_fetch_endpoint_checkpointed_restarts_when_boundary_page_changes(tmp_path) -> None:
    class MutableClient:
        def __init__(self) -> None:
            self.items = [
                {"id": "a", "updated_at": 1782250000},
                {"id": "b", "updated_at": 1782250001},
                {"id": "c", "updated_at": 1782250002},
            ]
            self.calls: list[int] = []

        def amo_api_get(self, *, path, params=None, limit=2):
            page = int((params or {}).get("page") or 1)
            self.calls.append(page)
            start = (page - 1) * limit
            rows = self.items[start : start + limit]
            payload = {"_embedded": {"leads": rows}}
            if start + limit < len(self.items):
                payload["_links"] = {"next": {"href": "next"}}
            return payload

    client = MutableClient()
    config = AmoIncrementalConfig(
        source_db=tmp_path / "source.sqlite",
        out_root=tmp_path / "out",
        mcp_env=tmp_path / "amo.env",
        max_pages=1,
        page_limit=2,
        sleep_sec=0.0,
    )
    lower_bound = datetime(2026, 6, 1, tzinfo=timezone.utc)
    next_checkpoint: dict = {}
    fetch_endpoint_checkpointed(
        client,
        key="amo_leads_updated_at",
        path="leads",
        embedded_key="leads",
        params={},
        lower_bound=lower_bound,
        config=config,
        checkpoint={},
        next_checkpoint=next_checkpoint,
    )

    client.items.insert(0, {"id": "x", "updated_at": 1782249999})
    restarted_checkpoint: dict = {}
    items, stats = fetch_endpoint_checkpointed(
        client,
        key="amo_leads_updated_at",
        path="leads",
        embedded_key="leads",
        params={},
        lower_bound=lower_bound,
        config=config,
        checkpoint={"endpoints": next_checkpoint},
        next_checkpoint=restarted_checkpoint,
    )

    assert stats["checkpoint_reset_reason"] == "pagination_universe_changed"
    assert stats["start_page_this_run"] == 1
    assert stats["carried_over_from_checkpoint"] == 0
    assert items == []  # An incomplete new interval is not confirmed output.
    saved = restarted_checkpoint["amo_leads_updated_at"]
    assert [item["id"] for item in saved["window_pending_items"]] == ["x", "a"]
    assert saved["upper_bound"] == next_checkpoint["amo_leads_updated_at"]["upper_bound"]
    assert saved["window_width"] == 43200
    assert stats["boundary_ids_not_in_cache"] == 1
    assert client.calls == [1, 1, 1, 1]


def test_updated_lead_versions_get_distinct_event_at() -> None:
    """Two versions of one AMO lead must carry different event_at, otherwise
    `ORDER BY event_at DESC, event_id DESC` picks a version by id hash and a
    stale card can present itself as the current one."""
    link_index = {("amo_contact_id", "30"): ("customer:known-contact",)}
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    def card(updated_at: int, price: int) -> dict:
        return {
            "_embedded": {
                "leads": [
                    {
                        "id": 42,
                        "name": "Lead",
                        "price": price,
                        "created_at": 1782250000,
                        "updated_at": updated_at,
                        "_embedded": {"contacts": [{"id": 30}]},
                    }
                ]
            }
        }

    first, _ = fetch_cards_source(
        FakeAmoClient(card(1782250001, 100), expected_path="leads"),
        path="leads",
        embedded_key="leads",
        entity_type="lead",
        cursor_name="amo_leads_updated_at",
        from_ts=NOW,
        link_index=link_index,
        config=config,
    )
    second, _ = fetch_cards_source(
        FakeAmoClient(card(1782259999, 200), expected_path="leads"),
        path="leads",
        embedded_key="leads",
        entity_type="lead",
        cursor_name="amo_leads_updated_at",
        from_ts=NOW,
        link_index=link_index,
        config=config,
    )

    assert first[0]["source_id"] != second[0]["source_id"]
    assert first[0]["event_at"] == first[0]["updated_at"]
    assert second[0]["event_at"] == second[0]["updated_at"]
    assert first[0]["event_at"] != second[0]["event_at"]

    normalizer = JsonlTimelineNormalizer("amocrm_snapshot")
    events = [
        normalizer.normalize(
            TimelineSourceRecord(
                source_system="amocrm_snapshot",
                source_ref="amocrm:lead:42",
                observed_at=NOW,
                payload=row,
            )
        ).events[0]
        for row in (first[0], second[0])
    ]
    assert events[0].event_at < events[1].event_at


def test_tasks_bootstrap_page_cap_resumes_with_frozen_lower_bound(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "staging.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()

    class TaskBacklogClient:
        def __init__(self):
            self.task_from_values = []

        def amo_api_get(self, *, path, params=None, limit=50):
            params = dict(params or {})
            if path != "tasks":
                return {"_embedded": {path: []}}
            self.task_from_values.append(params["filter[updated_at][from]"])
            page = int(params.get("page") or 1)
            start = (page - 1) * limit
            task_rows = [
                {
                    "id": f"task-{index}",
                    "entity_id": f"missing-lead-{index}",
                    "entity_type": "leads",
                    "text": "Действие",
                    "responsible_user_id": 7,
                    "complete_till": 1782388800,
                    "created_at": 1782280800,
                    "updated_at": 1782284400 + index,
                    "is_completed": False,
                }
                for index in range(start, min(start + limit, 5))
            ]
            payload = {"_embedded": {"tasks": task_rows}}
            if start + limit < 5:
                payload["_links"] = {"next": {"href": f"/api/v4/tasks?page={page + 1}"}}
            return payload

    client = TaskBacklogClient()
    monkeypatch.setattr(amo_incremental_module, "read_mcp_env", lambda _path: object())
    monkeypatch.setattr(amo_incremental_module, "AmoMcpClient", lambda _config: client)
    config = AmoIncrementalConfig(
        source_db=db_path,
        timeline_db=db_path,
        allowed_root=tmp_path,
        out_root=tmp_path / "out",
        mcp_env=tmp_path / "amo.env",
        copy_db=False,
        max_pages=1,
        page_limit=2,
        sleep_sec=0.0,
        tasks_snapshot=_write_verified_task_snapshot(tmp_path),
    )

    reports = [run_amo_incremental(config) for _ in range(3)]

    assert [report.get("apply_blocked", False) for report in reports] == [True, True, False]
    assert len({report["lower_bound"]["amo_tasks_updated_at"] for report in reports}) == 1
    assert len(set(client.task_from_values)) == 1
    assert reports[-1]["fetch"]["amo_tasks_updated_at"]["retry_cache_rows"] == 5
    assert reports[-1]["cursor_after"]["amo_tasks_updated_at"] is not None


def test_open_task_completion_updates_one_event_without_duplicate(tmp_path) -> None:
    db_path = tmp_path / "staging.sqlite"
    customer, opportunity = _seed_exact_amo_lead(db_path, tmp_path)
    link_index = {("amo_lead_id", "lead-1"): (customer.customer_id,)}
    opportunity_index = {
        "lead-1": (
            {"opportunity_id": opportunity.opportunity_id, "customer_id": customer.customer_id},
        )
    }
    common = {
        "timeline_db": db_path,
        "allowed_root": tmp_path,
        "tenant_id": "foton",
        "link_index": link_index,
        "opportunity_index": opportunity_index,
        "overlap_seconds": 300,
        "bootstrap_complete": True,
        "pending_link_gap_count": 0,
        "seed_report": {"sha256": "seed-sha"},
    }

    first = amo_incremental_module.import_amo_task_rows(
        **common,
        task_rows=[_task_row(is_completed=False)],
        fetch_upper_bound=NOW + timedelta(hours=1),
    )
    completed = _task_row(
        is_completed=True,
        updated_at="2026-06-24T09:00:00+00:00",
        result="Выполнено",
    )
    second = amo_incremental_module.import_amo_task_rows(
        **common,
        task_rows=[completed],
        fetch_upper_bound=NOW + timedelta(hours=2),
    )
    third = amo_incremental_module.import_amo_task_rows(
        **common,
        task_rows=[completed],
        fetch_upper_bound=NOW + timedelta(hours=3),
    )

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id,opportunity_id,record_json FROM timeline_events WHERE event_type='amo_task'"
        ).fetchall()
    assert len(rows) == 1
    record = json.loads(rows[0][2])["record"]
    assert rows[0][0] == customer.customer_id
    assert rows[0][1] == opportunity.opportunity_id
    assert record["completed"] is True
    assert "next_step" not in record
    assert first["changed_customer_count"] == 1
    assert second["changed_customer_count"] == 1
    assert third["changed_customer_count"] == 0
    assert third["write_status_counts"] == {"duplicate": 1}


@pytest.mark.parametrize(
    ("task", "link_customer", "reason"),
    [
        (_task_row(is_completed="pending"), "customer:task-owner", "unrecognized_completion_flag"),
        (_task_row(entity_type="contacts"), "customer:task-owner", "unsupported_entity_type"),
        (_task_row(), "customer:other", "ambiguous_lead"),
    ],
)
def test_task_poison_rows_never_create_cross_customer_event(
    tmp_path,
    task,
    link_customer,
    reason,
) -> None:
    db_path = tmp_path / "staging.sqlite"
    customer, opportunity = _seed_exact_amo_lead(db_path, tmp_path)
    report = amo_incremental_module.import_amo_task_rows(
        timeline_db=db_path,
        allowed_root=tmp_path,
        tenant_id="foton",
        task_rows=[task],
        link_index={("amo_lead_id", "lead-1"): (link_customer,)},
        opportunity_index={
            "lead-1": (
                {"opportunity_id": opportunity.opportunity_id, "customer_id": customer.customer_id},
            )
        },
        fetch_upper_bound=NOW + timedelta(hours=1),
        overlap_seconds=300,
        bootstrap_complete=True,
        pending_link_gap_count=1 if reason == "ambiguous_lead" else 0,
        seed_report={},
    )

    with sqlite3.connect(db_path) as con:
        count = con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type='amo_task'").fetchone()[0]
    assert count == 0
    assert report["outcome_counts"] == {reason: 1}


def test_tasks_snapshot_rejects_duplicate_task_ids_before_api(tmp_path) -> None:
    snapshot = _write_verified_task_snapshot(
        tmp_path,
        [_task_row(), _task_row(text="Конфликтующее действие")],
    )

    with pytest.raises(ValueError, match="duplicate task_id"):
        amo_incremental_module.load_amo_task_seed_snapshot(snapshot)


def test_task_cursor_is_db_truth_when_no_retry_checkpoint_exists(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "staging.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_ingestion_cursor(
            "foton",
            "amo_tasks_updated_at",
            last_cursor_ts=NOW,
            metadata={"bootstrap_complete": True, "cache_rows": 0},
        )

    class EmptyClient:
        def amo_api_get(self, *, path, params=None, limit=50):
            return {"_embedded": {path: []}}

    monkeypatch.setattr(amo_incremental_module, "read_mcp_env", lambda _path: object())
    monkeypatch.setattr(amo_incremental_module, "AmoMcpClient", lambda _config: EmptyClient())

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

    assert report["validation_ok"] is True
    assert report["fetch"]["amo_tasks_updated_at"]["balance_ok"] is True


class TimeWindowAmoClient:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def amo_api_get(self, *, path, params=None, limit=20):
        field = "created_at" if path == "events" else "updated_at"
        left, right, page = params[f"filter[{field}][from]"], params[f"filter[{field}][to]"], params.get("page", 1)
        self.calls.append((left, right, page))
        rows = [row for row in self.rows if left <= row[field] <= right]
        offset = (page - 1) * limit
        result = {"_embedded": {path: rows[offset:offset + limit]}}
        if len(rows) > offset + limit:
            result["_links"] = {"next": {"href": "next"}}
        return result


def _time_window_args(tmp_path, *, end=86399, max_pages=3, page_limit=10):
    lower = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base = int(lower.timestamp())
    config = AmoIncrementalConfig(source_db=tmp_path / "unused.sqlite", out_root=tmp_path / "cache",
        mcp_env=tmp_path / "unused.env", max_pages=max_pages, page_limit=page_limit, sleep_sec=0)
    params = {"filter[updated_at][from]": base, "order[id]": "asc"}
    fingerprint = amo_incremental_module.universe_fingerprint(path="leads", lower_bound=lower, params=params, page_limit=page_limit)
    entry = {"fingerprint": fingerprint, "upper_bound": datetime.fromtimestamp(base + end, timezone.utc).isoformat(),
        "items": [], "window_next": base, "window_width": 86400, "verified_windows": [], "pages_fetched": 0, "complete": False}
    return base, dict(key="leads", path="leads", embedded_key="leads", params=params,
        lower_bound=lower, config=config, checkpoint={"endpoints": {"leads": entry}})


@pytest.mark.parametrize("count", [1, 10, 100])
def test_time_windows_complete_without_repeating_confirmed_ranges(tmp_path, count) -> None:
    base, args = _time_window_args(tmp_path, end=99)
    client = TimeWindowAmoClient([{"id": i + 1, "updated_at": base + i} for i in range(count)])
    confirmed = set()
    for _ in range(32):
        before, saved = len(client.calls), {}
        rows, stats = fetch_endpoint_checkpointed(client, **args, next_checkpoint=saved)
        assert stats["pages_this_run"] <= 3 and stats["verification_pages"] <= 3
        assert stats["client_get_calls"] <= 6
        assert not any((left, right) in confirmed for left, right, _ in client.calls[before:])
        entry = saved["leads"]
        confirmed.update(tuple(item) for item in entry["verified_windows"])
        assert json.loads((tmp_path / "cache/amo_incremental_checkpoint.json").read_text())["endpoints"]["leads"] == entry
        args["checkpoint"] = {"endpoints": saved}
        if stats["complete"]:
            break
    else:
        pytest.fail("fixed ranges did not complete")
    assert [item["id"] for item in rows] == list(range(1, count + 1))
    assert entry["window_next"] == base + 100
    before = len(client.calls)
    rows2, stats2 = fetch_endpoint_checkpointed(client, **args, next_checkpoint={})
    assert rows2 == rows and stats2["complete"] is True and len(client.calls) == before


def test_window_cap_persists_reduced_width_without_completed_rows(tmp_path) -> None:
    base, args = _time_window_args(tmp_path, max_pages=1, page_limit=1)
    client = TimeWindowAmoClient([{"id": 1, "updated_at": base}, {"id": 2, "updated_at": base + 50000}])
    saved = {}
    _, stats = fetch_endpoint_checkpointed(client, **args, next_checkpoint=saved)
    assert stats["complete"] is False and saved["leads"]["window_width"] == 43200
    assert saved["leads"]["verified_windows"] == []
    upper = saved["leads"]["upper_bound"]
    args["checkpoint"], before, saved2 = {"endpoints": saved}, len(client.calls), {}
    rows, _ = fetch_endpoint_checkpointed(client, **args, next_checkpoint=saved2)
    assert client.calls[before] == (base, base + 43199, 1)
    assert rows == [{"id": 1, "updated_at": base}] and saved2["leads"]["upper_bound"] == upper


def test_one_second_over_budget_never_claims_complete(tmp_path) -> None:
    base, args = _time_window_args(tmp_path, end=0, max_pages=1, page_limit=1)
    client = TimeWindowAmoClient([{"id": 1, "updated_at": base}, {"id": 2, "updated_at": base}])
    rows, stats = fetch_endpoint_checkpointed(client, **args, next_checkpoint={})
    assert rows == [] and stats["complete"] is False
    assert stats["blocked_reason"] == "one_second_window_unproven" and len(client.calls) == 1


@pytest.mark.parametrize("where", ["middle", "terminal"])
def test_window_verification_rejects_page_or_terminal_mutation(tmp_path, where) -> None:
    base, args = _time_window_args(tmp_path, end=0, max_pages=3, page_limit=1)
    class ChangingClient(TimeWindowAmoClient):
        def amo_api_get(self, **kwargs):
            result = super().amo_api_get(**kwargs)
            if where == "middle" and len(self.calls) == 5:
                result["_embedded"]["leads"] = [{"id": "changed", "updated_at": base}]
            if where == "terminal" and len(self.calls) == 6:
                result["_links"] = {"next": {"href": "new-page"}}
            return result
    client, saved = ChangingClient([{"id": i, "updated_at": base} for i in range(3)]), {}
    _, stats = fetch_endpoint_checkpointed(client, **args, next_checkpoint=saved)
    assert stats["complete"] is False and stats["pagination_drift_detected"] is True
    assert saved["leads"]["verified_windows"] == []


def test_resume_prefix_mutating_after_anchor_is_not_confirmed(tmp_path) -> None:
    base, args = _time_window_args(tmp_path, end=10, max_pages=1, page_limit=1)
    initial = args["checkpoint"]["endpoints"]["leads"]
    old = {"id": 1, "updated_at": base}
    entry = {"fingerprint": initial["fingerprint"], "upper_bound": initial["upper_bound"], "next_page": 2,
        "last_page": 1, "last_page_anchor": amo_incremental_module.page_anchor([old]),
        "items": [old], "pages_fetched": 1, "complete": False}
    args["checkpoint"] = {"endpoints": {"leads": entry}}
    class ShrinkingClient(TimeWindowAmoClient):
        def amo_api_get(self, **kwargs):
            if len(self.calls) == 1:
                self.rows = self.rows[1:]
            return super().amo_api_get(**kwargs)
    client, saved = ShrinkingClient([old, {"id": 2, "updated_at": base}]), {}
    _, stats = fetch_endpoint_checkpointed(client, **args, next_checkpoint=saved)
    assert stats["complete"] is False and stats["boundary_probes"] == 2
    assert saved["leads"]["items"] == [] and old in saved["leads"]["unverified_items"]


def test_completed_endpoints_survive_fourth_endpoint_exception(tmp_path, monkeypatch) -> None:
    db = tmp_path / "timeline.sqlite"
    CustomerTimelineSQLiteStore(db, allowed_root=tmp_path).close()
    class FailingFourth(BigBacklogAmoClient):
        def amo_api_get(self, *, path, params=None, limit=20):
            if path == "tasks":
                raise RuntimeError("synthetic fourth endpoint failure")
            return super().amo_api_get(path=path, params=params, limit=limit)
    monkeypatch.setattr(amo_incremental_module, "read_mcp_env", lambda _: object())
    monkeypatch.setattr(amo_incremental_module, "AmoMcpClient", lambda _: FailingFourth(total_leads=0))
    config = AmoIncrementalConfig(source_db=db, timeline_db=db, allowed_root=tmp_path, copy_db=False,
        out_root=tmp_path / "cache", mcp_env=tmp_path / "unused", sleep_sec=0,
        tasks_snapshot=_write_verified_task_snapshot(tmp_path))
    with pytest.raises(RuntimeError, match="fourth endpoint"):
        run_amo_incremental(config)
    saved = amo_incremental_module.load_amo_incremental_checkpoint(config.out_root)["endpoints"]
    assert all(saved[key]["complete"] for key in ("amo_leads_updated_at", "amo_contacts_updated_at", "amo_events_created_at"))
    assert all(value is None for value in amo_incremental_module.load_cursor_snapshot(db, "foton").values())


def test_empty_nonterminal_window_never_confirms_coverage(tmp_path) -> None:
    _, args = _time_window_args(tmp_path, end=0, max_pages=1)
    class EmptyNext(TimeWindowAmoClient):
        def amo_api_get(self, **kwargs):
            return {"_embedded": {"leads": []}, "_links": {"next": {"href": "next"}}}
    saved = {}
    _, stats = fetch_endpoint_checkpointed(EmptyNext([]), **args, next_checkpoint=saved)
    assert stats["complete"] is False and saved["leads"]["verified_windows"] == []


@pytest.mark.parametrize("payload", [{"_embedded": {"leads": None}}, {"_embedded": {"leads": [None]}}, {"_embedded": []}])
def test_malformed_collection_is_not_silently_empty(tmp_path, payload) -> None:
    _, args = _time_window_args(tmp_path, end=0)
    class Malformed(TimeWindowAmoClient):
        def amo_api_get(self, **kwargs):
            return payload
    with pytest.raises(amo_incremental_module.AmoMcpError, match="malformed"):
        fetch_endpoint_checkpointed(Malformed([]), **args, next_checkpoint={})


def test_small_successful_window_recovers_width(tmp_path) -> None:
    base, args = _time_window_args(tmp_path, end=86399, max_pages=3)
    args["checkpoint"]["endpoints"]["leads"]["window_width"] = 1
    saved = {}
    _, stats = fetch_endpoint_checkpointed(TimeWindowAmoClient([]), **args, next_checkpoint=saved)
    assert saved["leads"]["window_next"] == base + 7
    assert saved["leads"]["window_width"] == 8 and stats["client_get_calls"] == 6


def test_retry_calls_respect_endpoint_budget(tmp_path, monkeypatch) -> None:
    base, args = _time_window_args(tmp_path, end=0, max_pages=2, page_limit=1)
    monkeypatch.setattr(amo_incremental_module.time, "sleep", lambda _: None)
    class RetryClient(TimeWindowAmoClient):
        attempts = 0
        def amo_api_get(self, **kwargs):
            self.attempts += 1
            if self.attempts % 2:
                raise amo_incremental_module.AmoMcpError("429 synthetic")
            return super().amo_api_get(**kwargs)
    client = RetryClient([{"id": 1, "updated_at": base}, {"id": 2, "updated_at": base}])
    saved = {}
    with pytest.raises(amo_incremental_module.AmoMcpError, match="budget exhausted"):
        fetch_endpoint_checkpointed(client, **args, next_checkpoint=saved)
    assert client.attempts == 6 and not saved


def test_completed_window_cache_requires_contiguous_coverage(tmp_path) -> None:
    base, args = _time_window_args(tmp_path, end=10)
    args["checkpoint"]["endpoints"]["leads"].update(complete=True, window_next=base + 11, verified_windows=[[base + 1, base + 10]])
    client = TimeWindowAmoClient([])
    with pytest.raises(ValueError, match="not contiguous"):
        fetch_endpoint_checkpointed(client, **args, next_checkpoint={})
    assert client.calls == []


@pytest.mark.parametrize("complete", [False, True])
def test_missing_legacy_anchor_is_not_empty_string_proof(tmp_path, complete) -> None:
    base, args = _time_window_args(tmp_path, end=0)
    entry = args["checkpoint"]["endpoints"]["leads"]
    for key in ("window_next", "window_width", "verified_windows"):
        del entry[key]
    entry.update(next_page=201, pages_fetched=200, complete=complete)
    client, saved = TimeWindowAmoClient([{"id": 1, "updated_at": base}]), {}
    rows, stats = fetch_endpoint_checkpointed(client, **args, next_checkpoint=saved)
    assert stats["checkpoint_reset_reason"] == "pagination_universe_changed"
    assert client.calls == [(base, base, 1), (base, base, 1)]
    assert rows == [{"id": 1, "updated_at": base}]


@pytest.mark.parametrize("left,right,conflicts,winner", [
    ({"id": 1, "created_at": 1, "name": "A"}, {"id": 1, "created_at": 1, "name": "B"}, 1, "A"),
    ({"id": 1, "updated_at": 1, "name": "A"}, {"id": 1, "updated_at": 1, "name": "B"}, 1, "A"),
    ({"id": 1, "updated_at": 1, "name": "A"}, {"id": 1, "updated_at": 2, "name": "B"}, 0, "B"),
])
def test_only_proven_newer_duplicate_versions_win(left, right, conflicts, winner) -> None:
    rows, _, actual = amo_incremental_module._dedupe_collection_items([left, right], allow_newer_versions=True)
    assert actual == conflicts and rows[0]["name"] == winner


def test_tasks_checkpoint_ahead_of_missing_cursor_blocks(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "staging.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    out_root = tmp_path / "out"
    out_root.mkdir()
    amo_incremental_module.save_amo_incremental_checkpoint(
        out_root,
        {
            "amo_tasks_cache": {
                "items": [],
                "seed_loaded": True,
                "bootstrap_complete": True,
                "seed_report": {"read_completed_at": NOW.isoformat()},
            }
        },
    )
    monkeypatch.setattr(amo_incremental_module, "read_mcp_env", lambda _path: object())
    monkeypatch.setattr(amo_incremental_module, "AmoMcpClient", lambda _config: object())

    with pytest.raises(ValueError, match="checkpoint is ahead"):
        run_amo_incremental(
            AmoIncrementalConfig(
                source_db=db_path,
                timeline_db=db_path,
                allowed_root=tmp_path,
                out_root=out_root,
                mcp_env=tmp_path / "amo.env",
                copy_db=False,
            )
        )
