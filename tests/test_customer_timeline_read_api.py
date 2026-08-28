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
        assert profile["manager_projection"]["amo_contact_ids"] == ["contact-raw-1"]
        assert profile["manager_projection"]["amo_lead_ids"] == ["lead-1", "lead-raw-1"]
        assert profile["manager_projection"]["schema_version"] == "customer_profile_manager_projection_v2"
        assert profile["manager_projection"]["manager_action"]["readiness_state"] == "review"
        assert "identity_conflict_open" in profile["manager_projection"]["manager_action"]["readiness_reason_codes"]
        assert {item["link_value"] for item in profile["manager_projection"]["identity_links"]} == {"contact-raw-1", "lead-raw-1"}
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
        assert len(profile["timeline"]["items"][0]["summary"]) > 500
        assert profile["timeline"]["items"][0]["call_analysis"]["structured_fields"]["student"]["grade_current"] == "9"
        assert profile["timeline"]["items"][0]["call_type"] == "sales_call"
        assert profile["timeline"]["items"][0]["call_history_eligible"] is True
        assert profile["timeline"]["items"][0]["artifacts"][0]["has_path"] is True
        assert profile["timeline"]["items"][0]["signals"][0]["allowed_for_bot"] is False
        assert profile["timeline"]["items"][0]["signals"][0]["requires_manager_review"] is True
        assert profile["signals"][0]["allowed_for_bot"] is False
        assert profile["signals"][0]["requires_manager_review"] is True
        assert {item["allowed_for_bot"] for item in profile["bot_context"]["items"]} == {False, True}
        assert {item["requires_manager_review"] for item in profile["bot_context"]["items"]} == {False, True}
        assert profile["bot_context"]["items"][0]["next_step_status"] == "needs_manager_review"
        assert all("display_text" not in item for item in profile["bot_context"]["items"])
        assert "Спорный текст шага" not in raw_text
        assert "path" not in profile["timeline"]["items"][0]["artifacts"][0]
        assert profile["readiness"]["bot_allowed_chunks"] == 1
        assert profile["readiness"]["bot_review_required_chunks"] == 1
        assert profile["readiness"]["open_conflicts"] == 1
        assert profile["readiness"]["safe_for_automatic_bot"] is False
        assert "provider_raw_payload" not in raw_text
        assert "record_json" not in raw_text
        assert "/not/read/transcript.json" not in raw_text
        assert "hidden" not in raw_text


def test_read_api_projects_same_exact_amo_task_manager_action(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    as_of = NOW + timedelta(minutes=5)
    due = NOW + timedelta(days=1)
    with sqlite3.connect(db_path) as con:
        opportunity_id = str(con.execute(
            "SELECT opportunity_id FROM customer_opportunities WHERE customer_id=? AND source_id='lead-1'",
            (customer_id,),
        ).fetchone()[0])
        con.execute("UPDATE timeline_conflicts SET status='resolved',resolved_at=?", (NOW.isoformat(),))
        con.execute(
            "INSERT OR REPLACE INTO ingestion_cursors "
            "(tenant_id,source_system,last_cursor_ts,updated_at,metadata_json) VALUES (?,?,?,?,?)",
            (
                "foton", "amo_tasks_updated_at", (as_of - timedelta(minutes=2)).isoformat(),
                (as_of - timedelta(minutes=1)).isoformat(),
                json.dumps({"metadata": {"bootstrap_complete": True, "last_status": "ok"}}),
            ),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO ingestion_runs
            (run_id,tenant_id,source_system,source_ref,run_kind,idempotency_key,status,
             started_at,finished_at,input_hash,accepted_count,rejected_count,output_ref,error,
             record_hash,record_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "run:read-api-task", "foton", "amocrm_snapshot", "amocrm:tasks:updated_at",
                "amo_tasks_incremental", "read-api-fixture", "completed",
                (as_of - timedelta(minutes=3)).isoformat(), (as_of - timedelta(minutes=1)).isoformat(),
                "fixture", 1, 0, None, None, "fixture", "{}",
            ),
        )
        con.commit()
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                event_type="amo_task",
                event_at=NOW + timedelta(minutes=2),
                source_system="amocrm_snapshot",
                source_id="task-read-api",
                source_ref="amo:task:task-read-api",
                direction="internal",
                actor_name="Анна Менеджер",
                actor_ref="amo:user:17",
                text_preview="Позвонить и согласовать расписание",
                summary="Open AMO task",
                match_status="strong_unique",
                record={
                    "action_text": "Позвонить и согласовать расписание",
                    "next_step": {
                        "action": "Позвонить и согласовать расписание",
                        "due": due.isoformat(),
                    },
                    "responsible_user_id": "17",
                    "responsible_user_name": "Анна Менеджер",
                    "complete_till": due.isoformat(),
                    "completed": False,
                    "provenance": {
                        "task_id": "task-read-api",
                        "entity_type": "leads",
                        "entity_id": "lead-1",
                        "opportunity_source_system": "amocrm_snapshot",
                        "opportunity_source_id": "lead-1",
                    },
                },
                metadata={"actor_role": "manager"},
                created_at=NOW,
            )
        )

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        profile = api.customer_profile("foton", customer_id, as_of=as_of)

    action = profile["manager_projection"]["manager_action"]
    assert action["readiness_state"] == "ready"
    assert action["action"] == "Позвонить и согласовать расписание"
    assert action["responsible_ref"] == "amo:user:17"
    assert action["due_at"] == due.isoformat()
    assert action["action_provenance"]["task_id"] == "task-read-api"
    assert profile["next_step_resolution"]["resolution_kind"] == "historical_hint"
    assert profile["next_step_resolution"]["status"] != "active"
    assert profile["next_step_resolution"]["action"] == ""
    assert profile["next_step_resolution"]["display_text"] == ""


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


def test_operational_list_and_search_hide_graduate_only_but_keep_mixed_family_history(
    tmp_path: Path,
) -> None:
    db_path, _customer_id = seed_timeline_db(tmp_path)
    scoped_customers = (
        ("customer:graduate", "Graduate only"),
        ("customer:mixed-graduate", "Mixed graduate"),
        ("customer:mixed-active", "Mixed active"),
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index, (customer_id, display_name) in enumerate(scoped_customers, start=1):
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status="strong",
                    display_name=display_name,
                    primary_phone=f"+790000001{index:02d}",
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="email_message",
                    event_at=NOW + timedelta(minutes=index),
                    source_system="mail_archive_stage2",
                    source_id=f"scope-poison-{index}",
                    direction="inbound",
                    subject="scope-poison operational search",
                    match_status="strong_unique",
                    created_at=NOW + timedelta(minutes=index),
                )
            )
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id=customer_id,
                    chunk_id=f"scope-bot-{index}",
                    source_system="customer_timeline_bot_safe_summary",
                    source_ref=f"botsafe:scope-{index}",
                    chunk_type="bot_safe_summary",
                    text=f"scope-poison bot context {index}",
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata={"brand_context_authorized": True, "client_safe": True},
                    created_at=NOW + timedelta(minutes=index),
                )
            )
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS family_links_v1 (
              tenant_id TEXT NOT NULL, family_id TEXT NOT NULL, customer_id TEXT NOT NULL,
              child_key TEXT NOT NULL, canonical_name TEXT NOT NULL, name_variants_json TEXT NOT NULL,
              grades_json TEXT NOT NULL, subjects_json TEXT NOT NULL, brand TEXT NOT NULL,
              status TEXT NOT NULL, confidence TEXT NOT NULL, reason TEXT NOT NULL,
              source_refs_json TEXT NOT NULL, evidence_count INTEGER NOT NULL, created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL, record_json TEXT NOT NULL,
              PRIMARY KEY (tenant_id, family_id, customer_id, child_key)
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS ix_family_links_v1_customer "
            "ON family_links_v1(tenant_id,customer_id,status,confidence)"
        )
        for family_id, customer_id in (
            ("family:graduate", "customer:graduate"),
            ("family:mixed", "customer:mixed-graduate"),
            ("family:mixed", "customer:mixed-active"),
        ):
            con.execute(
                "INSERT INTO family_members_v1 VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    "foton", family_id, customer_id, "confident", "high", "test",
                    NOW.isoformat(), NOW.isoformat(), f"hash:{customer_id}", "{}",
                ),
            )
        for family_id, customer_id, child_key, student_type in (
            ("family:graduate", "customer:graduate", "child:graduate", "Выпускник"),
            ("family:mixed", "customer:mixed-graduate", "child:mixed-graduate", "Выпускник"),
            ("family:mixed", "customer:mixed-active", "child:mixed-active", "8"),
        ):
            con.execute(
                "INSERT INTO family_links_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "foton", family_id, customer_id, child_key, child_key, "[]",
                    json.dumps([student_type], ensure_ascii=False), "[]", "foton", "confident", "high",
                    "test", "[]", 1, NOW.isoformat(), f"hash:{child_key}",
                    json.dumps({"grades": [student_type]}, ensure_ascii=False),
                ),
            )
        con.execute(
            "UPDATE customer_identities SET updated_at=? WHERE customer_id='customer:graduate'",
            ((NOW + timedelta(hours=1)).isoformat(),),
        )
        con.execute(
            """
            INSERT INTO timeline_events
            (event_id,dedupe_key,tenant_id,customer_id,opportunity_id,event_type,event_at,
             source_system,source_id,source_ref,direction,match_status,content_key,superseded_by,
             confidence,importance,subject,text_preview,summary,created_at,record_hash,record_json)
            SELECT 'event:graduate-limit-poison','dedupe:graduate-limit-poison',tenant_id,
                   customer_id,opportunity_id,event_type,?,source_system,'graduate-limit-poison',
                   source_ref,direction,match_status,NULL,superseded_by,confidence,importance,
                   subject,text_preview,summary,?,'hash:graduate-limit-poison',
                   json_set(record_json,'$.event_id','event:graduate-limit-poison',
                                       '$.source_id','graduate-limit-poison','$.event_at',?)
            FROM timeline_events
            WHERE customer_id='customer:graduate' AND source_id='scope-poison-1'
            """,
            (
                (NOW + timedelta(hours=1)).isoformat(),
                (NOW + timedelta(hours=1)).isoformat(),
                (NOW + timedelta(hours=1)).isoformat(),
            ),
        )
        con.commit()

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        traced_sql: list[str] = []
        api.store._con.set_trace_callback(traced_sql.append)
        listed = api.list_customers("foton", limit=100)
        searched = api.search(
            "foton", "scope-poison", scopes=("events",), as_of=NOW + timedelta(days=1), limit=100,
        )
        first_listed = api.list_customers("foton", limit=1)
        first_searched = api.search(
            "foton", "scope-poison", scopes=("events",), as_of=NOW + timedelta(days=1), limit=1,
        )
        graduate_history = api.customer_profile("foton", "customer:graduate", as_of=NOW + timedelta(days=1))
        graduate_bot = api.bot_context(
            "foton", "customer:graduate", allowed_only=True, as_of=NOW + timedelta(days=1)
        )
        graduate_internal = api.bot_context(
            "foton", "customer:graduate", allowed_only=False, as_of=NOW + timedelta(days=1)
        )
        mixed_bot = api.bot_context(
            "foton", "customer:mixed-active", allowed_only=True, as_of=NOW + timedelta(days=1)
        )
        api.store._con.set_trace_callback(None)

    listed_ids = {item["customer_id"] for item in listed["items"]}
    searched_ids = {item["record"]["customer_id"] for item in searched["result"]["items"]}
    assert "customer:graduate" not in listed_ids
    assert "customer:graduate" not in searched_ids
    assert {"customer:mixed-graduate", "customer:mixed-active"} <= listed_ids
    assert {"customer:mixed-graduate", "customer:mixed-active"} <= searched_ids
    assert len(first_listed["items"]) == 1
    assert first_listed["items"][0]["customer_id"] != "customer:graduate"
    assert len(first_searched["result"]["items"]) == 1
    assert first_searched["result"]["items"][0]["record"]["customer_id"] != "customer:graduate"
    assert graduate_history["found"] is True
    assert graduate_history["timeline"]["items"]
    assert graduate_bot["out_of_scope"] is True
    assert graduate_bot["summary"]["out_of_scope"] is True
    assert graduate_bot["items"] == []
    assert graduate_bot["summary"]["allowed_chunks"] == 0
    assert graduate_internal["out_of_scope"] is False
    assert graduate_internal["items"]
    assert mixed_bot["out_of_scope"] is False
    assert mixed_bot["items"]
    family_link_reads = [sql for sql in traced_sql if "FROM family_links_v1" in sql]
    assert family_link_reads
    assert all("customer_id IN" in sql for sql in family_link_reads)


def test_read_api_skips_malformed_record_without_crashing_customer_memory(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE bot_context_chunks SET record_json='{bad' WHERE allowed_for_bot=1"
        )
        con.commit()

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        result = api.bot_context("foton", customer_id, allowed_only=True)

    assert result["items"] == []


@pytest.mark.parametrize(
    ("source_system", "corrupt_event"),
    (
        ("mail_archive_stage2", False),
        ("mail_archive_stage2", True),
        ("customer_timeline_bot_safe_summary", False),
    ),
)
def test_bot_safe_boundary_rejects_malformed_protected_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_system: str,
    corrupt_event: bool,
) -> None:
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE", "1")
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE_ALLOW_TEST_PATHS", "1")
    db_path, customer_id = seed_timeline_db(tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        event = TimelineEvent(
            tenant_id="foton",
            customer_id=customer_id,
            event_type="email_message",
            event_at=NOW + timedelta(minutes=2),
            source_system="mail_archive_stage2",
            source_id=f"malformed-{source_system}-{corrupt_event}",
            direction="inbound",
            match_status="strong_unique",
            metadata={"brand_context_authorized": True, "client_safe": True},
            created_at=NOW + timedelta(minutes=2),
        )
        store.upsert_event(event)
        chunk = BotContextChunk(
            tenant_id="foton",
            customer_id=customer_id,
            chunk_id=f"malformed-{source_system}-{corrupt_event}",
            event_id=event.event_id if source_system == "mail_archive_stage2" else None,
            source_system=source_system,
            source_ref="malformed-probe",
            chunk_type="email_message",
            text="Эта поврежденная запись не должна быть видна.",
            allowed_for_bot=True,
            requires_manager_review=False,
            metadata={"brand_context_authorized": True},
            created_at=NOW + timedelta(minutes=2),
        )
        store.upsert_bot_context_chunk(chunk)

    with sqlite3.connect(db_path) as con:
        table = "timeline_events" if corrupt_event else "bot_context_chunks"
        key = event.event_id if corrupt_event else chunk.chunk_id
        key_column = "event_id" if corrupt_event else "chunk_id"
        con.execute(f"UPDATE {table} SET record_json='{{bad' WHERE {key_column}=?", (key,))
        con.commit()

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        result = api.bot_context("foton", customer_id, allowed_only=True, limit=50)

    assert all(item.get("chunk_id") != chunk.chunk_id for item in result["items"])


def test_read_api_bot_context_blocks_raw_mail_even_when_stored_open(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE", "1")
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE_ALLOW_TEST_PATHS", "1")
    db_path, customer_id = seed_timeline_db(tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        event = TimelineEvent(
            tenant_id="foton",
            customer_id=customer_id,
            event_type="email_message",
            event_at=NOW + timedelta(minutes=3),
            source_system="mail_archive_stage2",
            source_id="bab8a94ccfc211a7e15956076b3d7d00519bde54efc3fdc3e5a855fba546b093",
            direction="inbound",
            match_status="strong_unique",
            metadata={"brand_context_authorized": True},
            created_at=NOW + timedelta(minutes=3),
        )
        store.upsert_event(event)
        for chunk_id, source_ref, text in (
            ("mail-1", "a2v3_mail:120:bab8a94c", "Первый вариант письма."),
            ("mail-2", "mail_stage2:stage2_full:7450:bab8a94c", "Дубль того же письма."),
        ):
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id=customer_id,
                    chunk_id=chunk_id,
                    event_id=event.event_id,
                    source_system="mail_archive_stage2",
                    source_ref=source_ref,
                    chunk_type="email_message",
                    text=text,
                    summary=text,
                    event_at=NOW + timedelta(minutes=3),
                    relevance_tags=("email", "bot_visible", "mail_archive_stage2", "foton"),
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata={
                        "message_sha256": "bab8a94ccfc211a7e15956076b3d7d00519bde54efc3fdc3e5a855fba546b093",
                        "brand_context_authorized": True,
                        "client_safe": True,
                    },
                    created_at=NOW + timedelta(minutes=3),
                )
            )

    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        context = api.bot_context("foton", customer_id, allowed_only=True, limit=20)

    mail_items = [item for item in context["items"] if item.get("source_system") == "mail_archive_stage2"]
    assert mail_items == []
    assert context["summary"]["total_chunks"] > context["summary"]["allowed_chunks"]
    assert "Первый вариант письма" not in json.dumps(context, ensure_ascii=False)
    assert "Дубль того же письма" not in json.dumps(context, ensure_ascii=False)


def test_bot_safe_boundary_requires_boolean_brand_authorization_on_event_and_chunk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE", "1")
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE_ALLOW_TEST_PATHS", "1")
    db_path, customer_id = seed_timeline_db(tmp_path)
    cases = (
        ("auth-good", True, True),
        ("auth-event-false", False, True),
        ("auth-chunk-false", True, False),
        ("auth-missing", None, None),
        ("auth-string", "true", "true"),
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for suffix, event_auth, chunk_auth in cases:
            event_metadata = {} if event_auth is None else {"brand_context_authorized": event_auth}
            chunk_metadata = {"client_safe": True}
            if chunk_auth is not None:
                chunk_metadata["brand_context_authorized"] = chunk_auth
            event = TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type="email_message",
                event_at=NOW + timedelta(minutes=4),
                source_system="mail_archive_stage2",
                source_id=suffix,
                direction="inbound",
                match_status="strong_unique",
                metadata=event_metadata,
                created_at=NOW + timedelta(minutes=4),
            )
            store.upsert_event(event)
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id=customer_id,
                    chunk_id=suffix,
                    event_id=event.event_id,
                    source_system="mail_archive_stage2",
                    source_ref=f"mail:{suffix}",
                    chunk_type="email_message",
                    text=f"brandgateprobe {suffix}",
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata=chunk_metadata,
                    created_at=NOW + timedelta(minutes=4),
                )
            )
        store.upsert_bot_context_chunk(
            BotContextChunk(
                tenant_id="foton",
                customer_id=customer_id,
                chunk_id="auth-orphan",
                source_system="mail_archive_stage2",
                source_ref="mail:auth-orphan",
                chunk_type="email_message",
                text="brandgateprobe orphan",
                allowed_for_bot=True,
                requires_manager_review=False,
                metadata={"brand_context_authorized": True, "client_safe": True},
                created_at=NOW + timedelta(minutes=4),
            )
        )
        for suffix, authorized in (("summary-good", True), ("summary-missing", None)):
            metadata = {"client_safe": True}
            if authorized is not None:
                metadata["brand_context_authorized"] = authorized
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id=customer_id,
                    chunk_id=suffix,
                    source_system="customer_timeline_bot_safe_summary",
                    source_ref=f"botsafe:{suffix}",
                    chunk_type="bot_safe_summary",
                    text=f"brandgateprobe {suffix}",
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata=metadata,
                    created_at=NOW + timedelta(minutes=4),
                )
            )

    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        context = api.bot_context("foton", customer_id, allowed_only=True, limit=50)
        search = api.search(
            "foton",
            "brandgateprobe",
            customer_id=customer_id,
            scopes=("events", "bot_context", "signals"),
            allowed_for_bot=True,
            limit=50,
        )

    visible = {item["chunk_id"] for item in context["items"] if str(item.get("chunk_id") or "").startswith(("auth-", "summary-"))}
    assert visible == {"summary-good"}
    assert {item["scope"] for item in search["result"]["items"]} == {"bot_context"}
    assert {item["id"] for item in search["result"]["items"]} == {"summary-good"}


def test_bot_safe_reader_rejects_poisoned_raw_sources_even_with_all_env_bypasses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE", "1")
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE_ALLOW_TEST_PATHS", "1")
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_CHANNEL_HISTORY_BOT_VISIBLE", "1")
    monkeypatch.setenv("CUSTOMER_TIMELINE_E4B_CHANNEL_HISTORY_BOT_VISIBLE_ALLOW_TEST_PATHS", "1")
    db_path, customer_id = seed_timeline_db(tmp_path)
    forbidden_sources = (
        "mail_archive_stage2",
        "wappi_telegram",
        "wappi_max",
        "mango_processed_summary",
        "amocrm_event",
    )
    chunk_ids: list[str] = []
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index, source_system in enumerate((*forbidden_sources, "customer_timeline_bot_safe_summary")):
            event = TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type="system_note",
                event_at=NOW + timedelta(minutes=10 + index),
                source_system="customer_timeline_bot_safe_summary",
                source_id=f"read-boundary-{index}",
                direction="system",
                match_status="strong_unique",
                metadata={"brand_context_authorized": True},
                created_at=NOW + timedelta(minutes=10 + index),
            )
            store.upsert_event(event)
            chunk_id = f"read-boundary-{index}"
            chunk_ids.append(chunk_id)
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id=customer_id,
                    chunk_id=chunk_id,
                    event_id=event.event_id,
                    source_system="customer_timeline_bot_safe_summary",
                    source_ref=f"read-boundary:{index}",
                    chunk_type="bot_safe_summary",
                    text=f"readboundarypoison {source_system}",
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata={"brand_context_authorized": True, "client_safe": True},
                    created_at=NOW + timedelta(minutes=10 + index),
                )
            )
        store._con.executemany(  # noqa: SLF001 - poison fixture bypasses the writer gate intentionally.
            "UPDATE bot_context_chunks SET source_system=? WHERE chunk_id=?",
            zip(forbidden_sources, chunk_ids[:-1]),
        )
        store._con.commit()  # noqa: SLF001

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        context = api.bot_context("foton", customer_id, allowed_only=True, limit=50)
        fts = api.search(
            "foton",
            "readboundarypoison",
            customer_id=customer_id,
            allowed_for_bot=True,
            limit=50,
        )
        api.store._fts_enabled = False  # noqa: SLF001 - exercise the SQL fallback boundary too.
        fallback = api.search(
            "foton",
            "readboundarypoison",
            customer_id=customer_id,
            allowed_for_bot=True,
            limit=50,
        )

    expected = {chunk_ids[-1]}
    assert {item["chunk_id"] for item in context["items"] if item["chunk_id"] in chunk_ids} == expected
    assert {item["id"] for item in fts["result"]["items"]} == expected
    assert {item["id"] for item in fallback["result"]["items"]} == expected


def test_bot_safe_reader_uses_one_strict_as_of_cutoff_before_limit_and_search(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    future_at = NOW + timedelta(days=30)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for chunk_id, chunk_type, event_at in (
            ("safe-past", "bot_safe_summary", NOW - timedelta(minutes=1)),
            ("safe-future-purchase", "purchase_history", future_at),
            ("safe-malformed", "bot_safe_summary", NOW - timedelta(minutes=2)),
        ):
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id=customer_id,
                    chunk_id=chunk_id,
                    source_system="customer_timeline_bot_safe_summary",
                    source_ref=f"temporal:{chunk_id}",
                    chunk_type=chunk_type,
                    text=f"temporalprobe {chunk_id}",
                    event_at=event_at,
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata={"brand_context_authorized": True, "client_safe": True},
                    created_at=event_at,
                )
            )
        store._con.execute(  # noqa: SLF001 - malformed legacy fixture bypasses contracts intentionally.
            "UPDATE bot_context_chunks SET event_at='not-a-time',created_at='not-a-time' WHERE chunk_id='safe-malformed'"
        )
        store._con.commit()  # noqa: SLF001

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        current = api.bot_context("foton", customer_id, allowed_only=True, as_of=NOW, limit=50)
        current_search = api.search(
            "foton", "temporalprobe", customer_id=customer_id, allowed_for_bot=True, as_of=NOW, limit=50
        )
        later = api.bot_context(
            "foton", customer_id, allowed_only=True, as_of=future_at + timedelta(seconds=1), limit=50
        )
        api.store._fts_enabled = False  # noqa: SLF001 - prove the fallback uses the same cutoff.
        later_fallback = api.search(
            "foton",
            "temporalprobe",
            customer_id=customer_id,
            allowed_for_bot=True,
            as_of=future_at + timedelta(seconds=1),
            limit=50,
        )

    assert {item["chunk_id"] for item in current["items"] if item["chunk_id"].startswith("safe-")} == {"safe-past"}
    assert {item["id"] for item in current_search["result"]["items"]} == {"safe-past"}
    assert {item["chunk_id"] for item in later["items"] if item["chunk_id"].startswith("safe-")} == {
        "safe-past",
        "safe-future-purchase",
    }
    assert {item["id"] for item in later_fallback["result"]["items"]} == {
        "safe-past",
        "safe-future-purchase",
    }


def test_bot_safe_reader_fails_closed_on_client_safe_metadata(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for chunk_id, metadata in (
            ("client-safe-true", {"client_safe": True}),
            ("client-safe-false", {"client_safe": False}),
            ("client-safe-missing", {}),
            ("client-safe-string", {"client_safe": "true"}),
            ("client-safe-malformed", {}),
        ):
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id=customer_id,
                    chunk_id=chunk_id,
                    source_system="trusted_summary",
                    source_ref=f"client-safe:{chunk_id}",
                    chunk_type="bot_safe_summary",
                    text=f"clientsafeprobe {chunk_id}",
                    event_at=NOW - timedelta(minutes=1),
                    allowed_for_bot=True,
                    requires_manager_review=False,
                    metadata=metadata,
                    created_at=NOW - timedelta(minutes=1),
                )
            )
        store._con.execute(  # noqa: SLF001 - malformed legacy poison bypasses the writer contract.
            "UPDATE bot_context_chunks SET record_json='not-json' WHERE chunk_id='client-safe-malformed'"
        )
        store._con.commit()  # noqa: SLF001

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        context = api.bot_context("foton", customer_id, allowed_only=True, limit=50)
        fts = api.search(
            "foton",
            "clientsafeprobe",
            customer_id=customer_id,
            allowed_for_bot=True,
            limit=50,
        )
        api.store._fts_enabled = False  # noqa: SLF001 - exercise the same SQL fallback gate.
        fallback = api.search(
            "foton",
            "clientsafeprobe",
            customer_id=customer_id,
            allowed_for_bot=True,
            limit=50,
        )

    assert {item["chunk_id"] for item in context["items"] if item["chunk_id"].startswith("client-safe-")} == {
        "client-safe-true"
    }
    assert {item["id"] for item in fts["result"]["items"]} == {"client-safe-true"}
    assert {item["id"] for item in fallback["result"]["items"]} == {"client-safe-true"}


def test_read_api_summary_open_conflicts_is_global_not_recent_limit(tmp_path: Path) -> None:
    db_path, _ = seed_timeline_db(tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index in range(3):
            store.record_conflict(
                "foton",
                conflict_type="audit_probe",
                entity_refs=(f"probe:{index}",),
                actor="test",
            )

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        result = api.summary("foton", recent_limit=1)

    assert result["summary"]["open_conflicts"] == 4
    assert result["recent_conflicts"]["summary"]["open_conflicts"] == 4
    assert result["recent_conflicts"]["summary"]["recent_window"]["returned"] == 1


def test_read_api_conflict_owner_metrics_are_global_not_page_limited(tmp_path: Path) -> None:
    db_path, first_customer_id = seed_timeline_db(tmp_path)
    with sqlite3.connect(db_path) as con:
        second_customer_id = str(con.execute(
            "SELECT customer_id FROM customer_identities WHERE customer_id != ?",
            (first_customer_id,),
        ).fetchone()[0])
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index in range(205):
            store.record_conflict(
                "foton",
                conflict_type="owner_metric_probe",
                entity_refs=(
                    first_customer_id if index % 2 == 0 else second_customer_id,
                    f"probe:{index}",
                ),
                severity="high" if index % 2 == 0 else "low",
                actor="test",
            )
        store.record_conflict(
            "foton",
            conflict_type="resolved_owner_metric_probe",
            entity_refs=(f"customer:{first_customer_id}", "probe:resolved"),
            severity="critical",
            status="resolved",
            actor="test",
        )

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        result = api.list_conflicts("foton", limit=1)

    summary = result["summary"]
    assert len(result["items"]) == 1
    assert summary["total"] == 207
    assert summary["open_conflicts"] == 206
    assert summary["affected_customer_count"] == 2
    assert summary["open_affected_customer_count"] == 2
    assert summary["by_severity"] == {"critical": 1, "high": 103, "low": 102, "medium": 1}
    assert summary["open_by_severity"] == {"high": 103, "low": 102, "medium": 1}
    assert summary["recent_window"]["returned"] == 1

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        first_open = api.list_conflicts(
            "foton",
            customer_id=first_customer_id,
            status="open",
            conflict_type="owner_metric_probe",
            limit=1,
        )["summary"]
        resolved = api.list_conflicts(
            "foton",
            customer_id=first_customer_id,
            status="resolved",
            conflict_type="resolved_owner_metric_probe",
            limit=1,
        )["summary"]

    assert first_open["total"] == 103
    assert first_open["affected_customer_count"] == 1
    assert first_open["by_severity"] == {"high": 103}
    assert resolved["total"] == 1
    assert resolved["open_conflicts"] == 0
    assert resolved["affected_customer_count"] == 1
    assert resolved["open_affected_customer_count"] == 0
    assert resolved["open_by_severity"] == {}


def test_read_api_active_exact_link_conflict_blocks_customer_safety(tmp_path: Path) -> None:
    db_path, customer_id = seed_timeline_db(tmp_path)
    with sqlite3.connect(db_path) as con:
        con.execute("DELETE FROM timeline_conflicts")
        con.execute(
            "UPDATE identity_links SET source_ref='amocrm:contact:contact-raw-1' "
            "WHERE tenant_id='foton' AND link_type='amo_contact_id' AND link_value='contact-raw-1'"
        )
        con.execute(
            "INSERT INTO family_members_v1 VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "foton",
                "family:active-conflict-probe",
                customer_id,
                "confident",
                "high",
                "test",
                NOW.isoformat(),
                NOW.isoformat(),
                "hash:active-conflict-probe",
                "{}",
            ),
        )
        con.commit()
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.record_conflict(
            "foton",
            conflict_type="active_exact_link_probe",
            entity_refs=("amo:contact:contact-raw-1",),
            severity="high",
            status="active",
            actor="test",
        )
        store.record_conflict(
            "foton",
            conflict_type="active_family_ref_probe",
            entity_refs=("family:active-conflict-probe",),
            severity="medium",
            status="active",
            actor="test",
        )
        store.record_conflict(
            "foton",
            conflict_type="resolved_source_ref_probe",
            entity_refs=("amo:contact:contact-raw-1",),
            severity="low",
            status="resolved",
            actor="test",
        )

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        active = api.list_conflicts("foton", customer_id=customer_id, status="active")
        global_active = api.list_conflicts("foton", status="active", limit=1)
        global_resolved = api.list_conflicts("foton", status="resolved", limit=1)
        owner_summary = api.summary("foton", recent_limit=1)
        profile = api.customer_profile("foton", customer_id)

    assert len(active["items"]) == 2
    assert active["summary"]["open_conflicts"] == 2
    assert active["summary"]["open_affected_customer_count"] == 1
    assert global_active["summary"]["total"] == 2
    assert global_active["summary"]["open_conflicts"] == 2
    assert global_active["summary"]["affected_customer_count"] == 1
    assert global_active["summary"]["open_by_severity"] == {"high": 1, "medium": 1}
    assert global_resolved["summary"]["total"] == 1
    assert global_resolved["summary"]["affected_customer_count"] == 1
    assert global_resolved["summary"]["open_conflicts"] == 0
    assert owner_summary["summary"]["open_conflicts"] == 2
    assert profile["readiness"]["open_conflicts"] == 2
    assert profile["readiness"]["safe_for_automatic_bot"] is False


def test_read_api_reports_malformed_conflict_without_crashing(tmp_path: Path) -> None:
    db_path, _ = seed_timeline_db(tmp_path)
    with sqlite3.connect(db_path) as con:
        con.execute("DELETE FROM timeline_conflicts")
        con.execute(
            "INSERT INTO timeline_conflicts VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "timeline_conflict:malformed",
                "foton",
                "malformed_payload_probe",
                "medium",
                "open",
                NOW.isoformat(),
                None,
                "hash:malformed",
                "{bad json",
            ),
        )
        con.commit()

    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)
    ) as api:
        result = api.list_conflicts("foton", limit=10)

    assert result["items"] == []
    assert result["summary"]["total"] == 1
    assert result["summary"]["open_conflicts"] == 1
    assert result["summary"]["malformed_conflict_payload_count"] == 1
    assert result["summary"]["recent_window"]["returned"] == 0


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
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer.customer_id,
            link_type="amo_contact_id",
            link_value="contact-raw-1",
            source_system="amocrm_snapshot",
            source_ref="amo:contact:contact-raw-1",
            match_class="strong_unique",
            confidence=0.9,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer.customer_id,
            link_type="amo_lead_id",
            link_value="lead-raw-1",
            source_system="amocrm_snapshot",
            source_ref="amo:lead:lead-raw-1",
            match_class="strong_unique",
            confidence=0.85,
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
    long_call_summary = "Полный разбор звонка. " + ("Клиент обсуждал цену, формат и следующий шаг. " * 30)
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
        summary=long_call_summary,
        importance=3,
        match_status="strong_unique",
        confidence=0.9,
        record={
            "raw_payload": {"hidden": True},
            "audio_path": "/secret/audio.mp3",
            "call_analysis": {
                "history_summary": long_call_summary,
                "structured_fields": {"student": {"grade_current": "9"}},
                "call_type": "sales_call",
                "call_history_eligible": True,
                "objections": ["цена"],
                "next_step": "Перезвонить",
            },
            "call_type": "sales_call",
            "call_history_eligible": True,
        },
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
            metadata={
                "client_safe": True,
                "raw_file": "hidden",
                "next_step": {"status": "needs_manager_review", "display_text": "Спорный текст шага"},
            },
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


def seed_ready_manager_action(
    db_path: Path,
    tmp_path: Path,
    *,
    customer_id: str,
    as_of: datetime,
) -> None:
    """Shared synthetic fixture for a fully proven open AMO manager task."""
    due_at = as_of + timedelta(days=1)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute(
            "UPDATE customer_opportunities SET source_system='amocrm_snapshot',status='open',closed_at=NULL "
            "WHERE tenant_id='foton' AND customer_id=? AND source_id='lead-1'",
            (customer_id,),
        )
        opportunity_id = str(con.execute(
            "SELECT opportunity_id FROM customer_opportunities "
            "WHERE tenant_id='foton' AND customer_id=? AND source_id='lead-1'",
            (customer_id,),
        ).fetchone()["opportunity_id"])
        con.execute(
            "INSERT OR REPLACE INTO ingestion_cursors "
            "(tenant_id,source_system,last_cursor_ts,updated_at,metadata_json) VALUES (?,?,?,?,?)",
            (
                "foton", "amo_tasks_updated_at", (as_of - timedelta(minutes=2)).isoformat(),
                (as_of - timedelta(minutes=1)).isoformat(),
                json.dumps({"metadata": {"bootstrap_complete": True, "last_status": "ok"}}),
            ),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO ingestion_runs
            (run_id,tenant_id,source_system,source_ref,run_kind,idempotency_key,status,
             started_at,finished_at,input_hash,accepted_count,rejected_count,output_ref,error,
             record_hash,record_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "run:approval-ready-task", "foton", "amocrm_snapshot", "amocrm:tasks:updated_at",
                "amo_tasks_incremental", "approval-ready-fixture", "completed",
                (as_of - timedelta(minutes=3)).isoformat(), (as_of - timedelta(minutes=1)).isoformat(),
                "fixture", 1, 0, None, None, "fixture", "{}",
            ),
        )
        con.commit()
    task_id = "task:approval-ready"
    action = "Позвонить и согласовать расписание"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                event_type="amo_task",
                event_at=as_of - timedelta(minutes=2),
                source_system="amocrm_snapshot",
                source_id=task_id,
                source_ref=f"amo:task:{task_id}",
                direction="internal",
                actor_name="Анна Менеджер",
                actor_ref="amo:user:17",
                summary="Open AMO task",
                text_preview=action,
                record={
                    "action_text": action,
                    "next_step": {"action": action, "due": due_at.isoformat()},
                    "responsible_user_id": "17",
                    "responsible_user_name": "Анна Менеджер",
                    "complete_till": due_at.isoformat(),
                    "completed": False,
                    "provenance": {
                        "task_id": task_id,
                        "entity_type": "leads",
                        "entity_id": "lead-1",
                        "opportunity_source_system": "amocrm_snapshot",
                        "opportunity_source_id": "lead-1",
                    },
                },
                metadata={"actor_role": "manager"},
                match_status="strong_unique",
                created_at=as_of - timedelta(minutes=2),
            )
        )
