from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.store as store_module
from mango_mvp.customer_timeline import (
    CUSTOMER_TIMELINE_SQLITE_MIGRATION_ID,
    CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
    ArtifactType,
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    CustomerTimelineSQLiteStore,
    DerivedSignal,
    EventArtifact,
    ExtractionStatus,
    IdentityLink,
    IdentityStatus,
    OpportunityType,
    SignalSeverity,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)


NOW = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
SHA = "a" * 64


class StepClock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


def identity(*, tenant_id: str = "foton", phone: str = "+79161234567") -> CustomerIdentity:
    return CustomerIdentity(
        tenant_id=tenant_id,
        identity_status=IdentityStatus.STRONG,
        display_name="Иванова Мария",
        primary_phone=phone,
        primary_email=f"{tenant_id}@example.com",
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=1,
        created_at=NOW,
        updated_at=NOW,
    )


def identity_link(customer: CustomerIdentity) -> IdentityLink:
    return IdentityLink(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        link_type="phone",
        link_value=customer.primary_phone,
        source_system="tallanto_export",
        source_ref=f"Ученики.csv#{customer.customer_id}",
        match_class="strong_unique",
        confidence=0.95,
        first_seen_at=NOW,
        last_seen_at=NOW,
    )


def opportunity(customer: CustomerIdentity, *, source_id: str = "lead-1") -> CustomerOpportunity:
    return CustomerOpportunity(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        opportunity_type=OpportunityType.AMO_DEAL,
        source_system="amocrm_snapshot",
        source_id=source_id,
        title="ЕГЭ математика",
        status="open",
        opened_at=NOW,
        confidence=0.8,
    )


def event(
    customer: CustomerIdentity,
    opp: CustomerOpportunity | None = None,
    *,
    source_id: str = "call-1",
    summary: str = "Клиент спросил стоимость курса и попросил перезвонить.",
    tenant_id: str | None = None,
) -> TimelineEvent:
    return TimelineEvent(
        tenant_id=tenant_id or customer.tenant_id,
        customer_id=customer.customer_id,
        opportunity_id=opp.opportunity_id if opp else None,
        event_type=TimelineEventType.MANGO_CALL,
        event_at=NOW,
        source_system="mango",
        source_id=source_id,
        direction=TimelineDirection.INBOUND,
        actor_name="Клиент",
        subject="Вопрос про стоимость",
        text_preview="Сколько стоит подготовка к ЕГЭ?",
        summary=summary,
        importance=3,
        match_status="strong_unique",
        confidence=0.9,
        record={
            "visible": "ok",
            "raw_payload": {"secret": "must_not_be_stored"},
        },
        metadata={"provider_raw_payload": {"token": "must_not_be_stored"}},
        created_at=NOW,
    )


def artifact(ev: TimelineEvent, *, tenant_id: str | None = None) -> EventArtifact:
    return EventArtifact(
        tenant_id=tenant_id or ev.tenant_id,
        event_id=ev.event_id,
        artifact_type=ArtifactType.CALL_TRANSCRIPT_JSON,
        path="/not/read/transcript.json",
        sha256=SHA,
        size_bytes=128,
        mime_type="application/json",
        source_system="processing_export",
        source_ref=ev.event_id,
        extraction_status=ExtractionStatus.EXTRACTED,
        created_at=NOW,
    )


def signal(ev: TimelineEvent) -> DerivedSignal:
    return DerivedSignal(
        tenant_id=ev.tenant_id,
        customer_id=ev.customer_id,
        opportunity_id=ev.opportunity_id,
        event_id=ev.event_id,
        source_event_ids=(ev.event_id,),
        signal_type="price_interest",
        severity=SignalSeverity.HIGH,
        evidence_text="Клиент явно спросил стоимость.",
        confidence=0.88,
        requires_manager_review=True,
        metadata={"raw_payload": {"secret": "must_not_be_stored"}},
        created_at=NOW,
    )


def chunk(ev: TimelineEvent) -> BotContextChunk:
    return BotContextChunk(
        tenant_id=ev.tenant_id,
        customer_id=ev.customer_id,
        opportunity_id=ev.opportunity_id,
        event_id=ev.event_id,
        source_ref=ev.event_id,
        source_system=ev.source_system,
        chunk_type="sales_context",
        text="Клиент спрашивал стоимость и ждет звонок менеджера.",
        summary="Интерес к цене",
        event_at=ev.event_at,
        freshness_score=0.9,
        relevance_tags=("sales", "price"),
        allowed_for_bot=True,
        requires_manager_review=False,
        created_at=NOW,
    )


def open_store(tmp_path: Path) -> CustomerTimelineSQLiteStore:
    return CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=StepClock())


def table_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    return {row[0] for row in rows}


def test_sqlite_store_bootstraps_reopens_and_reports_safety(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    store.close()

    reopened = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    summary = reopened.summary()
    names = table_names(db_path)

    assert CUSTOMER_TIMELINE_SQLITE_MIGRATION_ID
    assert "schema_migrations" in names
    assert "customer_identities" in names
    assert "timeline_events" in names
    assert "timeline_conflicts" in names
    assert "customer_id_mappings" in names
    assert summary["schema_version"] == CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION
    assert summary["backend"] == "sqlite"
    assert summary["counts"]["schema_migrations"] == 1
    assert summary["counts"]["timeline_events"] == 0
    assert summary["counts"]["customer_id_mappings"] == 0
    assert summary["validation_ok"] is True
    assert summary["safety"]["write_crm"] is False
    assert summary["safety"]["write_tallanto"] is False
    assert summary["safety"]["write_runtime_db"] is False
    assert summary["safety"]["stable_runtime_writes"] is False
    assert summary["safety"]["store_raw_files_in_sqlite"] is False
    assert summary["safety"]["old_to_new_customer_id_mapping_required"] is True
    assert summary["safety"]["brand_blocks_identity_merge"] is False
    assert reopened._con.execute("PRAGMA foreign_key_check").fetchall() == []
    reopened.close()


def test_sqlite_store_uses_wal_and_read_only_mode_blocks_mutations(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer = identity()
    writable = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    writable.upsert_customer(customer)
    journal = writable._con.execute("PRAGMA journal_mode").fetchone()[0]
    writable.close()

    readonly = CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path)
    assert journal.lower() == "wal"
    assert readonly._con.execute("PRAGMA query_only").fetchone()[0] == 1
    assert readonly.summary()["counts"]["customer_identities"] == 1
    with pytest.raises(PermissionError, match="read-only"):
        readonly.upsert_customer(identity(phone="+79169876543"))
    with pytest.raises(PermissionError, match="read-only"):
        readonly.append_audit_log("foton", action="manual_note", entity_type="customer_identity")
    assert readonly.summary()["counts"]["customer_identities"] == 1
    readonly.close()


def test_read_only_missing_db_fails_without_creating_file(tmp_path: Path) -> None:
    db_path = tmp_path / "missing" / "customer_timeline.sqlite"

    with pytest.raises(sqlite3.OperationalError):
        CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path)

    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_bulk_write_defers_commit_and_rolls_back_on_error(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    first = identity(phone="+79160000001")

    with store.bulk_write():
        store.upsert_customer(first)
        with sqlite3.connect(db_path) as external:
            assert external.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0] == 0

    with sqlite3.connect(db_path) as external:
        assert external.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0] == 1

    with pytest.raises(RuntimeError, match="abort bulk"):
        with store.bulk_write():
            store.upsert_customer(identity(phone="+79160000002"))
            raise RuntimeError("abort bulk")

    with sqlite3.connect(db_path) as external:
        assert external.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0] == 1
    store.close()


def test_path_guard_rejects_runtime_outside_and_stable_runtime_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineSQLiteStore(tmp_path / "stable_runtime" / "customer_timeline.sqlite", allowed_root=tmp_path)

    runtime_dir = tmp_path / "stable_runtime"
    runtime_dir.mkdir()
    runtime_link = tmp_path / "timeline_link"
    runtime_link.symlink_to(runtime_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineSQLiteStore(runtime_link / "customer_timeline.sqlite", allowed_root=tmp_path)

    with pytest.raises(ValueError, match="runtime-looking"):
        CustomerTimelineSQLiteStore(tmp_path / "runtime.db", allowed_root=tmp_path)
    with pytest.raises(ValueError, match="runtime-looking"):
        CustomerTimelineSQLiteStore(tmp_path / "mango_product_appliance.sqlite", allowed_root=tmp_path)
    with pytest.raises(ValueError, match="allowed root"):
        CustomerTimelineSQLiteStore(tmp_path.parent / "outside_customer_timeline.sqlite", allowed_root=tmp_path)


def test_upserts_core_records_idempotently_after_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer = identity()
    link = identity_link(customer)
    opp = opportunity(customer)
    ev = event(customer, opp)
    art = artifact(ev)
    sig = signal(ev)
    ctx = chunk(ev)

    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    first_results = [
        store.upsert_customer(customer, actor="importer"),
        store.upsert_identity_link(link, actor="importer"),
        store.upsert_opportunity(opp, actor="importer"),
        store.upsert_event(ev, actor="importer"),
        store.upsert_artifact(art, actor="importer"),
        store.upsert_signal(sig, actor="importer"),
        store.upsert_bot_context_chunk(ctx, actor="importer"),
    ]
    store.close()

    reopened = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    second_results = [
        reopened.upsert_customer(customer, actor="importer"),
        reopened.upsert_identity_link(link, actor="importer"),
        reopened.upsert_opportunity(opp, actor="importer"),
        reopened.upsert_event(ev, actor="importer"),
        reopened.upsert_artifact(art, actor="importer"),
        reopened.upsert_signal(sig, actor="importer"),
        reopened.upsert_bot_context_chunk(ctx, actor="importer"),
    ]
    summary = reopened.summary()

    assert all(result.created is True and result.status == "created" for result in first_results)
    assert all(result.created is False and result.status == "duplicate" for result in second_results)
    assert summary["counts"]["customer_identities"] == 1
    assert summary["counts"]["identity_links"] == 1
    assert summary["counts"]["customer_opportunities"] == 1
    assert summary["counts"]["timeline_events"] == 1
    assert summary["counts"]["event_artifacts"] == 1
    assert summary["counts"]["derived_signals"] == 1
    assert summary["counts"]["bot_context_chunks"] == 1
    assert summary["counts"]["audit_log"] == 7
    assert reopened.get_customer("foton", customer.customer_id)["display_name"] == "Иванова Мария"
    assert reopened.get_event("foton", ev.event_id)["summary"] == ev.summary
    assert reopened.list_events_by_customer(
        "foton",
        customer.customer_id,
        include_artifacts=True,
        include_signals=True,
    )["items"][0]["artifacts"][0]["sha256"] == SHA
    reopened.close()


def test_upsert_updates_existing_record_without_changing_stable_id(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    store.upsert_customer(customer, actor="importer")
    updated = CustomerIdentity(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        identity_status=customer.identity_status,
        display_name="Иванова Мария Петровна",
        primary_phone=customer.primary_phone,
        primary_email=customer.primary_email,
        first_seen_at=customer.first_seen_at,
        last_seen_at=NOW + timedelta(days=1),
        touch_count=2,
        created_at=customer.created_at,
        updated_at=NOW + timedelta(days=1),
    )

    result = store.upsert_customer(updated, actor="importer")
    saved = store.get_customer("foton", customer.customer_id)
    audit = store.list_audit_log("foton", entity_type="customer_identity")["items"]

    assert result.created is False
    assert result.status == "updated"
    assert result.record_id == customer.customer_id
    assert saved["display_name"] == "Иванова Мария Петровна"
    assert saved["touch_count"] == 2
    assert [item["action"] for item in audit] == ["customer_identity_updated", "customer_identity_created"]
    store.close()


def test_customer_id_mapping_is_reversible_idempotent_and_guarded(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    target = identity(phone="+79160000003")
    other = identity(phone="+79160000004")
    store.upsert_customer(target)
    store.upsert_customer(other)

    first = store.record_customer_id_mapping(
        "foton",
        old_customer_id="customer:legacy-a",
        new_customer_id=target.customer_id,
        mapping_kind="merge",
        reason="phone_identity_union",
        source_refs=("amocrm:contact:1", "mango:call:1"),
        actor="identity_resolver",
    )
    second = store.record_customer_id_mapping(
        "foton",
        old_customer_id="customer:legacy-a",
        new_customer_id=target.customer_id,
        mapping_kind="merge",
        reason="phone_identity_union",
        source_refs=("amocrm:contact:1", "mango:call:1"),
        actor="identity_resolver",
    )
    store.record_customer_id_mapping(
        "foton",
        old_customer_id="customer:legacy-b",
        new_customer_id=target.customer_id,
        mapping_kind="merge",
        reason="phone_identity_union",
        source_refs=("tallanto:student:1",),
        actor="identity_resolver",
    )

    mappings = store.list_customer_id_mappings("foton")
    reverse = {target.customer_id: {item["old_customer_id"] for item in mappings if item["new_customer_id"] == target.customer_id}}

    assert first.created is True
    assert second.created is False
    assert second.status == "duplicate"
    assert store.summary()["counts"]["customer_id_mappings"] == 2
    assert {item["old_customer_id"] for item in mappings} == {"customer:legacy-a", "customer:legacy-b"}
    assert reverse == {target.customer_id: {"customer:legacy-a", "customer:legacy-b"}}
    assert mappings[0]["resolution_status"] == "active"
    assert mappings[0]["ingestion_run_id"] is None
    split = store.record_customer_id_mapping(
        "foton",
        old_customer_id="customer:legacy-a",
        new_customer_id=other.customer_id,
        mapping_kind="split",
        reason="manual_override",
    )
    assert split.created is True
    assert {
        item["new_customer_id"]
        for item in store.list_customer_id_mappings("foton", old_customer_id="customer:legacy-a")
    } == {target.customer_id, other.customer_id}
    with pytest.raises(ValueError, match="customer does not exist"):
        store.record_customer_id_mapping(
            "foton",
            old_customer_id="customer:legacy-missing",
            new_customer_id="customer:missing",
            reason="phone_identity_union",
        )
    third = identity(phone="+79160000005")
    store.upsert_customer(third)
    with pytest.raises(ValueError, match="already has active mapping"):
        store.record_customer_id_mapping(
            "foton",
            old_customer_id="customer:legacy-b",
            new_customer_id=third.customer_id,
            mapping_kind="alias",
            reason="manual_override",
        )
    store.close()


def test_store_never_persists_raw_payload_or_reads_artifact_files(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    customer = identity()
    ev = event(customer)
    ev = replace(
        ev,
        record={
            **ev.record,
            "telegram_message": {"text": "must_not_be_stored"},
            "nested": {
                "raw_update": {"token": "must_not_be_stored"},
                "callback_query": {"data": "must_not_be_stored"},
                "tallanto_raw_payload": {"cost": "must_not_be_stored"},
                "raw_finance": {"payment": "must_not_be_stored"},
                "safe_note": "kept",
            },
        },
        metadata={
            **ev.metadata,
            "telegram_raw_message": {"text": "must_not_be_stored"},
            "business_message": {"secret": "must_not_be_stored"},
            "most_finances_payload": {"payment_summa": "must_not_be_stored"},
            "most_abonements_payload": {"num_visit_left": "must_not_be_stored"},
        },
    )
    raw_chunk = replace(
        chunk(ev),
        metadata={
            "telegram_update_payload": {"update_id": "must_not_be_stored"},
            "raw_message": {"text": "must_not_be_stored"},
            "tallanto_api_response": {"records": "must_not_be_stored"},
            "safe_note": "kept",
        },
    )
    raw_file = tmp_path / "source.json"
    raw_file.write_text("raw-file-secret", encoding="utf-8")
    art = EventArtifact(
        tenant_id=ev.tenant_id,
        event_id=ev.event_id,
        artifact_type=ArtifactType.API_RAW_JSON,
        path=str(raw_file),
        sha256=SHA,
        source_system="mango",
        source_ref="call-1",
        extraction_status=ExtractionStatus.NOT_NEEDED,
        metadata={"file_bytes": "must_not_be_stored"},
        created_at=NOW,
    )

    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    store.upsert_customer(customer)
    store.upsert_event(ev)
    store.upsert_artifact(art)
    store.upsert_bot_context_chunk(raw_chunk)
    store.close()

    with sqlite3.connect(db_path) as con:
        dump = "\n".join(
            row[0]
            for row in con.execute(
                """
                SELECT record_json FROM timeline_events
                UNION ALL SELECT record_json FROM event_artifacts
                UNION ALL SELECT record_json FROM bot_context_chunks
                """
            )
        )

    assert "must_not_be_stored" not in dump
    assert "raw_payload" not in dump
    assert "provider_raw_payload" not in dump
    assert "telegram_message" not in dump
    assert "telegram_raw_message" not in dump
    assert "telegram_update_payload" not in dump
    assert "raw_update" not in dump
    assert "raw_message" not in dump
    assert "callback_query" not in dump
    assert "business_message" not in dump
    assert "tallanto_raw_payload" not in dump
    assert "tallanto_api_response" not in dump
    assert "raw_finance" not in dump
    assert "most_finances_payload" not in dump
    assert "most_abonements_payload" not in dump
    assert "file_bytes" not in dump
    assert "raw-file-secret" not in dump
    assert str(raw_file) in dump
    assert "safe_note" in dump


def test_search_uses_fts_or_fallback_for_events_signals_and_chunks(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    ev = event(customer)
    store.upsert_customer(customer)
    store.upsert_event(ev)
    store.upsert_signal(signal(ev))
    store.upsert_bot_context_chunk(chunk(ev))

    result = store.search_timeline("foton", "стоимость", limit=10)
    scopes = {item["scope"] for item in result["items"]}

    assert result["backend"] in {"fts5", "fallback_like"}
    assert "event" in scopes
    assert "signal" in scopes
    assert "bot_context" in scopes
    assert all(item["record"]["tenant_id"] == "foton" for item in result["items"])
    store.close()


def test_search_falls_back_when_fts_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "sqlite_fts5_available", lambda _con: False)
    store = open_store(tmp_path)
    customer = identity()
    ev = event(customer)
    store.upsert_customer(customer)
    store.upsert_event(ev)
    store.upsert_bot_context_chunk(chunk(ev))

    result = store.search_timeline("foton", "стоимость")

    assert result["backend"] == "fallback_like"
    assert {item["scope"] for item in result["items"]} == {"event", "bot_context"}
    store.close()


def test_search_and_unique_keys_are_tenant_scoped(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    foton_customer = identity(tenant_id="foton", phone="+79161234567")
    demo_customer = identity(tenant_id="demo", phone="+79161234567")
    foton_event = event(foton_customer, source_id="shared-call")
    demo_event = event(demo_customer, source_id="shared-call")

    store.upsert_customer(foton_customer)
    store.upsert_customer(demo_customer)
    store.upsert_event(foton_event)
    store.upsert_event(demo_event)

    foton_search = store.search_timeline("foton", "стоимость")
    demo_search = store.search_timeline("demo", "стоимость")

    assert store.summary()["counts"]["customer_identities"] == 2
    assert store.summary()["counts"]["timeline_events"] == 2
    assert {item["record"]["tenant_id"] for item in foton_search["items"]} == {"foton"}
    assert {item["record"]["tenant_id"] for item in demo_search["items"]} == {"demo"}
    store.close()


def test_duplicate_source_event_with_different_explicit_id_does_not_create_second_row(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    first = event(customer, source_id="call-duplicate")
    second = TimelineEvent(
        tenant_id=first.tenant_id,
        customer_id=first.customer_id,
        event_id="timeline_event:manual-different-id",
        event_type=first.event_type,
        event_at=first.event_at,
        source_system=first.source_system,
        source_id=first.source_id,
        direction=first.direction,
        summary="Новая версия с тем же source identity.",
        created_at=NOW,
    )
    store.upsert_customer(customer)
    created = store.upsert_event(first)
    duplicate = store.upsert_event(second)

    assert created.created is True
    assert duplicate.created is False
    assert duplicate.status == "duplicate"
    assert duplicate.record_id == first.event_id
    assert store.summary()["counts"]["timeline_events"] == 1
    assert store.get_event("foton", first.event_id)["summary"] == first.summary
    store.close()


def test_parent_validation_blocks_orphans_and_cross_tenant_references(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    ev = event(customer)

    with pytest.raises(ValueError, match="customer does not exist"):
        store.upsert_identity_link(identity_link(customer))
    with pytest.raises(ValueError, match="customer does not exist"):
        store.upsert_event(ev)

    store.upsert_customer(customer)
    store.upsert_event(ev)

    with pytest.raises(ValueError, match="event does not exist"):
        store.upsert_artifact(artifact(ev, tenant_id="demo"))
    with pytest.raises(TypeError, match="identity must be CustomerIdentity"):
        store.upsert_customer({"tenant_id": "foton"})  # type: ignore[arg-type]
    store.close()


def test_ingestion_runs_conflicts_and_audit_log_are_persistent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    run = store.start_ingestion_run(
        tenant_id="foton",
        source_system="mail_archive",
        source_ref="mail-batch-1",
        run_kind="dry_run_import",
        idempotency_key="mail-batch-1",
        input_hash=SHA,
        actor="mail_importer",
    )
    duplicate_run = store.start_ingestion_run(
        tenant_id="foton",
        source_system="mail_archive",
        source_ref="mail-batch-1",
        run_kind="dry_run_import",
        idempotency_key="mail-batch-1",
        input_hash=SHA,
        actor="mail_importer",
    )
    finished = store.finish_ingestion_run(
        run.run_id,
        status="completed",
        accepted_count=3,
        rejected_count=1,
        output_ref="reports/mail-batch-1.json",
        actor="mail_importer",
    )
    conflict = store.record_conflict(
        "foton",
        conflict_type="ambiguous_identity",
        entity_refs=("email:parent@example.com", "tallanto:student-1", "tallanto:student-2"),
        actor="identity_mapper",
        ingestion_run_id=run.run_id,
    )
    duplicate_conflict = store.record_conflict(
        "foton",
        conflict_type="ambiguous_identity",
        entity_refs=("email:parent@example.com", "tallanto:student-1", "tallanto:student-2"),
        actor="identity_mapper",
        ingestion_run_id=run.run_id,
    )
    store.close()

    reopened = CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path)
    runs = reopened.list_ingestion_runs("foton")["items"]
    audit = reopened.list_audit_log("foton")["items"]

    assert finished.status == "completed"
    assert duplicate_run.run_id == run.run_id
    assert conflict.created is True
    assert duplicate_conflict.created is False
    assert duplicate_conflict.status == "duplicate"
    assert reopened.summary()["counts"]["ingestion_runs"] == 1
    assert reopened.summary()["counts"]["timeline_conflicts"] == 1
    assert runs[0]["accepted_count"] == 3
    assert runs[0]["rejected_count"] == 1
    assert audit[0]["entity_type"] == "timeline_conflict"
    assert {item["actor"] for item in audit} >= {"mail_importer", "identity_mapper"}
    reopened.close()
