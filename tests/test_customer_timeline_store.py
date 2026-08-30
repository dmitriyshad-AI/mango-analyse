from __future__ import annotations

import json
import sqlite3
import stat
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.store as store_module
from mango_mvp.customer_timeline.store import customer_timeline_readonly_uri
from mango_mvp.customer_timeline import (
    CUSTOMER_TIMELINE_SQLITE_MIGRATION_ID,
    CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
    ArtifactType,
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    CustomerTimelineReadApi,
    CustomerTimelineReadApiConfig,
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
_MISSING = object()


class _BrokenIntegrityMapping(dict):
    def get(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("broken mapping")


def _valid_integrity_report() -> dict[str, object]:
    return {
        "schema_version": store_module.CUSTOMER_TIMELINE_INTEGRITY_SCHEMA_VERSION,
        "schema": {
            "database_schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "missing_tables": [],
            "missing_columns": {},
            "inspection_error": False,
            "checks_complete": True,
        },
        "violations": {},
        "violations_total": 0,
        "validation_ok": True,
    }


def test_customer_timeline_readonly_uri_never_uses_immutable(tmp_path: Path) -> None:
    db_path = tmp_path / "timeline with spaces.sqlite"
    assert customer_timeline_readonly_uri(db_path) == db_path.resolve().as_uri() + "?mode=ro"
    assert "immutable" not in customer_timeline_readonly_uri(db_path)


@pytest.mark.parametrize(
    ("value", "present", "invalid_literal"),
    (
        (None, 0, 0),
        ("\t\n", 0, 1),
        ("\tNone\n", 0, 1),
        ("\u00a0NULL\u2003", 0, 1),
        ("customer:none", 1, 0),
    ),
)
def test_customer_id_sql_policy_matches_python_whitespace(
    value: str | None,
    present: int,
    invalid_literal: int,
) -> None:
    with sqlite3.connect(":memory:") as con:
        row = con.execute(
            "WITH candidate(value) AS (VALUES (?)) "
            f"SELECT {store_module.customer_id_present_sql('value')}, "
            f"{store_module.customer_id_literal_invalid_sql('value')} FROM candidate",
            (value,),
        ).fetchone()
    assert row == (present, invalid_literal)


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


def email_event(
    customer: CustomerIdentity | None,
    *,
    source_id: str,
    subject: str = "Заявка с сайта",
    text_preview: str = "Клиент уточняет расписание группы.",
    summary: str = "Клиент уточнил расписание группы и попросил ответить.",
    event_at: datetime = NOW,
) -> TimelineEvent:
    return TimelineEvent(
        tenant_id=customer.tenant_id if customer else "foton",
        customer_id=customer.customer_id if customer else None,
        event_type=TimelineEventType.EMAIL_MESSAGE,
        event_at=event_at,
        source_system="mail_archive_stage2",
        source_id=source_id,
        direction=TimelineDirection.INBOUND,
        subject=subject,
        text_preview=text_preview,
        summary=summary,
        importance=2,
        match_status="strong_unique" if customer else "unmatched",
        confidence=0.9 if customer else None,
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
        metadata={"client_safe": True},
        created_at=NOW,
    )


def open_store(tmp_path: Path) -> CustomerTimelineSQLiteStore:
    return CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=StepClock())


@pytest.mark.parametrize(
    ("record_name", "table", "key_column", "physical_column", "damaged_value", "expected_value"),
    (
        ("customer", "customer_identities", "customer_id", "display_name", "Повреждено", "Иванова Мария"),
        ("opportunity", "customer_opportunities", "opportunity_id", "status", "damaged", "open"),
        (
            "event",
            "timeline_events",
            "event_id",
            "summary",
            "Повреждено",
            "Клиент спросил стоимость курса и попросил перезвонить.",
        ),
        ("signal", "derived_signals", "signal_id", "severity", "low", "high"),
        ("chunk", "bot_context_chunks", "chunk_id", "allowed_for_bot", 0, 1),
    ),
)
def test_public_upsert_repairs_json_and_materialized_column_drift(
    tmp_path: Path,
    record_name: str,
    table: str,
    key_column: str,
    physical_column: str,
    damaged_value: object,
    expected_value: object,
) -> None:
    store = open_store(tmp_path)
    customer = identity()
    opp = opportunity(customer)
    timeline_event = event(customer, opp)
    derived_signal = signal(timeline_event)
    bot_chunk = chunk(timeline_event)
    store.upsert_customer(customer)
    store.upsert_opportunity(opp)
    store.upsert_event(timeline_event)
    store.upsert_signal(derived_signal)
    store.upsert_bot_context_chunk(bot_chunk)
    records = {
        "customer": (customer, store.upsert_customer, customer.customer_id),
        "opportunity": (opp, store.upsert_opportunity, opp.opportunity_id),
        "event": (timeline_event, store.upsert_event, timeline_event.event_id),
        "signal": (derived_signal, store.upsert_signal, derived_signal.signal_id),
        "chunk": (bot_chunk, store.upsert_bot_context_chunk, bot_chunk.chunk_id),
    }
    record, writer, record_id = records[record_name]
    expected_json = store._con.execute(  # noqa: SLF001 - fixture damages one materialized row.
        f"SELECT record_json FROM {table} WHERE {key_column}=?",
        (record_id,),
    ).fetchone()[0]
    store._con.execute(  # noqa: SLF001
        f"UPDATE {table} SET {physical_column}=?,record_json='{{\"damaged\":true}}' WHERE {key_column}=?",
        (damaged_value, record_id),
    )
    store._con.commit()  # noqa: SLF001

    repaired = writer(record, actor="physical_repair_test")
    repeated = writer(record, actor="physical_repair_test")

    assert repaired.status == "updated"
    assert repeated.status == "duplicate"
    row = store._con.execute(  # noqa: SLF001
        f"SELECT {physical_column},record_json FROM {table} WHERE {key_column}=?",
        (record_id,),
    ).fetchone()
    assert row[physical_column] == expected_value
    assert row["record_json"] == expected_json
    audit = store._con.execute(  # noqa: SLF001
        "SELECT record_json FROM audit_log WHERE audit_id=?",
        (repaired.audit_id,),
    ).fetchone()[0]
    metadata = json.loads(audit)["metadata"]
    assert metadata["record_json_repaired"] is True
    assert physical_column in metadata["physical_columns_repaired"]
    store.close()


def seed_integrity_graph(tmp_path: Path) -> Path:
    store = open_store(tmp_path)
    for tenant_id, phones in (
        ("foton", ("+79000000101", "+79000000103")),
        ("unpk", ("+79000000102", "+79000000104")),
    ):
        for index, phone in enumerate(phones, start=1):
            customer = identity(tenant_id=tenant_id, phone=phone)
            opp = opportunity(customer, source_id=f"{tenant_id}-lead-{index}")
            ev = event(customer, opp, source_id=f"{tenant_id}-event-{index}")
            store.upsert_customer(customer)
            store.record_customer_id_mapping(
                tenant_id,
                old_customer_id=f"legacy:{tenant_id}:{index}",
                new_customer_id=customer.customer_id,
                mapping_kind="alias",
                reason="integrity_fixture",
            )
            store.upsert_identity_link(identity_link(customer))
            store.upsert_opportunity(opp)
            store.upsert_event(ev)
            store.upsert_artifact(artifact(ev))
            store.upsert_bot_context_chunk(chunk(ev))
            family_payload = {
                "schema_version": "family_graph_v1",
                "tenant_id": tenant_id,
                "family_id": f"family-{index}",
                "customer_id": customer.customer_id,
                "membership_status": "active",
                "confidence": "high",
                "reason": "test",
                "created_at": NOW.isoformat(),
                "updated_at": NOW.isoformat(),
            }
            store._con.execute(  # noqa: SLF001 - fixture covers logical owner relations.
                "INSERT INTO family_members_v1 VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    tenant_id,
                    f"family-{index}",
                    customer.customer_id,
                    "active",
                    "high",
                    "test",
                    NOW.isoformat(),
                    NOW.isoformat(),
                    store_module.stable_digest(family_payload),
                    store_module.json_dumps(family_payload),
                ),
            )
            replacement = event(customer, opp, source_id=f"{tenant_id}-replacement-{index}")
            store.upsert_event(replacement)
            store.mark_timeline_events_superseded(
                tenant_id,
                canonical_event_id=replacement.event_id,
                duplicate_event_ids=(ev.event_id,),
                actor="integrity_fixture",
            )
            store.upsert_signal(signal(replacement))
    store._commit()  # noqa: SLF001
    store.upsert_event(email_event(None, source_id="allowed-without-customer"))
    db_path = store.db_path
    store.close()
    return db_path


def test_integrity_report_valid_graph_is_deterministic_and_read_only(tmp_path: Path) -> None:
    db_path = seed_integrity_graph(tmp_path)
    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        first = store_module.customer_timeline_integrity_report(store._con)
        second = store_module.customer_timeline_integrity_report(store._con)
        query_only = store._con.execute("PRAGMA query_only").fetchone()[0]

    assert first == second
    assert first["violations"] == {}
    assert first["quarantine"] == {
        "events_without_customer_including_superseded": 1,
        "by_match_status": {"unmatched": 1},
    }
    assert first["validation_ok"] is True
    assert store_module.customer_timeline_integrity_report_ok(first) is True
    assert query_only == 1


@pytest.mark.parametrize(
    ("scope", "key", "value"),
    (
        ("replace", None, None),
        ("replace", None, []),
        ("replace", None, _BrokenIntegrityMapping()),
        ("report", "schema_version", "wrong"),
        ("report", "violations", []),
        ("report", "violations_total", False),
        ("report", "validation_ok", 1),
        ("schema", "missing_tables", ["timeline_events"]),
        ("schema", "inspection_error", True),
        ("schema", "checks_complete", False),
    ),
)
def test_integrity_report_ok_fails_closed(scope: str, key: str | None, value: object) -> None:
    report = _valid_integrity_report()
    candidate: object = report
    if scope == "replace":
        candidate = value
    elif key is not None:
        target = report if scope == "report" else report["schema"]
        assert isinstance(target, dict)
        if value is _MISSING:
            target.pop(key)
        else:
            target[key] = value

    assert store_module.customer_timeline_integrity_report_ok(candidate) is False


@pytest.mark.parametrize(
    ("child", "foreign_key", "parent", "parent_key", "code"),
    (
        ("customer_opportunities", "customer_id", "customer_identities", "customer_id", "opportunity_customer"),
        ("identity_links", "customer_id", "customer_identities", "customer_id", "identity_link_customer"),
        ("timeline_events", "customer_id", "customer_identities", "customer_id", "event_customer"),
        ("timeline_events", "opportunity_id", "customer_opportunities", "opportunity_id", "event_opportunity"),
        ("timeline_events", "superseded_by", "timeline_events", "event_id", "event_superseded_by"),
        ("event_artifacts", "event_id", "timeline_events", "event_id", "artifact_event"),
        ("derived_signals", "customer_id", "customer_identities", "customer_id", "signal_customer"),
        ("derived_signals", "opportunity_id", "customer_opportunities", "opportunity_id", "signal_opportunity"),
        ("derived_signals", "event_id", "timeline_events", "event_id", "signal_event"),
        ("bot_context_chunks", "customer_id", "customer_identities", "customer_id", "chunk_customer"),
        ("bot_context_chunks", "opportunity_id", "customer_opportunities", "opportunity_id", "chunk_opportunity"),
        ("bot_context_chunks", "event_id", "timeline_events", "event_id", "chunk_event"),
        (
            "customer_id_mappings",
            "new_customer_id",
            "customer_identities",
            "customer_id",
            "mapping_new_customer",
        ),
        ("family_members_v1", "customer_id", "customer_identities", "customer_id", "family_member_customer"),
    ),
)
def test_integrity_report_detects_missing_and_cross_tenant_links(
    tmp_path: Path,
    child: str,
    foreign_key: str,
    parent: str,
    parent_key: str,
    code: str,
) -> None:
    db_path = seed_integrity_graph(tmp_path)
    with sqlite3.connect(db_path) as con:
        target_rowid = con.execute(
            f"SELECT rowid FROM {child} WHERE tenant_id='foton' "
            f"AND {foreign_key} IS NOT NULL AND {foreign_key}!='' LIMIT 1"
        ).fetchone()[0]
        con.execute(f"UPDATE {child} SET {foreign_key}='missing-id' WHERE rowid=?", (target_rowid,))
        con.commit()
    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        missing = store_module.customer_timeline_integrity_report(store._con)
    assert missing["violations"][f"{code}_missing"] == 1

    with sqlite3.connect(db_path) as con:
        foreign_parent = con.execute(
            f"SELECT {parent_key} FROM {parent} WHERE tenant_id='unpk' LIMIT 1"
        ).fetchone()[0]
        con.execute(f"UPDATE {child} SET {foreign_key}=? WHERE rowid=?", (foreign_parent, target_rowid))
        con.commit()
    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        mismatch = store_module.customer_timeline_integrity_report(store._con)
    assert mismatch["violations"][f"{code}_tenant_mismatch"] == 1


@pytest.mark.parametrize(
    "marker",
    (
        "retired:wappi_expected_excluded:0123456789abcdef",
        "retired:wappi_verified_source_absent:0123456789abcdef",
    ),
)
def test_integrity_report_accepts_tagged_event_and_opaque_chunk_tombstones(
    tmp_path: Path,
    marker: str,
) -> None:
    db_path = seed_integrity_graph(tmp_path)
    with sqlite3.connect(db_path) as con:
        event_id = con.execute(
            "SELECT event_id FROM timeline_events WHERE tenant_id='foton' LIMIT 1"
        ).fetchone()[0]
        chunk_id = con.execute(
            "SELECT chunk_id FROM bot_context_chunks WHERE tenant_id='foton' LIMIT 1"
        ).fetchone()[0]
        con.execute(
            "UPDATE timeline_events SET source_system='wappi_telegram',"
            "superseded_by=?,"
            "record_json=json_set(record_json,'$.source_system','wappi_telegram') WHERE event_id=?",
            (marker, event_id),
        )
        con.execute(
            "UPDATE bot_context_chunks SET superseded_by='opaque:test_lifecycle' WHERE chunk_id=?",
            (chunk_id,),
        )
        con.commit()

    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        report = store_module.customer_timeline_integrity_report(store._con)

    assert report["violations"] == {}
    assert report["validation_ok"] is True


@pytest.mark.parametrize(
    ("source_system", "marker"),
    (
        ("wappi_telegram", "retired:test_lifecycle"),
        ("wappi_max", "retired:wappi_expected_excluded:0123456789abcdeg"),
        ("wappi_max", "retired:wappi_expected_excluded:0123456789abcde"),
        ("wappi_max", "retired:wappi_expected_excluded:0123456789abcdef0"),
        ("wappi_max", "retired:wappi_expected_excluded:0123456789ABCDEF"),
        ("mango", "retired:wappi_expected_excluded:0123456789abcdef"),
        ("", "retired:wappi_expected_excluded:0123456789abcdef"),
    ),
)
def test_integrity_report_rejects_unproven_event_retirement_markers(
    tmp_path: Path,
    source_system: str,
    marker: str,
) -> None:
    db_path = seed_integrity_graph(tmp_path)
    with sqlite3.connect(db_path) as con:
        event_id = con.execute(
            "SELECT event_id FROM timeline_events WHERE tenant_id='foton' LIMIT 1"
        ).fetchone()[0]
        con.execute(
            "UPDATE timeline_events SET source_system=?,superseded_by=?,"
            "record_json=json_set(record_json,'$.source_system',?) WHERE event_id=?",
            (source_system, marker, source_system, event_id),
        )
        con.commit()

    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        report = store_module.customer_timeline_integrity_report(store._con)

    assert report["violations"]["event_superseded_by_missing"] == 1
    assert report["validation_ok"] is False


def test_wappi_event_retirement_predicate_fails_closed_for_null_source() -> None:
    with sqlite3.connect(":memory:") as con:
        accepted = con.execute(
            "WITH c(source_system,superseded_by) AS (VALUES(NULL,?)) SELECT "
            + store_module._VALID_WAPPI_EVENT_RETIREMENT_SQL  # noqa: SLF001 - SQL contract test.
            + " FROM c",
            ("retired:wappi_expected_excluded:0123456789abcdef",),
        ).fetchone()[0]

    assert accepted == 0


def test_integrity_report_checks_all_signal_sources_and_confirmed_orphans(tmp_path: Path) -> None:
    db_path = seed_integrity_graph(tmp_path)
    with sqlite3.connect(db_path) as con:
        signal_rowid = con.execute(
            "SELECT rowid FROM derived_signals WHERE tenant_id='foton' LIMIT 1"
        ).fetchone()[0]
        con.execute(
            "UPDATE derived_signals SET record_json=json_set(record_json,'$.source_event_ids[0]','missing-id') "
            "WHERE rowid=?",
            (signal_rowid,),
        )
        con.execute(
            "UPDATE timeline_events SET match_status='strong_unique' WHERE source_id='allowed-without-customer'"
        )
        con.commit()
    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        report = store_module.customer_timeline_integrity_report(store._con)

    assert report["violations"]["signal_source_event_missing"] == 1
    assert report["violations"]["event_linked_without_customer"] == 1
    assert report["validation_ok"] is False


def test_integrity_report_rejects_active_signal_on_superseded_event(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity(phone="+79000000351")
    canonical = event(customer, source_id="integrity-canonical")
    duplicate = event(customer, source_id="integrity-duplicate")
    active_signal = signal(duplicate)
    store.upsert_customer(customer)
    store.upsert_event(canonical)
    store.upsert_event(duplicate)
    store.upsert_signal(active_signal)
    store._con.execute(  # noqa: SLF001 - inject one legacy invalid row for the integrity gate.
        "UPDATE timeline_events SET superseded_by=? WHERE event_id=?",
        (canonical.event_id, duplicate.event_id),
    )
    store._con.commit()  # noqa: SLF001

    report = store_module.customer_timeline_integrity_report(store._con)

    assert report["violations"]["active_signal_linked_to_superseded_event"] == 1
    assert report["validation_ok"] is False
    store.close()


def test_integrity_report_blocks_identity_link_materialized_owner_drift(tmp_path: Path) -> None:
    db_path = seed_integrity_graph(tmp_path)
    with sqlite3.connect(db_path) as con:
        link_rowid, original_customer = con.execute(
            "SELECT rowid,customer_id FROM identity_links WHERE tenant_id='foton' LIMIT 1"
        ).fetchone()
        foreign_customer = con.execute(
            "SELECT customer_id FROM customer_identities "
            "WHERE tenant_id='foton' AND customer_id!=? LIMIT 1",
            (original_customer,),
        ).fetchone()[0]
        con.execute(
            "UPDATE identity_links SET customer_id=? WHERE rowid=?",
            (foreign_customer, link_rowid),
        )
        con.commit()

    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        report = store_module.customer_timeline_integrity_report(store._con)

    assert report["violations"]["identity_link_record_identity_mismatch"] == 1
    assert report["validation_ok"] is False


@pytest.mark.parametrize("damage", ("physical_open_json_closed", "physical_closed_json_open"))
def test_integrity_report_blocks_bot_chunk_materialized_safety_drift(
    tmp_path: Path,
    damage: str,
) -> None:
    db_path = seed_integrity_graph(tmp_path)
    with sqlite3.connect(db_path) as con:
        chunk_id = con.execute("SELECT chunk_id FROM bot_context_chunks LIMIT 1").fetchone()[0]
        if damage == "physical_open_json_closed":
            con.execute(
                "UPDATE bot_context_chunks SET record_json=json_set(record_json,"
                "'$.allowed_for_bot',json('false'),'$.requires_manager_review',json('true')) "
                "WHERE chunk_id=?",
                (chunk_id,),
            )
        else:
            con.execute(
                "UPDATE bot_context_chunks SET allowed_for_bot=0,requires_manager_review=1 WHERE chunk_id=?",
                (chunk_id,),
            )
        con.commit()

    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        report = store_module.customer_timeline_integrity_report(store._con)

    assert report["violations"]["chunk_record_safety_mismatch"] == 1
    assert report["validation_ok"] is False


@pytest.mark.parametrize("damage", ("missing_schema", "missing_table", "malformed_json", "closed_connection"))
def test_integrity_report_schema_and_inspection_fail_closed(tmp_path: Path, damage: str) -> None:
    if damage == "missing_schema":
        db_path = tmp_path / "empty.sqlite"
        sqlite3.connect(db_path).close()
    else:
        db_path = seed_integrity_graph(tmp_path)
        with sqlite3.connect(db_path) as con:
            if damage == "missing_table":
                con.execute("DROP TABLE family_members_v1")
            elif damage == "malformed_json":
                con.execute("DROP INDEX ix_signals_multi_source")
                con.execute("UPDATE derived_signals SET record_json='{' WHERE tenant_id='foton'")
            con.commit()

    store = CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path)
    if damage == "closed_connection":
        store.close()
    report = store_module.customer_timeline_integrity_report(store._con)
    if damage != "closed_connection":
        store.close()

    assert report["validation_ok"] is False
    assert store_module.customer_timeline_integrity_report_ok(report) is False
    assert report["violations_total"] > 0


def test_store_restricts_writable_db_and_lock_permissions(tmp_path: Path) -> None:
    private_root = tmp_path / ".codex_local" / "staging"
    db_path = private_root / "customer_timeline.sqlite"

    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    try:
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(private_root.stat().st_mode) == 0o700
        assert stat.S_IMODE(db_path.with_suffix(".sqlite.writer.lock").stat().st_mode) == 0o600
    finally:
        store.close()


def table_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    return {row[0] for row in rows}


def column_names(db_path: Path, table: str) -> set[str]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def index_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
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
    assert {"status", "expires_at"} <= column_names(db_path, "derived_signals")
    assert {"content_key", "superseded_by"} <= column_names(db_path, "timeline_events")
    assert "superseded_by" in column_names(db_path, "bot_context_chunks")
    assert "ix_signals_customer_status_expiry" in index_names(db_path)
    assert "ix_signals_multi_source" in index_names(db_path)
    assert "ix_chunks_event_owner" in index_names(db_path)
    assert "ix_timeline_events_content_key" in index_names(db_path)
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


def test_content_key_backfill_is_explicit_batched_and_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    customer = identity()
    first = email_event(customer, source_id="mail-1")
    second = email_event(customer, source_id="mail-2", event_at=NOW + timedelta(minutes=3))
    store.upsert_customer(customer)
    store.upsert_event(first)
    store.upsert_event(second)
    store.close()

    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE timeline_events SET content_key = NULL")
        con.commit()

    reopened = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    assert reopened.count_missing_timeline_email_content_keys() == 2

    result = reopened.backfill_timeline_event_content_keys(batch_size=1)

    assert result == {"batches": 2, "rows_seen": 2, "rows_updated": 2}
    assert reopened.count_missing_timeline_email_content_keys() == 0
    assert reopened.backfill_timeline_event_content_keys(batch_size=1) == {
        "batches": 0,
        "rows_seen": 0,
        "rows_updated": 0,
    }
    reopened.close()


def test_bootstrap_adds_content_key_to_legacy_db_before_index_without_backfill(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE timeline_events (
              event_id TEXT PRIMARY KEY,
              dedupe_key TEXT NOT NULL UNIQUE,
              tenant_id TEXT NOT NULL,
              customer_id TEXT,
              opportunity_id TEXT,
              event_type TEXT NOT NULL,
              event_at TEXT NOT NULL,
              source_system TEXT NOT NULL,
              source_id TEXT NOT NULL,
              source_ref TEXT,
              direction TEXT NOT NULL,
              match_status TEXT NOT NULL,
              confidence REAL,
              importance INTEGER NOT NULL,
              subject TEXT,
              text_preview TEXT,
              summary TEXT,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO timeline_events (
              event_id, dedupe_key, tenant_id, customer_id, event_type, event_at,
              source_system, source_id, direction, match_status, importance,
              subject, text_preview, summary, created_at, record_hash, record_json
            )
            VALUES (
              'event-legacy', 'dedupe-legacy', 'foton', 'customer-1', 'email_message',
              ?, 'mail_archive_stage2', 'mail-1', 'inbound', 'strong_unique', 1,
              'Заявка', 'Текст письма', 'Клиент уточнил расписание.', ?, ?, ?
            )
            """,
            (NOW.isoformat(), NOW.isoformat(), SHA, json.dumps({"event_id": "event-legacy"})),
        )
        con.commit()

    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())

    assert {"content_key", "superseded_by"} <= column_names(db_path, "timeline_events")
    assert store.count_missing_timeline_email_content_keys() == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT content_key FROM timeline_events WHERE event_id = 'event-legacy'").fetchone()[0] is None
    store.close()


def test_derived_signal_status_migration_is_idempotent_and_reads_old_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    old_payload = {
        "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
        "signal_id": "derived_signal:old",
        "tenant_id": "foton",
        "customer_id": "customer:old",
        "opportunity_id": None,
        "event_id": None,
        "source_event_ids": [],
        "signal_type": "price_interest",
        "severity": "medium",
        "confidence": 0.7,
        "evidence_text": "Старая запись сигнала без status/expires_at.",
        "recommended_action": "Проверить вручную",
        "requires_manager_review": True,
        "metadata": {},
        "created_at": NOW.isoformat(),
    }
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE schema_migrations (
              migration_id TEXT PRIMARY KEY,
              schema_version TEXT NOT NULL,
              applied_at TEXT NOT NULL
            );
            CREATE TABLE derived_signals (
              signal_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              customer_id TEXT,
              opportunity_id TEXT,
              event_id TEXT,
              signal_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              confidence REAL,
              requires_manager_review INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            );
            """
        )
        con.execute(
            """
            INSERT INTO schema_migrations (migration_id, schema_version, applied_at)
            VALUES (?, ?, ?)
            """,
            ("20260512_001_customer_timeline_sqlite", CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION, NOW.isoformat()),
        )
        con.execute(
            """
            INSERT INTO derived_signals (
              signal_id, tenant_id, customer_id, opportunity_id, event_id,
              signal_type, severity, confidence, requires_manager_review,
              created_at, record_hash, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                old_payload["signal_id"],
                old_payload["tenant_id"],
                old_payload["customer_id"],
                old_payload["opportunity_id"],
                old_payload["event_id"],
                old_payload["signal_type"],
                old_payload["severity"],
                old_payload["confidence"],
                int(old_payload["requires_manager_review"]),
                old_payload["created_at"],
                "old-hash",
                json.dumps(old_payload, ensure_ascii=False),
            ),
        )

    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    row = store._con.execute(
        "SELECT status, expires_at, record_json FROM derived_signals WHERE signal_id = ?",
        (old_payload["signal_id"],),
    ).fetchone()
    assert {"status", "expires_at"} <= column_names(db_path, "derived_signals")
    assert "ix_signals_customer_status_expiry" in index_names(db_path)
    assert row["status"] == "active"
    assert row["expires_at"] is None
    assert "status" not in json.loads(row["record_json"])
    store.close()

    reopened = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    migration_rows = reopened._con.execute(
        "SELECT COUNT(*) FROM schema_migrations WHERE migration_id = ?",
        (CUSTOMER_TIMELINE_SQLITE_MIGRATION_ID,),
    ).fetchone()[0]
    migrated_rows = reopened._con.execute("SELECT COUNT(*) FROM derived_signals WHERE status = 'active'").fetchone()[0]
    assert migration_rows == 1
    assert migrated_rows == 1
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


def test_sqlite_store_allows_single_writer_and_read_only_observers(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    writer = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    try:
        with pytest.raises(RuntimeError, match="writer lock"):
            CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
        reader = CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path, clock=StepClock())
        try:
            assert reader.read_only is True
        finally:
            reader.close()
    finally:
        writer.close()

    reopened = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    reopened.close()


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


def test_bulk_write_keeps_fts_searchable_after_commit(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    ev = event(customer)

    with store.bulk_write():
        store.upsert_customer(customer)
        store.upsert_event(ev)
        store.upsert_bot_context_chunk(chunk(ev))

    result = store.search_timeline("foton", "стоимость", limit=10)
    scopes = {item["scope"] for item in result["items"]}

    assert result["backend"] in {"fts5", "fallback_like"}
    assert "event" in scopes
    assert "bot_context" in scopes
    store.close()


def test_bulk_write_updates_fts_incrementally_and_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = open_store(tmp_path)
    customer = identity()
    original = event(customer, summary="Исходная формулировка")
    store.upsert_customer(customer)
    store.upsert_event(original)
    monkeypatch.setattr(store, "_rebuild_fts_indexes", lambda: pytest.fail("global FTS rebuild"))

    with store.bulk_write():
        store.upsert_event(replace(original, summary="Точечное обновление"))
    assert store.search_timeline("foton", "обновление")["items"]

    with pytest.raises(RuntimeError, match="abort"):
        with store.bulk_write():
            store.upsert_event(replace(original, summary="Откат транзакции"))
            raise RuntimeError("abort")
    assert not store.search_timeline("foton", "транзакции")["items"]
    assert store.search_timeline("foton", "обновление")["items"]
    store.close()


def test_bulk_write_uses_rowid_point_sync_without_global_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = open_store(tmp_path)
    customer = identity()
    store.upsert_customer(customer)
    monkeypatch.setattr(store, "_rebuild_fts_indexes", lambda: pytest.fail("global FTS rebuild"))
    with store.bulk_write():
        for index in range(100):
            store.upsert_event(event(customer, source_id=f"bulk-{index}", summary=f"Пакет {index}"))
    assert len(store.search_timeline("foton", "Пакет", limit=200)["items"]) == 100
    assert store._con.execute(
        "SELECT COUNT(*) FROM timeline_event_fts WHERE tenant_id='foton'"
    ).fetchone()[0] == 100
    assert store._con.execute(
        "SELECT COUNT(*) FROM timeline_event_fts_keys WHERE fts_rowid IS NOT NULL"
    ).fetchone()[0] == 100
    store.close()


def test_new_fts_records_skip_linear_orphan_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = open_store(tmp_path)
    customer = identity()
    ev = event(customer, summary="Новая запись без полного скана")
    original_fetch_one = store._fetch_one

    def reject_linear_fts_lookup(query: str, params: tuple[object, ...] = ()):
        assert "FROM timeline_event_fts WHERE event_id" not in query
        assert "FROM bot_context_chunk_fts WHERE chunk_id" not in query
        return original_fetch_one(query, params)

    monkeypatch.setattr(store, "_fetch_one", reject_linear_fts_lookup)
    with store.bulk_write():
        store.upsert_customer(customer)
        store.upsert_event(ev)
        store.upsert_bot_context_chunk(chunk(ev))

    assert store.search_timeline("foton", "полного скана", limit=10)["items"]
    store.close()


def test_old_fts_key_schema_is_backfilled_without_retokenizing(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    original = event(customer, summary="Сохранённый индекс")
    store.upsert_customer(customer)
    store.upsert_event(original)
    store.close()

    db_path = tmp_path / "customer_timeline.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute("ALTER TABLE timeline_event_fts_keys RENAME TO old_event_fts_keys")
        con.execute("CREATE TABLE timeline_event_fts_keys(event_id TEXT PRIMARY KEY)")
        con.execute("INSERT INTO timeline_event_fts_keys SELECT event_id FROM old_event_fts_keys")
        con.execute("DROP TABLE old_event_fts_keys")
        con.execute("DROP TABLE bot_context_chunk_fts_keys")

    reopened = open_store(tmp_path)
    event_row = reopened._con.execute(
        "SELECT k.fts_rowid, f.rowid FROM timeline_event_fts_keys k "
        "JOIN timeline_event_fts f ON f.event_id=k.event_id WHERE k.event_id=?",
        (original.event_id,),
    ).fetchone()
    assert event_row[0] == event_row[1]
    assert reopened.search_timeline("foton", "Сохранённый")["items"]
    assert reopened._con.execute(
        "SELECT COUNT(*) FROM bot_context_chunk_fts_keys WHERE fts_rowid IS NOT NULL"
    ).fetchone()[0] == reopened._con.execute("SELECT COUNT(*) FROM bot_context_chunk_fts").fetchone()[0]
    reopened.close()


def test_missing_single_fts_key_is_repaired_without_duplicate(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    original = event(customer, summary="Старый текст")
    store.upsert_customer(customer)
    store.upsert_event(original)
    store._con.execute("DELETE FROM timeline_event_fts_keys WHERE event_id=?", (original.event_id,))
    store._con.commit()

    store.upsert_event(replace(original, summary="Новый текст"))

    assert store._con.execute(
        "SELECT COUNT(*) FROM timeline_event_fts WHERE event_id=?", (original.event_id,)
    ).fetchone()[0] == 1
    assert not store.search_timeline("foton", "Старый")["items"]
    assert store.search_timeline("foton", "Новый")["items"]
    store.close()


def test_missing_fts_key_blocks_point_delete_and_requests_rebuild(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    original = event(customer, summary="Старый текст")
    store.upsert_customer(customer)
    store.upsert_event(original)
    store._con.execute("DELETE FROM timeline_event_fts_keys WHERE event_id=?", (original.event_id,))

    assert store._delete_fts_rows_by_keys(
        fts_table="timeline_event_fts",
        key_table="timeline_event_fts_keys",
        key_column="event_id",
        keys=(original.event_id,),
    ) is False
    store.close()


def test_legacy_fts_rebuild_rollback_preserves_old_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = open_store(tmp_path)
    customer = identity()
    original = event(customer, summary="Старый индекс")
    store.upsert_customer(customer)
    store.upsert_event(original)
    original_bootstrap = store._bootstrap_fts

    def fail_after_create() -> None:
        original_bootstrap()
        raise RuntimeError("fts failed after create")

    monkeypatch.setattr(store, "_bootstrap_fts", fail_after_create)
    with pytest.raises(RuntimeError, match="fts failed after create"):
        with store.bulk_write():
            store._rebuild_fts_indexes()

    assert store.search_timeline("foton", "Старый")["items"]
    assert not store.search_timeline("foton", "Новый")["items"]
    assert store._con.execute("SELECT COUNT(*) FROM timeline_event_fts").fetchone()[0] == 1
    assert store._con.execute("SELECT COUNT(*) FROM timeline_event_fts_keys").fetchone()[0] == 1
    store.close()

    reopened = open_store(tmp_path)
    assert reopened.search_timeline("foton", "Старый")["items"]
    reopened.close()


def test_legacy_fts_rebuild_keeps_old_index_visible_until_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = open_store(tmp_path)
    customer = identity()
    original = event(customer, summary="До переключения")
    store.upsert_customer(customer)
    store.upsert_event(original)
    original_bootstrap = store._bootstrap_fts

    def bootstrap_and_check_reader() -> None:
        original_bootstrap()
        with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as reader:
            assert reader.execute("SELECT COUNT(*) FROM timeline_event_fts").fetchone()[0] == 1
            assert reader.execute("SELECT COUNT(*) FROM timeline_event_fts_keys").fetchone()[0] == 1

    monkeypatch.setattr(store, "_bootstrap_fts", bootstrap_and_check_reader)
    with store.bulk_write():
        store._rebuild_fts_indexes()
        store.upsert_event(replace(original, summary="После переключения"))

    assert not store.search_timeline("foton", "До переключения")["items"]
    assert store.search_timeline("foton", "После переключения")["items"]
    store.close()


def test_ingestion_cursor_persists_and_is_reported_in_summary(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    cursor = store.upsert_ingestion_cursor(
        "foton",
        "amocrm_snapshot",
        last_cursor_ts=NOW - timedelta(minutes=5),
        metadata={"max_source_ts": NOW.isoformat(), "last_status": "ok"},
    )
    store.close()

    reopened = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=StepClock())
    try:
        loaded = reopened.get_ingestion_cursor("foton", "amocrm_snapshot")
        cursors = reopened.list_ingestion_cursors("foton")
        summary = reopened.summary()
    finally:
        reopened.close()

    assert loaded is not None
    assert loaded.last_cursor_ts == cursor.last_cursor_ts
    assert loaded.metadata["last_status"] == "ok"
    assert cursors[0]["source_system"] == "amocrm_snapshot"
    assert summary["counts"]["ingestion_cursors"] == 1


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


def test_store_rejects_prod_writes_but_allows_read_only_and_resolved_symlink(tmp_path: Path) -> None:
    prod_dir = tmp_path / "customer_timeline_prod_20260722"
    prod_dir.mkdir()
    prod_db = prod_dir / "customer_timeline.sqlite"
    sqlite3.connect(prod_db).close()

    with CustomerTimelineSQLiteStore.open_read_only(prod_db, allowed_root=tmp_path):
        pass
    with pytest.raises(ValueError, match="snapshot-only"):
        CustomerTimelineSQLiteStore(prod_db, allowed_root=tmp_path)
    assert not prod_db.with_suffix(".sqlite.writer.lock").exists()

    alias = tmp_path / "timeline_alias"
    alias.symlink_to(prod_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="snapshot-only"):
        CustomerTimelineSQLiteStore(alias / prod_db.name, allowed_root=tmp_path)


def test_conflict_lookup_matches_exact_customer_ref_not_prefix(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    with sqlite3.connect(db_path) as con:
        for suffix, ref in (("one", "customer:1"), ("ten", "customer:10"), ("double", "customer:customer:1")):
            con.execute(
                "INSERT INTO timeline_conflicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"conflict:{suffix}",
                    "foton",
                    "ambiguous_identity",
                    "medium",
                    "open",
                    NOW.isoformat(),
                    None,
                    f"hash:{suffix}",
                    json.dumps({"conflict_id": f"conflict:{suffix}", "status": "open", "entity_refs": [ref]}),
                ),
            )
        con.commit()

    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        conflicts = store.list_conflicts_by_customer("foton", "customer:1", statuses=("open",))
        first_page = store.list_conflicts("foton", statuses=("open",), limit=2)
        second_page = store.list_conflicts(
            "foton", statuses=("open",), limit=2, cursor=first_page["next_cursor"]
        )

    assert {item["conflict_id"] for item in conflicts} == {"conflict:one", "conflict:double"}
    assert len(first_page["items"]) == 2
    assert len(second_page["items"]) == 1
    assert first_page["next_cursor"] == "2"
    assert second_page["next_cursor"] is None


def test_unresolved_tallanto_payment_is_part_of_family_conflict_gate(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:payment-owner",
                identity_status=IdentityStatus.STRONG,
            )
        )
        store.record_conflict(
            "foton",
            conflict_type="tallanto_payment_owner_unresolved",
            entity_refs=("customer:customer:payment-owner",),
        )

    with sqlite3.connect(db_path) as con:
        assert store_module.open_family_identity_conflict_customer_ids(con, "foton") == frozenset(
            {"customer:payment-owner"}
        )
        assert store_module.has_open_family_identity_conflict(
            con,
            "foton",
            family_id="",
            customer_ids=("customer:payment-owner",),
        )


@pytest.mark.parametrize(
    ("conflict_type", "blocked"),
    (
        ("tallanto_identity_ambiguous", True),
        ("telegram_identity_ambiguous", True),
        ("tallanto_attendance_api_identity_conflict", True),
        ("whatsapp_phone_ambiguous", True),
        ("tallanto_attendance_api_identity_infrastructure_gap", False),
        ("tallanto_attendance_api_identity_unmatched", False),
        ("ambiguous", False),
    ),
)
def test_family_conflict_gate_uses_explicit_business_types(
    tmp_path: Path,
    conflict_type: str,
    blocked: bool,
) -> None:
    db_path = tmp_path / f"{conflict_type}.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = identity()
        store.upsert_customer(customer)
        store.record_conflict(
            "foton",
            conflict_type=conflict_type,
            entity_refs=(f"customer:{customer.customer_id}",),
        )
    with sqlite3.connect(db_path) as con:
        actual = customer.customer_id in store_module.open_family_identity_conflict_customer_ids(con, "foton")
    assert actual is blocked


def test_family_conflict_gate_reconstructs_created_and_resolved_cutoff(tmp_path: Path) -> None:
    db_path = tmp_path / "conflict-cutoff.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = identity()
        store.upsert_customer(customer)
        store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=(f"customer:{customer.customer_id}",),
        )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE timeline_conflicts SET status='resolved',created_at=?,resolved_at=?",
            ((NOW + timedelta(hours=1)).isoformat(), (NOW + timedelta(hours=3)).isoformat()),
        )
        con.commit()
        before = store_module.open_family_identity_conflict_customer_ids(
            con, "foton", as_of=NOW.isoformat(),
        )
        during = store_module.open_family_identity_conflict_customer_ids(
            con, "foton", as_of=(NOW + timedelta(hours=2)).isoformat(),
        )
        after = store_module.open_family_identity_conflict_customer_ids(
            con, "foton", as_of=(NOW + timedelta(hours=4)).isoformat(),
        )
        scoped_before = store_module.has_open_family_identity_conflict(
            con,
            "foton",
            family_id="",
            customer_ids=(customer.customer_id,),
            as_of=NOW.isoformat(),
        )
        scoped_during = store_module.has_open_family_identity_conflict(
            con,
            "foton",
            family_id="",
            customer_ids=(customer.customer_id,),
            as_of=(NOW + timedelta(hours=2)).isoformat(),
        )
        scoped_after = store_module.has_open_family_identity_conflict(
            con,
            "foton",
            family_id="",
            customer_ids=(customer.customer_id,),
            as_of=(NOW + timedelta(hours=4)).isoformat(),
        )

    assert customer.customer_id not in before
    assert customer.customer_id in during
    assert customer.customer_id not in after
    assert scoped_before is False
    assert scoped_during is True
    assert scoped_after is False


def test_trusted_family_scope_uses_atomic_snapshot_not_materialization_time(tmp_path: Path) -> None:
    db_path = tmp_path / "family-cutoff.sqlite"
    first = identity(phone="+79160000021")
    second = identity(phone="+79160000022")
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(first)
        store.upsert_customer(second)
    with sqlite3.connect(db_path) as con:
        con.executemany(
            """
            INSERT INTO family_members_v1
            (tenant_id,family_id,customer_id,membership_status,confidence,reason,
             created_at,updated_at,record_hash,record_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ("foton", "family:cutoff", first.customer_id, "confident", "high", "test",
                 NOW.isoformat(), (NOW + timedelta(hours=2)).isoformat(), "hash:first", "{}"),
                ("foton", "family:cutoff", second.customer_id, "confident", "high", "test",
                 NOW.isoformat(), (NOW + timedelta(hours=2)).isoformat(), "hash:second", "{}"),
            ),
        )
        con.commit()
        before = store_module.trusted_family_customer_ids(
            con,
            tenant_id="foton",
            customer_id=first.customer_id,
            as_of=NOW + timedelta(hours=1),
        )
        after = store_module.trusted_family_customer_ids(
            con,
            tenant_id="foton",
            customer_id=first.customer_id,
            as_of=NOW + timedelta(hours=3),
        )

    assert before == tuple(sorted((first.customer_id, second.customer_id)))
    assert after == tuple(sorted((first.customer_id, second.customer_id)))


@pytest.mark.parametrize("family_table_present", [False, True])
def test_family_conflict_gate_is_addressed_with_or_without_family_table(
    tmp_path: Path,
    family_table_present: bool,
) -> None:
    db_path = tmp_path / f"customer_timeline_{family_table_present}.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        own = identity(phone="+79160000001")
        foreign = identity(phone="+79160000002")
        safe = identity(phone="+79160000003")
        for customer in (own, foreign, safe):
            store.upsert_customer(customer)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=own.customer_id,
                link_type="tallanto_student_id",
                link_value="student-00000001",
                source_system="tallanto_snapshot",
                source_ref="tallanto:student:student-00000001",
                match_class="strong_unique",
            )
        )
        store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=("tallanto_student:student-00000001",),
        )
        store.record_conflict(
            "foton",
            conflict_type="shared_family_phone",
            entity_refs=(f"customer:{foreign.customer_id}",),
        )
    if not family_table_present:
        with sqlite3.connect(db_path) as con:
            con.execute("DROP TABLE family_members_v1")

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        blocked = store_module.open_family_identity_conflict_customer_ids(con, "foton")
        assert own.customer_id in blocked
        assert foreign.customer_id in blocked
        assert safe.customer_id not in blocked
        exact_rows = store_module.authoritative_exact_identity_rows(
            con,
            "foton",
            link_types=("tallanto_student_id",),
        )
        assert exact_rows[0]["has_open_conflict"] == 1
        assert store_module.has_open_family_identity_conflict(
            con,
            "foton",
            family_id="",
            customer_ids=(own.customer_id,),
        )
        assert not store_module.has_open_family_identity_conflict(
            con,
            "foton",
            family_id="",
            customer_ids=(safe.customer_id,),
        )


def test_family_conflict_gate_does_not_match_unrelated_link_type_by_suffix(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = identity()
        store.upsert_customer(customer)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_contact_id",
                link_value="1234567890",
                source_system="synthetic",
                source_ref="synthetic:amo-contact",
                match_class="strong_unique",
            )
        )
        store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=("tallanto_student:1234567890",),
        )

    with sqlite3.connect(db_path) as con:
        assert customer.customer_id not in store_module.open_family_identity_conflict_customer_ids(con, "foton")


def test_authoritative_identity_rows_support_schema_without_family_members(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = identity()
        store.upsert_customer(customer)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_contact_id",
                link_value="7001",
                source_system="amocrm_snapshot",
                source_ref="amocrm:contact:7001",
                match_class="strong_unique",
            )
        )
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("DROP TABLE family_members_v1")
        rows = store_module.authoritative_exact_identity_rows(
            con,
            "foton",
            link_types=("amo_contact_id",),
        )

    assert len(rows) == 1
    assert rows[0]["customer_id"] == customer.customer_id
    assert rows[0]["owner_count"] == 1


def test_authoritative_identity_rows_block_direct_customer_conflict(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = identity()
        store.upsert_customer(customer)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_contact_id",
                link_value="7003",
                source_system="amocrm_snapshot",
                source_ref="amocrm:contact:7003",
                match_class="strong_unique",
            )
        )
        store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=(f"customer:{customer.customer_id}",),
        )

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = store_module.authoritative_exact_identity_rows(
            con,
            "foton",
            link_types=("amo_contact_id",),
        )

    assert len(rows) == 1
    assert rows[0]["has_open_conflict"] == 1


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


def test_identity_link_accumulates_time_range_without_replay_churn(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    store.upsert_customer(customer)
    base = identity_link(customer)
    later = replace(
        base,
        first_seen_at=NOW + timedelta(days=2),
        last_seen_at=NOW + timedelta(days=3),
    )
    earlier = replace(
        base,
        first_seen_at=NOW - timedelta(days=1),
        last_seen_at=NOW,
    )

    assert store.upsert_identity_link(later).status == "created"
    assert store.upsert_identity_link(earlier).status == "updated"
    assert store.upsert_identity_link(later).status == "duplicate"
    assert store.upsert_identity_link(earlier).status == "duplicate"

    with sqlite3.connect(store.db_path) as con:
        row = con.execute(
            "SELECT first_seen_at, last_seen_at FROM identity_links WHERE link_id = ?",
            (base.link_id,),
        ).fetchone()
    assert row == (
        (NOW - timedelta(days=1)).isoformat(),
        (NOW + timedelta(days=3)).isoformat(),
    )
    store.close()


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
        ingestion_run_id="run:first",
    )
    second = store.record_customer_id_mapping(
        "foton",
        old_customer_id="customer:legacy-a",
        new_customer_id=target.customer_id,
        mapping_kind="merge",
        reason="phone_identity_union",
        source_refs=("amocrm:contact:1", "mango:call:1"),
        actor="identity_resolver",
        ingestion_run_id="run:repeat",
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
    assert mappings[0]["ingestion_run_id"] == "run:first"
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


def test_customer_id_mapping_ignores_self_and_supersedes_historical_self(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    old = identity(phone="+79160000006")
    target = identity(phone="+79160000007")
    store.upsert_customer(old)
    store.upsert_customer(target)

    noop = store.record_customer_id_mapping(
        "foton",
        old_customer_id=old.customer_id,
        new_customer_id=old.customer_id,
        reason="unchanged",
    )
    assert noop.status == "duplicate"
    assert store.list_customer_id_mappings("foton") == ()

    mapping_id = store_module.stable_prefixed_id(
        "customer_id_mapping",
        {
            "tenant_id": "foton",
            "old_customer_id": old.customer_id,
            "new_customer_id": old.customer_id,
        },
    )
    historical = {
        "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
        "mapping_id": mapping_id,
        "tenant_id": "foton",
        "old_customer_id": old.customer_id,
        "new_customer_id": old.customer_id,
        "mapping_kind": "alias",
        "resolution_status": "active",
        "reason": "unchanged",
        "source_refs": [],
        "ingestion_run_id": None,
        "metadata": {},
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
    }
    store._upsert_record(
        table="customer_id_mappings",
        key_column="mapping_id",
        key_value=mapping_id,
        record_type="customer_id_mapping",
        tenant_id="foton",
        payload=historical,
        columns={
            "tenant_id": "foton",
            "old_customer_id": old.customer_id,
            "new_customer_id": old.customer_id,
            "mapping_kind": "alias",
            "resolution_status": "active",
            "reason": "unchanged",
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        },
        actor="legacy_test",
        ingestion_run_id=None,
    )

    store.record_customer_id_mapping(
        "foton",
        old_customer_id=old.customer_id,
        new_customer_id=target.customer_id,
        reason="tallanto_identity_union",
    )
    mappings = store.list_customer_id_mappings("foton", old_customer_id=old.customer_id)
    assert {(row["new_customer_id"], row["resolution_status"]) for row in mappings} == {
        (old.customer_id, "superseded"),
        (target.customer_id, "active"),
    }
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
                "whatsapp_update_payload": {"entry": "must_not_be_stored"},
                "wappi_raw_payload": {"entry": "must_not_be_stored"},
                "safe_note": "kept",
            },
        },
        metadata={
            **ev.metadata,
            "telegram_raw_message": {"text": "must_not_be_stored"},
            "business_message": {"secret": "must_not_be_stored"},
            "most_finances_payload": {"payment_summa": "must_not_be_stored"},
            "most_abonements_payload": {"num_visit_left": "must_not_be_stored"},
            "whatsapp_raw_message": {"text": "must_not_be_stored"},
            "wappi_message_payload": {"text": "must_not_be_stored"},
        },
    )
    raw_chunk = replace(
        chunk(ev),
        metadata={
            "telegram_update_payload": {"update_id": "must_not_be_stored"},
            "raw_message": {"text": "must_not_be_stored"},
            "tallanto_api_response": {"records": "must_not_be_stored"},
            "whatsapp_raw_payload": {"records": "must_not_be_stored"},
            "wappi_payload": {"records": "must_not_be_stored"},
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
    assert "whatsapp_update_payload" not in dump
    assert "whatsapp_raw_message" not in dump
    assert "whatsapp_raw_payload" not in dump
    assert "wappi_raw_payload" not in dump
    assert "wappi_message_payload" not in dump
    assert "wappi_payload" not in dump
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


def test_bot_context_search_requires_canonical_projection_in_fts_and_fallback(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    ev = event(customer)
    safe_chunk = replace(
        chunk(ev),
        chunk_id=None,
        source_ref="safe-context",
        text="Единый маркер контекста доступен боту.",
        summary="Единый маркер",
        allowed_for_bot=True,
        requires_manager_review=False,
    )
    blocked_chunk = replace(
        chunk(ev),
        chunk_id=None,
        source_ref="blocked-channel-context",
        text="Единый маркер контекста из канала требует проверки менеджера.",
        summary="Единый маркер",
        allowed_for_bot=False,
        requires_manager_review=True,
    )
    store.upsert_customer(customer)
    store.upsert_event(ev)
    store.upsert_bot_context_chunk(safe_chunk)
    store.upsert_bot_context_chunk(blocked_chunk)

    for mode in ("fts", "fallback"):
        bot_safe = store.search_timeline(
            "foton",
            "маркер",
            scopes=("bot_context",),
            allowed_for_bot=True,
            mode=mode,
            limit=10,
        )
        blocked = store.search_timeline(
            "foton",
            "маркер",
            scopes=("bot_context",),
            allowed_for_bot=False,
            mode=mode,
            limit=10,
        )
        assert bot_safe["items"] == []
        assert [item["record"]["source_ref"] for item in blocked["items"]] == ["blocked-channel-context"]
    store.close()


def test_revoking_event_chunks_keeps_active_event_in_fts(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    ev = replace(
        event(customer),
        summary="уникальноесобытие остаётся доступным",
        text_preview="уникальноесобытие",
    )
    context = replace(
        chunk(ev),
        chunk_id=None,
        source_ref="mail-context",
        text="уникальныйконтекст должен исчезнуть",
        summary="уникальныйконтекст",
        source_system="mail_archive_stage2",
        allowed_for_bot=False,
        requires_manager_review=True,
    )
    other_context = replace(
        chunk(ev),
        chunk_id=None,
        source_ref="trusted-context",
        text="другойконтекст должен остаться",
        summary="другойконтекст",
        source_system="trusted_summary",
        allowed_for_bot=True,
        requires_manager_review=False,
    )
    store.upsert_customer(customer)
    store.upsert_event(ev)
    store.upsert_bot_context_chunk(context)
    store.upsert_bot_context_chunk(other_context)

    assert store.search_timeline("foton", "уникальноесобытие", mode="fts")["items"]
    assert store.search_timeline("foton", "уникальныйконтекст", mode="fts")["items"]
    assert store.search_timeline("foton", "другойконтекст", mode="fts")["items"]
    with store.bulk_write():
        assert store.revoke_bot_context_chunks_for_event(
            "foton",
            event_id=ev.event_id,
            source_system="mail_archive_stage2",
            reason="identity_revalidated",
        ) == 1

    event_result = store.search_timeline("foton", "уникальноесобытие", mode="fts")
    chunk_result = store.search_timeline("foton", "уникальныйконтекст", mode="fts")
    assert {item["scope"] for item in event_result["items"]} == {"event"}
    assert chunk_result["items"] == []
    assert store.search_timeline("foton", "другойконтекст", mode="fts")["items"]

    store.upsert_bot_context_chunk(context)
    restored = store.search_timeline("foton", "уникальныйконтекст", mode="fts")
    assert {item["scope"] for item in restored["items"]} == {"bot_context"}
    store.close()


def test_soft_delete_hides_events_and_chunks_from_store_read_api_and_rebuilt_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=StepClock())
    customer = identity()
    first = event(customer, source_id="call-canon", summary="Каноническая строка без скрытого маркера.")
    duplicate = replace(
        event(customer, source_id="call-duplicate", summary="секретныйдубль нужно скрыть."),
        event_at=NOW + timedelta(minutes=2),
        text_preview="секретныйдубль в событии.",
        subject="Скрытый дубль",
    )
    visible_chunk = replace(chunk(first), source_ref="canon-context", text="Канонический контекст.")
    hidden_chunk = replace(
        chunk(duplicate),
        source_ref="hidden-context",
        text="секретныйдубль в chunk.",
        summary="секретныйдубль",
    )
    store.upsert_customer(customer)
    store.upsert_event(first)
    store.upsert_event(duplicate)
    store.upsert_bot_context_chunk(visible_chunk)
    store.upsert_bot_context_chunk(hidden_chunk)

    assert store.search_timeline("foton", "секретныйдубль", mode="fallback", limit=10)["items"]
    assert store.search_timeline("foton", "секретныйдубль", mode="fts", limit=10)["items"]

    result = store.mark_timeline_events_superseded(
        "foton",
        canonical_event_id=first.event_id,
        duplicate_event_ids=(duplicate.event_id,),
        actor="test",
    )
    store._rebuild_fts_indexes()
    store._con.commit()

    assert result["superseded_events"] == 1
    assert result["superseded_chunks"] == 1
    assert store.summary()["counts"]["timeline_events"] == 1
    assert [item["event_id"] for item in store.list_events_by_customer("foton", customer.customer_id)["items"]] == [
        first.event_id
    ]
    assert store.search_timeline("foton", "секретныйдубль", mode="fallback", limit=10)["items"] == []
    assert store.search_timeline("foton", "секретныйдубль", mode="fts", limit=10)["items"] == []
    store.close()

    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        profile = api.customer_profile("foton", customer.customer_id, event_limit=10, bot_context_limit=10)
        context = api.bot_context("foton", customer.customer_id, allowed_only=False, limit=10)

        assert [item["event_id"] for item in profile["timeline"]["items"]] == [first.event_id]
        assert context["summary"]["total_chunks"] == 1
        assert api._count("timeline_events", "tenant_id = ? AND customer_id = ?", ("foton", customer.customer_id)) == 1
        assert len(
            api._records(
                "bot_context_chunks",
                "tenant_id = ? AND customer_id = ?",
                ("foton", customer.customer_id),
                order_by="chunk_id",
                limit=10,
            )
        ) == 1


def test_supersession_is_finite_idempotent_and_preserves_opaque_chunk_tombstones(
    tmp_path: Path,
) -> None:
    store = open_store(tmp_path)
    customer = identity(phone="+79000000331")
    canonical = event(customer, source_id="finite-canonical")
    duplicate = event(customer, source_id="finite-duplicate")
    other = event(customer, source_id="finite-other")
    opaque_chunk = replace(chunk(duplicate), text="Контекст с независимым tombstone.")
    duplicate_signal = signal(duplicate)
    store.upsert_customer(customer)
    for item in (canonical, duplicate, other):
        store.upsert_event(item)
    store.upsert_signal(duplicate_signal)
    store.upsert_bot_context_chunk(opaque_chunk)
    store.retire_bot_context_chunk(opaque_chunk.chunk_id, reason="policy_revoked")
    marker_before = store._con.execute(
        "SELECT superseded_by FROM bot_context_chunks WHERE chunk_id=?", (opaque_chunk.chunk_id,)
    ).fetchone()[0]

    first = store.mark_timeline_events_superseded(
        customer.tenant_id,
        canonical_event_id=canonical.event_id,
        duplicate_event_ids=(duplicate.event_id,),
        actor="test",
    )
    audit_before = store._con.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action='timeline_events_superseded'"
    ).fetchone()[0]
    changes_before = store._con.total_changes
    repeated = store.mark_timeline_events_superseded(
        customer.tenant_id,
        canonical_event_id=canonical.event_id,
        duplicate_event_ids=(duplicate.event_id,),
        actor="test",
    )

    assert first["superseded_events"] == 1 and first["superseded_chunks"] == 0
    stale_signal = store._con.execute(
        "SELECT event_id,status,record_json FROM derived_signals WHERE signal_id=?",
        (duplicate_signal.signal_id,),
    ).fetchone()
    assert stale_signal["event_id"] is None and stale_signal["status"] == "stale"
    assert json.loads(stale_signal["record_json"])["source_event_ids"] == []
    assert repeated["superseded_events"] == 0 and repeated["superseded_chunks"] == 0
    assert repeated["audit_id"] is None
    assert store._con.total_changes == changes_before
    assert store._con.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action='timeline_events_superseded'"
    ).fetchone()[0] == audit_before
    assert store._con.execute(
        "SELECT superseded_by FROM bot_context_chunks WHERE chunk_id=?", (opaque_chunk.chunk_id,)
    ).fetchone()[0] == marker_before
    legacy_payload = json.loads(stale_signal["record_json"])
    legacy_payload.update(
        {
            "event_id": duplicate.event_id,
            "source_event_ids": [duplicate.event_id],
            "status": "active",
        }
    )
    store._con.execute(  # noqa: SLF001 - inject the exact legacy state the repair must converge.
        "UPDATE derived_signals SET event_id=?,status='active',record_json=? WHERE signal_id=?",
        (duplicate.event_id, json.dumps(legacy_payload), duplicate_signal.signal_id),
    )
    repaired = store.mark_timeline_events_superseded(
        customer.tenant_id,
        canonical_event_id=canonical.event_id,
        duplicate_event_ids=(duplicate.event_id,),
        actor="test",
    )
    repaired_signal = store._con.execute(
        "SELECT event_id,status,record_json FROM derived_signals WHERE signal_id=?",
        (duplicate_signal.signal_id,),
    ).fetchone()
    assert repaired["superseded_events"] == 0
    assert repaired["superseded_signals"] == 1
    assert repaired["superseded_chunks"] == 0
    assert repaired_signal["event_id"] is None and repaired_signal["status"] == "stale"
    assert json.loads(repaired_signal["record_json"])["source_event_ids"] == []
    with pytest.raises(ValueError, match="canonical timeline event must be active"):
        store.mark_timeline_events_superseded(
            customer.tenant_id,
            canonical_event_id=duplicate.event_id,
            duplicate_event_ids=(other.event_id,),
        )
    with pytest.raises(ValueError, match="different supersession"):
        store.mark_timeline_events_superseded(
            customer.tenant_id,
            canonical_event_id=other.event_id,
            duplicate_event_ids=(duplicate.event_id,),
        )
    with pytest.raises(ValueError, match="event is superseded"):
        store.upsert_signal(
            replace(
                signal(duplicate),
                signal_id=None,
                signal_type="late_signal",
            )
        )
    store.close()


def test_supersession_counts_one_multi_source_signal_once(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity(phone="+79000000332")
    canonical = event(customer, source_id="multi-source-canonical")
    first_duplicate = event(customer, source_id="multi-source-first")
    second_duplicate = event(customer, source_id="multi-source-second")
    multi_source = DerivedSignal(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        source_event_ids=(first_duplicate.event_id, second_duplicate.event_id),
        signal_type="multi_source_duplicate",
        severity=SignalSeverity.HIGH,
        evidence_text="Один вывод основан на двух дублях.",
        status="active",
        created_at=NOW,
    )
    store.upsert_customer(customer)
    for item in (canonical, first_duplicate, second_duplicate):
        store.upsert_event(item)
    store.upsert_signal(multi_source)

    first = store.mark_timeline_events_superseded(
        customer.tenant_id,
        canonical_event_id=canonical.event_id,
        duplicate_event_ids=(first_duplicate.event_id, second_duplicate.event_id),
    )
    repeated = store.mark_timeline_events_superseded(
        customer.tenant_id,
        canonical_event_id=canonical.event_id,
        duplicate_event_ids=(first_duplicate.event_id, second_duplicate.event_id),
    )

    assert first["superseded_signals"] == 1
    assert repeated["superseded_signals"] == 0
    store.close()


def test_soft_delete_rejects_none_customer_groups_before_write(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    first = email_event(None, source_id="web-form-1")
    duplicate = email_event(None, source_id="web-form-2", event_at=NOW + timedelta(minutes=1))
    store.upsert_event(first)
    store.upsert_event(duplicate)

    with pytest.raises(ValueError, match="customer_id NULL"):
        store.mark_timeline_events_superseded(
            "foton",
            canonical_event_id=first.event_id,
            duplicate_event_ids=(duplicate.event_id,),
            actor="test",
        )

    assert store.summary()["counts"]["timeline_events"] == 2
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


def test_email_content_duplicate_with_new_source_id_is_skipped(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    first = email_event(customer, source_id="mail-1")
    second = email_event(
        customer,
        source_id="mail-2",
        subject="  заявка   с сайта ",
        text_preview="Клиент уточняет расписание группы.",
        summary="Клиент уточнил расписание группы и попросил ответить.",
        event_at=NOW + timedelta(seconds=30),
    )

    store.upsert_customer(customer)
    created = store.upsert_event(first)
    duplicate = store.upsert_event(second)

    assert created.created is True
    assert duplicate.created is False
    assert duplicate.status == "duplicate"
    assert duplicate.record_id == first.event_id
    assert store.summary()["counts"]["timeline_events"] == 1
    stored = store.get_event("foton", first.event_id)
    assert stored["source_id"] == "mail-1"
    store.close()


def test_owner_change_rejected_as_duplicate_has_no_dependency_side_effects(
    tmp_path: Path,
) -> None:
    store = open_store(tmp_path)
    first = identity(phone="+79000000341")
    second = identity(phone="+79000000342")
    original = email_event(first, source_id="mail-owner-before")
    target = email_event(second, source_id="mail-owner-target")
    derived = signal(original)
    context = replace(
        chunk(original),
        allowed_for_bot=False,
        requires_manager_review=True,
    )
    for customer in (first, second):
        store.upsert_customer(customer)
    store.upsert_event(original)
    store.upsert_event(target)
    store.upsert_signal(derived)
    store.upsert_bot_context_chunk(context)
    audit_before = store._con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

    duplicate = store.upsert_event(
        replace(
            original,
            customer_id=second.customer_id,
            source_id=target.source_id,
        )
    )

    assert duplicate.status == "duplicate" and duplicate.record_id == target.event_id
    assert store._con.execute(
        "SELECT customer_id FROM timeline_events WHERE event_id=?", (original.event_id,)
    ).fetchone()[0] == first.customer_id
    assert tuple(
        store._con.execute(
            "SELECT event_id,status FROM derived_signals WHERE signal_id=?", (derived.signal_id,)
        ).fetchone()
    ) == (original.event_id, "active")
    assert tuple(
        store._con.execute(
            "SELECT event_id,superseded_by FROM bot_context_chunks WHERE chunk_id=?", (context.chunk_id,)
        ).fetchone()
    ) == (original.event_id, None)
    assert store._con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == audit_before
    store.close()


def test_email_content_duplicate_without_customer_is_not_skipped(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    first = email_event(None, source_id="web-form-1")
    second = email_event(None, source_id="web-form-2", event_at=NOW + timedelta(seconds=30))

    created = store.upsert_event(first)
    second_created = store.upsert_event(second)

    assert created.created is True
    assert second_created.created is True
    assert store.summary()["counts"]["timeline_events"] == 2
    store.close()


def test_email_content_key_requires_identical_text_preview(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    first = email_event(customer, source_id="mail-1", text_preview="Первый текст письма.")
    second = email_event(
        customer,
        source_id="mail-2",
        text_preview="Другой текст письма.",
        event_at=NOW + timedelta(seconds=30),
    )

    store.upsert_customer(customer)
    created = store.upsert_event(first)
    second_created = store.upsert_event(second)

    assert created.created is True
    assert second_created.created is True
    assert store.summary()["counts"]["timeline_events"] == 2
    store.close()


def test_email_content_duplicate_ignores_superseded_candidate(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    first = email_event(customer, source_id="mail-1", text_preview="Первый вариант.")
    hidden = email_event(customer, source_id="mail-2", text_preview="Второй вариант.")
    incoming = email_event(customer, source_id="mail-3", text_preview="Второй вариант.")
    store.upsert_customer(customer)
    assert store.upsert_event(first).created is True
    assert store.upsert_event(hidden).created is True
    store.mark_timeline_events_superseded(
        "foton",
        canonical_event_id=first.event_id,
        duplicate_event_ids=(hidden.event_id,),
        actor="test",
    )

    created = store.upsert_event(incoming)

    assert created.created is True
    assert created.record_id == incoming.event_id
    assert store.summary()["counts"]["timeline_events"] == 2
    store.close()


def test_email_content_key_normalizes_timezone_equivalent_minutes(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    first = email_event(customer, source_id="mail-utc", event_at=NOW)
    second = email_event(
        customer,
        source_id="mail-msk",
        event_at=(NOW + timedelta(seconds=30)).astimezone(timezone(timedelta(hours=3))),
    )

    store.upsert_customer(customer)
    created = store.upsert_event(first)
    duplicate = store.upsert_event(second)

    assert created.created is True
    assert duplicate.created is False
    assert duplicate.status == "duplicate"
    assert duplicate.record_id == first.event_id
    assert store.summary()["counts"]["timeline_events"] == 1
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


def test_writer_rejects_cross_owner_and_cross_tenant_graph_edges(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    first = identity(phone="+79000000301")
    second = identity(phone="+79000000302")
    demo = identity(tenant_id="demo", phone="+79000000303")
    opp = opportunity(first, source_id="owner-guard")
    ev = event(first, opp, source_id="owner-guard")
    context = chunk(ev)
    for customer in (first, second, demo):
        store.upsert_customer(customer)
    store.upsert_opportunity(opp)
    store.upsert_event(ev)
    store.upsert_bot_context_chunk(context)

    with pytest.raises(ValueError, match="opportunity owner does not match"):
        store.upsert_event(
            replace(
                event(second, source_id="foreign-opportunity"),
                opportunity_id=opp.opportunity_id,
            )
        )
    with pytest.raises(ValueError, match="event owner does not match"):
        store.upsert_signal(
            replace(
                signal(ev),
                signal_id=None,
                customer_id=second.customer_id,
                opportunity_id=None,
            )
        )
    with pytest.raises(ValueError, match="event owner does not match"):
        store.upsert_bot_context_chunk(
            replace(
                context,
                chunk_id=None,
                customer_id=second.customer_id,
                opportunity_id=None,
                source_ref="foreign-event-owner",
            )
        )
    with pytest.raises(ValueError, match="chunk owner does not match"):
        store.upsert_bot_context_chunk(
            replace(
                context,
                customer_id=second.customer_id,
                opportunity_id=None,
                event_id=None,
            )
        )
    with pytest.raises(ValueError, match="opportunity tenant does not match"):
        store.upsert_opportunity(
            replace(
                opp,
                tenant_id="demo",
                customer_id=demo.customer_id,
                opportunity_id=opp.opportunity_id,
            )
        )
    with pytest.raises(ValueError, match="event tenant does not match"):
        store.upsert_event(
            replace(
                ev,
                tenant_id="demo",
                customer_id=demo.customer_id,
                opportunity_id=None,
                event_id=ev.event_id,
            )
        )

    assert store.summary()["counts"]["timeline_events"] == 1
    assert store.summary()["counts"]["derived_signals"] == 0
    assert store.summary()["counts"]["bot_context_chunks"] == 1
    store.close()


def test_bot_context_accepts_summary_and_only_reactivates_an_active_same_owner_event(
    tmp_path: Path,
) -> None:
    store = open_store(tmp_path)
    customer = identity(phone="+79000000311")
    canonical = event(customer, source_id="chunk-canonical")
    duplicate = event(customer, source_id="chunk-duplicate")
    summary = BotContextChunk(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        source_ref="manager-summary",
        source_system="customer_timeline_summary",
        chunk_type="manager_only",
        text="Краткая сводка менеджеру.",
        allowed_for_bot=False,
        requires_manager_review=True,
        created_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_event(canonical)
    store.upsert_event(duplicate)
    store.upsert_bot_context_chunk(summary)
    store.retire_bot_context_chunk(summary.chunk_id, reason="older_snapshot")

    repeated = store.upsert_bot_context_chunk(summary)

    assert repeated.status == "duplicate"
    assert store._con.execute(
        "SELECT superseded_by FROM bot_context_chunks WHERE chunk_id=?", (summary.chunk_id,)
    ).fetchone()[0] is None
    store.mark_timeline_events_superseded(
        customer.tenant_id,
        canonical_event_id=canonical.event_id,
        duplicate_event_ids=(duplicate.event_id,),
    )
    with pytest.raises(ValueError, match="event is superseded"):
        store.upsert_bot_context_chunk(replace(chunk(duplicate), source_ref="hidden-event-context"))
    store.close()


def test_opportunity_owner_change_detaches_all_old_owner_dependencies(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    first = identity(phone="+79000000321")
    second = identity(phone="+79000000322")
    opp = opportunity(first, source_id="moving-opportunity")
    ev = event(first, opp, source_id="moving-opportunity")
    derived = signal(ev)
    context = chunk(ev)
    for customer in (first, second):
        store.upsert_customer(customer)
    store.upsert_opportunity(opp)
    store.upsert_event(ev)
    store.upsert_signal(derived)
    store.upsert_bot_context_chunk(context)

    store.upsert_opportunity(replace(opp, customer_id=second.customer_id))

    event_row = store._con.execute(
        "SELECT opportunity_id,record_json FROM timeline_events WHERE event_id=?", (ev.event_id,)
    ).fetchone()
    signal_row = store._con.execute(
        "SELECT opportunity_id,status,record_json FROM derived_signals WHERE signal_id=?",
        (derived.signal_id,),
    ).fetchone()
    chunk_row = store._con.execute(
        "SELECT opportunity_id,superseded_by,record_json FROM bot_context_chunks WHERE chunk_id=?",
        (context.chunk_id,),
    ).fetchone()
    assert event_row["opportunity_id"] is None
    assert json.loads(event_row["record_json"])["opportunity_id"] is None
    assert signal_row["opportunity_id"] is None and signal_row["status"] == "stale"
    assert json.loads(signal_row["record_json"])["opportunity_id"] is None
    assert chunk_row["opportunity_id"] is None and chunk_row["superseded_by"]
    assert json.loads(chunk_row["record_json"])["opportunity_id"] is None
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


def test_hidden_bot_chunk_does_not_retain_ambiguous_opportunity(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    customer = identity()
    opp = opportunity(customer)
    store.upsert_customer(customer)
    store.upsert_opportunity(opp)
    bot_chunk = BotContextChunk(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        opportunity_id=opp.opportunity_id,
        source_ref="old-summary",
        source_system="customer_timeline_summary",
        chunk_type="manager_only",
        text="Старая память",
        allowed_for_bot=False,
        requires_manager_review=True,
        created_at=NOW,
    )
    store.upsert_bot_context_chunk(bot_chunk)
    store.retire_bot_context_chunk(bot_chunk.chunk_id, reason="older_snapshot")

    result = store.delete_unreferenced_opportunity("amocrm_snapshot", opp.opportunity_id)

    assert result.status == "deleted"
    row = store._con.execute(
        "SELECT opportunity_id,superseded_by,record_json FROM bot_context_chunks WHERE chunk_id=?",
        (bot_chunk.chunk_id,),
    ).fetchone()
    assert row["opportunity_id"] is None
    assert row["superseded_by"] == "retired:older_snapshot"
    assert json.loads(row["record_json"])["opportunity_id"] is None
    store.close()


def test_event_owner_change_retires_old_customer_dependencies(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000001")
    second_customer = identity(phone="+79000000002")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    original = event(first_customer, source_id="owner-change")
    old_signal = signal(original)
    secondary_signal = replace(
        old_signal,
        signal_id=None,
        event_id=None,
        source_event_ids=(original.event_id,),
        signal_type="secondary_only",
    )
    old_chunk = chunk(original)
    store.upsert_event(original)
    store.upsert_signal(old_signal)
    store.upsert_signal(secondary_signal)
    store.upsert_bot_context_chunk(old_chunk)

    store.upsert_event(replace(original, customer_id=second_customer.customer_id))

    event_row = store._con.execute(
        "SELECT customer_id FROM timeline_events WHERE event_id=?", (original.event_id,)
    ).fetchone()
    signal_row = store._con.execute(
        "SELECT event_id,status,record_json FROM derived_signals WHERE signal_id=?", (old_signal.signal_id,)
    ).fetchone()
    secondary_row = store._con.execute(
        "SELECT event_id,status,record_json FROM derived_signals WHERE signal_id=?",
        (secondary_signal.signal_id,),
    ).fetchone()
    chunk_row = store._con.execute(
        "SELECT event_id,superseded_by,record_json FROM bot_context_chunks WHERE chunk_id=?", (old_chunk.chunk_id,)
    ).fetchone()
    assert event_row["customer_id"] == second_customer.customer_id
    assert signal_row["event_id"] is None
    assert signal_row["status"] == "stale"
    assert json.loads(signal_row["record_json"])["event_id"] is None
    assert secondary_row["event_id"] is None and secondary_row["status"] == "stale"
    assert json.loads(secondary_row["record_json"])["source_event_ids"] == []
    assert chunk_row["event_id"] is None
    assert chunk_row["superseded_by"] == f"event_owner_changed:{original.event_id}"
    assert json.loads(chunk_row["record_json"])["event_id"] is None
    assert store._con.execute(
        "SELECT COUNT(*) FROM bot_context_chunk_fts WHERE chunk_id=?", (old_chunk.chunk_id,)
    ).fetchone()[0] == 0
    store.close()


def test_event_owner_change_releases_cross_customer_supersession_without_full_fts_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000021")
    second_customer = identity(phone="+79000000022")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    canonical = event(first_customer, source_id="canonical-owner")
    duplicate = event(first_customer, source_id="duplicate-owner", summary="уникальныйдубль")
    old_chunk = replace(chunk(duplicate), text="уникальныйконтекстдубля")
    store.upsert_event(canonical)
    store.upsert_event(duplicate)
    store.upsert_bot_context_chunk(old_chunk)
    store.mark_timeline_events_superseded(
        "foton", canonical_event_id=canonical.event_id, duplicate_event_ids=(duplicate.event_id,)
    )
    monkeypatch.setattr(store, "_rebuild_fts_indexes", lambda: pytest.fail("full FTS rebuild"))

    store.upsert_event(replace(duplicate, customer_id=second_customer.customer_id))

    moved = store._con.execute(
        "SELECT customer_id,superseded_by FROM timeline_events WHERE event_id=?", (duplicate.event_id,)
    ).fetchone()
    retired_chunk = store._con.execute(
        "SELECT event_id,superseded_by FROM bot_context_chunks WHERE chunk_id=?", (old_chunk.chunk_id,)
    ).fetchone()
    assert tuple(moved) == (second_customer.customer_id, None)
    assert retired_chunk["event_id"] is None and retired_chunk["superseded_by"]
    assert store.search_timeline("foton", "уникальныйдубль", customer_id=second_customer.customer_id)["items"]
    assert store.search_timeline("foton", "уникальныйдубль", customer_id=first_customer.customer_id)["items"] == []
    assert store.search_timeline("foton", "уникальныйконтекстдубля")["items"] == []
    store.close()


def test_canonical_owner_change_reactivates_only_same_owner_duplicates(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000031")
    second_customer = identity(phone="+79000000032")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    canonical = event(first_customer, source_id="canonical-moves")
    duplicate = event(first_customer, source_id="duplicate-stays", summary="возвращенныйдубль")
    duplicate_chunk = replace(chunk(duplicate), text="возвращенныйконтекст")
    store.upsert_event(canonical)
    store.upsert_event(duplicate)
    store.upsert_bot_context_chunk(duplicate_chunk)
    store.mark_timeline_events_superseded(
        "foton", canonical_event_id=canonical.event_id, duplicate_event_ids=(duplicate.event_id,)
    )

    store.upsert_event(replace(canonical, customer_id=second_customer.customer_id))

    restored = store._con.execute(
        "SELECT superseded_by FROM timeline_events WHERE event_id=?", (duplicate.event_id,)
    ).fetchone()
    restored_chunk = store._con.execute(
        "SELECT superseded_by FROM bot_context_chunks WHERE chunk_id=?", (duplicate_chunk.chunk_id,)
    ).fetchone()
    assert restored["superseded_by"] is None
    assert restored_chunk["superseded_by"] is None
    assert store.search_timeline("foton", "возвращенныйдубль", customer_id=first_customer.customer_id)["items"]
    assert store.search_timeline("foton", "возвращенныйконтекст", customer_id=first_customer.customer_id)["items"]
    assert store.reconcile_event_dependency_owners("foton", actor="test") == 0
    store.close()


def test_canonical_owner_change_keeps_one_active_replacement_per_old_customer(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000034")
    second_customer = identity(phone="+79000000035")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    canonical = event(first_customer, source_id="multi-canonical")
    duplicates = tuple(
        replace(
            event(first_customer, source_id=f"multi-duplicate-{index}"),
            event_at=NOW + timedelta(minutes=index),
        )
        for index in (1, 2)
    )
    chunks = tuple(replace(chunk(item), text="мультидубль") for item in duplicates)
    store.upsert_event(canonical)
    for item in duplicates:
        store.upsert_event(item)
    for item in chunks:
        store.upsert_bot_context_chunk(item)
    store.mark_timeline_events_superseded(
        "foton",
        canonical_event_id=canonical.event_id,
        duplicate_event_ids=tuple(item.event_id for item in duplicates),
    )

    store.upsert_event(replace(canonical, customer_id=second_customer.customer_id))

    event_rows = store._con.execute(
        "SELECT event_id,superseded_by FROM timeline_events WHERE event_id IN (?,?) ORDER BY event_id",
        tuple(item.event_id for item in duplicates),
    ).fetchall()
    chunk_rows = store._con.execute(
        "SELECT chunk_id,superseded_by FROM bot_context_chunks WHERE chunk_id IN (?,?) ORDER BY chunk_id",
        tuple(item.chunk_id for item in chunks),
    ).fetchall()
    assert sum(row["superseded_by"] is None for row in event_rows) == 1
    replacement_id = next(str(row["event_id"]) for row in event_rows if row["superseded_by"] is None)
    assert {row["superseded_by"] for row in event_rows if row["event_id"] != replacement_id} == {replacement_id}
    assert sum(row["superseded_by"] is None for row in chunk_rows) == 1
    assert len(store.search_timeline("foton", "мультидубль", customer_id=first_customer.customer_id)["items"]) == 1
    store.close()


def test_shared_family_phone_does_not_change_valid_supersession(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_child = replace(identity(phone="+79000000041"), customer_id="customer:first-child")
    second_child = replace(identity(phone="+79000000041"), customer_id="customer:second-child")
    store.upsert_customer(first_child)
    store.upsert_customer(second_child)
    canonical = event(first_child, source_id="family-canonical")
    duplicate = event(first_child, source_id="family-duplicate")
    store.upsert_event(canonical)
    store.upsert_event(duplicate)
    store.mark_timeline_events_superseded(
        "foton", canonical_event_id=canonical.event_id, duplicate_event_ids=(duplicate.event_id,)
    )

    store.upsert_event(replace(canonical, summary="Обновлённый текст без смены владельца."))

    row = store._con.execute(
        "SELECT superseded_by FROM timeline_events WHERE event_id=?", (duplicate.event_id,)
    ).fetchone()
    assert row["superseded_by"] == canonical.event_id
    assert store._con.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0] == 2
    store.close()


def test_identity_quarantine_never_reactivates_hidden_duplicate(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    customer = identity(phone="+79000000051")
    store.upsert_customer(customer)
    canonical = event(customer, source_id="quarantined-canonical")
    duplicate = event(customer, source_id="quarantined-duplicate", summary="скрытыйкарантинныйдубль")
    store.upsert_event(canonical)
    store.upsert_event(duplicate)
    store.mark_timeline_events_superseded(
        "foton", canonical_event_id=canonical.event_id, duplicate_event_ids=(duplicate.event_id,)
    )

    store.quarantine_timeline_events_identity_conflict(
        "foton",
        source_system=canonical.source_system,
        source_id=canonical.source_id,
        reason="identity_conflict",
    )
    assert store.reconcile_event_dependency_owners("foton", actor="test") == 0
    store.upsert_event(replace(canonical, customer_id=None, summary="Повторный карантинный импорт."))

    hidden = store._con.execute(
        "SELECT superseded_by FROM timeline_events WHERE event_id=?", (duplicate.event_id,)
    ).fetchone()
    assert hidden["superseded_by"] == canonical.event_id
    assert store.search_timeline("foton", "скрытыйкарантинныйдубль")["items"] == []
    store.close()


def test_identity_conflict_quarantine_detaches_event_and_all_customer_memory(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    customer = identity(phone="+79000000011")
    original = event(customer, source_id="identity-conflict")
    same_source_other_type = replace(
        original,
        event_id=None,
        event_type=TimelineEventType.EMAIL_MESSAGE,
    )
    old_signal = signal(original)
    old_chunk = chunk(original)
    summary = BotContextChunk(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        source_ref=f"bot-safe:{customer.customer_id}",
        source_system="customer_timeline_bot_safe_summary",
        chunk_type="bot_safe_summary",
        text="Последнее занятие подтверждено.",
        allowed_for_bot=True,
        requires_manager_review=False,
        metadata={"brand_context_authorized": True},
        created_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_event(original)
    store.upsert_event(same_source_other_type)
    store.upsert_signal(old_signal)
    store.upsert_bot_context_chunk(old_chunk)
    store.upsert_bot_context_chunk(summary)

    first = store.quarantine_timeline_events_identity_conflict(
        "foton",
        source_system="mango",
        source_id="identity-conflict",
        reason="identity_conflict",
        actor="test",
    )
    first_hash = store._con.execute(
        "SELECT record_hash FROM timeline_events WHERE event_id=?", (original.event_id,)
    ).fetchone()[0]
    second = store.quarantine_timeline_events_identity_conflict(
        "foton",
        source_system="mango",
        source_id="identity-conflict",
        reason="identity_conflict",
        actor="test",
    )

    event_row = store._con.execute(
        "SELECT customer_id,opportunity_id,match_status,confidence,record_json,record_hash "
        "FROM timeline_events WHERE event_id=?",
        (original.event_id,),
    ).fetchone()
    signal_row = store._con.execute(
        "SELECT event_id,status FROM derived_signals WHERE signal_id=?", (old_signal.signal_id,)
    ).fetchone()
    chunk_row = store._con.execute(
        "SELECT event_id,superseded_by FROM bot_context_chunks WHERE chunk_id=?", (old_chunk.chunk_id,)
    ).fetchone()
    summary_row = store._con.execute(
        "SELECT allowed_for_bot,requires_manager_review,superseded_by,record_json "
        "FROM bot_context_chunks WHERE chunk_id=?",
        (summary.chunk_id,),
    ).fetchone()
    summary_audit = store._con.execute(
        "SELECT before_hash,after_hash FROM audit_log WHERE action='bot_context_chunk_retired' "
        "AND entity_id=? ORDER BY created_at DESC LIMIT 1",
        (summary.chunk_id,),
    ).fetchone()
    metadata = json.loads(event_row["record_json"])["metadata"]
    assert first == {
        "existing_event_quarantined": 2,
        "bot_context_chunks_revoked": 1,
        "bot_safe_summaries_revoked": 1,
    }
    assert second == {
        "existing_event_quarantined": 0,
        "bot_context_chunks_revoked": 0,
        "bot_safe_summaries_revoked": 0,
    }
    assert event_row["customer_id"] is None
    assert event_row["opportunity_id"] is None
    assert event_row["match_status"] == "ambiguous"
    assert float(event_row["confidence"]) == 0.0
    assert metadata["pending_attribution"] is True
    assert metadata["allowed_for_bot"] is False
    assert metadata["previous_customer_id"] == customer.customer_id
    assert store._con.execute(
        "SELECT COUNT(*) FROM timeline_events WHERE source_system='mango' AND source_id='identity-conflict' "
        "AND customer_id IS NULL AND match_status='ambiguous'"
    ).fetchone()[0] == 2
    assert event_row["record_hash"] == first_hash
    assert signal_row["event_id"] is None and signal_row["status"] == "stale"
    assert chunk_row["event_id"] is None and chunk_row["superseded_by"]
    assert summary_row["allowed_for_bot"] == 0
    assert summary_row["requires_manager_review"] == 1
    assert summary_row["superseded_by"]
    assert json.loads(summary_row["record_json"])["metadata"]["retired_reason"] == "identity_conflict"
    assert summary_audit["before_hash"] != summary_audit["after_hash"]
    store.close()


def test_identity_conflict_quarantine_retires_summary_without_source_event(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    customer = identity(phone="+79000000012")
    summary = BotContextChunk(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        source_ref=f"bot-safe:{customer.customer_id}",
        source_system="customer_timeline_bot_safe_summary",
        chunk_type="bot_safe_summary",
        text="Старая сводка.",
        allowed_for_bot=True,
        requires_manager_review=False,
        created_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_bot_context_chunk(summary)

    result = store.quarantine_timeline_events_identity_conflict(
        "foton",
        source_system="wappi_telegram",
        source_id="missing-event",
        reason="identity_conflict",
        previous_customer_id=customer.customer_id,
    )

    row = store._con.execute(
        "SELECT allowed_for_bot,requires_manager_review,superseded_by,record_json "
        "FROM bot_context_chunks WHERE chunk_id=?",
        (summary.chunk_id,),
    ).fetchone()
    assert result == {
        "existing_event_quarantined": 0,
        "bot_context_chunks_revoked": 0,
        "bot_safe_summaries_revoked": 1,
    }
    assert row["allowed_for_bot"] == 0
    assert row["requires_manager_review"] == 1
    assert row["superseded_by"]
    assert json.loads(row["record_json"])["metadata"]["retired_reason"] == "identity_conflict"
    store.close()


def test_event_owner_change_retires_signal_when_secondary_source_event_moves(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000021")
    second_customer = identity(phone="+79000000022")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    first_event = event(first_customer, source_id="hot-streak-first")
    last_event = event(first_customer, source_id="hot-streak-last")
    store.upsert_event(first_event)
    store.upsert_event(last_event)
    hot_streak = DerivedSignal(
        tenant_id="foton",
        customer_id=first_customer.customer_id,
        event_id=last_event.event_id,
        source_event_ids=(first_event.event_id, last_event.event_id),
        signal_type="hot_streak",
        severity="high",
        evidence_text="Два сообщения",
        created_at=NOW,
    )
    store.upsert_signal(hot_streak)

    store.upsert_event(replace(first_event, customer_id=second_customer.customer_id))

    row = store._con.execute(
        "SELECT event_id,status,record_json FROM derived_signals WHERE signal_id=?",
        (hot_streak.signal_id,),
    ).fetchone()
    payload = json.loads(row["record_json"])
    assert row["event_id"] == last_event.event_id
    assert row["status"] == "stale"
    assert payload["source_event_ids"] == [last_event.event_id]
    plan = " ".join(
        str(item)
        for row in store._con.execute(
            "EXPLAIN QUERY PLAN SELECT signal_id FROM derived_signals WHERE tenant_id=? "
            "AND json_array_length(record_json,'$.source_event_ids')>1 "
            "AND EXISTS (SELECT 1 FROM json_each(record_json,'$.source_event_ids') WHERE value=?)",
            ("foton", first_event.event_id),
        )
        for item in row
    )
    assert "ix_signals_multi_source" in plan
    store.close()


def test_reconcile_event_dependency_owners_repairs_legacy_mismatch(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000011")
    second_customer = identity(phone="+79000000012")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    original = event(first_customer, source_id="legacy-owner-change")
    old_signal = signal(original)
    old_chunk = chunk(original)
    store.upsert_event(original)
    store.upsert_signal(old_signal)
    store.upsert_bot_context_chunk(old_chunk)
    store._con.execute(
        "UPDATE timeline_events SET customer_id=? WHERE event_id=?",
        (second_customer.customer_id, original.event_id),
    )
    store._commit()

    repaired = store.reconcile_event_dependency_owners("foton", actor="test")

    signal_row = store._con.execute(
        "SELECT event_id,status FROM derived_signals WHERE signal_id=?", (old_signal.signal_id,)
    ).fetchone()
    chunk_row = store._con.execute(
        "SELECT event_id,superseded_by FROM bot_context_chunks WHERE chunk_id=?", (old_chunk.chunk_id,)
    ).fetchone()
    assert repaired == 1
    assert tuple(signal_row) == (None, "stale")
    assert tuple(chunk_row) == (None, f"event_owner_reconciled:{original.event_id}")
    assert store.reconcile_event_dependency_owners("foton", actor="test") == 0
    store.close()


def test_reconcile_event_dependency_owners_uses_multi_source_index(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000031")
    second_customer = identity(phone="+79000000032")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    first_event = event(first_customer, source_id="legacy-multi-first")
    last_event = event(first_customer, source_id="legacy-multi-last")
    store.upsert_event(first_event)
    store.upsert_event(last_event)
    multi_signal = DerivedSignal(
        tenant_id="foton",
        customer_id=first_customer.customer_id,
        event_id=last_event.event_id,
        source_event_ids=(first_event.event_id, last_event.event_id),
        signal_type="hot_streak",
        severity="high",
        evidence_text="Два события",
        created_at=NOW,
    )
    store.upsert_signal(multi_signal)
    store._con.execute(
        "UPDATE timeline_events SET customer_id=? WHERE event_id=?",
        (second_customer.customer_id, first_event.event_id),
    )
    store._commit()

    plan = " ".join(
        str(item)
        for row in store._con.execute(
            "EXPLAIN QUERY PLAN SELECT d.signal_id FROM derived_signals d,"
            "json_each(d.record_json,'$.source_event_ids') source_event "
            "WHERE d.tenant_id=? AND json_array_length(d.record_json,'$.source_event_ids')>1",
            ("foton",),
        )
        for item in row
    )
    assert "ix_signals_multi_source" in plan
    chunk_plan = " ".join(
        str(item)
        for row in store._con.execute(
            "EXPLAIN QUERY PLAN SELECT e.event_id FROM bot_context_chunks b "
            "JOIN timeline_events e ON e.tenant_id=b.tenant_id AND e.event_id=b.event_id "
            "WHERE b.tenant_id=? AND b.event_id IS NOT NULL",
            ("foton",),
        )
        for item in row
    )
    assert "ix_chunks_event_owner" in chunk_plan
    assert store.reconcile_event_dependency_owners("foton", actor="test") == 1
    assert store.reconcile_event_dependency_owners("foton", actor="test") == 0
    payload = json.loads(
        store._con.execute(
            "SELECT record_json FROM derived_signals WHERE signal_id=?", (multi_signal.signal_id,)
        ).fetchone()[0]
    )
    assert payload["source_event_ids"] == [last_event.event_id]
    store.close()


def test_reconcile_event_dependency_owners_rebuilds_missing_fts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    first_customer = identity(phone="+79000000021")
    second_customer = identity(phone="+79000000022")
    store.upsert_customer(first_customer)
    store.upsert_customer(second_customer)
    events = [event(first_customer, source_id=f"legacy-owner-{index}") for index in range(2)]
    chunks = [chunk(item) for item in events]
    for item, context in zip(events, chunks):
        store.upsert_event(item)
        store.upsert_bot_context_chunk(context)
        store._con.execute(
            "UPDATE timeline_events SET customer_id=? WHERE event_id=?",
            (second_customer.customer_id, item.event_id),
        )
        store._con.execute("DELETE FROM bot_context_chunk_fts_keys WHERE chunk_id=?", (context.chunk_id,))
    store._commit()

    rebuilds = 0
    original_rebuild = store._rebuild_fts_indexes

    def counted_rebuild() -> None:
        nonlocal rebuilds
        rebuilds += 1
        original_rebuild()

    monkeypatch.setattr(store, "_rebuild_fts_indexes", counted_rebuild)

    assert store.reconcile_event_dependency_owners("foton", actor="test") == 2
    assert rebuilds == 1
    assert store.search_timeline("foton", "стоимость", mode="fts")["backend"] == "fts5"
    store.close()


def test_unattributed_event_stays_hidden_after_full_fts_rebuild_and_fallback(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    customer = identity()
    hidden = replace(
        event(customer, source_id="nullish-owner"),
        subject="нулевойвладелец",
        text_preview="нулевойвладелец нельзя выдавать",
        summary="нулевойвладелец скрыт",
    )
    store.upsert_customer(customer)
    store.upsert_event(hidden)

    result = store.quarantine_timeline_events_identity_conflict(
        "foton",
        source_system=hidden.source_system,
        source_id=hidden.source_id,
        reason="textual_null_customer_id",
        previous_customer_id=customer.customer_id,
        actor="test",
    )
    assert result["existing_event_quarantined"] == 1
    assert store.search_timeline("foton", "нулевойвладелец", mode="fts")["items"] == []
    assert store.search_timeline("foton", "нулевойвладелец", mode="fallback")["items"] == []

    store._rebuild_fts_indexes()  # noqa: SLF001 - regression covers a full service rebuild.
    assert store.search_timeline("foton", "нулевойвладелец", mode="fts")["items"] == []
    assert store.search_timeline("foton", "нулевойвладелец", mode="fallback")["items"] == []

    payload = customer.to_json_dict()
    payload["customer_id"] = "None"
    store._con.execute(  # noqa: SLF001 - historical corruption fixture.
        "UPDATE customer_identities SET customer_id='None',record_json=? WHERE customer_id=?",
        (json.dumps(payload, ensure_ascii=False), customer.customer_id),
    )
    store._commit()  # noqa: SLF001
    assert store.list_customers("foton")["items"] == []
    integrity = store_module.customer_timeline_integrity_report(store._con)  # noqa: SLF001
    assert integrity["violations"]["nullish_customer_id_customer_identities_customer_id"] == 1
    store.close()
