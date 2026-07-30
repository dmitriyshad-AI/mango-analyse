from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    AmoSnapshotNormalizer,
    BotContextChunk,
    ChannelMessageNormalizer,
    CustomerIdentity,
    CustomerOpportunity,
    DerivedSignal,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    MailMessageNormalizer,
    MangoCallSummaryNormalizer,
    OpportunityType,
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
from mango_mvp.customer_timeline.ingestion import resolve_customer_identity_batches
from mango_mvp.customer_timeline.source_policy import is_non_contentful_call_record


NOW = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
SHA = "b" * 64


class FixedClock:
    def __call__(self) -> datetime:
        return NOW


class StepClock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def test_resolving_same_identity_conflict_twice_is_idempotent(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=StepClock())
    first = store.record_conflict(
        "foton", conflict_type="tallanto_identity_conflict",
        entity_refs=("tallanto_student_id:student-1",), status="resolved",
    )
    first_payload = store.list_conflicts("foton", statuses=("resolved",))["items"][0]
    second = store.record_conflict(
        "foton", conflict_type="tallanto_identity_conflict",
        entity_refs=("tallanto_student_id:student-1",), status="resolved",
    )
    second_payload = store.list_conflicts("foton", statuses=("resolved",))["items"][0]

    assert first.created is True
    assert second.status == "duplicate"
    assert second_payload["resolved_at"] == first_payload["resolved_at"]
    store.close()


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
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=FixedClock())
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
    phone_links = store.list_identity_links("foton", link_type="phone", link_value="+79161112233")
    conflicts = store.list_audit_log("foton", entity_type="timeline_conflict")["items"]

    assert before == after
    assert first.validation_ok is True
    assert first.source_unchanged is True
    assert second.source_unchanged is True
    assert summary["counts"]["customer_identities"] == 2
    assert summary["counts"]["identity_links"] == 6
    assert summary["counts"]["timeline_conflicts"] == 1
    assert summary["counts"]["customer_id_mappings"] == 0
    assert summary["counts"]["ingestion_runs"] == 1
    assert len({item["customer_id"] for item in email_links}) == 2
    assert {item["match_class"] for item in phone_links} == {"ambiguous"}
    assert store.list_customer_id_mappings("foton") == ()
    assert conflicts[0]["action"] == "timeline_conflict_created"
    assert second.write_status_counts["duplicate"] >= first.write_status_counts["created"]
    store.close()


def test_tallanto_normalizer_keeps_additional_contact_links() -> None:
    batch = TallantoSnapshotNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="tallanto_snapshot",
            source_ref="contacts.jsonl:1",
            observed_at=NOW,
            payload={
                "tallanto_id": "s-extra",
                "display_name": "Ученик",
                "primary_phone": "+7 916 111-22-33",
                "phone_extra": "+7 916 444-55-66",
                "primary_email": "primary@example.com",
                "email_extra": "parent@example.com | second@example.com",
                "amo_contact_id": "12345",
            },
        )
    )

    contact_links = {
        (link.link_type.value, link.link_value)
        for link in batch.identity_links
        if link.link_type.value in {"phone", "email"}
    }
    assert contact_links == {
        ("phone", "+79161112233"),
        ("phone", "+79164445566"),
        ("email", "primary@example.com"),
        ("email", "parent@example.com"),
        ("email", "second@example.com"),
    }
    amo_link = next(
        link
        for link in batch.identity_links
        if link.link_type.value == "amo_contact_id" and link.link_value == "12345"
    )
    assert amo_link.match_class == IdentityMatchClass.AMBIGUOUS
    assert amo_link.evidence == {"relationship": "family_amo_contact"}


@pytest.mark.parametrize("amo_contact_id", ("", "-", "abc", "0", "-1"))
def test_tallanto_normalizer_ignores_invalid_amo_contact_id(amo_contact_id: str) -> None:
    batch = TallantoSnapshotNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="tallanto_snapshot",
            source_ref="contacts.jsonl:1",
            observed_at=NOW,
            payload={"tallanto_id": "s-invalid-amo", "amo_contact_id": amo_contact_id},
        )
    )

    assert all(link.link_type.value != "amo_contact_id" for link in batch.identity_links)


def test_tallanto_sequential_shared_contact_demotes_prior_links(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)

    def record(student_id: str) -> TimelineSourceRecord:
        return TimelineSourceRecord(
            source_system="tallanto_snapshot",
            source_ref=f"contacts:{student_id}",
            observed_at=NOW,
            payload={
                "tallanto_id": student_id,
                "display_name": student_id,
                "email_extra": "family@example.com",
            },
        )

    service.import_records(
        (record("student-1"),),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="first",
        idempotency_key="first",
    )
    second = service.import_records(
        (record("student-2"),),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="second",
        idempotency_key="second",
    )
    repeat = service.import_records(
        (record("student-2"),),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="second",
        idempotency_key="second",
    )

    links = store.list_identity_links("foton", link_type="email", link_value="family@example.com")
    assert len({item["customer_id"] for item in links}) == 2
    assert {item["match_class"] for item in links} == {"ambiguous"}
    assert second.write_status_counts["updated"] >= 1
    assert repeat.write_status_counts.get("updated", 0) == 0
    store.close()


def test_tallanto_shared_phone_demotes_phone_aliases_across_sources(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    existing = CustomerIdentity(
        tenant_id="foton",
        customer_id="customer:master",
        identity_status=IdentityStatus.STRONG,
        display_name="Master customer",
    )
    store.upsert_customer(existing)
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=existing.customer_id,
            link_type="mango_client_phone",
            link_value="+79160000000",
            source_system="master_contacts_snapshot",
            source_ref="master:1",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.95,
        )
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=existing.customer_id,
            link_type="tallanto_student_id",
            link_value="student-existing",
            source_system="tallanto_snapshot",
            source_ref="tallanto:student:student-existing",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
        )
    )

    TimelineImportService(store).import_records(
        (
            TimelineSourceRecord(
                source_system="tallanto_snapshot",
                source_ref="contacts:student-1",
                observed_at=NOW,
                payload={
                    "tallanto_id": "student-1",
                    "display_name": "Student 1",
                    "phone": "+7 916 000-00-00",
                },
            ),
        ),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="tallanto",
        idempotency_key="tallanto",
    )

    links = [
        *store.list_identity_links("foton", link_type="phone", link_value="+79160000000"),
        *store.list_identity_links("foton", link_type="mango_client_phone", link_value="+79160000000"),
    ]
    assert len({item["customer_id"] for item in links}) == 2
    assert {item["match_class"] for item in links} == {"ambiguous"}
    store.close()


def test_tallanto_conflict_still_demotes_reused_family_contact(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)
    for customer_id in ("customer:student-1", "customer:student-2"):
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status=IdentityStatus.STRONG,
                display_name=customer_id,
            )
        )
    for customer_id, student_id in (
        ("customer:student-1", "student-1"),
        ("customer:student-2", "student-2"),
    ):
        source_ref = f"tallanto:student:{student_id}"
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type="tallanto_student_id",
                link_value=student_id,
                source_system="tallanto_snapshot",
                source_ref=source_ref,
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type="email",
                link_value="family@example.com",
                source_system="tallanto_snapshot",
                source_ref=source_ref,
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=0.95,
            )
        )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="customer:student-2",
            link_type="phone",
            link_value="+79160000000",
            source_system="amocrm_snapshot",
            source_ref="amo:contact:2",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.95,
        )
    )
    conflicting = TimelineSourceRecord(
        source_system="tallanto_snapshot",
        source_ref="contacts:student-1",
        observed_at=NOW,
        payload={
            "tallanto_id": "student-1",
            "display_name": "Student 2",
            "phone": "+79160000000",
            "email_extra": "family@example.com",
        },
    )
    result = service.import_records(
        (conflicting,),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="conflict",
        idempotency_key="conflict",
    )

    links = store.list_identity_links("foton", link_type="email", link_value="family@example.com")
    assert {item["match_class"] for item in links} == {"ambiguous"}
    assert result.normalized_counts["conflicts"] == 1
    assert result.write_status_counts["updated"] >= 1
    store.close()


def test_tallanto_import_demotes_shared_email_from_other_sources(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)
    for customer_id, source_system in (
        ("customer:master", "master_contacts_snapshot"),
        ("customer:mail", "mail_archive_stage2"),
    ):
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status=IdentityStatus.STRONG,
                display_name=customer_id,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type="email",
                link_value="shared@example.com",
                source_system=source_system,
                source_ref=f"{source_system}:1",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=0.95,
            )
        )

    result = service.import_records(
        (
            TimelineSourceRecord(
                source_system="tallanto_snapshot",
                source_ref="contacts:student-1",
                observed_at=NOW,
                payload={
                    "tallanto_id": "student-1",
                    "display_name": "Student 1",
                    "email_extra": "shared@example.com",
                },
            ),
        ),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="tallanto",
        idempotency_key="tallanto",
    )

    links = store.list_identity_links("foton", link_type="email", link_value="shared@example.com")
    assert {item["match_class"] for item in links} == {"ambiguous"}
    assert result.write_status_counts["updated"] == 2
    store.close()


def test_amo_normalizer_keeps_exact_messenger_identities() -> None:
    batch = AmoSnapshotNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="amocrm_snapshot",
            source_ref="amo.jsonl:1",
            observed_at=NOW,
            payload={
                "entity_id": "101",
                "entity_type": "contact",
                "customer_id": "customer:amo",
                "record": {
                    "custom_fields_values": [
                        {"field_name": "Telegram ID", "values": [{"value": "123456"}, {"value": "user-12x34"}]},
                        {"field_name": "Telegram username", "values": [{"value": "@Parent_Name"}]},
                        {"field_name": "Max User ID", "values": [{"value": "max-user-1"}]},
                    ]
                },
            },
        )
    )

    channel_links = {
        (link.link_type.value, link.link_value, link.match_class.value)
        for link in batch.identity_links
        if link.link_type.value in {"telegram_user_id", "telegram_username", "max_user_id"}
    }
    assert channel_links == {
        ("telegram_user_id", "123456", "strong_unique"),
        ("telegram_username", "parent_name", "strong_unique"),
        ("max_user_id", "max-user-1", "strong_unique"),
    }


def test_amo_normalizer_uses_organization_brand_and_fails_closed_on_mixed_value() -> None:
    def normalize(organization: str):
        return AmoSnapshotNormalizer(tenant_id="foton").normalize(
            TimelineSourceRecord(
                source_system="amocrm_snapshot",
                source_ref=f"amo:{organization}",
                observed_at=NOW,
                payload={
                    "entity_id": f"lead-{organization}",
                    "entity_type": "lead",
                    "record": {
                        "custom_fields_values": [
                            {"field_name": "Организация", "values": [{"value": organization}]},
                        ]
                    },
                },
            )
        )

    unpk = normalize("УНПК МФТИ")
    mixed = normalize("Фотон / УНПК")

    assert unpk.opportunities[0].product_context["brand"] == "unpk"
    assert unpk.events[0].record["brand"] == "unpk"
    assert mixed.opportunities[0].product_context["brand"] == "unknown"
    assert mixed.opportunities[0].product_context["brand_source"] == "amo_organization_conflict"

    known_and_unknown = AmoSnapshotNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="amocrm_snapshot",
            source_ref="amo:known-and-unknown",
            observed_at=NOW,
            payload={
                "entity_id": "lead-known-and-unknown",
                "entity_type": "lead",
                "record": {
                    "custom_fields_values": [{
                        "field_name": "Организация",
                        "values": [{"value": "Фотон"}, {"value": "ООО Ромашка"}],
                    }]
                },
            },
        )
    )
    assert known_and_unknown.opportunities[0].product_context["brand"] == "unknown"
    assert known_and_unknown.opportunities[0].product_context["brand_source"] == "amo_organization_conflict"


def test_tallanto_normalizer_lifts_filial_brand_without_guessing_mixed() -> None:
    def normalize(branch: object):
        return TallantoSnapshotNormalizer(tenant_id="foton").normalize(
            TimelineSourceRecord(
                source_system="tallanto_snapshot",
                source_ref=f"tallanto:{branch}",
                observed_at=NOW,
                payload={"entity_id": f"student-{branch}", "branch": branch, "name": "Test Student"},
            )
        )

    unpk = normalize({"sretenka": "Сретенка"})
    mixed = normalize({"foton": "Фотон", "mfti": "МФТИ"})

    assert unpk.events[0].record["brand"] == "unpk"
    assert unpk.opportunities[0].product_context["brand"] == "unpk"
    assert mixed.events[0].record["brand"] == "unknown"
    assert mixed.opportunities[0].product_context["brand_source"] == "tallanto_filial"


def test_local_source_loader_honors_limit(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(5)), encoding="utf-8")

    records = load_local_source_records(
        source,
        allowed_root=tmp_path,
        source_system="tallanto_snapshot",
        limit=2,
    )

    assert [record.payload["id"] for record in records] == [0, 1]


def test_tallanto_student_id_keeps_customer_when_contact_changes(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)

    def import_snapshot(phone: str, key: str) -> None:
        record = TimelineSourceRecord(
            source_system="tallanto_snapshot",
            source_ref="snapshot.jsonl#1",
            payload={
                "tallanto_id": "student-1",
                "display_name": "Ученик",
                "primary_phone": phone,
                "snapshot_at": NOW.isoformat(),
            },
            observed_at=NOW,
        )
        service.import_records(
            (record,),
            normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
            tenant_id="foton",
            source_ref=key,
            idempotency_key=key,
            actor="test",
        )

    import_snapshot("+7 916 111-22-33", "snapshot-1")
    first_customer_id = store.list_identity_links(
        "foton",
        link_type="tallanto_student_id",
        link_value="student-1",
    )[0]["customer_id"]
    first_touch_count = store.get_customer("foton", first_customer_id)["touch_count"]
    import_snapshot("+7 916 999-88-77", "snapshot-2")
    tallanto_links = store.list_identity_links(
        "foton",
        link_type="tallanto_student_id",
        link_value="student-1",
    )

    assert {link["customer_id"] for link in tallanto_links} == {first_customer_id}
    assert store.summary()["counts"]["customer_identities"] == 1
    assert store.get_customer("foton", first_customer_id)["touch_count"] == first_touch_count
    store.close()


def test_tallanto_does_not_merge_two_existing_customers_on_conflicting_contact(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)
    first = TimelineSourceRecord(
        source_system="tallanto_snapshot",
        source_ref="snapshot#first",
        payload={
            "tallanto_id": "student-conflict",
            "display_name": "Ученик",
            "primary_phone": "+7 916 111-22-33",
            "snapshot_at": NOW.isoformat(),
        },
        observed_at=NOW,
    )
    service.import_records(
        (first,),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="snapshot-first",
        idempotency_key="snapshot-first",
        actor="test",
    )
    first_customer = store.list_identity_links(
        "foton", link_type="tallanto_student_id", link_value="student-conflict"
    )[0]["customer_id"]
    other = CustomerIdentity(
        tenant_id="foton",
        identity_status="strong",
        display_name="Другой клиент",
        primary_phone="+79169998877",
        created_at=NOW,
        updated_at=NOW,
    )
    store.upsert_customer(other)
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=other.customer_id,
            link_type="phone",
            link_value="+79169998877",
            source_system="test",
            source_ref="other-phone",
        )
    )
    conflicting = TimelineSourceRecord(
        source_system="tallanto_snapshot",
        source_ref="snapshot#conflict",
        payload={
            "tallanto_id": "student-conflict",
            "display_name": "Ученик",
            "primary_phone": "+7 916 999-88-77",
            "snapshot_at": NOW.isoformat(),
        },
        observed_at=NOW,
    )

    report = service.import_records(
        (conflicting,),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="snapshot-conflict",
        idempotency_key="snapshot-conflict",
        actor="test",
    )

    links = store.list_identity_links("foton", link_type="tallanto_student_id", link_value="student-conflict")
    assert len(links) == 1 and links[0]["customer_id"] == first_customer
    assert store.summary()["counts"]["customer_identities"] == 2
    assert report.normalized_counts["conflicts"] == 1
    store.close()
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT conflict_type FROM timeline_conflicts").fetchone()[0] == "tallanto_identity_conflict"


def test_fresh_tallanto_card_repairs_historical_duplicate_exact_links(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    for customer_id, phone in (("customer:first", "+79161112233"), ("customer:wrong", "+79169998877")):
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton", customer_id=customer_id, identity_status="strong",
                primary_phone=phone, created_at=NOW, updated_at=NOW,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton", customer_id=customer_id, link_type="phone", link_value=phone,
                source_system="legacy", source_ref=f"legacy:phone:{customer_id}",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton", customer_id=customer_id, link_type="tallanto_student_id",
                link_value="student-repair", source_system="legacy",
                source_ref=f"legacy:tallanto:{customer_id}",
            )
        )
    assert len(store.list_conflicting_unique_identity_links("foton")) == 2
    record = TimelineSourceRecord(
        source_system="tallanto_snapshot", source_ref="snapshot#current",
        payload={
            "tallanto_id": "student-repair", "display_name": "Ученик",
            "primary_phone": "+79161112233", "snapshot_at": NOW.isoformat(),
        },
        observed_at=NOW,
    )
    service = TimelineImportService(store)
    unrelated = TimelineSourceRecord(
        source_system="tallanto_snapshot", source_ref="snapshot#other",
        payload={
            "tallanto_id": "student-other", "display_name": "Ученик",
            "primary_phone": "+79160000001", "snapshot_at": NOW.isoformat(),
        },
        observed_at=NOW,
    )
    service.import_records(
        (unrelated,), normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton", source_ref="unrelated", idempotency_key="unrelated", actor="test",
    )
    assert len(store.list_conflicting_unique_identity_links("foton")) == 2

    first = service.import_records(
        (record,), normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton", source_ref="repair-1", idempotency_key="repair-1", actor="test",
    )
    state_after_first = tuple(
        (row["link_id"], row["customer_id"], row["match_class"])
        for row in store.list_identity_links("foton", link_type="tallanto_student_id", link_value="student-repair")
    )
    second = service.import_records(
        (record,), normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton", source_ref="repair-2", idempotency_key="repair-2", actor="test",
    )
    state_after_second = tuple(
        (row["link_id"], row["customer_id"], row["match_class"])
        for row in store.list_identity_links("foton", link_type="tallanto_student_id", link_value="student-repair")
    )

    assert first.normalized_counts["conflicts"] == 1
    assert state_after_second == state_after_first
    assert sum(match_class == "strong_unique" for _, _, match_class in state_after_first) == 1
    assert sum(match_class == "ambiguous" for _, _, match_class in state_after_first) == 2
    assert {customer_id for _, customer_id, match_class in state_after_first if match_class == "strong_unique"} == {
        "customer:first"
    }
    assert store.list_conflicting_unique_identity_links("foton") == ()
    conflicts = store.list_conflicts("foton", statuses=("resolved",))["items"]
    assert [(row["conflict_type"], row["status"]) for row in conflicts] == [
        ("tallanto_identity_conflict", "resolved")
    ]
    assert conflicts[0]["resolved_at"] is not None
    assert store.summary()["counts"]["customer_identities"] == 3
    store.close()


def test_unrelated_import_does_not_change_stale_unique_exact_owners(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    for customer_id in ("customer:first", "customer:second"):
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id=customer_id, identity_status="strong")
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton", customer_id=customer_id, link_type="telegram_user_id",
                link_value="777", source_system="legacy", source_ref=f"legacy:{customer_id}",
            )
        )
    assert len(store.list_conflicting_unique_identity_links("foton")) == 2

    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="test", source_ref="other", payload={}),
        customers=(CustomerIdentity(tenant_id="foton", customer_id="customer:other", identity_status="partial"),),
    )
    resolved = resolve_customer_identity_batches((batch,), store=store).batches[0]
    for link in resolved.identity_links:
        store.upsert_identity_link(link)

    links = store.list_identity_links("foton", link_type="telegram_user_id", link_value="777")
    assert {link["match_class"] for link in links} == {"strong_unique"}
    assert len(store.list_conflicting_unique_identity_links("foton")) == 2
    store.close()


def test_fresh_unique_owner_conflicts_with_different_historical_owner(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    for customer_id in ("customer:current", "customer:stale"):
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id=customer_id, identity_status="strong")
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton", customer_id=customer_id, link_type="max_user_id",
                link_value="888", source_system="legacy", source_ref=f"legacy:{customer_id}",
            )
        )
    current_link = IdentityLink(
        tenant_id="foton", customer_id="customer:current", link_type="max_user_id",
        link_value="888", source_system="wappi_max", source_ref="chat:888",
    )
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="wappi_max", source_ref="chat:888", payload={}),
        customers=(CustomerIdentity(tenant_id="foton", customer_id="customer:current", identity_status="strong"),),
        identity_links=(current_link,),
    )

    resolved = resolve_customer_identity_batches((batch,), store=store).batches[0]
    for link in resolved.identity_links:
        store.upsert_identity_link(link)

    links = store.list_identity_links("foton", link_type="max_user_id", link_value="888")
    assert {link["match_class"] for link in links} == {"ambiguous"}
    assert {link["customer_id"] for link in links} == {None}
    assert store.list_conflicting_unique_identity_links("foton") == ()
    assert resolved.conflicts[0]["conflict_type"] == "ambiguous_identity"
    store.close()


def test_two_current_unique_owners_are_both_downgraded() -> None:
    batches = tuple(
        TimelineNormalizedBatch(
            source_record=TimelineSourceRecord(source_system="wappi_telegram", source_ref=customer_id, payload={}),
            customers=(CustomerIdentity(tenant_id="foton", customer_id=customer_id, identity_status="partial"),),
            identity_links=(
                IdentityLink(
                    tenant_id="foton", customer_id=customer_id, link_type="telegram_user_id",
                    link_value="999", source_system="wappi_telegram", source_ref=customer_id,
                ),
            ),
        )
        for customer_id in ("customer:first", "customer:second")
    )

    resolved = resolve_customer_identity_batches(batches)

    links = [link for batch in resolved.batches for link in batch.identity_links]
    assert {link.match_class for link in links} == {IdentityMatchClass.AMBIGUOUS}
    assert {link.customer_id for link in links} == {None}
    assert [conflict["conflict_type"] for conflict in resolved.batches[0].conflicts] == ["ambiguous_identity"]


def test_two_current_unique_owners_quarantine_events_and_drop_duplicate_opportunities(tmp_path: Path) -> None:
    batches = tuple(
        TimelineNormalizedBatch(
            source_record=TimelineSourceRecord(source_system="tallanto_snapshot", source_ref=customer_id, payload={}),
            customers=(CustomerIdentity(tenant_id="foton", customer_id=customer_id, identity_status="strong"),),
            identity_links=(
                IdentityLink(
                    tenant_id="foton", customer_id=customer_id, link_type="tallanto_student_id",
                    link_value="dup-student", source_system="tallanto_snapshot", source_ref=customer_id,
                ),
                IdentityLink(
                    tenant_id="foton", customer_id=customer_id, link_type="phone",
                    link_value=f"+7999000000{1 if customer_id.endswith('first') else 2}",
                    source_system="tallanto_snapshot", source_ref=customer_id,
                ),
            ),
            opportunities=(CustomerOpportunity(
                tenant_id="foton", customer_id=customer_id, opportunity_type=OpportunityType.TALLANTO_COURSE,
                source_system="tallanto_snapshot", source_id="student:dup-student", title="Курс",
            ),),
            events=(TimelineEvent(
                tenant_id="foton", customer_id=customer_id, event_type=TimelineEventType.TALLANTO_STUDENT_SNAPSHOT,
                event_at=NOW, source_system="tallanto_snapshot", source_id=customer_id,
                direction=TimelineDirection.SYSTEM, match_status=IdentityMatchClass.STRONG_UNIQUE,
            ),),
        )
        for customer_id in ("customer:first", "customer:second")
    )

    resolved = resolve_customer_identity_batches(batches)

    assert all(batch.opportunities == () for batch in resolved.batches)
    assert all(batch.events[0].customer_id is None for batch in resolved.batches)
    assert all(batch.events[0].match_status == IdentityMatchClass.AMBIGUOUS for batch in resolved.batches)
    assert all(batch.retired_opportunity_ids for batch in resolved.batches)
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    service = TimelineImportService(store)
    for batch in resolved.batches:
        service._apply_batch(batch, actor="test", ingestion_run_id=None)  # noqa: SLF001
    assert store.summary()["counts"]["customer_opportunities"] == 0
    assert {row["customer_id"] for row in store.list_identity_links("foton", link_type="tallanto_student_id")} == {None}
    store.close()


def test_existing_opportunity_id_is_reused_when_exact_owner_changes(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    store.upsert_customer(CustomerIdentity(tenant_id="foton", customer_id="customer:a", identity_status="strong"))
    existing = CustomerOpportunity(
        tenant_id="foton",
        customer_id="customer:a",
        opportunity_type=OpportunityType.TALLANTO_COURSE,
        source_system="tallanto_snapshot",
        source_id="student:1",
        title="Курс",
    )
    store.upsert_opportunity(existing)
    store.upsert_signal(DerivedSignal(
        tenant_id="foton", customer_id="customer:a", opportunity_id=existing.opportunity_id,
        signal_type="sales", severity="medium", evidence_text="Старая семья",
    ))
    store.upsert_bot_context_chunk(BotContextChunk(
        tenant_id="foton", customer_id="customer:a", opportunity_id=existing.opportunity_id,
        chunk_type="manager_only", text="Старая память", source_system="test_source",
        source_ref="test:old-owner", allowed_for_bot=False, requires_manager_review=True,
    ))
    incoming = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="tallanto_snapshot", source_ref="student:1", payload={}),
        customers=(CustomerIdentity(tenant_id="foton", customer_id="customer:b", identity_status="strong"),),
        opportunities=(replace(existing, customer_id="customer:b", opportunity_id=None),),
    )

    batch = resolve_customer_identity_batches((incoming,), store=store).batches[0]
    TimelineImportService(store)._apply_batch(batch, actor="test", ingestion_run_id=None)  # noqa: SLF001

    row = store.get_opportunity_by_source(
        "foton", source_system="tallanto_snapshot", source_id="student:1",
        opportunity_type=OpportunityType.TALLANTO_COURSE.value,
    )
    assert row is not None
    assert row["opportunity_id"] == existing.opportunity_id
    assert row["customer_id"] == "customer:b"
    signal = store._con.execute("SELECT customer_id,status,opportunity_id FROM derived_signals").fetchone()
    chunk = store._con.execute(
        "SELECT customer_id,superseded_by,opportunity_id FROM bot_context_chunks"
    ).fetchone()
    assert tuple(signal) == ("customer:a", "stale", None)
    assert chunk[0] == "customer:a" and chunk[1] and chunk[2] is None
    repeat = resolve_customer_identity_batches((incoming,), store=store).batches[0]
    TimelineImportService(store)._apply_batch(repeat, actor="test", ingestion_run_id=None)  # noqa: SLF001
    assert store._con.execute("SELECT COUNT(*) FROM derived_signals").fetchone()[0] == 1
    assert store._con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0] == 1
    store.close()


def test_manual_unique_owner_wins_over_automatic_owner() -> None:
    manual = IdentityLink(
        tenant_id="foton", customer_id="customer:manual", link_type="telegram_user_id",
        link_value="1000", source_system="manual_review", source_ref="manual:1000",
        match_class=IdentityMatchClass.MANUAL,
    )
    automatic = IdentityLink(
        tenant_id="foton", customer_id="customer:auto", link_type="telegram_user_id",
        link_value="1000", source_system="amocrm_snapshot", source_ref="amo:1000",
    )
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="test", source_ref="manual-priority", payload={}),
        customers=(
            CustomerIdentity(tenant_id="foton", customer_id="customer:manual", identity_status="strong"),
            CustomerIdentity(tenant_id="foton", customer_id="customer:auto", identity_status="strong"),
        ),
        identity_links=(manual, automatic),
    )

    links = resolve_customer_identity_batches((batch,)).batches[0].identity_links

    assert next(link for link in links if link.source_system == "manual_review").match_class == IdentityMatchClass.MANUAL
    automatic_result = next(link for link in links if link.source_system == "amocrm_snapshot")
    assert automatic_result.match_class == IdentityMatchClass.STRONG_UNIQUE
    assert automatic_result.customer_id == "customer:manual"


def test_historical_manual_unique_owner_wins_over_fresh_automatic_owner(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    store.upsert_customer(CustomerIdentity(tenant_id="foton", customer_id="customer:manual", identity_status="strong"))
    store.upsert_identity_link(IdentityLink(
        tenant_id="foton", customer_id="customer:manual", link_type="telegram_user_id",
        link_value="1002", source_system="manual_review", source_ref="manual:1002",
        match_class=IdentityMatchClass.MANUAL,
    ))
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="amocrm_snapshot", source_ref="amo:1002", payload={}),
        customers=(CustomerIdentity(tenant_id="foton", customer_id="customer:auto", identity_status="strong"),),
        identity_links=(IdentityLink(
            tenant_id="foton", customer_id="customer:auto", link_type="telegram_user_id",
            link_value="1002", source_system="amocrm_snapshot", source_ref="amo:1002",
        ),),
        events=(TimelineEvent(
            tenant_id="foton", customer_id="customer:auto", event_type=TimelineEventType.AMO_CONTACT_SNAPSHOT,
            event_at=NOW, source_system="amocrm_snapshot", source_id="1002",
            direction=TimelineDirection.SYSTEM, match_status=IdentityMatchClass.STRONG_UNIQUE,
        ),),
    )

    resolved = resolve_customer_identity_batches((batch,), store=store).batches[0]

    manual = store.list_identity_links("foton", link_type="telegram_user_id", link_value="1002")[0]
    automatic = next(link for link in resolved.identity_links if link.source_system == "amocrm_snapshot")
    assert manual["customer_id"] == "customer:manual" and manual["match_class"] == "manual"
    assert automatic.customer_id == "customer:manual"
    assert automatic.match_class == IdentityMatchClass.STRONG_UNIQUE
    assert resolved.events[0].customer_id == "customer:manual"
    assert {customer.customer_id for customer in resolved.customers} == {"customer:manual"}
    store.close()


@pytest.mark.parametrize("order", [("customer:first", "customer:second"), ("customer:second", "customer:first")])
def test_same_unique_source_ref_conflict_is_order_independent(order: tuple[str, str]) -> None:
    links = tuple(
        IdentityLink(
            tenant_id="foton", customer_id=customer_id, link_type="telegram_user_id",
            link_value="1003", source_system="amocrm_snapshot", source_ref="same:1003",
        )
        for customer_id in order
    )
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="amocrm_snapshot", source_ref="same:1003", payload={}),
        customers=tuple(
            CustomerIdentity(tenant_id="foton", customer_id=customer_id, identity_status="strong")
            for customer_id in order
        ),
        identity_links=links,
    )

    resolved = resolve_customer_identity_batches((batch,)).batches[0]

    assert {link.customer_id for link in resolved.identity_links} == {None}
    assert {link.match_class for link in resolved.identity_links} == {IdentityMatchClass.AMBIGUOUS}
    assert resolved.conflicts[0]["entity_refs"] == (
        "telegram_user_id:1003", "customer:first", "customer:second",
    )


@pytest.mark.parametrize("order", [("customer:first", "customer:second"), ("customer:second", "customer:first")])
def test_same_unique_source_ref_conflict_across_sequential_imports_is_ambiguous(
    tmp_path: Path,
    order: tuple[str, str],
) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    for customer_id, phone in zip(order, ("+79160000001", "+79160000002")):
        store.upsert_customer(CustomerIdentity(
            tenant_id="foton", customer_id=customer_id, identity_status="strong", primary_phone=phone,
        ))
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id=customer_id, link_type="phone", link_value=phone,
            source_system="verified_contact", source_ref=f"phone:{customer_id}",
        ))
    store.upsert_identity_link(IdentityLink(
        tenant_id="foton", customer_id=order[0], link_type="telegram_user_id", link_value="sequential",
        source_system="amocrm_snapshot", source_ref="same:sequential",
    ))
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="amocrm_snapshot", source_ref="same:sequential", payload={}),
        customers=(CustomerIdentity(tenant_id="foton", customer_id=order[1], identity_status="strong"),),
        identity_links=(IdentityLink(
            tenant_id="foton", customer_id=order[1], link_type="telegram_user_id", link_value="sequential",
            source_system="amocrm_snapshot", source_ref="same:sequential",
        ),),
    )

    resolved = resolve_customer_identity_batches((batch,), store=store).batches[0]

    assert resolved.identity_links[0].customer_id is None
    assert resolved.identity_links[0].match_class == IdentityMatchClass.AMBIGUOUS
    assert resolved.conflicts[0]["conflict_type"] == "ambiguous_identity"
    store.close()


def test_confirmed_unique_owner_resolves_only_matching_identity_conflict(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=StepClock())
    for value in ("resolved", "neighbor"):
        store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=(f"telegram_user_id:{value}", "customer:a", "customer:b"),
        )
    record = TimelineSourceRecord(source_system="amocrm_snapshot", source_ref="resolved", payload={})
    batch = TimelineNormalizedBatch(
        source_record=record,
        customers=(CustomerIdentity(tenant_id="foton", customer_id="customer:a", identity_status="strong"),),
        identity_links=(IdentityLink(
            tenant_id="foton", customer_id="customer:a", link_type="telegram_user_id", link_value="resolved",
            source_system="amocrm_snapshot", source_ref="resolved",
        ),),
    )
    normalizer = type(
        "IdentityNormalizer",
        (),
        {"source_system": "amocrm_snapshot", "normalize": lambda self, unused: batch},
    )()

    TimelineImportService(store).import_records(
        (record,), normalizer=normalizer, tenant_id="foton", source_ref="resolved", actor="test",
    )

    resolved = store.list_conflicts("foton", statuses=("resolved",))["items"]
    open_items = store.list_conflicts("foton", statuses=("open",))["items"]
    assert [item["entity_refs"][0] for item in resolved] == ["telegram_user_id:resolved"]
    assert [item["entity_refs"][0] for item in open_items] == ["telegram_user_id:neighbor"]
    store.close()


@pytest.mark.parametrize("order", [("amo", "wappi"), ("wappi", "amo")])
def test_customer_primary_phone_uses_source_authority_not_batch_order(order: tuple[str, str]) -> None:
    observations = {
        "amo": CustomerIdentity(
            tenant_id="foton", customer_id="customer:one", identity_status="strong",
            primary_phone="+79160000001", source_ref="amo:1",
            summary={"source_system": "amocrm_snapshot"}, created_at=NOW, updated_at=NOW,
        ),
        "wappi": CustomerIdentity(
            tenant_id="foton", customer_id="customer:one", identity_status="partial",
            primary_phone="+79160000002", source_ref="wappi:1",
            summary={"source_system": "wappi_telegram"}, created_at=NOW, updated_at=NOW,
        ),
    }
    batches = tuple(
        TimelineNormalizedBatch(
            source_record=TimelineSourceRecord(source_system=name, source_ref=name, payload={}),
            customers=(observations[name],),
        )
        for name in order
    )

    customers = [customer for batch in resolve_customer_identity_batches(batches).batches for customer in batch.customers]

    assert {customer.primary_phone for customer in customers} == {"+79160000001"}


def test_two_temporary_owners_merged_by_phone_keep_one_exact_unique_owner() -> None:
    phone = "+79160000001"
    batches = tuple(
        TimelineNormalizedBatch(
            source_record=TimelineSourceRecord(source_system="test", source_ref=customer_id, payload={}),
            customers=(
                CustomerIdentity(
                    tenant_id="foton", customer_id=customer_id, identity_status="strong", primary_phone=phone,
                ),
            ),
            identity_links=(
                IdentityLink(
                    tenant_id="foton", customer_id=customer_id, link_type="phone", link_value=phone,
                    source_system="test", source_ref=f"phone:{customer_id}",
                ),
                IdentityLink(
                    tenant_id="foton", customer_id=customer_id, link_type="telegram_user_id",
                    link_value="1001", source_system="test", source_ref=f"telegram:{customer_id}",
                ),
            ),
        )
        for customer_id in ("customer:first", "customer:second")
    )

    links = [link for batch in resolve_customer_identity_batches(batches).batches for link in batch.identity_links]
    telegram_links = [link for link in links if link.link_type.value == "telegram_user_id"]

    assert len({link.customer_id for link in telegram_links}) == 1
    assert {link.match_class for link in telegram_links} == {IdentityMatchClass.STRONG_UNIQUE}


def test_fresh_tallanto_card_keeps_exact_student_owner_when_amo_contact_exists(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(
        tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock()
    )
    service = TimelineImportService(store)
    tallanto = TallantoSnapshotNormalizer(tenant_id="foton")
    amo = AmoSnapshotNormalizer(tenant_id="foton")
    student = TimelineSourceRecord(
        source_system="tallanto_snapshot",
        source_ref="student:2001",
        payload={
            "tallanto_id": "2001",
            "display_name": "Анна Иванова",
            "primary_phone": "+79161112233",
            "snapshot_at": NOW.isoformat(),
        },
        observed_at=NOW,
    )
    service.import_records(
        (student,), normalizer=tallanto, tenant_id="foton",
        source_ref="student-first", idempotency_key="student-first", actor="test",
    )
    student_owner = store.list_identity_links(
        "foton", link_type="tallanto_student_id", link_value="2001"
    )[0]["customer_id"]
    service.import_records(
        (
            TimelineSourceRecord(
                source_system="amocrm_snapshot",
                source_ref="contact:1001",
                payload={
                    "entity_id": "1001",
                    "entity_type": "contact",
                    "name": "Ирина Иванова",
                    "phone": "+79169998877",
                    "updated_at": NOW.isoformat(),
                },
                observed_at=NOW,
            ),
        ),
        normalizer=amo,
        tenant_id="foton",
        source_ref="amo-first",
        idempotency_key="amo-first",
        actor="test",
    )
    amo_owner = store.list_identity_links(
        "foton", link_type="amo_contact_id", link_value="1001"
    )[0]["customer_id"]
    assert amo_owner != student_owner
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=student_owner,
            link_type="amo_contact_id",
            link_value="1001",
            source_system="tallanto_snapshot",
            source_ref="legacy:tallanto:student:2001:amo",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
        )
    )

    current = replace(
        student,
        source_ref="student:2001:current",
        payload={**student.payload, "amo_contact_id": "1001"},
    )
    first = service.import_records(
        (current,), normalizer=tallanto, tenant_id="foton",
        source_ref="student-current", idempotency_key="student-current", actor="test",
    )
    mapping_count = store.summary()["counts"]["customer_id_mappings"]
    second = service.import_records(
        (current,), normalizer=tallanto, tenant_id="foton",
        source_ref="student-repeat", idempotency_key="student-repeat", actor="test",
    )

    assert first.validation_ok is True
    assert second.validation_ok is True
    assert store.list_identity_links(
        "foton", link_type="tallanto_student_id", link_value="2001"
    )[0]["customer_id"] == student_owner
    amo_links = store.list_identity_links(
        "foton", link_type="amo_contact_id", link_value="1001"
    )
    assert {
        (row["customer_id"], row["match_class"])
        for row in amo_links
    } == {
        (amo_owner, "strong_unique"),
        (student_owner, "ambiguous"),
    }
    assert next(row for row in amo_links if row["customer_id"] == student_owner)["evidence"] == {
        "relationship": "family_amo_contact"
    }
    assert {
        row["match_class"]
        for row in amo_links
        if row["source_system"] == "tallanto_snapshot"
    } == {"ambiguous"}
    assert all(
        row["evidence"].get("relationship") == "family_amo_contact"
        for row in amo_links
        if row["source_system"] == "tallanto_snapshot"
    )
    assert store.summary()["counts"]["customer_id_mappings"] == mapping_count == 0
    store.close()


@pytest.mark.parametrize("student_order", (("2001", "2002"), ("2002", "2001")))
def test_shared_tallanto_amo_contact_never_merges_children(
    tmp_path: Path,
    student_order: tuple[str, str],
) -> None:
    store = CustomerTimelineSQLiteStore(
        tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock()
    )
    service = TimelineImportService(store)
    service.import_records(
        (
            TimelineSourceRecord(
                source_system="amocrm_snapshot",
                source_ref="contact:1001",
                payload={
                    "entity_id": "1001",
                    "entity_type": "contact",
                    "name": "Ирина Иванова",
                    "phone": "+79169998877",
                    "updated_at": NOW.isoformat(),
                },
                observed_at=NOW,
            ),
        ),
        normalizer=AmoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="amo-parent",
        idempotency_key="amo-parent",
        actor="test",
    )

    records = {
        student_id: TimelineSourceRecord(
            source_system="tallanto_snapshot",
            source_ref=f"student:{student_id}",
            payload={
                "tallanto_id": student_id,
                "display_name": f"Ученик {student_id}",
                "primary_phone": f"+7916000{student_id}",
                "amo_contact_id": "1001",
                "snapshot_at": NOW.isoformat(),
            },
            observed_at=NOW,
        )
        for student_id in student_order
    }
    normalizer = TallantoSnapshotNormalizer(tenant_id="foton")
    for pass_no in (1, 2):
        for student_id in student_order:
            service.import_records(
                (records[student_id],),
                normalizer=normalizer,
                tenant_id="foton",
                source_ref=f"student-{student_id}-pass-{pass_no}",
                idempotency_key=f"student-{student_id}-pass-{pass_no}",
                actor="test",
            )

    tallanto_links = [
        store.list_identity_links("foton", link_type="tallanto_student_id", link_value=student_id)[0]
        for student_id in student_order
    ]
    amo_links = store.list_identity_links("foton", link_type="amo_contact_id", link_value="1001")
    assert len({row["customer_id"] for row in tallanto_links}) == 2
    assert {row["match_class"] for row in tallanto_links} == {"strong_unique"}
    assert sum(row["match_class"] == "strong_unique" for row in amo_links) == 1
    assert sum(row["match_class"] == "ambiguous" for row in amo_links) == 2
    assert store.list_customer_id_mappings("foton") == ()
    store.close()


def test_tallanto_conflict_preserves_event_and_later_exact_contact_resolves_it(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=FixedClock())
    for customer_id in ("customer:first", "customer:wrong"):
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton", customer_id=customer_id, identity_status="strong",
                created_at=NOW, updated_at=NOW,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton", customer_id=customer_id, link_type="tallanto_student_id",
                link_value="student-retry", source_system="legacy",
                source_ref=f"legacy:tallanto:{customer_id}",
            )
        )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton", customer_id="customer:first", link_type="phone",
            link_value="+79161112233", source_system="legacy", source_ref="legacy:phone:first",
        )
    )
    service = TimelineImportService(store)
    normalizer = TallantoSnapshotNormalizer(tenant_id="foton")

    unresolved = TimelineSourceRecord(
        source_system="tallanto_snapshot", source_ref="snapshot#retry",
        payload={"tallanto_id": "student-retry", "display_name": "Ученик", "snapshot_at": NOW.isoformat()},
        observed_at=NOW,
    )
    first = service.import_records(
        (unresolved,), normalizer=normalizer, tenant_id="foton",
        source_ref="retry-1", idempotency_key="retry-1", actor="test",
    )
    assert first.normalized_counts["events"] == 1
    assert store.list_open_tallanto_identity_conflict_values("foton") == ("student-retry",)
    assert not [
        row for row in store.list_identity_links(
            "foton", link_type="tallanto_student_id", link_value="student-retry"
        )
        if row["match_class"] in {"strong_unique", "manual"}
    ]
    with sqlite3.connect(db_path) as con:
        event_before = con.execute(
            "SELECT event_id, customer_id, match_status FROM timeline_events WHERE source_id = 'student-retry'"
        ).fetchone()
    assert event_before is not None and event_before[2] == "ambiguous"

    resolved = TimelineSourceRecord(
        source_system="tallanto_snapshot", source_ref="snapshot#retry",
        payload={
            "tallanto_id": "student-retry", "display_name": "Ученик",
            "primary_phone": "+79161112233", "snapshot_at": NOW.isoformat(),
        },
        observed_at=NOW,
    )
    second = service.import_records(
        (resolved,), normalizer=normalizer, tenant_id="foton",
        source_ref="retry-2", idempotency_key="retry-2", actor="test",
    )
    links = store.list_identity_links("foton", link_type="tallanto_student_id", link_value="student-retry")
    with sqlite3.connect(db_path) as con:
        events_after = con.execute(
            "SELECT event_id, customer_id, match_status FROM timeline_events WHERE source_id = 'student-retry'"
        ).fetchall()

    assert second.normalized_counts["conflicts"] == 1
    assert store.list_open_tallanto_identity_conflict_values("foton") == ()
    assert sum(row["match_class"] == "strong_unique" for row in links) == 1
    assert [row[0] for row in events_after] == [event_before[0]]
    assert events_after[0][1:] == ("customer:first", "strong_unique")
    store.close()


def test_current_tallanto_owner_resolves_old_conflict_despite_stale_ambiguous_contact(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    for customer_id in ("customer:first", "customer:stale"):
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton", customer_id=customer_id, identity_status="strong",
                created_at=NOW, updated_at=NOW,
            )
        )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton", customer_id="customer:first", link_type="tallanto_student_id",
            link_value="student-current", source_system="tallanto_snapshot",
            source_ref="tallanto:student:student-current",
        )
    )
    for customer_id, match_class in (
        ("customer:first", IdentityMatchClass.STRONG_UNIQUE),
        ("customer:stale", IdentityMatchClass.AMBIGUOUS),
    ):
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton", customer_id=customer_id, link_type="phone",
                link_value="+79161112233", source_system="legacy",
                source_ref=f"legacy:phone:{customer_id}", match_class=match_class,
            )
        )
    store.record_conflict(
        "foton", conflict_type="tallanto_identity_conflict",
        entity_refs=("tallanto_student_id:student-current",), status="open",
    )

    TimelineImportService(store).import_records(
        (
            TimelineSourceRecord(
                source_system="tallanto_snapshot", source_ref="snapshot#current",
                payload={
                    "tallanto_id": "student-current", "display_name": "Ученик",
                    "primary_phone": "+79161112233", "snapshot_at": NOW.isoformat(),
                },
                observed_at=NOW,
            ),
        ),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton", source_ref="current", idempotency_key="current", actor="test",
    )

    assert store.list_open_tallanto_identity_conflict_values("foton") == ()
    store.close()


def test_tallanto_does_not_merge_phone_and_email_owned_by_different_customers(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=FixedClock())
    for customer_id, link_type, link_value in (
        ("customer:phone", "phone", "+79161112233"),
        ("customer:email", "email", "parent@example.com"),
    ):
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status="strong",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type=link_type,
                link_value=link_value,
                source_system="test",
                source_ref=f"test:{customer_id}",
            )
        )
    record = TimelineSourceRecord(
        source_system="tallanto_snapshot",
        source_ref="snapshot#cross-identity",
        payload={
            "tallanto_id": "student-new",
            "primary_phone": "+7 916 111-22-33",
            "primary_email": "parent@example.com",
            "snapshot_at": NOW.isoformat(),
        },
        observed_at=NOW,
    )

    report = TimelineImportService(store).import_records(
        (record,),
        normalizer=TallantoSnapshotNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="snapshot-cross-identity",
        idempotency_key="snapshot-cross-identity",
        actor="test",
    )

    assert report.normalized_counts["conflicts"] == 1
    links = store.list_identity_links("foton", link_type="tallanto_student_id", link_value="student-new")
    assert len(links) == 1 and links[0]["match_class"] == "ambiguous"
    assert links[0]["customer_id"] is None
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT match_status FROM timeline_events WHERE source_id = 'student-new'"
        ).fetchone()[0] == "ambiguous"
    assert store.list_open_tallanto_identity_conflict_values("foton") == ("student-new",)
    assert store.summary()["counts"]["customer_identities"] == 2
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
    assert set(report.changed_customer_ids) == {
        *(item["old_customer_id"] for item in mappings),
        *(item["new_customer_id"] for item in mappings),
    }
    assert len({item["customer_id"] for item in links}) == 1
    store.close()


def test_parent_and_single_tallanto_child_keep_separate_customer_ids_on_shared_phone() -> None:
    shared_phone = "+79162223344"
    child = CustomerIdentity(
        tenant_id="foton", customer_id="customer:child", identity_status="strong",
        display_name="Анна Иванова", primary_phone=shared_phone,
    )
    parent = CustomerIdentity(
        tenant_id="foton", customer_id="customer:parent", identity_status="strong",
        display_name="Ирина Иванова", primary_phone=shared_phone,
    )
    child_batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="tallanto_snapshot", source_ref="student:1", payload={}),
        customers=(child,),
        identity_links=(
            IdentityLink(
                tenant_id="foton", customer_id=child.customer_id, link_type="phone",
                link_value=shared_phone, source_system="tallanto_snapshot", source_ref="student:1",
            ),
            IdentityLink(
                tenant_id="foton", customer_id=child.customer_id, link_type="tallanto_student_id",
                link_value="student-1", source_system="tallanto_snapshot", source_ref="student:1",
            ),
        ),
    )
    parent_batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="amocrm_snapshot", source_ref="contact:1", payload={}),
        customers=(parent,),
        identity_links=(
            IdentityLink(
                tenant_id="foton", customer_id=parent.customer_id, link_type="phone",
                link_value=shared_phone, source_system="amocrm_snapshot", source_ref="contact:1",
            ),
            IdentityLink(
                tenant_id="foton", customer_id=parent.customer_id, link_type="amo_contact_id",
                link_value="1001", source_system="amocrm_snapshot", source_ref="contact:1",
            ),
        ),
    )

    result = resolve_customer_identity_batches((child_batch, parent_batch))

    assert result.mappings == ()
    phone_links = [link for batch in result.batches for link in batch.identity_links if link.link_type.value == "phone"]
    assert {link.customer_id for link in phone_links} == {"customer:child", "customer:parent"}
    assert {link.match_class for link in phone_links} == {IdentityMatchClass.AMBIGUOUS}


def test_existing_strong_customer_is_not_downgraded_by_same_customer_amo_lead(tmp_path: Path) -> None:
    customer_id = "customer:known-parent"
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id=customer_id,
            identity_status=IdentityStatus.STRONG,
            primary_phone="+79162223344",
            primary_email="parent@example.com",
            source_ref="amo:contact:1001",
        )
    )
    weak_lead = CustomerIdentity(
        tenant_id="foton",
        customer_id=customer_id,
        identity_status=IdentityStatus.PARTIAL,
        source_ref="amo:lead:5001",
    )
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="amocrm_snapshot", source_ref="lead:5001", payload={}),
        customers=(weak_lead,),
    )

    resolved = resolve_customer_identity_batches((batch,), store=store).batches[0].customers[0]

    assert resolved.identity_status == IdentityStatus.STRONG
    assert resolved.primary_phone == "+79162223344"
    assert resolved.primary_email == "parent@example.com"
    store.close()


def test_existing_strong_customer_stays_strong_after_repeat_wappi_observation(tmp_path: Path) -> None:
    customer_id = "customer:known-parent"
    db_path = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path, clock=FixedClock())
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id=customer_id,
            identity_status=IdentityStatus.STRONG,
            primary_phone="+79162223344",
            source_ref="amo:contact:1001",
        )
    )
    wappi_customer = CustomerIdentity(
        tenant_id="foton",
        customer_id=customer_id,
        identity_status=IdentityStatus.PARTIAL,
        source_ref="wappi:telegram:42",
    )
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="wappi_telegram", source_ref="chat:42", payload={}),
        customers=(wappi_customer,),
        identity_links=(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type="telegram_user_id",
                link_value="42",
                source_system="wappi_telegram",
                source_ref="chat:42",
                match_class=IdentityMatchClass.INFERRED,
            ),
        ),
    )

    first = resolve_customer_identity_batches((batch,), store=store).batches[0].customers[0]
    store.upsert_customer(first)
    second = resolve_customer_identity_batches((batch,), store=store).batches[0].customers[0]

    assert first.identity_status == second.identity_status == IdentityStatus.STRONG
    assert first.primary_phone == second.primary_phone == "+79162223344"
    store.close()


def test_same_customer_ambiguous_observation_still_blocks_identity(tmp_path: Path) -> None:
    customer_id = "customer:conflicted"
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id=customer_id,
            identity_status=IdentityStatus.STRONG,
            primary_phone="+79162223344",
            source_ref="amo:contact:1001",
        )
    )
    conflict = CustomerIdentity(
        tenant_id="foton",
        customer_id=customer_id,
        identity_status=IdentityStatus.AMBIGUOUS,
        source_ref="identity:conflict",
    )
    batch = TimelineNormalizedBatch(
        source_record=TimelineSourceRecord(source_system="identity_test", source_ref="conflict:1", payload={}),
        customers=(conflict,),
    )

    resolved = resolve_customer_identity_batches((batch,), store=store).batches[0].customers[0]

    assert resolved.identity_status == IdentityStatus.AMBIGUOUS
    assert resolved.primary_phone == "+79162223344"
    store.close()


@pytest.mark.parametrize("order", [("amo", "tallanto"), ("tallanto", "amo")])
def test_parent_child_identity_is_order_independent_and_repeat_safe(
    tmp_path: Path,
    order: tuple[str, str],
) -> None:
    shared_phone = "+79162223344"
    shared_email = "parent@example.com"
    records = {
        "amo": TimelineSourceRecord(
            source_system="amocrm_snapshot",
            source_ref="contact:1001",
            payload={
                "entity_id": "1001",
                "entity_type": "contact",
                "name": "Ирина Иванова",
                "phone": shared_phone,
                "email": shared_email,
                "updated_at": NOW.isoformat(),
            },
            observed_at=NOW,
        ),
        "tallanto": TimelineSourceRecord(
            source_system="tallanto_snapshot",
            source_ref="student:2001",
            payload={
                "entity_id": "2001",
                "name": "Анна Иванова",
                "parent_fio": "Ирина Иванова",
                "phone": shared_phone,
                "email": shared_email,
                "updated_at": NOW.isoformat(),
            },
            observed_at=NOW,
        ),
    }
    normalizers = {
        "amo": AmoSnapshotNormalizer(tenant_id="foton"),
        "tallanto": TallantoSnapshotNormalizer(tenant_id="foton"),
    }
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)
    try:
        for index, kind in enumerate(order, 1):
            service.import_records(
                (records[kind],),
                normalizer=normalizers[kind],
                tenant_id="foton",
                source_ref=f"{kind}-run-{index}",
                idempotency_key=f"{kind}-run-{index}",
                actor="test",
            )
        service.import_records(
            (records["tallanto"],),
            normalizer=normalizers["tallanto"],
            tenant_id="foton",
            source_ref="tallanto-repeat",
            idempotency_key="tallanto-repeat",
            actor="test",
        )

        assert store.summary()["counts"]["customer_identities"] == 2
        assert not [
            row
            for row in store.list_conflicts("foton", statuses=("open",))["items"]
            if row["conflict_type"] == "tallanto_identity_conflict"
        ]
        for link_type, value in (("phone", shared_phone), ("email", shared_email)):
            links = store.list_identity_links("foton", link_type=link_type, link_value=value)
            assert len({row["customer_id"] for row in links}) == 2
            assert {row["match_class"] for row in links} == {"ambiguous"}
        assert len(store.list_identity_links("foton", link_type="amo_contact_id", link_value="1001")) == 1
        assert len(store.list_identity_links("foton", link_type="tallanto_student_id", link_value="2001")) == 1
    finally:
        store.close()


def test_amo_contact_identity_survives_changed_phone_and_email(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    service = TimelineImportService(store)
    normalizer = AmoSnapshotNormalizer(tenant_id="foton")
    try:
        for index, (phone, email) in enumerate(
            (("+79161112233", "old@example.com"), ("+79164445566", "new@example.com")),
            start=1,
        ):
            service.import_records(
                (
                    TimelineSourceRecord(
                        source_system="amocrm_snapshot",
                        source_ref="contact:1001",
                        payload={
                            "entity_id": "1001",
                            "entity_type": "contact",
                            "name": "Ирина Иванова",
                            "phone": phone,
                            "email": email,
                            "updated_at": NOW.isoformat(),
                        },
                        observed_at=NOW,
                    ),
                ),
                normalizer=normalizer,
                tenant_id="foton",
                source_ref=f"amo-run-{index}",
                idempotency_key=f"amo-run-{index}",
                actor="test",
            )

        assert store.summary()["counts"]["customer_identities"] == 1
        links = store.list_identity_links("foton", link_type="amo_contact_id", link_value="1001")
        assert len({row["customer_id"] for row in links}) == 1
    finally:
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
    assert customers[0].get("display_name") != "Сделка ЕГЭ"
    amo_event = next(item for item in events if item["event_type"] == "amo_deal_stage")
    assert amo_event["subject"] == "Сделка ЕГЭ"
    assert {item["new_customer_id"] for item in mappings} == {customer_id}
    assert {item["reason"] for item in mappings} == {"phone_identity_union"}
    store.close()


def test_mango_increment_strict_identity_uses_existing_customer_without_creating_new_one(tmp_path: Path) -> None:
    existing = CustomerIdentity(
        tenant_id="foton",
        customer_id="customer:existing",
        identity_status=IdentityStatus.STRONG,
        primary_phone="+79163334455",
        source_ref="seed",
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=1,
        created_at=NOW,
        updated_at=NOW,
    )
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path, clock=FixedClock())
    store.upsert_customer(existing)
    service = TimelineImportService(store)
    record = TimelineSourceRecord(
        source_system="mango_processed_summary",
        source_ref="call#increment-1",
        payload={
            "call_id": "provider:increment-1",
            "phone": "+79163334455",
            "call_at": "2026-05-04T12:00:00+00:00",
            "summary": "Клиент уточнил стоимость.",
            "identity_authority": "existing_timeline_increment",
            "identity_resolved_by_increment": True,
            "match_class": "strong_unique",
            "customer_id": "customer:existing",
            "allowed_for_bot": False,
            "requires_manager_review": True,
        },
    )

    report = service.import_records(
        (record,),
        normalizer=MangoCallSummaryNormalizer(tenant_id="foton"),
        tenant_id="foton",
        source_ref="mango-increment",
        idempotency_key="mango-increment-1",
        actor="test",
    )

    events = store.list_events_by_customer("foton", "customer:existing", limit=10)["items"]
    chunks = store.search_timeline("foton", "стоимость", scopes=("bot_context",), mode="fallback", limit=10)["items"]
    assert report.normalized_counts["customers"] == 0
    assert store.summary()["counts"]["customer_identities"] == 1
    assert events[0]["source_system"] == "mango_processed_summary"
    assert events[0]["match_status"] == "strong_unique"
    chunk_record = chunks[0]["record"]
    assert chunk_record["allowed_for_bot"] is False
    assert chunk_record["requires_manager_review"] is True
    store.close()


def test_mango_increment_ambiguous_does_not_attach_or_create_customer() -> None:
    batch = MangoCallSummaryNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mango_processed_summary",
            source_ref="call#ambiguous",
            payload={
                "call_id": "provider:ambiguous",
                "phone": "+79163334455",
                "call_at": "2026-05-04T12:00:00+00:00",
                "summary": "Клиент уточнил стоимость.",
                "identity_authority": "existing_timeline_increment",
                "identity_resolved_by_increment": True,
                "match_class": "ambiguous",
                "identity_resolution_reason": "multiple_existing_customers",
                "allowed_for_bot": False,
            },
        )
    )

    assert batch.customers == ()
    assert batch.identity_links == ()
    assert batch.events[0].customer_id is None
    assert batch.events[0].match_status.value == "ambiguous"
    assert batch.bot_context_chunks == ()
    assert batch.conflicts[0]["conflict_type"] == "pending_attribution"


def test_mango_increment_non_conversation_has_no_summary_chunk_or_signal() -> None:
    batch = MangoCallSummaryNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mango_processed_summary",
            source_ref="call#non-conversation",
            payload={
                "call_id": "provider:non-conversation",
                "phone": "+79163334455",
                "call_at": "2026-05-04T12:00:00+00:00",
                "summary": "Автоответчик.",
                "call_type": "non_conversation",
                "identity_authority": "existing_timeline_increment",
                "identity_resolved_by_increment": True,
                "match_class": "strong_unique",
                "customer_id": "customer:existing",
                "next_step": "Перезвонить",
            },
        )
    )

    assert batch.events[0].summary is None
    assert batch.bot_context_chunks == ()
    assert batch.signals == ()


def test_mango_increment_explicit_non_contentful_overrides_sales_call_type() -> None:
    batch = MangoCallSummaryNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mango_processed_summary",
            source_ref="call#explicit-non-contentful",
            payload={
                "call_id": "provider:explicit-non-contentful",
                "call_at": "2026-05-04T12:00:00+00:00",
                "summary": "Автоответчик.",
                "call_type": "sales_call",
                "contentful": "Нет",
                "customer_id": "customer:existing",
                "match_class": "strong_unique",
            },
        )
    )

    assert batch.events[0].summary is None
    assert batch.bot_context_chunks == ()
    assert batch.signals == ()


def test_non_contentful_call_predicate_keeps_real_technical_service_and_unknown_calls() -> None:
    for call_type in ("technical_call", "service_call", "unknown", ""):
        assert not is_non_contentful_call_record({"record": {"call_type": call_type}})
    assert is_non_contentful_call_record({"record": {"call_type": "non_conversation"}})
    assert is_non_contentful_call_record({"record": {"contentful": "Нет", "call_type": "sales_call"}})


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
    assert next(link for link in batch.identity_links if link.link_type.value == "email").match_class == IdentityMatchClass.INFERRED
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


def test_mail_normalizer_does_not_trust_customer_id_derived_only_from_email() -> None:
    batch = MailMessageNormalizer(tenant_id="foton").normalize(
        TimelineSourceRecord(
            source_system="mail_archive",
            source_ref="mail#email-only",
            payload={
                "message_sha256": SHA,
                "resolved_customer_id": "customer:email-only",
                "from_email": "client@example.com",
                "to_email": "edu@kmipt.ru",
                "date_last": "2026-05-03T09:00:00+00:00",
                "allowed_for_bot": False,
            },
        )
    )

    assert batch.customers == ()
    assert batch.identity_links == ()
    assert batch.events == ()
    assert batch.conflicts[0]["conflict_type"] == "pending_attribution"


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


def test_mango_processed_summary_rejects_allowed_for_bot_true() -> None:
    with pytest.raises(ValueError, match="mango_processed_summary source records.*allowed_for_bot=False"):
        MangoCallSummaryNormalizer(tenant_id="foton").normalize(
            TimelineSourceRecord(
                source_system="mango_processed_summary",
                source_ref="mango#unsafe",
                payload={
                    "call_id": "provider:unsafe",
                    "phone": "+79163334455",
                    "call_at": "2026-05-04T12:00:00+00:00",
                    "summary": "Клиент уточнил стоимость.",
                    "identity_authority": "existing_timeline_increment",
                    "identity_resolved_by_increment": True,
                    "match_class": "strong_unique",
                    "customer_id": "customer:existing",
                    "allowed_for_bot": True,
                },
            )
        )


def test_amo_incremental_sources_reject_allowed_for_bot_true_in_children() -> None:
    source_record = TimelineSourceRecord(
        source_system="amo_events_created_at",
        source_ref="amocrm:event:unsafe",
        payload={"event_id": "unsafe"},
    )

    with pytest.raises(ValueError, match="amocrm_event timeline events.*allowed_for_bot=False"):
        TimelineNormalizedBatch(
            source_record=source_record,
            events=(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id="customer:amo-unsafe",
                    event_type=TimelineEventType.AMO_NOTE,
                    event_at=NOW,
                    source_system="amocrm_event",
                    source_id="evt-unsafe",
                    direction=TimelineDirection.INBOUND,
                    record={"allowed_for_bot": True},
                ),
            ),
        )

    with pytest.raises(ValueError, match="amocrm_event bot context chunks.*allowed_for_bot=False"):
        TimelineNormalizedBatch(
            source_record=source_record,
            bot_context_chunks=(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id="customer:amo-unsafe",
                    chunk_type="amo_event_raw",
                    text="AMO raw event must stay manager-only.",
                    source_ref="amocrm:event:unsafe",
                    source_system="amocrm_event",
                    allowed_for_bot=True,
                    requires_manager_review=False,
                ),
            ),
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
