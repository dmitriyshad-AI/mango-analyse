from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.family_graph as family_graph_module
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityMatchClass,
    TimelineEvent,
)
from mango_mvp.customer_timeline.family_graph import FamilyGraphConfig, _connect, build_family_graph
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def test_family_graph_assigns_single_child_family_with_high_confidence(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:one", phone="+79000000001")
    _seed_event(db_path, tmp_path, customer_id="customer:one", source_id="call-1", summary="Клиент спросил про расписание курса.")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:one", phone="+79000000001")
    _insert_field(profiles_db, profile_id="customer:one", field="child_name", value="Аня", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:one", field="grade", value="8", child_key="child_1")

    report = build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
            out_path=tmp_path / ".codex_local" / "staging" / "family.json",
        )
    )

    assert report["llm_calls_total"] == 0
    assert report["quick_check"] == "deferred_to_nightly_service"
    assert report["family_confidence_counts"]["high"] == 1
    with sqlite3.connect(db_path) as con:
        family = con.execute("SELECT canonical_name, status, confidence FROM family_links_v1").fetchone()
        member = con.execute(
            "SELECT membership_status, confidence FROM family_members_v1 WHERE customer_id='customer:one'"
        ).fetchone()
        event = con.execute("SELECT status, confidence, reason, child_key FROM event_child_attribution_v1").fetchone()
    assert family == ("Аня", "confident", "high")
    assert member == ("singleton", "medium")
    assert event[0:3] == ("matched", "high", "single_child_family")
    assert event[3]


def test_family_graph_reuses_normalized_amo_organization_brand(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:amo-brand", phone="+79000000009")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:amo-brand", phone="+79000000009")
    _insert_field(profiles_db, profile_id="customer:amo-brand", field="child_name", value="Анна", child_key="child_1")
    with sqlite3.connect(profiles_db) as con:
        con.execute("UPDATE profile_fields SET brand='unknown'")
        con.commit()
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT opportunity_id,record_json FROM customer_opportunities WHERE customer_id='customer:amo-brand'"
        ).fetchone()
        record = json.loads(row[1])
        record["product_context"]["brand"] = "unpk"
        con.execute(
            "UPDATE customer_opportunities SET record_json=? WHERE opportunity_id=?",
            (json.dumps(record, ensure_ascii=False), row[0]),
        )
        con.commit()

    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT brand FROM family_links_v1").fetchone()[0] == "unpk"


def test_family_graph_keeps_explicit_amo_brand_conflict_unknown(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:amo-conflict", phone="+79000000010")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:amo-conflict", phone="+79000000010")
    _insert_field(
        profiles_db,
        profile_id="customer:amo-conflict",
        field="child_name",
        value="Анна",
        child_key="child_1",
    )
    with sqlite3.connect(profiles_db) as con:
        con.execute("UPDATE profile_fields SET brand='unpk'")
        con.commit()
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT opportunity_id,record_json FROM customer_opportunities "
            "WHERE customer_id='customer:amo-conflict'"
        ).fetchone()
        record = json.loads(row[1])
        record["product_context"].update(
            {"brand": "unknown", "brand_source": "amo_organization_conflict"}
        )
        con.execute(
            "UPDATE customer_opportunities SET record_json=? WHERE opportunity_id=?",
            (json.dumps(record, ensure_ascii=False), row[0]),
        )
        con.commit()

    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT brand FROM family_links_v1").fetchone()[0] == "unknown"


def test_family_graph_keeps_amo_contact_brand_conflict_unknown(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:amo-contact", phone="+79000000011")
    _seed_event(
        db_path,
        tmp_path,
        customer_id="customer:amo-contact",
        source_id="amo-contact-conflict",
        summary="Снимок контакта AMO",
    )
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:amo-contact", phone="+79000000011")
    _insert_field(
        profiles_db,
        profile_id="customer:amo-contact",
        field="child_name",
        value="Анна",
        child_key="child_1",
    )
    with sqlite3.connect(profiles_db) as con:
        con.execute("UPDATE profile_fields SET brand='unpk'")
        con.commit()
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT event_id,record_json FROM timeline_events WHERE source_id='amo-contact-conflict'"
        ).fetchone()
        record = json.loads(row[1])
        record["source_system"] = "amocrm_snapshot"
        record["record"] = {"brand": "unknown", "brand_source": "amo_organization_conflict"}
        con.execute(
            "UPDATE timeline_events SET source_system='amocrm_snapshot',record_json=? WHERE event_id=?",
            (json.dumps(record, ensure_ascii=False), row[0]),
        )
        con.commit()

    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT brand FROM family_links_v1").fetchone()[0] == "unknown"


def test_family_graph_skips_identity_without_any_primary_evidence(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:real", phone="+79000000001")
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:empty-shell",
                identity_status="partial",
                created_at=NOW,
                updated_at=NOW,
            )
        )

    report = build_family_graph(
        FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True)
    )

    assert report["family_members_total"] == 1
    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT customer_id FROM family_members_v1 ORDER BY customer_id"
        ).fetchall() == [("customer:real",)]


def test_family_graph_removes_stale_member_without_primary_evidence(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:empty-shell",
                identity_status="partial",
                created_at=NOW,
                updated_at=NOW,
            )
        )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO family_members_v1 (tenant_id,family_id,customer_id,membership_status,"
            "confidence,reason,created_at,updated_at,record_hash,record_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "foton", "family:stale", "customer:empty-shell", "singleton", "medium",
                "stale", NOW.isoformat(), NOW.isoformat(), "hash", "{}",
            ),
        )
        con.commit()

    report = build_family_graph(
        FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True)
    )

    assert report["family_members_total"] == 0
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM family_members_v1").fetchone()[0] == 0


@pytest.mark.parametrize("match_status", ("strong_unique", "manual"))
def test_family_graph_groups_tallanto_siblings_by_parent_email(tmp_path: Path, match_status: str) -> None:
    db_path = _timeline_db(tmp_path)
    profiles_db = _profiles_db(tmp_path)
    for index, (customer_id, child_name) in enumerate(
        (("customer:child-a", "Анна Иванова"), ("customer:child-b", "Борис Иванов")),
        start=1,
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=f"+7900000002{index}")
        _insert_profile(profiles_db, profile_id=customer_id, phone=f"+7900000002{index}")
        _insert_field(profiles_db, profile_id=customer_id, field="child_name", value=child_name, child_key=f"child_{index}")
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            f"tallanto-{index}",
            "parent@example.com",
            match_status=match_status,
        )

    report = build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id, family_id FROM family_links_v1 ORDER BY customer_id"
        ).fetchall()
        members = con.execute(
            "SELECT customer_id, family_id, membership_status FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert len(rows) == 2
    assert len({row[1] for row in rows}) == 1
    assert len(members) == 2
    assert len({row[1] for row in members}) == 1
    assert {row[2] for row in members} == {"confident"}
    assert report["multi_customer_families"] == 1


def test_family_graph_reconciles_shared_phone_after_current_tallanto_cards(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    phone = "+79000000991"
    customer_ids = ("customer:sibling-a", "customer:sibling-b")
    student_ids = ("student-a", "student-b")
    for customer_id, student_id, student_name in zip(
        customer_ids,
        student_ids,
        ("Аглая Ким", "Ратмир Ким"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            student_id,
            "parent@example.com",
            student_name=student_name,
            student_phone=phone,
            first_name=student_name.split()[0],
            last_name="Ким",
        )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for customer_id, student_id, student_name in zip(
            customer_ids,
            student_ids,
            ("Анна Иванова", "Борис Иванов"),
        ):
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="phone",
                    link_value=phone,
                    source_system="tallanto_snapshot",
                    source_ref=f"legacy:{customer_id}:phone",
                    match_class=IdentityMatchClass.AMBIGUOUS,
                    confidence=0.5,
                )
            )
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="tallanto_student_snapshot",
                    event_at=NOW,
                    source_system="tallanto_snapshot",
                    source_id=f"tallanto:{customer_id}",
                    direction="system",
                    match_status="ambiguous",
                    record={},
                )
            )
        conflict = store.record_conflict(
            "foton",
            conflict_type="shared_family_phone",
            entity_refs=(
                f"phone_hash:{stable_digest({'phone': phone})[:16]}",
                *(f"customer:{customer_id}" for customer_id in customer_ids),
                *(f"tallanto_student:{student_id}" for student_id in student_ids),
            ),
            severity="high",
            metadata={
                "phone_hash": stable_digest({"phone": phone})[:16],
                "tallanto_student_ids": list(student_ids),
            },
        )

    first = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        status, first_hash = con.execute(
            "SELECT status,record_hash FROM timeline_conflicts WHERE conflict_id=?",
            (conflict.record_id,),
        ).fetchone()
        members = con.execute(
            "SELECT family_id,membership_status FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        repeated = store.record_conflict(
            "foton",
            conflict_type="shared_family_phone",
            entity_refs=(
                f"phone_hash:{stable_digest({'phone': phone})[:16]}",
                *(f"customer:{customer_id}" for customer_id in customer_ids),
                *(f"tallanto_student:{student_id}" for student_id in student_ids),
            ),
            severity="high",
            metadata={
                "phone_hash": stable_digest({"phone": phone})[:16],
                "tallanto_student_ids": list(student_ids),
            },
        )
    second = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        second_status, second_hash = con.execute(
            "SELECT status,record_hash FROM timeline_conflicts WHERE conflict_id=?",
            (conflict.record_id,),
        ).fetchone()

    assert status == "resolved"
    assert repeated.status == "updated"
    assert len({row[0] for row in members}) == 1
    assert {row[1] for row in members} == {"confident"}
    assert first["contact_conflict_reconciliation"]["resolved"] == 1
    assert second["contact_conflict_reconciliation"]["resolved"] == 1
    assert second["contact_conflict_reconciliation"]["reopened"] == 0
    assert second_status == "resolved"
    assert second_hash == first_hash


def test_family_graph_groups_exact_siblings_with_different_parent_names(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    phone = "+79000000993"
    for index, (customer_id, child_name, parent_name) in enumerate(
        (
            ("customer:kim-a", "Аглая Ким", "Ирина Ким"),
            ("customer:kim-b", "Ратмир Ким", "Ирина Ким"),
            ("customer:kim-c", "Элина Ким", "Сергей Ким"),
        ),
        start=1,
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            f"student-kim-{index}",
            "family-kim@example.com",
            parent_name,
            student_name=child_name,
            student_phone=phone,
            first_name=f"{child_name.split()[0]} Ильинична",
            last_name="Ким",
            student_type=f"Слушатель {index}",
        )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        conflict = store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=(
                "email:family-kim@example.com",
                f"phone:{phone}",
                "customer:kim-a",
                "customer:kim-b",
                "customer:kim-c",
            ),
            severity="high",
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT family_id,membership_status,reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
        conflict_status = con.execute(
            "SELECT status FROM timeline_conflicts WHERE conflict_id=?", (conflict.record_id,)
        ).fetchone()[0]
    assert len({row[0] for row in rows}) == 1
    assert {row[1] for row in rows} == {"confident"}
    assert [row[2] for row in rows].count("exact_tallanto_shared_contacts_family_core") == 1
    assert conflict_status == "resolved"


def test_family_graph_does_not_attach_diminutive_duplicate_to_family_core(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    phone = "+79000000991"
    for index, (customer_id, child_name, parent_name) in enumerate(
        (
            ("customer:anna", "Анна Иванова", "Ирина Иванова"),
            ("customer:boris", "Борис Иванов", "Ирина Иванова"),
            ("customer:anya", "Аня Иванова", "Сергей Иванов"),
        ),
        start=1,
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            f"student-duplicate-{index}",
            "duplicate-family@example.com",
            parent_name,
            student_name=child_name,
            student_phone=phone,
            first_name=child_name.split()[0],
            last_name="Иванова",
            student_type=f"Слушатель {index}",
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id,family_id,reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert len({row[1] for row in rows}) == 2
    assert dict((row[0], row[2]) for row in rows)["customer:anya"] == "single_customer_family"


@pytest.mark.parametrize(
    "student_names",
    (
        ("Анна Иванова", "Анна Иванова"),
        ("Анна Иванова", "Аня Иванова"),
        ("Александр Иванов", "Саша Иванов"),
        ("Евгений Иванов", "Женя Иванов"),
    ),
)
def test_family_graph_keeps_same_or_diminutive_name_shared_phone_for_manual_review(
    tmp_path: Path,
    student_names: tuple[str, str],
) -> None:
    db_path = _timeline_db(tmp_path)
    phone = "+79000000992"
    customer_ids = ("customer:duplicate-a", "customer:duplicate-b")
    student_ids = ("student-a", "student-b")
    for customer_id, student_id, student_name in zip(customer_ids, student_ids, student_names):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            student_id,
            "parent@example.com",
            student_name=student_name,
            student_phone=phone,
        )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for customer_id, student_id in zip(customer_ids, student_ids):
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="phone",
                    link_value=phone,
                    source_system="tallanto_snapshot",
                    source_ref=f"duplicate:{customer_id}",
                    match_class=IdentityMatchClass.AMBIGUOUS,
                    confidence=0.5,
                )
            )
        conflict = store.record_conflict(
            "foton",
            conflict_type="shared_family_phone",
            entity_refs=(
                f"phone_hash:{stable_digest({'phone': phone})[:16]}",
                *(f"customer:{customer_id}" for customer_id in customer_ids),
                *(f"tallanto_student:{student_id}" for student_id in student_ids),
            ),
            severity="high",
            metadata={
                "phone_hash": stable_digest({"phone": phone})[:16],
                "tallanto_student_ids": list(student_ids),
            },
        )

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        status = con.execute(
            "SELECT status FROM timeline_conflicts WHERE conflict_id=?",
            (conflict.record_id,),
        ).fetchone()[0]
    assert status == "open"
    assert report["contact_conflict_reconciliation"]["resolved"] == 0


def test_family_graph_keeps_contact_conflict_when_phone_and_email_have_different_owners(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id, phone, email, name in (
        ("customer:left", "student-left", "+79000000981", "left@example.com", "Анна Иванова"),
        ("customer:right", "student-right", "+79000000982", "right@example.com", "Борис Иванов"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(
            db_path, tmp_path, customer_id, student_id, email,
            student_name=name, student_phone=phone,
        )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        conflict = store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=("customer:customer:left", "customer:customer:right"),
            severity="high",
            metadata={"identifiers": ["phone:+79000000981", "email:right@example.com"]},
        )

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        status = con.execute(
            "SELECT status FROM timeline_conflicts WHERE conflict_id=?", (conflict.record_id,)
        ).fetchone()[0]
    assert status == "open"
    assert report["contact_conflict_reconciliation"]["resolved"] == 0


def test_family_graph_reopens_managed_contact_conflict_when_owner_moves_outside_original_group(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, phone in (
        ("customer:left", "+79000000971"),
        ("customer:right", "+79000000972"),
        ("customer:new-owner", "+79000000973"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
    _seed_tallanto_identity(
        db_path, tmp_path, "customer:new-owner", "student-new-owner", "owner@example.com",
        student_name="Анна Иванова", student_phone="+79000000973",
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        conflict = store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=("customer:customer:left", "customer:customer:right"),
            severity="high",
            metadata={"identifiers": ["phone:+79000000973"]},
        )
    with sqlite3.connect(db_path) as con:
        payload = json.loads(con.execute(
            "SELECT record_json FROM timeline_conflicts WHERE conflict_id=?", (conflict.record_id,)
        ).fetchone()[0])
        payload.update(status="resolved", resolved_at=NOW.isoformat())
        payload["metadata"].update(
            resolved_by="family_graph_v1_builder", resolution_reason="contact_no_longer_shared"
        )
        con.execute(
            "UPDATE timeline_conflicts SET status='resolved',resolved_at=?,record_hash=?,record_json=? WHERE conflict_id=?",
            (NOW.isoformat(), stable_digest(payload), json.dumps(payload), conflict.record_id),
        )

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        status = con.execute(
            "SELECT status FROM timeline_conflicts WHERE conflict_id=?", (conflict.record_id,)
        ).fetchone()[0]
    assert status == "open"
    assert report["contact_conflict_reconciliation"]["reopened"] == 1


def test_family_graph_uses_one_fallback_child_key_for_same_child_in_one_family(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    profiles_db = _profiles_db(tmp_path)
    for index, customer_id in enumerate(("customer:parent", "customer:student"), start=1):
        phone = f"+7900000003{index}"
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _insert_profile(profiles_db, profile_id=customer_id, phone=phone)
        _insert_field(
            profiles_db,
            profile_id=customer_id,
            field="child_name",
            value="Мария Иванова",
            child_key=f"profile_{index}",
        )
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            f"student-{index}",
            "one-family@example.com",
        )

    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT family_id,customer_id,child_key FROM family_links_v1 "
            "WHERE canonical_name='Мария Иванова' ORDER BY customer_id"
        ).fetchall()
    assert len(rows) == 2
    assert len({row[0] for row in rows}) == 1
    assert len({row[2] for row in rows}) == 1


def test_family_graph_uses_tallanto_amo_contact_as_family_relation(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    parent_id = "customer:parent"
    child_ids = ("customer:child-a", "customer:child-b")
    _seed_amo_parent(
        db_path,
        tmp_path,
        customer_id=parent_id,
        display_name="Ирина Иванова",
        email="parent@amo.example",
    )
    for index, child_id in enumerate(child_ids, 1):
        _seed_customer(db_path, tmp_path, customer_id=child_id, phone=f"+7900000010{index}")
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            child_id,
            f"student-{index}",
            f"child-{index}@example.com",
            parent_name=f"Родитель {index}",
            student_name=f"Ребёнок {index}",
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(DISTINCT family_id) FROM family_members_v1").fetchone()[0] == 3

    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for child_id in child_ids:
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=child_id,
                    link_type="amo_contact_id",
                    link_value=f"amo-{parent_id}",
                    source_system="tallanto_snapshot",
                    source_ref=f"tallanto:{child_id}:amo-family",
                    match_class=IdentityMatchClass.AMBIGUOUS,
                    confidence=1.0,
                    evidence={"relationship": "family_amo_contact"},
                )
            )

    first = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    second = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        members = con.execute(
            "SELECT customer_id, family_id FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
        exact_students = con.execute(
            "SELECT customer_id, link_value FROM identity_links "
            "WHERE link_type='tallanto_student_id' AND match_class='strong_unique' "
            "ORDER BY customer_id"
        ).fetchall()

    assert {row[0] for row in members} == {parent_id, *child_ids}
    assert len({row[1] for row in members}) == 1
    assert {row[0] for row in exact_students} == set(child_ids)
    assert first["multi_customer_families"] == second["multi_customer_families"] == 1


def test_family_graph_rejects_shared_strong_amo_contact_on_fresh_db(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    parent_ids = ("customer:parent-a", "customer:parent-b")
    child_id = "customer:child"
    for index, parent_id in enumerate(parent_ids, 1):
        _seed_amo_parent(
            db_path,
            tmp_path,
            customer_id=parent_id,
            display_name=f"Родитель {index}",
            email=f"parent-{index}@example.com",
        )
    _seed_customer(db_path, tmp_path, customer_id=child_id, phone="+79000000101")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        child_id,
        "student-1",
        "child@example.com",
        student_name="Ребёнок",
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for parent_id in parent_ids:
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=parent_id,
                    link_type="amo_contact_id",
                    link_value="shared-amo-contact",
                    source_system="amocrm_snapshot",
                    source_ref=f"amo:{parent_id}:shared",
                    match_class=IdentityMatchClass.STRONG_UNIQUE,
                    confidence=1.0,
                )
            )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=child_id,
                link_type="amo_contact_id",
                link_value="shared-amo-contact",
                source_system="tallanto_snapshot",
                source_ref="tallanto:child:amo-family",
                match_class=IdentityMatchClass.AMBIGUOUS,
                confidence=1.0,
                evidence={"relationship": "family_amo_contact"},
            )
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id, family_id, membership_status, reason "
            "FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()

    assert len({row[1] for row in rows}) == 3
    assert {row[0] for row in rows if row[2] == "conflict"} == set(parent_ids)
    assert next(row for row in rows if row[0] == child_id)[2:] == (
        "singleton",
        "single_customer_family",
    )


@pytest.mark.parametrize("match_status", ("ambiguous", "inferred"))
def test_family_root_rejects_non_strong_tallanto_snapshot(tmp_path: Path, match_status: str) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:risky", phone="+79000000021")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        "customer:risky",
        "student-risky",
        "parent@example.com",
        match_status=match_status,
    )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "DELETE FROM identity_links WHERE customer_id='customer:risky' "
            "AND link_type='tallanto_student_id'"
        )
    _seed_customer(db_path, tmp_path, customer_id="customer:safe", phone="+79000000022")
    _seed_tallanto_identity(db_path, tmp_path, "customer:safe", "student-safe", "parent@example.com")

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id, family_id, membership_status, reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    by_customer = {row[0]: row for row in rows}
    assert len({row[1] for row in rows}) == 2
    assert by_customer["customer:risky"][2:] == ("conflict", "tallanto_student_id_conflict")
    assert by_customer["customer:safe"][2:] == ("singleton", "single_customer_family")


def test_family_graph_accepts_late_exact_link_for_inferred_card_and_event(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    customer_id = "customer:late-exact"
    student_id = "student-late-exact"
    _seed_customer(db_path, tmp_path, customer_id=customer_id, phone="+79000000023")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        customer_id,
        student_id,
        "parent@example.com",
        match_status="inferred",
        student_name="Анна Иванова",
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        payment = TimelineEvent(
            tenant_id="foton",
            customer_id=customer_id,
            event_type="tallanto_payment",
            event_at=NOW,
            source_system="tallanto_payment",
            source_id="payment-late-exact",
            direction="system",
            match_status="inferred",
            record={"contact_id": student_id},
        )
        store.upsert_event(payment)

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        member = con.execute(
            "SELECT membership_status,reason FROM family_members_v1 WHERE customer_id=?",
            (customer_id,),
        ).fetchone()
        attribution = con.execute(
            "SELECT status,reason,child_key FROM event_child_attribution_v1 WHERE event_id=?",
            (payment.event_id,),
        ).fetchone()
    assert member == ("singleton", "single_customer_family")
    assert attribution[0:2] == ("matched", "exact_tallanto_identity")
    assert attribution[2]


def test_family_graph_keeps_canonical_owner_safe_from_stale_ambiguous_holder(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    family_customers = ("customer:canonical-a", "customer:canonical-b")
    for index, customer_id in enumerate(family_customers, start=1):
        _seed_customer(
            db_path,
            tmp_path,
            customer_id=customer_id,
            phone=f"+7900000003{index}",
        )
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            f"student-canonical-{index}",
            "canonical-family@example.com",
            student_name=("Анна Иванова", "Борис Иванов")[index - 1],
        )
    _seed_customer(
        db_path,
        tmp_path,
        customer_id="customer:stale-holder",
        phone="+79000000039",
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:stale-holder",
                link_type="tallanto_student_id",
                link_value="student-canonical-1",
                source_system="tallanto_snapshot",
                source_ref="historical:ambiguous-holder",
                match_class=IdentityMatchClass.AMBIGUOUS,
                confidence=0.5,
            )
        )
        conflict = store.record_conflict(
            "foton",
            conflict_type="ambiguous_identity",
            entity_refs=(
                "email:canonical-family@example.com",
                *family_customers,
            ),
            metadata={"identifiers": ["email:canonical-family@example.com"]},
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        members = con.execute(
            "SELECT customer_id,family_id,membership_status FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
        event_owners = con.execute(
            "SELECT source_id,customer_id FROM timeline_events "
            "WHERE event_type='tallanto_student_snapshot' ORDER BY source_id"
        ).fetchall()
        conflict_status = con.execute(
            "SELECT status FROM timeline_conflicts WHERE conflict_id=?",
            (conflict.record_id,),
        ).fetchone()[0]
    by_customer = {row[0]: row[1:] for row in members}
    assert by_customer[family_customers[0]][0] == by_customer[family_customers[1]][0]
    assert by_customer[family_customers[0]][1] == "confident"
    assert by_customer[family_customers[1]][1] == "confident"
    assert by_customer["customer:stale-holder"][1] == "conflict"
    assert event_owners == [
        ("student-canonical-1", family_customers[0]),
        ("student-canonical-2", family_customers[1]),
    ]
    assert conflict_status == "resolved"


def test_family_graph_rejects_inferred_card_when_exact_id_belongs_to_other_customer(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:other", phone="+79000000024")
    _seed_customer(db_path, tmp_path, customer_id="customer:risky", phone="+79000000025")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        "customer:risky",
        "student-other-owner",
        "parent@example.com",
        match_status="inferred",
        student_name="Анна Иванова",
    )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE identity_links SET customer_id='customer:other' "
            "WHERE link_type='tallanto_student_id' AND link_value='student-other-owner'"
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT membership_status FROM family_members_v1 WHERE customer_id='customer:risky'"
        ).fetchone() == ("conflict",)
        assert con.execute(
            "SELECT reason FROM event_child_attribution_v1 event_attr "
            "JOIN timeline_events event ON event.event_id=event_attr.event_id "
            "WHERE event.customer_id='customer:risky' AND event.source_system='tallanto_snapshot'"
        ).fetchone() == ("tallanto_student_id_not_in_family",)


def test_family_root_partial_run_marks_global_tallanto_conflict(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, phone in (("customer:left", "+79000000001"), ("customer:right", "+79000000002")):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(db_path, tmp_path, customer_id, "student-shared", "parent@example.com")

    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            customer_ids=("customer:left",),
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id, membership_status FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert rows == [("customer:left", "conflict"), ("customer:right", "conflict")]


def test_family_root_honors_open_tallanto_conflict_record(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id, phone in (
        ("customer:left", "student-left", "+79000000001"),
        ("customer:right", "student-right", "+79000000002"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(db_path, tmp_path, customer_id, student_id, "parent@example.com")
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.record_conflict(
            "foton",
            conflict_type="tallanto_identity_conflict",
            entity_refs=("tallanto_student_id:student-left",),
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = dict(
            con.execute("SELECT customer_id, membership_status FROM family_members_v1 ORDER BY customer_id")
        )
    assert rows == {"customer:left": "conflict", "customer:right": "singleton"}


def test_store_bootstrap_migrates_early_family_members_table(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE family_members_v1 (
              tenant_id TEXT NOT NULL, family_id TEXT NOT NULL, customer_id TEXT NOT NULL,
              membership_status TEXT NOT NULL, confidence TEXT NOT NULL, reason TEXT NOT NULL,
              created_at TEXT NOT NULL, record_hash TEXT NOT NULL, record_json TEXT NOT NULL,
              PRIMARY KEY (tenant_id, customer_id)
            )
            """
        )

    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path):
        pass

    with sqlite3.connect(db_path) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(family_members_v1)")}
    assert "updated_at" in columns


def test_family_graph_does_not_merge_same_surname_with_different_parents(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    phone = "+79000000994"
    for customer_id, student_id, child_name, parent_name in (
        ("customer:left", "student-left", "Анна Иванова", "Ирина Иванова"),
        ("customer:right", "student-right", "Мария Иванова", "Ольга Петрова"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            student_id,
            "shared@example.com",
            parent_name,
            student_name=child_name,
            student_phone=phone,
            first_name=child_name.split()[0],
            last_name="Иванова",
            student_type=f"Слушатель {student_id}",
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT family_id, membership_status FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert len({row[0] for row in rows}) == 2
    assert {row[1] for row in rows} == {"singleton"}


def test_family_graph_ignores_stale_tallanto_contact_links(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id, email in (
        ("customer:a", "student-a", "old@example.com"),
        ("customer:b", "student-b", "old@example.com"),
        ("customer:c", "student-c", "isolated@example.com"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=f"+7900000000{len(customer_id)}")
        _seed_tallanto_identity(db_path, tmp_path, customer_id, student_id, email)
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        before = dict(con.execute("SELECT customer_id,family_id FROM family_members_v1"))
    assert before["customer:a"] == before["customer:b"]
    assert before["customer:c"] != before["customer:b"]

    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for customer_id in ("customer:b", "customer:c"):
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="email",
                    link_value="new@example.com",
                    source_system="tallanto_snapshot",
                    source_ref=f"stale:{customer_id}",
                    match_class="strong_unique",
                    confidence=1.0,
                )
            )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        after = dict(con.execute("SELECT customer_id,family_id FROM family_members_v1"))
    assert after["customer:a"] == after["customer:b"]
    assert after["customer:c"] != after["customer:b"]


def test_family_graph_attaches_exact_amo_parent_and_attributes_parent_event_to_child(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    profiles_db = _profiles_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:child", phone="+79000000041")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        "customer:child",
        "student-child",
        "parent@example.com",
        "Ирина Иванова",
    )
    _insert_profile(profiles_db, profile_id="customer:child", phone="+79000000041")
    _insert_field(
        profiles_db,
        profile_id="customer:child",
        field="child_name",
        value="Анна",
        child_key="child_1",
    )
    _seed_amo_parent(
        db_path,
        tmp_path,
        customer_id="customer:parent",
        display_name="Ирина Иванова",
        email="parent@example.com",
    )
    _seed_event(
        db_path,
        tmp_path,
        customer_id="customer:parent",
        source_id="parent-call",
        summary="Ирина уточнила расписание для Анны.",
    )

    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
        )
    )
    _seed_event(
        db_path, tmp_path, customer_id="customer:parent", source_id="parent-call-late",
        summary="Ирина повторно уточнила расписание для Анны.",
    )
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        members = con.execute(
            "SELECT customer_id, family_id, reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
        event_owners = con.execute(
            """
            SELECT a.customer_id
            FROM event_child_attribution_v1 a
            JOIN timeline_events e ON e.event_id=a.event_id
            WHERE e.source_id IN ('parent-call','parent-call-late') AND a.status='matched'
            ORDER BY e.source_id
            """
        ).fetchall()
    assert len({row[1] for row in members}) == 1
    assert dict((row[0], row[2]) for row in members)["customer:parent"] == "exact_amo_parent_name_and_phone_or_email"
    assert event_owners == [("customer:child",), ("customer:child",)]


def test_family_graph_does_not_attach_amo_contact_with_different_parent_name(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:child", phone="+79000000042")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        "customer:child",
        "student-child",
        "parent@example.com",
        "Ирина Иванова",
    )
    _seed_amo_parent(
        db_path,
        tmp_path,
        customer_id="customer:other",
        display_name="Мария Петрова",
        email="parent@example.com",
    )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        families = con.execute("SELECT family_id FROM family_members_v1 ORDER BY customer_id").fetchall()
    assert len({row[0] for row in families}) == 2


def test_family_graph_does_not_treat_tallanto_student_as_amo_parent(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:child", phone="+79000000043")
    _seed_tallanto_identity(
        db_path, tmp_path, "customer:child", "student-child", "parent@example.com", "Ирина Иванова"
    )
    _seed_amo_parent(
        db_path, tmp_path, customer_id="customer:mixed", display_name="Ирина Иванова",
        email="parent@example.com",
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton", customer_id="customer:mixed", link_type="tallanto_student_id",
                link_value="student-mixed", source_system="tallanto_snapshot", source_ref="student:mixed",
                match_class="strong_unique", confidence=1.0,
            )
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        families = con.execute("SELECT family_id FROM family_members_v1 ORDER BY customer_id").fetchall()
    assert len({row[0] for row in families}) == 2


def test_family_graph_removes_stale_family_edge_when_parent_evidence_changes(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id in (
        ("customer:left", "student-left"),
        ("customer:right", "student-right"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=f"+7900000000{len(customer_id)}")
        _seed_tallanto_identity(db_path, tmp_path, customer_id, student_id, "shared@example.com")
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(DISTINCT family_id) FROM family_members_v1").fetchone()[0] == 1
        event_id = con.execute(
            "SELECT event_id FROM timeline_events WHERE customer_id='customer:right' AND source_system='tallanto_snapshot'"
        ).fetchone()[0]
        payload = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE event_id=?", (event_id,)).fetchone()[0])
        payload["record"]["payload"]["parent_fio"] = "Мария Петрова"
        con.execute("UPDATE timeline_events SET record_json=? WHERE event_id=?", (json.dumps(payload), event_id))

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT family_id, membership_status FROM family_members_v1").fetchall()
    assert len({row[0] for row in rows}) == 2
    assert {row[1] for row in rows} == {"singleton"}


def test_family_graph_preserves_existing_child_graph_without_profiles_db(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    profiles_db = _profiles_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:one", phone="+79000000031")
    _insert_profile(profiles_db, profile_id="customer:one", phone="+79000000031")
    _insert_field(profiles_db, profile_id="customer:one", field="child_name", value="Анна", child_key="child_1")

    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=profiles_db,
            apply=True,
        )
    )
    report = build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        child_rows = con.execute(
            "SELECT canonical_name FROM family_links_v1 WHERE customer_id='customer:one'"
        ).fetchall()
        member_rows = con.execute(
            "SELECT membership_status FROM family_members_v1 WHERE customer_id='customer:one'"
        ).fetchall()
    assert child_rows == [("Анна",)]
    assert member_rows == [("singleton",)]
    assert report["existing_family_links"] == 1
    assert report["child_graph_preserved_without_profiles"] is True
    assert report["child_graph_write_applied"] is False


def test_family_graph_apply_with_selection_rebuilds_global_family_root(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:first", phone="+79000000041")
    _seed_tallanto_identity(db_path, tmp_path, "customer:first", "student-first", "parent@example.com")
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    _seed_customer(db_path, tmp_path, customer_id="customer:second", phone="+79000000042")
    _seed_tallanto_identity(db_path, tmp_path, "customer:second", "student-second", "parent@example.com")
    build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            customer_ids=("customer:second",),
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        roots = dict(con.execute("SELECT customer_id, family_id FROM family_members_v1"))
    assert set(roots) == {"customer:first", "customer:second"}
    assert len(set(roots.values())) == 1


def test_family_graph_missing_profiles_path_preserves_existing_children(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    profiles_db = _profiles_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:one", phone="+79000000043")
    _insert_profile(profiles_db, profile_id="customer:one", phone="+79000000043")
    _insert_field(profiles_db, profile_id="customer:one", field="child_name", value="Анна", child_key="child_1")
    build_family_graph(
        FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True)
    )

    report = build_family_graph(
        FamilyGraphConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            profiles_db=tmp_path / "missing.sqlite",
            apply=True,
        )
    )

    with sqlite3.connect(db_path) as con:
        children = con.execute("SELECT canonical_name FROM family_links_v1").fetchall()
    assert children == [("Анна",)]
    assert report["child_graph_preserved_without_profiles"] is True


def test_family_root_is_stable_when_new_child_is_added(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id, phone in (
        ("customer:b", "student-b", "+79000000002"),
        ("customer:c", "student-c", "+79000000003"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(db_path, tmp_path, customer_id, student_id, "parent@example.com")
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        original = dict(con.execute("SELECT customer_id, family_id FROM family_members_v1"))

    _seed_customer(db_path, tmp_path, customer_id="customer:a", phone="+79000000001")
    _seed_tallanto_identity(db_path, tmp_path, "customer:a", "student-a", "parent@example.com")
    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        current = dict(con.execute("SELECT customer_id, family_id FROM family_members_v1"))
    assert report["families_total"] == 1
    assert current["customer:b"] == original["customer:b"]
    assert current["customer:c"] == original["customer:c"]
    assert current["customer:a"] == original["customer:b"]


def test_family_root_rerun_is_idempotent(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:one", phone="+79000000001")
    _seed_tallanto_identity(db_path, tmp_path, "customer:one", "student-one", "parent@example.com")

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        before = con.execute("SELECT * FROM family_members_v1").fetchall()
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        after = con.execute("SELECT * FROM family_members_v1").fetchall()
    assert after == before


def test_family_links_and_child_attribution_rerun_is_idempotent(tmp_path: Path) -> None:
    """BLOK C3 idempotency: rebuilding the family graph twice on an unchanged input
    must not change family_links_v1 (child_key rows) or event_child_attribution_v1,
    and two distinct children must keep two distinct, stable child_key values across
    both runs (complements test_family_root_rerun_is_idempotent, which only covers
    family_members_v1/family_id)."""
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:siblings-rerun", phone="+79000000771")
    _seed_event(
        db_path,
        tmp_path,
        customer_id="customer:siblings-rerun",
        source_id="call-rerun-1",
        summary="Обсуждали занятия для Никиты.",
    )
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:siblings-rerun", phone="+79000000771")
    _insert_field(profiles_db, profile_id="customer:siblings-rerun", field="child_name", value="Кулаков Никита", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:siblings-rerun", field="child_name", value="Кулакова Дарья", child_key="child_2")
    config = FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True)

    first_report = build_family_graph(config)
    with sqlite3.connect(db_path) as con:
        family_links_before = con.execute(
            "SELECT * FROM family_links_v1 ORDER BY customer_id, child_key"
        ).fetchall()
        attribution_before = con.execute(
            "SELECT * FROM event_child_attribution_v1 ORDER BY event_id"
        ).fetchall()

    second_report = build_family_graph(config)
    with sqlite3.connect(db_path) as con:
        family_links_after = con.execute(
            "SELECT * FROM family_links_v1 ORDER BY customer_id, child_key"
        ).fetchall()
        attribution_after = con.execute(
            "SELECT * FROM event_child_attribution_v1 ORDER BY event_id"
        ).fetchall()

    assert first_report["family_links_total"] == 2
    assert second_report["family_links_total"] == 2
    assert family_links_before  # fixture sanity: rows actually exist
    assert family_links_after == family_links_before
    assert attribution_after == attribution_before
    child_keys = {row[3] for row in family_links_before}
    assert len(child_keys) == 2  # the two siblings never collapse onto one child_key


def test_persisted_family_traits_do_not_grow_on_rerun(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    profiles_db = _profiles_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:traits", phone="+79000000772")
    _insert_profile(profiles_db, profile_id="customer:traits", phone="+79000000772")
    _insert_field(
        profiles_db,
        profile_id="customer:traits",
        field="child_name",
        value="Анна Иванова",
        child_key="child_1",
    )
    _insert_field(
        profiles_db,
        profile_id="customer:traits",
        field="grade",
        value="8",
        child_key="child_1",
    )
    build_family_graph(
        FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True)
    )
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE a2v3_mail_event_facts (tenant_id TEXT,event_id TEXT,customer_id TEXT,"
            "student_name TEXT,grade TEXT,subject_area TEXT,email_brand TEXT)"
        )
        con.execute(
            "INSERT INTO a2v3_mail_event_facts VALUES (?,?,?,?,?,?,?)",
            ("foton", "mail-1", "customer:traits", "Анна Иванова", "9", "", "unknown"),
        )

    config = FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True)
    build_family_graph(config)
    with sqlite3.connect(db_path) as con:
        before = con.execute("SELECT grades_json,record_hash,record_json FROM family_links_v1").fetchone()
    build_family_graph(config)
    with sqlite3.connect(db_path) as con:
        after = con.execute("SELECT grades_json,record_hash,record_json FROM family_links_v1").fetchone()

    assert json.loads(before[0]) == ["8", "9"]
    assert after == before


def test_family_root_rejects_conflicting_tallanto_student_id(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, phone in (("customer:left", "+79000000001"), ("customer:right", "+79000000002")):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(db_path, tmp_path, customer_id, "student-shared", "parent@example.com")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT family_id, membership_status, reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert len({row[0] for row in rows}) == 2
    assert {row[1] for row in rows} == {"conflict"}
    assert {row[2] for row in rows} == {"tallanto_student_id_conflict"}
    assert report["family_membership_status_counts"] == {"conflict": 2}


def test_family_root_accepts_multiple_students_for_one_parent(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:family", phone="+79000000001")
    _seed_tallanto_identity(db_path, tmp_path, "customer:family", "student-one", "parent@example.com")
    _seed_tallanto_identity(db_path, tmp_path, "customer:family", "student-two", "parent@example.com")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT membership_status, reason FROM family_members_v1 WHERE customer_id='customer:family'"
        ).fetchone()
    assert row == ("singleton", "single_customer_family")
    assert report["family_membership_status_counts"] == {"singleton": 1}


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    (("Анна Иванова", "Мария Иванова"), ("Даня", "Даниил Иванов")),
)
def test_family_graph_keeps_historical_tallanto_students_distinct_under_one_customer(
    tmp_path: Path,
    first_name: str,
    second_name: str,
) -> None:
    db_path = _timeline_db(tmp_path)
    customer_id = "customer:historically-merged"
    _seed_customer(db_path, tmp_path, customer_id=customer_id, phone="+79000000003")
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for student_id, name in (("student-one", first_name), ("student-two", second_name)):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="tallanto_student_snapshot",
                    event_at=NOW,
                    source_system="tallanto_snapshot",
                    source_id=student_id,
                    source_ref=f"tallanto:contact:{student_id}",
                    direction="system",
                    match_status="strong_unique",
                    record={"payload": {"display_name": name}},
                )
            )

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        children = con.execute(
            "SELECT canonical_name,child_key FROM family_links_v1 WHERE customer_id=? ORDER BY canonical_name",
            (customer_id,),
        ).fetchall()
    assert {row[0] for row in children} == {first_name, second_name}
    assert len({row[1] for row in children}) == 2
    assert report["family_links_total"] == 2


def test_profile_name_cannot_bridge_two_exact_tallanto_children(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    customer_id = "customer:bridge"
    _seed_customer(db_path, tmp_path, customer_id=customer_id, phone="+79000000003")
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for student_id in ("student-one", "student-two"):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="tallanto_student_snapshot",
                    event_at=NOW,
                    source_system="tallanto_snapshot",
                    source_id=student_id,
                    direction="system",
                    match_status="strong_unique",
                    record={"payload": {"display_name": "Иван Иванов"}},
                )
            )
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id=customer_id, phone="+79000000003")
    _insert_field(profiles_db, profile_id=customer_id, field="child_name", value="Иван Иванов", child_key="profile-child")

    build_family_graph(
        FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True)
    )
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT child_key,record_json FROM family_links_v1 WHERE customer_id=? ORDER BY child_key",
            (customer_id,),
        ).fetchall()
    assert len(rows) == 2
    assert {tuple(json.loads(row[1])["tallanto_student_ids"]) for row in rows} == {
        ("student-one",),
        ("student-two",),
    }


def test_wrong_historical_attribution_cannot_reuse_another_students_key(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    customer_id = "customer:wrong-history"
    _seed_customer(db_path, tmp_path, customer_id=customer_id, phone="+79000000004")
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for student_id, name in (("student-one", "Анна Иванова"), ("student-two", "Мария Иванова")):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="tallanto_student_snapshot",
                    event_at=NOW,
                    source_system="tallanto_snapshot",
                    source_id=student_id,
                    direction="system",
                    match_status="strong_unique",
                    record={"payload": {"display_name": name}},
                )
            )
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        event_ids = dict(con.execute("SELECT source_id,event_id FROM timeline_events"))
        key_one = con.execute(
            "SELECT child_key FROM event_child_attribution_v1 WHERE event_id=?",
            (event_ids["student-one"],),
        ).fetchone()[0]
        con.execute("DELETE FROM event_child_attribution_v1 WHERE event_id=?", (event_ids["student-one"],))
        con.execute(
            "UPDATE event_child_attribution_v1 SET child_key=? WHERE event_id=?",
            (key_one, event_ids["student-two"]),
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        child_keys = {
            row[0] for row in con.execute("SELECT child_key FROM family_links_v1 WHERE customer_id=?", (customer_id,))
        }
    assert len(child_keys) == 2


def test_exact_tallanto_child_key_survives_customer_relink(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:old", phone="+79000000005")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        "customer:old",
        "student-moving",
        "parent@example.com",
        student_name="Анна Иванова",
    )
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        before = con.execute(
            "SELECT child_key FROM family_links_v1 WHERE customer_id='customer:old'"
        ).fetchone()[0]

    _seed_customer(db_path, tmp_path, customer_id="customer:new", phone="+79000000006")
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE timeline_events SET customer_id='customer:new' WHERE source_id='student-moving'"
        )
        con.execute(
            "UPDATE identity_links SET customer_id='customer:new' WHERE source_system='tallanto_snapshot'"
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id,child_key FROM family_links_v1 ORDER BY customer_id"
        ).fetchall()
    assert rows == [("customer:new", before)]


def test_family_root_merges_two_persisted_roots_on_exact_tallanto_proof(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id, phone in (
        ("customer:left", "student-left", "+79000000001"),
        ("customer:right", "student-right", "+79000000002"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(db_path, tmp_path, customer_id, student_id, f"{student_id}@example.com")
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        original = dict(con.execute("SELECT customer_id, family_id FROM family_members_v1"))
        for event_id, record_json in con.execute(
            "SELECT event_id,record_json FROM timeline_events WHERE event_type='tallanto_student_snapshot'"
        ).fetchall():
            payload = json.loads(record_json)
            payload["record"]["payload"]["primary_email"] = "parent@example.com"
            con.execute("UPDATE timeline_events SET record_json=? WHERE event_id=?", (json.dumps(payload), event_id))

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id, family_id, membership_status, reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert len(set(original.values())) == 2
    assert len({row[1] for row in rows}) == 1
    assert rows[0][1] in set(original.values())
    assert {row[2] for row in rows} == {"confident"}
    assert {row[3] for row in rows} == {"exact_tallanto_parent_name_and_phone_or_email"}


def test_family_graph_attributes_tallanto_events_only_by_exact_student_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id, phone, student_name in (
        ("customer:left", "student-left", "+79000000011", "Анна Иванова"),
        ("customer:right", "student-right", "+79000000012", "Мария Иванова"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=phone)
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            student_id,
            "parent@example.com",
            student_name=student_name,
        )
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        right_customer, right_key = con.execute(
            "SELECT customer_id,child_key FROM family_links_v1 WHERE canonical_name='Мария Иванова'"
        ).fetchone()
        legacy_key = "child:stable-existing"
        con.execute(
            "UPDATE family_links_v1 SET child_key=? WHERE customer_id=? AND child_key=?",
            (legacy_key, right_customer, right_key),
        )
        con.execute(
            "UPDATE event_child_attribution_v1 SET child_key=? WHERE customer_id=? AND child_key=?",
            (legacy_key, right_customer, right_key),
        )

    event_records = (
        ("payment-exact", "tallanto_payment", {"contact_id": "student-right"}, "strong_unique"),
        ("abonement-exact", "tallanto_abonement", {"contact_id": "student-right"}, "manual"),
        ("attendance-xls-exact", "tallanto_attendance", {"logical_key": {"tallanto_id": "student-right"}}, "strong_unique"),
        ("attendance-api-exact", "tallanto_attendance", {"tallanto_student_id": "student-right"}, "manual"),
        ("payment-name-trap", "tallanto_payment", {"contact_id": "unknown-student"}, "strong_unique"),
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for source_id, event_type, record, match_status in event_records:
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id="customer:left",
                    event_type=event_type,
                    event_at=NOW,
                    source_system="tallanto_test",
                    source_id=source_id,
                    direction="system",
                    match_status=match_status,
                    summary="Оплата Марии Ивановой",
                    record=record,
                )
            )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT event.source_id, attribution.customer_id, attribution.child_key, attribution.reason
            FROM timeline_events AS event
            JOIN event_child_attribution_v1 AS attribution ON attribution.event_id=event.event_id
            WHERE event.source_id IN (?,?,?,?,?,?) ORDER BY event.source_id
            """,
            ("student-right", *(item[0] for item in event_records)),
        ).fetchall()
        stable_key = con.execute(
            "SELECT child_key FROM family_links_v1 WHERE customer_id=?",
            (right_customer,),
        ).fetchone()[0]
    exact = [row for row in rows if row[0] != "payment-name-trap"]
    name_trap = next(row for row in rows if row[0] == "payment-name-trap")
    assert stable_key == legacy_key
    assert {(row[1], row[2], row[3]) for row in exact} == {
        (right_customer, legacy_key, "exact_tallanto_identity")
    }
    assert name_trap[2:] == ("", "tallanto_student_id_not_in_family")

    monkeypatch.setattr(family_graph_module, "_event_tallanto_student_id", lambda _event: "")
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        disabled = con.execute(
            "SELECT child_key,reason FROM event_child_attribution_v1 AS attribution "
            "JOIN timeline_events AS event ON event.event_id=attribution.event_id "
            "WHERE event.source_id='payment-exact'"
        ).fetchone()
    assert disabled == ("", "missing_exact_tallanto_student_id")


def test_tallanto_payment_before_student_card_stays_in_explicit_quarantine(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:no-family", phone="+79000000091")
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        event = TimelineEvent(
            tenant_id="foton",
            customer_id="customer:no-family",
            event_type="tallanto_payment",
            event_at=NOW,
            source_system="tallanto_crm_call",
            source_id="payment-orphan",
            direction="system",
            match_status="strong_unique",
            record={"contact_id": "student-late", "amount": 1000},
        )
        store.upsert_event(event)

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT status,child_key,reason FROM event_child_attribution_v1 WHERE event_id=?",
            (event.event_id,),
        ).fetchone()
    assert row == ("ambiguous", "", "tallanto_student_id_not_in_family")


def test_master_contact_tallanto_summary_is_not_a_student_identity(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    customer_id = "customer:master-summary"
    _seed_customer(db_path, tmp_path, customer_id=customer_id, phone="+79000000092")
    _seed_tallanto_identity(
        db_path,
        tmp_path,
        customer_id,
        "student-real",
        "parent@example.com",
        student_name="Анна Иванова",
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        summary = TimelineEvent(
            tenant_id="foton",
            customer_id=customer_id,
            event_type="tallanto_student_snapshot",
            event_at=NOW,
            source_system="tallanto_snapshot",
            source_id=f"tallanto:{customer_id}",
            source_ref=f"master_contact:{customer_id}:tallanto",
            direction="system",
            match_status="strong_unique",
        )
        store.upsert_event(summary)

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT status,child_key,reason FROM event_child_attribution_v1 WHERE event_id=?",
            (summary.event_id,),
        ).fetchone()
    assert row == ("ambiguous", "", "missing_exact_tallanto_student_id")


def test_family_root_preserves_distinct_roots_on_shared_amo_contact_id(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:child", phone="+79000000031")
    _seed_tallanto_identity(db_path, tmp_path, "customer:child", "student-child", "parent@example.com")
    for customer_id, email in (
        ("customer:parent-left", "left@example.com"),
        ("customer:parent-right", "right@example.com"),
    ):
        _seed_amo_parent(
            db_path,
            tmp_path,
            customer_id=customer_id,
            display_name="Ирина Иванова",
            email=email,
        )
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    with sqlite3.connect(db_path) as con:
        original = dict(con.execute("SELECT customer_id, family_id FROM family_members_v1"))
        assert len(set(original.values())) == 3
        con.execute(
            "UPDATE identity_links SET link_value='parent@example.com' "
            "WHERE source_system='amocrm_snapshot' AND link_type='email'"
        )
        con.execute(
            "UPDATE identity_links SET link_value='amo-shared' "
            "WHERE source_system='amocrm_snapshot' AND link_type='amo_contact_id'"
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id, family_id, membership_status, reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert {row[0]: row[1] for row in rows} == original
    parent_rows = [row for row in rows if row[0].startswith("customer:parent-")]
    assert len(parent_rows) == 2
    assert {row[2] for row in parent_rows} == {"conflict"}
    assert {row[3] for row in parent_rows} == {"shared_amo_contact_across_customers"}


def test_family_graph_generated_at_ignores_future_source_rows(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:future", phone="+79000000011")
    _seed_event(db_path, tmp_path, customer_id="customer:future", source_id="future", summary="Тест.")
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE timeline_events SET created_at='2099-01-01T00:00:00+00:00'")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    assert report["generated_at"] != "2099-01-01T00:00:00+00:00"


def test_family_graph_never_marks_multiple_children_high_without_unique_mention(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:multi", phone="+79000000002")
    _seed_event(db_path, tmp_path, customer_id="customer:multi", source_id="call-1", summary="Нужно подобрать курс ребёнку.")
    _seed_event(db_path, tmp_path, customer_id="customer:multi", source_id="call-2", summary="Миша интересуется физикой.")
    _seed_event(
        db_path,
        tmp_path,
        customer_id="customer:multi",
        source_id="call-3",
        summary="Миша просил физику. Младшая дочь учится в 10 классе и хочет математику.",
    )
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:multi", phone="+79000000002")
    _insert_field(profiles_db, profile_id="customer:multi", field="child_name", value="Миша", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:multi", field="child_name", value="Даня", child_key="child_2")

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    with sqlite3.connect(db_path) as con:
        family_conf = con.execute("SELECT confidence, COUNT(*) FROM family_links_v1 GROUP BY confidence").fetchall()
        rows = con.execute(
            "SELECT status, confidence, reason FROM event_child_attribution_v1 ORDER BY event_id"
        ).fetchall()

    assert family_conf == [("medium", 2)]
    assert ("ambiguous", "low", "child_relevant_but_no_unique_name") in rows
    assert ("ambiguous", "low", "named_child_plus_other_child_reference") in rows
    assert ("matched", "medium", "unique_child_name_mention") in rows
    assert all(row[1] != "high" for row in rows)


def test_family_graph_blocks_named_child_plus_unknown_other_child(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:single-known", phone="+79000000012")
    _seed_event(
        db_path,
        tmp_path,
        customer_id="customer:single-known",
        source_id="call-mixed",
        summary="Миша просил физику. Младшая дочь хочет математику.",
    )
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:single-known", phone="+79000000012")
    _insert_field(profiles_db, profile_id="customer:single-known", field="child_name", value="Миша", child_key="child_1")

    build_family_graph(
        FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True)
    )

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT status, reason FROM event_child_attribution_v1 WHERE event_id IS NOT NULL"
        ).fetchone()
    assert row == ("ambiguous", "named_child_plus_other_child_reference")


def test_family_graph_merges_safe_name_variants_inside_family(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:daniel", phone="+79000000004")
    _seed_event(db_path, tmp_path, customer_id="customer:daniel", source_id="call-1", summary="Даня хочет курс по математике.")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:daniel", phone="+79000000004")
    _insert_field(profiles_db, profile_id="customer:daniel", field="child_name", value="Орёл Даниил", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:daniel", field="child_name", value="Дениил Романович Орел", child_key="child_2")
    _insert_field(profiles_db, profile_id="customer:daniel", field="child_name", value="Орлов Данил Романович", child_key="child_3")
    _insert_field(profiles_db, profile_id="customer:daniel", field="child_name", value="Дан", child_key="child_4")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    assert report["family_links_total"] == 1
    assert report["family_confidence_counts"]["high"] == 1
    with sqlite3.connect(db_path) as con:
        family = con.execute("SELECT canonical_name, status, confidence, name_variants_json FROM family_links_v1").fetchone()
        event = con.execute("SELECT status, confidence, reason FROM event_child_attribution_v1").fetchone()
    assert family[0] == "Орёл Даниил"
    assert family[1:3] == ("confident", "high")
    assert set(json.loads(family[3])) == {"Орёл Даниил", "Дениил Романович Орел", "Орлов Данил Романович", "Дан"}
    assert event == ("matched", "high", "unique_child_name_mention")


def test_family_graph_does_not_merge_different_children_with_similar_surname(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:siblings", phone="+79000000005")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:siblings", phone="+79000000005")
    _insert_field(profiles_db, profile_id="customer:siblings", field="child_name", value="Кулаков Никита", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:siblings", field="child_name", value="Кулакова Дарья", child_key="child_2")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    assert report["family_links_total"] == 2
    assert report["family_confidence_counts"]["medium"] == 2
    with sqlite3.connect(db_path) as con:
        names = [row[0] for row in con.execute("SELECT canonical_name FROM family_links_v1 ORDER BY canonical_name").fetchall()]
    assert names == ["Кулаков Никита", "Кулакова Дарья"]


def test_family_graph_does_not_merge_conflicting_full_names_through_patronymic_bridge(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:bridge", phone="+79000000009")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:bridge", phone="+79000000009")
    _insert_field(
        profiles_db,
        profile_id="customer:bridge",
        field="child_name",
        value="Иванов Даниил Сергеевич",
        child_key="child_1",
    )
    _insert_field(
        profiles_db,
        profile_id="customer:bridge",
        field="child_name",
        value="Даниил Сергеевич",
        child_key="child_2",
    )
    _insert_field(
        profiles_db,
        profile_id="customer:bridge",
        field="child_name",
        value="Петров Даниил Сергеевич",
        child_key="child_3",
    )

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    assert report["family_links_total"] == 3
    assert report["family_confidence_counts"] == {"low": 1, "medium": 2}
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT canonical_name, status, confidence, record_json FROM family_links_v1 ORDER BY canonical_name"
        ).fetchall()
    by_name = {row[0]: row for row in rows}
    assert by_name["Иванов Даниил Сергеевич"][1:3] == ("needs_review", "medium")
    assert by_name["Петров Даниил Сергеевич"][1:3] == ("needs_review", "medium")
    assert by_name["Даниил Сергеевич"][1:3] == ("excluded", "low")
    assert "ambiguous_patronymic_bridge" in json.loads(by_name["Даниил Сергеевич"][3])["suspicious_reasons"]


def test_family_graph_excludes_weak_one_off_child_candidate_on_identity_risk(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:partial", phone="+79000000007", identity_status="partial")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:partial", phone="+79000000007")
    _insert_field(profiles_db, profile_id="customer:partial", field="child_name", value="Никита", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:partial", field="child_name", value="Никита Афанаско", child_key="child_2")
    _insert_field(profiles_db, profile_id="customer:partial", field="child_name", value="Алина", child_key="child_3")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    assert report["family_status_counts"] == {"ambiguous": 1, "excluded": 1}
    with sqlite3.connect(db_path) as con:
        rows = {
            row[0]: json.loads(row[1])["suspicious_reasons"]
            for row in con.execute("SELECT canonical_name, record_json FROM family_links_v1")
        }
    assert rows["Алина"] == ["weak_one_off_child_candidate"]


def test_family_graph_keeps_one_off_full_name_without_duplicate_traits(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:two-kids", phone="+79000000008", identity_status="partial")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:two-kids", phone="+79000000008")
    _insert_field(profiles_db, profile_id="customer:two-kids", field="child_name", value="Кирилл", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:two-kids", field="child_name", value="Гусев Кирилл", child_key="child_2")
    _insert_field(profiles_db, profile_id="customer:two-kids", field="child_name", value="Александр Рязанов", child_key="child_3")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    assert report["family_status_counts"] == {"ambiguous": 2}
    with sqlite3.connect(db_path) as con:
        names = [row[0] for row in con.execute("SELECT canonical_name FROM family_links_v1 ORDER BY canonical_name").fetchall()]
    assert names == ["Александр Рязанов", "Гусев Кирилл"]


def test_family_graph_excludes_parent_like_and_initials_names(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:risk", phone="+79000000003")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:risk", phone="+79000000003")
    _insert_field(profiles_db, profile_id="customer:risk", field="parent_name", value="Татьяна Юрьевна", child_key="")
    _insert_field(profiles_db, profile_id="customer:risk", field="child_name", value="Синицына Татьяна", child_key="child_1")
    _insert_field(profiles_db, profile_id="customer:risk", field="child_name", value="Камаренцев Э.Н.", child_key="child_2")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    assert report["family_status_counts"]["excluded"] == 2
    with sqlite3.connect(db_path) as con:
        payloads = [json.loads(row[0]) for row in con.execute("SELECT record_json FROM family_links_v1")]
    reasons = {reason for payload in payloads for reason in payload["suspicious_reasons"]}
    assert "same_as_parent_name" in reasons
    assert "initials_possible_adult_or_teacher" in reasons


def test_family_graph_excludes_null_literal_child_name(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    _seed_customer(db_path, tmp_path, customer_id="customer:null", phone="+79000000006")
    profiles_db = _profiles_db(tmp_path)
    _insert_profile(profiles_db, profile_id="customer:null", phone="+79000000006")
    _insert_field(profiles_db, profile_id="customer:null", field="child_name", value="null", child_key="child_1")

    report = build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, profiles_db=profiles_db, apply=True))

    assert report["family_status_counts"]["excluded"] == 1
    with sqlite3.connect(db_path) as con:
        payload = json.loads(con.execute("SELECT record_json FROM family_links_v1").fetchone()[0])
    assert payload["canonical_name"] == "null"
    assert "empty_or_null_name" in payload["suspicious_reasons"]


def test_family_graph_apply_requires_staging_path(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path):
        pass

    with pytest.raises(ValueError, match=".codex_local/staging"):
        build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))


def _timeline_db(tmp_path: Path) -> Path:
    stage = tmp_path / ".codex_local" / "staging"
    stage.mkdir(parents=True)
    db_path = stage / "customer_timeline_staging.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path):
        pass
    return db_path


def _seed_customer(db_path: Path, tmp_path: Path, *, customer_id: str, phone: str, identity_status: str = "strong") -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status=identity_status,
                primary_phone=phone,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id=customer_id,
                opportunity_type="amo_deal",
                source_system="amocrm_snapshot",
                source_id=f"lead-{customer_id}",
                title="Курс для ребёнка",
                status="open",
                opened_at=NOW,
            )
        )


def _seed_event(db_path: Path, tmp_path: Path, *, customer_id: str, source_id: str, summary: str) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type="mango_call",
                event_at=NOW,
                source_system="mango_processed_summary",
                source_id=source_id,
                direction="inbound",
                subject="Звонок",
                text_preview=summary,
                summary=summary,
                match_status="strong_unique",
                importance=3,
                record={"summary": summary},
            )
        )


def _profiles_db(tmp_path: Path) -> Path:
    path = tmp_path / "profiles.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE customer_profiles (
              profile_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              primary_phone TEXT,
              display_name TEXT,
              built_at TEXT NOT NULL,
              build_id TEXT NOT NULL,
              source_event_count INTEGER NOT NULL,
              last_event_at TEXT
            );
            CREATE TABLE profile_fields (
              field_id TEXT PRIMARY KEY,
              profile_id TEXT NOT NULL,
              field TEXT NOT NULL,
              value TEXT NOT NULL,
              child_key TEXT NOT NULL DEFAULT '',
              brand TEXT NOT NULL DEFAULT 'unknown',
              source_system TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              event_at TEXT NOT NULL,
              quote TEXT NOT NULL DEFAULT '',
              superseded_by TEXT NOT NULL DEFAULT ''
            );
            """
        )
    return path


def _seed_tallanto_identity(
    db_path: Path,
    tmp_path: Path,
    customer_id: str,
    student_id: str,
    parent_email: str,
    parent_name: str = "Ирина Иванова",
    match_status: str = "strong_unique",
    student_name: str = "",
    student_phone: str = "",
    first_name: str = "",
    last_name: str = "",
    student_type: str = "",
) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for link_type, link_value in (("tallanto_student_id", student_id), ("email", parent_email)):
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type=link_type,
                    link_value=link_value,
                    source_system="tallanto_snapshot",
                    source_ref=f"{customer_id}:{student_id}:{link_type}",
                    match_class=(
                        IdentityMatchClass.STRONG_UNIQUE
                        if link_type == "tallanto_student_id"
                        else IdentityMatchClass.AMBIGUOUS
                    ),
                    confidence=1.0,
                )
            )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type="tallanto_student_snapshot",
                event_at=NOW,
                source_system="tallanto_snapshot",
                source_id=student_id,
                source_ref=f"tallanto:student:{student_id}",
                direction="system",
                match_status=match_status,
                record={
                    "payload": {
                        "parent_fio": parent_name,
                        "primary_email": parent_email,
                        **({"primary_phone": student_phone} if student_phone else {}),
                        **({"display_name": student_name} if student_name else {}),
                        **({"first_name": first_name} if first_name else {}),
                        **({"last_name": last_name} if last_name else {}),
                        **({"student_type": student_type} if student_type else {}),
                    }
                },
            )
        )


def _seed_amo_parent(
    db_path: Path,
    tmp_path: Path,
    *,
    customer_id: str,
    display_name: str,
    email: str,
) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status="strong",
                display_name=display_name,
                primary_email=email,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        for link_type, link_value in (("amo_contact_id", f"amo-{customer_id}"), ("email", email)):
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type=link_type,
                    link_value=link_value,
                    source_system="amocrm_snapshot",
                    source_ref=f"{customer_id}:{link_type}",
                    match_class=(
                        IdentityMatchClass.STRONG_UNIQUE
                        if link_type == "amo_contact_id"
                        else IdentityMatchClass.AMBIGUOUS
                    ),
                    confidence=1.0,
                )
            )


def _insert_profile(path: Path, *, profile_id: str, phone: str) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            "INSERT INTO customer_profiles VALUES (?, 'foton', ?, '', ?, 'test', 1, ?)",
            (profile_id, phone, NOW.isoformat(), NOW.isoformat()),
        )


def _insert_field(path: Path, *, profile_id: str, field: str, value: str, child_key: str) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            "INSERT INTO profile_fields VALUES (?, ?, ?, ?, ?, 'foton', 'fixture', ?, ?, '', '')",
            (f"{profile_id}:{field}:{child_key}:{value}", profile_id, field, value, child_key, f"src:{field}", NOW.isoformat()),
        )


def test_family_graph_readonly_connection_sees_active_wal(tmp_path: Path) -> None:
    db = tmp_path / "wal.sqlite"
    writer = sqlite3.connect(db)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE marker(value TEXT)")
        writer.commit()
        writer.execute("INSERT INTO marker VALUES ('visible')")
        writer.commit()
        assert Path(f"{db}-wal").exists()

        with _connect(db, write=False) as reader:
            value = reader.execute("SELECT value FROM marker").fetchone()[0]
    finally:
        writer.close()

    assert value == "visible"
