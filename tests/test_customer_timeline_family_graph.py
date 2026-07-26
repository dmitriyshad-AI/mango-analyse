from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityMatchClass,
    TimelineEvent,
)
from mango_mvp.customer_timeline.family_graph import FamilyGraphConfig, build_family_graph
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


def test_family_graph_groups_tallanto_siblings_by_parent_email(tmp_path: Path) -> None:
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


def test_family_graph_does_not_merge_shared_email_with_different_parents(tmp_path: Path) -> None:
    db_path = _timeline_db(tmp_path)
    for customer_id, student_id, parent_name in (
        ("customer:left", "student-left", "Ирина Иванова"),
        ("customer:right", "student-right", "Мария Петрова"),
    ):
        _seed_customer(db_path, tmp_path, customer_id=customer_id, phone=f"+7900000000{len(customer_id)}")
        _seed_tallanto_identity(
            db_path,
            tmp_path,
            customer_id,
            student_id,
            "shared@example.com",
            parent_name,
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT family_id, membership_status FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert len({row[0] for row in rows}) == 2
    assert {row[1] for row in rows} == {"singleton"}


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


def test_family_root_does_not_merge_two_persisted_roots(tmp_path: Path) -> None:
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
        con.execute(
            "UPDATE identity_links SET link_value='parent@example.com' "
            "WHERE source_system='tallanto_snapshot' AND link_type='email'"
        )

    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT customer_id, family_id, membership_status, reason FROM family_members_v1 ORDER BY customer_id"
        ).fetchall()
    assert {row[0]: row[1] for row in rows} == original
    assert {row[2] for row in rows} == {"conflict"}
    assert {row[3] for row in rows} == {"conflicting_persisted_family_roots"}


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
                direction="system",
                match_status=match_status,
                record={"payload": {"parent_fio": parent_name}},
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
