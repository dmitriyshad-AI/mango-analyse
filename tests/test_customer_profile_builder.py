from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_profile import CustomerProfileBuilder, CustomerProfileBuildOptions
from mango_mvp.customer_profile.build_cli import safe_field_preview
from mango_mvp.customer_profile.contracts import ProfileFieldCandidate
from mango_mvp.customer_profile.store import CustomerProfileSQLiteStore
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


def _timeline_db(tmp_path: Path, *, duplicate_phone_customer: bool = False) -> Path:
    db = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    customer = CustomerIdentity(
        tenant_id="foton",
        customer_id="cust-1",
        identity_status=IdentityStatus.STRONG,
        display_name="Клиент",
        primary_phone="+79990000000",
        source_ref="test",
        first_seen_at=NOW,
        last_seen_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="cust-1",
            link_type="phone",
            link_value="+79990000000",
            source_system="test",
            source_ref="test:phone",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=1.0,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )
    store.upsert_event(
        TimelineEvent(
            tenant_id="foton",
            customer_id="cust-1",
            event_type=TimelineEventType.MANGO_CALL,
            event_at=NOW,
            source_system="mango_processed_summary",
            source_id="100",
            source_ref="mango:100",
            direction=TimelineDirection.INBOUND,
            record={"brand": "foton"},
            created_at=NOW,
        )
    )
    if duplicate_phone_customer:
        second = CustomerIdentity(
            tenant_id="foton",
            customer_id="cust-2",
            identity_status=IdentityStatus.STRONG,
            display_name="Другой клиент",
            primary_phone="+79990000000",
            source_ref="test2",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        store.upsert_customer(second)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="cust-2",
                link_type="phone",
                link_value="+79990000000",
                source_system="test",
                source_ref="test2:phone",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
                first_seen_at=NOW,
                last_seen_at=NOW,
            )
        )
    store.close()
    return db


def _master_calls_db(tmp_path: Path, rows: list[tuple[int, str, dict]]) -> Path:
    db = tmp_path / "canonical_calls_master.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            """
            CREATE TABLE canonical_calls (
              canonical_call_id INTEGER PRIMARY KEY,
              phone TEXT,
              started_at TEXT,
              analysis_status TEXT,
              analysis_json TEXT
            )
            """
        )
        for call_id, started_at, analysis in rows:
            phone = analysis.pop("_phone", "+79990000000")
            con.execute(
                "INSERT INTO canonical_calls VALUES (?, ?, ?, ?, ?)",
                (call_id, phone, started_at, "done", json.dumps(analysis, ensure_ascii=False)),
            )
        con.commit()
    finally:
        con.close()
    return db


def _analysis(*, grade: str = "8", child_name: str = "Ребенок", subjects: list[str] | None = None) -> dict:
    return {
        "structured_fields": {
            "people": {"parent_fio": "Родитель", "child_fio": child_name},
            "student": {"grade_current": grade},
            "interests": {"subjects": subjects or ["математика"], "format": ["онлайн"]},
            "next_step": {"action": "Отправить материалы"},
        },
        "target_product": "годовые курсы",
        "objections": ["цена"],
    }


def _children_analysis(children: list[dict], *, parent_name: str = "Родитель") -> dict:
    return {
        "structured_fields": {
            "children": children,
            "people": {"parent_fio": parent_name},
            "interests": {},
            "next_step": {},
        },
        "target_product": "годовые курсы",
    }


def test_builder_marks_superseded_conflicting_grade_and_is_idempotent(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="8")),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"
    options = CustomerProfileBuildOptions(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=master_db,
        customer_ids=("cust-1",),
        build_id="test-build",
    )

    first = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_first = store.active_fields("cust-1")
        summary_first = store.summary()
    second = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_second = store.active_fields("cust-1")
        all_grade_rows = sqlite3.connect(profiles_db).execute(
            "SELECT value, superseded_by FROM profile_fields WHERE field='grade' ORDER BY event_at"
        ).fetchall()

    active_grade = [row for row in active_first if row["field"] == "grade"]
    assert first["build_id"] == "test-build"
    assert second["fields_written"] == first["fields_written"]
    assert active_first == active_second
    assert summary_first["counts"]["customer_profiles"] == 1
    assert active_grade[0]["value"] == "8"
    assert all_grade_rows[0][0] == "7"
    assert all_grade_rows[0][1]
    assert all_grade_rows[1][0] == "8"
    assert all_grade_rows[1][1] == ""


def test_builder_marks_duplicate_child_slots_as_merge_candidate(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7", child_name="Рома")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="8", child_name="Роман")),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"
    options = CustomerProfileBuildOptions(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=master_db,
        customer_ids=("cust-1",),
        build_id="test-build",
    )

    first = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_first = store.active_fields("cust-1")
    second = CustomerProfileBuilder(options).build()
    with CustomerProfileSQLiteStore(profiles_db) as store:
        active_second = store.active_fields("cust-1")

    child_keys = {
        row["child_key"]
        for row in active_first
        if row["field"] in {"child_name", "grade", "subject"}
    }
    marker_rows = [row for row in active_first if row["field"] == "child_slot_merge_candidate"]
    active_grade = [row for row in active_first if row["field"] == "grade"]

    assert len(child_keys) == 1
    assert len(marker_rows) == 1
    assert "merge_candidate" in marker_rows[0]["value"]
    assert active_grade[0]["value"] == "8"
    assert first["child_slot_merge"]["profiles_with_2plus_children_before"] == 1
    assert first["child_slot_merge"]["profiles_with_2plus_children_after"] == 0
    assert first["child_slot_merge"]["merge_candidate_groups"] == 1
    assert second["fields_written"] == first["fields_written"]
    assert active_second == active_first


def test_builder_does_not_merge_different_children_or_ambiguous_diminutive(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (
                100,
                "2026-01-10T10:00:00+00:00",
                _children_analysis(
                    [
                        {"name": "Ермаков Тимур", "grade": "7", "subjects": ["математика"]},
                        {"name": "Ермаков Олег", "grade": "7", "subjects": ["физика"]},
                    ]
                ),
            ),
            (
                101,
                "2026-02-10T10:00:00+00:00",
                _children_analysis(
                    [
                        {"name": "Саша", "grade": "8", "subjects": ["математика"]},
                        {"name": "Александр", "grade": "8", "subjects": ["физика"]},
                    ]
                ),
            ),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    assert report["child_slot_merge"]["merge_candidate_groups"] == 0
    assert not [row for row in fields if row["field"] == "child_slot_merge_candidate"]
    assert {row["value"] for row in fields if row["field"] == "child_name"} == {
        "Ермаков Тимур",
        "Ермаков Олег",
        "Саша",
        "Александр",
    }


def test_builder_child_slot_merge_stays_within_profile(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path, duplicate_phone_customer=True)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7", child_name="Рома")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="8", child_name="Роман")),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1", "cust-2"),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields_1 = store.active_fields("cust-1")
        fields_2 = store.active_fields("cust-2")

    assert report["ambiguous_calls"] == 2
    assert report["child_slot_merge"]["merge_candidate_groups"] == 0
    assert not [row for row in fields_1 + fields_2 if row["source_system"] == "mango_processed_summary"]


def test_builder_keeps_explicit_children_separate_by_child_key(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (
                100,
                "2026-01-10T10:00:00+00:00",
                {
                    "structured_fields": {
                        "children": [
                            {"name": "Первый", "grade": "7", "subjects": ["математика"]},
                            {"name": "Второй", "grade": "9", "subjects": ["физика"]},
                        ],
                        "people": {},
                        "interests": {},
                        "next_step": {},
                    }
                },
            )
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    names = {row["value"]: row["child_key"] for row in fields if row["field"] == "child_name"}
    grades = {row["child_key"]: row["value"] for row in fields if row["field"] == "grade"}
    assert set(names) == {"Первый", "Второй"}
    assert names["Первый"] != names["Второй"]
    assert grades[names["Первый"]] == "7"
    assert grades[names["Второй"]] == "9"


def test_builder_does_not_mix_single_child_from_different_calls(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(
        tmp_path,
        [
            (100, "2026-01-10T10:00:00+00:00", _analysis(grade="7", child_name="Первый")),
            (101, "2026-02-10T10:00:00+00:00", _analysis(grade="9", child_name="Второй", subjects=["физика"])),
        ],
    )
    profiles_db = tmp_path / "profiles.sqlite"

    CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    grades = {row["value"] for row in fields if row["field"] == "grade"}
    assert {"7", "9"}.issubset(grades)


def test_builder_skips_ambiguous_phone_and_counts_it(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path, duplicate_phone_customer=True)
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", _analysis())])
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1", "cust-2"),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields_1 = store.active_fields("cust-1")
        fields_2 = store.active_fields("cust-2")
    assert report["ambiguous_calls"] == 1
    assert not [row for row in fields_1 + fields_2 if row["source_system"] == "mango_processed_summary"]


def test_builder_matches_master_call_by_last_10_digits(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    analysis = _analysis()
    analysis["_phone"] = "9990000000"
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", analysis)])
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    assert report["unmatched_calls"] == 0
    with CustomerProfileSQLiteStore(profiles_db) as store:
        assert any(row["field"] == "grade" for row in store.active_fields("cust-1"))


def test_builder_counts_master_call_without_phone_as_unmatched(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    analysis = _analysis()
    analysis["_phone"] = None
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", analysis)])
    profiles_db = tmp_path / "profiles.sqlite"

    report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    with CustomerProfileSQLiteStore(profiles_db) as store:
        fields = store.active_fields("cust-1")

    assert report["unmatched_calls"] == 1
    assert not [row for row in fields if row["source_system"] == "mango_processed_summary"]


def test_store_enforces_profile_field_foreign_key(tmp_path: Path) -> None:
    profiles_db = tmp_path / "profiles.sqlite"
    with CustomerProfileSQLiteStore(profiles_db):
        pass
    con = sqlite3.connect(profiles_db)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                """
                INSERT INTO profile_fields (
                  field_id, profile_id, field, value, child_key, brand, source_system,
                  source_ref, event_at, quote, superseded_by
                ) VALUES ('f1', 'missing', 'grade', '8', '', 'unknown', 'test', 'test', ?, '', '')
                """,
                (NOW.isoformat(),),
            )
    finally:
        con.close()


def test_cli_field_preview_does_not_expose_raw_value() -> None:
    preview = safe_field_preview({"field": "child_name", "value": "Иван Петров", "brand": "foton"})

    assert preview["field"] == "child_name"
    assert preview["has_value"] is True
    assert preview["value_len"] == len("Иван Петров")
    assert "value" not in preview


def test_profile_field_requires_origin_and_timezone() -> None:
    with pytest.raises(ValueError, match="source_ref"):
        ProfileFieldCandidate(
            profile_id="cust-1",
            field="grade",
            value="8",
            source_system="mango_processed_summary",
            source_ref="",
            event_at=NOW,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        ProfileFieldCandidate(
            profile_id="cust-1",
            field="grade",
            value="8",
            source_system="mango_processed_summary",
            source_ref="mango:1",
            event_at=datetime(2026, 1, 1, 10, 0),
        )


def test_profile_build_records_timeline_sha256(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    master_db = _master_calls_db(tmp_path, [(100, "2026-01-10T10:00:00+00:00", _analysis())])
    profiles_db = tmp_path / "profiles.sqlite"

    CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_db,
            customer_ids=("cust-1",),
        )
    ).build()

    con = sqlite3.connect(profiles_db)
    try:
        row = con.execute(
            "SELECT build_id, timeline_db_sha256, profiles_built FROM profile_builds"
        ).fetchone()
    finally:
        con.close()
    assert row[0]
    assert len(row[1]) == 64
    assert row[2] == 1
