from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.contracts import CustomerIdentity, CustomerOpportunity, TimelineEvent
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
        event = con.execute("SELECT status, confidence, reason, child_key FROM event_child_attribution_v1").fetchone()
    assert family == ("Аня", "confident", "high")
    assert event[0:3] == ("matched", "high", "single_child_family")
    assert event[3]


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
    assert ("matched", "medium", "unique_child_name_mention") in rows
    assert all(row[1] != "high" for row in rows)


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
