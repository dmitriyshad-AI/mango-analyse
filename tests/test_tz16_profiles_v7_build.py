from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_profile.contracts import ProfileFieldCandidate, ProfileSnapshot
from mango_mvp.customer_profile.store import CustomerProfileSQLiteStore
from scripts.build_tz16_profiles_v7 import (
    anonymized_examples,
    hash_directory,
    profile_content_signature,
    profile_metrics,
)


NOW = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)


def test_profile_metrics_counts_coverage_children_merge_and_superseded(tmp_path: Path) -> None:
    db = tmp_path / "profiles.sqlite"
    seed_profiles(db)

    metrics = profile_metrics(db)

    assert metrics["profile_count"] == 2
    assert metrics["coverage_profiles_by_field"]["child_name"] == 2
    assert metrics["coverage_profiles_by_field"]["grade"] == 1
    assert metrics["profiles_with_2plus_children"] == 1
    assert metrics["merge_candidate_profiles"] == 1
    assert metrics["superseded_fields"] == 1
    assert metrics["superseded_by_field"]["next_step"] == 1


def test_profile_metrics_open_db_under_path_with_space(tmp_path: Path) -> None:
    root = tmp_path / "profiles with space"
    root.mkdir()
    db = root / "profiles.sqlite"
    seed_profiles(db)

    metrics = profile_metrics(db)

    assert metrics["profile_count"] == 2
    assert metrics["coverage_profiles_by_field"]["child_name"] == 2


def test_anonymized_examples_do_not_expose_raw_values(tmp_path: Path) -> None:
    db = tmp_path / "profiles.sqlite"
    seed_profiles(db)

    examples = anonymized_examples(db, limit=5)
    raw = json.dumps(examples, ensure_ascii=False)

    assert len(examples) == 2
    assert "Анна" not in raw
    assert "+7999" not in raw
    assert "profile_example_1" in raw
    assert examples[0]["has_phone"] is True


def test_profile_content_signature_ignores_build_timestamps_for_same_content(tmp_path: Path) -> None:
    left = tmp_path / "left.sqlite"
    right = tmp_path / "right.sqlite"
    seed_profiles(left, build_id="left-build")
    seed_profiles(right, build_id="right-build")

    assert profile_content_signature(left) == profile_content_signature(right)


def test_hash_directory_detects_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.mkdir()
    (path / "a.txt").write_text("one", encoding="utf-8")
    before = hash_directory(path)
    (path / "a.txt").write_text("two", encoding="utf-8")
    after = hash_directory(path)

    assert before != after
    assert before["files"]["a.txt"]["sha256"] != after["files"]["a.txt"]["sha256"]


def seed_profiles(path: Path, *, build_id: str = "build") -> None:
    with CustomerProfileSQLiteStore(path) as store:
        store.replace_profiles(
            build_id=build_id,
            built_at=NOW,
            timeline_db_path=path,
            timeline_db_sha256="timeline",
            profiles=[
                ProfileSnapshot("profile-1", "foton", primary_phone="+79990000000", display_name="Анна", source_event_count=3),
                ProfileSnapshot("profile-2", "foton", primary_phone="", display_name="Без телефона", source_event_count=1),
            ],
            fields=[
                field("profile-1", "child_name", "Анна", child_key="child_1"),
                field("profile-1", "grade", "7", child_key="child_1"),
                field("profile-1", "child_name", "Аня", child_key="child_2"),
                field("profile-1", "child_slot_merge_candidate", "{\"marker\":\"merge_candidate\"}", child_key="child_1"),
                field("profile-1", "next_step", "старое", superseded_by="winner"),
                field("profile-1", "next_step", "новое", field_id="winner"),
                field("profile-2", "child_name", "Иван", child_key="child_1"),
            ],
        )


def field(
    profile_id: str,
    name: str,
    value: str,
    *,
    child_key: str = "",
    field_id: str | None = None,
    superseded_by: str = "",
) -> ProfileFieldCandidate:
    return ProfileFieldCandidate(
        profile_id=profile_id,
        field=name,
        value=value,
        child_key=child_key,
        source_system="test",
        source_ref=f"test:{profile_id}:{name}:{child_key}:{value}",
        event_at=NOW,
        field_id=field_id,
        superseded_by=superseded_by,
    )
