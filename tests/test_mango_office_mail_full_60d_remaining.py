from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.mango_office_mail_full_60d_remaining import (
    iter_age_windows,
    is_batch_complete,
    load_existing_completed_batch,
    prune_archive_rows_before,
)


def test_full_mail_batch_complete_allows_live_plan_growth_under_limit() -> None:
    report = {
        "errors": [],
        "messages_found_since": 4,
        "messages_attempted": 4,
        "messages_inserted_or_seen": 4,
        "messages_excluded_by_sha256": 0,
    }

    assert is_batch_complete(report, {"verification_pass": True}, max_messages=250) is True


def test_full_mail_batch_complete_blocks_when_live_count_exceeds_limit() -> None:
    report = {
        "errors": [],
        "messages_found_since": 251,
        "messages_attempted": 250,
        "messages_inserted_or_seen": 250,
        "messages_excluded_by_sha256": 0,
    }

    assert is_batch_complete(report, {"verification_pass": True}, max_messages=250) is False


def test_full_mail_batch_complete_requires_all_attempted_messages_accounted() -> None:
    report = {
        "errors": [],
        "messages_found_since": 4,
        "messages_attempted": 4,
        "messages_inserted_or_seen": 3,
        "messages_excluded_by_sha256": 0,
    }

    assert is_batch_complete(report, {"verification_pass": True}, max_messages=250) is False


def test_iter_age_windows_supports_sparse_old_mail_ranges() -> None:
    assert iter_age_windows(since_days=730, older_than_days=365, window_days=31)[:3] == [
        (730, 699),
        (699, 668),
        (668, 637),
    ]
    assert iter_age_windows(since_days=370, older_than_days=365, window_days=31) == [
        (370, 365)
    ]


def test_load_existing_completed_batch_requires_matching_plan_count(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch_reports"
    batch_dir.mkdir()
    report = {
        "errors": [],
        "messages_found_since": 4,
        "messages_attempted": 4,
        "messages_inserted_or_seen": 3,
        "messages_excluded_by_sha256": 1,
    }
    verification = {"verification_pass": True}
    (batch_dir / "INBOX_d7_to_d6_ingest.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    (batch_dir / "INBOX_d7_to_d6_verification.json").write_text(
        json.dumps(verification),
        encoding="utf-8",
    )

    assert (
        load_existing_completed_batch(
            batch_dir,
            "INBOX_d7_to_d6",
            max_messages=250,
            planned_message_count=4,
        )
        == (report, verification)
    )
    assert (
        load_existing_completed_batch(
            batch_dir,
            "INBOX_d7_to_d6",
            max_messages=250,
            planned_message_count=5,
        )
        is None
    )


def test_prune_archive_rows_before_removes_only_stale_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "mail_archive.sqlite"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE messages (sha256 TEXT PRIMARY KEY, updated_at TEXT NOT NULL);
            CREATE TABLE message_sources (
              source_key TEXT PRIMARY KEY,
              message_sha256 TEXT NOT NULL
            );
            CREATE TABLE message_participants (message_sha256 TEXT NOT NULL);
            CREATE TABLE attachments (message_sha256 TEXT NOT NULL);
            CREATE TABLE message_matches (message_sha256 TEXT PRIMARY KEY);
            """
        )
        con.executemany(
            "INSERT INTO messages (sha256, updated_at) VALUES (?, ?)",
            [
                ("old", "2026-05-13T00:00:00+00:00"),
                ("new", "2026-05-13T01:00:00+00:00"),
            ],
        )
        con.executemany(
            "INSERT INTO message_sources (source_key, message_sha256) VALUES (?, ?)",
            [("s-old", "old"), ("s-new", "new")],
        )
        con.executemany(
            "INSERT INTO message_participants (message_sha256) VALUES (?)",
            [("old",), ("new",)],
        )
        con.executemany(
            "INSERT INTO attachments (message_sha256) VALUES (?)",
            [("old",), ("new",)],
        )
        con.executemany(
            "INSERT INTO message_matches (message_sha256) VALUES (?)",
            [("old",), ("new",)],
        )
        con.commit()
    (tmp_path / "raw_eml" / "ol").mkdir(parents=True)
    (tmp_path / "raw_eml" / "ol" / "old.eml").write_bytes(b"raw")
    (tmp_path / "raw_eml" / "ne").mkdir(parents=True)
    (tmp_path / "raw_eml" / "ne" / "new.eml").write_bytes(b"raw")
    (tmp_path / "extracted_text").mkdir()
    (tmp_path / "extracted_text" / "old.txt").write_text("text", encoding="utf-8")
    (tmp_path / "extracted_text" / "new.txt").write_text("text", encoding="utf-8")
    (tmp_path / "attachments" / "old").mkdir(parents=True)
    (tmp_path / "attachments" / "old" / "part.bin").write_bytes(b"attachment")
    (tmp_path / "attachments" / "new").mkdir(parents=True)
    (tmp_path / "attachments" / "new" / "part.bin").write_bytes(b"attachment")

    report = prune_archive_rows_before(
        db_path,
        cutoff_iso="2026-05-13T00:30:00+00:00",
    )

    assert report == {
        "attachment_dirs": 1,
        "attachment_files": 1,
        "messages": 1,
        "message_sources": 1,
        "message_participants": 1,
        "attachments": 1,
        "message_matches": 1,
        "raw_eml_files": 1,
        "text_files": 1,
    }
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT sha256 FROM messages").fetchall() == [("new",)]
        assert con.execute("SELECT message_sha256 FROM attachments").fetchall() == [("new",)]
    assert not (tmp_path / "raw_eml" / "ol" / "old.eml").exists()
    assert not (tmp_path / "extracted_text" / "old.txt").exists()
    assert not (tmp_path / "attachments" / "old").exists()
    assert (tmp_path / "raw_eml" / "ne" / "new.eml").exists()
    assert (tmp_path / "extracted_text" / "new.txt").exists()
    assert (tmp_path / "attachments" / "new" / "part.bin").exists()
