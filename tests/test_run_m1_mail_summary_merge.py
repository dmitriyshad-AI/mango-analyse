from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.run_m1_mail_summary_merge import _mail_archive_envelope_for_row


def _write_archive(tmp_path: Path, *, sha: str) -> Path:
    archive_db = tmp_path / "archive" / "mail_archive.sqlite"
    archive_db.parent.mkdir(parents=True)
    with sqlite3.connect(archive_db) as con:
        con.executescript(
            """
            CREATE TABLE message_participants (
              message_sha256 TEXT,
              header_name TEXT,
              display_name TEXT,
              email_normalized TEXT,
              domain TEXT
            );
            """
        )
        con.execute(
            "INSERT INTO message_participants VALUES (?, 'from', 'Parent', 'parent@example.com', 'example.com')",
            (sha,),
        )
        con.execute(
            "INSERT INTO message_participants VALUES (?, 'to', 'Foton', 'edu@kmipt.ru', 'kmipt.ru')",
            (sha,),
        )
    return archive_db


def test_m1_mail_merge_reads_email_envelope_from_stage2_record_archive_db(tmp_path: Path) -> None:
    sha = "d" * 64
    archive_db = _write_archive(tmp_path, sha=sha)

    envelope = _mail_archive_envelope_for_row(
        {"record": {"stage2_enrich_archive_db": str(archive_db)}},
        message_sha=sha,
        direction="inbound",
        archive_cache={},
    )

    assert envelope["contact_email"] == "parent@example.com"
    assert envelope["contact_source"] == "header_from"
    assert envelope["contact_reason"] == "inbound_external_from"
    assert envelope["from_email"] == "parent@example.com"
    assert envelope["from_domain"] == "example.com"
    assert envelope["to_emails"] == ["edu@kmipt.ru"]
    assert envelope["to_domains"] == ["kmipt.ru"]
