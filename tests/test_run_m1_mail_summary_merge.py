from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.run_m1_mail_summary_merge import M1MailMergeConfig, _mail_archive_envelope_for_row, _parse_args


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


def test_m1_mail_merge_accepts_explicit_tallanto_identity_db(tmp_path: Path) -> None:
    identity_db = tmp_path / "identity.sqlite"
    args = _parse_args(
        [
            "--timeline-db",
            str(tmp_path / ".codex_local" / "staging" / "timeline.sqlite"),
            "--prod-timeline-db",
            str(tmp_path / "prod.sqlite"),
            "--allowed-root",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--tallanto-identity-db",
            str(identity_db),
        ]
    )
    config = M1MailMergeConfig(
        archive=args.archive,
        external_manifest=args.external_manifest,
        timeline_db=args.timeline_db,
        prod_timeline_db=args.prod_timeline_db,
        allowed_root=args.allowed_root,
        out_dir=args.out_dir,
        tallanto_identity_db=args.tallanto_identity_db,
    )

    assert config.tallanto_identity_db == identity_db
