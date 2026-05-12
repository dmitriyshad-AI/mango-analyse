from __future__ import annotations

import csv
import sqlite3
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Sequence

import pytest

from mango_mvp.productization.mail_archive import (
    FULL_MESSAGE_FETCH_QUERY,
    MailArchiveIngestConfig,
    MailMatchingReportConfig,
    TallantoIdentityMapConfig,
    build_mail_archive_ingest,
    build_mail_matching_report,
    build_tallanto_identity_map,
    extract_email_addresses,
    extract_phone_numbers,
    normalize_email,
    normalize_phone,
)
from mango_mvp.productization.mail_imap_snapshot import MailImapCredentials


def test_mail_archive_identity_ingest_and_matching_are_read_only(tmp_path: Path) -> None:
    tallanto_csv = tmp_path / "students.csv"
    _write_tallanto_csv(
        tallanto_csv,
        [
            {
                "ID": "T-1",
                "amoCRM ID": "A-1",
                "Имя": "Анна",
                "Фамилия": "Иванова",
                "ФИО родителя": "Мария Иванова",
                "Тип ученика": "active",
                "Ответственный(ая)": "Настя",
                "Ответственный(ая) (ID)": "M-1",
                "Группа(ID)": "G-1",
                "E-mail": "Client <client@example.com>",
                "Другой E-mail": "",
                "Тел. (родителя)": "+7 (999) 000-00-00",
            }
        ],
    )

    identity_report = build_tallanto_identity_map(
        TallantoIdentityMapConfig(
            tallanto_csv_path=tallanto_csv,
            out_dir=tmp_path / "identity",
            encoding="utf-8",
            delimiter="\t",
        )
    )
    assert identity_report["row_count"] == 1
    assert identity_report["identity_values"]["email"]["strong_unique"] == 1
    assert identity_report["privacy"]["raw_export_copied"] is False

    fake_imap = FakeImapClient([_raw_message()])
    ingest_report = build_mail_archive_ingest(
        credentials=MailImapCredentials(
            host="mail.example.test",
            port=993,
            email_address="school@kmipt.ru",
            password="not-written",
        ),
        config=MailArchiveIngestConfig(
            out_dir=tmp_path / "archive",
            mailbox="INBOX",
            mailbox_label="INBOX",
            since_days=7,
            max_messages=10,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )

    assert ingest_report["safety"]["fetch_uses_body_peek"] is True
    assert ingest_report["safety"]["send_mail"] is False
    assert ingest_report["safety"]["delete_or_move_mail"] is False
    assert ingest_report["safety"]["password_written"] is False
    assert fake_imap.selected_readonly == [("INBOX", True)]
    assert fake_imap.fetch_queries == [FULL_MESSAGE_FETCH_QUERY]
    assert ingest_report["messages_inserted_or_seen"] == 1
    assert ingest_report["raw_eml_written"] == 1
    assert ingest_report["text_files_written"] == 1

    archive_db = Path(ingest_report["paths"]["archive_db"])
    with sqlite3.connect(archive_db) as con:
        con.row_factory = sqlite3.Row
        message = con.execute("select * from messages").fetchone()
        assert message["message_kind"] == "external"
        participants = list(con.execute("select email_normalized from message_participants order by email_normalized"))
        assert [row["email_normalized"] for row in participants] == ["client@example.com", "school@kmipt.ru"]

    match_report = build_mail_matching_report(
        MailMatchingReportConfig(
            archive_db_path=archive_db,
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "matching",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )

    assert match_report["message_count"] == 1
    assert match_report["counts"]["strong_unique"] == 1
    assert match_report["matched_message_count"] == 1
    assert match_report["privacy"]["raw_emails_written"] is False


def test_mail_archive_normalizers_and_stable_runtime_guards(tmp_path: Path) -> None:
    assert normalize_email("Client <CLIENT@Example.COM>") == "client@example.com"
    assert extract_email_addresses("a@example.com; B <b@example.com>") == ["a@example.com", "b@example.com"]
    assert normalize_phone("8 (999) 000-00-00") == "+79990000000"
    assert extract_phone_numbers("main +7 999 000-00-00 / +7 999 111-22-33") == [
        "+79990000000",
        "+79991112233",
    ]

    tallanto_csv = tmp_path / "students.csv"
    _write_tallanto_csv(tallanto_csv, [{"ID": "T-1", "E-mail": "client@example.com"}])

    with pytest.raises(ValueError, match="stable_runtime"):
        build_tallanto_identity_map(
            TallantoIdentityMapConfig(
                tallanto_csv_path=tallanto_csv,
                out_dir=tmp_path / "stable_runtime" / "mail_identity",
                encoding="utf-8",
                delimiter="\t",
            )
        )


def test_mail_archive_identity_map_duplicate_handling(tmp_path: Path) -> None:
    tallanto_csv = tmp_path / "students.csv"
    _write_tallanto_csv(
        tallanto_csv,
        [
            {"ID": "T-1", "E-mail": "shared@example.com"},
            {"ID": "T-2", "Другой E-mail": "shared@example.com"},
            {"ID": "T-3", "E-mail": ""},
        ],
    )

    report = build_tallanto_identity_map(
        TallantoIdentityMapConfig(
            tallanto_csv_path=tallanto_csv,
            out_dir=tmp_path / "identity",
            encoding="utf-8",
            delimiter="\t",
        )
    )

    assert report["identity_values"]["email"] == {"duplicate": 1, "strong_unique": 0}
    assert report["row_identity_classes"]["email"] == {
        "duplicate": 2,
        "missing": 1,
        "strong_unique": 0,
    }
    with sqlite3.connect(tmp_path / "identity" / "tallanto_email_identity_map.sqlite") as con:
        row = con.execute(
            "select match_class, candidate_count from identity_values where kind = 'email'"
        ).fetchone()
        assert row == ("duplicate", 2)


def test_mail_archive_ingest_rerun_is_idempotent_and_does_not_leak_password(
    tmp_path: Path,
) -> None:
    fake_imap = FakeImapClient([_raw_message()])
    credentials = MailImapCredentials(
        host="mail.example.test",
        port=993,
        email_address="school@kmipt.ru",
        password="not-written",
    )
    config = MailArchiveIngestConfig(
        out_dir=tmp_path / "archive",
        mailbox="INBOX",
        mailbox_label="INBOX",
        since_days=7,
        max_messages=10,
        account_label="test",
        internal_domains=("kmipt.ru",),
    )

    first = build_mail_archive_ingest(credentials=credentials, config=config, client=fake_imap)
    second = build_mail_archive_ingest(credentials=credentials, config=config, client=fake_imap)

    assert first["raw_eml_written"] == 1
    assert first["attachments_written"] == 1
    assert second["raw_eml_written"] == 0
    assert second["attachments_written"] == 0
    assert second["text_files_written"] == 0

    archive_db = tmp_path / "archive" / "mail_archive.sqlite"
    with sqlite3.connect(archive_db) as con:
        assert con.execute("select count(*) from messages").fetchone()[0] == 1
        assert con.execute("select count(*) from message_sources").fetchone()[0] == 1
        assert con.execute("select count(*) from attachments").fetchone()[0] == 1

    assert "not-written" not in (tmp_path / "archive" / "mail_ingest_report.json").read_text(
        encoding="utf-8"
    )
    assert b"not-written" not in archive_db.read_bytes()


class FakeImapClient:
    def __init__(self, messages: list[bytes]) -> None:
        self.messages = messages
        self.selected_readonly: list[tuple[str, bool]] = []
        self.fetch_queries: list[str] = []

    def login(self, user: str, password: str) -> tuple[str, Sequence[bytes]]:
        assert user == "school@kmipt.ru"
        assert password == "not-written"
        return "OK", [b"logged in"]

    def list(self) -> tuple[str, Sequence[bytes]]:
        return "OK", []

    def select(self, mailbox: str, readonly: bool = False) -> tuple[str, Sequence[bytes]]:
        self.selected_readonly.append((mailbox, readonly))
        return "OK", [str(len(self.messages)).encode("ascii")]

    def search(self, charset: str | None, *criteria: str) -> tuple[str, Sequence[bytes]]:
        assert charset is None
        assert criteria[0] == "SINCE"
        return "OK", [b" ".join(str(index).encode("ascii") for index in range(1, len(self.messages) + 1))]

    def fetch(self, message_set: bytes | str, message_parts: str) -> tuple[str, Sequence[Any]]:
        self.fetch_queries.append(message_parts)
        index = int(message_set) - 1
        return "OK", [(b"1 FETCH", self.messages[index])]

    def close(self) -> tuple[str, Sequence[bytes]]:
        return "OK", []

    def logout(self) -> tuple[str, Sequence[bytes]]:
        return "OK", []


def _raw_message() -> bytes:
    message = EmailMessage()
    message["Message-ID"] = "<m-1@example.com>"
    message["Date"] = "Tue, 12 May 2026 10:00:00 +0300"
    message["From"] = "Client <client@example.com>"
    message["To"] = "School <school@kmipt.ru>"
    message["Subject"] = "Материалы по лагерю"
    message.set_content("Здравствуйте, пришлите материалы по летнему лагерю.")
    message.add_attachment(
        b"synthetic attachment bytes",
        maintype="application",
        subtype="octet-stream",
        filename="attachment.bin",
    )
    return message.as_bytes()


def _write_tallanto_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
