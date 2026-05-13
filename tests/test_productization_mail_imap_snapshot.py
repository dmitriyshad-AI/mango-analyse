from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from mango_mvp.productization.mail_imap_snapshot import (
    HEADER_FETCH_QUERY,
    MailImapCredentials,
    MailImapSnapshotConfig,
    build_mail_imap_snapshot,
    decode_imap_modified_utf7,
    parse_mailbox_list_line,
    quote_imap_mailbox_name,
)
from scripts import mango_office_mail_imap_snapshot


class FakeImapClient:
    def __init__(self) -> None:
        self.logged_in: tuple[str, str] | None = None
        self.selected: list[tuple[str, bool]] = []
        self.searches: list[tuple[str, tuple[str, ...]]] = []
        self.fetches: list[tuple[bytes | str, str]] = []
        self.closed = 0
        self.logged_out = False

    def login(self, user: str, password: str) -> tuple[str, Sequence[bytes]]:
        self.logged_in = (user, password)
        return "OK", [b"logged in"]

    def list(self) -> tuple[str, Sequence[bytes]]:
        return (
            "OK",
            [
                b'(\\HasNoChildren) "/" "INBOX"',
                b'(\\HasNoChildren) "/" "&BBsEEgQo-"',
                b'(\\Noselect \\HasChildren) "/" "Archive"',
            ],
        )

    def select(self, mailbox: str, readonly: bool = False) -> tuple[str, Sequence[bytes]]:
        self.selected.append((mailbox, readonly))
        totals = {'"INBOX"': b"3", '"&BBsEEgQo-"': b"2"}
        return "OK", [totals[mailbox]]

    def search(self, charset: Optional[str], *criteria: str) -> tuple[str, Sequence[bytes]]:
        self.searches.append((str(charset), criteria))
        if criteria == ("UNSEEN",):
            return "OK", [b"2"]
        if criteria[0] == "SINCE":
            return "OK", [b"1 2"]
        raise AssertionError(f"Unexpected search criteria: {criteria}")

    def fetch(self, message_set: bytes | str, message_parts: str) -> tuple[str, Sequence[Any]]:
        self.fetches.append((message_set, message_parts))
        assert message_parts == HEADER_FETCH_QUERY
        assert "BODY.PEEK" in message_parts
        assert "BODY[TEXT]" not in message_parts
        payload = (
            b"Message-ID: <m-1@example.local>\r\n"
            b"Date: Tue, 12 May 2026 10:00:00 +0300\r\n"
            b"From: Sender <sender@example.local>\r\n"
            b"To: edu@kmipt.ru\r\n"
            b"Subject: =?utf-8?b?0KLQtdGB0YLQvtCy0L7QtSDQv9C40YHRjNC80L4=?=\r\n"
            b"\r\n"
        )
        return "OK", [(b"1 (BODY[HEADER] {128}", payload)]

    def close(self) -> tuple[str, Sequence[bytes]]:
        self.closed += 1
        return "OK", [b"closed"]

    def logout(self) -> tuple[str, Sequence[bytes]]:
        self.logged_out = True
        return "BYE", [b"logout"]


def test_decode_imap_modified_utf7_decodes_reg_ru_folder_names() -> None:
    assert decode_imap_modified_utf7("&BBsEEgQo-") == "ЛВШ"
    assert decode_imap_modified_utf7("&BBcEEgQo-") == "ЗВШ"
    assert decode_imap_modified_utf7("&BCgEMAQxBDsEPgQ9BEs-") == "Шаблоны"
    assert decode_imap_modified_utf7("Archive &- More") == "Archive & More"


def test_parse_mailbox_list_line_keeps_raw_name_and_decodes_display_name() -> None:
    parsed = parse_mailbox_list_line(b'(\\HasNoChildren \\UnMarked) "." "&BBsEEgQo-"')

    assert parsed["name_raw"] == '"&BBsEEgQo-"'
    assert parsed["name"] == "ЛВШ"
    assert parsed["flags"] == ["\\HasNoChildren", "\\UnMarked"]
    assert parsed["delimiter"] == "."


def test_quote_imap_mailbox_name_quotes_spaces_and_escapes_quotes() -> None:
    assert quote_imap_mailbox_name("INBOX") == "INBOX"
    assert quote_imap_mailbox_name('"&BBsEEgQo-"') == '"&BBsEEgQo-"'
    assert quote_imap_mailbox_name("Sent Messages") == '"Sent Messages"'
    assert quote_imap_mailbox_name('Тариф "Премиум 10"') == '"Тариф \\"Премиум 10\\""'


def test_build_mail_imap_snapshot_is_read_only_and_header_only(tmp_path: Path) -> None:
    fake = FakeImapClient()
    out_dir = tmp_path / "mail_snapshot"

    report = build_mail_imap_snapshot(
        credentials=MailImapCredentials(
            host="mail.hosting.reg.ru",
            port=993,
            email_address="edu@kmipt.ru",
            password="secret-password",
        ),
        config=MailImapSnapshotConfig(
            out_dir=out_dir,
            since_days=30,
            header_sample_limit_per_mailbox=1,
            account_label="regru_edu",
        ),
        client=fake,
    )

    assert report["schema_version"] == "mail_imap_snapshot_v1"
    assert report["mailbox_count"] == 3
    assert report["total_messages"] == 5
    assert report["total_since"] == 4
    assert report["header_sample_rows"] == 2
    assert report["safety"] == {
        "readonly_select": True,
        "download_bodies": False,
        "download_attachments": False,
        "send_mail": False,
        "delete_or_move_mail": False,
        "write_crm": False,
    }
    assert fake.selected == [('"INBOX"', True), ('"&BBsEEgQo-"', True)]
    assert fake.closed == 2
    assert fake.logged_out is True
    assert all(fetch_query == HEADER_FETCH_QUERY for _msg_id, fetch_query in fake.fetches)

    report_text = (out_dir / "imap_dry_run_report.json").read_text(encoding="utf-8")
    headers_text = (out_dir / "imap_headers_sample.jsonl").read_text(encoding="utf-8")
    assert "secret-password" not in report_text
    assert "secret-password" not in headers_text
    assert "Тестовое письмо" in headers_text
    assert '"mailbox": "ЛВШ"' in headers_text
    assert json.loads(report_text)["paths"]["headers_sample"].endswith("imap_headers_sample.jsonl")


def test_mail_imap_snapshot_cli_requires_password_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MAIL_IMAP_PASSWORD", raising=False)
    monkeypatch.setenv("MAIL_IMAP_EMAIL", "edu@kmipt.ru")

    rc = mango_office_mail_imap_snapshot.main(["--out-dir", str(tmp_path / "out")])

    assert rc == 2
