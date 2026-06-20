from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import subprocess
import zipfile
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from mango_mvp.productization.mail_archive import (
    FULL_MESSAGE_FETCH_QUERY,
    MailArchiveIngestConfig,
    MailArchivePreflightConfig,
    MailArchiveVerificationConfig,
    MailAttachmentImageOcrPlanConfig,
    MailAttachmentOcrPreflightConfig,
    MailAttachmentOcrPilotConfig,
    MailAttachmentParsePlanConfig,
    MailAttachmentPdfExtractConfig,
    MailAttachmentStage6PlanConfig,
    MailAttachmentTextExtractConfig,
    MailAttachmentTextIndexConfig,
    MailCustomerHistoryHandoffConfig,
    MailCustomerRelinkPreviewConfig,
    MailMangoBridgePreviewConfig,
    MailMatchingReportConfig,
    MailPhoneLiftPreviewConfig,
    MangoPhoneIndexPreviewConfig,
    TallantoIdentityMapConfig,
    build_mail_archive_ingest,
    build_mail_archive_preflight,
    build_mail_attachment_image_ocr_plan,
    build_mail_attachment_ocr_preflight,
    build_mail_attachment_ocr_pilot,
    build_mail_attachment_parse_plan,
    build_mail_attachment_pdf_extract,
    build_mail_attachment_stage6_plan,
    build_mail_attachment_text_extract,
    build_mail_attachment_text_index,
    build_mail_customer_history_handoff,
    build_mail_customer_relink_preview,
    build_mail_mango_bridge_preview,
    build_mail_matching_report,
    build_mail_phone_lift_preview,
    build_mango_phone_index_preview,
    build_tallanto_identity_map,
    extract_email_addresses,
    extract_phone_numbers,
    iter_attachment_parts,
    is_transient_imap_fetch_error,
    message_metadata,
    message_participants,
    normalize_extracted_attachment_text,
    normalize_email,
    normalize_phone,
    open_imap_client_with_retries,
    run_tesseract_ocr,
    verify_mail_archive_pilot,
)
from mango_mvp.productization.mail_imap_snapshot import MailImapCredentials
from scripts import mango_office_mail_archive


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
    assert identity_report["row_coverage"]["rows_with_any_strong_email_or_phone"] == 1
    assert all(identity_report["sanity_checks"].values())
    assert identity_report["audit_readiness"]["pass"] is True
    assert len(identity_report["source_file"]["sha256"]) == 64
    assert len(identity_report["artifact_integrity"]["identity_db_sha256"]) == 64
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

    rerun_match_report = build_mail_matching_report(
        MailMatchingReportConfig(
            archive_db_path=archive_db,
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "matching",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )
    assert rerun_match_report["counts"] == match_report["counts"]
    with sqlite3.connect(archive_db) as con:
        assert con.execute("select count(*) from message_matches").fetchone()[0] == 1

    handoff = build_mail_customer_history_handoff(
        MailCustomerHistoryHandoffConfig(
            archive_db_paths=[archive_db],
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "history_handoff",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )
    assert handoff["message_count"] == 1
    assert handoff["counts"]["strong_unique"] == 1
    assert handoff["safety"]["write_crm"] is False
    assert handoff["safety"]["write_tallanto"] is False
    assert handoff["privacy"]["raw_emails_written"] is False
    with sqlite3.connect(tmp_path / "history_handoff" / "mail_customer_history_handoff.sqlite") as con:
        assert con.execute("select count(*) from v_strong_customer_mail_links").fetchone()[0] == 1
        assert con.execute("select count(*) from v_manual_review_mail_links").fetchone()[0] == 0


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
            {
                "ID": "T-1",
                "E-mail": "shared@example.com",
                "Тел. цифровой (моб.)": "+7 999 000-00-00",
            },
            {
                "ID": "T-2",
                "Другой E-mail": "shared@example.com",
                "Тел. (родителя)": "8 999 000 00 00",
            },
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
    assert report["identity_values"]["phone"] == {"duplicate": 1, "strong_unique": 0}
    assert report["row_identity_classes"]["phone"] == {
        "duplicate": 2,
        "missing": 1,
        "strong_unique": 0,
    }
    with sqlite3.connect(tmp_path / "identity" / "tallanto_email_identity_map.sqlite") as con:
        row = con.execute(
            "select match_class, candidate_count from identity_values where kind = 'email'"
        ).fetchone()
        assert row == ("duplicate", 2)
        row = con.execute(
            "select match_class, candidate_count from identity_values where kind = 'phone'"
        ).fetchone()
        assert row == ("duplicate", 2)
    assert not (tmp_path / "identity" / "tallanto_email_identity_map.sqlite-wal").exists()
    assert not (tmp_path / "identity" / "tallanto_email_identity_map.sqlite-shm").exists()


def test_mail_archive_ingest_rerun_is_idempotent_and_does_not_leak_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
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

    verification = verify_mail_archive_pilot(
        MailArchiveVerificationConfig(
            archive_dir=tmp_path / "archive",
            expected_max_messages=1,
        )
    )
    assert verification["verification_pass"] is True
    assert verification["blocking_risks"] == []
    assert verification["db_counts"]["messages"] == 1
    assert verification["file_counts"]["raw_eml"] == 1
    assert verification["privacy"]["verification_opens_raw_eml"] is False
    assert (tmp_path / "archive" / "mail_archive_verification.json").exists()


def test_mail_archive_ingest_can_exclude_control_message_sha256s(tmp_path: Path) -> None:
    raw = _raw_message()
    excluded_sha256 = hashlib.sha256(raw).hexdigest()
    fake_imap = FakeImapClient([raw])

    report = build_mail_archive_ingest(
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
            exclude_message_sha256s=(excluded_sha256,),
        ),
        client=fake_imap,
    )

    assert report["messages_attempted"] == 1
    assert report["messages_excluded_by_sha256"] == 1
    assert report["messages_inserted_or_seen"] == 0
    assert report["raw_eml_written"] == 0
    assert report["attachments_written"] == 0
    assert report["text_files_written"] == 0
    assert not list((tmp_path / "archive" / "raw_eml").glob("**/*.eml"))
    with sqlite3.connect(tmp_path / "archive" / "mail_archive.sqlite") as con:
        assert con.execute("select count(*) from messages").fetchone()[0] == 0
        assert con.execute("select count(*) from message_sources").fetchone()[0] == 0


def test_mail_archive_zero_max_messages_fetches_nothing(tmp_path: Path) -> None:
    fake_imap = FakeImapClient([_raw_message()])

    report = build_mail_archive_ingest(
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
            max_messages=0,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )

    assert report["messages_found_since"] == 1
    assert report["messages_attempted"] == 0
    assert report["raw_eml_written"] == 0
    assert fake_imap.fetch_queries == []


def test_mail_archive_detects_transient_imap_fetch_errors() -> None:
    transient = type("abort", (Exception,), {})("command: FETCH => socket error: EOF")
    timeout = TimeoutError("The handshake operation timed out")
    dns = OSError("[Errno 8] nodename nor servname provided, or not known")
    permanent = ValueError("invalid payload")

    assert is_transient_imap_fetch_error(transient) is True
    assert is_transient_imap_fetch_error(timeout) is True
    assert is_transient_imap_fetch_error(dns) is True
    assert is_transient_imap_fetch_error(permanent) is False


def test_mail_archive_retries_transient_imap_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    class FakeClient:
        def __init__(self, host: str, port: int) -> None:
            calls.append((host, port))
            if len(calls) == 1:
                raise TimeoutError("The handshake operation timed out")

    monkeypatch.setattr("mango_mvp.productization.mail_archive.ImapLibClient", FakeClient)

    client, retries = open_imap_client_with_retries(
        MailImapCredentials(
            host="mail.example.test",
            port=993,
            email_address="school@example.test",
            password="not-written",
        ),
        attempts=2,
        delay_seconds=0,
    )

    assert isinstance(client, FakeClient)
    assert retries == 1
    assert calls == [("mail.example.test", 993), ("mail.example.test", 993)]


def test_mail_archive_tolerates_broken_attachment_headers() -> None:
    class BrokenPart:
        def is_multipart(self) -> bool:
            return False

        def get_content_disposition(self) -> str:
            raise AttributeError("'str' object has no attribute 'token_type'")

        def get_filename(self) -> str:
            raise AttributeError("'str' object has no attribute 'token_type'")

        def get(self, name: str, default: str = "") -> str:
            if name == "Content-Disposition":
                return "attachment; filename=broken.pdf"
            return default

    class BrokenMessage:
        def __init__(self, part: BrokenPart) -> None:
            self.part = part

        def is_multipart(self) -> bool:
            return True

        def walk(self) -> list[Any]:
            return [self, self.part]

    part = BrokenPart()

    assert iter_attachment_parts(BrokenMessage(part)) == [part]  # type: ignore[arg-type]


def test_mail_archive_tolerates_broken_address_headers() -> None:
    class BrokenHeaderMessage:
        def get(self, name: str) -> str:
            raise AttributeError("'str' object has no attribute 'token_type'")

        def get_all(self, name: str, default: list[str]) -> list[str]:
            raise AttributeError("'str' object has no attribute 'token_type'")

        def raw_items(self) -> list[tuple[str, str]]:
            return [
                ("From", "Broken <broken@example.com>"),
                ("To", "Client <client@example.com>"),
                ("Subject", "Legacy message"),
            ]

    msg = BrokenHeaderMessage()

    assert message_metadata(msg)["from"] == "Broken <broken@example.com>"  # type: ignore[arg-type]
    assert [row["email_normalized"] for row in message_participants(msg)] == [  # type: ignore[arg-type]
        "broken@example.com",
        "client@example.com",
    ]


def test_mail_archive_preflight_blocks_missing_secret_and_unsafe_limits(tmp_path: Path) -> None:
    report = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=tmp_path / "mail_pilot",
            email_address="",
            password_env_present=False,
            since_days=31,
            max_messages=0,
        )
    )

    assert report["preflight_pass"] is False
    assert "missing_or_invalid_mailbox_email" in report["blocking_risks"]
    assert "missing_password_env" in report["blocking_risks"]
    assert "pilot_window_days_must_be_1_to_7" in report["blocking_risks"]
    assert "pilot_max_messages_must_be_1_to_5" in report["blocking_risks"]
    assert "out_dir_not_git_ignored" in report["blocking_risks"]
    assert report["safety"]["network_calls"] is False
    assert report["safety"]["password_written"] is False
    assert (tmp_path / "mail_pilot" / "mail_archive_preflight.json").exists()


def test_mail_archive_preflight_passes_for_small_ignored_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    identity_db = tmp_path / "_external_handoffs" / "identity.sqlite"
    identity_db.parent.mkdir(parents=True)
    identity_db.write_bytes(b"sqlite placeholder")

    report = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=tmp_path / "_external_handoffs" / "pilot",
            email_address="school@kmipt.ru",
            password_env_present=True,
            since_days=3,
            max_messages=1,
            identity_db_path=identity_db,
        )
    )

    assert report["preflight_pass"] is True
    assert report["blocking_risks"] == []
    assert report["checks"]["out_dir_git_ignored"] is True
    assert report["checks"]["identity_db_exists"] is True
    assert report["requested_pilot"]["email_present"] is True
    text = (tmp_path / "_external_handoffs" / "pilot" / "mail_archive_preflight.json").read_text(
        encoding="utf-8"
    )
    assert "MAIL_IMAP_PASSWORD" in text
    assert "sqlite placeholder" not in text


def test_mail_archive_preflight_passes_for_explicit_large_ignored_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    identity_db = tmp_path / "_external_handoffs" / "identity.sqlite"
    identity_db.parent.mkdir(parents=True)
    identity_db.write_bytes(b"sqlite placeholder")

    report = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=tmp_path / "_external_handoffs" / "batch",
            email_address="school@kmipt.ru",
            password_env_present=True,
            since_days=30,
            max_messages=500,
            identity_db_path=identity_db,
            allow_large_batch=True,
        )
    )

    assert report["preflight_pass"] is True
    assert report["batch_mode"] == "approved_large_batch"
    assert report["blocking_risks"] == []
    assert "--allow-large-batch" in report["recommended_command"]

    blocked = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=tmp_path / "_external_handoffs" / "batch_too_large",
            email_address="school@kmipt.ru",
            password_env_present=True,
            since_days=30,
            max_messages=501,
            identity_db_path=identity_db,
            allow_large_batch=True,
        )
    )
    assert blocked["preflight_pass"] is False
    assert "batch_max_messages_must_be_1_to_500" in blocked["blocking_risks"]


def test_mail_archive_preflight_and_ingest_support_older_date_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    identity_db = tmp_path / "_external_handoffs" / "identity.sqlite"
    identity_db.parent.mkdir(parents=True)
    identity_db.write_bytes(b"sqlite placeholder")

    preflight = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=tmp_path / "_external_handoffs" / "older_batch",
            email_address="school@kmipt.ru",
            password_env_present=True,
            since_days=60,
            before_days=30,
            max_messages=100,
            identity_db_path=identity_db,
            allow_large_batch=True,
        )
    )

    assert preflight["preflight_pass"] is True
    assert preflight["requested_pilot"]["window_days"] == 30
    assert "--before-days 30" in preflight["recommended_command"]

    fake_imap = FakeImapClient([_raw_message()])
    ingest = build_mail_archive_ingest(
        credentials=MailImapCredentials(
            host="mail.example.test",
            port=993,
            email_address="school@kmipt.ru",
            password="not-written",
        ),
        config=MailArchiveIngestConfig(
            out_dir=tmp_path / "_external_handoffs" / "older_batch",
            mailbox="Sent Messages",
            mailbox_label="Sent Messages",
            since_days=60,
            before_days=30,
            max_messages=1,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )

    assert ingest["before_days"] == 30
    assert ingest["window_days"] == 30
    assert fake_imap.search_criteria
    assert fake_imap.search_criteria[0][0] == "SINCE"
    assert fake_imap.search_criteria[0][2] == "BEFORE"


def test_mail_archive_preflight_sanitizes_invalid_password_env_name(tmp_path: Path) -> None:
    report = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=tmp_path / "mail_pilot",
            email_address="school@kmipt.ru",
            password_env_name="literal-secret-value",
            password_env_present=False,
            since_days=3,
            max_messages=1,
        )
    )

    assert "invalid_password_env_name" in report["blocking_risks"]
    assert report["checks"]["password_env_name"] == "<invalid_env_var_name>"
    assert "literal-secret-value" not in str(report)


def test_mail_archive_cli_blocks_unsafe_ingest_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAIL_IMAP_PASSWORD", "not-written")

    rc = mango_office_mail_archive.main(
        [
            "ingest",
            "--email",
            "school@kmipt.ru",
            "--since-days",
            "31",
            "--max-messages",
            "25",
            "--out-dir",
            str(tmp_path / "archive"),
            "--dotenv",
            "",
        ]
    )

    assert rc == 2


def test_mail_archive_cli_blocks_nonignored_ingest_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAIL_IMAP_PASSWORD", "not-written")

    rc = mango_office_mail_archive.main(
        [
            "ingest",
            "--email",
            "school@kmipt.ru",
            "--since-days",
            "3",
            "--max-messages",
            "1",
            "--out-dir",
            str(tmp_path / "not_ignored_archive"),
            "--dotenv",
            "",
        ]
    )

    assert rc == 2
    assert not (tmp_path / "not_ignored_archive" / "raw_eml").exists()


def test_mail_archive_live_ingest_requires_git_ignored_output_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="git-ignored"):
        build_mail_archive_ingest(
            credentials=MailImapCredentials(
                host="mail.example.test",
                port=993,
                email_address="school@kmipt.ru",
                password="not-written",
            ),
            config=MailArchiveIngestConfig(
                out_dir=tmp_path / "not_ignored_archive",
                max_messages=0,
            ),
            client=None,
        )
    assert not (tmp_path / "not_ignored_archive").exists()


def test_mail_matching_report_counts_ambiguous_missing_internal_and_service(
    tmp_path: Path,
) -> None:
    archive_db, identity_db = _build_mixed_match_archive(tmp_path)

    match_report = build_mail_matching_report(
        MailMatchingReportConfig(
            archive_db_path=archive_db,
            identity_db_path=identity_db,
            out_dir=tmp_path / "matching",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )

    assert match_report["message_count"] == 5
    assert match_report["matched_message_count"] == 2
    assert match_report["counts"] == {
        "strong_unique": 1,
        "ambiguous": 1,
        "missing": 1,
        "internal_or_service": 2,
    }
    assert match_report["message_kind_counts"] == {"external": 3, "internal": 1, "service": 1}
    assert match_report["match_class_by_message_kind"] == {
        "external": {"ambiguous": 1, "missing": 1, "strong_unique": 1},
        "internal": {"internal_or_service": 1},
        "service": {"internal_or_service": 1},
    }
    with sqlite3.connect(archive_db) as con:
        match_counts = dict(
            con.execute(
                "select match_class, count(*) from message_matches group by match_class"
            ).fetchall()
        )
        kind_counts = dict(
            con.execute("select message_kind, count(*) from messages group by message_kind").fetchall()
        )
    assert match_counts == match_report["counts"]
    assert kind_counts == {"external": 3, "internal": 1, "service": 1}


def test_mail_customer_history_handoff_marks_manual_review_and_excluded(
    tmp_path: Path,
) -> None:
    archive_db, identity_db = _build_mixed_match_archive(tmp_path)

    handoff = build_mail_customer_history_handoff(
        MailCustomerHistoryHandoffConfig(
            archive_db_paths=[archive_db],
            identity_db_path=identity_db,
            out_dir=tmp_path / "history_handoff",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )

    assert handoff["counts"] == {
        "strong_unique": 1,
        "ambiguous": 1,
        "missing": 1,
        "internal_or_service": 2,
    }
    with sqlite3.connect(tmp_path / "history_handoff" / "mail_customer_history_handoff.sqlite") as con:
        status_counts = dict(
            con.execute(
                "select link_status, count(*) from mail_customer_links group by link_status"
            ).fetchall()
        )
        blocked_reasons = dict(
            con.execute(
                """
                select blocked_reason, count(*)
                from mail_customer_links
                where blocked_reason <> ''
                group by blocked_reason
                """
            ).fetchall()
        )
        assert con.execute("select count(*) from v_strong_customer_mail_links").fetchone()[0] == 1
        assert con.execute("select count(*) from v_manual_review_mail_links").fetchone()[0] == 2
    assert status_counts == {"excluded": 2, "manual_review": 2, "ready": 1}
    assert blocked_reasons == {
        "ambiguous_identity_match": 1,
        "internal_or_service_message": 2,
        "missing_identity_match": 1,
    }


def test_mail_mango_bridge_preview_resolves_and_blocks_by_phone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    handoff_db, identity_db = _build_bridge_mail_handoff(tmp_path)
    product_db = _build_bridge_product_db(tmp_path)

    report = build_mail_mango_bridge_preview(
        MailMangoBridgePreviewConfig(
            mail_handoff_db_path=handoff_db,
            identity_db_path=identity_db,
            product_db_path=product_db,
            out_dir=tmp_path / "_external_handoffs" / "bridge_preview",
        )
    )

    assert report["schema_version"] == "mail_mango_bridge_preview_v1"
    assert report["candidate_count"] == 4
    assert report["counts"] == {
        "resolved": 2,
        "blocked": 2,
        "no_phone_for_candidate": 1,
        "phone_multiple_candidates": 1,
        "mango_no_phone_match": 0,
    }
    assert report["mail_link_reconciliation"]["pass"] is True
    assert report["mango_source_counts"]["capture_rows_seen"] == 2
    assert report["mango_source_counts"]["capture_rows_with_normalized_phone"] == 2
    assert report["mango_source_counts"]["product_calls_with_filename_phone"] == 1
    assert report["mango_source_counts"]["distinct_product_filename_phones"] == 1
    assert report["artifact_counts"]["mango_call_refs_written"] == 2
    assert report["safety"]["source_sqlite_mode"] == "mode=ro"
    assert report["safety"]["source_db_attached_to_writer"] is False
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["write_tallanto"] is False
    assert report["safety"]["runtime_db_writes"] is False
    assert report["safety"]["run_asr"] is False
    assert report["privacy"]["raw_emails_written"] is False
    assert report["privacy"]["raw_payloads_written"] is False
    assert "+79990000000" not in json.dumps(report, ensure_ascii=False)

    preview_db = tmp_path / "_external_handoffs" / "bridge_preview" / "mail_mango_bridge_preview.sqlite"
    with sqlite3.connect(preview_db) as con:
        status_counts = dict(
            con.execute(
                "select bridge_status, count(*) from candidate_mango_preview group by bridge_status"
            ).fetchall()
        )
        blocked_counts = dict(
            con.execute(
                "select blocked_reason, count(*) from blocked_bridge_candidates group by blocked_reason"
            ).fetchall()
        )
        assert con.execute("select count(*) from v_resolved_mail_mango_links").fetchone()[0] == 2
        assert con.execute("select count(*) from v_manual_review_mail_mango_links").fetchone()[0] == 2
        assert con.execute("select count(*) from mango_call_refs").fetchone()[0] == 2
        assert con.execute("select count(*) from candidate_phone_refs").fetchone()[0] == 3
    assert status_counts == {"blocked": 2, "resolved": 2}
    assert blocked_counts == {
        "no_phone_for_candidate": 1,
        "phone_multiple_candidates": 1,
    }


def test_mail_mango_bridge_preview_blocks_missing_inputs_and_unsafe_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    handoff_db, identity_db = _build_bridge_mail_handoff(tmp_path)
    product_db = _build_bridge_product_db(tmp_path)

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_mango_bridge_preview(
            MailMangoBridgePreviewConfig(
                mail_handoff_db_path=handoff_db,
                identity_db_path=identity_db,
                product_db_path=product_db,
                out_dir=tmp_path / "bridge_preview",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_mango_bridge_preview(
            MailMangoBridgePreviewConfig(
                mail_handoff_db_path=handoff_db,
                identity_db_path=identity_db,
                product_db_path=product_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "bridge_preview",
            )
        )
    missing_identity = tmp_path / "_external_handoffs" / "missing_identity.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_mango_bridge_preview(
            MailMangoBridgePreviewConfig(
                mail_handoff_db_path=handoff_db,
                identity_db_path=missing_identity,
                product_db_path=product_db,
                out_dir=tmp_path / "_external_handoffs" / "bridge_preview_missing",
            )
        )
    assert not missing_identity.exists()


def test_mango_phone_index_preview_extends_bridge_from_recording_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    handoff_db, identity_db, product_db = _build_phone_index_bridge_fixture(tmp_path)
    recording_root = tmp_path / "_external_handoffs" / "mango_recordings"
    recording_root.mkdir(parents=True)
    (
        recording_root / "2026-05-07__13-00-00__79993334455__Manager_77.mp3"
    ).write_bytes(b"synthetic audio bytes are not opened")

    index_report = build_mango_phone_index_preview(
        MangoPhoneIndexPreviewConfig(
            product_db_path=product_db,
            recording_roots=[recording_root],
            out_dir=tmp_path / "_external_handoffs" / "mango_phone_index",
        )
    )

    assert index_report["schema_version"] == "mango_phone_index_preview_v1"
    assert index_report["source_file_counts"]["recording_files_seen"] == 1
    assert index_report["source_file_counts"]["recording_files_with_filename_phone"] == 1
    assert index_report["source_file_counts"]["distinct_recording_filename_phones"] == 1
    assert index_report["phone_index_counts"]["distinct_normalized_phones"] == 2
    assert index_report["safety"]["source_sqlite_mode"] == "mode=ro"
    assert index_report["safety"]["open_audio_files"] is False
    assert index_report["safety"]["scan_filenames_only"] is True
    assert index_report["privacy"]["raw_phones_written_to_json"] is False
    assert "+79993334455" not in json.dumps(index_report, ensure_ascii=False)

    baseline = build_mail_mango_bridge_preview(
        MailMangoBridgePreviewConfig(
            mail_handoff_db_path=handoff_db,
            identity_db_path=identity_db,
            product_db_path=product_db,
            out_dir=tmp_path / "_external_handoffs" / "bridge_baseline",
        )
    )
    assert baseline["counts"]["resolved"] == 1
    assert baseline["counts"]["mango_no_phone_match"] == 1

    expanded = build_mail_mango_bridge_preview(
        MailMangoBridgePreviewConfig(
            mail_handoff_db_path=handoff_db,
            identity_db_path=identity_db,
            product_db_path=product_db,
            out_dir=tmp_path / "_external_handoffs" / "bridge_expanded",
            mango_phone_index_db_path=Path(index_report["paths"]["index_db"]),
        )
    )

    assert expanded["candidate_count"] == 2
    assert expanded["counts"]["resolved"] == 2
    assert expanded["counts"]["mango_no_phone_match"] == 0
    assert expanded["mango_source_counts"]["phone_index_enabled"] == 1
    assert expanded["mango_source_counts"]["phone_index_rows_added_after_dedupe"] >= 1
    assert "+79993334455" not in json.dumps(expanded, ensure_ascii=False)

    index_db = Path(index_report["paths"]["index_db"])
    with sqlite3.connect(index_db) as con:
        row = con.execute(
            "select phone_sha256, source_filename_sha256, source_path_sha256 from mango_phone_call_refs "
            "where source_kind = 'recording_filename'"
        ).fetchone()
    assert row is not None
    assert all(len(value) == 64 for value in row)


def test_mango_phone_index_preview_blocks_unsafe_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    product_db = _build_phone_index_product_db(tmp_path)
    recording_root = tmp_path / "_external_handoffs" / "mango_recordings"
    recording_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mango_phone_index_preview(
            MangoPhoneIndexPreviewConfig(
                product_db_path=product_db,
                recording_roots=[recording_root],
                out_dir=tmp_path / "mango_phone_index",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mango_phone_index_preview(
            MangoPhoneIndexPreviewConfig(
                product_db_path=product_db,
                recording_roots=[recording_root],
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "mango_phone_index",
            )
        )


def test_mail_phone_lift_preview_lifts_manual_messages_from_text_phones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, identity_db = _build_phone_lift_archive(tmp_path)

    report = build_mail_phone_lift_preview(
        MailPhoneLiftPreviewConfig(
            archive_db_paths=[archive_db],
            identity_db_path=identity_db,
            out_dir=tmp_path / "_external_handoffs" / "phone_lift_preview",
        )
    )

    assert report["schema_version"] == "mail_phone_lift_preview_v1"
    assert report["evaluated_message_count"] == 5
    assert report["by_original_match_class"] == {"ambiguous": 1, "missing": 4}
    assert report["lift_class_counts"] == {
        "phone_strong_unique": 2,
        "phone_ambiguous": 1,
        "phone_no_identity_match": 1,
        "no_phone_detected": 1,
        "no_text": 0,
    }
    assert report["potential_lift"] == {
        "strong_unique_messages": 2,
        "manual_review_messages": 3,
        "ambiguous_to_phone_strong_unique": 1,
        "missing_to_phone_strong_unique": 1,
    }
    assert report["text_access_counts"]["text_files_read"] == 5
    assert report["artifact_counts"]["extracted_phone_values_seen"] == 4
    assert report["artifact_counts"]["identity_phone_value_matches"] == 3
    assert report["safety"]["source_sqlite_mode"] == "mode=ro"
    assert report["safety"]["source_db_attached_to_writer"] is False
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["write_tallanto"] is False
    assert report["safety"]["open_attachments"] is False
    assert report["privacy"]["raw_phones_written"] is False
    assert "+79990000000" not in json.dumps(report, ensure_ascii=False)
    assert "999" not in json.dumps(report, ensure_ascii=False)

    preview_db = tmp_path / "_external_handoffs" / "phone_lift_preview" / "mail_phone_lift_preview.sqlite"
    with sqlite3.connect(preview_db) as con:
        lift_counts = dict(
            con.execute(
                "select lift_class, count(*) from message_phone_lift_preview group by lift_class"
            ).fetchall()
        )
        assert con.execute("select count(*) from v_phone_strong_unique_lift").fetchone()[0] == 2
        assert con.execute("select count(*) from v_phone_manual_review_lift").fetchone()[0] == 3
        stored = con.execute(
            """
            select phone_sha256_json, candidate_keys_json
            from message_phone_lift_preview
            where lift_class = 'phone_strong_unique'
            order by message_sha256
            limit 1
            """
        ).fetchone()
    assert lift_counts["phone_strong_unique"] == 2
    assert "+79990000000" not in stored[0]
    assert len(json.loads(stored[0])[0]) == 64
    assert json.loads(stored[1])


def test_mail_phone_lift_preview_blocks_missing_inputs_and_unsafe_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, identity_db = _build_phone_lift_archive(tmp_path)

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_phone_lift_preview(
            MailPhoneLiftPreviewConfig(
                archive_db_paths=[archive_db],
                identity_db_path=identity_db,
                out_dir=tmp_path / "phone_lift_preview",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_phone_lift_preview(
            MailPhoneLiftPreviewConfig(
                archive_db_paths=[archive_db],
                identity_db_path=identity_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "phone_lift_preview",
            )
        )
    missing_identity = tmp_path / "_external_handoffs" / "missing_identity.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_phone_lift_preview(
            MailPhoneLiftPreviewConfig(
                archive_db_paths=[archive_db],
                identity_db_path=missing_identity,
                out_dir=tmp_path / "_external_handoffs" / "phone_lift_preview_missing",
            )
        )
    assert not missing_identity.exists()
    assert not (tmp_path / "_external_handoffs" / "phone_lift_preview_missing").exists()


def test_mail_customer_relink_preview_links_only_unique_tallanto_phone_and_learns_email(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, identity_db = _build_phone_lift_archive(tmp_path)
    handoff = build_mail_customer_history_handoff(
        MailCustomerHistoryHandoffConfig(
            archive_db_paths=[archive_db],
            identity_db_path=identity_db,
            out_dir=tmp_path / "history_handoff",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )

    config = MailCustomerRelinkPreviewConfig(
        mail_handoff_db_path=Path(handoff["paths"]["handoff_db"]),
        identity_db_path=identity_db,
        out_dir=tmp_path / "_external_handoffs" / "customer_relink_preview",
    )
    report = build_mail_customer_relink_preview(config)
    rerun = build_mail_customer_relink_preview(config)

    assert report["schema_version"] == "mail_customer_relink_preview_v1"
    assert report["baseline"]["manual_review"] == 5
    assert report["after_preview"]["new_links"] == 2
    assert report["address_book"]["learned_values"] == 1
    assert report["unmatched_reasons"]["duplicate_identity_value"] == 1
    assert report["unmatched_reasons"]["identity_value_missing"] == 1
    assert report["unmatched_reasons"]["no_phone_signal"] == 1
    assert rerun["after_preview"] == report["after_preview"]
    assert rerun["address_book"]["learned_values"] == report["address_book"]["learned_values"]
    assert report["safety"]["write_tallanto"] is False
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["raw_personal_values_in_public_report"] is False

    preview_db = tmp_path / "_external_handoffs" / "customer_relink_preview" / "mail_customer_relink_preview.sqlite"
    with sqlite3.connect(preview_db) as con:
        linked = con.execute("select count(*) from v_linked_mail_relinks").fetchone()[0]
        learned = con.execute("select kind, value from learned_address_book_values").fetchall()
        duplicate_reason = con.execute(
            "select reason from mail_relink_decisions where reason = 'duplicate_identity_value'"
        ).fetchone()
    assert linked == 2
    assert learned == [("email", "missing@example.com")]
    assert duplicate_reason == ("duplicate_identity_value",)
    assert not (preview_db.parent / "mail_customer_relink_preview.sqlite-wal").exists()


def test_mail_customer_relink_preview_blocks_cross_brand_common_phone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    tallanto_csv = tmp_path / "brand_students.csv"
    _write_tallanto_csv(
        tallanto_csv,
        [
            {
                "ID": "F-1",
                "Филиал": "Фотон",
                "E-mail": "foton@example.com",
                "Тел. (родителя)": "+7 999 100-00-00",
            },
            {
                "ID": "U-1",
                "Филиал": "УНПК МФТИ",
                "E-mail": "unpk@example.com",
                "Тел. (родителя)": "8 999 100 00 00",
            },
        ],
    )
    identity_report = build_tallanto_identity_map(
        TallantoIdentityMapConfig(
            tallanto_csv_path=tallanto_csv,
            out_dir=tmp_path / "identity_brand",
            encoding="utf-8",
            delimiter="\t",
        )
    )
    fake_imap = FakeImapClient(
        [
            _raw_message(
                message_id="m-cross-brand",
                from_addr="Missing <missing@example.com>",
                body="Подскажите по оплате, телефон +7 999 100-00-00.",
            )
        ]
    )
    ingest_report = build_mail_archive_ingest(
        credentials=MailImapCredentials(
            host="mail.example.test",
            port=993,
            email_address="school@kmipt.ru",
            password="not-written",
        ),
        config=MailArchiveIngestConfig(
            out_dir=tmp_path / "_external_handoffs" / "brand_archive",
            mailbox="INBOX",
            mailbox_label="INBOX",
            since_days=7,
            max_messages=1,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )
    handoff = build_mail_customer_history_handoff(
        MailCustomerHistoryHandoffConfig(
            archive_db_paths=[Path(ingest_report["paths"]["archive_db"])],
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "brand_handoff",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )

    report = build_mail_customer_relink_preview(
        MailCustomerRelinkPreviewConfig(
            mail_handoff_db_path=Path(handoff["paths"]["handoff_db"]),
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "_external_handoffs" / "brand_relink",
        )
    )

    assert report["after_preview"]["new_links"] == 0
    assert report["unmatched_reasons"] == {"brand_conflict": 1}


def test_mail_customer_relink_preview_live_lookup_is_read_only_and_blocks_ambiguous_contacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, identity_db = _build_phone_lift_archive(tmp_path)
    handoff = build_mail_customer_history_handoff(
        MailCustomerHistoryHandoffConfig(
            archive_db_paths=[archive_db],
            identity_db_path=identity_db,
            out_dir=tmp_path / "live_handoff",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )
    calls: list[str] = []

    def fake_live_lookup(phone: str) -> list[dict[str, str]]:
        calls.append(phone)
        if phone == "+79995556677":
            return [{"id": "L-1"}, {"id": "L-2"}]
        return []

    report = build_mail_customer_relink_preview(
        MailCustomerRelinkPreviewConfig(
            mail_handoff_db_path=Path(handoff["paths"]["handoff_db"]),
            identity_db_path=identity_db,
            out_dir=tmp_path / "_external_handoffs" / "live_relink",
            live_phone_lookup=fake_live_lookup,
        )
    )

    assert "+79995556677" in calls
    assert report["live_tallanto"]["enabled"] is True
    assert report["live_tallanto"]["counts"]["ambiguous"] == 1
    assert report["unmatched_reasons"]["live_multiple_contacts"] == 1
    assert report["safety"]["write_tallanto"] is False


def test_mail_attachment_parse_plan_classifies_without_raw_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db = _build_attachment_plan_archive(tmp_path)

    config = MailAttachmentParsePlanConfig(
        archive_db_paths=[archive_db],
        out_dir=tmp_path / "_external_handoffs" / "attachment_plan",
        max_size_bytes=20_000_000,
    )
    report = build_mail_attachment_parse_plan(config)
    rerun = build_mail_attachment_parse_plan(config)

    assert report["schema_version"] == "mail_attachment_parse_plan_v1"
    assert report["attachment_count"] == 7
    assert report["parse_plan_counts"] == {
        "parse_later": 2,
        "manual_review": 1,
        "blocked": 4,
    }
    assert rerun["parse_plan_counts"] == report["parse_plan_counts"]
    assert report["safety"]["open_attachments"] is False
    assert report["safety"]["read_attachment_bytes"] is False
    assert report["privacy"]["raw_filenames_written"] is False
    assert report["privacy"]["raw_paths_written"] is False

    report_text = (
        tmp_path
        / "_external_handoffs"
        / "attachment_plan"
        / "mail_attachment_parse_plan_report.json"
    ).read_text(encoding="utf-8")
    assert "invoice" not in report_text
    assert "../" not in report_text

    with sqlite3.connect(
        tmp_path / "_external_handoffs" / "attachment_plan" / "mail_attachment_parse_plan.sqlite"
    ) as con:
        con.row_factory = sqlite3.Row
        rows = list(
            con.execute(
                """
                SELECT extension, action, risk_level, risk_reasons_json, filename_sha256
                FROM attachment_parse_plan
                ORDER BY part_index
                """
            )
        )
        assert con.execute("select count(*) from v_attachment_parse_queue").fetchone()[0] == 2
        assert con.execute("select count(*) from v_attachment_manual_review").fetchone()[0] == 1
        assert con.execute("select count(*) from v_attachment_blocked").fetchone()[0] == 4
    assert rows[0]["extension"] == ".pdf"
    assert rows[0]["action"] == "parse_later"
    assert len(rows[0]["filename_sha256"]) == 64
    assert rows[2]["action"] == "blocked"
    assert "double_extension" in json.loads(rows[2]["risk_reasons_json"])
    assert rows[4]["action"] == "manual_review"
    assert "legacy_office_manual_review" in json.loads(rows[4]["risk_reasons_json"])
    assert "size_limit_exceeded" in json.loads(rows[5]["risk_reasons_json"])
    assert "path_traversal_filename" in json.loads(rows[6]["risk_reasons_json"])


def test_mail_attachment_parse_plan_blocks_unsafe_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db = _build_attachment_plan_archive(tmp_path)

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_parse_plan(
            MailAttachmentParsePlanConfig(
                archive_db_paths=[archive_db],
                out_dir=tmp_path / "attachment_plan",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_parse_plan(
            MailAttachmentParsePlanConfig(
                archive_db_paths=[archive_db],
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_plan",
            )
        )


def test_mail_attachment_text_extract_parses_allowlisted_synthetic_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_text_extract_fixture(tmp_path)

    config = MailAttachmentTextExtractConfig(
        archive_db_paths=[archive_db],
        parse_plan_db_path=plan_db,
        out_dir=tmp_path / "_external_handoffs" / "attachment_text_extract",
        max_text_chars_per_attachment=5_000,
    )
    report = build_mail_attachment_text_extract(config)
    rerun = build_mail_attachment_text_extract(config)

    assert report["schema_version"] == "mail_attachment_text_extract_v1"
    assert report["parse_plan_queue_count"] == 5
    assert report["stage_supported_queue_count"] == 4
    assert report["stage_skipped_queue_count"] == 1
    assert report["status_counts"]["extracted"] == 4
    assert rerun["status_counts"] == report["status_counts"]
    assert report["safety"]["read_raw_eml"] is False
    assert report["safety"]["read_extracted_mail_text"] is False
    assert report["safety"]["parse_pdf"] is False
    assert report["privacy"]["raw_filenames_written"] is False
    assert report["privacy"]["raw_source_attachment_paths_written"] is False

    out_dir = tmp_path / "_external_handoffs" / "attachment_text_extract"
    report_text = (out_dir / "mail_attachment_text_extract_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov 89990000000" not in report_text
    assert "TXT safe line" not in report_text
    assert "Docx safe phrase" not in report_text
    assert "Xlsx safe phrase" not in report_text

    with sqlite3.connect(out_dir / "mail_attachment_text_extract.sqlite") as con:
        con.row_factory = sqlite3.Row
        rows = list(
            con.execute(
                """
                SELECT extension, status, status_reason, parser, derived_text_path
                FROM attachment_text_extracts
                ORDER BY extension
                """
            )
        )
        assert con.execute("select count(*) from v_attachment_text_ready").fetchone()[0] == 4
        assert con.execute("select count(*) from v_attachment_text_needs_review").fetchone()[0] == 0

    assert {row["extension"] for row in rows} == {".txt", ".csv", ".docx", ".xlsx"}
    assert {row["status"] for row in rows} == {"extracted"}
    db_dump = "\n".join(str(dict(row)) for row in rows)
    assert "Client Ivanov 89990000000" not in db_dump
    assert "source_sensitive_segment" not in db_dump

    extracted_text = "\n".join(
        Path(row["derived_text_path"]).read_text(encoding="utf-8") for row in rows
    )
    assert "TXT safe line" in extracted_text
    assert "csv safe value" in extracted_text
    assert "Docx safe phrase" in extracted_text
    assert "Xlsx safe phrase" in extracted_text


def test_mail_attachment_text_extract_limits_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_text_extract_limit_fixture(tmp_path)

    out_dir = tmp_path / "_external_handoffs" / "attachment_text_extract_limits"
    report = build_mail_attachment_text_extract(
        MailAttachmentTextExtractConfig(
            archive_db_paths=[archive_db],
            parse_plan_db_path=plan_db,
            out_dir=out_dir,
            max_attachment_bytes=1_000,
            max_text_chars_per_attachment=12,
        )
    )

    assert report["stage_supported_queue_count"] == 4
    assert report["status_counts"]["extracted"] == 1
    assert report["status_counts"]["skipped"] == 1
    assert report["status_counts"]["blocked_safety"] == 1
    assert report["status_counts"]["parse_error"] == 1
    assert report["status_reason_counts"]["stage_size_limit_exceeded"] == 1
    assert report["status_reason_counts"]["office_macro_payload_detected"] == 1
    assert report["status_reason_counts"]["invalid_office_zip"] == 1
    assert report["warning_counts"]["text_truncated"] == 1

    with sqlite3.connect(out_dir / "mail_attachment_text_extract.sqlite") as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["status_reason"]: row
            for row in con.execute(
                """
                SELECT status, status_reason, text_chars, text_truncated, derived_text_path
                FROM attachment_text_extracts
                """
            )
    }
    assert rows["ok"]["text_chars"] == 12
    assert rows["ok"]["text_truncated"] == 1
    assert "BBB" in Path(rows["ok"]["derived_text_path"]).read_text(encoding="utf-8")
    assert rows["stage_size_limit_exceeded"]["status"] == "skipped"
    assert rows["office_macro_payload_detected"]["status"] == "blocked_safety"
    assert rows["invalid_office_zip"]["status"] == "parse_error"


def test_mail_attachment_text_extract_blocks_unsafe_output_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_text_extract_fixture(tmp_path)

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_text_extract(
            MailAttachmentTextExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "attachment_text_extract",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_text_extract(
            MailAttachmentTextExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_text_extract",
            )
        )
    with pytest.raises(ValueError, match="supports only"):
        build_mail_attachment_text_extract(
            MailAttachmentTextExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_text_extract_pdf",
                stage_extensions=(".pdf",),
            )
        )
    missing_plan = tmp_path / "_external_handoffs" / "missing_plan.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_attachment_text_extract(
            MailAttachmentTextExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=missing_plan,
                out_dir=tmp_path / "_external_handoffs" / "attachment_text_extract_missing",
            )
        )
    assert not (tmp_path / "_external_handoffs" / "attachment_text_extract_missing").exists()


def test_mail_attachment_pdf_extract_parses_synthetic_pdf_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_pdf_extract_fixture(
        tmp_path,
        [
            (
                "m_pdf",
                1,
                "Client Ivanov 89990000000.pdf",
                "application/pdf",
                _minimal_pdf_bytes(["PDF safe phrase"]),
            )
        ],
    )

    config = MailAttachmentPdfExtractConfig(
        archive_db_paths=[archive_db],
        parse_plan_db_path=plan_db,
        out_dir=tmp_path / "_external_handoffs" / "attachment_pdf_extract",
        pdf_timeout_seconds=0,
    )
    report = build_mail_attachment_pdf_extract(config)
    rerun = build_mail_attachment_pdf_extract(config)

    assert report["schema_version"] == "mail_attachment_pdf_extract_v1"
    assert report["parse_plan_queue_count"] == 1
    assert report["stage_supported_queue_count"] == 1
    assert report["status_counts"]["extracted"] == 1
    assert report["parser_counts"] == {"pypdf_text": 1}
    assert report["artifact_counts"]["derived_text_files_written"] == 1
    assert rerun["status_counts"] == report["status_counts"]
    assert rerun["artifact_counts"]["derived_text_files_written"] == 0
    assert report["safety"]["parse_pdf"] is True
    assert report["safety"]["render_pdf"] is False
    assert report["safety"]["run_ocr"] is False
    assert report["safety"]["parse_images"] is False
    assert report["privacy"]["raw_filenames_written"] is False
    assert report["privacy"]["raw_source_attachment_paths_written"] is False

    out_dir = tmp_path / "_external_handoffs" / "attachment_pdf_extract"
    report_text = (out_dir / "mail_attachment_pdf_extract_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov" not in report_text
    assert "89990000000" not in report_text
    assert "PDF safe phrase" not in report_text

    with sqlite3.connect(out_dir / "mail_attachment_pdf_extract.sqlite") as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT status, status_reason, parser, page_count, pages_processed,
                   text_sha256, derived_text_path
            FROM attachment_pdf_extracts
            """
        ).fetchone()
        assert con.execute("select count(*) from v_attachment_pdf_text_ready").fetchone()[0] == 1
    row_dump = str(dict(row))
    assert "Client Ivanov" not in row_dump
    assert "89990000000" not in row_dump
    assert "PDF safe phrase" not in row_dump
    assert row["status"] == "extracted"
    assert row["status_reason"] == "ok"
    assert row["parser"] == "pypdf_text"
    assert row["page_count"] == 1
    assert row["pages_processed"] == 1
    assert len(row["text_sha256"]) == 64
    assert "PDF safe phrase" in Path(row["derived_text_path"]).read_text(encoding="utf-8")


def test_mail_attachment_pdf_extract_limits_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_pdf_extract_fixture(
        tmp_path,
        [
            (
                "m_big",
                1,
                "big.pdf",
                "application/pdf",
                _minimal_pdf_bytes(["big"]) + (b" " * 3_000),
            ),
            (
                "m_pages",
                1,
                "many_pages.pdf",
                "application/pdf",
                _minimal_pdf_bytes(["one", "two", "three"]),
            ),
            (
                "m_long",
                1,
                "long.pdf",
                "application/pdf",
                _minimal_pdf_bytes(["B" * 80]),
            ),
            (
                "m_corrupt",
                1,
                "corrupt.pdf",
                "application/pdf",
                b"%PDF-1.4\nnot a valid xref\n%%EOF",
            ),
            (
                "m_js",
                1,
                "active.pdf",
                "application/pdf",
                _minimal_pdf_bytes(["safe"]) + b"\n/JavaScript\n",
            ),
            (
                "m_encrypted",
                1,
                "encrypted.pdf",
                "application/pdf",
                _encrypted_pdf_bytes(),
            ),
        ],
    )

    out_dir = tmp_path / "_external_handoffs" / "attachment_pdf_extract_limits"
    report = build_mail_attachment_pdf_extract(
        MailAttachmentPdfExtractConfig(
            archive_db_paths=[archive_db],
            parse_plan_db_path=plan_db,
            out_dir=out_dir,
            max_attachment_bytes=2_500,
            max_pdf_pages=2,
            max_text_chars_per_attachment=12,
            pdf_timeout_seconds=0,
        )
    )

    assert report["stage_supported_queue_count"] == 6
    assert report["status_counts"]["extracted"] == 1
    assert report["status_counts"]["skipped"] == 1
    assert report["status_counts"]["blocked_safety"] == 3
    assert report["status_counts"]["parse_error"] == 1
    assert report["status_reason_counts"]["stage_size_limit_exceeded"] == 1
    assert report["status_reason_counts"]["pdf_page_limit_exceeded"] == 1
    assert report["status_reason_counts"]["pdf_javascript_declared"] == 1
    assert report["status_reason_counts"]["pdf_encrypted"] == 1
    assert report["warning_counts"]["text_truncated"] == 1

    with sqlite3.connect(out_dir / "mail_attachment_pdf_extract.sqlite") as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["status_reason"]: row
            for row in con.execute(
                """
                SELECT status, status_reason, text_chars, text_truncated, derived_text_path
                FROM attachment_pdf_extracts
                """
            )
        }
    assert rows["ok"]["text_chars"] == 12
    assert rows["ok"]["text_truncated"] == 1
    assert "BBB" in Path(rows["ok"]["derived_text_path"]).read_text(encoding="utf-8")
    assert rows["stage_size_limit_exceeded"]["status"] == "skipped"
    assert rows["pdf_page_limit_exceeded"]["status"] == "blocked_safety"
    assert rows["pdf_javascript_declared"]["status"] == "blocked_safety"
    assert rows["pdf_encrypted"]["status"] == "blocked_safety"
    assert normalize_extracted_attachment_text("bad\ud83dtext") == "bad?text"


def test_mail_attachment_pdf_extract_blocks_unsafe_output_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_pdf_extract_fixture(
        tmp_path,
        [("m_pdf", 1, "safe.pdf", "application/pdf", _minimal_pdf_bytes(["safe"]))],
    )

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_pdf_extract(
            MailAttachmentPdfExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "attachment_pdf_extract",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_pdf_extract(
            MailAttachmentPdfExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_pdf_extract",
            )
        )
    with pytest.raises(ValueError, match="supports only"):
        build_mail_attachment_pdf_extract(
            MailAttachmentPdfExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_pdf_extract_png",
                stage_extensions=(".png",),
            )
        )
    missing_plan = tmp_path / "_external_handoffs" / "missing_pdf_plan.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_attachment_pdf_extract(
            MailAttachmentPdfExtractConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=missing_plan,
                out_dir=tmp_path / "_external_handoffs" / "attachment_pdf_extract_missing",
            )
        )
    assert not (tmp_path / "_external_handoffs" / "attachment_pdf_extract_missing").exists()


def test_mail_attachment_image_ocr_plan_inspects_headers_without_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_image_ocr_fixture(
        tmp_path,
        [
            (
                "m_png",
                1,
                "Client Ivanov 89990000000.png",
                "image/png",
                _minimal_png_bytes(640, 480, with_exif=True),
            ),
            (
                "m_jpg",
                1,
                "scan.jpg",
                "image/jpeg",
                _minimal_jpeg_bytes(800, 600, with_exif=True),
            ),
            (
                "m_webp",
                1,
                "image.webp",
                "image/webp",
                _minimal_webp_vp8x_bytes(320, 240),
            ),
        ],
    )

    config = MailAttachmentImageOcrPlanConfig(
        archive_db_paths=[archive_db],
        parse_plan_db_path=plan_db,
        out_dir=tmp_path / "_external_handoffs" / "attachment_image_ocr_plan",
        inspect_headers=True,
    )
    report = build_mail_attachment_image_ocr_plan(config)
    rerun = build_mail_attachment_image_ocr_plan(config)

    assert report["schema_version"] == "mail_attachment_image_ocr_plan_v1"
    assert report["parse_plan_queue_count"] == 3
    assert report["stage_supported_queue_count"] == 3
    assert report["status_counts"]["planned"] == 3
    assert report["image_format_counts"] == {"jpeg": 1, "png": 1, "webp": 1}
    assert report["warning_counts"]["image_exif_declared_not_read"] == 2
    assert report["safety"]["run_ocr"] is False
    assert report["safety"]["ocr_enabled"] is False
    assert report["safety"]["decode_full_images"] is False
    assert report["safety"]["extract_exif"] is False
    assert report["safety"]["write_thumbnails"] is False
    assert report["privacy"]["raw_filenames_written"] is False
    assert report["privacy"]["raw_source_attachment_paths_written"] is False
    assert rerun["status_counts"] == report["status_counts"]

    out_dir = tmp_path / "_external_handoffs" / "attachment_image_ocr_plan"
    report_text = (out_dir / "mail_attachment_image_ocr_plan_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov" not in report_text
    assert "89990000000" not in report_text
    assert "scan.jpg" not in report_text

    with sqlite3.connect(out_dir / "mail_attachment_image_ocr_plan.sqlite") as con:
        con.row_factory = sqlite3.Row
        rows = list(
            con.execute(
                """
                SELECT image_format, width, height, status, status_reason, ocr_status,
                       warnings_json
                FROM attachment_image_ocr_plan
                ORDER BY image_format
                """
            )
        )
        assert con.execute("select count(*) from v_attachment_image_ocr_candidates").fetchone()[0] == 3
        assert con.execute("select count(*) from v_attachment_image_ocr_needs_review").fetchone()[0] == 0
    row_dump = "\n".join(str(dict(row)) for row in rows)
    assert "Client Ivanov" not in row_dump
    assert "89990000000" not in row_dump
    assert {row["ocr_status"] for row in rows} == {"disabled"}
    assert {row["status_reason"] for row in rows} == {"ocr_disabled"}


def test_mail_attachment_image_ocr_plan_default_is_metadata_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_image_ocr_fixture(
        tmp_path,
        [("m_png", 1, "Client Ivanov 89990000000.png", "image/png", _minimal_png_bytes(64, 64))],
    )

    out_dir = tmp_path / "_external_handoffs" / "attachment_image_ocr_plan_metadata"
    report = build_mail_attachment_image_ocr_plan(
        MailAttachmentImageOcrPlanConfig(
            archive_db_paths=[archive_db],
            parse_plan_db_path=plan_db,
            out_dir=out_dir,
        )
    )

    assert report["status_counts"]["planned"] == 1
    assert report["image_format_counts"] == {"not_inspected": 1}
    assert report["artifact_counts"]["attachment_bytes_read"] == 0
    assert report["safety"]["inspect_image_headers_only"] is False
    assert report["safety"]["run_ocr"] is False
    assert report["safety"]["write_images"] is False
    assert report["safety"]["write_thumbnails"] is False

    report_text = (out_dir / "mail_attachment_image_ocr_plan_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov" not in report_text
    assert "89990000000" not in report_text
    with sqlite3.connect(out_dir / "mail_attachment_image_ocr_plan.sqlite") as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT image_format, width, height, status, status_reason, ocr_status
            FROM attachment_image_ocr_plan
            """
        ).fetchone()
    assert row["image_format"] == "not_inspected"
    assert row["width"] == 0
    assert row["height"] == 0
    assert row["status"] == "planned"
    assert row["status_reason"] == "ocr_disabled"
    assert row["ocr_status"] == "disabled"


def test_mail_attachment_image_ocr_plan_limits_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_image_ocr_fixture(
        tmp_path,
        [
            ("m_big", 1, "big.png", "image/png", _minimal_png_bytes(100, 100) + b" " * 200),
            ("m_dim", 1, "dimension.png", "image/png", _minimal_png_bytes(30_000, 100)),
            ("m_pixels", 1, "pixels.jpg", "image/jpeg", _minimal_jpeg_bytes(10_000, 10_000)),
            ("m_bad", 1, "bad.png", "image/png", b"not an image"),
        ],
    )

    out_dir = tmp_path / "_external_handoffs" / "attachment_image_ocr_plan_limits"
    report = build_mail_attachment_image_ocr_plan(
        MailAttachmentImageOcrPlanConfig(
            archive_db_paths=[archive_db],
            parse_plan_db_path=plan_db,
            out_dir=out_dir,
            max_attachment_bytes=120,
            max_image_dimension=20_000,
            max_image_pixels=50_000_000,
            inspect_headers=True,
        )
    )

    assert report["stage_supported_queue_count"] == 4
    assert report["status_counts"]["skipped"] == 1
    assert report["status_counts"]["blocked_safety"] == 2
    assert report["status_counts"]["parse_error"] == 1
    assert report["status_reason_counts"]["stage_size_limit_exceeded"] == 1
    assert report["status_reason_counts"]["image_dimension_limit_exceeded"] == 1
    assert report["status_reason_counts"]["image_pixel_limit_exceeded"] == 1
    assert report["status_reason_counts"]["invalid_image_header"] == 1
    assert report["artifact_counts"]["derived_image_files_written"] == 0
    assert report["artifact_counts"]["ocr_text_files_written"] == 0

    with sqlite3.connect(out_dir / "mail_attachment_image_ocr_plan.sqlite") as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["status_reason"]: row["status"]
            for row in con.execute(
                "SELECT status, status_reason FROM attachment_image_ocr_plan"
            )
        }
    assert rows["stage_size_limit_exceeded"] == "skipped"
    assert rows["image_dimension_limit_exceeded"] == "blocked_safety"
    assert rows["image_pixel_limit_exceeded"] == "blocked_safety"
    assert rows["invalid_image_header"] == "parse_error"


def test_mail_attachment_image_ocr_plan_blocks_unsafe_output_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, plan_db = _build_attachment_image_ocr_fixture(
        tmp_path,
        [("m_png", 1, "safe.png", "image/png", _minimal_png_bytes(64, 64))],
    )

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_image_ocr_plan(
            MailAttachmentImageOcrPlanConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "attachment_image_ocr_plan",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_image_ocr_plan(
            MailAttachmentImageOcrPlanConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_image_ocr_plan",
            )
        )
    with pytest.raises(ValueError, match="supports only"):
        build_mail_attachment_image_ocr_plan(
            MailAttachmentImageOcrPlanConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=plan_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_image_ocr_plan_pdf",
                stage_extensions=(".pdf",),
            )
        )
    missing_plan = tmp_path / "_external_handoffs" / "missing_image_plan.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_attachment_image_ocr_plan(
            MailAttachmentImageOcrPlanConfig(
                archive_db_paths=[archive_db],
                parse_plan_db_path=missing_plan,
                out_dir=tmp_path / "_external_handoffs" / "attachment_image_ocr_plan_missing",
            )
        )
    assert not (tmp_path / "_external_handoffs" / "attachment_image_ocr_plan_missing").exists()


def test_mail_attachment_text_index_merges_extract_metadata_without_reading_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    text_db, pdf_db, image_db, parse_plan_db = _build_attachment_text_index_fixture(tmp_path)

    out_dir = tmp_path / "_external_handoffs" / "attachment_text_index"
    config = MailAttachmentTextIndexConfig(
        text_extract_db_paths=[text_db],
        pdf_extract_db_paths=[pdf_db],
        image_ocr_plan_db_paths=[image_db],
        parse_plan_db_path=parse_plan_db,
        out_dir=out_dir,
    )
    report = build_mail_attachment_text_index(config)
    rerun = build_mail_attachment_text_index(config)

    assert report["schema_version"] == "mail_attachment_text_index_v1"
    assert report["source_text_extract_count"] == 1
    assert report["source_pdf_extract_count"] == 1
    assert report["source_image_ocr_plan_count"] == 1
    assert report["parse_plan_queue_count"] == 6
    assert report["source_stage_counts"] == {
        "image_ocr_plan": 1,
        "pdf_extract": 2,
        "text_extract": 2,
    }
    assert report["text_status_counts"] == {
        "available": 2,
        "blocked": 1,
        "empty_text": 1,
        "ocr_pending": 1,
    }
    assert report["coverage_counts"]["covered_source_rows"] == 5
    assert report["coverage_counts"]["available_text_rows"] == 2
    assert report["coverage_counts"]["ocr_pending_rows"] == 1
    assert report["coverage_counts"]["parse_later_rows_without_stage5_source"] == 1
    assert rerun["text_status_counts"] == report["text_status_counts"]
    assert rerun["artifact_counts"]["rows_written"] == report["artifact_counts"]["rows_written"]
    assert report["safety"]["read_raw_eml"] is False
    assert report["safety"]["read_attachment_bytes"] is False
    assert report["safety"]["read_existing_derived_text"] is False
    assert report["safety"]["run_ocr"] is False
    assert report["privacy"]["raw_text_written_to_sqlite"] is False
    assert report["privacy"]["derived_text_paths_written"] is False
    assert report["privacy"]["derived_text_path_hashes_written"] is True

    report_text = (out_dir / "mail_attachment_text_index_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov" not in report_text
    assert "89990000000" not in report_text
    assert "source_sensitive_segment" not in report_text

    with sqlite3.connect(out_dir / "mail_attachment_text_index.sqlite") as con:
        con.row_factory = sqlite3.Row
        assert con.execute("select count(*) from stage5_sources").fetchone()[0] == 3
        assert con.execute("select count(*) from attachment_text_index").fetchone()[0] == 5
        assert con.execute("select count(*) from v_attachment_text_available").fetchone()[0] == 2
        assert con.execute("select count(*) from v_attachment_text_needs_review").fetchone()[0] == 3
        assert con.execute("select count(*) from v_attachment_ocr_pending").fetchone()[0] == 1
        rows = list(
            con.execute(
                """
                SELECT source_stage, source_status, text_status, derived_text_path_sha256
                FROM attachment_text_index
                ORDER BY source_stage, message_sha256
                """
            )
        )
    db_dump = "\n".join(str(dict(row)) for row in rows)
    assert "Client Ivanov" not in db_dump
    assert "89990000000" not in db_dump
    assert "source_sensitive_segment" not in db_dump
    assert all(
        len(row["derived_text_path_sha256"]) in {0, 64}
        for row in rows
    )


def test_mail_attachment_text_index_blocks_unsafe_output_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    text_db, _pdf_db, _image_db, _parse_plan_db = _build_attachment_text_index_fixture(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        build_mail_attachment_text_index(
            MailAttachmentTextIndexConfig(
                out_dir=tmp_path / "_external_handoffs" / "attachment_text_index_empty",
            )
        )
    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_text_index(
            MailAttachmentTextIndexConfig(
                text_extract_db_paths=[text_db],
                out_dir=tmp_path / "attachment_text_index",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_text_index(
            MailAttachmentTextIndexConfig(
                text_extract_db_paths=[text_db],
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_text_index",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_text_index(
            MailAttachmentTextIndexConfig(
                text_extract_db_paths=[
                    tmp_path / "stable_runtime" / "_external_handoffs" / "text.sqlite"
                ],
                out_dir=tmp_path / "_external_handoffs" / "attachment_text_index_stable_input",
            )
        )
    missing_db = tmp_path / "_external_handoffs" / "missing_text_extract.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_attachment_text_index(
            MailAttachmentTextIndexConfig(
                text_extract_db_paths=[missing_db],
                out_dir=tmp_path / "_external_handoffs" / "attachment_text_index_missing",
            )
        )
    assert not (tmp_path / "_external_handoffs" / "attachment_text_index_missing").exists()

    bad_db = tmp_path / "_external_handoffs" / "bad_text_extract.sqlite"
    _write_bad_schema_db(bad_db)
    with pytest.raises(ValueError, match="schema mismatch"):
        build_mail_attachment_text_index(
            MailAttachmentTextIndexConfig(
                text_extract_db_paths=[bad_db],
                out_dir=tmp_path / "_external_handoffs" / "attachment_text_index_bad_schema",
            )
        )


def test_mail_attachment_stage6_plan_summarizes_gaps_and_selects_small_ocr_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    parse_plan_db, text_index_db = _build_attachment_stage6_fixture(tmp_path)

    out_dir = tmp_path / "_external_handoffs" / "attachment_stage6_plan"
    config = MailAttachmentStage6PlanConfig(
        parse_plan_db_path=parse_plan_db,
        text_index_db_path=text_index_db,
        out_dir=out_dir,
        ocr_pilot_limit=3,
        max_pilot_attachment_bytes=1_000,
    )
    report = build_mail_attachment_stage6_plan(config)
    rerun = build_mail_attachment_stage6_plan(config)

    assert report["schema_version"] == "mail_attachment_stage6_plan_v1"
    assert report["parse_plan_queue_count"] == 10
    assert report["text_index_row_count"] == 6
    assert report["gap_count"] == 4
    assert report["gap_extension_counts"] == {
        ".gif": 1,
        ".heic": 1,
        ".html": 1,
        ".ics": 1,
    }
    assert report["gap_class_counts"] == {
        "calendar_invite_format": 1,
        "unsafe_markup_format": 1,
        "unsupported_image_format": 2,
    }
    assert report["ocr_pilot"]["candidate_count"] == 5
    assert report["ocr_pilot"]["eligible_count"] == 4
    assert report["ocr_pilot"]["selected_count"] == 3
    assert report["ocr_pilot"]["pilot_status_counts"] == {
        "deferred": 1,
        "excluded": 1,
        "selected": 3,
    }
    assert rerun["gap_extension_counts"] == report["gap_extension_counts"]
    assert rerun["ocr_pilot"]["pilot_status_counts"] == report["ocr_pilot"]["pilot_status_counts"]
    assert report["artifact_counts"]["attachment_bytes_read"] == 0
    assert report["artifact_counts"]["ocr_text_files_written"] == 0
    assert report["safety"]["run_ocr"] is False
    assert report["safety"]["read_attachment_bytes"] is False
    assert report["safety"]["decode_full_images"] is False
    assert report["privacy"]["raw_filenames_written"] is False
    assert report["privacy"]["raw_source_attachment_paths_written"] is False

    report_text = (out_dir / "mail_attachment_stage6_plan_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov" not in report_text
    assert "89990000000" not in report_text
    assert "source_sensitive_segment" not in report_text

    with sqlite3.connect(out_dir / "mail_attachment_stage6_plan.sqlite") as con:
        con.row_factory = sqlite3.Row
        selected = list(
            con.execute(
                """
                SELECT extension, pilot_rank, pilot_status
                FROM v_stage6_ocr_pilot_selected
                ORDER BY pilot_rank
                """
            )
        )
        assert con.execute("select count(*) from stage6_gap_plan").fetchone()[0] == 4
        assert con.execute("select count(*) from v_stage6_gap_needs_decision").fetchone()[0] == 4
        assert con.execute("select count(*) from stage6_ocr_pilot_plan").fetchone()[0] == 5
        assert con.execute("select count(*) from v_stage6_ocr_pilot_deferred").fetchone()[0] == 2
    assert [row["pilot_rank"] for row in selected] == [1, 2, 3]
    assert [row["extension"] for row in selected] == [".jpeg", ".jpg", ".png"]

    large_only = build_mail_attachment_stage6_plan(
        MailAttachmentStage6PlanConfig(
            parse_plan_db_path=parse_plan_db,
            text_index_db_path=text_index_db,
            out_dir=tmp_path / "_external_handoffs" / "attachment_stage6_large_only",
            ocr_pilot_limit=10,
            min_pilot_attachment_bytes=1_000,
            max_pilot_attachment_bytes=3_000,
        )
    )
    assert large_only["ocr_pilot"]["candidate_count"] == 5
    assert large_only["ocr_pilot"]["eligible_count"] == 1
    assert large_only["ocr_pilot"]["selected_count"] == 1
    assert large_only["ocr_pilot"]["min_attachment_bytes"] == 1_000
    assert large_only["ocr_pilot"]["max_attachment_bytes"] == 3_000
    assert large_only["ocr_pilot"]["pilot_status_counts"] == {
        "excluded": 4,
        "selected": 1,
    }


def test_mail_attachment_stage6_plan_blocks_unsafe_output_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    parse_plan_db, text_index_db = _build_attachment_stage6_fixture(tmp_path)

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_stage6_plan(
            MailAttachmentStage6PlanConfig(
                parse_plan_db_path=parse_plan_db,
                text_index_db_path=text_index_db,
                out_dir=tmp_path / "attachment_stage6_plan",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_stage6_plan(
            MailAttachmentStage6PlanConfig(
                parse_plan_db_path=parse_plan_db,
                text_index_db_path=text_index_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_stage6_plan",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_stage6_plan(
            MailAttachmentStage6PlanConfig(
                parse_plan_db_path=tmp_path / "stable_runtime" / "_external_handoffs" / "plan.sqlite",
                text_index_db_path=text_index_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_stage6_plan_stable_input",
            )
        )
    with pytest.raises(ValueError, match="supports only"):
        build_mail_attachment_stage6_plan(
            MailAttachmentStage6PlanConfig(
                parse_plan_db_path=parse_plan_db,
                text_index_db_path=text_index_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_stage6_plan_bad_extension",
                pilot_extensions=(".gif",),
            )
        )
    missing_index = tmp_path / "_external_handoffs" / "missing_text_index.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_attachment_stage6_plan(
            MailAttachmentStage6PlanConfig(
                parse_plan_db_path=parse_plan_db,
                text_index_db_path=missing_index,
                out_dir=tmp_path / "_external_handoffs" / "attachment_stage6_plan_missing",
            )
        )
    assert not (tmp_path / "_external_handoffs" / "attachment_stage6_plan_missing").exists()

    bad_db = tmp_path / "_external_handoffs" / "bad_stage6_index.sqlite"
    _write_bad_schema_db(bad_db)
    with pytest.raises(ValueError, match="schema mismatch"):
        build_mail_attachment_stage6_plan(
            MailAttachmentStage6PlanConfig(
                parse_plan_db_path=parse_plan_db,
                text_index_db_path=bad_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_stage6_plan_bad_schema",
            )
        )


def test_mail_attachment_ocr_preflight_verifies_selected_hashes_without_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, stage6_plan_db = _build_attachment_ocr_preflight_fixture(tmp_path)

    out_dir = tmp_path / "_external_handoffs" / "attachment_ocr_preflight"
    config = MailAttachmentOcrPreflightConfig(
        archive_db_paths=[archive_db],
        stage6_plan_db_path=stage6_plan_db,
        out_dir=out_dir,
        max_candidates=2,
        max_attachment_bytes=10_000,
    )
    report = build_mail_attachment_ocr_preflight(config)
    rerun = build_mail_attachment_ocr_preflight(config)

    assert report["schema_version"] == "mail_attachment_ocr_preflight_v1"
    assert report["selected_candidate_count"] == 2
    assert report["status_counts"] == {
        "blocked_safety": 0,
        "skipped": 0,
        "verified": 2,
    }
    assert report["artifact_counts"]["sha256_verified"] == 2
    assert report["artifact_counts"]["attachment_bytes_read"] > 0
    assert report["artifact_counts"]["ocr_text_files_written"] == 0
    assert report["safety"]["run_ocr"] is False
    assert report["safety"]["ocr_enabled"] is False
    assert report["safety"]["decode_full_images"] is False
    assert report["safety"]["extract_exif"] is False
    assert report["safety"]["read_attachment_bytes_scope"] == "selected_ocr_pilot_candidates_only"
    assert report["privacy"]["raw_filenames_written"] is False
    assert report["privacy"]["raw_source_attachment_paths_written"] is False
    assert rerun["status_counts"] == report["status_counts"]

    report_text = (out_dir / "mail_attachment_ocr_preflight_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov" not in report_text
    assert "89990000000" not in report_text
    assert "source_sensitive_segment" not in report_text

    with sqlite3.connect(out_dir / "mail_attachment_ocr_preflight.sqlite") as con:
        con.row_factory = sqlite3.Row
        assert con.execute("select count(*) from v_attachment_ocr_preflight_verified").fetchone()[0] == 2
        assert con.execute("select count(*) from v_attachment_ocr_preflight_needs_review").fetchone()[0] == 0
        rows = list(
            con.execute(
                """
                SELECT extension, preflight_status, status_reason, sha256_verified
                FROM attachment_ocr_preflight
                ORDER BY pilot_rank
                """
            )
        )
        db_dump = "\n".join(con.iterdump())
    row_dump = "\n".join(str(dict(row)) for row in rows)
    assert "Client Ivanov" not in row_dump
    assert "89990000000" not in row_dump
    assert "source_sensitive_segment" not in db_dump
    assert {row["status_reason"] for row in rows} == {"sha256_verified"}


def test_mail_attachment_ocr_preflight_blocks_unsafe_output_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, stage6_plan_db = _build_attachment_ocr_preflight_fixture(tmp_path)

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_ocr_preflight(
            MailAttachmentOcrPreflightConfig(
                archive_db_paths=[archive_db],
                stage6_plan_db_path=stage6_plan_db,
                out_dir=tmp_path / "attachment_ocr_preflight",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_ocr_preflight(
            MailAttachmentOcrPreflightConfig(
                archive_db_paths=[archive_db],
                stage6_plan_db_path=stage6_plan_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_ocr_preflight",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_ocr_preflight(
            MailAttachmentOcrPreflightConfig(
                archive_db_paths=[tmp_path / "stable_runtime" / "_external_handoffs" / "archive.sqlite"],
                stage6_plan_db_path=stage6_plan_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_ocr_preflight_stable_input",
            )
        )
    missing_stage6 = tmp_path / "_external_handoffs" / "missing_stage6.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_attachment_ocr_preflight(
            MailAttachmentOcrPreflightConfig(
                archive_db_paths=[archive_db],
                stage6_plan_db_path=missing_stage6,
                out_dir=tmp_path / "_external_handoffs" / "attachment_ocr_preflight_missing",
            )
        )
    assert not (tmp_path / "_external_handoffs" / "attachment_ocr_preflight_missing").exists()

    bad_db = tmp_path / "_external_handoffs" / "bad_ocr_preflight_stage6.sqlite"
    _write_bad_schema_db(bad_db)
    with pytest.raises(ValueError, match="schema mismatch"):
        build_mail_attachment_ocr_preflight(
            MailAttachmentOcrPreflightConfig(
                archive_db_paths=[archive_db],
                stage6_plan_db_path=bad_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_ocr_preflight_bad_schema",
            )
        )


def test_mail_attachment_ocr_pilot_runs_tesseract_only_for_verified_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    ocr_calls: list[Path] = []

    def fake_ocr(
        path: Path,
        *,
        languages: str,
        page_segmentation_mode: int,
        timeout_seconds: int,
        tesseract_thread_limit: int,
    ) -> dict[str, Any]:
        ocr_calls.append(path)
        return {
            "returncode": 0,
            "text": "OCR safe phrase Client Ivanov 89990000000",
            "stderr_present": True,
        }

    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.run_tesseract_ocr",
        fake_ocr,
    )
    archive_db, stage6_plan_db = _build_attachment_ocr_preflight_fixture(tmp_path)
    preflight_dir = tmp_path / "_external_handoffs" / "attachment_ocr_preflight"
    build_mail_attachment_ocr_preflight(
        MailAttachmentOcrPreflightConfig(
            archive_db_paths=[archive_db],
            stage6_plan_db_path=stage6_plan_db,
            out_dir=preflight_dir,
        )
    )

    out_dir = tmp_path / "_external_handoffs" / "attachment_ocr_pilot"
    config = MailAttachmentOcrPilotConfig(
        archive_db_paths=[archive_db],
        ocr_preflight_db_path=preflight_dir / "mail_attachment_ocr_preflight.sqlite",
        out_dir=out_dir,
        languages="rus+eng",
        tesseract_timeout_seconds=1,
        workers=2,
        tesseract_thread_limit=1,
    )
    report = build_mail_attachment_ocr_pilot(config)
    rerun = build_mail_attachment_ocr_pilot(config)
    reused = build_mail_attachment_ocr_pilot(
        MailAttachmentOcrPilotConfig(
            archive_db_paths=[archive_db],
            ocr_preflight_db_path=preflight_dir / "mail_attachment_ocr_preflight.sqlite",
            out_dir=out_dir,
            reuse_existing_ocr_text=True,
        )
    )

    assert report["schema_version"] == "mail_attachment_ocr_pilot_v1"
    assert report["verified_candidate_count"] == 2
    assert report["status_counts"]["extracted"] == 2
    assert report["warning_counts"] == {"tesseract_stderr_present": 2}
    assert report["artifact_counts"]["ocr_text_files_written"] == 2
    assert report["artifact_counts"]["ocr_text_files_reused"] == 0
    assert report["artifact_counts"]["images_written"] == 0
    assert report["artifact_counts"]["thumbnails_written"] == 0
    assert rerun["status_counts"] == report["status_counts"]
    assert rerun["artifact_counts"]["ocr_text_files_written"] == 0
    assert len(ocr_calls) == 4
    assert reused["status_counts"]["extracted"] == 2
    assert reused["status_reason_counts"] == {"existing_ocr_text_reused": 2}
    assert reused["artifact_counts"]["ocr_text_files_reused"] == 2
    assert reused["artifact_counts"]["attachment_bytes_submitted_to_ocr"] == 0
    assert reused["safety"]["read_existing_derived_text"] is True
    assert report["ocr"]["languages"] == "rus+eng"
    assert report["ocr"]["workers"] == 2
    assert report["ocr"]["tesseract_thread_limit"] == 1
    assert report["ocr"]["processing_wall_seconds"] >= 0
    assert report["safety"]["run_ocr"] is True
    assert report["safety"]["ocr_enabled"] is True
    assert report["safety"]["decode_full_images_scope"] == "tesseract_cli_only"
    assert report["safety"]["write_images"] is False
    assert report["privacy"]["raw_ocr_text_written_to_json"] is False
    assert report["privacy"]["raw_ocr_text_written_to_sqlite"] is False

    report_text = (out_dir / "mail_attachment_ocr_pilot_report.json").read_text(
        encoding="utf-8"
    )
    assert "Client Ivanov" not in report_text
    assert "89990000000" not in report_text

    with sqlite3.connect(out_dir / "mail_attachment_ocr_pilot.sqlite") as con:
        con.row_factory = sqlite3.Row
        assert con.execute("select count(*) from v_attachment_ocr_pilot_extracted").fetchone()[0] == 2
        assert con.execute("select count(*) from v_attachment_ocr_pilot_needs_review").fetchone()[0] == 0
        rows = list(
            con.execute(
                """
                SELECT status, status_reason, text_sha256, text_chars,
                       derived_text_path_sha256
                FROM attachment_ocr_pilot
                ORDER BY pilot_rank
                """
            )
        )
        db_dump = "\n".join(con.iterdump())
    row_dump = "\n".join(str(dict(row)) for row in rows)
    assert "Client Ivanov" not in row_dump
    assert "89990000000" not in row_dump
    assert "source_sensitive_segment" not in db_dump
    assert {row["status_reason"] for row in rows} == {"existing_ocr_text_reused"}
    assert all(len(row["derived_text_path_sha256"]) == 64 for row in rows)
    extracted_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (out_dir / "attachment_ocr_text").rglob("*.txt")
    )
    assert "OCR safe phrase" in extracted_text
    assert "Client Ivanov" in extracted_text


def test_mail_attachment_ocr_pilot_rechecks_sha256_before_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    ocr_calls: list[Path] = []

    def fake_ocr(
        path: Path,
        *,
        languages: str,
        page_segmentation_mode: int,
        timeout_seconds: int,
        tesseract_thread_limit: int,
    ) -> dict[str, Any]:
        ocr_calls.append(path)
        return {"returncode": 0, "text": "should not run", "stderr_present": False}

    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.run_tesseract_ocr",
        fake_ocr,
    )
    archive_db, stage6_plan_db = _build_attachment_ocr_preflight_fixture(tmp_path)
    preflight_dir = tmp_path / "_external_handoffs" / "attachment_ocr_preflight_sha_guard"
    build_mail_attachment_ocr_preflight(
        MailAttachmentOcrPreflightConfig(
            archive_db_paths=[archive_db],
            stage6_plan_db_path=stage6_plan_db,
            out_dir=preflight_dir,
        )
    )

    with sqlite3.connect(archive_db) as con:
        attachment_paths = [
            Path(row[0])
            for row in con.execute("SELECT path FROM attachments ORDER BY message_sha256, part_index")
        ]
    for attachment_path in attachment_paths:
        payload = attachment_path.read_bytes()
        replacement_first_byte = b"\x00" if payload[:1] != b"\x00" else b"\x01"
        attachment_path.write_bytes(replacement_first_byte + payload[1:])

    out_dir = tmp_path / "_external_handoffs" / "attachment_ocr_pilot_sha_guard"
    report = build_mail_attachment_ocr_pilot(
        MailAttachmentOcrPilotConfig(
            archive_db_paths=[archive_db],
            ocr_preflight_db_path=preflight_dir / "mail_attachment_ocr_preflight.sqlite",
            out_dir=out_dir,
        )
    )

    assert ocr_calls == []
    assert report["status_counts"]["blocked_safety"] == 2
    assert report["status_reason_counts"] == {"attachment_sha256_mismatch": 2}
    assert report["artifact_counts"]["attachment_bytes_submitted_to_ocr"] == 0
    assert report["artifact_counts"]["ocr_text_files_written"] == 0


def test_run_tesseract_ocr_hides_paths_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "_external_handoffs" / "source_sensitive_segment" / "image.bin"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"not-an-image")

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd=["tesseract", str(image_path), "stdout"],
            timeout=1,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(TimeoutError) as exc_info:
        run_tesseract_ocr(
            image_path,
            languages="rus+eng",
            page_segmentation_mode=6,
            timeout_seconds=1,
            tesseract_thread_limit=1,
        )

    assert "tesseract_timeout" in str(exc_info.value)
    assert "source_sensitive_segment" not in str(exc_info.value)


def test_mail_attachment_ocr_pilot_blocks_unsafe_output_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    archive_db, stage6_plan_db = _build_attachment_ocr_preflight_fixture(tmp_path)
    preflight_dir = tmp_path / "_external_handoffs" / "attachment_ocr_preflight_for_pilot"
    build_mail_attachment_ocr_preflight(
        MailAttachmentOcrPreflightConfig(
            archive_db_paths=[archive_db],
            stage6_plan_db_path=stage6_plan_db,
            out_dir=preflight_dir,
        )
    )
    preflight_db = preflight_dir / "mail_attachment_ocr_preflight.sqlite"

    with pytest.raises(ValueError, match="_external_handoffs"):
        build_mail_attachment_ocr_pilot(
            MailAttachmentOcrPilotConfig(
                archive_db_paths=[archive_db],
                ocr_preflight_db_path=preflight_db,
                out_dir=tmp_path / "attachment_ocr_pilot",
            )
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_attachment_ocr_pilot(
            MailAttachmentOcrPilotConfig(
                archive_db_paths=[archive_db],
                ocr_preflight_db_path=preflight_db,
                out_dir=tmp_path / "stable_runtime" / "_external_handoffs" / "attachment_ocr_pilot",
            )
        )
    missing_preflight = tmp_path / "_external_handoffs" / "missing_ocr_preflight.sqlite"
    with pytest.raises(FileNotFoundError):
        build_mail_attachment_ocr_pilot(
            MailAttachmentOcrPilotConfig(
                archive_db_paths=[archive_db],
                ocr_preflight_db_path=missing_preflight,
                out_dir=tmp_path / "_external_handoffs" / "attachment_ocr_pilot_missing",
            )
        )
    assert not (tmp_path / "_external_handoffs" / "attachment_ocr_pilot_missing").exists()
    bad_db = tmp_path / "_external_handoffs" / "bad_ocr_pilot_preflight.sqlite"
    _write_bad_schema_db(bad_db)
    with pytest.raises(ValueError, match="schema mismatch"):
        build_mail_attachment_ocr_pilot(
            MailAttachmentOcrPilotConfig(
                archive_db_paths=[archive_db],
                ocr_preflight_db_path=bad_db,
                out_dir=tmp_path / "_external_handoffs" / "attachment_ocr_pilot_bad_schema",
            )
        )


def test_mail_archive_ingest_records_fetch_errors_and_keeps_successes(tmp_path: Path) -> None:
    fake_imap = FakeImapClient([_raw_message(), None])

    report = build_mail_archive_ingest(
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
            max_messages=2,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )

    assert report["messages_attempted"] == 2
    assert report["messages_inserted_or_seen"] == 1
    assert len(report["errors"]) == 1
    assert "IMAP FETCH failed" in report["errors"][0]["error"]
    with sqlite3.connect(tmp_path / "archive" / "mail_archive.sqlite") as con:
        assert con.execute("select count(*) from messages").fetchone()[0] == 1


def test_mail_archive_verification_blocks_bad_safety_and_raw_db_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mango_mvp.productization.mail_archive.git_check_ignored",
        lambda _path: True,
    )
    report = build_mail_archive_ingest(
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
            max_messages=1,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=FakeImapClient([_raw_message()]),
    )

    report_path = tmp_path / "archive" / "mail_ingest_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["safety"]["readonly_select"] = False
    payload["safety"]["fetch_uses_body_peek"] = False
    payload["safety"]["write_tallanto"] = True
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for raw_path in Path(report["paths"]["raw_eml_dir"]).glob("*/*.eml"):
        raw_path.unlink()

    verification = verify_mail_archive_pilot(
        MailArchiveVerificationConfig(
            archive_dir=tmp_path / "archive",
            expected_max_messages=1,
        )
    )

    assert verification["verification_pass"] is False
    assert "readonly_select_not_confirmed" in verification["blocking_risks"]
    assert "body_peek_not_confirmed" in verification["blocking_risks"]
    assert "write_tallanto_not_false" in verification["blocking_risks"]
    assert "raw_eml_count_mismatch" in verification["blocking_risks"]


def test_mail_archive_stable_runtime_guards_cover_ingest_and_matching(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_archive_ingest(
            credentials=MailImapCredentials(
                host="mail.example.test",
                port=993,
                email_address="school@kmipt.ru",
                password="not-written",
            ),
            config=MailArchiveIngestConfig(
                out_dir=tmp_path / "stable_runtime" / "archive",
                max_messages=0,
            ),
            client=FakeImapClient([_raw_message()]),
        )

    with pytest.raises(ValueError, match="stable_runtime"):
        build_mail_matching_report(
            MailMatchingReportConfig(
                archive_db_path=tmp_path / "stable_runtime" / "mail_archive.sqlite",
                identity_db_path=tmp_path / "identity.sqlite",
                out_dir=tmp_path / "matching",
            )
        )


class FakeImapClient:
    def __init__(self, messages: list[bytes | None]) -> None:
        self.messages = messages
        self.selected_readonly: list[tuple[str, bool]] = []
        self.fetch_queries: list[str] = []
        self.search_criteria: list[tuple[str, ...]] = []

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
        self.search_criteria.append(tuple(criteria))
        return "OK", [b" ".join(str(index).encode("ascii") for index in range(1, len(self.messages) + 1))]

    def fetch(self, message_set: bytes | str, message_parts: str) -> tuple[str, Sequence[Any]]:
        self.fetch_queries.append(message_parts)
        index = int(message_set) - 1
        if self.messages[index] is None:
            return "NO", [b"fetch failed"]
        return "OK", [(b"1 FETCH", self.messages[index])]

    def store(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("STORE must not be called")

    def copy(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("COPY must not be called")

    def expunge(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("EXPUNGE must not be called")

    def delete(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("DELETE must not be called")

    def append(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("APPEND must not be called")

    def close(self) -> tuple[str, Sequence[bytes]]:
        return "OK", []

    def logout(self) -> tuple[str, Sequence[bytes]]:
        return "OK", []


def _build_attachment_plan_archive(tmp_path: Path) -> Path:
    db_path = tmp_path / "_external_handoffs" / "attachment_source" / "mail_archive.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE attachments (
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              filename TEXT,
              content_type TEXT,
              content_disposition TEXT,
              size_bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              path TEXT NOT NULL,
              PRIMARY KEY (message_sha256, part_index)
            );
            """
        )
        rows = [
            ("m1", 1, "invoice.pdf", "application/pdf", 20_000_000),
            (
                "m1",
                2,
                "report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                128,
            ),
            ("m2", 3, "invoice.pdf.exe", "application/pdf", 128),
            ("m3", 4, "archive.zip", "application/zip", 128),
            ("m4", 5, "legacy.doc", "application/msword", 128),
            ("m5", 6, "big.pdf", "application/pdf", 20_000_001),
            ("m6", 7, "../traverse.pdf", "application/pdf", 128),
        ]
        con.executemany(
            """
            INSERT INTO attachments (
              message_sha256, part_index, filename, content_type,
              content_disposition, size_bytes, sha256, path
            ) VALUES (?, ?, ?, ?, 'attachment', ?, ?, '')
            """,
            [
                (
                    message_sha,
                    part_index,
                    filename,
                    content_type,
                    size_bytes,
                    hashlib.sha256(f"{message_sha}:{part_index}".encode("utf-8")).hexdigest(),
                )
                for message_sha, part_index, filename, content_type, size_bytes in rows
            ],
        )
        con.commit()
    return db_path


def _build_attachment_text_extract_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "_external_handoffs" / "attachment_text_source"
    attachments = [
        (
            "m_txt",
            1,
            "Client Ivanov 89990000000.txt",
            "text/plain",
            b"TXT safe line\nsecond line",
        ),
        (
            "m_csv",
            1,
            "table.csv",
            "text/csv",
            b"label,value\nrow,csv safe value\n",
        ),
        (
            "m_docx",
            1,
            "doc.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _minimal_docx_bytes("Docx safe phrase"),
        ),
        (
            "m_xlsx",
            1,
            "book.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            _minimal_xlsx_bytes("Xlsx safe phrase"),
        ),
        (
            "m_pdf",
            1,
            "future.pdf",
            "application/pdf",
            b"%PDF-not-parsed-in-stage-two",
        ),
    ]
    archive_db = _write_attachment_text_archive(root, attachments)
    report = build_mail_attachment_parse_plan(
        MailAttachmentParsePlanConfig(
            archive_db_paths=[archive_db],
            out_dir=tmp_path / "_external_handoffs" / "attachment_text_plan",
        )
    )
    assert report["parse_plan_counts"]["parse_later"] == 5
    return archive_db, tmp_path / "_external_handoffs" / "attachment_text_plan" / "mail_attachment_parse_plan.sqlite"


def _build_attachment_text_extract_limit_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "_external_handoffs" / "attachment_text_limit_source"
    attachments = [
        ("m_big", 1, "big.txt", "text/plain", b"A" * 1_200),
        ("m_long", 1, "long.txt", "text/plain", b"B" * 40),
        (
            "m_macro",
            1,
            "macro.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _minimal_docx_bytes("macro hidden", extra_entries={"word/vbaProject.bin": b"macro"}),
        ),
        (
            "m_bad",
            1,
            "bad.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            b"not a zip",
        ),
    ]
    archive_db = _write_attachment_text_archive(root, attachments)
    build_mail_attachment_parse_plan(
        MailAttachmentParsePlanConfig(
            archive_db_paths=[archive_db],
            out_dir=tmp_path / "_external_handoffs" / "attachment_text_limit_plan",
        )
    )
    return archive_db, tmp_path / "_external_handoffs" / "attachment_text_limit_plan" / "mail_attachment_parse_plan.sqlite"


def _build_attachment_pdf_extract_fixture(
    tmp_path: Path,
    attachments: Sequence[tuple[str, int, str, str, bytes]],
) -> tuple[Path, Path]:
    root = tmp_path / "_external_handoffs" / "attachment_pdf_source"
    archive_db = _write_attachment_text_archive(root, attachments)
    build_mail_attachment_parse_plan(
        MailAttachmentParsePlanConfig(
            archive_db_paths=[archive_db],
            out_dir=tmp_path / "_external_handoffs" / "attachment_pdf_plan",
        )
    )
    return archive_db, tmp_path / "_external_handoffs" / "attachment_pdf_plan" / "mail_attachment_parse_plan.sqlite"


def _build_attachment_image_ocr_fixture(
    tmp_path: Path,
    attachments: Sequence[tuple[str, int, str, str, bytes]],
) -> tuple[Path, Path]:
    root = tmp_path / "_external_handoffs" / "attachment_image_source_sensitive_segment"
    archive_db = _write_attachment_text_archive(root, attachments)
    build_mail_attachment_parse_plan(
        MailAttachmentParsePlanConfig(
            archive_db_paths=[archive_db],
            out_dir=tmp_path / "_external_handoffs" / "attachment_image_plan",
        )
    )
    return archive_db, tmp_path / "_external_handoffs" / "attachment_image_plan" / "mail_attachment_parse_plan.sqlite"


def _build_attachment_text_index_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "_external_handoffs" / "attachment_text_index_source_sensitive_segment"
    text_db = root / "stage2" / "mail_attachment_text_extract.sqlite"
    pdf_db = root / "stage3" / "mail_attachment_pdf_extract.sqlite"
    image_db = root / "stage4" / "mail_attachment_image_ocr_plan.sqlite"
    parse_plan_db = root / "stage1" / "mail_attachment_parse_plan.sqlite"

    sensitive_text_path = (
        root
        / "stage2"
        / "attachment_text"
        / "source_sensitive_segment"
        / "Client Ivanov 89990000000.txt"
    )
    pdf_text_path = root / "stage3" / "attachment_pdf_text" / "pdf-safe.txt"
    _write_text_extract_db(
        text_db,
        [
            {
                "source_archive_id": "archive_text",
                "message_sha256": "m_text",
                "part_index": 1,
                "attachment_sha256": "a_text",
                "extension": ".txt",
                "declared_content_type": "text/plain",
                "size_bytes": 12,
                "parser": "text_plain",
                "status": "extracted",
                "status_reason": "ok",
                "warnings_json": "[]",
                "text_sha256": hashlib.sha256(b"Client Ivanov 89990000000").hexdigest(),
                "text_chars": 28,
                "text_truncated": 0,
                "derived_text_path": str(sensitive_text_path),
            },
            {
                "source_archive_id": "archive_text",
                "message_sha256": "m_blocked",
                "part_index": 2,
                "attachment_sha256": "a_blocked",
                "extension": ".docx",
                "declared_content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "size_bytes": 256,
                "parser": "",
                "status": "blocked_safety",
                "status_reason": "office_external_relationship_declared",
                "warnings_json": '["office_external_relationship_declared"]',
                "text_sha256": "",
                "text_chars": 0,
                "text_truncated": 0,
                "derived_text_path": "",
            },
        ],
    )
    _write_pdf_extract_db(
        pdf_db,
        [
            {
                "source_archive_id": "archive_pdf",
                "message_sha256": "m_pdf",
                "part_index": 1,
                "attachment_sha256": "a_pdf",
                "extension": ".pdf",
                "declared_content_type": "application/pdf",
                "size_bytes": 512,
                "parser": "pypdf_text",
                "status": "extracted",
                "status_reason": "ok",
                "warnings_json": "[]",
                "page_count": 1,
                "pages_processed": 1,
                "text_sha256": hashlib.sha256(b"PDF safe phrase").hexdigest(),
                "text_chars": 15,
                "text_truncated": 0,
                "derived_text_path": str(pdf_text_path),
            },
            {
                "source_archive_id": "archive_pdf",
                "message_sha256": "m_empty",
                "part_index": 2,
                "attachment_sha256": "a_empty",
                "extension": ".pdf",
                "declared_content_type": "application/pdf",
                "size_bytes": 128,
                "parser": "pypdf_text",
                "status": "empty_text",
                "status_reason": "no_text_extracted",
                "warnings_json": "[]",
                "page_count": 1,
                "pages_processed": 1,
                "text_sha256": "",
                "text_chars": 0,
                "text_truncated": 0,
                "derived_text_path": "",
            },
        ],
    )
    _write_image_ocr_plan_db(
        image_db,
        [
            {
                "source_archive_id": "archive_image",
                "message_sha256": "m_image",
                "part_index": 1,
                "attachment_sha256": "a_image",
                "extension": ".png",
                "declared_content_type": "image/png",
                "size_bytes": 64,
                "image_format": "not_inspected",
                "width": 0,
                "height": 0,
                "pixel_count": 0,
                "status": "planned",
                "status_reason": "ocr_disabled",
                "ocr_status": "disabled",
                "warnings_json": "[]",
            }
        ],
    )
    _write_attachment_text_index_parse_plan(parse_plan_db, parse_later_count=6)
    return text_db, pdf_db, image_db, parse_plan_db


def _build_attachment_stage6_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "_external_handoffs" / "attachment_stage6_source_sensitive_segment"
    parse_plan_db = root / "stage1" / "mail_attachment_parse_plan.sqlite"
    text_index_db = root / "stage5" / "mail_attachment_text_index.sqlite"
    image_rows = [
        ("archive_image", "m_jpeg", 1, "a_jpeg", ".jpeg", "image/jpeg", 300, "image_ocr_plan", "ocr_pending"),
        ("archive_image", "m_jpg", 1, "a_jpg", ".jpg", "image/jpeg", 200, "image_ocr_plan", "ocr_pending"),
        ("archive_image", "m_png", 1, "a_png", ".png", "image/png", 100, "image_ocr_plan", "ocr_pending"),
        ("archive_image", "m_webp", 1, "a_webp", ".webp", "image/webp", 50, "image_ocr_plan", "ocr_pending"),
        ("archive_image", "m_big", 1, "a_big", ".jpg", "image/jpeg", 2_000, "image_ocr_plan", "ocr_pending"),
    ]
    available_row = (
        "archive_text",
        "m_text",
        1,
        "a_text",
        ".txt",
        "text/plain",
        20,
        "text_extract",
        "available",
    )
    _write_stage6_parse_plan_db(
        parse_plan_db,
        [
            *[
                {
                    "source_archive_id": source_archive_id,
                    "message_sha256": message_sha256,
                    "part_index": part_index,
                    "attachment_sha256": attachment_sha256,
                    "extension": extension,
                    "declared_content_type": content_type,
                    "size_bytes": size_bytes,
                }
                for (
                    source_archive_id,
                    message_sha256,
                    part_index,
                    attachment_sha256,
                    extension,
                    content_type,
                    size_bytes,
                    _source_stage,
                    _text_status,
                ) in [*image_rows, available_row]
            ],
            {
                "source_archive_id": "archive_gap",
                "message_sha256": "m_gif",
                "part_index": 1,
                "attachment_sha256": "a_gif",
                "extension": ".gif",
                "declared_content_type": "image/gif",
                "size_bytes": 500,
            },
            {
                "source_archive_id": "archive_gap",
                "message_sha256": "m_heic",
                "part_index": 1,
                "attachment_sha256": "a_heic",
                "extension": ".heic",
                "declared_content_type": "image/heic",
                "size_bytes": 500,
            },
            {
                "source_archive_id": "archive_gap",
                "message_sha256": "m_ics",
                "part_index": 1,
                "attachment_sha256": "a_ics",
                "extension": ".ics",
                "declared_content_type": "application/ics",
                "size_bytes": 500,
            },
            {
                "source_archive_id": "archive_gap",
                "message_sha256": "m_html",
                "part_index": 1,
                "attachment_sha256": "a_html",
                "extension": ".html",
                "declared_content_type": "text/html",
                "size_bytes": 500,
            },
        ],
    )
    _write_stage6_text_index_db(
        text_index_db,
        [
            *[
                {
                    "source_stage": source_stage,
                    "source_archive_id": source_archive_id,
                    "message_sha256": message_sha256,
                    "part_index": part_index,
                    "attachment_sha256": attachment_sha256,
                    "extension": extension,
                    "declared_content_type": content_type,
                    "size_bytes": size_bytes,
                    "text_status": text_status,
                }
                for (
                    source_archive_id,
                    message_sha256,
                    part_index,
                    attachment_sha256,
                    extension,
                    content_type,
                    size_bytes,
                    source_stage,
                    text_status,
                ) in image_rows
            ],
            {
                "source_stage": available_row[7],
                "source_archive_id": available_row[0],
                "message_sha256": available_row[1],
                "part_index": available_row[2],
                "attachment_sha256": available_row[3],
                "extension": available_row[4],
                "declared_content_type": available_row[5],
                "size_bytes": available_row[6],
                "text_status": available_row[8],
            },
        ],
    )
    return parse_plan_db, text_index_db


def _build_attachment_ocr_preflight_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "_external_handoffs" / "attachment_ocr_preflight_source_sensitive_segment"
    attachments = [
        (
            "m_png",
            1,
            "Client Ivanov 89990000000.png",
            "image/png",
            _minimal_png_bytes(64, 64),
        ),
        (
            "m_jpg",
            1,
            "scan.jpg",
            "image/jpeg",
            _minimal_jpeg_bytes(64, 64),
        ),
    ]
    archive_db = _write_attachment_text_archive(root, attachments)
    source_archive_id = hashlib.sha256(
        str(Path(archive_db).resolve(strict=False)).encode("utf-8")
    ).hexdigest()[:16]
    selected_rows = []
    for rank, (message_sha, part_index, _filename, content_type, payload) in enumerate(
        attachments,
        start=1,
    ):
        attachment_sha = hashlib.sha256(payload).hexdigest()
        selected_rows.append(
            {
                "source_archive_id": source_archive_id,
                "message_sha256": message_sha,
                "part_index": part_index,
                "attachment_sha256": attachment_sha,
                "extension": ".png" if content_type == "image/png" else ".jpg",
                "declared_content_type": content_type,
                "size_bytes": len(payload),
                "pilot_rank": rank,
            }
        )
    stage6_plan_db = root / "stage6" / "mail_attachment_stage6_plan.sqlite"
    _write_ocr_preflight_stage6_plan_db(stage6_plan_db, selected_rows)
    return archive_db, stage6_plan_db


def _write_ocr_preflight_stage6_plan_db(
    db_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE stage6_ocr_pilot_plan (
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              attachment_sha256 TEXT NOT NULL,
              extension TEXT NOT NULL,
              declared_content_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              pilot_status TEXT NOT NULL,
              pilot_rank INTEGER NOT NULL,
              pilot_batch_label TEXT NOT NULL,
              status_reason TEXT NOT NULL,
              planned_at TEXT NOT NULL,
              PRIMARY KEY (source_archive_id, message_sha256, part_index, attachment_sha256)
            );
            """
        )
        con.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', 'mail_attachment_stage6_plan_v1')"
        )
        for row in rows:
            con.execute(
                """
                INSERT INTO stage6_ocr_pilot_plan (
                  source_archive_id, message_sha256, part_index, attachment_sha256,
                  extension, declared_content_type, size_bytes, pilot_status,
                  pilot_rank, pilot_batch_label, status_reason, planned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'selected', ?, 'pilot_001',
                          'balanced_extension_sample', '2026-05-13T00:00:00+00:00')
                """,
                (
                    row["source_archive_id"],
                    row["message_sha256"],
                    int(row["part_index"]),
                    row["attachment_sha256"],
                    row["extension"],
                    row["declared_content_type"],
                    int(row["size_bytes"]),
                    int(row["pilot_rank"]),
                ),
            )
        con.commit()


def _write_stage6_parse_plan_db(db_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE attachment_parse_plan (
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              attachment_sha256 TEXT NOT NULL,
              filename_sha256 TEXT NOT NULL,
              extension TEXT NOT NULL,
              declared_content_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              risk_level TEXT NOT NULL,
              action TEXT NOT NULL,
              risk_reasons_json TEXT NOT NULL,
              planned_at TEXT NOT NULL,
              PRIMARY KEY (source_archive_id, message_sha256, part_index, attachment_sha256)
            );
            """
        )
        con.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', 'mail_attachment_parse_plan_v1')"
        )
        for row in rows:
            con.execute(
                """
                INSERT INTO attachment_parse_plan (
                  source_archive_id, message_sha256, part_index, attachment_sha256,
                  filename_sha256, extension, declared_content_type, size_bytes,
                  risk_level, action, risk_reasons_json, planned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'low', 'parse_later', '[]', '2026-05-13T00:00:00+00:00')
                """,
                (
                    row["source_archive_id"],
                    row["message_sha256"],
                    int(row["part_index"]),
                    row["attachment_sha256"],
                    hashlib.sha256(
                        f"{row['message_sha256']}:{row['part_index']}".encode("utf-8")
                    ).hexdigest(),
                    row["extension"],
                    row["declared_content_type"],
                    int(row["size_bytes"]),
                ),
            )
        con.commit()


def _write_stage6_text_index_db(db_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE attachment_text_index (
              source_db_id TEXT NOT NULL,
              source_stage TEXT NOT NULL,
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              attachment_sha256 TEXT NOT NULL,
              extension TEXT NOT NULL,
              declared_content_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              parser TEXT NOT NULL,
              source_status TEXT NOT NULL,
              source_status_reason TEXT NOT NULL,
              text_status TEXT NOT NULL,
              text_sha256 TEXT NOT NULL,
              text_chars INTEGER NOT NULL,
              text_truncated INTEGER NOT NULL,
              derived_text_path_sha256 TEXT NOT NULL,
              page_count INTEGER NOT NULL,
              pages_processed INTEGER NOT NULL,
              warnings_json TEXT NOT NULL,
              indexed_at TEXT NOT NULL,
              PRIMARY KEY (
                source_db_id, source_stage, source_archive_id,
                message_sha256, part_index, attachment_sha256
              )
            );
            """
        )
        con.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', 'mail_attachment_text_index_v1')"
        )
        for row in rows:
            con.execute(
                """
                INSERT INTO attachment_text_index (
                  source_db_id, source_stage, source_archive_id, message_sha256,
                  part_index, attachment_sha256, extension, declared_content_type,
                  size_bytes, parser, source_status, source_status_reason,
                  text_status, text_sha256, text_chars, text_truncated,
                  derived_text_path_sha256, page_count, pages_processed,
                  warnings_json, indexed_at
                ) VALUES (
                  'source_db', ?, ?, ?, ?, ?, ?, ?, ?, '', 'planned',
                  'ocr_disabled', ?, '', 0, 0, '', 0, 0, '[]',
                  '2026-05-13T00:00:00+00:00'
                )
                """,
                (
                    row["source_stage"],
                    row["source_archive_id"],
                    row["message_sha256"],
                    int(row["part_index"]),
                    row["attachment_sha256"],
                    row["extension"],
                    row["declared_content_type"],
                    int(row["size_bytes"]),
                    row["text_status"],
                ),
            )
        con.commit()


def _write_text_extract_db(db_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE attachment_text_extracts (
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              attachment_sha256 TEXT NOT NULL,
              extension TEXT NOT NULL,
              declared_content_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              parser TEXT NOT NULL,
              status TEXT NOT NULL,
              status_reason TEXT NOT NULL,
              warnings_json TEXT NOT NULL,
              text_sha256 TEXT NOT NULL,
              text_chars INTEGER NOT NULL,
              text_truncated INTEGER NOT NULL,
              derived_text_path TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              PRIMARY KEY (source_archive_id, message_sha256, part_index, attachment_sha256)
            );
            """
        )
        con.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', 'mail_attachment_text_extract_v1')"
        )
        for row in rows:
            con.execute(
                """
                INSERT INTO attachment_text_extracts (
                  source_archive_id, message_sha256, part_index, attachment_sha256,
                  extension, declared_content_type, size_bytes, parser, status,
                  status_reason, warnings_json, text_sha256, text_chars,
                  text_truncated, derived_text_path, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-05-13T00:00:00+00:00')
                """,
                (
                    row["source_archive_id"],
                    row["message_sha256"],
                    int(row["part_index"]),
                    row["attachment_sha256"],
                    row["extension"],
                    row["declared_content_type"],
                    int(row["size_bytes"]),
                    row["parser"],
                    row["status"],
                    row["status_reason"],
                    row["warnings_json"],
                    row["text_sha256"],
                    int(row["text_chars"]),
                    int(row["text_truncated"]),
                    row["derived_text_path"],
                ),
            )
        con.commit()


def _write_pdf_extract_db(db_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE attachment_pdf_extracts (
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              attachment_sha256 TEXT NOT NULL,
              extension TEXT NOT NULL,
              declared_content_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              parser TEXT NOT NULL,
              status TEXT NOT NULL,
              status_reason TEXT NOT NULL,
              warnings_json TEXT NOT NULL,
              page_count INTEGER NOT NULL,
              pages_processed INTEGER NOT NULL,
              text_sha256 TEXT NOT NULL,
              text_chars INTEGER NOT NULL,
              text_truncated INTEGER NOT NULL,
              derived_text_path TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              PRIMARY KEY (source_archive_id, message_sha256, part_index, attachment_sha256)
            );
            """
        )
        con.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', 'mail_attachment_pdf_extract_v1')"
        )
        for row in rows:
            con.execute(
                """
                INSERT INTO attachment_pdf_extracts (
                  source_archive_id, message_sha256, part_index, attachment_sha256,
                  extension, declared_content_type, size_bytes, parser, status,
                  status_reason, warnings_json, page_count, pages_processed,
                  text_sha256, text_chars, text_truncated, derived_text_path,
                  extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-05-13T00:00:00+00:00')
                """,
                (
                    row["source_archive_id"],
                    row["message_sha256"],
                    int(row["part_index"]),
                    row["attachment_sha256"],
                    row["extension"],
                    row["declared_content_type"],
                    int(row["size_bytes"]),
                    row["parser"],
                    row["status"],
                    row["status_reason"],
                    row["warnings_json"],
                    int(row["page_count"]),
                    int(row["pages_processed"]),
                    row["text_sha256"],
                    int(row["text_chars"]),
                    int(row["text_truncated"]),
                    row["derived_text_path"],
                ),
            )
        con.commit()


def _write_image_ocr_plan_db(db_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE attachment_image_ocr_plan (
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              attachment_sha256 TEXT NOT NULL,
              extension TEXT NOT NULL,
              declared_content_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              image_format TEXT NOT NULL,
              width INTEGER NOT NULL,
              height INTEGER NOT NULL,
              pixel_count INTEGER NOT NULL,
              status TEXT NOT NULL,
              status_reason TEXT NOT NULL,
              ocr_status TEXT NOT NULL,
              warnings_json TEXT NOT NULL,
              planned_at TEXT NOT NULL,
              PRIMARY KEY (source_archive_id, message_sha256, part_index, attachment_sha256)
            );
            """
        )
        con.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', 'mail_attachment_image_ocr_plan_v1')"
        )
        for row in rows:
            con.execute(
                """
                INSERT INTO attachment_image_ocr_plan (
                  source_archive_id, message_sha256, part_index, attachment_sha256,
                  extension, declared_content_type, size_bytes, image_format,
                  width, height, pixel_count, status, status_reason, ocr_status,
                  warnings_json, planned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-05-13T00:00:00+00:00')
                """,
                (
                    row["source_archive_id"],
                    row["message_sha256"],
                    int(row["part_index"]),
                    row["attachment_sha256"],
                    row["extension"],
                    row["declared_content_type"],
                    int(row["size_bytes"]),
                    row["image_format"],
                    int(row["width"]),
                    int(row["height"]),
                    int(row["pixel_count"]),
                    row["status"],
                    row["status_reason"],
                    row["ocr_status"],
                    row["warnings_json"],
                ),
            )
        con.commit()


def _write_attachment_text_index_parse_plan(db_path: Path, *, parse_later_count: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE attachment_parse_plan (
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              attachment_sha256 TEXT NOT NULL,
              action TEXT NOT NULL
            );
            """
        )
        con.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', 'mail_attachment_parse_plan_v1')"
        )
        con.executemany(
            """
            INSERT INTO attachment_parse_plan (
              source_archive_id, message_sha256, part_index, attachment_sha256, action
            ) VALUES ('archive', ?, 1, ?, 'parse_later')
            """,
            [(f"m_{index}", f"a_{index}") for index in range(parse_later_count)],
        )
        con.commit()


def _write_bad_schema_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta (key, value) VALUES ('schema_version', 'bad_schema_v1');
            """
        )


def _write_attachment_text_archive(
    root: Path,
    attachments: Sequence[tuple[str, int, str, str, bytes]],
) -> Path:
    db_path = root / "archive" / "mail_archive.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE attachments (
              message_sha256 TEXT NOT NULL,
              part_index INTEGER NOT NULL,
              filename TEXT,
              content_type TEXT,
              content_disposition TEXT,
              size_bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              path TEXT NOT NULL,
              PRIMARY KEY (message_sha256, part_index)
            );
            """
        )
        for message_sha, part_index, filename, content_type, payload in attachments:
            attachment_sha = hashlib.sha256(payload).hexdigest()
            path = (
                root
                / "archive"
                / "attachments"
                / message_sha
                / f"part_{part_index:03d}_{attachment_sha[:16]}.bin"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            con.execute(
                """
                INSERT INTO attachments (
                  message_sha256, part_index, filename, content_type,
                  content_disposition, size_bytes, sha256, path
                ) VALUES (?, ?, ?, ?, 'attachment', ?, ?, ?)
                """,
                (
                    message_sha,
                    part_index,
                    filename,
                    content_type,
                    len(payload),
                    attachment_sha,
                    str(path),
                ),
            )
        con.commit()
    return db_path


def _minimal_docx_bytes(text: str, *, extra_entries: Mapping[str, bytes] | None = None) -> bytes:
    body = "".join(
        f"<w:p><w:r><w:t>{chunk}</w:t></w:r></w:p>"
        for chunk in text.replace("&", "&amp;").replace("<", "&lt;").splitlines()
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", document_xml.encode("utf-8"))
        for name, value in (extra_entries or {}).items():
            zf.writestr(name, value)
    return payload.getvalue()


def _minimal_xlsx_bytes(text: str) -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["label", "value"])
    worksheet.append(["row", text])
    payload = io.BytesIO()
    workbook.save(payload)
    workbook.close()
    return payload.getvalue()


def _minimal_pdf_bytes(pages: Sequence[str]) -> bytes:
    def esc(value: str) -> str:
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    page_refs: list[str] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"")
    objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    for index, text in enumerate(pages):
        page_id = 4 + index * 2
        content_id = page_id + 1
        page_refs.append(f"{page_id} 0 R")
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({esc(text.replace(chr(10), ' '))}) Tj ET"
        ).encode("latin-1", errors="replace")
        objects.append(
            (
                f"{page_id} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>\n"
                "endobj\n"
            ).encode("ascii")
        )
        objects.append(
            (
                f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n"
            ).encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        )
    objects[1] = (
        f"2 0 obj\n<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {len(page_refs)} >>\n"
        "endobj\n"
    ).encode("ascii")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def _encrypted_pdf_bytes() -> bytes:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    payload = io.BytesIO()
    writer.write(payload)
    return payload.getvalue()


def _minimal_png_bytes(width: int, height: int, *, with_exif: bool = False) -> bytes:
    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )
    chunks = [
        len(ihdr).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00\x00\x00\x00",
    ]
    if with_exif:
        exif = b"Exif\x00\x00SYNTHETIC"
        chunks.append(len(exif).to_bytes(4, "big") + b"eXIf" + exif + b"\x00\x00\x00\x00")
    chunks.append(b"\x00\x00\x00\x00IEND\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _minimal_jpeg_bytes(width: int, height: int, *, with_exif: bool = False) -> bytes:
    parts = [b"\xff\xd8"]
    if with_exif:
        app1 = b"Exif\x00\x00"
        parts.append(b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1)
    sof = (
        b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00"
    )
    parts.append(b"\xff\xc0" + (len(sof) + 2).to_bytes(2, "big") + sof)
    parts.append(b"\xff\xd9")
    return b"".join(parts)


def _minimal_webp_vp8x_bytes(width: int, height: int) -> bytes:
    body = (
        b"VP8X"
        + (10).to_bytes(4, "little")
        + b"\x00\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    return b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WEBP" + body


def _build_mixed_match_archive(tmp_path: Path) -> tuple[Path, Path]:
    tallanto_csv = tmp_path / "students.csv"
    _write_tallanto_csv(
        tallanto_csv,
        [
            {"ID": "T-1", "E-mail": "unique@example.com"},
            {"ID": "T-2", "E-mail": "shared@example.com"},
            {"ID": "T-3", "E-mail": "shared@example.com"},
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
    fake_imap = FakeImapClient(
        [
            _raw_message(message_id="m-unique", from_addr="Unique <unique@example.com>"),
            _raw_message(message_id="m-shared", from_addr="Shared <shared@example.com>"),
            _raw_message(message_id="m-missing", from_addr="Missing <missing@example.com>"),
            _raw_message(
                message_id="m-internal",
                from_addr="Colleague <colleague@kmipt.ru>",
                to_addr="School <school@kmipt.ru>",
            ),
            _raw_message(message_id="m-service", from_addr="No Reply <no-reply@example.com>"),
        ]
    )
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
            max_messages=5,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )
    return Path(ingest_report["paths"]["archive_db"]), Path(identity_report["paths"]["identity_db"])


def _build_phone_lift_archive(tmp_path: Path) -> tuple[Path, Path]:
    tallanto_csv = tmp_path / "phone_lift_students.csv"
    _write_tallanto_csv(
        tallanto_csv,
        [
            {
                "ID": "T-1",
                "E-mail": "unique@example.com",
                "Тел. (родителя)": "8 999 000 00 00",
            },
            {
                "ID": "T-2",
                "E-mail": "shared@example.com",
                "Тел. (родителя)": "+7 999 111-22-33",
            },
            {
                "ID": "T-3",
                "E-mail": "shared@example.com",
                "Тел. (родителя)": "+7 999 333-44-55",
            },
            {
                "ID": "T-4",
                "E-mail": "dup-one@example.com",
                "Тел. (родителя)": "+7 999 222-33-44",
            },
            {
                "ID": "T-5",
                "E-mail": "dup-two@example.com",
                "Тел. (родителя)": "8 999 222 33 44",
            },
        ],
    )
    identity_report = build_tallanto_identity_map(
        TallantoIdentityMapConfig(
            tallanto_csv_path=tallanto_csv,
            out_dir=tmp_path / "identity_phone_lift",
            encoding="utf-8",
            delimiter="\t",
        )
    )
    fake_imap = FakeImapClient(
        [
            _raw_message(
                message_id="m-ambiguous-phone-strong",
                from_addr="Shared <shared@example.com>",
                body="Перезвоните по номеру +7 999 000-00-00.",
            ),
            _raw_message(
                message_id="m-missing-phone-strong",
                from_addr="Missing <missing@example.com>",
                body="Удобный телефон: +7 999 111-22-33.",
            ),
            _raw_message(
                message_id="m-missing-no-phone",
                from_addr="Missing Two <missing-two@example.com>",
                body="Телефон не указан, нужна консультация по договору.",
            ),
            _raw_message(
                message_id="m-missing-phone-duplicate",
                from_addr="Missing Three <missing-three@example.com>",
                body="Контактный номер +7 999 222-33-44.",
            ),
            _raw_message(
                message_id="m-missing-phone-no-identity",
                from_addr="Missing Four <missing-four@example.com>",
                body="Можно связаться по +7 999 555-66-77.",
            ),
        ]
    )
    ingest_report = build_mail_archive_ingest(
        credentials=MailImapCredentials(
            host="mail.example.test",
            port=993,
            email_address="school@kmipt.ru",
            password="not-written",
        ),
        config=MailArchiveIngestConfig(
            out_dir=tmp_path / "_external_handoffs" / "phone_lift_archive",
            mailbox="INBOX",
            mailbox_label="INBOX",
            since_days=7,
            max_messages=5,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )
    archive_db = Path(ingest_report["paths"]["archive_db"])
    build_mail_matching_report(
        MailMatchingReportConfig(
            archive_db_path=archive_db,
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "_external_handoffs" / "phone_lift_matching",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )
    return archive_db, Path(identity_report["paths"]["identity_db"])


def _build_bridge_mail_handoff(tmp_path: Path) -> tuple[Path, Path]:
    tallanto_csv = tmp_path / "bridge_students.csv"
    _write_tallanto_csv(
        tallanto_csv,
        [
            {
                "ID": "T-1",
                "amoCRM ID": "A-1",
                "E-mail": "unique@example.com",
                "Тел. (родителя)": "8 999 000 00 00",
            },
            {
                "ID": "T-2",
                "amoCRM ID": "A-2",
                "E-mail": "dup-phone@example.com",
                "Тел. (родителя)": "+7 999 111-22-33",
            },
            {
                "ID": "T-3",
                "amoCRM ID": "A-3",
                "E-mail": "other@example.com",
                "Тел. (родителя)": "+7 999 111-22-33",
            },
            {
                "ID": "T-4",
                "amoCRM ID": "A-4",
                "E-mail": "no-phone@example.com",
            },
            {
                "ID": "T-5",
                "amoCRM ID": "A-5",
                "E-mail": "no-mango@example.com",
                "Тел. (родителя)": "+7 999 222-33-44",
            },
        ],
    )
    identity_report = build_tallanto_identity_map(
        TallantoIdentityMapConfig(
            tallanto_csv_path=tallanto_csv,
            out_dir=tmp_path / "identity_bridge",
            encoding="utf-8",
            delimiter="\t",
        )
    )
    fake_imap = FakeImapClient(
        [
            _raw_message(message_id="m-unique", from_addr="Unique <unique@example.com>"),
            _raw_message(message_id="m-dup-phone", from_addr="Dup <dup-phone@example.com>"),
            _raw_message(message_id="m-no-phone", from_addr="No Phone <no-phone@example.com>"),
            _raw_message(message_id="m-no-mango", from_addr="No Mango <no-mango@example.com>"),
            _raw_message(message_id="m-missing", from_addr="Missing <missing@example.com>"),
            _raw_message(
                message_id="m-internal",
                from_addr="Colleague <colleague@kmipt.ru>",
                to_addr="School <school@kmipt.ru>",
            ),
        ]
    )
    ingest_report = build_mail_archive_ingest(
        credentials=MailImapCredentials(
            host="mail.example.test",
            port=993,
            email_address="school@kmipt.ru",
            password="not-written",
        ),
        config=MailArchiveIngestConfig(
            out_dir=tmp_path / "bridge_archive",
            mailbox="INBOX",
            mailbox_label="INBOX",
            since_days=7,
            max_messages=6,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )
    handoff = build_mail_customer_history_handoff(
        MailCustomerHistoryHandoffConfig(
            archive_db_paths=[Path(ingest_report["paths"]["archive_db"])],
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "bridge_handoff",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )
    return Path(handoff["paths"]["handoff_db"]), Path(identity_report["paths"]["identity_db"])


def _build_phone_index_bridge_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    tallanto_csv = tmp_path / "phone_index_students.csv"
    _write_tallanto_csv(
        tallanto_csv,
        [
            {
                "ID": "T-1",
                "amoCRM ID": "A-1",
                "E-mail": "product-phone@example.com",
                "Тел. (родителя)": "8 999 000 00 00",
            },
            {
                "ID": "T-2",
                "amoCRM ID": "A-2",
                "E-mail": "index-phone@example.com",
                "Тел. (родителя)": "+7 999 333-44-55",
            },
        ],
    )
    identity_report = build_tallanto_identity_map(
        TallantoIdentityMapConfig(
            tallanto_csv_path=tallanto_csv,
            out_dir=tmp_path / "identity_phone_index",
            encoding="utf-8",
            delimiter="\t",
        )
    )
    fake_imap = FakeImapClient(
        [
            _raw_message(message_id="m-product", from_addr="Product <product-phone@example.com>"),
            _raw_message(message_id="m-index", from_addr="Index <index-phone@example.com>"),
        ]
    )
    ingest_report = build_mail_archive_ingest(
        credentials=MailImapCredentials(
            host="mail.example.test",
            port=993,
            email_address="school@kmipt.ru",
            password="not-written",
        ),
        config=MailArchiveIngestConfig(
            out_dir=tmp_path / "phone_index_archive",
            mailbox="INBOX",
            mailbox_label="INBOX",
            since_days=7,
            max_messages=2,
            account_label="test",
            internal_domains=("kmipt.ru",),
        ),
        client=fake_imap,
    )
    handoff = build_mail_customer_history_handoff(
        MailCustomerHistoryHandoffConfig(
            archive_db_paths=[Path(ingest_report["paths"]["archive_db"])],
            identity_db_path=Path(identity_report["paths"]["identity_db"]),
            out_dir=tmp_path / "phone_index_handoff",
            mailbox_email="school@kmipt.ru",
            internal_domains=("kmipt.ru",),
        )
    )
    return (
        Path(handoff["paths"]["handoff_db"]),
        Path(identity_report["paths"]["identity_db"]),
        _build_phone_index_product_db(tmp_path),
    )


def _build_phone_index_product_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "phone_index_product.sqlite"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE capture_inbox_items (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL,
              provider TEXT NOT NULL,
              event_key TEXT NOT NULL,
              provider_call_id TEXT NOT NULL,
              status TEXT NOT NULL,
              source_job_run_id INTEGER,
              source_report_ref TEXT,
              raw_payload_ref TEXT,
              started_at TEXT,
              ended_at TEXT,
              direction TEXT,
              client_phone TEXT,
              manager_ref TEXT,
              recording_ref TEXT,
              recording_url TEXT,
              audio_ref TEXT,
              decision_reason TEXT,
              candidate_json TEXT,
              event_json TEXT,
              first_seen_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL,
              enqueue_count INTEGER NOT NULL DEFAULT 1,
              reserved_by TEXT,
              reserved_at TEXT,
              error TEXT,
              UNIQUE(tenant_id, provider, event_key)
            );
            CREATE TABLE product_calls (
              tenant_id TEXT NOT NULL,
              telephony_provider TEXT NOT NULL,
              provider_call_id TEXT NOT NULL,
              event_key TEXT NOT NULL,
              recording_id TEXT,
              source_filename TEXT NOT NULL,
              started_at TEXT,
              duration_sec REAL,
              manager_extension TEXT,
              manager_display_name TEXT,
              crm_owner_id INTEGER,
              crm_owner_name TEXT,
              crm_match_status TEXT,
              raw_payload_ref TEXT,
              source_repository_ref TEXT NOT NULL,
              imported_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, telephony_provider, provider_call_id),
              UNIQUE(event_key)
            );
            """
        )
        con.execute(
            """
            INSERT INTO capture_inbox_items (
              tenant_id, provider, event_key, provider_call_id, status,
              raw_payload_ref, started_at, ended_at, direction, client_phone,
              manager_ref, recording_ref, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "demo",
                "mango",
                "event-product",
                "call-product",
                "ready_for_capture",
                "raw-ref-product",
                "2026-05-07T10:00:00+00:00",
                "2026-05-07T10:05:00+00:00",
                "inbound",
                "8 999 000 00 00",
                "manager-1",
                "recording-1",
                "2026-05-07T10:01:00+00:00",
                "2026-05-07T10:01:00+00:00",
            ),
        )
        con.execute(
            """
            INSERT INTO product_calls (
              tenant_id, telephony_provider, provider_call_id, event_key,
              recording_id, source_filename, started_at, duration_sec,
              manager_extension, manager_display_name, crm_owner_id,
              crm_owner_name, crm_match_status, raw_payload_ref,
              source_repository_ref, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "demo",
                "mango",
                "call-product",
                "event-product",
                "recording-1",
                "call-product.mp3",
                "2026-05-07T10:00:00+00:00",
                300.0,
                "101",
                "Manager One",
                42,
                "Owner",
                "matched",
                "raw-ref-product",
                "repo-ref",
                "2026-05-07T12:00:00+00:00",
                "2026-05-07T12:00:00+00:00",
            ),
        )
        con.commit()
    return db_path


def _build_bridge_product_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "mango_product.sqlite"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE capture_inbox_items (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL,
              provider TEXT NOT NULL,
              event_key TEXT NOT NULL,
              provider_call_id TEXT NOT NULL,
              status TEXT NOT NULL,
              source_job_run_id INTEGER,
              source_report_ref TEXT,
              raw_payload_ref TEXT,
              started_at TEXT,
              ended_at TEXT,
              direction TEXT,
              client_phone TEXT,
              manager_ref TEXT,
              recording_ref TEXT,
              recording_url TEXT,
              audio_ref TEXT,
              decision_reason TEXT,
              candidate_json TEXT,
              event_json TEXT,
              first_seen_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL,
              enqueue_count INTEGER NOT NULL DEFAULT 1,
              reserved_by TEXT,
              reserved_at TEXT,
              error TEXT,
              UNIQUE(tenant_id, provider, event_key)
            );
            CREATE TABLE product_calls (
              tenant_id TEXT NOT NULL,
              telephony_provider TEXT NOT NULL,
              provider_call_id TEXT NOT NULL,
              event_key TEXT NOT NULL,
              recording_id TEXT,
              source_filename TEXT NOT NULL,
              started_at TEXT,
              duration_sec REAL,
              manager_extension TEXT,
              manager_display_name TEXT,
              crm_owner_id INTEGER,
              crm_owner_name TEXT,
              crm_match_status TEXT,
              raw_payload_ref TEXT,
              source_repository_ref TEXT NOT NULL,
              imported_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, telephony_provider, provider_call_id),
              UNIQUE(event_key)
            );
            """
        )
        con.execute(
            """
            INSERT INTO capture_inbox_items (
              tenant_id, provider, event_key, provider_call_id, status,
              raw_payload_ref, started_at, ended_at, direction, client_phone,
              manager_ref, recording_ref, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "demo",
                "mango",
                "event-1",
                "call-1",
                "ready_for_capture",
                "raw-ref-1",
                "2026-05-07T10:00:00+00:00",
                "2026-05-07T10:05:00+00:00",
                "inbound",
                "8 999 000 00 00",
                "manager-1",
                "recording-1",
                "2026-05-07T10:01:00+00:00",
                "2026-05-07T10:01:00+00:00",
            ),
        )
        con.execute(
            """
            INSERT INTO capture_inbox_items (
              tenant_id, provider, event_key, provider_call_id, status,
              raw_payload_ref, started_at, ended_at, direction, client_phone,
              manager_ref, recording_ref, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "demo",
                "mango",
                "event-dup",
                "call-dup",
                "ready_for_capture",
                "raw-ref-dup",
                "2026-05-07T11:00:00+00:00",
                "2026-05-07T11:05:00+00:00",
                "outbound",
                "+7 999 111-22-33",
                "manager-2",
                "recording-dup",
                "2026-05-07T11:01:00+00:00",
                "2026-05-07T11:01:00+00:00",
            ),
        )
        con.execute(
            """
            INSERT INTO product_calls (
              tenant_id, telephony_provider, provider_call_id, event_key,
              recording_id, source_filename, started_at, duration_sec,
              manager_extension, manager_display_name, crm_owner_id,
              crm_owner_name, crm_match_status, raw_payload_ref,
              source_repository_ref, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "demo",
                "mango",
                "call-1",
                "event-1",
                "recording-1",
                "call-1.mp3",
                "2026-05-07T10:00:00+00:00",
                300.0,
                "101",
                "Manager One",
                42,
                "Owner",
                "matched",
                "raw-ref-product",
                "repo-ref",
                "2026-05-07T12:00:00+00:00",
                "2026-05-07T12:00:00+00:00",
            ),
        )
        con.execute(
            """
            INSERT INTO product_calls (
              tenant_id, telephony_provider, provider_call_id, event_key,
              recording_id, source_filename, started_at, duration_sec,
              manager_extension, manager_display_name, crm_owner_id,
              crm_owner_name, crm_match_status, raw_payload_ref,
              source_repository_ref, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "demo",
                "mango",
                "call-filename",
                "event-filename",
                "recording-filename",
                "2026-05-07__12-00-00__79992223344__Manager_99.mp3",
                "2026-05-07T12:00:00+00:00",
                180.0,
                "102",
                "Manager Two",
                43,
                "Owner Two",
                "matched",
                "raw-ref-product-2",
                "repo-ref-2",
                "2026-05-07T12:30:00+00:00",
                "2026-05-07T12:30:00+00:00",
            ),
        )
        con.commit()
    return db_path


def _raw_message(
    *,
    message_id: str = "m-1",
    from_addr: str = "Client <client@example.com>",
    to_addr: str = "School <school@kmipt.ru>",
    subject: str = "Материалы по лагерю",
    body: str = "Здравствуйте, пришлите материалы по летнему лагерю.",
) -> bytes:
    message = EmailMessage()
    message["Message-ID"] = f"<{message_id}@example.com>"
    message["Date"] = "Tue, 12 May 2026 10:00:00 +0300"
    message["From"] = from_addr
    message["To"] = to_addr
    message["Subject"] = subject
    message.set_content(body)
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
