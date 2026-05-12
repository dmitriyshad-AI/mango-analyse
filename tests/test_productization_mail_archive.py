from __future__ import annotations

import csv
import json
import sqlite3
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Sequence

import pytest

from mango_mvp.productization.mail_archive import (
    FULL_MESSAGE_FETCH_QUERY,
    MailArchiveIngestConfig,
    MailArchivePreflightConfig,
    MailArchiveVerificationConfig,
    MailCustomerHistoryHandoffConfig,
    MailMangoBridgePreviewConfig,
    MailMatchingReportConfig,
    MailPhoneLiftPreviewConfig,
    MangoPhoneIndexPreviewConfig,
    TallantoIdentityMapConfig,
    build_mail_archive_ingest,
    build_mail_archive_preflight,
    build_mail_customer_history_handoff,
    build_mail_mango_bridge_preview,
    build_mail_matching_report,
    build_mail_phone_lift_preview,
    build_mango_phone_index_preview,
    build_tallanto_identity_map,
    extract_email_addresses,
    extract_phone_numbers,
    normalize_email,
    normalize_phone,
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
    assert "pilot_since_days_must_be_1_to_7" in report["blocking_risks"]
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
            max_messages=100,
            identity_db_path=identity_db,
            allow_large_batch=True,
        )
    )

    assert report["preflight_pass"] is True
    assert report["batch_mode"] == "approved_large_batch"
    assert report["blocking_risks"] == []
    assert "--allow-large-batch" in report["recommended_command"]


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
