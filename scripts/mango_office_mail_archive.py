#!/usr/bin/env python3
"""Build read-only mail archive and Tallanto identity matching artifacts.

The IMAP ingest command downloads messages with BODY.PEEK[] into a local raw
archive. It never sends, deletes, moves, marks as read, or writes CRM/Tallanto.
Passwords are read only from an environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mail_archive import (  # noqa: E402
    MailArchiveIngestConfig,
    MailMatchingReportConfig,
    TallantoIdentityMapConfig,
    build_mail_archive_ingest,
    build_mail_matching_report,
    build_tallanto_identity_map,
)
from mango_mvp.productization.mail_imap_snapshot import MailImapCredentials  # noqa: E402


DEFAULT_HOST = "mail.hosting.reg.ru"
DEFAULT_PORT = 993
DEFAULT_PASSWORD_ENV = "MAIL_IMAP_PASSWORD"
DEFAULT_ACCOUNT_LABEL = "regru_edu"
DEFAULT_TALLANTO_CSV = (
    "_external_handoffs/tallanto_students_export_2026-05-12/Ученики.csv"
)
DEFAULT_MAIL_ARCHIVE_ROOT = "_external_handoffs/mail_archive_2026-05-12"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "identity-map":
        return run_identity_map(args)
    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "match-report":
        return run_match_report(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def run_identity_map(args: argparse.Namespace) -> int:
    try:
        report = build_tallanto_identity_map(
            TallantoIdentityMapConfig(
                tallanto_csv_path=Path(args.tallanto_csv),
                out_dir=Path(args.out_dir or default_identity_out_dir(args.account_label)),
                encoding=args.encoding,
                delimiter=args.delimiter,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL identity map failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_ingest(args: argparse.Namespace) -> int:
    password = os.environ.get(args.password_env, "")
    email_address = args.email or os.environ.get("MAIL_IMAP_EMAIL", "")
    host = args.host or os.environ.get("MAIL_IMAP_HOST", DEFAULT_HOST)
    port = args.port or int(os.environ.get("MAIL_IMAP_PORT", DEFAULT_PORT))
    if not email_address:
        print("MAIL archive ingest failed: missing --email or MAIL_IMAP_EMAIL.", file=sys.stderr)
        return 2
    if not password:
        print(
            f"MAIL archive ingest failed: missing password env {args.password_env}.",
            file=sys.stderr,
        )
        return 2

    try:
        report = build_mail_archive_ingest(
            credentials=MailImapCredentials(
                host=host,
                port=port,
                email_address=email_address,
                password=password,
            ),
            config=MailArchiveIngestConfig(
                out_dir=Path(args.out_dir or default_ingest_out_dir(args.account_label)),
                mailbox=args.mailbox,
                mailbox_label=args.mailbox_label or args.mailbox.strip('"'),
                since_days=args.since_days,
                max_messages=args.max_messages,
                account_label=args.account_label,
                internal_domains=tuple(args.internal_domain or []),
                extracted_text_max_chars=args.extracted_text_max_chars,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL archive ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0 if not report.get("errors") else 1


def run_match_report(args: argparse.Namespace) -> int:
    try:
        report = build_mail_matching_report(
            MailMatchingReportConfig(
                archive_db_path=Path(args.archive_db),
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir or default_match_out_dir(args.account_label)),
                mailbox_email=args.email or os.environ.get("MAIL_IMAP_EMAIL", ""),
                internal_domains=tuple(args.internal_domain or []),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL matching report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def print_summary(report: Mapping[str, object]) -> None:
    safe_keys = (
        "schema_version",
        "created_at",
        "row_count",
        "identity_values",
        "row_identity_classes",
        "message_count",
        "matched_message_count",
        "distinct_matched_candidates",
        "counts",
        "account_label",
        "mailbox",
        "since_days",
        "max_messages",
        "messages_found_since",
        "messages_attempted",
        "messages_inserted_or_seen",
        "raw_eml_written",
        "attachments_written",
        "text_files_written",
        "errors",
        "safety",
        "paths",
    )
    print(
        json.dumps(
            {key: report[key] for key in safe_keys if key in report},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def default_identity_out_dir(account_label: str) -> str:
    return f"{DEFAULT_MAIL_ARCHIVE_ROOT}/{safe_label(account_label)}/identity_map"


def default_ingest_out_dir(account_label: str) -> str:
    suffix = datetime.now().strftime("%Y%m%d")
    return f"{DEFAULT_MAIL_ARCHIVE_ROOT}/{safe_label(account_label)}/pilot_{suffix}"


def default_match_out_dir(account_label: str) -> str:
    suffix = datetime.now().strftime("%Y%m%d")
    return f"{DEFAULT_MAIL_ARCHIVE_ROOT}/{safe_label(account_label)}/matching_{suffix}"


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only mail archive artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity-map", help="Build Tallanto email/phone map.")
    identity.add_argument("--tallanto-csv", default=DEFAULT_TALLANTO_CSV)
    identity.add_argument("--out-dir")
    identity.add_argument("--encoding", default="cp1251")
    identity.add_argument("--delimiter", default="\t")
    identity.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)

    ingest = subparsers.add_parser("ingest", help="Ingest one read-only IMAP pilot batch.")
    ingest.add_argument("--host", help=f"Defaults to MAIL_IMAP_HOST or {DEFAULT_HOST}.")
    ingest.add_argument("--port", type=int, help=f"Defaults to MAIL_IMAP_PORT or {DEFAULT_PORT}.")
    ingest.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    ingest.add_argument(
        "--password-env",
        default=DEFAULT_PASSWORD_ENV,
        help=f"Environment variable that contains the mailbox password. Default: {DEFAULT_PASSWORD_ENV}.",
    )
    ingest.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    ingest.add_argument("--mailbox", default="INBOX")
    ingest.add_argument("--mailbox-label")
    ingest.add_argument("--out-dir")
    ingest.add_argument("--since-days", type=int, default=31)
    ingest.add_argument("--max-messages", type=int, default=25)
    ingest.add_argument("--internal-domain", action="append", default=["kmipt.ru"])
    ingest.add_argument("--extracted-text-max-chars", type=int, default=250_000)

    match = subparsers.add_parser("match-report", help="Build mail to Tallanto matching report.")
    match.add_argument("--archive-db", required=True)
    match.add_argument("--identity-db", required=True)
    match.add_argument("--out-dir")
    match.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    match.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    match.add_argument("--internal-domain", action="append", default=["kmipt.ru"])

    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
