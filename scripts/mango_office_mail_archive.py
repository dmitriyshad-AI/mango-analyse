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
    MailArchivePreflightConfig,
    MailArchiveVerificationConfig,
    MailCustomerHistoryHandoffConfig,
    MailMangoBridgePreviewConfig,
    MailPhoneLiftPreviewConfig,
    MangoPhoneIndexPreviewConfig,
    MailMatchingReportConfig,
    TallantoIdentityMapConfig,
    build_mail_archive_ingest,
    build_mail_archive_preflight,
    build_mail_customer_history_handoff,
    build_mail_mango_bridge_preview,
    build_mail_phone_lift_preview,
    build_mango_phone_index_preview,
    build_mail_matching_report,
    build_tallanto_identity_map,
    valid_env_var_name,
    verify_mail_archive_pilot,
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
DEFAULT_MANGO_PRODUCT_DB = (
    "_local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite"
)
DEFAULT_MANGO_RECORDING_ROOT = "_local_archive_mango_api_downloads_20260507"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "identity-map":
        return run_identity_map(args)
    if args.command == "preflight":
        return run_preflight(args)
    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "verify-pilot":
        return run_verify_pilot(args)
    if args.command == "match-report":
        return run_match_report(args)
    if args.command == "history-handoff":
        return run_history_handoff(args)
    if args.command == "mango-bridge-preview":
        return run_mango_bridge_preview(args)
    if args.command == "mango-phone-index-preview":
        return run_mango_phone_index_preview(args)
    if args.command == "phone-lift-preview":
        return run_phone_lift_preview(args)
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


def run_preflight(args: argparse.Namespace) -> int:
    load_dotenv_file(Path(args.dotenv)) if args.dotenv else None
    email_address = args.email or os.environ.get("MAIL_IMAP_EMAIL", "")
    host = args.host or os.environ.get("MAIL_IMAP_HOST", DEFAULT_HOST)
    port = resolve_port(args.port)
    password_env_valid = valid_env_var_name(args.password_env)
    report = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=Path(args.out_dir or default_ingest_out_dir(args.account_label)),
            mailbox=args.mailbox,
            since_days=args.since_days,
            max_messages=args.max_messages,
            account_label=args.account_label,
            host=host,
            port=port,
            email_address=email_address,
            password_env_name=args.password_env,
            password_env_present=password_env_valid and bool(os.environ.get(args.password_env)),
            identity_db_path=Path(args.identity_db) if args.identity_db else None,
            allow_large_batch=args.allow_large_batch,
        )
    )
    print_summary(report)
    return 0 if report.get("preflight_pass") else 2


def run_ingest(args: argparse.Namespace) -> int:
    load_dotenv_file(Path(args.dotenv)) if args.dotenv else None
    if not valid_env_var_name(args.password_env):
        print("MAIL archive ingest failed: invalid password env variable name.", file=sys.stderr)
        return 2
    email_address = args.email or os.environ.get("MAIL_IMAP_EMAIL", "")
    host = args.host or os.environ.get("MAIL_IMAP_HOST", DEFAULT_HOST)
    port = resolve_port(args.port)
    out_dir = Path(args.out_dir or default_ingest_out_dir(args.account_label))
    preflight = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=out_dir,
            mailbox=args.mailbox,
            since_days=args.since_days,
            max_messages=args.max_messages,
            account_label=args.account_label,
            host=host,
            port=port,
            email_address=email_address,
            password_env_name=args.password_env,
            password_env_present=bool(os.environ.get(args.password_env)),
            allow_large_batch=args.allow_large_batch,
        )
    )
    if not preflight.get("preflight_pass"):
        print(
            "MAIL archive ingest failed: preflight checks did not pass.",
            file=sys.stderr,
        )
        print_summary(preflight)
        return 2
    password = os.environ.get(args.password_env, "")

    try:
        report = build_mail_archive_ingest(
            credentials=MailImapCredentials(
                host=host,
                port=port,
                email_address=email_address,
                password=password,
            ),
            config=MailArchiveIngestConfig(
                out_dir=out_dir,
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


def run_verify_pilot(args: argparse.Namespace) -> int:
    try:
        report = verify_mail_archive_pilot(
            MailArchiveVerificationConfig(
                archive_dir=Path(args.archive_dir),
                expected_max_messages=args.expected_max_messages,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL archive verify failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0 if report.get("verification_pass") else 1


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


def run_history_handoff(args: argparse.Namespace) -> int:
    try:
        report = build_mail_customer_history_handoff(
            MailCustomerHistoryHandoffConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir),
                mailbox_email=args.email or os.environ.get("MAIL_IMAP_EMAIL", ""),
                internal_domains=tuple(args.internal_domain or []),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL customer history handoff failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_mango_bridge_preview(args: argparse.Namespace) -> int:
    try:
        report = build_mail_mango_bridge_preview(
            MailMangoBridgePreviewConfig(
                mail_handoff_db_path=Path(args.mail_handoff_db),
                identity_db_path=Path(args.identity_db),
                product_db_path=Path(args.product_db),
                out_dir=Path(args.out_dir),
                max_call_refs_per_candidate=args.max_call_refs_per_candidate,
                mango_phone_index_db_path=(
                    Path(args.mango_phone_index_db) if args.mango_phone_index_db else None
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL Mango bridge preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_mango_phone_index_preview(args: argparse.Namespace) -> int:
    try:
        report = build_mango_phone_index_preview(
            MangoPhoneIndexPreviewConfig(
                product_db_path=Path(args.product_db),
                recording_roots=[
                    Path(path)
                    for path in (args.recording_root or [DEFAULT_MANGO_RECORDING_ROOT])
                ],
                out_dir=Path(args.out_dir),
                include_product_db=not args.skip_product_db,
                include_recording_filenames=not args.skip_recording_filenames,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Mango phone index preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_phone_lift_preview(args: argparse.Namespace) -> int:
    try:
        report = build_mail_phone_lift_preview(
            MailPhoneLiftPreviewConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir),
                max_text_chars_per_message=args.max_text_chars_per_message,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL phone lift preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def print_summary(report: Mapping[str, object]) -> None:
    safe_keys = (
        "schema_version",
        "created_at",
        "preflight_pass",
        "verification_pass",
        "blocking_risks",
        "warnings",
        "batch_mode",
        "requested_pilot",
        "checks",
        "recommended_command",
        "archive_dir",
        "expected_max_messages",
        "ingest_summary",
        "db_counts",
        "file_counts",
        "provenance",
        "source_file",
        "row_count",
        "duplicate_tallanto_id_values",
        "source_id_quality",
        "identity_values",
        "row_identity_classes",
        "row_coverage",
        "identity_link_counts",
        "sanity_checks",
        "audit_readiness",
        "message_count",
        "source_archive_count",
        "candidate_count",
        "matched_message_count",
        "distinct_matched_candidates",
        "distinct_candidate_keys",
        "evaluated_message_count",
        "counts",
        "message_kind_counts",
        "match_class_by_message_kind",
        "mail_link_reconciliation",
        "mango_source_counts",
        "product_source_counts",
        "by_original_match_class",
        "lift_class_counts",
        "potential_lift",
        "text_access_counts",
        "source_file_counts",
        "phone_index_counts",
        "artifact_counts",
        "artifact_integrity",
        "pii_artifacts",
        "privacy",
        "columns_used",
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


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def resolve_port(arg_port: int | None) -> int:
    if arg_port:
        return int(arg_port)
    raw = os.environ.get("MAIL_IMAP_PORT", "")
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        return 0


def is_small_pilot(since_days: int, max_messages: int) -> bool:
    return 1 <= int(since_days) <= 7 and 1 <= int(max_messages) <= 5


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only mail archive artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity-map", help="Build Tallanto email/phone map.")
    identity.add_argument("--tallanto-csv", default=DEFAULT_TALLANTO_CSV)
    identity.add_argument("--out-dir")
    identity.add_argument("--encoding", default="cp1251")
    identity.add_argument("--delimiter", default="\t")
    identity.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)

    preflight = subparsers.add_parser("preflight", help="Check a live IMAP pilot without network calls.")
    preflight.add_argument("--host", help=f"Defaults to MAIL_IMAP_HOST or {DEFAULT_HOST}.")
    preflight.add_argument("--port", type=int, help=f"Defaults to MAIL_IMAP_PORT or {DEFAULT_PORT}.")
    preflight.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    preflight.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV)
    preflight.add_argument("--dotenv", default=".env", help="Optional dotenv file to load into env.")
    preflight.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    preflight.add_argument("--mailbox", default="INBOX")
    preflight.add_argument("--out-dir")
    preflight.add_argument("--since-days", type=int, default=3)
    preflight.add_argument("--max-messages", type=int, default=1)
    preflight.add_argument("--identity-db")
    preflight.add_argument(
        "--allow-large-batch",
        action="store_true",
        help="Approve a controlled non-pilot preflight window after pilot review.",
    )

    ingest = subparsers.add_parser("ingest", help="Ingest one read-only IMAP pilot batch.")
    ingest.add_argument("--host", help=f"Defaults to MAIL_IMAP_HOST or {DEFAULT_HOST}.")
    ingest.add_argument("--port", type=int, help=f"Defaults to MAIL_IMAP_PORT or {DEFAULT_PORT}.")
    ingest.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    ingest.add_argument(
        "--password-env",
        default=DEFAULT_PASSWORD_ENV,
        help=f"Environment variable that contains the mailbox password. Default: {DEFAULT_PASSWORD_ENV}.",
    )
    ingest.add_argument("--dotenv", default=".env", help="Optional dotenv file to load into env.")
    ingest.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    ingest.add_argument("--mailbox", default="INBOX")
    ingest.add_argument("--mailbox-label")
    ingest.add_argument("--out-dir")
    ingest.add_argument("--since-days", type=int, default=3)
    ingest.add_argument("--max-messages", type=int, default=1)
    ingest.add_argument(
        "--allow-large-batch",
        action="store_true",
        help="Allow non-pilot windows after a successful small pilot review.",
    )
    ingest.add_argument("--internal-domain", action="append", default=["kmipt.ru"])
    ingest.add_argument("--extracted-text-max-chars", type=int, default=250_000)

    verify = subparsers.add_parser("verify-pilot", help="Verify a completed mail archive pilot.")
    verify.add_argument("--archive-dir", required=True)
    verify.add_argument("--expected-max-messages", type=int, default=5)

    match = subparsers.add_parser("match-report", help="Build mail to Tallanto matching report.")
    match.add_argument("--archive-db", required=True)
    match.add_argument("--identity-db", required=True)
    match.add_argument("--out-dir")
    match.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    match.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    match.add_argument("--internal-domain", action="append", default=["kmipt.ru"])

    handoff = subparsers.add_parser(
        "history-handoff",
        help="Build read-only mail-to-customer history handoff from archive DBs.",
    )
    handoff.add_argument("--archive-db", action="append", required=True)
    handoff.add_argument("--identity-db", required=True)
    handoff.add_argument("--out-dir", required=True)
    handoff.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    handoff.add_argument("--internal-domain", action="append", default=["kmipt.ru"])

    bridge = subparsers.add_parser(
        "mango-bridge-preview",
        help="Build read-only mail-to-Mango phone bridge preview.",
    )
    bridge.add_argument("--mail-handoff-db", required=True)
    bridge.add_argument("--identity-db", required=True)
    bridge.add_argument("--product-db", default=DEFAULT_MANGO_PRODUCT_DB)
    bridge.add_argument("--out-dir", required=True)
    bridge.add_argument("--max-call-refs-per-candidate", type=int, default=50)
    bridge.add_argument("--mango-phone-index-db")

    phone_index = subparsers.add_parser(
        "mango-phone-index-preview",
        help="Build a read-only Mango phone index preview from product DB and local recording filenames.",
    )
    phone_index.add_argument("--product-db", default=DEFAULT_MANGO_PRODUCT_DB)
    phone_index.add_argument("--recording-root", action="append")
    phone_index.add_argument("--out-dir", required=True)
    phone_index.add_argument("--skip-product-db", action="store_true")
    phone_index.add_argument("--skip-recording-filenames", action="store_true")

    phone_lift = subparsers.add_parser(
        "phone-lift-preview",
        help="Preview phone-based lifts for ambiguous/missing mail matches.",
    )
    phone_lift.add_argument("--archive-db", action="append", required=True)
    phone_lift.add_argument("--identity-db", required=True)
    phone_lift.add_argument("--out-dir", required=True)
    phone_lift.add_argument("--max-text-chars-per-message", type=int, default=250_000)

    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
