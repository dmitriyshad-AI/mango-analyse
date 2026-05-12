from __future__ import annotations

import csv
import email
import hashlib
import json
import platform
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata
from mango_mvp.productization.mail_imap_snapshot import (
    HEADER_FETCH_QUERY,
    ImapClient,
    ImapLibClient,
    MailImapCredentials,
    clean_header,
    first_fetch_payload,
    parse_header_payload,
    parse_search_ids,
)


MAIL_ARCHIVE_SCHEMA_VERSION = "mail_archive_v1"
TALLANTO_IDENTITY_MAP_SCHEMA_VERSION = "tallanto_email_identity_map_v1"
MAIL_MATCHING_REPORT_SCHEMA_VERSION = "mail_matching_report_v1"
MAIL_MANGO_BRIDGE_PREVIEW_SCHEMA_VERSION = "mail_mango_bridge_preview_v1"
MAIL_PHONE_LIFT_PREVIEW_SCHEMA_VERSION = "mail_phone_lift_preview_v1"
MANGO_PHONE_INDEX_PREVIEW_SCHEMA_VERSION = "mango_phone_index_preview_v1"
FULL_MESSAGE_FETCH_QUERY = "(BODY.PEEK[])"

DEFAULT_TALLANTO_EMAIL_COLUMNS = ("E-mail", "Другой E-mail")
DEFAULT_TALLANTO_PHONE_COLUMNS = (
    "Тел. (родителя)",
    "Тел. (доп.)",
    "Тел. (дом.)",
    "Другой тел.",
    "Помощник - тел.",
    "Тел. цифровой (моб.)",
    "Тел. цифровой (доп.)",
)
DEFAULT_TALLANTO_CANDIDATE_COLUMNS = (
    "ID",
    "amoCRM ID",
    "Имя",
    "Фамилия",
    "ФИО родителя",
    "Тип ученика",
    "Ответственный(ая)",
    "Ответственный(ая) (ID)",
    "Группа(ID)",
)
DEFAULT_INTERNAL_DOMAINS = ("kmipt.ru",)
SERVICE_LOCAL_PARTS = (
    "no-reply",
    "noreply",
    "mailer-daemon",
    "postmaster",
    "notification",
    "notifications",
    "robot",
)
PILOT_SINCE_DAYS_RANGE = (1, 7)
PILOT_MAX_MESSAGES_RANGE = (1, 5)
LARGE_BATCH_SINCE_DAYS_RANGE = (1, 31)
LARGE_BATCH_MAX_MESSAGES_RANGE = (1, 250)


@dataclass(frozen=True)
class TallantoIdentityMapConfig:
    tallanto_csv_path: Path
    out_dir: Path
    encoding: str = "cp1251"
    delimiter: str = "\t"
    email_columns: Sequence[str] = DEFAULT_TALLANTO_EMAIL_COLUMNS
    phone_columns: Sequence[str] = DEFAULT_TALLANTO_PHONE_COLUMNS
    candidate_columns: Sequence[str] = DEFAULT_TALLANTO_CANDIDATE_COLUMNS


@dataclass(frozen=True)
class MailArchiveIngestConfig:
    out_dir: Path
    mailbox: str = "INBOX"
    mailbox_label: str = "INBOX"
    since_days: int = 30
    max_messages: int = 25
    account_label: str = "regru_edu"
    internal_domains: Sequence[str] = DEFAULT_INTERNAL_DOMAINS
    extracted_text_max_chars: int = 250_000


@dataclass(frozen=True)
class MailArchivePreflightConfig:
    out_dir: Path
    mailbox: str = "INBOX"
    since_days: int = 3
    max_messages: int = 1
    account_label: str = "regru_edu"
    host: str = "mail.hosting.reg.ru"
    port: int = 993
    email_address: str = ""
    password_env_name: str = "MAIL_IMAP_PASSWORD"
    password_env_present: bool = False
    identity_db_path: Optional[Path] = None
    allow_large_batch: bool = False


@dataclass(frozen=True)
class MailMatchingReportConfig:
    archive_db_path: Path
    identity_db_path: Path
    out_dir: Path
    mailbox_email: str = ""
    internal_domains: Sequence[str] = DEFAULT_INTERNAL_DOMAINS


@dataclass(frozen=True)
class MailArchiveVerificationConfig:
    archive_dir: Path
    expected_max_messages: int = 5


@dataclass(frozen=True)
class MailCustomerHistoryHandoffConfig:
    archive_db_paths: Sequence[Path]
    identity_db_path: Path
    out_dir: Path
    mailbox_email: str = ""
    internal_domains: Sequence[str] = DEFAULT_INTERNAL_DOMAINS


@dataclass(frozen=True)
class MailMangoBridgePreviewConfig:
    mail_handoff_db_path: Path
    identity_db_path: Path
    product_db_path: Path
    out_dir: Path
    max_call_refs_per_candidate: int = 50
    mango_phone_index_db_path: Optional[Path] = None


@dataclass(frozen=True)
class MailPhoneLiftPreviewConfig:
    archive_db_paths: Sequence[Path]
    identity_db_path: Path
    out_dir: Path
    max_text_chars_per_message: int = 250_000


@dataclass(frozen=True)
class MangoPhoneIndexPreviewConfig:
    product_db_path: Path
    recording_roots: Sequence[Path]
    out_dir: Path
    include_product_db: bool = True
    include_recording_filenames: bool = True


def normalize_email(value: object) -> str:
    text = clean_text(value).strip().strip(";,")
    if not text:
        return ""
    text = re.sub(r"^mailto:", "", text, flags=re.IGNORECASE).strip()
    if "<" in text and ">" in text:
        parsed = getaddresses([text])
        if parsed:
            text = parsed[0][1]
    text = text.strip().strip("<>;,").casefold()
    if not re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z0-9-]+", text):
        return ""
    local, domain = text.rsplit("@", 1)
    domain = domain.strip(".")
    if not local or not domain:
        return ""
    return f"{local}@{domain}"


def extract_email_addresses(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    found: list[str] = []
    for _name, address in getaddresses([text.replace(";", ",").replace("\n", ",")]):
        normalized = normalize_email(address)
        if normalized:
            found.append(normalized)
    for match in re.findall(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9-]+",
        text,
    ):
        normalized = normalize_email(match)
        if normalized:
            found.append(normalized)
    return unique_preserving_order(found)


def normalize_phone(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    digits = re.sub(r"\D+", "", text)
    if len(digits) == 11 and digits[0] in {"7", "8"}:
        return "+7" + digits[-10:]
    if len(digits) == 10:
        return "+7" + digits
    if 11 <= len(digits) <= 15:
        return "+" + digits
    return ""


def extract_phone_numbers(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    candidates: list[str] = []
    for part in re.split(r"[,;\n\r/|]+", text):
        normalized = normalize_phone(part)
        if normalized:
            candidates.append(normalized)
    for match in re.findall(r"\+?\d[\d\s().-]{8,}\d", text):
        normalized = normalize_phone(match)
        if normalized:
            candidates.append(normalized)
    return unique_preserving_order(candidates)


def build_tallanto_identity_map(config: TallantoIdentityMapConfig) -> Mapping[str, Any]:
    guard_not_stable_runtime(config.out_dir, "identity map output directory")
    out_dir = config.out_dir.resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "tallanto_email_identity_map.sqlite"
    report_path = out_dir / "tallanto_email_identity_map_report.json"

    rows = read_tallanto_rows(
        config.tallanto_csv_path,
        encoding=config.encoding,
        delimiter=config.delimiter,
    )
    tallanto_id_counts = Counter(clean_text(row.get("ID")) for row in rows if clean_text(row.get("ID")))
    candidate_rows: list[dict[str, Any]] = []
    identity_links: dict[tuple[str, str], dict[str, Any]] = {}
    row_identity_classes: dict[str, dict[str, str]] = {}

    for row_number, row in enumerate(rows, start=2):
        candidate_key = candidate_key_for_row(row_number, row, tallanto_id_counts=tallanto_id_counts)
        candidate_payload = {
            column: clean_text(row.get(column))
            for column in config.candidate_columns
            if column in row
        }
        candidate_rows.append(
            {
                "candidate_key": candidate_key,
                "row_number": row_number,
                "tallanto_id": candidate_payload.get("ID", ""),
                "amocrm_id": candidate_payload.get("amoCRM ID", ""),
                "first_name": candidate_payload.get("Имя", ""),
                "last_name": candidate_payload.get("Фамилия", ""),
                "parent_name": candidate_payload.get("ФИО родителя", ""),
                "student_type": candidate_payload.get("Тип ученика", ""),
                "manager": candidate_payload.get("Ответственный(ая)", ""),
                "manager_id": candidate_payload.get("Ответственный(ая) (ID)", ""),
                "group_id": candidate_payload.get("Группа(ID)", ""),
                "candidate_json": json.dumps(
                    candidate_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

        email_values = collect_row_identity_values(
            row,
            columns=config.email_columns,
            extractor=extract_email_addresses,
        )
        phone_values = collect_row_identity_values(
            row,
            columns=config.phone_columns,
            extractor=extract_phone_numbers,
        )
        for value, columns in email_values.items():
            append_identity_link(identity_links, "email", value, candidate_key, columns)
        for value, columns in phone_values.items():
            append_identity_link(identity_links, "phone", value, candidate_key, columns)
        row_identity_classes[candidate_key] = {
            "email": "missing" if not email_values else "pending",
            "phone": "missing" if not phone_values else "pending",
        }

    for item in identity_links.values():
        candidate_count = len(item["candidate_keys"])
        item["match_class"] = "strong_unique" if candidate_count == 1 else "duplicate"
        for candidate_key in item["candidate_keys"]:
            kind = item["kind"]
            current = row_identity_classes[candidate_key][kind]
            if current == "missing":
                row_identity_classes[candidate_key][kind] = item["match_class"]
            elif item["match_class"] == "duplicate":
                row_identity_classes[candidate_key][kind] = "duplicate"
            elif current == "pending":
                row_identity_classes[candidate_key][kind] = "strong_unique"

    for classes in row_identity_classes.values():
        for kind in ("email", "phone"):
            if classes[kind] == "pending":
                classes[kind] = "strong_unique"

    write_identity_map_db(
        db_path,
        source_path=config.tallanto_csv_path,
        candidate_rows=candidate_rows,
        identity_links=identity_links,
        row_identity_classes=row_identity_classes,
    )
    remove_sqlite_sidecars(db_path)

    report = build_identity_map_report(
        db_path=db_path,
        report_path=report_path,
        source_path=config.tallanto_csv_path,
        candidate_rows=candidate_rows,
        identity_links=identity_links,
        row_identity_classes=row_identity_classes,
        config=config,
        duplicate_tallanto_id_values=sum(1 for count in tallanto_id_counts.values() if count > 1),
    )
    write_json(report_path, report)
    return report


def build_mail_archive_preflight(config: MailArchivePreflightConfig) -> Mapping[str, Any]:
    out_dir = config.out_dir.resolve(strict=False)
    blocking_risks: list[str] = []
    warnings: list[str] = []

    try:
        guard_not_stable_runtime(out_dir, "mail archive preflight output directory")
        out_dir_not_stable_runtime = True
    except ValueError:
        out_dir_not_stable_runtime = False
        blocking_risks.append("out_dir_under_stable_runtime")

    out_dir_under_external_handoffs = "_external_handoffs" in out_dir.parts
    if not out_dir_under_external_handoffs:
        warnings.append("out_dir_not_under_external_handoffs")

    out_dir_git_ignored = git_check_ignored(out_dir)
    if not out_dir_git_ignored:
        blocking_risks.append("out_dir_not_git_ignored")

    email_normalized = normalize_email(config.email_address)
    if not email_normalized:
        blocking_risks.append("missing_or_invalid_mailbox_email")
    password_env_name = (
        config.password_env_name
        if valid_env_var_name(config.password_env_name)
        else "<invalid_env_var_name>"
    )
    if not valid_env_var_name(config.password_env_name):
        blocking_risks.append("invalid_password_env_name")
    elif not config.password_env_present:
        blocking_risks.append("missing_password_env")
    if not clean_text(config.mailbox):
        blocking_risks.append("missing_mailbox")
    if not clean_text(config.host):
        blocking_risks.append("missing_imap_host")
    if int(config.port or 0) <= 0:
        blocking_risks.append("invalid_imap_port")
    since_min, since_max = (
        LARGE_BATCH_SINCE_DAYS_RANGE if config.allow_large_batch else PILOT_SINCE_DAYS_RANGE
    )
    messages_min, messages_max = (
        LARGE_BATCH_MAX_MESSAGES_RANGE if config.allow_large_batch else PILOT_MAX_MESSAGES_RANGE
    )
    if not (since_min <= int(config.since_days) <= since_max):
        blocking_risks.append(
            f"{'batch' if config.allow_large_batch else 'pilot'}_since_days_must_be_"
            f"{since_min}_to_{since_max}"
        )
    if not (messages_min <= int(config.max_messages) <= messages_max):
        blocking_risks.append(
            f"{'batch' if config.allow_large_batch else 'pilot'}_max_messages_must_be_"
            f"{messages_min}_to_{messages_max}"
        )

    identity_db_exists = None
    identity_db_not_stable_runtime = None
    if config.identity_db_path is not None:
        identity_db = config.identity_db_path.resolve(strict=False)
        identity_db_exists = identity_db.exists()
        if not identity_db_exists:
            blocking_risks.append("identity_db_missing")
        try:
            guard_not_stable_runtime(identity_db, "identity database")
            identity_db_not_stable_runtime = True
        except ValueError:
            identity_db_not_stable_runtime = False
            blocking_risks.append("identity_db_under_stable_runtime")

    if out_dir.exists():
        warnings.append("out_dir_already_exists")

    report_path = out_dir / "mail_archive_preflight.json"
    report = {
        "schema_version": "mail_archive_preflight_v1",
        "created_at": utc_now(),
        "account_label": config.account_label,
        "preflight_pass": not blocking_risks,
        "blocking_risks": blocking_risks,
        "warnings": warnings,
        "batch_mode": "approved_large_batch" if config.allow_large_batch else "pilot",
        "requested_pilot": {
            "host": config.host,
            "port": config.port,
            "email_present": bool(email_normalized),
            "mailbox": config.mailbox,
            "since_days": int(config.since_days),
            "max_messages": int(config.max_messages),
        },
        "checks": {
            "password_env_name": password_env_name,
            "password_env_present": bool(config.password_env_present),
            "out_dir_not_stable_runtime": out_dir_not_stable_runtime,
            "out_dir_under_external_handoffs": out_dir_under_external_handoffs,
            "out_dir_git_ignored": out_dir_git_ignored,
            "identity_db_exists": identity_db_exists,
            "identity_db_not_stable_runtime": identity_db_not_stable_runtime,
        },
        "safety": {
            "network_calls": False,
            "imap_login": False,
            "send_mail": False,
            "delete_or_move_mail": False,
            "write_crm": False,
            "write_tallanto": False,
            "password_written": False,
            "run_asr": False,
            "run_ra": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
        },
        "recommended_command": (
            "python3 scripts/mango_office_mail_archive.py ingest "
            f"--email <redacted> --mailbox {shell_safe_token(config.mailbox)} "
            f"--since-days {int(config.since_days)} --max-messages {int(config.max_messages)} "
            f"--out-dir {shell_safe_token(str(config.out_dir))}"
            f"{' --allow-large-batch' if config.allow_large_batch else ''}"
        ),
        "paths": {
            "out_dir": str(out_dir),
            "preflight_report": str(report_path),
            "identity_db": str(config.identity_db_path) if config.identity_db_path else "",
        },
        "privacy": {
            "contains_password": False,
            "contains_raw_mail": False,
            "contains_raw_personal_values": False,
        },
    }
    if out_dir_not_stable_runtime:
        write_json(report_path, report)
    return report


def build_mail_archive_ingest(
    *,
    credentials: MailImapCredentials,
    config: MailArchiveIngestConfig,
    client: Optional[ImapClient] = None,
) -> Mapping[str, Any]:
    guard_not_stable_runtime(config.out_dir, "mail archive output directory")
    out_dir = config.out_dir.resolve(strict=False)
    if client is None:
        guard_git_ignored_output(out_dir, "mail archive output directory")
    raw_dir = out_dir / "raw_eml"
    attachment_dir = out_dir / "attachments"
    text_dir = out_dir / "extracted_text"
    for path in (raw_dir, attachment_dir, text_dir):
        path.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "mail_archive.sqlite"
    report_path = out_dir / "mail_ingest_report.json"

    since_days = max(1, int(config.since_days))
    since_dt = (datetime.now(timezone.utc) - timedelta(days=since_days)).date()
    since_imap = since_dt.strftime("%d-%b-%Y")
    max_messages = max(0, int(config.max_messages))

    report: dict[str, Any] = {
        "schema_version": MAIL_ARCHIVE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "account_label": config.account_label,
        "host": credentials.host,
        "port": credentials.port,
        "email": credentials.email_address,
        "mailbox": config.mailbox_label,
        "mailbox_raw": config.mailbox,
        "since_days": since_days,
        "since_imap": since_imap,
        "max_messages": max_messages,
        "safety": {
            "readonly_select": True,
            "fetch_uses_body_peek": True,
            "send_mail": False,
            "delete_or_move_mail": False,
            "write_crm": False,
            "write_tallanto": False,
            "open_attachments": False,
            "password_written": False,
            "run_asr": False,
            "run_ra": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
        },
        "messages_found_since": 0,
        "messages_attempted": 0,
        "messages_inserted_or_seen": 0,
        "raw_eml_written": 0,
        "attachments_written": 0,
        "text_files_written": 0,
        "errors": [],
        "paths": {
            "archive_db": str(db_path),
            "raw_eml_dir": str(raw_dir),
            "attachments_dir": str(attachment_dir),
            "extracted_text_dir": str(text_dir),
            "report": str(report_path),
        },
    }

    init_mail_archive_db(db_path)
    imap = client or ImapLibClient(host=credentials.host, port=credentials.port)
    try:
        login_status, _ = imap.login(credentials.email_address, credentials.password)
        report["login_status"] = login_status
        select_status, select_data = imap.select(config.mailbox, readonly=True)
        report["select_status"] = select_status
        report["mailbox_total_messages"] = int(select_data[0]) if select_data and select_data[0] else 0
        if select_status != "OK":
            raise RuntimeError(f"IMAP SELECT failed for {config.mailbox}: {select_status}")
        search_status, search_data = imap.search(None, "SINCE", since_imap)
        report["search_status"] = search_status
        message_ids = parse_search_ids(search_status, search_data)
        report["messages_found_since"] = len(message_ids)
        selected_ids = message_ids[-max_messages:] if max_messages > 0 else []
        report["messages_attempted"] = len(selected_ids)

        for msg_id in selected_ids:
            try:
                fetched = ingest_one_message(
                    imap,
                    msg_id=msg_id,
                    credentials=credentials,
                    config=config,
                    db_path=db_path,
                    raw_dir=raw_dir,
                    attachment_dir=attachment_dir,
                    text_dir=text_dir,
                )
                report["messages_inserted_or_seen"] += 1
                report["raw_eml_written"] += int(bool(fetched["raw_eml_written"]))
                report["attachments_written"] += int(fetched["attachments_written"])
                report["text_files_written"] += int(bool(fetched["text_file_written"]))
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(
                    {
                        "imap_seq": msg_id.decode("ascii", "ignore"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        try:
            imap.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass

    write_json(report_path, report)
    return report


def build_mail_matching_report(config: MailMatchingReportConfig) -> Mapping[str, Any]:
    guard_not_stable_runtime(config.out_dir, "matching report output directory")
    guard_not_stable_runtime(config.archive_db_path, "mail archive database")
    out_dir = config.out_dir.resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mail_matching_report.json"

    identity_index = load_identity_index(config.identity_db_path)
    internal_domains = normalize_domains(config.internal_domains, config.mailbox_email)

    with sqlite3.connect(str(config.archive_db_path)) as con:
        con.row_factory = sqlite3.Row
        init_mail_match_tables(con)
        con.execute("DELETE FROM message_matches")
        messages = list(con.execute("SELECT * FROM messages ORDER BY message_date_iso, sha256"))
        counts: dict[str, int] = {
            "strong_unique": 0,
            "ambiguous": 0,
            "missing": 0,
            "internal_or_service": 0,
        }
        kind_counts: Counter[str] = Counter()
        class_by_kind: dict[str, Counter[str]] = {}
        candidate_messages: set[tuple[str, str]] = set()
        for message in messages:
            participants = list(
                con.execute(
                    "SELECT * FROM message_participants WHERE message_sha256 = ?",
                    (message["sha256"],),
                )
            )
            external_emails = [
                row["email_normalized"]
                for row in participants
                if row["email_normalized"]
                and not email_domain_is_internal(row["email_normalized"], internal_domains)
            ]
            external_emails = unique_preserving_order(external_emails)
            message_kind = clean_text(message["message_kind"])
            kind_counts[message_kind] += 1
            if message_kind in {"internal", "service"}:
                match_class = "internal_or_service"
                candidate_keys: list[str] = []
            else:
                candidate_keys = unique_preserving_order(
                    candidate_key
                    for address in external_emails
                    for candidate_key in identity_index.get(address, [])
                )
                if len(candidate_keys) == 1:
                    match_class = "strong_unique"
                elif len(candidate_keys) > 1:
                    match_class = "ambiguous"
                else:
                    match_class = "missing"
            counts[match_class] += 1
            class_by_kind.setdefault(message_kind, Counter())[match_class] += 1
            con.execute(
                """
                INSERT OR REPLACE INTO message_matches (
                  message_sha256, match_class, candidate_count, matched_email_count, matched_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message["sha256"],
                    match_class,
                    len(candidate_keys),
                    sum(1 for address in external_emails if address in identity_index),
                    utc_now(),
                ),
            )
            for candidate_key in candidate_keys:
                candidate_messages.add((candidate_key, message["sha256"]))
        con.commit()

    report = {
        "schema_version": MAIL_MATCHING_REPORT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "archive_db_path": str(config.archive_db_path),
        "identity_db_path": str(config.identity_db_path),
        "message_count": len(messages),
        "matched_message_count": counts["strong_unique"] + counts["ambiguous"],
        "distinct_matched_candidates": len({candidate for candidate, _sha in candidate_messages}),
        "counts": counts,
        "message_kind_counts": {kind: int(count) for kind, count in sorted(kind_counts.items())},
        "match_class_by_message_kind": {
            kind: {match_class: int(count) for match_class, count in sorted(class_counts.items())}
            for kind, class_counts in sorted(class_by_kind.items())
        },
        "privacy": {
            "contains_raw_personal_values": False,
            "raw_emails_written": False,
            "raw_names_written": False,
            "raw_phones_written": False,
        },
        "paths": {"report": str(report_path)},
    }
    write_json(report_path, report)
    return report


def verify_mail_archive_pilot(config: MailArchiveVerificationConfig) -> Mapping[str, Any]:
    archive_dir = config.archive_dir.resolve(strict=False)
    report_path = archive_dir / "mail_ingest_report.json"
    db_path = archive_dir / "mail_archive.sqlite"
    verification_path = archive_dir / "mail_archive_verification.json"
    raw_dir = archive_dir / "raw_eml"
    text_dir = archive_dir / "extracted_text"
    attachment_dir = archive_dir / "attachments"

    blocking_risks: list[str] = []
    warnings: list[str] = []
    try:
        guard_not_stable_runtime(archive_dir, "mail archive verification directory")
        archive_dir_not_stable_runtime = True
    except ValueError:
        archive_dir_not_stable_runtime = False
        blocking_risks.append("archive_dir_under_stable_runtime")

    archive_dir_git_ignored = git_check_ignored(archive_dir)
    if not archive_dir_git_ignored:
        blocking_risks.append("archive_dir_not_git_ignored")

    ingest_report: Mapping[str, Any] = {}
    if not report_path.exists():
        blocking_risks.append("mail_ingest_report_missing")
    else:
        try:
            ingest_report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blocking_risks.append("mail_ingest_report_invalid_json")

    if not db_path.exists():
        blocking_risks.append("mail_archive_db_missing")

    safety = ingest_report.get("safety") if isinstance(ingest_report, Mapping) else {}
    if isinstance(safety, Mapping):
        if safety.get("readonly_select") is not True:
            blocking_risks.append("readonly_select_not_confirmed")
        if safety.get("fetch_uses_body_peek") is not True:
            blocking_risks.append("body_peek_not_confirmed")
        for key in (
            "send_mail",
            "delete_or_move_mail",
            "write_crm",
            "write_tallanto",
            "password_written",
            "run_asr",
            "run_ra",
            "runtime_db_writes",
            "stable_runtime_writes",
        ):
            if safety.get(key) is not False:
                blocking_risks.append(f"{key}_not_false")
    elif ingest_report:
        blocking_risks.append("safety_block_missing")

    attempted = int(ingest_report.get("messages_attempted") or 0) if ingest_report else 0
    expected_max = int(config.expected_max_messages)
    if expected_max > 0 and attempted > expected_max:
        blocking_risks.append("messages_attempted_exceeds_expected_max")
    if ingest_report and ingest_report.get("errors"):
        warnings.append("ingest_report_contains_errors")

    db_counts = empty_archive_db_counts()
    db_schema_ok = False
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as con:
                db_counts = archive_db_counts(con)
                db_schema_ok = archive_db_has_required_tables(con)
        except sqlite3.Error:
            blocking_risks.append("mail_archive_db_unreadable")
    if db_path.exists() and not db_schema_ok:
        blocking_risks.append("mail_archive_db_missing_required_tables")

    raw_file_count = count_files(raw_dir, "*.eml")
    text_file_count = count_files(text_dir, "*.txt")
    attachment_file_count = count_files(attachment_dir, "*.bin")
    if db_counts["messages"] != raw_file_count:
        blocking_risks.append("raw_eml_count_mismatch")
    if db_counts["messages"] and text_file_count > db_counts["messages"]:
        warnings.append("text_file_count_exceeds_message_count")
    if db_counts["attachments"] != attachment_file_count:
        warnings.append("attachment_file_count_differs_from_db_count")
    if db_counts["messages"] and db_counts["message_sources"] < db_counts["messages"]:
        blocking_risks.append("message_sources_less_than_messages")

    report_text = report_path.read_text(encoding="utf-8", errors="ignore") if report_path.exists() else ""
    if re.search(r"(?i)password\\s*[:=]", report_text):
        warnings.append("report_contains_password_key_text")

    verification = {
        "schema_version": "mail_archive_verification_v1",
        "created_at": utc_now(),
        "verification_pass": not blocking_risks,
        "blocking_risks": blocking_risks,
        "warnings": warnings,
        "archive_dir": str(archive_dir),
        "expected_max_messages": expected_max,
        "checks": {
            "archive_dir_not_stable_runtime": archive_dir_not_stable_runtime,
            "archive_dir_git_ignored": archive_dir_git_ignored,
            "mail_ingest_report_exists": report_path.exists(),
            "mail_archive_db_exists": db_path.exists(),
            "mail_archive_db_schema_ok": db_schema_ok,
            "safety_block_present": isinstance(safety, Mapping),
        },
        "ingest_summary": {
            "messages_found_since": ingest_report.get("messages_found_since", 0),
            "messages_attempted": attempted,
            "messages_inserted_or_seen": ingest_report.get("messages_inserted_or_seen", 0),
            "raw_eml_written": ingest_report.get("raw_eml_written", 0),
            "attachments_written": ingest_report.get("attachments_written", 0),
            "text_files_written": ingest_report.get("text_files_written", 0),
            "error_count": len(ingest_report.get("errors") or []) if ingest_report else 0,
        },
        "db_counts": db_counts,
        "file_counts": {
            "raw_eml": raw_file_count,
            "extracted_text": text_file_count,
            "attachments": attachment_file_count,
        },
        "paths": {
            "archive_db": str(db_path),
            "ingest_report": str(report_path),
            "verification_report": str(verification_path),
            "raw_eml_dir": str(raw_dir),
            "extracted_text_dir": str(text_dir),
            "attachments_dir": str(attachment_dir),
        },
        "privacy": {
            "verification_opens_raw_eml": False,
            "verification_opens_attachments": False,
            "contains_password": False,
            "contains_raw_mail": False,
            "contains_raw_personal_values": False,
        },
    }
    if archive_dir_not_stable_runtime:
        write_json(verification_path, verification)
    return verification


def build_mail_customer_history_handoff(
    config: MailCustomerHistoryHandoffConfig,
) -> Mapping[str, Any]:
    guard_not_stable_runtime(config.out_dir, "customer history handoff output directory")
    guard_not_stable_runtime(config.identity_db_path, "identity database")
    out_dir = config.out_dir.resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_db_path = out_dir / "mail_customer_history_handoff.sqlite"
    report_path = out_dir / "mail_customer_history_handoff_report.json"

    identity_index = load_identity_index(config.identity_db_path)
    internal_domains = normalize_domains(config.internal_domains, config.mailbox_email)
    source_paths = [path.resolve(strict=False) for path in config.archive_db_paths]
    for source_path in source_paths:
        guard_not_stable_runtime(source_path, "mail archive database")
        if not source_path.exists():
            raise FileNotFoundError(f"mail archive database not found: {source_path}")

    with sqlite3.connect(str(out_db_path)) as out:
        out.row_factory = sqlite3.Row
        out.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_archives (
              source_archive_id INTEGER PRIMARY KEY AUTOINCREMENT,
              archive_db_path TEXT NOT NULL UNIQUE,
              message_count INTEGER NOT NULL,
              imported_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mail_customer_links (
              source_archive_id INTEGER NOT NULL,
              message_sha256 TEXT NOT NULL,
              mailbox TEXT,
              message_date_iso TEXT,
              message_kind TEXT NOT NULL,
              match_class TEXT NOT NULL,
              candidate_count INTEGER NOT NULL,
              matched_email_count INTEGER NOT NULL,
              candidate_keys_json TEXT NOT NULL,
              link_status TEXT NOT NULL,
              blocked_reason TEXT,
              PRIMARY KEY (source_archive_id, message_sha256)
            );
            DELETE FROM meta;
            DELETE FROM source_archives;
            DELETE FROM mail_customer_links;
            """
        )
        out.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", "mail_customer_history_handoff_v1"),
                ("created_at", utc_now()),
                ("identity_db_path", str(config.identity_db_path)),
            ],
        )

        for source_path in source_paths:
            with open_sqlite_readonly(source_path) as src:
                src.row_factory = sqlite3.Row
                messages = list(src.execute("SELECT * FROM messages ORDER BY message_date_iso, sha256"))
                out.execute(
                    """
                    INSERT INTO source_archives (archive_db_path, message_count, imported_at)
                    VALUES (?, ?, ?)
                    """,
                    (str(source_path), len(messages), utc_now()),
                )
                source_archive_id = int(out.execute("SELECT last_insert_rowid()").fetchone()[0])
                for message in messages:
                    participants = list(
                        src.execute(
                            "SELECT * FROM message_participants WHERE message_sha256 = ?",
                            (message["sha256"],),
                        )
                    )
                    match = classify_message_against_identity(
                        message_kind=clean_text(message["message_kind"]),
                        participants=participants,
                        identity_index=identity_index,
                        internal_domains=internal_domains,
                    )
                    out.execute(
                        """
                        INSERT INTO mail_customer_links (
                          source_archive_id, message_sha256, mailbox, message_date_iso,
                          message_kind, match_class, candidate_count, matched_email_count,
                          candidate_keys_json, link_status, blocked_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_archive_id,
                            message["sha256"],
                            clean_text(message["mailbox"]),
                            clean_text(message["message_date_iso"]),
                            clean_text(message["message_kind"]),
                            match["match_class"],
                            len(match["candidate_keys"]),
                            int(match["matched_email_count"]),
                            json.dumps(match["candidate_keys"], ensure_ascii=False),
                            link_status_for_match_class(match["match_class"]),
                            blocked_reason_for_match_class(match["match_class"]),
                        ),
                    )
        out.executescript(
            """
            CREATE VIEW IF NOT EXISTS v_strong_customer_mail_links AS
            SELECT * FROM mail_customer_links WHERE match_class = 'strong_unique';
            CREATE VIEW IF NOT EXISTS v_manual_review_mail_links AS
            SELECT * FROM mail_customer_links WHERE match_class IN ('ambiguous', 'missing');
            """
        )
        counts = dict(
            out.execute(
                "SELECT match_class, COUNT(*) FROM mail_customer_links GROUP BY match_class"
            ).fetchall()
        )
        kind_counts = dict(
            out.execute(
                "SELECT message_kind, COUNT(*) FROM mail_customer_links GROUP BY message_kind"
            ).fetchall()
        )
        total_links = int(out.execute("SELECT COUNT(*) FROM mail_customer_links").fetchone()[0])
        distinct_candidates = {
            candidate
            for row in out.execute("SELECT candidate_keys_json FROM mail_customer_links")
            for candidate in json.loads(row[0] or "[]")
        }
        out.commit()

    remove_sqlite_sidecars(out_db_path)
    report = {
        "schema_version": "mail_customer_history_handoff_v1",
        "created_at": utc_now(),
        "source_archive_count": len(source_paths),
        "message_count": total_links,
        "counts": {
            "strong_unique": int(counts.get("strong_unique", 0)),
            "ambiguous": int(counts.get("ambiguous", 0)),
            "missing": int(counts.get("missing", 0)),
            "internal_or_service": int(counts.get("internal_or_service", 0)),
        },
        "message_kind_counts": {str(k): int(v) for k, v in kind_counts.items()},
        "distinct_candidate_keys": len(distinct_candidates),
        "safety": {
            "read_only_source_archives": True,
            "write_crm": False,
            "write_tallanto": False,
            "live_crm_reads": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "store_raw_files_in_sqlite": False,
        },
        "privacy": {
            "contains_raw_personal_values": False,
            "raw_emails_written": False,
            "raw_names_written": False,
            "raw_phones_written": False,
            "raw_mail_written": False,
            "handoff_sqlite_contains_identity_refs": True,
        },
        "paths": {
            "handoff_db": str(out_db_path),
            "report": str(report_path),
        },
    }
    write_json(report_path, report)
    return report


def build_mango_phone_index_preview(config: MangoPhoneIndexPreviewConfig) -> Mapping[str, Any]:
    guard_not_stable_runtime(config.product_db_path, "Mango product database")
    for root in config.recording_roots:
        guard_not_stable_runtime(root, "Mango recording root")
    guard_not_stable_runtime(config.out_dir, "Mango phone index preview output directory")
    out_dir = config.out_dir.resolve(strict=False)
    guard_external_handoffs_output(out_dir, "Mango phone index preview output directory")
    guard_git_ignored_output(out_dir, "Mango phone index preview output directory")
    if config.include_product_db and not config.product_db_path.exists():
        raise FileNotFoundError(f"Mango product database not found: {config.product_db_path}")

    recording_roots = [Path(root).resolve(strict=False) for root in config.recording_roots]
    if config.include_recording_filenames:
        for root in recording_roots:
            if not root.exists():
                raise FileNotFoundError(f"Mango recording root not found: {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_db_path = out_dir / "mango_phone_index_preview.sqlite"
    report_path = out_dir / "mango_phone_index_preview_report.json"

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    product_source_counts: Mapping[str, int] = {}
    product_call_refs_seen = 0
    product_distinct_phones: set[str] = set()
    if config.include_product_db:
        product_by_phone, product_source_counts = load_mango_calls_by_phone(config.product_db_path)
        for phone, calls in product_by_phone.items():
            product_distinct_phones.add(phone)
            for call in calls:
                product_call_refs_seen += 1
                row = mango_phone_index_row_from_call(
                    phone=phone,
                    call=call,
                    source_kind=f"product_db:{clean_text(call.get('source_table')) or 'unknown'}",
                    source_path_sha256="",
                    source_filename_sha256="",
                    source_root_index=None,
                )
                rows_by_key.setdefault((phone, row["call_ref_key"]), row)

    recording_stats = (
        scan_recording_filename_phone_refs(recording_roots)
        if config.include_recording_filenames
        else {
            "rows": [],
            "files_seen": 0,
            "files_with_filename_phone": 0,
            "distinct_filename_phones": 0,
        }
    )
    for row in recording_stats["rows"]:
        rows_by_key.setdefault((row["normalized_phone"], row["call_ref_key"]), row)

    source_kind_counts = Counter(row["source_kind"] for row in rows_by_key.values())
    distinct_phones = {phone for phone, _call_ref in rows_by_key}

    with sqlite3.connect(str(out_db_path)) as out:
        out.row_factory = sqlite3.Row
        out.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mango_phone_call_refs (
              normalized_phone TEXT NOT NULL,
              phone_sha256 TEXT NOT NULL,
              call_ref_key TEXT NOT NULL,
              source_kind TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              provider TEXT NOT NULL,
              event_key TEXT NOT NULL,
              provider_call_id TEXT NOT NULL,
              started_at TEXT,
              direction TEXT,
              manager_ref TEXT,
              recording_ref TEXT,
              status TEXT,
              product_call_started_at TEXT,
              duration_sec REAL,
              manager_extension TEXT,
              crm_owner_id INTEGER,
              crm_match_status TEXT,
              source_path_sha256 TEXT,
              source_filename_sha256 TEXT,
              source_root_index INTEGER,
              PRIMARY KEY (normalized_phone, call_ref_key)
            );
            DELETE FROM meta;
            DELETE FROM mango_phone_call_refs;
            """
        )
        out.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", MANGO_PHONE_INDEX_PREVIEW_SCHEMA_VERSION),
                ("created_at", utc_now()),
                ("product_db_path", str(config.product_db_path)),
                ("recording_root_count", str(len(recording_roots))),
            ],
        )
        out.executemany(
            """
            INSERT INTO mango_phone_call_refs (
              normalized_phone, phone_sha256, call_ref_key, source_kind,
              tenant_id, provider, event_key, provider_call_id, started_at,
              direction, manager_ref, recording_ref, status,
              product_call_started_at, duration_sec, manager_extension,
              crm_owner_id, crm_match_status, source_path_sha256,
              source_filename_sha256, source_root_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["normalized_phone"],
                    row["phone_sha256"],
                    row["call_ref_key"],
                    row["source_kind"],
                    row["tenant_id"],
                    row["provider"],
                    row["event_key"],
                    row["provider_call_id"],
                    row["started_at"],
                    row["direction"],
                    row["manager_ref"],
                    row["recording_ref"],
                    row["status"],
                    row["product_call_started_at"],
                    row["duration_sec"],
                    row["manager_extension"],
                    row["crm_owner_id"],
                    row["crm_match_status"],
                    row["source_path_sha256"],
                    row["source_filename_sha256"],
                    row["source_root_index"],
                )
                for row in sorted(
                    rows_by_key.values(),
                    key=lambda item: (
                        clean_text(item.get("normalized_phone")),
                        clean_text(item.get("started_at")),
                        clean_text(item.get("call_ref_key")),
                    ),
                )
            ],
        )
        out.executescript(
            """
            CREATE VIEW IF NOT EXISTS v_mango_phone_index_source_counts AS
            SELECT source_kind, COUNT(*) AS call_refs, COUNT(DISTINCT normalized_phone) AS phones
            FROM mango_phone_call_refs
            GROUP BY source_kind;
            """
        )
        out.commit()

    remove_sqlite_sidecars(out_db_path)
    report = {
        "schema_version": MANGO_PHONE_INDEX_PREVIEW_SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_file_counts": {
            "recording_root_count": len(recording_roots),
            "recording_files_seen": int(recording_stats["files_seen"]),
            "recording_files_with_filename_phone": int(recording_stats["files_with_filename_phone"]),
            "distinct_recording_filename_phones": int(recording_stats["distinct_filename_phones"]),
            "product_call_refs_seen": product_call_refs_seen,
            "distinct_product_db_phones": len(product_distinct_phones),
        },
        "product_source_counts": dict(product_source_counts),
        "phone_index_counts": {
            "rows_written": len(rows_by_key),
            "distinct_normalized_phones": len(distinct_phones),
            "duplicate_call_refs_collapsed": (
                product_call_refs_seen + len(recording_stats["rows"]) - len(rows_by_key)
            ),
            "source_kind_counts": {kind: int(count) for kind, count in sorted(source_kind_counts.items())},
        },
        "safety": {
            "source_sqlite_mode": "mode=ro",
            "source_sqlite_query_only": True,
            "read_only_product_db": True,
            "open_audio_files": False,
            "scan_filenames_only": True,
            "write_crm": False,
            "write_tallanto": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "run_asr": False,
            "run_ra": False,
        },
        "privacy": {
            "contains_raw_personal_values": False,
            "raw_phones_written_to_json": False,
            "raw_filenames_written_to_json": False,
            "raw_paths_written_to_json": False,
            "phone_index_sqlite_contains_phone_refs": True,
            "source_paths_stored_as_sha256": True,
            "source_filenames_stored_as_sha256": True,
        },
        "paths": {
            "index_db": str(out_db_path),
            "report": str(report_path),
            "product_db": str(config.product_db_path),
            "recording_root_count": len(recording_roots),
        },
    }
    write_json(report_path, report)
    return report


def build_mail_mango_bridge_preview(config: MailMangoBridgePreviewConfig) -> Mapping[str, Any]:
    guard_not_stable_runtime(config.mail_handoff_db_path, "mail handoff database")
    guard_not_stable_runtime(config.identity_db_path, "identity database")
    guard_not_stable_runtime(config.product_db_path, "Mango product database")
    if config.mango_phone_index_db_path is not None:
        guard_not_stable_runtime(config.mango_phone_index_db_path, "Mango phone index preview database")
    guard_not_stable_runtime(config.out_dir, "mail Mango bridge preview output directory")
    out_dir = config.out_dir.resolve(strict=False)
    guard_external_handoffs_output(out_dir, "mail Mango bridge preview output directory")
    guard_git_ignored_output(out_dir, "mail Mango bridge preview output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_db_path = out_dir / "mail_mango_bridge_preview.sqlite"
    report_path = out_dir / "mail_mango_bridge_preview_report.json"

    for source_path, label in (
        (config.mail_handoff_db_path, "mail handoff database"),
        (config.identity_db_path, "identity database"),
        (config.product_db_path, "Mango product database"),
        (config.mango_phone_index_db_path, "Mango phone index preview database"),
    ):
        if source_path is not None and not source_path.exists():
            raise FileNotFoundError(f"{label} not found: {source_path}")

    mail_summary = load_mail_handoff_candidate_summary(config.mail_handoff_db_path)
    candidate_refs = load_identity_candidate_refs(config.identity_db_path)
    candidate_phones = load_candidate_phone_links(config.identity_db_path)
    mango_by_phone, mango_source_counts = load_mango_calls_by_phone(config.product_db_path)
    if config.mango_phone_index_db_path is not None:
        index_by_phone, index_counts = load_mango_phone_index_calls_by_phone(
            config.mango_phone_index_db_path
        )
        mango_by_phone, index_rows_added = merge_mango_calls_by_phone(mango_by_phone, index_by_phone)
        mango_source_counts = {
            **mango_source_counts,
            "phone_index_enabled": 1,
            "phone_index_call_refs_loaded": int(index_counts["call_refs_loaded"]),
            "phone_index_distinct_phones": int(index_counts["distinct_normalized_phones"]),
            "phone_index_rows_added_after_dedupe": index_rows_added,
            "combined_distinct_normalized_phones": len(mango_by_phone),
        }
    else:
        mango_source_counts = {
            **mango_source_counts,
            "phone_index_enabled": 0,
            "combined_distinct_normalized_phones": len(mango_by_phone),
        }

    status_counts: Counter[str] = Counter()
    blocked_reason_counts: Counter[str] = Counter()
    phone_ref_rows = 0
    call_ref_rows = 0
    resolved_candidates_with_calls = 0

    with sqlite3.connect(str(out_db_path)) as out:
        out.row_factory = sqlite3.Row
        out.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidate_mango_preview (
              candidate_key TEXT PRIMARY KEY,
              tallanto_id TEXT,
              amocrm_id TEXT,
              mail_message_count INTEGER NOT NULL,
              first_mail_date_iso TEXT,
              last_mail_date_iso TEXT,
              phone_value_count INTEGER NOT NULL,
              strong_phone_value_count INTEGER NOT NULL,
              duplicate_phone_value_count INTEGER NOT NULL,
              mango_call_count INTEGER NOT NULL,
              first_call_started_at TEXT,
              last_call_started_at TEXT,
              bridge_status TEXT NOT NULL,
              blocked_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS candidate_phone_refs (
              candidate_key TEXT NOT NULL,
              normalized_phone TEXT NOT NULL,
              phone_sha256 TEXT NOT NULL,
              phone_match_class TEXT NOT NULL,
              phone_candidate_count INTEGER NOT NULL,
              PRIMARY KEY (candidate_key, normalized_phone)
            );
            CREATE TABLE IF NOT EXISTS mango_call_refs (
              candidate_key TEXT NOT NULL,
              normalized_phone TEXT NOT NULL,
              call_ref_key TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              provider TEXT NOT NULL,
              event_key TEXT NOT NULL,
              provider_call_id TEXT NOT NULL,
              started_at TEXT,
              direction TEXT,
              manager_ref TEXT,
              recording_ref TEXT,
              status TEXT,
              product_call_started_at TEXT,
              duration_sec REAL,
              manager_extension TEXT,
              crm_owner_id INTEGER,
              crm_match_status TEXT,
              PRIMARY KEY (candidate_key, normalized_phone, call_ref_key)
            );
            CREATE TABLE IF NOT EXISTS blocked_bridge_candidates (
              candidate_key TEXT PRIMARY KEY,
              blocked_reason TEXT NOT NULL
            );
            DELETE FROM meta;
            DELETE FROM candidate_mango_preview;
            DELETE FROM candidate_phone_refs;
            DELETE FROM mango_call_refs;
            DELETE FROM blocked_bridge_candidates;
            """
        )
        out.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", MAIL_MANGO_BRIDGE_PREVIEW_SCHEMA_VERSION),
                ("created_at", utc_now()),
                ("mail_handoff_db_path", str(config.mail_handoff_db_path)),
                ("identity_db_path", str(config.identity_db_path)),
                ("product_db_path", str(config.product_db_path)),
                ("mango_phone_index_db_path", str(config.mango_phone_index_db_path or "")),
            ],
        )

        for candidate_key, mail in sorted(mail_summary["candidate_summaries"].items()):
            refs = candidate_refs.get(candidate_key, {})
            phone_rows = candidate_phones.get(candidate_key, [])
            strong_phone_values = [
                row["normalized_phone"]
                for row in phone_rows
                if row["match_class"] == "strong_unique"
            ]
            duplicate_phone_rows = [
                row for row in phone_rows if row["match_class"] != "strong_unique"
            ]
            calls_by_key: dict[str, Mapping[str, Any]] = {}
            for phone in strong_phone_values:
                for call in mango_by_phone.get(phone, []):
                    calls_by_key.setdefault(str(call["call_ref_key"]), call)
            calls = sorted(
                calls_by_key.values(),
                key=lambda row: (clean_text(row.get("started_at")), clean_text(row.get("call_ref_key"))),
            )

            if not phone_rows:
                bridge_status = "blocked"
                blocked_reason = "no_phone_for_candidate"
            elif not strong_phone_values:
                bridge_status = "blocked"
                blocked_reason = "phone_multiple_candidates"
            elif not calls:
                bridge_status = "blocked"
                blocked_reason = "mango_no_phone_match"
            else:
                bridge_status = "resolved"
                blocked_reason = ""
                resolved_candidates_with_calls += 1

            status_counts[bridge_status] += 1
            if blocked_reason:
                blocked_reason_counts[blocked_reason] += 1
                out.execute(
                    """
                    INSERT INTO blocked_bridge_candidates (candidate_key, blocked_reason)
                    VALUES (?, ?)
                    """,
                    (candidate_key, blocked_reason),
                )

            out.execute(
                """
                INSERT INTO candidate_mango_preview (
                  candidate_key, tallanto_id, amocrm_id, mail_message_count,
                  first_mail_date_iso, last_mail_date_iso, phone_value_count,
                  strong_phone_value_count, duplicate_phone_value_count, mango_call_count,
                  first_call_started_at, last_call_started_at, bridge_status, blocked_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_key,
                    clean_text(refs.get("tallanto_id")),
                    clean_text(refs.get("amocrm_id")),
                    int(mail["mail_message_count"]),
                    clean_text(mail["first_mail_date_iso"]),
                    clean_text(mail["last_mail_date_iso"]),
                    len(phone_rows),
                    len(strong_phone_values),
                    len(duplicate_phone_rows),
                    len(calls),
                    clean_text(calls[0].get("started_at")) if calls else "",
                    clean_text(calls[-1].get("started_at")) if calls else "",
                    bridge_status,
                    blocked_reason,
                ),
            )
            for row in phone_rows:
                out.execute(
                    """
                    INSERT INTO candidate_phone_refs (
                      candidate_key, normalized_phone, phone_sha256,
                      phone_match_class, phone_candidate_count
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_key,
                        row["normalized_phone"],
                        hashlib.sha256(row["normalized_phone"].encode("utf-8")).hexdigest(),
                        row["match_class"],
                        int(row["candidate_count"]),
                    ),
                )
                phone_ref_rows += 1
            for call in calls[: max(0, int(config.max_call_refs_per_candidate))]:
                out.execute(
                    """
                    INSERT INTO mango_call_refs (
                      candidate_key, normalized_phone, call_ref_key, tenant_id, provider,
                      event_key, provider_call_id, started_at, direction, manager_ref,
                      recording_ref, status, product_call_started_at, duration_sec,
                      manager_extension, crm_owner_id, crm_match_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_key,
                        call["normalized_phone"],
                        call["call_ref_key"],
                        call["tenant_id"],
                        call["provider"],
                        call["event_key"],
                        call["provider_call_id"],
                        call["started_at"],
                        call["direction"],
                        call["manager_ref"],
                        call["recording_ref"],
                        call["status"],
                        call["product_call_started_at"],
                        call["duration_sec"],
                        call["manager_extension"],
                        call["crm_owner_id"],
                        call["crm_match_status"],
                    ),
                )
                call_ref_rows += 1
        out.executescript(
            """
            CREATE VIEW IF NOT EXISTS v_resolved_mail_mango_links AS
            SELECT * FROM candidate_mango_preview WHERE bridge_status = 'resolved';
            CREATE VIEW IF NOT EXISTS v_manual_review_mail_mango_links AS
            SELECT * FROM candidate_mango_preview WHERE bridge_status <> 'resolved';
            """
        )
        out.commit()

    remove_sqlite_sidecars(out_db_path)
    candidate_count = len(mail_summary["candidate_summaries"])
    report = {
        "schema_version": MAIL_MANGO_BRIDGE_PREVIEW_SCHEMA_VERSION,
        "created_at": utc_now(),
        "candidate_count": candidate_count,
        "mail_link_reconciliation": {
            **mail_summary["link_status_counts"],
            "total": int(mail_summary["total_links"]),
            "pass": (
                int(mail_summary["total_links"])
                == sum(int(value) for value in mail_summary["link_status_counts"].values())
            ),
        },
        "counts": {
            "resolved": int(status_counts.get("resolved", 0)),
            "blocked": candidate_count - int(status_counts.get("resolved", 0)),
            "no_phone_for_candidate": int(blocked_reason_counts.get("no_phone_for_candidate", 0)),
            "phone_multiple_candidates": int(blocked_reason_counts.get("phone_multiple_candidates", 0)),
            "mango_no_phone_match": int(blocked_reason_counts.get("mango_no_phone_match", 0)),
        },
        "mango_source_counts": mango_source_counts,
        "artifact_counts": {
            "candidate_phone_refs": phone_ref_rows,
            "mango_call_refs_written": call_ref_rows,
            "resolved_candidates_with_calls": resolved_candidates_with_calls,
        },
        "safety": {
            "source_sqlite_mode": "mode=ro",
            "source_sqlite_query_only": True,
            "source_db_attached_to_writer": False,
            "read_only_source_archives": True,
            "read_only_identity_db": True,
            "read_only_product_db": True,
            "write_crm": False,
            "write_tallanto": False,
            "live_crm_reads": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "run_asr": False,
            "run_ra": False,
            "store_raw_files_in_sqlite": False,
        },
        "privacy": {
            "contains_raw_personal_values": False,
            "raw_emails_written": False,
            "raw_names_written": False,
            "raw_mail_written": False,
            "raw_payloads_written": False,
            "raw_audio_written": False,
            "bridge_sqlite_contains_phone_refs": True,
            "bridge_sqlite_contains_customer_refs": True,
        },
        "paths": {
            "preview_db": str(out_db_path),
            "report": str(report_path),
            "mail_handoff_db": str(config.mail_handoff_db_path),
            "identity_db": str(config.identity_db_path),
            "product_db": str(config.product_db_path),
        },
    }
    write_json(report_path, report)
    return report


def build_mail_phone_lift_preview(config: MailPhoneLiftPreviewConfig) -> Mapping[str, Any]:
    for archive_db_path in config.archive_db_paths:
        guard_not_stable_runtime(archive_db_path, "mail archive database")
    guard_not_stable_runtime(config.identity_db_path, "identity database")
    guard_not_stable_runtime(config.out_dir, "mail phone lift preview output directory")
    out_dir = config.out_dir.resolve(strict=False)
    guard_external_handoffs_output(out_dir, "mail phone lift preview output directory")
    guard_git_ignored_output(out_dir, "mail phone lift preview output directory")

    source_paths = [Path(path).resolve(strict=False) for path in config.archive_db_paths]
    if not source_paths:
        raise ValueError("at least one mail archive database is required")
    for source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"mail archive database not found: {source_path}")
    if not config.identity_db_path.exists():
        raise FileNotFoundError(f"identity database not found: {config.identity_db_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_db_path = out_dir / "mail_phone_lift_preview.sqlite"
    report_path = out_dir / "mail_phone_lift_preview_report.json"

    phone_identity = load_phone_identity_index(config.identity_db_path)
    source_rows = load_mail_phone_lift_source_messages(source_paths)

    original_counts: Counter[str] = Counter()
    lift_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    distinct_phone_hashes: set[str] = set()
    text_files_read = 0
    text_files_missing = 0
    text_files_unreadable = 0
    text_paths_rejected = 0
    total_extracted_phone_values = 0
    total_identity_phone_matches = 0

    with sqlite3.connect(str(out_db_path)) as out:
        out.row_factory = sqlite3.Row
        out.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message_phone_lift_preview (
              source_archive_id TEXT NOT NULL,
              message_sha256 TEXT NOT NULL,
              message_date_iso TEXT,
              original_match_class TEXT NOT NULL,
              text_available INTEGER NOT NULL,
              text_path_status TEXT NOT NULL,
              extracted_phone_count INTEGER NOT NULL,
              identity_phone_match_count INTEGER NOT NULL,
              identity_candidate_count INTEGER NOT NULL,
              lift_class TEXT NOT NULL,
              phone_sha256_json TEXT NOT NULL,
              candidate_keys_json TEXT NOT NULL,
              evaluated_at TEXT NOT NULL,
              PRIMARY KEY (source_archive_id, message_sha256)
            );
            DELETE FROM meta;
            DELETE FROM message_phone_lift_preview;
            """
        )
        out.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", MAIL_PHONE_LIFT_PREVIEW_SCHEMA_VERSION),
                ("created_at", utc_now()),
                ("identity_db_path", str(config.identity_db_path)),
                ("source_archive_count", str(len(source_paths))),
            ],
        )

        for row in source_rows:
            original_match_class = clean_text(row["match_class"])
            original_counts[original_match_class] += 1
            source_counts[clean_text(row["source_archive_id"])] += 1

            text, text_path_status = read_safe_extracted_text(
                row["extracted_text_path"],
                archive_db_path=Path(row["source_archive_path"]),
                max_chars=config.max_text_chars_per_message,
            )
            text_available = bool(text)
            if text_path_status == "read":
                text_files_read += 1
            elif text_path_status == "missing":
                text_files_missing += 1
            elif text_path_status == "rejected":
                text_paths_rejected += 1
            elif text_path_status == "unreadable":
                text_files_unreadable += 1

            phone_values = extract_phone_numbers(text)
            phone_hashes = [
                hashlib.sha256(phone.encode("utf-8")).hexdigest()
                for phone in phone_values
            ]
            distinct_phone_hashes.update(phone_hashes)
            matched_phone_count = sum(1 for phone in phone_values if phone in phone_identity)
            total_extracted_phone_values += len(phone_values)
            total_identity_phone_matches += matched_phone_count
            candidate_keys = unique_preserving_order(
                candidate_key
                for phone in phone_values
                for candidate_key in phone_identity.get(phone, {}).get("candidate_keys", [])
            )

            if text_path_status != "read":
                lift_class = "no_text"
            elif not phone_values:
                lift_class = "no_phone_detected"
            elif not candidate_keys:
                lift_class = "phone_no_identity_match"
            elif len(candidate_keys) == 1:
                lift_class = "phone_strong_unique"
            else:
                lift_class = "phone_ambiguous"
            lift_counts[lift_class] += 1

            out.execute(
                """
                INSERT INTO message_phone_lift_preview (
                  source_archive_id, message_sha256, message_date_iso,
                  original_match_class, text_available, text_path_status,
                  extracted_phone_count, identity_phone_match_count,
                  identity_candidate_count, lift_class, phone_sha256_json,
                  candidate_keys_json, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["source_archive_id"],
                    row["message_sha256"],
                    row["message_date_iso"],
                    original_match_class,
                    int(text_available),
                    text_path_status,
                    len(phone_values),
                    matched_phone_count,
                    len(candidate_keys),
                    lift_class,
                    json.dumps(phone_hashes, ensure_ascii=False),
                    json.dumps(candidate_keys, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )
        out.executescript(
            """
            CREATE VIEW IF NOT EXISTS v_phone_strong_unique_lift AS
            SELECT * FROM message_phone_lift_preview
            WHERE lift_class = 'phone_strong_unique';
            CREATE VIEW IF NOT EXISTS v_phone_manual_review_lift AS
            SELECT * FROM message_phone_lift_preview
            WHERE lift_class <> 'phone_strong_unique';
            """
        )
        out.commit()

    remove_sqlite_sidecars(out_db_path)
    evaluated_count = len(source_rows)
    strong_lift_count = int(lift_counts.get("phone_strong_unique", 0))
    report = {
        "schema_version": MAIL_PHONE_LIFT_PREVIEW_SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_archive_count": len(source_paths),
        "evaluated_message_count": evaluated_count,
        "by_original_match_class": {
            "ambiguous": int(original_counts.get("ambiguous", 0)),
            "missing": int(original_counts.get("missing", 0)),
        },
        "lift_class_counts": {
            "phone_strong_unique": strong_lift_count,
            "phone_ambiguous": int(lift_counts.get("phone_ambiguous", 0)),
            "phone_no_identity_match": int(lift_counts.get("phone_no_identity_match", 0)),
            "no_phone_detected": int(lift_counts.get("no_phone_detected", 0)),
            "no_text": int(lift_counts.get("no_text", 0)),
        },
        "potential_lift": {
            "strong_unique_messages": strong_lift_count,
            "manual_review_messages": evaluated_count - strong_lift_count,
            "ambiguous_to_phone_strong_unique": count_phone_lift_rows(
                out_db_path,
                original_match_class="ambiguous",
                lift_class="phone_strong_unique",
            ),
            "missing_to_phone_strong_unique": count_phone_lift_rows(
                out_db_path,
                original_match_class="missing",
                lift_class="phone_strong_unique",
            ),
        },
        "text_access_counts": {
            "text_files_read": text_files_read,
            "text_files_missing": text_files_missing,
            "text_files_unreadable": text_files_unreadable,
            "text_paths_rejected": text_paths_rejected,
        },
        "artifact_counts": {
            "preview_rows_written": evaluated_count,
            "extracted_phone_values_seen": total_extracted_phone_values,
            "identity_phone_value_matches": total_identity_phone_matches,
            "distinct_phone_hashes": len(distinct_phone_hashes),
        },
        "safety": {
            "source_sqlite_mode": "mode=ro",
            "source_sqlite_query_only": True,
            "source_db_attached_to_writer": False,
            "read_only_source_archives": True,
            "read_only_identity_db": True,
            "write_crm": False,
            "write_tallanto": False,
            "live_crm_reads": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "run_asr": False,
            "run_ra": False,
            "open_attachments": False,
            "read_extracted_text_files": True,
            "store_raw_files_in_sqlite": False,
        },
        "privacy": {
            "contains_raw_personal_values": False,
            "raw_emails_written": False,
            "raw_names_written": False,
            "raw_phones_written": False,
            "raw_mail_written": False,
            "raw_text_written": False,
            "phone_hashes_written": True,
            "preview_sqlite_contains_candidate_refs": True,
        },
        "paths": {
            "preview_db": str(out_db_path),
            "report": str(report_path),
            "identity_db": str(config.identity_db_path),
            "archive_db_count": len(source_paths),
        },
    }
    write_json(report_path, report)
    return report


def ingest_one_message(
    imap: ImapClient,
    *,
    msg_id: bytes,
    credentials: MailImapCredentials,
    config: MailArchiveIngestConfig,
    db_path: Path,
    raw_dir: Path,
    attachment_dir: Path,
    text_dir: Path,
) -> Mapping[str, Any]:
    fetch_status, fetch_data = imap.fetch(msg_id, FULL_MESSAGE_FETCH_QUERY)
    if fetch_status != "OK":
        raise RuntimeError(f"IMAP FETCH failed: {fetch_status}")
    raw = first_fetch_payload(fetch_data)
    if not raw:
        raise RuntimeError("IMAP FETCH returned empty payload")
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    raw_path = raw_message_path(raw_dir, raw_sha256)
    raw_written = write_bytes_once(raw_path, raw)

    msg = email.message_from_bytes(raw, policy=policy.default)
    metadata = message_metadata(msg)
    source_key = hashlib.sha256(
        "|".join(
            [
                config.account_label,
                config.mailbox,
                msg_id.decode("ascii", "ignore"),
                metadata["message_id"],
                raw_sha256,
            ]
        ).encode("utf-8")
    ).hexdigest()
    message_kind = classify_message_kind(
        mailbox_email=credentials.email_address,
        participants=message_participants(msg),
        configured_internal_domains=config.internal_domains,
    )
    extracted_text = extract_message_text(msg, max_chars=config.extracted_text_max_chars)
    text_path = text_dir / f"{raw_sha256}.txt"
    text_written = write_text_once(text_path, extracted_text) if extracted_text else False
    attachments_written = save_attachments(msg, raw_sha256=raw_sha256, attachment_dir=attachment_dir)

    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        upsert_message(
            con,
            sha256=raw_sha256,
            source_key=source_key,
            msg_id=msg_id,
            config=config,
            metadata=metadata,
            message_kind=message_kind,
            raw_path=raw_path,
            text_path=text_path if extracted_text else None,
            raw_size_bytes=len(raw),
            extracted_text_chars=len(extracted_text),
        )
        replace_participants(con, raw_sha256, message_participants(msg))
        replace_attachments(con, raw_sha256, msg, attachment_dir / raw_sha256)
        con.commit()

    return {
        "sha256": raw_sha256,
        "raw_eml_written": raw_written,
        "text_file_written": text_written,
        "attachments_written": attachments_written,
    }


def read_tallanto_rows(path: Path, *, encoding: str, delimiter: str) -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        return [dict(row) for row in reader]


def collect_row_identity_values(
    row: Mapping[str, Any],
    *,
    columns: Sequence[str],
    extractor: Any,
) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for column in columns:
        if column not in row:
            continue
        for value in extractor(row.get(column)):
            values.setdefault(value, [])
            if column not in values[value]:
                values[value].append(column)
    return values


def append_identity_link(
    identity_links: dict[tuple[str, str], dict[str, Any]],
    kind: str,
    value: str,
    candidate_key: str,
    columns: Sequence[str],
) -> None:
    item = identity_links.setdefault(
        (kind, value),
        {
            "kind": kind,
            "value": value,
            "candidate_keys": [],
            "columns_by_candidate": {},
            "match_class": "",
        },
    )
    if candidate_key not in item["candidate_keys"]:
        item["candidate_keys"].append(candidate_key)
    item["columns_by_candidate"].setdefault(candidate_key, [])
    for column in columns:
        if column not in item["columns_by_candidate"][candidate_key]:
            item["columns_by_candidate"][candidate_key].append(column)


def write_identity_map_db(
    db_path: Path,
    *,
    source_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    identity_links: Mapping[tuple[str, str], Mapping[str, Any]],
    row_identity_classes: Mapping[str, Mapping[str, str]],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        con.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS identity_candidates (
              candidate_key TEXT PRIMARY KEY,
              row_number INTEGER NOT NULL,
              tallanto_id TEXT,
              amocrm_id TEXT,
              first_name TEXT,
              last_name TEXT,
              parent_name TEXT,
              student_type TEXT,
              manager TEXT,
              manager_id TEXT,
              group_id TEXT,
              email_class TEXT NOT NULL,
              phone_class TEXT NOT NULL,
              candidate_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS identity_values (
              kind TEXT NOT NULL,
              value TEXT NOT NULL,
              match_class TEXT NOT NULL,
              candidate_count INTEGER NOT NULL,
              PRIMARY KEY (kind, value)
            );
            CREATE TABLE IF NOT EXISTS identity_links (
              kind TEXT NOT NULL,
              value TEXT NOT NULL,
              candidate_key TEXT NOT NULL,
              source_columns_json TEXT NOT NULL,
              PRIMARY KEY (kind, value, candidate_key)
            );
            DELETE FROM meta;
            DELETE FROM identity_candidates;
            DELETE FROM identity_values;
            DELETE FROM identity_links;
            """
        )
        con.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", TALLANTO_IDENTITY_MAP_SCHEMA_VERSION),
                ("created_at", utc_now()),
                ("source_path", str(source_path)),
            ],
        )
        for row in candidate_rows:
            classes = row_identity_classes[row["candidate_key"]]
            con.execute(
                """
                INSERT INTO identity_candidates (
                  candidate_key, row_number, tallanto_id, amocrm_id, first_name, last_name,
                  parent_name, student_type, manager, manager_id, group_id,
                  email_class, phone_class, candidate_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["candidate_key"],
                    row["row_number"],
                    row.get("tallanto_id", ""),
                    row.get("amocrm_id", ""),
                    row.get("first_name", ""),
                    row.get("last_name", ""),
                    row.get("parent_name", ""),
                    row.get("student_type", ""),
                    row.get("manager", ""),
                    row.get("manager_id", ""),
                    row.get("group_id", ""),
                    classes["email"],
                    classes["phone"],
                    row["candidate_json"],
                ),
            )
        for item in identity_links.values():
            con.execute(
                """
                INSERT INTO identity_values (kind, value, match_class, candidate_count)
                VALUES (?, ?, ?, ?)
                """,
                (
                    item["kind"],
                    item["value"],
                    item["match_class"],
                    len(item["candidate_keys"]),
                ),
            )
            for candidate_key, columns in item["columns_by_candidate"].items():
                con.execute(
                    """
                    INSERT INTO identity_links (
                      kind, value, candidate_key, source_columns_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        item["kind"],
                        item["value"],
                        candidate_key,
                        json.dumps(columns, ensure_ascii=False),
                    ),
                )
        con.commit()


def build_identity_map_report(
    *,
    db_path: Path,
    report_path: Path,
    source_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    identity_links: Mapping[tuple[str, str], Mapping[str, Any]],
    row_identity_classes: Mapping[str, Mapping[str, str]],
    config: TallantoIdentityMapConfig,
    duplicate_tallanto_id_values: int = 0,
) -> Mapping[str, Any]:
    values_by_kind: dict[str, dict[str, int]] = {
        "email": {"strong_unique": 0, "duplicate": 0},
        "phone": {"strong_unique": 0, "duplicate": 0},
    }
    for item in identity_links.values():
        values_by_kind[item["kind"]][item["match_class"]] += 1

    row_counts: dict[str, dict[str, int]] = {
        "email": {"strong_unique": 0, "duplicate": 0, "missing": 0},
        "phone": {"strong_unique": 0, "duplicate": 0, "missing": 0},
    }
    rows_with_any_strong = 0
    rows_with_both_strong = 0
    for classes in row_identity_classes.values():
        row_counts["email"][classes["email"]] += 1
        row_counts["phone"][classes["phone"]] += 1
        email_strong = classes["email"] == "strong_unique"
        phone_strong = classes["phone"] == "strong_unique"
        rows_with_any_strong += int(email_strong or phone_strong)
        rows_with_both_strong += int(email_strong and phone_strong)

    source_size = source_path.stat().st_size if source_path.exists() else 0
    identity_db_size = db_path.stat().st_size if db_path.exists() else 0
    sanity_checks = {
        "email_row_classes_equal_row_count": sum(row_counts["email"].values()) == len(candidate_rows),
        "phone_row_classes_equal_row_count": sum(row_counts["phone"].values()) == len(candidate_rows),
        "identity_values_equal_links_by_value": len(identity_links) == sum(values_by_kind[kind][klass] for kind in values_by_kind for klass in values_by_kind[kind]),
        "source_file_exists": source_path.exists(),
        "identity_db_exists": db_path.exists(),
    }
    warnings = build_identity_report_warnings(
        duplicate_tallanto_id_values=duplicate_tallanto_id_values,
        values_by_kind=values_by_kind,
        row_counts=row_counts,
    )
    blocking_risks = [name for name, passed in sanity_checks.items() if not passed]

    return {
        "schema_version": TALLANTO_IDENTITY_MAP_SCHEMA_VERSION,
        "created_at": utc_now(),
        "provenance": build_local_provenance(),
        "source_path": str(source_path),
        "source_file": {
            "size_bytes": source_size,
            "sha256": sha256_file(source_path) if source_path.exists() else "",
        },
        "row_count": len(candidate_rows),
        "duplicate_tallanto_id_values": duplicate_tallanto_id_values,
        "source_id_quality": {
            "duplicate_tallanto_id_values": duplicate_tallanto_id_values,
        },
        "identity_values": values_by_kind,
        "row_identity_classes": row_counts,
        "row_coverage": {
            "rows_with_contact_email": len(candidate_rows) - row_counts["email"]["missing"],
            "rows_without_contact_email": row_counts["email"]["missing"],
            "rows_with_contact_phone": len(candidate_rows) - row_counts["phone"]["missing"],
            "rows_without_contact_phone": row_counts["phone"]["missing"],
            "rows_with_strong_unique_email": row_counts["email"]["strong_unique"],
            "rows_with_strong_unique_phone": row_counts["phone"]["strong_unique"],
            "rows_with_any_strong_email_or_phone": rows_with_any_strong,
            "rows_with_both_strong_email_and_phone": rows_with_both_strong,
        },
        "identity_link_counts": {
            "candidate_rows": len(candidate_rows),
            "identity_values": len(identity_links),
            "identity_links": sum(len(item["candidate_keys"]) for item in identity_links.values()),
        },
        "sanity_checks": sanity_checks,
        "columns_used": {
            "emails": list(config.email_columns),
            "phones": list(config.phone_columns),
            "candidate": list(config.candidate_columns),
        },
        "artifact_integrity": {
            "identity_db_size_bytes": identity_db_size,
            "identity_db_sha256": sha256_file(db_path) if db_path.exists() else "",
        },
        "audit_readiness": {
            "pass": not blocking_risks,
            "blocking_risks": blocking_risks,
            "warnings": warnings,
            "recommended_next_action": (
                "run_small_imap_pilot"
                if not blocking_risks
                else "fix_identity_map_sanity_checks_before_imap_pilot"
            ),
        },
        "pii_artifacts": {
            "source_csv": {
                "path": str(source_path),
                "contains_pii": True,
                "commit_safe": False,
            },
            "identity_db": {
                "path": str(db_path),
                "contains_pii": True,
                "commit_safe": False,
            },
            "identity_report": {
                "path": str(report_path),
                "contains_pii": False,
                "commit_safe": False,
                "reason": "operational handoff report under ignored raw artifact root",
            },
        },
        "privacy": {
            "report_contains_raw_personal_values": False,
            "sqlite_contains_selected_identity_values": True,
            "raw_history_columns_written": False,
            "raw_export_copied": False,
        },
        "paths": {
            "identity_db": str(db_path),
            "report": str(report_path),
        },
    }


def build_identity_report_warnings(
    *,
    duplicate_tallanto_id_values: int,
    values_by_kind: Mapping[str, Mapping[str, int]],
    row_counts: Mapping[str, Mapping[str, int]],
) -> list[str]:
    warnings: list[str] = []
    if duplicate_tallanto_id_values:
        warnings.append("duplicate_tallanto_id_values_present")
    for kind in ("email", "phone"):
        if int(values_by_kind[kind].get("duplicate") or 0):
            warnings.append(f"duplicate_{kind}_identity_values_present")
        if int(row_counts[kind].get("missing") or 0):
            warnings.append(f"rows_missing_{kind}_identity_present")
    warnings.append("identity_sqlite_contains_pii_keep_out_of_git")
    return warnings


def build_local_provenance() -> Mapping[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git": best_effort_git_provenance(),
    }


def best_effort_git_provenance() -> Mapping[str, Any]:
    def run_git(args: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:  # noqa: BLE001
            return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short_line_count": len([line for line in status.splitlines() if line.strip()]),
    }


def git_check_ignored(path: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return False
    return completed.returncode == 0


def guard_git_ignored_output(path: Path, label: str) -> None:
    if not git_check_ignored(path):
        raise ValueError(f"{label} must be git-ignored before writing raw mail artifacts")


def guard_external_handoffs_output(path: Path, label: str) -> None:
    if "_external_handoffs" not in path.resolve(strict=False).parts:
        raise ValueError(f"{label} must be under _external_handoffs")


def open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve(strict=False)), safe='/:')}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def init_mail_archive_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
              sha256 TEXT PRIMARY KEY,
              message_id TEXT,
              message_date_header TEXT,
              message_date_iso TEXT,
              subject TEXT,
              from_header TEXT,
              to_header TEXT,
              cc_header TEXT,
              mailbox TEXT NOT NULL,
              mailbox_raw TEXT NOT NULL,
              message_kind TEXT NOT NULL,
              raw_eml_path TEXT NOT NULL,
              extracted_text_path TEXT,
              raw_size_bytes INTEGER NOT NULL,
              extracted_text_chars INTEGER NOT NULL,
              first_ingested_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message_sources (
              source_key TEXT PRIMARY KEY,
              message_sha256 TEXT NOT NULL,
              account_label TEXT NOT NULL,
              mailbox TEXT NOT NULL,
              mailbox_raw TEXT NOT NULL,
              imap_seq TEXT NOT NULL,
              source_message_id TEXT,
              ingested_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message_participants (
              message_sha256 TEXT NOT NULL,
              header_name TEXT NOT NULL,
              display_name TEXT,
              email_raw TEXT,
              email_normalized TEXT,
              domain TEXT,
              PRIMARY KEY (message_sha256, header_name, email_normalized, email_raw)
            );
            CREATE TABLE IF NOT EXISTS attachments (
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
        con.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", MAIL_ARCHIVE_SCHEMA_VERSION),
                ("updated_at", utc_now()),
            ],
        )
        init_mail_match_tables(con)
        con.commit()


def init_mail_match_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS message_matches (
          message_sha256 TEXT PRIMARY KEY,
          match_class TEXT NOT NULL,
          candidate_count INTEGER NOT NULL,
          matched_email_count INTEGER NOT NULL,
          matched_at TEXT NOT NULL
        );
        """
    )


def empty_archive_db_counts() -> dict[str, int]:
    return {
        "messages": 0,
        "message_sources": 0,
        "message_participants": 0,
        "attachments": 0,
        "message_matches": 0,
    }


def archive_db_counts(con: sqlite3.Connection) -> dict[str, int]:
    counts = empty_archive_db_counts()
    for table in counts:
        try:
            counts[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.Error:
            counts[table] = 0
    return counts


def archive_db_has_required_tables(con: sqlite3.Connection) -> bool:
    required = {
        "messages",
        "message_sources",
        "message_participants",
        "attachments",
        "message_matches",
    }
    rows = con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    present = {str(row[0]) for row in rows}
    return required.issubset(present)


def upsert_message(
    con: sqlite3.Connection,
    *,
    sha256: str,
    source_key: str,
    msg_id: bytes,
    config: MailArchiveIngestConfig,
    metadata: Mapping[str, str],
    message_kind: str,
    raw_path: Path,
    text_path: Optional[Path],
    raw_size_bytes: int,
    extracted_text_chars: int,
) -> None:
    now = utc_now()
    existing = con.execute("SELECT first_ingested_at FROM messages WHERE sha256 = ?", (sha256,)).fetchone()
    first_ingested_at = existing["first_ingested_at"] if existing else now
    con.execute(
        """
        INSERT OR REPLACE INTO messages (
          sha256, message_id, message_date_header, message_date_iso, subject,
          from_header, to_header, cc_header, mailbox, mailbox_raw, message_kind,
          raw_eml_path, extracted_text_path, raw_size_bytes, extracted_text_chars,
          first_ingested_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sha256,
            metadata["message_id"],
            metadata["date"],
            metadata["date_iso"],
            metadata["subject"],
            metadata["from"],
            metadata["to"],
            metadata["cc"],
            config.mailbox_label,
            config.mailbox,
            message_kind,
            str(raw_path),
            str(text_path) if text_path else "",
            raw_size_bytes,
            extracted_text_chars,
            first_ingested_at,
            now,
        ),
    )
    con.execute(
        """
        INSERT OR REPLACE INTO message_sources (
          source_key, message_sha256, account_label, mailbox, mailbox_raw,
          imap_seq, source_message_id, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_key,
            sha256,
            config.account_label,
            config.mailbox_label,
            config.mailbox,
            msg_id.decode("ascii", "ignore"),
            metadata["message_id"],
            now,
        ),
    )


def replace_participants(
    con: sqlite3.Connection,
    message_sha256: str,
    participants: Sequence[Mapping[str, str]],
) -> None:
    con.execute("DELETE FROM message_participants WHERE message_sha256 = ?", (message_sha256,))
    for row in participants:
        con.execute(
            """
            INSERT OR REPLACE INTO message_participants (
              message_sha256, header_name, display_name, email_raw, email_normalized, domain
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_sha256,
                row["header_name"],
                row["display_name"],
                row["email_raw"],
                row["email_normalized"],
                domain_from_email(row["email_normalized"]),
            ),
        )


def replace_attachments(
    con: sqlite3.Connection,
    message_sha256: str,
    msg: Message,
    attachment_message_dir: Path,
) -> None:
    con.execute("DELETE FROM attachments WHERE message_sha256 = ?", (message_sha256,))
    for part_index, part in enumerate(iter_attachment_parts(msg), start=1):
        payload = part.get_payload(decode=True) or b""
        attachment_sha = hashlib.sha256(payload).hexdigest()
        filename = clean_header(part.get_filename()) or ""
        path = attachment_message_dir / f"part_{part_index:03d}_{attachment_sha[:16]}.bin"
        con.execute(
            """
            INSERT OR REPLACE INTO attachments (
              message_sha256, part_index, filename, content_type, content_disposition,
              size_bytes, sha256, path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_sha256,
                part_index,
                filename,
                clean_text(part.get_content_type()),
                clean_text(part.get("Content-Disposition")),
                len(payload),
                attachment_sha,
                str(path),
            ),
        )


def save_attachments(msg: Message, *, raw_sha256: str, attachment_dir: Path) -> int:
    written = 0
    message_dir = attachment_dir / raw_sha256
    for part_index, part in enumerate(iter_attachment_parts(msg), start=1):
        payload = part.get_payload(decode=True) or b""
        attachment_sha = hashlib.sha256(payload).hexdigest()
        path = message_dir / f"part_{part_index:03d}_{attachment_sha[:16]}.bin"
        if write_bytes_once(path, payload):
            written += 1
    return written


def iter_attachment_parts(msg: Message) -> list[Message]:
    parts: list[Message] = []
    if not msg.is_multipart():
        return parts
    for part in msg.walk():
        if part.is_multipart():
            continue
        disposition = clean_text(part.get_content_disposition()).lower()
        filename = clean_header(part.get_filename())
        if disposition == "attachment" or filename:
            parts.append(part)
    return parts


def message_metadata(msg: Message) -> dict[str, str]:
    date_header = clean_header(msg.get("Date"))
    return {
        "message_id": clean_header(msg.get("Message-ID")),
        "date": date_header,
        "date_iso": parse_email_date_iso(date_header),
        "from": clean_header(msg.get("From")),
        "to": clean_header(msg.get("To")),
        "cc": clean_header(msg.get("Cc")),
        "subject": clean_header(msg.get("Subject")),
    }


def message_participants(msg: Message) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for header_name in ("From", "To", "Cc", "Bcc", "Reply-To"):
        values = msg.get_all(header_name, [])
        for display_name, address in getaddresses(values):
            normalized = normalize_email(address)
            rows.append(
                {
                    "header_name": header_name,
                    "display_name": clean_header(display_name),
                    "email_raw": clean_text(address),
                    "email_normalized": normalized,
                }
            )
    return rows


def classify_message_kind(
    *,
    mailbox_email: str,
    participants: Sequence[Mapping[str, str]],
    configured_internal_domains: Sequence[str],
) -> str:
    internal_domains = normalize_domains(configured_internal_domains, mailbox_email)
    normalized = [row["email_normalized"] for row in participants if row.get("email_normalized")]
    if any(is_service_email(address) for address in normalized):
        return "service"
    external = [address for address in normalized if not email_domain_is_internal(address, internal_domains)]
    if normalized and not external:
        return "internal"
    return "external"


def extract_message_text(msg: Message, *, max_chars: int) -> str:
    chunks: list[str] = []
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.is_multipart():
            continue
        if clean_text(part.get_content_disposition()).lower() == "attachment":
            continue
        content_type = clean_text(part.get_content_type()).lower()
        if content_type not in {"text/plain", "text/html"}:
            continue
        try:
            text = part.get_content()
        except Exception:  # noqa: BLE001
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
        text = clean_html_text(text) if content_type == "text/html" else str(text)
        text = text.strip()
        if text:
            chunks.append(text)
        if sum(len(chunk) for chunk in chunks) >= max_chars:
            break
    joined = "\n\n--- part ---\n\n".join(chunks)
    return joined[:max(0, max_chars)]


def clean_html_text(text: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", without_tags).strip()


def load_identity_index(identity_db_path: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    with open_sqlite_readonly(identity_db_path) as con:
        for row in con.execute(
            """
            SELECT value, candidate_key
            FROM identity_links
            WHERE kind = 'email'
            ORDER BY value, candidate_key
            """
        ):
            index.setdefault(row["value"], []).append(row["candidate_key"])
    return index


def load_phone_identity_index(identity_db_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with open_sqlite_readonly(identity_db_path) as con:
        for row in con.execute(
            """
            SELECT il.value, il.candidate_key, iv.match_class, iv.candidate_count
            FROM identity_links il
            JOIN identity_values iv
              ON iv.kind = il.kind AND iv.value = il.value
            WHERE il.kind = 'phone'
            ORDER BY il.value, il.candidate_key
            """
        ):
            phone = clean_text(row["value"])
            item = index.setdefault(
                phone,
                {
                    "candidate_keys": [],
                    "match_class": clean_text(row["match_class"]),
                    "candidate_count": int(row["candidate_count"]),
                },
            )
            item["candidate_keys"].append(clean_text(row["candidate_key"]))
    return index


def load_mail_phone_lift_source_messages(archive_db_paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for archive_db_path in archive_db_paths:
        source_path = Path(archive_db_path).resolve(strict=False)
        source_archive_id = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
        with open_sqlite_readonly(source_path) as con:
            required_tables = {"messages", "message_matches"}
            present = {
                clean_text(row["name"])
                for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            missing = sorted(required_tables - present)
            if missing:
                raise ValueError(
                    f"mail archive database is missing required tables for phone lift: {', '.join(missing)}"
                )
            for row in con.execute(
                """
                SELECT
                  m.sha256 AS message_sha256,
                  m.message_date_iso,
                  m.extracted_text_path,
                  mm.match_class
                FROM message_matches mm
                JOIN messages m ON m.sha256 = mm.message_sha256
                WHERE mm.match_class IN ('ambiguous', 'missing')
                ORDER BY m.message_date_iso, m.sha256
                """
            ):
                rows.append(
                    {
                        "source_archive_id": source_archive_id,
                        "source_archive_path": str(source_path),
                        "message_sha256": clean_text(row["message_sha256"]),
                        "message_date_iso": clean_text(row["message_date_iso"]),
                        "extracted_text_path": clean_text(row["extracted_text_path"]),
                        "match_class": clean_text(row["match_class"]),
                    }
                )
    return rows


def read_safe_extracted_text(
    value: object,
    *,
    archive_db_path: Path,
    max_chars: int,
) -> tuple[str, str]:
    raw_path = clean_text(value)
    if not raw_path:
        return "", "missing"
    path = Path(raw_path).resolve(strict=False)
    archive_dir = archive_db_path.resolve(strict=False).parent
    try:
        path.relative_to(archive_dir)
    except ValueError:
        return "", "rejected"
    if "_external_handoffs" not in path.parts:
        return "", "rejected"
    if "stable_runtime" in {part.casefold() for part in path.parts}:
        return "", "rejected"
    if "extracted_text" not in path.parts:
        return "", "rejected"
    if path.suffix.casefold() != ".txt":
        return "", "rejected"
    if not path.exists():
        return "", "missing"
    try:
        return path.read_text(encoding="utf-8", errors="replace")[: max(0, int(max_chars))], "read"
    except OSError:
        return "", "unreadable"


def count_phone_lift_rows(
    preview_db_path: Path,
    *,
    original_match_class: str,
    lift_class: str,
) -> int:
    with open_sqlite_readonly(preview_db_path) as con:
        return int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM message_phone_lift_preview
                WHERE original_match_class = ? AND lift_class = ?
                """,
                (original_match_class, lift_class),
            ).fetchone()[0]
        )


def load_mail_handoff_candidate_summary(mail_handoff_db_path: Path) -> dict[str, Any]:
    with open_sqlite_readonly(mail_handoff_db_path) as con:
        link_status_counts = {
            "ready": 0,
            "manual_review": 0,
            "excluded": 0,
        }
        link_status_counts.update(
            {
                str(row["link_status"]): int(row["count"])
                for row in con.execute(
                    """
                    SELECT link_status, COUNT(*) AS count
                    FROM mail_customer_links
                    GROUP BY link_status
                    """
                )
            }
        )
        total_links = int(con.execute("SELECT COUNT(*) FROM mail_customer_links").fetchone()[0])
        candidate_summaries: dict[str, dict[str, Any]] = {}
        anomalies = 0
        for row in con.execute(
            """
            SELECT message_sha256, message_date_iso, candidate_keys_json
            FROM mail_customer_links
            WHERE match_class = 'strong_unique'
            ORDER BY message_date_iso, message_sha256
            """
        ):
            try:
                candidate_keys = json.loads(row["candidate_keys_json"] or "[]")
            except json.JSONDecodeError:
                candidate_keys = []
            if len(candidate_keys) != 1:
                anomalies += 1
                continue
            candidate_key = clean_text(candidate_keys[0])
            if not candidate_key:
                anomalies += 1
                continue
            summary = candidate_summaries.setdefault(
                candidate_key,
                {
                    "mail_message_count": 0,
                    "first_mail_date_iso": "",
                    "last_mail_date_iso": "",
                    "message_sha256_refs": [],
                },
            )
            message_date = clean_text(row["message_date_iso"])
            summary["mail_message_count"] += 1
            if message_date and (
                not summary["first_mail_date_iso"] or message_date < summary["first_mail_date_iso"]
            ):
                summary["first_mail_date_iso"] = message_date
            if message_date and (
                not summary["last_mail_date_iso"] or message_date > summary["last_mail_date_iso"]
            ):
                summary["last_mail_date_iso"] = message_date
            summary["message_sha256_refs"].append(clean_text(row["message_sha256"]))
    return {
        "candidate_summaries": candidate_summaries,
        "link_status_counts": link_status_counts,
        "total_links": total_links,
        "strong_candidate_key_anomalies": anomalies,
    }


def load_identity_candidate_refs(identity_db_path: Path) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    with open_sqlite_readonly(identity_db_path) as con:
        for row in con.execute(
            """
            SELECT candidate_key, tallanto_id, amocrm_id
            FROM identity_candidates
            ORDER BY candidate_key
            """
        ):
            refs[row["candidate_key"]] = {
                "tallanto_id": clean_text(row["tallanto_id"]),
                "amocrm_id": clean_text(row["amocrm_id"]),
            }
    return refs


def load_candidate_phone_links(identity_db_path: Path) -> dict[str, list[dict[str, Any]]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    with open_sqlite_readonly(identity_db_path) as con:
        for row in con.execute(
            """
            SELECT il.candidate_key, il.value, iv.match_class, iv.candidate_count
            FROM identity_links il
            JOIN identity_values iv
              ON iv.kind = il.kind AND iv.value = il.value
            WHERE il.kind = 'phone'
            ORDER BY il.candidate_key, il.value
            """
        ):
            by_candidate.setdefault(row["candidate_key"], []).append(
                {
                    "normalized_phone": clean_text(row["value"]),
                    "match_class": clean_text(row["match_class"]),
                    "candidate_count": int(row["candidate_count"]),
                }
            )
    return by_candidate


def load_mango_calls_by_phone(product_db_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    by_phone: dict[str, dict[str, dict[str, Any]]] = {}
    counts = {
        "capture_rows_seen": 0,
        "capture_rows_with_phone": 0,
        "capture_rows_with_normalized_phone": 0,
        "distinct_normalized_phones": 0,
        "product_calls_seen": 0,
        "product_calls_with_filename_phone": 0,
        "distinct_product_filename_phones": 0,
        "capture_rows_joined_to_product_calls": 0,
    }
    with open_sqlite_readonly(product_db_path) as con:
        counts["product_calls_seen"] = int(con.execute("SELECT COUNT(*) FROM product_calls").fetchone()[0])
        rows = list(
            con.execute(
                """
                SELECT
                  c.tenant_id,
                  c.provider,
                  c.event_key,
                  c.provider_call_id,
                  c.started_at,
                  c.direction,
                  c.client_phone,
                  c.manager_ref,
                  c.recording_ref,
                  c.status,
                  pc.started_at AS product_call_started_at,
                  pc.duration_sec,
                  pc.manager_extension,
                  pc.crm_owner_id,
                  pc.crm_match_status
                FROM capture_inbox_items c
                LEFT JOIN product_calls pc
                  ON pc.tenant_id = c.tenant_id
                 AND pc.telephony_provider = c.provider
                 AND (
                   pc.event_key = c.event_key
                   OR pc.provider_call_id = c.provider_call_id
                 )
                ORDER BY c.started_at, c.event_key, c.provider_call_id
                """
            )
        )
        product_rows = list(
            con.execute(
                """
                SELECT
                  tenant_id,
                  telephony_provider AS provider,
                  event_key,
                  provider_call_id,
                  recording_id,
                  source_filename,
                  started_at,
                  duration_sec,
                  manager_extension,
                  crm_owner_id,
                  crm_match_status
                FROM product_calls
                ORDER BY started_at, event_key, provider_call_id
                """
            )
        )
    counts["capture_rows_seen"] = len(rows)
    seen_capture_keys: set[tuple[str, str, str]] = set()
    normalized_capture_keys: set[tuple[str, str, str]] = set()
    joined_capture_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        capture_key = (
            clean_text(row["tenant_id"]),
            clean_text(row["provider"]),
            clean_text(row["event_key"]),
        )
        if capture_key not in seen_capture_keys:
            seen_capture_keys.add(capture_key)
            if clean_text(row["client_phone"]):
                counts["capture_rows_with_phone"] += 1
        normalized_phone = normalize_phone(row["client_phone"])
        if not normalized_phone:
            continue
        if capture_key not in normalized_capture_keys:
            normalized_capture_keys.add(capture_key)
            counts["capture_rows_with_normalized_phone"] += 1
        call_ref_key = "|".join(
            [
                clean_text(row["tenant_id"]),
                clean_text(row["provider"]),
                clean_text(row["event_key"]),
                clean_text(row["provider_call_id"]),
            ]
        )
        if clean_text(row["product_call_started_at"]):
            joined_capture_keys.add(capture_key)
        by_phone.setdefault(normalized_phone, {})
        by_phone[normalized_phone].setdefault(
            call_ref_key,
            {
                "source_table": "capture_inbox_items",
                "normalized_phone": normalized_phone,
                "call_ref_key": call_ref_key,
                "tenant_id": clean_text(row["tenant_id"]),
                "provider": clean_text(row["provider"]),
                "event_key": clean_text(row["event_key"]),
                "provider_call_id": clean_text(row["provider_call_id"]),
                "started_at": clean_text(row["started_at"]),
                "direction": clean_text(row["direction"]),
                "manager_ref": clean_text(row["manager_ref"]),
                "recording_ref": clean_text(row["recording_ref"]),
                "status": clean_text(row["status"]),
                "product_call_started_at": clean_text(row["product_call_started_at"]),
                "duration_sec": row["duration_sec"],
                "manager_extension": clean_text(row["manager_extension"]),
                "crm_owner_id": row["crm_owner_id"],
                "crm_match_status": clean_text(row["crm_match_status"]),
            },
        )
    product_filename_phones: set[str] = set()
    for row in product_rows:
        filename_meta = parse_filename_metadata(clean_text(row["source_filename"]))
        normalized_phone = normalize_phone(filename_meta.get("phone"))
        if not normalized_phone:
            continue
        counts["product_calls_with_filename_phone"] += 1
        product_filename_phones.add(normalized_phone)
        call_ref_key = "|".join(
            [
                clean_text(row["tenant_id"]),
                clean_text(row["provider"]),
                clean_text(row["event_key"]),
                clean_text(row["provider_call_id"]),
            ]
        )
        by_phone.setdefault(normalized_phone, {})
        by_phone[normalized_phone].setdefault(
            call_ref_key,
            {
                "source_table": "product_calls_filename",
                "normalized_phone": normalized_phone,
                "call_ref_key": call_ref_key,
                "tenant_id": clean_text(row["tenant_id"]),
                "provider": clean_text(row["provider"]),
                "event_key": clean_text(row["event_key"]),
                "provider_call_id": clean_text(row["provider_call_id"]),
                "started_at": clean_text(row["started_at"]),
                "direction": "",
                "manager_ref": clean_text(row["manager_extension"]),
                "recording_ref": clean_text(row["recording_id"]),
                "status": "",
                "product_call_started_at": clean_text(row["started_at"]),
                "duration_sec": row["duration_sec"],
                "manager_extension": clean_text(row["manager_extension"]),
                "crm_owner_id": row["crm_owner_id"],
                "crm_match_status": clean_text(row["crm_match_status"]),
            },
        )
    counts["distinct_normalized_phones"] = len(by_phone)
    counts["distinct_product_filename_phones"] = len(product_filename_phones)
    counts["capture_rows_joined_to_product_calls"] = len(joined_capture_keys)
    return (
        {
            phone: sorted(
                calls.values(),
                key=lambda item: (clean_text(item.get("started_at")), clean_text(item.get("call_ref_key"))),
            )
            for phone, calls in by_phone.items()
        },
        counts,
    )


def mango_phone_index_row_from_call(
    *,
    phone: str,
    call: Mapping[str, Any],
    source_kind: str,
    source_path_sha256: str,
    source_filename_sha256: str,
    source_root_index: Optional[int],
) -> dict[str, Any]:
    normalized_phone = normalize_phone(phone)
    call_ref_key = clean_text(call.get("call_ref_key"))
    if not call_ref_key:
        call_ref_key = hashlib.sha256(
            json.dumps(call, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return {
        "normalized_phone": normalized_phone,
        "phone_sha256": hashlib.sha256(normalized_phone.encode("utf-8")).hexdigest(),
        "call_ref_key": call_ref_key,
        "source_kind": clean_text(source_kind),
        "tenant_id": clean_text(call.get("tenant_id")) or "local_mango_archive",
        "provider": clean_text(call.get("provider")) or "mango",
        "event_key": clean_text(call.get("event_key")) or call_ref_key,
        "provider_call_id": clean_text(call.get("provider_call_id")) or call_ref_key,
        "started_at": clean_text(call.get("started_at")),
        "direction": clean_text(call.get("direction")),
        "manager_ref": clean_text(call.get("manager_ref")),
        "recording_ref": clean_text(call.get("recording_ref")),
        "status": clean_text(call.get("status")),
        "product_call_started_at": clean_text(call.get("product_call_started_at")),
        "duration_sec": call.get("duration_sec"),
        "manager_extension": clean_text(call.get("manager_extension")),
        "crm_owner_id": call.get("crm_owner_id"),
        "crm_match_status": clean_text(call.get("crm_match_status")),
        "source_path_sha256": clean_text(source_path_sha256),
        "source_filename_sha256": clean_text(source_filename_sha256),
        "source_root_index": source_root_index,
    }


def scan_recording_filename_phone_refs(recording_roots: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    files_seen = 0
    files_with_filename_phone = 0
    distinct_filename_phones: set[str] = set()
    for root_index, root in enumerate(recording_roots):
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                continue
            files_seen += 1
            metadata = parse_filename_metadata(path.name)
            normalized_phone = normalize_phone(metadata.get("phone"))
            if not normalized_phone:
                continue
            files_with_filename_phone += 1
            distinct_filename_phones.add(normalized_phone)
            started_at = metadata.get("started_at")
            if isinstance(started_at, datetime):
                started_at_text = started_at.isoformat()
            else:
                started_at_text = clean_text(started_at)
            filename_sha256 = hashlib.sha256(path.name.encode("utf-8")).hexdigest()
            path_sha256 = hashlib.sha256(str(path.resolve(strict=False)).encode("utf-8")).hexdigest()
            rows.append(
                mango_phone_index_row_from_call(
                    phone=normalized_phone,
                    call={
                        "call_ref_key": f"recording_filename|{filename_sha256}",
                        "tenant_id": "local_mango_archive",
                        "provider": "mango",
                        "event_key": filename_sha256,
                        "provider_call_id": clean_text(metadata.get("source_call_id")) or filename_sha256,
                        "started_at": started_at_text,
                        "direction": "",
                        "manager_ref": clean_text(metadata.get("manager_name")),
                        "recording_ref": f"filename_sha256:{filename_sha256}",
                        "status": "filename_indexed",
                        "product_call_started_at": started_at_text,
                        "duration_sec": None,
                        "manager_extension": "",
                        "crm_owner_id": None,
                        "crm_match_status": "",
                    },
                    source_kind="recording_filename",
                    source_path_sha256=path_sha256,
                    source_filename_sha256=filename_sha256,
                    source_root_index=root_index,
                )
            )
    return {
        "rows": rows,
        "files_seen": files_seen,
        "files_with_filename_phone": files_with_filename_phone,
        "distinct_filename_phones": len(distinct_filename_phones),
    }


def load_mango_phone_index_calls_by_phone(
    index_db_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    by_phone: dict[str, dict[str, dict[str, Any]]] = {}
    with open_sqlite_readonly(index_db_path) as con:
        rows = list(
            con.execute(
                """
                SELECT
                  normalized_phone,
                  call_ref_key,
                  source_kind,
                  tenant_id,
                  provider,
                  event_key,
                  provider_call_id,
                  started_at,
                  direction,
                  manager_ref,
                  recording_ref,
                  status,
                  product_call_started_at,
                  duration_sec,
                  manager_extension,
                  crm_owner_id,
                  crm_match_status
                FROM mango_phone_call_refs
                ORDER BY started_at, call_ref_key
                """
            )
        )
    for row in rows:
        normalized_phone = clean_text(row["normalized_phone"])
        if not normalized_phone:
            continue
        call_ref_key = clean_text(row["call_ref_key"])
        by_phone.setdefault(normalized_phone, {})
        by_phone[normalized_phone].setdefault(
            call_ref_key,
            {
                "source_table": clean_text(row["source_kind"]),
                "normalized_phone": normalized_phone,
                "call_ref_key": call_ref_key,
                "tenant_id": clean_text(row["tenant_id"]),
                "provider": clean_text(row["provider"]),
                "event_key": clean_text(row["event_key"]),
                "provider_call_id": clean_text(row["provider_call_id"]),
                "started_at": clean_text(row["started_at"]),
                "direction": clean_text(row["direction"]),
                "manager_ref": clean_text(row["manager_ref"]),
                "recording_ref": clean_text(row["recording_ref"]),
                "status": clean_text(row["status"]),
                "product_call_started_at": clean_text(row["product_call_started_at"]),
                "duration_sec": row["duration_sec"],
                "manager_extension": clean_text(row["manager_extension"]),
                "crm_owner_id": row["crm_owner_id"],
                "crm_match_status": clean_text(row["crm_match_status"]),
            },
        )
    return (
        {
            phone: sorted(
                calls.values(),
                key=lambda item: (clean_text(item.get("started_at")), clean_text(item.get("call_ref_key"))),
            )
            for phone, calls in by_phone.items()
        },
        {
            "call_refs_loaded": len(rows),
            "distinct_normalized_phones": len(by_phone),
        },
    )


def merge_mango_calls_by_phone(
    base: Mapping[str, Sequence[Mapping[str, Any]]],
    extra: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for source in (base, extra):
        for phone, calls in source.items():
            for call in calls:
                normalized_phone = normalize_phone(phone) or clean_text(phone)
                call_ref_key = clean_text(call.get("call_ref_key"))
                if not normalized_phone or not call_ref_key:
                    continue
                merged.setdefault(normalized_phone, {})
                merged[normalized_phone].setdefault(call_ref_key, dict(call))
    base_keys = {
        (normalize_phone(phone) or clean_text(phone), clean_text(call.get("call_ref_key")))
        for phone, calls in base.items()
        for call in calls
        if clean_text(call.get("call_ref_key"))
    }
    merged_keys = {
        (phone, call_ref_key)
        for phone, calls in merged.items()
        for call_ref_key in calls
    }
    return (
        {
            phone: sorted(
                calls.values(),
                key=lambda item: (clean_text(item.get("started_at")), clean_text(item.get("call_ref_key"))),
            )
            for phone, calls in merged.items()
        },
        len(merged_keys - base_keys),
    )


def classify_message_against_identity(
    *,
    message_kind: str,
    participants: Sequence[Mapping[str, Any]],
    identity_index: Mapping[str, Sequence[str]],
    internal_domains: set[str],
) -> dict[str, Any]:
    external_emails = [
        clean_text(row["email_normalized"])
        for row in participants
        if row["email_normalized"]
        and not email_domain_is_internal(clean_text(row["email_normalized"]), internal_domains)
    ]
    external_emails = unique_preserving_order(external_emails)
    if message_kind in {"internal", "service"}:
        return {
            "match_class": "internal_or_service",
            "candidate_keys": [],
            "matched_email_count": 0,
        }
    candidate_keys = unique_preserving_order(
        candidate_key
        for address in external_emails
        for candidate_key in identity_index.get(address, [])
    )
    if len(candidate_keys) == 1:
        match_class = "strong_unique"
    elif len(candidate_keys) > 1:
        match_class = "ambiguous"
    else:
        match_class = "missing"
    return {
        "match_class": match_class,
        "candidate_keys": candidate_keys,
        "matched_email_count": sum(1 for address in external_emails if address in identity_index),
    }


def link_status_for_match_class(match_class: str) -> str:
    if match_class == "strong_unique":
        return "ready"
    if match_class in {"ambiguous", "missing"}:
        return "manual_review"
    return "excluded"


def blocked_reason_for_match_class(match_class: str) -> str:
    return {
        "ambiguous": "ambiguous_identity_match",
        "missing": "missing_identity_match",
        "internal_or_service": "internal_or_service_message",
    }.get(match_class, "")


def raw_message_path(raw_dir: Path, raw_sha256: str) -> Path:
    return raw_dir / raw_sha256[:2] / f"{raw_sha256}.eml"


def write_bytes_once(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_bytes(payload)
    return True


def write_text_once(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_text(text, encoding="utf-8")
    return True


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_sqlite_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{db_path}{suffix}")
        try:
            if sidecar.exists():
                sidecar.unlink()
        except OSError:
            pass


def parse_email_date_iso(date_header: str) -> str:
    if not date_header:
        return ""
    try:
        parsed = parsedate_to_datetime(date_header)
    except Exception:  # noqa: BLE001
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def candidate_key_for_row(
    row_number: int,
    row: Mapping[str, Any],
    *,
    tallanto_id_counts: Mapping[str, int],
) -> str:
    tallanto_id = clean_text(row.get("ID"))
    if tallanto_id:
        if int(tallanto_id_counts.get(tallanto_id) or 0) > 1:
            return f"tallanto:{tallanto_id}:row:{row_number}"
        return f"tallanto:{tallanto_id}"
    digest = hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"row:{row_number}:{digest}"


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def shell_safe_token(value: str) -> str:
    text = clean_text(value)
    if not text:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_./:@=+-]+", text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def valid_env_var_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", clean_text(value)))


def unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_domains(domains: Sequence[str], mailbox_email: str = "") -> set[str]:
    result = {
        domain.casefold().strip().lstrip("@")
        for domain in domains
        if clean_text(domain)
    }
    mailbox_domain = domain_from_email(normalize_email(mailbox_email))
    if mailbox_domain:
        result.add(mailbox_domain)
    return result


def domain_from_email(address: str) -> str:
    normalized = normalize_email(address)
    if "@" not in normalized:
        return ""
    return normalized.rsplit("@", 1)[1]


def email_domain_is_internal(address: str, internal_domains: set[str]) -> bool:
    domain = domain_from_email(address)
    return bool(domain and domain in internal_domains)


def is_service_email(address: str) -> bool:
    normalized = normalize_email(address)
    if "@" not in normalized:
        return False
    local = normalized.split("@", 1)[0]
    return any(token in local for token in SERVICE_LOCAL_PARTS)


def guard_not_stable_runtime(path: Path, label: str) -> None:
    parts = {part.casefold() for part in path.resolve(strict=False).parts}
    if "stable_runtime" in parts:
        raise ValueError(f"{label} must not be under stable_runtime")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "FULL_MESSAGE_FETCH_QUERY",
    "MAIL_ARCHIVE_SCHEMA_VERSION",
    "MAIL_MANGO_BRIDGE_PREVIEW_SCHEMA_VERSION",
    "MAIL_MATCHING_REPORT_SCHEMA_VERSION",
    "MAIL_PHONE_LIFT_PREVIEW_SCHEMA_VERSION",
    "MANGO_PHONE_INDEX_PREVIEW_SCHEMA_VERSION",
    "TALLANTO_IDENTITY_MAP_SCHEMA_VERSION",
    "MailArchiveIngestConfig",
    "MailArchivePreflightConfig",
    "MailArchiveVerificationConfig",
    "MailCustomerHistoryHandoffConfig",
    "MailMangoBridgePreviewConfig",
    "MailMatchingReportConfig",
    "MailPhoneLiftPreviewConfig",
    "MangoPhoneIndexPreviewConfig",
    "TallantoIdentityMapConfig",
    "build_mail_archive_ingest",
    "build_mail_archive_preflight",
    "build_mail_customer_history_handoff",
    "build_mail_mango_bridge_preview",
    "build_mail_matching_report",
    "build_mail_phone_lift_preview",
    "build_mango_phone_index_preview",
    "build_tallanto_identity_map",
    "extract_email_addresses",
    "extract_phone_numbers",
    "normalize_email",
    "normalize_phone",
    "valid_env_var_name",
    "verify_mail_archive_pilot",
]
