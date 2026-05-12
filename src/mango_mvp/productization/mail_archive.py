from __future__ import annotations

import csv
import email
import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
class MailMatchingReportConfig:
    archive_db_path: Path
    identity_db_path: Path
    out_dir: Path
    mailbox_email: str = ""
    internal_domains: Sequence[str] = DEFAULT_INTERNAL_DOMAINS


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


def build_mail_archive_ingest(
    *,
    credentials: MailImapCredentials,
    config: MailArchiveIngestConfig,
    client: Optional[ImapClient] = None,
) -> Mapping[str, Any]:
    guard_not_stable_runtime(config.out_dir, "mail archive output directory")
    out_dir = config.out_dir.resolve(strict=False)
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
            "open_attachments": False,
            "password_written": False,
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
        selected_ids = message_ids[-max_messages:] if max_messages else message_ids
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
            PRAGMA journal_mode=WAL;
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
    for classes in row_identity_classes.values():
        row_counts["email"][classes["email"]] += 1
        row_counts["phone"][classes["phone"]] += 1

    return {
        "schema_version": TALLANTO_IDENTITY_MAP_SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_path": str(source_path),
        "row_count": len(candidate_rows),
        "duplicate_tallanto_id_values": duplicate_tallanto_id_values,
        "identity_values": values_by_kind,
        "row_identity_classes": row_counts,
        "columns_used": {
            "emails": list(config.email_columns),
            "phones": list(config.phone_columns),
            "candidate": list(config.candidate_columns),
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
    with sqlite3.connect(str(identity_db_path)) as con:
        con.row_factory = sqlite3.Row
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
    "MAIL_MATCHING_REPORT_SCHEMA_VERSION",
    "TALLANTO_IDENTITY_MAP_SCHEMA_VERSION",
    "MailArchiveIngestConfig",
    "MailMatchingReportConfig",
    "TallantoIdentityMapConfig",
    "build_mail_archive_ingest",
    "build_mail_matching_report",
    "build_tallanto_identity_map",
    "extract_email_addresses",
    "extract_phone_numbers",
    "normalize_email",
    "normalize_phone",
]
