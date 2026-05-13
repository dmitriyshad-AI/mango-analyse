from __future__ import annotations

import base64
import email
import imaplib
import json
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence


MAIL_IMAP_SNAPSHOT_SCHEMA_VERSION = "mail_imap_snapshot_v1"
HEADER_FETCH_QUERY = "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID DATE FROM TO CC SUBJECT)])"


class ImapClient(Protocol):
    def login(self, user: str, password: str) -> tuple[str, Sequence[bytes]]: ...

    def list(self) -> tuple[str, Sequence[bytes]]: ...

    def select(self, mailbox: str, readonly: bool = False) -> tuple[str, Sequence[bytes]]: ...

    def search(self, charset: Optional[str], *criteria: str) -> tuple[str, Sequence[bytes]]: ...

    def fetch(self, message_set: bytes | str, message_parts: str) -> tuple[str, Sequence[Any]]: ...

    def close(self) -> tuple[str, Sequence[bytes]]: ...

    def logout(self) -> tuple[str, Sequence[bytes]]: ...


@dataclass(frozen=True)
class MailImapCredentials:
    host: str
    port: int
    email_address: str
    password: str


@dataclass(frozen=True)
class MailImapSnapshotConfig:
    out_dir: Path
    since_days: int = 30
    header_sample_limit_per_mailbox: int = 10
    account_label: str = "default"


class ImapLibClient:
    def __init__(self, *, host: str, port: int, timeout: int = 30) -> None:
        context = ssl.create_default_context()
        self._imap = imaplib.IMAP4_SSL(host, port, ssl_context=context, timeout=timeout)

    def login(self, user: str, password: str) -> tuple[str, Sequence[bytes]]:
        return self._imap.login(user, password)

    def list(self) -> tuple[str, Sequence[bytes]]:
        return self._imap.list()

    def select(self, mailbox: str, readonly: bool = False) -> tuple[str, Sequence[bytes]]:
        return self._imap.select(quote_imap_mailbox_name(mailbox), readonly=readonly)

    def search(self, charset: Optional[str], *criteria: str) -> tuple[str, Sequence[bytes]]:
        return self._imap.search(charset, *criteria)

    def fetch(self, message_set: bytes | str, message_parts: str) -> tuple[str, Sequence[Any]]:
        return self._imap.fetch(message_set, message_parts)

    def close(self) -> tuple[str, Sequence[bytes]]:
        return self._imap.close()

    def logout(self) -> tuple[str, Sequence[bytes]]:
        return self._imap.logout()


def build_mail_imap_snapshot(
    *,
    credentials: MailImapCredentials,
    config: MailImapSnapshotConfig,
    client: Optional[ImapClient] = None,
) -> Mapping[str, Any]:
    """Build a read-only IMAP mailbox/header snapshot.

    The function intentionally fetches only message headers via BODY.PEEK and
    never downloads bodies, attachments, or mutates mailbox state.
    """

    out_dir = config.out_dir.resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "imap_dry_run_report.json"
    headers_path = out_dir / "imap_headers_sample.jsonl"

    since_days = max(1, int(config.since_days))
    since_dt = (datetime.now(timezone.utc) - timedelta(days=since_days)).date()
    since_imap = since_dt.strftime("%d-%b-%Y")
    header_limit = max(0, int(config.header_sample_limit_per_mailbox))

    imap = client or ImapLibClient(host=credentials.host, port=credentials.port)
    header_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": MAIL_IMAP_SNAPSHOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "account_label": config.account_label,
        "host": credentials.host,
        "port": credentials.port,
        "email": credentials.email_address,
        "since_days": since_days,
        "since_imap": since_imap,
        "safety": {
            "readonly_select": True,
            "download_bodies": False,
            "download_attachments": False,
            "send_mail": False,
            "delete_or_move_mail": False,
            "write_crm": False,
        },
        "mailboxes": [],
        "errors": [],
    }

    try:
        login_status, _ = imap.login(credentials.email_address, credentials.password)
        report["login_status"] = login_status
        list_status, boxes = imap.list()
        report["list_status"] = list_status
        if list_status != "OK":
            raise RuntimeError(f"IMAP LIST failed: {list_status}")

        parsed_boxes = [parse_mailbox_list_line(line) for line in boxes or []]
        report["mailbox_count"] = len(parsed_boxes)
        for mailbox in parsed_boxes:
            item = dict(mailbox)
            flags = set(item.get("flags") or [])
            selectable = "\\Noselect" not in flags
            item["selectable"] = selectable
            if not selectable:
                report["mailboxes"].append(item)
                continue
            try:
                selected = collect_mailbox_snapshot(
                    imap,
                    mailbox_raw=str(item["name_raw"]),
                    mailbox_name=str(item["name"]),
                    since_imap=since_imap,
                    header_sample_limit=header_limit,
                )
                item.update(selected["mailbox"])
                header_rows.extend(selected["headers"])
            except Exception as exc:  # noqa: BLE001
                item["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append({"mailbox": item.get("name"), "error": item["error"]})
            finally:
                try:
                    imap.close()
                except Exception:  # noqa: BLE001
                    pass
            report["mailboxes"].append(item)
    finally:
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass

    with headers_path.open("w", encoding="utf-8") as fh:
        for row in header_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report["header_sample_rows"] = len(header_rows)
    report["total_since"] = sum(int(row.get("since_count") or 0) for row in report["mailboxes"])
    report["total_messages"] = sum(
        int(row.get("total_messages") or 0) for row in report["mailboxes"]
    )
    report["paths"] = {
        "report": str(report_path),
        "headers_sample": str(headers_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def collect_mailbox_snapshot(
    imap: ImapClient,
    *,
    mailbox_raw: str,
    mailbox_name: str,
    since_imap: str,
    header_sample_limit: int,
) -> Mapping[str, Any]:
    select_status, data = imap.select(mailbox_raw, readonly=True)
    item: dict[str, Any] = {"select_status": select_status}
    if select_status != "OK":
        return {"mailbox": item, "headers": []}

    item["total_messages"] = int(data[0]) if data and data[0] else 0
    unseen_status, unseen_data = imap.search(None, "UNSEEN")
    item["unseen_status"] = unseen_status
    item["unseen_count"] = count_search_ids(unseen_status, unseen_data)

    since_status, since_data = imap.search(None, "SINCE", since_imap)
    ids = parse_search_ids(since_status, since_data)
    item["since_status"] = since_status
    item["since_count"] = len(ids)

    headers: list[dict[str, Any]] = []
    for msg_id in ids[-header_sample_limit:] if header_sample_limit else []:
        fetch_status, fetch_data = imap.fetch(msg_id, HEADER_FETCH_QUERY)
        if fetch_status != "OK":
            continue
        payload = first_fetch_payload(fetch_data)
        if not payload:
            continue
        row = parse_header_payload(payload)
        row.update(
            {
                "mailbox": mailbox_name,
                "mailbox_raw": mailbox_raw,
                "imap_seq": msg_id.decode("ascii", "ignore"),
            }
        )
        headers.append(row)
    return {"mailbox": item, "headers": headers}


def parse_search_ids(status: str, data: Sequence[bytes]) -> list[bytes]:
    if status != "OK" or not data:
        return []
    return list((data[0] or b"").split())


def count_search_ids(status: str, data: Sequence[bytes]) -> int:
    return len(parse_search_ids(status, data))


def first_fetch_payload(fetch_data: Sequence[Any]) -> bytes:
    for part in fetch_data or []:
        if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], bytes):
            return part[1]
    return b""


def parse_header_payload(payload: bytes) -> dict[str, str]:
    msg = email.message_from_bytes(payload)
    return {
        "message_id": clean_header(msg.get("Message-ID")),
        "date": clean_header(msg.get("Date")),
        "from": clean_header(msg.get("From")),
        "to": clean_header(msg.get("To")),
        "cc": clean_header(msg.get("Cc")),
        "subject": clean_header(msg.get("Subject")),
    }


def clean_header(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:  # noqa: BLE001
        return str(value).strip()


def parse_mailbox_list_line(line: bytes | str) -> dict[str, Any]:
    text = line.decode("utf-8", "ignore") if isinstance(line, bytes) else str(line)
    match = re.match(
        r"\((?P<flags>.*?)\)\s+\"?(?P<delimiter>[^\" ]*)\"?\s+(?P<name>.+)$",
        text,
    )
    if not match:
        return {
            "raw": text,
            "name_raw": text,
            "name": decode_imap_modified_utf7(text),
            "flags": [],
            "delimiter": None,
        }
    name_raw = match.group("name").strip()
    name = "" if name_raw.upper() == "NIL" else decode_imap_modified_utf7(unquote_imap_name(name_raw))
    flags = [flag for flag in match.group("flags").split() if flag]
    return {
        "raw": text,
        "name_raw": name_raw,
        "name": name,
        "flags": flags,
        "delimiter": match.group("delimiter"),
    }


def unquote_imap_name(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.replace(r"\"", '"')


def quote_imap_mailbox_name(value: str) -> str:
    text = value.strip()
    if text.upper() == "INBOX":
        return "INBOX"
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', r"\"") + '"'


def decode_imap_modified_utf7(value: str) -> str:
    def replace_token(match: re.Match[str]) -> str:
        token = match.group(1)
        if token == "":
            return "&"
        data = token.replace(",", "/")
        data += "=" * ((4 - len(data) % 4) % 4)
        try:
            return base64.b64decode(data).decode("utf-16-be")
        except Exception:  # noqa: BLE001
            return match.group(0)

    return re.sub(r"&([^-]*)-", replace_token, value)


__all__ = [
    "HEADER_FETCH_QUERY",
    "MAIL_IMAP_SNAPSHOT_SCHEMA_VERSION",
    "ImapClient",
    "ImapLibClient",
    "MailImapCredentials",
    "MailImapSnapshotConfig",
    "build_mail_imap_snapshot",
    "decode_imap_modified_utf7",
    "parse_mailbox_list_line",
    "quote_imap_mailbox_name",
]
