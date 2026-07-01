from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable

from scripts.email_pipeline.classification import (
    ClassificationInput,
    EML_HDR_BYTES,
    RE_AUTO_SUBMITTED,
    RE_CAMPAIGN,
    RE_LIST_UNSUB,
    RE_PRECEDENCE_BULK,
    classify_message,
    local_of,
    participants_for,
)


DEFAULT_SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")
DEFAULT_PROD_TIMELINE = DEFAULT_SOURCE_ROOT / "product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"

FULL_ARCHIVES = (
    "full_60d_remaining_20260513_v2",
    "full_180_to_60d_20260513",
    "full_365_to_180d_20260513",
    "full_730_to_365d_20260513",
    "full_730_to_365d_sent_20260513",
    "full_730_to_365d_other_mailboxes_20260513",
    "full_older_than_730d_20260513",
)


@dataclass(frozen=True)
class ArchiveSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ArchiveMessage:
    message_sha256: str
    source_archive: str
    message_id: str
    in_reply_to: str
    references: str
    list_unsubscribe: bool
    precedence: str
    subject: str
    mailbox: str
    message_kind: str
    date_iso: str
    direction: str
    klass: str
    classification_reason: str
    from_email: str
    from_domain: str
    to_domains: tuple[str, ...]
    body_chars: int
    attachment_count: int
    extracted_text_path: Path | None
    raw_eml_path: Path | None


def default_archive_specs(source_root: Path = DEFAULT_SOURCE_ROOT) -> list[ArchiveSpec]:
    base = source_root / "_external_handoffs/mail_archive_2026-05-12/regru_edu"
    specs = [ArchiveSpec(name, base / name / "archive/mail_archive.sqlite") for name in FULL_ARCHIVES]
    specs.append(
        ArchiveSpec(
            "incremental_20260513_to_20260620",
            source_root
            / "_external_handoffs/mail_archive_2026-06-20/regru_edu/incremental_20260513_to_20260620/archive/mail_archive.sqlite",
        )
    )
    return specs


def existing_archive_paths(specs: Iterable[ArchiveSpec]) -> list[Path]:
    paths = [spec.path for spec in specs if spec.path.exists()]
    if not paths:
        raise FileNotFoundError("No mail archive SQLite files found")
    return paths


def resolve_data_path(value: str | None, *, source_root: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    if raw.exists():
        return raw
    text = str(value)
    markers = (
        "/Users/dmitrijfabarisov/Projects/Mango analyse/",
        "/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore/",
    )
    for marker in markers:
        if marker in text:
            suffix = text.split(marker, 1)[1]
            for root in (source_root, repo_root):
                candidate = root / suffix
                if candidate.exists():
                    return candidate
    if not raw.is_absolute():
        for root in (source_root, repo_root):
            candidate = root / raw
            if candidate.exists():
                return candidate
    return raw


def load_archive_messages(
    spec: ArchiveSpec,
    *,
    source_root: Path,
    repo_root: Path,
    outbound_templates: set[str],
) -> list[ArchiveMessage]:
    if not spec.path.exists():
        return []
    with sqlite3.connect(f"file:{spec.path}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        participants = participants_for(con)
        rows = con.execute(
            "SELECT sha256, message_id, subject, mailbox, message_kind, message_date_iso, "
            "extracted_text_chars, extracted_text_path, raw_eml_path FROM messages"
        ).fetchall()
        attachment_counts = {
            str(sha): int(count)
            for sha, count in con.execute("SELECT message_sha256, count(*) FROM attachments GROUP BY message_sha256")
        }
        output: list[ArchiveMessage] = []
        for sha, message_id, subject, mailbox, kind, date_iso, body_chars, extracted_path, eml_path in rows:
            record = participants.get(sha, {"from": None, "to": [], "cc": [], "reply_to": None})
            from_record = record.get("from") or ("", "", "")
            from_email = str(from_record[1] or "")
            from_domain = str(from_record[2] or "").lower()
            to_records = record.get("to") or []
            to_domains = tuple(str(item[2] or "").lower() for item in to_records)
            is_outbound = (
                from_domain in {"kmipt.ru"}
                or mailbox in ("Sent", "Sent Messages", "Drafts", "Templates")
                or "Шаблоны" in (mailbox or "")
            )
            resolved_eml = resolve_data_path(eml_path, source_root=source_root, repo_root=repo_root)
            eml_meta = scan_eml_metadata(resolved_eml)
            eml_flags = {"list_unsub": False, "bulk": False, "auto": False, "campaign": False}
            if not is_outbound and kind != "internal" and from_domain not in {"kmipt.ru"}:
                eml_flags = {key: bool(eml_meta.get(key)) for key in ("list_unsub", "bulk", "auto", "campaign")}
            classification_input = ClassificationInput(
                kind=kind or "",
                mailbox=mailbox or "",
                from_email=from_email,
                from_dom=from_domain,
                from_local=local_of(from_email),
                to_doms=to_domains,
                subject=subject or "",
                body_chars=int(body_chars or 0),
                eml_flags=eml_flags,
                is_outbound=is_outbound,
            )
            klass, reason = classify_message(classification_input, outbound_templates)
            output.append(
                ArchiveMessage(
                    message_sha256=sha,
                    source_archive=spec.name,
                    message_id=str(message_id or eml_meta.get("message_id") or ""),
                    in_reply_to=str(eml_meta.get("in_reply_to") or ""),
                    references=str(eml_meta.get("references") or ""),
                    list_unsubscribe=bool(eml_meta.get("list_unsub")),
                    precedence=str(eml_meta.get("precedence") or ""),
                    subject=subject or "",
                    mailbox=mailbox or "",
                    message_kind=kind or "",
                    date_iso=date_iso or "",
                    direction="outbound" if is_outbound else "inbound",
                    klass=klass,
                    classification_reason=reason,
                    from_email=from_email,
                    from_domain=from_domain,
                    to_domains=to_domains,
                    body_chars=int(body_chars or 0),
                    attachment_count=int(attachment_counts.get(str(sha), 0)),
                    extracted_text_path=resolve_data_path(extracted_path, source_root=source_root, repo_root=repo_root),
                    raw_eml_path=resolved_eml,
                )
            )
        return output


def scan_eml_metadata(path: Path | str | None) -> dict[str, object]:
    flags: dict[str, object] = {
        "list_unsub": False,
        "bulk": False,
        "auto": False,
        "campaign": False,
        "message_id": "",
        "in_reply_to": "",
        "references": "",
        "precedence": "",
    }
    if not path:
        return flags
    try:
        head = Path(path).read_bytes()[:EML_HDR_BYTES]
    except Exception:
        return flags
    sep = head.find(b"\r\n\r\n")
    if sep == -1:
        sep = head.find(b"\n\n")
    header = head[:sep] if sep != -1 else head
    flags["list_unsub"] = bool(RE_LIST_UNSUB.search(header))
    flags["bulk"] = bool(RE_PRECEDENCE_BULK.search(header))
    flags["auto"] = bool(RE_AUTO_SUBMITTED.search(header))
    flags["campaign"] = bool(RE_CAMPAIGN.search(header))
    try:
        msg = BytesParser(policy=policy.default).parsebytes(header + b"\r\n\r\n")
    except Exception:
        return flags
    flags["message_id"] = str(msg.get("Message-ID") or "").strip()
    flags["in_reply_to"] = str(msg.get("In-Reply-To") or "").strip()
    flags["references"] = str(msg.get("References") or "").strip()
    flags["precedence"] = str(msg.get("Precedence") or "").strip().lower()
    return flags


def read_text(path: Path | None, *, limit: int | None = 10000) -> str:
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if limit is None:
            return text
        return text[:limit]
    except Exception:
        return ""


def check_prod_timeline_readonly(prod_db: Path) -> dict[str, object]:
    before = prod_db.stat()
    with sqlite3.connect(f"file:{prod_db}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        email_events = int(
            con.execute("SELECT count(*) FROM timeline_events WHERE event_type='email_message'").fetchone()[0]
        )
    after = prod_db.stat()
    return {
        "path": str(prod_db),
        "quick_check": quick_check,
        "email_events": email_events,
        "mtime_before": int(before.st_mtime),
        "mtime_after": int(after.st_mtime),
        "size_before": int(before.st_size),
        "size_after": int(after.st_size),
        "mtime_unchanged": before.st_mtime == after.st_mtime,
        "size_unchanged": before.st_size == after.st_size,
    }
