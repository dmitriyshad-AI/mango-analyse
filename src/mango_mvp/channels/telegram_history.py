from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import (
    ChannelAttachment,
    ChannelMessage,
    stable_digest,
)
from mango_mvp.channels.persistence import ChannelSQLiteStore, channel_sqlite_safety_contract


TELEGRAM_HISTORY_SCHEMA_VERSION = "telegram_history_archive_v1"
TELEGRAM_HISTORY_CHANNEL = "telegram_history"
TELEGRAM_IDENTITY_STRONG_UNIQUE = "strong_unique"
TELEGRAM_IDENTITY_AMBIGUOUS = "ambiguous"
TELEGRAM_IDENTITY_UNMATCHED = "unmatched"

SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "api_key",
    "apikey",
    "пароль",
    "токен",
    "ключ",
    "хэш пароля",
)


@dataclass(frozen=True)
class TelegramHistoryInventory:
    export_id: str
    export_root_name: str
    dialogs_total: int
    messages_total: int
    dialog_date_start: Optional[str]
    dialog_date_end: Optional[str]
    message_date_start: Optional[str]
    message_date_end: Optional[str]
    peer_kind_counts: Mapping[str, int]
    message_peer_kind_counts: Mapping[str, int]
    direction_counts: Mapping[str, int]
    content_counts: Mapping[str, int]
    dialog_field_presence: Mapping[str, int]
    message_field_presence: Mapping[str, int]
    identity_field_presence: Mapping[str, int]
    safety: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "peer_kind_counts", dict(self.peer_kind_counts))
        object.__setattr__(self, "message_peer_kind_counts", dict(self.message_peer_kind_counts))
        object.__setattr__(self, "direction_counts", dict(self.direction_counts))
        object.__setattr__(self, "content_counts", dict(self.content_counts))
        object.__setattr__(self, "dialog_field_presence", dict(self.dialog_field_presence))
        object.__setattr__(self, "message_field_presence", dict(self.message_field_presence))
        object.__setattr__(self, "identity_field_presence", dict(self.identity_field_presence))
        object.__setattr__(self, "safety", dict(self.safety))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True)
class TelegramHistoryImportResult:
    db_path: str
    messages_seen: int
    messages_created: int
    messages_duplicate: int
    messages_skipped_empty: int
    messages_skipped_invalid: int
    safety: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "safety", dict(self.safety))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True)
class TelegramIdentityObservation:
    channel_thread_id: str
    telegram_user_id: Optional[str] = None
    username: Optional[str] = None
    phone: Optional[str] = None
    display_name: Optional[str] = None
    source_refs: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_thread_id", require_clean_text(self.channel_thread_id, "channel_thread_id"))
        object.__setattr__(self, "telegram_user_id", optional_clean_text(self.telegram_user_id))
        object.__setattr__(self, "username", normalize_username(self.username))
        object.__setattr__(self, "phone", normalize_phone(self.phone))
        object.__setattr__(self, "display_name", optional_clean_text(self.display_name))
        object.__setattr__(self, "source_refs", tuple(str(item) for item in self.source_refs if str(item).strip()))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self, *, include_display_name: bool = False) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
            "channel_thread_id": self.channel_thread_id,
            "telegram_user_id": self.telegram_user_id,
            "username": self.username,
            "phone": self.phone,
            "display_name_present": bool(self.display_name),
            "source_refs": list(self.source_refs),
            "metadata": dict(self.metadata),
        }
        if include_display_name:
            payload["display_name"] = self.display_name
        return payload


@dataclass(frozen=True)
class CustomerIdentityRecord:
    customer_id: str
    source_system: str
    phones: Sequence[str] = field(default_factory=tuple)
    telegram_user_ids: Sequence[str] = field(default_factory=tuple)
    telegram_usernames: Sequence[str] = field(default_factory=tuple)
    names: Sequence[str] = field(default_factory=tuple)
    source_refs: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "customer_id", require_clean_text(self.customer_id, "customer_id"))
        object.__setattr__(self, "source_system", normalize_keyish(self.source_system, "source_system"))
        object.__setattr__(self, "phones", tuple(sorted({item for item in map(normalize_phone, self.phones) if item})))
        object.__setattr__(
            self,
            "telegram_user_ids",
            tuple(sorted({item for item in map(optional_clean_text, self.telegram_user_ids) if item})),
        )
        object.__setattr__(
            self,
            "telegram_usernames",
            tuple(sorted({item for item in map(normalize_username, self.telegram_usernames) if item})),
        )
        object.__setattr__(self, "names", tuple(item for item in map(optional_clean_text, self.names) if item))
        object.__setattr__(self, "source_refs", tuple(str(item) for item in self.source_refs if str(item).strip()))
        object.__setattr__(self, "metadata", scrub_sensitive_report_payload(dict(self.metadata)))

    def to_json_dict(self, *, include_names: bool = False) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
            "customer_id": self.customer_id,
            "source_system": self.source_system,
            "phones": list(self.phones),
            "telegram_user_ids": list(self.telegram_user_ids),
            "telegram_usernames": list(self.telegram_usernames),
            "names_present": bool(self.names),
            "source_refs": list(self.source_refs),
            "metadata": dict(self.metadata),
        }
        if include_names:
            payload["names"] = list(self.names)
        return payload


@dataclass(frozen=True)
class TelegramIdentityLink:
    channel_thread_id: str
    match_class: str
    candidate_customer_ids: Sequence[str] = field(default_factory=tuple)
    confidence: float = 0.0
    evidence_keys: Sequence[str] = field(default_factory=tuple)
    conflict_flags: Sequence[str] = field(default_factory=tuple)
    source_refs: Sequence[str] = field(default_factory=tuple)
    telegram_user_id: Optional[str] = None
    username: Optional[str] = None
    phone: Optional[str] = None
    display_name_present: bool = False

    def __post_init__(self) -> None:
        match_class = normalize_keyish(self.match_class, "match_class")
        if match_class not in {
            TELEGRAM_IDENTITY_STRONG_UNIQUE,
            TELEGRAM_IDENTITY_AMBIGUOUS,
            TELEGRAM_IDENTITY_UNMATCHED,
        }:
            raise ValueError(f"unsupported Telegram identity match_class: {match_class!r}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "channel_thread_id", require_clean_text(self.channel_thread_id, "channel_thread_id"))
        object.__setattr__(self, "match_class", match_class)
        object.__setattr__(self, "candidate_customer_ids", tuple(sorted(set(self.candidate_customer_ids))))
        object.__setattr__(self, "evidence_keys", tuple(sorted(set(self.evidence_keys))))
        object.__setattr__(self, "conflict_flags", tuple(sorted(set(self.conflict_flags))))
        object.__setattr__(self, "source_refs", tuple(str(item) for item in self.source_refs if str(item).strip()))
        object.__setattr__(self, "telegram_user_id", optional_clean_text(self.telegram_user_id))
        object.__setattr__(self, "username", normalize_username(self.username))
        object.__setattr__(self, "phone", normalize_phone(self.phone))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
            "channel_thread_id": self.channel_thread_id,
            "match_class": self.match_class,
            "candidate_customer_ids": list(self.candidate_customer_ids),
            "confidence": self.confidence,
            "evidence_keys": list(self.evidence_keys),
            "conflict_flags": list(self.conflict_flags),
            "source_refs": list(self.source_refs),
            "telegram_user_id": self.telegram_user_id,
            "username": self.username,
            "phone": self.phone,
            "display_name_present": self.display_name_present,
        }


def build_telegram_history_inventory(export_dir: Path | str) -> TelegramHistoryInventory:
    root = Path(export_dir)
    summary = read_json_object(root / "summary.json")
    export_id = telegram_history_export_id(summary)

    dialog_presence: Counter[str] = Counter()
    message_presence: Counter[str] = Counter()
    peer_kind_counts: Counter[str] = Counter()
    message_peer_kind_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    identity_presence: Counter[str] = Counter()
    dialogs_total = 0
    messages_total = 0
    dialog_date_start: Optional[str] = None
    dialog_date_end: Optional[str] = None
    message_date_start: Optional[str] = None
    message_date_end: Optional[str] = None

    for dialog in iter_jsonl_objects(root / "dialogs.jsonl"):
        dialogs_total += 1
        dialog_presence.update(key for key, value in dialog.items() if value not in (None, ""))
        peer_kind_counts[str(dialog.get("peer_kind") or "unknown")] += 1
        dialog_date_start, dialog_date_end = minmax_iso_text(
            optional_clean_text(dialog.get("top_message_date")),
            dialog_date_start,
            dialog_date_end,
        )
        if dialog.get("dialog_id") not in (None, ""):
            identity_presence["telegram_id"] += 1
        if optional_clean_text(dialog.get("name")):
            identity_presence["name"] += 1
        if optional_clean_text(dialog.get("username")):
            identity_presence["username"] += 1
        if optional_clean_text(dialog.get("phone")):
            identity_presence["phone"] += 1

    for message in iter_jsonl_objects(root / "messages.jsonl"):
        messages_total += 1
        message_presence.update(key for key, value in message.items() if value not in (None, ""))
        message_peer_kind_counts[str(message.get("peer_kind") or "unknown")] += 1
        direction_counts["outbound" if bool(message.get("out")) else "inbound"] += 1
        text_present = bool(optional_clean_text(message.get("text")))
        media_present = bool(message.get("has_media"))
        if text_present and media_present:
            content_counts["text_and_media"] += 1
        elif text_present:
            content_counts["text_only"] += 1
        elif media_present:
            content_counts["media_only"] += 1
        else:
            content_counts["empty_no_media"] += 1
        message_date_start, message_date_end = minmax_iso_text(
            optional_clean_text(message.get("date")),
            message_date_start,
            message_date_end,
        )

    return TelegramHistoryInventory(
        export_id=export_id,
        export_root_name=root.name,
        dialogs_total=dialogs_total,
        messages_total=messages_total,
        dialog_date_start=dialog_date_start,
        dialog_date_end=dialog_date_end,
        message_date_start=message_date_start,
        message_date_end=message_date_end,
        peer_kind_counts=dict(peer_kind_counts),
        message_peer_kind_counts=dict(message_peer_kind_counts),
        direction_counts=dict(direction_counts),
        content_counts=dict(content_counts),
        dialog_field_presence=dict(dialog_presence),
        message_field_presence=dict(message_presence),
        identity_field_presence=dict(identity_presence),
        safety=telegram_history_safety_contract(),
    )


def iter_telegram_history_messages(export_dir: Path | str) -> Iterable[ChannelMessage]:
    root = Path(export_dir)
    export_id = telegram_history_export_id(read_json_object(root / "summary.json"))
    for line_number, row in enumerate(iter_jsonl_objects(root / "messages.jsonl"), start=1):
        parsed = telegram_history_message_from_row(row, export_id=export_id, source_line=line_number)
        if parsed is not None:
            yield parsed


def telegram_history_message_from_row(
    row: Mapping[str, Any],
    *,
    export_id: str,
    source_line: int,
) -> Optional[ChannelMessage]:
    text = optional_clean_text(row.get("text")) or ""
    has_media = bool(row.get("has_media"))
    if not text and not has_media:
        return None

    dialog_id = require_clean_text(row.get("dialog_id"), "dialog_id")
    message_id = require_clean_text(row.get("message_id"), "message_id")
    sender_id = optional_clean_text(row.get("sender_id")) or dialog_id
    source_ref = f"telegram_history:{export_id}:messages.jsonl:{source_line}"
    attachments: tuple[ChannelAttachment, ...] = ()
    if has_media:
        attachments = (
            ChannelAttachment(
                kind="telegram_history_media",
                uri=f"telegram-history:media:{dialog_id}:{message_id}",
                metadata={
                    "source_ref": source_ref,
                    "media_present": True,
                    "media_payload_archived": False,
                },
            ),
        )
    return ChannelMessage(
        channel=TELEGRAM_HISTORY_CHANNEL,
        channel_message_id=message_id,
        channel_thread_id=dialog_id,
        channel_user_id=sender_id,
        direction="outbound" if bool(row.get("out")) else "inbound",
        text=text,
        received_at=parse_iso_datetime(row.get("date"), "date"),
        attachments=attachments,
        raw_payload={
            "source_ref": source_ref,
            "raw_payload_persist_allowed": False,
        },
        metadata={
            "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
            "parser_mode": "read_only",
            "source_ref": source_ref,
            "export_id": export_id,
            "telegram_dialog_id": dialog_id,
            "telegram_sender_id": sender_id,
            "telegram_message_id": message_id,
            "telegram_peer_kind": optional_clean_text(row.get("peer_kind")),
            "telegram_reply_to_msg_id": optional_clean_text(row.get("reply_to_msg_id")),
            "telegram_dialog_name_present": bool(optional_clean_text(row.get("dialog_name"))),
            "has_media": has_media,
            "raw_payload_persisted": False,
            "network_calls": False,
            "telegram_api_called": False,
            "write_crm": False,
            "write_tallanto": False,
        },
    )


def import_telegram_history_export(
    export_dir: Path | str,
    db_path: Path | str,
    *,
    limit: Optional[int] = None,
    actor: str = "telegram_history_import",
) -> TelegramHistoryImportResult:
    root = Path(export_dir)
    target = Path(db_path)
    export_id = telegram_history_export_id(read_json_object(root / "summary.json"))
    messages_seen = 0
    created = 0
    duplicates = 0
    skipped_empty = 0
    skipped_invalid = 0

    with ChannelSQLiteStore(target) as store:
        for line_number, row in enumerate(iter_jsonl_objects(root / "messages.jsonl"), start=1):
            if limit is not None and messages_seen >= limit:
                break
            messages_seen += 1
            try:
                message = telegram_history_message_from_row(row, export_id=export_id, source_line=line_number)
            except (TypeError, ValueError, KeyError):
                skipped_invalid += 1
                continue
            if message is None:
                skipped_empty += 1
                continue
            result = store.upsert_message(message, actor=actor)
            if result.created:
                created += 1
            elif result.status == "duplicate":
                duplicates += 1

    return TelegramHistoryImportResult(
        db_path=str(target),
        messages_seen=messages_seen,
        messages_created=created,
        messages_duplicate=duplicates,
        messages_skipped_empty=skipped_empty,
        messages_skipped_invalid=skipped_invalid,
        safety=telegram_history_safety_contract(),
    )


def read_telegram_dialog_identity_observations(export_dir: Path | str) -> tuple[TelegramIdentityObservation, ...]:
    root = Path(export_dir)
    export_id = telegram_history_export_id(read_json_object(root / "summary.json"))
    observations: list[TelegramIdentityObservation] = []
    for line_number, row in enumerate(iter_jsonl_objects(root / "dialogs.jsonl"), start=1):
        dialog_id = optional_clean_text(row.get("dialog_id"))
        if not dialog_id:
            continue
        observations.append(
            TelegramIdentityObservation(
                channel_thread_id=dialog_id,
                telegram_user_id=dialog_id if bool(row.get("is_user")) else None,
                username=row.get("username"),
                phone=row.get("phone"),
                display_name=row.get("name"),
                source_refs=(f"telegram_history:{export_id}:dialogs.jsonl:{line_number}",),
                metadata={
                    "peer_kind": optional_clean_text(row.get("peer_kind")),
                    "is_user": bool(row.get("is_user")),
                    "is_group": bool(row.get("is_group")),
                    "is_channel": bool(row.get("is_channel")),
                    "display_name_present": bool(optional_clean_text(row.get("name"))),
                },
            )
        )
    return tuple(observations)


def read_tallanto_identity_records(
    csv_path: Path | str,
    *,
    source_system: str = "tallanto",
    encoding: Optional[str] = None,
) -> tuple[CustomerIdentityRecord, ...]:
    path = Path(csv_path)
    text_encoding = encoding or detect_text_encoding(path)
    records: list[CustomerIdentityRecord] = []
    with path.open("r", encoding=text_encoding, newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        for row_number, row in enumerate(reader, start=2):
            tallanto_id = optional_clean_text(row.get("ID"))
            if not tallanto_id:
                continue
            telegram_user_ids = values_from_row(row, ("Telegram ID",))
            telegram_names = values_from_row(row, ("Telegram",))
            usernames = [item for item in (extract_username(value) for value in telegram_names) if item]
            records.append(
                CustomerIdentityRecord(
                    customer_id=f"tallanto:{tallanto_id}",
                    source_system=source_system,
                    phones=values_from_row(
                        row,
                        (
                            "Тел. цифровой (моб.)",
                            "Тел. цифровой (доп.)",
                            "Тел. (родителя)",
                            "Тел. (доп.)",
                            "Тел. (дом.)",
                            "Другой тел.",
                        ),
                    ),
                    telegram_user_ids=telegram_user_ids,
                    telegram_usernames=usernames,
                    names=values_from_row(row, ("ФИО родителя", "Имя", "Фамилия")),
                    source_refs=(f"tallanto_csv:{path.name}:{row_number}",),
                    metadata={
                        "telegram_subscribed": normalize_boolish(row.get("Подписан в Telegram")),
                        "has_telegram_field": bool(telegram_names),
                        "has_telegram_id": bool(telegram_user_ids),
                    },
                )
            )
    return tuple(records)


def build_telegram_identity_links(
    observations: Sequence[TelegramIdentityObservation],
    candidates: Sequence[CustomerIdentityRecord],
) -> tuple[TelegramIdentityLink, ...]:
    phone_index = build_candidate_index(candidates, "phones")
    telegram_id_index = build_candidate_index(candidates, "telegram_user_ids")
    username_index = build_candidate_index(candidates, "telegram_usernames")

    links: list[TelegramIdentityLink] = []
    for observation in observations:
        candidate_evidence: dict[str, set[str]] = defaultdict(set)
        conflict_flags: set[str] = set()

        if observation.telegram_user_id:
            matched = telegram_id_index.get(observation.telegram_user_id, set())
            add_matches(candidate_evidence, matched, "telegram_user_id")
            if len(matched) > 1:
                conflict_flags.add("telegram_user_id_conflict")
        if observation.phone:
            matched = phone_index.get(observation.phone, set())
            add_matches(candidate_evidence, matched, "phone")
            if len(matched) > 1:
                conflict_flags.add("phone_conflict")
        if observation.username:
            matched = username_index.get(observation.username, set())
            add_matches(candidate_evidence, matched, "username")
            if len(matched) > 1:
                conflict_flags.add("username_conflict")

        candidate_ids = tuple(sorted(candidate_evidence))
        evidence_keys = tuple(sorted({key for values in candidate_evidence.values() for key in values}))
        if len(candidate_ids) == 1:
            match_class = TELEGRAM_IDENTITY_STRONG_UNIQUE
            confidence = confidence_for_evidence(evidence_keys)
        elif len(candidate_ids) > 1:
            match_class = TELEGRAM_IDENTITY_AMBIGUOUS
            confidence = 0.45
            conflict_flags.add("multiple_candidate_customers")
            if evidence_keys:
                conflict_flags.add("evidence_disagreement")
        else:
            match_class = TELEGRAM_IDENTITY_UNMATCHED
            confidence = 0.0
            if observation.display_name and not any((observation.telegram_user_id, observation.username, observation.phone)):
                conflict_flags.add("name_only_not_matched")

        links.append(
            TelegramIdentityLink(
                channel_thread_id=observation.channel_thread_id,
                match_class=match_class,
                candidate_customer_ids=candidate_ids,
                confidence=confidence,
                evidence_keys=evidence_keys,
                conflict_flags=tuple(conflict_flags),
                source_refs=observation.source_refs,
                telegram_user_id=observation.telegram_user_id,
                username=observation.username,
                phone=observation.phone,
                display_name_present=bool(observation.display_name),
            )
        )
    return tuple(links)


def build_telegram_matching_report(
    links: Sequence[TelegramIdentityLink],
    *,
    high_utility_thread_ids: Sequence[str] = (),
) -> Mapping[str, Any]:
    high_utility = set(str(item) for item in high_utility_thread_ids)
    class_counts = Counter(link.match_class for link in links)
    conflict_counts = Counter(flag for link in links for flag in link.conflict_flags)
    high_utility_counts = Counter(
        link.match_class for link in links if link.channel_thread_id in high_utility
    )
    return {
        "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
        "links_total": len(links),
        "class_counts": dict(class_counts),
        "conflict_counts": dict(conflict_counts),
        "high_utility_threads_total": len(high_utility),
        "high_utility_class_counts": dict(high_utility_counts),
        "safety": telegram_history_safety_contract(),
    }


def telegram_message_timeline_event(
    message: ChannelMessage,
    *,
    identity_link: Optional[TelegramIdentityLink] = None,
    include_text_preview: bool = False,
) -> Mapping[str, Any]:
    if message.channel != TELEGRAM_HISTORY_CHANNEL:
        raise ValueError(f"expected {TELEGRAM_HISTORY_CHANNEL} message, got {message.channel!r}")
    link_payload = identity_link.to_json_dict() if identity_link else None
    return {
        "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
        "event_type": "telegram_message",
        "source_system": TELEGRAM_HISTORY_CHANNEL,
        "source_id": message.idempotency_key,
        "source_ref": message.metadata.get("source_ref"),
        "event_at": message.received_at.isoformat(),
        "direction": message.direction.value,
        "channel_thread_id": message.channel_thread_id,
        "channel_message_id": message.channel_message_id,
        "actor_ref": message.channel_user_id,
        "text_preview": message.text[:240] if include_text_preview else None,
        "text_preview_redacted": not include_text_preview,
        "attachment_count": len(message.attachments),
        "identity_link": link_payload,
        "confidence": identity_link.confidence if identity_link else 0.0,
        "conflict_flags": list(identity_link.conflict_flags) if identity_link else (),
        "safety": telegram_history_safety_contract(),
    }


def telegram_identity_link_timeline_record(link: TelegramIdentityLink) -> Mapping[str, Any]:
    return {
        "schema_version": TELEGRAM_HISTORY_SCHEMA_VERSION,
        "event_type": "telegram_identity_link",
        "source_system": TELEGRAM_HISTORY_CHANNEL,
        "channel_thread_id": link.channel_thread_id,
        "match_class": link.match_class,
        "candidate_customer_ids": list(link.candidate_customer_ids),
        "confidence": link.confidence,
        "evidence_keys": list(link.evidence_keys),
        "conflict_flags": list(link.conflict_flags),
        "source_refs": list(link.source_refs),
        "safety": telegram_history_safety_contract(),
    }


def telegram_history_safety_contract() -> Mapping[str, Any]:
    return {
        **channel_sqlite_safety_contract(),
        "network_calls": False,
        "telegram_api_called": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "imports_legacy_rag": False,
        "imports_legacy_bot_source": False,
        "report_contains_message_text": False,
    }


def scrub_sensitive_report_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            lowered = text_key.strip().lower()
            if any(part in lowered for part in SENSITIVE_KEY_PARTS):
                result[text_key] = "[redacted]"
            else:
                result[text_key] = scrub_sensitive_report_payload(item)
        return result
    if isinstance(value, (list, tuple)):
        return [scrub_sensitive_report_payload(item) for item in value]
    return value


def read_json_object(path: Path) -> Mapping[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return parsed


def iter_jsonl_objects(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield parsed


def telegram_history_export_id(summary: Mapping[str, Any]) -> str:
    payload = {
        "since": optional_clean_text(summary.get("since")),
        "total_dialogs": summary.get("total_dialogs"),
        "total_messages": summary.get("total_messages"),
        "finished_at": optional_clean_text(summary.get("finished_at")),
    }
    return stable_digest(payload)[:16]


def parse_iso_datetime(value: Any, field_name: str) -> datetime:
    text = require_clean_text(value, field_name)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def minmax_iso_text(value: Optional[str], current_min: Optional[str], current_max: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not value:
        return current_min, current_max
    next_min = value if current_min is None or value < current_min else current_min
    next_max = value if current_max is None or value > current_max else current_max
    return next_min, next_max


def build_candidate_index(candidates: Sequence[CustomerIdentityRecord], attr_name: str) -> Mapping[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for value in getattr(candidate, attr_name):
            index[str(value)].add(candidate.customer_id)
    return index


def add_matches(candidate_evidence: dict[str, set[str]], candidate_ids: Iterable[str], evidence_key: str) -> None:
    for candidate_id in candidate_ids:
        candidate_evidence[candidate_id].add(evidence_key)


def confidence_for_evidence(evidence_keys: Sequence[str]) -> float:
    keys = set(evidence_keys)
    if "phone" in keys and "telegram_user_id" in keys:
        return 0.98
    if "phone" in keys:
        return 0.96
    if "telegram_user_id" in keys:
        return 0.94
    if "username" in keys:
        return 0.78
    return 0.0


def values_from_row(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for column in columns:
        value = optional_clean_text(row.get(column))
        if value:
            values.append(value)
    return tuple(values)


def detect_text_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1251", "utf-16"):
        try:
            with path.open("r", encoding=encoding) as handle:
                handle.read(4096)
            return encoding
        except UnicodeError:
            continue
    return "utf-8-sig"


def normalize_boolish(value: Any) -> Optional[bool]:
    text = optional_clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y", "да", "истина"}:
        return True
    if lowered in {"0", "false", "no", "n", "нет", "ложь"}:
        return False
    return None


def extract_username(value: Any) -> Optional[str]:
    text = optional_clean_text(value)
    if not text:
        return None
    match = re.search(r"(?:t\.me/|telegram\.me/|@)([A-Za-z0-9_]{3,32})", text)
    if match:
        return normalize_username(match.group(1))
    if re.fullmatch(r"[A-Za-z0-9_]{3,32}", text):
        return normalize_username(text)
    return None


def normalize_username(value: Any) -> Optional[str]:
    text = optional_clean_text(value)
    if not text:
        return None
    text = text.strip().lstrip("@").lower()
    return text or None


def normalize_phone(value: Any) -> Optional[str]:
    text = optional_clean_text(value)
    if not text:
        return None
    digits = re.sub(r"\D+", "", text)
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    return f"+{digits}"


def optional_clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def require_clean_text(value: Any, field_name: str) -> str:
    text = optional_clean_text(value)
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def normalize_keyish(value: Any, field_name: str) -> str:
    text = require_clean_text(value, field_name).strip().lower()
    text = re.sub(r"[^a-z0-9_.:-]+", "_", text)
    text = text.strip("_")
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text
