#!/usr/bin/env python3
"""Import a local WhatsApp txt export into the customer timeline.

The script is read-only by default. Use --apply to write the local timeline DB.
It does not call AMO, Tallanto, Wappi, ASR, LLM, or any network API.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast
from urllib.parse import quote

from mango_mvp.customer_timeline.contracts import (
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
)
from mango_mvp.customer_timeline.ids import normalize_key, stable_digest
from mango_mvp.customer_timeline.import_cli import safety_ok
from mango_mvp.customer_timeline.ingestion import (
    ChannelMessageNormalizer,
    TimelineImportReport,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineSourceRecord,
    build_source_inventory,
    guard_customer_timeline_source_path,
    timeline_ingestion_safety_contract,
)
from mango_mvp.customer_timeline.safety import (
    blocked_live_actions,
    guard_customer_timeline_output_path,
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    guard_customer_timeline_sqlite_path,
)
from mango_mvp.utils.phone import normalize_phone


WHATSAPP_TIMELINE_IMPORT_SCHEMA_VERSION = "whatsapp_timeline_import_v1"
SOURCE_SYSTEM = "channel_snapshot"
CHANNEL = "whatsapp"
CHAT_HDR = re.compile(r"^===== CHAT: (.+?) =====$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
SKIP_RE = re.compile(r"Not supported WhatsApp internal message")


@dataclass(frozen=True)
class WhatsAppImportConfig:
    source: Path
    timeline_db: Path
    allowed_root: Path
    tenant_id: str
    brand: str = "unpk"
    apply: bool = False
    actor: str = "whatsapp_timeline_import"

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        source = guard_customer_timeline_source_path(Path(self.source).expanduser(), root)
        timeline_db = guard_customer_timeline_sqlite_path(Path(self.timeline_db).expanduser())
        timeline_db = guard_customer_timeline_output_path(timeline_db, root)
        if source == timeline_db:
            raise ValueError("source path and timeline DB path must be different")
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timeline_db", timeline_db)
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "brand", normalize_brand(self.brand))


@dataclass(frozen=True)
class WhatsAppParsedMessage:
    chat_name: str
    chat_key: str
    chat_phone: Optional[str]
    date: str
    time: str
    ordinal_in_minute: int
    sender: str
    text: str
    brand: str

    @property
    def direction(self) -> str:
        return "outbound" if self.sender == "You" else "inbound"

    @property
    def channel_message_id(self) -> str:
        return f"{self.chat_key}:{self.date}T{self.time}:{self.ordinal_in_minute}"

    @property
    def source_id(self) -> str:
        return f"{CHANNEL}:{self.channel_message_id}"

    @property
    def received_at(self) -> str:
        return f"{self.date}T{self.time}:00+00:00"

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "channel": CHANNEL,
            "channel_shared": True,
            "channel_thread_id": self.chat_key,
            "channel_message_id": self.channel_message_id,
            "channel_user_id": self.chat_phone or self.chat_key,
            "received_at": self.received_at,
            "direction": self.direction,
            "display_name": self.chat_name,
            "text": self.text,
            "brand_hint": self.brand,
            "chat_name": self.chat_name,
            "chat_key": self.chat_key,
            "chat_phone": self.chat_phone,
            "sender": self.sender,
            "linked_by_phone": bool(self.chat_phone),
        }


@dataclass
class WhatsAppParseStats:
    chats_seen: int = 0
    messages_seen: int = 0
    records_built: int = 0
    skipped_service: int = 0
    skipped_empty: int = 0
    skipped_malformed: int = 0
    linked_by_phone: int = 0
    session_only: int = 0
    chat_names: set[str] = field(default_factory=set)
    phone_chats: set[str] = field(default_factory=set)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "chats_seen": self.chats_seen,
            "unique_chats": len(self.chat_names),
            "messages_seen": self.messages_seen,
            "records_built": self.records_built,
            "skipped_service": self.skipped_service,
            "skipped_empty": self.skipped_empty,
            "skipped_malformed": self.skipped_malformed,
            "linked_by_phone": self.linked_by_phone,
            "session_only": self.session_only,
            "chats_linked_by_phone": len(self.phone_chats),
            "chats_session_only": len(self.chat_names - self.phone_chats),
        }


@dataclass(frozen=True)
class PhoneCustomerLookup:
    unique_customer_ids: Mapping[str, str]
    ambiguous_customer_ids: Mapping[str, tuple[str, ...]]

    @property
    def ambiguous_phone_matches(self) -> int:
        return len(self.ambiguous_customer_ids)


class WhatsAppTimelineNormalizer:
    source_system = SOURCE_SYSTEM

    def __init__(
        self,
        *,
        tenant_id: str,
        phone_customer_ids: Mapping[str, str] | None = None,
        ambiguous_phone_customer_ids: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")
        self._channel_normalizer = ChannelMessageNormalizer(tenant_id=self.tenant_id)
        self._phone_customer_ids = dict(phone_customer_ids or {})
        self._ambiguous_phone_customer_ids = {
            str(phone): tuple(sorted(str(customer_id) for customer_id in customer_ids))
            for phone, customer_ids in (ambiguous_phone_customer_ids or {}).items()
        }

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        batch = self._channel_normalizer.normalize(record)
        payload = record.payload.get("message") if isinstance(record.payload.get("message"), Mapping) else record.payload
        phone = payload.get("chat_phone")
        brand = normalize_brand(payload.get("brand_hint") or "unknown")

        customers = batch.customers
        links = batch.identity_links
        conflicts = batch.conflicts
        customer_id = batch.customers[0].customer_id if batch.customers else ""
        if phone and customers:
            phone_value = str(phone)
            base_customer = customers[0]
            existing_customer_id = self._phone_customer_ids.get(phone_value)
            ambiguous_customer_ids = self._ambiguous_phone_customer_ids.get(phone_value, ())
            event_at = batch.events[0].event_at if batch.events else base_customer.first_seen_at
            phone_evidence = {
                "linked_by_phone": True,
                "chat_name": payload.get("chat_name"),
                "source_channel": CHANNEL,
            }
            if ambiguous_customer_ids:
                customer_id = base_customer.customer_id
                customers = tuple(
                    replace(
                        item,
                        identity_status=IdentityStatus.AMBIGUOUS,
                        primary_phone=phone_value,
                        metadata={
                            **item.metadata,
                            "identity_conflict": "whatsapp_phone_ambiguous",
                            "candidate_customer_ids": list(ambiguous_customer_ids),
                        },
                    )
                    if item.customer_id == base_customer.customer_id
                    else item
                    for item in customers
                )
                links = tuple(
                    replace(
                        link,
                        customer_id=customer_id,
                        match_class=IdentityMatchClass.AMBIGUOUS,
                        confidence=0.0,
                        evidence={
                            **link.evidence,
                            "ambiguous_phone": True,
                            "candidate_customer_ids": list(ambiguous_customer_ids),
                        },
                    )
                    for link in links
                )
                links = (
                    *links,
                    IdentityLink(
                        tenant_id=self.tenant_id,
                        customer_id=customer_id,
                        link_type="whatsapp_phone",
                        link_value=phone_value,
                        source_system=self.source_system,
                        source_ref=record.source_ref,
                        match_class=IdentityMatchClass.AMBIGUOUS,
                        confidence=0.0,
                        evidence={**phone_evidence, "candidate_customer_ids": list(ambiguous_customer_ids)},
                        first_seen_at=event_at,
                        last_seen_at=event_at,
                    ),
                )
                conflicts = (
                    *conflicts,
                    {
                        "tenant_id": self.tenant_id,
                        "conflict_type": "whatsapp_phone_ambiguous",
                        "entity_refs": (
                            f"whatsapp_phone:{phone_value}",
                            record.source_ref,
                            *ambiguous_customer_ids,
                        ),
                        "severity": "medium",
                        "status": "open",
                        "summary": "WhatsApp chat phone matches multiple existing customers",
                        "metadata": {
                            "phone": phone_value,
                            "candidate_customer_ids": list(ambiguous_customer_ids),
                            "source_ref": record.source_ref,
                        },
                    },
                )
            elif existing_customer_id:
                customer_id = existing_customer_id
                customers = ()
                links = tuple(replace(link, customer_id=customer_id) for link in links)
                links = (
                    *links,
                    *self._phone_links(
                        phone_value,
                        customer_id=customer_id,
                        source_ref=record.source_ref,
                        event_at=event_at,
                        match_class=IdentityMatchClass.STRONG_UNIQUE,
                        confidence=0.9,
                        evidence=phone_evidence,
                    ),
                )
            else:
                customer_id = base_customer.customer_id
                customers = tuple(
                    replace(item, identity_status=IdentityStatus.PARTIAL, primary_phone=phone_value)
                    if item.customer_id == base_customer.customer_id
                    else item
                    for item in customers
                )
                links = tuple(replace(link, customer_id=customer_id) for link in links)
                links = (
                    *links,
                    *self._phone_links(
                        phone_value,
                        customer_id=customer_id,
                        source_ref=record.source_ref,
                        event_at=event_at,
                        match_class=IdentityMatchClass.STRONG_UNIQUE,
                        confidence=0.9,
                        evidence=phone_evidence,
                    ),
                )

        events = tuple(
            replace(
                event,
                customer_id=customer_id or event.customer_id,
                match_status=IdentityMatchClass.AMBIGUOUS
                if phone and str(phone) in self._ambiguous_phone_customer_ids
                else (
                    IdentityMatchClass.STRONG_UNIQUE
                    if phone and str(phone) in self._phone_customer_ids
                    else event.match_status
                ),
                confidence=0.0
                if phone and str(phone) in self._ambiguous_phone_customer_ids
                else (0.9 if phone and str(phone) in self._phone_customer_ids else event.confidence),
                metadata={**event.metadata, "brand": brand, "channel": CHANNEL, "channel_shared": True},
            )
            for event in batch.events
        )
        chunks = tuple(
            replace(
                chunk,
                customer_id=customer_id or chunk.customer_id,
                relevance_tags=tuple(dict.fromkeys((*chunk.relevance_tags, f"brand:{brand}", "channel_shared:true"))),
                metadata={**chunk.metadata, "brand": brand, "channel": CHANNEL, "channel_shared": True},
            )
            for chunk in batch.bot_context_chunks
        )
        return TimelineNormalizedBatch(
            source_record=batch.source_record,
            customers=customers,
            identity_links=links,
            opportunities=batch.opportunities,
            events=events,
            artifacts=batch.artifacts,
            signals=batch.signals,
            bot_context_chunks=chunks,
            conflicts=conflicts,
        )

    def _phone_links(
        self,
        phone: str,
        *,
        customer_id: str,
        source_ref: str,
        event_at: Any,
        match_class: IdentityMatchClass,
        confidence: float,
        evidence: Mapping[str, Any],
    ) -> tuple[IdentityLink, IdentityLink]:
        return (
            IdentityLink(
                tenant_id=self.tenant_id,
                customer_id=customer_id,
                link_type="phone",
                link_value=phone,
                source_system=self.source_system,
                source_ref=source_ref,
                match_class=match_class,
                confidence=confidence,
                evidence=evidence,
                first_seen_at=event_at,
                last_seen_at=event_at,
            ),
            IdentityLink(
                tenant_id=self.tenant_id,
                customer_id=customer_id,
                link_type="whatsapp_phone",
                link_value=phone,
                source_system=self.source_system,
                source_ref=source_ref,
                match_class=match_class,
                confidence=confidence,
                evidence=evidence,
                first_seen_at=event_at,
                last_seen_at=event_at,
            ),
        )


class _DryRunStore:
    """Placeholder for TimelineImportService dry-run mode.

    TimelineImportService does not touch the store when dry_run=True; using this
    object keeps dry-run from creating the target SQLite file.
    """


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        report = run_whatsapp_import(config)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - compact CLI error for operators.
        print(f"whatsapp timeline import failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a local all_whatsapp_chats.txt export into customer_timeline.sqlite. "
            "Defaults to dry-run; use --apply to write the local timeline DB."
        )
    )
    parser.add_argument("--source", required=True, help="Path to all_whatsapp_chats.txt")
    parser.add_argument("--timeline-db", required=True, help="Target local customer timeline SQLite DB")
    parser.add_argument("--allowed-root", required=True, help="Root that must contain source and timeline DB")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--brand", choices=("foton", "unpk", "unknown"), default="unpk")
    parser.add_argument("--apply", action="store_true", help="Actually write the local timeline DB")
    parser.add_argument("--actor", default="whatsapp_timeline_import")
    return parser


def config_from_args(args: argparse.Namespace) -> WhatsAppImportConfig:
    return WhatsAppImportConfig(
        source=Path(args.source),
        timeline_db=Path(args.timeline_db),
        allowed_root=Path(args.allowed_root),
        tenant_id=args.tenant_id,
        brand=args.brand,
        apply=bool(args.apply),
        actor=args.actor,
    )


def run_whatsapp_import(config: WhatsAppImportConfig) -> Mapping[str, Any]:
    text = read_text_with_fallback(config.source)
    records, parse_stats = parse_whatsapp_export_text(
        text,
        source_path=config.source,
        brand=config.brand,
    )
    phones = {str(record.payload.get("chat_phone") or "") for record in records if record.payload.get("chat_phone")}
    phone_lookup = load_phone_customer_lookup(
        config.timeline_db,
        tenant_id=config.tenant_id,
        phones=phones,
    )
    phone_customer_ids = dict(phone_lookup.unique_customer_ids)
    normalizer = WhatsAppTimelineNormalizer(
        tenant_id=config.tenant_id,
        phone_customer_ids=phone_customer_ids,
        ambiguous_phone_customer_ids=phone_lookup.ambiguous_customer_ids,
    )
    idempotency_key = stable_digest(
        {
            "tenant_id": config.tenant_id,
            "source_system": SOURCE_SYSTEM,
            "source_path": str(config.source),
            "records": [record.to_json_dict() for record in records],
        }
    )
    store_summary_before: Optional[Mapping[str, Any]] = None
    store_summary_after: Optional[Mapping[str, Any]] = None
    source_inventory_before = build_source_inventory(records)
    if config.apply:
        store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root)
        try:
            store_summary_before = store.summary()
            import_report = TimelineImportService(store).import_records(
                records,
                normalizer=normalizer,
                tenant_id=config.tenant_id,
                source_ref=config.source.name,
                idempotency_key=idempotency_key,
                dry_run=False,
                actor=config.actor,
            )
            store_summary_after = store.summary()
        finally:
            store.close()
    else:
        import_report = TimelineImportService(cast(CustomerTimelineSQLiteStore, _DryRunStore())).import_records(
            records,
            normalizer=normalizer,
            tenant_id=config.tenant_id,
            source_ref=config.source.name,
            idempotency_key=idempotency_key,
            dry_run=True,
            actor=config.actor,
        )

    source_inventory_after = build_source_inventory(records)
    validation_ok = (
        import_report.validation_ok
        and import_report.source_unchanged
        and safety_ok(timeline_ingestion_safety_contract())
    )
    return build_report(
        config=config,
        parse_stats=parse_stats,
        import_report=import_report,
        validation_ok=validation_ok,
        phone_lookup=phone_lookup,
        source_inventory_before=source_inventory_before,
        source_inventory_after=source_inventory_after,
        store_summary_before=store_summary_before,
        store_summary_after=store_summary_after,
    )


def build_report(
    *,
    config: WhatsAppImportConfig,
    parse_stats: WhatsAppParseStats,
    import_report: TimelineImportReport,
    validation_ok: bool,
    phone_lookup: PhoneCustomerLookup,
    source_inventory_before: Sequence[Mapping[str, Any]],
    source_inventory_after: Sequence[Mapping[str, Any]],
    store_summary_before: Optional[Mapping[str, Any]],
    store_summary_after: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    safety = timeline_ingestion_safety_contract()
    status = "completed_with_warnings" if parse_stats.skipped_malformed else "completed"
    return {
        "schema_version": WHATSAPP_TIMELINE_IMPORT_SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run_preview",
        "dry_run": not config.apply,
        "validation_ok": validation_ok,
        "summary": {
            "validation_ok": validation_ok,
            "status": status if validation_ok else "completed_with_warnings",
            "tenant_id": config.tenant_id,
            "brand": config.brand,
            "source_system": SOURCE_SYSTEM,
            "records_loaded": parse_stats.records_built,
            "messages_seen": parse_stats.messages_seen,
            "skipped_service": parse_stats.skipped_service,
            "skipped_empty": parse_stats.skipped_empty,
            "skipped_malformed": parse_stats.skipped_malformed,
            "linked_by_phone": parse_stats.linked_by_phone,
            "session_only": parse_stats.session_only,
            "write_applied": config.apply,
            "writes_applied": sum(int(value) for value in import_report.write_status_counts.values()) if config.apply else 0,
            "source_unchanged": import_report.source_unchanged,
            "safety_ok": safety_ok(safety),
        },
        "parser": parse_stats.to_json_dict(),
        "paths": {
            "source": str(config.source),
            "timeline_db": str(config.timeline_db),
            "allowed_root": str(config.allowed_root),
        },
        "source": {
            "inventory": {
                "before": list(source_inventory_before),
                "after": list(source_inventory_after),
                "unchanged": import_report.source_unchanged,
            },
        },
        "import_report": import_report.to_json_dict(),
        "links": {
            "unique_existing_phone_matches": len(phone_lookup.unique_customer_ids),
            "ambiguous_phone_matches": phone_lookup.ambiguous_phone_matches,
        },
        "store_summary_before": store_summary_before,
        "store_summary_after": store_summary_after,
        "safety": {
            **safety,
            "write_product_timeline_db": config.apply,
            "default_mode": "dry_run_preview",
            "requires_apply_for_db_write": True,
            "blocked_live_actions": blocked_live_actions(),
            "ok": safety_ok(safety),
        },
    }


def parse_whatsapp_export_file(
    path: Path | str,
    *,
    allowed_root: Path | str,
    brand: str = "unpk",
) -> tuple[TimelineSourceRecord, ...]:
    source_path = guard_customer_timeline_source_path(path, allowed_root)
    records, _stats = parse_whatsapp_export_text(
        read_text_with_fallback(source_path),
        source_path=source_path,
        brand=brand,
    )
    return records


def load_unique_phone_customer_ids(
    db_path: Path,
    *,
    tenant_id: str,
    phones: set[str],
) -> dict[str, str]:
    return dict(load_phone_customer_lookup(db_path, tenant_id=tenant_id, phones=phones).unique_customer_ids)


def load_phone_customer_lookup(
    db_path: Path,
    *,
    tenant_id: str,
    phones: set[str],
) -> PhoneCustomerLookup:
    if not phones or not db_path.exists():
        return PhoneCustomerLookup(unique_customer_ids={}, ambiguous_customer_ids={})
    rows = query_existing_phone_rows(db_path, tenant_id=tenant_id, phones=phones)
    by_phone: dict[str, set[str]] = {}
    for phone, customer_id in rows:
        if phone and customer_id:
            by_phone.setdefault(phone, set()).add(customer_id)
    return PhoneCustomerLookup(
        unique_customer_ids={phone: next(iter(customer_ids)) for phone, customer_ids in by_phone.items() if len(customer_ids) == 1},
        ambiguous_customer_ids={phone: tuple(sorted(customer_ids)) for phone, customer_ids in by_phone.items() if len(customer_ids) > 1},
    )


def query_existing_phone_rows(db_path: Path, *, tenant_id: str, phones: set[str]) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    with open_readonly_sqlite(db_path) as con:
        if sqlite_table_exists(con, "customer_identities"):
            for chunk in chunks(sorted(phones), 800):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    (str(row["primary_phone"]), str(row["customer_id"]))
                    for row in con.execute(
                        f"""
                        SELECT primary_phone, customer_id
                        FROM customer_identities
                        WHERE tenant_id = ? AND primary_phone IN ({placeholders})
                        """,
                        (tenant_id, *chunk),
                    )
                    if row["primary_phone"] and row["customer_id"]
                )
        if sqlite_table_exists(con, "identity_links"):
            for chunk in chunks(sorted(phones), 800):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    (str(row["link_value"]), str(row["customer_id"]))
                    for row in con.execute(
                        f"""
                        SELECT link_value, customer_id
                        FROM identity_links
                        WHERE tenant_id = ?
                          AND link_type IN ('phone', 'mango_client_phone')
                          AND link_value IN ({placeholders})
                        """,
                        (tenant_id, *chunk),
                    )
                    if row["link_value"] and row["customer_id"]
                )
    return tuple(rows)


def open_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path.resolve(strict=False)), safe='/:')}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def sqlite_table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def chunks(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(values[idx : idx + size]) for idx in range(0, len(values), size))


def parse_whatsapp_export_text(
    text: str,
    *,
    source_path: Path | str,
    brand: str = "unpk",
) -> tuple[tuple[TimelineSourceRecord, ...], WhatsAppParseStats]:
    normalized_brand = normalize_brand(brand)
    source = str(Path(source_path).resolve(strict=False))
    lines = text.splitlines()
    current_chat: Optional[str] = None
    current_date: Optional[str] = None
    minute_counts: dict[tuple[str, str, str], int] = {}
    records: list[TimelineSourceRecord] = []
    stats = WhatsAppParseStats()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        header = CHAT_HDR.match(stripped)
        if header:
            current_chat = header.group(1).strip()
            current_date = None
            stats.chats_seen += 1
            stats.chat_names.add(current_chat)
            chat_phone = normalize_chat_name_phone(current_chat)
            if chat_phone:
                stats.phone_chats.add(current_chat)
            i += 1
            continue

        if current_chat is None:
            i += 1
            continue

        if DATE_RE.match(stripped):
            current_date = stripped
            i += 1
            continue

        if current_date and TIME_RE.match(stripped):
            if i + 1 >= len(lines):
                stats.skipped_malformed += 1
                i += 1
                continue
            sender = lines[i + 1].strip()
            if not sender or is_delimiter_line(sender):
                stats.skipped_malformed += 1
                i += 1
                continue
            time_value = stripped
            i += 2
            text_lines: list[str] = []
            while i < len(lines):
                next_line = lines[i].strip()
                if is_delimiter_line(next_line):
                    break
                text_lines.append(lines[i].rstrip("\r"))
                i += 1
            stats.messages_seen += 1
            key = (current_chat, current_date, time_value)
            minute_counts[key] = minute_counts.get(key, 0) + 1
            ordinal = minute_counts[key]
            message_text = "\n".join(text_lines).strip()
            if SKIP_RE.search(message_text):
                stats.skipped_service += 1
                continue
            if not message_text:
                stats.skipped_empty += 1
                continue
            parsed = build_parsed_message(
                chat_name=current_chat,
                date=current_date,
                time_value=time_value,
                ordinal=ordinal,
                sender=sender,
                text=message_text,
                brand=normalized_brand,
            )
            records.append(
                TimelineSourceRecord(
                    source_system=SOURCE_SYSTEM,
                    source_ref=parsed.source_id,
                    payload=parsed.to_payload(),
                    source_path=source,
                )
            )
            stats.records_built += 1
            if parsed.chat_phone:
                stats.linked_by_phone += 1
            else:
                stats.session_only += 1
            continue

        i += 1
    return tuple(records), stats


def build_parsed_message(
    *,
    chat_name: str,
    date: str,
    time_value: str,
    ordinal: int,
    sender: str,
    text: str,
    brand: str,
) -> WhatsAppParsedMessage:
    phone = normalize_chat_name_phone(chat_name)
    chat_key = phone or chat_name
    return WhatsAppParsedMessage(
        chat_name=chat_name,
        chat_key=chat_key,
        chat_phone=phone,
        date=date,
        time=time_value,
        ordinal_in_minute=ordinal,
        sender=sender,
        text=text,
        brand=brand,
    )


def normalize_chat_name_phone(chat_name: str) -> Optional[str]:
    digits = re.sub(r"\D+", "", chat_name)
    if not 10 <= len(digits) <= 15:
        return None
    return normalize_phone(chat_name)


def is_delimiter_line(value: str) -> bool:
    return bool(CHAT_HDR.match(value) or DATE_RE.match(value) or TIME_RE.match(value))


def read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251")


def normalize_brand(value: Any) -> str:
    brand = normalize_key(value or "unknown", "brand")
    if brand not in {"foton", "unpk", "unknown"}:
        raise ValueError("brand must be one of: foton, unpk, unknown")
    return brand


if __name__ == "__main__":
    raise SystemExit(main())
