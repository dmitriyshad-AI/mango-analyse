from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.channels.telegram_history import build_telegram_history_inventory, normalize_username
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    CustomerOpportunity,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    TimelineParticipant,
)
from mango_mvp.customer_timeline.ids import normalize_key, require_text
from mango_mvp.customer_timeline.import_cli import (
    build_timeline_import_preview,
    normalized_total,
    timeline_import_cli_safety_contract,
)
from mango_mvp.customer_timeline.ingestion import (
    ChannelMessageNormalizer,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineSourceRecord,
    file_sha256,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    guard_customer_timeline_sqlite_path,
)
from mango_mvp.utils.phone import normalize_phone


TELEGRAM_EXPORT_TIMELINE_IMPORT_SCHEMA_VERSION = "telegram_export_timeline_import_v1"
SOURCE_SYSTEM = "telegram_history"
SOURCE_KIND = "channel_snapshot"
CHANNEL = "telegram"
BRANDS = {"foton", "unpk", "unknown"}


@dataclass(frozen=True)
class TelegramExportImportConfig:
    export_dir: Path
    allowed_root: Path
    timeline_db: Path
    identity_export_dir: Optional[Path] = None
    tenant_id: str = "foton"
    brand: str = "unpk"
    apply: bool = False
    out_path: Optional[Path] = None
    actor: str = "telegram_export_timeline_import"
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).resolve(strict=False)
        export_dir = guard_customer_timeline_output_path(self.export_dir, root)
        if not export_dir.exists() or not export_dir.is_dir():
            raise ValueError(f"telegram export dir must exist and be a directory: {export_dir}")
        dialogs_path = export_dir / "dialogs.jsonl"
        messages_path = export_dir / "messages.jsonl"
        for path in (dialogs_path, messages_path):
            if not path.exists() or not path.is_file():
                raise ValueError(f"telegram export must contain {path.name}: {path}")
        raw_identity_export_dir = self.identity_export_dir if self.identity_export_dir else autodetect_identity_export_dir(export_dir)
        identity_export_dir = guard_customer_timeline_output_path(raw_identity_export_dir, root) if raw_identity_export_dir else None
        if identity_export_dir is not None:
            identity_dialogs_path = identity_export_dir / "dialogs.jsonl"
            if not identity_dialogs_path.exists() or not identity_dialogs_path.is_file():
                raise ValueError(f"telegram identity export must contain dialogs.jsonl: {identity_dialogs_path}")
        timeline_db = guard_customer_timeline_output_path(guard_customer_timeline_sqlite_path(self.timeline_db), root)
        out_path = guard_customer_timeline_output_path(self.out_path, root) if self.out_path else None
        if out_path and out_path in {dialogs_path, messages_path, timeline_db}:
            raise ValueError("report output path must not overwrite source files or timeline DB")
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "export_dir", export_dir)
        object.__setattr__(self, "identity_export_dir", identity_export_dir)
        object.__setattr__(self, "timeline_db", timeline_db)
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "brand", normalize_brand(self.brand))
        object.__setattr__(self, "actor", require_text(self.actor, "actor"))
        object.__setattr__(self, "out_path", out_path)

    @property
    def dialogs_path(self) -> Path:
        return self.export_dir / "dialogs.jsonl"

    @property
    def messages_path(self) -> Path:
        return self.export_dir / "messages.jsonl"

    @property
    def identity_dialogs_path(self) -> Optional[Path]:
        if self.identity_export_dir is None:
            return None
        return self.identity_export_dir / "dialogs.jsonl"

    @property
    def source_ref(self) -> str:
        return f"telegram_export:{self.export_dir.name}:{self.brand}"


@dataclass
class TelegramBuildCounters:
    dialogs: int = 0
    messages: int = 0
    skipped: int = 0
    groups: int = 0
    linked_by_phone: int = 0
    linked_by_username: int = 0
    ambiguous_dialogs: int = 0
    unmatched_dialogs: int = 0
    dialog_events: int = 0
    session_only: int = 0
    duplicates: int = 0
    bad_jsonl_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "dialogs": self.dialogs,
            "messages": self.messages,
            "imported": 0,
            "skipped": self.skipped,
            "groups": self.groups,
            "linked_by_phone": self.linked_by_phone,
            "linked_by_username": self.linked_by_username,
            "ambiguous_dialogs": self.ambiguous_dialogs,
            "unmatched_dialogs": self.unmatched_dialogs,
            "dialog_events": self.dialog_events,
            "session_only": self.session_only,
            "duplicates": self.duplicates,
            "bad_jsonl_rows": self.bad_jsonl_rows,
        }


@dataclass(frozen=True)
class JsonlReadResult:
    rows: tuple[Mapping[str, Any], ...]
    nonempty_lines: int
    bad_rows: int


@dataclass(frozen=True)
class PhoneCustomerLookup:
    unique_customer_ids: Mapping[str, str]
    ambiguous_phone_matches: int
    candidate_customer_ids: Mapping[str, Sequence[str]] = field(default_factory=dict)
    ambiguous_phones: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unique_customer_ids", dict(self.unique_customer_ids))
        object.__setattr__(
            self,
            "candidate_customer_ids",
            {key: tuple(value) for key, value in self.candidate_customer_ids.items()},
        )
        object.__setattr__(self, "ambiguous_phones", tuple(self.ambiguous_phones))


@dataclass(frozen=True)
class UsernameCustomerLookup:
    unique_customer_ids: Mapping[str, str]
    ambiguous_usernames: Sequence[str] = field(default_factory=tuple)
    candidate_customer_ids: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unique_customer_ids", dict(self.unique_customer_ids))
        object.__setattr__(self, "ambiguous_usernames", tuple(self.ambiguous_usernames))
        object.__setattr__(
            self,
            "candidate_customer_ids",
            {key: tuple(value) for key, value in self.candidate_customer_ids.items()},
        )


@dataclass(frozen=True)
class TelegramIdentityResolution:
    match_class: IdentityMatchClass
    customer_id: Optional[str]
    candidate_customer_ids: Sequence[str]
    confidence: float
    evidence_keys: Sequence[str]
    conflict_flags: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_class", IdentityMatchClass(self.match_class))
        object.__setattr__(self, "customer_id", optional_str(self.customer_id))
        object.__setattr__(self, "candidate_customer_ids", tuple(sorted(set(self.candidate_customer_ids))))
        object.__setattr__(self, "evidence_keys", tuple(sorted(set(self.evidence_keys))))
        object.__setattr__(self, "conflict_flags", tuple(sorted(set(self.conflict_flags))))


class TelegramExportTimelineNormalizer:
    source_system = SOURCE_SYSTEM

    def __init__(
        self,
        *,
        tenant_id: str,
        phone_lookup: PhoneCustomerLookup,
        username_lookup: UsernameCustomerLookup,
    ) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")
        self._base = ChannelMessageNormalizer(tenant_id=self.tenant_id)
        self._phone_lookup = phone_lookup
        self._username_lookup = username_lookup

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        batch = self._base.normalize(record)
        payload = record.payload
        source_id = require_text(payload.get("timeline_source_id"), "timeline_source_id")
        dialog_id = require_text(payload.get("telegram_dialog_id"), "telegram_dialog_id")
        brand = normalize_brand(payload.get("brand_hint"))
        phone = normalize_phone(str(payload.get("dialog_phone_normalized") or payload.get("dialog_phone") or ""))
        username = normalize_username(payload.get("dialog_username") or payload.get("sender_username"))
        resolution = self._resolve_identity(phone=phone, username=username)

        if not batch.customers or not batch.events:
            return batch

        base_customer = batch.customers[0]
        if resolution.customer_id:
            customer_id = resolution.customer_id
            customers: tuple[CustomerIdentity, ...] = ()
        else:
            customer = replace(
                base_customer,
                customer_id=None,
                source_ref=f"{SOURCE_SYSTEM}:dialog:{dialog_id}",
                identity_status=identity_status_for_resolution(resolution),
                primary_phone=phone,
                summary={
                    **base_customer.summary,
                    "source_system": SOURCE_SYSTEM,
                    "channel": CHANNEL,
                    "brand": brand,
                    "identity_resolution": resolution.match_class.value,
                    "evidence_keys": list(resolution.evidence_keys),
                },
                metadata={
                    **base_customer.metadata,
                    "source_system": SOURCE_SYSTEM,
                    "channel": CHANNEL,
                    "thread_id": dialog_id,
                    "user_id": dialog_id,
                    "brand": brand,
                    "identity_resolution": resolution.match_class.value,
                    "conflict_flags": list(resolution.conflict_flags),
                    "has_phone": bool(phone),
                    "has_username": bool(username),
                },
            )
            customer_id = customer.customer_id
            customers = (customer,)

        base_event = batch.events[0]
        match_class = resolution.match_class
        confidence = resolution.confidence
        event = replace(
            base_event,
            event_id=None,
            customer_id=customer_id,
            source_system=SOURCE_SYSTEM,
            source_id=source_id,
            source_ref=f"{SOURCE_SYSTEM}:{dialog_id}:{payload.get('telegram_message_id')}",
            match_status=match_class,
            confidence=confidence,
            record={"message": safe_telegram_message_record(payload)},
            metadata={
                **base_event.metadata,
                "source_system": SOURCE_SYSTEM,
                "brand": brand,
                "identity_resolution": match_class.value,
                "evidence_keys": list(resolution.evidence_keys),
                "conflict_flags": list(resolution.conflict_flags),
                "telegram_dialog_id": dialog_id,
            },
        )

        links = [
            replace(
                link,
                link_id=None,
                customer_id=customer_id,
                source_system=SOURCE_SYSTEM,
                match_class=match_class,
                confidence=confidence
                if match_class == IdentityMatchClass.UNMATCHED or link.link_type.value == "telegram_user_id"
                else min(0.7, max(confidence, 0.5)),
                evidence={"evidence_keys": list(resolution.evidence_keys), "conflict_flags": list(resolution.conflict_flags)},
            )
            for link in batch.identity_links
        ]
        if username:
            links.append(
                IdentityLink(
                    tenant_id=self.tenant_id,
                    customer_id=customer_id,
                    link_type="telegram_username",
                    link_value=username,
                    source_system=SOURCE_SYSTEM,
                    source_ref=f"{SOURCE_SYSTEM}:{dialog_id}:username",
                    match_class=match_class,
                    confidence=confidence if match_class != IdentityMatchClass.UNMATCHED else 0.0,
                    evidence={"evidence_keys": list(resolution.evidence_keys), "conflict_flags": list(resolution.conflict_flags)},
                    first_seen_at=event.event_at,
                    last_seen_at=event.event_at,
                )
            )
        if phone:
            links.append(
                IdentityLink(
                    tenant_id=self.tenant_id,
                    customer_id=customer_id,
                    link_type="phone",
                    link_value=phone,
                    source_system=SOURCE_SYSTEM,
                    source_ref=f"{SOURCE_SYSTEM}:{dialog_id}:phone",
                    match_class=match_class,
                    confidence=confidence if match_class != IdentityMatchClass.UNMATCHED else 0.0,
                    evidence={"evidence_keys": list(resolution.evidence_keys), "conflict_flags": list(resolution.conflict_flags)},
                    first_seen_at=event.event_at,
                    last_seen_at=event.event_at,
                )
            )

        opportunities: tuple[CustomerOpportunity, ...] = ()
        dialog_events: tuple[TimelineEvent, ...] = ()
        opportunity_id: Optional[str] = None
        if bool(payload.get("emit_dialog_event")):
            opportunity = CustomerOpportunity(
                tenant_id=self.tenant_id,
                customer_id=customer_id,
                opportunity_type=OpportunityType.TELEGRAM_DIALOG,
                source_system=SOURCE_SYSTEM,
                source_id=f"telegram_dialog:{dialog_id}",
                title="Telegram dialog",
                status="open",
                product_context={"brand": brand, "channel": CHANNEL},
                opened_at=event.event_at,
                confidence=confidence,
                evidence={"source_ref": f"{SOURCE_SYSTEM}:dialog:{dialog_id}", "identity_resolution": match_class.value},
            )
            opportunity_id = opportunity.opportunity_id
            opportunities = (opportunity,)
            dialog_events = (
                TimelineEvent(
                    tenant_id=self.tenant_id,
                    customer_id=customer_id,
                    opportunity_id=opportunity_id,
                    event_type=TimelineEventType.TELEGRAM_DIALOG,
                    event_at=event.event_at,
                    source_system=SOURCE_SYSTEM,
                    source_id=f"telegram_dialog:{dialog_id}",
                    source_ref=f"{SOURCE_SYSTEM}:dialog:{dialog_id}",
                    direction=TimelineDirection.SYSTEM,
                    participants=(TimelineParticipant(role="client", ref=dialog_id, channel=CHANNEL),),
                    subject="Telegram dialog",
                    text_preview=None,
                    summary="Telegram dialog imported from local archive",
                    match_status=match_class,
                    confidence=confidence,
                    record={
                        "dialog": {
                            "telegram_dialog_id": dialog_id,
                            "brand": brand,
                            "has_phone": bool(phone),
                            "has_username": bool(username),
                            "identity_resolution": match_class.value,
                            "evidence_keys": list(resolution.evidence_keys),
                            "conflict_flags": list(resolution.conflict_flags),
                        }
                    },
                    metadata={
                        "source_system": SOURCE_SYSTEM,
                        "brand": brand,
                        "identity_resolution": match_class.value,
                        "candidate_customer_count": len(resolution.candidate_customer_ids),
                    },
                    created_at=event.event_at,
                ),
            )
        if opportunity_id:
            event = replace(event, opportunity_id=opportunity_id, event_id=None)

        brand_tag = f"brand:{brand}"
        chunks = tuple(
            replace(
                chunk,
                chunk_id=None,
                customer_id=customer_id,
                event_id=event.event_id,
                source_system=SOURCE_SYSTEM,
                relevance_tags=unique_tags((*chunk.relevance_tags, brand_tag)),
                metadata={
                    **chunk.metadata,
                    "brand": brand,
                    "identity_resolution": match_class.value,
                    "allowed_for_bot_reason": "telegram_history_manager_only",
                },
            )
            for chunk in batch.bot_context_chunks
        )
        conflicts = tuple(batch.conflicts)
        if match_class == IdentityMatchClass.AMBIGUOUS:
            refs = [f"{SOURCE_SYSTEM}:dialog:{dialog_id}"]
            refs.extend(f"customer:{item}" for item in resolution.candidate_customer_ids)
            conflicts = (
                *conflicts,
                {
                    "tenant_id": self.tenant_id,
                    "conflict_type": "telegram_identity_ambiguous",
                    "entity_refs": tuple(refs),
                    "severity": "medium",
                    "status": "open",
                    "summary": "Telegram dialog has ambiguous phone or username evidence",
                    "metadata": {
                        "channel": CHANNEL,
                        "thread_id": dialog_id,
                        "candidate_customer_count": len(resolution.candidate_customer_ids),
                        "evidence_keys": list(resolution.evidence_keys),
                        "conflict_flags": list(resolution.conflict_flags),
                    },
                },
            )

        return TimelineNormalizedBatch(
            source_record=batch.source_record,
            customers=customers,
            identity_links=tuple(links),
            opportunities=opportunities,
            events=(*dialog_events, event),
            artifacts=batch.artifacts,
            signals=batch.signals,
            bot_context_chunks=chunks,
            conflicts=conflicts,
        )

    def _resolve_identity(self, *, phone: Optional[str], username: Optional[str]) -> TelegramIdentityResolution:
        return resolve_telegram_identity(
            phone=phone,
            username=username,
            phone_lookup=self._phone_lookup,
            username_lookup=self._username_lookup,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        report = run_telegram_export_import(config)
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if config.out_path:
            config.out_path.parent.mkdir(parents=True, exist_ok=True)
            config.out_path.write_text(f"{text}\n", encoding="utf-8")
        else:
            print(text)
        return 0 if report["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - CLI should return a compact operator-facing error.
        print(f"telegram export timeline import failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import Telegram JSONL export into customer_timeline.sqlite through the existing "
            "ChannelMessageNormalizer and TimelineImportService. Defaults to dry-run."
        )
    )
    parser.add_argument("--export-dir", required=True, help="Directory containing dialogs.jsonl and messages.jsonl.")
    parser.add_argument(
        "--identity-export-dir",
        help=(
            "Optional directory containing a richer dialogs.jsonl with phones/usernames. "
            "Defaults to sibling *_max or *_with_contacts when present."
        ),
    )
    parser.add_argument("--allowed-root", default=".", help="Safety root for source, report and timeline DB paths.")
    parser.add_argument(
        "--timeline-db",
        help="Target timeline DB. Defaults to <allowed-root>/customer_timeline/customer_timeline.sqlite.",
    )
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--brand", choices=sorted(BRANDS), default="unpk")
    parser.add_argument("--apply", action="store_true", help="Write into the local timeline DB.")
    parser.add_argument("--out", help="Optional JSON report path. If omitted, report is printed to stdout.")
    parser.add_argument("--actor", default="telegram_export_timeline_import")
    parser.add_argument("--idempotency-key", help="Optional stable key for repeated imports.")
    return parser


def config_from_args(args: argparse.Namespace) -> TelegramExportImportConfig:
    allowed_root = Path(args.allowed_root)
    timeline_db = Path(args.timeline_db) if args.timeline_db else allowed_root / "customer_timeline" / "customer_timeline.sqlite"
    return TelegramExportImportConfig(
        export_dir=Path(args.export_dir),
        allowed_root=allowed_root,
        timeline_db=timeline_db,
        identity_export_dir=Path(args.identity_export_dir) if args.identity_export_dir else None,
        tenant_id=args.tenant_id,
        brand=args.brand,
        apply=bool(args.apply),
        out_path=Path(args.out) if args.out else None,
        actor=args.actor,
        idempotency_key=args.idempotency_key,
    )


def run_telegram_export_import(config: TelegramExportImportConfig) -> Mapping[str, Any]:
    inventory_paths = [config.dialogs_path, config.messages_path]
    if config.identity_dialogs_path is not None and config.identity_dialogs_path != config.dialogs_path:
        inventory_paths.append(config.identity_dialogs_path)
    source_inventory_before = source_file_inventory(*inventory_paths)
    dialog_read = read_jsonl_lenient(config.dialogs_path)
    identity_dialog_read: Optional[JsonlReadResult] = None
    if config.identity_dialogs_path is not None and config.identity_dialogs_path != config.dialogs_path:
        identity_dialog_read = read_jsonl_lenient(config.identity_dialogs_path)
    dialog_rows = merge_dialog_identity_rows(dialog_read.rows, identity_dialog_read.rows if identity_dialog_read else ())
    dialogs = dialogs_by_id(dialog_rows)
    message_read = read_jsonl_lenient(config.messages_path)
    messages = message_read.rows
    records, counters = build_timeline_records(
        dialogs=dialogs,
        dialog_count=dialog_read.nonempty_lines,
        messages=messages,
        message_count=message_read.nonempty_lines,
        brand=config.brand,
        source_path=config.messages_path,
    )
    counters.bad_jsonl_rows = dialog_read.bad_rows + message_read.bad_rows + (identity_dialog_read.bad_rows if identity_dialog_read else 0)
    counters.skipped += message_read.bad_rows
    source_ids = tuple(str(record.payload["timeline_source_id"]) for record in records)
    existing_source_ids = load_existing_message_source_ids(
        config.timeline_db,
        tenant_id=config.tenant_id,
        source_ids=source_ids,
    )
    existing_duplicate_count = sum(1 for source_id in source_ids if source_id in existing_source_ids)
    counters.duplicates += existing_duplicate_count

    phones = {str(record.payload.get("dialog_phone_normalized") or "") for record in records if record.payload.get("dialog_phone_normalized")}
    usernames = {str(record.payload.get("dialog_username") or "") for record in records if record.payload.get("dialog_username")}
    phone_lookup = load_phone_customer_lookup(
        config.timeline_db,
        tenant_id=config.tenant_id,
        phones=phones,
    )
    username_lookup = load_username_customer_lookup(
        config.timeline_db,
        tenant_id=config.tenant_id,
        usernames=usernames,
    )
    normalizer = TelegramExportTimelineNormalizer(
        tenant_id=config.tenant_id,
        phone_lookup=phone_lookup,
        username_lookup=username_lookup,
    )
    preview = build_timeline_import_preview(
        records,
        normalizer=normalizer,
        tenant_id=config.tenant_id,
        source_ref=config.source_ref,
        source_kind=SOURCE_KIND,
        timeline_db=config.timeline_db,
    )

    store_summary_before: Optional[Mapping[str, Any]] = None
    store_summary_after: Optional[Mapping[str, Any]] = None
    import_report: Optional[Mapping[str, Any]] = None
    report_payload: Mapping[str, Any] = preview
    mode = "dry_run_preview"
    if config.apply:
        store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root)
        try:
            store_summary_before = store.summary()
            service_report = TimelineImportService(store).import_records(
                records,
                normalizer=normalizer,
                tenant_id=config.tenant_id,
                source_ref=config.source_ref,
                idempotency_key=config.idempotency_key or preview["input_hash"],
                dry_run=False,
                actor=config.actor,
            )
            import_report = service_report.to_json_dict()
            report_payload = import_report
            store_summary_after = store.summary()
        finally:
            store.close()
        mode = "apply"

    source_inventory_after = source_file_inventory(*inventory_paths)
    source_unchanged = source_inventory_before == source_inventory_after
    accepted_count = int(report_payload["accepted_count"])
    imported = max(0, accepted_count - existing_duplicate_count)
    counter_payload = counters.to_dict()
    counter_payload["imported"] = imported
    match_counts = normalized_match_counts(records, phone_lookup=phone_lookup, username_lookup=username_lookup)
    counter_payload["ambiguous_dialogs"] = int(match_counts.get("ambiguous_dialogs", 0))
    counter_payload["unmatched_dialogs"] = int(match_counts.get("unmatched_dialogs", 0))
    history_inventory = safe_telegram_history_inventory(config.export_dir)
    safety = timeline_import_cli_safety_contract(write_product_timeline_db=config.apply)
    validation_ok = bool(report_payload["validation_ok"]) and source_unchanged
    return {
        "schema_version": TELEGRAM_EXPORT_TIMELINE_IMPORT_SCHEMA_VERSION,
        "mode": mode,
        "dry_run": not config.apply,
        "validation_ok": validation_ok,
        "counters": counter_payload,
        "summary": {
            "validation_ok": validation_ok,
            "status": "completed" if validation_ok else "completed_with_warnings",
            "dry_run": not config.apply,
            "tenant_id": config.tenant_id,
            "brand": config.brand,
            "source_system": SOURCE_SYSTEM,
            "source_ref": config.source_ref,
            "records_accepted": accepted_count,
            "records_rejected": int(report_payload["rejected_count"]),
            "normalized_total": normalized_total(report_payload["normalized_counts"]),
            "writes_applied": sum(int(value) for value in report_payload["write_status_counts"].values())
            if config.apply
            else 0,
            "source_unchanged": source_unchanged,
        },
        "source": {
            "export_dir": str(config.export_dir),
            "identity_export_dir": str(config.identity_export_dir) if config.identity_export_dir else None,
            "dialogs_path": str(config.dialogs_path),
            "messages_path": str(config.messages_path),
            "identity_dialogs_path": str(config.identity_dialogs_path) if config.identity_dialogs_path else None,
            "telegram_history_inventory": history_inventory,
            "inventory": {
                "unchanged": source_unchanged,
                "before": list(source_inventory_before),
                "after": list(source_inventory_after),
            },
        },
        "links": {
            "unique_existing_phone_matches": len(phone_lookup.unique_customer_ids),
            "ambiguous_phone_matches": phone_lookup.ambiguous_phone_matches,
            "unique_existing_username_matches": len(username_lookup.unique_customer_ids),
            "ambiguous_username_matches": len(username_lookup.ambiguous_usernames),
            "existing_duplicate_source_ids": existing_duplicate_count,
            "message_match_counts": dict(match_counts),
        },
        "normalization": {
            "schema_version": "customer_timeline_ingestion_v1",
            "counts": dict(report_payload["normalized_counts"]),
            "by_source_record": preview["by_source_record"],
        },
        "writes": {
            "target": {
                "db_path": str(config.timeline_db),
                "allowed_root": str(config.allowed_root),
                "schema_version": "customer_timeline_sqlite_v1",
            },
            "applied": config.apply,
            "planned_counts_by_type": preview["operation_plan"]["counts"],
            "status_counts": dict(report_payload["write_status_counts"]),
            "items": preview["operation_plan"]["items"],
        },
        "errors": list(report_payload["errors"]),
        "paths": {
            "allowed_root": str(config.allowed_root),
            "timeline_db": str(config.timeline_db),
            "report_out": str(config.out_path) if config.out_path else None,
        },
        "preview": preview,
        "import_report": import_report or preview,
        "store_summary_before": store_summary_before,
        "store_summary_after": store_summary_after,
        "source_unchanged": source_unchanged,
        "safety": {
            **safety,
            "ok": all(
                safety.get(action) is False
                for action in (
                    "write_crm",
                    "write_tallanto",
                    "send_email",
                    "send_messenger",
                    "run_asr",
                    "run_ra",
                    "mutate_stable_runtime",
                    "stable_runtime_writes",
                )
            ),
        },
    }


def build_timeline_records(
    *,
    dialogs: Mapping[str, Mapping[str, Any]],
    dialog_count: Optional[int] = None,
    messages: Sequence[Mapping[str, Any]],
    message_count: Optional[int] = None,
    brand: str,
    source_path: Path,
) -> tuple[tuple[TimelineSourceRecord, ...], TelegramBuildCounters]:
    counters = TelegramBuildCounters(
        dialogs=len(dialogs) if dialog_count is None else dialog_count,
        messages=len(messages) if message_count is None else message_count,
    )
    records: list[TimelineSourceRecord] = []
    seen_source_ids: set[str] = set()
    seen_dialog_ids: set[str] = set()
    dialog_resolution_hints: dict[str, str] = {}
    for msg in messages:
        dialog_id = optional_str(msg.get("dialog_id"))
        message_id = optional_str(msg.get("message_id"))
        received_at = optional_str(msg.get("date"))
        if not dialog_id or not message_id or not received_at:
            counters.skipped += 1
            continue
        dialog = dialogs.get(dialog_id, {})
        message_peer_kind = str(msg.get("peer_kind") or "").strip().lower()
        dialog_peer_kind = str(dialog.get("peer_kind") or "").strip().lower()
        if not is_user_peer(message_peer_kind, dialog_peer_kind):
            counters.groups += 1
            counters.skipped += 1
            continue
        text = str(msg.get("text") or "").strip()
        if not text:
            counters.skipped += 1
            continue
        payload = tg_message_to_payload(msg, dialog, brand)
        source_id = str(payload["timeline_source_id"])
        if source_id in seen_source_ids:
            counters.duplicates += 1
            counters.skipped += 1
            continue
        seen_source_ids.add(source_id)
        payload["emit_dialog_event"] = dialog_id not in seen_dialog_ids
        if payload["emit_dialog_event"]:
            counters.dialog_events += 1
            seen_dialog_ids.add(dialog_id)
        dialog_key = "phone" if payload.get("dialog_phone_normalized") else "username" if payload.get("dialog_username") else "unmatched"
        if dialog_id not in dialog_resolution_hints:
            dialog_resolution_hints[dialog_id] = dialog_key
        if payload.get("dialog_phone_normalized"):
            counters.linked_by_phone += 1
        if payload.get("dialog_username"):
            counters.linked_by_username += 1
        if not payload.get("dialog_phone_normalized") and not payload.get("dialog_username"):
            counters.session_only += 1
        records.append(
            TimelineSourceRecord(
                source_system=SOURCE_SYSTEM,
                source_ref=source_id,
                payload=payload,
                source_path=str(source_path),
            )
        )
    counters.unmatched_dialogs = sum(1 for value in dialog_resolution_hints.values() if value == "unmatched")
    return tuple(records), counters


def tg_message_to_payload(msg: Mapping[str, Any], dialog: Mapping[str, Any], brand: str) -> dict[str, Any]:
    dialog_id = text_id(msg.get("dialog_id") or dialog.get("dialog_id"))
    message_id = text_id(msg.get("message_id"))
    phone = normalize_phone(str(dialog.get("phone") or ""))
    display_name = (
        optional_str(msg.get("dialog_name"))
        or optional_str(dialog.get("title"))
        or optional_str(dialog.get("name"))
        or ""
    )
    username = normalize_username(dialog.get("username") or msg.get("sender_username"))
    return {
        "channel": CHANNEL,
        "channel_thread_id": dialog_id,
        "channel_message_id": message_id,
        "channel_user_id": dialog_id,
        "received_at": require_text(msg.get("date"), "date"),
        "direction": "outbound" if msg.get("out") else "inbound",
        "display_name": display_name,
        "text": str(msg.get("text") or "").strip(),
        "brand_hint": normalize_brand(brand),
        "timeline_source_id": telegram_source_id(dialog_id, message_id),
        "telegram_dialog_id": dialog_id,
        "telegram_message_id": message_id,
        "sender_id": optional_str(msg.get("sender_id")),
        "sender_username": optional_str(msg.get("sender_username")),
        "dialog_username": username,
        "dialog_phone": optional_str(dialog.get("phone")),
        "dialog_phone_normalized": phone,
    }


def load_dialogs(path: Path) -> dict[str, Mapping[str, Any]]:
    return dialogs_by_id(read_jsonl(path))


def dialogs_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    dialogs: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        dialog_id = text_id(row.get("dialog_id"))
        dialogs[dialog_id] = row
    return dialogs


def merge_dialog_identity_rows(
    base_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if not identity_rows:
        return tuple(base_rows)
    identity_by_id = {text_id(row.get("dialog_id")): row for row in identity_rows if optional_str(row.get("dialog_id"))}
    merged: list[Mapping[str, Any]] = []
    for row in base_rows:
        dialog_id = text_id(row.get("dialog_id"))
        rich = identity_by_id.get(dialog_id)
        if not rich:
            merged.append(row)
            continue
        payload = dict(row)
        for key in ("username", "phone", "first_name", "last_name", "is_contact", "is_mutual_contact", "is_bot"):
            value = rich.get(key)
            if value not in (None, ""):
                payload[key] = value
        merged.append(payload)
    return tuple(merged)


def read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    return read_jsonl_lenient(path).rows


def read_jsonl_lenient(path: Path) -> JsonlReadResult:
    rows: list[Mapping[str, Any]] = []
    nonempty_lines = 0
    bad_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            nonempty_lines += 1
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            if not isinstance(payload, Mapping):
                bad_rows += 1
                continue
            rows.append(dict(payload))
    return JsonlReadResult(rows=tuple(rows), nonempty_lines=nonempty_lines, bad_rows=bad_rows)


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
        return PhoneCustomerLookup(unique_customer_ids={}, ambiguous_phone_matches=0)
    tenant = normalize_key(tenant_id, "tenant_id")
    rows = query_existing_phone_rows(db_path, tenant_id=tenant, phones=phones)
    by_phone: dict[str, set[str]] = {}
    ambiguous_values: set[str] = set()
    for phone, customer_id, match_class in rows:
        if phone and customer_id:
            by_phone.setdefault(phone, set()).add(customer_id)
        if match_class in {"ambiguous", "duplicate"}:
            ambiguous_values.add(phone)
    unique_customer_ids = {
        phone: next(iter(customer_ids))
        for phone, customer_ids in by_phone.items()
        if len(customer_ids) == 1 and phone not in ambiguous_values
    }
    ambiguous_phones = tuple(
        sorted(phone for phone, customer_ids in by_phone.items() if len(customer_ids) > 1 or phone in ambiguous_values)
    )
    return PhoneCustomerLookup(
        unique_customer_ids=unique_customer_ids,
        ambiguous_phone_matches=len(ambiguous_phones),
        candidate_customer_ids={phone: tuple(sorted(customer_ids)) for phone, customer_ids in by_phone.items()},
        ambiguous_phones=ambiguous_phones,
    )


def query_existing_phone_rows(db_path: Path, *, tenant_id: str, phones: set[str]) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    with open_readonly_sqlite(db_path) as con:
        if sqlite_table_exists(con, "customer_identities"):
            for chunk in chunks(sorted(phones), 800):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    (str(row["primary_phone"]), str(row["customer_id"]), "strong_unique")
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
                    (str(row["link_value"]), str(row["customer_id"]), str(row["match_class"] or "strong_unique"))
                    for row in con.execute(
                        f"""
                        SELECT link_value, customer_id, match_class
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


def load_username_customer_lookup(
    db_path: Path,
    *,
    tenant_id: str,
    usernames: set[str],
) -> UsernameCustomerLookup:
    normalized_usernames = {item for item in (normalize_username(value) for value in usernames) if item}
    if not normalized_usernames or not db_path.exists():
        return UsernameCustomerLookup(unique_customer_ids={})
    tenant = normalize_key(tenant_id, "tenant_id")
    rows = query_existing_username_rows(db_path, tenant_id=tenant, usernames=normalized_usernames)
    by_username: dict[str, set[str]] = {}
    ambiguous_values: set[str] = set()
    for username, customer_id, match_class in rows:
        if username and customer_id:
            by_username.setdefault(username, set()).add(customer_id)
        if match_class in {"ambiguous", "duplicate"}:
            ambiguous_values.add(username)
    ambiguous_usernames = tuple(
        sorted(username for username, customer_ids in by_username.items() if len(customer_ids) > 1 or username in ambiguous_values)
    )
    return UsernameCustomerLookup(
        unique_customer_ids={
            username: next(iter(customer_ids))
            for username, customer_ids in by_username.items()
            if len(customer_ids) == 1 and username not in ambiguous_values
        },
        ambiguous_usernames=ambiguous_usernames,
        candidate_customer_ids={username: tuple(sorted(customer_ids)) for username, customer_ids in by_username.items()},
    )


def query_existing_username_rows(
    db_path: Path,
    *,
    tenant_id: str,
    usernames: set[str],
) -> tuple[tuple[str, str, str], ...]:
    if not usernames:
        return ()
    rows: list[tuple[str, str, str]] = []
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "identity_links"):
            return ()
        for chunk in chunks(sorted(usernames), 800):
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                (str(row["link_value"]), str(row["customer_id"]), str(row["match_class"] or "strong_unique"))
                for row in con.execute(
                    f"""
                    SELECT link_value, customer_id, match_class
                    FROM identity_links
                    WHERE tenant_id = ?
                      AND link_type = 'telegram_username'
                      AND link_value IN ({placeholders})
                    """,
                    (tenant_id, *chunk),
                )
                if row["link_value"] and row["customer_id"]
            )
    return tuple(rows)


def load_existing_message_source_ids(
    db_path: Path,
    *,
    tenant_id: str,
    source_ids: Sequence[str],
) -> set[str]:
    if not source_ids or not db_path.exists():
        return set()
    found: set[str] = set()
    tenant = normalize_key(tenant_id, "tenant_id")
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "timeline_events"):
            return set()
        for chunk in chunks(tuple(dict.fromkeys(source_ids)), 800):
            placeholders = ",".join("?" for _ in chunk)
            found.update(
                str(row["source_id"])
                for row in con.execute(
                    f"""
                    SELECT source_id
                    FROM timeline_events
                    WHERE tenant_id = ?
                      AND source_system = ?
                      AND event_type = 'telegram_message'
                      AND source_id IN ({placeholders})
                    """,
                    (tenant, SOURCE_SYSTEM, *chunk),
                )
            )
    return found


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


def source_file_inventory(*paths: Path) -> tuple[Mapping[str, Any], ...]:
    items: list[Mapping[str, Any]] = []
    for path in paths:
        stat = path.stat()
        items.append(
            {
                "path": str(path),
                "exists": True,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path),
            }
        )
    return tuple(items)


def chunks(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(values[idx : idx + size]) for idx in range(0, len(values), size))


def unique_tags(tags: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = normalize_key(tag, "relevance tag")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def is_user_peer(message_peer_kind: str, dialog_peer_kind: str) -> bool:
    kinds = tuple(kind for kind in (message_peer_kind, dialog_peer_kind) if kind)
    return bool(kinds) and all(kind == "user" for kind in kinds)


def telegram_source_id(dialog_id: str, message_id: str) -> str:
    return f"{CHANNEL}:{require_text(dialog_id, 'dialog_id')}:{require_text(message_id, 'message_id')}"


def normalize_brand(value: Any) -> str:
    brand = str(value or "unknown").strip().lower()
    if brand not in BRANDS:
        raise ValueError(f"unsupported brand: {value!r}")
    return brand


def identity_status_for_resolution(resolution: TelegramIdentityResolution) -> IdentityStatus:
    if resolution.match_class == IdentityMatchClass.AMBIGUOUS:
        return IdentityStatus.AMBIGUOUS
    if resolution.match_class == IdentityMatchClass.UNMATCHED:
        return IdentityStatus.UNMATCHED
    return IdentityStatus.PARTIAL


def resolve_telegram_identity(
    *,
    phone: Optional[str],
    username: Optional[str],
    phone_lookup: PhoneCustomerLookup,
    username_lookup: UsernameCustomerLookup,
) -> TelegramIdentityResolution:
    candidate_evidence: dict[str, set[str]] = {}
    conflict_flags: set[str] = set()
    if phone:
        if phone in phone_lookup.ambiguous_phones:
            conflict_flags.add("phone_ambiguous")
            for candidate_id in phone_lookup.candidate_customer_ids.get(phone, ()):
                candidate_evidence.setdefault(candidate_id, set()).add("phone")
        elif phone in phone_lookup.unique_customer_ids:
            candidate_evidence.setdefault(phone_lookup.unique_customer_ids[phone], set()).add("phone")
    if username:
        if username in username_lookup.ambiguous_usernames:
            conflict_flags.add("username_ambiguous")
            for candidate_id in username_lookup.candidate_customer_ids.get(username, ()):
                candidate_evidence.setdefault(candidate_id, set()).add("username")
        elif username in username_lookup.unique_customer_ids:
            candidate_evidence.setdefault(username_lookup.unique_customer_ids[username], set()).add("username")
    candidate_ids = tuple(sorted(candidate_evidence))
    evidence_keys = tuple(sorted({key for values in candidate_evidence.values() for key in values}))
    if conflict_flags or len(candidate_ids) > 1:
        if len(candidate_ids) > 1:
            conflict_flags.add("multiple_candidate_customers")
        return TelegramIdentityResolution(
            match_class=IdentityMatchClass.AMBIGUOUS,
            customer_id=None,
            candidate_customer_ids=candidate_ids,
            confidence=0.45,
            evidence_keys=evidence_keys or tuple(key for key, value in (("phone", phone), ("username", username)) if value),
            conflict_flags=tuple(conflict_flags),
        )
    if len(candidate_ids) == 1:
        confidence = 0.96 if "phone" in evidence_keys else 0.8
        return TelegramIdentityResolution(
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            customer_id=candidate_ids[0],
            candidate_customer_ids=candidate_ids,
            confidence=confidence,
            evidence_keys=evidence_keys,
        )
    return TelegramIdentityResolution(
        match_class=IdentityMatchClass.UNMATCHED,
        customer_id=None,
        candidate_customer_ids=(),
        confidence=0.0,
        evidence_keys=tuple(key for key, value in (("phone", phone), ("username", username)) if value),
        conflict_flags=("no_customer_match",),
    )


def safe_telegram_message_record(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    text = str(payload.get("text") or "")
    return {
        "telegram_dialog_id": optional_str(payload.get("telegram_dialog_id")),
        "telegram_message_id": optional_str(payload.get("telegram_message_id")),
        "sender_id_present": bool(optional_str(payload.get("sender_id"))),
        "sender_username_present": bool(optional_str(payload.get("sender_username"))),
        "dialog_phone_present": bool(payload.get("dialog_phone_normalized")),
        "dialog_username_present": bool(payload.get("dialog_username")),
        "brand": normalize_brand(payload.get("brand_hint")),
        "direction": optional_str(payload.get("direction")),
        "received_at": optional_str(payload.get("received_at")),
        "text_length": len(text),
        "has_text": bool(text.strip()),
    }


def autodetect_identity_export_dir(export_dir: Path) -> Optional[Path]:
    if export_dir.name.endswith(("_max", "_with_contacts")):
        return None
    for suffix in ("_max", "_with_contacts"):
        candidate = export_dir.with_name(f"{export_dir.name}{suffix}")
        if (candidate / "dialogs.jsonl").exists():
            return candidate.resolve(strict=False)
    return None


def safe_telegram_history_inventory(export_dir: Path) -> Optional[Mapping[str, Any]]:
    if not (export_dir / "summary.json").exists():
        return {"status": "missing_summary_json", "export_root_name": export_dir.name}
    try:
        return dict(build_telegram_history_inventory(export_dir).to_json_dict())
    except Exception as exc:  # noqa: BLE001 - inventory is report-only.
        return {"status": "inventory_error", "error_type": type(exc).__name__, "export_root_name": export_dir.name}


def normalized_match_counts(
    records: Sequence[TimelineSourceRecord],
    *,
    phone_lookup: PhoneCustomerLookup,
    username_lookup: UsernameCustomerLookup,
) -> Mapping[str, int]:
    by_dialog: dict[str, TelegramIdentityResolution] = {}
    for record in records:
        payload = record.payload
        dialog_id = optional_str(payload.get("telegram_dialog_id"))
        if not dialog_id or dialog_id in by_dialog:
            continue
        phone = normalize_phone(str(payload.get("dialog_phone_normalized") or payload.get("dialog_phone") or ""))
        username = normalize_username(payload.get("dialog_username") or payload.get("sender_username"))
        by_dialog[dialog_id] = resolve_telegram_identity(
            phone=phone,
            username=username,
            phone_lookup=phone_lookup,
            username_lookup=username_lookup,
        )
    counts = Counter(resolution.match_class.value for resolution in by_dialog.values())
    return {
        "dialogs_total": len(by_dialog),
        "strong_unique_dialogs": counts.get(IdentityMatchClass.STRONG_UNIQUE.value, 0),
        "ambiguous_dialogs": counts.get(IdentityMatchClass.AMBIGUOUS.value, 0),
        "unmatched_dialogs": counts.get(IdentityMatchClass.UNMATCHED.value, 0),
    }


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def text_id(value: Any) -> str:
    return require_text(value, "id")


__all__ = [
    "TelegramExportImportConfig",
    "TelegramExportTimelineNormalizer",
    "build_timeline_records",
    "main",
    "run_telegram_export_import",
    "telegram_source_id",
    "tg_message_to_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
