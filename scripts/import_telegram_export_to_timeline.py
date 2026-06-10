from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
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
SOURCE_SYSTEM = "channel_snapshot"
SOURCE_KIND = "channel_snapshot"
CHANNEL = "telegram"
BRANDS = {"foton", "unpk", "unknown"}


@dataclass(frozen=True)
class TelegramExportImportConfig:
    export_dir: Path
    allowed_root: Path
    timeline_db: Path
    tenant_id: str = "foton"
    brand: str = "unknown"
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
        timeline_db = guard_customer_timeline_output_path(guard_customer_timeline_sqlite_path(self.timeline_db), root)
        out_path = guard_customer_timeline_output_path(self.out_path, root) if self.out_path else None
        if out_path and out_path in {dialogs_path, messages_path, timeline_db}:
            raise ValueError("report output path must not overwrite source files or timeline DB")
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "export_dir", export_dir)
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
    def source_ref(self) -> str:
        return f"telegram_export:{self.export_dir.name}:{self.brand}"


@dataclass
class TelegramBuildCounters:
    dialogs: int = 0
    messages: int = 0
    skipped: int = 0
    groups: int = 0
    linked_by_phone: int = 0
    session_only: int = 0
    duplicates: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "dialogs": self.dialogs,
            "messages": self.messages,
            "imported": 0,
            "skipped": self.skipped,
            "groups": self.groups,
            "linked_by_phone": self.linked_by_phone,
            "session_only": self.session_only,
            "duplicates": self.duplicates,
        }


class TelegramExportTimelineNormalizer:
    source_system = SOURCE_SYSTEM

    def __init__(self, *, tenant_id: str, phone_customer_ids: Mapping[str, str]) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")
        self._base = ChannelMessageNormalizer(tenant_id=self.tenant_id)
        self._phone_customer_ids = dict(phone_customer_ids)

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        batch = self._base.normalize(record)
        payload = record.payload
        source_id = require_text(payload.get("timeline_source_id"), "timeline_source_id")
        dialog_id = require_text(payload.get("telegram_dialog_id"), "telegram_dialog_id")
        brand = normalize_brand(payload.get("brand_hint"))
        phone = normalize_phone(str(payload.get("dialog_phone_normalized") or payload.get("dialog_phone") or ""))

        if not batch.customers or not batch.events:
            return batch

        base_customer = batch.customers[0]
        existing_customer_id = self._phone_customer_ids.get(phone or "")
        if existing_customer_id:
            customer_id = existing_customer_id
            customers: tuple[CustomerIdentity, ...] = ()
        elif phone:
            customer = replace(
                base_customer,
                customer_id=None,
                identity_status=IdentityStatus.STRONG,
                primary_phone=phone,
            )
            customer_id = customer.customer_id
            customers = (customer,)
        else:
            customer_id = base_customer.customer_id
            customers = tuple(batch.customers)

        base_event = batch.events[0]
        event = replace(
            base_event,
            event_id=None,
            customer_id=customer_id,
            source_id=source_id,
            metadata={**base_event.metadata, "brand": brand},
        )

        links = [replace(link, customer_id=customer_id) for link in batch.identity_links]
        if phone:
            links.append(
                IdentityLink(
                    tenant_id=self.tenant_id,
                    customer_id=customer_id,
                    link_type="phone",
                    link_value=phone,
                    source_system=self.source_system,
                    source_ref=f"channel:{CHANNEL}:{dialog_id}:phone",
                    match_class=IdentityMatchClass.STRONG_UNIQUE,
                    confidence=0.9,
                    first_seen_at=event.event_at,
                    last_seen_at=event.event_at,
                )
            )

        brand_tag = f"brand:{brand}"
        chunks = tuple(
            replace(
                chunk,
                chunk_id=None,
                customer_id=customer_id,
                event_id=event.event_id,
                relevance_tags=unique_tags((*chunk.relevance_tags, brand_tag)),
                metadata={**chunk.metadata, "brand": brand},
            )
            for chunk in batch.bot_context_chunks
        )

        return TimelineNormalizedBatch(
            source_record=batch.source_record,
            customers=customers,
            identity_links=tuple(links),
            opportunities=batch.opportunities,
            events=(event,),
            artifacts=batch.artifacts,
            signals=batch.signals,
            bot_context_chunks=chunks,
            conflicts=batch.conflicts,
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
    parser.add_argument("--allowed-root", default=".", help="Safety root for source, report and timeline DB paths.")
    parser.add_argument(
        "--timeline-db",
        help="Target timeline DB. Defaults to <allowed-root>/customer_timeline/customer_timeline.sqlite.",
    )
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--brand", choices=sorted(BRANDS), default="unknown")
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
        tenant_id=args.tenant_id,
        brand=args.brand,
        apply=bool(args.apply),
        out_path=Path(args.out) if args.out else None,
        actor=args.actor,
        idempotency_key=args.idempotency_key,
    )


def run_telegram_export_import(config: TelegramExportImportConfig) -> Mapping[str, Any]:
    source_inventory_before = source_file_inventory(config.dialogs_path, config.messages_path)
    dialog_rows = read_jsonl(config.dialogs_path)
    dialogs = dialogs_by_id(dialog_rows)
    messages = read_jsonl(config.messages_path)
    records, counters = build_timeline_records(
        dialogs=dialogs,
        dialog_count=len(dialog_rows),
        messages=messages,
        brand=config.brand,
        source_path=config.messages_path,
    )
    source_ids = tuple(str(record.payload["timeline_source_id"]) for record in records)
    existing_source_ids = load_existing_message_source_ids(
        config.timeline_db,
        tenant_id=config.tenant_id,
        source_ids=source_ids,
    )
    existing_duplicate_count = sum(1 for source_id in source_ids if source_id in existing_source_ids)
    counters.duplicates += existing_duplicate_count

    phones = {str(record.payload.get("dialog_phone_normalized") or "") for record in records if record.payload.get("dialog_phone_normalized")}
    phone_customer_ids = load_unique_phone_customer_ids(
        config.timeline_db,
        tenant_id=config.tenant_id,
        phones=phones,
    )
    normalizer = TelegramExportTimelineNormalizer(
        tenant_id=config.tenant_id,
        phone_customer_ids=phone_customer_ids,
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

    source_inventory_after = source_file_inventory(config.dialogs_path, config.messages_path)
    source_unchanged = source_inventory_before == source_inventory_after
    accepted_count = int(report_payload["accepted_count"])
    imported = max(0, accepted_count - existing_duplicate_count)
    counter_payload = counters.to_dict()
    counter_payload["imported"] = imported
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
            "dialogs_path": str(config.dialogs_path),
            "messages_path": str(config.messages_path),
            "inventory": {
                "unchanged": source_unchanged,
                "before": list(source_inventory_before),
                "after": list(source_inventory_after),
            },
        },
        "links": {
            "unique_existing_phone_matches": len(phone_customer_ids),
            "existing_duplicate_source_ids": existing_duplicate_count,
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
    brand: str,
    source_path: Path,
) -> tuple[tuple[TimelineSourceRecord, ...], TelegramBuildCounters]:
    counters = TelegramBuildCounters(dialogs=len(dialogs) if dialog_count is None else dialog_count, messages=len(messages))
    records: list[TimelineSourceRecord] = []
    seen_source_ids: set[str] = set()
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
        if payload.get("dialog_phone_normalized"):
            counters.linked_by_phone += 1
        else:
            counters.session_only += 1
        records.append(
            TimelineSourceRecord(
                source_system=SOURCE_SYSTEM,
                source_ref=source_id,
                payload=payload,
                source_path=str(source_path),
            )
        )
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


def read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path.name} line {lineno} must contain a JSON object")
            rows.append(dict(payload))
    return tuple(rows)


def load_unique_phone_customer_ids(
    db_path: Path,
    *,
    tenant_id: str,
    phones: set[str],
) -> dict[str, str]:
    if not phones or not db_path.exists():
        return {}
    tenant = normalize_key(tenant_id, "tenant_id")
    rows = query_existing_phone_rows(db_path, tenant_id=tenant, phones=phones)
    by_phone: dict[str, set[str]] = {}
    for phone, customer_id in rows:
        if phone and customer_id:
            by_phone.setdefault(phone, set()).add(customer_id)
    return {phone: next(iter(customer_ids)) for phone, customer_ids in by_phone.items() if len(customer_ids) == 1}


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
