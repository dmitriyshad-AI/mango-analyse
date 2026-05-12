from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.ids import normalize_key, require_text, stable_digest
from mango_mvp.customer_timeline.ingestion import (
    AmoSnapshotNormalizer,
    ChannelMessageNormalizer,
    MailMessageNormalizer,
    MangoCallSummaryNormalizer,
    TallantoSnapshotNormalizer,
    TimelineImportError,
    TimelineImportReport,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineNormalizer,
    TimelineSourceRecord,
    build_source_inventory,
    infer_identity_conflicts,
    load_local_source_records,
    load_sqlite_source_records,
    merge_counts,
    timeline_ingestion_safety_contract,
    zero_normalized_counts,
)
from mango_mvp.customer_timeline.safety import blocked_live_actions, guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    guard_customer_timeline_sqlite_path,
)


CUSTOMER_TIMELINE_IMPORT_CLI_SCHEMA_VERSION = "customer_timeline_import_cli_v1"
SOURCE_KIND_TO_SYSTEM = {
    "amocrm_snapshot": "amocrm_snapshot",
    "tallanto_snapshot": "tallanto_snapshot",
    "channel_snapshot": "channel_snapshot",
    "mail_archive": "mail_archive",
    "mango_processed_summary": "mango_processed_summary",
}


@dataclass(frozen=True)
class TimelineImportCliConfig:
    tenant_id: str
    source_kind: str
    source_path: Path
    allowed_root: Path
    timeline_db: Path
    source_ref: str
    out_path: Optional[Path] = None
    apply: bool = False
    actor: str = "customer_timeline_import_cli"
    idempotency_key: Optional[str] = None
    source_ref_prefix: Optional[str] = None
    csv_encoding: str = "utf-8-sig"
    csv_delimiter: Optional[str] = None
    sqlite_table: Optional[str] = None
    sqlite_source_ref_column: Optional[str] = None
    sqlite_where: Optional[str] = None
    limit: Optional[int] = None
    observed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        tenant = normalize_key(self.tenant_id, "tenant_id")
        source_kind = normalize_key(self.source_kind, "source_kind")
        if source_kind not in SOURCE_KIND_TO_SYSTEM:
            raise ValueError(f"unsupported source_kind: {self.source_kind}")
        root = Path(self.allowed_root).resolve(strict=False)
        source_path = guard_customer_timeline_output_path(self.source_path, root)
        timeline_db = guard_customer_timeline_sqlite_path(self.timeline_db)
        timeline_db = guard_customer_timeline_output_path(timeline_db, root)
        out_path = guard_customer_timeline_output_path(self.out_path, root) if self.out_path else None
        if out_path and out_path == source_path:
            raise ValueError("report output path must not overwrite source path")
        if out_path and out_path == timeline_db:
            raise ValueError("report output path must not overwrite timeline DB")
        if source_path == timeline_db:
            raise ValueError("source path and timeline DB path must be different")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "timeline_db", timeline_db)
        object.__setattr__(self, "source_ref", require_text(self.source_ref, "source_ref"))
        object.__setattr__(self, "actor", require_text(self.actor, "actor"))
        object.__setattr__(self, "out_path", out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        report = run_timeline_import_cli(config)
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if config.out_path:
            config.out_path.parent.mkdir(parents=True, exist_ok=True)
            config.out_path.write_text(f"{text}\n", encoding="utf-8")
        else:
            print(text)
        return 0 if report["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - CLI should return a compact operator-facing error.
        print(f"customer timeline import failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import local read-only snapshots into the isolated customer_timeline.sqlite. "
            "Defaults to dry-run preview; use --apply to write the product timeline DB."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--source-kind", required=True, choices=sorted(SOURCE_KIND_TO_SYSTEM))
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--timeline-db", help="Target product timeline DB. Defaults to <allowed-root>/customer_timeline/customer_timeline.sqlite.")
    parser.add_argument("--source-ref", help="Logical import source ref. Defaults to source filename.")
    parser.add_argument("--source-ref-prefix", help="Optional per-row source_ref prefix for JSON/JSONL/CSV files.")
    parser.add_argument("--out", help="Optional JSON report path. If omitted, report is printed to stdout.")
    parser.add_argument("--apply", action="store_true", help="Actually upsert into customer_timeline.sqlite.")
    parser.add_argument("--actor", default="customer_timeline_import_cli")
    parser.add_argument("--idempotency-key", help="Optional stable key for repeated imports.")
    parser.add_argument("--csv-encoding", default="utf-8-sig")
    parser.add_argument("--csv-delimiter", help="CSV delimiter. Use '\\t' for tab.")
    parser.add_argument("--sqlite-table", help="Read-only SQLite source table name.")
    parser.add_argument("--sqlite-source-ref-column", help="Column used as per-row source_ref for SQLite sources.")
    parser.add_argument("--sqlite-where", help="Optional read-only SQL WHERE clause without the WHERE keyword.")
    parser.add_argument("--limit", type=int, help="Optional max source rows to read.")
    parser.add_argument("--observed-at", help="Optional ISO datetime used as source observed_at fallback.")
    return parser


def config_from_args(args: argparse.Namespace) -> TimelineImportCliConfig:
    allowed_root = Path(args.allowed_root)
    timeline_db = Path(args.timeline_db) if args.timeline_db else allowed_root / "customer_timeline" / "customer_timeline.sqlite"
    source_path = Path(args.source_path)
    source_ref = args.source_ref or source_path.name
    return TimelineImportCliConfig(
        tenant_id=args.tenant_id,
        source_kind=args.source_kind,
        source_path=source_path,
        allowed_root=allowed_root,
        timeline_db=timeline_db,
        source_ref=source_ref,
        out_path=Path(args.out) if args.out else None,
        apply=bool(args.apply),
        actor=args.actor,
        idempotency_key=args.idempotency_key,
        source_ref_prefix=args.source_ref_prefix,
        csv_encoding=args.csv_encoding,
        csv_delimiter=decode_delimiter(args.csv_delimiter),
        sqlite_table=args.sqlite_table,
        sqlite_source_ref_column=args.sqlite_source_ref_column,
        sqlite_where=args.sqlite_where,
        limit=args.limit,
        observed_at=parse_datetime(args.observed_at) if args.observed_at else None,
    )


def run_timeline_import_cli(config: TimelineImportCliConfig) -> Mapping[str, Any]:
    records = load_records_for_config(config)
    normalizer = normalizer_for_source_kind(config.source_kind, tenant_id=config.tenant_id)
    preview = build_timeline_import_preview(
        records,
        normalizer=normalizer,
        tenant_id=config.tenant_id,
        source_ref=config.source_ref,
        source_kind=config.source_kind,
        timeline_db=config.timeline_db,
    )
    source_inventory_before = tuple(dict(item) for item in preview["source_inventory_before"])
    store_summary_before: Optional[Mapping[str, Any]] = None
    store_summary_after: Optional[Mapping[str, Any]] = None
    import_report: Optional[TimelineImportReport] = None
    mode = "dry_run_preview"
    if config.apply:
        store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root)
        try:
            store_summary_before = store.summary()
            import_report = TimelineImportService(store).import_records(
                records,
                normalizer=normalizer,
                tenant_id=config.tenant_id,
                source_ref=config.source_ref,
                idempotency_key=config.idempotency_key or preview["input_hash"],
                dry_run=False,
                actor=config.actor,
            )
            store_summary_after = store.summary()
        finally:
            store.close()
        mode = "apply"

    source_inventory_after = tuple(dict(item) for item in build_source_inventory(records))
    report_payload = import_report.to_json_dict() if import_report else preview
    source_unchanged = source_inventory_before == source_inventory_after
    safety_flags = timeline_import_cli_safety_contract(write_product_timeline_db=config.apply)
    operation_plan = preview["operation_plan"]
    errors = report_payload["errors"]
    normalized_counts = report_payload["normalized_counts"]
    write_status_counts = report_payload["write_status_counts"]
    return {
        "schema_version": CUSTOMER_TIMELINE_IMPORT_CLI_SCHEMA_VERSION,
        "mode": mode,
        "dry_run": not config.apply,
        "validation_ok": bool(report_payload["validation_ok"]) and source_unchanged,
        "summary": {
            "validation_ok": bool(report_payload["validation_ok"]) and source_unchanged,
            "status": "completed" if bool(report_payload["validation_ok"]) and source_unchanged else "completed_with_warnings",
            "dry_run": not config.apply,
            "tenant_id": config.tenant_id,
            "source_system": SOURCE_KIND_TO_SYSTEM[config.source_kind],
            "source_ref": config.source_ref,
            "records_read": len(records),
            "records_accepted": report_payload["accepted_count"],
            "records_rejected": report_payload["rejected_count"],
            "normalized_total": normalized_total(normalized_counts),
            "writes_planned": len(operation_plan["items"]),
            "writes_applied": sum(int(value) for value in write_status_counts.values()) if config.apply else 0,
            "conflicts_open": preview["conflicts"]["counts_by_status"].get("open", 0),
            "source_unchanged": source_unchanged,
            "safety_ok": safety_ok(safety_flags),
        },
        "run": {
            "run_id": report_payload.get("run_id"),
            "run_kind": "timeline_import",
            "idempotency_key": config.idempotency_key or preview["input_hash"],
            "input_hash": report_payload["input_hash"],
        },
        "source": {
            "loader": source_loader_name(config),
            "allowed_root": str(config.allowed_root),
            "records": [source_record_preview(record) for record in records],
            "inventory": {
                "checkable": bool(source_inventory_before),
                "unchanged": source_unchanged,
                "before": list(source_inventory_before),
                "after": list(source_inventory_after),
                "changed_paths": changed_inventory_paths(source_inventory_before, source_inventory_after),
            },
            "sqlite": {
                "table_name": config.sqlite_table,
                "source_ref_column": config.sqlite_source_ref_column,
                "where_sql": config.sqlite_where,
                "limit": config.limit,
            }
            if source_loader_name(config) == "sqlite_table"
            else None,
        },
        "normalization": {
            "schema_version": "customer_timeline_ingestion_v1",
            "counts": dict(normalized_counts),
            "by_source_record": preview["by_source_record"],
        },
        "writes": {
            "target": {
                "db_path": str(config.timeline_db),
                "allowed_root": str(config.allowed_root),
                "schema_version": "customer_timeline_sqlite_v1",
            },
            "applied": config.apply,
            "planned_counts_by_type": operation_plan["counts"],
            "status_counts": dict(write_status_counts),
            "items": operation_plan["items"],
        },
        "conflicts": preview["conflicts"],
        "errors": list(errors),
        "tenant_id": config.tenant_id,
        "source_kind": config.source_kind,
        "source_system": SOURCE_KIND_TO_SYSTEM[config.source_kind],
        "source_ref": config.source_ref,
        "paths": {
            "allowed_root": str(config.allowed_root),
            "source_path": str(config.source_path),
            "timeline_db": str(config.timeline_db),
            "report_out": str(config.out_path) if config.out_path else None,
        },
        "operator_summary": {
            "records_loaded": len(records),
            "accepted_count": report_payload["accepted_count"],
            "rejected_count": report_payload["rejected_count"],
            "conflicts": report_payload["normalized_counts"].get("conflicts", 0),
            "source_unchanged": source_unchanged,
            "write_applied": config.apply,
        },
        "import_report": report_payload,
        "preview": preview,
        "store_summary_before": store_summary_before,
        "store_summary_after": store_summary_after,
        "source_inventory_before": list(source_inventory_before),
        "source_inventory_after": list(source_inventory_after),
        "source_unchanged": source_unchanged,
        "safety": {
            **safety_flags,
            "ok": safety_ok(safety_flags),
            "blocked_live_actions": blocked_live_actions(),
        },
    }


def load_records_for_config(config: TimelineImportCliConfig) -> tuple[TimelineSourceRecord, ...]:
    source_system = SOURCE_KIND_TO_SYSTEM[config.source_kind]
    if config.source_path.suffix.casefold() in {".sqlite", ".db", ".sqlite3"}:
        if not config.sqlite_table:
            raise ValueError("--sqlite-table is required for SQLite sources")
        return load_sqlite_source_records(
            config.source_path,
            allowed_root=config.allowed_root,
            source_system=source_system,
            table_name=config.sqlite_table,
            source_ref_column=config.sqlite_source_ref_column,
            where_sql=config.sqlite_where,
            limit=config.limit,
            observed_at=config.observed_at,
        )
    if config.sqlite_table:
        raise ValueError("--sqlite-table can only be used with .sqlite/.db/.sqlite3 sources")
    return load_local_source_records(
        config.source_path,
        allowed_root=config.allowed_root,
        source_system=source_system,
        source_ref_prefix=config.source_ref_prefix,
        csv_encoding=config.csv_encoding,
        csv_delimiter=config.csv_delimiter,
        observed_at=config.observed_at,
    )


def normalizer_for_source_kind(source_kind: str, *, tenant_id: str) -> TimelineNormalizer:
    normalized = normalize_key(source_kind, "source_kind")
    if normalized == "amocrm_snapshot":
        return AmoSnapshotNormalizer(tenant_id=tenant_id)
    if normalized == "tallanto_snapshot":
        return TallantoSnapshotNormalizer(tenant_id=tenant_id)
    if normalized == "channel_snapshot":
        return ChannelMessageNormalizer(tenant_id=tenant_id)
    if normalized == "mail_archive":
        return MailMessageNormalizer(tenant_id=tenant_id)
    if normalized == "mango_processed_summary":
        return MangoCallSummaryNormalizer(tenant_id=tenant_id)
    raise ValueError(f"unsupported source_kind: {source_kind}")


def build_timeline_import_preview(
    records: Sequence[TimelineSourceRecord],
    *,
    normalizer: TimelineNormalizer,
    tenant_id: str,
    source_ref: str,
    source_kind: str,
    timeline_db: Path,
) -> Mapping[str, Any]:
    tenant = normalize_key(tenant_id, "tenant_id")
    normalized_source_ref = require_text(source_ref, "source_ref")
    normalized_records = tuple(records)
    input_hash = stable_digest(
        {
            "tenant_id": tenant,
            "source_system": normalizer.source_system,
            "source_ref": normalized_source_ref,
            "records": [record.to_json_dict() for record in normalized_records],
        }
    )
    source_inventory_before = build_source_inventory(normalized_records)
    accepted = 0
    errors: list[TimelineImportError] = []
    batches: list[TimelineNormalizedBatch] = []
    normalized_counts = zero_normalized_counts()
    for record in normalized_records:
        try:
            batch = normalizer.normalize(record)
            batches.append(batch)
            merge_counts(normalized_counts, batch.counts())
            accepted += 1
        except Exception as exc:  # noqa: BLE001 - keep preview per-row and operator-readable.
            errors.append(TimelineImportError(source_ref=record.source_ref, error_type=type(exc).__name__, message=str(exc)))
    inferred_conflicts = infer_identity_conflicts(batches)
    normalized_counts["conflicts"] = normalized_counts.get("conflicts", 0) + len(inferred_conflicts)
    source_inventory_after = build_source_inventory(normalized_records)
    operation_plan = build_operation_plan(batches, inferred_conflicts)
    conflicts = build_conflict_report(batches, inferred_conflicts)
    return {
        "schema_version": CUSTOMER_TIMELINE_IMPORT_CLI_SCHEMA_VERSION,
        "preview_schema_version": "customer_timeline_import_preview_v1",
        "dry_run": True,
        "run_id": None,
        "source_kind": normalize_key(source_kind, "source_kind"),
        "source_system": normalize_key(normalizer.source_system, "source_system"),
        "source_ref": normalized_source_ref,
        "timeline_db": str(guard_customer_timeline_sqlite_path(timeline_db)),
        "input_hash": input_hash,
        "idempotency_key": input_hash,
        "accepted_count": accepted,
        "rejected_count": len(errors),
        "normalized_counts": dict(normalized_counts),
        "write_status_counts": {
            "would_upsert": len(operation_plan["items"]),
            "would_record_conflict": operation_plan["counts"].get("timeline_conflict", 0),
        },
        "errors": [item.to_json_dict() for item in errors],
        "source_inventory": list(source_inventory_after),
        "source_inventory_before": list(source_inventory_before),
        "source_inventory_after": list(source_inventory_after),
        "source_unchanged": source_inventory_before == source_inventory_after,
        "operation_plan": operation_plan,
        "by_source_record": [normalized_source_record_summary(batch) for batch in batches],
        "conflicts": conflicts,
        "validation_ok": len(errors) == 0,
        "safety": timeline_import_cli_safety_contract(write_product_timeline_db=False),
    }


def build_operation_plan(
    batches: Sequence[TimelineNormalizedBatch],
    inferred_conflicts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    items: list[Mapping[str, Any]] = []
    for batch in batches:
        source_ref = batch.source_record.source_ref
        for customer in batch.customers:
            items.append(operation_item("upsert_customer", "customer_identity", customer.customer_id, source_ref))
        for link in batch.identity_links:
            items.append(operation_item("upsert_identity_link", "identity_link", link.link_id, source_ref))
        for opportunity in batch.opportunities:
            items.append(operation_item("upsert_opportunity", "customer_opportunity", opportunity.opportunity_id, source_ref))
        for event in batch.events:
            items.append(operation_item("upsert_event", "timeline_event", event.event_id, source_ref))
        for artifact in batch.artifacts:
            items.append(operation_item("upsert_artifact", "event_artifact", artifact.artifact_id, source_ref))
        for signal in batch.signals:
            items.append(operation_item("upsert_signal", "derived_signal", signal.signal_id, source_ref))
        for chunk in batch.bot_context_chunks:
            items.append(operation_item("upsert_bot_context_chunk", "bot_context_chunk", chunk.chunk_id, source_ref))
        for conflict in batch.conflicts:
            items.append(
                {
                    "operation": "record_conflict",
                    "record_type": "timeline_conflict",
                    "source_ref": source_ref,
                    "conflict_type": conflict.get("conflict_type"),
                    "entity_refs": list(conflict.get("entity_refs") or ()),
                }
            )
    for conflict in inferred_conflicts:
        items.append(
            {
                "operation": "record_inferred_conflict",
                "record_type": "timeline_conflict",
                "source_ref": "inferred_identity_conflicts",
                "conflict_type": conflict.get("conflict_type"),
                "entity_refs": list(conflict.get("entity_refs") or ()),
            }
        )
    return {
        "schema_version": "customer_timeline_import_operation_plan_v1",
        "items": items,
        "counts": count_operations(items),
    }


def operation_item(operation: str, record_type: str, record_id: Optional[str], source_ref: str) -> Mapping[str, Any]:
    return {
        "operation": operation,
        "record_type": record_type,
        "record_id": record_id,
        "source_ref": source_ref,
    }


def count_operations(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("record_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_conflict_report(
    batches: Sequence[TimelineNormalizedBatch],
    inferred_conflicts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    items: list[Mapping[str, Any]] = []
    for batch in batches:
        for conflict in batch.conflicts:
            items.append(conflict_preview(conflict, origin=batch.source_record.source_ref))
    for conflict in inferred_conflicts:
        items.append(conflict_preview(conflict, origin="inferred_identity"))
    counts_by_status: dict[str, int] = {}
    counts_by_type: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "open")
        conflict_type = str(item.get("conflict_type") or "unknown")
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
        counts_by_type[conflict_type] = counts_by_type.get(conflict_type, 0) + 1
    return {
        "schema_version": "customer_timeline_import_conflicts_v1",
        "auto_merge": False,
        "counts_by_status": counts_by_status,
        "counts_by_type": counts_by_type,
        "items": items,
    }


def conflict_preview(conflict: Mapping[str, Any], *, origin: str) -> Mapping[str, Any]:
    return {
        "origin": origin,
        "conflict_type": conflict.get("conflict_type"),
        "severity": conflict.get("severity") or "medium",
        "status": conflict.get("status") or "open",
        "entity_refs": list(conflict.get("entity_refs") or ()),
        "summary": conflict.get("summary"),
        "metadata": dict(conflict.get("metadata") or {}),
    }


def normalized_source_record_summary(batch: TimelineNormalizedBatch) -> Mapping[str, Any]:
    counts = dict(batch.counts())
    return {
        "source_ref": batch.source_record.source_ref,
        "source_system": batch.source_record.source_system,
        "payload_hash": batch.source_record.payload_hash,
        "counts": counts,
        "conflict_count": counts.get("conflicts", 0),
    }


def source_record_preview(record: TimelineSourceRecord) -> Mapping[str, Any]:
    return {
        "source_ref": record.source_ref,
        "source_system": record.source_system,
        "source_path": record.source_path,
        "payload_hash": record.payload_hash,
        "observed_at": record.observed_at.isoformat() if record.observed_at else None,
    }


def changed_inventory_paths(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> list[str]:
    before_by_path = {str(item.get("path")): dict(item) for item in before}
    after_by_path = {str(item.get("path")): dict(item) for item in after}
    paths = sorted(set(before_by_path) | set(after_by_path))
    return [path for path in paths if before_by_path.get(path) != after_by_path.get(path)]


def source_loader_name(config: TimelineImportCliConfig) -> str:
    return "sqlite_table" if config.source_path.suffix.casefold() in {".sqlite", ".db", ".sqlite3"} else "local_file"


def normalized_total(counts: Mapping[str, int]) -> int:
    return sum(int(value) for value in counts.values())


def safety_ok(flags: Mapping[str, Any]) -> bool:
    return all(flags.get(action) is False for action in blocked_live_actions()) and flags.get("read_only_source_systems") is True


def timeline_import_cli_safety_contract(*, write_product_timeline_db: bool) -> Mapping[str, Any]:
    return {
        **timeline_ingestion_safety_contract(),
        "schema_version": CUSTOMER_TIMELINE_IMPORT_CLI_SCHEMA_VERSION,
        "write_product_timeline_db": bool(write_product_timeline_db),
        "default_mode": "dry_run_preview",
        "requires_apply_for_db_write": True,
        "raw_source_payload_in_report": False,
        "report_contains_operation_plan": True,
    }


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def decode_delimiter(value: Optional[str]) -> Optional[str]:
    if value == "\\t":
        return "\t"
    return value


__all__ = [
    "CUSTOMER_TIMELINE_IMPORT_CLI_SCHEMA_VERSION",
    "SOURCE_KIND_TO_SYSTEM",
    "TimelineImportCliConfig",
    "build_operation_plan",
    "build_parser",
    "build_timeline_import_preview",
    "config_from_args",
    "decode_delimiter",
    "load_records_for_config",
    "main",
    "normalizer_for_source_kind",
    "parse_datetime",
    "run_timeline_import_cli",
    "timeline_import_cli_safety_contract",
]
