#!/usr/bin/env python3
"""Import read-only Tallanto payment/abonement snapshots into customer_timeline.

The importer consumes a local JSON snapshot, or stdin, that was produced by the
read-only crm_call.sh MCP wrapper. It does not call AMO, Tallanto, ASR, LLM, or
any network API itself. Use --apply to write only the local customer timeline DB.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast
from urllib.parse import quote

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    TimelineParticipant,
)
from mango_mvp.customer_timeline.ids import normalize_key, optional_text, stable_digest
from mango_mvp.customer_timeline.import_cli import safety_ok
from mango_mvp.customer_timeline.ingestion import (
    TimelineImportReport,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineSourceRecord,
    build_source_inventory,
    compact_text,
    guard_customer_timeline_source_path,
    parse_source_datetime,
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


TALLANTO_PAYMENTS_IMPORT_SCHEMA_VERSION = "tallanto_payments_timeline_import_v1"
SOURCE_SYSTEM = "tallanto_crm_call"
SOURCE_KIND = "tallanto_readonly_mcp_snapshot"
PAYMENT_MODULE = "most_finances"
ABONEMENT_MODULE = "most_abonements"
CLASS_MODULE = "most_class"


@dataclass(frozen=True)
class TallantoPaymentsImportConfig:
    source: Optional[Path]
    timeline_db: Path
    allowed_root: Path
    tenant_id: str
    apply: bool = False
    actor: str = "tallanto_payments_timeline_import"
    source_label: str = "crm_call.sh:tallanto_select"

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        source = guard_customer_timeline_source_path(Path(self.source).expanduser(), root) if self.source else None
        timeline_db = guard_customer_timeline_sqlite_path(Path(self.timeline_db).expanduser())
        timeline_db = guard_customer_timeline_output_path(timeline_db, root)
        if source and source == timeline_db:
            raise ValueError("source path and timeline DB path must be different")
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timeline_db", timeline_db)
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "actor", optional_text(self.actor) or "tallanto_payments_timeline_import")
        object.__setattr__(self, "source_label", optional_text(self.source_label) or "crm_call.sh:tallanto_select")


@dataclass
class TallantoSnapshotStats:
    payment_rows: int = 0
    abonement_rows: int = 0
    class_rows: int = 0
    payment_events: int = 0
    abonement_events: int = 0
    skipped: int = 0
    linked_existing: int = 0
    ambiguous_contact_ids: int = 0
    unmatched_contact_ids: int = 0
    amount_fields_in_events: int = 0
    balance_fields_in_events: int = 0
    bot_safe_amount_leaks: int = 0

    def to_json_dict(self) -> Mapping[str, int]:
        return {
            "payment_rows": self.payment_rows,
            "abonement_rows": self.abonement_rows,
            "class_rows": self.class_rows,
            "payment_events": self.payment_events,
            "abonement_events": self.abonement_events,
            "skipped": self.skipped,
            "linked_existing": self.linked_existing,
            "ambiguous_contact_ids": self.ambiguous_contact_ids,
            "unmatched_contact_ids": self.unmatched_contact_ids,
            "amount_fields_in_events": self.amount_fields_in_events,
            "balance_fields_in_events": self.balance_fields_in_events,
            "bot_safe_amount_leaks": self.bot_safe_amount_leaks,
        }


@dataclass(frozen=True)
class TallantoCustomerLookup:
    unique_customer_ids: Mapping[str, str]
    ambiguous_customer_ids: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unique_customer_ids", dict(self.unique_customer_ids))
        object.__setattr__(
            self,
            "ambiguous_customer_ids",
            {key: tuple(sorted(set(value))) for key, value in self.ambiguous_customer_ids.items()},
        )


@dataclass(frozen=True)
class TallantoIdentityResolution:
    customer_id: Optional[str]
    match_class: IdentityMatchClass
    identity_status: IdentityStatus
    confidence: float
    candidate_customer_ids: Sequence[str] = ()


class TallantoPaymentsTimelineNormalizer:
    source_system = SOURCE_SYSTEM

    def __init__(
        self,
        *,
        tenant_id: str,
        customer_lookup: TallantoCustomerLookup,
        class_lookup: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")
        self._customer_lookup = customer_lookup
        self._class_lookup = {str(key): dict(value) for key, value in class_lookup.items()}

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        module = normalize_key(record.payload.get("_tallanto_module") or record.source_system, "tallanto_module")
        if module == PAYMENT_MODULE:
            return self._normalize_payment(record)
        if module == ABONEMENT_MODULE:
            return self._normalize_abonement(record)
        raise ValueError(f"unsupported Tallanto module for B2: {module}")

    def _normalize_payment(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = dict(record.payload)
        payment_id = require_nonempty(first_value(payload, ("id", "finance_id", "payment_id")), "Tallanto payment id")
        contact_id = clean_text(first_value(payload, ("contact_id", "student_id", "tallanto_id", "Contact_ID")))
        abonement_id = clean_text(first_value(payload, ("most_abonements_id", "abonement_id", "subscription_id")))
        class_id = clean_text(first_value(payload, ("most_class_id", "class_id")))
        event_at = parse_source_datetime(
            first_value(payload, ("date_payment", "date_entered", "date_modified", "event_at")),
            record.observed_at,
        )
        resolution = self._resolve_contact(contact_id)
        source_ref = f"tallanto:{PAYMENT_MODULE}:{payment_id}"
        customer, link = self._customer_and_link(
            contact_id=contact_id,
            source_ref=source_ref,
            event_at=event_at,
            resolution=resolution,
            fallback_ref=f"tallanto:payment:{payment_id}",
        )
        customer_id = resolution.customer_id or customer.customer_id
        amount = money_value(first_value(payload, ("cost", "payment_summa", "amount", "summa")))
        direction = clean_text(first_value(payload, ("direction", "direction_translated")))
        status = clean_text(first_value(payload, ("print_check_status", "payment_status", "status")))
        payment_type = clean_text(first_value(payload, ("type_translated", "type")))
        class_info = self._class_lookup.get(class_id or "", {})
        opportunity = CustomerOpportunity(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_type=OpportunityType.TALLANTO_COURSE,
            source_system=SOURCE_SYSTEM,
            source_id=f"payment:{payment_id}",
            title=compact_text(
                first_value(class_info, ("name", "cource_name", "subject_name"))
                or first_value(payload, ("name",))
                or "Tallanto payment",
                limit=160,
            ),
            status=direction or status,
            product_context=compact_mapping(
                {
                    "amount": amount,
                    "currency": "RUB" if amount is not None else None,
                    "payment_direction": direction,
                    "payment_status": status,
                    "payment_type": payment_type,
                    "abonement_id": abonement_id,
                    "class_id": class_id,
                    "class_name": first_value(class_info, ("name", "cource_name")),
                    "subject_name": first_value(class_info, ("subject_name",)),
                }
            ),
            opened_at=event_at,
            confidence=resolution.confidence,
            evidence={"source_ref": source_ref, "raw_payload_persisted": False},
        )
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type=TimelineEventType.TALLANTO_PAYMENT,
            event_at=event_at,
            source_system=SOURCE_SYSTEM,
            source_id=f"{PAYMENT_MODULE}:{payment_id}",
            source_ref=source_ref,
            direction=TimelineDirection.SYSTEM,
            participants=participant_for_contact(contact_id),
            subject="Tallanto payment",
            text_preview=compact_text(first_value(payload, ("name",)), limit=240),
            summary=compact_text(direction or status or "Tallanto finance operation imported", limit=240),
            match_status=resolution.match_class,
            confidence=resolution.confidence,
            record=compact_mapping(
                {
                    "module": PAYMENT_MODULE,
                    "payment_id": payment_id,
                    "contact_id": contact_id,
                    "amount": amount,
                    "currency": "RUB" if amount is not None else None,
                    "payment_direction": direction,
                    "payment_status": status,
                    "payment_type": payment_type,
                    "abonement_id": abonement_id,
                    "class_id": class_id,
                    "class_name": first_value(class_info, ("name", "cource_name")),
                    "source_fields": "safe_projection",
                    "raw_payload_persisted": False,
                }
            ),
            metadata={
                "source_module": PAYMENT_MODULE,
                "source_kind": SOURCE_KIND,
                "identity_resolution": resolution.match_class.value,
                "raw_payload_persisted": False,
            },
            created_at=event_at,
        )
        conflicts = self._conflicts(contact_id=contact_id, source_ref=source_ref, resolution=resolution)
        return TimelineNormalizedBatch(
            source_record=record,
            customers=() if resolution.customer_id else (customer,),
            identity_links=(link,) if link else (),
            opportunities=(opportunity,),
            events=(event,),
            conflicts=conflicts,
        )

    def _normalize_abonement(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = dict(record.payload)
        abonement_id = require_nonempty(first_value(payload, ("id", "abonement_id", "most_abonements_id")), "Tallanto abonement id")
        contact_id = clean_text(first_value(payload, ("contact_id", "student_id", "tallanto_id", "Contact_ID")))
        event_at = parse_source_datetime(
            first_value(payload, ("date_modified", "date_entered", "start_date", "finish_date", "event_at")),
            record.observed_at,
        )
        resolution = self._resolve_contact(contact_id)
        source_ref = f"tallanto:{ABONEMENT_MODULE}:{abonement_id}"
        customer, link = self._customer_and_link(
            contact_id=contact_id,
            source_ref=source_ref,
            event_at=event_at,
            resolution=resolution,
            fallback_ref=f"tallanto:abonement:{abonement_id}",
        )
        customer_id = resolution.customer_id or customer.customer_id
        amount = money_value(first_value(payload, ("cost", "price", "amount")))
        discount = money_value(first_value(payload, ("discount",)))
        visits_total = numeric_value(first_value(payload, ("num_visit",)))
        visits_left = numeric_value(first_value(payload, ("num_visit_left", "visits_left")))
        status = clean_text(first_value(payload, ("status", "type_translated", "type", "form_translated", "form")))
        title = compact_text(first_value(payload, ("name", "rate_translated", "category_translated")) or "Tallanto abonement", limit=160)
        opportunity = CustomerOpportunity(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_type=OpportunityType.TALLANTO_COURSE,
            source_system=SOURCE_SYSTEM,
            source_id=f"abonement:{abonement_id}",
            title=title,
            status=status,
            product_context=compact_mapping(
                {
                    "abonement_id": abonement_id,
                    "amount": amount,
                    "discount": discount,
                    "visits_total": visits_total,
                    "visits_left": visits_left,
                    "start_date": clean_text(payload.get("start_date")),
                    "finish_date": clean_text(payload.get("finish_date")),
                    "filial": flatten_label(payload.get("filial")),
                    "currency": "RUB" if amount is not None else None,
                }
            ),
            opened_at=event_at,
            confidence=resolution.confidence,
            evidence={"source_ref": source_ref, "raw_payload_persisted": False},
        )
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type=TimelineEventType.TALLANTO_ABONEMENT,
            event_at=event_at,
            source_system=SOURCE_SYSTEM,
            source_id=f"{ABONEMENT_MODULE}:{abonement_id}",
            source_ref=source_ref,
            direction=TimelineDirection.SYSTEM,
            participants=participant_for_contact(contact_id),
            subject="Tallanto abonement",
            text_preview=title,
            summary=compact_text(status or "Tallanto abonement imported", limit=240),
            match_status=resolution.match_class,
            confidence=resolution.confidence,
            record=compact_mapping(
                {
                    "module": ABONEMENT_MODULE,
                    "abonement_id": abonement_id,
                    "contact_id": contact_id,
                    "amount": amount,
                    "discount": discount,
                    "currency": "RUB" if amount is not None else None,
                    "visits_total": visits_total,
                    "visits_left": visits_left,
                    "start_date": clean_text(payload.get("start_date")),
                    "finish_date": clean_text(payload.get("finish_date")),
                    "filial": flatten_label(payload.get("filial")),
                    "source_fields": "safe_projection",
                    "raw_payload_persisted": False,
                }
            ),
            metadata={
                "source_module": ABONEMENT_MODULE,
                "source_kind": SOURCE_KIND,
                "identity_resolution": resolution.match_class.value,
                "raw_payload_persisted": False,
            },
            created_at=event_at,
        )
        conflicts = self._conflicts(contact_id=contact_id, source_ref=source_ref, resolution=resolution)
        return TimelineNormalizedBatch(
            source_record=record,
            customers=() if resolution.customer_id else (customer,),
            identity_links=(link,) if link else (),
            opportunities=(opportunity,),
            events=(event,),
            conflicts=conflicts,
        )

    def _resolve_contact(self, contact_id: Optional[str]) -> TallantoIdentityResolution:
        if not contact_id:
            return TallantoIdentityResolution(
                customer_id=None,
                match_class=IdentityMatchClass.UNMATCHED,
                identity_status=IdentityStatus.UNMATCHED,
                confidence=0.0,
            )
        existing = self._customer_lookup.unique_customer_ids.get(contact_id)
        if existing:
            return TallantoIdentityResolution(
                customer_id=existing,
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                identity_status=IdentityStatus.STRONG,
                confidence=0.98,
            )
        candidates = tuple(self._customer_lookup.ambiguous_customer_ids.get(contact_id, ()))
        if candidates:
            return TallantoIdentityResolution(
                customer_id=None,
                match_class=IdentityMatchClass.AMBIGUOUS,
                identity_status=IdentityStatus.AMBIGUOUS,
                confidence=0.0,
                candidate_customer_ids=candidates,
            )
        return TallantoIdentityResolution(
            customer_id=None,
            match_class=IdentityMatchClass.UNMATCHED,
            identity_status=IdentityStatus.PARTIAL,
            confidence=0.55,
        )

    def _customer_and_link(
        self,
        *,
        contact_id: Optional[str],
        source_ref: str,
        event_at: datetime,
        resolution: TallantoIdentityResolution,
        fallback_ref: str,
    ) -> tuple[CustomerIdentity, Optional[IdentityLink]]:
        customer = CustomerIdentity(
            tenant_id=self.tenant_id,
            customer_id=resolution.customer_id,
            identity_status=resolution.identity_status,
            display_name=None,
            source_ref=f"tallanto:contact:{contact_id}" if contact_id else fallback_ref,
            first_seen_at=event_at,
            last_seen_at=event_at,
            touch_count=1,
            summary={
                "source_system": SOURCE_SYSTEM,
                "source_kind": SOURCE_KIND,
                "identity_resolution": resolution.match_class.value,
            },
            metadata={
                "source_ref": source_ref,
                "contact_id_present": bool(contact_id),
                "candidate_customer_count": len(resolution.candidate_customer_ids),
            },
            created_at=event_at,
            updated_at=event_at,
        )
        if not contact_id:
            return customer, None
        return customer, IdentityLink(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            link_type="tallanto_student_id",
            link_value=contact_id,
            source_system=SOURCE_SYSTEM,
            source_ref=source_ref,
            match_class=resolution.match_class,
            confidence=resolution.confidence,
            evidence={
                "source_kind": SOURCE_KIND,
                "candidate_customer_ids": list(resolution.candidate_customer_ids),
            },
            first_seen_at=event_at,
            last_seen_at=event_at,
        )

    def _conflicts(
        self,
        *,
        contact_id: Optional[str],
        source_ref: str,
        resolution: TallantoIdentityResolution,
    ) -> tuple[Mapping[str, Any], ...]:
        if resolution.match_class != IdentityMatchClass.AMBIGUOUS:
            return ()
        refs = [source_ref]
        if contact_id:
            refs.append(f"tallanto_student_id:{contact_id}")
        refs.extend(f"customer:{item}" for item in resolution.candidate_customer_ids)
        return (
            {
                "tenant_id": self.tenant_id,
                "conflict_type": "tallanto_identity_ambiguous",
                "entity_refs": tuple(refs),
                "severity": "medium",
                "status": "open",
                "summary": "Tallanto contact id is linked to multiple customer identities; no first-match merge",
                "metadata": {
                    "contact_id": contact_id,
                    "candidate_customer_count": len(resolution.candidate_customer_ids),
                    "source_kind": SOURCE_KIND,
                },
            },
        )


class _DryRunStore:
    """Placeholder for TimelineImportService dry-run mode."""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        report = run_tallanto_payments_import(config, stdin_text=sys.stdin.read() if args.source == "-" else None)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - compact CLI error for operators.
        print(f"tallanto payments timeline import failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a read-only Tallanto most_finances/most_abonements JSON snapshot into "
            "customer_timeline.sqlite. Defaults to dry-run; use --apply to write local DB."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to JSON snapshot produced by crm_call.sh, or '-' to read the snapshot from stdin.",
    )
    parser.add_argument("--timeline-db", required=True, help="Target local customer timeline SQLite DB")
    parser.add_argument("--allowed-root", required=True, help="Root that must contain source and timeline DB")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--apply", action="store_true", help="Actually write the local timeline DB")
    parser.add_argument("--actor", default="tallanto_payments_timeline_import")
    parser.add_argument("--source-label", default="crm_call.sh:tallanto_select")
    return parser


def config_from_args(args: argparse.Namespace) -> TallantoPaymentsImportConfig:
    return TallantoPaymentsImportConfig(
        source=None if args.source == "-" else Path(args.source),
        timeline_db=Path(args.timeline_db),
        allowed_root=Path(args.allowed_root),
        tenant_id=args.tenant_id,
        apply=bool(args.apply),
        actor=args.actor,
        source_label=args.source_label,
    )


def run_tallanto_payments_import(
    config: TallantoPaymentsImportConfig,
    *,
    stdin_text: Optional[str] = None,
) -> Mapping[str, Any]:
    snapshot = load_snapshot(config.source, stdin_text=stdin_text)
    records, stats, class_lookup = build_tallanto_records(
        snapshot,
        source_path=str(config.source) if config.source else None,
    )
    contact_ids = {
        str(record.payload.get("contact_id"))
        for record in records
        if optional_text(record.payload.get("contact_id"))
    }
    customer_lookup = load_tallanto_customer_lookup(
        config.timeline_db,
        tenant_id=config.tenant_id,
        contact_ids=contact_ids,
    )
    stats.linked_existing = len(customer_lookup.unique_customer_ids)
    stats.ambiguous_contact_ids = len(customer_lookup.ambiguous_customer_ids)
    stats.unmatched_contact_ids = len(contact_ids - set(customer_lookup.unique_customer_ids) - set(customer_lookup.ambiguous_customer_ids))
    normalizer = TallantoPaymentsTimelineNormalizer(
        tenant_id=config.tenant_id,
        customer_lookup=customer_lookup,
        class_lookup=class_lookup,
    )
    idempotency_key = stable_digest(
        {
            "tenant_id": config.tenant_id,
            "source_system": SOURCE_SYSTEM,
            "source_kind": SOURCE_KIND,
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
                source_ref=config.source_label,
                idempotency_key=idempotency_key,
                dry_run=False,
                actor=config.actor,
            )
            store_summary_after = store.summary()
            stats.bot_safe_amount_leaks = count_bot_safe_amount_leaks(config.timeline_db)
        finally:
            store.close()
    else:
        import_report = TimelineImportService(cast(CustomerTimelineSQLiteStore, _DryRunStore())).import_records(
            records,
            normalizer=normalizer,
            tenant_id=config.tenant_id,
            source_ref=config.source_label,
            idempotency_key=idempotency_key,
            dry_run=True,
            actor=config.actor,
        )
    source_inventory_after = build_source_inventory(records)
    normalized_counts = import_report.normalized_counts
    stats.payment_events = int(normalized_counts.get("events", 0)) - stats.abonement_rows
    stats.abonement_events = stats.abonement_rows
    stats.amount_fields_in_events = stats.payment_rows + stats.abonement_rows
    stats.balance_fields_in_events = stats.abonement_rows
    validation_ok = (
        import_report.validation_ok
        and import_report.source_unchanged
        and stats.bot_safe_amount_leaks == 0
        and safety_ok(timeline_ingestion_safety_contract())
    )
    safety = timeline_ingestion_safety_contract()
    import_report_payload = import_report.to_json_dict()
    import_report_payload["safety"] = {
        **dict(import_report_payload.get("safety") or {}),
        "write_product_timeline_db": config.apply,
    }
    return {
        "schema_version": TALLANTO_PAYMENTS_IMPORT_SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run_preview",
        "dry_run": not config.apply,
        "validation_ok": validation_ok,
        "summary": {
            "validation_ok": validation_ok,
            "status": "completed" if validation_ok else "completed_with_warnings",
            "tenant_id": config.tenant_id,
            "source_system": SOURCE_SYSTEM,
            "source_kind": SOURCE_KIND,
            "records_loaded": len(records),
            "payment_events": stats.payment_events,
            "abonement_events": stats.abonement_events,
            "write_applied": config.apply,
            "writes_applied": sum(int(value) for value in import_report.write_status_counts.values()) if config.apply else 0,
            "source_unchanged": import_report.source_unchanged,
            "bot_safe_amount_leaks": stats.bot_safe_amount_leaks,
            "safety_ok": safety_ok(safety),
        },
        "source": {
            "label": config.source_label,
            "path": str(config.source) if config.source else "stdin",
            "crm_call_sh_expected": True,
            "modules": [PAYMENT_MODULE, ABONEMENT_MODULE, CLASS_MODULE],
            "inventory": {
                "before": list(source_inventory_before),
                "after": list(source_inventory_after),
                "unchanged": source_inventory_before == source_inventory_after,
            },
        },
        "stats": stats.to_json_dict(),
        "links": {
            "unique_existing_tallanto_matches": len(customer_lookup.unique_customer_ids),
            "ambiguous_tallanto_matches": len(customer_lookup.ambiguous_customer_ids),
            "unmatched_tallanto_contact_ids": stats.unmatched_contact_ids,
        },
        "import_report": import_report_payload,
        "store_summary_before": store_summary_before,
        "store_summary_after": store_summary_after,
        "safety": {
            **safety,
            "write_product_timeline_db": config.apply,
            "default_mode": "dry_run_preview",
            "requires_apply_for_db_write": True,
            "blocked_live_actions": blocked_live_actions(),
            "raw_source_payload_in_report": False,
            "bot_safe_payment_amounts": False,
            "ok": safety_ok(safety),
        },
    }


def load_snapshot(source: Optional[Path], *, stdin_text: Optional[str]) -> Mapping[str, Any]:
    if source is None:
        text = stdin_text or ""
        if not text.strip():
            raise ValueError("stdin source is empty")
        payload = json.loads(text)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Tallanto snapshot must be a JSON object")
    return payload


def build_tallanto_records(
    snapshot: Mapping[str, Any],
    *,
    source_path: Optional[str],
) -> tuple[tuple[TimelineSourceRecord, ...], TallantoSnapshotStats, Mapping[str, Mapping[str, Any]]]:
    payment_rows = extract_module_rows(snapshot, PAYMENT_MODULE, aliases=("finances", "payments", "most_finances"))
    abonement_rows = extract_module_rows(snapshot, ABONEMENT_MODULE, aliases=("abonements", "subscriptions", "most_abonements"))
    class_rows = extract_module_rows(snapshot, CLASS_MODULE, aliases=("classes", "lessons", "most_class"))
    class_lookup = {
        require_nonempty(first_value(row, ("id", "class_id", "most_class_id")), "Tallanto class id"): sanitize_source_row(row)
        for row in class_rows
        if optional_text(first_value(row, ("id", "class_id", "most_class_id")))
    }
    records: list[TimelineSourceRecord] = []
    stats = TallantoSnapshotStats(payment_rows=len(payment_rows), abonement_rows=len(abonement_rows), class_rows=len(class_rows))
    for module, rows in ((PAYMENT_MODULE, payment_rows), (ABONEMENT_MODULE, abonement_rows)):
        for idx, row in enumerate(rows, start=1):
            sanitized = sanitize_source_row(row)
            record_id = clean_text(first_value(sanitized, ("id", "finance_id", "payment_id", "abonement_id", "most_abonements_id")))
            if not record_id:
                stats.skipped += 1
                continue
            sanitized["_tallanto_module"] = module
            records.append(
                TimelineSourceRecord(
                    source_system=SOURCE_SYSTEM,
                    source_ref=f"tallanto:{module}:{record_id}",
                    payload=sanitized,
                    source_path=source_path,
                )
            )
    return tuple(records), stats, class_lookup


def extract_module_rows(snapshot: Mapping[str, Any], module: str, *, aliases: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    keys = (module, *aliases)
    for key in keys:
        if key in snapshot:
            rows.extend(extract_rows(snapshot[key], expected_module=module))
    mcp_responses = snapshot.get("mcp_responses")
    if isinstance(mcp_responses, Mapping):
        for key in keys:
            if key in mcp_responses:
                rows.extend(extract_rows(mcp_responses[key], expected_module=module))
    if not rows and str(snapshot.get("module") or "") == module:
        rows.extend(extract_rows(snapshot, expected_module=module))
    seen: set[str] = set()
    unique: list[Mapping[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        key = str(first_value(row, ("id", "finance_id", "payment_id", "abonement_id", "most_abonements_id")) or idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return tuple(unique)


def extract_rows(value: Any, *, expected_module: str) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [sanitize_source_row(item) for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    if "result" in value:
        return extract_rows(value["result"], expected_module=expected_module)
    content = value.get("content")
    if isinstance(content, list):
        rows: list[Mapping[str, Any]] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                try:
                    rows.extend(extract_rows(json.loads(text), expected_module=expected_module))
                except json.JSONDecodeError:
                    continue
            else:
                rows.extend(extract_rows(item, expected_module=expected_module))
        return rows
    if str(value.get("module") or "") and str(value.get("module") or "") != expected_module:
        return []
    for key in ("records", "rows", "items", "data", "entry_list"):
        rows_value = value.get(key)
        if isinstance(rows_value, list):
            return [sanitize_source_row(item) for item in rows_value if isinstance(item, Mapping)]
    return [sanitize_source_row(value)] if "id" in value else []


def sanitize_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = flatten_sugar_row(row)
    allowed_keys = {
        "id",
        "finance_id",
        "payment_id",
        "abonement_id",
        "most_abonements_id",
        "most_class_id",
        "class_id",
        "contact_id",
        "student_id",
        "tallanto_id",
        "Contact_ID",
        "name",
        "cost",
        "payment_summa",
        "amount",
        "summa",
        "discount",
        "date_entered",
        "date_modified",
        "date_payment",
        "event_at",
        "direction",
        "direction_translated",
        "status",
        "payment_status",
        "print_check_status",
        "type",
        "type_translated",
        "num_visit",
        "num_visit_left",
        "start_date",
        "finish_date",
        "filial",
        "cource_name",
        "subject_name",
        "course",
        "group",
        "_tallanto_module",
    }
    return {str(key): value for key, value in source.items() if str(key) in allowed_keys}


def flatten_sugar_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    name_value_list = row.get("name_value_list")
    if not isinstance(name_value_list, Mapping):
        return dict(row)
    result: dict[str, Any] = {}
    if row.get("id"):
        result["id"] = row.get("id")
    for key, value in name_value_list.items():
        if isinstance(value, Mapping) and "value" in value:
            result[str(key)] = value.get("value")
        else:
            result[str(key)] = value
    return result


def load_tallanto_customer_lookup(
    db_path: Path,
    *,
    tenant_id: str,
    contact_ids: set[str],
) -> TallantoCustomerLookup:
    if not contact_ids or not db_path.exists():
        return TallantoCustomerLookup(unique_customer_ids={}, ambiguous_customer_ids={})
    rows: list[tuple[str, str]] = []
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "identity_links"):
            return TallantoCustomerLookup(unique_customer_ids={}, ambiguous_customer_ids={})
        for chunk in chunks(sorted(contact_ids), 800):
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                (str(row["link_value"]), str(row["customer_id"]))
                for row in con.execute(
                    f"""
                    SELECT link_value, customer_id
                    FROM identity_links
                    WHERE tenant_id = ?
                      AND link_type = 'tallanto_student_id'
                      AND link_value IN ({placeholders})
                    """,
                    (tenant_id, *chunk),
                )
                if row["link_value"] and row["customer_id"]
            )
    by_contact: dict[str, set[str]] = {}
    for contact_id, customer_id in rows:
        by_contact.setdefault(contact_id, set()).add(customer_id)
    return TallantoCustomerLookup(
        unique_customer_ids={contact_id: next(iter(customer_ids)) for contact_id, customer_ids in by_contact.items() if len(customer_ids) == 1},
        ambiguous_customer_ids={contact_id: tuple(sorted(customer_ids)) for contact_id, customer_ids in by_contact.items() if len(customer_ids) > 1},
    )


def count_bot_safe_amount_leaks(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "bot_context_chunks"):
            return 0
        row = con.execute(
            """
            SELECT COUNT(*) AS count
            FROM bot_context_chunks
            WHERE allowed_for_bot = 1
              AND (
                record_json LIKE '%"amount"%'
                OR record_json LIKE '%"cost"%'
                OR record_json LIKE '%"visits_left"%'
                OR record_json LIKE '%"num_visit_left"%'
              )
            """
        ).fetchone()
    return int(row["count"]) if row else 0


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


def participant_for_contact(contact_id: Optional[str]) -> tuple[TimelineParticipant, ...]:
    if not contact_id:
        return ()
    return (TimelineParticipant(role="client", ref=contact_id, channel="tallanto"),)


def first_value(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def require_nonempty(value: Any, field_name: str) -> str:
    text = clean_text(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def clean_text(value: Any) -> Optional[str]:
    text = optional_text(value)
    if not text:
        return None
    return " ".join(text.split())


def money_value(value: Any) -> Optional[int | float]:
    number = decimal_value(value)
    if number is None:
        return None
    if number == number.to_integral_value():
        return int(number)
    return float(number)


def numeric_value(value: Any) -> Optional[int | float]:
    return money_value(value)


def decimal_value(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value).replace(" ", "").replace(",", "."))
    except (InvalidOperation, ValueError):
        return None


def flatten_label(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        labels = [clean_text(item) for item in value.values()]
        return ", ".join(item for item in labels if item) or None
    if isinstance(value, list):
        labels = [clean_text(item) for item in value]
        return ", ".join(item for item in labels if item) or None
    return clean_text(value)


def compact_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if item in (None, "", [], {}):
            continue
        result[key] = item
    return result


if __name__ == "__main__":
    raise SystemExit(main())
