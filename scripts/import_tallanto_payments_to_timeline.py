#!/usr/bin/env python3
"""Import Tallanto payments/abonements from a local snapshot or read-only API.

Use --apply to write only the local staging Customer Timeline DB. The importer
never writes to Tallanto and never calls ASR or an LLM.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast

from mango_mvp.customer_timeline.contracts import (
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
from mango_mvp.customer_timeline.ids import normalize_key, optional_text, stable_digest, stable_prefixed_id
from mango_mvp.customer_timeline.import_cli import safety_ok
from mango_mvp.customer_timeline.ingestion import (
    TimelineImportReport,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineSourceRecord,
    TALLANTO_TIMEZONE,
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
    authoritative_exact_identity_rows,
    customer_timeline_readonly_uri,
    guard_customer_timeline_sqlite_path,
)


TALLANTO_PAYMENTS_IMPORT_SCHEMA_VERSION = "tallanto_payments_timeline_import_v1"
MONEY_PRODUCT_BACKFILL_VERSION = "abonement_titles_attendance_class_v1"
SOURCE_SYSTEM = "tallanto_crm_call"
SOURCE_KIND = "tallanto_readonly_mcp_snapshot"
PAYMENT_MODULE = "most_finances"
ABONEMENT_MODULE = "most_abonements"
CLASS_MODULE = "most_class"
MONEY_SELECT_FIELDS = {
    PAYMENT_MODULE: (
        "id", "contact_id", "name", "cost", "date_payment", "date_modified",
        "direction", "direction_translated", "type", "type_translated",
        "print_check_status", "payment_status", "status", "most_abonements_id", "most_class_id",
    ),
    ABONEMENT_MODULE: (
        "id", "contact_id", "name", "cost", "discount", "num_visit", "num_visit_left",
        "start_date", "finish_date", "date_modified", "type", "type_translated", "status",
        "rate_translated", "category_translated", "form", "form_translated", "filial",
    ),
}


@dataclass(frozen=True)
class TallantoPaymentsImportConfig:
    source: Optional[Path]
    timeline_db: Path
    allowed_root: Path
    tenant_id: str
    apply: bool = False
    allow_test_paths: bool = False
    actor: str = "tallanto_payments_timeline_import"
    source_label: str = "crm_call.sh:tallanto_select"

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        source = guard_customer_timeline_source_path(Path(self.source).expanduser(), root) if self.source else None
        timeline_db = guard_customer_timeline_sqlite_path(Path(self.timeline_db).expanduser())
        timeline_db = guard_customer_timeline_output_path(timeline_db, root)
        if self.apply:
            assert_tallanto_import_apply_staging_path(timeline_db, root, allow_test_paths=self.allow_test_paths)
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
    unresolved_payment_owners: int = 0
    amount_fields_in_events: int = 0
    balance_fields_in_events: int = 0
    bot_safe_amount_leaks: int = 0
    local_unowned_payment_retries: int = 0
    local_unowned_abonement_retries: int = 0
    local_weak_payment_retries: int = 0
    local_weak_abonement_retries: int = 0
    local_product_payment_retries: int = 0
    local_product_abonement_retries: int = 0
    payment_titles_resolved: int = 0
    payment_titles_unresolved: int = 0
    payment_class_relations_resolved: int = 0
    payment_class_relations_unresolved: int = 0
    abonement_titles_resolved: int = 0
    abonement_titles_unresolved: int = 0
    abonement_class_relations_resolved: int = 0
    abonement_class_relations_unresolved: int = 0

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
            "unresolved_payment_owners": self.unresolved_payment_owners,
            "amount_fields_in_events": self.amount_fields_in_events,
            "balance_fields_in_events": self.balance_fields_in_events,
            "bot_safe_amount_leaks": self.bot_safe_amount_leaks,
            "local_unowned_payment_retries": self.local_unowned_payment_retries,
            "local_unowned_abonement_retries": self.local_unowned_abonement_retries,
            "local_weak_payment_retries": self.local_weak_payment_retries,
            "local_weak_abonement_retries": self.local_weak_abonement_retries,
            "local_product_payment_retries": self.local_product_payment_retries,
            "local_product_abonement_retries": self.local_product_abonement_retries,
            "payment_titles_resolved": self.payment_titles_resolved,
            "payment_titles_unresolved": self.payment_titles_unresolved,
            "payment_class_relations_resolved": self.payment_class_relations_resolved,
            "payment_class_relations_unresolved": self.payment_class_relations_unresolved,
            "abonement_titles_resolved": self.abonement_titles_resolved,
            "abonement_titles_unresolved": self.abonement_titles_unresolved,
            "abonement_class_relations_resolved": self.abonement_class_relations_resolved,
            "abonement_class_relations_unresolved": self.abonement_class_relations_unresolved,
        }


@dataclass(frozen=True)
class TallantoCustomerLookup:
    unique_customer_ids: Mapping[str, str]
    unique_match_classes: Mapping[str, IdentityMatchClass]
    ambiguous_customer_ids: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unique_customer_ids", dict(self.unique_customer_ids))
        object.__setattr__(
            self,
            "unique_match_classes",
            {key: IdentityMatchClass(value) for key, value in self.unique_match_classes.items()},
        )
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
        opportunity_lookup: Mapping[str, str] = (),
    ) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")
        self._customer_lookup = customer_lookup
        self._class_lookup = {str(key): dict(value) for key, value in class_lookup.items()}
        self._opportunity_lookup = dict(opportunity_lookup)

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
        abonement_contact_id = clean_text(payload.get("_abonement_contact_id"))
        abonement_id = clean_text(first_value(payload, ("most_abonements_id", "abonement_id", "subscription_id")))
        class_id = clean_text(first_value(payload, ("most_class_id", "class_id")))
        event_at = parse_source_datetime(
            first_value(payload, ("date_payment", "date_entered", "date_modified", "event_at")),
            record.observed_at,
            naive_timezone=TALLANTO_TIMEZONE,
        )
        unresolved_owner = payload.get("_contact_id_source") in {"unresolved_abonement", "unresolved_owner"}
        resolution = TallantoIdentityResolution(
            customer_id=None,
            match_class=IdentityMatchClass.AMBIGUOUS,
            identity_status=IdentityStatus.AMBIGUOUS,
            confidence=0.0,
        ) if unresolved_owner else (
            self._resolve_conflicting_contacts(contact_id, abonement_contact_id)
            if contact_id and abonement_contact_id and contact_id != abonement_contact_id
            else self._resolve_contact(contact_id)
        )
        source_ref = f"tallanto:{PAYMENT_MODULE}:{payment_id}"
        link = self._identity_link(
            contact_id=contact_id,
            source_ref=source_ref,
            event_at=event_at,
            resolution=resolution,
        )
        customer_id = resolution.customer_id
        amount = money_value(first_value(payload, ("cost", "payment_summa", "amount", "summa")))
        direction = clean_text(first_value(payload, ("direction", "direction_translated")))
        status = clean_text(first_value(payload, ("print_check_status", "payment_status", "status")))
        payment_type = clean_text(first_value(payload, ("type_translated", "type")))
        class_info = self._class_lookup.get(class_id or "", {})
        class_name = (
            first_value(class_info, ("name", "cource_name"))
            or first_value(payload, ("_abonement_name", "class_name", "name"))
        )
        title = compact_text(
            first_value(class_info, ("name", "cource_name", "subject_name"))
            or first_value(payload, ("_abonement_name", "name"))
            or "Tallanto payment",
            limit=160,
        )
        opportunity = CustomerOpportunity(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_type=OpportunityType.TALLANTO_COURSE,
            opportunity_id=self._opportunity_id(f"payment:{payment_id}"),
            source_system=SOURCE_SYSTEM,
            source_id=f"payment:{payment_id}",
            title=title,
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
                    "class_name": class_name,
                    "subject_name": first_value(class_info, ("subject_name",)),
                }
            ),
            opened_at=event_at,
            confidence=resolution.confidence,
            evidence={"source_ref": source_ref, "raw_payload_persisted": False},
        ) if customer_id else None
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id if opportunity else None,
            event_type=TimelineEventType.TALLANTO_PAYMENT,
            event_at=event_at,
            source_system=SOURCE_SYSTEM,
            source_id=f"{PAYMENT_MODULE}:{payment_id}",
            source_ref=source_ref,
            direction=TimelineDirection.SYSTEM,
            participants=participant_for_contact(contact_id),
            subject=title,
            text_preview=title,
            summary=compact_text(direction or status or "Tallanto finance operation imported", limit=240),
            match_status=resolution.match_class,
            confidence=resolution.confidence,
            record=compact_mapping(
                {
                    "module": PAYMENT_MODULE,
                    "payment_id": payment_id,
                    "contact_id": contact_id,
                    "contact_id_source": payload.get("_contact_id_source") or ("direct" if contact_id else "missing"),
                    "contact_id_conflict": bool(contact_id and abonement_contact_id and contact_id != abonement_contact_id),
                    "amount": amount,
                    "currency": "RUB" if amount is not None else None,
                    "payment_direction": direction,
                    "payment_status": status,
                    "payment_type": payment_type,
                    "abonement_id": abonement_id,
                    "class_id": class_id,
                    "class_name": class_name,
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
        conflicts = self._conflicts(
            contact_id=contact_id,
            alternate_contact_id=abonement_contact_id,
            abonement_id=abonement_id,
            source_ref=source_ref,
            resolution=resolution,
        )
        return TimelineNormalizedBatch(
            source_record=record,
            identity_links=(link,) if link else (),
            opportunities=(opportunity,) if opportunity else (),
            events=(event,),
            conflicts=conflicts,
            retired_opportunity_ids=() if opportunity else (self._opportunity_id(f"payment:{payment_id}"),),
        )

    def _normalize_abonement(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = dict(record.payload)
        abonement_id = require_nonempty(first_value(payload, ("id", "abonement_id", "most_abonements_id")), "Tallanto abonement id")
        contact_id = clean_text(first_value(payload, ("contact_id", "student_id", "tallanto_id", "Contact_ID")))
        class_id = clean_text(first_value(payload, ("most_class_id", "class_id")))
        class_info = self._class_lookup.get(class_id or "", {})
        event_at = parse_source_datetime(
            first_value(payload, ("date_modified", "date_entered", "start_date", "finish_date", "event_at")),
            record.observed_at,
            naive_timezone=TALLANTO_TIMEZONE,
        )
        resolution = self._resolve_contact(contact_id)
        source_ref = f"tallanto:{ABONEMENT_MODULE}:{abonement_id}"
        link = self._identity_link(
            contact_id=contact_id,
            source_ref=source_ref,
            event_at=event_at,
            resolution=resolution,
        )
        customer_id = resolution.customer_id
        amount = money_value(first_value(payload, ("cost", "price", "amount")))
        discount = money_value(first_value(payload, ("discount",)))
        visits_total = numeric_value(first_value(payload, ("num_visit",)))
        visits_left = numeric_value(first_value(payload, ("num_visit_left", "visits_left")))
        status = clean_text(first_value(payload, ("status", "type_translated", "type", "form_translated", "form")))
        title = compact_text(
            first_value(class_info, ("name", "cource_name", "subject_name"))
            or first_value(payload, ("name", "rate_translated", "category_translated"))
            or "Tallanto abonement",
            limit=160,
        )
        opportunity = CustomerOpportunity(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_type=OpportunityType.TALLANTO_COURSE,
            opportunity_id=self._opportunity_id(f"abonement:{abonement_id}"),
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
                    "class_id": class_id,
                    "class_name": first_value(class_info, ("name", "cource_name")),
                    "subject_name": first_value(class_info, ("subject_name",)),
                    "currency": "RUB" if amount is not None else None,
                }
            ),
            opened_at=event_at,
            confidence=resolution.confidence,
            evidence={"source_ref": source_ref, "raw_payload_persisted": False},
        ) if customer_id else None
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id if opportunity else None,
            event_type=TimelineEventType.TALLANTO_ABONEMENT,
            event_at=event_at,
            source_system=SOURCE_SYSTEM,
            source_id=f"{ABONEMENT_MODULE}:{abonement_id}",
            source_ref=source_ref,
            direction=TimelineDirection.SYSTEM,
            participants=participant_for_contact(contact_id),
            subject=title,
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
                    "class_id": class_id,
                    "class_name": first_value(class_info, ("name", "cource_name")),
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
            identity_links=(link,) if link else (),
            opportunities=(opportunity,) if opportunity else (),
            events=(event,),
            conflicts=conflicts,
            retired_opportunity_ids=() if opportunity else (self._opportunity_id(f"abonement:{abonement_id}"),),
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
                match_class=self._customer_lookup.unique_match_classes[contact_id],
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

    def _resolve_conflicting_contacts(self, *contact_ids: Optional[str]) -> TallantoIdentityResolution:
        candidates: set[str] = set()
        for contact_id in filter(None, contact_ids):
            existing = self._customer_lookup.unique_customer_ids.get(str(contact_id))
            if existing:
                candidates.add(existing)
            candidates.update(self._customer_lookup.ambiguous_customer_ids.get(str(contact_id), ()))
        return TallantoIdentityResolution(
            customer_id=None,
            match_class=IdentityMatchClass.AMBIGUOUS,
            identity_status=IdentityStatus.AMBIGUOUS,
            confidence=0.0,
            candidate_customer_ids=tuple(sorted(candidates)),
        )

    def _opportunity_id(self, source_id: str) -> str:
        return self._opportunity_lookup.get(source_id) or stable_tallanto_opportunity_id(self.tenant_id, source_id)

    def _identity_link(
        self,
        *,
        contact_id: Optional[str],
        source_ref: str,
        event_at: datetime,
        resolution: TallantoIdentityResolution,
    ) -> Optional[IdentityLink]:
        if not contact_id:
            return None
        return IdentityLink(
            tenant_id=self.tenant_id,
            customer_id=resolution.customer_id,
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
        alternate_contact_id: Optional[str] = None,
        abonement_id: Optional[str] = None,
        source_ref: str,
        resolution: TallantoIdentityResolution,
    ) -> tuple[Mapping[str, Any], ...]:
        is_payment = source_ref.startswith(f"tallanto:{PAYMENT_MODULE}:")
        if resolution.customer_id:
            return ()
        if resolution.match_class == IdentityMatchClass.UNMATCHED and not is_payment:
            return ()
        if resolution.match_class != IdentityMatchClass.AMBIGUOUS and not is_payment:
            return ()
        unresolved_payment = is_payment and not resolution.candidate_customer_ids
        refs = [source_ref]
        if abonement_id:
            refs.append(f"tallanto_abonement_id:{abonement_id}")
        if not unresolved_payment:
            if contact_id:
                refs.append(f"tallanto_student_id:{contact_id}")
            if alternate_contact_id and alternate_contact_id != contact_id:
                refs.append(f"tallanto_student_id:{alternate_contact_id}")
            refs.extend(f"customer:{item}" for item in resolution.candidate_customer_ids)
        return (
            {
                "tenant_id": self.tenant_id,
                "conflict_type": (
                    "tallanto_payment_owner_unresolved" if unresolved_payment else "tallanto_identity_ambiguous"
                ),
                "entity_refs": tuple(refs),
                "severity": "medium",
                "status": "open",
                "summary": (
                    "Tallanto payment owner is unresolved and must be retried after identity updates"
                    if unresolved_payment
                    else "Tallanto contact id is linked to multiple customer identities; no first-match merge"
                ),
                "metadata": {
                    "contact_id": contact_id,
                    "alternate_contact_id": alternate_contact_id,
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
        if args.tallanto_api_env:
            report = run_tallanto_money_api_increment(
                config,
                env_file=Path(args.tallanto_api_env),
            )
        else:
            report = run_tallanto_payments_import(
                config,
                stdin_text=sys.stdin.read() if args.source == "-" else None,
            )
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
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        help="Path to JSON snapshot produced by crm_call.sh, or '-' to read the snapshot from stdin.",
    )
    source_group.add_argument(
        "--tallanto-api-env",
        help="Read-only CRM_TALLANTO_* env file; API rows stay in memory and are not persisted raw.",
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
        source=None if args.source in (None, "-") else Path(args.source),
        timeline_db=Path(args.timeline_db),
        allowed_root=Path(args.allowed_root),
        tenant_id=args.tenant_id,
        apply=bool(args.apply),
        actor=args.actor,
        source_label=("tallanto_api:get_entry_list" if args.tallanto_api_env else args.source_label),
    )


def run_tallanto_money_api_increment(
    config: TallantoPaymentsImportConfig,
    *,
    env_file: Path,
    client: Any = None,
) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    previous = _latest_money_sync(config.timeline_db, config.tenant_id)
    product_backfill = _money_product_backfill_required(config.timeline_db, config.tenant_id, previous)
    api_mode = "product_backfill" if product_backfill else ("incremental" if previous else "full")
    fetch_from = previous - timedelta(minutes=5) if previous else None
    upper_bound = started.astimezone(TALLANTO_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    if client is None:
        from mango_mvp.integrations.amo_wappi_phase1 import load_env_file

        load_env_file(env_file, override=True)
        from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, build_tallanto_api_config

        client = TallantoApiClient(build_tallanto_api_config())
    snapshot: dict[str, list[Mapping[str, Any]]] = {}
    api_stats: dict[str, Mapping[str, Any]] = {}
    module_modes: dict[str, str] = {}
    fetch_started = time.monotonic()
    for module in (PAYMENT_MODULE, ABONEMENT_MODULE):
        module_fetch_from = None if product_backfill else fetch_from
        module_modes[module] = "full" if module_fetch_from is None else "incremental"
        module_query = (
            f"{module}.date_modified >= '{module_fetch_from.astimezone(TALLANTO_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}' "
            f"AND {module}.date_modified <= '{upper_bound}'"
            if module_fetch_from
            else f"{module}.date_modified <= '{upper_bound}'"
        )
        rows, stats = fetch_tallanto_module_strict(
            client,
            module=module,
            query=module_query,
            select_fields=MONEY_SELECT_FIELDS[module],
        )
        snapshot[module] = list(rows)
        api_stats[module] = {**stats, "mode": module_modes[module]}
    class_ids = _class_ids_needing_lookup(snapshot, config.timeline_db, config.tenant_id)
    if class_ids:
        class_rows, class_stats = fetch_tallanto_module_ids_strict(
            client,
            module=CLASS_MODULE,
            ids=class_ids,
            select_fields=("id", "name", "cource_name", "subject_name"),
        )
        snapshot[CLASS_MODULE] = list(class_rows)
        api_stats[CLASS_MODULE] = {**class_stats, "mode": "id_batches"}
    import_started = time.monotonic()
    report = dict(
        run_tallanto_payments_import(
            config,
            stdin_text=json.dumps(snapshot, ensure_ascii=False),
            retry_products=product_backfill,
        )
    )
    if report.get("validation_ok") and config.apply:
        with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root) as store:
            store.upsert_ingestion_cursor(
                config.tenant_id,
                "tallanto_money_api",
                last_cursor_ts=started,
                metadata={
                    "last_status": "ok",
                    "mode": api_mode,
                    "product_backfill_version": MONEY_PRODUCT_BACKFILL_VERSION,
                },
                actor=config.actor,
            )
    report["api"] = {
        "modules": api_stats,
        "raw_payload_persisted": False,
        "full_rescan": any(mode == "full" for mode in module_modes.values()),
        "performance": {
            "mode": api_mode,
            "module_modes": module_modes,
            "rows": {"fetched": sum(item["records"] for item in api_stats.values())},
            "seconds": {
                "fetch": round(import_started - fetch_started, 3),
                "write": round(time.monotonic() - import_started, 3),
                "total": round(time.monotonic() - fetch_started, 3),
            },
            "pages": sum(item["pages"] for item in api_stats.values()),
            "cursor_used": previous is not None,
            "product_backfill": product_backfill,
        },
    }
    report["source"] = {
        **dict(report.get("source") or {}),
        "label": "tallanto_api:get_entry_list",
        "path": None,
        "crm_call_sh_expected": False,
    }
    report["safety"] = {
        **dict(report.get("safety") or {}),
        "network_calls": True,
        "write_tallanto": False,
        "raw_source_payload_in_report": False,
    }
    return report


def fetch_tallanto_module_strict(
    client: Any,
    *,
    module: str,
    query: str | None = None,
    select_fields: Sequence[str] | None = None,
    page_delay_seconds: float = 0.0,
    rate_limit_wait_seconds: float = 30.0,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, int]]:
    rows: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    offset = 0
    pages = 0
    expected_total: int | None = None
    while True:
        if pages >= 10_000:
            raise ValueError(f"Tallanto {module} pagination exceeded 10000 pages")
        for rate_limit_attempt in range(3):
            try:
                payload = client.get_entry_list(
                    module=module,
                    select_fields=select_fields,
                    query=query,
                    order_by="date_modified ASC,id ASC",
                    offset=offset,
                )
                break
            except ValueError as exc:
                if getattr(exc, "category", None) != "rate_limited" or rate_limit_attempt == 2:
                    raise
                time.sleep(rate_limit_wait_seconds * (rate_limit_attempt + 1))
        pages += 1
        raw_rows = payload.get("entry_list")
        if not isinstance(raw_rows, list):
            raise ValueError(f"Tallanto {module} response misses entry_list")
        total_raw = payload.get("total_count")
        if total_raw in (None, ""):
            raise ValueError(f"Tallanto {module} response misses total_count")
        total = int(total_raw)
        if total < 0:
            raise ValueError(f"Tallanto {module} returned negative total_count")
        if expected_total is None:
            expected_total = total
        elif expected_total != total:
            raise ValueError(f"Tallanto {module} total_count changed during pagination")
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"Tallanto {module} returned a non-object row")
            row = sanitize_source_row(raw_row)
            row_id = clean_text(
                first_value(row, ("id", "finance_id", "payment_id", "abonement_id", "most_abonements_id"))
            )
            if not row_id or row_id in seen_ids:
                raise ValueError(f"Tallanto {module} pagination returned missing or duplicate id")
            seen_ids.add(row_id)
            rows.append(row)
        if len(rows) == expected_total:
            break
        next_raw = payload.get("next_offset")
        if next_raw in (None, ""):
            break
        next_offset = int(next_raw)
        if next_offset <= offset:
            raise ValueError(f"Tallanto {module} pagination did not advance")
        offset = next_offset
        if page_delay_seconds > 0:
            time.sleep(page_delay_seconds)
    if len(rows) != expected_total:
        raise ValueError(
            f"Tallanto {module} pagination incomplete: expected {expected_total}, got {len(rows)}"
        )
    return tuple(rows), {"pages": pages, "records": len(rows)}


def fetch_tallanto_module_ids_strict(
    client: Any,
    *,
    module: str,
    ids: set[str],
    select_fields: Sequence[str],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, int]]:
    rows: list[Mapping[str, Any]] = []
    pages = 0
    for batch in chunks(sorted(ids), 100):
        fetched, stats = fetch_tallanto_module_strict(
            client,
            module=module,
            query=f"{module}.id IN (" + ",".join(f"'{_api_quote(value)}'" for value in batch) + ")",
            select_fields=select_fields,
        )
        unexpected = {
            clean_text(first_value(row, ("id", f"{module}_id"))) for row in fetched
        } - set(batch)
        if unexpected:
            raise ValueError(f"Tallanto {module} batch returned an unrequested id")
        rows.extend(fetched)
        pages += int(stats["pages"])
    return tuple(rows), {"pages": pages, "records": len(rows), "batches": len(chunks(sorted(ids), 100))}


def _api_quote(value: str) -> str:
    return value.replace("'", "''")


def _latest_money_sync(db_path: Path, tenant_id: str) -> datetime | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as con:
            row = con.execute(
                "SELECT last_cursor_ts FROM ingestion_cursors WHERE tenant_id=? AND source_system='tallanto_money_api'",
                (tenant_id,),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    return datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")) if row else None


def _money_product_backfill_required(
    db_path: Path,
    tenant_id: str,
    previous: datetime | None,
) -> bool:
    if previous is None or not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as con:
            row = con.execute(
                "SELECT metadata_json FROM ingestion_cursors "
                "WHERE tenant_id=? AND source_system='tallanto_money_api'",
                (normalize_key(tenant_id, "tenant_id"),),
            ).fetchone()
    except sqlite3.OperationalError:
        return False
    if row is None:
        return True
    try:
        payload = json.loads(str(row[0] or "{}"))
    except json.JSONDecodeError:
        return True
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    state = metadata if isinstance(metadata, Mapping) else payload
    return not isinstance(state, Mapping) or state.get("product_backfill_version") != MONEY_PRODUCT_BACKFILL_VERSION


def _class_ids_needing_lookup(
    snapshot: Mapping[str, Any],
    db_path: Path,
    tenant_id: str,
) -> set[str]:
    class_ids = {
        value
        for row in extract_module_rows(snapshot, PAYMENT_MODULE, aliases=("finances", "payments"))
        if (value := clean_text(first_value(row, ("most_class_id", "class_id"))))
    }
    if not db_path.exists():
        return class_ids
    try:
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as con:
            rows = con.execute(
                "SELECT DISTINCT json_extract(record_json,'$.record.class_id') FROM timeline_events "
                "WHERE tenant_id=? AND source_system=? AND event_type='tallanto_payment' "
                "AND coalesce(superseded_by,'')='' "
                "AND coalesce(json_extract(record_json,'$.record.class_id'),'')!='' "
                "AND (coalesce(json_extract(record_json,'$.record.class_name'),'')='' "
                "OR subject='Tallanto payment')",
                (normalize_key(tenant_id, "tenant_id"), SOURCE_SYSTEM),
            ).fetchall()
            known_rows = con.execute(
                "SELECT DISTINCT json_extract(record_json,'$.record.class_id') FROM timeline_events "
                "WHERE tenant_id=? AND source_system=? AND event_type='tallanto_payment' "
                "AND coalesce(superseded_by,'')='' "
                "AND coalesce(json_extract(record_json,'$.record.class_id'),'')!='' "
                "AND coalesce(json_extract(record_json,'$.record.class_name'),'')!='' "
                "AND subject!='Tallanto payment'",
                (normalize_key(tenant_id, "tenant_id"), SOURCE_SYSTEM),
            ).fetchall()
    except sqlite3.OperationalError:
        return class_ids
    class_ids.update(clean_text(row[0]) for row in rows if clean_text(row[0]))
    known_ids = {clean_text(row[0]) for row in known_rows if clean_text(row[0])}
    return class_ids - known_ids


def assert_tallanto_import_apply_staging_path(path: Path, allowed_root: Path, *, allow_test_paths: bool = False) -> None:
    resolved = path.resolve(strict=False)
    if any("customer_timeline_prod_" in part for part in resolved.parts):
        raise ValueError(f"refusing to apply Tallanto import to prod timeline path: {resolved}")
    if allow_test_paths:
        return
    root = allowed_root.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Tallanto import DB must stay under allowed root: {root}") from exc
    parts = tuple(part.casefold() for part in resolved.parts)
    if not any(part == ".codex_local" and parts[index + 1] == "staging" for index, part in enumerate(parts[:-1])):
        raise ValueError("Tallanto import apply is allowed only for .codex_local/staging paths")


def include_local_unowned_payment_rows(
    snapshot: Mapping[str, Any],
    *,
    db_path: Path,
    tenant_id: str,
    retry_products: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, int]]:
    """Retry local money rows whose identity or product evidence can now improve."""
    payment_rows = list(extract_module_rows(snapshot, PAYMENT_MODULE, aliases=("finances", "payments")))
    abonement_rows = list(
        extract_module_rows(snapshot, ABONEMENT_MODULE, aliases=("abonements", "subscriptions"))
    )
    current_payment_ids = {
        clean_text(first_value(row, ("id", "finance_id", "payment_id")))
        for row in payment_rows
    }
    current_abonement_ids = {
        clean_text(first_value(row, ("id", "abonement_id", "most_abonements_id")))
        for row in abonement_rows
    }
    abonement_contacts = dict(load_existing_abonement_contacts(db_path, tenant_id))
    abonement_contacts.update({
        abonement_id: contact_id
        for row in abonement_rows
        if (abonement_id := clean_text(first_value(row, ("id", "abonement_id", "most_abonements_id"))))
        and (contact_id := clean_text(first_value(row, ("contact_id", "student_id", "tallanto_id"))))
    })

    def retry_contact(record: Mapping[str, Any]) -> str:
        return clean_text(record.get("contact_id")) or abonement_contacts.get(
            clean_text(record.get("abonement_id")), ""
        )
    payment_retries: list[Mapping[str, Any]] = []
    abonement_retries: list[Mapping[str, Any]] = []
    retry_counts = {
        "unowned_payment": 0,
        "unowned_abonement": 0,
        "weak_payment": 0,
        "weak_abonement": 0,
        "product_payment": 0,
        "product_abonement": 0,
    }
    if db_path.exists():
        try:
            with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as con:
                product_condition = (
                    " OR (e.event_type='tallanto_payment' AND e.subject='Tallanto payment') "
                    "OR (e.event_type='tallanto_abonement' AND e.subject='Tallanto abonement')"
                    if retry_products
                    else ""
                )
                rows = con.execute(
                    "SELECT event_type,event_at,subject,record_json,e.customer_id,e.match_status,"
                    "CASE WHEN e.customer_id IS NULL OR EXISTS ("
                    "SELECT 1 FROM customer_identities c WHERE c.tenant_id=e.tenant_id "
                    "AND c.customer_id=e.customer_id AND coalesce(c.display_name,'')='' "
                    "AND coalesce(c.primary_phone,'')='' AND coalesce(c.primary_email,'')='' "
                    "AND json_extract(c.record_json,'$.summary.source_system')=? "
                    "AND NOT EXISTS (SELECT 1 FROM identity_links l WHERE l.tenant_id=c.tenant_id "
                    "AND l.customer_id=c.customer_id AND (l.source_system!=? "
                    "OR l.match_class IN ('strong_unique','manual')))) THEN 'unowned' "
                    "WHEN e.match_status NOT IN ('strong_unique','manual') THEN 'weak' "
                    "ELSE 'product' END AS retry_kind FROM timeline_events e "
                    "WHERE e.tenant_id=? AND e.source_system=? "
                    "AND e.event_type IN ('tallanto_payment','tallanto_abonement') "
                    "AND coalesce(e.superseded_by,'')='' AND (e.customer_id IS NULL "
                    "OR e.match_status NOT IN ('strong_unique','manual') OR EXISTS ("
                    "SELECT 1 FROM customer_identities c WHERE c.tenant_id=e.tenant_id "
                    "AND c.customer_id=e.customer_id AND coalesce(c.display_name,'')='' "
                    "AND coalesce(c.primary_phone,'')='' AND coalesce(c.primary_email,'')='' "
                    "AND json_extract(c.record_json,'$.summary.source_system')=? "
                    "AND NOT EXISTS (SELECT 1 FROM identity_links l WHERE l.tenant_id=c.tenant_id "
                    "AND l.customer_id=c.customer_id AND (l.source_system!=? "
                    "OR l.match_class IN ('strong_unique','manual')))) "
                    + product_condition + ")",
                    (
                        SOURCE_SYSTEM,
                        SOURCE_SYSTEM,
                        normalize_key(tenant_id, "tenant_id"),
                        SOURCE_SYSTEM,
                        SOURCE_SYSTEM,
                        SOURCE_SYSTEM,
                    ),
                ).fetchall()
        except sqlite3.OperationalError:
            rows = ()
        parsed_rows = [
            (*row, dict((json.loads(str(row[3] or "{}"))).get("record") or {}))
            for row in rows
        ]
        retry_identity = load_tallanto_customer_lookup(
            db_path,
            tenant_id=tenant_id,
            contact_ids={
                contact_id
                for *_, retry_kind, record in parsed_rows
                if retry_kind in {"unowned", "weak"}
                and (contact_id := retry_contact(record))
            },
        )
        for event_type, event_at, subject, _record_json, customer_id, _match_status, retry_kind, record in parsed_rows:
            if retry_kind in {"unowned", "weak"}:
                contact_id = retry_contact(record)
                improved_owner = retry_identity.unique_customer_ids.get(contact_id)
                if not improved_owner or (retry_kind == "weak" and improved_owner == customer_id and _match_status in {"strong_unique", "manual"}):
                    continue
            if event_type == "tallanto_payment":
                payment_id = clean_text(record.get("payment_id"))
                if not payment_id or payment_id in current_payment_ids:
                    continue
                payment_retries.append(
                    compact_mapping(
                        {
                            "id": payment_id,
                            "contact_id": retry_contact(record),
                            "cost": record.get("amount"),
                            "direction": record.get("payment_direction"),
                            "payment_status": record.get("payment_status"),
                            "type": record.get("payment_type"),
                            "most_abonements_id": record.get("abonement_id"),
                            "most_class_id": record.get("class_id"),
                            "name": subject,
                            "class_name": record.get("class_name") or subject,
                            "event_at": event_at,
                        }
                    )
                )
                retry_counts[f"{retry_kind}_payment"] += 1
                continue
            abonement_id = clean_text(record.get("abonement_id"))
            if not abonement_id or abonement_id in current_abonement_ids:
                continue
            abonement_retries.append(
                compact_mapping(
                    {
                        "id": abonement_id,
                        "contact_id": record.get("contact_id"),
                        "name": subject,
                        "cost": record.get("amount"),
                        "discount": record.get("discount"),
                        "num_visit": record.get("visits_total"),
                        "num_visit_left": record.get("visits_left"),
                        "start_date": record.get("start_date"),
                        "finish_date": record.get("finish_date"),
                        "filial": record.get("filial"),
                        "event_at": event_at,
                    }
                )
            )
            retry_counts[f"{retry_kind}_abonement"] += 1
    return {
        PAYMENT_MODULE: [*payment_rows, *payment_retries],
        ABONEMENT_MODULE: [*abonement_rows, *abonement_retries],
        CLASS_MODULE: list(extract_module_rows(snapshot, CLASS_MODULE, aliases=("classes", "lessons"))),
    }, retry_counts


def run_tallanto_payments_import(
    config: TallantoPaymentsImportConfig,
    *,
    stdin_text: Optional[str] = None,
    retry_products: bool = False,
) -> Mapping[str, Any]:
    snapshot = load_snapshot(config.source, stdin_text=stdin_text)
    snapshot, retry_counts = include_local_unowned_payment_rows(
        snapshot,
        db_path=config.timeline_db,
        tenant_id=config.tenant_id,
        retry_products=retry_products,
    )
    existing_class_lookup, existing_abonement_classes = load_existing_money_class_context(
        config.timeline_db, config.tenant_id
    )
    records, stats, class_lookup = build_tallanto_records(
        snapshot,
        source_path=str(config.source) if config.source else None,
        existing_abonement_contacts=load_existing_abonement_contacts(config.timeline_db, config.tenant_id),
        existing_class_lookup=existing_class_lookup,
        existing_abonement_classes=existing_abonement_classes,
    )
    stats.local_unowned_payment_retries = retry_counts["unowned_payment"]
    stats.local_unowned_abonement_retries = retry_counts["unowned_abonement"]
    stats.local_weak_payment_retries = retry_counts["weak_payment"]
    stats.local_weak_abonement_retries = retry_counts["weak_abonement"]
    stats.local_product_payment_retries = retry_counts["product_payment"]
    stats.local_product_abonement_retries = retry_counts["product_abonement"]
    contact_ids = {
        str(record.payload.get(key))
        for record in records
        for key in ("contact_id", "_abonement_contact_id")
        if optional_text(record.payload.get(key))
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
        opportunity_lookup=load_tallanto_opportunity_lookup(
            config.timeline_db,
            tenant_id=config.tenant_id,
            source_ids=tallanto_opportunity_source_ids(records),
        ),
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
    resolved_owner_conflicts = 0
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
        resolved_owner_conflicts = close_relinked_payment_owner_conflicts(
            config.timeline_db,
            tenant_id=config.tenant_id,
            records=records,
            customer_lookup=customer_lookup,
        )
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
            "resolved_payment_owner_conflicts": resolved_owner_conflicts,
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


def close_relinked_payment_owner_conflicts(
    db_path: Path,
    *,
    tenant_id: str,
    records: Sequence[TimelineSourceRecord],
    customer_lookup: TallantoCustomerLookup,
) -> int:
    resolved_refs: set[str] = set()
    for record in records:
        contact_id = optional_text(record.payload.get("contact_id"))
        if record.payload.get("_tallanto_module") != PAYMENT_MODULE or not contact_id:
            continue
        if contact_id not in customer_lookup.unique_customer_ids:
            continue
        alternate_contact_id = optional_text(record.payload.get("_abonement_contact_id"))
        if alternate_contact_id and alternate_contact_id != contact_id:
            continue
        resolved_refs.add(record.source_ref)
    if not resolved_refs:
        return 0
    with CustomerTimelineSQLiteStore(db_path, allowed_root=db_path.parent) as store:
        return store.resolve_conflicts_by_refs(
            tenant_id,
            conflict_types=("tallanto_payment_owner_unresolved", "tallanto_identity_ambiguous"),
            entity_refs=tuple(sorted(resolved_refs)),
            actor="tallanto_payments_conflict_resolver",
        )


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


def stable_tallanto_opportunity_id(tenant_id: str, source_id: str) -> str:
    return stable_prefixed_id(
        "customer_opportunity",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "opportunity_type": OpportunityType.TALLANTO_COURSE.value,
            "source_system": SOURCE_SYSTEM,
            "source_id": source_id,
        },
    )


def build_tallanto_records(
    snapshot: Mapping[str, Any],
    *,
    source_path: Optional[str],
    existing_abonement_contacts: Optional[Mapping[str, str]] = None,
    existing_class_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
    existing_abonement_classes: Optional[Mapping[str, str]] = None,
) -> tuple[tuple[TimelineSourceRecord, ...], TallantoSnapshotStats, Mapping[str, Mapping[str, Any]]]:
    payment_rows = extract_module_rows(snapshot, PAYMENT_MODULE, aliases=("finances", "payments", "most_finances"))
    abonement_rows = extract_module_rows(snapshot, ABONEMENT_MODULE, aliases=("abonements", "subscriptions", "most_abonements"))
    class_rows = extract_module_rows(snapshot, CLASS_MODULE, aliases=("classes", "lessons", "most_class"))
    class_lookup = {
        **{str(key): dict(value) for key, value in (existing_class_lookup or {}).items()},
        **{
        require_nonempty(first_value(row, ("id", "class_id", "most_class_id")), "Tallanto class id"): sanitize_source_row(row)
        for row in class_rows
        if optional_text(first_value(row, ("id", "class_id", "most_class_id")))
        },
    }
    classes_by_abonement: dict[str, set[str]] = {
        str(key): {str(value)} for key, value in (existing_abonement_classes or {}).items()
    }
    abonement_names: dict[str, str] = {}
    for row in payment_rows:
        abonement_id = clean_text(first_value(row, ("most_abonements_id", "abonement_id")))
        class_id = clean_text(first_value(row, ("most_class_id", "class_id")))
        if abonement_id and class_id:
            classes_by_abonement.setdefault(abonement_id, set()).add(class_id)
    abonement_contacts: dict[str, Optional[str]] = dict(existing_abonement_contacts or {})
    for row in abonement_rows:
        abonement_id = clean_text(first_value(row, ("id", "abonement_id", "most_abonements_id")))
        if abonement_id:
            abonement_contacts[abonement_id] = clean_text(
                first_value(row, ("contact_id", "student_id", "tallanto_id", "Contact_ID"))
            )
            name = clean_text(first_value(row, ("name", "rate_translated", "category_translated")))
            if name:
                abonement_names[abonement_id] = name
    records: list[TimelineSourceRecord] = []
    stats = TallantoSnapshotStats(payment_rows=len(payment_rows), abonement_rows=len(abonement_rows), class_rows=len(class_rows))
    for module, rows in ((PAYMENT_MODULE, payment_rows), (ABONEMENT_MODULE, abonement_rows)):
        for idx, row in enumerate(rows, start=1):
            sanitized = sanitize_source_row(row)
            record_id = clean_text(first_value(sanitized, ("id", "finance_id", "payment_id", "abonement_id", "most_abonements_id")))
            if not record_id:
                stats.skipped += 1
                continue
            if module == PAYMENT_MODULE:
                abonement_id = clean_text(first_value(sanitized, ("most_abonements_id", "abonement_id")))
                abonement_contact = abonement_contacts.get(abonement_id or "")
                if abonement_contact:
                    sanitized["_abonement_contact_id"] = abonement_contact
                    if not clean_text(first_value(sanitized, ("contact_id", "student_id", "tallanto_id", "Contact_ID"))):
                        sanitized["contact_id"] = abonement_contact
                        sanitized["_contact_id_source"] = "abonement"
                elif not clean_text(first_value(sanitized, ("contact_id", "student_id", "tallanto_id", "Contact_ID"))):
                    sanitized["_contact_id_source"] = "unresolved_abonement" if abonement_id else "unresolved_owner"
                    stats.unresolved_payment_owners += 1
                class_id = clean_text(first_value(sanitized, ("most_class_id", "class_id")))
                related_class_ids = classes_by_abonement.get(abonement_id or "", set())
                if not class_id and len(related_class_ids) == 1:
                    class_id = next(iter(related_class_ids))
                    sanitized["most_class_id"] = class_id
                abonement_name = abonement_names.get(abonement_id or "")
                if abonement_name:
                    sanitized["_abonement_name"] = abonement_name
                if class_id and class_id in class_lookup:
                    stats.payment_class_relations_resolved += 1
                else:
                    stats.payment_class_relations_unresolved += 1
                if (
                    class_id
                    and clean_text(first_value(class_lookup.get(class_id, {}), ("name", "cource_name")))
                ) or abonement_name:
                    stats.payment_titles_resolved += 1
                else:
                    stats.payment_titles_unresolved += 1
            else:
                abonement_id = clean_text(first_value(sanitized, ("id", "abonement_id", "most_abonements_id")))
                class_ids = classes_by_abonement.get(abonement_id or "", set())
                abonement_name = clean_text(first_value(sanitized, ("name", "rate_translated", "category_translated")))
                if len(class_ids) == 1:
                    sanitized["most_class_id"] = next(iter(class_ids))
                if len(class_ids) == 1 and sanitized["most_class_id"] in class_lookup:
                    stats.abonement_class_relations_resolved += 1
                else:
                    stats.abonement_class_relations_unresolved += 1
                if abonement_name:
                    stats.abonement_titles_resolved += 1
                else:
                    stats.abonement_titles_unresolved += 1
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


def load_existing_abonement_contacts(db_path: Path, tenant_id: str) -> Mapping[str, str]:
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT source_id, record_json FROM timeline_events WHERE tenant_id=? AND source_system=? "
            "AND source_id LIKE 'most_abonements:%' AND coalesce(superseded_by,'')=''",
            (tenant_id, SOURCE_SYSTEM),
        ).fetchall()
    result: dict[str, str] = {}
    for source_id, record_json in rows:
        record = (json.loads(record_json).get("record") or {}) if record_json else {}
        contact_id = clean_text(record.get("contact_id"))
        if contact_id:
            result[str(source_id).split(":", 1)[-1]] = contact_id
    return result


def load_existing_money_class_context(
    db_path: Path,
    tenant_id: str,
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, str]]:
    if not db_path.exists():
        return {}, {}
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT event_type,subject,record_json FROM timeline_events WHERE tenant_id=? "
            "AND coalesce(superseded_by,'')='' AND ((source_system=? AND event_type='tallanto_payment' "
            "AND coalesce(json_extract(record_json,'$.record.class_id'),'')!='') "
            "OR (source_system='tallanto_attendance_api' AND event_type='tallanto_attendance' "
            "AND coalesce(json_extract(record_json,'$.record.abonement_id'),'')!='' "
            "AND coalesce(json_extract(record_json,'$.record.most_class_id'),'')!=''))",
            (tenant_id, SOURCE_SYSTEM),
        ).fetchall()
    class_lookup: dict[str, Mapping[str, Any]] = {}
    abonement_classes: dict[str, str] = {}
    ambiguous_abonements: set[str] = set()
    for event_type, subject, record_json in rows:
        record = (json.loads(record_json).get("record") or {}) if record_json else {}
        class_id = clean_text(record.get("class_id") or record.get("most_class_id"))
        class_name = clean_text(record.get("class_name") or (subject if event_type == "tallanto_attendance" else "")) or ""
        abonement_id = clean_text(record.get("abonement_id"))
        if class_id and class_name and class_name.casefold() not in {"tallanto payment", "оплата за абонемент"}:
            class_lookup[class_id] = {"id": class_id, "name": class_name}
        if class_id and abonement_id:
            if abonement_id in ambiguous_abonements:
                continue
            previous = abonement_classes.get(abonement_id)
            if previous is None or previous == class_id:
                abonement_classes[abonement_id] = class_id
            else:
                abonement_classes.pop(abonement_id, None)
                ambiguous_abonements.add(abonement_id)
    return class_lookup, abonement_classes


def extract_module_rows(snapshot: Mapping[str, Any], module: str, *, aliases: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    keys = tuple(dict.fromkeys((module, *aliases)))
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
    seen: dict[str, Mapping[str, Any]] = {}
    unique: list[Mapping[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        key = str(first_value(row, ("id", "finance_id", "payment_id", "abonement_id", "most_abonements_id")) or idx)
        sanitized = sanitize_source_row(row)
        if key in seen:
            if sanitized != seen[key]:
                raise ValueError(f"Tallanto {module} conflicting duplicate id: {key}")
            continue
        seen[key] = sanitized
        unique.append(sanitized)
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
        return TallantoCustomerLookup(unique_customer_ids={}, unique_match_classes={}, ambiguous_customer_ids={})
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "identity_links"):
            return TallantoCustomerLookup(unique_customer_ids={}, unique_match_classes={}, ambiguous_customer_ids={})
        exact_rows = tuple(
            row for row in authoritative_exact_identity_rows(
                con, tenant_id, link_types=("tallanto_student_id",)
            )
            if str(row["link_value"]) in contact_ids
        )
    by_contact: dict[str, set[str]] = {}
    match_classes: dict[str, set[IdentityMatchClass]] = {}
    blocked_contacts: set[str] = set()
    for row in exact_rows:
        contact_id = str(row["link_value"])
        customer_id = str(row["customer_id"])
        match_class = str(row["match_class"])
        by_contact.setdefault(contact_id, set()).add(customer_id)
        match_classes.setdefault(contact_id, set()).add(IdentityMatchClass(match_class))
        if int(row["owner_count"]) != 1 or bool(row["has_open_conflict"]):
            blocked_contacts.add(contact_id)
    unique_customer_ids = {
        contact_id: next(iter(customer_ids))
        for contact_id, customer_ids in by_contact.items()
        if len(customer_ids) == 1 and contact_id not in blocked_contacts
    }
    return TallantoCustomerLookup(
        unique_customer_ids=unique_customer_ids,
        unique_match_classes={
            contact_id: IdentityMatchClass.STRONG_UNIQUE
            if IdentityMatchClass.STRONG_UNIQUE in match_classes[contact_id]
            else IdentityMatchClass.MANUAL
            for contact_id in unique_customer_ids
        },
        ambiguous_customer_ids={
            contact_id: tuple(sorted(customer_ids))
            for contact_id, customer_ids in by_contact.items()
            if len(customer_ids) > 1 or contact_id in blocked_contacts
        },
    )


def tallanto_opportunity_source_ids(records: Sequence[TimelineSourceRecord]) -> set[str]:
    source_ids: set[str] = set()
    for record in records:
        module = normalize_key(record.payload.get("_tallanto_module") or record.source_system, "tallanto_module")
        if module == PAYMENT_MODULE:
            record_id = optional_text(first_value(record.payload, ("id", "finance_id", "payment_id")))
            if record_id:
                source_ids.add(f"payment:{record_id}")
        elif module == ABONEMENT_MODULE:
            record_id = optional_text(first_value(record.payload, ("id", "abonement_id", "most_abonements_id")))
            if record_id:
                source_ids.add(f"abonement:{record_id}")
    return source_ids


def load_tallanto_opportunity_lookup(
    db_path: Path,
    *,
    tenant_id: str,
    source_ids: set[str],
) -> Mapping[str, str]:
    if not source_ids or not db_path.exists():
        return {}
    lookup: dict[str, str] = {}
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "customer_opportunities"):
            return {}
        for chunk in chunks(sorted(source_ids), 800):
            placeholders = ",".join("?" for _ in chunk)
            for row in con.execute(
                f"""
                SELECT source_id, opportunity_id
                FROM customer_opportunities
                WHERE tenant_id = ?
                  AND source_system = ?
                  AND opportunity_type = ?
                  AND source_id IN ({placeholders})
                """,
                (tenant_id, SOURCE_SYSTEM, OpportunityType.TALLANTO_COURSE.value, *chunk),
            ):
                if row["source_id"] and row["opportunity_id"]:
                    lookup[str(row["source_id"])] = str(row["opportunity_id"])
    return lookup


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
    con = sqlite3.connect(customer_timeline_readonly_uri(db_path), uri=True, timeout=15)
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
