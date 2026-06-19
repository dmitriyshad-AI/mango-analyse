from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.ids import normalize_email, normalize_key, stable_digest
from mango_mvp.customer_timeline.safety import customer_timeline_safety_contract, guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore, customer_timeline_sqlite_safety_contract
from mango_mvp.utils.phone import normalize_phone


CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION = "canonical_readonly_customer_timeline_v1"
DEFAULT_OUT_ROOT = Path("product_data/customer_timeline/canonical_readonly_20260521_v5")
OFFLINE_BRAND_INFER_MODE = "cyrillic_v2"
MASTER_CONTACT_SOURCE = "master_contacts_snapshot"
MANGO_SOURCE = "mango_processed_summary"
AMO_SOURCE = "amocrm_snapshot"
TALLANTO_SOURCE = "tallanto_snapshot"
MAIL_SOURCE = "mail_archive"
DEFAULT_TENANT_ID = "foton"
HISTORY_CALL_TYPES = frozenset({"sales_call", "existing_client_progress", "technical_call"})


@dataclass(frozen=True)
class CanonicalReadonlyTimelineConfig:
    project_root: Path
    out_root: Path = DEFAULT_OUT_ROOT
    timeline_db: Optional[Path] = None
    tenant_id: str = DEFAULT_TENANT_ID
    current_runtime_json: Optional[Path] = None
    master_contacts_csv: Optional[Path] = None
    master_calls_csv: Optional[Path] = None
    canonical_calls_db: Optional[Path] = None
    amo_contacts_csv: Optional[Path] = None
    amo_deals_csv: Optional[Path] = None
    mail_handoff_db: Optional[Path] = None
    mail_bridge_db: Optional[Path] = None
    generated_at: Optional[datetime] = None
    max_call_events_per_contact: int = 0


def build_canonical_readonly_customer_timeline(config: CanonicalReadonlyTimelineConfig) -> Mapping[str, Any]:
    resolved = resolve_config(config)
    generated_at = resolved.generated_at or datetime.now(timezone.utc)
    resolved.out_root.mkdir(parents=True, exist_ok=True)

    contacts = read_csv_rows(resolved.master_contacts_csv)
    customers_by_key = build_customer_index(
        contacts,
        tenant_id=resolved.tenant_id,
        generated_at=generated_at,
    )
    customer_phones = tuple(item["phone"] for item in customers_by_key.values())
    known_phones = set(customer_phones)
    calls_by_phone = read_calls_by_phone(
        resolved.master_calls_csv,
        known_phones=known_phones,
        max_per_phone=resolved.max_call_events_per_contact,
        canonical_calls_db=resolved.canonical_calls_db,
    )
    amo_contacts_by_phone = read_amo_contacts_by_phone(resolved.amo_contacts_csv, known_phones=known_phones)
    amo_deals_by_contact_id = read_amo_deals_by_contact_id(resolved.amo_deals_csv)
    duplicate_amo_contact_ids, duplicate_amo_lead_ids = duplicate_amo_ids_across_sources(
        contacts,
        amo_contacts_by_phone=amo_contacts_by_phone,
        amo_deals_by_contact_id=amo_deals_by_contact_id,
    )
    shared_amo_reasons_by_phone = shared_amo_reasons_by_phone_from_sources(
        contacts,
        amo_contacts_by_phone=amo_contacts_by_phone,
        amo_deals_by_contact_id=amo_deals_by_contact_id,
        duplicate_amo_contact_ids=duplicate_amo_contact_ids,
        duplicate_amo_lead_ids=duplicate_amo_lead_ids,
    )
    mail_by_phone = read_mail_aggregates_by_phone(resolved.mail_bridge_db, known_phones=known_phones)

    source_manifest = build_source_manifest(
        {
            "current_runtime_json": resolved.current_runtime_json,
            "master_contacts_csv": resolved.master_contacts_csv,
            "master_calls_csv": resolved.master_calls_csv,
            "canonical_calls_db": resolved.canonical_calls_db,
            "amo_contacts_csv": resolved.amo_contacts_csv,
            "amo_deals_csv": resolved.amo_deals_csv,
            "mail_handoff_db": resolved.mail_handoff_db,
            "mail_bridge_db": resolved.mail_bridge_db,
        }
    )
    input_hash = stable_digest(
        {
            "schema_version": CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION,
            "sources": source_manifest,
            "tenant_id": resolved.tenant_id,
            "known_phones": len(known_phones),
            "customer_entries": len(customers_by_key),
            "generated_at": generated_at.isoformat(),
        }
    )

    store = CustomerTimelineSQLiteStore(resolved.timeline_db, allowed_root=resolved.out_root)
    imported_counts: Counter[str] = Counter()
    write_status_counts: Counter[str] = Counter()
    manual_review_counts: Counter[str] = Counter()
    source_customer_counts: Counter[str] = Counter()
    source_event_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    run: Any = None
    try:
        run = store.start_ingestion_run(
            tenant_id=resolved.tenant_id,
            source_system="canonical_readonly_customer_timeline",
            source_ref=resolved.out_root.name,
            run_kind="canonical_readonly_timeline_import",
            idempotency_key=input_hash,
            input_hash=input_hash,
            started_at=generated_at,
            metadata={
                "schema_version": CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION,
                "sources": source_manifest,
                "safety": canonical_readonly_timeline_safety_contract(write_customer_timeline_db=True),
            },
            actor="canonical_readonly_timeline_import",
        )
        family_groups = shared_family_phone_groups(customers_by_key)
        family_phones = set(family_groups)
        family_amo_contact_ids = {
            phone: {safe_text(contact.get("contact_id")) for contact in amo_contacts_by_phone.get(phone, ()) if safe_text(contact.get("contact_id"))}
            for phone in family_phones
        }
        family_amo_lead_ids = {
            phone: {
                lead_id
                for contact in amo_contacts_by_phone.get(phone, ())
                for lead_id in split_ids(contact.get("linked_lead_ids"))
            }
            for phone in family_phones
        }
        for phone, items in family_groups.items():
            customer_ids = tuple(item["customer"].customer_id for item in items)
            tallanto_ids = tuple(sorted({tallanto_id for item in items for tallanto_id in split_ids(item["row"].get("ID Tallanto"))}))
            result = store.record_conflict(
                resolved.tenant_id,
                conflict_type="shared_family_phone",
                entity_refs=(
                    f"phone_hash:{stable_digest({'phone': phone})[:16]}",
                    *(f"customer:{customer_id}" for customer_id in customer_ids),
                    *(f"tallanto_student:{tallanto_id}" for tallanto_id in tallanto_ids),
                ),
                severity="high",
                status="open",
                summary="One phone is linked to multiple Tallanto students; customers stay split.",
                metadata={"phone_hash": stable_digest({"phone": phone})[:16], "tallanto_student_ids": list(tallanto_ids)},
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=run.run_id,
            )
            imported_counts[result.record_type] += 1
            write_status_counts[result.status] += 1
            manual_review_counts.update(["shared_family_phone"])

        for customer_key in sorted(customers_by_key):
            item = customers_by_key[customer_key]
            phone = item["phone"]
            row = item["row"]
            customer = item["customer"]
            brand = infer_offline_brand(row)
            brand_counts[brand] += 1
            effective_duplicate_amo_contact_ids = set(duplicate_amo_contact_ids)
            effective_duplicate_amo_lead_ids = set(duplicate_amo_lead_ids)
            if phone in family_phones:
                effective_duplicate_amo_contact_ids.update(family_amo_contact_ids.get(phone, set()))
                effective_duplicate_amo_lead_ids.update(family_amo_lead_ids.get(phone, set()))
            reasons = manual_review_reasons(
                row=row,
                calls=calls_by_phone.get(phone, ()),
                mail=mail_by_phone.get(phone),
                duplicate_amo_contact_ids=effective_duplicate_amo_contact_ids,
                duplicate_amo_lead_ids=effective_duplicate_amo_lead_ids,
                extra_reasons=tuple(shared_amo_reasons_by_phone.get(phone, ()))
                + (("shared_family_phone",) if phone in family_phones else ()),
            )
            manual_review_counts.update(reasons)
            for result in upsert_customer_bundle(
                store,
                tenant_id=resolved.tenant_id,
                customer=customer,
                row=row,
                brand=brand,
                generated_at=generated_at,
                ingestion_run_id=run.run_id,
                duplicate_amo_contact_ids=effective_duplicate_amo_contact_ids,
                duplicate_amo_lead_ids=effective_duplicate_amo_lead_ids,
                phone_match_class=IdentityMatchClass.AMBIGUOUS if phone in family_phones else IdentityMatchClass.STRONG_UNIQUE,
                phone_confidence=0.55 if phone in family_phones else 0.95,
            ):
                imported_counts[result.record_type] += 1
                write_status_counts[result.status] += 1
            mapping_result = store.record_customer_id_mapping(
                resolved.tenant_id,
                old_customer_id=customer.customer_id,
                new_customer_id=customer.customer_id,
                mapping_kind="alias",
                reason="canonical_readonly_identity",
                source_refs=(customer.source_ref or customer.customer_id,),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=run.run_id,
            )
            imported_counts[mapping_result.record_type] += 1
            write_status_counts[mapping_result.status] += 1
            if phone in family_phones:
                split_mapping = store.record_customer_id_mapping(
                    resolved.tenant_id,
                    old_customer_id=legacy_phone_customer_id(resolved.tenant_id, phone=phone, row=row),
                    new_customer_id=customer.customer_id,
                    mapping_kind="split",
                    reason="shared_family_phone",
                    source_refs=(customer.source_ref or customer.customer_id,),
                    metadata={"phone_hash": stable_digest({"phone": phone})[:16]},
                    actor="canonical_readonly_timeline_import",
                    ingestion_run_id=run.run_id,
                )
                imported_counts[split_mapping.record_type] += 1
                write_status_counts[split_mapping.status] += 1
            source_customer_counts[MASTER_CONTACT_SOURCE] += 1

            for call in calls_by_phone.get(phone, ()):
                event = call_event(
                    tenant_id=resolved.tenant_id,
                    customer_id=customer.customer_id,
                    call=call,
                    brand=brand,
                    fallback_at=generated_at,
                    source_id_suffix=customer.customer_id if phone in family_phones else None,
                    match_class=IdentityMatchClass.AMBIGUOUS if phone in family_phones else IdentityMatchClass.STRONG_UNIQUE,
                    confidence=0.55 if phone in family_phones else 0.9,
                )
                result = store.upsert_event(event, actor="canonical_readonly_timeline_import", ingestion_run_id=run.run_id)
                imported_counts[result.record_type] += 1
                write_status_counts[result.status] += 1
                source_event_counts[MANGO_SOURCE] += 1
                if event.summary:
                    chunk = BotContextChunk(
                        tenant_id=resolved.tenant_id,
                        customer_id=customer.customer_id,
                        event_id=event.event_id,
                        source_ref=event.source_ref,
                        source_system=MANGO_SOURCE,
                        chunk_type="mango_call_summary",
                        text=event.summary,
                        summary=event.summary[:160],
                        event_at=event.event_at,
                        freshness_score=0.8,
                        relevance_tags=("mango", "call", brand),
                        allowed_for_bot=False,
                        requires_manager_review=True,
                        created_at=generated_at,
                    )
                    result = store.upsert_bot_context_chunk(
                        chunk,
                        actor="canonical_readonly_timeline_import",
                        ingestion_run_id=run.run_id,
                    )
                    imported_counts[result.record_type] += 1
                    write_status_counts[result.status] += 1

            for amo_contact in amo_contacts_by_phone.get(phone, ()):
                for result in upsert_amo_snapshot(
                    store,
                    tenant_id=resolved.tenant_id,
                    customer_id=customer.customer_id,
                    amo_contact=amo_contact,
                    deals_by_contact_id=amo_deals_by_contact_id,
                    brand=brand,
                    generated_at=generated_at,
                    ingestion_run_id=run.run_id,
                    duplicate_amo_contact_ids=effective_duplicate_amo_contact_ids,
                    duplicate_amo_lead_ids=effective_duplicate_amo_lead_ids,
                ):
                    imported_counts[result.record_type] += 1
                    write_status_counts[result.status] += 1
                source_event_counts[AMO_SOURCE] += 1

            mail = mail_by_phone.get(phone)
            if mail:
                event = mail_aggregate_event(
                    tenant_id=resolved.tenant_id,
                    customer_id=customer.customer_id,
                    mail=mail,
                    brand=brand,
                    fallback_at=generated_at,
                )
                result = store.upsert_event(event, actor="canonical_readonly_timeline_import", ingestion_run_id=run.run_id)
                imported_counts[result.record_type] += 1
                write_status_counts[result.status] += 1
                source_event_counts[MAIL_SOURCE] += 1

        store.finish_ingestion_run(
            run.run_id,
            status="completed",
            accepted_count=len(customers_by_key),
            rejected_count=0,
            output_ref=str(resolved.timeline_db),
            finished_at=generated_at,
            metadata={
                "imported_counts": dict(imported_counts),
                "write_status_counts": dict(write_status_counts),
                "manual_review_counts": dict(manual_review_counts),
            },
            actor="canonical_readonly_timeline_import",
        )
        store_summary = store.summary()
    except Exception as exc:
        if run is not None:
            finished_at = max(generated_at, datetime.now(timezone.utc))
            store.finish_ingestion_run(
                run.run_id,
                status="failed",
                accepted_count=sum(source_customer_counts.values()),
                rejected_count=max(0, len(customers_by_key) - sum(source_customer_counts.values())),
                output_ref=str(resolved.timeline_db),
                error=f"{type(exc).__name__}: {exc}",
                finished_at=finished_at,
                metadata={
                    "imported_counts": dict(imported_counts),
                    "write_status_counts": dict(write_status_counts),
                    "manual_review_counts": dict(manual_review_counts),
                },
                actor="canonical_readonly_timeline_import",
            )
        raise
    finally:
        store.close()

    coverage = build_coverage_report(
        config=resolved,
        generated_at=generated_at,
        source_manifest=source_manifest,
        phones=customer_phones,
        calls_by_phone=calls_by_phone,
        amo_contacts_by_phone=amo_contacts_by_phone,
        mail_by_phone=mail_by_phone,
        contacts=contacts,
        imported_counts=imported_counts,
        write_status_counts=write_status_counts,
        manual_review_counts=manual_review_counts,
        source_event_counts=source_event_counts,
        source_customer_counts=source_customer_counts,
        brand_counts=brand_counts,
        store_summary=store_summary,
        duplicate_amo_contact_ids=duplicate_amo_contact_ids,
        duplicate_amo_lead_ids=duplicate_amo_lead_ids,
        shared_amo_reasons_by_phone=shared_amo_reasons_by_phone,
    )
    import_report = {
        "schema_version": CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "tenant_id": resolved.tenant_id,
        "mode": "apply_local_customer_timeline_db",
        "paths": {
            "out_root": str(resolved.out_root),
            "timeline_db": str(resolved.timeline_db),
            "coverage_report_json": str(resolved.out_root / "coverage_report.json"),
            "coverage_report_md": str(resolved.out_root / "coverage_report.md"),
            "source_manifest_json": str(resolved.out_root / "source_manifest.json"),
            "import_report_json": str(resolved.out_root / "import_report.json"),
        },
        "summary": coverage["summary"],
        "safety": canonical_readonly_timeline_safety_contract(write_customer_timeline_db=True),
    }
    write_json(resolved.out_root / "source_manifest.json", source_manifest)
    write_json(resolved.out_root / "coverage_report.json", coverage)
    write_json(resolved.out_root / "import_report.json", import_report)
    (resolved.out_root / "coverage_report.md").write_text(render_coverage_markdown(coverage), encoding="utf-8")
    return import_report


def resolve_config(config: CanonicalReadonlyTimelineConfig) -> CanonicalReadonlyTimelineConfig:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    out_root = guard_customer_timeline_output_path(project_root / config.out_root if not config.out_root.is_absolute() else config.out_root, project_root)
    timeline_db = config.timeline_db or out_root / "customer_timeline.sqlite"
    timeline_db = guard_customer_timeline_output_path(timeline_db, out_root)
    current_runtime = resolve_existing(config.current_runtime_json or project_root / "stable_runtime" / "CURRENT_RUNTIME.json")
    runtime = json.loads(current_runtime.read_text(encoding="utf-8"))
    paths = runtime.get("paths") if isinstance(runtime.get("paths"), Mapping) else {}
    active_export_root = Path(str(paths.get("active_export_root") or "")).expanduser()
    if not active_export_root:
        raise ValueError("CURRENT_RUNTIME.json does not contain paths.active_export_root")
    active_export_root = active_export_root.resolve(strict=False)
    master_contacts = resolve_existing(config.master_contacts_csv or active_export_root / "master_contacts_ru.csv")
    master_calls = resolve_existing(config.master_calls_csv or active_export_root / "master_calls_ru.csv")
    amo_root = project_root / "stable_runtime" / "deal_aware_amo_live_snapshot_20260513_v2"
    amo_contacts = resolve_existing(config.amo_contacts_csv or amo_root / "amo_contacts_snapshot.csv")
    amo_deals = resolve_existing(config.amo_deals_csv or amo_root / "amo_deals_snapshot.csv")
    mail_root = project_root / "_external_handoffs" / "mail_archive_2026-05-12" / "regru_edu" / "full_all_mail_combined_20260513"
    mail_handoff = resolve_existing(
        config.mail_handoff_db
        or mail_root / "customer_history_handoff_full_all_mail" / "mail_customer_history_handoff.sqlite"
    )
    mail_bridge = resolve_existing(
        config.mail_bridge_db
        or mail_root / "mango_bridge_preview_full_all_mail_extended_phone_index" / "mail_mango_bridge_preview.sqlite"
    )
    return CanonicalReadonlyTimelineConfig(
        project_root=project_root,
        out_root=out_root,
        timeline_db=timeline_db,
        tenant_id=normalize_key(config.tenant_id, "tenant_id"),
        current_runtime_json=current_runtime,
        master_contacts_csv=master_contacts,
        master_calls_csv=master_calls,
        canonical_calls_db=resolve_existing(config.canonical_calls_db) if config.canonical_calls_db else None,
        amo_contacts_csv=amo_contacts,
        amo_deals_csv=amo_deals,
        mail_handoff_db=mail_handoff,
        mail_bridge_db=mail_bridge,
        generated_at=config.generated_at,
        max_call_events_per_contact=max(0, int(config.max_call_events_per_contact)),
    )


def build_customer_index(
    rows: Sequence[Mapping[str, str]],
    *,
    tenant_id: str,
    generated_at: datetime,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    rows_by_phone: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        phone = normalize_phone(row.get("Телефон клиента", ""))
        if not phone:
            continue
        rows_by_phone[phone].append(row)
    for phone, phone_rows in sorted(rows_by_phone.items()):
        if family_phone_group(phone_rows):
            grouped_rows = family_customer_rows(phone, phone_rows)
        else:
            grouped_rows = ((phone, merge_contact_rows(phone_rows)),)
        for customer_key, row in grouped_rows:
            result[customer_key] = build_customer_index_item(
                row,
                tenant_id=tenant_id,
                phone=phone,
                customer_key=customer_key,
                generated_at=generated_at,
            )
    return result


def build_customer_index_item(
    row: Mapping[str, str],
    *,
    tenant_id: str,
    phone: str,
    customer_key: str,
    generated_at: datetime,
) -> dict[str, Any]:
    first_seen = parse_datetime_guess(row.get("Первый звонок")) or generated_at
    last_seen = parse_datetime_guess(row.get("Последний звонок")) or first_seen
    email = normalize_email(row.get("Email"))
    display_name = safe_text(row.get("ФИО родителя") or row.get("ФИО родителя Tallanto") or row.get("Контакт Tallanto")) or None
    status = customer_identity_status_from_tallanto(row, phone=phone, email=email)
    brands = tuple(split_ids(row.get("_brand_history"))) or (infer_offline_brand(row),)
    customer = CustomerIdentity(
        tenant_id=tenant_id,
        identity_status=status,
        display_name=display_name,
        primary_phone=phone,
        primary_email=email or None,
        source_ref=f"master_contact:{customer_key}",
        first_seen_at=first_seen,
        last_seen_at=last_seen,
        touch_count=int_or_zero(row.get("Всего звонков в истории")),
        summary={
            "source_system": MASTER_CONTACT_SOURCE,
            "call_count": int_or_zero(row.get("Всего звонков в истории")),
            "contentful_call_count": int_or_zero(row.get("Содержательных звонков в истории")),
            "tallanto_match_status": safe_text(row.get("Статус матчинга Tallanto")),
            "amo_contact_id_count": len(split_ids(row.get("AMO contact IDs"))),
            "amo_lead_id_count": len(split_ids(row.get("AMO lead IDs"))),
            "brand": infer_offline_brand(row),
            "brands": list(brands),
        },
        metadata={"source": MASTER_CONTACT_SOURCE, "brands": list(brands)},
        created_at=generated_at,
        updated_at=generated_at,
    )
    return {"customer": customer, "row": dict(row), "phone": phone, "customer_key": customer_key}


def legacy_phone_customer_id(tenant_id: str, *, phone: str, row: Mapping[str, str]) -> str:
    return CustomerIdentity(
        tenant_id=tenant_id,
        identity_status=IdentityStatus.STRONG,
        primary_phone=phone,
        source_ref=f"master_contact:{phone}",
    ).customer_id


def customer_identity_status_from_tallanto(
    row: Mapping[str, str],
    *,
    phone: str,
    email: str,
) -> IdentityStatus:
    match_class = tallanto_match_class(row.get("Статус матчинга Tallanto"))
    if match_class in {IdentityMatchClass.AMBIGUOUS, IdentityMatchClass.DUPLICATE}:
        return IdentityStatus.AMBIGUOUS
    if match_class == IdentityMatchClass.UNMATCHED:
        return IdentityStatus.PARTIAL if phone or email else IdentityStatus.UNMATCHED
    if match_class == IdentityMatchClass.STRONG_UNIQUE:
        return IdentityStatus.STRONG if phone or email else IdentityStatus.PARTIAL
    return IdentityStatus.PARTIAL if phone or email else IdentityStatus.UNMATCHED


def canonical_customer_key(phone: str, row: Mapping[str, str], ordinal: int) -> str:
    tallanto_ids = split_ids(row.get("ID Tallanto"))
    if tallanto_ids:
        return f"{phone}:tallanto:{'+'.join(tallanto_ids)}"
    return f"{phone}:row:{ordinal}"


def merge_contact_rows(rows: Sequence[Mapping[str, str]]) -> Mapping[str, str]:
    merged: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            text = safe_text(value)
            if text and not merged.get(key):
                merged[key] = text
    for key in ("ID Tallanto", "AMO contact IDs", "AMO lead IDs"):
        values = sorted({item for row in rows for item in split_ids(row.get(key))})
        if values:
            merged[key] = ";".join(values)
    brands = sorted({brand for row in rows if (brand := infer_offline_brand(row)) != "unknown"})
    if brands:
        merged["_brand_history"] = ";".join(brands)
    return merged


def family_customer_rows(phone: str, rows: Sequence[Mapping[str, str]]) -> tuple[tuple[str, Mapping[str, str]], ...]:
    by_tallanto_id: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    no_tallanto_rows: list[Mapping[str, str]] = []
    for row in rows:
        tallanto_ids = split_ids(row.get("ID Tallanto"))
        if not tallanto_ids:
            no_tallanto_rows.append(row)
            continue
        for tallanto_id in tallanto_ids:
            split_row = dict(row)
            split_row["ID Tallanto"] = tallanto_id
            split_row["_family_source_tallanto_ids"] = ";".join(tallanto_ids)
            by_tallanto_id[tallanto_id].append(split_row)

    result: list[tuple[str, Mapping[str, str]]] = []
    for tallanto_id in sorted(by_tallanto_id):
        row = merge_contact_rows(by_tallanto_id[tallanto_id])
        result.append((f"{phone}:tallanto:{tallanto_id}", row))
    for ordinal, row in enumerate(no_tallanto_rows, start=1):
        result.append((canonical_customer_key(phone, row, ordinal), row))
    return tuple(result)


def family_phone_group(rows: Sequence[Mapping[str, str]]) -> bool:
    tallanto_ids = {item for row in rows for item in split_ids(row.get("ID Tallanto"))}
    if len(tallanto_ids) > 1:
        return True
    return any(tallanto_multiple_candidate(row) for row in rows)


def tallanto_multiple_candidate(row: Mapping[str, str]) -> bool:
    status = safe_text(row.get("Статус матчинга Tallanto")).casefold()
    if any(marker in status for marker in ("multiple", "ambiguous", "many", "несколько", "duplicate")):
        return True
    return int_or_zero(row.get("Количество кандидатов Tallanto")) > 1


def shared_family_phone_groups(customers_by_key: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in customers_by_key.values():
        grouped[str(item["phone"])].append(item)
    return {
        phone: tuple(items)
        for phone, items in grouped.items()
        if len(items) > 1 and len({tallanto_id for item in items for tallanto_id in split_ids(item["row"].get("ID Tallanto"))}) > 1
    }


def upsert_customer_bundle(
    store: CustomerTimelineSQLiteStore,
    *,
    tenant_id: str,
    customer: CustomerIdentity,
    row: Mapping[str, str],
    brand: str,
    generated_at: datetime,
    ingestion_run_id: str,
    duplicate_amo_contact_ids: set[str] | None = None,
    duplicate_amo_lead_ids: set[str] | None = None,
    phone_match_class: IdentityMatchClass = IdentityMatchClass.STRONG_UNIQUE,
    phone_confidence: float = 0.95,
) -> list[Any]:
    results: list[Any] = []
    duplicate_amo_contact_ids = duplicate_amo_contact_ids or set()
    duplicate_amo_lead_ids = duplicate_amo_lead_ids or set()
    results.append(store.upsert_customer(customer, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
    source_ref = customer.source_ref or customer.customer_id
    if customer.primary_phone:
        results.append(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer.customer_id,
                    link_type="phone",
                    link_value=customer.primary_phone,
                    source_system=MASTER_CONTACT_SOURCE,
                    source_ref=source_ref,
                    match_class=phone_match_class,
                    confidence=phone_confidence,
                    first_seen_at=customer.first_seen_at,
                    last_seen_at=customer.last_seen_at,
                ),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
        results.append(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer.customer_id,
                    link_type="mango_client_phone",
                    link_value=customer.primary_phone,
                    source_system=MANGO_SOURCE,
                    source_ref=source_ref,
                    match_class=phone_match_class,
                    confidence=phone_confidence,
                    first_seen_at=customer.first_seen_at,
                    last_seen_at=customer.last_seen_at,
                ),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    if customer.primary_email:
        results.append(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer.customer_id,
                    link_type="email",
                    link_value=customer.primary_email,
                    source_system=MASTER_CONTACT_SOURCE,
                    source_ref=source_ref,
                    match_class=IdentityMatchClass.STRONG_UNIQUE,
                    confidence=0.8,
                    first_seen_at=customer.first_seen_at,
                    last_seen_at=customer.last_seen_at,
                ),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    for tallanto_id in split_ids(row.get("ID Tallanto")):
        results.append(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer.customer_id,
                    link_type="tallanto_student_id",
                    link_value=tallanto_id,
                    source_system=TALLANTO_SOURCE,
                    source_ref=f"master_contact:{customer.customer_id}:tallanto",
                    match_class=tallanto_match_class(row.get("Статус матчинга Tallanto")),
                    confidence=0.9,
                    first_seen_at=customer.first_seen_at,
                    last_seen_at=customer.last_seen_at,
                ),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    for contact_id in split_ids(row.get("AMO contact IDs")):
        is_duplicate_contact = contact_id in duplicate_amo_contact_ids
        results.append(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer.customer_id,
                    link_type="amo_contact_id",
                    link_value=contact_id,
                    source_system=AMO_SOURCE,
                    source_ref=f"master_contact:{customer.customer_id}:amo_contact",
                    match_class=(
                        IdentityMatchClass.STRONG_UNIQUE
                        if len(split_ids(row.get("AMO contact IDs"))) == 1 and not is_duplicate_contact
                        else IdentityMatchClass.AMBIGUOUS
                    ),
                    confidence=0.85,
                    first_seen_at=customer.first_seen_at,
                    last_seen_at=customer.last_seen_at,
                ),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    for lead_id in split_ids(row.get("AMO lead IDs")):
        is_duplicate_lead = lead_id in duplicate_amo_lead_ids
        if not is_duplicate_lead:
            opportunity = CustomerOpportunity(
                tenant_id=tenant_id,
                customer_id=customer.customer_id,
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system=AMO_SOURCE,
                source_id=lead_id,
                title=safe_text(row.get("Рекомендуемый продукт")) or f"AMO deal {lead_id}",
                status=safe_text(row.get("Причина статуса AMO") or row.get("AMO entity policy")),
                product_context={"brand": brand, "products_of_interest": safe_text(row.get("Продукты интереса"))},
                opened_at=customer.first_seen_at,
                confidence=0.7,
                evidence={"source_system": MASTER_CONTACT_SOURCE},
            )
            results.append(store.upsert_opportunity(opportunity, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
        results.append(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer.customer_id,
                    link_type="amo_lead_id",
                    link_value=lead_id,
                    source_system=AMO_SOURCE,
                    source_ref=f"master_contact:{customer.customer_id}:amo_lead",
                    match_class=(
                        IdentityMatchClass.STRONG_UNIQUE
                        if len(split_ids(row.get("AMO lead IDs"))) == 1 and not is_duplicate_lead
                        else IdentityMatchClass.AMBIGUOUS
                    ),
                    confidence=0.55 if is_duplicate_lead else 0.8,
                    first_seen_at=customer.first_seen_at,
                    last_seen_at=customer.last_seen_at,
                ),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    contact_event = TimelineEvent(
        tenant_id=tenant_id,
        customer_id=customer.customer_id,
        event_type=TimelineEventType.AMO_CONTACT_SNAPSHOT,
        event_at=customer.last_seen_at or generated_at,
        source_system=MASTER_CONTACT_SOURCE,
        source_id=customer.customer_id,
        source_ref=source_ref,
        direction=TimelineDirection.SYSTEM,
        subject="Master contact snapshot",
        text_preview=compact_join(
            [
                f"Звонков: {safe_text(row.get('Всего звонков в истории'))}",
                f"Tallanto: {safe_text(row.get('Статус матчинга Tallanto'))}",
                f"AMO: {safe_text(row.get('AMO entity policy'))}",
            ]
        ),
        summary=safe_text(row.get("Краткая история общения"))[:500],
        match_status=IdentityMatchClass.STRONG_UNIQUE,
        confidence=0.85,
        record={"brand": brand, "manual_review_reasons": manual_review_reasons(row=row, calls=(), mail=None)},
        metadata={"brand": brand},
        created_at=generated_at,
    )
    results.append(store.upsert_event(contact_event, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
    if safe_text(row.get("Статус матчинга Tallanto")) or safe_text(row.get("ID Tallanto")):
        tallanto_event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer.customer_id,
            event_type=TimelineEventType.TALLANTO_STUDENT_SNAPSHOT,
            event_at=customer.last_seen_at or generated_at,
            source_system=TALLANTO_SOURCE,
            source_id=f"tallanto:{customer.customer_id}",
            source_ref=f"master_contact:{customer.customer_id}:tallanto",
            direction=TimelineDirection.SYSTEM,
            subject="Tallanto snapshot from master contact",
            text_preview=compact_join(
                [
                    f"Статус: {safe_text(row.get('Статус матчинга Tallanto'))}",
                    f"Тип ученика: {safe_text(row.get('Тип ученика Tallanto'))}",
                    f"Филиал: {safe_text(row.get('Филиал Tallanto'))}",
                ]
            ),
            summary=safe_text(row.get("Статус матчинга Tallanto")),
            match_status=tallanto_match_class(row.get("Статус матчинга Tallanto")),
            confidence=0.85,
            record={"brand": brand, "candidate_count": safe_text(row.get("Количество кандидатов Tallanto"))},
            metadata={"brand": brand},
            created_at=generated_at,
        )
        results.append(store.upsert_event(tallanto_event, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
    if safe_text(row.get("Краткая история общения")):
        chunk = BotContextChunk(
            tenant_id=tenant_id,
            customer_id=customer.customer_id,
            event_id=contact_event.event_id,
            source_ref=contact_event.source_ref,
            source_system=MASTER_CONTACT_SOURCE,
            chunk_type="customer_history_summary",
            text=safe_text(row.get("Краткая история общения"))[:2000],
            summary=safe_text(row.get("Краткая история общения"))[:160],
            event_at=contact_event.event_at,
            freshness_score=0.7,
            relevance_tags=("customer_history", brand),
            allowed_for_bot=False,
            requires_manager_review=True,
            created_at=generated_at,
        )
        results.append(store.upsert_bot_context_chunk(chunk, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
    return results


def read_calls_by_phone(
    path: Path,
    *,
    known_phones: set[str],
    max_per_phone: int = 0,
    canonical_calls_db: Optional[Path] = None,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    canonical_by_ref = read_canonical_call_analysis_by_ref(canonical_calls_db) if canonical_calls_db else {}
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        phone = normalize_phone(row.get("Телефон клиента", ""))
        if phone and phone in known_phones:
            grouped[phone].append(enrich_call_row_with_canonical_analysis(row, canonical_by_ref))
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for phone, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda row: parse_datetime_guess(row.get("Дата и время звонка")) or datetime.min.replace(tzinfo=timezone.utc))
        if max_per_phone:
            rows_sorted = rows_sorted[-max_per_phone:]
        result[phone] = tuple(rows_sorted)
    return result


def read_canonical_call_analysis_by_ref(path: Optional[Path]) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT source_filename, source_file, canonical_call_id, amocrm_contact_id, amocrm_lead_id, analysis_json
            FROM canonical_calls
            WHERE analysis_json IS NOT NULL AND analysis_json != ''
            """
        ).fetchall()
    for row in rows:
        try:
            analysis = json.loads(str(row["analysis_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(analysis, Mapping):
            continue
        payload = {
            "analysis": dict(analysis),
            "canonical_call_id": row["canonical_call_id"],
            "amocrm_contact_id": row["amocrm_contact_id"],
            "amocrm_lead_id": row["amocrm_lead_id"],
        }
        for ref in (row["source_filename"], row["source_file"]):
            text = safe_text(ref)
            if text:
                result[text] = payload
                result[Path(text).name] = payload
    return result


def enrich_call_row_with_canonical_analysis(
    row: Mapping[str, str],
    canonical_by_ref: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not canonical_by_ref:
        return dict(row)
    refs = (
        safe_text(row.get("Имя исходного файла")),
        safe_text(row.get("Путь к записи")),
        Path(safe_text(row.get("Путь к записи"))).name if safe_text(row.get("Путь к записи")) else "",
    )
    match: Mapping[str, Any] | None = None
    for ref in refs:
        if ref and ref in canonical_by_ref:
            match = canonical_by_ref[ref]
            break
    if match is None:
        return dict(row)
    enriched: dict[str, Any] = dict(row)
    enriched["__canonical_call_analysis"] = match.get("analysis") or {}
    enriched["__canonical_call_id"] = match.get("canonical_call_id")
    enriched["__canonical_amocrm_contact_id"] = match.get("amocrm_contact_id")
    enriched["__canonical_amocrm_lead_id"] = match.get("amocrm_lead_id")
    return enriched


def call_event(
    *,
    tenant_id: str,
    customer_id: str,
    call: Mapping[str, Any],
    brand: str,
    fallback_at: datetime,
    source_id_suffix: Optional[str] = None,
    match_class: IdentityMatchClass = IdentityMatchClass.STRONG_UNIQUE,
    confidence: float = 0.9,
) -> TimelineEvent:
    event_at = parse_datetime_guess(call.get("Дата и время звонка")) or fallback_at
    call_id = safe_text(call.get("ID звонка")) or stable_digest({"customer_id": customer_id, "event_at": event_at.isoformat()})[:16]
    source_id = call_id if not source_id_suffix else f"{call_id}:{source_id_suffix}"
    direction = call_direction(call.get("Направление звонка"))
    call_analysis = canonical_call_analysis_for_event(call)
    summary = safe_text(
        call_analysis.get("history_summary")
        or call.get("Краткое резюме разговора")
        or call_analysis.get("summary")
        or call.get("Следующий шаг")
        or call.get("Тип звонка")
    )
    record: dict[str, Any] = {
        "brand": brand,
        "duration_sec": int_or_zero(call.get("Длительность, сек")),
        "contentful": safe_text(call.get("Содержательный звонок")),
        "manual_review_required": safe_text(call.get("Нужна ручная проверка")),
    }
    if call_analysis:
        record["call_analysis"] = call_analysis
        record["call_type"] = call_analysis.get("call_type")
        record["call_history_eligible"] = bool(call_analysis.get("call_history_eligible"))
    canonical_call_id = call.get("__canonical_call_id")
    if canonical_call_id not in (None, ""):
        record["canonical_call_id"] = canonical_call_id
    canonical_amo_contact_id = safe_text(call.get("__canonical_amocrm_contact_id"))
    canonical_amo_lead_id = safe_text(call.get("__canonical_amocrm_lead_id"))
    if canonical_amo_contact_id:
        record["canonical_amocrm_contact_id"] = canonical_amo_contact_id
    if canonical_amo_lead_id:
        record["canonical_amocrm_lead_id"] = canonical_amo_lead_id
    return TimelineEvent(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_type=TimelineEventType.MANGO_CALL,
        event_at=event_at,
        source_system=MANGO_SOURCE,
        source_id=source_id,
        source_ref=f"mango:{source_id}",
        direction=direction,
        actor_name=safe_text(call.get("Менеджер")),
        subject=safe_text(call.get("Тип звонка") or call.get("Рекомендуемый продукт"))[:160],
        text_preview=summary[:240],
        summary=summary,
        importance=1 if safe_text(call.get("Содержательный звонок")).lower() == "да" else 0,
        match_status=match_class,
        confidence=confidence,
        record=record,
        metadata={"brand": brand},
        created_at=fallback_at,
    )


def canonical_call_analysis_for_event(call: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = call.get("__canonical_call_analysis")
    if not isinstance(raw, Mapping):
        return {}
    return {
        "history_summary": safe_text(raw.get("history_summary")),
        "history_short": safe_text(raw.get("history_short")),
        "summary": safe_text(raw.get("summary")),
        "structured_fields": json_safe_value(raw.get("structured_fields")),
        "crm_blocks": json_safe_value(raw.get("crm_blocks")),
        "objections": json_safe_value(raw.get("objections")),
        "pain_points": json_safe_value(raw.get("pain_points")),
        "next_step": safe_text(raw.get("next_step")),
        "interests": json_safe_value(raw.get("interests")),
        "target_product": safe_text(raw.get("target_product")),
        "budget": json_safe_value(raw.get("budget")),
        "student_grade": safe_text(raw.get("student_grade")),
        "follow_up_score": json_safe_value(raw.get("follow_up_score")),
        "follow_up_reason": safe_text(raw.get("follow_up_reason")),
        "needs_review": json_safe_value(raw.get("needs_review")),
        "review_reasons": json_safe_value(raw.get("review_reasons")),
        "quality_flags": json_safe_value(raw.get("quality_flags")),
        "call_type": canonical_call_type(raw),
        "call_history_eligible": canonical_call_type(raw) in HISTORY_CALL_TYPES,
        "analysis_schema_version": safe_text(raw.get("analysis_schema_version")),
    }


def canonical_call_type(analysis: Mapping[str, Any]) -> str:
    quality_current = analysis.get("call_quality_current")
    if isinstance(quality_current, Mapping):
        call_type = safe_text(quality_current.get("call_type"))
        if call_type:
            return call_type
    quality_flags = analysis.get("quality_flags")
    if isinstance(quality_flags, Mapping):
        return safe_text(quality_flags.get("call_type"))
    return ""


def json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {safe_text(key): json_safe_value(item) for key, item in value.items() if safe_text(key)}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return safe_text(value)


def read_amo_contacts_by_phone(path: Path, *, known_phones: set[str]) -> dict[str, tuple[Mapping[str, str], ...]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        phones = extract_phones_from_text(row.get("phones", ""))
        for phone in phones:
            if phone in known_phones:
                grouped[phone].append(row)
    return {phone: tuple(rows) for phone, rows in grouped.items()}


def read_amo_deals_by_contact_id(path: Path) -> dict[str, tuple[Mapping[str, str], ...]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        for contact_id in split_ids(row.get("linked_contact_ids")):
            grouped[contact_id].append(row)
    return {contact_id: tuple(rows) for contact_id, rows in grouped.items()}


def upsert_amo_snapshot(
    store: CustomerTimelineSQLiteStore,
    *,
    tenant_id: str,
    customer_id: str,
    amo_contact: Mapping[str, str],
    deals_by_contact_id: Mapping[str, Sequence[Mapping[str, str]]],
    brand: str,
    generated_at: datetime,
    ingestion_run_id: str,
    duplicate_amo_contact_ids: set[str] | None = None,
    duplicate_amo_lead_ids: set[str] | None = None,
) -> list[Any]:
    results: list[Any] = []
    duplicate_amo_contact_ids = duplicate_amo_contact_ids or set()
    duplicate_amo_lead_ids = duplicate_amo_lead_ids or set()
    contact_id = safe_text(amo_contact.get("contact_id"))
    if not contact_id:
        return results
    is_duplicate_contact = contact_id in duplicate_amo_contact_ids
    contact_created_at = parse_unix_or_iso(amo_contact.get("created_at"))
    updated_at = parse_unix_or_iso(amo_contact.get("updated_at")) or contact_created_at or generated_at
    contact_first_seen_at, updated_at, contact_date_corrected = ordered_datetime_pair(
        contact_created_at or updated_at,
        updated_at,
    )
    results.append(
        store.upsert_identity_link(
            IdentityLink(
                tenant_id=tenant_id,
                customer_id=customer_id,
                link_type="amo_contact_id",
                link_value=contact_id,
                source_system=AMO_SOURCE,
                source_ref=f"amo:contact:{contact_id}" if not is_duplicate_contact else f"amo:contact:{contact_id}:customer:{customer_id}",
                match_class=IdentityMatchClass.AMBIGUOUS if is_duplicate_contact else IdentityMatchClass.STRONG_UNIQUE,
                confidence=0.55 if is_duplicate_contact else 0.85,
                first_seen_at=contact_first_seen_at,
                last_seen_at=updated_at,
            ),
            actor="canonical_readonly_timeline_import",
            ingestion_run_id=ingestion_run_id,
        )
    )
    event = TimelineEvent(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_type=TimelineEventType.AMO_CONTACT_SNAPSHOT,
        event_at=updated_at,
        source_system=AMO_SOURCE,
        source_id=contact_id if not is_duplicate_contact else f"{contact_id}:{customer_id}",
        source_ref=f"amo:contact:{contact_id}" if not is_duplicate_contact else f"amo:contact:{contact_id}:customer:{customer_id}",
        direction=TimelineDirection.SYSTEM,
        subject="AMO contact snapshot",
        text_preview=compact_join(
            [
                f"Ответственный: {safe_text(amo_contact.get('responsible_user_name'))}",
                f"Сделок: {len(split_ids(amo_contact.get('linked_lead_ids')))}",
            ]
        ),
        summary="Read-only AMO contact snapshot",
        match_status=IdentityMatchClass.AMBIGUOUS if is_duplicate_contact else IdentityMatchClass.STRONG_UNIQUE,
        confidence=0.55 if is_duplicate_contact else 0.85,
        record={
            "brand": brand,
            "linked_lead_count": len(split_ids(amo_contact.get("linked_lead_ids"))),
            "date_order_corrected": contact_date_corrected,
            "shared_contact_across_customers": is_duplicate_contact,
        },
        metadata={"brand": brand},
        created_at=generated_at,
    )
    results.append(store.upsert_event(event, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
    for deal in deals_by_contact_id.get(contact_id, ()):
        lead_id = safe_text(deal.get("lead_id"))
        if not lead_id:
            continue
        is_duplicate_lead = lead_id in duplicate_amo_lead_ids
        deal_brand = infer_offline_brand({"contact_brand": brand, "lead_name": deal.get("lead_name"), "pipeline_name": deal.get("pipeline_name")})
        deal_created_at = parse_unix_or_iso(deal.get("created_at"))
        deal_updated_at = parse_unix_or_iso(deal.get("updated_at")) or deal_created_at or updated_at
        opened_at, deal_seen_at, deal_date_corrected = ordered_datetime_pair(
            deal_created_at or deal_updated_at,
            deal_updated_at,
        )
        closed_at = parse_unix_or_iso(deal.get("closed_at"))
        closed_date_ignored = bool(closed_at and closed_at < opened_at)
        if closed_date_ignored:
            closed_at = None
        opportunity: CustomerOpportunity | None = None
        if not is_duplicate_lead:
            opportunity = CustomerOpportunity(
                tenant_id=tenant_id,
                customer_id=customer_id,
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system=AMO_SOURCE,
                source_id=lead_id,
                title=safe_text(deal.get("lead_name")) or f"AMO deal {lead_id}",
                status=safe_text(deal.get("status_name")),
                product_context={"brand": deal_brand, "pipeline": safe_text(deal.get("pipeline_name"))},
                opened_at=opened_at,
                closed_at=closed_at,
                confidence=0.8,
                evidence={"source": "amo_snapshot"},
            )
            results.append(store.upsert_opportunity(opportunity, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
        results.append(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer_id,
                    link_type="amo_lead_id",
                    link_value=lead_id,
                    source_system=AMO_SOURCE,
                    source_ref=f"amo:lead:{lead_id}" if not is_duplicate_lead else f"amo:lead:{lead_id}:customer:{customer_id}",
                    match_class=IdentityMatchClass.AMBIGUOUS if is_duplicate_lead else IdentityMatchClass.STRONG_UNIQUE,
                    confidence=0.55 if is_duplicate_lead else 0.8,
                    first_seen_at=opened_at,
                    last_seen_at=deal_seen_at,
                ),
                actor="canonical_readonly_timeline_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
        deal_event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id if opportunity else None,
            event_type=TimelineEventType.AMO_DEAL_STAGE,
            event_at=deal_seen_at,
            source_system=AMO_SOURCE,
            source_id=lead_id if not is_duplicate_lead else f"{lead_id}:{customer_id}",
            source_ref=f"amo:lead:{lead_id}" if not is_duplicate_lead else f"amo:lead:{lead_id}:customer:{customer_id}",
            direction=TimelineDirection.SYSTEM,
            subject=safe_text(deal.get("lead_name"))[:160] or "AMO deal",
            text_preview=compact_join([safe_text(deal.get("pipeline_name")), safe_text(deal.get("status_name"))])[:240],
            summary=safe_text(deal.get("status_name")),
            stage_after=safe_text(deal.get("status_name")),
            match_status=IdentityMatchClass.AMBIGUOUS if is_duplicate_lead else IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.55 if is_duplicate_lead else 0.8,
            record={
                "brand": deal_brand,
                "price_present": bool(safe_text(deal.get("price"))),
                "loss_reason_present": bool(safe_text(deal.get("loss_reason"))),
                "date_order_corrected": deal_date_corrected,
                "closed_date_ignored": closed_date_ignored,
                "shared_lead_across_customers": is_duplicate_lead,
            },
            metadata={"brand": deal_brand},
            created_at=generated_at,
        )
        results.append(store.upsert_event(deal_event, actor="canonical_readonly_timeline_import", ingestion_run_id=ingestion_run_id))
    return results


def read_mail_aggregates_by_phone(path: Path, *, known_phones: set[str]) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    by_phone: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        query = """
            SELECT p.normalized_phone,
                   c.candidate_key,
                   c.mail_message_count,
                   c.first_mail_date_iso,
                   c.last_mail_date_iso,
                   c.bridge_status,
                   c.blocked_reason,
                   c.tallanto_id,
                   c.amocrm_id
            FROM candidate_phone_refs p
            JOIN candidate_mango_preview c ON c.candidate_key = p.candidate_key
            WHERE p.phone_match_class = 'strong_unique'
        """
        for row in con.execute(query):
            phone = normalize_phone(row["normalized_phone"])
            if not phone or phone not in known_phones:
                continue
            current = by_phone.setdefault(
                phone,
                {
                    "candidate_count": 0,
                    "mail_message_count": 0,
                    "first_mail_date_iso": "",
                    "last_mail_date_iso": "",
                    "bridge_status_counts": Counter(),
                    "blocked_reason_counts": Counter(),
                },
            )
            current["candidate_count"] += 1
            current["mail_message_count"] += int_or_zero(row["mail_message_count"])
            current["bridge_status_counts"][safe_text(row["bridge_status"]) or "unknown"] += 1
            if safe_text(row["blocked_reason"]):
                current["blocked_reason_counts"][safe_text(row["blocked_reason"])] += 1
            first = safe_text(row["first_mail_date_iso"])
            last = safe_text(row["last_mail_date_iso"])
            if first and (not current["first_mail_date_iso"] or first < current["first_mail_date_iso"]):
                current["first_mail_date_iso"] = first
            if last and (not current["last_mail_date_iso"] or last > current["last_mail_date_iso"]):
                current["last_mail_date_iso"] = last
    return {
        phone: {
            **value,
            "bridge_status_counts": dict(value["bridge_status_counts"]),
            "blocked_reason_counts": dict(value["blocked_reason_counts"]),
        }
        for phone, value in by_phone.items()
    }


def mail_aggregate_event(
    *,
    tenant_id: str,
    customer_id: str,
    mail: Mapping[str, Any],
    brand: str,
    fallback_at: datetime,
) -> TimelineEvent:
    event_at = parse_datetime_guess(mail.get("last_mail_date_iso")) or fallback_at
    source_id = stable_digest({"customer_id": customer_id, "mail": dict(mail)})[:24]
    count = int(mail.get("mail_message_count") or 0)
    return TimelineEvent(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_type=TimelineEventType.EMAIL_MESSAGE,
        event_at=event_at,
        source_system=MAIL_SOURCE,
        source_id=source_id,
        source_ref=f"mail_aggregate:{source_id}",
        direction=TimelineDirection.SYSTEM,
        subject="Email aggregate",
        text_preview=f"Связанных писем: {count}",
        summary=f"Email handoff: {count} сообщений; кандидатов: {int(mail.get('candidate_count') or 0)}",
        match_status=IdentityMatchClass.STRONG_UNIQUE,
        confidence=0.75,
        record={
            "brand": brand,
            "candidate_count": int(mail.get("candidate_count") or 0),
            "mail_message_count": count,
            "first_mail_date_iso": safe_text(mail.get("first_mail_date_iso")),
            "last_mail_date_iso": safe_text(mail.get("last_mail_date_iso")),
            "bridge_status_counts": dict(mail.get("bridge_status_counts") or {}),
            "blocked_reason_counts": dict(mail.get("blocked_reason_counts") or {}),
        },
        metadata={"brand": brand, "contains_raw_mail": False},
        created_at=fallback_at,
    )


def build_coverage_report(
    *,
    config: CanonicalReadonlyTimelineConfig,
    generated_at: datetime,
    source_manifest: Mapping[str, Any],
    phones: Sequence[str],
    calls_by_phone: Mapping[str, Sequence[Mapping[str, str]]],
    amo_contacts_by_phone: Mapping[str, Sequence[Mapping[str, str]]],
    mail_by_phone: Mapping[str, Mapping[str, Any]],
    contacts: Sequence[Mapping[str, str]],
    imported_counts: Counter[str],
    write_status_counts: Counter[str],
    manual_review_counts: Counter[str],
    source_event_counts: Counter[str],
    source_customer_counts: Counter[str],
    brand_counts: Counter[str],
    store_summary: Mapping[str, Any],
    duplicate_amo_contact_ids: set[str],
    duplicate_amo_lead_ids: set[str],
    shared_amo_reasons_by_phone: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    total = len(phones)
    tallanto_count = sum(1 for row in contacts if safe_text(row.get("ID Tallanto")) or safe_text(row.get("Статус матчинга Tallanto")))
    amo_count = sum(1 for phone in phones if amo_contacts_by_phone.get(phone))
    mail_count = sum(1 for phone in phones if mail_by_phone.get(phone))
    call_count = sum(1 for phone in phones if calls_by_phone.get(phone))
    summary = {
        "total_customers": total,
        "with_mango_calls": call_count,
        "with_tallanto_context": tallanto_count,
        "with_amo_context": amo_count,
        "with_email_context": mail_count,
        "manual_review_customers_estimated": sum(
            1
            for row in contacts
            if manual_review_reasons(
                row=row,
                calls=calls_by_phone.get(normalize_phone(row.get("Телефон клиента", "")), ()),
                mail=mail_by_phone.get(normalize_phone(row.get("Телефон клиента", ""))),
                duplicate_amo_contact_ids=duplicate_amo_contact_ids,
                duplicate_amo_lead_ids=duplicate_amo_lead_ids,
                extra_reasons=shared_amo_reasons_by_phone.get(normalize_phone(row.get("Телефон клиента", "")), ()),
            )
        ),
        "unknown_brand_customers": int(brand_counts.get("unknown", 0)),
        "source_manual_review_required_customers": sum(
            1 for row in contacts if safe_text(row.get("Нужна ручная проверка")).lower() == "да"
        ),
        "shared_amo_contact_customers": sum(
            1 for reasons in shared_amo_reasons_by_phone.values() if "shared_amo_contact_across_customers" in reasons
        ),
        "shared_amo_lead_customers": sum(
            1 for reasons in shared_amo_reasons_by_phone.values() if "shared_amo_lead_across_customers" in reasons
        ),
        "duplicate_amo_contact_ids": len(duplicate_amo_contact_ids),
        "duplicate_amo_lead_ids": len(duplicate_amo_lead_ids),
        "timeline_primary_read_enabled_allowed": False,
        "timeline_preview_enabled_recommendation": "keep_disabled_until_manual_review_reasons_are_triaged",
    }
    return {
        "schema_version": CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "tenant_id": config.tenant_id,
        "summary": summary,
        "coverage_ratios": {
            "with_mango_calls": ratio(call_count, total),
            "with_tallanto_context": ratio(tallanto_count, total),
            "with_amo_context": ratio(amo_count, total),
            "with_email_context": ratio(mail_count, total),
        },
        "brand_counts": dict(brand_counts),
        "source_customer_counts": dict(source_customer_counts),
        "source_event_counts": dict(source_event_counts),
        "manual_review_reason_counts": dict(manual_review_counts),
        "write_status_counts": dict(write_status_counts),
        "imported_counts": dict(imported_counts),
        "store_counts": dict(store_summary.get("counts", {})),
        "source_freshness": source_freshness(source_manifest, generated_at),
        "primary_read_blockers": primary_read_blockers(
            summary=summary,
            total_customers=total,
            source_manifest=source_manifest,
        ),
        "source_manifest_summary": {
            key: {
                "exists": value.get("exists"),
                "size_bytes": value.get("size_bytes"),
                "row_count": value.get("row_count"),
                "sha256": value.get("sha256"),
            }
            for key, value in source_manifest.items()
            if isinstance(value, Mapping)
        },
        "safety": canonical_readonly_timeline_safety_contract(write_customer_timeline_db=False),
        "limitations": [
            "SQLite contains local customer refs and must stay ignored.",
            "Reports contain aggregate counts only and no raw phones, email values, names, mail text, OCR text, or Telegram raw data.",
            "Telegram is intentionally not imported in v5.",
            "Mail context uses the aggregate Mango bridge DB; mail_handoff_db is included in manifest/freshness but raw mail/link rows are not imported.",
            "with_mango_calls is expected to be 100% because master_contacts is built from Mango phone history.",
            "AMO snapshot freshness must be reviewed before using timeline as a primary answer context.",
            "timeline_primary_read_enabled remains disabled.",
        ],
    }


def render_coverage_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    ratios = report["coverage_ratios"]
    lines = [
        "# Canonical read-only customer_timeline coverage",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- total_customers: `{summary['total_customers']}`",
        f"- with_mango_calls: `{summary['with_mango_calls']}` ({ratios['with_mango_calls']})",
        f"- with_tallanto_context: `{summary['with_tallanto_context']}` ({ratios['with_tallanto_context']})",
        f"- with_amo_context: `{summary['with_amo_context']}` ({ratios['with_amo_context']})",
        f"- with_email_context: `{summary['with_email_context']}` ({ratios['with_email_context']})",
        f"- manual_review_customers_estimated: `{summary['manual_review_customers_estimated']}`",
        f"- timeline_primary_read_enabled_allowed: `{summary['timeline_primary_read_enabled_allowed']}`",
        "",
        "## Manual Review Reasons",
    ]
    for key, value in sorted(report.get("manual_review_reason_counts", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Primary Read Blockers"])
    for item in report.get("primary_read_blockers", ()):
        lines.append(f"- `{item['reason']}`: `{item['value']}`")
    lines.extend(["", "## Source Freshness"])
    for key, value in sorted(report.get("source_freshness", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Safety", "- No AMO/Tallanto/CRM writes.", "- No Telegram/email send.", "- No ASR/R+A.", "- No stable_runtime mutation."])
    return "\n".join(lines) + "\n"


def manual_review_reasons(
    *,
    row: Mapping[str, str],
    calls: Sequence[Mapping[str, str]],
    mail: Optional[Mapping[str, Any]],
    duplicate_amo_contact_ids: set[str] | None = None,
    duplicate_amo_lead_ids: set[str] | None = None,
    extra_reasons: Sequence[str] = (),
) -> list[str]:
    reasons: list[str] = []
    duplicate_amo_contact_ids = duplicate_amo_contact_ids or set()
    duplicate_amo_lead_ids = duplicate_amo_lead_ids or set()
    if safe_text(row.get("Нужна ручная проверка")).lower() == "да":
        reasons.append("source_manual_review_required")
    if len(split_ids(row.get("ID Tallanto"))) > 1 or tallanto_multiple_candidate(row):
        reasons.append("multiple_tallanto_candidates")
    if safe_text(row.get("Статус матчинга Tallanto")).lower() in {"", "missing", "not_found", "none", "нет"}:
        reasons.append("missing_tallanto_context")
    if len(split_ids(row.get("AMO contact IDs"))) > 1:
        reasons.append("multiple_amo_contacts")
    if any(contact_id in duplicate_amo_contact_ids for contact_id in split_ids(row.get("AMO contact IDs"))):
        reasons.append("shared_amo_contact_across_customers")
    if len(split_ids(row.get("AMO lead IDs"))) > 1:
        reasons.append("multiple_amo_deals")
    if any(lead_id in duplicate_amo_lead_ids for lead_id in split_ids(row.get("AMO lead IDs"))):
        reasons.append("shared_amo_lead_across_customers")
    reasons.extend(extra_reasons)
    if mail and mail.get("blocked_reason_counts"):
        reasons.append("email_bridge_blocked_or_ambiguous")
    if not calls:
        reasons.append("no_mango_calls")
    return sorted(set(reasons))


def canonical_readonly_timeline_safety_contract(*, write_customer_timeline_db: bool) -> Mapping[str, Any]:
    return {
        **customer_timeline_safety_contract(),
        **customer_timeline_sqlite_safety_contract(),
        "schema_version": CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION,
        "read_stable_runtime_as_source": True,
        "write_customer_timeline_db": bool(write_customer_timeline_db),
        "write_product_timeline_db": bool(write_customer_timeline_db),
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "run_asr": False,
        "run_ra": False,
        "stable_runtime_writes": False,
        "raw_personal_values_in_reports": False,
        "telegram_import_enabled": False,
        "timeline_preview_enabled_allowed": False,
        "timeline_primary_read_enabled_allowed": False,
    }


def build_source_manifest(paths: Mapping[str, Optional[Path]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for key, path in paths.items():
        if path is None:
            manifest[key] = {"exists": False}
            continue
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            stat = path.stat()
            item.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": file_sha256(path),
                }
            )
            if path.suffix.lower() == ".csv":
                item["row_count"] = csv_row_count(path)
            elif path.suffix.lower() in {".sqlite", ".db", ".sqlite3"}:
                item["table_counts"] = sqlite_table_counts(path)
        manifest[key] = item
    return manifest


def duplicate_ids_across_phones(rows: Sequence[Mapping[str, str]], column: str) -> set[str]:
    phones_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        phone = normalize_phone(row.get("Телефон клиента", ""))
        if not phone:
            continue
        for value in split_ids(row.get(column)):
            phones_by_id[value].add(phone)
    return {value for value, phones in phones_by_id.items() if len(phones) > 1}


def duplicate_amo_ids_across_sources(
    rows: Sequence[Mapping[str, str]],
    *,
    amo_contacts_by_phone: Mapping[str, Sequence[Mapping[str, str]]],
    amo_deals_by_contact_id: Mapping[str, Sequence[Mapping[str, str]]],
) -> tuple[set[str], set[str]]:
    contact_phones_by_id: dict[str, set[str]] = defaultdict(set)
    lead_phones_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        phone = normalize_phone(row.get("Телефон клиента", ""))
        if not phone:
            continue
        for contact_id in split_ids(row.get("AMO contact IDs")):
            contact_phones_by_id[contact_id].add(phone)
        for lead_id in split_ids(row.get("AMO lead IDs")):
            lead_phones_by_id[lead_id].add(phone)
    for phone, amo_contacts in amo_contacts_by_phone.items():
        for amo_contact in amo_contacts:
            contact_id = safe_text(amo_contact.get("contact_id"))
            if contact_id:
                contact_phones_by_id[contact_id].add(phone)
            for lead_id in split_ids(amo_contact.get("linked_lead_ids")):
                lead_phones_by_id[lead_id].add(phone)
            for deal in amo_deals_by_contact_id.get(contact_id, ()):
                lead_id = safe_text(deal.get("lead_id"))
                if lead_id:
                    lead_phones_by_id[lead_id].add(phone)
    duplicate_contacts = {value for value, phones in contact_phones_by_id.items() if len(phones) > 1}
    duplicate_leads = {value for value, phones in lead_phones_by_id.items() if len(phones) > 1}
    return duplicate_contacts, duplicate_leads


def shared_amo_reasons_by_phone_from_sources(
    rows: Sequence[Mapping[str, str]],
    *,
    amo_contacts_by_phone: Mapping[str, Sequence[Mapping[str, str]]],
    amo_deals_by_contact_id: Mapping[str, Sequence[Mapping[str, str]]],
    duplicate_amo_contact_ids: set[str],
    duplicate_amo_lead_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    reasons_by_phone: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        phone = normalize_phone(row.get("Телефон клиента", ""))
        if not phone:
            continue
        if any(contact_id in duplicate_amo_contact_ids for contact_id in split_ids(row.get("AMO contact IDs"))):
            reasons_by_phone[phone].add("shared_amo_contact_across_customers")
        if any(lead_id in duplicate_amo_lead_ids for lead_id in split_ids(row.get("AMO lead IDs"))):
            reasons_by_phone[phone].add("shared_amo_lead_across_customers")
    for phone, amo_contacts in amo_contacts_by_phone.items():
        for amo_contact in amo_contacts:
            contact_id = safe_text(amo_contact.get("contact_id"))
            if contact_id in duplicate_amo_contact_ids:
                reasons_by_phone[phone].add("shared_amo_contact_across_customers")
            if any(lead_id in duplicate_amo_lead_ids for lead_id in split_ids(amo_contact.get("linked_lead_ids"))):
                reasons_by_phone[phone].add("shared_amo_lead_across_customers")
            for deal in amo_deals_by_contact_id.get(contact_id, ()):
                lead_id = safe_text(deal.get("lead_id"))
                if lead_id in duplicate_amo_lead_ids:
                    reasons_by_phone[phone].add("shared_amo_lead_across_customers")
    return {phone: tuple(sorted(reasons)) for phone, reasons in reasons_by_phone.items()}


def primary_read_blockers(
    *,
    summary: Mapping[str, Any],
    total_customers: int,
    source_manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    manual_review_count = int(summary.get("manual_review_customers_estimated") or 0)
    unknown_brand_count = int(summary.get("unknown_brand_customers") or 0)
    return [
        {
            "reason": "manual_review_customers_estimated",
            "value": manual_review_count,
            "ratio": ratio(manual_review_count, total_customers),
        },
        {
            "reason": "unknown_brand_customers",
            "value": unknown_brand_count,
            "ratio": ratio(unknown_brand_count, total_customers),
        },
        {
            "reason": "shared_amo_contact_customers",
            "value": int(summary.get("shared_amo_contact_customers") or 0),
            "duplicate_ids": int(summary.get("duplicate_amo_contact_ids") or 0),
        },
        {
            "reason": "shared_amo_lead_customers",
            "value": int(summary.get("shared_amo_lead_customers") or 0),
            "duplicate_ids": int(summary.get("duplicate_amo_lead_ids") or 0),
        },
        {
            "reason": "amo_snapshot_freshness_requires_review",
            "value": bool(source_manifest.get("amo_contacts_csv", {}).get("mtime_ns")),
        },
    ]


def source_freshness(source_manifest: Mapping[str, Any], generated_at: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in source_manifest.items():
        if not isinstance(value, Mapping) or not value.get("mtime_ns"):
            continue
        mtime = datetime.fromtimestamp(int(value["mtime_ns"]) / 1_000_000_000, tz=timezone.utc)
        result[key] = {
            "mtime": mtime.isoformat(),
            "age_days": round((generated_at - mtime).total_seconds() / 86400, 2),
        }
    return result


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def sqlite_table_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only = ON")
        tables = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        for table in tables:
            if table.startswith("sqlite_"):
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
                counts[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if not resolved.is_file():
        raise ValueError(f"expected file path: {resolved}")
    return resolved


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def split_ids(value: Any) -> list[str]:
    return [part for part in re.split(r"[|,;\s]+", safe_text(value)) if part]


def extract_phones_from_text(value: Any) -> list[str]:
    phones: list[str] = []
    for raw in re.findall(r"(?:\+?7|8)?[\s(\\-]*\d{3}[\s)\\-]*\d{3}[\s\\-]*\d{2}[\s\\-]*\d{2}", safe_text(value)):
        phone = normalize_phone(raw)
        if phone:
            phones.append(phone)
    return sorted(set(phones))


def parse_datetime_guess(value: Any) -> Optional[datetime]:
    text = safe_text(value)
    if not text:
        return None
    candidates = [text, text.replace(" ", "T")]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_unix_or_iso(value: Any) -> Optional[datetime]:
    text = safe_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d{10,13}", text):
        number = int(text)
        if number > 10_000_000_000:
            number = number // 1000
        return datetime.fromtimestamp(number, tz=timezone.utc)
    return parse_datetime_guess(text)


def ordered_datetime_pair(first: datetime, last: datetime) -> tuple[datetime, datetime, bool]:
    if last < first:
        return last, first, True
    return first, last, False


def tallanto_match_class(value: Any) -> IdentityMatchClass:
    text = safe_text(value).lower()
    if any(marker in text for marker in ("no_exact", "no exact", "not_exact", "not exact", "no_match", "no match", "unmatched")):
        return IdentityMatchClass.UNMATCHED
    if text in {"", "missing", "not_found", "none", "нет"}:
        return IdentityMatchClass.UNMATCHED
    if any(marker in text for marker in ("multiple", "ambiguous", "duplicate", "несколько")):
        return IdentityMatchClass.AMBIGUOUS
    if any(marker in text for marker in ("single", "exact", "strong", "точ")):
        return IdentityMatchClass.STRONG_UNIQUE
    return IdentityMatchClass.INFERRED


def call_direction(value: Any) -> TimelineDirection:
    text = safe_text(value).lower()
    if "исход" in text or "out" in text:
        return TimelineDirection.OUTBOUND
    if "вход" in text or "in" in text:
        return TimelineDirection.INBOUND
    return TimelineDirection.SYSTEM


def infer_brand(values: Iterable[Any], *, mode: str = "legacy") -> str:
    text = " ".join(safe_text(value).lower() for value in values)
    normalized_mode = safe_text(mode).casefold() or "legacy"
    if normalized_mode == "cyrillic_v2":
        return infer_brand_cyrillic_v2(text)
    if "унпк" in text or "unpk" in text:
        return "unpk"
    if "фотон" in text or "foton" in text:
        return "foton"
    return "unknown"


def infer_offline_brand(values: Mapping[str, Any] | Iterable[Any]) -> str:
    if isinstance(values, Mapping):
        return infer_brand_cyrillic_v2_record(values)
    return infer_brand(values, mode=OFFLINE_BRAND_INFER_MODE)


def infer_brand_cyrillic_v2(text: str) -> str:
    hits = brand_root_hits(safe_text(text))
    has_foton = hits["foton"]
    has_unpk = hits["unpk"]
    if has_foton and has_unpk:
        return "unknown"
    if has_foton:
        return "foton"
    if has_unpk:
        return "unpk"
    return "unknown"


def infer_brand_cyrillic_v2_record(row: Mapping[str, Any]) -> str:
    foton_score = 0
    unpk_score = 0
    explicit_unpk = False
    short_mixed = False
    long_mixed_foton = False
    long_mixed_unpk = False

    for key, value in row.items():
        text = safe_text(value)
        if not text:
            continue
        hits = brand_root_hits(text)
        field = safe_text(key).casefold().replace("ё", "е")
        if hits["explicit_unpk"]:
            explicit_unpk = True
        weight = 1
        if "филиал" in field or "branch" in field:
            weight = 1
        if hits["foton"] and hits["unpk"]:
            if len(text) <= 160 or hits["explicit_unpk"]:
                short_mixed = True
            else:
                long_mixed_foton = True
                long_mixed_unpk = True
            continue
        if hits["foton"]:
            foton_score += weight
        elif hits["unpk"]:
            unpk_score += weight

    if short_mixed:
        return "unknown"
    if foton_score and unpk_score:
        if explicit_unpk:
            return "unknown"
        return "foton"
    if foton_score:
        return "foton"
    if long_mixed_foton and long_mixed_unpk and not explicit_unpk:
        return "foton"
    if unpk_score:
        return "unpk"
    if long_mixed_unpk:
        return "unknown"
    return "unknown"


def brand_root_hits(text: str) -> Mapping[str, bool]:
    normalized = safe_text(text).casefold().replace("ё", "е")
    compact = re.sub(r"\s+", "", normalized)
    has_foton = has_foton_root(normalized)
    has_unpk_token = "унпк" in compact or "unpk" in compact
    has_mfti = "мфти" in compact
    return {
        "foton": has_foton,
        "unpk": has_unpk_token or has_mfti,
        "explicit_unpk": has_unpk_token,
        "mfti": has_mfti,
    }


def has_foton_root(compact_text: str) -> bool:
    for token in ("foton", "фотон"):
        start = 0
        while True:
            index = compact_text.find(token, start)
            if index < 0:
                break
            suffix = compact_text[index + len(token) :]
            if not suffix or not re.match(r"[a-zа-я0-9]", suffix[0]):
                return True
            if token == "фотон":
                if suffix.startswith(("а", "у", "е", "ы")) and (len(suffix) == 1 or not re.match(r"[а-я0-9]", suffix[1])):
                    return True
                if suffix.startswith("ом") and (len(suffix) == 2 or not re.match(r"[а-я0-9]", suffix[2])):
                    return True
            start = index + len(token)
    return False


def compact_join(parts: Sequence[Any]) -> str:
    return " | ".join(safe_text(part) for part in parts if safe_text(part))[:500]


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value).replace(",", ".")))
    except ValueError:
        return 0


def safe_text(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION",
    "CanonicalReadonlyTimelineConfig",
    "build_canonical_readonly_customer_timeline",
    "canonical_readonly_timeline_safety_contract",
    "duplicate_amo_ids_across_sources",
    "duplicate_ids_across_phones",
    "primary_read_blockers",
    "resolve_config",
    "shared_amo_reasons_by_phone_from_sources",
    "split_ids",
]
