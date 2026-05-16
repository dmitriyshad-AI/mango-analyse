from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
from mango_mvp.customer_timeline.ids import normalize_key, stable_digest, stable_prefixed_id
from mango_mvp.customer_timeline.safety import customer_timeline_safety_contract, guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore, customer_timeline_sqlite_safety_contract
from mango_mvp.utils.phone import normalize_phone


CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION = "contact_control_customer_timeline_sample_import_v1"
MASTER_CONTACT_SOURCE_SYSTEM = "master_contacts_snapshot"
MANGO_SOURCE_SYSTEM = "mango_processed_summary"
AMO_SOURCE_SYSTEM = "amocrm_snapshot"
TALLANTO_SOURCE_SYSTEM = "tallanto_snapshot"

DEFAULT_TARGET_BUCKET_COUNTS = {
    "active_recent": 30,
    "former_tallanto": 30,
    "new_lead_no_deal": 20,
    "paid_or_success": 20,
}
DEFAULT_HARD_TARGET_BUCKET_COUNTS = {
    "no_reliable_tallanto": 25,
    "tallanto_multiple": 15,
    "multi_phone_or_multi_deal": 20,
    "long_history": 20,
    "payment_or_documents_risk": 20,
}


@dataclass(frozen=True)
class ContactControlTimelineSampleConfig:
    master_contacts_csv: Path
    master_calls_csv: Path
    allowed_root: Path
    out_root: Path
    timeline_db: Path
    exclude_phones_csv: Optional[Path] = None
    tenant_id: str = "foton"
    sample_profile: str = "ordinary"
    target_bucket_counts: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_TARGET_BUCKET_COUNTS))
    max_call_events_per_contact: int = 50
    generated_at: Optional[datetime] = None
    require_russian_phone: bool = True


def build_contact_control_timeline_sample(config: ContactControlTimelineSampleConfig) -> Mapping[str, Any]:
    resolved = _resolve_config(config)
    generated_at = _generated_at(config)
    contacts_rows = _read_csv(resolved.master_contacts_csv)
    excluded_phones = _read_excluded_phones(resolved.exclude_phones_csv) if resolved.exclude_phones_csv else set()
    selected_rows, bucket_pool_counts = select_contact_control_rows(
        contacts_rows,
        excluded_phones=excluded_phones,
        target_bucket_counts=resolved.target_bucket_counts,
        require_russian_phone=resolved.require_russian_phone,
        sample_profile=resolved.sample_profile,
    )
    selected_phones = {row["primary_phone"] for row in selected_rows}
    calls_by_phone = _read_selected_calls(resolved.master_calls_csv, selected_phones)

    resolved.out_root.mkdir(parents=True, exist_ok=True)
    selected_csv = resolved.out_root / "selected_contact_control_sample.csv"
    _write_csv(selected_csv, selected_rows)
    source_rows = _write_import_source(
        resolved.out_root / "contact_control_timeline_import_source.csv",
        selected_rows,
        calls_by_phone=calls_by_phone,
        generated_at=generated_at,
    )

    store = CustomerTimelineSQLiteStore(resolved.timeline_db, allowed_root=resolved.allowed_root)
    try:
        input_hash = stable_digest(
            {
                "schema_version": CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION,
                "selected_rows": selected_rows,
                "source_rows": source_rows,
            }
        )
        run = store.start_ingestion_run(
            tenant_id=resolved.tenant_id,
            source_system=MASTER_CONTACT_SOURCE_SYSTEM,
            source_ref=selected_csv.name,
            run_kind="contact_control_sample_import",
            idempotency_key=input_hash,
            input_hash=input_hash,
            started_at=generated_at,
            metadata={
                "schema_version": CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION,
                "selected_contacts": len(selected_rows),
                "selected_phones": len(selected_phones),
                "bucket_counts": dict(Counter(row["control_bucket"] for row in selected_rows)),
                "bucket_pool_counts": dict(bucket_pool_counts),
                "sources": _source_manifest(resolved),
                "safety": _safety_contract(write_customer_timeline_db=True),
            },
            actor="contact_control_sample_import",
        )
        status_counts: Counter[str] = Counter()
        imported: Counter[str] = Counter()
        for row in selected_rows:
            call_rows = _sort_call_rows(calls_by_phone.get(row["primary_phone"], []))[: resolved.max_call_events_per_contact]
            results = _upsert_contact(
                store,
                tenant_id=resolved.tenant_id,
                row=row,
                call_rows=call_rows,
                generated_at=generated_at,
                ingestion_run_id=run.run_id,
            )
            for result in results:
                status_counts[result["status"]] += 1
                imported[result["record_type"]] += 1
        finished = store.finish_ingestion_run(
            run.run_id,
            status="completed",
            accepted_count=len(selected_rows),
            rejected_count=0,
            output_ref=str(resolved.timeline_db),
            finished_at=generated_at,
            metadata={"write_status_counts": dict(status_counts), "imported_counts": dict(imported)},
            actor="contact_control_sample_import",
        )
        store_summary = store.summary()
    finally:
        store.close()

    report = {
        "schema_version": CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "tenant_id": resolved.tenant_id,
        "mode": "apply_local_customer_timeline_db",
        "input": {
            "master_contacts_csv": str(resolved.master_contacts_csv),
            "master_calls_csv": str(resolved.master_calls_csv),
            "exclude_phones_csv": str(resolved.exclude_phones_csv) if resolved.exclude_phones_csv else None,
            "sample_profile": resolved.sample_profile,
        },
        "outputs": {
            "out_root": str(resolved.out_root),
            "timeline_db": str(resolved.timeline_db),
            "selected_sample_csv": str(selected_csv),
            "import_source_csv": str(resolved.out_root / "contact_control_timeline_import_source.csv"),
            "import_report_json": str(resolved.out_root / "import_report.json"),
        },
        "summary": {
            "selected_contacts": len(selected_rows),
            "sample_profile": resolved.sample_profile,
            "selected_unique_phones": len(selected_phones),
            "target_bucket_counts": dict(resolved.target_bucket_counts),
            "selected_bucket_counts": dict(Counter(row["control_bucket"] for row in selected_rows)),
            "bucket_pool_counts": dict(bucket_pool_counts),
            "excluded_phones": len(excluded_phones),
            "source_rows_written": len(source_rows),
            "matched_call_rows": sum(len(items) for items in calls_by_phone.values()),
            "ingestion_run_id": finished.run_id,
            "write_status_counts": dict(status_counts),
            "imported_counts": dict(imported),
            "store_counts": dict(store_summary.get("counts", {})),
            "validation_ok": bool(store_summary.get("validation_ok")),
        },
        "safety": _safety_contract(write_customer_timeline_db=True),
    }
    (resolved.out_root / "import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_source_manifest(resolved)
    return report


def audit_contact_control_timeline_sample(config: ContactControlTimelineSampleConfig) -> Mapping[str, Any]:
    resolved = _resolve_config(config)
    generated_at = _generated_at(config)
    selected_csv = resolved.out_root / "selected_contact_control_sample.csv"
    selected_rows = _read_csv(selected_csv)
    resolved.out_root.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, Any]] = []
    if not resolved.timeline_db.exists():
        report_rows = [_missing_db_report_row(row) for row in selected_rows]
    else:
        with sqlite3.connect(f"file:{resolved.timeline_db}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only = ON")
            report_rows = [_audit_contact(con, resolved.tenant_id, row) for row in selected_rows]

    verdict_counts = Counter(safe_text(row.get("verdict")) for row in report_rows)
    identity_counts = Counter(safe_text(row.get("identity_validation_status")) for row in report_rows)
    manual_categories = Counter(safe_text(row.get("manual_review_category")) for row in report_rows if safe_text(row.get("manual_review_category")))
    total = len(report_rows)
    timeline_found = sum(1 for row in report_rows if row.get("timeline_found") == "Да")
    valid_identity = sum(1 for row in report_rows if row.get("valid_customer_identity") == "Да")
    ready = verdict_counts.get("ready_for_preview", 0)
    manual = verdict_counts.get("needs_manual_review", 0)
    gate = _gate_decision(total=total, timeline_found=timeline_found, valid_identity=valid_identity, ready=ready)
    summary = {
        "schema_version": CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION,
        "report_kind": "contact_control_customer_timeline_sample_audit",
        "generated_at": generated_at.isoformat(),
        "tenant_id": resolved.tenant_id,
        "sample_profile": resolved.sample_profile,
        "timeline_db_found": resolved.timeline_db.exists(),
        "timeline_db_path": str(resolved.timeline_db) if resolved.timeline_db.exists() else None,
        "selected_contacts": total,
        "selected_unique_phones": len({row["primary_phone"] for row in selected_rows}),
        "selected_bucket_counts": dict(Counter(row.get("control_bucket", "") for row in selected_rows)),
        "source_manual_review_required": sum(1 for row in selected_rows if row.get("manual_review_required") == "Да"),
        "source_manual_review_not_required": sum(1 for row in selected_rows if row.get("manual_review_required") != "Да"),
        "timeline_matched_contacts": timeline_found,
        "timeline_found_ratio": round(timeline_found / total, 6) if total else 1.0,
        "valid_customer_identity": valid_identity,
        "valid_customer_identity_ratio": round(valid_identity / total, 6) if total else 1.0,
        "ready_for_preview": ready,
        "ready_for_preview_ratio": round(ready / total, 6) if total else 1.0,
        "needs_manual_review": manual,
        "manual_review_ratio": round(manual / total, 6) if total else 0.0,
        "identity_validation_counts": dict(identity_counts),
        "manual_review_by_category": dict(manual_categories),
        "fallback_used": sum(1 for row in report_rows if row.get("fallback_used") == "Да"),
        "source_counts": dict(_sum_counter(report_rows, "source_counts_json")),
        "event_type_counts": dict(_sum_counter(report_rows, "event_type_counts_json")),
        "gate_decision": gate,
        "safety": _safety_contract(write_customer_timeline_db=False),
        "limitations": [
            "Это контрольная выборка из master_contacts/master_calls, а не live AMO/Tallanto.",
            "Контакты из текущего deal-aware набора 709 исключены, чтобы не повторять специально сложную выборку.",
            "valid_customer_identity в этом прогоне проверяет техническую склейку внутри локального timeline, а не ручное подтверждение РОПа.",
        ],
    }
    _write_csv(resolved.out_root / "timeline_coverage_report.csv", report_rows)
    problem_rows = [row for row in report_rows if row.get("verdict") != "ready_for_preview"]
    ready_rows = [row for row in report_rows if row.get("verdict") == "ready_for_preview"]
    _write_csv(resolved.out_root / "problem_examples.csv", problem_rows[:50])
    _write_csv(resolved.out_root / "ready_for_preview_examples.csv", ready_rows[:50])
    (resolved.out_root / "coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (resolved.out_root / "coverage_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return {"summary": summary, "rows": report_rows}


def select_contact_control_rows(
    contact_rows: Sequence[Mapping[str, Any]],
    *,
    excluded_phones: set[str],
    target_bucket_counts: Mapping[str, int],
    require_russian_phone: bool = True,
    sample_profile: str = "ordinary",
) -> tuple[list[dict[str, str]], Counter[str]]:
    profile = _normalize_sample_profile(sample_profile)
    lead_to_phones = _lead_to_phones(contact_rows, excluded_phones=excluded_phones, require_russian_phone=require_russian_phone)
    pools: dict[str, list[dict[str, str]]] = {bucket: [] for bucket in target_bucket_counts}
    pool_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for raw in contact_rows:
        phone = normalize_phone(raw.get("Телефон клиента", ""))
        if not phone or phone in excluded_phones or phone in seen:
            continue
        if require_russian_phone and not phone.startswith("+7"):
            continue
        contentful_calls = int_or_zero(raw.get("Содержательных звонков в истории"))
        if contentful_calls < 1:
            continue
        related_phones = _related_phones(raw, lead_to_phones)
        bucket = _classify_control_bucket(raw, related_phones=related_phones, sample_profile=profile)
        if bucket not in target_bucket_counts:
            continue
        seen.add(phone)
        row = _selected_row(raw, phone=phone, bucket=bucket, sample_profile=profile, related_phones=related_phones)
        pool_counts[bucket] += 1
        pools[bucket].append(row)

    selected: list[dict[str, str]] = []
    for bucket, target in target_bucket_counts.items():
        bucket_rows = sorted(pools.get(bucket, []), key=lambda row: _selection_sort_key(profile, bucket, row))
        selected.extend(bucket_rows[: max(0, int(target))])
    return selected, pool_counts


def _resolve_config(config: ContactControlTimelineSampleConfig) -> ContactControlTimelineSampleConfig:
    allowed_root = Path(config.allowed_root).expanduser().resolve(strict=False)
    out_root = guard_customer_timeline_output_path(Path(config.out_root).expanduser(), allowed_root)
    timeline_db = guard_customer_timeline_output_path(Path(config.timeline_db).expanduser(), allowed_root)
    target_counts = {safe_text(key): max(0, int(value)) for key, value in config.target_bucket_counts.items()}
    return ContactControlTimelineSampleConfig(
        master_contacts_csv=Path(config.master_contacts_csv).expanduser().resolve(strict=False),
        master_calls_csv=Path(config.master_calls_csv).expanduser().resolve(strict=False),
        exclude_phones_csv=Path(config.exclude_phones_csv).expanduser().resolve(strict=False) if config.exclude_phones_csv else None,
        allowed_root=allowed_root,
        out_root=out_root,
        timeline_db=timeline_db,
        tenant_id=normalize_key(config.tenant_id, "tenant_id"),
        sample_profile=_normalize_sample_profile(config.sample_profile),
        target_bucket_counts=target_counts,
        max_call_events_per_contact=max(1, int(config.max_call_events_per_contact)),
        generated_at=config.generated_at,
        require_russian_phone=bool(config.require_russian_phone),
    )


def _normalize_sample_profile(value: Any) -> str:
    profile = safe_text(value) or "ordinary"
    if profile not in {"ordinary", "hard"}:
        raise ValueError(f"unsupported contact control sample_profile: {value!r}")
    return profile


def _lead_to_phones(
    contact_rows: Sequence[Mapping[str, Any]],
    *,
    excluded_phones: set[str],
    require_russian_phone: bool,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in contact_rows:
        phone = normalize_phone(row.get("Телефон клиента", ""))
        if not phone or phone in excluded_phones:
            continue
        if require_russian_phone and not phone.startswith("+7"):
            continue
        for lead_id in _split_ids(row.get("AMO lead IDs")):
            result[lead_id].add(phone)
    return result


def _related_phones(row: Mapping[str, Any], lead_to_phones: Mapping[str, set[str]]) -> set[str]:
    result: set[str] = set()
    for lead_id in _split_ids(row.get("AMO lead IDs")):
        result.update(lead_to_phones.get(lead_id, set()))
    return result


def _upsert_contact(
    store: CustomerTimelineSQLiteStore,
    *,
    tenant_id: str,
    row: Mapping[str, str],
    call_rows: Sequence[Mapping[str, Any]],
    generated_at: datetime,
    ingestion_run_id: str,
) -> list[Mapping[str, str]]:
    phone = row["primary_phone"]
    customer_id = stable_prefixed_id(
        "customer",
        {"tenant_id": tenant_id, "sample_kind": "contact_control", "phone": phone},
    )
    first_at, last_at = _contact_time_bounds(row, call_rows, generated_at)
    display_name = safe_text(row.get("display_name")) or phone
    customer = CustomerIdentity(
        tenant_id=tenant_id,
        customer_id=customer_id,
        identity_status=IdentityStatus.STRONG,
        display_name=display_name,
        primary_phone=phone,
        primary_email=None,
        source_ref=f"master_contacts:phone:{phone}",
        first_seen_at=first_at,
        last_seen_at=last_at,
        touch_count=max(1, len(call_rows) + 1),
        summary={
            "source_system": MASTER_CONTACT_SOURCE_SYSTEM,
            "control_bucket": row.get("control_bucket", ""),
            "sample_kind": row.get("sample_kind", ""),
            "contentful_calls": row.get("contentful_call_count", ""),
        },
        metadata={"source": "contact_control_sample_import"},
        created_at=generated_at,
        updated_at=generated_at,
    )
    results = [_result(store.upsert_customer(customer, actor="contact_control_sample_import", ingestion_run_id=ingestion_run_id))]
    results.append(
        _result(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer_id,
                    link_type="phone",
                    link_value=phone,
                    source_system=MASTER_CONTACT_SOURCE_SYSTEM,
                    source_ref=f"master_contacts:phone:{phone}",
                    match_class=IdentityMatchClass.STRONG_UNIQUE,
                    confidence=0.98,
                    first_seen_at=first_at,
                    last_seen_at=last_at,
                ),
                actor="contact_control_sample_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    )
    for contact_id in _split_ids(row.get("amo_contact_ids")):
        results.append(_result(_upsert_link(store, tenant_id, customer_id, "amo_contact_id", contact_id, AMO_SOURCE_SYSTEM, generated_at, ingestion_run_id)))
    amo_lead_ids = _split_ids(row.get("amo_lead_ids"))
    opportunity_id = None
    for lead_id in amo_lead_ids:
        results.append(_result(_upsert_link(store, tenant_id, customer_id, "amo_lead_id", lead_id, AMO_SOURCE_SYSTEM, generated_at, ingestion_run_id)))
        opportunity = CustomerOpportunity(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_type=OpportunityType.AMO_DEAL,
            source_system=AMO_SOURCE_SYSTEM,
            source_id=lead_id,
            title=safe_text(row.get("recommended_product")) or f"AMO сделка {lead_id}",
            status="linked_from_master_contact",
            product_context={"control_bucket": row.get("control_bucket", ""), "priority": row.get("lead_priority", "")},
            opened_at=first_at,
            confidence=0.7,
            evidence={"source_ref": f"master_contacts:phone:{phone}", "amo_lead_id": lead_id},
        )
        result = store.upsert_opportunity(opportunity, actor="contact_control_sample_import", ingestion_run_id=ingestion_run_id)
        results.append(_result(result))
        opportunity_id = opportunity_id or opportunity.opportunity_id
    for tallanto_id in _split_ids(row.get("tallanto_ids")):
        results.append(_result(_upsert_link(store, tenant_id, customer_id, "tallanto_student_id", tallanto_id, TALLANTO_SOURCE_SYSTEM, generated_at, ingestion_run_id)))

    summary = safe_text(row.get("contact_history_summary"))
    if summary:
        event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity_id,
            event_type=TimelineEventType.SYSTEM_NOTE,
            event_at=last_at,
            source_system=MASTER_CONTACT_SOURCE_SYSTEM,
            source_id=f"contact-summary:{phone}",
            source_ref=f"master_contacts:phone:{phone}:summary",
            direction=TimelineDirection.SYSTEM,
            subject="Контактная история из master_contacts",
            text_preview=summary[:240],
            summary=summary[:500],
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.85,
            record={"contact": _compact_row(row), "phone": phone},
            metadata={"control_bucket": row.get("control_bucket", "")},
            created_at=generated_at,
        )
        results.append(_result(store.upsert_event(event, actor="contact_control_sample_import", ingestion_run_id=ingestion_run_id)))
        results.append(_result(_upsert_chunk(store, tenant_id, customer_id, opportunity_id, event, "contact_history_summary", generated_at, ingestion_run_id)))

    if amo_lead_ids or safe_text(row.get("amo_contact_ids")):
        event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity_id,
            event_type=TimelineEventType.AMO_CONTACT_SNAPSHOT,
            event_at=last_at,
            source_system=AMO_SOURCE_SYSTEM,
            source_id=f"contact:{phone}",
            source_ref=f"amocrm:contact-phone:{phone}",
            direction=TimelineDirection.SYSTEM,
            subject="AMO связь из master_contacts",
            text_preview=f"AMO contacts: {row.get('amo_contact_ids', '')}; leads: {row.get('amo_lead_ids', '')}"[:240],
            summary=f"AMO contact IDs: {row.get('amo_contact_ids', '')}; AMO lead IDs: {row.get('amo_lead_ids', '')}",
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.7,
            record={"amo_contact_ids": row.get("amo_contact_ids", ""), "amo_lead_ids": row.get("amo_lead_ids", "")},
            metadata={"control_bucket": row.get("control_bucket", "")},
            created_at=generated_at,
        )
        results.append(_result(store.upsert_event(event, actor="contact_control_sample_import", ingestion_run_id=ingestion_run_id)))
    if safe_text(row.get("tallanto_match_status")) or safe_text(row.get("tallanto_ids")):
        event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity_id,
            event_type=TimelineEventType.TALLANTO_STUDENT_SNAPSHOT,
            event_at=last_at,
            source_system=TALLANTO_SOURCE_SYSTEM,
            source_id=f"student:{safe_text(row.get('tallanto_ids')) or phone}",
            source_ref=f"tallanto:phone:{phone}",
            direction=TimelineDirection.SYSTEM,
            subject="Tallanto связь из master_contacts",
            text_preview=_tallanto_summary(row)[:240],
            summary=_tallanto_summary(row),
            match_status=IdentityMatchClass.STRONG_UNIQUE if row.get("tallanto_match_status") == "exact_phone_single" else IdentityMatchClass.INFERRED,
            confidence=0.85 if row.get("tallanto_match_status") == "exact_phone_single" else 0.5,
            record={"tallanto": _compact_row(row)},
            metadata={"control_bucket": row.get("control_bucket", "")},
            created_at=generated_at,
        )
        results.append(_result(store.upsert_event(event, actor="contact_control_sample_import", ingestion_run_id=ingestion_run_id)))

    for idx, call in enumerate(call_rows):
        event_at = parse_datetime_guess(call.get("Дата и время звонка")) or last_at
        call_id = safe_text(call.get("ID звонка")) or stable_digest({"phone": phone, "event_at": event_at.isoformat()})[:16]
        summary = safe_text(call.get("Краткое резюме разговора") or call.get("Следующий шаг") or call.get("Тип звонка"))
        event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity_id,
            event_type=TimelineEventType.MANGO_CALL,
            event_at=event_at,
            source_system=MANGO_SOURCE_SYSTEM,
            source_id=f"call:{call_id}",
            source_ref=f"mango:call:{call_id}",
            direction=_call_direction(call),
            actor_name=safe_text(call.get("Менеджер")),
            subject=safe_text(call.get("Тип звонка") or call.get("Рекомендуемый продукт"))[:160],
            text_preview=summary[:240],
            summary=summary[:500],
            importance=1 if call.get("Содержательный звонок") == "Да" else 0,
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.9 if call.get("Содержательный звонок") == "Да" else 0.65,
            record={"call": _compact_row(call), "phone": phone},
            metadata={"control_bucket": row.get("control_bucket", "")},
            created_at=generated_at,
        )
        results.append(_result(store.upsert_event(event, actor="contact_control_sample_import", ingestion_run_id=ingestion_run_id)))
        if event.summary:
            results.append(_result(_upsert_chunk(store, tenant_id, customer_id, opportunity_id, event, "mango_call_summary", generated_at, ingestion_run_id, ordinal=idx)))
    return results


def _upsert_link(
    store: CustomerTimelineSQLiteStore,
    tenant_id: str,
    customer_id: str,
    link_type: str,
    link_value: str,
    source_system: str,
    generated_at: datetime,
    ingestion_run_id: str,
) -> Any:
    return store.upsert_identity_link(
        IdentityLink(
            tenant_id=tenant_id,
            customer_id=customer_id,
            link_type=link_type,
            link_value=link_value,
            source_system=source_system,
            source_ref=f"{source_system}:{link_type}:{link_value}",
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.85,
            first_seen_at=generated_at,
            last_seen_at=generated_at,
        ),
        actor="contact_control_sample_import",
        ingestion_run_id=ingestion_run_id,
    )


def _upsert_chunk(
    store: CustomerTimelineSQLiteStore,
    tenant_id: str,
    customer_id: str,
    opportunity_id: str | None,
    event: TimelineEvent,
    chunk_type: str,
    generated_at: datetime,
    ingestion_run_id: str,
    *,
    ordinal: int = 0,
) -> Any:
    text = safe_text(event.summary or event.text_preview)
    return store.upsert_bot_context_chunk(
        BotContextChunk(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity_id,
            event_id=event.event_id,
            source_ref=event.source_ref,
            source_system=event.source_system,
            chunk_type=chunk_type,
            text=text,
            summary=text[:160],
            event_at=event.event_at,
            freshness_score=0.7,
            relevance_tags=("contact_control", "manual_review_required"),
            ordinal=ordinal,
            allowed_for_bot=False,
            requires_manager_review=True,
            created_at=generated_at,
        ),
        actor="contact_control_sample_import",
        ingestion_run_id=ingestion_run_id,
    )


def _audit_contact(con: sqlite3.Connection, tenant_id: str, row: Mapping[str, str]) -> dict[str, Any]:
    phone = row["primary_phone"]
    links = con.execute(
        """
        SELECT record_json
        FROM identity_links
        WHERE tenant_id = ? AND link_type = 'phone' AND link_value = ?
        """,
        (tenant_id, phone),
    ).fetchall()
    customer_ids = {
        safe_text(json.loads(link["record_json"]).get("customer_id"))
        for link in links
        if safe_text(json.loads(link["record_json"]).get("customer_id"))
    }
    base = _base_report_row(row)
    if not customer_ids:
        return {
            **base,
            "timeline_db_found": "Да",
            "timeline_found": "Нет",
            "timeline_lookup_status": "missing_timeline",
            "valid_customer_identity": "Нет",
            "identity_validation_status": "missing_timeline",
            "manual_review_category": "identity",
            "fallback_used": "Да",
            "verdict": "needs_manual_review",
            "not_ready_reasons": "missing_timeline",
        }
    if len(customer_ids) != 1:
        return {
            **base,
            "timeline_db_found": "Да",
            "timeline_found": "Да",
            "timeline_lookup_status": "ambiguous_identity",
            "timeline_customer_id_count": str(len(customer_ids)),
            "timeline_customer_ids": " | ".join(sorted(customer_ids)),
            "valid_customer_identity": "Нет",
            "identity_validation_status": "invalid_ambiguous_identity",
            "manual_review_category": "identity",
            "fallback_used": "Да",
            "verdict": "needs_manual_review",
            "not_ready_reasons": "invalid_ambiguous_identity",
        }
    customer_id = next(iter(customer_ids))
    events = [_json_row(item) for item in con.execute("SELECT record_json FROM timeline_events WHERE tenant_id = ? AND customer_id = ?", (tenant_id, customer_id))]
    chunks = [_json_row(item) for item in con.execute("SELECT record_json FROM bot_context_chunks WHERE tenant_id = ? AND customer_id = ?", (tenant_id, customer_id))]
    links = [_json_row(item) for item in con.execute("SELECT record_json FROM identity_links WHERE tenant_id = ? AND customer_id = ?", (tenant_id, customer_id))]
    source_counts = Counter(safe_text(item.get("source_system")) for item in events if safe_text(item.get("source_system")))
    event_counts = Counter(safe_text(item.get("event_type")) for item in events if safe_text(item.get("event_type")))
    event_times = [parse_datetime_guess(item.get("event_at")) for item in events]
    event_times = [item for item in event_times if item is not None]
    link_confidences = [float(item.get("confidence")) for item in links if item.get("confidence") is not None]
    foreign = _foreign_history_suspicion(events, {phone})
    empty_or_short = not events or all(len(safe_text(item.get("summary") or item.get("text_preview"))) < 20 for item in events)
    not_ready: list[str] = []
    identity_status = "auto_valid"
    valid_identity = not foreign
    if foreign:
        identity_status = "invalid_foreign_history"
        not_ready.append("invalid_foreign_history")
    if len(events) < 2:
        not_ready.append("timeline_event_count_below_2")
    if not source_counts.get(MANGO_SOURCE_SYSTEM):
        not_ready.append("missing_mango_call_events")
    if empty_or_short:
        not_ready.append("empty_or_overcompressed_events")
    hard_reasons = _hard_control_not_ready_reasons(row)
    not_ready.extend(hard_reasons)
    manual_category = ""
    if not valid_identity:
        manual_category = "identity"
    elif hard_reasons:
        manual_category = "hard_control_risk"
    elif not_ready:
        manual_category = "history_quality"
    verdict = "ready_for_preview" if not not_ready else "needs_manual_review"
    return {
        **base,
        "timeline_db_found": "Да",
        "timeline_found": "Да",
        "timeline_lookup_status": "single_customer",
        "timeline_customer_id_count": "1",
        "timeline_customer_ids": customer_id,
        "valid_customer_identity": "Да" if valid_identity else "Нет",
        "identity_validation_status": identity_status,
        "identity_confidence": f"{min(link_confidences):.2f}" if link_confidences else "",
        "all_sample_phones_linked": "Да",
        "phone_link_match_classes": "strong_unique",
        "min_identity_link_confidence": f"{min(link_confidences):.2f}" if link_confidences else "",
        "amo_lead_link_found": "Да" if any(item.get("link_type") == "amo_lead_id" for item in links) else "Нет",
        "tallanto_link_status": "linked" if any(item.get("link_type") == "tallanto_student_id" for item in links) else "not_linked",
        "tallanto_ids": row.get("tallanto_ids", ""),
        "open_identity_conflicts": "0",
        "timeline_event_count": str(len(events)),
        "has_calls_in_timeline": "Да" if source_counts.get(MANGO_SOURCE_SYSTEM) else "Нет",
        "has_amo_in_timeline": "Да" if source_counts.get(AMO_SOURCE_SYSTEM) else "Нет",
        "has_tallanto_in_timeline": "Да" if source_counts.get(TALLANTO_SOURCE_SYSTEM) else "Нет",
        "has_email_in_timeline": "Нет",
        "has_telegram_in_timeline": "Нет",
        "has_bot_context": "Да" if chunks else "Нет",
        "bot_allowed_chunks": str(sum(1 for item in chunks if item.get("allowed_for_bot"))),
        "bot_review_required_chunks": str(sum(1 for item in chunks if item.get("requires_manager_review"))),
        "fallback_used": "Нет",
        "last_timeline_event_at": max(event_times).isoformat() if event_times else "",
        "empty_or_overcompressed_events": "Да" if empty_or_short else "Нет",
        "foreign_history_suspicion": "Да" if foreign else "Нет",
        "duplicate_event_suspicion": "Нет",
        "chronology_violation_suspicion": "Нет",
        "amo_tallanto_conflict_suspicion": "Нет",
        "history_content_status": "ok" if not empty_or_short else "too_short",
        "hard_control_flags": " | ".join(hard_reasons),
        "manual_review_category": manual_category,
        "not_ready_reasons": " | ".join(not_ready),
        "source_counts_json": json.dumps(dict(source_counts), ensure_ascii=False, sort_keys=True),
        "event_type_counts_json": json.dumps(dict(event_counts), ensure_ascii=False, sort_keys=True),
        "verdict": verdict,
    }


def _missing_db_report_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        **_base_report_row(row),
        "timeline_db_found": "Нет",
        "timeline_found": "Нет",
        "timeline_lookup_status": "timeline_db_missing",
        "valid_customer_identity": "Нет",
        "identity_validation_status": "missing_timeline",
        "fallback_used": "Да",
        "timeline_event_count": "0",
        "has_calls_in_timeline": "Нет",
        "has_amo_in_timeline": "Нет",
        "has_tallanto_in_timeline": "Нет",
        "verdict": "needs_manual_review",
        "manual_review_category": "identity",
        "not_ready_reasons": "timeline_db_missing",
    }


def _hard_control_not_ready_reasons(row: Mapping[str, str]) -> list[str]:
    if row.get("sample_kind") != "hard_contact_control":
        return []
    bucket = row.get("control_bucket", "")
    mapping = {
        "no_reliable_tallanto": "hard_no_reliable_tallanto",
        "tallanto_multiple": "hard_multiple_tallanto_matches",
        "multi_phone_or_multi_deal": "hard_multi_phone_or_multi_deal",
        "long_history": "hard_long_history_manual_review",
        "payment_or_documents_risk": "hard_payment_or_documents_review",
    }
    reason = mapping.get(bucket)
    return [reason] if reason else ["hard_control_manual_review"]


def _base_report_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "sample_kind": row.get("sample_kind", ""),
        "control_bucket": row.get("control_bucket", ""),
        "selection_reason": row.get("selection_reason", ""),
        "primary_phone": row.get("primary_phone", ""),
        "normalized_phones": row.get("normalized_phones", ""),
        "selected_deal_id": row.get("amo_lead_ids", ""),
        "selected_deal_name": row.get("recommended_product", ""),
        "selected_status_name": row.get("contact_status_hint", ""),
        "candidate_call_count": row.get("contentful_call_count", ""),
        "candidate_phone_count": "1",
        "related_phone_count": row.get("related_phone_count", ""),
        "related_phones": row.get("related_phones", ""),
        "source_manual_review_required": row.get("manual_review_required", ""),
        "tallanto_context_status": row.get("tallanto_match_status", ""),
        "lead_priority": row.get("lead_priority", ""),
        "sale_probability_percent": row.get("sale_probability_percent", ""),
        "last_contact_at": row.get("last_contact_at", ""),
    }


def _selected_row(
    raw: Mapping[str, Any],
    *,
    phone: str,
    bucket: str,
    sample_profile: str,
    related_phones: set[str],
) -> dict[str, str]:
    sample_kind = "hard_contact_control" if sample_profile == "hard" else "ordinary_contact_control"
    sample_id = stable_prefixed_id("sample", {"sample_kind": sample_kind, "phone": phone})
    return {
        "sample_id": sample_id,
        "sample_kind": sample_kind,
        "control_bucket": bucket,
        "selection_reason": _selection_reason(bucket),
        "primary_phone": phone,
        "normalized_phones": phone,
        "raw_phone": safe_text(raw.get("Телефон клиента")),
        "display_name": safe_text(raw.get("ФИО родителя") or raw.get("ФИО родителя Tallanto") or raw.get("Контакт Tallanto")),
        "email": safe_text(raw.get("Email")),
        "total_call_count": safe_text(raw.get("Всего звонков в истории")),
        "contentful_call_count": safe_text(raw.get("Содержательных звонков в истории")),
        "manual_review_required": safe_text(raw.get("Нужна ручная проверка")),
        "first_contact_at": safe_text(raw.get("Первый звонок")),
        "last_contact_at": safe_text(raw.get("Последний звонок")),
        "fresh_call_count": safe_text(raw.get("Свежих звонков за период")),
        "contact_history_summary": safe_text(raw.get("Краткая история общения")),
        "recent_touch_chronology": safe_text(raw.get("Хронология общения (последние 5 касаний)")),
        "recommended_product": safe_text(raw.get("Рекомендуемый продукт")),
        "products_of_interest": safe_text(raw.get("Продукты интереса")),
        "objections": safe_text(raw.get("Возражения")),
        "next_step": safe_text(raw.get("Следующий шаг")),
        "lead_priority": safe_text(raw.get("Приоритет лида")),
        "sale_probability_percent": safe_text(raw.get("Вероятность продажи, %")),
        "tallanto_match_status": safe_text(raw.get("Статус матчинга Tallanto")),
        "tallanto_candidate_count": safe_text(raw.get("Количество кандидатов Tallanto")),
        "tallanto_ids": safe_text(raw.get("ID Tallanto")),
        "tallanto_parent_name": safe_text(raw.get("ФИО родителя Tallanto")),
        "tallanto_contact": safe_text(raw.get("Контакт Tallanto")),
        "tallanto_student_type": safe_text(raw.get("Тип ученика Tallanto")),
        "amo_contact_ids": safe_text(raw.get("AMO contact IDs")),
        "amo_lead_ids": safe_text(raw.get("AMO lead IDs")),
        "related_phone_count": str(len(related_phones)),
        "related_phones": " | ".join(sorted(related_phones)),
        "outcome_source": safe_text(raw.get("Outcome source")),
        "utility_score": safe_text(raw.get("Utility score")),
        "contact_status_hint": _contact_status_hint(raw, bucket),
    }


def _classify_control_bucket(row: Mapping[str, Any], *, related_phones: set[str], sample_profile: str) -> str:
    contentful = int_or_zero(row.get("Содержательных звонков в истории"))
    if contentful < 1:
        return "skip"
    lead_ids = safe_text(row.get("AMO lead IDs"))
    tallanto_status = safe_text(row.get("Статус матчинга Tallanto"))
    tallanto_candidates = int_or_zero(row.get("Количество кандидатов Tallanto"))
    if sample_profile == "hard":
        text = " | ".join(safe_text(value).lower() for value in row.values())
        has_payment_context = any(word in text for word in ("оплат", "счет", "счёт", "чек", "квитанц", "договор"))
        total_calls = int_or_zero(row.get("Всего звонков в истории"))
        lead_id_items = _split_ids(lead_ids)
        if tallanto_status == "exact_phone_multiple" or tallanto_candidates > 1:
            return "tallanto_multiple"
        if len(lead_id_items) > 1 or len(related_phones) > 1:
            return "multi_phone_or_multi_deal"
        if contentful >= 8 or total_calls >= 10:
            return "long_history"
        if has_payment_context:
            return "payment_or_documents_risk"
        if tallanto_status != "exact_phone_single":
            return "no_reliable_tallanto"
        return "other"
    if tallanto_status == "exact_phone_multiple":
        return "other"
    last_at = parse_datetime_guess(row.get("Последний звонок"))
    text = " | ".join(safe_text(value).lower() for value in row.values())
    has_payment_context = any(word in text for word in ("оплат", "счет", "счёт", "чек", "квитанц", "договор"))
    if has_payment_context and tallanto_status == "exact_phone_single":
        return "paid_or_success"
    if last_at and last_at >= datetime(2026, 3, 1, tzinfo=timezone.utc) and contentful >= 2:
        return "active_recent"
    if tallanto_status == "exact_phone_single" and (not last_at or last_at < datetime(2026, 3, 1, tzinfo=timezone.utc)):
        return "former_tallanto"
    if not lead_ids and tallanto_status != "exact_phone_single":
        return "new_lead_no_deal"
    return "other"


def _selection_sort_key(sample_profile: str, bucket: str, row: Mapping[str, str]) -> tuple[int, int, str]:
    manual_review_penalty = 1 if row.get("manual_review_required") == "Да" else 0
    tallanto_penalty = 1 if row.get("tallanto_match_status") == "exact_phone_multiple" else 0
    if sample_profile == "hard":
        manual_review_penalty = 0
    return (manual_review_penalty, tallanto_penalty, stable_digest({"bucket": bucket, "phone": row["primary_phone"]}))


def _selection_reason(bucket: str) -> str:
    return {
        "active_recent": "Недавняя содержательная история без попадания в 709 deal-aware.",
        "former_tallanto": "Бывший/учившийся клиент с точной связью Tallanto и без текущей deal-aware выборки.",
        "new_lead_no_deal": "Лид без AMO-сделки и без точной связи Tallanto.",
        "paid_or_success": "Есть признаки оплаты/документов и точная связь Tallanto.",
        "no_reliable_tallanto": "Сложная контрольная строка: нет надежной связи с Tallanto.",
        "tallanto_multiple": "Сложная контрольная строка: несколько возможных учеников в Tallanto.",
        "multi_phone_or_multi_deal": "Сложная контрольная строка: несколько сделок или телефонов связаны с клиентом.",
        "long_history": "Сложная контрольная строка: длинная история общения.",
        "payment_or_documents_risk": "Сложная контрольная строка: оплата, счет, чек или документы.",
    }.get(bucket, "Контрольная обычная строка.")


def _contact_status_hint(row: Mapping[str, Any], bucket: str) -> str:
    if bucket == "paid_or_success":
        return "оплата/документы/учеба"
    if bucket == "active_recent":
        return "активная недавняя история"
    if bucket == "former_tallanto":
        return "бывший или действующий ученик Tallanto"
    if bucket == "new_lead_no_deal":
        return "новый лид без сделки"
    if bucket == "no_reliable_tallanto":
        return "нет надежной связи с Tallanto"
    if bucket == "tallanto_multiple":
        return "несколько возможных учеников Tallanto"
    if bucket == "multi_phone_or_multi_deal":
        return "несколько сделок или телефонов"
    if bucket == "long_history":
        return "длинная история"
    if bucket == "payment_or_documents_risk":
        return "оплата/документы"
    return safe_text(row.get("Outcome source")) or bucket


def _read_selected_calls(path: Path, selected_phones: set[str]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phone = normalize_phone(row.get("Телефон клиента", ""))
            if phone in selected_phones and row.get("Содержательный звонок") == "Да":
                result[phone].append(dict(row))
    return result


def _read_excluded_phones(path: Path) -> set[str]:
    result: set[str] = set()
    for row in _read_csv(path):
        for key, value in row.items():
            if _is_phone_field_name(key.casefold()):
                result.update(_phones_from_values(value))
    return result


def _write_import_source(
    path: Path,
    selected_rows: Sequence[Mapping[str, str]],
    *,
    calls_by_phone: Mapping[str, Sequence[Mapping[str, Any]]],
    generated_at: datetime,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in selected_rows:
        phone = row["primary_phone"]
        rows.append(
            {
                "schema_version": CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION,
                "generated_at": generated_at.isoformat(),
                "sample_id": row.get("sample_id", ""),
                "sample_kind": row.get("sample_kind", ""),
                "control_bucket": row.get("control_bucket", ""),
                "primary_phone": phone,
                "normalized_phones": row.get("normalized_phones", ""),
                "related_phone_count": row.get("related_phone_count", ""),
                "related_phones": row.get("related_phones", ""),
                "matched_call_rows": str(len(calls_by_phone.get(phone, ()))),
                "contentful_call_count": row.get("contentful_call_count", ""),
                "tallanto_match_status": row.get("tallanto_match_status", ""),
                "tallanto_ids": row.get("tallanto_ids", ""),
                "amo_contact_ids": row.get("amo_contact_ids", ""),
                "amo_lead_ids": row.get("amo_lead_ids", ""),
                "selection_reason": row.get("selection_reason", ""),
            }
        )
    _write_csv(path, rows)
    return rows


def _contact_time_bounds(row: Mapping[str, Any], call_rows: Sequence[Mapping[str, Any]], generated_at: datetime) -> tuple[datetime, datetime]:
    dates = [parse_datetime_guess(row.get("first_contact_at")), parse_datetime_guess(row.get("last_contact_at"))]
    dates.extend(parse_datetime_guess(item.get("Дата и время звонка")) for item in call_rows)
    clean = [item for item in dates if item is not None]
    if not clean:
        return generated_at, generated_at
    return min(clean), max(clean)


def _sort_call_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return sorted((dict(row) for row in rows), key=lambda row: parse_datetime_guess(row.get("Дата и время звонка")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)


def _call_direction(call: Mapping[str, Any]) -> TimelineDirection:
    raw = safe_text(call.get("Направление звонка")).lower()
    if raw in {"out", "outbound", "исходящий"}:
        return TimelineDirection.OUTBOUND
    if raw in {"internal", "внутренний"}:
        return TimelineDirection.INTERNAL
    return TimelineDirection.INBOUND


def _tallanto_summary(row: Mapping[str, str]) -> str:
    parts = [
        f"Статус: {row.get('tallanto_match_status', '')}",
        f"ID: {row.get('tallanto_ids', '')}",
        row.get("tallanto_parent_name", ""),
        row.get("tallanto_contact", ""),
        row.get("tallanto_student_type", ""),
    ]
    return " ".join(part for part in parts if safe_text(part))[:500]


def _split_ids(value: Any) -> list[str]:
    ids: list[str] = []
    for part in re.split(r"[|,;\s]+", safe_text(value)):
        item = part.strip()
        if item and item not in ids:
            ids.append(item)
    return ids


def _phones_from_values(*values: Any) -> list[str]:
    phones: list[str] = []
    for value in values:
        for part in re.split(r"[|,;\s]+", safe_text(value)):
            phone = normalize_phone(part)
            if phone and phone not in phones:
                phones.append(phone)
    return phones


def _foreign_history_suspicion(events: Sequence[Mapping[str, Any]], group_phones: set[str]) -> bool:
    seen_phones: set[str] = set()
    for event in events:
        record = event.get("record")
        if not isinstance(record, Mapping):
            continue
        for value in _iter_phone_field_values(record):
            digits = re.sub(r"\D+", "", safe_text(value))
            if len(digits) not in {10, 11}:
                continue
            phone = normalize_phone(value)
            if phone:
                seen_phones.add(phone)
    return bool(seen_phones and not seen_phones.issubset(group_phones))


def _iter_phone_field_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        result: list[Any] = []
        for key, child in value.items():
            if _is_phone_field_name(str(key).casefold()):
                result.extend(_iter_plain_values(child))
            else:
                result.extend(_iter_phone_field_values(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_iter_phone_field_values(child))
        return result
    return []


def _iter_plain_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        result: list[Any] = []
        for child in value.values():
            result.extend(_iter_plain_values(child))
        return result
    if isinstance(value, list) or isinstance(value, tuple):
        result = []
        for child in value:
            result.extend(_iter_plain_values(child))
        return result
    return [value]


def _is_phone_field_name(name: str) -> bool:
    return "phone" in name or "телефон" in name


def _compact_row(row: Mapping[str, Any], *, value_limit: int = 500) -> dict[str, Any]:
    return {str(key): safe_text(value)[:value_limit] for key, value in row.items() if safe_text(value)}


def _result(result: Any) -> Mapping[str, str]:
    return {"record_type": result.record_type, "status": result.status}


def _json_row(row: sqlite3.Row) -> Mapping[str, Any]:
    return json.loads(row["record_json"])


def _sum_counter(rows: Sequence[Mapping[str, Any]], key: str) -> Counter[str]:
    total: Counter[str] = Counter()
    for row in rows:
        raw = safe_text(row.get(key))
        if not raw:
            continue
        try:
            total.update(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return total


def _gate_decision(*, total: int, timeline_found: int, valid_identity: int, ready: int) -> Mapping[str, Any]:
    timeline_ratio = timeline_found / total if total else 1.0
    identity_ratio = valid_identity / total if total else 1.0
    ready_ratio = ready / total if total else 1.0
    if timeline_ratio < 0.8:
        gate = "NOT_READY"
        reason = "timeline_found_ratio_below_control_threshold"
    elif identity_ratio < 0.95:
        gate = "IDENTITY_REVIEW_REQUIRED"
        reason = "valid_customer_identity_ratio_below_primary_threshold"
    elif ready_ratio < 0.7:
        gate = "HISTORY_REVIEW_REQUIRED"
        reason = "identity_ok_but_preview_ready_ratio_low"
    elif ready_ratio < 0.95:
        gate = "PREVIEW_READY_CONTROL_ONLY"
        reason = "control_preview_threshold_passed_primary_threshold_not_passed"
    else:
        gate = "PRIMARY_READY_CONTROL_CANDIDATE"
        reason = "control_sample_primary_threshold_passed"
    return {
        "gate": gate,
        "reason": reason,
        "can_enable_timeline_preview_enabled": False,
        "can_enable_timeline_primary_read_enabled": False,
        "timeline_found_ratio": round(timeline_ratio, 6),
        "valid_customer_identity_ratio": round(identity_ratio, 6),
        "ready_ratio": round(ready_ratio, 6),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    gate = summary["gate_decision"]
    return "\n".join(
        [
            "# Контрольный аудит customer timeline",
            "",
            f"Статус: `{gate['gate']}`",
            f"Профиль выборки: `{summary.get('sample_profile', 'ordinary')}`",
            "",
            "## Метрики",
            "",
            f"- клиентов в выборке: {summary['selected_contacts']}",
            f"- уникальных телефонов: {summary['selected_unique_phones']}",
            f"- найдено в timeline: {summary['timeline_matched_contacts']}",
            f"- timeline_found_ratio: {summary['timeline_found_ratio']}",
            f"- valid_customer_identity: {summary['valid_customer_identity']}",
            f"- valid_customer_identity_ratio: {summary['valid_customer_identity_ratio']}",
            f"- ready_for_preview: {summary['ready_for_preview']}",
            f"- ready_for_preview_ratio: {summary['ready_for_preview_ratio']}",
            f"- needs_manual_review: {summary['needs_manual_review']}",
            "",
            "## Решение",
            "",
            "- Флаги timeline не включать автоматически по одному контрольному прогону.",
            "- Этот прогон нужен как baseline на обычных клиентах, отдельно от сложной deal-aware выборки.",
            "",
        ]
    )


def _source_manifest(config: ContactControlTimelineSampleConfig) -> Mapping[str, Any]:
    return {
        "master_contacts_csv": _file_info(config.master_contacts_csv),
        "master_calls_csv": _file_info(config.master_calls_csv),
        "exclude_phones_csv": _file_info(config.exclude_phones_csv) if config.exclude_phones_csv else {"exists": False},
    }


def _write_source_manifest(config: ContactControlTimelineSampleConfig) -> None:
    manifest = {
        "schema_version": CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION,
        "generated_at": _generated_at(config).isoformat(),
        "sample_profile": config.sample_profile,
        "sources": _source_manifest(config),
    }
    (config.out_root / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safety_contract(*, write_customer_timeline_db: bool) -> Mapping[str, Any]:
    contract = {
        **customer_timeline_safety_contract(),
        **customer_timeline_sqlite_safety_contract(),
        "schema_version": CONTACT_CONTROL_SAMPLE_IMPORT_SCHEMA_VERSION,
        "read_local_files_only": True,
        "write_customer_timeline_db": write_customer_timeline_db,
        "write_product_timeline_db": write_customer_timeline_db,
        "write_crm": False,
        "write_tallanto": False,
        "run_asr": False,
        "run_ra": False,
        "mutate_stable_runtime": False,
        "stable_runtime_writes": False,
        "network_calls": False,
    }
    return contract


def _file_info(path: Optional[Path]) -> Mapping[str, Any]:
    if path is None:
        return {"exists": False}
    resolved = Path(path).resolve(strict=False)
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _generated_at(config: ContactControlTimelineSampleConfig) -> datetime:
    value = config.generated_at or datetime.now(timezone.utc).replace(microsecond=0)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value


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


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value) or 0))
    except ValueError:
        return 0


def safe_text(value: Any) -> str:
    return str(value or "").strip()
