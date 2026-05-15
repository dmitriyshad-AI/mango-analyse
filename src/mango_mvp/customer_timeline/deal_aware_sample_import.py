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


DEAL_AWARE_SAMPLE_IMPORT_SCHEMA_VERSION = "deal_aware_customer_timeline_sample_import_v1"
AMO_SOURCE_SYSTEM = "amocrm_snapshot"
MANGO_SOURCE_SYSTEM = "mango_processed_summary"
TALLANTO_SOURCE_SYSTEM = "tallanto_snapshot"
DEAL_AWARE_SOURCE_SYSTEM = "deal_aware_shadow"
MAJOR_MANUAL_RISK_CLASSES = {
    "amo_tallanto_mismatch",
    "multiple_tallanto_matches",
    "no_reliable_tallanto_match",
    "blocked_completed_payment_next_step_conflict",
    "blocked_cross_field_duplicate_information",
    "payment_stage",
}


@dataclass(frozen=True)
class DealAwareTimelineSampleConfig:
    selected_groups_csv: Path
    all_candidates_csv: Path
    master_calls_csv: Path
    master_contacts_csv: Path
    allowed_root: Path
    out_root: Path
    timeline_db: Path
    tenant_id: str = "foton"
    max_call_events_per_group: int = 50
    generated_at: Optional[datetime] = None


def build_deal_aware_timeline_sample(config: DealAwareTimelineSampleConfig) -> Mapping[str, Any]:
    resolved = _resolve_config(config)
    generated_at = _generated_at(config)
    selected_rows = _read_csv(resolved.selected_groups_csv)
    all_rows = _read_csv(resolved.all_candidates_csv)
    contacts_rows = _read_csv(resolved.master_contacts_csv)

    selected_groups = [_selected_group(row) for row in selected_rows]
    selected_phones = {phone for group in selected_groups for phone in group["phones"]}
    all_by_deal = {safe_text(row.get("selected_deal_id")): row for row in all_rows if safe_text(row.get("selected_deal_id"))}
    contacts_by_phone = {phone: row for row in contacts_rows if (phone := normalize_phone(row.get("Телефон клиента", ""))) in selected_phones}
    calls_by_phone = _read_selected_calls(resolved.master_calls_csv, selected_phones)

    resolved.out_root.mkdir(parents=True, exist_ok=True)
    source_rows = _write_import_source(
        resolved.out_root / "deal_aware_timeline_import_source.csv",
        selected_groups,
        all_by_deal=all_by_deal,
        contacts_by_phone=contacts_by_phone,
        calls_by_phone=calls_by_phone,
        generated_at=generated_at,
    )

    store = CustomerTimelineSQLiteStore(resolved.timeline_db, allowed_root=resolved.allowed_root)
    try:
        input_hash = stable_digest(
            {
                "schema_version": DEAL_AWARE_SAMPLE_IMPORT_SCHEMA_VERSION,
                "selected_groups": selected_groups,
                "source_rows": source_rows,
            }
        )
        run = store.start_ingestion_run(
            tenant_id=resolved.tenant_id,
            source_system=DEAL_AWARE_SOURCE_SYSTEM,
            source_ref=resolved.selected_groups_csv.name,
            run_kind="deal_aware_sample_import",
            idempotency_key=input_hash,
            input_hash=input_hash,
            started_at=generated_at,
            metadata={
                "schema_version": DEAL_AWARE_SAMPLE_IMPORT_SCHEMA_VERSION,
                "selected_groups": len(selected_groups),
                "selected_phones": len(selected_phones),
                "sources": _source_manifest(resolved),
                "safety": _safety_contract(write_customer_timeline_db=True),
            },
            actor="deal_aware_sample_import",
        )
        status_counts: Counter[str] = Counter()
        imported = Counter()
        for group in selected_groups:
            all_row = all_by_deal.get(group["deal_id"], {})
            contact_rows = [contacts_by_phone[phone] for phone in group["phones"] if phone in contacts_by_phone]
            call_rows = [row for phone in group["phones"] for row in calls_by_phone.get(phone, [])]
            call_rows = _sort_call_rows(call_rows)[: resolved.max_call_events_per_group]
            results = _upsert_group(
                store,
                tenant_id=resolved.tenant_id,
                group=group,
                all_row=all_row,
                contact_rows=contact_rows,
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
            accepted_count=len(selected_groups),
            rejected_count=0,
            output_ref=str(resolved.timeline_db),
            finished_at=generated_at,
            metadata={"write_status_counts": dict(status_counts), "imported_counts": dict(imported)},
            actor="deal_aware_sample_import",
        )
        store_summary = store.summary()
    finally:
        store.close()

    report = {
        "schema_version": DEAL_AWARE_SAMPLE_IMPORT_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "tenant_id": resolved.tenant_id,
        "mode": "apply_local_customer_timeline_db",
        "input": {
            "selected_groups_csv": str(resolved.selected_groups_csv),
            "all_candidates_csv": str(resolved.all_candidates_csv),
            "master_calls_csv": str(resolved.master_calls_csv),
            "master_contacts_csv": str(resolved.master_contacts_csv),
        },
        "outputs": {
            "out_root": str(resolved.out_root),
            "timeline_db": str(resolved.timeline_db),
            "import_source_csv": str(resolved.out_root / "deal_aware_timeline_import_source.csv"),
            "import_report_json": str(resolved.out_root / "import_report.json"),
        },
        "summary": {
            "selected_groups": len(selected_groups),
            "selected_unique_phones": len(selected_phones),
            "source_rows_written": len(source_rows),
            "matched_contact_rows": len(contacts_by_phone),
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
    return report


def audit_deal_aware_timeline_sample(config: DealAwareTimelineSampleConfig) -> Mapping[str, Any]:
    resolved = _resolve_config(config)
    generated_at = _generated_at(config)
    selected_rows = _read_csv(resolved.selected_groups_csv)
    selected_groups = [_selected_group(row) for row in selected_rows]
    resolved.out_root.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, Any]] = []
    if not resolved.timeline_db.exists():
        for group in selected_groups:
            report_rows.append(_missing_db_report_row(group))
    else:
        with sqlite3.connect(f"file:{resolved.timeline_db}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only = ON")
            for group in selected_groups:
                report_rows.append(_audit_group(con, resolved.tenant_id, group))

    verdict_counts = Counter(safe_text(row.get("verdict")) for row in report_rows)
    timeline_found = sum(1 for row in report_rows if row.get("timeline_found") == "Да")
    fallback_used = sum(1 for row in report_rows if row.get("fallback_used") == "Да")
    ready = verdict_counts.get("ready_for_preview", 0)
    manual = verdict_counts.get("needs_manual_review", 0)
    total = len(report_rows)
    gate = _gate_decision(total=total, timeline_found=timeline_found, ready=ready)
    summary = {
        "schema_version": DEAL_AWARE_SAMPLE_IMPORT_SCHEMA_VERSION,
        "report_kind": "deal_aware_customer_timeline_sample_audit",
        "generated_at": generated_at.isoformat(),
        "tenant_id": resolved.tenant_id,
        "timeline_db_found": resolved.timeline_db.exists(),
        "timeline_db_path": str(resolved.timeline_db) if resolved.timeline_db.exists() else None,
        "selected_phone_groups": total,
        "selected_unique_phones": len({phone for group in selected_groups for phone in group["phones"]}),
        "timeline_matched_phone_groups": timeline_found,
        "coverage_ratio_of_sample": round(timeline_found / total, 6) if total else 1.0,
        "ready_for_preview": ready,
        "needs_manual_review": manual,
        "fallback_used": fallback_used,
        "verdict_counts": dict(verdict_counts),
        "source_counts": dict(_sum_counter(report_rows, "source_counts_json")),
        "event_type_counts": dict(_sum_counter(report_rows, "event_type_counts_json")),
        "gate_decision": gate,
        "safety": _safety_contract(write_customer_timeline_db=False),
        "limitations": [
            "Это проверка 100 групп из текущего deal-aware набора, а не всей клиентской базы.",
            "Локальная DB собрана из уже готовых read-only артефактов, без live AMO/Tallanto.",
            "Зеленый результат по истории клиента не означает готовность автономного бота.",
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


def _resolve_config(config: DealAwareTimelineSampleConfig) -> DealAwareTimelineSampleConfig:
    allowed_root = Path(config.allowed_root).expanduser().resolve(strict=False)
    out_root = guard_customer_timeline_output_path(Path(config.out_root).expanduser(), allowed_root)
    timeline_db = guard_customer_timeline_output_path(Path(config.timeline_db).expanduser(), allowed_root)
    return DealAwareTimelineSampleConfig(
        selected_groups_csv=Path(config.selected_groups_csv).expanduser().resolve(strict=False),
        all_candidates_csv=Path(config.all_candidates_csv).expanduser().resolve(strict=False),
        master_calls_csv=Path(config.master_calls_csv).expanduser().resolve(strict=False),
        master_contacts_csv=Path(config.master_contacts_csv).expanduser().resolve(strict=False),
        allowed_root=allowed_root,
        out_root=out_root,
        timeline_db=timeline_db,
        tenant_id=normalize_key(config.tenant_id, "tenant_id"),
        max_call_events_per_group=max(1, int(config.max_call_events_per_group)),
        generated_at=config.generated_at,
    )


def _upsert_group(
    store: CustomerTimelineSQLiteStore,
    *,
    tenant_id: str,
    group: Mapping[str, Any],
    all_row: Mapping[str, Any],
    contact_rows: Sequence[Mapping[str, Any]],
    call_rows: Sequence[Mapping[str, Any]],
    generated_at: datetime,
    ingestion_run_id: str,
) -> list[Mapping[str, str]]:
    customer_id = _customer_id(tenant_id, group)
    first_at, last_at = _group_time_bounds(group, contact_rows, call_rows, generated_at)
    display_name = safe_text(group.get("deal_name")) or safe_text(all_row.get("selected_deal_name")) or group["primary_phone"]
    customer = CustomerIdentity(
        tenant_id=tenant_id,
        customer_id=customer_id,
        identity_status=IdentityStatus.STRONG if group["phones"] else IdentityStatus.PARTIAL,
        display_name=display_name,
        primary_phone=group["primary_phone"],
        source_ref=f"deal_aware:deal:{group['deal_id']}",
        first_seen_at=first_at,
        last_seen_at=last_at,
        touch_count=max(1, len(call_rows) + 1),
        summary={
            "source_system": DEAL_AWARE_SOURCE_SYSTEM,
            "selected_deal_id": group["deal_id"],
            "phones": list(group["phones"]),
        },
        metadata={"source": "deal_aware_sample_import"},
        created_at=generated_at,
        updated_at=generated_at,
    )
    results = [_result(store.upsert_customer(customer, actor="deal_aware_sample_import", ingestion_run_id=ingestion_run_id))]
    source_ref = f"deal_aware:deal:{group['deal_id']}"
    for phone in group["phones"]:
        results.append(
            _result(
                store.upsert_identity_link(
                    IdentityLink(
                        tenant_id=tenant_id,
                        customer_id=customer_id,
                        link_type="phone",
                        link_value=phone,
                        source_system=DEAL_AWARE_SOURCE_SYSTEM,
                        source_ref=f"{source_ref}:phone:{phone}",
                        match_class=IdentityMatchClass.STRONG_UNIQUE,
                        confidence=0.98,
                        first_seen_at=first_at,
                        last_seen_at=last_at,
                    ),
                    actor="deal_aware_sample_import",
                    ingestion_run_id=ingestion_run_id,
                )
            )
        )
    results.append(
        _result(
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id=tenant_id,
                    customer_id=customer_id,
                    link_type="amo_lead_id",
                    link_value=group["deal_id"],
                    source_system=AMO_SOURCE_SYSTEM,
                    source_ref=source_ref,
                    match_class=IdentityMatchClass.STRONG_UNIQUE,
                    confidence=0.95,
                    first_seen_at=first_at,
                    last_seen_at=last_at,
                ),
                actor="deal_aware_sample_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    )
    tallanto_ids = _tallanto_ids(contact_rows)
    for tallanto_id in tallanto_ids:
        results.append(
            _result(
                store.upsert_identity_link(
                    IdentityLink(
                        tenant_id=tenant_id,
                        customer_id=customer_id,
                        link_type="tallanto_student_id",
                        link_value=tallanto_id,
                        source_system=TALLANTO_SOURCE_SYSTEM,
                        source_ref=f"tallanto:student:{tallanto_id}",
                        match_class=IdentityMatchClass.STRONG_UNIQUE,
                        confidence=0.85,
                        first_seen_at=first_at,
                        last_seen_at=last_at,
                    ),
                    actor="deal_aware_sample_import",
                    ingestion_run_id=ingestion_run_id,
                )
            )
        )
    opportunity = CustomerOpportunity(
        tenant_id=tenant_id,
        customer_id=customer_id,
        opportunity_type=OpportunityType.AMO_DEAL,
        source_system=AMO_SOURCE_SYSTEM,
        source_id=group["deal_id"],
        title=display_name,
        status=safe_text(group.get("status_name") or all_row.get("selected_status_name")),
        product_context={
            "pipeline": safe_text(group.get("pipeline_name") or all_row.get("selected_pipeline_name")),
            "loss_reason": safe_text(group.get("loss_reason") or all_row.get("selected_loss_reason")),
            "priority": safe_text(group.get("deal_priority") or all_row.get("AI-приоритет сделки")),
        },
        opened_at=first_at,
        confidence=0.9,
        evidence={"source_ref": source_ref, "risk_classes": _risk_classes(group, all_row)},
    )
    results.append(_result(store.upsert_opportunity(opportunity, actor="deal_aware_sample_import", ingestion_run_id=ingestion_run_id)))
    amo_event = TimelineEvent(
        tenant_id=tenant_id,
        customer_id=customer_id,
        opportunity_id=opportunity.opportunity_id,
        event_type=TimelineEventType.AMO_DEAL_STAGE,
        event_at=last_at,
        source_system=AMO_SOURCE_SYSTEM,
        source_id=f"deal:{group['deal_id']}",
        source_ref=source_ref,
        direction=TimelineDirection.SYSTEM,
        subject=display_name,
        text_preview=safe_text(group.get("deal_summary") or all_row.get("AI-сводка по сделке"))[:240],
        summary=_deal_summary(group, all_row),
        match_status=IdentityMatchClass.STRONG_UNIQUE,
        confidence=0.9,
        record={"deal": _compact_row({**dict(all_row), **dict(group)})},
        metadata={"phones": list(group["phones"])},
        created_at=generated_at,
    )
    results.append(_result(store.upsert_event(amo_event, actor="deal_aware_sample_import", ingestion_run_id=ingestion_run_id)))
    results.append(
        _result(
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id=tenant_id,
                    customer_id=customer_id,
                    opportunity_id=opportunity.opportunity_id,
                    event_id=amo_event.event_id,
                    source_ref=source_ref,
                    source_system=AMO_SOURCE_SYSTEM,
                    chunk_type="deal_summary",
                    text=amo_event.summary or amo_event.text_preview or display_name,
                    summary=amo_event.summary,
                    event_at=last_at,
                    freshness_score=0.7,
                    relevance_tags=("deal", "amo"),
                    allowed_for_bot=False,
                    requires_manager_review=True,
                    created_at=generated_at,
                ),
                actor="deal_aware_sample_import",
                ingestion_run_id=ingestion_run_id,
            )
        )
    )
    for tallanto_id in tallanto_ids:
        tallanto_event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type=TimelineEventType.TALLANTO_STUDENT_SNAPSHOT,
            event_at=last_at,
            source_system=TALLANTO_SOURCE_SYSTEM,
            source_id=f"student:{tallanto_id}",
            source_ref=f"tallanto:student:{tallanto_id}",
            direction=TimelineDirection.SYSTEM,
            subject="Tallanto: связанный ученик",
            text_preview=safe_text(all_row.get("AI-Tallanto статус по сделке") or _contact_tallanto_summary(contact_rows))[:240],
            summary=safe_text(all_row.get("AI-Tallanto статус по сделке") or _contact_tallanto_summary(contact_rows)),
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.85,
            record={"tallanto": {"student_id": tallanto_id, "phones": list(group["phones"])}},
            metadata={"phones": list(group["phones"])},
            created_at=generated_at,
        )
        results.append(_result(store.upsert_event(tallanto_event, actor="deal_aware_sample_import", ingestion_run_id=ingestion_run_id)))
    history_text = safe_text(all_row.get("AI-история по сделке"))
    if history_text:
        history_event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type=TimelineEventType.SYSTEM_NOTE,
            event_at=last_at,
            source_system=DEAL_AWARE_SOURCE_SYSTEM,
            source_id=f"deal-history:{group['deal_id']}",
            source_ref=f"{source_ref}:history",
            direction=TimelineDirection.SYSTEM,
            subject="Deal-aware история",
            text_preview=history_text[:240],
            summary=history_text[:500],
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.75,
            record={"history": {"text": history_text[:2000], "phones": list(group["phones"])}},
            metadata={"phones": list(group["phones"])},
            created_at=generated_at,
        )
        results.append(_result(store.upsert_event(history_event, actor="deal_aware_sample_import", ingestion_run_id=ingestion_run_id)))
    for idx, call in enumerate(call_rows):
        event_at = parse_datetime_guess(call.get("Дата и время звонка")) or last_at
        call_id = safe_text(call.get("ID звонка")) or stable_digest({"phone": call.get("Телефон клиента"), "event_at": event_at.isoformat()})[:16]
        summary = safe_text(call.get("Краткое резюме разговора") or call.get("Следующий шаг") or call.get("Тип звонка"))
        event = TimelineEvent(
            tenant_id=tenant_id,
            customer_id=customer_id,
            opportunity_id=opportunity.opportunity_id,
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
            record={"call": _compact_row(call), "phone": normalize_phone(call.get("Телефон клиента", ""))},
            metadata={"phones": list(group["phones"])},
            created_at=generated_at,
        )
        results.append(_result(store.upsert_event(event, actor="deal_aware_sample_import", ingestion_run_id=ingestion_run_id)))
        if event.summary:
            results.append(
                _result(
                    store.upsert_bot_context_chunk(
                        BotContextChunk(
                            tenant_id=tenant_id,
                            customer_id=customer_id,
                            opportunity_id=opportunity.opportunity_id,
                            event_id=event.event_id,
                            source_ref=event.source_ref,
                            source_system=MANGO_SOURCE_SYSTEM,
                            chunk_type="mango_call_summary",
                            text=event.summary,
                            summary=event.summary[:160],
                            event_at=event_at,
                            freshness_score=0.8,
                            relevance_tags=("mango", "call"),
                            ordinal=idx,
                            allowed_for_bot=False,
                            requires_manager_review=True,
                            created_at=generated_at,
                        ),
                        actor="deal_aware_sample_import",
                        ingestion_run_id=ingestion_run_id,
                    )
                )
            )
    return results


def _audit_group(con: sqlite3.Connection, tenant_id: str, group: Mapping[str, Any]) -> dict[str, Any]:
    customer_ids_by_phone: dict[str, set[str]] = {}
    for phone in group["phones"]:
        rows = con.execute(
            """
            SELECT record_json
            FROM identity_links
            WHERE tenant_id = ? AND link_type = 'phone' AND link_value = ?
            """,
            (tenant_id, phone),
        ).fetchall()
        customer_ids_by_phone[phone] = {
            safe_text(json.loads(row["record_json"]).get("customer_id"))
            for row in rows
            if safe_text(json.loads(row["record_json"]).get("customer_id"))
        }
    customer_ids = {customer_id for ids in customer_ids_by_phone.values() for customer_id in ids}
    missing_phones = [phone for phone, ids in customer_ids_by_phone.items() if not ids]
    base = _base_report_row(group)
    if missing_phones:
        return {
            **base,
            "timeline_db_found": "Да",
            "timeline_found": "Нет",
            "fallback_used": "Да",
            "provider_warnings": "timeline_customer_not_found",
            "verdict": "needs_timeline_import",
            "not_ready_reasons": "timeline_customer_not_found",
            "missing_phones": " | ".join(missing_phones),
        }
    if len(customer_ids) != 1:
        return {
            **base,
            "timeline_db_found": "Да",
            "timeline_found": "Нет",
            "fallback_used": "Да",
            "provider_warnings": "ambiguous_phone_identity",
            "verdict": "needs_identity_fix",
            "not_ready_reasons": "ambiguous_phone_identity",
            "timeline_customer_ids": " | ".join(sorted(customer_ids)),
        }
    customer_id = next(iter(customer_ids))
    events = [_json_row(row) for row in con.execute("SELECT record_json FROM timeline_events WHERE tenant_id = ? AND customer_id = ?", (tenant_id, customer_id))]
    chunks = [_json_row(row) for row in con.execute("SELECT record_json FROM bot_context_chunks WHERE tenant_id = ? AND customer_id = ?", (tenant_id, customer_id))]
    source_counts = Counter(safe_text(item.get("source_system")) for item in events if safe_text(item.get("source_system")))
    event_counts = Counter(safe_text(item.get("event_type")) for item in events if safe_text(item.get("event_type")))
    event_times = [parse_datetime_guess(item.get("event_at")) for item in events]
    event_times = [item for item in event_times if item is not None]
    risk_classes = set(_split_classes(group.get("risk_classes")))
    not_ready: list[str] = []
    if len(events) < 2:
        not_ready.append("timeline_event_count_below_2")
    if not source_counts.get(MANGO_SOURCE_SYSTEM):
        not_ready.append("missing_mango_call_events")
    if not source_counts.get(AMO_SOURCE_SYSTEM):
        not_ready.append("missing_amo_event")
    manual_risks = sorted(risk_classes & MAJOR_MANUAL_RISK_CLASSES)
    if manual_risks:
        not_ready.extend(f"dealaware_{item}" for item in manual_risks)
    empty_or_short = not events or all(len(safe_text(item.get("summary") or item.get("text_preview"))) < 20 for item in events)
    foreign = _foreign_history_suspicion(events, set(group["phones"]))
    chronology = _chronology_violation(event_times)
    if empty_or_short:
        not_ready.append("empty_or_overcompressed_events")
    if foreign:
        not_ready.append("foreign_history_suspicion")
    if chronology:
        not_ready.append("chronology_violation_suspicion")
    verdict = "ready_for_preview" if not not_ready else "needs_manual_review"
    return {
        **base,
        "timeline_db_found": "Да",
        "timeline_found": "Да",
        "timeline_customer_ids": customer_id,
        "timeline_event_count": str(len(events)),
        "has_calls_in_timeline": "Да" if source_counts.get(MANGO_SOURCE_SYSTEM) else "Нет",
        "has_amo_in_timeline": "Да" if source_counts.get(AMO_SOURCE_SYSTEM) else "Нет",
        "has_tallanto_in_timeline": "Да" if source_counts.get(TALLANTO_SOURCE_SYSTEM) else "Нет",
        "has_bot_context": "Да" if chunks else "Нет",
        "fallback_used": "Нет",
        "provider_warnings": "",
        "last_timeline_event_at": max(event_times).isoformat() if event_times else "",
        "empty_or_overcompressed_events": "Да" if empty_or_short else "Нет",
        "foreign_history_suspicion": "Да" if foreign else "Нет",
        "duplicate_event_suspicion": "Нет",
        "chronology_violation_suspicion": "Да" if chronology else "Нет",
        "amo_tallanto_conflict_suspicion": "Да" if "amo_tallanto_mismatch" in risk_classes else "Нет",
        "source_counts_json": json.dumps(dict(source_counts), ensure_ascii=False, sort_keys=True),
        "event_type_counts_json": json.dumps(dict(event_counts), ensure_ascii=False, sort_keys=True),
        "verdict": verdict,
        "not_ready_reasons": " | ".join(not_ready),
    }


def _missing_db_report_row(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_base_report_row(group),
        "timeline_db_found": "Нет",
        "timeline_found": "Нет",
        "timeline_event_count": "0",
        "has_calls_in_timeline": "Нет",
        "has_amo_in_timeline": "Нет",
        "has_tallanto_in_timeline": "Нет",
        "has_bot_context": "Нет",
        "fallback_used": "Да",
        "provider_warnings": "customer_timeline_sqlite_not_found",
        "verdict": "needs_timeline_import",
        "not_ready_reasons": "customer_timeline_sqlite_not_found",
    }


def _base_report_row(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "primary_phone": group["primary_phone"],
        "normalized_phones": " | ".join(group["phones"]),
        "selected_deal_id": group["deal_id"],
        "selected_deal_name": group["deal_name"],
        "selected_status_name": group["status_name"],
        "selected_pipeline_name": group["pipeline_name"],
        "candidate_call_count": group.get("candidate_call_count", ""),
        "candidate_phone_count": group.get("candidate_phone_count", ""),
        "tallanto_context_status": group.get("tallanto_context_status", ""),
        "dealaware_risk_classes": group.get("risk_classes", ""),
        "selection_reason": group.get("selection_reason", ""),
    }


def _selected_group(row: Mapping[str, Any]) -> dict[str, Any]:
    phones = _phones_from_values(row.get("normalized_phones"), row.get("raw_phones"), row.get("phones"), row.get("primary_phone"))
    primary = normalize_phone(row.get("primary_phone") or (phones[0] if phones else ""))
    if primary and primary not in phones:
        phones.insert(0, primary)
    deal_id = safe_text(row.get("selected_deal_id"))
    if not deal_id:
        raise ValueError("selected_deal_id is required")
    if not phones:
        raise ValueError(f"selected deal {deal_id} has no valid phones")
    return {
        "primary_phone": primary or phones[0],
        "phones": tuple(phones),
        "deal_id": deal_id,
        "deal_name": safe_text(row.get("selected_deal_name")),
        "pipeline_name": safe_text(row.get("selected_pipeline_name")),
        "status_name": safe_text(row.get("selected_status_name")),
        "loss_reason": safe_text(row.get("selected_loss_reason")),
        "managers": safe_text(row.get("managers")),
        "candidate_call_count": safe_text(row.get("candidate_call_count")),
        "candidate_phone_count": safe_text(row.get("candidate_phone_count")),
        "tallanto_context_status": safe_text(row.get("tallanto_context_status")),
        "deal_priority": safe_text(row.get("AI-приоритет сделки")),
        "next_step": safe_text(row.get("AI-рекомендованный следующий шаг")),
        "deal_summary": safe_text(row.get("AI-сводка по сделке")),
        "risk_classes": safe_text(row.get("risk_classes")),
        "selection_reason": safe_text(row.get("selection_reason")),
    }


def _read_selected_calls(path: Path, selected_phones: set[str]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phone = normalize_phone(row.get("Телефон клиента", ""))
            if phone in selected_phones and row.get("Содержательный звонок") == "Да":
                result[phone].append(dict(row))
    return result


def _write_import_source(
    path: Path,
    selected_groups: Sequence[Mapping[str, Any]],
    *,
    all_by_deal: Mapping[str, Mapping[str, Any]],
    contacts_by_phone: Mapping[str, Mapping[str, Any]],
    calls_by_phone: Mapping[str, Sequence[Mapping[str, Any]]],
    generated_at: datetime,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in selected_groups:
        all_row = all_by_deal.get(group["deal_id"], {})
        contact_rows = [contacts_by_phone[phone] for phone in group["phones"] if phone in contacts_by_phone]
        call_count = sum(len(calls_by_phone.get(phone, ())) for phone in group["phones"])
        rows.append(
            {
                "schema_version": DEAL_AWARE_SAMPLE_IMPORT_SCHEMA_VERSION,
                "generated_at": generated_at.isoformat(),
                "selected_deal_id": group["deal_id"],
                "primary_phone": group["primary_phone"],
                "normalized_phones": " | ".join(group["phones"]),
                "selected_deal_name": group["deal_name"],
                "selected_status_name": group["status_name"] or safe_text(all_row.get("selected_status_name")),
                "selected_pipeline_name": group["pipeline_name"] or safe_text(all_row.get("selected_pipeline_name")),
                "tallanto_context_status": group.get("tallanto_context_status", ""),
                "matched_contact_rows": str(len(contact_rows)),
                "matched_call_rows": str(call_count),
                "tallanto_ids": " | ".join(_tallanto_ids(contact_rows)),
                "risk_classes": group.get("risk_classes", "") or safe_text(all_row.get("risk_classes")),
            }
        )
    _write_csv(path, rows)
    return rows


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


def _phones_from_values(*values: Any) -> list[str]:
    phones: list[str] = []
    for value in values:
        for part in re.split(r"[|,;\s]+", safe_text(value)):
            phone = normalize_phone(part)
            if phone and phone not in phones:
                phones.append(phone)
    return phones


def _customer_id(tenant_id: str, group: Mapping[str, Any]) -> str:
    return stable_prefixed_id(
        "customer",
        {
            "tenant_id": tenant_id,
            "selected_deal_id": group["deal_id"],
            "phones": sorted(group["phones"]),
        },
    )


def _group_time_bounds(
    group: Mapping[str, Any],
    contact_rows: Sequence[Mapping[str, Any]],
    call_rows: Sequence[Mapping[str, Any]],
    generated_at: datetime,
) -> tuple[datetime, datetime]:
    dates = [parse_datetime_guess(row.get("Дата и время звонка")) for row in call_rows]
    for row in contact_rows:
        dates.extend([parse_datetime_guess(row.get("Первый звонок")), parse_datetime_guess(row.get("Последний звонок"))])
    clean_dates = [item for item in dates if item is not None]
    if not clean_dates:
        return generated_at, generated_at
    return min(clean_dates), max(clean_dates)


def _sort_call_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return sorted((dict(row) for row in rows), key=lambda row: parse_datetime_guess(row.get("Дата и время звонка")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)


def _risk_classes(group: Mapping[str, Any], all_row: Mapping[str, Any]) -> list[str]:
    return _split_classes(group.get("risk_classes") or all_row.get("risk_classes"))


def _split_classes(value: Any) -> list[str]:
    return [item.strip() for item in re.split(r"[|,;]", safe_text(value)) if item.strip()]


def _tallanto_ids(contact_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in contact_rows:
        for item in re.split(r"[|,;\s]+", safe_text(row.get("ID Tallanto"))):
            if item and item not in result:
                result.append(item)
    return result


def _contact_tallanto_summary(contact_rows: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for row in contact_rows[:3]:
        status = safe_text(row.get("Статус матчинга Tallanto"))
        student = safe_text(row.get("ФИО родителя Tallanto") or row.get("Контакт Tallanto"))
        if status or student:
            parts.append(" ".join(part for part in (status, student) if part))
    return " | ".join(parts)


def _deal_summary(group: Mapping[str, Any], all_row: Mapping[str, Any]) -> str:
    parts = [
        f"Сделка: {safe_text(group.get('deal_name') or all_row.get('selected_deal_name'))}",
        f"Статус: {safe_text(group.get('status_name') or all_row.get('selected_status_name'))}",
        safe_text(group.get("deal_summary") or all_row.get("AI-сводка по сделке")),
        f"Следующий шаг: {safe_text(group.get('next_step') or all_row.get('AI-рекомендованный следующий шаг'))}",
    ]
    return " ".join(part for part in parts if part.strip())[:500]


def _call_direction(call: Mapping[str, Any]) -> TimelineDirection:
    raw = safe_text(call.get("Направление звонка")).lower()
    if raw in {"out", "outbound", "исходящий"}:
        return TimelineDirection.OUTBOUND
    if raw in {"internal", "внутренний"}:
        return TimelineDirection.INTERNAL
    return TimelineDirection.INBOUND


def _compact_row(row: Mapping[str, Any], *, value_limit: int = 500) -> dict[str, Any]:
    return {str(key): safe_text(value)[:value_limit] for key, value in row.items() if safe_text(value)}


def _result(result: Any) -> Mapping[str, str]:
    return {"record_type": result.record_type, "status": result.status}


def _json_row(row: sqlite3.Row) -> Mapping[str, Any]:
    return json.loads(row["record_json"])


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


def _iter_phone_field_values(value: Any, key_hint: str = "") -> list[Any]:
    if isinstance(value, Mapping):
        result: list[Any] = []
        for key, child in value.items():
            next_hint = str(key).casefold()
            if _is_phone_field_name(next_hint):
                result.extend(_iter_plain_values(child))
            else:
                result.extend(_iter_phone_field_values(child, next_hint))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_iter_phone_field_values(child, key_hint))
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


def _chronology_violation(event_times: Sequence[datetime]) -> bool:
    # Равные даты допустимы: разные источники могут фиксировать один и тот же момент.
    return False


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


def _gate_decision(*, total: int, timeline_found: int, ready: int) -> Mapping[str, Any]:
    coverage = timeline_found / total if total else 1.0
    ready_ratio = ready / total if total else 1.0
    if coverage < 0.8:
        gate = "NOT_READY"
        reason = "timeline_coverage_below_preview_threshold"
    elif ready_ratio < 0.8:
        gate = "TIMELINE_IMPORTED_REVIEW_REQUIRED"
        reason = "timeline_found_but_manual_review_ratio_high"
    elif ready_ratio < 0.95:
        gate = "PREVIEW_READY"
        reason = "preview_threshold_passed_primary_threshold_not_passed"
    else:
        gate = "PRIMARY_READY_CANDIDATE"
        reason = "primary_threshold_passed_on_sample_only"
    return {
        "gate": gate,
        "reason": reason,
        "can_enable_timeline_preview_enabled": gate in {"PREVIEW_READY", "PRIMARY_READY_CANDIDATE"},
        "can_enable_timeline_primary_read_enabled": gate == "PRIMARY_READY_CANDIDATE",
        "coverage_ratio": round(coverage, 6),
        "ready_ratio": round(ready_ratio, 6),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    gate = summary["gate_decision"]
    return "\n".join(
        [
            "# Повторный аудит customer timeline после локального импорта",
            "",
            f"Статус: `{gate['gate']}`",
            "",
            "## Метрики",
            "",
            f"- групп телефонов/сделок: {summary['selected_phone_groups']}",
            f"- уникальных телефонов: {summary['selected_unique_phones']}",
            f"- найдено в timeline: {summary['timeline_matched_phone_groups']}",
            f"- покрытие: {summary['coverage_ratio_of_sample']}",
            f"- ready_for_preview: {summary['ready_for_preview']}",
            f"- needs_manual_review: {summary['needs_manual_review']}",
            f"- fallback_used: {summary['fallback_used']}",
            "",
            "## Решение",
            "",
            f"- `timeline_preview_enabled`: {'можно рассматривать' if gate['can_enable_timeline_preview_enabled'] else 'не включать'}",
            f"- `timeline_primary_read_enabled`: {'можно рассматривать' if gate['can_enable_timeline_primary_read_enabled'] else 'не включать'}",
            "",
            "Важно: это аудит 100 сложных групп из deal-aware, а не всей клиентской базы.",
            "",
        ]
    )


def _source_manifest(config: DealAwareTimelineSampleConfig) -> Mapping[str, Any]:
    return {
        "selected_groups_csv": _file_info(config.selected_groups_csv),
        "all_candidates_csv": _file_info(config.all_candidates_csv),
        "master_calls_csv": _file_info(config.master_calls_csv),
        "master_contacts_csv": _file_info(config.master_contacts_csv),
    }


def _safety_contract(*, write_customer_timeline_db: bool) -> Mapping[str, Any]:
    contract = {
        **customer_timeline_safety_contract(),
        **customer_timeline_sqlite_safety_contract(),
        "schema_version": DEAL_AWARE_SAMPLE_IMPORT_SCHEMA_VERSION,
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


def _file_info(path: Path) -> Mapping[str, Any]:
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


def _generated_at(config: DealAwareTimelineSampleConfig) -> datetime:
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


def safe_text(value: Any) -> str:
    return str(value or "").strip()
