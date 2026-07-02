from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityMatchClass,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.ids import (
    normalize_email,
    normalize_identity_value,
    normalize_key,
    stable_digest,
)
from mango_mvp.customer_timeline.ingestion import (
    compact_text,
    customer_identity_from_json,
)
from mango_mvp.customer_timeline.canonical_readonly_import import infer_offline_brand
from mango_mvp.customer_timeline.mail_stage2_ingest import (
    file_sha256,
    write_json,
    _parse_event_at,
    _sha16,
)
from mango_mvp.customer_timeline.next_step_resolver import resolve_customer_next_step
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    existing_timeline_email_content_signatures,
    parse_datetime,
    scrub_timeline_persisted_json,
    timeline_email_content_signature,
)


A2V3_MAIL_INGEST_SCHEMA_VERSION = "a2v3_mail_timeline_ingest_v1"
A2V3_MAIL_SOURCE_SYSTEM = "mail_archive_stage2"
A2V3_DEDUPE_SOURCE_SYSTEMS = ("mail_archive", "mail_archive_stage2")
DEFAULT_A2V3_INPUT = Path(".codex_local/email_pipeline/A2v3_100_review_full_storage.jsonl")
DEFAULT_TALLANTO_IDENTITY_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/"
    "_external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite"
)
DEFAULT_PROD_TIMELINE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/"
    "product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"
)
QUALIFYING_AMO_STAGES = {
    "Перспектива",
    "Оплата получена",
    "Успешно",
    "Ожидание оплаты",
    "В работе",
    "Принимают решение",
    "Переговоры",
    "Заключение договора",
    "Запись в группу",
    "Аудит",
}
BLOCKED_MEMORY_STATUSES = {
    "broadcast_not_usable",
    "quote_only",
    "thin_ack",
    "attachment_only",
    "financial_unverified",
    "needs_thread_context",
}


@dataclass(frozen=True)
class A2V3MailIngestConfig:
    input_jsonl: Path
    prod_timeline_db: Path
    timeline_db_path: Path
    allowed_root: Path
    out_dir: Path
    tallanto_identity_db: Optional[Path] = DEFAULT_TALLANTO_IDENTITY_DB
    tenant_id: str = "foton"
    source_ref: str = "a2v3_100_review_20260701"
    enrich_existing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_jsonl", Path(self.input_jsonl).expanduser())
        object.__setattr__(self, "prod_timeline_db", Path(self.prod_timeline_db).expanduser())
        object.__setattr__(self, "timeline_db_path", Path(self.timeline_db_path).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        if self.tallanto_identity_db is not None:
            object.__setattr__(self, "tallanto_identity_db", Path(self.tallanto_identity_db).expanduser())
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))


@dataclass(frozen=True)
class CustomerResolution:
    outcome: str
    reason: str
    customer_id: Optional[str] = None
    method: Optional[str] = None
    tallanto_id: Optional[str] = None
    link_email_value: Optional[str] = None
    blocked: bool = False
    ambiguous: bool = False


@dataclass(frozen=True)
class PlannedA2V3MailEvent:
    row: Mapping[str, Any]
    event: TimelineEvent
    customer: Optional[CustomerIdentity]
    identity_links: tuple[IdentityLink, ...]
    opportunity: Optional[CustomerOpportunity]
    chunk: Optional[BotContextChunk]
    resolution: CustomerResolution
    prod_duplicate_keys: tuple[str, ...] = ()
    bot_eligible_candidate: bool = False
    bot_gate_reason: str = ""
    customer_brand: str = "unknown"
    customer_brand_source: str = "unknown"
    customer_brand_reason: str = "not_resolved"


def ensure_not_prod_apply_path(path: Path, *, allowed_root: Path | None = None) -> None:
    resolved_path = Path(path).expanduser().resolve(strict=False)
    resolved = str(resolved_path)
    if "customer_timeline_prod_" in resolved:
        raise ValueError(f"refusing to apply A2 mail ingest to prod timeline path: {resolved}")
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve(strict=False)
        try:
            resolved_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"refusing to apply A2 mail ingest outside allowed_root: {resolved}") from exc


def load_a2v3_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def prod_readonly_check(db_path: Path) -> Mapping[str, Any]:
    path = Path(db_path).expanduser()
    before_sha = file_sha256(path)
    with _prod_connection(path) as con:
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        email_events = int(
            con.execute("SELECT count(*) FROM timeline_events WHERE event_type = 'email_message'").fetchone()[0]
        )
        email_links = int(con.execute("SELECT count(*) FROM identity_links WHERE link_type = 'email'").fetchone()[0])
    after_sha = file_sha256(path)
    return {
        "path": str(path),
        "quick_check": quick_check,
        "email_events": email_events,
        "email_identity_links": email_links,
        "sha256_before": before_sha,
        "sha256_after": after_sha,
        "sha256_unchanged": before_sha == after_sha,
        "sqlite_uri_mode": "mode=ro&immutable=1",
        "query_only": True,
    }


def plan_a2v3_mail_ingest(config: A2V3MailIngestConfig) -> tuple[list[PlannedA2V3MailEvent], Mapping[str, Any]]:
    rows = load_a2v3_rows(config.input_jsonl)
    ensure_not_prod_apply_path(config.timeline_db_path, allowed_root=config.allowed_root)
    normalized_contacts = _collect_contact_values(rows)
    tallanto_matches = _load_tallanto_identity_matches(
        config.tallanto_identity_db,
        emails=normalized_contacts["email"],
        phones=normalized_contacts["phone"],
    )
    tallanto_ids = _tallanto_ids_from_matches(tallanto_matches)
    with _prod_connection(config.prod_timeline_db) as prod:
        snapshot = _load_prod_snapshot(
            prod,
            tenant_id=config.tenant_id,
            emails=normalized_contacts["email"],
            phones=normalized_contacts["phone"],
            message_shas={_message_sha(row) for row in rows},
            tallanto_ids=tallanto_ids,
        )

    plans: list[PlannedA2V3MailEvent] = []
    counters: Counter[str] = Counter(input_rows=len(rows))
    for row in rows:
        resolution = _resolve_customer(row, snapshot=snapshot, tallanto_matches=tallanto_matches)
        message_sha = _message_sha(row)
        prod_duplicate_keys = tuple(sorted(snapshot["prod_duplicate_keys_by_sha"].get(message_sha, ())))
        if prod_duplicate_keys:
            counters["prod_duplicate_messages"] += 1
        if resolution.outcome == "linked":
            counters["linked"] += 1
        elif resolution.outcome == "blocked":
            counters["blocked"] += 1
        elif resolution.outcome == "new_contact":
            counters["new_contact"] += 1
        else:
            counters["unmatched"] += 1

        event_at = _parse_event_at(row)
        source_ref = f"a2v3_mail:{int(row.get('_line_number') or 0)}:{message_sha[:16]}"
        linked_customer_id = resolution.customer_id if resolution.outcome == "linked" else None
        customer = _customer_for_resolution(resolution, snapshot)
        qualified = bool(
            linked_customer_id and snapshot["qualified_by_customer"].get(linked_customer_id, {}).get("qualified")
        )
        customer_brand, customer_brand_source, customer_brand_reason = _resolve_customer_brand(
            row,
            resolution=resolution,
            snapshot=snapshot,
        )
        counters[f"customer_brand.{customer_brand}"] += 1
        if resolution.outcome == "linked" and customer_brand == "unknown":
            counters[f"linked_customer_brand_unknown.{customer_brand_reason}"] += 1
        bot_eligible_candidate, bot_gate_reason = _bot_visibility(
            row,
            resolution=resolution,
            qualified=qualified,
            customer_brand=customer_brand,
        )
        counters[f"bot_gate.{bot_gate_reason}"] += 1
        if bot_eligible_candidate:
            counters["bot_eligible_candidate"] += 1
        else:
            counters["bot_ineligible_candidate"] += 1
        new_links = _identity_links_for_row(
            row,
            tenant_id=config.tenant_id,
            resolution=resolution,
            source_ref=source_ref,
            event_at=event_at,
            existing_email_values=snapshot["existing_email_values"],
        )
        counters["new_email_links_planned"] += sum(1 for link in new_links if link.link_type.value == "email")
        opportunity = None
        chunk = None
        if linked_customer_id:
            thread_id = _thread_id(row)
            opportunity = CustomerOpportunity(
                tenant_id=config.tenant_id,
                customer_id=linked_customer_id,
                opportunity_type=OpportunityType.MAIL_THREAD,
                source_system=A2V3_MAIL_SOURCE_SYSTEM,
                source_id=thread_id,
                title=compact_text(row.get("subject_full") or "Email thread", limit=180),
                status="observed",
                product_context={
                    "brand": customer_brand,
                    "brand_source": customer_brand_source,
                    "brand_reason": customer_brand_reason,
                    "email_brand": _brand(row),
                },
                opened_at=event_at,
                confidence=0.75,
                evidence={"message_sha256": message_sha, "thread_id_stable": True},
            )
        event = TimelineEvent(
            tenant_id=config.tenant_id,
            customer_id=linked_customer_id,
            opportunity_id=opportunity.opportunity_id if opportunity else None,
            event_type=TimelineEventType.EMAIL_MESSAGE,
            event_at=event_at,
            source_system=A2V3_MAIL_SOURCE_SYSTEM,
            source_id=message_sha,
            source_ref=source_ref,
            direction=_direction(row),
            subject=compact_text(row.get("subject_full") or "Email message", limit=220),
            text_preview=compact_text(_summary_text(row), limit=260),
            summary=compact_text(_summary_text(row), limit=700),
            match_status=_match_status_for_resolution(resolution),
            confidence=0.95 if resolution.outcome == "linked" else 0.0,
            record=_event_record(
                row,
                resolution=resolution,
                qualified=qualified,
                bot_eligible_candidate=bot_eligible_candidate,
                bot_gate_reason=bot_gate_reason,
                customer_brand=customer_brand,
                customer_brand_source=customer_brand_source,
                customer_brand_reason=customer_brand_reason,
            ),
            metadata={
                "a2v3_mail_ingest": True,
                "brand": customer_brand,
                "brand_source": customer_brand_source,
                "brand_reason": customer_brand_reason,
                "pending_attribution": resolution.outcome != "linked",
                "pending_reason": None if resolution.outcome == "linked" else resolution.reason,
            },
            created_at=event_at,
        )
        if linked_customer_id and _chunk_text(row):
            chunk = BotContextChunk(
                tenant_id=config.tenant_id,
                customer_id=linked_customer_id,
                opportunity_id=opportunity.opportunity_id if opportunity else None,
                event_id=event.event_id,
                source_ref=source_ref,
                source_system=A2V3_MAIL_SOURCE_SYSTEM,
                chunk_type="email_message",
                text=_chunk_text(row),
                summary=compact_text(_summary_text(row), limit=700),
                event_at=event_at,
                freshness_score=0.72,
                relevance_tags=_relevance_tags(row, bot_eligible_candidate=bot_eligible_candidate),
                allowed_for_bot=False,
                requires_manager_review=True,
                metadata={
                    "message_sha256": message_sha,
                    "memory_status": _quality(row).get("memory_status"),
                    "bot_eligible_candidate": bot_eligible_candidate,
                    "bot_gate_reason": bot_gate_reason,
                    "brand": customer_brand,
                    "brand_source": customer_brand_source,
                    "email_brand": _brand(row),
                    "thread_id": _thread_id(row),
                    "safe_next_step_note": _quality(row).get("safe_next_step_note"),
                    "requires_human_confirmation": bool(_quality(row).get("requires_human_confirmation")),
                },
                created_at=event_at,
            )
            counters["chunks_planned"] += 1
        plans.append(
            PlannedA2V3MailEvent(
                row=row,
                event=event,
                customer=customer,
                identity_links=tuple(new_links),
                opportunity=opportunity,
                chunk=chunk,
                resolution=resolution,
                prod_duplicate_keys=prod_duplicate_keys,
                bot_eligible_candidate=bot_eligible_candidate,
                bot_gate_reason=bot_gate_reason,
                customer_brand=customer_brand,
                customer_brand_source=customer_brand_source,
                customer_brand_reason=customer_brand_reason,
            )
        )

    duplicate_in_input = len(rows) - len({_message_sha(row) for row in rows})
    counters["duplicate_messages_in_input"] = duplicate_in_input
    return plans, {
        "schema_version": A2V3_MAIL_INGEST_SCHEMA_VERSION,
        "counts": dict(counters),
        "prod_snapshot": snapshot["summary"],
        "tallanto_identity_matches": tallanto_matches["summary"],
    }


def create_test_db_backup(config: A2V3MailIngestConfig, *, label: str = "a2v3_mail") -> Mapping[str, Any]:
    ensure_not_prod_apply_path(config.timeline_db_path, allowed_root=config.allowed_root)
    if not config.timeline_db_path.exists():
        raise FileNotFoundError(f"test timeline DB does not exist: {config.timeline_db_path}")
    backup_dir = config.out_dir / "backups" / f"{label}_{_now_stamp()}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_db = backup_dir / config.timeline_db_path.name
    with sqlite3.connect(f"file:{config.timeline_db_path}?mode=ro", uri=True) as source, sqlite3.connect(backup_db) as target:
        source.backup(target)
    manifest = {
        "schema_version": A2V3_MAIL_INGEST_SCHEMA_VERSION,
        "kind": "a2v3_test_db_backup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_db_path": str(config.timeline_db_path.resolve(strict=False)),
        "backup_db_path": str(backup_db.resolve(strict=False)),
        "backup_sha256": file_sha256(backup_db),
        "test_db_sha256_after_backup": file_sha256(config.timeline_db_path),
    }
    manifest_path = backup_dir / "backup_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def validate_a2v3_mail_ingest(config: A2V3MailIngestConfig) -> Mapping[str, Any]:
    plans, report = plan_a2v3_mail_ingest(config)
    test_existing = existing_event_dedupe_keys(config.timeline_db_path)
    test_existing_content = existing_timeline_email_content_signatures(config.timeline_db_path)
    seen: set[str] = set()
    seen_content: set[str] = set()
    would_create = 0
    would_skip_existing = 0
    would_skip_existing_content = 0
    would_skip_content_in_input = 0
    would_skip_prod_duplicate = 0
    for plan in plans:
        key = plan.event.dedupe_key
        if key in seen or key in test_existing:
            would_skip_existing += 1
            continue
        seen.add(key)
        content_signature = timeline_email_content_signature(
            tenant_id=plan.event.tenant_id,
            customer_id=plan.event.customer_id,
            event_type=plan.event.event_type.value,
            event_at=plan.event.event_at,
            subject=plan.event.subject,
            summary=plan.event.summary,
            text_preview=plan.event.text_preview,
        )
        if content_signature and content_signature in seen_content:
            would_skip_content_in_input += 1
            continue
        if content_signature and content_signature in test_existing_content:
            would_skip_existing_content += 1
            continue
        if content_signature:
            seen_content.add(content_signature)
        if plan.prod_duplicate_keys:
            would_skip_prod_duplicate += 1
        would_create += 1
    validation = {
        **report,
        "mode": "validate",
        "input_jsonl": str(config.input_jsonl),
        "test_db_path": str(config.timeline_db_path),
        "would_create_events": would_create,
        "would_skip_existing_test_events": would_skip_existing,
        "would_skip_existing_content_events": would_skip_existing_content,
        "would_skip_content_events_in_input": would_skip_content_in_input,
        "would_skip_prod_duplicate_events": would_skip_prod_duplicate,
    }
    write_json(config.out_dir / "validate_report.json", validation)
    return validation


def apply_a2v3_mail_ingest(config: A2V3MailIngestConfig, *, backup_manifest_path: Path) -> Mapping[str, Any]:
    ensure_not_prod_apply_path(config.timeline_db_path, allowed_root=config.allowed_root)
    _validate_backup_manifest(config, backup_manifest_path)
    plans, planning_report = plan_a2v3_mail_ingest(config)
    test_existing = existing_event_dedupe_keys(config.timeline_db_path)
    test_existing_content = existing_timeline_email_content_signatures(config.timeline_db_path)
    existing_event_actions = _classify_existing_a2v3_events(config.timeline_db_path, plans)
    seen: set[str] = set()
    seen_content: set[str] = set()
    selected: list[PlannedA2V3MailEvent] = []
    enrich_conflicts: list[Mapping[str, Any]] = []
    enrich_diffs: list[Mapping[str, Any]] = []
    counters: Counter[str] = Counter()
    for plan in plans:
        key = plan.event.dedupe_key
        if key in seen:
            counters["skipped_duplicate_in_input"] += 1
            continue
        seen.add(key)
        existing_action = existing_event_actions.get(_message_sha(plan.row))
        if existing_action is not None and existing_action["status"] == "identical":
            counters["skipped_identical_events"] += 1
            continue
        if existing_action is not None and existing_action["status"] == "hash_mismatch":
            if not config.enrich_existing:
                counters["hash_mismatch_events"] += 1
                enrich_conflicts.append(existing_action)
                continue
            counters["enrich_existing_events"] += 1
            enrich_diffs.append(_a2v3_enrich_diff(plan, existing_action))
            plan = replace(plan, event=_merged_existing_event(plan.event, existing_action))
        elif key in test_existing:
            counters["skipped_existing_test"] += 1
            continue
        content_signature = timeline_email_content_signature(
            tenant_id=plan.event.tenant_id,
            customer_id=plan.event.customer_id,
            event_type=plan.event.event_type.value,
            event_at=plan.event.event_at,
            subject=plan.event.subject,
            summary=plan.event.summary,
            text_preview=plan.event.text_preview,
        )
        if content_signature and content_signature in seen_content:
            counters["skipped_content_duplicate_in_input"] += 1
            continue
        if content_signature and content_signature in test_existing_content:
            counters["skipped_existing_content"] += 1
            continue
        if content_signature:
            seen_content.add(content_signature)
        if plan.prod_duplicate_keys:
            counters["prod_duplicate_events_observed"] += 1
        selected.append(plan)

    if enrich_conflicts:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(config.out_dir / "enrich_existing_required.json", {"conflicts": enrich_conflicts})
        ids = ", ".join(str(item["event_id"]) for item in enrich_conflicts[:10])
        raise ValueError(f"enrich_existing required for existing A2v3 event hash mismatch: {ids}")
    if enrich_diffs:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(config.out_dir / "enrich_existing_diff.json", {"diffs": enrich_diffs})

    store = CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=config.allowed_root)
    accepted_plans: list[PlannedA2V3MailEvent] = []
    facts_upserted = 0
    try:
        input_hash = _input_fingerprint(config)
        run = store.start_ingestion_run(
            tenant_id=config.tenant_id,
            source_system=A2V3_MAIL_SOURCE_SYSTEM,
            source_ref=config.source_ref,
            run_kind="a2v3_mail_test_apply",
            idempotency_key=f"{config.source_ref}:{input_hash}",
            input_hash=input_hash,
            metadata={"input_jsonl": str(config.input_jsonl), "backup_manifest": str(backup_manifest_path)},
            actor="a2v3_mail_ingest_runner",
        )
        with store.bulk_write():
            for plan in selected:
                if plan.customer is not None:
                    store.upsert_customer(plan.customer, actor="a2v3_mail_ingest_runner", ingestion_run_id=run.run_id)
                for link in plan.identity_links:
                    link_result = store.upsert_identity_link(
                        link,
                        actor="a2v3_mail_ingest_runner",
                        ingestion_run_id=run.run_id,
                    )
                    if link_result.created:
                        counters["created_identity_links"] += 1
                    else:
                        counters["updated_identity_links"] += 1
                if plan.opportunity is not None:
                    store.upsert_opportunity(plan.opportunity, actor="a2v3_mail_ingest_runner", ingestion_run_id=run.run_id)
                event_result = store.upsert_event(
                    plan.event,
                    actor="a2v3_mail_ingest_runner",
                    ingestion_run_id=run.run_id,
                )
                if event_result.created:
                    counters["created_events"] += 1
                    accepted_plans.append(plan)
                elif event_result.status == "duplicate" and event_result.record_id != plan.event.event_id:
                    counters["skipped_content_duplicate_events"] += 1
                    continue
                if plan.chunk is not None:
                    chunk_result = store.upsert_bot_context_chunk(
                        plan.chunk,
                        actor="a2v3_mail_ingest_runner",
                        ingestion_run_id=run.run_id,
                    )
                    if chunk_result.created:
                        counters["created_chunks"] += 1
                        if plan.bot_eligible_candidate:
                            counters["created_bot_eligible_candidate_chunks"] += 1
            fact_plans = _plans_with_existing_a2v3_events(store._con, plans)
            facts_upserted = _upsert_a2v3_event_facts(
                config.timeline_db_path,
                fact_plans,
                connection=store._con,
                allowed_root=config.allowed_root,
            )
            counters["upserted_a2v3_event_facts"] += facts_upserted
            counters["reconciled_a2v3_event_facts"] += max(0, facts_upserted - len(accepted_plans))
        store.finish_ingestion_run(
            run.run_id,
            status="completed",
            accepted_count=int(counters.get("created_events", 0)),
            rejected_count=sum(value for key, value in counters.items() if key.startswith("skipped_")),
            output_ref=str(config.out_dir / "apply_report.json"),
            metadata={"counts": dict(counters)},
            actor="a2v3_mail_ingest_runner",
        )
    finally:
        store.close()
    report = {
        **planning_report,
        "mode": "apply",
        "test_db_path": str(config.timeline_db_path),
        "selected_events": len(selected),
        "counts": {**planning_report["counts"], **dict(counters)},
    }
    write_json(config.out_dir / "apply_report.json", report)
    return report


def _classify_existing_a2v3_events(
    db_path: Path,
    plans: Sequence[PlannedA2V3MailEvent],
) -> Mapping[str, Mapping[str, Any]]:
    path = Path(db_path)
    if not path.exists() or not plans:
        return {}
    source_ids = [_message_sha(plan.row) for plan in plans]
    placeholders = ",".join("?" for _ in source_ids)
    by_sha = {_message_sha(plan.row): plan for plan in plans}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='timeline_events'").fetchone()
        if not exists:
            return {}
        rows = con.execute(
            f"""
            SELECT event_id, source_id, record_hash, record_json
            FROM timeline_events
            WHERE source_system = ?
              AND source_id IN ({placeholders})
            """,
            (A2V3_MAIL_SOURCE_SYSTEM, *source_ids),
        ).fetchall()
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        message_sha = str(row["source_id"]).lower()
        plan = by_sha.get(message_sha)
        if plan is None:
            continue
        incoming_hash = stable_digest(scrub_timeline_persisted_json(plan.event.to_json_dict()))
        existing_hash = str(row["record_hash"])
        result[message_sha] = {
            "status": "identical" if incoming_hash == existing_hash else "hash_mismatch",
            "event_id": str(row["event_id"]),
            "message_sha256": message_sha,
            "existing_hash": existing_hash,
            "incoming_hash": incoming_hash,
            "record_json": str(row["record_json"]),
        }
    return result


def _merged_existing_event(event: TimelineEvent, existing_action: Mapping[str, Any]) -> TimelineEvent:
    existing_payload = json.loads(str(existing_action["record_json"]))
    existing_metadata = existing_payload.get("metadata") if isinstance(existing_payload.get("metadata"), Mapping) else {}
    existing_record = existing_payload.get("record") if isinstance(existing_payload.get("record"), Mapping) else {}
    merged_metadata = {**dict(event.metadata), **dict(existing_metadata)}
    existing_record_dict = dict(existing_record)
    existing_record_dict.pop("bot_visibility", None)
    merged_record = {**existing_record_dict, **dict(event.record)}
    existing_source_refs = existing_payload.get("source_refs")
    if not isinstance(existing_source_refs, Sequence) or isinstance(existing_source_refs, (str, bytes, bytearray)):
        existing_source_refs = (existing_payload.get("source_ref") or event.source_ref,)
    return replace(
        event,
        event_at=parse_datetime(str(existing_payload["event_at"]), "event_at"),
        source_system=str(existing_payload["source_system"]),
        source_id=str(existing_payload["source_id"]),
        source_ref=str(existing_payload.get("source_ref") or event.source_ref),
        source_refs=tuple(str(item) for item in existing_source_refs if item),
        direction=str(existing_payload["direction"]),
        created_at=parse_datetime(str(existing_payload["created_at"]), "created_at"),
        metadata=merged_metadata,
        record=merged_record,
    )


def _a2v3_enrich_diff(plan: PlannedA2V3MailEvent, existing_action: Mapping[str, Any]) -> Mapping[str, Any]:
    existing_payload = json.loads(str(existing_action["record_json"]))
    incoming_payload = plan.event.to_json_dict()
    fields = ("subject", "text_preview", "summary")
    changes = {}
    for field in fields:
        if existing_payload.get(field) != incoming_payload.get(field):
            changes[field] = {
                "old": compact_text(existing_payload.get(field), limit=80),
                "new": compact_text(incoming_payload.get(field), limit=80),
            }
    old_metadata = existing_payload.get("metadata") if isinstance(existing_payload.get("metadata"), Mapping) else {}
    new_metadata = incoming_payload.get("metadata") if isinstance(incoming_payload.get("metadata"), Mapping) else {}
    preserved_metadata_keys = sorted(set(old_metadata) & set(new_metadata))
    added_metadata_keys = sorted(set(new_metadata) - set(old_metadata))
    return {
        "event_id": existing_action["event_id"],
        "message_sha256": str(existing_action["message_sha256"])[:16],
        "existing_hash": existing_action["existing_hash"],
        "incoming_hash": existing_action["incoming_hash"],
        "changes": changes,
        "preserved_existing_metadata_keys": preserved_metadata_keys,
        "added_metadata_keys": added_metadata_keys,
    }


def _upsert_a2v3_event_facts(
    db_path: Path,
    plans: Sequence[PlannedA2V3MailEvent],
    *,
    allowed_root: Path,
    connection: sqlite3.Connection | None = None,
) -> int:
    ensure_not_prod_apply_path(db_path, allowed_root=allowed_root)
    if not plans:
        _ensure_a2v3_event_facts_table(db_path, connection=connection)
        return 0
    owns_connection = connection is None
    con = connection or sqlite3.connect(Path(db_path))
    try:
        _ensure_a2v3_event_facts_table(db_path, connection=con)
        rows = [_a2v3_event_fact_row(plan) for plan in plans]
        con.executemany(
            """
            INSERT INTO a2v3_mail_event_facts (
              event_id, tenant_id, customer_id, message_sha256, event_type_detail,
              money_direction, amount_kind, amount_rub, money_amounts_rub_json,
              amount_uncertain, email_brand, email_brand_source, customer_brand,
              customer_brand_source, customer_brand_reason, contact_email, contact_phone,
              contact_name, student_name, grade, subject_area, memory_status,
              bot_visible, bot_gate_reason, identity_outcome, identity_reason,
              created_at
            )
            VALUES (
              :event_id, :tenant_id, :customer_id, :message_sha256, :event_type_detail,
              :money_direction, :amount_kind, :amount_rub, :money_amounts_rub_json,
              :amount_uncertain, :email_brand, :email_brand_source, :customer_brand,
              :customer_brand_source, :customer_brand_reason, :contact_email, :contact_phone,
              :contact_name, :student_name, :grade, :subject_area, :memory_status,
              :bot_visible, :bot_gate_reason, :identity_outcome, :identity_reason,
              :created_at
            )
            ON CONFLICT(message_sha256) DO UPDATE SET
              event_id = excluded.event_id,
              customer_id = excluded.customer_id,
              event_type_detail = excluded.event_type_detail,
              money_direction = excluded.money_direction,
              amount_kind = excluded.amount_kind,
              amount_rub = excluded.amount_rub,
              money_amounts_rub_json = excluded.money_amounts_rub_json,
              amount_uncertain = excluded.amount_uncertain,
              email_brand = excluded.email_brand,
              email_brand_source = excluded.email_brand_source,
              customer_brand = excluded.customer_brand,
              customer_brand_source = excluded.customer_brand_source,
              customer_brand_reason = excluded.customer_brand_reason,
              contact_email = excluded.contact_email,
              contact_phone = excluded.contact_phone,
              contact_name = excluded.contact_name,
              student_name = excluded.student_name,
              grade = excluded.grade,
              subject_area = excluded.subject_area,
              memory_status = excluded.memory_status,
              bot_visible = excluded.bot_visible,
              bot_gate_reason = excluded.bot_gate_reason,
              identity_outcome = excluded.identity_outcome,
              identity_reason = excluded.identity_reason
            """,
            rows,
        )
        if owns_connection:
            con.commit()
    finally:
        if owns_connection:
            con.close()
    return len(plans)


def reconcile_a2v3_event_facts(
    db_path: Path,
    plans: Sequence[PlannedA2V3MailEvent],
    *,
    allowed_root: Path,
) -> Mapping[str, Any]:
    ensure_not_prod_apply_path(db_path, allowed_root=allowed_root)
    with sqlite3.connect(Path(db_path)) as con:
        con.row_factory = sqlite3.Row
        fact_plans = _plans_with_existing_a2v3_events(con, plans)
        upserted = _upsert_a2v3_event_facts(db_path, fact_plans, connection=con, allowed_root=allowed_root)
        con.commit()
    return {
        "plans": len(plans),
        "events_found": len(fact_plans),
        "facts_upserted": upserted,
    }


def _plans_with_existing_a2v3_events(
    con: sqlite3.Connection,
    plans: Sequence[PlannedA2V3MailEvent],
) -> list[PlannedA2V3MailEvent]:
    if not plans:
        return []
    source_ids = [_message_sha(plan.row) for plan in plans]
    placeholders = ",".join("?" for _ in source_ids)
    rows = con.execute(
        f"""
        SELECT source_id
        FROM timeline_events
        WHERE source_system = ?
          AND source_id IN ({placeholders})
        """,
        (A2V3_MAIL_SOURCE_SYSTEM, *source_ids),
    ).fetchall()
    existing = {str(row["source_id"]).lower() for row in rows}
    return [plan for plan in plans if _message_sha(plan.row) in existing]


def _ensure_a2v3_event_facts_table(db_path: Path, *, connection: sqlite3.Connection | None = None) -> None:
    owns_connection = connection is None
    con = connection or sqlite3.connect(Path(db_path))
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS a2v3_mail_event_facts (
              event_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              customer_id TEXT,
              message_sha256 TEXT NOT NULL UNIQUE,
              event_type_detail TEXT,
              money_direction TEXT,
              amount_kind TEXT,
              amount_rub REAL,
              money_amounts_rub_json TEXT NOT NULL,
              amount_uncertain INTEGER NOT NULL,
              email_brand TEXT NOT NULL,
              email_brand_source TEXT,
              customer_brand TEXT NOT NULL,
              customer_brand_source TEXT NOT NULL,
              customer_brand_reason TEXT NOT NULL,
              contact_email TEXT,
              contact_phone TEXT,
              contact_name TEXT,
              student_name TEXT,
              grade TEXT,
              subject_area TEXT,
              memory_status TEXT,
              bot_visible INTEGER NOT NULL,
              bot_gate_reason TEXT NOT NULL,
              identity_outcome TEXT NOT NULL,
              identity_reason TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_a2v3_mail_event_facts_customer
              ON a2v3_mail_event_facts(tenant_id, customer_id, customer_brand);
            CREATE INDEX IF NOT EXISTS idx_a2v3_mail_event_facts_type_amount
              ON a2v3_mail_event_facts(tenant_id, event_type_detail, amount_kind, amount_rub);
            CREATE INDEX IF NOT EXISTS idx_a2v3_mail_event_facts_bot_visible
              ON a2v3_mail_event_facts(tenant_id, bot_visible, customer_brand);
            """
        )
        if owns_connection:
            con.commit()
    finally:
        if owns_connection:
            con.close()


def _a2v3_event_fact_row(plan: PlannedA2V3MailEvent) -> Mapping[str, Any]:
    row = plan.row
    payload = row.get("summary_payload") or {}
    quality = _quality(row)
    return {
        "event_id": plan.event.event_id,
        "tenant_id": plan.event.tenant_id,
        "customer_id": plan.resolution.customer_id if plan.resolution.outcome == "linked" else None,
        "message_sha256": _message_sha(row),
        "event_type_detail": _detail_event_type(row),
        "money_direction": payload.get("money_direction"),
        "amount_kind": payload.get("amount_kind"),
        "amount_rub": payload.get("amount_rub"),
        "money_amounts_rub_json": json.dumps(quality.get("money_amounts_rub") or [], ensure_ascii=False),
        "amount_uncertain": 1 if quality.get("amount_uncertain") else 0,
        "email_brand": _brand(row),
        "email_brand_source": row.get("brand_source") or row.get("raw_infer_offline_brand"),
        "customer_brand": plan.customer_brand,
        "customer_brand_source": plan.customer_brand_source,
        "customer_brand_reason": plan.customer_brand_reason,
        "contact_email": row.get("contact_email"),
        "contact_phone": row.get("contact_phone"),
        "contact_name": row.get("contact_name"),
        "student_name": payload.get("student_name"),
        "grade": payload.get("grade"),
        "subject_area": payload.get("subject_area"),
        "memory_status": quality.get("memory_status"),
        "bot_visible": 1 if plan.bot_eligible_candidate else 0,
        "bot_gate_reason": plan.bot_gate_reason,
        "identity_outcome": plan.resolution.outcome,
        "identity_reason": plan.resolution.reason,
        "created_at": plan.event.created_at.isoformat(),
    }


def existing_event_dedupe_keys(db_path: Path) -> set[str]:
    path = Path(db_path)
    if not path.exists():
        return set()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='timeline_events'").fetchone()
        if not exists:
            return set()
        return {str(row["dedupe_key"]) for row in con.execute("SELECT dedupe_key FROM timeline_events")}


def build_local_client_review(config: A2V3MailIngestConfig, plans: Sequence[PlannedA2V3MailEvent]) -> Mapping[str, Any]:
    fact_by_sha = _a2v3_facts_by_message_sha(config.timeline_db_path)
    grouped: dict[str, list[PlannedA2V3MailEvent]] = defaultdict(list)
    for plan in plans:
        key = f"customer:{plan.resolution.customer_id}" if plan.resolution.outcome == "linked" and plan.resolution.customer_id else (
            f"unresolved:{plan.resolution.outcome}:{plan.event.source_id[:16]}"
        )
        grouped[key].append(plan)
    review_rows: list[dict[str, Any]] = []
    with _prod_connection(config.prod_timeline_db) as prod:
        for group_id in sorted(grouped):
            new_plans = grouped[group_id]
            customer_id = new_plans[0].resolution.customer_id if group_id.startswith("customer:") else None
            existing = (
                _customer_history_summary(prod, tenant_id=config.tenant_id, customer_id=customer_id)
                if customer_id
                else _empty_customer_history_summary()
            )
            existing_events = existing.get("events") or []
            combined_timeline = _combined_timeline(existing_events, new_plans, fact_by_sha=fact_by_sha)
            next_step = _review_next_step(existing_events, new_plans, customer_id=customer_id)
            review_rows.append(
                {
                    "review_group_id": group_id,
                    "customer_id": customer_id,
                    "contact": _review_contact(new_plans),
                    "brand_values": sorted({_brand(plan.row) for plan in new_plans}),
                    "existing_before": existing,
                    "new_email_events": [_review_new_event(plan, fact_by_sha=fact_by_sha) for plan in new_plans],
                    "combined_timeline": combined_timeline,
                    "combined_timeline_count": len(combined_timeline),
                    "resolution": dict(Counter(plan.resolution.outcome for plan in new_plans)),
                    "resolution_reasons": sorted({plan.resolution.reason for plan in new_plans}),
                    "current_next_step": next_step,
                    "bot_eligible_candidate_new_chunks": sum(1 for plan in new_plans if plan.bot_eligible_candidate),
                }
            )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = config.out_dir / "timeline_100_clients_review.jsonl"
    csv_path = config.out_dir / "timeline_100_clients_review.csv"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in review_rows) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "review_group_id",
            "customer_id",
            "contact_email",
            "contact_phone",
            "contact_name",
            "brands",
            "resolution",
            "resolution_reasons",
            "existing_event_count",
            "existing_event_type_counts",
            "existing_last_items",
            "new_email_count",
            "new_email_summaries",
            "combined_timeline_count",
            "combined_timeline",
            "current_next_step_status",
            "current_next_step",
            "current_next_step_reason",
            "bot_eligible_candidate_new_chunks",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_rows:
            contact = row["contact"]
            writer.writerow(
                {
                    "review_group_id": row["review_group_id"],
                    "customer_id": row["customer_id"],
                    "contact_email": contact.get("email", ""),
                    "contact_phone": contact.get("phone", ""),
                    "contact_name": contact.get("name", ""),
                    "brands": ", ".join(row["brand_values"]),
                    "resolution": json.dumps(row["resolution"], ensure_ascii=False, sort_keys=True),
                    "resolution_reasons": ", ".join(row["resolution_reasons"]),
                    "existing_event_count": row["existing_before"]["event_count"],
                    "existing_event_type_counts": json.dumps(
                        row["existing_before"]["event_type_counts"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "existing_last_items": json.dumps(row["existing_before"]["last_items"], ensure_ascii=False),
                    "new_email_count": len(row["new_email_events"]),
                    "new_email_summaries": json.dumps(row["new_email_events"], ensure_ascii=False),
                    "combined_timeline_count": row["combined_timeline_count"],
                    "combined_timeline": json.dumps(row["combined_timeline"], ensure_ascii=False),
                    "current_next_step_status": row["current_next_step"].get("status", ""),
                    "current_next_step": row["current_next_step"].get("display_text", ""),
                    "current_next_step_reason": row["current_next_step"].get("reason_code", ""),
                    "bot_eligible_candidate_new_chunks": row["bot_eligible_candidate_new_chunks"],
                }
            )
    return {
        "rows": len(review_rows),
        "new_email_events": sum(len(row["new_email_events"]) for row in review_rows),
        "linked_groups": sum(1 for row in review_rows if row["customer_id"]),
        "unresolved_groups": sum(1 for row in review_rows if not row["customer_id"]),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
    }


def _a2v3_facts_by_message_sha(db_path: Path) -> Mapping[str, Mapping[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return {}
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
        ).fetchone()
        if not exists:
            return {}
        facts: dict[str, Mapping[str, Any]] = {}
        for row in con.execute("SELECT * FROM a2v3_mail_event_facts"):
            payload = dict(row)
            payload["money_amounts_rub"] = json.loads(payload.get("money_amounts_rub_json") or "[]")
            facts[str(payload["message_sha256"]).lower()] = payload
        return facts


def verify_test_db(config: A2V3MailIngestConfig) -> Mapping[str, Any]:
    with sqlite3.connect(f"file:{config.timeline_db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        counts = {
            table: int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in (
                "customer_identities",
                "identity_links",
                "customer_opportunities",
                "timeline_events",
                "bot_context_chunks",
                "ingestion_runs",
            )
        }
        bad_sync_chunks = int(
            con.execute(
                """
                SELECT count(*)
                FROM bot_context_chunks
                WHERE (allowed_for_bot = 1 AND requires_manager_review != 0)
                   OR (allowed_for_bot = 0 AND requires_manager_review != 1)
                """
            ).fetchone()[0]
        )
        visible_not_usable = int(
            con.execute(
                """
                SELECT count(*)
                FROM bot_context_chunks
                WHERE allowed_for_bot = 1
                  AND json_extract(record_json, '$.metadata.memory_status') != 'usable_memory'
                """
            ).fetchone()[0]
        )
        match_status_counts = {
            str(row["match_status"]): int(row["c"])
            for row in con.execute("SELECT match_status, count(*) AS c FROM timeline_events GROUP BY match_status")
        }
        a2_facts_table_exists = bool(
            con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
            ).fetchone()
        )
        a2_fact_counts: Mapping[str, Any] = {}
        if a2_facts_table_exists:
            a2_fact_counts = {
                "rows": int(con.execute("SELECT count(*) FROM a2v3_mail_event_facts").fetchone()[0]),
                "bot_eligible_candidate": int(
                    con.execute("SELECT count(*) FROM a2v3_mail_event_facts WHERE bot_visible = 1").fetchone()[0]
                ),
                "unknown_linked_customer_brand": int(
                    con.execute(
                        """
                        SELECT count(*)
                        FROM a2v3_mail_event_facts
                        WHERE identity_outcome = 'linked' AND customer_brand = 'unknown'
                        """
                    ).fetchone()[0]
                ),
                "amount_rows": int(
                    con.execute(
                        """
                        SELECT count(*)
                        FROM a2v3_mail_event_facts
                        WHERE amount_rub IS NOT NULL OR money_amounts_rub_json != '[]'
                        """
                    ).fetchone()[0]
                ),
                "event_type_detail_counts": {
                    str(row["event_type_detail"]): int(row["c"])
                    for row in con.execute(
                        """
                        SELECT event_type_detail, count(*) AS c
                        FROM a2v3_mail_event_facts
                        GROUP BY event_type_detail
                        ORDER BY event_type_detail
                        """
                    )
                },
                "customer_brand_counts": {
                    str(row["customer_brand"]): int(row["c"])
                    for row in con.execute(
                        """
                        SELECT customer_brand, count(*) AS c
                        FROM a2v3_mail_event_facts
                        GROUP BY customer_brand
                        ORDER BY customer_brand
                        """
                    )
                },
                "bot_gate_counts": {
                    str(row["bot_gate_reason"]): int(row["c"])
                    for row in con.execute(
                        """
                        SELECT bot_gate_reason, count(*) AS c
                        FROM a2v3_mail_event_facts
                        GROUP BY bot_gate_reason
                        ORDER BY bot_gate_reason
                        """
                    )
                },
            }
        chunk_gate_counts = [
            {
                "allowed_for_bot": int(row["allowed_for_bot"]),
                "requires_manager_review": int(row["requires_manager_review"]),
                "count": int(row["c"]),
            }
            for row in con.execute(
                """
                SELECT allowed_for_bot, requires_manager_review, count(*) AS c
                FROM bot_context_chunks
                GROUP BY allowed_for_bot, requires_manager_review
                ORDER BY allowed_for_bot, requires_manager_review
                """
            )
        ]
    report = {
        "quick_check": quick_check,
        "counts": counts,
        "bad_sync_chunks": bad_sync_chunks,
        "visible_not_usable": visible_not_usable,
        "match_status_counts": match_status_counts,
        "a2_fact_counts": a2_fact_counts,
        "chunk_gate_counts": chunk_gate_counts,
    }
    write_json(config.out_dir / "test_db_verification.json", report)
    return report


def write_foton_report(
    path: Path,
    *,
    prod_check_before: Mapping[str, Any],
    prod_check_after: Mapping[str, Any],
    validate_report: Mapping[str, Any],
    first_apply_report: Mapping[str, Any],
    second_apply_report: Mapping[str, Any],
    review_report: Mapping[str, Any],
    test_db_verification: Mapping[str, Any],
    test_db_path: Path,
) -> None:
    counts = first_apply_report.get("counts", {})
    a2_fact_counts = test_db_verification.get("a2_fact_counts") or {}
    lines = [
        "# A2-v3: тестовое вливание 100 писем в customer_timeline",
        "",
        "## Безопасность",
        f"- Прод-БД открывалась только `mode=ro&immutable=1`, `query_only=ON`: {prod_check_before.get('quick_check')} / {prod_check_after.get('quick_check')}.",
        f"- sha256 прода до: `{prod_check_before.get('sha256_before')}`",
        f"- sha256 прода после: `{prod_check_after.get('sha256_after')}`",
        f"- sha256 совпал: `{prod_check_before.get('sha256_before') == prod_check_after.get('sha256_after')}`",
        f"- Тест-БД вне прода: `{test_db_path}`",
        "- AMO/Tallanto/CRM/live-write/client sends: 0.",
        "",
        "## Вход",
        f"- A2-v3 input rows: `{validate_report.get('counts', {}).get('input_rows')}`",
        f"- Дедуп против prod (`mail_archive` + `mail_archive_stage2`): `{validate_report.get('would_skip_prod_duplicate_events')}` уже существующих.",
        "",
        "## Привязка",
        f"- linked: `{counts.get('linked', 0)}`",
        f"- unmatched: `{counts.get('unmatched', 0)}`",
        f"- blocked: `{counts.get('blocked', 0)}`",
        f"- new_contact: `{counts.get('new_contact', 0)}`",
        f"- новых email identity_links в тест-БД запланировано: `{counts.get('new_email_links_planned', 0)}`",
        f"- identity_links создано при apply: `{counts.get('created_identity_links', 0)}`",
        f"- identity_links обновлено/повторно подтверждено при apply: `{counts.get('updated_identity_links', 0)}`",
        "",
        "## Bot visibility gate",
        f"- bot-eligible candidate chunks: `{counts.get('created_bot_eligible_candidate_chunks', 0)}`",
        f"- bot-ineligible candidate rows: `{counts.get('bot_ineligible_candidate', 0)}`",
        "- Правило Э1 v4: все `mail_archive_stage2` chunks записаны `allowed_for_bot=0`, `requires_manager_review=1`.",
        "- `_bot_visibility` в Э1 является только диагностикой `bot_eligible_candidate`; открытие боту переносится в Э4б.",
        "",
        "## A2 first-class факты в тест-БД",
        f"- a2v3_mail_event_facts rows: `{a2_fact_counts.get('rows', 0)}`",
        f"- event_type_detail counts: `{a2_fact_counts.get('event_type_detail_counts', {})}`",
        f"- customer_brand counts: `{a2_fact_counts.get('customer_brand_counts', {})}`",
        f"- linked с unresolved customer_brand: `{a2_fact_counts.get('unknown_linked_customer_brand', 0)}`",
        f"- linked unknown-brand reasons: `{ {key.replace('linked_customer_brand_unknown.', ''): value for key, value in counts.items() if key.startswith('linked_customer_brand_unknown.')} }`",
        f"- rows with amount fields: `{a2_fact_counts.get('amount_rows', 0)}`",
        f"- bot gate counts: `{a2_fact_counts.get('bot_gate_counts', {})}`",
        "- Базовый `timeline_events.event_type` остаётся `email_message`, потому что контракт TimelineEventType не принимает `payment/refund`; бизнес-тип хранится в `a2v3_mail_event_facts.event_type_detail`.",
        "",
        "## Apply в тест-БД",
        f"- selected events first apply: `{first_apply_report.get('selected_events')}`",
        f"- created_events first apply: `{counts.get('created_events', 0)}`",
        f"- created_chunks first apply: `{counts.get('created_chunks', 0)}`",
        f"- repeat apply selected_events: `{second_apply_report.get('selected_events')}`",
        f"- repeat apply created_events: `{second_apply_report.get('counts', {}).get('created_events', 0)}`",
        f"- test DB quick_check: `{test_db_verification.get('quick_check')}`",
        f"- test DB row counts: `{test_db_verification.get('counts')}`",
        f"- bad sync chunks: `{test_db_verification.get('bad_sync_chunks')}`",
        f"- visible-not-usable chunks: `{test_db_verification.get('visible_not_usable')}`",
        f"- chunk gate counts: `{test_db_verification.get('chunk_gate_counts')}`",
        "",
        "## Локальная таблица для ручного просмотра Дмитрия",
        f"- rows grouped by client/unresolved identity: `{review_report.get('rows')}`",
        f"- new email events represented: `{review_report.get('new_email_events')}`",
        f"- linked client rows: `{review_report.get('linked_groups')}`",
        f"- unresolved/blocked rows: `{review_report.get('unresolved_groups')}`",
        f"- JSONL с ПДн: `{review_report.get('jsonl')}`",
        f"- CSV с ПДн: `{review_report.get('csv')}`",
        "- Таблица содержит: контакт+бренд, память до письма, новые email-события, объединённую ленту и текущий next-step по объединённому виду.",
        "",
        "## Ограничения",
        "- Это контрольный тест на свежей тест-БД, не решение о prod-вливании.",
        "- Непривязанные/спорные письма сохранены только как тестовый outcome, боту не видны.",
        "- `next_step` и агрегат покупок остаются Phase 2 в timeline.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prod_connection(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro&immutable=1", uri=True, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def _collect_contact_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    values = {"email": set(), "phone": set()}
    for row in rows:
        email = normalize_email(row.get("contact_email"))
        if email:
            values["email"].add(email)
        phone = _normalize_optional_identity("phone", row.get("contact_phone"))
        if phone:
            values["phone"].add(phone)
    return values


def _load_prod_snapshot(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    emails: set[str],
    phones: set[str],
    message_shas: set[str],
    tallanto_ids: set[str],
) -> Mapping[str, Any]:
    direct_links = _load_direct_identity_links(con, tenant_id=tenant_id, emails=emails, phones=phones)
    tallanto_to_customers = _load_tallanto_customer_links(con, tenant_id=tenant_id, tallanto_ids=tallanto_ids)
    prod_duplicate_keys_by_sha = _load_prod_duplicate_keys(con, tenant_id=tenant_id, message_shas=message_shas)
    customer_ids = {
        item["customer_id"]
        for links_by_value in direct_links.values()
        for links in links_by_value.values()
        for item in links
        if item.get("customer_id")
    }
    for ids in tallanto_to_customers.values():
        customer_ids.update(ids)
    customer_payload_by_id = _load_customer_payloads(con, tenant_id=tenant_id, customer_ids=customer_ids)
    qualified_by_customer = {
        customer_id: _customer_qualification(con, tenant_id=tenant_id, customer_id=customer_id)
        for customer_id in customer_payload_by_id
    }
    brand_profile_by_customer = _load_customer_brand_profiles(con, tenant_id=tenant_id, customer_ids=customer_ids)
    return {
        "direct_links": direct_links,
        "tallanto_to_customers": tallanto_to_customers,
        "customer_payload_by_id": customer_payload_by_id,
        "qualified_by_customer": qualified_by_customer,
        "brand_profile_by_customer": brand_profile_by_customer,
        "prod_duplicate_keys_by_sha": prod_duplicate_keys_by_sha,
        "existing_email_values": set(emails) & set(direct_links.get("email", {})),
        "summary": {
            "direct_email_values_requested": len(emails),
            "direct_phone_values_requested": len(phones),
            "direct_email_values_found": len(direct_links.get("email", {})),
            "direct_phone_values_found": len(direct_links.get("phone", {})),
            "tallanto_customer_links": len(tallanto_to_customers),
            "prod_duplicate_messages": len(prod_duplicate_keys_by_sha),
            "customer_payloads_loaded": len(customer_payload_by_id),
        },
    }


def _load_customer_brand_profiles(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: set[str],
) -> Mapping[str, Mapping[str, Any]]:
    if not customer_ids:
        return {}
    profiles: dict[str, Mapping[str, Any]] = {}
    for customer_id in sorted(customer_ids):
        counts: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        opportunity_rows = con.execute(
            """
            SELECT title, record_json
            FROM customer_opportunities
            WHERE tenant_id = ? AND customer_id = ?
            """,
            (tenant_id, customer_id),
        ).fetchall()
        for row in opportunity_rows:
            payload = _safe_json_object(row["record_json"])
            product_context = payload.get("product_context") if isinstance(payload.get("product_context"), Mapping) else {}
            brand = _normalize_known_brand(product_context.get("brand"))
            if brand == "unknown":
                brand = _infer_brand_from_mapping({"title": row["title"], "product_context": product_context})
            if brand in {"foton", "unpk"}:
                counts[brand] += 3
                sources[f"opportunity:{brand}"] += 1
        event_rows = con.execute(
            """
            SELECT subject, text_preview, summary, record_json
            FROM timeline_events
            WHERE tenant_id = ? AND customer_id = ?
            """,
            (tenant_id, customer_id),
        ).fetchall()
        for row in event_rows:
            payload = _safe_json_object(row["record_json"])
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
            brand = _normalize_known_brand(metadata.get("brand") or record.get("brand"))
            source = "event_explicit_brand"
            if brand == "unknown":
                brand = _infer_brand_from_mapping(
                    {
                        "subject": row["subject"],
                        "text_preview": row["text_preview"],
                        "summary": row["summary"],
                        "record_summary": record.get("summary"),
                        "full_clean_text": record.get("full_clean_text"),
                    }
                )
                source = "event_content_infer"
            if brand in {"foton", "unpk"}:
                counts[brand] += 1
                sources[f"{source}:{brand}"] += 1
        known = {brand: count for brand, count in counts.items() if brand in {"foton", "unpk"} and count > 0}
        if not known:
            profiles[customer_id] = {
                "brand": "unknown",
                "source": "customer_history",
                "reason": "no_known_brand_in_history",
                "counts": dict(counts),
            }
            continue
        if len(known) > 1:
            profiles[customer_id] = {
                "brand": "unknown",
                "source": "customer_history",
                "reason": "mixed_brand_history",
                "counts": dict(counts),
                "sources": dict(sources),
            }
            continue
        brand = next(iter(known))
        profiles[customer_id] = {
            "brand": brand,
            "source": "customer_history",
            "reason": "single_known_brand_in_history",
            "counts": dict(counts),
            "sources": dict(sources),
        }
    return profiles


def _safe_json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _infer_brand_from_mapping(values: Mapping[str, Any]) -> str:
    return _normalize_known_brand(infer_offline_brand(values))


def _normalize_known_brand(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "мфти"}:
        return "unpk"
    return "unknown"


def _load_direct_identity_links(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    emails: set[str],
    phones: set[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {"email": defaultdict(list), "phone": defaultdict(list)}
    for link_type, values in (("email", emails), ("phone", phones)):
        if not values:
            continue
        placeholders = ",".join("?" for _ in values)
        for row in con.execute(
            f"""
            SELECT link_value, customer_id, match_class, record_json
            FROM identity_links
            WHERE tenant_id = ? AND link_type = ? AND link_value IN ({placeholders})
            """,
            (tenant_id, link_type, *sorted(values)),
        ):
            result[link_type][str(row["link_value"])].append(
                {
                    "customer_id": row["customer_id"],
                    "match_class": row["match_class"],
                    "record": json.loads(row["record_json"]),
                }
            )
    return result


def _load_tallanto_customer_links(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    tallanto_ids: set[str],
) -> dict[str, set[str]]:
    if not tallanto_ids:
        return {}
    placeholders = ",".join("?" for _ in tallanto_ids)
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in con.execute(
        f"""
        SELECT link_value, customer_id
        FROM identity_links
        WHERE tenant_id = ?
          AND link_type = 'tallanto_student_id'
          AND link_value IN ({placeholders})
          AND customer_id IS NOT NULL AND customer_id != ''
        """,
        (tenant_id, *sorted(tallanto_ids)),
    ):
        grouped[str(row["link_value"])].add(str(row["customer_id"]))
    return grouped


def _load_prod_duplicate_keys(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    message_shas: set[str],
) -> dict[str, set[str]]:
    if not message_shas:
        return {}
    placeholders = ",".join("?" for _ in message_shas)
    system_placeholders = ",".join("?" for _ in A2V3_DEDUPE_SOURCE_SYSTEMS)
    result: dict[str, set[str]] = defaultdict(set)
    for row in con.execute(
        f"""
        SELECT source_id, dedupe_key
        FROM timeline_events
        WHERE tenant_id = ?
          AND event_type = 'email_message'
          AND source_system IN ({system_placeholders})
          AND source_id IN ({placeholders})
        """,
        (tenant_id, *A2V3_DEDUPE_SOURCE_SYSTEMS, *sorted(message_shas)),
    ):
        result[str(row["source_id"])].add(str(row["dedupe_key"]))
    return result


def _load_customer_payloads(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    if not customer_ids:
        return {}
    placeholders = ",".join("?" for _ in customer_ids)
    payloads: dict[str, Mapping[str, Any]] = {}
    for row in con.execute(
        f"""
        SELECT customer_id, record_json
        FROM customer_identities
        WHERE tenant_id = ? AND customer_id IN ({placeholders})
        """,
        (tenant_id, *sorted(customer_ids)),
    ):
        payloads[str(row["customer_id"])] = json.loads(row["record_json"])
    return payloads


def _customer_qualification(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> Mapping[str, Any]:
    directions = {
        str(row["direction"]): int(row["c"])
        for row in con.execute(
            """
            SELECT direction, count(*) AS c
            FROM timeline_events
            WHERE tenant_id = ? AND customer_id = ?
            GROUP BY direction
            """,
            (tenant_id, customer_id),
        )
    }
    stages = [
        str(row["stage_after"])
        for row in con.execute(
            """
            SELECT DISTINCT json_extract(record_json, '$.stage_after') AS stage_after
            FROM timeline_events
            WHERE tenant_id = ? AND customer_id = ? AND event_type = 'amo_deal_stage'
            """,
            (tenant_id, customer_id),
        )
        if row["stage_after"]
    ]
    has_allow_stage = any(stage in QUALIFYING_AMO_STAGES for stage in stages)
    has_two_way = bool(directions.get("inbound", 0) > 0 and directions.get("outbound", 0) > 0)
    return {
        "qualified": has_allow_stage or has_two_way,
        "has_allow_stage": has_allow_stage,
        "has_two_way": has_two_way,
        "directions": directions,
        "stages": stages[:20],
    }


def _load_tallanto_identity_matches(
    identity_db: Optional[Path],
    *,
    emails: set[str],
    phones: set[str],
) -> Mapping[str, Any]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {"email": {}, "phone": {}}
    if identity_db is None or not Path(identity_db).exists():
        return {"matches": result, "summary": {"identity_db_present": False}}
    with sqlite3.connect(f"file:{Path(identity_db)}?mode=ro", uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        for kind, values in (("email", emails), ("phone", phones)):
            for value in sorted(values):
                row = con.execute(
                    """
                    SELECT match_class, candidate_count
                    FROM identity_values
                    WHERE kind = ? AND value = ?
                    """,
                    (kind, value),
                ).fetchone()
                if row is None:
                    continue
                candidates = con.execute(
                    """
                    SELECT c.tallanto_id
                    FROM identity_links l
                    JOIN identity_candidates c ON c.candidate_key = l.candidate_key
                    WHERE l.kind = ? AND l.value = ?
                    """,
                    (kind, value),
                ).fetchall()
                tallanto_ids = sorted({str(item["tallanto_id"]) for item in candidates if item["tallanto_id"]})
                result[kind][value] = {
                    "match_class": str(row["match_class"]),
                    "candidate_count": int(row["candidate_count"]),
                    "tallanto_ids": tallanto_ids,
                }
    return {
        "matches": result,
        "summary": {
            "identity_db_present": True,
            "email_values_found": len(result["email"]),
            "phone_values_found": len(result["phone"]),
        },
    }


def _tallanto_ids_from_matches(tallanto_matches: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for matches_by_value in (tallanto_matches.get("matches") or {}).values():
        for match in matches_by_value.values():
            values.update(str(item) for item in (match.get("tallanto_ids") or ()) if item)
    return values


def _resolve_customer(
    row: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    tallanto_matches: Mapping[str, Any],
) -> CustomerResolution:
    if bool(row.get("contact_ambiguous")):
        return CustomerResolution("blocked", "contact_ambiguous", blocked=True, ambiguous=True)
    email = normalize_email(row.get("contact_email"))
    phone = _normalize_optional_identity("phone", row.get("contact_phone"))
    for kind, value in (("email", email), ("phone", phone)):
        if not value:
            continue
        direct = snapshot["direct_links"].get(kind, {}).get(value, [])
        if direct:
            return _resolution_from_direct_links(kind, value, direct, snapshot=snapshot)
    for kind, value in (("email", email), ("phone", phone)):
        if not value:
            continue
        match = tallanto_matches["matches"].get(kind, {}).get(value)
        if match:
            resolution = _resolution_from_tallanto_match(kind, value, match, snapshot=snapshot)
            if resolution.outcome != "unmatched":
                return resolution
    if email or phone:
        return CustomerResolution("unmatched", "no_customer_match_for_contact")
    return CustomerResolution("unmatched", "contact_missing")


def _resolution_from_direct_links(
    kind: str,
    value: str,
    links: Sequence[Mapping[str, Any]],
    *,
    snapshot: Mapping[str, Any],
) -> CustomerResolution:
    customer_ids = {str(item.get("customer_id")) for item in links if item.get("customer_id")}
    if len(customer_ids) != 1:
        return CustomerResolution("blocked", f"direct_{kind}_multiple_customers", blocked=True, ambiguous=True)
    customer_id = next(iter(customer_ids))
    if any(str(item.get("match_class")) == "ambiguous" for item in links):
        return CustomerResolution("blocked", f"direct_{kind}_ambiguous_link", blocked=True, ambiguous=True)
    if _customer_is_ambiguous(customer_id, snapshot):
        return CustomerResolution("blocked", "customer_identity_ambiguous", customer_id=customer_id, blocked=True, ambiguous=True)
    return CustomerResolution("linked", f"direct_{kind}_identity_link", customer_id=customer_id, method=f"direct_{kind}")


def _resolution_from_tallanto_match(
    kind: str,
    value: str,
    match: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
) -> CustomerResolution:
    if match.get("match_class") != "strong_unique" or int(match.get("candidate_count") or 0) != 1:
        return CustomerResolution("blocked", f"tallanto_{kind}_not_strong_unique", blocked=True, ambiguous=True)
    tallanto_ids = tuple(str(item) for item in match.get("tallanto_ids") or () if item)
    if len(tallanto_ids) != 1:
        return CustomerResolution("blocked", f"tallanto_{kind}_multiple_candidates", blocked=True, ambiguous=True)
    tallanto_id = tallanto_ids[0]
    customers = snapshot["tallanto_to_customers"].get(tallanto_id, set())
    if len(customers) != 1:
        return CustomerResolution("blocked", "tallanto_id_not_unique_in_timeline", tallanto_id=tallanto_id, blocked=True, ambiguous=True)
    customer_id = next(iter(customers))
    if _customer_is_ambiguous(customer_id, snapshot):
        return CustomerResolution("blocked", "customer_identity_ambiguous", customer_id=customer_id, tallanto_id=tallanto_id, blocked=True, ambiguous=True)
    return CustomerResolution(
        "linked",
        f"tallanto_{kind}_strong_unique",
        customer_id=customer_id,
        method=f"tallanto_{kind}",
        tallanto_id=tallanto_id,
        link_email_value=value if kind == "email" else None,
    )


def _customer_is_ambiguous(customer_id: str, snapshot: Mapping[str, Any]) -> bool:
    payload = snapshot["customer_payload_by_id"].get(customer_id) or {}
    return str(payload.get("identity_status") or "").lower() == "ambiguous"


def _customer_for_resolution(
    resolution: CustomerResolution,
    snapshot: Mapping[str, Any],
) -> Optional[CustomerIdentity]:
    if resolution.outcome != "linked" or not resolution.customer_id:
        return None
    payload = snapshot["customer_payload_by_id"].get(resolution.customer_id)
    if not payload:
        return None
    return customer_identity_from_json(payload)


def _identity_links_for_row(
    row: Mapping[str, Any],
    *,
    tenant_id: str,
    resolution: CustomerResolution,
    source_ref: str,
    event_at: datetime,
    existing_email_values: set[str],
) -> tuple[IdentityLink, ...]:
    if resolution.outcome != "linked" or not resolution.customer_id or not resolution.link_email_value:
        return ()
    email = normalize_email(resolution.link_email_value)
    if not email or email in existing_email_values:
        return ()
    stable_source_ref = f"a2v3_contact_email:{_sha16(email)}"
    return (
        IdentityLink(
            tenant_id=tenant_id,
            customer_id=resolution.customer_id,
            link_type="email",
            link_value=email,
            source_system=A2V3_MAIL_SOURCE_SYSTEM,
            source_ref=stable_source_ref,
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.9,
            evidence={
                "source": "a2v3_contact_email_tallanto_bridge",
                "message_sha256": _message_sha(row),
                "tallanto_id_hash": _sha16(resolution.tallanto_id or ""),
            },
            first_seen_at=event_at,
            last_seen_at=event_at,
        ),
    )


def _bot_visibility(
    row: Mapping[str, Any],
    *,
    resolution: CustomerResolution,
    qualified: bool,
    customer_brand: str,
) -> tuple[bool, str]:
    status = str(_quality(row).get("memory_status") or "")
    if status != "usable_memory":
        return False, f"memory_status_{status or 'missing'}"
    if resolution.outcome != "linked" or not resolution.customer_id:
        return False, f"identity_{resolution.outcome}"
    if resolution.ambiguous or resolution.blocked:
        return False, "identity_ambiguous_or_blocked"
    if customer_brand not in {"foton", "unpk"}:
        return False, "customer_brand_unknown"
    if not qualified:
        return False, "customer_not_qualified"
    return True, "usable_linked_qualified"


def _event_record(
    row: Mapping[str, Any],
    *,
    resolution: CustomerResolution,
    qualified: bool,
    bot_eligible_candidate: bool,
    bot_gate_reason: str,
    customer_brand: str,
    customer_brand_source: str,
    customer_brand_reason: str,
) -> Mapping[str, Any]:
    payload = row.get("summary_payload") or {}
    quality = _quality(row)
    return {
        "source": "A2v3_email_pipeline",
        "message_sha256": _message_sha(row),
        "thread_id": _thread_id(row),
        "thread_basis": quality.get("thread_basis"),
        "brand": _brand(row),
        "brand_source": row.get("brand_source") or row.get("raw_infer_offline_brand"),
        "customer_brand": customer_brand,
        "customer_brand_source": customer_brand_source,
        "customer_brand_reason": customer_brand_reason,
        "direction": row.get("direction"),
        "subject": row.get("subject_full"),
        "full_clean_text": row.get("full_clean_text"),
        "summary": payload.get("summary"),
        "topic": payload.get("topic"),
        "next_step_model": payload.get("next_step"),
        "safe_next_step_note": quality.get("safe_next_step_note"),
        "event_type_detail": payload.get("event_type"),
        "money_direction": payload.get("money_direction"),
        "amount_kind": payload.get("amount_kind"),
        "amount_rub": payload.get("amount_rub"),
        "money_amounts_rub": quality.get("money_amounts_rub"),
        "amount_uncertain": quality.get("amount_uncertain"),
        "student_name": payload.get("student_name"),
        "grade": payload.get("grade"),
        "subject_area": payload.get("subject_area"),
        "contact_email": row.get("contact_email"),
        "contact_phone": row.get("contact_phone"),
        "contact_name": row.get("contact_name"),
        "quality": quality,
        "identity_resolution": {
            "outcome": resolution.outcome,
            "reason": resolution.reason,
            "method": resolution.method,
            "customer_id": resolution.customer_id,
            "blocked": resolution.blocked,
            "ambiguous": resolution.ambiguous,
        },
        "bot_eligible_candidate": {
            "qualified": qualified,
            "eligible": bot_eligible_candidate,
            "reason": bot_gate_reason,
        },
    }


def _detail_event_type(row: Mapping[str, Any]) -> str:
    payload = row.get("summary_payload") or {}
    value = str(payload.get("event_type") or "other").strip().lower()
    return value or "other"


def _resolve_customer_brand(
    row: Mapping[str, Any],
    *,
    resolution: CustomerResolution,
    snapshot: Mapping[str, Any],
) -> tuple[str, str, str]:
    if resolution.outcome == "linked" and resolution.customer_id:
        profile = (snapshot.get("brand_profile_by_customer") or {}).get(resolution.customer_id) or {}
        brand = _normalize_known_brand(profile.get("brand"))
        if brand in {"foton", "unpk"}:
            return brand, str(profile.get("source") or "customer_history"), str(profile.get("reason") or "resolved_from_history")
        reason = str(profile.get("reason") or "no_known_brand_in_history")
        email_brand = _brand(row)
        if email_brand in {"foton", "unpk"} and reason != "mixed_brand_history":
            return email_brand, "a2v3_email_content", "history_unknown_email_content_known"
        return "unknown", str(profile.get("source") or "customer_history"), reason
    email_brand = _brand(row)
    if email_brand in {"foton", "unpk"}:
        return email_brand, "a2v3_email_content_unlinked", "unlinked_email_content_known"
    return "unknown", "unresolved_identity", resolution.reason


def _match_status_for_resolution(resolution: CustomerResolution) -> IdentityMatchClass:
    if resolution.outcome == "linked":
        return IdentityMatchClass.STRONG_UNIQUE
    if resolution.ambiguous or resolution.blocked:
        return IdentityMatchClass.AMBIGUOUS
    return IdentityMatchClass.UNMATCHED


def _quality(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("quality") or {}


def _message_sha(row: Mapping[str, Any]) -> str:
    value = str(row.get("message_sha256") or "").strip().lower()
    if not value:
        raise ValueError("A2v3 row missing message_sha256")
    return value


def _summary_text(row: Mapping[str, Any]) -> str:
    payload = row.get("summary_payload") or {}
    return str(payload.get("summary") or row.get("summary") or row.get("subject_full") or "")


def _chunk_text(row: Mapping[str, Any]) -> str:
    payload = row.get("summary_payload") or {}
    parts = [
        str(payload.get("summary") or "").strip(),
        f"Тема: {payload.get('topic')}" if payload.get("topic") else "",
        f"Безопасная заметка: {_quality(row).get('safe_next_step_note')}" if _quality(row).get("safe_next_step_note") else "",
    ]
    return compact_text("\n".join(part for part in parts if part), limit=1200)


def _thread_id(row: Mapping[str, Any]) -> str:
    return str(_quality(row).get("thread_id") or _message_sha(row))


def _brand(row: Mapping[str, Any]) -> str:
    return _normalize_known_brand(row.get("brand"))


def _direction(row: Mapping[str, Any]) -> TimelineDirection:
    value = str(row.get("direction") or "").strip().lower()
    if value == "outbound":
        return TimelineDirection.OUTBOUND
    if value == "system":
        return TimelineDirection.SYSTEM
    return TimelineDirection.INBOUND


def _relevance_tags(row: Mapping[str, Any], *, bot_eligible_candidate: bool) -> tuple[str, ...]:
    payload = row.get("summary_payload") or {}
    tags = ["email", _brand(row), str(_quality(row).get("memory_status") or "unknown")]
    if payload.get("event_type"):
        tags.append(str(payload["event_type"]))
    tags.append("bot_eligible_candidate" if bot_eligible_candidate else "manager_review")
    return tuple(tag for tag in tags if tag)


def _normalize_optional_identity(kind: str, value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return normalize_identity_value(kind, value)
    except ValueError:
        return ""


def _input_fingerprint(config: A2V3MailIngestConfig) -> str:
    payload = {
        "schema_version": A2V3_MAIL_INGEST_SCHEMA_VERSION,
        "input_jsonl": file_sha256(config.input_jsonl),
        "tenant_id": config.tenant_id,
        "source_ref": config.source_ref,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_backup_manifest(config: A2V3MailIngestConfig, manifest_path: Path) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "a2v3_test_db_backup":
        raise ValueError("backup manifest kind mismatch")
    if Path(str(manifest.get("test_db_path"))).resolve(strict=False) != config.timeline_db_path.resolve(strict=False):
        raise ValueError("backup manifest test DB mismatch")
    backup_db = Path(str(manifest.get("backup_db_path")))
    if not backup_db.exists():
        raise FileNotFoundError(f"backup DB not found: {backup_db}")
    if file_sha256(backup_db) != manifest.get("backup_sha256"):
        raise ValueError("backup sha256 mismatch")


def _customer_history_summary(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
) -> Mapping[str, Any]:
    events = _customer_history_events(con, tenant_id=tenant_id, customer_id=customer_id)
    counts = {
        str(row["event_type"]): int(row["c"])
        for row in con.execute(
            """
            SELECT event_type, count(*) AS c
            FROM timeline_events
            WHERE tenant_id = ? AND customer_id = ?
            GROUP BY event_type
            """,
            (tenant_id, customer_id),
        )
    }
    return {
        "event_count": sum(counts.values()),
        "event_type_counts": counts,
        "last_items": [_timeline_review_item(event, source="existing_prod") for event in events[:3]],
        "events": events,
    }


def _empty_customer_history_summary() -> Mapping[str, Any]:
    return {"event_count": 0, "event_type_counts": {}, "last_items": [], "events": []}


def _customer_history_events(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
) -> list[Mapping[str, Any]]:
    rows = con.execute(
        """
        SELECT
          event_id,
          customer_id,
          event_at,
          event_type,
          source_system,
          source_id,
          source_ref,
          direction,
          match_status,
          subject,
          text_preview,
          summary,
          record_json
        FROM timeline_events
        WHERE tenant_id = ? AND customer_id = ?
        ORDER BY event_at DESC, event_id DESC
        """,
        (tenant_id, customer_id),
    ).fetchall()
    events: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            payload = dict(json.loads(row["record_json"] or "{}"))
        except json.JSONDecodeError:
            payload = {}
        for key in (
            "event_id",
            "customer_id",
            "event_at",
            "event_type",
            "source_system",
            "source_id",
            "source_ref",
            "direction",
            "match_status",
            "subject",
            "text_preview",
            "summary",
        ):
            if payload.get(key) in (None, ""):
                payload[key] = row[key]
        events.append(payload)
    return events


def _combined_timeline(
    existing_events: Sequence[Mapping[str, Any]],
    new_plans: Sequence[PlannedA2V3MailEvent],
    *,
    fact_by_sha: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    # New email items are intentionally shown first: Dmitry reviews what would be added over current memory.
    new_items = [
        _timeline_review_item(_new_event_for_review(plan, fact_by_sha=fact_by_sha), source="new_a2v3_email")
        for plan in sorted(new_plans, key=lambda item: str(item.event.event_at.isoformat()), reverse=True)
    ]
    existing_items = [_timeline_review_item(event, source="existing_prod") for event in existing_events]
    return [*new_items, *existing_items]


def _review_next_step(
    existing_events: Sequence[Mapping[str, Any]],
    new_plans: Sequence[PlannedA2V3MailEvent],
    *,
    customer_id: Optional[str],
) -> Mapping[str, Any]:
    if not customer_id:
        return {
            "status": "not_applicable",
            "display_text": "Нет привязанного клиента",
            "reason_code": "unresolved_identity",
        }
    events = [dict(event) for event in existing_events]
    events.extend(_new_event_for_review(plan) for plan in new_plans)
    return dict(
        resolve_customer_next_step(
            events,
            readiness={"open_conflicts": 0},
            customer_id=customer_id,
        ).to_json_dict()
    )


def _new_event_for_review(
    plan: PlannedA2V3MailEvent,
    *,
    fact_by_sha: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    event = dict(plan.event.to_json_dict())
    record = dict(event.get("record") or {})
    fact = (fact_by_sha or {}).get(_message_sha(plan.row)) or {}
    record["safe_next_step_note"] = _quality(plan.row).get("safe_next_step_note")
    record["next_step_model"] = (plan.row.get("summary_payload") or {}).get("next_step")
    record["memory_status"] = fact.get("memory_status") or _quality(plan.row).get("memory_status")
    record["event_type_detail"] = fact.get("event_type_detail") or record.get("event_type_detail")
    record["amount_rub"] = fact.get("amount_rub") if fact.get("amount_rub") is not None else record.get("amount_rub")
    record["amount_kind"] = fact.get("amount_kind") or record.get("amount_kind")
    record["money_amounts_rub"] = fact.get("money_amounts_rub") or record.get("money_amounts_rub")
    record["brand"] = fact.get("email_brand") or record.get("brand")
    record["customer_brand"] = fact.get("customer_brand") or record.get("customer_brand")
    record["customer_brand_source"] = fact.get("customer_brand_source") or record.get("customer_brand_source")
    record["customer_brand_reason"] = fact.get("customer_brand_reason") or record.get("customer_brand_reason")
    record["contact_email"] = fact.get("contact_email") or record.get("contact_email")
    record["contact_phone"] = fact.get("contact_phone") or record.get("contact_phone")
    record["student_name"] = fact.get("student_name") or record.get("student_name")
    record["grade"] = fact.get("grade") or record.get("grade")
    record["subject_area"] = fact.get("subject_area") or record.get("subject_area")
    record["identity_resolution"] = {
        "outcome": fact.get("identity_outcome") or plan.resolution.outcome,
        "reason": fact.get("identity_reason") or plan.resolution.reason,
        "method": plan.resolution.method,
        "customer_id": plan.resolution.customer_id,
        "blocked": plan.resolution.blocked,
        "ambiguous": plan.resolution.ambiguous,
    }
    record["bot_eligible_candidate"] = {
        "eligible": bool(fact.get("bot_visible")) if fact else plan.bot_eligible_candidate,
        "qualified": (plan.resolution.outcome == "linked"),
        "reason": fact.get("bot_gate_reason") or plan.bot_gate_reason,
    }
    event["record"] = record
    return event


def _timeline_review_item(event: Mapping[str, Any], *, source: str) -> Mapping[str, Any]:
    record = event.get("record") or {}
    if not isinstance(record, Mapping):
        record = {}
    metadata = event.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    bot_eligible = record.get("bot_eligible_candidate") or {}
    if not isinstance(bot_eligible, Mapping):
        bot_eligible = {}
    identity_resolution = record.get("identity_resolution") or {}
    if not isinstance(identity_resolution, Mapping):
        identity_resolution = {}
    return {
        "source": source,
        "event_id": event.get("event_id") or event.get("source_id"),
        "event_at": event.get("event_at"),
        "event_type": event.get("event_type"),
        "event_type_detail": record.get("event_type_detail"),
        "source_system": event.get("source_system"),
        "direction": event.get("direction"),
        "subject": compact_text(event.get("subject"), limit=160),
        "summary": compact_text(event.get("summary") or record.get("summary"), limit=260),
        "email_brand": record.get("brand"),
        "customer_brand": record.get("customer_brand") or (metadata.get("brand") if metadata else None),
        "customer_brand_source": record.get("customer_brand_source") or (metadata.get("brand_source") if metadata else None),
        "customer_brand_reason": record.get("customer_brand_reason") or (metadata.get("brand_reason") if metadata else None),
        "safe_next_step_note": compact_text(record.get("safe_next_step_note"), limit=220),
        "amount_rub": record.get("amount_rub"),
        "amount_kind": record.get("amount_kind"),
        "memory_status": record.get("memory_status") or (metadata.get("memory_status") if metadata else None),
        "bot_eligible_candidate": bot_eligible.get("eligible"),
        "resolution": identity_resolution.get("outcome"),
        "resolution_reason": identity_resolution.get("reason"),
    }


def _review_contact(plans: Sequence[PlannedA2V3MailEvent]) -> Mapping[str, Any]:
    for plan in plans:
        row = plan.row
        return {
            "email": row.get("contact_email"),
            "phone": row.get("contact_phone"),
            "name": row.get("contact_name"),
        }
    return {}


def _review_new_event(
    plan: PlannedA2V3MailEvent,
    *,
    fact_by_sha: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    row = plan.row
    payload = row.get("summary_payload") or {}
    fact = fact_by_sha.get(_message_sha(row)) or {}
    return {
        "message_sha256": _message_sha(row),
        "date_iso": row.get("date_iso"),
        "event_type": fact.get("event_type_detail") or payload.get("event_type"),
        "summary": payload.get("summary"),
        "safe_next_step_note": _quality(row).get("safe_next_step_note"),
        "amount_rub": fact.get("amount_rub") if fact.get("amount_rub") is not None else payload.get("amount_rub"),
        "amount_kind": fact.get("amount_kind") or payload.get("amount_kind"),
        "money_amounts_rub": fact.get("money_amounts_rub") or _quality(row).get("money_amounts_rub"),
        "memory_status": fact.get("memory_status") or _quality(row).get("memory_status"),
        "email_brand": fact.get("email_brand") or _brand(row),
        "customer_brand": fact.get("customer_brand") or plan.customer_brand,
        "customer_brand_source": fact.get("customer_brand_source") or plan.customer_brand_source,
        "customer_brand_reason": fact.get("customer_brand_reason") or plan.customer_brand_reason,
        "bot_eligible_candidate": bool(fact.get("bot_visible")) if fact else plan.bot_eligible_candidate,
        "bot_gate_reason": fact.get("bot_gate_reason") or plan.bot_gate_reason,
        "resolution": fact.get("identity_outcome") or plan.resolution.outcome,
        "resolution_reason": fact.get("identity_reason") or plan.resolution.reason,
    }


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
