from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.existing_clients.amo_step1_snapshot import (
    AmoMcpClient,
    AmoMcpError,
    embedded_items,
    read_mcp_env,
)
from mango_mvp.customer_timeline.ids import normalize_email, stable_digest
from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    completed_import_source_names,
    run_nightly_incremental,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_writable_path
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.utils.phone import normalize_phone


AMO_INCREMENTAL_SCHEMA_VERSION = "customer_timeline_amo_incremental_v1"
AMO_EVENT_TYPES = frozenset(
    {
        "incoming_chat_message",
        "outgoing_chat_message",
        "common_note_added",
        "incoming_mail",
        "outgoing_mail",
    }
)
DEFAULT_SOURCE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/"
    "customer_timeline_prod_20260621/customer_timeline.sqlite"
)


@dataclass(frozen=True)
class AmoIncrementalConfig:
    source_db: Path
    out_root: Path
    mcp_env: Path
    timeline_db: Optional[Path] = None
    mcp_transport: Optional[str] = None
    allowed_root: Optional[Path] = None
    tenant_id: str = "foton"
    safety_overlap_seconds: int = 300
    page_limit: int = 20
    max_pages: int = 2
    sleep_sec: float = 1.05
    since: Optional[datetime] = None
    copy_db: bool = True


# D1: bounded checkpoint continuation for page_cap_hit. A single run only
# reads up to `max_pages` pages per endpoint; when an endpoint is not fully
# read yet, its accumulated raw items + resume page are persisted here
# (never in the timeline DB) so the *next* run continues from where this one
# stopped instead of re-reading from page 1. `page_cap_hit=true` must never
# advance the final ingestion cursor and must never be reported as "ok" --
# see the `all_complete` gate in run_amo_incremental below, which is
# unchanged in spirit from the original all-or-nothing gate, just now fed by
# resumable per-endpoint fetches instead of always starting at page 1.
AMO_INCREMENTAL_CHECKPOINT_SCHEMA_VERSION = "customer_timeline_amo_incremental_checkpoint_v1"


def _checkpoint_path(out_root: Path) -> Path:
    return out_root / "amo_incremental_checkpoint.json"


def load_amo_incremental_checkpoint(out_root: Path) -> Mapping[str, Any]:
    path = _checkpoint_path(out_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    if payload.get("schema_version") != AMO_INCREMENTAL_CHECKPOINT_SCHEMA_VERSION:
        return {}
    return payload


def save_amo_incremental_checkpoint(out_root: Path, endpoints: Mapping[str, Any]) -> None:
    path = _checkpoint_path(out_root)
    if not endpoints:
        if path.exists():
            path.unlink()
        return
    _atomic_write_text(
        path,
        json.dumps(
            {"schema_version": AMO_INCREMENTAL_CHECKPOINT_SCHEMA_VERSION, "endpoints": dict(endpoints)},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def universe_fingerprint(
    *,
    path: str,
    lower_bound: datetime,
    params: Optional[Mapping[str, Any]] = None,
    page_limit: Optional[int] = None,
) -> str:
    # D1: freezes the paginated "universe" (endpoint + lower_bound) at the
    # start of a checkpoint cycle. A repeat call with a different lower_bound
    # (the cursor advanced, e.g. from a prior fully-completed cycle) yields a
    # different fingerprint, which _resume_state treats as "no usable
    # checkpoint" -- an incomplete checkpoint from a stale window is never
    # silently resumed into a different one.
    return stable_digest(
        {
            "path": path,
            "lower_bound": lower_bound.isoformat(),
            "params": dict(params or {}),
            "page_limit": page_limit,
        }
    )


def page_anchor(items: Sequence[Mapping[str, Any]]) -> str:
    return stable_digest(list(items))


def _checkpoint_entry(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    endpoints = checkpoint.get("endpoints")
    entry = endpoints.get(key) if isinstance(endpoints, Mapping) else None
    return entry if isinstance(entry, Mapping) else {}


def _resume_state(
    checkpoint: Mapping[str, Any], *, key: str, fingerprint: str
) -> tuple[int, list[Mapping[str, Any]], int, bool]:
    endpoints = checkpoint.get("endpoints")
    entry = endpoints.get(key) if isinstance(endpoints, Mapping) else None
    if not isinstance(entry, Mapping) or entry.get("fingerprint") != fingerprint:
        return 1, [], 0, False
    start_page = max(1, int(entry.get("next_page") or 1))
    items = [item for item in (entry.get("items") or ()) if isinstance(item, Mapping)]
    pages_fetched = max(0, int(entry.get("pages_fetched") or 0))
    return start_page, items, pages_fetched, bool(entry.get("complete"))


def fetch_endpoint_checkpointed(
    client: AmoMcpClient,
    *,
    key: str,
    path: str,
    embedded_key: str,
    params: Mapping[str, Any],
    lower_bound: datetime,
    config: AmoIncrementalConfig,
    checkpoint: Mapping[str, Any],
    next_checkpoint: dict[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """D1: bounded per-run pagination that resumes from a persisted
    start_page/accumulated-items checkpoint instead of re-reading from page 1
    every run. `complete` is True only once fetch_collection finishes this
    endpoint's pagination with no page_cap_hit; run_amo_incremental only
    treats the whole cycle as ok (and only then advances the cursor) once
    every endpoint reports complete=True -- a partial pass here always saves
    a checkpoint and never confirms freshness.
    """
    fingerprint = universe_fingerprint(
        path=path,
        lower_bound=lower_bound,
        params=params,
        page_limit=config.page_limit,
    )
    entry = _checkpoint_entry(checkpoint, key)
    start_page, carried_items, pages_before, already_complete = _resume_state(
        checkpoint, key=key, fingerprint=fingerprint
    )
    upper_bound = parse_iso(str(entry.get("upper_bound"))) if start_page > 1 and entry.get("upper_bound") else datetime.now(timezone.utc)
    timestamp_field = "created_at" if path == "events" else "updated_at"
    effective_params = {
        **dict(params),
        f"filter[{timestamp_field}][to]": int(upper_bound.timestamp()),
    }
    if already_complete:
        next_checkpoint[key] = dict(entry)
        return carried_items, {
            "pages": pages_before,
            "pages_this_run": 0,
            "start_page_this_run": start_page,
            "max_pages": max(1, int(config.max_pages)),
            "page_cap_hit": False,
            "complete": True,
            "fetched": len(carried_items),
            "fetched_this_run": 0,
            "carried_over_from_checkpoint": len(carried_items),
            "upper_bound": upper_bound.isoformat(),
            "checkpoint_reset_reason": None,
            "pagination_drift_detected": False,
        }
    checkpoint_reset_reason: Optional[str] = None
    if start_page > 1:
        anchor_page = int(entry.get("last_page") or 0)
        expected_anchor = str(entry.get("last_page_anchor") or "")
        if anchor_page < 1 or not expected_anchor:
            current_anchor = ""
        else:
            current_payload = _fetch_collection_page(
                client,
                path=path,
                params=effective_params,
                page=anchor_page,
                config=config,
            )
            current_anchor = page_anchor(embedded_items(current_payload, embedded_key))
        if current_anchor != expected_anchor:
            start_page, carried_items, pages_before = 1, [], 0
            upper_bound = datetime.now(timezone.utc)
            effective_params[f"filter[{timestamp_field}][to]"] = int(upper_bound.timestamp())
            checkpoint_reset_reason = "pagination_universe_changed"
    page_snapshots: dict[int, list[Mapping[str, Any]]] = {}
    if path == "events":
        batch_items, batch_pages, page_cap_hit = fetch_events_collection(
            client,
            from_ts=lower_bound,
            config=config,
            start_page=start_page,
            params_override=effective_params,
            page_snapshots=page_snapshots,
        )
    else:
        batch_items, batch_pages, page_cap_hit = fetch_collection(
            client,
            path=path,
            embedded_key=embedded_key,
            params=effective_params,
            config=config,
            start_page=start_page,
            page_snapshots=page_snapshots,
        )
    all_items, identical_duplicates, conflicting_duplicates = _dedupe_collection_items(
        carried_items + list(batch_items)
    )
    total_pages = pages_before + batch_pages
    drift_detected = conflicting_duplicates > 0
    complete = not page_cap_hit and not drift_detected
    if not drift_detected:
        last_page = start_page + batch_pages - 1
        next_checkpoint[key] = {
            "fingerprint": fingerprint,
            "upper_bound": upper_bound.isoformat(),
            "next_page": start_page + batch_pages,
            "last_page": last_page,
            "last_page_anchor": page_anchor(page_snapshots.get(last_page, ())),
            "items": all_items,
            "pages_fetched": total_pages,
            "complete": complete,
        }
    stats = {
        "pages": total_pages,
        "pages_this_run": batch_pages,
        "start_page_this_run": start_page,
        "max_pages": max(1, int(config.max_pages)),
        "page_cap_hit": page_cap_hit,
        "complete": complete,
        "fetched": len(all_items),
        "fetched_this_run": len(batch_items),
        "carried_over_from_checkpoint": len(carried_items),
        "upper_bound": upper_bound.isoformat(),
        "checkpoint_reset_reason": checkpoint_reset_reason,
        "pagination_drift_detected": drift_detected,
        "identical_duplicates_collapsed": identical_duplicates,
        "conflicting_duplicates": conflicting_duplicates,
    }
    return all_items, stats


def _dedupe_collection_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int, int]:
    result: list[Mapping[str, Any]] = []
    positions: dict[str, int] = {}
    identical = conflicting = 0
    for item in items:
        item_id = clean_id(item.get("id"))
        if not item_id or item_id not in positions:
            if item_id:
                positions[item_id] = len(result)
            result.append(item)
            continue
        if stable_digest(result[positions[item_id]]) == stable_digest(item):
            identical += 1
        else:
            conflicting += 1
    return result, identical, conflicting


def run_amo_incremental(config: AmoIncrementalConfig) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    out_root = config.out_root.expanduser().resolve(strict=False)
    timeline_db = (
        config.timeline_db.expanduser().resolve(strict=False)
        if config.timeline_db is not None
        else out_root / "customer_timeline.sqlite"
    )
    guard_customer_timeline_writable_path(timeline_db)
    if config.timeline_db is not None and config.copy_db:
        raise ValueError("explicit timeline_db requires copy_db=False; refusing to overwrite target DB")
    out_root.mkdir(parents=True, exist_ok=True)
    if config.copy_db:
        backup_sqlite(config.source_db, timeline_db)
    if not timeline_db.exists():
        raise FileNotFoundError(f"timeline DB does not exist: {timeline_db}")
    allowed_root = (
        config.allowed_root.expanduser().resolve(strict=False)
        if config.allowed_root is not None
        else out_root
    )

    mcp_config = read_mcp_env(config.mcp_env)
    if config.mcp_transport:
        mcp_config = replace(mcp_config, transport=config.mcp_transport)
    client = AmoMcpClient(mcp_config)
    link_index_before = load_amo_link_index(timeline_db, tenant_id=config.tenant_id)
    cursor_before = load_cursor_snapshot(timeline_db, config.tenant_id)
    lower_bound = resolve_lower_bounds(cursor_before, config)
    source_dir = out_root / "amo_incremental_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "amo_leads_updated_at": source_dir / "amo_leads_updated_at.jsonl",
        "amo_contacts_updated_at": source_dir / "amo_contacts_updated_at.jsonl",
        "amo_events_created_at": source_dir / "amo_events_created_at.jsonl",
    }

    checkpoint = load_amo_incremental_checkpoint(out_root)
    next_checkpoint: dict[str, Any] = {}
    pending_leads = [
        item
        for item in (_checkpoint_entry(checkpoint, "amo_leads_pending").get("items") or ())
        if isinstance(item, Mapping)
    ]
    if pending_leads:
        next_checkpoint["amo_leads_pending"] = {"items": pending_leads}
    lead_items, lead_fetch_stats = fetch_endpoint_checkpointed(
        client,
        key="amo_leads_updated_at",
        path="leads",
        embedded_key="leads",
        params={
            "filter[updated_at][from]": int(lower_bound["amo_leads_updated_at"].timestamp()),
            "order[id]": "asc",
            "with": "contacts",
        },
        lower_bound=lower_bound["amo_leads_updated_at"],
        config=config,
        checkpoint=checkpoint,
        next_checkpoint=next_checkpoint,
    )
    contact_items, contact_fetch_stats = fetch_endpoint_checkpointed(
        client,
        key="amo_contacts_updated_at",
        path="contacts",
        embedded_key="contacts",
        params={
            "filter[updated_at][from]": int(lower_bound["amo_contacts_updated_at"].timestamp()),
            "order[id]": "asc",
            "with": "leads",
        },
        lower_bound=lower_bound["amo_contacts_updated_at"],
        config=config,
        checkpoint=checkpoint,
        next_checkpoint=next_checkpoint,
    )
    event_items, event_fetch_stats = fetch_endpoint_checkpointed(
        client,
        key="amo_events_created_at",
        path="events",
        embedded_key="events",
        params={
            "filter[created_at][from]": int(lower_bound["amo_events_created_at"].timestamp()),
            "filter[type][]": sorted(AMO_EVENT_TYPES),
            "order[id]": "asc",
        },
        lower_bound=lower_bound["amo_events_created_at"],
        config=config,
        checkpoint=checkpoint,
        next_checkpoint=next_checkpoint,
    )
    lead_pages, lead_page_cap_hit = lead_fetch_stats["pages"], lead_fetch_stats["page_cap_hit"]
    contact_pages, contact_page_cap_hit = contact_fetch_stats["pages"], contact_fetch_stats["page_cap_hit"]
    event_pages, event_page_cap_hit = event_fetch_stats["pages"], event_fetch_stats["page_cap_hit"]
    event_prefetch = (event_items, event_pages, event_page_cap_hit)
    fetch_report: dict[str, Any] = {
        "amo_leads_updated_at": {"endpoint": "/api/v4/leads", **lead_fetch_stats},
        "amo_contacts_updated_at": {"endpoint": "/api/v4/contacts", **contact_fetch_stats},
        "amo_events_created_at": {"endpoint": "/api/v4/events", **event_fetch_stats},
    }
    all_complete = all(stats.get("complete") for stats in fetch_report.values())
    # Keep both completed and incomplete endpoints until all DB imports have
    # succeeded. Otherwise a short endpoint restarts from page 1 on every run
    # while a longer endpoint is still walking its backlog.
    save_amo_incremental_checkpoint(out_root, next_checkpoint)
    if not all_complete:
        finished = datetime.now(timezone.utc)
        report = {
            "schema_version": AMO_INCREMENTAL_SCHEMA_VERSION,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 3),
            "timeline_db": str(timeline_db),
            "source_db": str(config.source_db),
            "cursor_before": cursor_before,
            "cursor_after": load_cursor_snapshot(timeline_db, config.tenant_id),
            "lower_bound": {key: value.isoformat() for key, value in lower_bound.items()},
            "fetch": fetch_report,
            "validation_ok": False,
            "apply_blocked": True,
            "blocked_reason": (
                "pagination_universe_changed"
                if any(stats.get("pagination_drift_detected") or stats.get("checkpoint_reset_reason") for stats in fetch_report.values())
                else "page_cap_hit"
            ),
            "checkpoint": {
                "path": str(_checkpoint_path(out_root)),
                "pending_endpoints": sorted(
                    key for key, stats in fetch_report.items() if not stats.get("complete")
                ),
                "note": "bounded checkpoint saved; the next run resumes from the persisted page, not page 1",
            },
            "safety": {
                "amo_write": False,
                "tallanto_write": False,
                "crm_write": False,
                "staging_db_write": False,
                "notes_endpoint_used": False,
                "bot_safe_summary_created": False,
            },
        }
        write_json(out_root / "amo_incremental_report.json", report)
        return report
    lead_items, _, _ = _dedupe_collection_items([*lead_items, *pending_leads])
    contact_rows, contact_stats = normalize_cards_source(
        contact_items,
        pages=contact_pages,
        page_cap_hit=contact_page_cap_hit,
        path="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_updated_at",
        link_index=link_index_before,
        config=config,
    )
    contact_fetched_ids = set(contact_stats.pop("_fetched_entity_ids", ()))
    fetch_report["amo_contacts_updated_at"] = {**contact_fetch_stats, **contact_stats}
    write_jsonl(paths["amo_contacts_updated_at"], contact_rows)
    contacts_config = nightly_config_for_sources(
        timeline_db=timeline_db,
        out_root=out_root,
        allowed_root=allowed_root,
        tenant_id=config.tenant_id,
        overlap_seconds=config.safety_overlap_seconds,
        paths=paths,
        source_names=("amo_contacts_updated_at",),
    )
    contacts_first = run_nightly_incremental(contacts_config)
    link_index_after_contacts = load_amo_link_index(timeline_db, tenant_id=config.tenant_id)

    lead_rows, lead_stats = normalize_cards_source(
        lead_items,
        pages=lead_pages,
        page_cap_hit=lead_page_cap_hit,
        path="leads",
        entity_type="lead",
        cursor_name="amo_leads_updated_at",
        link_index=link_index_after_contacts,
        config=config,
    )
    lead_fetched_ids = set(lead_stats.pop("_fetched_entity_ids", ()))
    skipped_lead_ids = set(lead_stats.pop("_skipped_entity_ids", ()))
    pending_leads = [item for item in lead_items if clean_id(item.get("id")) in skipped_lead_ids]
    lead_stats["pending_retry_count"] = len(pending_leads)
    fetch_report["amo_leads_updated_at"] = {**lead_fetch_stats, **lead_stats}
    write_jsonl(paths["amo_leads_updated_at"], lead_rows)
    cards_config = nightly_config_for_sources(
        timeline_db=timeline_db,
        out_root=out_root,
        allowed_root=allowed_root,
        tenant_id=config.tenant_id,
        overlap_seconds=config.safety_overlap_seconds,
        paths=paths,
        source_names=("amo_leads_updated_at", "amo_contacts_updated_at"),
    )
    cards_first = run_nightly_incremental(cards_config)
    link_index_after_cards = load_amo_link_index(timeline_db, tenant_id=config.tenant_id)
    opportunity_index_after_cards = load_amo_opportunity_index(timeline_db, tenant_id=config.tenant_id)

    event_rows, event_stats = fetch_events_source(
        client,
        from_ts=lower_bound["amo_events_created_at"],
        link_index=link_index_after_cards,
        opportunity_index=opportunity_index_after_cards,
        diagnostic_link_index_before=link_index_before,
        fetched_entity_ids={"lead": lead_fetched_ids, "contact": contact_fetched_ids},
        config=config,
        prefetched=event_prefetch,
    )
    fetch_report["amo_events_created_at"] = {**event_fetch_stats, **event_stats}

    write_jsonl(paths["amo_events_created_at"], event_rows)
    events_config = nightly_config_for_sources(
        timeline_db=timeline_db,
        out_root=out_root,
        allowed_root=allowed_root,
        tenant_id=config.tenant_id,
        overlap_seconds=config.safety_overlap_seconds,
        paths=paths,
        source_names=("amo_events_created_at",),
    )
    events_first = run_nightly_incremental(events_config)
    validation_ok = all(
        item.get("gate_passed") is True
        for item in (contacts_first, cards_first, events_first)
    )
    if validation_ok:
        # The private pending queue now owns unresolved leads. Advance the
        # network watermark to the proven fetch boundary so they are retried
        # locally instead of forcing the same AMO pages to be downloaded again.
        fetch_boundary = parse_iso(lead_fetch_stats["upper_bound"]) - timedelta(
            seconds=config.safety_overlap_seconds
        )
        with CustomerTimelineSQLiteStore(timeline_db, allowed_root=allowed_root) as store:
            cursor = store.get_ingestion_cursor(config.tenant_id, "amo_leads_updated_at")
            metadata = dict(cursor.metadata if cursor else {})
            metadata.update(
                {
                    "last_status": "ok",
                    "fetch_complete_upper_bound": lead_fetch_stats["upper_bound"],
                    "pending_lead_retries": len(pending_leads),
                }
            )
            store.upsert_ingestion_cursor(
                config.tenant_id,
                "amo_leads_updated_at",
                last_cursor_ts=max(fetch_boundary, cursor.last_cursor_ts if cursor else fetch_boundary),
                metadata=metadata,
                actor="customer_timeline_amo_incremental",
            )
        save_amo_incremental_checkpoint(
            out_root,
            {"amo_leads_pending": {"items": pending_leads}} if pending_leads else {},
        )
    cursor_after = load_cursor_snapshot(timeline_db, config.tenant_id)
    examples = sample_inserted_examples(timeline_db, config.tenant_id, limit=10)
    finished = datetime.now(timezone.utc)
    completed_import_sources = sorted({
        source
        for current in (contacts_first, cards_first, events_first)
        for source in completed_import_source_names(current.get("imports", ()))
    })
    report = {
        "schema_version": AMO_INCREMENTAL_SCHEMA_VERSION,
        "validation_ok": validation_ok,
        "complete": True,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "completed_import_sources": completed_import_sources,
        "timeline_db": str(timeline_db),
        "source_db": str(config.source_db),
        "endpoints": {
            "leads": "/api/v4/leads filter[updated_at][from]",
            "contacts": "/api/v4/contacts filter[updated_at][from]",
            "events": "/api/v4/events filter[created_at][from]",
            "notes": "not_used_whitelist_not_extended",
        },
        "cursor_before": cursor_before,
        "cursor_after": cursor_after,
        "lower_bound": {key: value.isoformat() for key, value in lower_bound.items()},
        "fetch": fetch_report,
        "link_index": {
            "before_entry_count": link_index_entry_count(link_index_before),
            "after_cards_entry_count": link_index_entry_count(link_index_after_cards),
        },
        "source_files": {key: str(path) for key, path in paths.items()},
        "first_run": {
            "contacts_bootstrap": compact_nightly_report(contacts_first),
            "cards": compact_nightly_report(cards_first),
            "events": compact_nightly_report(events_first),
            "affected_customer_count": int(cards_first.get("affected_customer_count") or 0)
            + int(events_first.get("affected_customer_count") or 0),
            "changed_customer_count": int(cards_first.get("changed_customer_count") or 0)
            + int(events_first.get("changed_customer_count") or 0),
        },
        "event_body_status": body_status_counts(event_rows),
        "examples": examples,
        "checkpoint": {
            "path": str(_checkpoint_path(out_root)),
            "pending_endpoints": [] if validation_ok else ["database_import"],
            "pending_lead_retries": len(pending_leads),
            "cleared": validation_ok and not pending_leads,
        },
        "identity_resolution": {
            "complete": not pending_leads,
            "pending_lead_retries": len(pending_leads),
            "pending_state": "private_checkpoint" if pending_leads else "none",
        },
        "safety": {
            "amo_write": False,
            "tallanto_write": False,
            "crm_write": False,
            "notes_endpoint_used": False,
            "bot_safe_summary_created": False,
            "test_copy_only": True,
        },
    }
    write_json(out_root / "amo_incremental_report.json", report)
    return report


def nightly_config_for_sources(
    *,
    timeline_db: Path,
    out_root: Path,
    allowed_root: Path,
    tenant_id: str,
    overlap_seconds: int,
    paths: Mapping[str, Path],
    source_names: Sequence[str],
    preserve_cursor_sources: Sequence[str] = (),
) -> NightlyIncrementalConfig:
    source_templates = {
        "amo_leads_updated_at": {
            "source_ref": "amocrm:leads:updated_at",
            "normalizer": "amo_snapshot",
        },
        "amo_contacts_updated_at": {
            "source_ref": "amocrm:contacts:updated_at",
            "normalizer": "amo_snapshot",
        },
        "amo_events_created_at": {
            "source_ref": "amocrm:events:created_at",
            "normalizer": "amo_event",
        },
    }
    sources = []
    for name in source_names:
        template = source_templates[name]
        sources.append(
            IncrementalSourceConfig(
                name=name,
                source_system=name,
                path=paths[name],
                tenant_id=tenant_id,
                source_ref=template["source_ref"],
                normalizer=template["normalizer"],
                preserve_cursor=name in preserve_cursor_sources,
            )
        )
    return NightlyIncrementalConfig(
        timeline_db=timeline_db,
        allowed_root=allowed_root,
        tenant_id=tenant_id,
        journal_path=out_root / "amo_incremental_journal.jsonl",
        safety_margin_seconds=overlap_seconds,
        sources=tuple(sources),
    )


def backup_sqlite(source: Path, target: Path) -> None:
    source = source.expanduser().resolve(strict=False)
    target = target.expanduser().resolve(strict=False)
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src, sqlite3.connect(target) as dst:
        src.backup(dst)


def load_amo_link_index(db_path: Path, *, tenant_id: str) -> Mapping[tuple[str, str], tuple[str, ...]]:
    result: dict[tuple[str, str], set[str]] = {}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        for row in con.execute(
            """
            SELECT link_type, link_value, customer_id
            FROM identity_links
            WHERE tenant_id = ?
              AND link_type IN (
                'amo_lead_id', 'amo_contact_id', 'phone', 'mango_client_phone',
                'whatsapp_phone', 'email'
              )
            """,
            (tenant_id,),
        ):
            link_type = str(row["link_type"])
            canonical_type = "phone" if link_type in {"phone", "mango_client_phone", "whatsapp_phone"} else link_type
            result.setdefault((canonical_type, str(row["link_value"])), set()).add(str(row["customer_id"]))
    return {key: tuple(sorted(values)) for key, values in result.items()}


def load_amo_opportunity_index(db_path: Path, *, tenant_id: str) -> Mapping[str, tuple[Mapping[str, str], ...]]:
    result: dict[str, list[Mapping[str, str]]] = {}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        for row in con.execute(
            """
            SELECT source_id, opportunity_id, customer_id
            FROM customer_opportunities
            WHERE tenant_id = ?
              AND source_system = 'amocrm_snapshot'
              AND opportunity_type = 'amo_deal'
              AND source_id IS NOT NULL
              AND source_id != ''
            """,
            (tenant_id,),
        ):
            result.setdefault(str(row["source_id"]), []).append(
                {
                    "opportunity_id": str(row["opportunity_id"]),
                    "customer_id": str(row["customer_id"]),
                }
            )
    return {key: tuple(values) for key, values in result.items()}


def load_cursor_snapshot(db_path: Path, tenant_id: str) -> Mapping[str, Optional[str]]:
    wanted = ("amo_leads_updated_at", "amo_contacts_updated_at", "amo_events_created_at")
    result: dict[str, Optional[str]] = {key: None for key in wanted}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        try:
            rows = con.execute(
                "SELECT source_system, last_cursor_ts FROM ingestion_cursors WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return result
    for row in rows:
        source = str(row["source_system"])
        if source in result:
            result[source] = str(row["last_cursor_ts"])
    return result


def resolve_lower_bounds(cursor_before: Mapping[str, Optional[str]], config: AmoIncrementalConfig) -> Mapping[str, datetime]:
    fallback = config.since or (datetime.now(timezone.utc) - timedelta(hours=24))
    result: dict[str, datetime] = {}
    for key in ("amo_leads_updated_at", "amo_contacts_updated_at", "amo_events_created_at"):
        raw = cursor_before.get(key)
        result[key] = parse_iso(raw) if raw else fallback
    return result


def fetch_cards_source(
    client: AmoMcpClient,
    *,
    path: str,
    embedded_key: str,
    entity_type: str,
    cursor_name: str,
    from_ts: datetime,
    link_index: Mapping[tuple[str, str], tuple[str, ...]],
    config: AmoIncrementalConfig,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    items, pages, page_cap_hit = fetch_collection(
        client,
        path=path,
        embedded_key=embedded_key,
        params={
            "filter[updated_at][from]": int(from_ts.timestamp()),
            "order[id]": "asc",
            "with": "contacts" if entity_type == "lead" else "leads",
        },
        config=config,
    )
    return normalize_cards_source(
        items,
        pages=pages,
        page_cap_hit=page_cap_hit,
        path=path,
        entity_type=entity_type,
        cursor_name=cursor_name,
        link_index=link_index,
        config=config,
    )


def normalize_cards_source(
    items: Sequence[Mapping[str, Any]],
    *,
    pages: int,
    page_cap_hit: bool,
    path: str,
    entity_type: str,
    cursor_name: str,
    link_index: Mapping[tuple[str, str], tuple[str, ...]],
    config: AmoIncrementalConfig,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    skipped = Counter()
    resolution_counts = Counter()
    contact_identity_diagnostics = Counter()
    contact_identity_by_entity: Mapping[str, Mapping[str, str]] = {}
    if entity_type == "contact":
        contact_identity_by_entity, contact_identity_diagnostics = safe_contact_identity_fields(
            items,
            link_index=link_index,
        )
    fetched_entity_ids: set[str] = set()
    skipped_entity_ids: set[str] = set()
    for item in items:
        entity_id = clean_id(item.get("id"))
        if not entity_id:
            skipped["missing_id"] += 1
            continue
        fetched_entity_ids.add(entity_id)
        link_type = "amo_lead_id" if entity_type == "lead" else "amo_contact_id"
        customers, resolution = resolve_card_customers(item, entity_type=entity_type, entity_id=entity_id, link_index=link_index)
        resolution_counts[resolution] += 1
        if len(customers) > 1:
            skipped["ambiguous"] += 1
            skipped_entity_ids.add(entity_id)
            continue
        updated_at = epoch_to_iso(item.get("updated_at") or item.get("created_at"))
        significant_hash = stable_digest(significant_card_payload(item))
        contact_identity = contact_identity_by_entity.get(entity_id, {})
        row = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "source_id": f"{entity_type}:{entity_id}:{updated_at}:{significant_hash[:12]}",
            "customer_id": customers[0] if len(customers) == 1 else None,
            "name": item.get("name"),
            "status": item.get("status_id"),
            "pipeline": item.get("pipeline_id"),
            "created_at": epoch_to_iso(item.get("created_at")) or updated_at,
            # Without an explicit event_at the normalizer falls back to
            # created_at, so every version of one card lands on the same
            # timestamp and "ORDER BY event_at DESC" cannot tell them apart.
            "event_at": updated_at,
            "updated_at": updated_at,
            "source_ref": f"amocrm:{entity_type}:{entity_id}",
            "record": scrub_item(item),
            "source_cursor": cursor_name,
            **contact_identity,
        }
        if not row["customer_id"] and entity_type == "lead":
            skipped["unmatched"] += 1
            skipped_entity_ids.add(entity_id)
            continue
        rows.append(row)
    return rows, {
        "endpoint": f"/api/v4/{path}",
        "pages": pages,
        "max_pages": max(1, int(config.max_pages)),
        "page_cap_hit": page_cap_hit,
        "fetched": len(items),
        "fetched_entity_count": len(fetched_entity_ids),
        "normalized": len(rows),
        "skipped": dict(skipped),
        "contact_identity_diagnostics": dict(contact_identity_diagnostics),
        "resolution_counts": dict(resolution_counts),
        "_fetched_entity_ids": tuple(sorted(fetched_entity_ids)),
        "_skipped_entity_ids": tuple(sorted(skipped_entity_ids)),
    }


def contact_identity_fields(item: Mapping[str, Any]) -> tuple[dict[str, str], Counter[str]]:
    normalized: dict[str, set[str]] = {"phone": set(), "email": set()}
    diagnostics: Counter[str] = Counter()
    for field in item.get("custom_fields_values") or ():
        if not isinstance(field, Mapping):
            continue
        kind = str(field.get("field_code") or "").strip().lower()
        if kind not in normalized:
            continue
        normalizer = normalize_phone if kind == "phone" else normalize_email
        for value_item in field.get("values") or ():
            raw = value_item.get("value") if isinstance(value_item, Mapping) else value_item
            value = normalizer(raw)
            if value:
                normalized[kind].add(value)
            elif str(raw or "").strip():
                diagnostics[f"{kind}_invalid_values_skipped"] += 1

    selected: dict[str, str] = {}
    for kind, values in normalized.items():
        if len(values) == 1:
            selected[kind] = next(iter(values))
        elif len(values) > 1:
            diagnostics[f"{kind}_ambiguous_contacts"] += 1
    return selected, diagnostics


def safe_contact_identity_fields(
    items: Sequence[Mapping[str, Any]],
    *,
    link_index: Mapping[tuple[str, str], tuple[str, ...]],
) -> tuple[Mapping[str, Mapping[str, str]], Counter[str]]:
    candidates: dict[str, dict[str, str]] = {}
    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    diagnostics: Counter[str] = Counter()
    for item in items:
        entity_id = clean_id(item.get("id"))
        if not entity_id:
            continue
        fields, item_diagnostics = contact_identity_fields(item)
        diagnostics.update(item_diagnostics)
        candidates[entity_id] = fields
        customers, _ = resolve_card_customers(
            item,
            entity_type="contact",
            entity_id=entity_id,
            link_index=link_index,
        )
        owner_refs = set(customers) or {f"unresolved:{entity_id}"}
        for kind, value in fields.items():
            owners.setdefault((kind, value), set()).update(owner_refs)

    selected: dict[str, dict[str, str]] = {}
    for entity_id, fields in candidates.items():
        for kind, value in fields.items():
            value_owners = owners[(kind, value)] | set(link_index.get((kind, value), ()))
            if len(value_owners) == 1:
                selected.setdefault(entity_id, {})[kind] = value
                diagnostics[f"{kind}_selected"] += 1
            else:
                diagnostics[f"{kind}_cross_customer_ambiguous"] += 1
    return selected, diagnostics


def resolve_card_customers(
    item: Mapping[str, Any],
    *,
    entity_type: str,
    entity_id: str,
    link_index: Mapping[tuple[str, str], tuple[str, ...]],
) -> tuple[tuple[str, ...], str]:
    link_type = "amo_lead_id" if entity_type == "lead" else "amo_contact_id"
    direct = link_index.get((link_type, entity_id), ())
    if direct:
        return direct, "direct_identity_link" if len(direct) == 1 else "direct_ambiguous"
    if entity_type != "lead":
        lead_customers: set[str] = set()
        for lead_id in embedded_entity_ids(item, "leads"):
            lead_customers.update(link_index.get(("amo_lead_id", lead_id), ()))
        if len(lead_customers) == 1:
            return tuple(lead_customers), "embedded_lead_identity_link"
        if len(lead_customers) > 1:
            return tuple(sorted(lead_customers)), "embedded_lead_ambiguous"
        return (), "new_contact_identity"
    contact_customers: set[str] = set()
    for contact_id in embedded_entity_ids(item, "contacts"):
        contact_customers.update(link_index.get(("amo_contact_id", contact_id), ()))
    if len(contact_customers) == 1:
        return tuple(sorted(contact_customers)), "embedded_contact_identity_link"
    if len(contact_customers) > 1:
        return tuple(sorted(contact_customers)), "embedded_contact_ambiguous"
    return (), "unmatched"


def embedded_entity_ids(item: Mapping[str, Any], key: str) -> tuple[str, ...]:
    embedded = item.get("_embedded")
    if not isinstance(embedded, Mapping):
        return ()
    values = embedded.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    result: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            entity_id = clean_id(value.get("id"))
            if entity_id:
                result.append(entity_id)
    return tuple(result)


def fetch_events_source(
    client: AmoMcpClient,
    *,
    from_ts: datetime,
    link_index: Mapping[tuple[str, str], tuple[str, ...]],
    opportunity_index: Optional[Mapping[str, tuple[Mapping[str, str], ...]]] = None,
    diagnostic_link_index_before: Optional[Mapping[tuple[str, str], tuple[str, ...]]] = None,
    fetched_entity_ids: Optional[Mapping[str, set[str]]] = None,
    config: AmoIncrementalConfig,
    prefetched: tuple[list[Mapping[str, Any]], int, bool] | None = None,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    items, pages, page_cap_hit = prefetched or fetch_events_collection(
        client,
        from_ts=from_ts,
        config=config,
    )
    rows: list[Mapping[str, Any]] = []
    skipped = Counter()
    skipped_by_type: Counter[str] = Counter()
    mapping_counts = Counter()
    mapping_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    mapping_common_note = Counter()
    diagnostic_samples: list[Mapping[str, Any]] = []
    before_index = diagnostic_link_index_before or {}
    opportunities = opportunity_index or {}
    fetched_ids = fetched_entity_ids or {}
    opportunity_counts = Counter()
    for item in items:
        amo_type = str(item.get("type") or "").strip()
        if amo_type not in AMO_EVENT_TYPES:
            skipped["unsupported_type"] += 1
            skipped_by_type[amo_type or "unknown"] += 1
            category = "unsupported_type"
            mapping_counts[category] += 1
            mapping_by_type[amo_type or "unknown"][category] += 1
            continue
        entity_type = str(item.get("entity_type") or "").strip()
        entity_id = clean_id(item.get("entity_id"))
        event_id = clean_id(item.get("id"))
        if entity_type not in {"lead", "contact"} or not entity_id or not event_id:
            skipped["missing_entity"] += 1
            skipped_by_type[amo_type] += 1
            category = "missing_entity"
            mapping_counts[category] += 1
            mapping_by_type[amo_type][category] += 1
            add_diagnostic_sample(diagnostic_samples, item, category=category, in_fetched_card=False, before_count=0, after_count=0)
            continue
        link_type = "amo_lead_id" if entity_type == "lead" else "amo_contact_id"
        before_count = len(before_index.get((link_type, entity_id), ()))
        customers = link_index.get((link_type, entity_id), ())
        in_fetched_card = entity_id in fetched_ids.get(entity_type, set())
        category = event_mapping_category(
            before_count=before_count,
            after_count=len(customers),
            in_fetched_card=in_fetched_card,
        )
        mapping_counts[category] += 1
        mapping_by_type[amo_type][category] += 1
        if amo_type == "common_note_added":
            mapping_common_note[category] += 1
        if len(customers) == 0:
            skipped["unmatched"] += 1
            skipped_by_type[amo_type] += 1
            add_diagnostic_sample(
                diagnostic_samples,
                item,
                category=category,
                in_fetched_card=in_fetched_card,
                before_count=before_count,
                after_count=0,
            )
            continue
        if len(customers) > 1:
            skipped["ambiguous"] += 1
            skipped_by_type[amo_type] += 1
            add_diagnostic_sample(
                diagnostic_samples,
                item,
                category=category,
                in_fetched_card=in_fetched_card,
                before_count=before_count,
                after_count=len(customers),
            )
            continue
        body_status = "note_body_missing" if amo_type.startswith("common_note") else "event_only"
        opportunity_id = resolve_event_opportunity_id(
            entity_type=entity_type,
            entity_id=entity_id,
            customer_id=customers[0],
            opportunity_index=opportunities,
        )
        opportunity_counts["mapped" if opportunity_id else "missing_or_not_applicable"] += 1
        rows.append(
            {
                "event_id": event_id,
                "customer_id": customers[0],
                "entity_type": entity_type,
                "entity_id": entity_id,
                "opportunity_id": opportunity_id,
                "amo_event_type": amo_type,
                "created_at": epoch_to_iso(item.get("created_at")),
                "event_at": epoch_to_iso(item.get("created_at")),
                "updated_at": epoch_to_iso(item.get("created_at")),
                "source_ref": f"amocrm:event:{event_id}",
                "source_body_status": body_status,
                "subject": amo_type,
                "summary": event_summary(item, body_status=body_status),
                "text_preview": event_summary(item, body_status=body_status),
                "record": scrub_item(item),
                "source_cursor": "amo_events_created_at",
            }
        )
    return rows, {
        "endpoint": "/api/v4/events",
        "pages": pages,
        "max_pages": max(1, int(config.max_pages)),
        "page_cap_hit": page_cap_hit,
        "fetched": len(items),
        "normalized": len(rows),
        "skipped": dict(skipped),
        "fetched_type_counts": dict(Counter(str(item.get("type") or "unknown") for item in items)),
        "normalized_type_counts": dict(Counter(str(row.get("amo_event_type") or "unknown") for row in rows)),
        "skipped_type_counts": dict(skipped_by_type),
        "mapping_diagnostics_counts": dict(mapping_counts),
        "mapping_diagnostics_by_type": {key: dict(value) for key, value in mapping_by_type.items()},
        "common_note_added_mapping_diagnostics": dict(mapping_common_note),
        "opportunity_mapping_counts": dict(opportunity_counts),
        "diagnostic_samples": diagnostic_samples[:20],
        "source_body_status_counts": body_status_counts(rows),
    }


def fetch_events_collection(
    client: AmoMcpClient,
    *,
    from_ts: datetime,
    config: AmoIncrementalConfig,
    start_page: int = 1,
    params_override: Optional[Mapping[str, Any]] = None,
    page_snapshots: Optional[dict[int, list[Mapping[str, Any]]]] = None,
) -> tuple[list[Mapping[str, Any]], int, bool]:
    return fetch_collection(
        client,
        path="events",
        embedded_key="events",
        params={
            "filter[created_at][from]": int(from_ts.timestamp()),
            "filter[type][]": sorted(AMO_EVENT_TYPES),
            "order[id]": "asc",
            **dict(params_override or {}),
        },
        config=config,
        start_page=start_page,
        page_snapshots=page_snapshots,
    )


def resolve_event_opportunity_id(
    *,
    entity_type: str,
    entity_id: str,
    customer_id: str,
    opportunity_index: Mapping[str, tuple[Mapping[str, str], ...]],
) -> Optional[str]:
    if entity_type != "lead":
        return None
    matches = [
        item
        for item in opportunity_index.get(entity_id, ())
        if str(item.get("customer_id") or "") == customer_id and str(item.get("opportunity_id") or "")
    ]
    if len(matches) != 1:
        return None
    return str(matches[0]["opportunity_id"])


def event_mapping_category(*, before_count: int, after_count: int, in_fetched_card: bool) -> str:
    if after_count == 1:
        return "mapped_before" if before_count == 1 else "mapped_after_card_import"
    if after_count > 1:
        return "ambiguous_before" if before_count > 1 else "ambiguous_after_card_import"
    return "fetched_card_but_no_link_after" if in_fetched_card else "entity_not_in_fetched_cards"


def add_diagnostic_sample(
    samples: list[Mapping[str, Any]],
    item: Mapping[str, Any],
    *,
    category: str,
    in_fetched_card: bool,
    before_count: int,
    after_count: int,
) -> None:
    if len(samples) >= 20:
        return
    entity_id = clean_id(item.get("entity_id"))
    samples.append(
        {
            "event_id_masked": mask_id(clean_id(item.get("id"))),
            "event_type": str(item.get("type") or "unknown"),
            "entity_type": str(item.get("entity_type") or "unknown"),
            "entity_id_masked": mask_id(entity_id),
            "in_fetched_card": bool(in_fetched_card),
            "source_link_count": before_count,
            "after_card_import_link_count": after_count,
            "category": category,
        }
    )


def link_index_entry_count(index: Mapping[tuple[str, str], tuple[str, ...]]) -> int:
    return sum(1 for customers in index.values() if len(customers) == 1)


def fetch_collection(
    client: AmoMcpClient,
    *,
    path: str,
    embedded_key: str,
    params: Mapping[str, Any],
    config: AmoIncrementalConfig,
    start_page: int = 1,
    page_snapshots: Optional[dict[int, list[Mapping[str, Any]]]] = None,
) -> tuple[list[Mapping[str, Any]], int, bool]:
    items: list[Mapping[str, Any]] = []
    pages = 0
    max_pages = max(1, int(config.max_pages))
    # D1: start_page lets a caller resume a bounded pagination window from a
    # persisted checkpoint instead of always starting at page 1; default of 1
    # reproduces the exact previous behavior/tests unchanged.
    first_page = max(1, int(start_page))
    last_page = first_page + max_pages - 1
    page_cap_hit = False
    for page in range(first_page, last_page + 1):
        payload = _fetch_collection_page(client, path=path, params=params, page=page, config=config)
        pages += 1
        page_items = embedded_items(payload, embedded_key)
        if page_snapshots is not None:
            page_snapshots[page] = list(page_items)
        if not page_items:
            break
        items.extend(page_items)
        links = payload.get("_links") if isinstance(payload, Mapping) else {}
        if not isinstance(links, Mapping) or not isinstance(links.get("next"), Mapping):
            break
        if page >= last_page:
            page_cap_hit = True
            break
        time.sleep(config.sleep_sec)
    return items, pages, page_cap_hit


def _fetch_collection_page(
    client: AmoMcpClient,
    *,
    path: str,
    params: Mapping[str, Any],
    page: int,
    config: AmoIncrementalConfig,
) -> Mapping[str, Any]:
    try:
        return client.amo_api_get(
            path=path,
            params={**dict(params), "page": page},
            limit=config.page_limit,
        )
    except AmoMcpError as exc:
        text = str(exc).lower()
        if "429" not in text and "timed out" not in text and "timeout" not in text:
            raise
        time.sleep(max(2.0, config.sleep_sec * 3))
        return client.amo_api_get(
            path=path,
            params={**dict(params), "page": page},
            limit=config.page_limit,
        )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def parse_iso(value: Optional[str]) -> datetime:
    if not value:
        raise ValueError("empty datetime")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def epoch_to_iso(value: Any) -> Optional[str]:
    if value in {None, ""}:
        return None
    try:
        return datetime.fromtimestamp(int(value), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return str(value)


def clean_id(value: Any) -> str:
    return str(value or "").strip()


def scrub_item(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in item.items() if key not in {"request_id"}}


def significant_card_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "name": item.get("name"),
        "price": item.get("price"),
        "status_id": item.get("status_id"),
        "pipeline_id": item.get("pipeline_id"),
        "responsible_user_id": item.get("responsible_user_id"),
        "custom_fields_values": item.get("custom_fields_values"),
        "updated_at": item.get("updated_at"),
    }


def event_summary(item: Mapping[str, Any], *, body_status: str) -> str:
    event_type = str(item.get("type") or "amo_event")
    entity_type = str(item.get("entity_type") or "entity")
    suffix = "body missing" if body_status == "note_body_missing" else "event only"
    return f"AMO {event_type} for {entity_type}; {suffix}"


def compact_nightly_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "affected_customer_count": report.get("affected_customer_count"),
        "changed_customer_count": report.get("changed_customer_count"),
        "sources": report.get("sources"),
        "imports": [
            {
                "source_system": item.get("source_system"),
                "accepted_count": item.get("accepted_count"),
                "write_status_counts": item.get("write_status_counts"),
            }
            for item in report.get("imports", ())
        ],
        "cursor_updates": report.get("cursor_updates"),
        "source_errors": report.get("source_errors"),
        "safety": report.get("safety"),
    }


def body_status_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts = Counter(str(row.get("source_body_status") or "unknown") for row in rows)
    return {
        "event_only": int(counts.get("event_only") or 0),
        "note_body_missing": int(counts.get("note_body_missing") or 0),
    }


def sample_inserted_examples(db_path: Path, tenant_id: str, *, limit: int) -> list[Mapping[str, Any]]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        rows = con.execute(
            """
            SELECT customer_id, event_type, event_at, source_system, source_id, summary
            FROM timeline_events
            WHERE tenant_id = ?
              AND source_system IN ('amocrm_snapshot', 'amocrm_event')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (tenant_id, limit),
        ).fetchall()
    result = []
    for row in rows:
        result.append(
            {
                "customer_id_masked": mask_id(str(row["customer_id"])),
                "event_type": row["event_type"],
                "event_at": row["event_at"],
                "source_system": row["source_system"],
                "source_id_masked": mask_id(str(row["source_id"])),
                "summary": row["summary"],
            }
        )
    return result


def mask_id(value: str) -> str:
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


__all__ = [
    "AMO_INCREMENTAL_SCHEMA_VERSION",
    "AmoIncrementalConfig",
    "run_amo_incremental",
]
