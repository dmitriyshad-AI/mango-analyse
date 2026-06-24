from __future__ import annotations

import json
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.existing_clients.amo_step1_snapshot import (
    AmoMcpClient,
    AmoMcpError,
    embedded_items,
    read_mcp_env,
)
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    run_nightly_incremental,
)


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
    tenant_id: str = "foton"
    safety_overlap_seconds: int = 300
    page_limit: int = 20
    max_pages: int = 2
    sleep_sec: float = 1.05
    since: Optional[datetime] = None
    copy_db: bool = True


def run_amo_incremental(config: AmoIncrementalConfig) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    out_root = config.out_root.expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)
    timeline_db = out_root / "customer_timeline.sqlite"
    if config.copy_db:
        backup_sqlite(config.source_db, timeline_db)
    if not timeline_db.exists():
        raise FileNotFoundError(f"timeline DB does not exist: {timeline_db}")

    client = AmoMcpClient(read_mcp_env(config.mcp_env))
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

    fetch_report: dict[str, Any] = {}
    lead_rows, lead_stats = fetch_cards_source(
        client,
        path="leads",
        embedded_key="leads",
        entity_type="lead",
        cursor_name="amo_leads_updated_at",
        from_ts=lower_bound["amo_leads_updated_at"],
        link_index=link_index_before,
        config=config,
    )
    contact_rows, contact_stats = fetch_cards_source(
        client,
        path="contacts",
        embedded_key="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_updated_at",
        from_ts=lower_bound["amo_contacts_updated_at"],
        link_index=link_index_before,
        config=config,
    )
    lead_fetched_ids = set(lead_stats.pop("_fetched_entity_ids", ()))
    contact_fetched_ids = set(contact_stats.pop("_fetched_entity_ids", ()))
    write_jsonl(paths["amo_leads_updated_at"], lead_rows)
    write_jsonl(paths["amo_contacts_updated_at"], contact_rows)
    cards_config = nightly_config_for_sources(
        timeline_db=timeline_db,
        out_root=out_root,
        tenant_id=config.tenant_id,
        overlap_seconds=config.safety_overlap_seconds,
        paths=paths,
        source_names=("amo_leads_updated_at", "amo_contacts_updated_at"),
    )
    cards_first = run_nightly_incremental(cards_config)
    link_index_after_cards = load_amo_link_index(timeline_db, tenant_id=config.tenant_id)

    event_rows, event_stats = fetch_events_source(
        client,
        from_ts=lower_bound["amo_events_created_at"],
        link_index=link_index_after_cards,
        diagnostic_link_index_before=link_index_before,
        fetched_entity_ids={"lead": lead_fetched_ids, "contact": contact_fetched_ids},
        config=config,
    )
    fetch_report["amo_leads_updated_at"] = lead_stats
    fetch_report["amo_contacts_updated_at"] = contact_stats
    fetch_report["amo_events_created_at"] = event_stats

    write_jsonl(paths["amo_events_created_at"], event_rows)
    events_config = nightly_config_for_sources(
        timeline_db=timeline_db,
        out_root=out_root,
        tenant_id=config.tenant_id,
        overlap_seconds=config.safety_overlap_seconds,
        paths=paths,
        source_names=("amo_events_created_at",),
    )
    events_first = run_nightly_incremental(events_config)
    all_config = nightly_config_for_sources(
        timeline_db=timeline_db,
        out_root=out_root,
        tenant_id=config.tenant_id,
        overlap_seconds=config.safety_overlap_seconds,
        paths=paths,
        source_names=("amo_leads_updated_at", "amo_contacts_updated_at", "amo_events_created_at"),
    )
    second = run_nightly_incremental(all_config)
    cursor_after = load_cursor_snapshot(timeline_db, config.tenant_id)
    examples = sample_inserted_examples(timeline_db, config.tenant_id, limit=10)
    finished = datetime.now(timezone.utc)
    report = {
        "schema_version": AMO_INCREMENTAL_SCHEMA_VERSION,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
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
            "cards": compact_nightly_report(cards_first),
            "events": compact_nightly_report(events_first),
            "affected_customer_count": int(cards_first.get("affected_customer_count") or 0)
            + int(events_first.get("affected_customer_count") or 0),
            "changed_customer_count": int(cards_first.get("changed_customer_count") or 0)
            + int(events_first.get("changed_customer_count") or 0),
        },
        "second_run": compact_nightly_report(second),
        "repeat_run_duplicates": import_duplicate_count(second),
        "event_body_status": body_status_counts(event_rows),
        "examples": examples,
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
    tenant_id: str,
    overlap_seconds: int,
    paths: Mapping[str, Path],
    source_names: Sequence[str],
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
            )
        )
    return NightlyIncrementalConfig(
        timeline_db=timeline_db,
        allowed_root=out_root,
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
              AND link_type IN ('amo_lead_id', 'amo_contact_id')
            """,
            (tenant_id,),
        ):
            result.setdefault((str(row["link_type"]), str(row["link_value"])), set()).add(str(row["customer_id"]))
    return {key: tuple(sorted(values)) for key, values in result.items()}


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
    items, pages = fetch_collection(
        client,
        path=path,
        embedded_key=embedded_key,
        params={
            "filter[updated_at][from]": int(from_ts.timestamp()),
            "order[updated_at]": "asc",
            "with": "contacts" if entity_type == "lead" else "leads",
        },
        config=config,
    )
    rows: list[Mapping[str, Any]] = []
    skipped = Counter()
    resolution_counts = Counter()
    fetched_entity_ids: set[str] = set()
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
            continue
        updated_at = epoch_to_iso(item.get("updated_at") or item.get("created_at"))
        significant_hash = stable_digest(significant_card_payload(item))
        row = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "source_id": f"{entity_type}:{entity_id}:{updated_at}:{significant_hash[:12]}",
            "customer_id": customers[0] if len(customers) == 1 else None,
            "name": item.get("name"),
            "status": item.get("status_id"),
            "pipeline": item.get("pipeline_id"),
            "created_at": epoch_to_iso(item.get("created_at")) or updated_at,
            "updated_at": updated_at,
            "source_ref": f"amocrm:{entity_type}:{entity_id}",
            "record": scrub_item(item),
            "source_cursor": cursor_name,
        }
        if not row["customer_id"] and entity_type == "lead":
            skipped["unmatched"] += 1
            continue
        rows.append(row)
    return rows, {
        "endpoint": f"/api/v4/{path}",
        "pages": pages,
        "fetched": len(items),
        "fetched_entity_count": len(fetched_entity_ids),
        "normalized": len(rows),
        "skipped": dict(skipped),
        "resolution_counts": dict(resolution_counts),
        "_fetched_entity_ids": tuple(sorted(fetched_entity_ids)),
    }


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
    diagnostic_link_index_before: Optional[Mapping[tuple[str, str], tuple[str, ...]]] = None,
    fetched_entity_ids: Optional[Mapping[str, set[str]]] = None,
    config: AmoIncrementalConfig,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    items, pages = fetch_collection(
        client,
        path="events",
        embedded_key="events",
        params={
            "filter[created_at][from]": int(from_ts.timestamp()),
            "filter[type][]": sorted(AMO_EVENT_TYPES),
            "order[created_at]": "asc",
        },
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
    fetched_ids = fetched_entity_ids or {}
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
        rows.append(
            {
                "event_id": event_id,
                "customer_id": customers[0],
                "entity_type": entity_type,
                "entity_id": entity_id,
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
        "fetched": len(items),
        "normalized": len(rows),
        "skipped": dict(skipped),
        "fetched_type_counts": dict(Counter(str(item.get("type") or "unknown") for item in items)),
        "normalized_type_counts": dict(Counter(str(row.get("amo_event_type") or "unknown") for row in rows)),
        "skipped_type_counts": dict(skipped_by_type),
        "mapping_diagnostics_counts": dict(mapping_counts),
        "mapping_diagnostics_by_type": {key: dict(value) for key, value in mapping_by_type.items()},
        "common_note_added_mapping_diagnostics": dict(mapping_common_note),
        "diagnostic_samples": diagnostic_samples[:20],
        "source_body_status_counts": body_status_counts(rows),
    }


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
) -> tuple[list[Mapping[str, Any]], int]:
    items: list[Mapping[str, Any]] = []
    pages = 0
    for page in range(1, max(1, config.max_pages) + 1):
        try:
            payload = client.amo_api_get(path=path, params={**dict(params), "page": page}, limit=config.page_limit)
        except AmoMcpError as exc:
            if "429" in str(exc):
                time.sleep(max(2.0, config.sleep_sec * 3))
                payload = client.amo_api_get(path=path, params={**dict(params), "page": page}, limit=config.page_limit)
            else:
                raise
        pages += 1
        page_items = embedded_items(payload, embedded_key)
        if not page_items:
            break
        items.extend(page_items)
        links = payload.get("_links") if isinstance(payload, Mapping) else {}
        if not isinstance(links, Mapping) or not isinstance(links.get("next"), Mapping):
            break
        time.sleep(config.sleep_sec)
    return items, pages


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def import_duplicate_count(report: Mapping[str, Any]) -> int:
    total = 0
    for item in report.get("imports", ()):
        counts = item.get("write_status_counts") if isinstance(item, Mapping) else {}
        if isinstance(counts, Mapping):
            total += int(counts.get("duplicate") or 0)
    return total


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
