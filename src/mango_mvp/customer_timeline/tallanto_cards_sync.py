from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, TallantoApiError
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    run_nightly_incremental,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_writable_path
from mango_mvp.customer_timeline.tallanto_attendance_import import _build_tallanto_client


# BLOCK C: the real tallanto_cards_sync step. Reuses the existing read-only
# TallantoApiClient (mango_mvp.amocrm_runtime.tallanto_api), the existing
# TallantoSnapshotNormalizer / TimelineImportService import contract
# (mango_mvp.customer_timeline.ingestion), and the existing generic
# run_nightly_incremental step (mango_mvp.customer_timeline.nightly_incremental)
# -- no second API client, no second XLS/JSON parser, no second identity
# engine. This module only adds: (1) a thin adapter from raw Tallanto
# Contact fields to the flat payload shape TallantoSnapshotNormalizer already
# consumes (the same shape scripts/normalize_tallanto_contacts.py produces
# from the bootstrap Contacts *.xls), and (2) bounded/checkpointed pagination
# over the Contact module so a run never blocks forever on a large contact
# list and never claims freshness on a partial read.
#
# Design note on "daily increment": Tallanto's Contact module exposes no
# confirmed date-filter query syntax in this codebase, so this step walks
# the FULL current Contact module every completed cycle (bounded across
# possibly several runs for a large universe -- see the checkpoint below)
# rather than guessing at an unverified server-side filter. Freshness is not
# an unbounded re-scan risk here: TimelineImportService's dedupe_key/
# record_hash upsert already makes an unchanged contact a pure no-op
# ("duplicate", zero new events), which is exactly what "a repeat run after
# a full cycle with the same universe creates zero new events" requires.
TALLANTO_CARDS_SYNC_SCHEMA_VERSION = "customer_timeline_tallanto_cards_sync_v1"
# Cursor bucket for THIS daily live-API cycle. Deliberately distinct from
# TallantoSnapshotNormalizer.source_system ("tallanto_snapshot", used on the
# events/identity links it emits -- unchanged, so identity resolution still
# merges correctly with any other tallanto_snapshot-sourced data) and from
# any manual bootstrap `customer_timeline_import.py --source-kind
# tallanto_snapshot` run (which never touches ingestion_cursors at all). A
# manual bootstrap import can therefore never be mistaken for proof this
# live cycle ran -- "Contacts 10.07.2026.xls = bootstrap only, NOT proof of
# freshness".
TALLANTO_CARDS_SOURCE_SYSTEM = "tallanto_cards_daily"
DEFAULT_ORDER_BY = "id ASC"
DEFAULT_SELECT_FIELDS = (
    "id",
    "first_name",
    "last_name",
    "phone_mobile",
    "phone_work",
    "email1",
    "email2",
    "marital_status_c",
    "amo_id",
    "barcode",
    "type_client_c",
    "filial",
    "subject1_name",
    "subject2_name",
    "subject3_name",
    "subject4_name",
    "subject5_name",
    "subject6_name",
    "interests_c",
    "source",
    "date_entered",
    "date_modified",
    "assigned_user_id",
    "assigned_user_name",
)


@dataclass(frozen=True)
class TallantoCardsSyncConfig:
    timeline_db: Path
    out_root: Path
    allowed_root: Optional[Path] = None
    tenant_id: str = "foton"
    client: Optional[TallantoApiClient] = None
    tallanto_env_file: Optional[Path] = None
    select_fields: Optional[Sequence[str]] = None
    max_pages: int = 5
    safety_margin_seconds: int = 300
    actor: str = "tallanto_cards_sync"


def _checkpoint_path(out_root: Path) -> Path:
    return out_root / "tallanto_cards_sync_checkpoint.json"


def load_tallanto_cards_checkpoint(out_root: Path) -> Mapping[str, Any]:
    path = _checkpoint_path(out_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    if payload.get("schema_version") != TALLANTO_CARDS_SYNC_SCHEMA_VERSION:
        return {}
    return payload


def save_tallanto_cards_checkpoint(out_root: Path, state: Optional[Mapping[str, Any]]) -> None:
    path = _checkpoint_path(out_root)
    if not state:
        if path.exists():
            path.unlink()
        return
    _atomic_write_text(
        path,
        json.dumps(
            {"schema_version": TALLANTO_CARDS_SYNC_SCHEMA_VERSION, **dict(state)},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def universe_fingerprint(*, select_fields: Optional[Sequence[str]], order_by: str) -> str:
    # Block C: freezes the paginated Contact-module "universe" (field
    # selection + ordering -- the only two things that vary this fetch,
    # since there is no time-window filter, see module docstring) at the
    # start of a checkpoint cycle. A fingerprint mismatch (e.g. select_fields
    # changed in a later code revision) discards the old, incomplete
    # checkpoint and restarts at offset 0 instead of silently resuming a
    # different read into it.
    return stable_digest({"module": "Contact", "select_fields": list(select_fields or ()), "order_by": order_by})


def _fetch_contact_page_batch(
    client: TallantoApiClient,
    *,
    select_fields: Optional[Sequence[str]],
    order_by: str,
    start_offset: int,
    max_pages: int,
    expected_total_count: Optional[int] = None,
) -> tuple[list[Mapping[str, Any]], int, bool, int, int, Mapping[str, Any]]:
    """Bounded (<= max_pages), validated read of the Tallanto Contact module.

    Mirrors the pagination-completeness checks in
    tallanto_attendance_import._fetch_complete_entry_list (result_count/
    total_count consistency, no repeated offsets, monotonic next_offset) --
    same validation discipline, adapted to stop after max_pages pages and
    report a resume offset instead of looping unconditionally to
    completion. Raises ValueError (fail-loud) on any pagination-shape
    inconsistency rather than silently treating partial data as complete.

    Returns rows, pages, completion, resume offset, total count and the last
    page anchor used to validate a later resume.
    """
    rows: list[Mapping[str, Any]] = []
    offset = max(0, int(start_offset))
    total_count: Optional[int] = expected_total_count
    pages = 0
    seen_offsets: set[int] = set()
    last_anchor: Mapping[str, Any] = {}
    for _ in range(max(1, int(max_pages))):
        if offset in seen_offsets:
            raise ValueError(f"Tallanto Contact pagination repeated offset: {offset}")
        seen_offsets.add(offset)
        payload = client.get_entry_list(module="Contact", select_fields=select_fields, order_by=order_by, offset=offset)
        pages += 1
        page = payload.get("entry_list")
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise ValueError("Tallanto Contact page misses a valid entry_list")
        try:
            page_count = int(payload["result_count"])
            page_total = int(payload["total_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Tallanto Contact page misses result_count/total_count") from exc
        if page_count != len(page):
            raise ValueError("Tallanto Contact result_count mismatch")
        if total_count is None:
            total_count = page_total
        elif total_count != page_total:
            raise ValueError("Tallanto Contact total_count changed during pagination")
        rows.extend(page)
        last_anchor = contact_page_anchor(page, offset=offset, total_count=page_total, next_offset=payload.get("next_offset"))
        # NOTE: completeness must compare the ABSOLUTE position reached in
        # the full universe (offset + len(page)) against total_count, not
        # len(rows) -- rows only accumulates within *this bounded call*, so
        # on a resumed run (start_offset > 0) len(rows) alone would never
        # reach total_count even once the module has actually been read in
        # full across earlier runs.
        consumed = offset + len(page)
        if not page:
            return rows, pages, True, offset, page_total, last_anchor
        if consumed > total_count:
            raise ValueError("Tallanto Contact returned more rows than total_count")
        next_offset_raw = payload.get("next_offset")
        if consumed == total_count or next_offset_raw in (None, ""):
            if consumed != total_count:
                raise ValueError("Tallanto Contact pagination ended before total_count")
            return rows, pages, True, consumed, page_total, last_anchor
        try:
            next_offset = int(next_offset_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Tallanto Contact next_offset is invalid") from exc
        if next_offset <= offset:
            raise ValueError("Tallanto Contact next_offset did not increase")
        offset = next_offset
    if total_count is None:
        raise ValueError("Tallanto Contact pagination did not report total_count")
    return rows, pages, False, offset, total_count, last_anchor


def contact_page_anchor(
    page: Sequence[Mapping[str, Any]], *, offset: int, total_count: int, next_offset: object
) -> Mapping[str, Any]:
    return {
        "offset": int(offset),
        "total_count": int(total_count),
        "next_offset": None if next_offset in (None, "") else int(next_offset),
        "digest": stable_digest(list(page)),
    }


def _resume_anchor_matches(
    client: TallantoApiClient,
    *,
    select_fields: Optional[Sequence[str]],
    order_by: str,
    anchor: Mapping[str, Any],
) -> bool:
    if not anchor:
        return False
    offset = int(anchor.get("offset") or 0)
    payload = client.get_entry_list(
        module="Contact", select_fields=select_fields, order_by=order_by, offset=offset
    )
    page = payload.get("entry_list")
    if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
        return False
    try:
        current = contact_page_anchor(
            page,
            offset=offset,
            total_count=int(payload["total_count"]),
            next_offset=payload.get("next_offset"),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return current == dict(anchor)


_PHONE_RAW_KEYS = ("phone_mobile", "phone_work")
_EMAIL_RAW_KEYS = ("email1", "email2")


def _first_nonempty(raw: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = raw.get(key)
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return None


def map_raw_contact_to_snapshot_payload(raw: Mapping[str, Any], *, snapshot_at: str) -> Optional[Mapping[str, Any]]:
    """Adapts one raw Tallanto (SugarCRM-style) Contact record -- as returned
    by TallantoApiClient.get_entry_list(module="Contact") -- into the exact
    flat contract shape TallantoSnapshotNormalizer already consumes (the
    same field names scripts/normalize_tallanto_contacts.py produces from
    the bootstrap Contacts *.xls: tallanto_id/primary_phone/primary_email/
    display_name/...). This is a field-name adapter, not a second parser or
    a second identity-resolution path: the actual normalization, identity
    linking, and conflict detection all still happen inside
    TallantoSnapshotNormalizer / resolve_customer_identity_batches, reused
    unchanged.

    Returns None when the raw record has no usable id -- the caller must
    skip it (counted as unmatched), never fuzzy-match or link it by name;
    TallantoSnapshotNormalizer requires a non-empty entity_id and would
    otherwise raise for the whole batch.

    NOTE on idempotency: the returned payload deliberately does NOT include
    a "snapshot_at"/"captured_at" key set to "now". TallantoSnapshotNormalizer
    prioritizes those keys (over updated_at/created_at) for the emitted
    event's event_at, and the whole payload feeds the event's record_hash --
    a wall-clock "now" value there would make every daily run's hash differ
    even when the underlying contact is byte-for-byte unchanged, breaking
    "repeated import doesn't increase raw events". event_at instead comes
    from the contact's own Tallanto-side date_modified (stable across runs,
    only changes when the contact actually changes); `snapshot_at` is used
    only as a last-resort fallback when Tallanto provides no timestamp at
    all for a record.
    """
    entity_id = _first_nonempty(raw, ("id",))
    if not entity_id:
        return None
    phone = _first_nonempty(raw, _PHONE_RAW_KEYS)
    extra_phones = [
        value
        for key in _PHONE_RAW_KEYS
        if (value := _first_nonempty(raw, (key,))) and value != phone
    ]
    email = _first_nonempty(raw, _EMAIL_RAW_KEYS)
    extra_emails = [
        value
        for key in _EMAIL_RAW_KEYS
        if (value := _first_nonempty(raw, (key,))) and value != email
    ]
    first_name = _first_nonempty(raw, ("first_name",))
    last_name = _first_nonempty(raw, ("last_name",))
    display_name = " ".join(part for part in (first_name, last_name) if part) or None
    updated_at = _first_nonempty(raw, ("date_modified",)) or snapshot_at
    created_at = _first_nonempty(raw, ("date_entered",)) or updated_at
    return {
        "tallanto_id": entity_id,
        "display_name": display_name,
        "first_name": first_name,
        "last_name": last_name,
        "primary_phone": phone,
        "phone_extra": " | ".join(dict.fromkeys(extra_phones)) or None,
        "primary_email": email,
        "email_extra": " | ".join(dict.fromkeys(extra_emails)) or None,
        "parent_fio": _first_nonempty(raw, ("marital_status_c",)),
        "student_type": _first_nonempty(raw, ("type_client_c",)),
        "interests": _first_nonempty(raw, ("interests_c",)),
        "branch": _first_nonempty(raw, ("filial",)),
        "subjects": ", ".join(
            value
            for index in range(1, 7)
            if (value := _first_nonempty(raw, (f"subject{index}_name",)))
        )
        or None,
        "source": _first_nonempty(raw, ("source",)),
        "amo_contact_id": _first_nonempty(raw, ("amo_id",)),
        "barcode": _first_nonempty(raw, ("barcode",)),
        "responsible": _first_nonempty(raw, ("assigned_user_name", "assigned_user_id")),
        "created_at": created_at,
        "updated_at": updated_at,
        "match_class": "strong_unique" if (phone or email) else "unmatched",
    }


def run_tallanto_cards_sync(config: TallantoCardsSyncConfig) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    timeline_db = config.timeline_db.expanduser().resolve(strict=False)
    guard_customer_timeline_writable_path(timeline_db)
    out_root = config.out_root.expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)
    allowed_root = (
        config.allowed_root.expanduser().resolve(strict=False) if config.allowed_root is not None else out_root
    )

    select_fields = tuple(config.select_fields or DEFAULT_SELECT_FIELDS)
    checkpoint = load_tallanto_cards_checkpoint(out_root)
    fingerprint = universe_fingerprint(select_fields=select_fields, order_by=DEFAULT_ORDER_BY)
    if checkpoint.get("fingerprint") == fingerprint:
        start_offset = max(0, int(checkpoint.get("next_offset") or 0))
        carried_items = [item for item in (checkpoint.get("items") or ()) if isinstance(item, Mapping)]
        pages_before = max(0, int(checkpoint.get("pages_fetched") or 0))
    else:
        start_offset = 0
        carried_items = []
        pages_before = 0

    if config.client is not None:
        client = config.client
    elif config.tallanto_env_file is not None:
        client = _build_tallanto_client(config.tallanto_env_file.expanduser())
    else:
        raise ValueError("Tallanto cards sync requires tallanto_env_file")
    checkpoint_reset_reason: Optional[str] = None
    expected_total_count = (
        int(checkpoint.get("expected_total_count"))
        if checkpoint.get("fingerprint") == fingerprint and checkpoint.get("expected_total_count") is not None
        else None
    )
    if start_offset and not _resume_anchor_matches(
        client,
        select_fields=select_fields,
        order_by=DEFAULT_ORDER_BY,
        anchor=checkpoint.get("last_page_anchor") if isinstance(checkpoint.get("last_page_anchor"), Mapping) else {},
    ):
        start_offset = 0
        carried_items = []
        pages_before = 0
        expected_total_count = None
        checkpoint_reset_reason = "pagination_universe_changed"

    try:
        batch_rows, batch_pages, complete, resume_offset, total_count, last_page_anchor = _fetch_contact_page_batch(
            client,
            select_fields=select_fields,
            order_by=DEFAULT_ORDER_BY,
            start_offset=start_offset,
            max_pages=config.max_pages,
            expected_total_count=expected_total_count,
        )
    except (TallantoApiError, ValueError) as exc:
        # D3-style safe failure: never leak the raw exception body, never
        # touch the checkpoint or the timeline DB, never publish "latest" --
        # the source simply fails.
        return _finish(
            out_root,
            started=started,
            timeline_db=timeline_db,
            extra={
                "validation_ok": False,
                "source_failed": True,
                "failure_category": getattr(exc, "category", None) or "invalid_pagination",
                "checked": 0,
                "checked_with_id": 0,
                "skipped_missing_id": 0,
                "updated": 0,
                "unchanged": 0,
                "unmatched": 0,
                "conflict": 0,
                "cursor_time": None,
                "safety": _safety(staging_db_write=False),
            },
        )

    all_items = carried_items + list(batch_rows)
    total_pages = pages_before + batch_pages

    if not complete:
        save_tallanto_cards_checkpoint(
            out_root,
            {
                "fingerprint": fingerprint,
                "next_offset": resume_offset,
                "items": all_items,
                "pages_fetched": total_pages,
                "expected_total_count": total_count,
                "last_page_anchor": last_page_anchor,
            },
        )
        return _finish(
            out_root,
            started=started,
            timeline_db=timeline_db,
            extra={
                "validation_ok": False,
                "apply_blocked": True,
                "blocked_reason": "page_cap_hit",
                "complete": False,
                "checked": len(all_items),
                "checked_with_id": 0,
                "skipped_missing_id": 0,
                "updated": 0,
                "unchanged": 0,
                "unmatched": 0,
                "conflict": 0,
                "cursor_time": None,
                "checkpoint": {
                    "path": str(_checkpoint_path(out_root)),
                    "pages_fetched": total_pages,
                    "next_offset": resume_offset,
                    "checkpoint_reset_reason": checkpoint_reset_reason,
                    "note": "bounded checkpoint saved; the next run resumes from the persisted offset, not offset 0",
                },
                "safety": _safety(staging_db_write=False),
            },
        )

    identifiers = [str(item.get("id") or "").strip() for item in all_items]
    identifiers_with_id = [value for value in identifiers if value]
    if len(all_items) != total_count or len(set(identifiers_with_id)) != len(identifiers_with_id):
        save_tallanto_cards_checkpoint(out_root, None)
        return _finish(
            out_root,
            started=started,
            timeline_db=timeline_db,
            extra={
                "validation_ok": False,
                "apply_blocked": True,
                "blocked_reason": "pagination_universe_changed",
                "complete": False,
                "checked": len(all_items),
                "cursor_time": None,
                "safety": _safety(staging_db_write=False),
            },
        )

    # Keep the complete source checkpoint until the DB import succeeds.
    save_tallanto_cards_checkpoint(
        out_root,
        {
            "fingerprint": fingerprint,
            "next_offset": resume_offset,
            "items": all_items,
            "pages_fetched": total_pages,
            "expected_total_count": total_count,
            "last_page_anchor": last_page_anchor,
        },
    )

    # Full fixed universe read -- map and import via
    # the existing generic nightly_incremental + TallantoSnapshotNormalizer
    # contract (never a second parser/importer/identity engine).
    snapshot_at = started.isoformat()
    mapped_rows: list[Mapping[str, Any]] = []
    skipped_missing_id = 0
    for raw in all_items:
        mapped = map_raw_contact_to_snapshot_payload(raw, snapshot_at=snapshot_at)
        if mapped is None:
            skipped_missing_id += 1
            continue
        mapped_rows.append(mapped)

    if len(mapped_rows) != len(all_items):
        return _finish(
            out_root,
            started=started,
            timeline_db=timeline_db,
            extra={
                "validation_ok": False,
                "apply_blocked": True,
                "blocked_reason": "contacts_missing_stable_id",
                "complete": False,
                "checked": len(all_items),
                "checked_with_id": len(mapped_rows),
                "skipped_missing_id": skipped_missing_id,
                "updated": 0,
                "unchanged": 0,
                "unmatched": skipped_missing_id,
                "conflict": 0,
                "cursor_time": None,
                "safety": _safety(staging_db_write=False),
            },
        )

    source_path = out_root / "tallanto_cards_sources" / "tallanto_contacts_daily.jsonl"
    write_jsonl(source_path, mapped_rows)

    tallanto_ids = [str(row["tallanto_id"]) for row in mapped_rows]
    before_hashes = _tallanto_event_hashes(timeline_db, tenant_id=config.tenant_id, tallanto_ids=tallanto_ids)

    nightly_config = NightlyIncrementalConfig(
        timeline_db=timeline_db,
        allowed_root=allowed_root,
        tenant_id=config.tenant_id,
        journal_path=out_root / "tallanto_cards_sync_journal.jsonl",
        safety_margin_seconds=config.safety_margin_seconds,
        actor=config.actor,
        sources=(
            IncrementalSourceConfig(
                name="tallanto_cards_daily",
                source_system=TALLANTO_CARDS_SOURCE_SYSTEM,
                path=source_path,
                tenant_id=config.tenant_id,
                source_ref="tallanto:contacts:daily",
                normalizer="tallanto_snapshot",
                # See module docstring: this step reads the full current
                # Contact module every completed cycle, so there is no
                # source-side time window to advance through -- idempotent
                # upsert alone makes a repeat of unchanged rows a no-op.
                ignore_cursor=True,
            ),
        ),
    )
    imported = run_nightly_incremental(nightly_config)
    after_hashes = _tallanto_event_hashes(timeline_db, tenant_id=config.tenant_id, tallanto_ids=tallanto_ids)
    # Proof counts are per-CHECKED-CONTACT (matching "checked"/"unmatched"/
    # "conflict", all counted in the same unit), computed from the actual
    # persisted timeline_events row hash before vs. after this cycle's
    # import -- not from write_status_counts, which aggregates every
    # sub-record type (customer/links/opportunity/event/...) together and
    # would overcount relative to "how many contacts changed".
    updated = sum(1 for tid in tallanto_ids if before_hashes.get(tid) != after_hashes.get(tid))
    unchanged = len(tallanto_ids) - updated
    import_reports = [item for item in (imported.get("imports") or ()) if isinstance(item, Mapping)]
    normalized_counts = import_reports[0].get("normalized_counts") if import_reports else {}
    normalized_counts = normalized_counts if isinstance(normalized_counts, Mapping) else {}
    conflict_count = int(normalized_counts.get("conflicts") or 0)
    unmatched = skipped_missing_id + sum(1 for row in mapped_rows if row.get("match_class") == "unmatched")

    finished = datetime.now(timezone.utc)
    validation_ok = bool(imported.get("gate_passed"))
    if validation_ok:
        save_tallanto_cards_checkpoint(out_root, None)
    return _finish(
        out_root,
        started=started,
        timeline_db=timeline_db,
        extra={
            "validation_ok": validation_ok,
            "complete": validation_ok,
            "checked": len(all_items),
            "checked_with_id": len(mapped_rows),
            "skipped_missing_id": skipped_missing_id,
            "updated": updated,
            "unchanged": unchanged,
            "unmatched": unmatched,
            "conflict": conflict_count,
            "cursor_time": finished.isoformat() if validation_ok else None,
            "total_pages_read": total_pages,
            "checkpoint": {
                "path": str(_checkpoint_path(out_root)),
                "pending_endpoints": [] if validation_ok else [TALLANTO_CARDS_SOURCE_SYSTEM],
                "cleared": validation_ok,
            },
            "import": compact_import_report(imported),
            "safety": _safety(staging_db_write=True),
        },
    )


def _tallanto_event_hashes(db_path: Path, *, tenant_id: str, tallanto_ids: Sequence[str]) -> Mapping[str, str]:
    if not tallanto_ids or not db_path.exists():
        return {}
    placeholders = ",".join("?" for _ in tallanto_ids)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only = ON")
        rows = con.execute(
            f"""
            SELECT source_id, record_hash FROM timeline_events
            WHERE tenant_id = ? AND source_system = 'tallanto_snapshot' AND source_id IN ({placeholders})
            """,
            (tenant_id, *tallanto_ids),
        ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _finish(
    out_root: Path,
    *,
    started: datetime,
    timeline_db: Path,
    extra: Mapping[str, Any],
) -> Mapping[str, Any]:
    finished = datetime.now(timezone.utc)
    report = {
        "schema_version": TALLANTO_CARDS_SYNC_SCHEMA_VERSION,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "timeline_db": str(timeline_db),
        **dict(extra),
    }
    write_json(out_root / "tallanto_cards_sync_report.json", report)
    return report


def _safety(*, staging_db_write: bool) -> Mapping[str, bool]:
    return {
        "amo_write": False,
        "tallanto_write": False,
        "crm_write": False,
        "staging_db_write": staging_db_write,
        "network_calls": True,
    }


def compact_import_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "affected_customer_count": report.get("affected_customer_count"),
        "changed_customer_count": report.get("changed_customer_count"),
        "sources": report.get("sources"),
        "imports": [
            {
                "source_system": item.get("source_system"),
                "accepted_count": item.get("accepted_count"),
                "rejected_count": item.get("rejected_count"),
                "write_status_counts": item.get("write_status_counts"),
                "normalized_counts": item.get("normalized_counts"),
            }
            for item in report.get("imports", ())
            if isinstance(item, Mapping)
        ],
        "cursor_updates": report.get("cursor_updates"),
        "source_errors": report.get("source_errors"),
        "gate_passed": report.get("gate_passed"),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


__all__ = [
    "TALLANTO_CARDS_SYNC_SCHEMA_VERSION",
    "TALLANTO_CARDS_SOURCE_SYSTEM",
    "DEFAULT_SELECT_FIELDS",
    "TallantoCardsSyncConfig",
    "load_tallanto_cards_checkpoint",
    "save_tallanto_cards_checkpoint",
    "universe_fingerprint",
    "map_raw_contact_to_snapshot_payload",
    "run_tallanto_cards_sync",
]
