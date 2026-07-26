from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.ids import normalize_key, stable_digest
from mango_mvp.customer_timeline.wappi_history_import import (
    WAPPI_EXACT_AMO_AUTHORITIES,
    WAPPI_RESOLVED_LINK_STATUSES,
    WAPPI_TECHNICAL_LINK_STATUSES,
    _widget_linkage_status,
    load_wappi_widget_links,
)


BREAKDOWN_SCHEMA_VERSION = "wappi_unmatched_link_breakdown_v1"
EXCLUSION_CATEGORIES = ("employee", "test", "system")
BREAKDOWN_ROW_FIELDS = (
    "channel",
    "profile_id",
    "chat_id_digest",
    "contact_id_present",
    "lead_count",
    "status",
    "resolution_source",
    "reason",
)


def load_wappi_chat_exclusions(path: Path | None) -> Mapping[tuple[str, str], str]:
    """Read an explicit employee/test/system chat exclusion list (BLOK C1).

    The file is optional. A missing file yields zero exclusions: every chat then
    falls through to `_conclusive_reason`, so an absent stoplist can never silently
    drop a chat from the count -- it only means the chat still needs a conclusive
    reason instead of being pre-classified as excluded. Never network I/O.
    """
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Wappi chat exclusion file must be a JSON object keyed by category")
    result: dict[tuple[str, str], str] = {}
    for category in EXCLUSION_CATEGORIES:
        entries = payload.get(category) or ()
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            raise ValueError(f"Wappi chat exclusion category {category!r} must be a list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"Wappi chat exclusion entries must be objects: {entry!r}")
            channel = normalize_key(entry.get("channel"), "channel")
            chat_id = str(entry.get("chat_id") or "").strip()
            if not chat_id:
                raise ValueError("Wappi chat exclusion entry requires a non-empty chat_id")
            key = (channel, chat_id)
            if key in result:
                raise ValueError(f"Wappi chat exclusion id listed more than once: {key!r}")
            result[key] = category
    return result


def _conclusive_reason(status: str, resolution_source: str, lead_ids: Sequence[str] = ()) -> str:
    """Map a locally-cached `wappi_amo_links` row to one specific, non-generic reason.

    `wappi_amo_links.status` persists the *raw* widget-lookup outcome written by
    `collect_wappi_widget_links` -- "resolved", "missing", "conflict", "candidate",
    or a technical failure code -- never the resolved_contact_only/resolved_one_lead/
    resolved_multiple_leads split used for reporting (BLOK A1: the real cache stores
    bare `status='resolved'`, so counting only the three-way split against it silently
    gave linked=0). Raw "resolved" is expanded here via the existing canonical
    `_widget_linkage_status`, the same normalizer `collect_wappi_widget_links` itself
    uses, so there is no second, parallel status-mapping table to drift out of sync.

    Never returns a bare "unmatched"/"pending" bucket: every branch names the exact
    technical or business condition, per BLOK C1 ("оставшиеся дать по конечным
    причинам"). `WAPPI_RESOLVED_LINK_STATUSES` is the only bucket ever counted as
    linked/strong; `candidate`/`conflict`/technical statuses never are (см. тест
    test_breakdown_never_treats_candidate_or_conflict_as_linked).
    """
    if status == "resolved":
        status = _widget_linkage_status(status, lead_ids)
    if status in WAPPI_RESOLVED_LINK_STATUSES:
        authority = resolution_source if resolution_source in WAPPI_EXACT_AMO_AUTHORITIES else resolution_source or "unknown_source"
        return f"linked_{status}_via_{authority}"
    if status == "candidate":
        return "candidate_awaiting_amo_talk_confirmation"
    if status == "conflict":
        return "conflicting_amo_relation_manual_review_required"
    if status in WAPPI_TECHNICAL_LINK_STATUSES:
        return f"technical_lookup_failure_{status}_retry_needed"
    if status == "missing":
        return "no_amo_contact_found_via_widget_lookup"
    return f"unrecognized_link_status_{status or 'empty'}"


def build_wappi_unmatched_breakdown(
    *,
    widget_link_db: Path,
    exclusions_path: Path | None = None,
) -> Mapping[str, Any]:
    """Offline, read-only inventory + conclusive-reason breakdown of every personal
    Wappi chat already checked against the AMO widget contract (BLOK C1).

    Zero network calls: only the locally persisted `wappi_amo_links` cache
    (`load_wappi_widget_links`, offline by construction) and an optional local
    employee/test/system stoplist are read. Chats that were never persisted into
    `wappi_amo_links` (e.g. because the raw chat catalogue itself was never fetched)
    are out of scope for this function -- that gap requires a live Wappi
    `list_chats`/AMO widget call, which is an owner-gated step, not something this
    function performs.
    """
    links = load_wappi_widget_links(widget_link_db)
    exclusions = load_wappi_chat_exclusions(exclusions_path)
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    linked_count = 0
    for (channel, profile_id, chat_id), link in sorted(links.items()):
        status = str(link.get("status") or "")
        lead_ids = tuple(link.get("lead_ids") or ())
        resolution_source = str(link.get("resolution_source") or "")
        exclusion = exclusions.get((channel, chat_id))
        if exclusion:
            reason = f"excluded_{exclusion}"
            exclusion_counts[exclusion] += 1
        else:
            reason = _conclusive_reason(status, resolution_source, lead_ids)
            # Single source of truth for "linked": whatever `_conclusive_reason`
            # classified as linked, not a second independent status-set check that
            # can silently drift out of sync with it (that drift was BLOK A1's bug).
            if reason.startswith("linked_"):
                linked_count += 1
        reason_counts[reason] += 1
        rows.append(
            {
                "channel": channel,
                "profile_id": profile_id,
                "chat_id_digest": stable_digest({"chat_id": chat_id})[:16],
                "contact_id_present": bool(link.get("contact_id")),
                "lead_count": len(lead_ids),
                "status": status,
                "resolution_source": resolution_source,
                "reason": reason,
            }
        )
    excluded_total = sum(exclusion_counts.values())
    return {
        "schema_version": BREAKDOWN_SCHEMA_VERSION,
        "catalog_scope": "local_widget_link_cache",
        "cached_chats_total": len(rows),
        # Backward-compatible alias. It is not the full live Wappi catalogue.
        "chats_total": len(rows),
        "linked": linked_count,
        "excluded_total": excluded_total,
        "excluded_by_category": dict(sorted(exclusion_counts.items())),
        "remaining_needs_conclusive_reason": len(rows) - linked_count - excluded_total,
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": tuple(rows),
    }


def write_wappi_unmatched_breakdown(out_dir: Path, report: Mapping[str, Any]) -> Mapping[str, str]:
    """Persist the per-chat CSV (digested chat ids only, no raw ids/names/phones) and
    an aggregate-only JSON summary. Row-level output must stay under `.codex_local`,
    matching the existing `write_hint_pack` convention.
    """
    target = Path(out_dir).expanduser().resolve(strict=False)
    if ".codex_local" not in target.parts:
        raise ValueError("Wappi unmatched breakdown rows must stay under .codex_local")
    target.mkdir(parents=True, exist_ok=True)
    rows_path = target / "wappi_unmatched_breakdown.csv"
    summary_path = target / "wappi_unmatched_breakdown_summary.json"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BREAKDOWN_ROW_FIELDS))
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(row)
    rows_path.chmod(0o600)
    summary = {key: value for key, value in report.items() if key != "rows"}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.chmod(0o600)
    return {"rows_csv": str(rows_path), "summary_json": str(summary_path)}
