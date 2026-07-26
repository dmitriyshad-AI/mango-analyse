from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.wappi_history_import import (
    _ensure_wappi_widget_link_schema,
    load_wappi_widget_links,
)
from mango_mvp.customer_timeline.wappi_unmatched_breakdown import (
    build_wappi_unmatched_breakdown,
    load_wappi_chat_exclusions,
    write_wappi_unmatched_breakdown,
)


def _seed_link(
    db_path: Path,
    *,
    channel: str,
    profile_id: str,
    chat_id: str,
    status: str,
    contact_id: str = "",
    lead_ids: tuple[str, ...] = (),
    resolution_source: str = "wappi_widget",
) -> None:
    con = sqlite3.connect(db_path)
    try:
        _ensure_wappi_widget_link_schema(con)
        con.execute(
            """
            INSERT INTO wappi_amo_links
              (channel, profile_id, chat_id, contact_id, lead_ids_json, status, checked_at,
               response_sha256, resolution_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel,
                profile_id,
                chat_id,
                contact_id or None,
                json.dumps(list(lead_ids)),
                status,
                "2026-07-26T00:00:00+00:00",
                "deadbeef",
                resolution_source,
            ),
        )
        con.commit()
    finally:
        con.close()


def test_breakdown_never_treats_candidate_or_conflict_as_linked(tmp_path: Path) -> None:
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-candidate", status="candidate", contact_id="111", lead_ids=("9001",), resolution_source="amo_event_sequence_candidate")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-conflict", status="conflict", contact_id="112", lead_ids=("9002",), resolution_source="wappi_widget_conflict")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-missing", status="missing")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-auth", status="auth_error")

    report = build_wappi_unmatched_breakdown(widget_link_db=db_path)

    assert report["catalog_scope"] == "local_widget_link_cache"
    assert report["cached_chats_total"] == 4
    assert report["chats_total"] == 4
    assert report["linked"] == 0
    reasons = report["reason_counts"]
    assert reasons["candidate_awaiting_amo_talk_confirmation"] == 1
    assert reasons["conflicting_amo_relation_manual_review_required"] == 1
    assert reasons["no_amo_contact_found_via_widget_lookup"] == 1
    assert reasons["technical_lookup_failure_auth_error_retry_needed"] == 1
    # No reason string in this report is a bare/generic bucket.
    for reason in reasons:
        assert reason not in {"unmatched", "pending", "pending_attribution"}
    # No row was silently dropped: every row has a specific, non-empty reason.
    assert sum(reasons.values()) == report["chats_total"]


def test_breakdown_counts_only_resolved_statuses_as_linked(tmp_path: Path) -> None:
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-1", status="resolved_one_lead", contact_id="200", lead_ids=("9101",), resolution_source="wappi_amo_widget")
    _seed_link(db_path, channel="max", profile_id="p2", chat_id="c-2", status="resolved_contact_only", contact_id="201", resolution_source="amo_talk_authoritative")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-3", status="resolved_multiple_leads", contact_id="202", lead_ids=("9102", "9103"), resolution_source="wappi_amo_widget")

    report = build_wappi_unmatched_breakdown(widget_link_db=db_path)

    assert report["linked"] == 3
    assert report["remaining_needs_conclusive_reason"] == 0
    assert report["reason_counts"]["linked_resolved_one_lead_via_wappi_amo_widget"] == 1
    assert report["reason_counts"]["linked_resolved_contact_only_via_amo_talk_authoritative"] == 1
    assert report["reason_counts"]["linked_resolved_multiple_leads_via_wappi_amo_widget"] == 1


def test_breakdown_treats_raw_resolved_status_as_linked_by_lead_count(tmp_path: Path) -> None:
    """BLOK A1 regression guard: the real `wappi_amo_links` cache persists the raw
    widget-lookup outcome `status='resolved'` (see `collect_wappi_widget_links` in
    wappi_history_import.py), never the resolved_contact_only/resolved_one_lead/
    resolved_multiple_leads split used only for reporting. Before the fix, raw
    "resolved" fell through to "unrecognized_link_status_resolved" and every real
    linked chat was silently excluded from `linked` (host cache: linked=0 instead
    of 2623)."""
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="raw-contact-only", status="resolved", contact_id="500", lead_ids=(), resolution_source="wappi_widget")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="raw-one-lead", status="resolved", contact_id="501", lead_ids=("9301",), resolution_source="wappi_widget")
    _seed_link(db_path, channel="max", profile_id="p2", chat_id="raw-multi-lead", status="resolved", contact_id="502", lead_ids=("9302", "9303"), resolution_source="wappi_widget")

    report = build_wappi_unmatched_breakdown(widget_link_db=db_path)

    assert report["chats_total"] == 3
    assert report["linked"] == 3
    assert report["remaining_needs_conclusive_reason"] == 0
    assert report["reason_counts"]["linked_resolved_contact_only_via_wappi_widget"] == 1
    assert report["reason_counts"]["linked_resolved_one_lead_via_wappi_widget"] == 1
    assert report["reason_counts"]["linked_resolved_multiple_leads_via_wappi_widget"] == 1
    # Raw status is preserved verbatim in the row (never rewritten to the
    # normalized substatus), and no raw chat id ever appears in a row.
    for row in report["rows"]:
        assert row["status"] == "resolved"
        assert "raw-" not in json.dumps(row)


def test_breakdown_excludes_employee_test_system_explicitly_and_not_in_remainder(tmp_path: Path) -> None:
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="staff-1", status="missing")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="qa-bot", status="missing")
    _seed_link(db_path, channel="max", profile_id="p2", chat_id="system-hook", status="missing")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="real-lead", status="missing")

    exclusions_path = tmp_path / "exclusions.json"
    exclusions_path.write_text(
        json.dumps(
            {
                "employee": [{"channel": "telegram", "chat_id": "staff-1"}],
                "test": [{"channel": "telegram", "chat_id": "qa-bot"}],
                "system": [{"channel": "max", "chat_id": "system-hook"}],
            }
        ),
        encoding="utf-8",
    )

    report = build_wappi_unmatched_breakdown(widget_link_db=db_path, exclusions_path=exclusions_path)

    assert report["excluded_total"] == 3
    assert report["excluded_by_category"] == {"employee": 1, "system": 1, "test": 1}
    assert report["remaining_needs_conclusive_reason"] == 1
    assert report["reason_counts"]["excluded_employee"] == 1
    assert report["reason_counts"]["excluded_test"] == 1
    assert report["reason_counts"]["excluded_system"] == 1
    # The one genuine remaining chat still gets its own conclusive (non-excluded) reason.
    assert report["reason_counts"]["no_amo_contact_found_via_widget_lookup"] == 1


def test_missing_exclusions_file_fails_open_to_conclusive_reason_not_silent_drop(tmp_path: Path) -> None:
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-1", status="missing")

    report = build_wappi_unmatched_breakdown(widget_link_db=db_path, exclusions_path=tmp_path / "does_not_exist.json")

    assert report["excluded_total"] == 0
    assert report["chats_total"] == 1
    assert report["remaining_needs_conclusive_reason"] == 1


def test_load_wappi_chat_exclusions_rejects_duplicate_chat_id(tmp_path: Path) -> None:
    exclusions_path = tmp_path / "exclusions.json"
    exclusions_path.write_text(
        json.dumps(
            {
                "employee": [{"channel": "telegram", "chat_id": "dup-1"}],
                "test": [{"channel": "telegram", "chat_id": "dup-1"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="listed more than once"):
        load_wappi_chat_exclusions(exclusions_path)


def test_breakdown_is_idempotent_and_read_only(tmp_path: Path) -> None:
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-1", status="missing")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-2", status="resolved_one_lead", contact_id="300", lead_ids=("9201",), resolution_source="wappi_amo_widget")

    before_bytes = db_path.read_bytes()
    first = build_wappi_unmatched_breakdown(widget_link_db=db_path)
    second = build_wappi_unmatched_breakdown(widget_link_db=db_path)
    after_bytes = db_path.read_bytes()

    assert first == second
    assert before_bytes == after_bytes


def test_load_wappi_widget_links_is_strictly_read_only_and_leaves_file_untouched(tmp_path: Path) -> None:
    """BLOK A2: `load_wappi_widget_links` must never CREATE/ALTER/commit against the
    cache file. Hash *and* mtime/size are checked before/after -- either alone could
    pass by accident (e.g. a no-op transaction that still bumps mtime, or a rewrite
    that happens to reproduce the same bytes)."""
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-1", status="resolved", contact_id="700", lead_ids=("9501",), resolution_source="wappi_widget")
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="c-2", status="missing")

    before_hash = hashlib.sha256(db_path.read_bytes()).hexdigest()
    before_stat = db_path.stat()

    first = load_wappi_widget_links(db_path)
    second = load_wappi_widget_links(db_path)
    build_wappi_unmatched_breakdown(widget_link_db=db_path)

    after_hash = hashlib.sha256(db_path.read_bytes()).hexdigest()
    after_stat = db_path.stat()

    assert first == second
    assert first[("telegram", "p1", "c-1")]["status"] == "resolved"
    assert before_hash == after_hash
    assert before_stat.st_mtime_ns == after_stat.st_mtime_ns
    assert before_stat.st_size == after_stat.st_size


def test_load_wappi_widget_links_missing_table_raises_diagnostic_error_not_silent_create(tmp_path: Path) -> None:
    """BLOK A2: a file that exists but has no `wappi_amo_links` table (wrong path,
    unrelated sqlite file) must fail loudly, never fall back to silently CREATE
    TABLE-ing an empty cache -- that would both violate read-only and hide a
    misconfiguration behind a quietly-empty report."""
    db_path = tmp_path / "not_a_link_cache.sqlite"
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        con.commit()
    finally:
        con.close()
    before_bytes = db_path.read_bytes()

    with pytest.raises(ValueError, match="wappi_amo_links"):
        load_wappi_widget_links(db_path)
    with pytest.raises(ValueError, match="wappi_amo_links"):
        build_wappi_unmatched_breakdown(widget_link_db=db_path)

    assert db_path.read_bytes() == before_bytes
    with sqlite3.connect(db_path) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "wappi_amo_links" not in tables


def test_write_wappi_unmatched_breakdown_requires_codex_local_and_omits_raw_chat_id(tmp_path: Path) -> None:
    db_path = tmp_path / "wappi_amo_links.sqlite"
    _seed_link(db_path, channel="telegram", profile_id="p1", chat_id="super-secret-chat-id", status="missing")
    report = build_wappi_unmatched_breakdown(widget_link_db=db_path)

    with pytest.raises(ValueError, match=r"\.codex_local"):
        write_wappi_unmatched_breakdown(tmp_path / "exports", report)

    paths = write_wappi_unmatched_breakdown(tmp_path / ".codex_local" / "wappi_breakdown", report)
    raw_csv_text = Path(paths["rows_csv"]).read_text(encoding="utf-8")
    assert "super-secret-chat-id" not in raw_csv_text
    rows = list(csv.DictReader(Path(paths["rows_csv"]).open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["reason"] == "no_amo_contact_found_via_widget_lookup"
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert "rows" not in summary
    assert summary["chats_total"] == 1
