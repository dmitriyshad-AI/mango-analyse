from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.customer_profile.contracts import ProfileSnapshot
from mango_mvp.customer_profile.store import CustomerProfileSQLiteStore, sha256_file
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore

import scripts.refresh_customer_profiles as refresh


NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


def test_since_rebuilds_only_customers_with_created_at_after_since(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    store = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    try:
        _add_customer(store, "cust-old")
        _add_customer(store, "cust-new")
        _add_event(store, "cust-old", event_id="old", created_at=NOW - timedelta(hours=2))
        _add_event(store, "cust-new", event_id="new", created_at=NOW - timedelta(minutes=10))
    finally:
        store.close()
    profiles_db = tmp_path / "profiles.sqlite"
    _seed_profiles(profiles_db, timeline_db, ("cust-old", "cust-new"), build_id="old-build")

    report = refresh.refresh_since(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=None,
        tenant_id="foton",
        since=NOW - timedelta(hours=1),
        build_id="new-build",
    )

    assert report["selected_customer_ids"] == ["cust-new"]
    assert report["build"]["profiles_built"] == 1
    assert _profile_build_ids(profiles_db) == {"cust-old": "old-build", "cust-new": "new-build"}


def test_detect_quiet_dialogs_and_journal_refresh_reports_bad_active_unmatched(tmp_path: Path) -> None:
    timeline_db = _timeline_db(tmp_path)
    store = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    try:
        _add_customer(store, "cust-telegram")
        _add_customer(store, "cust-session")
        _add_customer(store, "cust-active")
        _add_customer(store, "cust-ambiguous-a")
        _add_customer(store, "cust-ambiguous-b")
        _add_identity_link(store, "cust-telegram", "telegram_user_id", "100")
        _add_identity_link(store, "cust-session", "channel_session_id", "telegram:200")
        _add_identity_link(store, "cust-active", "telegram_user_id", "101")
        _add_identity_link(store, "cust-ambiguous-a", "channel_session_id", "telegram:777")
        _add_identity_link(store, "cust-ambiguous-b", "telegram_user_id", "777")
        _add_event(store, "cust-telegram", event_id="telegram", created_at=NOW - timedelta(days=1))
        _add_event(store, "cust-session", event_id="session", created_at=NOW - timedelta(days=1))
        _add_event(store, "cust-active", event_id="active", created_at=NOW - timedelta(days=1))
        _add_event(store, "cust-ambiguous-a", event_id="amb-a", created_at=NOW - timedelta(days=1))
        _add_event(store, "cust-ambiguous-b", event_id="amb-b", created_at=NOW - timedelta(days=1))
    finally:
        store.close()
    profiles_db = tmp_path / "profiles.sqlite"
    _seed_profiles(
        profiles_db,
        timeline_db,
        ("cust-telegram", "cust-session", "cust-active", "cust-ambiguous-a", "cust-ambiguous-b"),
        build_id="old-build",
    )
    journal_path = tmp_path / "journal.copy.jsonl"
    _write_jsonl(
        journal_path,
        [
            {"profile_id": "journal-telegram", "chat_id": "100", "created_at": (NOW - timedelta(minutes=40)).isoformat()},
            {"profile_id": "journal-session", "chat_id": "200", "created_at": (NOW - timedelta(minutes=31)).isoformat()},
            {"profile_id": "journal-active", "chat_id": "101", "created_at": (NOW - timedelta(minutes=5)).isoformat()},
            {"profile_id": "journal-unmatched", "chat_id": "999", "created_at": (NOW - timedelta(minutes=45)).isoformat()},
            {"profile_id": "journal-ambiguous", "chat_id": "777", "created_at": (NOW - timedelta(minutes=45)).isoformat()},
            {"profile_id": "bad-ts", "chat_id": "300", "created_at": "not-an-iso-date"},
            {"profile_id": "", "chat_id": "301", "created_at": (NOW - timedelta(minutes=45)).isoformat()},
            "{not json",
        ],
    )

    rows, bad_rows = refresh.read_journal_jsonl(journal_path)
    quiet_pairs = refresh.detect_quiet_dialogs(rows, NOW, quiet_minutes=30)
    report = refresh.refresh_from_journal(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=None,
        tenant_id="foton",
        journal_path=journal_path,
        now_utc=NOW,
        quiet_minutes=30,
        build_id="journal-build",
    )

    assert bad_rows == 1
    assert ("journal-telegram", "100") in quiet_pairs
    assert ("journal-session", "200") in quiet_pairs
    assert ("journal-active", "101") not in quiet_pairs
    assert report["journal_rows_bad"] == 1
    assert report["journal_rows_ignored"] == 2
    assert report["selected_customer_ids"] == ["cust-telegram", "cust-session"]
    assert [item["chat_id"] for item in report["unmatched_quiet_pairs"]] == ["999"]
    assert [item["chat_id"] for item in report["ambiguous_quiet_pairs"]] == ["777"]
    assert _profile_build_ids(profiles_db) == {
        "cust-ambiguous-a": "old-build",
        "cust-ambiguous-b": "old-build",
        "cust-active": "old-build",
        "cust-session": "journal-build",
        "cust-telegram": "journal-build",
    }


def test_quiet_minutes_default_is_30_and_cli_override_wins() -> None:
    rows = [
        {"profile_id": "journal-active", "chat_id": "101", "created_at": (NOW - timedelta(minutes=20)).isoformat()},
    ]
    base_args = [
        "--timeline-db",
        "timeline.sqlite",
        "--profiles-db",
        "profiles.sqlite",
        "--from-journal",
        "journal.copy.jsonl",
    ]

    default_args = refresh.build_parser().parse_args(base_args)
    override_args = refresh.build_parser().parse_args([*base_args, "--quiet-minutes", "15"])

    assert refresh.DEFAULT_QUIET_MINUTES == 30
    assert default_args.quiet_minutes == 30
    assert override_args.quiet_minutes == 15
    assert refresh.detect_quiet_dialogs(rows, NOW) == []
    assert refresh.detect_quiet_dialogs(rows, NOW, quiet_minutes=15) == [("journal-active", "101")]


def test_refresh_customer_profiles_does_not_import_or_use_live_draft_module() -> None:
    source_path = Path(refresh.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            forbidden_modules.append(node.module)

    assert "mango_mvp.integrations.draft_loop" not in forbidden_modules
    assert "scripts.run_amo_wappi_draft_loop" not in forbidden_modules
    assert "mango_mvp.integrations.draft_loop" not in source
    assert "run_amo_wappi_draft_loop" not in source


def _timeline_db(tmp_path: Path) -> Path:
    return tmp_path / "customer_timeline.sqlite"


def _add_customer(store: CustomerTimelineSQLiteStore, customer_id: str) -> None:
    store.upsert_customer(
        CustomerIdentity(
            tenant_id="foton",
            customer_id=customer_id,
            identity_status=IdentityStatus.STRONG,
            display_name=customer_id,
            source_ref=f"test:{customer_id}",
            first_seen_at=NOW,
            last_seen_at=NOW,
            created_at=NOW,
            updated_at=NOW,
        )
    )


def _add_identity_link(store: CustomerTimelineSQLiteStore, customer_id: str, link_type: str, link_value: str) -> None:
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer_id,
            link_type=link_type,
            link_value=link_value,
            source_system="test",
            source_ref=f"test:{link_type}:{link_value}",
            match_class=IdentityMatchClass.INFERRED,
            confidence=0.8,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )


def _add_event(store: CustomerTimelineSQLiteStore, customer_id: str, *, event_id: str, created_at: datetime) -> None:
    store.upsert_event(
        TimelineEvent(
            tenant_id="foton",
            customer_id=customer_id,
            event_type=TimelineEventType.SYSTEM_NOTE,
            event_at=created_at,
            source_system="test",
            source_id=event_id,
            source_ref=f"test:{event_id}",
            direction=TimelineDirection.SYSTEM,
            summary=event_id,
            created_at=created_at,
        )
    )


def _seed_profiles(profiles_db: Path, timeline_db: Path, profile_ids: tuple[str, ...], *, build_id: str) -> None:
    with CustomerProfileSQLiteStore(profiles_db) as store:
        store.replace_profiles(
            build_id=build_id,
            built_at=NOW - timedelta(days=1),
            timeline_db_path=timeline_db,
            timeline_db_sha256=sha256_file(timeline_db),
            profiles=[
                ProfileSnapshot(
                    profile_id=profile_id,
                    tenant_id="foton",
                    display_name=profile_id,
                    source_event_count=1,
                    last_event_at=NOW - timedelta(days=1),
                )
                for profile_id in profile_ids
            ],
            fields=(),
            notes="seed",
        )


def _profile_build_ids(profiles_db: Path) -> dict[str, str]:
    con = sqlite3.connect(profiles_db)
    try:
        rows = con.execute("SELECT profile_id, build_id FROM customer_profiles ORDER BY profile_id").fetchall()
    finally:
        con.close()
    return {str(profile_id): str(build_id) for profile_id, build_id in rows}


def _write_jsonl(path: Path, rows: list[dict[str, object] | str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
