from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.customer_timeline import CustomerIdentity, CustomerTimelineSQLiteStore, IdentityStatus
from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    run_nightly_incremental,
    single_run_lock,
)


NOW = datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc)


def customer(customer_id: str = "customer:test-1") -> CustomerIdentity:
    return CustomerIdentity(
        tenant_id="foton",
        customer_id=customer_id,
        identity_status=IdentityStatus.STRONG,
        display_name="Тестовый клиент",
        primary_phone="+79161234567",
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=1,
        created_at=NOW,
        updated_at=NOW,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def base_config(tmp_path: Path, source_path: Path) -> NightlyIncrementalConfig:
    return NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="amo_updates",
                source_system="amocrm_snapshot",
                path=source_path,
                source_ref="test:amo_updates",
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
        safety_margin_seconds=60,
        lock_timeout_seconds=2,
    )


def seed_customer(tmp_path: Path, customer_id: str = "customer:test-1") -> None:
    with CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path) as store:
        store.upsert_customer(customer(customer_id))


def event_count(tmp_path: Path) -> int:
    import sqlite3

    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        return int(con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0])


def event_counts_by_match_status(tmp_path: Path) -> dict[str, int]:
    import sqlite3

    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        return {
            str(status): int(count)
            for status, count in con.execute("SELECT match_status, COUNT(*) FROM timeline_events GROUP BY match_status")
        }


def conflict_count(tmp_path: Path) -> int:
    import sqlite3

    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        return int(con.execute("SELECT COUNT(*) FROM timeline_conflicts WHERE conflict_type='pending_attribution'").fetchone()[0])


def test_nightly_incremental_uses_overlap_and_repeat_adds_no_duplicates(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "amo_updates.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "source_id": "lead-1",
                "customer_id": "customer:test-1",
                "event_type": "amo_deal_stage",
                "created_at": "2026-06-21T10:00:00+00:00",
                "updated_at": "2026-06-21T10:00:00+00:00",
                "summary": "Сделка создана",
            },
            {
                "source_id": "lead-2",
                "customer_id": "customer:test-1",
                "event_type": "amo_deal_stage",
                "created_at": "2026-06-21T10:05:00+00:00",
                "updated_at": "2026-06-21T10:05:00+00:00",
                "summary": "Сделка обновлена",
            },
        ],
    )

    first = run_nightly_incremental(base_config(tmp_path, source_path))
    second = run_nightly_incremental(base_config(tmp_path, source_path))

    assert first["changed_customer_ids"] == ["customer:test-1"]
    assert second["changed_customer_ids"] == []
    assert event_count(tmp_path) == 2
    assert second["imports"][0]["write_status_counts"]["duplicate"] >= 1
    cursor = second["cursor_updates"][0]
    assert cursor["last_cursor_ts"] == "2026-06-21T10:04:00+00:00"


def test_nightly_incremental_uses_updated_at_not_only_created_at(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "amo_updates.jsonl"
    with CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path) as store:
        store.upsert_ingestion_cursor(
            "foton",
            "amocrm_snapshot",
            last_cursor_ts=datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc),
        )
    write_jsonl(
        source_path,
        [
            {
                "source_id": "lead-old-created",
                "customer_id": "customer:test-1",
                "event_type": "amo_deal_stage",
                "created_at": "2026-06-20T09:00:00+00:00",
                "updated_at": "2026-06-21T10:10:00+00:00",
                "summary": "Старая сделка обновлена ночью",
            }
        ],
    )

    report = run_nightly_incremental(base_config(tmp_path, source_path))

    assert report["sources"][0]["rows_selected"] == 1
    assert report["changed_customer_ids"] == ["customer:test-1"]
    assert event_count(tmp_path) == 1


def test_nightly_incremental_uses_mango_normalizer_for_load_and_import(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mango_calls.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "call_id": "provider:call-1",
                "customer_id": "customer:test-1",
                "match_class": "strong_unique",
                "identity_authority": "existing_timeline_increment",
                "identity_resolved_by_increment": True,
                "phone": "+79161234567",
                "call_at": "2026-06-21T10:05:00+00:00",
                "updated_at": "2026-06-21T10:05:00+00:00",
                "summary": "Клиент уточнил расписание.",
                "allowed_for_bot": False,
                "requires_manager_review": True,
            }
        ],
    )
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="mango_calls",
                source_system="mango_processed_summary",
                path=source_path,
                source_ref="test:mango_calls",
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
        safety_margin_seconds=60,
        lock_timeout_seconds=2,
    )

    first = run_nightly_incremental(config)
    second = run_nightly_incremental(config)

    with CustomerTimelineSQLiteStore.open_read_only(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path) as store:
        events = store.list_events_by_customer("foton", "customer:test-1", limit=10)["items"]
        chunks = store.search_timeline("foton", "расписание", scopes=("bot_context",), mode="fallback", limit=10)["items"]
        summary = store.summary()
    assert first["changed_customer_ids"] == ["customer:test-1"]
    assert second["changed_customer_ids"] == []
    assert summary["counts"]["customer_identities"] == 1
    assert events[0]["event_type"] == "mango_call"
    assert events[0]["source_system"] == "mango_processed_summary"
    chunk_record = chunks[0]["record"]
    assert chunk_record["allowed_for_bot"] is False
    assert chunk_record["requires_manager_review"] is True


def test_nightly_incremental_mango_ambiguous_imports_event_and_conflict_without_customer(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mango_calls.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "call_id": "provider:call-ambiguous",
                "match_class": "ambiguous",
                "identity_authority": "existing_timeline_increment",
                "identity_resolved_by_increment": True,
                "identity_resolution_reason": "multiple_existing_customers",
                "phone": "+79161234567",
                "call_at": "2026-06-21T10:05:00+00:00",
                "updated_at": "2026-06-21T10:05:00+00:00",
                "summary": "Клиент уточнил расписание.",
                "allowed_for_bot": False,
                "requires_manager_review": True,
            }
        ],
    )
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="mango_calls",
                source_system="mango_processed_summary",
                path=source_path,
                source_ref="test:mango_calls",
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
        safety_margin_seconds=60,
        lock_timeout_seconds=2,
    )

    report = run_nightly_incremental(config)

    assert report["source_errors"] == []
    assert report["changed_customer_ids"] == []
    assert event_counts_by_match_status(tmp_path) == {"ambiguous": 1}
    assert conflict_count(tmp_path) == 1


def test_nightly_incremental_unavailable_source_skips_and_alerts_after_two_failures(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    missing = tmp_path / "missing.jsonl"
    config = base_config(tmp_path, missing)

    first = run_nightly_incremental(config)
    second = run_nightly_incremental(config)

    assert first["source_errors"] == [{"source": "amo_updates", "reason": "source_unavailable"}]
    assert second["source_errors"] == [{"source": "amo_updates", "reason": "source_unavailable"}]
    with CustomerTimelineSQLiteStore.open_read_only(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path) as store:
        cursor = store.get_ingestion_cursor("foton", "amocrm_snapshot")
    assert cursor is not None
    assert cursor.metadata["consecutive_failures"] == 2
    assert cursor.metadata["alert"] is True


def test_single_run_lock_waits_for_existing_holder(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with single_run_lock(db_path, timeout_seconds=2):
            entered.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(timeout=1)
    started = time.monotonic()
    result: dict[str, float] = {}

    def waiter() -> None:
        with single_run_lock(db_path, timeout_seconds=2) as info:
            result["waited"] = float(info["waited_seconds"])

    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()
    time.sleep(0.15)
    release.set()
    thread.join(timeout=2)
    waiter_thread.join(timeout=2)

    assert time.monotonic() - started >= 0.1
    assert result["waited"] > 0
