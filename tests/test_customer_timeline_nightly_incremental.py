from __future__ import annotations

import json
import sqlite3
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


def test_nightly_incremental_tracks_cursor_per_source_ref(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    first_path = tmp_path / "mail_first.jsonl"
    second_path = tmp_path / "mail_second.jsonl"
    write_jsonl(
        first_path,
        [
            {
                "source_id": "first",
                "customer_id": "customer:test-1",
                "event_type": "system_note",
                "event_at": "2026-06-21T10:00:00+00:00",
                "updated_at": "2026-06-21T10:00:00+00:00",
                "summary": "Первый файл.",
            }
        ],
    )
    write_jsonl(
        second_path,
        [
            {
                "source_id": "second",
                "customer_id": "customer:test-1",
                "event_type": "system_note",
                "event_at": "2026-06-21T09:00:00+00:00",
                "updated_at": "2026-06-21T09:00:00+00:00",
                "summary": "Второй файл старше первого, но новый для своего source_ref.",
            }
        ],
    )
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="mail_ref_a",
                source_system="mail_archive_stage2",
                path=first_path,
                source_ref="mail:ref-a",
            ),
            IncrementalSourceConfig(
                name="mail_ref_b",
                source_system="mail_archive_stage2",
                path=second_path,
                source_ref="mail:ref-b",
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
        safety_margin_seconds=0,
    )

    report = run_nightly_incremental(config)

    assert [source["rows_selected"] for source in report["sources"]] == [1, 1]
    assert event_count(tmp_path) == 2


def test_nightly_incremental_imports_mail_archive_stage2_manager_only(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail_stage2.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "message_sha256": "a" * 64,
                "customer_id": "customer:test-1",
                "date_last": "2026-06-21T11:00:00+00:00",
                "subject": "Вопрос по расписанию",
                "summary": "Клиент уточнил расписание.",
                "brand": "foton",
            }
        ],
    )
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="mail_stage2",
                source_system="mail_archive_stage2",
                path=source_path,
                source_ref="nightly-test:mail",
                normalizer="mail_archive_stage2",
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
        safety_margin_seconds=0,
    )

    first = run_nightly_incremental(config)
    second = run_nightly_incremental(config)

    assert first["changed_customer_ids"] == ["customer:test-1"]
    assert second["changed_customer_ids"] == []
    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        event = con.execute(
            "SELECT event_type, source_system, source_id FROM timeline_events WHERE source_id = ?",
            ("a" * 64,),
        ).fetchone()
        chunk = con.execute(
            "SELECT allowed_for_bot, requires_manager_review FROM bot_context_chunks"
        ).fetchone()
    assert event == ("email_message", "mail_archive_stage2", "a" * 64)
    assert chunk == (0, 1)


def test_nightly_incremental_unavailable_source_skips_and_alerts_after_two_failures(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    missing = tmp_path / "missing.jsonl"
    config = base_config(tmp_path, missing)

    first = run_nightly_incremental(config)
    second = run_nightly_incremental(config)

    expected_error = {
        "source": "amo_updates",
        "source_system": "amocrm_snapshot",
        "required": True,
        "reason": "source_unavailable",
    }
    assert first["source_errors"] == [expected_error]
    assert second["source_errors"] == [expected_error]
    with CustomerTimelineSQLiteStore.open_read_only(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path) as store:
        cursor = store.get_ingestion_cursor("foton", "amocrm_snapshot")
    assert cursor is not None
    assert cursor.metadata["consecutive_failures"] == 2
    assert cursor.metadata["alert"] is True


def test_nightly_incremental_fail_soft_keeps_other_sources_running(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    bad_path = tmp_path / "bad.jsonl"
    good_path = tmp_path / "good.jsonl"
    bad_path.write_text("{not-json}\n", encoding="utf-8")
    write_jsonl(
        good_path,
        [
            {
                "source_id": "good-event-1",
                "customer_id": "customer:test-1",
                "event_type": "system_note",
                "event_at": "2026-06-21T10:00:00+00:00",
                "updated_at": "2026-06-21T10:00:00+00:00",
                "direction": "system",
                "summary": "Второй источник должен импортироваться.",
            }
        ],
    )
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="bad_json",
                source_system="bad_json_source",
                path=bad_path,
                source_ref="test:bad-json",
            ),
            IncrementalSourceConfig(
                name="good_json",
                source_system="good_json_source",
                path=good_path,
                source_ref="test:good-json",
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
        safety_margin_seconds=0,
        lock_timeout_seconds=2,
    )

    report = run_nightly_incremental(config)

    assert report["overall_status"] == "partial"
    assert report["failed_required_sources"] == ["bad_json"]
    assert report["source_errors"][0]["reason"] == "source_exception:JSONDecodeError"
    assert report["sources"][0]["status"] == "failed"
    assert report["sources"][1]["status"] == "ok"
    assert report["changed_customer_ids"] == ["customer:test-1"]
    assert event_count(tmp_path) == 1


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
