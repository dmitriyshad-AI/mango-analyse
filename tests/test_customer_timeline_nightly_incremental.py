from __future__ import annotations

import importlib.util
import json
import hashlib
import sqlite3
import threading
import time
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.nightly_incremental as nightly_incremental_module
from mango_mvp.customer_timeline import CustomerIdentity, CustomerTimelineSQLiteStore, IdentityStatus
from mango_mvp.customer_timeline.nightly_incremental import (
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    run_nightly_incremental,
    single_run_lock,
)
from scripts import run_customer_timeline_nightly_incremental as nightly_cli


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source_proof(
    manifest_path: Path,
    *,
    source_path: Path,
    finished_at: datetime,
    rows_written: int = 0,
    max_event_at: str | None = None,
) -> None:
    builder_manifest = manifest_path.with_name("builder_manifest.json")
    builder_manifest.write_text('{"status":"ok"}\n', encoding="utf-8")
    runtime = {"head": "test-head", "worktree": "/test/worktree"}
    download_manifest = manifest_path.with_name("mail_download_manifest.json")
    download_manifest.write_text(
        json.dumps(
            {
                "status": "ok",
                "truncated": False,
                "errors": 0,
                "runtime": runtime,
                "mailbox_reports": {"inbox": {"status": "ok"}, "sent": {"status": "ok"}},
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "finished_at": finished_at.isoformat(),
                "rows_written": rows_written,
                "max_event_at": max_event_at or (
                    "2026-08-28T10:00:00+00:00" if rows_written else None
                ),
                "output_jsonl": str(source_path),
                "output_sha256": sha256_file(source_path),
                "builder_manifest": str(builder_manifest),
                "builder_manifest_sha256": sha256_file(builder_manifest),
                "runtime": runtime,
                "download_manifest": str(download_manifest),
                "download_manifest_sha256": sha256_file(download_manifest),
            }
        ),
        encoding="utf-8",
    )


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
                "raw_payload": {"must_not_affect_change_detection": True},
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
    performance = second["sources"][0]["performance"]
    assert performance["mode"] == "incremental"
    assert performance["rows"]["fetched"] == 2
    assert performance["rows"]["changed_customers"] == 0
    assert performance["seconds"]["total"] >= 0
    assert performance["cursor_used"] is True
    cursor = second["cursor_updates"][0]
    assert cursor["last_cursor_ts"] == "2026-06-21T10:04:00+00:00"


def test_nightly_empty_success_records_source_freshness_without_rebuild(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "empty.jsonl"
    source_path.write_text("", encoding="utf-8")

    report = run_nightly_incremental(base_config(tmp_path, source_path))

    assert report["gate_passed"] is True
    assert report["changed_customer_ids"] == []
    assert report["rebuild"]["selected_customer_count"] == 0
    assert report["imports"][0]["accepted_count"] == 0
    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        assert con.execute(
            "SELECT status FROM ingestion_runs WHERE source_ref='test:amo_updates'"
        ).fetchone()[0] == "completed"
        cursor = con.execute(
            "SELECT last_cursor_ts, updated_at FROM ingestion_cursors WHERE source_system='amocrm_snapshot'"
        ).fetchone()
        assert cursor[0] == "1970-01-01T00:00:00+00:00"
        assert cursor[1]


def test_nightly_missing_only_source_ignores_and_preserves_cursor(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "old_missing.jsonl"
    write_jsonl(
        source_path,
        [{
            "source_id": "old-missing",
            "customer_id": "customer:test-1",
            "event_type": "amo_deal_stage",
            "created_at": "2020-01-01T00:00:00+00:00",
            "updated_at": "2020-01-01T00:00:00+00:00",
            "summary": "Старое отсутствующее событие",
        }],
    )
    with CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path) as store:
        before = store.upsert_ingestion_cursor(
            "foton",
            "amocrm_snapshot",
            last_cursor_ts=datetime(2026, 7, 12, tzinfo=timezone.utc),
            metadata={"sentinel": "keep"},
        ).to_json_dict()
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(IncrementalSourceConfig(
            name="old_missing",
            source_system="amocrm_snapshot",
            path=source_path,
            source_ref="backfill:missing",
            ignore_cursor=True,
            preserve_cursor=True,
        ),),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
    )

    first = run_nightly_incremental(config)
    second = run_nightly_incremental(config)

    with CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path) as store:
        after = store.get_ingestion_cursor("foton", "amocrm_snapshot").to_json_dict()
    assert first["sources"][0]["rows_selected"] == 1
    assert first["cursor_updates"] == second["cursor_updates"] == []
    assert before == after
    assert event_count(tmp_path) == 1


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


@pytest.mark.parametrize(
    ("event_at", "updated_at"),
    (
        ("2026-08-30T09:00:00+00:00", "2026-08-31T10:00:00+00:00"),
        ("2026-08-31T10:00:00+00:00", "2026-08-31T10:00:00+00:00"),
    ),
)
def test_mail_proof_uses_business_time_and_cursor_uses_update_time(
    tmp_path: Path,
    event_at: str,
    updated_at: str,
) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail_stage2.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "message_sha256": "b" * 64,
                "customer_id": "customer:test-1",
                "event_at": event_at,
                "updated_at": updated_at,
                "subject": "Вопрос по расписанию",
            }
        ],
    )
    proof_path = tmp_path / "mail_process_manifest.json"
    write_source_proof(
        proof_path,
        source_path=source_path,
        finished_at=datetime.now(timezone.utc),
        rows_written=1,
        max_event_at=event_at,
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
                proof_manifest_path=proof_path,
                proof_manifest_sha256=sha256_file(proof_path),
                proof_max_age_hours=72,
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
        safety_margin_seconds=0,
    )

    report = run_nightly_incremental(config)

    with sqlite3.connect(config.timeline_db) as con:
        stored_event_at, record_json = con.execute(
            "SELECT event_at, record_json FROM timeline_events WHERE source_id = ?",
            ("b" * 64,),
        ).fetchone()
        cursor_at = con.execute(
            "SELECT last_cursor_ts FROM ingestion_cursors "
            "WHERE source_system = 'mail_archive_stage2'"
        ).fetchone()[0]
    assert report["gate_passed"] is True
    assert report["sources"][0]["artifact_proof"]["max_event_at_verified"] == event_at
    assert stored_event_at == event_at
    assert json.loads(record_json)["metadata"]["source_updated_at"] == updated_at
    assert cursor_at == updated_at


def test_mail_invalid_business_time_does_not_fall_back_to_updated_at(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail_stage2.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "message_sha256": "c" * 64,
                "customer_id": "customer:test-1",
                "event_at": "not-a-date",
                "updated_at": "2026-08-31T10:00:00+00:00",
                "subject": "Некорректная дата письма",
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
                normalizer="mail_archive_stage2",
            ),
        ),
        journal_path=tmp_path / "nightly" / "journal.jsonl",
    )

    report = run_nightly_incremental(config)

    assert report["gate_passed"] is False
    assert report["source_errors"][0]["reason"] == "source_exception:ValueError"
    with sqlite3.connect(config.timeline_db) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0] == 0


def test_nightly_incremental_preserves_mail_link_enrich_pending_state(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail_stage2_pending.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "message_sha256": "c" * 64,
                "date_last": "2026-06-21T11:00:00+00:00",
                "subject": "Вопрос по расписанию",
                "summary": "Клиент уточнил расписание.",
                "brand": "unknown",
                "match_status": "unmatched",
                "pending_attribution": True,
                "pending_reason": "no_strong_identity_match",
                "fresh_relink": True,
                "mail_link_enrich": {
                    "schema_version": "mail_link_enrich_v1",
                    "outcome": "unmatched",
                    "reason": "no_strong_identity_match",
                },
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

    run_nightly_incremental(config)

    with sqlite3.connect(tmp_path / "customer_timeline.sqlite") as con:
        con.row_factory = sqlite3.Row
        event = con.execute(
            "SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id = ?",
            ("c" * 64,),
        ).fetchone()
    payload = json.loads(event["record_json"])
    assert event["customer_id"] is None
    assert event["match_status"] == "unmatched"
    assert payload["metadata"]["pending_reason"] == "no_strong_identity_match"
    assert payload["metadata"]["fresh_relink"] is True
    assert payload["metadata"]["mail_link_enrich"]["outcome"] == "unmatched"


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


def test_proved_source_is_verified_immediately_before_read(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail.jsonl"
    source_path.write_text("", encoding="utf-8")
    proof_path = tmp_path / "mail_process_manifest.json"
    write_source_proof(proof_path, source_path=source_path, finished_at=datetime.now(timezone.utc))
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="mail_stage2",
                source_system="mail_archive_stage2",
                path=source_path,
                normalizer="mail_archive_stage2",
                proof_manifest_path=proof_path,
                proof_manifest_sha256=sha256_file(proof_path),
                proof_max_age_hours=72,
            ),
        ),
        journal_path=tmp_path / "nightly/journal.jsonl",
    )

    report = run_nightly_incremental(config)

    assert report["gate_passed"] is True
    proof = report["sources"][0]["artifact_proof"]
    assert proof["status"] == "ok"
    assert proof["manifest_sha256"] == sha256_file(proof_path)
    assert proof["output_sha256"] == sha256_file(source_path)


def test_mail_proof_rejects_new_failed_download_after_old_process_success(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail.jsonl"
    source_path.write_text("", encoding="utf-8")
    proof_path = tmp_path / "mail_process_manifest.json"
    download_path = tmp_path / "mail_download_manifest.json"
    runtime = {"head": "abc", "worktree": "/repo"}
    download_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "truncated": False,
                "errors": 0,
                "runtime": runtime,
                "mailbox_reports": {"inbox": {"status": "ok"}, "sent": {"status": "ok"}},
            }
        ),
        encoding="utf-8",
    )
    write_source_proof(proof_path, source_path=source_path, finished_at=datetime.now(timezone.utc))
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "runtime": runtime,
            "download_manifest": str(download_path),
            "download_manifest_sha256": sha256_file(download_path),
        }
    )
    proof_path.write_text(json.dumps(payload), encoding="utf-8")
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(IncrementalSourceConfig(
            name="mail_stage2", source_system="mail_archive_stage2", path=source_path,
            normalizer="mail_archive_stage2", proof_manifest_path=proof_path,
            proof_manifest_sha256=sha256_file(proof_path), proof_max_age_hours=72,
        ),),
        journal_path=tmp_path / "nightly/journal.jsonl",
    )
    failed = json.loads(download_path.read_text(encoding="utf-8"))
    failed["status"] = "failed"
    download_path.write_text(json.dumps(failed), encoding="utf-8")

    report = run_nightly_incremental(config)

    assert report["gate_passed"] is False
    assert report["sources"][0]["artifact_proof"]["reason"] == "download_lineage_mismatch"


def test_mail_proof_without_download_lineage_is_unavailable(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail.jsonl"
    source_path.write_text("", encoding="utf-8")
    proof_path = tmp_path / "mail_process_manifest.json"
    write_source_proof(proof_path, source_path=source_path, finished_at=datetime.now(timezone.utc))
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    payload.pop("download_manifest")
    payload.pop("download_manifest_sha256")
    proof_path.write_text(json.dumps(payload), encoding="utf-8")
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(IncrementalSourceConfig(
            name="mail_stage2", source_system="mail_archive_stage2", path=source_path,
            normalizer="mail_archive_stage2", proof_manifest_path=proof_path,
            proof_manifest_sha256=sha256_file(proof_path), proof_max_age_hours=72,
        ),),
        journal_path=tmp_path / "nightly/journal.jsonl",
    )

    report = run_nightly_incremental(config)

    assert report["gate_passed"] is False
    assert report["sources"][0]["artifact_proof"]["reason"] == "download_lineage_missing"


def test_proved_source_reads_verified_fd_across_atomic_path_replace(
    tmp_path: Path, monkeypatch
) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail.jsonl"
    source_path.write_text("", encoding="utf-8")
    proof_path = tmp_path / "mail_process_manifest.json"
    write_source_proof(proof_path, source_path=source_path, finished_at=datetime.now(timezone.utc))
    original = nightly_incremental_module.source_artifact_proof

    def replace_after_proof(source, **kwargs):
        result = original(source, **kwargs)
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text('{"unexpected":true}\n', encoding="utf-8")
        replacement.replace(source_path)
        return result

    monkeypatch.setattr(nightly_incremental_module, "source_artifact_proof", replace_after_proof)
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(IncrementalSourceConfig(
            name="mail_stage2", source_system="mail_archive_stage2", path=source_path,
            normalizer="mail_archive_stage2", proof_manifest_path=proof_path,
            proof_manifest_sha256=sha256_file(proof_path), proof_max_age_hours=72,
        ),),
        journal_path=tmp_path / "nightly/journal.jsonl",
    )

    report = run_nightly_incremental(config)

    assert report["source_errors"] == []
    assert report["gate_passed"] is True
    assert report["sources"][0]["rows_total"] == 0


def test_proved_source_blocks_in_place_mutation_after_verification(
    tmp_path: Path, monkeypatch
) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail.jsonl"
    source_path.write_text("", encoding="utf-8")
    proof_path = tmp_path / "mail_process_manifest.json"
    write_source_proof(proof_path, source_path=source_path, finished_at=datetime.now(timezone.utc))
    original = nightly_incremental_module.source_artifact_proof

    def mutate_after_proof(source, **kwargs):
        result = original(source, **kwargs)
        source_path.write_text('{"unexpected":true}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(nightly_incremental_module, "source_artifact_proof", mutate_after_proof)
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(IncrementalSourceConfig(
            name="mail_stage2", source_system="mail_archive_stage2", path=source_path,
            normalizer="mail_archive_stage2", proof_manifest_path=proof_path,
            proof_manifest_sha256=sha256_file(proof_path), proof_max_age_hours=72,
        ),),
        journal_path=tmp_path / "nightly/journal.jsonl",
    )

    report = run_nightly_incremental(config)

    assert report["gate_passed"] is False
    assert report["source_errors"][0]["reason"] == "source_unavailable"


@pytest.mark.parametrize(
    ("manifest_field", "manifest_value", "expected_reason"),
    (
        ("rows_written", 0, "output_verification_failed"),
        ("max_event_at", "2026-08-27T10:00:00+00:00", "max_event_at_mismatch"),
    ),
)
def test_proved_source_blocks_manifest_jsonl_balance_mismatch(
    tmp_path: Path,
    manifest_field: str,
    manifest_value: object,
    expected_reason: str,
) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail.jsonl"
    write_jsonl(
        source_path,
        [{
            "source_id": "mail-proof-row",
            "customer_id": "customer:1",
            "event_at": "2026-08-28T10:00:00+00:00",
        }],
    )
    proof_path = tmp_path / "mail_process_manifest.json"
    write_source_proof(
        proof_path,
        source_path=source_path,
        finished_at=datetime.now(timezone.utc),
        rows_written=1,
    )
    proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_payload[manifest_field] = manifest_value
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(IncrementalSourceConfig(
            name="mail_stage2",
            source_system="mail_archive_stage2",
            path=source_path,
            normalizer="mail_archive_stage2",
            proof_manifest_path=proof_path,
            proof_manifest_sha256=sha256_file(proof_path),
            proof_max_age_hours=72,
        ),),
        journal_path=tmp_path / "nightly/journal.jsonl",
    )

    report = run_nightly_incremental(config)

    assert report["gate_passed"] is False
    assert report["sources"][0]["artifact_proof"]["reason"] == expected_reason


def test_stale_proved_source_never_advances_existing_cursor(tmp_path: Path) -> None:
    seed_customer(tmp_path)
    source_path = tmp_path / "mail.jsonl"
    source_path.write_text("", encoding="utf-8")
    proof_path = tmp_path / "mail_process_manifest.json"
    write_source_proof(
        proof_path,
        source_path=source_path,
        finished_at=datetime.now(timezone.utc) - timedelta(hours=73),
    )
    with CustomerTimelineSQLiteStore(
        tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path
    ) as store:
        before = store.upsert_ingestion_cursor(
            "foton",
            "mail_archive_stage2",
            last_cursor_ts=datetime(2026, 8, 1, tzinfo=timezone.utc),
            metadata={"sentinel": "keep"},
        ).to_json_dict()
    config = NightlyIncrementalConfig(
        timeline_db=tmp_path / "customer_timeline.sqlite",
        allowed_root=tmp_path,
        sources=(
            IncrementalSourceConfig(
                name="mail_stage2",
                source_system="mail_archive_stage2",
                path=source_path,
                normalizer="mail_archive_stage2",
                proof_manifest_path=proof_path,
                proof_manifest_sha256=sha256_file(proof_path),
                proof_max_age_hours=72,
            ),
        ),
        journal_path=tmp_path / "nightly/journal.jsonl",
    )

    report = run_nightly_incremental(config)

    assert report["gate_passed"] is False
    assert report["source_errors"][0]["reason"] == "source_stale"
    with CustomerTimelineSQLiteStore.open_read_only(
        tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path
    ) as store:
        after = store.get_ingestion_cursor("foton", "mail_archive_stage2").to_json_dict()
    assert after == before


def test_nightly_incremental_cli_returns_nonzero_when_required_gate_fails(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "timeline_db": str(tmp_path / "timeline.sqlite"),
                "journal_path": str(tmp_path / "journal.jsonl"),
                "sources": [
                    {
                        "source_system": "mail_archive_stage2",
                        "path": str(tmp_path / "missing.jsonl"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        nightly_cli,
        "run_nightly_incremental",
        lambda _config: {
            "gate_passed": False,
            "failed_required_sources": ["mail_archive_stage2"],
            "overall_status": "partial",
        },
    )

    assert nightly_cli.main(["--config", str(config), "--summary-only"]) == 1


def test_nightly_service_cli_summary_only_keeps_data_quality_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_service = types.ModuleType("mango_mvp.customer_timeline.nightly_service")
    fake_service.service_config_from_json = lambda _path: object()
    fake_service.run_nightly_service = lambda _config: {
        "schema_version": "customer_timeline_nightly_service_v1",
        "run_id": "run-1",
        "overall_status": "ok",
        "data_quality_status": "pass_with_notes",
        "partial_failure": False,
        "failed_required_steps": [],
        "required_sources_check": {"missing": ["email"], "degraded": ["email"]},
        "degraded_steps": [{"name": "mail_archive_incremental"}],
        "degraded_sources": {"email": {"status": "degraded"}},
        "duration_seconds": 1.0,
        "steps": [],
        "snapshot_manifest": {"latest_published": True},
        "safety": {"writes_prod_db": False},
    }
    monkeypatch.setitem(sys.modules, fake_service.__name__, fake_service)
    script = Path(__file__).resolve().parents[1] / "scripts/run_customer_timeline_nightly_service.py"
    spec = importlib.util.spec_from_file_location("nightly_service_cli_test", script)
    nightly_service_cli = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(nightly_service_cli)

    rc = nightly_service_cli.main(["--config", str(tmp_path / "config.json"), "--summary-only"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["data_quality_status"] == "pass_with_notes"
    assert payload["required_sources_check"]["degraded"] == ["email"]
    assert payload["degraded_steps"][0]["name"] == "mail_archive_incremental"
    assert payload["degraded_sources"]["email"]["status"] == "degraded"


def test_mail_archive_stage2_proof_requires_pinned_manifest_sha(tmp_path: Path) -> None:
    source_path = tmp_path / "mail.jsonl"
    proof_path = tmp_path / "mail_process_manifest.json"

    with pytest.raises(ValueError, match="mail_archive_stage2 proof_manifest_sha256"):
        IncrementalSourceConfig(
            name="mail_stage2",
            source_system="mail_archive_stage2",
            path=source_path,
            normalizer="mail_archive_stage2",
            proof_manifest_path=proof_path,
            proof_max_age_hours=72,
        )


def test_nightly_incremental_cli_rejects_canonical_staging_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "timeline_db": str(tmp_path / "customer_timeline_staging.sqlite"),
                "allowed_root": str(tmp_path),
                "journal_path": str(tmp_path / "journal.jsonl"),
                "sources": [
                    {
                        "source_system": "mail_archive_stage2",
                        "path": str(tmp_path / "mail.jsonl"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        nightly_cli,
        "run_nightly_incremental",
        lambda _config: pytest.fail("direct CLI must stop before opening the writer"),
    )

    with pytest.raises(RuntimeError, match="owned_by_nightly_service"):
        nightly_cli.main(["--config", str(config)])


def test_nightly_incremental_cli_starts_without_external_pythonpath() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/run_customer_timeline_nightly_incremental.py"), "--help"],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout


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
