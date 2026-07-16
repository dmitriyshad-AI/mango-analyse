from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.publish_snapshot import build_snapshot, flip, preflight, reader_smoke, rollback
from scripts.publish_snapshot import common as publish_common
from scripts.publish_snapshot.common import backup_plan_report, classify_publish_worktree_status, copy_verified, run_command
from tests.test_customer_timeline_read_api import seed_timeline_db


def _config(tmp_path: Path, prod: Path, staging: Path) -> Path:
    cfg = {
        "schema_version": "publish_snapshot_config_v1",
        "package_name": "test",
        "tenant_id": "foton",
        "staging_db": str(staging),
        "prod_db": str(prod),
        "snapshot_root": str(tmp_path / "snapshots"),
        "backup_root": str(tmp_path / "prod_backups"),
        "backup_async_copy_root": str(tmp_path / "openclaw_backups"),
        "required_free_copies": 1,
        "count_tables": ["customer_identities", "timeline_events", "bot_context_chunks"],
        "control_customers": [{"customer_id": "customer:0", "expected_found": True}],
        "readers": [],
    }
    path = tmp_path / "publish_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_build_snapshot_vacuum_into_and_manifest_then_reader_smoke(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["control_customers"] = [
        {
            "customer_id": staging_customer,
            "expected_found": True,
            "expected_counts": {
                "events_total": 1,
                "bot_context_chunks_total": 2,
                "allowed_chunks": 1,
                "review_required_chunks": 1,
                "derived_signals_total": 1,
            },
        }
    ]
    cfg.write_text(json.dumps(payload), encoding="utf-8")

    report, ok = build_snapshot.build_snapshot(cfg, execute=True, snapshot_name="prod_test")

    assert ok is True
    snapshot_db = Path(report["snapshot_db"])
    manifest = json.loads((snapshot_db.parent / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quick_check"] == "ok"
    assert manifest["counts"]["timeline_events"] >= 1

    smoke_report, smoke_ok = reader_smoke.smoke(cfg, snapshot_db=snapshot_db)
    assert smoke_ok is True
    assert smoke_report["internal_control_customers"][0]["found"] is True
    assert smoke_report["internal_control_customers"][0]["count_mismatches"] == {}

    payload["control_customers"][0]["expected_counts"]["events_total"] = 999
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    mismatch_report, mismatch_ok = reader_smoke.smoke(cfg, snapshot_db=snapshot_db)
    assert mismatch_ok is False
    assert mismatch_report["internal_control_customers"][0]["count_mismatches"]["events_total"]["actual"] == 1


def test_reader_smoke_blocks_mail_allowed_when_a2_facts_require_review(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["control_customers"] = [{"customer_id": staging_customer, "expected_found": True}]
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    with sqlite3.connect(staging) as con:
        event_id = con.execute("SELECT event_id FROM timeline_events LIMIT 1").fetchone()[0]
        con.execute(
            """
            INSERT INTO bot_context_chunks (
              chunk_id, tenant_id, customer_id, opportunity_id, event_id,
              source_system, source_ref, chunk_type, event_at, freshness_score,
              allowed_for_bot, requires_manager_review, ordinal, created_at,
              record_hash, record_json
            )
            VALUES (?, 'foton', ?, NULL, ?, 'mail_archive_stage2', 'mail:1',
              'email_message', '2026-05-12T12:00:00+00:00', 0.7,
              1, 0, 0, '2026-05-12T12:00:00+00:00', ?, ?)
            """,
            (
                "mail-sensitive",
                staging_customer,
                event_id,
                "f" * 64,
                json.dumps({"allowed_for_bot": True, "summary": "internal"}, ensure_ascii=False),
            ),
        )
        con.execute(
            """
            CREATE TABLE a2v3_mail_event_facts (
              event_id TEXT PRIMARY KEY,
              client_safe INTEGER NOT NULL,
              bot_visible INTEGER NOT NULL,
              client_safe_reason TEXT NOT NULL,
              sensitivity_tags_json TEXT NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO a2v3_mail_event_facts VALUES (?, 0, 0, ?, ?)",
            (event_id, "has_manager_note", json.dumps(["manager_action_required", "has_manager_note"], ensure_ascii=False)),
        )
        con.commit()

    smoke_report, smoke_ok = reader_smoke.smoke(cfg, snapshot_db=staging)

    assert smoke_ok is False
    gate = smoke_report["mail_allowed_safety_gate"]
    assert gate["ok"] is False
    assert gate["violations"]["allowed_mail_forbidden_primary_reason"] == 1
    assert gate["violations"]["allowed_mail_bot_visible_false"] == 1
    assert gate["violations"]["allowed_mail_unapproved_client_unsafe_reason"] == 1


def test_reader_smoke_allows_variant_b_money_but_blocks_secret_mail_tags(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["control_customers"] = [{"customer_id": staging_customer, "expected_found": True}]
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    with sqlite3.connect(staging) as con:
        event_id = con.execute("SELECT event_id FROM timeline_events LIMIT 1").fetchone()[0]
        con.execute(
            """
            INSERT INTO bot_context_chunks (
              chunk_id, tenant_id, customer_id, opportunity_id, event_id,
              source_system, source_ref, chunk_type, event_at, freshness_score,
              allowed_for_bot, requires_manager_review, ordinal, created_at,
              record_hash, record_json
            )
            VALUES (?, 'foton', ?, NULL, ?, 'mail_archive_stage2', 'mail:money',
              'email_message', '2026-05-12T12:00:00+00:00', 0.7,
              1, 0, 0, '2026-05-12T12:00:00+00:00', ?, ?)
            """,
            (
                "mail-money",
                staging_customer,
                event_id,
                "f" * 64,
                json.dumps({"allowed_for_bot": True, "summary": "money"}, ensure_ascii=False),
            ),
        )
        con.execute(
            """
            CREATE TABLE a2v3_mail_event_facts (
              event_id TEXT PRIMARY KEY,
              client_safe INTEGER NOT NULL,
              bot_visible INTEGER NOT NULL,
              client_safe_reason TEXT NOT NULL,
              sensitivity_tags_json TEXT NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO a2v3_mail_event_facts VALUES (?, 0, 1, ?, ?)",
            (
                event_id,
                "sensitive_money",
                json.dumps(["sensitive_money", "manager_action_required"], ensure_ascii=False),
            ),
        )
        con.commit()

    money_report, money_ok = reader_smoke.smoke(cfg, snapshot_db=staging)

    assert money_ok is True
    money_gate = money_report["mail_allowed_safety_gate"]
    assert money_gate["ok"] is True
    assert money_gate["counts"]["allowed_mail_variant_b_client_unsafe"] == 1
    assert money_gate["violations"] == {}

    with sqlite3.connect(staging) as con:
        event_id = con.execute("SELECT event_id FROM timeline_events LIMIT 1").fetchone()[0]
        con.execute(
            "UPDATE a2v3_mail_event_facts SET sensitivity_tags_json = ? WHERE event_id = ?",
            (json.dumps(["sensitive_credentials"], ensure_ascii=False), event_id),
        )
        con.commit()

    secret_report, secret_ok = reader_smoke.smoke(cfg, snapshot_db=staging)

    assert secret_ok is False
    secret_gate = secret_report["mail_allowed_safety_gate"]
    assert secret_gate["ok"] is False
    assert secret_gate["violations"]["allowed_mail_secret_tags"] == 1


def test_reader_smoke_allows_strong_known_brand_mango_processed_chunks(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["control_customers"] = [{"customer_id": staging_customer, "expected_found": True}]
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    with sqlite3.connect(staging) as con:
        event_id = con.execute("SELECT event_id FROM timeline_events LIMIT 1").fetchone()[0]
        con.execute(
            """
            UPDATE timeline_events
            SET source_system = 'mango_processed_summary',
                match_status = 'strong_unique'
            WHERE event_id = ?
            """,
            (event_id,),
        )
        con.execute(
            """
            INSERT INTO bot_context_chunks (
              chunk_id, tenant_id, customer_id, opportunity_id, event_id,
              source_system, source_ref, chunk_type, event_at, freshness_score,
              allowed_for_bot, requires_manager_review, ordinal, created_at,
              record_hash, record_json
            )
            VALUES (?, 'foton', ?, NULL, ?, 'mango_processed_summary', 'mango:1',
              'mango_call_summary', '2026-05-12T12:00:00+00:00', 0.7,
              1, 0, 0, '2026-05-12T12:00:00+00:00', ?, ?)
            """,
            (
                "mango-strong",
                staging_customer,
                event_id,
                "f" * 64,
                json.dumps(
                    {
                        "allowed_for_bot": True,
                        "requires_manager_review": False,
                        "metadata": {"content_brand": "foton"},
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        con.commit()

    smoke_report, smoke_ok = reader_smoke.smoke(cfg, snapshot_db=staging)

    assert smoke_ok is True
    gate = smoke_report["mango_processed_allowed_safety_gate"]
    assert gate["ok"] is True
    assert gate["counts"]["allowed_mango_processed_chunks"] == 1
    assert gate["violations"] == {}


def test_reader_smoke_blocks_mango_processed_non_strong_but_allows_unknown_brand_metric(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["control_customers"] = [{"customer_id": staging_customer, "expected_found": True}]
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    with sqlite3.connect(staging) as con:
        event_id = con.execute("SELECT event_id FROM timeline_events LIMIT 1").fetchone()[0]
        con.execute(
            """
            UPDATE timeline_events
            SET source_system = 'mango_processed_summary',
                match_status = 'ambiguous'
            WHERE event_id = ?
            """,
            (event_id,),
        )
        con.execute(
            """
            INSERT INTO bot_context_chunks (
              chunk_id, tenant_id, customer_id, opportunity_id, event_id,
              source_system, source_ref, chunk_type, event_at, freshness_score,
              allowed_for_bot, requires_manager_review, ordinal, created_at,
              record_hash, record_json
            )
            VALUES (?, 'foton', ?, NULL, ?, 'mango_processed_summary', 'mango:1',
              'mango_call_summary', '2026-05-12T12:00:00+00:00', 0.7,
              1, 0, 0, '2026-05-12T12:00:00+00:00', ?, ?)
            """,
            (
                "mango-ambiguous",
                staging_customer,
                event_id,
                "f" * 64,
                json.dumps(
                    {
                        "allowed_for_bot": True,
                        "requires_manager_review": False,
                        "metadata": {"content_brand": "unknown"},
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        con.commit()

    smoke_report, smoke_ok = reader_smoke.smoke(cfg, snapshot_db=staging)

    assert smoke_ok is False
    gate = smoke_report["mango_processed_allowed_safety_gate"]
    assert gate["ok"] is False
    assert gate["violations"]["allowed_mango_processed_non_strong_match"] == 1
    assert "allowed_mango_processed_unknown_brand_metric" not in gate["violations"]
    assert gate["counts"]["allowed_mango_processed_unknown_brand_metric"] == 1


def test_reader_smoke_blocks_mango_processed_corrupted_identity_contract(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["control_customers"] = [{"customer_id": staging_customer, "expected_found": True}]
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    with sqlite3.connect(staging) as con:
        event_id = con.execute("SELECT event_id FROM timeline_events LIMIT 1").fetchone()[0]
        con.execute(
            """
            UPDATE timeline_events
            SET source_system = 'mango_processed_summary',
                match_status = 'strong_unique'
            WHERE event_id = ?
            """,
            (event_id,),
        )
        con.execute(
            "UPDATE customer_identities SET identity_status = 'ambiguous' WHERE customer_id = ?",
            (staging_customer,),
        )
        rows = [
            ("mango-wrong-type", staging_customer, "wrong_call_summary"),
            ("mango-mismatch", "customer:other", "mango_call_summary"),
            ("mango-bad-identity", staging_customer, "mango_call_summary"),
        ]
        for chunk_id, customer_id, chunk_type in rows:
            con.execute(
                """
                INSERT INTO bot_context_chunks (
                  chunk_id, tenant_id, customer_id, opportunity_id, event_id,
                  source_system, source_ref, chunk_type, event_at, freshness_score,
                  allowed_for_bot, requires_manager_review, ordinal, created_at,
                  record_hash, record_json
                )
                VALUES (?, 'foton', ?, NULL, ?, 'mango_processed_summary', 'mango:bad',
                  ?, '2026-05-12T12:00:00+00:00', 0.7,
                  1, 0, 0, '2026-05-12T12:00:00+00:00', ?, ?)
                """,
                (
                    chunk_id,
                    customer_id,
                    event_id,
                    chunk_type,
                    "f" * 64,
                    json.dumps(
                        {
                            "allowed_for_bot": True,
                            "requires_manager_review": False,
                            "metadata": {"content_brand": "unknown"},
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        con.commit()

    smoke_report, smoke_ok = reader_smoke.smoke(cfg, snapshot_db=staging)

    assert smoke_ok is False
    gate = smoke_report["mango_processed_allowed_safety_gate"]
    assert gate["ok"] is False
    assert gate["violations"]["allowed_mango_processed_wrong_chunk_type"] == 1
    assert gate["violations"]["allowed_mango_processed_customer_mismatch"] == 1
    assert gate["violations"]["allowed_mango_processed_missing_identity"] == 3


def test_preflight_blocks_dirty_reader_worktree(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, _staging_customer = seed_timeline_db(staging_dir)
    cfg = json.loads(_config(tmp_path, prod, staging).read_text(encoding="utf-8"))
    cfg["readers"] = [{"name": "reader", "worktree": str(tmp_path), "stop_command": ["true"], "start_command": ["true"]}]
    cfg_path = tmp_path / "publish_config_dirty.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    report, ok = preflight.build_report(cfg_path)

    assert ok is False
    assert report["readers"][0]["git_status_clean"] is False


def test_publish_worktree_status_allows_data_untracked() -> None:
    status = (
        "?? product_data/telegram_dynamic_test_sets/sample.jsonl\n"
        "?? tasks/_running/task.md\n"
        "?? README.local.md\n"
    )

    report = classify_publish_worktree_status(status)

    assert report["clean_for_publish"] is True
    assert report["tracked_blockers"] == []
    assert report["untracked_code_blockers"] == []
    assert report["untracked_allowed"] == [
        "?? product_data/telegram_dynamic_test_sets/sample.jsonl",
        "?? tasks/_running/task.md",
        "?? README.local.md",
    ]


def test_publish_worktree_status_blocks_code_untracked_and_tracked_changes() -> None:
    status = (
        " M docs/DECISIONS_LOG.md\n"
        "?? src/mango_mvp/new_module.py\n"
        "?? scripts/new_tool.py\n"
        "?? product_data/telegram_dynamic_test_sets/sample.jsonl\n"
    )

    report = classify_publish_worktree_status(status)

    assert report["clean_for_publish"] is False
    assert report["tracked_blockers"] == [" M docs/DECISIONS_LOG.md"]
    assert report["untracked_code_blockers"] == ["?? src/mango_mvp/new_module.py", "?? scripts/new_tool.py"]
    assert report["untracked_allowed"] == ["?? product_data/telegram_dynamic_test_sets/sample.jsonl"]


def test_backup_plan_allows_same_disk_with_verified_async_copy(tmp_path: Path) -> None:
    source = tmp_path / "prod.sqlite"
    source.write_text("backup-source", encoding="utf-8")

    report = backup_plan_report(
        source,
        tmp_path / "prod_backups",
        tmp_path / "openclaw_backups",
        required_bytes=source.stat().st_size,
    )
    first = copy_verified(source, tmp_path / "prod_backups" / "copy.sqlite")
    second = copy_verified(Path(first["target"]), tmp_path / "openclaw_backups" / "copy.sqlite")

    assert report["ok"] is True
    assert report["policy"] == "same_disk_verified_backup_plus_yandex_async_copy"
    assert first["source_sha256"] == first["target_sha256"]
    assert second["source_sha256"] == second["target_sha256"]


def test_flip_process_pattern_counts_exactly_one(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = (
            "101 python3 scripts/run_amo_wappi_draft_loop.py --loop\n"
            "202 python3 scripts/other.py\n"
        )
        stderr = ""

    monkeypatch.setattr(flip.subprocess, "run", lambda *args, **kwargs: Result())

    checks = flip.process_pattern_counts(["scripts/run_amo_wappi_draft_loop.py"])

    assert checks == [
        {
            "pattern": "scripts/run_amo_wappi_draft_loop.py",
            "count": 1,
            "matches": ["101 python3 scripts/run_amo_wappi_draft_loop.py --loop"],
        }
    ]


def test_flip_blocks_if_lsof_reappears_before_replace(monkeypatch, tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, _staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["readers"] = [{"name": "reader", "worktree": str(tmp_path), "stop_command": ["true"], "start_command": ["true"]}]
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    original_sha = flip.sha256_file(prod)
    calls = {"count": 0}

    def fake_lsof(_path: Path) -> list[str]:
        calls["count"] += 1
        return [] if calls["count"] == 1 else ["python 123 reopened customer_timeline.sqlite"]

    monkeypatch.setattr(flip, "lsof_holders", fake_lsof)

    report, ok = flip.flip(cfg, snapshot_db=staging, execute=True)

    assert ok is False
    assert report["status"] == "blocked_lsof_before_replace"
    assert flip.sha256_file(prod) == original_sha


def test_replace_sqlite_retries_only_transient_open(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    con = sqlite3.connect(source)
    try:
        con.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        con.execute("INSERT INTO sample VALUES ('new')")
        con.commit()
    finally:
        con.close()
    target.write_bytes(b"old")
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_quick_check(_path: Path) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return "ok"

    monkeypatch.setattr(publish_common, "writable_quick_check", fake_quick_check)
    monkeypatch.setattr(publish_common.time, "sleep", sleeps.append)

    report = publish_common.replace_sqlite_verified(source, target)

    assert report["ok"] is True
    assert report["replace_completed"] is True
    assert report["quick_check_attempts"] == 2
    assert report["quick_check_errors"][0]["transient_open"] is True
    assert sleeps == [2.0]
    assert report["read_only_open"] is True
    con = sqlite3.connect(target)
    try:
        assert con.execute("SELECT value FROM sample").fetchone()[0] == "new"
    finally:
        con.close()


def test_replace_sqlite_verifies_checkpointed_wal_snapshot_without_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    con = sqlite3.connect(source)
    try:
        assert con.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        con.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        con.execute("INSERT INTO sample VALUES ('kept')")
        con.commit()
        assert con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() == (0, 0, 0)
    finally:
        con.close()
    for suffix in ("-wal", "-shm"):
        Path(str(source) + suffix).unlink(missing_ok=True)
    target.write_bytes(b"old")

    report = publish_common.replace_sqlite_verified(source, target, attempts=1, delay_seconds=0)

    assert report["ok"] is True
    assert report["quick_check"] == "ok"
    assert report["post_replace_sidecars"] == []
    with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as con:
        assert con.execute("SELECT value FROM sample").fetchone()[0] == "kept"


def test_replace_sqlite_does_not_retry_other_sqlite_errors(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    sleeps: list[float] = []

    def locked_quick_check(_path: Path) -> str:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(publish_common, "writable_quick_check", locked_quick_check)
    monkeypatch.setattr(publish_common.time, "sleep", sleeps.append)

    report = publish_common.replace_sqlite_verified(source, target)

    assert report["ok"] is False
    assert report["status"] == "quick_check_exception"
    assert report["quick_check_attempts"] == 1
    assert report["exception"]["transient_open"] is False
    assert sleeps == []


def test_replace_sqlite_stops_after_configured_transient_attempts(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    sleeps: list[float] = []

    def unavailable_quick_check(_path: Path) -> str:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(publish_common, "writable_quick_check", unavailable_quick_check)
    monkeypatch.setattr(publish_common.time, "sleep", sleeps.append)

    report = publish_common.replace_sqlite_verified(source, target, attempts=3, delay_seconds=0.5)

    assert report["ok"] is False
    assert report["status"] == "quick_check_exception"
    assert report["quick_check_attempts"] == 3
    assert len(report["quick_check_errors"]) == 3
    assert sleeps == [0.5, 0.5]


def test_replace_sqlite_fails_if_post_replace_sidecar_exists(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    Path(str(target) + "-wal").write_bytes(b"stale")
    monkeypatch.setattr(publish_common, "writable_quick_check", lambda _path: "ok")

    report = publish_common.replace_sqlite_verified(source, target)

    assert report["ok"] is False
    assert report["status"] == "post_replace_sidecars_present"
    assert report["replace_completed"] is True
    assert report["post_replace_sidecars"] == [str(target) + "-wal"]


def test_flip_and_rollback_report_post_replace_failure(monkeypatch, tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, _staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    failure = {
        "ok": False,
        "status": "quick_check_exception",
        "replace_completed": True,
        "quick_check": None,
        "quick_check_attempts": 1,
        "quick_check_errors": [],
        "post_replace_sidecars": [],
        "sha256": None,
        "exception": {"type": "OperationalError", "message": "database is locked", "attempt": 1},
    }
    monkeypatch.setattr(flip, "replace_sqlite_verified", lambda *_args, **_kwargs: failure)

    flip_report, flip_ok = flip.flip(cfg, snapshot_db=staging, execute=True)

    assert flip_ok is False
    assert flip_report["status"] == "failed_post_replace_verification"
    assert flip_report["post_replace_verification"] == failure
    assert "backup_db" in flip_report

    monkeypatch.setattr(rollback, "replace_sqlite_verified", lambda *_args, **_kwargs: failure)
    rollback_report, rollback_ok = rollback.rollback(cfg, backup_db=staging, execute=True)

    assert rollback_ok is False
    assert rollback_report["status"] == "failed"
    assert rollback_report["post_replace_verification"] == failure


def test_run_command_reports_timeout_instead_of_raising() -> None:
    result = run_command(["python3", "-c", "import time; time.sleep(1)"], timeout=0.01)

    assert result["rc"] == 124
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 0.01
