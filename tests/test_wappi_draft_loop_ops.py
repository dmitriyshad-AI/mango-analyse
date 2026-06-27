from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.wappi_draft_loop_ops as ops
from mango_mvp.integrations.draft_loop import DraftLoopConfig, DraftLoopKey, DraftLoopPair, DraftLoopProfile


def _config(tmp_path: Path) -> DraftLoopConfig:
    profile = DraftLoopProfile(profile_id="profile-foton", brand="foton", channel="telegram")
    key = DraftLoopKey(profile.profile_id, "chat-1")
    return DraftLoopConfig(
        profiles={profile.profile_id: profile},
        pairs={key: DraftLoopPair(key=key, lead_id="123", contact_id="456", expected_brand="foton")},
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "journal.jsonl",
        manager_edit_log_path=tmp_path / "manager_edits.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        stop_path=tmp_path / "STOP_DRAFT_LOOP",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_runtime_passport_uses_actual_process_env_and_redacts_secrets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    processes = [
        ops.ProcessInfo(pid=7, ppid=0, command="/usr/bin/screen -S mango_draft_loop"),
        ops.ProcessInfo(pid=42, ppid=7, command="python3 scripts/run_amo_wappi_draft_loop.py --loop"),
    ]

    passport = ops.build_runtime_passport(
        repo_root=Path.cwd(),
        config=config,
        process_lister=lambda: processes,
        env_reader=lambda _pid: (
            {
                "TELEGRAM_BOT_SAFE_CRM_CONTEXT": "1",
                "DRAFT_LOOP_AUTO_RESOLVER": "1",
                "DRAFT_LOOP_AUTO_RESOLVER_ALLOW_ALL": "0",
                "TELEGRAM_BOT_TOKEN": "secret-token",
                "UNRELATED_FLAG": "must-not-leak",
            },
            "fake_process_env",
        ),
    )

    assert passport["process"]["found"] is True
    assert passport["process"]["pid"] == 42
    assert passport["process"]["launch_path"] == "scripts/run_amo_wappi_draft_loop.py"
    assert passport["process"]["screen"]["detected"] is True
    env = passport["runtime_env"]["values"]
    assert env["DRAFT_LOOP_AUTO_RESOLVER"] == "1"
    assert env["DRAFT_LOOP_AUTO_RESOLVER_ALLOW_ALL"] == "0"
    assert env["TELEGRAM_BOT_SAFE_CRM_CONTEXT"] == "1"
    assert env["TELEGRAM_BOT_TOKEN"] == "[REDACTED]"
    assert "UNRELATED_FLAG" not in env


def test_runtime_passport_prefers_python_runner_over_screen_wrapper(tmp_path: Path) -> None:
    config = _config(tmp_path)
    wrapper_command = (
        "SCREEN -dmS mango_wappi_observe bash -lc "
        "python3 scripts/run_amo_wappi_draft_loop.py --loop --dry-run"
    )
    processes = [
        ops.ProcessInfo(pid=10, ppid=1, command=wrapper_command),
        ops.ProcessInfo(pid=11, ppid=10, command="bash -lc python3 scripts/run_amo_wappi_draft_loop.py --loop --dry-run"),
        ops.ProcessInfo(pid=12, ppid=11, command="python3 scripts/run_amo_wappi_draft_loop.py --loop --dry-run"),
    ]

    passport = ops.build_runtime_passport(
        repo_root=Path.cwd(),
        config=config,
        process_lister=lambda: processes,
        env_reader=lambda pid: ({"DRAFT_LOOP_AUTO_RESOLVER": "0"} if pid == 12 else {"DRAFT_LOOP_AUTO_RESOLVER": "wrong"}, "fake"),
    )

    assert passport["process"]["pid"] == 12
    assert passport["process"]["screen"]["detected"] is True
    assert passport["runtime_env"]["values"]["DRAFT_LOOP_AUTO_RESOLVER"] == "0"


def test_daily_report_includes_required_resolver_reasons_and_state_counts(tmp_path: Path) -> None:
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    journal = tmp_path / "journal.jsonl"
    heartbeat = tmp_path / "heartbeat.json"
    state = tmp_path / "state.json"
    _write_jsonl(
        journal,
        [
            {
                "event": "pair_missing",
                "created_at": (now - timedelta(hours=1)).isoformat(),
                "auto_candidate": {"reason": "amo_chat_event_sequence_unconfirmed"},
            },
            {"event": "pair_missing", "created_at": now.isoformat(), "auto_candidate": {"reason": "amo_chat_event_rate_limited"}},
            {"event": "pair_missing", "created_at": now.isoformat(), "auto_candidate": {"reason": "amo_chat_event_ambiguous"}},
            {"event": "brand_pair_mismatch", "created_at": now.isoformat()},
            {"event": "pair_missing", "created_at": now.isoformat(), "auto_candidate": {"reason": "closed_lead"}},
            {"event": "pair_missing", "created_at": now.isoformat(), "auto_candidate": {"reason": "max_phone_missing"}},
            {"event": "pair_quarantined", "created_at": now.isoformat(), "status": "skipped"},
            {"event": "draft_created", "created_at": now.isoformat(), "status": "dry_run"},
            {"event": "note_written", "created_at": now.isoformat(), "status": "note_written"},
            {"event": "note_retry_failed", "created_at": now.isoformat(), "status": "manual_review", "error": "HTTP 500"},
        ],
    )
    heartbeat.write_text(json.dumps({"last_cycle_at": (now - timedelta(seconds=30)).isoformat(), "status": "ok"}), encoding="utf-8")
    state.write_text(
        json.dumps(
            {
                "quarantined_pairs": {"profile:chat": {"reason": "allowlist_desync"}},
                "pending_notes": {"profile\tchat\tm1": {"status": "note_pending"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = ops.build_daily_report(journal_path=journal, heartbeat_path=heartbeat, state_path=state, now=now)

    assert report["alive"]["fresh"] is True
    assert report["counts"]["draft_created"] == 1
    assert report["counts"]["notes_written"] >= 1
    assert report["counts"]["pair_missing"] == 5
    assert report["counts"]["errors"] == 1
    reasons = report["resolver_reasons"]
    for key in ops.REQUIRED_RESOLVER_REASON_KEYS:
        assert key in reasons
    assert reasons["amo_chat_event_sequence_unconfirmed"] == 1
    assert reasons["amo_chat_event_rate_limited"] == 1
    assert reasons["amo_chat_event_ambiguous"] == 1
    assert reasons["brand_mismatch"] == 1
    assert reasons["closed_lead"] == 1
    assert reasons["max_phone_missing"] == 1
    assert reasons["quarantined_pairs"] == 2
    assert reasons["pending_notes"] == 1
    assert reasons["quarantined"] == 2
    assert reasons["pending"] == 1


def test_daily_report_accepts_short_quarantined_and_pending_reason_aliases(tmp_path: Path) -> None:
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    journal = tmp_path / "journal.jsonl"
    heartbeat = tmp_path / "heartbeat.json"
    state = tmp_path / "state.json"
    _write_jsonl(
        journal,
        [
            {"event": "pair_missing", "created_at": now.isoformat(), "auto_resolver_reason": "quarantined"},
            {"event": "pair_missing", "created_at": now.isoformat(), "auto_resolver_reason": "pending"},
        ],
    )
    heartbeat.write_text(json.dumps({"last_cycle_at": now.isoformat(), "status": "ok"}), encoding="utf-8")
    state.write_text("{}", encoding="utf-8")

    reasons = ops.build_daily_report(journal_path=journal, heartbeat_path=heartbeat, state_path=state, now=now)["resolver_reasons"]

    assert reasons["quarantined_pairs"] == 1
    assert reasons["quarantined"] == 1
    assert reasons["pending_notes"] == 1
    assert reasons["pending"] == 1


def test_daily_report_includes_heartbeat_auto_resolver_counts(tmp_path: Path) -> None:
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    journal = tmp_path / "journal.jsonl"
    heartbeat = tmp_path / "heartbeat.json"
    state = tmp_path / "state.json"
    _write_jsonl(journal, [{"event": "pair_missing", "created_at": now.isoformat(), "status": "skipped"}])
    heartbeat.write_text(
        json.dumps(
            {
                "last_cycle_at": now.isoformat(),
                "status": "ok",
                "summary": {"auto_resolver_counts": {"not_enabled": 7, "amo_chat_event_ambiguous": 2}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state.write_text("{}", encoding="utf-8")

    reasons = ops.build_daily_report(journal_path=journal, heartbeat_path=heartbeat, state_path=state, now=now)["resolver_reasons"]

    assert reasons["not_enabled"] == 7
    assert reasons["amo_chat_event_ambiguous"] == 2


def test_quality_table_rows_keep_required_fields_and_manager_reply(tmp_path: Path) -> None:
    rows = ops.build_quality_rows(
        journal_rows=[
            {
                "event": "draft_created",
                "created_at": "2026-06-25T12:00:00+00:00",
                "lead_id": "123",
                "contact_id": "456",
                "profile_id": "profile-foton",
                "chat_id": "chat-secret-999999",
                "message_id": "m1",
                "route": "draft_for_manager",
                "safety_flags": ["client_safe_fact_verified", "draft_only"],
                "bot_draft_text": "Черновик менеджеру",
            }
        ],
        manager_edit_rows=[
            {
                "profile_id": "profile-foton",
                "chat_id": "chat-secret-999999",
                "message_id": "m1",
                "manager_sent_text": "Ответ менеджера",
            }
        ],
    )
    target = tmp_path / "quality.csv"
    ops.write_quality_csv(rows, target)

    with target.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))

    assert loaded[0].keys() == set(ops.QUALITY_COLUMNS)
    assert loaded[0]["chat_suffix"] == "999999"
    assert loaded[0]["draft_text"] == "Черновик менеджеру"
    assert loaded[0]["manager_reply_if_seen"] == "Ответ менеджера"
    assert loaded[0]["manual_label"] == ""
