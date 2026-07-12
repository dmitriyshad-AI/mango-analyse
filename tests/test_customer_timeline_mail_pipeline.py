from __future__ import annotations

import json
import sqlite3
import plistlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts import run_customer_timeline_mail_download as download
from scripts import run_customer_timeline_mail_process as process
from scripts import run_customer_timeline_mail_import as mail_import
from scripts import run_customer_timeline_mail_chain as mail_chain


class FakeDiscoveryImap:
    def __init__(self, *, host: str, port: int) -> None:
        assert host == "mail.example.test"
        assert port == 993

    def login(self, _user: str, _password: str) -> tuple[str, Sequence[bytes]]:
        return "OK", []

    def list(self) -> tuple[str, Sequence[bytes]]:
        return "OK", [
            b'(\\HasNoChildren) "/" "INBOX"',
            b'(\\HasNoChildren \\Sent) "/" "Sent"',
            b'(\\HasNoChildren) "/" "Sent Messages"',
        ]

    def logout(self) -> tuple[str, Sequence[bytes]]:
        return "BYE", []


def test_mail_download_discovers_exact_required_mailboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(download, "ImapLibClient", FakeDiscoveryImap)

    result = download.discover_required_mailboxes(
        host="mail.example.test",
        port=993,
        email_address="hidden@example.test",
        password="hidden",
        sent_name="Sent",
    )

    assert result == {
        "inbox": {"name": "INBOX", "raw": '"INBOX"'},
        "sent": {"name": "Sent", "raw": '"Sent"'},
    }


def test_mail_download_fails_on_ambiguous_sent_mailbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Ambiguous(FakeDiscoveryImap):
        def list(self) -> tuple[str, Sequence[bytes]]:
            return "OK", [b'() "/" "INBOX"', b'() "/" "Sent"', b'() "/" "Sent"']

    monkeypatch.setattr(download, "ImapLibClient", Ambiguous)

    with pytest.raises(RuntimeError, match="mailbox_sent_match_count=2"):
        download.discover_required_mailboxes(
            host="mail.example.test",
            port=993,
            email_address="hidden@example.test",
            password="hidden",
            sent_name="Sent",
        )


def test_mail_download_dry_run_has_no_network_or_secret_requirement(tmp_path: Path) -> None:
    report = download.execute(
        download.parse_args(
            [
                "--code-root",
                str(download.ROOT),
                "--data-root",
                str(tmp_path / "data"),
                "--state-dir",
                str(tmp_path / "state"),
            ]
        )
    )

    assert report["status"] == "dry_run"
    assert report["network_calls"] is False
    assert report["max_messages"] is None
    assert not (tmp_path / "state").exists()


def test_mail_download_updates_cursor_only_after_both_mailboxes_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "mail.env"
    secret.write_text("unused=true\n", encoding="utf-8")
    secret.chmod(0o600)
    monkeypatch.setattr(
        download,
        "load_dotenv_file",
        lambda _path: __import__("os").environ.update(
            {
                "MAIL_IMAP_EMAIL": "hidden@example.test",
                "MAIL_IMAP_PASSWORD": "hidden",
                "MAIL_IMAP_HOST": "mail.example.test",
                "MAIL_IMAP_PORT": "993",
            }
        ),
    )
    monkeypatch.setattr(
        download,
        "discover_required_mailboxes",
        lambda **_kwargs: {
            "inbox": {"name": "INBOX", "raw": '"INBOX"'},
            "sent": {"name": "Sent", "raw": '"Sent"'},
        },
    )
    monkeypatch.setattr(
        download,
        "run_ingest",
        lambda **_kwargs: (
            0,
            {
                "messages_found_since": 2,
                "messages_attempted": 2,
                "messages_inserted_or_seen": 2,
                "errors": [],
                "selection_truncated": False,
            },
        ),
    )

    state = tmp_path / ".codex_local/staging/mail_pipeline"
    report = download.execute(
        download.parse_args(
            [
                "--apply",
                "--code-root",
                str(download.ROOT),
                "--data-root",
                str(tmp_path / "data"),
                "--state-dir",
                str(state),
                "--dotenv",
                str(secret),
            ]
        )
    )

    assert report["status"] == "ok"
    assert set(report["mailbox_reports"]) == {"inbox", "sent"}
    assert json.loads((state / "mail_download_cursor.json").read_text())["cursor_kind"] == (
        "overlap_waterline_sha"
    )


def test_mail_download_failure_does_not_advance_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "mail.env"
    secret.write_text("unused=true\n", encoding="utf-8")
    secret.chmod(0o600)
    monkeypatch.setattr(
        download,
        "load_dotenv_file",
        lambda _path: __import__("os").environ.update(
            {"MAIL_IMAP_EMAIL": "x", "MAIL_IMAP_PASSWORD": "x"}
        ),
    )
    monkeypatch.setattr(
        download,
        "discover_required_mailboxes",
        lambda **_kwargs: {
            "inbox": {"name": "INBOX", "raw": '"INBOX"'},
            "sent": {"name": "Sent", "raw": '"Sent"'},
        },
    )
    calls = iter(
        [
            (0, {"messages_attempted": 1, "messages_inserted_or_seen": 1, "errors": []}),
            (1, {"messages_attempted": 1, "errors": [{"error": "safe"}]}),
        ]
    )
    monkeypatch.setattr(download, "run_ingest", lambda **_kwargs: next(calls))

    state = tmp_path / "state"
    report = download.execute(
        download.parse_args(
            [
                "--apply",
                "--code-root",
                str(download.ROOT),
                "--data-root",
                str(tmp_path / "data"),
                "--state-dir",
                str(state),
                "--dotenv",
                str(secret),
            ]
        )
    )

    assert report["status"] == "failed"
    assert not (state / "mail_download_cursor.json").exists()


def _write_archive(path: Path, *, sha: str, event_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE messages (
              sha256 TEXT PRIMARY KEY,
              message_date_iso TEXT,
              subject TEXT,
              message_kind TEXT,
              mailbox TEXT,
              extracted_text_path TEXT,
              updated_at TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha, event_at, "Тема", "external", "INBOX", "", event_at),
        )


def _write_timeline_with_cursor(path: Path, cursor: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE ingestion_cursors (
              tenant_id TEXT,
              source_system TEXT,
              last_cursor_ts TEXT,
              updated_at TEXT,
              metadata_json TEXT,
              PRIMARY KEY (tenant_id, source_system)
            );
            CREATE TABLE timeline_events (
              source_id TEXT,
              customer_id TEXT,
              match_status TEXT,
              confidence REAL,
              record_json TEXT,
              source_system TEXT
            );
            """
        )
        con.execute(
            "INSERT INTO ingestion_cursors VALUES (?, ?, ?, ?, ?)",
            (
                "foton",
                "mail_archive_stage2",
                cursor,
                cursor,
                json.dumps(
                    {
                        "source_refs": {
                            "mail_pipeline:mail_archive_stage2": {"last_cursor_ts": cursor}
                        }
                    }
                ),
            ),
        )


def test_mail_process_reuses_builder_and_timeline_cursor(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    state = tmp_path / ".codex_local/staging/mail_pipeline"
    canonical = data_root / download.CANONICAL_RELATIVE_ROOT / "archive/mail_archive.sqlite"
    incoming = data_root / download.CANONICAL_RELATIVE_ROOT / "incoming/regru_edu/inbox/mail_archive.sqlite"
    _write_archive(canonical, sha="a" * 64, event_at="2026-07-12T10:02:00+00:00")
    _write_archive(incoming, sha="b" * 64, event_at="2026-07-12T10:03:00+00:00")
    timeline = tmp_path / ".codex_local/staging/customer_timeline.sqlite"
    _write_timeline_with_cursor(timeline, "2026-07-12T10:05:00+00:00")
    state.mkdir(parents=True)
    runtime = download.runtime_identity(download.ROOT)
    download.atomic_write_json(
        state / "mail_download_manifest.json",
        {
            "status": "ok",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "truncated": False,
            "errors": 0,
            "mailbox_reports": {"inbox": {"status": "ok"}, "sent": {"status": "ok"}},
            "archive_db_paths": [str(incoming)],
        },
    )

    report = process.execute(
        process.parse_args(
            [
                "--code-root",
                str(download.ROOT),
                "--data-root",
                str(data_root),
                "--state-dir",
                str(state),
                "--timeline-db",
                str(timeline),
            ]
        )
    )

    assert report["status"] == "ok"
    assert report["cursor_start_with_overlap"] == "2026-07-12T10:00:00+00:00"
    assert report["rows_written"] == 2
    assert len(report["archive_databases"]) == 2
    config = json.loads(Path(report["config"]).read_text(encoding="utf-8"))
    assert [item["source_system"] for item in config["sources"]] == ["mail_archive_stage2"]


def test_mail_process_requires_explicit_bootstrap_when_cursor_missing(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.sqlite"
    with sqlite3.connect(timeline):
        pass

    with pytest.raises(RuntimeError, match="mail_cursor_unavailable_and_no_bootstrap"):
        process.read_cursor(timeline, bootstrap=None, overlap_seconds=300)


def test_mail_process_rejects_prod_or_non_staging_timeline_paths(tmp_path: Path) -> None:
    state = tmp_path / ".codex_local/staging/mail_pipeline"
    with pytest.raises(RuntimeError, match="timeline_db_outside_codex_staging"):
        process.staging_root_for(
            state_dir=state,
            timeline_db=tmp_path / "product_data/customer_timeline/customer_timeline_prod_1/db.sqlite",
        )
    with pytest.raises(RuntimeError, match="mail_state_dir_not_under_codex_staging"):
        process.staging_root_for(
            state_dir=tmp_path / "shared/mail_pipeline",
            timeline_db=tmp_path / "shared/timeline.sqlite",
        )


def test_mail_process_rejects_failed_download_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "runtime": {"head": "x", "worktree": "y"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="download_manifest_not_ok"):
        process.load_success_manifest(
            path,
            expected_runtime={"head": "x", "worktree": "y"},
            max_age_hours=4,
        )


def test_mail_import_rejects_non_mail_config(tmp_path: Path) -> None:
    state = tmp_path / ".codex_local/staging/mail_pipeline"
    process_dir = state / "process"
    process_dir.mkdir(parents=True)
    runtime = {"head": "abc", "worktree": "tree"}
    config = process_dir / "mail_incremental_config.json"
    download.atomic_write_json(
        config,
        {
            "runtime": runtime,
            "timeline_db": str(tmp_path / "timeline.sqlite"),
            "sources": [
                {"source_system": "mail_archive_stage2", "required": True},
                {"source_system": "amocrm_snapshot", "required": True},
            ],
        },
    )
    download.atomic_write_json(
        state / "mail_process_manifest.json",
        {
            "status": "ok",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "config": str(config),
            "config_sha256": download.sha256_file(config),
        },
    )

    with pytest.raises(RuntimeError, match="process_config_not_mail_only"):
        mail_import.load_inputs(state_dir=state, runtime=runtime, max_age_hours=4)


def test_mail_import_is_fail_loud_when_incremental_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / ".codex_local/staging/mail_pipeline"
    process_dir = state / "process"
    process_dir.mkdir(parents=True)
    runtime = download.runtime_identity(download.ROOT)
    timeline = tmp_path / ".codex_local/staging/customer_timeline.sqlite"
    _write_timeline_with_cursor(timeline, "2026-07-12T10:05:00+00:00")
    source = process_dir / "increment.jsonl"
    source.write_text("", encoding="utf-8")
    config = process_dir / "mail_incremental_config.json"
    download.atomic_write_json(
        config,
        {
            "runtime": runtime,
            "timeline_db": str(timeline),
            "allowed_root": str(tmp_path / ".codex_local/staging"),
            "journal_path": str(process_dir / "mail_incremental_journal.jsonl"),
            "sources": [
                {
                    "source_system": "mail_archive_stage2",
                    "normalizer": "mail_archive_stage2",
                    "required": True,
                    "path": str(source),
                }
            ],
        },
    )
    download.atomic_write_json(
        state / "mail_process_manifest.json",
        {
            "status": "ok",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "config": str(config),
            "config_sha256": download.sha256_file(config),
            "rows_written": 0,
            "output_jsonl": str(source),
            "output_sha256": download.sha256_file(source),
        },
    )

    class Result:
        returncode = 1
        stdout = json.dumps(
            {
                "overall_status": "partial",
                "gate_passed": False,
                "failed_required_sources": ["mail_archive_stage2"],
            }
        )

    monkeypatch.setattr(mail_import, "run_incremental", lambda *_args, **_kwargs: Result())
    monkeypatch.setattr(
        mail_import,
        "enrich_mail_links",
        lambda **_kwargs: pytest.fail("enrich must not run after failed incremental gate"),
    )

    report = mail_import.execute(
        mail_import.parse_args(
            ["--code-root", str(download.ROOT), "--state-dir", str(state)]
        )
    )

    assert report["status"] == "failed"
    assert report["gate_passed"] is False
    assert report["cursor_before"] == report["cursor_after"]


def test_mail_import_runs_existing_link_enrich_and_preserves_bot_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / ".codex_local/staging/mail_pipeline"
    process_dir = state / "process"
    process_dir.mkdir(parents=True)
    runtime = download.runtime_identity(download.ROOT)
    timeline = tmp_path / ".codex_local/staging/customer_timeline.sqlite"
    _write_timeline_with_cursor(timeline, "2026-07-12T10:05:00+00:00")
    source = process_dir / "increment.jsonl"
    source.write_text("", encoding="utf-8")
    config = process_dir / "mail_incremental_config.json"
    download.atomic_write_json(
        config,
        {
            "runtime": runtime,
            "timeline_db": str(timeline),
            "allowed_root": str(tmp_path / ".codex_local/staging"),
            "journal_path": str(process_dir / "journal.jsonl"),
            "sources": [
                {
                    "source_system": "mail_archive_stage2",
                    "normalizer": "mail_archive_stage2",
                    "required": True,
                    "path": str(source),
                }
            ],
        },
    )
    download.atomic_write_json(
        state / "mail_process_manifest.json",
        {
            "status": "ok",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "config": str(config),
            "config_sha256": download.sha256_file(config),
            "output_jsonl": str(source),
            "output_sha256": download.sha256_file(source),
            "rows_written": 1,
        },
    )

    class Result:
        returncode = 0
        stdout = json.dumps(
            {"overall_status": "ok", "gate_passed": True, "failed_required_sources": []}
        )

    monkeypatch.setattr(mail_import, "run_incremental", lambda *_args, **_kwargs: Result())
    monkeypatch.setattr(
        mail_import,
        "enrich_mail_links",
        lambda **_kwargs: {
            "target_events": 1,
            "counts": {"planned.strong": 1},
            "apply": {"counts": {"updated_events": 1, "created_chunks": 1}},
            "safety": {
                "allowed_for_bot_changed": False,
                "mail_stage2_allowed_for_bot_changed": False,
            },
        },
    )

    report = mail_import.execute(
        mail_import.parse_args(
            ["--code-root", str(download.ROOT), "--state-dir", str(state)]
        )
    )

    assert report["status"] == "ok"
    assert report["mail_link_enrich"] == {
        "status": "ok",
        "error": None,
        "target_events": 1,
        "planned": {"strong": 1, "weak_email": 0, "unmatched": 0, "blocked": 0},
        "updated_events": 1,
        "created_chunks": 1,
        "visibility_changed": False,
    }


def test_mail_import_execute_restores_full_cursor_after_enrich_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / ".codex_local/staging/mail_pipeline"
    process_dir = state / "process"
    process_dir.mkdir(parents=True)
    runtime = download.runtime_identity(download.ROOT)
    timeline = tmp_path / ".codex_local/staging/customer_timeline.sqlite"
    _write_timeline_with_cursor(timeline, "2026-07-12T10:05:00+00:00")
    before = mail_import.read_mail_cursor_state(timeline)
    source = process_dir / "increment.jsonl"
    source.write_text("", encoding="utf-8")
    config = process_dir / "mail_incremental_config.json"
    download.atomic_write_json(
        config,
        {
            "runtime": runtime,
            "timeline_db": str(timeline),
            "allowed_root": str(tmp_path / ".codex_local/staging"),
            "journal_path": str(process_dir / "journal.jsonl"),
            "sources": [
                {
                    "source_system": "mail_archive_stage2",
                    "normalizer": "mail_archive_stage2",
                    "required": True,
                    "path": str(source),
                }
            ],
        },
    )
    download.atomic_write_json(
        state / "mail_process_manifest.json",
        {
            "status": "ok",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "config": str(config),
            "config_sha256": download.sha256_file(config),
            "output_jsonl": str(source),
            "output_sha256": download.sha256_file(source),
            "rows_written": 1,
        },
    )

    class Result:
        returncode = 0
        stdout = json.dumps(
            {"overall_status": "ok", "gate_passed": True, "failed_required_sources": []}
        )

    def advance_both_cursors(*_args: object, **_kwargs: object) -> Result:
        with sqlite3.connect(timeline) as con:
            con.execute(
                "UPDATE ingestion_cursors SET last_cursor_ts=?, metadata_json=? WHERE tenant_id=? AND source_system=?",
                (
                    "2026-07-12T11:00:00+00:00",
                    json.dumps(
                        {
                            "source_refs": {
                                "mail_pipeline:mail_archive_stage2": {
                                    "last_cursor_ts": "2026-07-12T11:00:00+00:00"
                                }
                            }
                        }
                    ),
                    "foton",
                    "mail_archive_stage2",
                ),
            )
        return Result()

    monkeypatch.setattr(mail_import, "run_incremental", advance_both_cursors)
    monkeypatch.setattr(
        mail_import,
        "enrich_mail_links",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("late stage failed")),
    )

    report = mail_import.execute(
        mail_import.parse_args(
            ["--code-root", str(download.ROOT), "--state-dir", str(state)]
        )
    )

    assert report["status"] == "failed"
    assert mail_import.read_mail_cursor_state(timeline) == before


def test_mail_chain_stops_after_busy_stage_and_does_not_start_next(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(task: str) -> mail_chain.StageRun:
        calls.append(task)
        if task == "mail-download":
            return mail_chain.StageRun(task=task, rc=0, payload={"status": "ok"})
        return mail_chain.StageRun(
            task=task,
            rc=75,
            payload={"status": "already_running", "stop_reason": "already_running"},
        )

    report = mail_chain.run_chain(
        lock_path=tmp_path / "mail_chain.lock",
        runner=runner,
        preflight=lambda _task: "",
    )

    assert report["status"] == "stopped"
    assert report["stop_reason"] == "mail-process:already_running"
    assert calls == ["mail-download", "mail-process"]
    assert report["stages"][1]["status"] == "stopped"


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (download, []),
        (process, ["--data-root", "/tmp/test-mail-data"]),
        (mail_import, []),
    ],
)
def test_mail_stage_main_reports_busy_lock_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module: object,
    argv: list[str],
) -> None:
    monkeypatch.setattr(module, "execute", lambda _args: (_ for _ in ()).throw(RuntimeError("mail_download_already_running")))

    assert module.main(argv) == 75
    assert json.loads(capsys.readouterr().out) == {
        "status": "already_running",
        "stop_reason": "already_running",
    }


def test_mail_chain_reports_already_running_when_chain_lock_is_busy(tmp_path: Path) -> None:
    lock_path = tmp_path / "mail_chain.lock"
    with mail_chain.chain_lock(lock_path) as acquired:
        assert acquired is True
        report = mail_chain.run_chain(
            lock_path=lock_path,
            runner=lambda _task: pytest.fail("stage must not start when chain lock is busy"),
            preflight=lambda _task: "",
        )

    assert report["status"] == "stopped"
    assert report["result"] == "already_running"
    assert report["stop_reason"] == "already_running"
    assert report["stages"] == []


def test_mail_chain_stale_manifest_blocks_next_stage(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(task: str) -> mail_chain.StageRun:
        calls.append(task)
        return mail_chain.StageRun(task=task, rc=0, payload={"status": "ok"})

    def preflight(task: str) -> str:
        return "stage manifest is stale" if task == "mail-process" else ""

    report = mail_chain.run_chain(
        lock_path=tmp_path / "mail_chain.lock",
        runner=runner,
        preflight=preflight,
    )

    assert report["status"] == "stopped"
    assert report["stop_reason"] == "mail-process:stage manifest is stale"
    assert calls == ["mail-download"]
    assert report["stages"][1] == {
        "task": "mail-process",
        "status": "stopped",
        "stop_reason": "stage manifest is stale",
        "started": False,
    }


def test_mail_chain_detects_manifest_stale_by_actual_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = {"head": "abc", "worktree": "tree"}
    old = datetime.now(timezone.utc) - timedelta(hours=5)
    (tmp_path / "mail_download_manifest.json").write_text(
        json.dumps({"status": "ok", "finished_at": old.isoformat(), "runtime": runtime}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mail_chain.codex_task, "MAIL_STATE_DIR", tmp_path)
    monkeypatch.setattr(mail_chain.codex_task, "current_runtime", lambda: runtime)

    assert mail_chain.stage_preflight_stop_reason("mail-process") == "stage manifest is stale"


def test_mail_chain_first_unsuccessful_stage_stops_chain(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(task: str) -> mail_chain.StageRun:
        calls.append(task)
        if task == "mail-download":
            return mail_chain.StageRun(task=task, rc=0, payload={"status": "ok"})
        return mail_chain.StageRun(task=task, rc=2, payload={"status": "failed", "stop_reason": "gate_failed"})

    report = mail_chain.run_chain(
        lock_path=tmp_path / "mail_chain.lock",
        runner=runner,
        preflight=lambda _task: "",
    )

    assert report["status"] == "stopped"
    assert report["stop_reason"] == "mail-process:gate_failed"
    assert calls == ["mail-download", "mail-process"]
    assert [stage["task"] for stage in report["stages"]] == ["mail-download", "mail-process"]


def test_mail_launchd_uses_single_chain_calendar_trigger_and_deprecates_split_plists() -> None:
    deploy = download.ROOT / "deploy/customer_timeline_daily_captures"
    chain_path = deploy / "com.mango.customer-timeline-mail-chain.plist.template"
    with chain_path.open("rb") as fh:
        chain_payload = plistlib.load(fh)
    assert chain_payload["Label"] == "com.mango.customer-timeline-mail-chain"
    assert chain_payload["ProgramArguments"] == [
        "/usr/bin/python3",
        str(download.ROOT / "scripts/run_customer_timeline_mail_chain.py"),
    ]
    assert chain_payload["StartCalendarInterval"] == {"Hour": 2, "Minute": 0}

    split_payloads = []
    for suffix in ("download", "process", "import"):
        path = deploy / f"com.mango.customer-timeline-mail-{suffix}.plist.template"
        with path.open("rb") as fh:
            split_payloads.append(plistlib.load(fh))
    assert all(payload.get("Disabled") is True for payload in split_payloads)
    assert all("StartCalendarInterval" not in payload for payload in split_payloads)
    assert all(payload["WorkingDirectory"] == str(download.ROOT) for payload in [chain_payload, *split_payloads])
