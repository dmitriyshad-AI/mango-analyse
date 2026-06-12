from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts import build_tz19_tail_bundle as bundle


def test_select_tail_calls_uses_old_ge60_non_blacklist_only(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.sqlite"
    master = tmp_path / "master.sqlite"
    previous = tmp_path / "previous"
    seed_timeline(timeline, [1, 2, 3, 4, 5, 6, 7])
    seed_master(
        master,
        [
            call_row(1, "2025-05-31T10:00:00+00:00", 60, "done", 100, "v6"),
            call_row(2, "2025-06-01T00:00:00+00:00", 60, "done", 100, "v6"),
            call_row(3, "2025-05-01T10:00:00+00:00", 59, "done", 100, "v6"),
            call_row(4, "2025-05-01T10:00:00+00:00", 60, "done", 100, "v7"),
            call_row(5, "2025-05-01T10:00:00+00:00", 60, "failed", 100, "v6"),
            call_row(6, "2025-05-01T10:00:00+00:00", 60, "done", 0, "v6"),
            call_row(7, "2025-05-01T10:00:00+00:00", 60, "done", 100, "v6"),
        ],
    )
    seed_previous(previous, "7\n")
    cfg = config(tmp_path, timeline, master, previous, expected_calls=1)

    selected = bundle.select_tail_calls(cfg, {7})

    assert [item.call_id for item in selected] == [1]


def test_build_tail_bundle_writes_manifest_task_ready_and_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeline = tmp_path / "timeline.sqlite"
    master = tmp_path / "master.sqlite"
    previous = tmp_path / "previous"
    seed_timeline(timeline, [11, 12])
    seed_master(
        master,
        [
            call_row(11, "2025-05-01T10:00:00+00:00", 60, "done", 100, "v6"),
            call_row(12, "2025-05-02T10:00:00+00:00", 70, "done", 200, "v6"),
        ],
    )
    seed_previous(previous, "999\n")
    monkeypatch.setattr(bundle, "current_git_commit", lambda _repo: "a" * 40)
    monkeypatch.setattr(bundle, "create_git_archive", fake_git_archive)

    cfg = config(tmp_path, timeline, master, previous, expected_calls=2)
    first = bundle.build_tail_bundle(cfg)
    manifest_path = cfg.bundle_dir / "data" / "manifest.json"
    first_manifest = manifest_path.read_text(encoding="utf-8")
    second = bundle.build_tail_bundle(cfg)

    assert first["manifest"]["rows"] == 2
    assert first["manifest"]["transcript_chars_sum"] == 300
    assert first["manifest"]["safety"]["blacklist_overlap"] == 0
    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert manifest_path.read_text(encoding="utf-8") == first_manifest
    assert (cfg.bundle_dir / "data" / "slice_zone.db").exists()
    assert (cfg.bundle_dir / "data" / "slice_zone.db.zip").exists()
    task = cfg.inbox_m1_dir / f"{bundle.TASK_ID}.task.yaml"
    ready = Path(str(task) + ".ready")
    assert task.exists()
    assert ready.read_text(encoding="utf-8") == f"sha256:{bundle.sha256_text(task.read_text(encoding='utf-8'))}\n"
    assert "expected_calls: 2" in task.read_text(encoding="utf-8")


def test_build_tail_bundle_stops_on_wrong_manifest_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeline = tmp_path / "timeline.sqlite"
    master = tmp_path / "master.sqlite"
    previous = tmp_path / "previous"
    seed_timeline(timeline, [21])
    seed_master(master, [call_row(21, "2025-05-01T10:00:00+00:00", 60, "done", 100, "v6")])
    seed_previous(previous, "")
    monkeypatch.setattr(bundle, "current_git_commit", lambda _repo: "b" * 40)
    monkeypatch.setattr(bundle, "create_git_archive", fake_git_archive)

    with pytest.raises(RuntimeError, match="tail call count mismatch"):
        bundle.build_tail_bundle(config(tmp_path, timeline, master, previous, expected_calls=2))


def test_build_tail_bundle_refuses_to_overwrite_different_generated_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeline = tmp_path / "timeline.sqlite"
    master = tmp_path / "master.sqlite"
    previous = tmp_path / "previous"
    seed_timeline(timeline, [31])
    seed_master(master, [call_row(31, "2025-05-01T10:00:00+00:00", 60, "done", 100, "v6")])
    seed_previous(previous, "")
    cfg = config(tmp_path, timeline, master, previous, expected_calls=1)
    cfg.bundle_dir.mkdir(parents=True)
    (cfg.bundle_dir / "README_RUN.md").write_text("foreign content", encoding="utf-8")
    monkeypatch.setattr(bundle, "current_git_commit", lambda _repo: "c" * 40)
    monkeypatch.setattr(bundle, "create_git_archive", fake_git_archive)

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        bundle.build_tail_bundle(cfg)


def test_real_prompt_sha_helper_is_stable() -> None:
    prompt = bundle.SYSTEM_PROMPT_FULL.strip() + "\n"

    assert bundle.ANALYZE_PROMPT_VERSION_FULL == "v7"
    assert bundle.sha256_text(prompt) == bundle.sha256_text(prompt)


def config(
    tmp_path: Path,
    timeline: Path,
    master: Path,
    previous: Path,
    *,
    expected_calls: int,
) -> bundle.TailBundleConfig:
    return bundle.TailBundleConfig(
        timeline_db=timeline,
        master_calls_db=master,
        previous_rerun_package=previous,
        bundle_dir=tmp_path / "bundle",
        inbox_m1_dir=tmp_path / "inbox_m1",
        repo_root=tmp_path,
        expected_calls=expected_calls,
    )


def fake_git_archive(_repo_root: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake archive")


def call_row(
    call_id: int,
    started_at: str,
    duration_sec: int,
    analysis_status: str,
    transcript_chars: int,
    prompt_version: str,
) -> tuple:
    return (
        call_id,
        f"file-{call_id}",
        f"call-{call_id}.mp3",
        started_at,
        duration_sec,
        analysis_status,
        "done",
        "done",
        transcript_chars,
        f"MANAGER: hello {call_id}\nCLIENT: reply",
        "{}",
        json.dumps({"analysis_meta": {"analysis_prompt_version": prompt_version}}),
        "2026-01-01T00:00:00+00:00",
    )


def seed_timeline(path: Path, call_ids: list[int]) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customer_opportunities (
          opportunity_id TEXT PRIMARY KEY,
          tenant_id TEXT,
          customer_id TEXT,
          opportunity_type TEXT,
          source_system TEXT,
          source_id TEXT,
          title TEXT,
          status TEXT,
          opened_at TEXT,
          closed_at TEXT,
          confidence REAL,
          record_hash TEXT,
          record_json TEXT
        );
        CREATE TABLE identity_links (
          link_id TEXT PRIMARY KEY,
          tenant_id TEXT,
          customer_id TEXT,
          link_type TEXT,
          link_value TEXT,
          source_system TEXT,
          source_ref TEXT,
          match_class TEXT,
          confidence REAL,
          first_seen_at TEXT,
          last_seen_at TEXT,
          record_hash TEXT,
          record_json TEXT
        );
        CREATE TABLE timeline_events (
          event_id TEXT PRIMARY KEY,
          dedupe_key TEXT,
          tenant_id TEXT,
          customer_id TEXT,
          opportunity_id TEXT,
          event_type TEXT,
          event_at TEXT,
          source_system TEXT,
          source_id TEXT,
          source_ref TEXT,
          direction TEXT,
          match_status TEXT,
          confidence REAL,
          importance INTEGER,
          subject TEXT,
          text_preview TEXT,
          summary TEXT,
          created_at TEXT,
          record_hash TEXT,
          record_json TEXT
        );
        """
    )
    con.execute(
        "INSERT INTO customer_opportunities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("opp1", "foton", "cust1", "amo_deal", "amocrm_snapshot", "1", "", "В работе", "", "", 1, "h", "{}"),
    )
    con.executemany(
        "INSERT INTO timeline_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                f"e{cid}",
                f"d{cid}",
                "foton",
                "cust1",
                "",
                "mango_call",
                "2026-01-01T00:00:00+00:00",
                "mango_processed_summary",
                str(cid),
                f"mango:{cid}",
                "system",
                "strong_unique",
                1,
                0,
                "",
                "",
                "",
                "",
                "h",
                "{}",
            )
            for cid in call_ids
        ],
    )
    con.commit()
    con.close()


def seed_master(path: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE canonical_calls (
            canonical_call_id INTEGER PRIMARY KEY,
            source_file TEXT,
            source_filename TEXT,
            started_at TEXT,
            duration_sec REAL,
            analysis_status TEXT,
            transcription_status TEXT,
            resolve_status TEXT,
            transcript_chars INTEGER,
            transcript_text TEXT,
            transcript_variants_json TEXT,
            analysis_json TEXT,
            created_at TEXT
        );
        CREATE INDEX idx_canonical_calls_source_filename on canonical_calls(source_filename);
        """
    )
    con.executemany("INSERT INTO canonical_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def seed_previous(path: Path, blacklist_text: str) -> None:
    (path / "scripts_pkg").mkdir(parents=True)
    (path / "blacklist_77.txt").write_text(blacklist_text, encoding="utf-8")
    (path / "scripts_pkg" / "export_analyze_results.py").write_text("export", encoding="utf-8")
    (path / "scripts_pkg" / "m1_run_cli_shim.sh").write_text("shim", encoding="utf-8")
