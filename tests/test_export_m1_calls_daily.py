from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("daily_export", ROOT / "scripts/export_m1_calls_daily.py")
daily_export = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = daily_export
SPEC.loader.exec_module(daily_export)

SCHEMA = """
CREATE TABLE call_records (
 id INTEGER NOT NULL PRIMARY KEY, source_file VARCHAR(1024) NOT NULL,
 source_filename VARCHAR(255) NOT NULL, source_call_id VARCHAR(128),
 audio_codec VARCHAR(64), sample_rate INTEGER, channels INTEGER, duration_sec FLOAT,
 phone VARCHAR(64), manager_name VARCHAR(255), direction VARCHAR(32), started_at DATETIME,
 transcription_status VARCHAR(16) NOT NULL, resolve_status VARCHAR(16) NOT NULL,
 analysis_status VARCHAR(16) NOT NULL, sync_status VARCHAR(16) NOT NULL,
 transcribe_attempts INTEGER NOT NULL, resolve_attempts INTEGER NOT NULL,
 analyze_attempts INTEGER NOT NULL, sync_attempts INTEGER NOT NULL,
 pipeline_stage VARCHAR(32), pipeline_worker_id VARCHAR(64), pipeline_claimed_at DATETIME,
 analysis_worker_id VARCHAR(64), analysis_claimed_at DATETIME, next_retry_at DATETIME,
 dead_letter_stage VARCHAR(16), transcript_manager TEXT, transcript_client TEXT,
 transcript_text TEXT, transcript_variants_json TEXT, resolve_json TEXT,
 resolve_quality_score FLOAT, analysis_json TEXT, amocrm_contact_id INTEGER,
 amocrm_lead_id INTEGER, last_error TEXT, created_at DATETIME NOT NULL,
 updated_at DATETIME NOT NULL
);
CREATE INDEX ix_call_records_source_call_id ON call_records(source_call_id);
CREATE UNIQUE INDEX ix_call_records_source_file ON call_records(source_file);
"""


def row(number: int, *, summary: str | None = None, controlled: bool = False, source_id: str | None = None):
    source_id = source_id or f"call-{number}"
    source_file = f"/tmp/{'controlled-mango-call-test/working/audio' if controlled else 'live'}/{number}.mp3"
    return {
        "id": number, "source_file": source_file, "source_filename": f"{number}.mp3",
        "source_call_id": source_id, "audio_codec": "mp3", "sample_rate": 16000,
        "channels": 2, "duration_sec": 40.0, "phone": "+70000000000",
        "manager_name": "Менеджер", "direction": "inbound",
        "started_at": f"2026-09-24 10:{number % 60:02d}:00", "transcription_status": "done",
        "resolve_status": "done", "analysis_status": "done", "sync_status": "done",
        "transcribe_attempts": 1, "resolve_attempts": 1, "analyze_attempts": 1,
        "sync_attempts": 1, "pipeline_stage": None, "pipeline_worker_id": None,
        "pipeline_claimed_at": None, "analysis_worker_id": None, "analysis_claimed_at": None,
        "next_retry_at": None, "dead_letter_stage": None, "transcript_manager": "Добрый день",
        "transcript_client": "Здравствуйте", "transcript_text": "Полный диалог",
        "transcript_variants_json": "{}", "resolve_json": "{}", "resolve_quality_score": 1.0,
        "analysis_json": json.dumps({"summary": summary or f"Итог {number}"}, ensure_ascii=False),
        "amocrm_contact_id": number, "amocrm_lead_id": number, "last_error": None,
        "created_at": "2026-09-24 11:00:00", "updated_at": f"2026-09-24 11:{number % 60:02d}:00",
    }


def make_db(path: Path, rows: list[dict]) -> None:
    con = sqlite3.connect(path); con.executescript(SCHEMA)
    columns = list(rows[0]) if rows else [item[1] for item in con.execute("PRAGMA table_info(call_records)")]
    if rows:
        con.executemany(
            f"INSERT INTO call_records ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            [[item[name] for name in columns] for item in rows],
        )
    con.commit(); con.close()


def packaged_count(path: Path) -> int:
    con = sqlite3.connect(path / "call_records_snapshot.sqlite")
    value = con.execute("SELECT count(*) FROM call_records").fetchone()[0]
    assert con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    con.close(); return value


def test_progressive_1_10_daily_and_empty(tmp_path: Path) -> None:
    baseline, live = tmp_path / "baseline.sqlite", tmp_path / "live.sqlite"
    make_db(baseline, [row(1), row(2)])
    live_rows = [row(1), row(2, summary="Изменено")] + [row(i) for i in range(3, 15)] + [row(15, controlled=True)]
    make_db(live, live_rows)
    previous = daily_export.scan_db(baseline, None).fingerprints

    one = daily_export.scan_db(live, previous, limit=1); one_dir = tmp_path / "one"; one_dir.mkdir()
    daily_export.write_package(one, one_dir); assert packaged_count(one_dir) == 1

    ten = daily_export.scan_db(live, previous, limit=10); ten_dir = tmp_path / "ten"; ten_dir.mkdir()
    daily_export.write_package(ten, ten_dir); assert packaged_count(ten_dir) == 10

    full = daily_export.scan_db(live, previous); full_dir = tmp_path / "full"; full_dir.mkdir()
    meta = daily_export.write_package(full, full_dir)
    assert (full.counts["new"], full.counts["changed"], full.counts["controlled_excluded"]) == (12, 1, 1)
    assert full.counts["overlap"] == 1
    assert meta["rows"] == packaged_count(full_dir) == 14

    empty = daily_export.scan_db(live, full.fingerprints); empty_dir = tmp_path / "empty"; empty_dir.mkdir()
    daily_export.write_package(empty, empty_dir)
    assert empty.counts["new"] == empty.counts["changed"] == packaged_count(empty_dir) == 0


def test_invalid_analysis_and_duplicate_id_fail_closed(tmp_path: Path) -> None:
    broken = row(1); broken["analysis_json"] = "not-json"
    invalid = tmp_path / "invalid.sqlite"; make_db(invalid, [broken])
    with pytest.raises(RuntimeError, match="invalid_done_analysis_json"):
        daily_export.scan_db(invalid, {})

    duplicate = tmp_path / "duplicate.sqlite"; make_db(duplicate, [row(1, source_id="same"), row(2, source_id="same")])
    with pytest.raises(RuntimeError, match="duplicate_ready_source_call_id"):
        daily_export.scan_db(duplicate, {})

    empty_object = row(3); empty_object["analysis_json"] = "{}"
    empty = tmp_path / "empty-object.sqlite"; make_db(empty, [empty_object])
    with pytest.raises(RuntimeError, match="invalid_done_analysis_json"):
        daily_export.scan_db(empty, {})

    dirty = row(4, source_id=" call-4 ")
    dirty_db = tmp_path / "dirty-id.sqlite"; make_db(dirty_db, [dirty])
    with pytest.raises(RuntimeError, match="noncanonical_ready_source_call_id"):
        daily_export.scan_db(dirty_db, {})


def test_all_controlled_paths_are_excluded_before_validation(tmp_path: Path) -> None:
    controlled = row(1)
    controlled["source_file"] = "/tmp/controlled-mango-call-pilot/audio/1.mp3"
    controlled["analysis_json"] = "broken-but-excluded"
    controlled_ready = row(2)
    controlled_ready["source_file"] = "/tmp/controlled-mango-call-pilot/audio/2.mp3"
    live = tmp_path / "controlled.sqlite"; make_db(live, [controlled, controlled_ready])
    scan = daily_export.scan_db(live, {})
    assert scan.rows == []
    assert scan.counts["controlled_excluded"] == 1


def test_baseline_manifest_is_verified_before_bootstrap(tmp_path: Path) -> None:
    baseline = tmp_path / "call_records_snapshot.sqlite"; make_db(baseline, [row(1)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": daily_export.SCHEMA_VERSION,
        "status": "READY_FOR_M4_READ_ONLY_IMPORT_REVIEW",
        "snapshot": {"file": baseline.name, "sha256": daily_export.sha256(baseline), "size_bytes": baseline.stat().st_size},
    }), encoding="utf-8")
    assert len(daily_export.verified_baseline(baseline, manifest)) == 1
    payload = json.loads(manifest.read_text()); payload["snapshot"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="baseline_snapshot_fingerprint_mismatch"):
        daily_export.verified_baseline(baseline, manifest)


def test_yandex_proof_requires_remote_hash_size_and_idle(tmp_path: Path) -> None:
    log = tmp_path / "sync_core.log"; rel = "OpenClaw/pkg/call_records_snapshot.sqlite"
    digest, size = "a" * 64, 123
    log.write_text(
        f'REMOTE "{rel}" object {digest[:8]} {size} file\n'
        f'PROP "{rel}" lh={digest[:8]},rh={digest[:8]},id=object\nCore IDLE\n',
        encoding="utf-8",
    )
    proof = daily_export.wait_yandex(log, 0, log.stat().st_ino, {rel: (digest, size)}, time.time() + 1)
    assert proof["status"] == "remote_hash_size_and_idle_confirmed"
