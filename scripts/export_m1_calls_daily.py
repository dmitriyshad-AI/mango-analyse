#!/usr/bin/env python3
"""Publish a fail-closed daily call_records delta for the M4 Timeline writer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "m1_to_m4_processed_calls_sqlite_snapshot_v1"
GENERATOR = "m1_daily_calls_export_v1"
OVERLAP_ROWS = 25
BASE_READY = """transcription_status='done' AND analysis_status='done'
AND length(trim(coalesce(source_call_id,'')))>0 AND length(trim(coalesce(transcript_text,'')))>0
AND CASE WHEN json_valid(analysis_json)=1 AND json_type(analysis_json)='object'
  THEN EXISTS(SELECT 1 FROM json_each(analysis_json)) ELSE 0 END
AND length(trim(coalesce(started_at,'')))>0 AND length(trim(coalesce(updated_at,'')))>0"""
READY = BASE_READY + " AND coalesce(source_file,'') NOT LIKE '%/controlled-mango-call-%'"
FP_FIELDS = ("source_call_id", "started_at", "updated_at", "phone", "manager_name", "direction",
             "duration_sec", "transcript_manager", "transcript_client", "transcript_text",
             "transcript_variants_json", "resolve_json", "resolve_quality_score", "analysis_json",
             "amocrm_contact_id", "amocrm_lead_id")


@dataclass
class Scan:
    columns: list[str]
    schema: str
    rows: list[tuple]
    kinds: list[str]
    fingerprints: dict[str, str]
    counts: dict[str, int]
    finished_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path); path.chmod(0o600)


def parse_date(value: object) -> None:
    text = str(value or "").strip()
    if text.endswith("Z"): text = text[:-1] + "+00:00"
    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError("invalid_ready_call_datetime") from exc


def row_fingerprint(row: sqlite3.Row) -> str:
    raw = json.dumps([row[key] for key in FP_FIELDS], ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def scan_db(path: Path, previous: dict[str, str] | None, limit: int | None = None) -> Scan:
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row; con.execute("PRAGMA query_only=ON"); con.execute("BEGIN")
    if con.execute("PRAGMA quick_check").fetchone()[0] != "ok": raise RuntimeError("source_quick_check_failed")
    columns = [row[1] for row in con.execute("PRAGMA table_info(call_records)")]
    missing = (set(FP_FIELDS) | {"id", "source_file", "transcription_status", "analysis_status"}) - set(columns)
    if missing: raise RuntimeError(f"source_schema_missing:{','.join(sorted(missing))}")
    candidate = "analysis_status='done' AND transcription_status='done' AND length(trim(coalesce(source_call_id,'')))>0 AND length(trim(coalesce(transcript_text,'')))>0 AND coalesce(source_file,'') NOT LIKE '%/controlled-mango-call-%'"
    bad = con.execute(f"SELECT id FROM call_records WHERE {candidate} AND (json_valid(analysis_json)=0 OR CASE WHEN json_valid(analysis_json)=1 THEN json_type(analysis_json)!='object' OR NOT EXISTS(SELECT 1 FROM json_each(analysis_json)) ELSE 1 END) LIMIT 1").fetchone()
    if bad: raise RuntimeError("invalid_done_analysis_json")
    dirty_id = con.execute(f"SELECT id FROM call_records WHERE {READY} AND source_call_id!=trim(source_call_id) LIMIT 1").fetchone()
    if dirty_id: raise RuntimeError("noncanonical_ready_source_call_id")
    dup = con.execute(f"SELECT trim(source_call_id) FROM call_records WHERE {READY} GROUP BY trim(source_call_id) HAVING count(*)>1 LIMIT 1").fetchone()
    if dup: raise RuntimeError("duplicate_ready_source_call_id")
    schema = "\n\n".join(row[0].rstrip(";") + ";" for row in con.execute("SELECT sql FROM sqlite_master WHERE tbl_name='call_records' AND sql IS NOT NULL ORDER BY type='table' DESC,name")) + "\n"
    total = con.execute("SELECT count(*) FROM call_records").fetchone()[0]
    controlled = con.execute(f"SELECT count(*) FROM call_records WHERE {BASE_READY} AND source_file LIKE '%/controlled-mango-call-%'").fetchone()[0]
    ready = con.execute(f"SELECT count(*) FROM call_records WHERE {READY}").fetchone()[0]
    fingerprints: dict[str, str] = {}; selected: list[tuple] = []; kinds: list[str] = []
    unchanged_tail: list[tuple] = []; all_new = all_changed = 0
    for row in con.execute(f"SELECT * FROM call_records WHERE {READY} ORDER BY updated_at,id"):
        parse_date(row["started_at"]); parse_date(row["updated_at"])
        source_id, digest = str(row["source_call_id"]).strip(), row_fingerprint(row)
        fingerprints[source_id] = digest
        if previous is None: continue
        if previous.get(source_id) != digest:
            kind = "new" if source_id not in previous else "changed"
            all_new += kind == "new"; all_changed += kind == "changed"
            if limit is None or len(selected) < limit: selected.append(tuple(row)); kinds.append(kind)
        elif limit is None:
            unchanged_tail.append(tuple(row)); unchanged_tail = unchanged_tail[-OVERLAP_ROWS:]
    if previous is not None and limit is None and selected:
        selected.extend(unchanged_tail); kinds.extend(["overlap"] * len(unchanged_tail))
    con.close()
    counts = {"total": total, "ready": ready, "not_ready": total - ready - controlled,
              "controlled_excluded": controlled, "new": kinds.count("new"),
              "changed": kinds.count("changed"), "overlap": kinds.count("overlap"),
              "unchanged": ready - all_new - all_changed,
              "omitted_changes": all_new + all_changed - kinds.count("new") - kinds.count("changed")}
    return Scan(columns, schema, selected, kinds, fingerprints, counts, utc_now())


def write_package(scan: Scan, root: Path) -> dict:
    db = root / "call_records_snapshot.sqlite"; schema_file = root / "schema.sql"
    schema_file.write_text("-- compatible call_records subset; no audio files\n" + scan.schema, encoding="utf-8")
    out = sqlite3.connect(db); out.executescript(scan.schema)
    if scan.rows:
        quoted = ",".join(f'"{name}"' for name in scan.columns)
        out.executemany(f"INSERT INTO call_records ({quoted}) VALUES ({','.join('?' for _ in scan.columns)})", scan.rows)
    out.commit(); quick = out.execute("PRAGMA quick_check").fetchone()[0]
    fk = len(out.execute("PRAGMA foreign_key_check").fetchall()); rows = out.execute("SELECT count(*) FROM call_records").fetchone()[0]
    duplicates = out.execute("SELECT count(*) FROM (SELECT source_call_id FROM call_records GROUP BY source_call_id HAVING count(*)>1)").fetchone()[0]
    indexes = out.execute("SELECT count(*) FROM sqlite_master WHERE type='index' AND tbl_name='call_records'").fetchone()[0]; out.close()
    if quick != "ok" or fk or duplicates or rows != len(scan.rows): raise RuntimeError("package_validation_failed")
    db.chmod(0o400); schema_file.chmod(0o600)
    sums = root / "SHA256SUMS"; sums.write_text(f"{sha256(db)}  {db.name}\n{sha256(schema_file)}  {schema_file.name}\n", encoding="ascii"); sums.chmod(0o600)
    return {"file": db.name, "sha256": sha256(db), "size_bytes": db.stat().st_size,
            "quick_check": quick, "foreign_key_violations": fk, "rows": rows,
            "columns": len(scan.columns), "indexes": indexes,
            "schema_sha256": sha256(schema_file), "sha256sums_sha256": sha256(sums)}


def verified_baseline(db: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); snapshot = manifest.get("snapshot") or {}
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "READY_FOR_M4_READ_ONLY_IMPORT_REVIEW":
        raise RuntimeError("baseline_manifest_contract_mismatch")
    if snapshot.get("file") != db.name or snapshot.get("sha256") != sha256(db) or snapshot.get("size_bytes") != db.stat().st_size:
        raise RuntimeError("baseline_snapshot_fingerprint_mismatch")
    con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    if con.execute("PRAGMA quick_check").fetchone()[0] != "ok" or con.execute("PRAGMA foreign_key_check").fetchone():
        raise RuntimeError("baseline_sqlite_integrity_failed")
    con.close(); return scan_db(db, None).fingerprints


def wait_yandex(log: Path, offset: int, inode: int, files: dict[str, tuple[str, int]], deadline: float) -> dict:
    while time.time() < deadline:
        stat = log.stat()
        if stat.st_ino != inode or stat.st_size < offset: raise RuntimeError("yandex_sync_log_rotated")
        with log.open("r", encoding="utf-8", errors="ignore") as handle: handle.seek(offset); text = handle.read()
        positions = []
        for rel, (digest, size) in files.items():
            remote = re.search(rf'REMOTE "{re.escape(rel)}" \S+ {digest[:8]} {size} file', text)
            prop = re.search(rf'PROP "{re.escape(rel)}" lh={digest[:8]},rh={digest[:8]}', text)
            if not remote or not prop: break
            positions.extend((remote.end(), prop.end()))
        else:
            idle = text.rfind("Core IDLE")
            if positions and idle > max(positions):
                return {"status": "remote_hash_size_and_idle_confirmed", "confirmed_at_utc": utc_now()}
        time.sleep(2)
    raise RuntimeError("yandex_sync_confirmation_timeout")


def publish(args: argparse.Namespace) -> dict:
    state, output = Path(args.state_dir).expanduser(), Path(args.output_root).expanduser()
    state.mkdir(parents=True, exist_ok=True, mode=0o700); output.mkdir(parents=True, exist_ok=True)
    journal_path = state / "journal.json"; journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    msk = datetime.now(ZoneInfo("Europe/Moscow")); day = msk.date().isoformat()
    if not args.manual and journal.get("last_delivery_msk_date") == day: return {"status": "already_delivered_today"}
    if not args.manual and not (90 <= msk.hour * 60 + msk.minute <= 175): raise RuntimeError("outside_01_30_02_55_msk_window")
    previous = journal.get("fingerprints")
    if previous is None: previous = verified_baseline(Path(args.baseline_db).expanduser(), Path(args.baseline_manifest).expanduser())
    scan = scan_db(Path(args.source_db).expanduser(), previous)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); name = f"M1_to_M4_processed_calls_{stamp}"
    package = output / name
    if package.exists(): raise RuntimeError("package_path_exists")
    log = Path(args.sync_log).expanduser(); mark = log.stat()
    deadline = time.time() + 1800 if args.manual else msk.replace(hour=2, minute=55, second=0, microsecond=0).timestamp()
    with tempfile.TemporaryDirectory(prefix="m1-calls-export-", dir=state) as tmp_name:
        staging = Path(tmp_name); snapshot = write_package(scan, staging); package.mkdir(mode=0o700)
        for filename in (snapshot["file"], "schema.sql", "SHA256SUMS"):
            partial = package / f".{filename}.partial"; shutil.copy2(staging / filename, partial); os.replace(partial, package / filename)
    payload_files = {str((package / f).relative_to(output.parent)): (sha256(package / f), (package / f).stat().st_size) for f in (snapshot["file"], "schema.sql", "SHA256SUMS")}
    copied_db = package / snapshot["file"]
    if sha256(copied_db) != snapshot["sha256"] or copied_db.stat().st_size != snapshot["size_bytes"]:
        raise RuntimeError("package_copy_fingerprint_mismatch")
    payload_sync = wait_yandex(log, mark.st_size, mark.st_ino, payload_files, deadline)
    if time.time() >= deadline: raise RuntimeError("delivery_deadline_reached_before_manifest")
    manifest = {"schema_version": SCHEMA_VERSION, "status": "READY_FOR_M4_READ_ONLY_IMPORT_REVIEW",
                "generator": GENERATOR, "created_at_utc": utc_now(), "package_name": name,
                "source": {"canonical_sqlite": str(Path(args.source_db).expanduser()), "read_only": True,
                           "source_mutated_by_export": False, "runs_asr_or_analysis": False, "writes_timeline": False},
                "snapshot": {**snapshot, "backup_finished_at_utc": scan.finished_at},
                "delta": {**scan.counts, "fingerprint_fields": list(FP_FIELDS),
                          "source_call_id_set_sha256": hashlib.sha256("\n".join(sorted(str(row[scan.columns.index('source_call_id')]) for row in scan.rows)).encode()).hexdigest()},
                "transfer": {"manifest_written_last": True, "payload_sync": payload_sync, "contains_audio": False, "contains_credentials": False}}
    manifest_mark = log.stat(); atomic_json(package / "manifest.json", manifest)
    manifest_sync = wait_yandex(log, manifest_mark.st_size, manifest_mark.st_ino,
                               {str((package / "manifest.json").relative_to(output.parent)): (sha256(package / "manifest.json"), (package / "manifest.json").stat().st_size)}, deadline)
    history = (journal.get("confirmed_packages") or []) + [name]
    atomic_json(journal_path, {"schema_version": 1, "fingerprints": scan.fingerprints,
                              "last_delivery_msk_date": day, "last_package": name,
                              "last_manifest_sha256": sha256(package / "manifest.json"),
                              "confirmed_packages": history[-3:], "confirmed_at_utc": manifest_sync["confirmed_at_utc"]})
    return {"status": "published", "package": str(package), "snapshot": snapshot,
            "manifest_sha256": sha256(package / "manifest.json"), "delta": scan.counts,
            "payload_sync": payload_sync, "manifest_sync": manifest_sync}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); p.add_argument("--source-db", required=True); p.add_argument("--baseline-db", required=True); p.add_argument("--baseline-manifest", required=True)
    p.add_argument("--output-root", required=True); p.add_argument("--state-dir", required=True); p.add_argument("--sync-log", required=True)
    p.add_argument("--manual", action="store_true"); return p


def main() -> int:
    os.umask(0o077); args = parser().parse_args(); state = Path(args.state_dir).expanduser(); state.mkdir(parents=True, exist_ok=True)
    expected = os.environ.get("M1_CALLS_EXPORT_EXPECTED_HEAD")
    if expected and subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip() != expected: raise RuntimeError("code_sha_mismatch")
    with (state / "export.lock").open("a+") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: print(json.dumps({"status": "already_running"})); return 0
        try: report = publish(args); atomic_json(state / "last_success.json", report); print(json.dumps(report, ensure_ascii=False)); return 0
        except Exception as exc:
            report = {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc), "failed_at_utc": utc_now()}
            atomic_json(state / "last_failure.json", report); print(json.dumps(report, ensure_ascii=False)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
