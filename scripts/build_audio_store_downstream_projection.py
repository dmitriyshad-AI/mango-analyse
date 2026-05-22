#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.audio_store import AudioStoreIndex  # noqa: E402
from mango_mvp.productization.processing_handoff import (  # noqa: E402
    ASR_HANDOFF_STATUS,
    PROCESSING_HANDOFF_SCHEMA_VERSION,
)


DEFAULT_CURRENT_RUNTIME = ROOT / "stable_runtime" / "CURRENT_RUNTIME.json"


def default_canonical_db() -> str:
    if DEFAULT_CURRENT_RUNTIME.exists():
        payload = json.loads(DEFAULT_CURRENT_RUNTIME.read_text(encoding="utf-8"))
        value = clean((payload.get("paths") or {}).get("canonical_db"))
        if value:
            return value
    raise FileNotFoundError(f"Cannot resolve current canonical DB from {DEFAULT_CURRENT_RUNTIME}")


def clean(value: object) -> str:
    return str(value or "").strip()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root))
    except ValueError:
        return str(path.resolve(strict=False))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_canonical_rows(db_path: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            select canonical_call_id, source_filename, source_file, started_at, phone,
                   manager_name, duration_sec, is_actionable, canonical_status
              from canonical_calls
             order by started_at, canonical_call_id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def build_projection(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve(strict=False)
    out_dir = Path(args.out_dir).resolve(strict=False)
    mapping_csv = Path(args.audio_store_mapping).resolve(strict=False)
    canonical_db = Path(args.canonical_db).resolve(strict=False)
    queue_csv = Path(args.new_queue_csv).resolve(strict=False)
    index = AudioStoreIndex(mapping_csv=mapping_csv, project_root=project_root)

    canonical_rows = load_canonical_rows(canonical_db)
    projection_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in canonical_rows:
        record = index.resolve(
            record_type="canonical_call",
            record_id=clean(row.get("canonical_call_id")),
            source_audio_path=clean(row.get("source_file")),
            source_filename=clean(row.get("source_filename")),
        )
        if record is None:
            unresolved.append({"record_type": "canonical_call", **row, "reason": "audio_store_mapping_not_found"})
            continue
        audio_path = index.canonical_path(record)
        projection_rows.append(
            {
                **row,
                "original_source_file": clean(row.get("source_file")),
                "original_source_filename": clean(row.get("source_filename")),
                "audio_store_source_file": rel(audio_path, project_root),
                "audio_store_source_filename": audio_path.name,
                "audio_store_sha256": record.sha256,
                "audio_store_size_bytes": record.size_bytes,
                "audio_store_exists": str(audio_path.exists()).lower(),
            }
        )

    queue_rows = read_csv(queue_csv)
    handoff_rows: list[dict[str, Any]] = []
    for row in queue_rows:
        record = index.resolve(
            record_type="new_mango_queue",
            record_id=clean(row.get("queue_item_id")),
            queue_item_id=clean(row.get("queue_item_id")),
            event_key=clean(row.get("event_key")),
            provider_call_id=clean(row.get("provider_call_id")),
            source_audio_path=clean(row.get("audio_path")),
            sha256=clean(row.get("audio_sha256")),
        )
        if record is None:
            unresolved.append({"record_type": "new_mango_queue", **row, "reason": "audio_store_mapping_not_found"})
            continue
        audio_path = index.canonical_path(record)
        item = {
            "schema_version": PROCESSING_HANDOFF_SCHEMA_VERSION,
            "queue_status": ASR_HANDOFF_STATUS,
            "queue_item_id": clean(row.get("queue_item_id")),
            "asset_id": 0,
            "tenant_id": "foton",
            "provider": "mango",
            "event_key": clean(row.get("event_key")),
            "provider_call_id": clean(row.get("provider_call_id")),
            "recording_id": clean(row.get("recording_id")) or clean(row.get("recording_ref")),
            "package_ref": "canonical_audio_store_20260516_v1",
            "audio_path": rel(audio_path, project_root),
            "source_filename": audio_path.name,
            "checksum_sha256": record.sha256,
            "size_bytes": record.size_bytes,
            "duration_sec": None,
            "started_at": clean(row.get("started_at_utc")) or None,
            "direction": None,
            "client_phone": clean(row.get("client_phone")) or None,
            "manager_ref": clean(row.get("manager_ref")) or None,
            "manager_name": clean(row.get("manager_ref")) or None,
            "planned_outputs_rel": {
                "transcript_json": f"outputs/foton/mango/{clean(row.get('queue_item_id'))}.transcript.json",
                "transcript_txt": f"outputs/foton/mango/{clean(row.get('queue_item_id'))}.transcript.txt",
                "asr_audit_json": f"outputs/foton/mango/{clean(row.get('queue_item_id'))}.asr_audit.json",
            },
            "source_refs": {
                "original_audio_path": clean(row.get("audio_path")),
                "audio_store_mapping": rel(mapping_csv, project_root),
                "source_manifest": clean(row.get("source_manifest")),
            },
        }
        if args.verify_checksum:
            actual = sha256_file(audio_path)
            if actual != record.sha256:
                unresolved.append({"record_type": "new_mango_queue", **row, "reason": "audio_store_checksum_mismatch"})
                continue
        handoff_rows.append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    projection_csv = out_dir / "canonical_calls_audio_store_projection.csv"
    handoff_jsonl = out_dir / "new_mango_processing_handoff_audio_store.jsonl"
    unresolved_csv = out_dir / "audio_store_projection_unresolved.csv"
    write_csv(
        projection_csv,
        projection_rows,
        [
            "canonical_call_id",
            "source_filename",
            "source_file",
            "started_at",
            "phone",
            "manager_name",
            "duration_sec",
            "is_actionable",
            "canonical_status",
            "original_source_file",
            "original_source_filename",
            "audio_store_source_file",
            "audio_store_source_filename",
            "audio_store_sha256",
            "audio_store_size_bytes",
            "audio_store_exists",
        ],
    )
    write_jsonl(handoff_jsonl, handoff_rows)
    if unresolved:
        write_csv(unresolved_csv, unresolved, sorted({key for row in unresolved for key in row.keys()}))
    elif unresolved_csv.exists():
        unresolved_csv.unlink()

    target_missing = []
    for row in projection_rows:
        p = project_root / clean(row.get("audio_store_source_file"))
        if not p.exists() or p.stat().st_size != int(row.get("audio_store_size_bytes") or 0):
            target_missing.append(row)
    for row in handoff_rows:
        p = project_root / clean(row.get("audio_path"))
        if not p.exists() or p.stat().st_size != int(row.get("size_bytes") or 0):
            target_missing.append(row)

    summary = {
        "schema_version": "audio_store_downstream_projection_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "audio_store_mapping": rel(mapping_csv, project_root),
        "canonical_db": rel(canonical_db, project_root),
        "new_queue_csv": rel(queue_csv, project_root),
        "canonical_rows_seen": len(canonical_rows),
        "canonical_rows_projected": len(projection_rows),
        "new_queue_rows_seen": len(queue_rows),
        "new_queue_handoff_rows": len(handoff_rows),
        "unresolved_rows": len(unresolved),
        "target_missing_or_size_mismatch": len(target_missing),
        "handoff_queue_status_counts": dict(Counter(row.get("queue_status") for row in handoff_rows)),
        "outputs": {
            "canonical_projection_csv": rel(projection_csv, project_root),
            "new_mango_processing_handoff_jsonl": rel(handoff_jsonl, project_root),
            "unresolved_csv": rel(unresolved_csv, project_root) if unresolved else None,
        },
        "safety": {
            "stable_runtime_modified": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_tallanto": False,
            "deleted_files": False,
            "moved_files": False,
        },
        "validation_ok": len(unresolved) == 0 and len(target_missing) == 0,
    }
    report_path = out_dir / "downstream_projection_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only downstream projection using canonical audio store paths.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--audio-store-mapping", default="product_data/canonical_audio_store_20260516_v1/audio_store_mapping.csv")
    parser.add_argument("--canonical-db", default=default_canonical_db())
    parser.add_argument("--new-queue-csv", default="product_data/mango_audio_update_20260516_v1/asr_handoff_new_calls_20260516.csv")
    parser.add_argument("--out-dir", default="product_data/canonical_audio_store_20260516_v1/downstream_projection")
    parser.add_argument("--no-verify-checksum", action="store_true")
    args = parser.parse_args(argv)
    args.verify_checksum = not args.no_verify_checksum
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_projection(args)
    except Exception as exc:
        print(f"audio-store downstream projection failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("validation_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
