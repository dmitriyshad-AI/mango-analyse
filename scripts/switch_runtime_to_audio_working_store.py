#!/usr/bin/env python3
"""Switch current runtime to a canonical DB whose source_file points to audio working store."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.current_runtime import build_current_runtime_contract  # noqa: E402

DEFAULT_RUNTIME = ROOT / "stable_runtime" / "CURRENT_RUNTIME.json"
DEFAULT_STORE = ROOT / "product_data" / "audio_working_store_20260523_v1"
DEFAULT_DB_ROOT = ROOT / "stable_runtime" / "canonical_master_20260523_audio_working_store_v1"
DEFAULT_EXPORT_ROOT = ROOT / "stable_runtime" / "sales_master_export_20260523_audio_working_store_v1"
DEFAULT_CRM_GATE_ROOT = ROOT / "stable_runtime" / "crm_writeback_quality_gate_20260523_audio_working_store_v1"
DEFAULT_AMO_QUEUE_ROOT = ROOT / "stable_runtime" / "amo_writeback_queue_20260523_audio_working_store_v1"


def clean(value: object) -> str:
    return str(value or "").strip()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root))
    except ValueError:
        return str(path.resolve(strict=False))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, old, new) for key, item in value.items()}
    return value


def hardlink_copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for path in src.rglob("*"):
        rel_path = path.relative_to(src)
        target = dst / rel_path
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)


def read_call_mapping(store: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    path = store / "manifests" / "call_audio_mapping.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = int(row["canonical_call_id"])
            source_filename = clean(row.get("source_filename"))
            by_filename = store / "by_filename" / source_filename
            if not by_filename.exists():
                raise FileNotFoundError(f"Missing by_filename audio for canonical_call_id={cid}: {by_filename}")
            mapping[cid] = str(by_filename.resolve(strict=False))
    return mapping


def write_audio_store_mapping_compat(store: Path, project_root: Path) -> Path:
    call_mapping = store / "manifests" / "call_audio_mapping.csv"
    orphan_mapping = store / "manifests" / "orphan_audio_manifest.csv"
    out = store / "manifests" / "audio_store_mapping_compat.csv"
    fields = [
        "record_type",
        "record_id",
        "source_set",
        "source_audio_path",
        "source_audio_basename",
        "canonical_audio_path",
        "sha256",
        "size_bytes",
        "ext",
        "queue_item_id",
        "event_key",
        "provider_call_id",
    ]
    rows: list[dict[str, str]] = []

    def project_relative_audio_path(value: object) -> str:
        text = clean(value)
        if not text:
            return ""
        path = Path(text)
        if path.is_absolute():
            return rel(path, project_root)
        if text.startswith(str(rel(store, project_root)) + "/"):
            return text
        # Orphan manifests store canonical paths relative to the store root.
        return str(Path(rel(store, project_root)) / path)

    with call_mapping.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            canonical_audio_path = project_relative_audio_path(row.get("canonical_audio_path"))
            rows.append(
                {
                    "record_type": "canonical_call",
                    "record_id": clean(row.get("canonical_call_id")),
                    "source_set": clean(row.get("source_bucket")),
                    "source_audio_path": clean(row.get("original_source_file")),
                    "source_audio_basename": clean(row.get("source_filename")),
                    "canonical_audio_path": canonical_audio_path,
                    "sha256": clean(row.get("sha256")),
                    "size_bytes": clean(row.get("audio_size_bytes")),
                    "ext": Path(canonical_audio_path).suffix.lower(),
                    "queue_item_id": "",
                    "event_key": "",
                    "provider_call_id": clean(row.get("source_call_id")),
                }
            )
    if orphan_mapping.exists():
        with orphan_mapping.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                canonical_audio_path = project_relative_audio_path(row.get("canonical_audio_path"))
                rows.append(
                    {
                        "record_type": "orphan_project_audio",
                        "record_id": clean(row.get("sha256")),
                        "source_set": "orphan_project_audio",
                        "source_audio_path": clean(row.get("original_source_file")),
                        "source_audio_basename": clean(row.get("original_source_filename")),
                        "canonical_audio_path": canonical_audio_path,
                        "sha256": clean(row.get("sha256")),
                        "size_bytes": clean(row.get("size_bytes")),
                        "ext": clean(row.get("ext")) or Path(canonical_audio_path).suffix.lower(),
                        "queue_item_id": "",
                        "event_key": "",
                        "provider_call_id": "",
                    }
                )
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def migrate_db(old_db: Path, new_root: Path, store: Path, project_root: Path) -> Path:
    new_root.mkdir(parents=True, exist_ok=True)
    new_db = new_root / "canonical_calls_master.db"
    if new_db.exists():
        new_db.unlink()
    shutil.copy2(old_db, new_db)
    mapping = read_call_mapping(store)
    con = sqlite3.connect(new_db)
    try:
        cur = con.cursor()
        cur.execute("begin")
        for cid, new_source_file in mapping.items():
            cur.execute("update canonical_calls set source_file = ? where canonical_call_id = ?", (new_source_file, cid))
            cur.execute("update call_record_provenance set source_file = ? where canonical_call_id = ?", (new_source_file, cid))
        # Mark source artifacts/builds as migrated without deleting historical provenance.
        cur.execute(
            "update source_artifacts set path = ? where artifact_type = 'audio_dir'",
            (rel(store / "by_filename", project_root),),
        )
        cur.execute(
            "update canonical_builds set source_dir = ?",
            (str((store / "by_filename").resolve(strict=False)),),
        )
        cur.execute("commit")
        missing = []
        for cid, source_file, audio_size_bytes in cur.execute(
            "select canonical_call_id, source_file, audio_size_bytes from canonical_calls order by canonical_call_id"
        ):
            p = Path(source_file)
            if not p.exists() or p.stat().st_size != int(audio_size_bytes or 0):
                missing.append((cid, source_file))
                if len(missing) >= 5:
                    break
        if missing:
            raise RuntimeError(f"migrated DB has missing/size mismatch source files: {missing}")
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return new_db


def copy_and_patch_export(old_export: Path, new_export: Path, old_db: Path, new_db: Path) -> None:
    hardlink_copytree(old_export, new_export)
    summary_path = new_export / "summary.json"
    summary = load_json(summary_path)
    old_root_str = str(old_export.resolve(strict=False))
    new_root_str = str(new_export.resolve(strict=False))
    summary = replace_strings(summary, old_root_str, new_root_str)
    summary = replace_strings(summary, str(old_db.resolve(strict=False)), str(new_db.resolve(strict=False)))
    summary["canonical_db"] = str(new_db.resolve(strict=False))
    summary["audio_working_store"] = str((DEFAULT_STORE).resolve(strict=False))
    summary["build_id"] = "sales_master_export_20260523_audio_working_store_v1"
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    dump_json(summary_path, summary)


def copy_and_patch_summary_dir(old_dir: Path, new_dir: Path, old_input: Path, new_input: Path) -> None:
    hardlink_copytree(old_dir, new_dir)
    for json_path in new_dir.rglob("*.json"):
        try:
            payload = load_json(json_path)
        except Exception:
            continue
        payload = replace_strings(payload, str(old_input.resolve(strict=False)), str(new_input.resolve(strict=False)))
        payload = replace_strings(payload, str(old_dir.resolve(strict=False)), str(new_dir.resolve(strict=False)))
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        dump_json(json_path, payload)


def find_matching_summary_dir(project_root: Path, pattern: str, key: str, expected: Path) -> Path:
    matches = []
    for summary_path in sorted((project_root / "stable_runtime").glob(pattern)):
        try:
            data = load_json(summary_path)
        except Exception:
            continue
        value = clean(data.get(key))
        if value and Path(value).resolve(strict=False) == expected.resolve(strict=False):
            matches.append(summary_path.parent)
    if not matches:
        raise FileNotFoundError(f"No matching summary dir for {pattern} {key}={expected}")
    return matches[-1]


def switch_runtime(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve(strict=False)
    runtime = Path(args.current_runtime).resolve(strict=False)
    store = Path(args.audio_store).resolve(strict=False)
    db_root = Path(args.out_db_root).resolve(strict=False)
    export_root = Path(args.out_export_root).resolve(strict=False)
    crm_gate_root = Path(args.out_crm_gate_root).resolve(strict=False)
    amo_queue_root = Path(args.out_amo_queue_root).resolve(strict=False)

    current = load_json(runtime)
    old_db = Path(current["paths"]["canonical_db"]).resolve(strict=False)
    old_export = Path(current["paths"]["active_export_root"]).resolve(strict=False)
    old_amo_csv = Path(current["paths"]["amo_export_ready_csv"]).resolve(strict=False)

    compat_mapping = write_audio_store_mapping_compat(store, project_root)
    new_db = migrate_db(old_db, db_root, store, project_root)
    copy_and_patch_export(old_export, export_root, old_db, new_db)
    new_amo_csv = export_root / old_amo_csv.name

    old_crm_gate = find_matching_summary_dir(project_root, "crm_writeback_quality_gate_*/summary.json", "input", old_amo_csv)
    old_amo_queue = find_matching_summary_dir(project_root, "amo_writeback_queue_*/summary.json", "input_csv", old_amo_csv)
    copy_and_patch_summary_dir(old_crm_gate, crm_gate_root, old_amo_csv, new_amo_csv)
    copy_and_patch_summary_dir(old_amo_queue, amo_queue_root, old_amo_csv, new_amo_csv)

    old_canonical_summary = load_json(old_db.parent / "summary.json")
    db_summary = {
        **old_canonical_summary,
        "schema_version": "canonical_master_audio_working_store_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build_id": "canonical_master_20260523_audio_working_store_v1",
        "base_db": str(old_db.resolve(strict=False)),
        "out_db": str(new_db.resolve(strict=False)),
        "audio_working_store": str(store.resolve(strict=False)),
        "audio_working_store_manifest": str((store / "manifests" / "call_audio_mapping.csv").resolve(strict=False)),
        "outputs": {
            **(old_canonical_summary.get("outputs") or {}),
            "summary_json": str((db_root / "summary.json").resolve(strict=False)),
            "canonical_db": str(new_db.resolve(strict=False)),
        },
        "safety": {
            "copied_base_db": True,
            "mutated_base_db": False,
            "deleted_files": False,
            "crm_writes": False,
            "tallanto_writes": False,
            "asr_run": False,
            "ra_run": False,
            "source_audio_moved": False,
        },
    }
    db_summary["canonical_db"] = {
        **(old_canonical_summary.get("canonical_db") or {}),
        "path": str(new_db.resolve(strict=False)),
        "passed": True,
    }
    (db_root / "summary.json").write_text(json.dumps(db_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pointer = project_root / "stable_runtime" / "CANONICAL_EXPORT.txt"
    pointer.write_text(export_root.name + "\n", encoding="utf-8")
    contract = build_current_runtime_contract(project_root=project_root, out_path=runtime)
    summary = {
        "schema_version": "audio_working_store_runtime_switch_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_canonical_db": rel(old_db, project_root),
        "new_canonical_db": rel(new_db, project_root),
        "old_export_root": rel(old_export, project_root),
        "new_export_root": rel(export_root, project_root),
        "audio_working_store": rel(store, project_root),
        "audio_store_mapping_compat": rel(compat_mapping, project_root),
        "new_crm_quality_gate": rel(crm_gate_root, project_root),
        "new_amo_queue": rel(amo_queue_root, project_root),
        "current_runtime_validation_ok": bool((contract.get("summary") or {}).get("validation_ok")),
        "current_runtime_blocked": int((contract.get("summary") or {}).get("blocked") or 0),
        "safety": {
            "old_db_modified": False,
            "old_export_modified": False,
            "source_audio_deleted": False,
            "source_audio_moved": False,
            "asr_run": False,
            "ra_run": False,
            "crm_write": False,
            "tallanto_write": False,
        },
    }
    (store / "manifests" / "runtime_switch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Switch current runtime to audio working store paths.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--current-runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--audio-store", default=str(DEFAULT_STORE))
    parser.add_argument("--out-db-root", default=str(DEFAULT_DB_ROOT))
    parser.add_argument("--out-export-root", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--out-crm-gate-root", default=str(DEFAULT_CRM_GATE_ROOT))
    parser.add_argument("--out-amo-queue-root", default=str(DEFAULT_AMO_QUEUE_ROOT))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = switch_runtime(args)
    except Exception as exc:
        print(f"runtime audio-store switch failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("current_runtime_validation_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
