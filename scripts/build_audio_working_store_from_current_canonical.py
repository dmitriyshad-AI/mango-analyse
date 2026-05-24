#!/usr/bin/env python3
"""Build a single deduplicated working audio folder from the current canonical DB.

Safety properties:
- reads current runtime/canonical DB only;
- does not delete or move source audio;
- creates hardlinks by default, falling back to copy only when hardlink is not possible;
- writes manifests so every canonical call can be traced back to the original source file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT_RUNTIME = ROOT / "stable_runtime" / "CURRENT_RUNTIME.json"
DEFAULT_OUT_ROOT = ROOT / "product_data" / "audio_working_store_20260523_v1"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def clean(value: object) -> str:
    return str(value or "").strip()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root))
    except ValueError:
        return str(path.resolve(strict=False))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_current_canonical_db(runtime_path: Path) -> Path:
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    value = clean((payload.get("paths") or {}).get("canonical_db"))
    if not value:
        raise ValueError(f"canonical_db is not set in {runtime_path}")
    db = Path(value).expanduser().resolve(strict=False)
    if not db.exists():
        raise FileNotFoundError(f"canonical DB does not exist: {db}")
    return db


def load_canonical_rows(db_path: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            select canonical_call_id, source_call_id, source_filename, source_file,
                   started_at, phone, manager_name, duration_sec, direction,
                   is_actionable, canonical_status, audio_size_bytes, audio_mtime,
                   selected_source_db, selected_call_record_id,
                   transcription_status, resolve_status, analysis_status
              from canonical_calls
             order by canonical_call_id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def bucket_for_source(source_file: str) -> str:
    if "canonical_audio_store_20260516_v1" in source_file:
        return "canonical_audio_store_20260516"
    if "mango_update_after_20260512_20260521_v1" in source_file:
        return "mango_update_20260521"
    if "2026-03-09--26" in source_file:
        return "main_2026-03-09--26"
    if "_local_archive_mango_api" in source_file:
        return "local_archive_mango_api"
    return "other"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_link_or_copy(src: Path, dst: Path, *, mode: str) -> str:
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise ValueError(f"existing target has different size: {dst}")
        return "existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(src, dst)
        return "copy_fallback"


def build_store(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve(strict=False)
    runtime_path = Path(args.current_runtime).resolve(strict=False)
    out_root = Path(args.out_root).resolve(strict=False)
    audio_root = out_root / "audio"
    by_filename_root = out_root / "by_filename"
    manifests_root = out_root / "manifests"
    canonical_db = Path(args.canonical_db).resolve(strict=False) if args.canonical_db else load_current_canonical_db(runtime_path)
    rows = load_canonical_rows(canonical_db)

    call_rows: list[dict[str, Any]] = []
    unique_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    missing_rows: list[dict[str, Any]] = []
    size_mismatch_rows: list[dict[str, Any]] = []
    invalid_ext_rows: list[dict[str, Any]] = []
    link_actions = Counter()
    source_buckets = Counter()

    started = datetime.now(timezone.utc)
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        source_file = clean(row.get("source_file"))
        source_path = Path(source_file).expanduser().resolve(strict=False)
        source_bucket = bucket_for_source(source_file)
        source_buckets[source_bucket] += 1
        base = source_path.name
        ext = source_path.suffix.lower() or Path(clean(row.get("source_filename"))).suffix.lower()
        if ext not in AUDIO_EXTS:
            invalid_ext_rows.append({**row, "source_file": source_file, "reason": f"unexpected audio extension: {ext}"})
        if not source_path.exists():
            missing_rows.append({**row, "source_file": source_file, "reason": "source_file_missing"})
            continue
        size = source_path.stat().st_size
        expected_size = int(row.get("audio_size_bytes") or 0)
        if expected_size and size != expected_size:
            size_mismatch_rows.append({**row, "source_file": source_file, "actual_size_bytes": size, "reason": "source_size_mismatch"})
            continue
        sha = sha256_file(source_path)
        unique_key = (sha, ext)
        target_rel = Path("audio") / sha[:2] / f"{sha}{ext}"
        target_path = out_root / target_rel
        action = ensure_link_or_copy(source_path, target_path, mode=args.mode)
        link_actions[action] += 1
        unique_rows_by_key.setdefault(
            unique_key,
            {
                "sha256": sha,
                "ext": ext,
                "size_bytes": size,
                "canonical_audio_path": str(target_rel),
                "first_source_file": rel(source_path, project_root),
                "first_source_filename": base,
                "source_bucket": source_bucket,
                "link_action": action,
            },
        )
        call_rows.append(
            {
                "canonical_call_id": row.get("canonical_call_id"),
                "source_call_id": clean(row.get("source_call_id")),
                "started_at": clean(row.get("started_at")),
                "phone": clean(row.get("phone")),
                "manager_name": clean(row.get("manager_name")),
                "duration_sec": row.get("duration_sec"),
                "direction": clean(row.get("direction")),
                "is_actionable": row.get("is_actionable"),
                "canonical_status": clean(row.get("canonical_status")),
                "transcription_status": clean(row.get("transcription_status")),
                "resolve_status": clean(row.get("resolve_status")),
                "analysis_status": clean(row.get("analysis_status")),
                "source_bucket": source_bucket,
                "source_filename": clean(row.get("source_filename")),
                "original_source_file": rel(source_path, project_root),
                "audio_size_bytes": size,
                "sha256": sha,
                "canonical_audio_path": str(target_rel),
                "canonical_audio_abs_path": str(target_path),
                "by_filename_path": str(Path("by_filename") / clean(row.get("source_filename"))),
                "selected_source_db": clean(row.get("selected_source_db")),
                "selected_call_record_id": clean(row.get("selected_call_record_id")),
            }
        )
        if args.progress_every and (idx % args.progress_every == 0 or idx == total):
            print(json.dumps({"processed": idx, "total": total, "unique_audio": len(unique_rows_by_key)}, ensure_ascii=False), flush=True)

    duplicate_counts = Counter((row["sha256"], Path(row["canonical_audio_path"]).suffix.lower()) for row in call_rows)
    for row in call_rows:
        row["duplicate_group_size"] = duplicate_counts[(row["sha256"], Path(row["canonical_audio_path"]).suffix.lower())]
        row["is_exact_duplicate_audio"] = "Да" if row["duplicate_group_size"] > 1 else "Нет"

    unique_rows = []
    for row in unique_rows_by_key.values():
        sha = row["sha256"]
        ext = row["ext"]
        row = dict(row)
        row["call_count"] = duplicate_counts[(sha, ext)]
        row["is_shared_by_multiple_calls"] = "Да" if row["call_count"] > 1 else "Нет"
        unique_rows.append(row)
    unique_rows.sort(key=lambda r: (r["sha256"], r["ext"]))

    by_filename_created = 0
    by_filename_existing = 0
    by_filename_errors: list[dict[str, str]] = []
    by_filename_root.mkdir(parents=True, exist_ok=True)
    for row in call_rows:
        source_filename = clean(row.get("source_filename"))
        canonical_audio_path = clean(row.get("canonical_audio_path"))
        if not source_filename or not canonical_audio_path:
            continue
        link_path = by_filename_root / source_filename
        target = Path("..") / canonical_audio_path
        try:
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink() and os.readlink(link_path) == str(target):
                    by_filename_existing += 1
                    continue
                link_path.unlink()
            link_path.symlink_to(target)
            by_filename_created += 1
        except OSError as exc:
            by_filename_errors.append(
                {
                    "canonical_call_id": clean(row.get("canonical_call_id")),
                    "source_filename": source_filename,
                    "target": str(target),
                    "error": repr(exc),
                }
            )

    write_csv(
        manifests_root / "call_audio_mapping.csv",
        call_rows,
        [
            "canonical_call_id",
            "source_call_id",
            "started_at",
            "phone",
            "manager_name",
            "duration_sec",
            "direction",
            "is_actionable",
            "canonical_status",
            "transcription_status",
            "resolve_status",
            "analysis_status",
            "source_bucket",
            "source_filename",
            "original_source_file",
            "audio_size_bytes",
            "sha256",
            "canonical_audio_path",
            "canonical_audio_abs_path",
            "by_filename_path",
            "duplicate_group_size",
            "is_exact_duplicate_audio",
            "selected_source_db",
            "selected_call_record_id",
        ],
    )
    write_csv(
        manifests_root / "unique_audio_manifest.csv",
        unique_rows,
        [
            "sha256",
            "ext",
            "size_bytes",
            "canonical_audio_path",
            "first_source_file",
            "first_source_filename",
            "source_bucket",
            "link_action",
            "call_count",
            "is_shared_by_multiple_calls",
        ],
    )
    if missing_rows:
        write_csv(manifests_root / "missing_source_files.csv", missing_rows, sorted({k for r in missing_rows for k in r.keys()}))
    if size_mismatch_rows:
        write_csv(manifests_root / "size_mismatch_source_files.csv", size_mismatch_rows, sorted({k for r in size_mismatch_rows for k in r.keys()}))
    if invalid_ext_rows:
        write_csv(manifests_root / "unexpected_extension_files.csv", invalid_ext_rows, sorted({k for r in invalid_ext_rows for k in r.keys()}))
    if by_filename_errors:
        write_csv(
            manifests_root / "by_filename_link_errors.csv",
            by_filename_errors,
            ["canonical_call_id", "source_filename", "target", "error"],
        )

    summary = {
        "schema_version": "audio_working_store_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "current_runtime": rel(runtime_path, project_root),
        "canonical_db": rel(canonical_db, project_root),
        "out_root": rel(out_root, project_root),
        "audio_root": rel(audio_root, project_root),
        "by_filename_root": rel(by_filename_root, project_root),
        "source_rows": total,
        "call_rows_mapped": len(call_rows),
        "unique_audio_files": len(unique_rows),
        "exact_duplicate_call_rows": sum(1 for row in call_rows if row["duplicate_group_size"] > 1),
        "exact_duplicate_groups": sum(1 for row in unique_rows if row["call_count"] > 1),
        "missing_source_files": len(missing_rows),
        "size_mismatch_source_files": len(size_mismatch_rows),
        "unexpected_extension_files": len(invalid_ext_rows),
        "source_bucket_counts": dict(source_buckets),
        "link_action_counts": dict(link_actions),
        "by_filename_links_created": by_filename_created,
        "by_filename_links_existing": by_filename_existing,
        "by_filename_link_errors": len(by_filename_errors),
        "mode": args.mode,
        "outputs": {
            "call_audio_mapping_csv": rel(manifests_root / "call_audio_mapping.csv", project_root),
            "unique_audio_manifest_csv": rel(manifests_root / "unique_audio_manifest.csv", project_root),
            "summary_json": rel(manifests_root / "summary.json", project_root),
            "by_filename_root": rel(by_filename_root, project_root),
        },
        "safety": {
            "deleted_files": False,
            "moved_source_files": False,
            "source_audio_modified": False,
            "stable_runtime_modified": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_tallanto": False,
        },
        "validation_ok": len(call_rows) == total
        and not missing_rows
        and not size_mismatch_rows
        and not invalid_ext_rows
        and not by_filename_errors,
        "started_at": started.isoformat(),
    }
    (manifests_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "README.md").write_text(readme_text(summary), encoding="utf-8")
    return summary


def readme_text(summary: dict[str, Any]) -> str:
    return f"""# Audio Working Store

Единая рабочая папка аудиозаписей текущей canonical DB.

Ключевые свойства:

- исходные аудиофайлы не удалялись и не перемещались;
- внутри `audio/` лежит один файл на один уникальный SHA-256;
- `manifests/call_audio_mapping.csv` связывает каждый canonical_call_id с рабочим аудиофайлом;
- `manifests/unique_audio_manifest.csv` содержит список уникальных аудио;
- `by_filename/` содержит символические ссылки с исходными именами файлов на `audio/`;
- сборка не запускала ASR, Resolve+Analyze, CRM/AMO/Tallanto write.

Итог:

- строк в canonical DB: `{summary['source_rows']}`;
- привязано звонков: `{summary['call_rows_mapped']}`;
- уникальных аудиофайлов: `{summary['unique_audio_files']}`;
- групп точных дублей: `{summary['exact_duplicate_groups']}`;
- validation_ok: `{summary['validation_ok']}`.
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deduplicated audio working store from current canonical DB.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--current-runtime", default=str(DEFAULT_CURRENT_RUNTIME))
    parser.add_argument("--canonical-db", default="")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--mode", choices=["auto", "hardlink", "copy"], default="auto")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_store(args)
    except Exception as exc:
        print(f"audio working store build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("validation_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
