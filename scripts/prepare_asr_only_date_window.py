from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


def _db_has_call_records(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(
            "select name from sqlite_master where type='table' and name='call_records'"
        ).fetchone()
    except sqlite3.Error:
        return False
    return bool(row)


def _collect_done_filenames(db_paths: list[Path], *, skip_root: Path) -> tuple[set[str], list[str], list[str]]:
    done: set[str] = set()
    usable: list[str] = []
    errors: list[str] = []
    for db_path in sorted(db_paths):
        try:
            db_path.resolve().relative_to(skip_root)
            continue
        except ValueError:
            pass
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            errors.append(f"{db_path}: open: {exc}")
            continue
        try:
            if not _db_has_call_records(conn):
                continue
            scanned = 0
            for (source_filename,) in conn.execute(
                """
                select source_filename
                  from call_records
                 where lower(coalesce(transcription_status, '')) = 'done'
                   and source_filename is not null
                   and source_filename != ''
                """
            ):
                filename = str(source_filename or "").strip()
                if filename:
                    done.add(filename)
                    scanned += 1
            if scanned:
                usable.append(str(db_path))
        except sqlite3.Error as exc:
            errors.append(f"{db_path}: scan: {exc}")
        finally:
            conn.close()
    return done, usable, errors


def _iter_audio_files(source_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()


def _started_at(path: Path) -> datetime | None:
    value = parse_filename_metadata(path.name).get("started_at")
    return value if isinstance(value, datetime) else None


def _month_key(value: datetime | None) -> str:
    return value.strftime("%Y-%m") if value else "unknown"


def _jsonable_datetime(value: Any) -> str:
    return value.isoformat(sep=" ") if isinstance(value, datetime) else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ASR-only symlink batch for untranscribed calls in a date window."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--db-search-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0, help="Optional max selected calls")
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio files into batch_asr_only instead of creating symlinks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    db_search_root = Path(args.db_search_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    batch_dir = out_root / "batch_asr_only"
    transcripts_dir = out_root / "transcripts"
    db_path = out_root / f"{out_root.name}.db"
    selected_calls_csv = out_root / "selected_calls.csv"
    phones_csv = out_root / "phones.csv"
    manifest_path = out_root / "selection_manifest.json"

    start_date = datetime.fromisoformat(args.start_date).date()
    end_date = datetime.fromisoformat(args.end_date).date()
    if end_date < start_date:
        raise SystemExit("--end-date must be >= --start-date")

    out_root.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    _clean_dir(batch_dir)

    db_paths = [path for path in db_search_root.rglob("*.db") if path.is_file()]
    done_filenames, usable_dbs, db_errors = _collect_done_filenames(db_paths, skip_root=out_root)

    malformed: list[str] = []
    window_files: list[Path] = []
    already_done = 0
    selected: list[Path] = []
    for path in _iter_audio_files(source_dir):
        started_at = _started_at(path)
        if started_at is None:
            malformed.append(path.name)
            continue
        if not (start_date <= started_at.date() <= end_date):
            continue
        window_files.append(path)
        if path.name in done_filenames:
            already_done += 1
            continue
        selected.append(path)

    selected.sort(key=lambda path: (_started_at(path) or datetime.min, path.name))
    eligible_untranscribed = len(selected)
    if args.limit and args.limit > 0:
        selected = selected[: int(args.limit)]

    for path in selected:
        link_path = batch_dir / path.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        if args.copy_audio:
            shutil.copy2(path, link_path)
        else:
            link_path.symlink_to(path)

    phones = sorted(
        {
            str(parse_filename_metadata(path.name).get("phone") or "").strip()
            for path in selected
            if str(parse_filename_metadata(path.name).get("phone") or "").strip()
        }
    )

    with selected_calls_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_filename", "source_file", "phone", "manager_name", "started_at"])
        for path in selected:
            meta = parse_filename_metadata(path.name)
            writer.writerow(
                [
                    path.name,
                    str(path),
                    str(meta.get("phone") or "").strip(),
                    str(meta.get("manager_name") or "").strip(),
                    _jsonable_datetime(meta.get("started_at")),
                ]
            )

    with phones_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phone"])
        for phone in phones:
            writer.writerow([phone])

    selected_months = Counter(_month_key(_started_at(path)) for path in selected)
    window_months = Counter(_month_key(_started_at(path)) for path in window_files)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "ASR-only date-window batch: calls with no successful transcription in local call_records DBs by source_filename",
        "source_dir": str(source_dir),
        "out_root": str(out_root),
        "batch_dir": str(batch_dir),
        "db_path": str(db_path),
        "transcripts_dir": str(transcripts_dir),
        "date_window": {"start": args.start_date, "end": args.end_date},
        "total_archive_calls_in_window": len(window_files),
        "archive_calls_by_month": dict(sorted(window_months.items())),
        "already_transcribed_by_filename": already_done,
        "eligible_untranscribed_calls": eligible_untranscribed,
        "limit": int(args.limit or 0) or None,
        "selected_untranscribed_calls": len(selected),
        "selected_by_month": dict(sorted(selected_months.items())),
        "unique_phones": len(phones),
        "malformed_filenames_skipped": len(malformed),
        "malformed_filename_samples": malformed[:20],
        "known_done_filenames": len(done_filenames),
        "db_files_scanned": len(db_paths),
        "usable_db_files_scanned": len(usable_dbs),
        "usable_db_files": usable_dbs,
        "db_scan_errors": db_errors,
        "selected_calls_csv": str(selected_calls_csv),
        "phones_csv": str(phones_csv),
        "selection_order": "oldest-to-newest by started_at for deterministic ASR-only processing",
        "batch_audio_mode": "copy" if args.copy_audio else "symlink",
        "pipeline_stages": {
            "transcribe": True,
            "backfill_second_asr": True,
            "resolve": False,
            "analyze": False,
            "sync": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
