from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


def _parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _discover_dbs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("**/*.db")
        if path.is_file() and "venv" not in path.parts
    )


def _db_has_call_records(db_path: Path) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        names = {row[0] for row in cur.execute("select name from sqlite_master where type='table'")}
        conn.close()
        return "call_records" in names
    except sqlite3.Error:
        return False


def _iter_recognized_calls(db_path: Path) -> Iterable[tuple[str, str, Optional[str]]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        for row in cur.execute(
            "select source_file, source_filename, phone from call_records where transcription_status='done'"
        ):
            source_file = str(row[0] or "")
            source_filename = str(row[1] or "")
            phone = str(row[2] or "").strip() or None
            yield source_file, source_filename, phone
    finally:
        conn.close()


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all calls from source-dir for phones that appear in already recognized calls "
            "from the specified date window, then build a symlink batch with full history."
        )
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument(
        "--db",
        action="append",
        default=[],
        help="Optional explicit DB path. If omitted, all *.db files under workspace-root are scanned.",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    window_start = _parse_date(args.start_date)
    window_end = _parse_date(args.end_date)
    explicit_dbs = [Path(item).expanduser().resolve() for item in args.db]
    db_paths = explicit_dbs or _discover_dbs(workspace_root)
    db_paths = [path for path in db_paths if _db_has_call_records(path)]

    recognized_keys: set[str] = set()
    phones: set[str] = set()
    source_db_hits: Counter[str] = Counter()

    for db_path in db_paths:
        for source_file, source_filename, phone in _iter_recognized_calls(db_path):
            meta = parse_filename_metadata(source_filename)
            started_at = meta.get("started_at")
            if not started_at:
                continue
            current_date = started_at.date()
            if current_date < window_start or current_date > window_end:
                continue
            key = source_file or source_filename
            if not key or key in recognized_keys:
                continue
            recognized_keys.add(key)
            normalized_phone = (phone or meta.get("phone") or "").strip()
            if not normalized_phone:
                continue
            phones.add(normalized_phone)
            source_db_hits[str(db_path)] += 1

    selected_files: list[Path] = []
    selected_dates: list[str] = []
    for path in sorted(source_dir.iterdir(), key=lambda item: item.name, reverse=True):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        meta = parse_filename_metadata(path.name)
        normalized_phone = str(meta.get("phone") or "").strip()
        if not normalized_phone or normalized_phone not in phones:
            continue
        selected_files.append(path)
        started_at = meta.get("started_at")
        if started_at:
            selected_dates.append(started_at.date().isoformat())

    batch_dir = out_root / "batch_full_history"
    _clean_output_dir(batch_dir)

    for path in selected_files:
        link_path = batch_dir / path.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(path)

    manifest = {
        "workspace_root": str(workspace_root),
        "source_dir": str(source_dir),
        "date_window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "db_count": len(db_paths),
        "db_paths": [str(path) for path in db_paths],
        "recognized_calls_window_unique": len(recognized_keys),
        "unique_phones": len(phones),
        "matched_files_in_whole_folder": len(selected_files),
        "matched_date_min": min(selected_dates) if selected_dates else None,
        "matched_date_max": max(selected_dates) if selected_dates else None,
        "batch_dir": str(batch_dir),
        "selected_files": [path.name for path in selected_files],
        "top_source_dbs": source_db_hits.most_common(20),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
