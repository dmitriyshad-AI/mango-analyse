from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        name = Path(str(href)).name
        if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
            self.hrefs.append(name)


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()


def _iter_audio_files(path: Path) -> list[Path]:
    return sorted(
        [
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda item: item.name,
    )


def _extract_dates(paths: list[Path]) -> list[str]:
    dates: list[str] = []
    for path in paths:
        started_at = parse_filename_metadata(path.name).get("started_at")
        if started_at is not None:
            dates.append(started_at.date().isoformat())
    return dates


def _safe_scalar(cur: sqlite3.Cursor, query: str) -> int:
    value = cur.execute(query).fetchone()
    if not value:
        return 0
    return int(value[0] or 0)


def _db_has_call_records(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    try:
        row = cur.execute(
            "select name from sqlite_master where type='table' and name='call_records'"
        ).fetchone()
    except sqlite3.Error:
        return False
    return bool(row)


def _collect_db_index(db_paths: list[Path]) -> tuple[dict[str, dict[str, bool | int]], list[str]]:
    by_filename: dict[str, dict[str, bool | int]] = {}
    usable_dbs: list[str] = []
    for db_path in sorted(db_paths):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            if not _db_has_call_records(conn):
                continue
            cur = conn.cursor()
            try:
                rows = cur.execute(
                    """
                    select
                        source_filename,
                        transcription_status,
                        resolve_status,
                        analysis_status
                    from call_records
                    """
                )
            except sqlite3.Error:
                continue
            seen_any = 0
            for source_filename, transcription_status, resolve_status, analysis_status in rows:
                filename = str(source_filename or "").strip()
                if not filename:
                    continue
                state = by_filename.setdefault(
                    filename,
                    {
                        "seen_in_any_db": False,
                        "transcription_done": False,
                        "resolve_terminal": False,
                        "analysis_done": False,
                        "db_hits": 0,
                    },
                )
                state["seen_in_any_db"] = True
                state["db_hits"] = int(state["db_hits"]) + 1
                if str(transcription_status or "").strip().lower() == "done":
                    state["transcription_done"] = True
                if str(resolve_status or "").strip().lower() in {"done", "skipped", "manual"}:
                    state["resolve_terminal"] = True
                if str(analysis_status or "").strip().lower() == "done":
                    state["analysis_done"] = True
                seen_any += 1
            if seen_any:
                usable_dbs.append(str(db_path))
        finally:
            conn.close()
    return by_filename, usable_dbs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect full phone history for one repaired messages(N) archive, exclude calls that "
            "already have transcription done in existing DBs, and build a new full-cycle batch."
        )
    )
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--normalized-source-dir", required=True)
    parser.add_argument("--db-search-root", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    archive_dir = Path(args.archive_dir).expanduser().resolve()
    normalized_source_dir = Path(args.normalized_source_dir).expanduser().resolve()
    db_search_root = Path(args.db_search_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    index_path = archive_dir / "index.html"
    if not index_path.exists():
        raise SystemExit(f"Missing index.html in {archive_dir}")

    parser_html = _IndexParser()
    parser_html.feed(index_path.read_text("utf-8"))
    href_names = [Path(name).name for name in parser_html.hrefs]
    unique_href_names = list(dict.fromkeys(href_names))
    duplicate_ref_overflow = len(href_names) - len(unique_href_names)
    duplicate_ref_names = [name for name, count in Counter(href_names).items() if count > 1]

    archive_files: list[Path] = []
    archive_missing: list[str] = []
    for name in unique_href_names:
        source_path = normalized_source_dir / name
        if source_path.exists():
            archive_files.append(source_path)
        else:
            archive_missing.append(name)

    phones = sorted(
        {
            str(parse_filename_metadata(path.name).get("phone") or "").strip()
            for path in archive_files
            if str(parse_filename_metadata(path.name).get("phone") or "").strip()
        }
    )
    phone_set = set(phones)

    all_history_files: list[Path] = []
    for path in _iter_audio_files(normalized_source_dir):
        phone = str(parse_filename_metadata(path.name).get("phone") or "").strip()
        if phone and phone in phone_set:
            all_history_files.append(path)

    db_paths = [path for path in db_search_root.rglob("*.db") if path.is_file()]
    db_index, usable_dbs = _collect_db_index(db_paths)

    selected_files: list[Path] = []
    already_done = 0
    seen_but_not_done = 0
    selected_from_archive = 0
    archive_file_names = {path.name for path in archive_files}
    for path in all_history_files:
        state = db_index.get(path.name, {})
        transcription_done = bool(state.get("transcription_done", False))
        if transcription_done:
            already_done += 1
            continue
        if bool(state.get("seen_in_any_db", False)):
            seen_but_not_done += 1
        selected_files.append(path)
        if path.name in archive_file_names:
            selected_from_archive += 1

    batch_dir = out_root / "batch_full_cycle"
    phones_csv = out_root / "phones.csv"
    selected_calls_csv = out_root / "selected_calls.csv"
    manifest_path = out_root / "selection_manifest.json"
    out_root.mkdir(parents=True, exist_ok=True)
    _clean_output_dir(batch_dir)

    for path in selected_files:
        link_path = batch_dir / path.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(path)

    with phones_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phone"])
        for phone in phones:
            writer.writerow([phone])

    with selected_calls_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "source_filename",
                "source_file",
                "phone",
                "manager_name",
                "started_at",
                "is_archive_call",
                "seen_in_any_db",
                "transcription_done_elsewhere",
                "resolve_terminal_elsewhere",
                "analysis_done_elsewhere",
                "db_hits",
            ]
        )
        for path in selected_files:
            meta = parse_filename_metadata(path.name)
            state = db_index.get(path.name, {})
            started_at = meta.get("started_at")
            writer.writerow(
                [
                    path.name,
                    str(path),
                    str(meta.get("phone") or "").strip(),
                    str(meta.get("manager_name") or "").strip(),
                    started_at.isoformat(sep=" ") if started_at is not None else "",
                    "1" if path.name in archive_file_names else "0",
                    "1" if bool(state.get("seen_in_any_db", False)) else "0",
                    "1" if bool(state.get("transcription_done", False)) else "0",
                    "1" if bool(state.get("resolve_terminal", False)) else "0",
                    "1" if bool(state.get("analysis_done", False)) else "0",
                    int(state.get("db_hits", 0) or 0),
                ]
            )

    archive_dates = _extract_dates(archive_files)
    matched_dates = _extract_dates(all_history_files)
    selected_dates = _extract_dates(selected_files)

    manifest = {
        "archive_dir": str(archive_dir),
        "normalized_source_dir": str(normalized_source_dir),
        "db_search_root": str(db_search_root),
        "out_root": str(out_root),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "archive_audio_refs": len(href_names),
        "unique_archive_audio_refs": len(unique_href_names),
        "duplicate_ref_overflow": duplicate_ref_overflow,
        "duplicate_ref_samples": duplicate_ref_names[:20],
        "archive_files_found_in_normalized_dir": len(archive_files),
        "archive_missing_files": len(archive_missing),
        "archive_missing_samples": archive_missing[:20],
        "archive_date_min": min(archive_dates) if archive_dates else None,
        "archive_date_max": max(archive_dates) if archive_dates else None,
        "unique_phones": len(phones),
        "matched_history_files": len(all_history_files),
        "matched_history_date_min": min(matched_dates) if matched_dates else None,
        "matched_history_date_max": max(matched_dates) if matched_dates else None,
        "already_transcribed_elsewhere": already_done,
        "seen_in_db_but_not_transcribed_done": seen_but_not_done,
        "selected_untranscribed_files": len(selected_files),
        "selected_from_archive_calls": selected_from_archive,
        "selected_history_only_calls": max(len(selected_files) - selected_from_archive, 0),
        "selected_date_min": min(selected_dates) if selected_dates else None,
        "selected_date_max": max(selected_dates) if selected_dates else None,
        "db_files_scanned": len(db_paths),
        "usable_db_files_scanned": len(usable_dbs),
        "batch_dir": str(batch_dir),
        "phones_csv": str(phones_csv),
        "selected_calls_csv": str(selected_calls_csv),
        "selected_files": [path.name for path in selected_files],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
