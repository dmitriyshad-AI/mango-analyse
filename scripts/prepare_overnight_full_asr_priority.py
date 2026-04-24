from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


def _db_has_call_records(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(
            "select name from sqlite_master where type='table' and name='call_records'"
        ).fetchone()
    except sqlite3.Error:
        return False
    return bool(row)


def _collect_known_filenames(db_search_root: Path) -> tuple[set[str], list[str]]:
    known: set[str] = set()
    usable_dbs: list[str] = []
    for db_path in sorted(db_search_root.rglob("*.db")):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            if not _db_has_call_records(conn):
                continue
            usable_dbs.append(str(db_path))
            rows = conn.execute(
                "select source_filename from call_records where source_filename is not null and trim(source_filename) != ''"
            )
            for (source_filename,) in rows:
                filename = str(source_filename or "").strip()
                if filename:
                    known.add(filename)
        finally:
            conn.close()
    return known, usable_dbs


def _iter_candidates(source_dir: Path, known: set[str]) -> list[tuple[datetime, Path, str, str]]:
    items: list[tuple[datetime, Path, str, str]] = []
    for path in source_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name in known:
            continue
        meta = parse_filename_metadata(path.name)
        started_at = meta.get("started_at")
        phone = str(meta.get("phone") or "").strip()
        manager_name = str(meta.get("manager_name") or "").strip()
        if not phone or started_at is None:
            continue
        items.append((started_at, path, phone, manager_name))
    items.sort(key=lambda item: item[0], reverse=True)
    return items


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare exactly N newest unique raw audio calls that were never seen in any local DB."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--db-search-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    db_search_root = Path(args.db_search_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    batch_size = max(1, int(args.batch_size))

    known, usable_dbs = _collect_known_filenames(db_search_root)
    candidates = _iter_candidates(source_dir, known)
    if len(candidates) < batch_size:
        raise SystemExit(
            f"Not enough unseen raw calls in {source_dir}: need {batch_size}, found {len(candidates)}"
        )

    selected = candidates[:batch_size]
    batch_dir = out_root / "batch_full_cycle"
    _clean_dir(batch_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for _started_at, source_path, _phone, _manager_name in selected:
        link_path = batch_dir / source_path.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(source_path)

    phones = sorted({phone for _started_at, _path, phone, _manager_name in selected})
    phones_csv = out_root / "phones.csv"
    with phones_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phone"])
        for phone in phones:
            writer.writerow([phone])

    selected_calls_csv = out_root / "selected_calls.csv"
    with selected_calls_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_filename", "source_file", "phone", "manager_name", "started_at"])
        for started_at, source_path, phone, manager_name in selected:
            writer.writerow(
                [
                    source_path.name,
                    str(source_path),
                    phone,
                    manager_name,
                    started_at.isoformat(sep=" "),
                ]
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_strategy": (
            f"exactly {batch_size} unique unprocessed raw audio calls; canonical paths only; "
            "require phone in filename; sorted by newest started_at desc"
        ),
        "source_dir": str(source_dir),
        "db_search_root": str(db_search_root),
        "output_root": str(out_root),
        "batch_dir": str(batch_dir),
        "total_selected": len(selected),
        "unique_phones": len(phones),
        "date_min": selected[-1][0].isoformat(sep=" "),
        "date_max": selected[0][0].isoformat(sep=" "),
        "usable_db_files_scanned": len(usable_dbs),
        "all_selected_not_in_any_db_by_filename": True,
        "selected_calls_csv": str(selected_calls_csv),
        "phones_csv": str(phones_csv),
        "source_roots_used": [source_dir.name],
    }
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
