from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all calls from source-dir for phones found in an existing DB and build "
            "a symlink batch with full phone history."
        )
    )
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_db = Path(args.source_db).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    batch_dir = out_root / "batch_phone_history"
    phones_csv = out_root / "phones.csv"
    manifest_path = out_root / "selection_manifest.json"

    conn = sqlite3.connect(source_db)
    cur = conn.cursor()
    phones = sorted(
        {
            str(row[0] or "").strip()
            for row in cur.execute(
                "select phone from call_records where phone is not null and trim(phone) != ''"
            )
            if str(row[0] or "").strip()
        }
    )
    cur.execute("select count(*) from call_records")
    source_calls_total = int(cur.fetchone()[0] or 0)
    conn.close()

    selected_files: list[Path] = []
    selected_dates: list[str] = []
    phone_set = set(phones)
    for path in sorted(source_dir.iterdir(), key=lambda item: item.name, reverse=True):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        meta = parse_filename_metadata(path.name)
        phone = str(meta.get("phone") or "").strip()
        if not phone or phone not in phone_set:
            continue
        selected_files.append(path)
        started_at = meta.get("started_at")
        if started_at is not None:
            selected_dates.append(started_at.date().isoformat())

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

    manifest = {
        "source_db": str(source_db),
        "source_dir": str(source_dir),
        "out_root": str(out_root),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_db_calls_total": source_calls_total,
        "unique_phones": len(phones),
        "matched_files_in_whole_folder": len(selected_files),
        "matched_date_min": min(selected_dates) if selected_dates else None,
        "matched_date_max": max(selected_dates) if selected_dates else None,
        "batch_dir": str(batch_dir),
        "phones_csv": str(phones_csv),
        "selected_files": [path.name for path in selected_files],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
