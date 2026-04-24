from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".mp4"}


def _collect_known_filenames(db_paths: list[Path]) -> set[str]:
    known: set[str] = set()
    for db_path in db_paths:
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            rows = cur.execute("select source_filename from call_records").fetchall()
            known.update(str(row[0]) for row in rows if row and row[0])
            conn.close()
        except sqlite3.Error:
            continue
    return known


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--known-db", action="append", default=[])
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    batch_size = max(1, int(args.batch_size))
    known_dbs = [Path(item).expanduser().resolve() for item in args.known_db]

    out_root.mkdir(parents=True, exist_ok=True)
    batch_dir = out_root / f"batch_{batch_size}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    known = _collect_known_filenames(known_dbs)
    candidates = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS and path.name not in known
        ),
        key=lambda item: item.name,
        reverse=True,
    )
    selected = candidates[:batch_size]

    for existing in batch_dir.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()

    for path in selected:
        link_path = batch_dir / path.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(path)

    manifest = {
        "source_dir": str(source_dir),
        "known_dbs": [str(path) for path in known_dbs],
        "batch_size": batch_size,
        "known_match_count": len(known),
        "selected_count": len(selected),
        "batch_dir": str(batch_dir),
        "selected_files": [path.name for path in selected],
    }
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
