from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".mp4",
}


def _known_filenames(db_path: Path) -> set[str]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    return {
        str(row[0])
        for row in cur.execute(
            "SELECT source_filename FROM call_records WHERE source_filename IS NOT NULL"
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare 500+500 newest untranscribed audio batches")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--known-db", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    known_db = Path(args.known_db)
    out_root = Path(args.out_root)
    mini_dir = out_root / "mini_500"
    full_dir = out_root / "full_500"
    mini_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    known = _known_filenames(known_db)
    candidates = [
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and path.name not in known
    ]
    candidates.sort(key=lambda item: item.name, reverse=True)

    needed = int(args.batch_size) * 2
    if len(candidates) < needed:
        raise SystemExit(
            f"Not enough fresh audio files in {source_dir}: need {needed}, found {len(candidates)}"
        )

    selected = candidates[:needed]
    mini = selected[0::2][: args.batch_size]
    full = selected[1::2][: args.batch_size]

    for directory in (mini_dir, full_dir):
        for item in directory.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()

    for batch, directory in ((mini, mini_dir), (full, full_dir)):
        for source in batch:
            target = directory / source.name
            target.symlink_to(source)

    manifest = {
        "source_dir": str(source_dir),
        "known_db": str(known_db),
        "batch_size": int(args.batch_size),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "mini_dir": str(mini_dir),
        "full_dir": str(full_dir),
        "mini_files": [path.name for path in mini],
        "full_files": [path.name for path in full],
    }
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
