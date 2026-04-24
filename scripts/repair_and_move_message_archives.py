from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from mango_mvp.utils.filename_repair import repair_filename_display


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag != "a":
            return
        attrs_map = dict(attrs)
        href = attrs_map.get("href")
        if not href:
            return
        name = Path(str(href)).name
        if Path(name).suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            self.hrefs.append(name)


def _timestamp_key(name: str) -> tuple[str, str] | None:
    parts = Path(name).stem.split("__")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _parts4(name: str) -> list[str]:
    return Path(name).stem.split("__")


def _choose_manifest_name(actual_name: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]

    repaired = repair_filename_display(actual_name)
    if repaired in unique:
        return repaired

    actual_parts = _parts4(actual_name)
    if len(actual_parts) >= 4:
        actual_tail = set(actual_parts[2:])
        scored: list[tuple[int, str]] = []
        for candidate in unique:
            candidate_parts = _parts4(candidate)
            overlap = len(actual_tail & set(candidate_parts[2:])) if len(candidate_parts) >= 4 else 0
            scored.append((overlap, candidate))
        scored.sort(reverse=True)
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return scored[0][1]

    return unique[0]


def _iter_audio_files(folder: Path) -> Iterable[Path]:
    for item in sorted(folder.iterdir()):
        if not item.is_file():
            continue
        if item.name == "index.html":
            continue
        if item.suffix.lower() not in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            continue
        yield item


def repair_and_move_archives(*, source_dirs: list[Path], target_dir: Path, dry_run: bool) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped_existing = 0
    unresolved = 0
    source_index_files = 0
    moved_index_files = 0
    unresolved_samples: list[dict[str, str]] = []

    for folder in source_dirs:
        index_path = folder / "index.html"
        if not index_path.exists():
            continue
        source_index_files += 1

        parser = _IndexParser()
        parser.feed(index_path.read_text("utf-8"))
        manifest_by_ts: dict[tuple[str, str] | None, list[str]] = defaultdict(list)
        for name in parser.hrefs:
            manifest_by_ts[_timestamp_key(name)].append(name)

        for source_file in _iter_audio_files(folder):
            candidates = manifest_by_ts.get(_timestamp_key(source_file.name), [])
            manifest_name = _choose_manifest_name(source_file.name, candidates)
            if manifest_name is None:
                unresolved += 1
                if len(unresolved_samples) < 20:
                    unresolved_samples.append(
                        {"folder": folder.name, "source": source_file.name}
                    )
                continue

            target_file = target_dir / manifest_name
            if target_file.exists():
                if source_file.stat().st_size == target_file.stat().st_size:
                    skipped_existing += 1
                    if not dry_run:
                        source_file.unlink()
                    continue
                stem = target_file.stem
                suffix = target_file.suffix
                duplicate_idx = 2
                while True:
                    candidate = target_dir / f"{stem}__dup{duplicate_idx}{suffix}"
                    if not candidate.exists():
                        target_file = candidate
                        break
                    duplicate_idx += 1

            if not dry_run:
                shutil.move(str(source_file), str(target_file))
            moved += 1

        moved_index_target = target_dir / f"{folder.name}_index.html"
        if not moved_index_target.exists():
            moved_index_files += 1
            if not dry_run:
                shutil.copy2(index_path, moved_index_target)

    return {
        "target_dir": str(target_dir),
        "moved": moved,
        "skipped_existing": skipped_existing,
        "unresolved": unresolved,
        "source_index_files": source_index_files,
        "copied_index_files": moved_index_files,
        "unresolved_samples": unresolved_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair Mango message archive filenames using index.html manifests and move audio files into one target folder."
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Folder that will receive repaired audio files.",
    )
    parser.add_argument(
        "source_dirs",
        nargs="+",
        help="One or more message archive folders (for example messages(1) messages(2)).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the move plan but do not change the filesystem.",
    )
    args = parser.parse_args()

    result = repair_and_move_archives(
        source_dirs=[Path(item).expanduser().resolve() for item in args.source_dirs],
        target_dir=Path(args.target_dir).expanduser().resolve(),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
