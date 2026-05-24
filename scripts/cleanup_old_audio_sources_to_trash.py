#!/usr/bin/env python3
"""Move legacy audio copies to macOS Trash after SHA coverage verification.

The script is intentionally conservative: it only moves audio files/directories
whose content hash already exists in the current audio working store.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "product_data" / "audio_working_store_20260523_v1"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

FILE_GLOB_TARGETS = [
    "2026-03-09--26/*.mp3",
    "2026-03-05-21-06-49-ч1/*.mp3",
    "2026-03-05-21-06-49-ч2/*.mp3",
    "product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/audio/*.mp3",
]

DIR_TARGETS = [
    "product_data/canonical_audio_store_20260516_v1/audio",
    "_local_archive_mango_api_downloads_20260507/recordings",
    "_local_archive_mango_api_downloads_20260507/quarantine_import/audio",
    "_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_pack_stage13/audio",
    "_local_archive_mango_api_downloads_20260507/product_appliance/live_mango_capture_20260511/asr_ui_batch_20260511_v1/audio",
    "_local_archive_mango_api_downloads_20260507/product_appliance/live_mango_capture_20260511/recordings",
    "_local_archive_mango_api_downloads_20260507/product_appliance/live_mango_capture_20260516_incremental_v1/recordings",
    "_local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recordings",
    "_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage11/audio",
    "_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/audio",
    "product_data/mango_incremental_4_asr_ra_20260516_v1/audio",
    "product_data/mango_new_21_asr_ra_20260516_v1/audio",
]


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root))
    except ValueError:
        return str(path.resolve(strict=False))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def store_hashes(store: Path) -> set[str]:
    audio_root = store / "audio"
    hashes = {path.stem.lower() for path in audio_root.glob("*/*") if path.is_file()}
    if not hashes:
        raise RuntimeError(f"No audio hashes found under {audio_root}")
    return hashes


def iter_audio(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
        yield path
    elif path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() in AUDIO_EXTS:
                yield child


def collect_candidates() -> tuple[list[Path], list[Path]]:
    file_targets: list[Path] = []
    for pattern in FILE_GLOB_TARGETS:
        file_targets.extend(sorted(ROOT.glob(pattern)))
    dir_targets = [ROOT / value for value in DIR_TARGETS if (ROOT / value).exists()]
    return file_targets, dir_targets


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(1, 10_000):
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find unique destination for {path}")


def verify_covered(paths: Iterable[Path], known_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source in paths:
        source = source.resolve(strict=False)
        if source in seen:
            continue
        seen.add(source)
        if not source.exists():
            continue
        digest = sha256_file(source)
        row = {
            "source_path": rel(source),
            "sha256": digest,
            "size_bytes": source.stat().st_size,
            "covered_by_audio_working_store": digest in known_hashes,
        }
        if digest in known_hashes:
            rows.append(row)
        else:
            missing.append(row)
    return rows, missing


def move_file(source: Path, trash_root: Path) -> Path:
    destination = unique_destination(trash_root / "files" / rel(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination


def move_dir(source: Path, trash_root: Path) -> Path:
    destination = unique_destination(trash_root / "dirs" / rel(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()}) if rows else ["source_path"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store).resolve(strict=False)
    known_hashes = store_hashes(store)
    file_targets, dir_targets = collect_candidates()

    all_audio_paths = list(file_targets)
    for directory in dir_targets:
        all_audio_paths.extend(iter_audio(directory))
    covered_rows, missing_rows = verify_covered(all_audio_paths, known_hashes)
    if missing_rows:
        report_dir = ROOT / "docs"
        missing_path = report_dir / "AUDIO_WORKING_STORE_CLEANUP_BLOCKED_MISSING_2026-05-23.csv"
        write_csv(missing_path, missing_rows)
        raise RuntimeError(f"Refusing to move legacy audio: {len(missing_rows)} files are not covered. See {missing_path}")

    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trash_root = Path.home() / ".Trash" / f"MangoAnalyse_audio_cleanup_{generated_at}"
    moved_rows: list[dict[str, Any]] = []

    if args.dry_run:
        for row in covered_rows:
            moved_rows.append({**row, "operation": "dry_run", "trash_path": ""})
    else:
        # Move whole directory targets first and skip file-glob entries that are inside them.
        moved_dirs: list[Path] = []
        for directory in dir_targets:
            audio_files = list(iter_audio(directory))
            if not audio_files:
                continue
            destination = move_dir(directory, trash_root)
            moved_dirs.append(directory.resolve(strict=False))
            moved_rows.append(
                {
                    "source_path": rel(directory),
                    "trash_path": str(destination),
                    "operation": "move_dir_to_trash",
                    "audio_files": len(audio_files),
                    "size_bytes": sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()),
                    "covered_by_audio_working_store": True,
                }
            )
        for source in file_targets:
            resolved = source.resolve(strict=False)
            if not resolved.exists():
                continue
            if any(str(resolved).startswith(str(moved_dir) + "/") for moved_dir in moved_dirs):
                continue
            destination = move_file(resolved, trash_root)
            moved_rows.append(
                {
                    "source_path": rel(source),
                    "trash_path": str(destination),
                    "operation": "move_file_to_trash",
                    "sha256": sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                    "covered_by_audio_working_store": True,
                }
            )

    report_dir = ROOT / "docs"
    moved_csv = report_dir / "AUDIO_WORKING_STORE_OLD_AUDIO_MOVED_2026-05-23.csv"
    write_csv(moved_csv, moved_rows)
    summary = {
        "schema_version": "audio_working_store_old_audio_cleanup_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audio_working_store": rel(store),
        "dry_run": bool(args.dry_run),
        "trash_root": str(trash_root),
        "candidate_audio_files": len(covered_rows),
        "missing_uncovered_audio_files": len(missing_rows),
        "moved_entries": len(moved_rows),
        "moved_csv": rel(moved_csv),
        "file_glob_targets": FILE_GLOB_TARGETS,
        "dir_targets": DIR_TARGETS,
        "safety": {
            "hash_coverage_required": True,
            "missing_audio_aborts": True,
            "stable_runtime_modified": False,
            "asr_run": False,
            "ra_run": False,
            "crm_write": False,
            "tallanto_write": False,
        },
    }
    summary_path = report_dir / "AUDIO_WORKING_STORE_OLD_AUDIO_CLEANUP_SUMMARY_2026-05-23.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move covered legacy audio sources to macOS Trash.")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
