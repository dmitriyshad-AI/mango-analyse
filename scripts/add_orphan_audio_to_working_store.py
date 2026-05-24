#!/usr/bin/env python3
"""Add non-canonical project audio files to the audio working store.

This keeps the canonical call mapping unchanged and records extra audio under
manifests/orphan_audio_manifest.csv. It is intended for old local folders that
may contain audio not present in the current canonical DB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "product_data" / "audio_working_store_20260523_v1"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root))
    except ValueError:
        return str(path.resolve(strict=False))


def read_known_hashes(store: Path) -> set[str]:
    known: set[str] = set()
    for csv_path in [store / "manifests" / "unique_audio_manifest.csv", store / "manifests" / "orphan_audio_manifest.csv"]:
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                sha = str(row.get("sha256") or "").strip().lower()
                if sha:
                    known.add(sha)
    return known


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def add_orphans(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve(strict=False)
    store = Path(args.store).resolve(strict=False)
    audio_root = store / "audio"
    orphan_links = store / "orphan_by_filename"
    manifest_path = store / "manifests" / "orphan_audio_manifest.csv"
    known = read_known_hashes(store)
    rows: list[dict[str, Any]] = []
    seen_this_run: set[str] = set()
    scanned = 0
    duplicates = 0
    added = 0
    link_errors = 0
    for raw_root in args.audio_roots:
        root = Path(raw_root).expanduser().resolve(strict=False)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
                continue
            scanned += 1
            sha = sha256_file(path)
            if sha in known or sha in seen_this_run:
                duplicates += 1
                continue
            seen_this_run.add(sha)
            ext = path.suffix.lower()
            target_rel = Path("audio") / sha[:2] / f"{sha}{ext}"
            target = store / target_rel
            action = link_or_copy(path, target)
            link_name = f"{sha[:12]}__{path.name}"
            link_path = orphan_links / link_name
            try:
                link_path.parent.mkdir(parents=True, exist_ok=True)
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(Path("..") / target_rel)
            except OSError:
                link_errors += 1
            added += 1
            rows.append(
                {
                    "sha256": sha,
                    "ext": ext,
                    "size_bytes": path.stat().st_size,
                    "canonical_audio_path": str(target_rel),
                    "orphan_by_filename_path": str(Path("orphan_by_filename") / link_name),
                    "original_source_file": rel(path, project_root),
                    "original_source_filename": path.name,
                    "source_root": rel(root, project_root),
                    "link_action": action,
                }
            )
    fields = [
        "sha256",
        "ext",
        "size_bytes",
        "canonical_audio_path",
        "orphan_by_filename_path",
        "original_source_file",
        "original_source_filename",
        "source_root",
        "link_action",
    ]
    if rows:
        append_csv(manifest_path, rows, fields)
    summary_path = store / "manifests" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["orphan_audio_manifest_csv"] = rel(manifest_path, project_root)
    summary["orphan_by_filename_root"] = rel(orphan_links, project_root)
    summary["orphan_audio_last_added_at"] = datetime.now(timezone.utc).isoformat()
    summary["orphan_audio_last_scan"] = {
        "scanned_audio_files": scanned,
        "duplicates_already_in_store": duplicates,
        "new_orphan_audio_added": added,
        "link_errors": link_errors,
        "source_roots": [rel(Path(p), project_root) for p in args.audio_roots],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary["orphan_audio_last_scan"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add orphan project audio files to the audio working store.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--audio-root", dest="audio_roots", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = add_orphans(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("link_errors", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
