from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter


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
        if Path(name).suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            self.hrefs.append(name)


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a symlink batch for one messages(N) archive using its index.html and the normalized target folder."
    )
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--normalized-source-dir", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir).expanduser().resolve()
    normalized_source_dir = Path(args.normalized_source_dir).expanduser().resolve()
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

    batch_dir = out_root / "batch"
    _clean_output_dir(batch_dir)

    linked = 0
    missing: list[str] = []
    selected_dates: list[str] = []
    for name in unique_href_names:
        source_path = normalized_source_dir / name
        if not source_path.exists():
            missing.append(name)
            continue
        link_path = batch_dir / name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(source_path)
        linked += 1
        parts = source_path.stem.split("__")
        if parts:
            selected_dates.append(parts[0])

    manifest = {
        "archive_dir": str(archive_dir),
        "normalized_source_dir": str(normalized_source_dir),
        "out_root": str(out_root),
        "archive_audio_refs": len(href_names),
        "unique_archive_audio_refs": len(unique_href_names),
        "duplicate_ref_overflow": duplicate_ref_overflow,
        "duplicate_ref_samples": duplicate_ref_names[:20],
        "linked_files": linked,
        "missing_files": len(missing),
        "missing_samples": missing[:20],
        "batch_dir": str(batch_dir),
        "date_min": min(selected_dates) if selected_dates else None,
        "date_max": max(selected_dates) if selected_dates else None,
        "selected_files": unique_href_names,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
