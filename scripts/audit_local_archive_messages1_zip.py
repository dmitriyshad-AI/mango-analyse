#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "_local_archive_20260424" / "source_archives" / "messages(1).zip"
STORE_ROOT = PROJECT_ROOT / "product_data" / "audio_working_store_20260523_v1"
UNIQUE_MANIFEST = STORE_ROOT / "manifests" / "unique_audio_manifest.csv"
ORPHAN_MANIFEST = STORE_ROOT / "manifests" / "orphan_audio_manifest.csv"
OUT_CSV = PROJECT_ROOT / "docs" / "LOCAL_ARCHIVE_MESSAGES1_ZIP_AUDIT_2026-05-23.csv"
OUT_JSON = PROJECT_ROOT / "docs" / "LOCAL_ARCHIVE_MESSAGES1_ZIP_AUDIT_2026-05-23.json"


def _load_store_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in [UNIQUE_MANIFEST, ORPHAN_MANIFEST]:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                value = (row.get("sha256") or "").strip().lower()
                if value:
                    hashes.add(value)
    return hashes


def _hash_zip_member(zf: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with zf.open(name) as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    store_hashes = _load_store_hashes()
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            sha = _hash_zip_member(zf, info.filename)
            covered = sha in store_hashes
            rows.append(
                {
                    "zip_member": info.filename,
                    "ext": ext,
                    "size_bytes": info.file_size,
                    "sha256": sha,
                    "covered_by_audio_working_store": "yes" if covered else "no",
                }
            )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["zip_member", "ext", "size_bytes", "sha256", "covered_by_audio_working_store"],
        )
        writer.writeheader()
        writer.writerows(rows)
    audio_rows = [row for row in rows if row["ext"] in {".mp3", ".wav", ".m4a", ".ogg"}]
    uncovered_audio = [row for row in audio_rows if row["covered_by_audio_working_store"] == "no"]
    ext_counts: dict[str, int] = {}
    for row in rows:
        ext_counts[str(row["ext"])] = ext_counts.get(str(row["ext"]), 0) + 1
    summary = {
        "schema_version": "local_archive_messages1_zip_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zip_path": str(ZIP_PATH),
        "zip_exists": ZIP_PATH.exists(),
        "zip_size_bytes": ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0,
        "store_root": str(STORE_ROOT),
        "store_hashes": len(store_hashes),
        "zip_files": len(rows),
        "zip_audio_files": len(audio_rows),
        "zip_non_audio_files": len(rows) - len(audio_rows),
        "extension_counts": ext_counts,
        "covered_audio_files": len(audio_rows) - len(uncovered_audio),
        "uncovered_audio_files": len(uncovered_audio),
        "uncovered_audio_bytes": sum(int(row["size_bytes"]) for row in uncovered_audio),
        "passed_all_audio_covered": len(uncovered_audio) == 0,
        "report_csv": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed_all_audio_covered"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
