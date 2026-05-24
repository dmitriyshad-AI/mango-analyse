#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINTER = PROJECT_ROOT / "stable_runtime" / "CANONICAL_EXPORT.txt"
DOCS = PROJECT_ROOT / "docs"
REPORT_CSV = DOCS / "ACTIVE_EXPORT_AUDIO_PATH_REPAIR_2026-05-23.csv"
REPORT_JSON = DOCS / "ACTIVE_EXPORT_AUDIO_PATH_REPAIR_2026-05-23.json"


def main() -> int:
    export_name = POINTER.read_text(encoding="utf-8").strip()
    export_root = PROJECT_ROOT / "stable_runtime" / export_name
    summary = json.loads((export_root / "summary.json").read_text(encoding="utf-8"))
    db_path = Path(summary["canonical_db"])
    calls_csv = export_root / "master_calls_ru.csv"
    audio_store = Path(summary["audio_working_store"]).resolve()

    con = sqlite3.connect(db_path)
    mapping = {
        str(row[0]): str(row[1])
        for row in con.execute("select canonical_call_id, source_file from canonical_calls where is_actionable = 1")
    }
    con.close()

    tmp = calls_csv.with_suffix(".csv.tmp")
    changed_rows: list[dict[str, str]] = []
    missing_mapping = 0
    missing_files = 0
    total_rows = 0
    with calls_csv.open("r", encoding="utf-8-sig", newline="") as src, tmp.open("w", encoding="utf-8-sig", newline="") as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise RuntimeError("master_calls_ru.csv has no header")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total_rows += 1
            call_id = str(row.get("ID звонка") or "").strip()
            new_path = mapping.get(call_id)
            if not new_path:
                missing_mapping += 1
                writer.writerow(row)
                continue
            old_path = str(row.get("Путь к записи") or "")
            new_resolved = Path(new_path).resolve()
            try:
                new_resolved.relative_to(audio_store)
            except ValueError as exc:
                raise RuntimeError(f"New source_file is outside audio working store for call {call_id}: {new_path}") from exc
            if not new_resolved.exists():
                missing_files += 1
            if old_path != str(new_resolved):
                changed_rows.append(
                    {
                        "ID звонка": call_id,
                        "Имя исходного файла": row.get("Имя исходного файла", ""),
                        "old_path": old_path,
                        "new_path": str(new_resolved),
                        "new_path_exists": str(new_resolved.exists()),
                    }
                )
                row["Путь к записи"] = str(new_resolved)
            writer.writerow(row)
    os.replace(tmp, calls_csv)

    DOCS.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ID звонка", "Имя исходного файла", "old_path", "new_path", "new_path_exists"])
        writer.writeheader()
        writer.writerows(changed_rows)
    report = {
        "schema_version": "active_export_audio_path_repair_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_root": str(export_root),
        "canonical_db": str(db_path),
        "audio_working_store": str(audio_store),
        "master_calls_csv": str(calls_csv),
        "total_rows": total_rows,
        "changed_rows": len(changed_rows),
        "missing_mapping": missing_mapping,
        "missing_new_files": missing_files,
        "report_csv": str(REPORT_CSV),
        "passed": missing_mapping == 0 and missing_files == 0,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
