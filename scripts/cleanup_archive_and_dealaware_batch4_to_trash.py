#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIRM_ENV = "CONFIRM_CLEANUP_BATCH4_ARCHIVE_DEALAWARE"
CONFIRM_TOKEN = "MOVE_ARCHIVE_DEALAWARE_BATCH4_TO_TRASH"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
DEST_ROOT = Path.home() / ".Trash" / f"MangoAnalyse_archive_dealaware_cleanup_batch4_{RUN_ID}"
CSV_MANIFEST = PROJECT_ROOT / "docs" / "ARCHIVE_DEALAWARE_CLEANUP_BATCH4_MOVED_2026-05-23.csv"
JSON_SUMMARY = PROJECT_ROOT / "docs" / "ARCHIVE_DEALAWARE_CLEANUP_BATCH4_SUMMARY_2026-05-23.json"
MD_MANIFEST = PROJECT_ROOT / "docs" / "ARCHIVE_DEALAWARE_CLEANUP_BATCH4_MANIFEST_2026-05-23.md"

CANDIDATES: list[tuple[str, str]] = [
    ("_local_archive_20260424/legacy_outputs", "derived legacy outputs/test transcripts; unique source zip is kept"),
    ("_local_archive_20260424/old_db_backups", "old DB backups not used by current runtime; unique source zip is kept"),
    ("_local_archive_20260424/old_test_dbs", "old test DBs not used by current runtime; unique source zip is kept"),
    ("_local_archive_20260424/processed_message_exports", "derived html exports; source zip with unique audio is kept"),
    ("stable_runtime/deal_aware_stage2_attribution_20260514_selector_fix_phase1", "heavy intermediate selector-fix Phase1 artifact; test now uses small frozen fixture"),
    ("stable_runtime/deal_aware_stage2_attribution_20260514_selector_fix_phase2", "heavy intermediate selector-fix Phase2 artifact; confidence fixture preserved"),
    ("stable_runtime/deal_aware_stage3_deal_state_20260514_selector_fix_phase2", "heavy intermediate selector-fix Stage3 artifact; no current runtime dependency"),
    ("stable_runtime/deal_aware_stage4_preview_20260514_selector_fix_phase2", "intermediate selector-fix Stage4 preview artifact; superseded by later review/writeback layers"),
    ("stable_runtime/deal_aware_stage5_quality_gate_20260514_selector_fix_phase2", "intermediate selector-fix Stage5 quality artifact; superseded by later/current gates"),
    ("stable_runtime/deal_aware_stage6_writeback_preflight_20260514_selector_fix_phase2", "intermediate selector-fix Stage6 preflight artifact; not live evidence/current runtime"),
]

PROTECTED: list[str] = [
    "_local_archive_20260424/source_archives/messages(1).zip",
    "product_data/audio_working_store_20260523_v1",
    "stable_runtime/canonical_master_20260523_audio_working_store_v1",
    "stable_runtime/sales_master_export_20260523_audio_working_store_v1",
    "stable_runtime/amo_writeback_queue_20260523_audio_working_store_v1",
    "stable_runtime/crm_writeback_quality_gate_20260523_audio_working_store_v1",
    "stable_runtime/deal_aware_stage100_rop_final_20260514_v1",
    "stable_runtime/deal_aware_stage709_all_batches_20260514_v1",
    "stable_runtime/deal_aware_stage709_review_20260514_selector_fix_phase2",
]


def _resolve(rel: str) -> Path:
    return (PROJECT_ROOT / rel).resolve()


def _is_protected(path: Path) -> bool:
    resolved = path.resolve()
    for rel in PROTECTED:
        protected = _resolve(rel)
        if resolved == protected:
            return True
        try:
            protected.relative_to(resolved)
            return True
        except ValueError:
            pass
        try:
            resolved.relative_to(protected)
            return True
        except ValueError:
            pass
    return False


def _path_size_and_count(path: Path) -> tuple[int, int]:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size, 1
    total = 0
    count = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
                count += 1
            except OSError:
                pass
    return total, count


def main() -> int:
    if os.environ.get(CONFIRM_ENV) != CONFIRM_TOKEN:
        raise SystemExit(f"Refusing cleanup without {CONFIRM_ENV}={CONFIRM_TOKEN}")
    rows: list[dict[str, object]] = []
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    for rel, reason in CANDIDATES:
        src = _resolve(rel)
        row: dict[str, object] = {"relative_path": rel, "reason": reason}
        if not src.exists() and not src.is_symlink():
            row.update({"action": "skip_missing", "bytes": 0, "file_count": 0, "trash_path": ""})
            rows.append(row)
            continue
        if _is_protected(src):
            raise RuntimeError(f"Refusing to move protected path or parent: {rel}")
        size, count = _path_size_and_count(src)
        dest = DEST_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        row.update({"action": "moved_to_trash", "bytes": size, "file_count": count, "trash_path": str(dest)})
        rows.append(row)
    CSV_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CSV_MANIFEST.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "action", "bytes", "file_count", "reason", "trash_path"])
        writer.writeheader()
        writer.writerows(rows)
    moved = [row for row in rows if row["action"] == "moved_to_trash"]
    total_bytes = sum(int(row["bytes"]) for row in moved)
    summary = {
        "schema_version": "archive_dealaware_cleanup_batch4_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trash_root": str(DEST_ROOT),
        "moved_count": len(moved),
        "total_bytes_moved": total_bytes,
        "total_gib_moved": round(total_bytes / (1024 ** 3), 3),
        "protected_kept": PROTECTED,
        "csv_manifest": str(CSV_MANIFEST),
        "md_manifest": str(MD_MANIFEST),
    }
    JSON_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Archive + deal-aware cleanup batch 4 - 2026-05-23",
        "",
        "Способ: перенос в macOS Trash, не безвозвратное удаление.",
        "",
        f"Trash root: `{DEST_ROOT}`",
        f"Moved: `{len(moved)}` paths",
        f"Total moved: `{summary['total_gib_moved']}` GiB",
        "",
        "## Важное решение по `_local_archive_20260424`",
        "",
        "`messages(1).zip` не удалён: аудит SHA-256 показал `231` уникальную mp3 относительно текущего `audio_working_store`.",
        "Удалены только производные legacy outputs / old DB / html exports внутри `_local_archive_20260424`.",
        "",
        "## Moved paths",
        "",
        "| Путь | Размер, байт | Файлов | Причина |",
        "|---|---:|---:|---|",
    ]
    for row in moved:
        lines.append(f"| `{row['relative_path']}` | {row['bytes']} | {row['file_count']} | {str(row['reason']).replace('|', '/')} |")
    lines.extend(["", "## Protected / kept", ""])
    lines.extend(f"- `{item}`" for item in PROTECTED)
    MD_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
