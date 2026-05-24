#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIRM_ENV = "CONFIRM_RUNTIME_CLEANUP_BATCH3"
CONFIRM_TOKEN = "MOVE_RUNTIME_CLEANUP_BATCH3_TO_TRASH"
TRASH_ROOT = Path.home() / ".Trash"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
DEST_ROOT = TRASH_ROOT / f"MangoAnalyse_runtime_cleanup_batch3_{RUN_ID}"
CSV_MANIFEST = PROJECT_ROOT / "docs" / "RUNTIME_CLEANUP_BATCH3_MOVED_2026-05-23.csv"
JSON_SUMMARY = PROJECT_ROOT / "docs" / "RUNTIME_CLEANUP_BATCH3_SUMMARY_2026-05-23.json"
MD_MANIFEST = PROJECT_ROOT / "docs" / "RUNTIME_CLEANUP_BATCH3_MANIFEST_2026-05-23.md"

CANDIDATES: list[tuple[str, str]] = [
    ("stable_runtime/canonical_master_20260521_after_mango_update_v1", "superseded by canonical_master_20260523_audio_working_store_v1; DB compare matched except audio source path"),
    ("stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance", "superseded by sales_master_export_20260523_audio_working_store_v1; key CSVs have identical content hashes"),
    ("product_data/canonical_audio_store_20260516_v1", "old audio-store metadata/projection; actual audio already moved to the 20260523 working store and old audio dir was moved to Trash"),
    ("stable_runtime/ra_pending_mango_api_20260517_v1", "old pending R+A layer; current runtime has missing ASR/R+A = 0"),
    ("stable_runtime/history_remaining_excl_done_20260407", "historical remaining-history layer; no current runtime dependency"),
    ("stable_runtime/start_remaining_history_resolve4.sh", "legacy launcher that points to removed history_remaining_excl_done_20260407"),
    ("АКТУАЛЬНО_AI_review.xlsx", "stale root Excel; empty technical sheet, replaced by current runtime artifacts"),
    ("АКТУАЛЬНО_AMO_ready.xlsx", "stale root AMO Excel; write_amo_ready_contacts now defaults to active CANONICAL_EXPORT CSV"),
    ("АКТУАЛЬНО_Tallanto_match_issues.xlsx", "stale root Tallanto issue export; current contact/review data is in active 20260523 export"),
    ("АКТУАЛЬНО_Звонки_общая_таблица.xlsx", "stale root calls Excel; replaced by active master_calls_ru.csv with 65,939 actionable rows"),
    ("АКТУАЛЬНО_История_еще_не_добита.xlsx", "stale root Excel; empty technical sheet"),
    ("АКТУАЛЬНО_Контакты_для_продаж.xlsx", "stale root contacts Excel; replaced by active master_contacts_ru.csv"),
    ("АКТУАЛЬНО_Полный_пакет_экспорта.xlsx", "stale root all-in-one export; replaced by active 20260523 CSV export folder"),
    ("АКТУАЛЬНО_РОП_очередь_сделок_30д_live.xlsx", "stale root ROP deal queue; current AMO/deal-aware layers use current runtime/audit packs"),
    ("stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v2_hybrid_reuse", "intermediate Stage12 KB export superseded by v11 frozen gate and current runtime"),
    ("stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v4_stage15_hardened", "intermediate Stage15 KB export superseded by later safety/frozen-gate layers"),
    ("stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v5_claude_safety_fix", "intermediate sanitizer fix export superseded by later frozen-gate layers"),
    ("stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v6_claude_safety_fix", "intermediate sanitizer fix export superseded by later frozen-gate layers"),
    ("stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v7_location_fix", "intermediate sanitizer fix export superseded by later frozen-gate layers"),
    ("stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v8_orphan_surname_fix", "intermediate sanitizer fix export superseded by v10/v11 frozen-gate layers; docs keep historical report"),
    ("stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v10_fixpoint", "intermediate fixpoint export superseded by v11 frozen-gate layer"),
    ("stable_runtime/amo_live_stage51_summary_repair_20260512_v1", "superseded dry-run repair pack; no approval/readback history, replaced by stage51_textarea_repair_v2"),
    ("stable_runtime/amo_live_stage55_20260511_v2_wrong_person_gate_check", "failed wrong-person gate check pack; no downstream refs"),
    ("stable_runtime/crm_writeback_quality_gate_20260510_v5_product_gate", "early product gate superseded by later/current CRM gates; no refs"),
    ("stable_runtime/amo_writeback_queue_20260516_after_mango_update_v1", "old AMO queue superseded by current 20260523 AMO queue"),
    ("stable_runtime/amo_writeback_queue_20260521_after_mango_update_v4_runtime_acceptance", "old AMO queue superseded by current 20260523 AMO queue"),
    ("stable_runtime/crm_writeback_quality_gate_20260513_human_history_v1", "old human-history CRM quality gate superseded by current 20260523 gate"),
    ("stable_runtime/crm_writeback_quality_gate_20260513_human_history_v2", "old human-history CRM quality gate superseded by current 20260523 gate"),
    ("stable_runtime/crm_writeback_quality_gate_20260513_human_history_v3", "old human-history CRM quality gate superseded by current 20260523 gate"),
    ("stable_runtime/crm_writeback_quality_gate_20260521_after_mango_update_v4_runtime_acceptance", "old CRM quality gate superseded by current 20260523 gate"),
]


def _read_current_runtime() -> dict:
    path = PROJECT_ROOT / "stable_runtime" / "CURRENT_RUNTIME.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _protected_paths() -> set[Path]:
    runtime = _read_current_runtime()
    protected: set[Path] = set()
    for value in (runtime.get("paths") or {}).values():
        if isinstance(value, str) and value:
            protected.add(Path(value).expanduser().resolve())
    protected.update(
        {
            PROJECT_ROOT / "stable_runtime" / "CURRENT_RUNTIME.json",
            PROJECT_ROOT / "stable_runtime" / "CANONICAL_EXPORT.txt",
            PROJECT_ROOT / "product_data" / "audio_working_store_20260523_v1",
            PROJECT_ROOT / "product_data" / "CURRENT_AUDIO_WORKING_STORE.txt",
            PROJECT_ROOT / "product_data" / "customer_timeline" / "canonical_readonly_20260521_v5",
            PROJECT_ROOT / "_external_handoffs" / "mail_archive_2026-05-12",
            PROJECT_ROOT / "telegram_exports (2)",
            PROJECT_ROOT / "TP UNPK DataExport_2026-05-21",
            PROJECT_ROOT / "stable_runtime" / "sales_master_export_20260523_audio_working_store_v1",
            PROJECT_ROOT / "stable_runtime" / "canonical_master_20260523_audio_working_store_v1",
            PROJECT_ROOT / "stable_runtime" / "crm_writeback_quality_gate_20260523_audio_working_store_v1",
            PROJECT_ROOT / "stable_runtime" / "amo_writeback_queue_20260523_audio_working_store_v1",
            PROJECT_ROOT / "stable_runtime" / "sales_insight_knowledge_base_after_quality_backfill_20260510_v11_frozen_gate",
            PROJECT_ROOT / "stable_runtime" / "rop_validation_pack_after_quality_backfill_20260510_v11_frozen_gate",
        }
    )
    return {path.resolve() for path in protected if path}


def _is_protected(path: Path, protected: set[Path]) -> bool:
    resolved = path.resolve()
    for item in protected:
        if resolved == item:
            return True
        try:
            resolved.relative_to(item)
            return True
        except ValueError:
            pass
    return False


def _path_size_and_count(path: Path) -> tuple[int, int]:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size, 1
    total = 0
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
                count += 1
            except OSError:
                pass
        for name in dirs:
            dp = Path(root) / name
            if dp.is_symlink():
                try:
                    total += dp.lstat().st_size
                    count += 1
                except OSError:
                    pass
    return total, count


def _unique_destination(relative: Path) -> Path:
    dest = DEST_ROOT / relative
    if not dest.exists():
        return dest
    suffix = 1
    while True:
        candidate = dest.with_name(f"{dest.name}__dup{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def main() -> int:
    if os.environ.get(CONFIRM_ENV) != CONFIRM_TOKEN:
        raise SystemExit(
            f"Refusing cleanup without {CONFIRM_ENV}={CONFIRM_TOKEN}. "
            "This one-off script moves files to macOS Trash."
        )
    protected = _protected_paths()
    rows: list[dict[str, object]] = []
    moved = []
    skipped = []
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for rel_text, reason in CANDIDATES:
        rel = Path(rel_text)
        source = (PROJECT_ROOT / rel).resolve()
        row: dict[str, object] = {"relative_path": rel_text, "reason": reason}
        if not source.exists() and not source.is_symlink():
            row.update({"action": "skip_missing", "bytes": 0, "file_count": 0, "trash_path": ""})
            skipped.append(rel_text)
            rows.append(row)
            continue
        if _is_protected(source, protected):
            raise RuntimeError(f"Refusing to move protected path: {rel_text}")
        size, count = _path_size_and_count(source)
        dest = _unique_destination(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        row.update({"action": "moved_to_trash", "bytes": size, "file_count": count, "trash_path": str(dest)})
        moved.append(rel_text)
        rows.append(row)

    CSV_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CSV_MANIFEST.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "action", "bytes", "file_count", "reason", "trash_path"])
        writer.writeheader()
        writer.writerows(rows)

    total_bytes = sum(int(row.get("bytes") or 0) for row in rows if row.get("action") == "moved_to_trash")
    summary = {
        "schema_version": "runtime_cleanup_batch3_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trash_root": str(DEST_ROOT),
        "moved_count": len(moved),
        "skipped_missing_count": len(skipped),
        "total_bytes_moved": total_bytes,
        "total_gib_moved": round(total_bytes / (1024 ** 3), 3),
        "csv_manifest": str(CSV_MANIFEST),
        "md_manifest": str(MD_MANIFEST),
        "moved": moved,
        "skipped_missing": skipped,
    }
    JSON_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Runtime cleanup batch 3 - 2026-05-23",
        "",
        "Способ: перенос в macOS Trash, не безвозвратное удаление.",
        "",
        f"Trash root: `{DEST_ROOT}`",
        f"Moved: `{len(moved)}` paths",
        f"Skipped missing: `{len(skipped)}` paths",
        f"Total moved: `{summary['total_gib_moved']}` GiB",
        "",
        "## Moved paths",
        "",
        "| Путь | Размер, байт | Файлов | Причина |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        if row.get("action") != "moved_to_trash":
            continue
        lines.append(
            f"| `{row['relative_path']}` | {row['bytes']} | {row['file_count']} | {str(row['reason']).replace('|', '/')} |"
        )
    if skipped:
        lines.extend(["", "## Skipped missing", ""])
        lines.extend(f"- `{item}`" for item in skipped)
    lines.extend(
        [
            "",
            "## Protected current layers",
            "",
            "Не трогались: текущий `CURRENT_RUNTIME.json`, активный export/canonical DB 20260523, текущий audio working store, mail archive, Telegram exports, customer timeline, текущий KB v11/ROP pack v11.",
        ]
    )
    MD_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
