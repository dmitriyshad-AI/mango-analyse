from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROLLBACK_VERSION = "hard_gate_rollback_manifest_v1"
DEFAULT_COLUMNS = [
    "id",
    "source_filename",
    "source_file",
    "phone",
    "manager_name",
    "started_at",
    "transcription_status",
    "resolve_status",
    "analysis_status",
    "sync_status",
    "resolve_json",
    "resolve_quality_score",
    "analysis_json",
    "analyze_attempts",
    "dead_letter_stage",
    "last_error",
    "next_retry_at",
    "updated_at",
]


@dataclass(frozen=True)
class HardGateBackupManifestConfig:
    apply_plan_csv: Path
    out_root: Path
    project_root: Path = Path(".")
    queue_filter: str | None = None
    hash_db_files: bool = True


def build_hard_gate_backup_manifest(config: HardGateBackupManifestConfig) -> dict[str, Any]:
    project_root = config.project_root.expanduser().resolve()
    apply_plan_path = _resolve_path(config.apply_plan_csv, project_root)
    out_root = _resolve_path(config.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if not apply_plan_path.exists():
        raise FileNotFoundError(f"apply_plan_csv not found: {apply_plan_path}")

    input_rows = _read_csv(apply_plan_path)
    selected_rows = [
        row
        for row in input_rows
        if not config.queue_filter or _clean(row.get("queue")) == config.queue_filter
    ]
    by_db: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        by_db[_clean(row.get("db"))].append(row)

    db_manifest_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    queue_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    schema_warnings: list[str] = []

    for row in selected_rows:
        queue_counts[_clean(row.get("queue")) or "unknown"] += 1
        risk_counts[_clean(row.get("risk_level")) or "unknown"] += 1

    for db_value, rows in sorted(by_db.items()):
        db_path = _resolve_path(Path(db_value), project_root) if db_value else project_root / "__missing_db__"
        db_info = _db_file_info(db_path, project_root=project_root, hash_file=config.hash_db_files)
        found_count = 0
        missing_count = 0
        if not db_value or not db_path.exists():
            for plan_row in rows:
                missing_rows.append(_missing_row(plan_row, reason="db_missing"))
                missing_count += 1
        else:
            try:
                con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
                con.row_factory = sqlite3.Row
                try:
                    columns = _existing_columns(con, "call_records")
                    if not columns:
                        schema_warnings.append(f"{db_value}: call_records table missing")
                        for plan_row in rows:
                            missing_rows.append(_missing_row(plan_row, reason="call_records_missing"))
                            missing_count += 1
                    else:
                        select_columns = [column for column in DEFAULT_COLUMNS if column in columns]
                        missing_required = [column for column in ("id", "source_filename", "analysis_json") if column not in columns]
                        if missing_required:
                            schema_warnings.append(f"{db_value}: missing columns {','.join(missing_required)}")
                        for plan_row in rows:
                            db_row = _fetch_snapshot_row(con, plan_row, select_columns=select_columns)
                            if db_row is None:
                                missing_rows.append(_missing_row(plan_row, reason="row_missing"))
                                missing_count += 1
                                continue
                            found_count += 1
                            rollback_rows.append(_rollback_row(plan_row, db_value=db_value, db_row=db_row, select_columns=select_columns))
                finally:
                    con.close()
            except sqlite3.Error as exc:
                schema_warnings.append(f"{db_value}: sqlite_error:{exc}")
                for plan_row in rows:
                    missing_rows.append(_missing_row(plan_row, reason=f"sqlite_error:{exc}"))
                    missing_count += 1
        db_manifest_rows.append(
            {
                **db_info,
                "candidate_rows": len(rows),
                "rollback_rows_found": found_count,
                "missing_rows": missing_count,
                "backup_required": True,
                "recommended_backup_filename": _recommended_backup_filename(db_path),
            }
        )

    outputs = {
        "summary_json": out_root / "summary.json",
        "report_markdown": out_root / "PHASE7_BACKUP_ROLLBACK_MANIFEST.md",
        "db_manifest_csv": out_root / "db_manifest.csv",
        "rollback_snapshot_csv": out_root / "rollback_snapshot.csv",
        "rollback_snapshot_jsonl": out_root / "rollback_snapshot.jsonl",
        "missing_rows_csv": out_root / "missing_rows.csv",
        "queue_summary_csv": out_root / "queue_summary.csv",
        "risk_summary_csv": out_root / "risk_summary.csv",
        "backup_copy_plan_sh": out_root / "backup_copy_plan.sh",
        "restore_notes_md": out_root / "ROLLBACK_RESTORE_NOTES.md",
    }
    _write_csv(outputs["db_manifest_csv"], db_manifest_rows)
    _write_csv(outputs["rollback_snapshot_csv"], rollback_rows)
    _write_jsonl(outputs["rollback_snapshot_jsonl"], rollback_rows)
    _write_csv(outputs["missing_rows_csv"], missing_rows)
    _write_csv(outputs["queue_summary_csv"], _counter_rows(queue_counts, "queue"))
    _write_csv(outputs["risk_summary_csv"], _counter_rows(risk_counts, "risk_level"))
    outputs["backup_copy_plan_sh"].write_text(_backup_copy_plan(db_manifest_rows), encoding="utf-8")
    outputs["restore_notes_md"].write_text(_restore_notes(), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "backup_rollback_manifest_read_only",
        "manifest_version": ROLLBACK_VERSION,
        "apply_plan_csv": str(apply_plan_path),
        "project_root": str(project_root),
        "queue_filter": config.queue_filter,
        "hash_db_files": config.hash_db_files,
        "input_rows": len(input_rows),
        "selected_rows": len(selected_rows),
        "dbs_affected": len(by_db),
        "rollback_rows": len(rollback_rows),
        "missing_rows": len(missing_rows),
        "queue_counts": dict(queue_counts.most_common()),
        "risk_counts": dict(risk_counts.most_common()),
        "schema_warnings": schema_warnings,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_markdown"].write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _db_file_info(db_path: Path, *, project_root: Path, hash_file: bool) -> dict[str, Any]:
    exists = db_path.exists()
    stat = db_path.stat() if exists else None
    return {
        "db": _rel(db_path, project_root) if exists else _clean(db_path),
        "db_abs_path": str(db_path),
        "exists": exists,
        "size_bytes": stat.st_size if stat else "",
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat() if stat else "",
        "sha256": _sha256_file(db_path) if exists and hash_file else "",
    }


def _fetch_snapshot_row(
    con: sqlite3.Connection,
    plan_row: dict[str, str],
    *,
    select_columns: list[str],
) -> sqlite3.Row | None:
    if not select_columns:
        return None
    call_id = int(float(_clean(plan_row.get("id")) or 0))
    source_filename = _clean(plan_row.get("source_filename"))
    quoted = ", ".join(f'"{column}"' for column in select_columns)
    return con.execute(
        f"""
        select {quoted}
          from call_records
         where id = ?
           and source_filename = ?
        """,
        (call_id, source_filename),
    ).fetchone()


def _rollback_row(
    plan_row: dict[str, str],
    *,
    db_value: str,
    db_row: sqlite3.Row,
    select_columns: list[str],
) -> dict[str, Any]:
    analysis_json = _clean(db_row["analysis_json"]) if "analysis_json" in select_columns else ""
    resolve_json = _clean(db_row["resolve_json"]) if "resolve_json" in select_columns else ""
    snapshot = {column: db_row[column] for column in select_columns}
    return {
        "rollback_version": ROLLBACK_VERSION,
        "db": db_value,
        "id": _clean(plan_row.get("id")),
        "source_filename": _clean(plan_row.get("source_filename")),
        "audit_id": _clean(plan_row.get("audit_id")),
        "task_id": _clean(plan_row.get("task_id")),
        "queue": _clean(plan_row.get("queue")),
        "risk_level": _clean(plan_row.get("risk_level")),
        "review_hash": _clean(plan_row.get("review_hash")),
        "current_call_type": _clean(plan_row.get("current_call_type")),
        "planned_normalized_call_type": _clean(plan_row.get("normalized_call_type")),
        "before_analysis_status": _clean(snapshot.get("analysis_status")),
        "before_resolve_status": _clean(snapshot.get("resolve_status")),
        "before_sync_status": _clean(snapshot.get("sync_status")),
        "before_analyze_attempts": _clean(snapshot.get("analyze_attempts")),
        "before_resolve_quality_score": _clean(snapshot.get("resolve_quality_score")),
        "before_dead_letter_stage": _clean(snapshot.get("dead_letter_stage")),
        "before_last_error": _clean(snapshot.get("last_error")),
        "before_next_retry_at": _clean(snapshot.get("next_retry_at")),
        "before_updated_at": _clean(snapshot.get("updated_at")),
        "before_analysis_json_sha256": _sha256_text(analysis_json),
        "before_resolve_json_sha256": _sha256_text(resolve_json),
        "before_analysis_json": analysis_json,
        "before_resolve_json": resolve_json,
        "before_snapshot_json": json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), default=str),
    }


def _missing_row(plan_row: dict[str, str], *, reason: str) -> dict[str, Any]:
    return {
        "db": _clean(plan_row.get("db")),
        "id": _clean(plan_row.get("id")),
        "source_filename": _clean(plan_row.get("source_filename")),
        "audit_id": _clean(plan_row.get("audit_id")),
        "task_id": _clean(plan_row.get("task_id")),
        "queue": _clean(plan_row.get("queue")),
        "reason": reason,
    }


def _existing_columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in con.execute(f"pragma table_info({table})")}
    except sqlite3.Error:
        return set()


def _recommended_backup_filename(db_path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{db_path.stem}.before_hard_gate_backfill_{timestamp}{db_path.suffix}"


def _backup_copy_plan(db_rows: list[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated plan only. Review paths before running.",
        'BACKUP_DIR="${1:-stable_runtime/backups/hard_gate_backfill_phase7}"',
        'mkdir -p "$BACKUP_DIR"',
        "",
    ]
    for row in db_rows:
        if not row.get("exists"):
            continue
        source = str(row.get("db_abs_path"))
        target = str(row.get("recommended_backup_filename"))
        lines.append(f"cp -p {json.dumps(source, ensure_ascii=False)} \"$BACKUP_DIR\"/{json.dumps(target, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _restore_notes() -> str:
    return """# Rollback Restore Notes

This package is read-only and did not change SQLite files.

Before any staged apply:

1. Copy affected DB files using `backup_copy_plan.sh` or another verified backup process.
2. Keep `db_manifest.csv`, `rollback_snapshot.csv`, and `rollback_snapshot.jsonl` with the backup.
3. After apply, rollback can be performed either by restoring full DB copies or by restoring per-row fields from `rollback_snapshot.jsonl`.
4. Prefer full DB restore for early staged batches. Per-row restore should be implemented as a separate audited script before use.

Fields captured per row include previous `analysis_json`, `resolve_json`, statuses, retry/error fields and hashes.
"""


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 7 Backup / Rollback Manifest",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Manifest version: `{summary['manifest_version']}`",
        f"- Apply plan: `{summary['apply_plan_csv']}`",
        f"- Selected rows: `{summary['selected_rows']}`",
        f"- DBs affected: `{summary['dbs_affected']}`",
        f"- Rollback rows captured: `{summary['rollback_rows']}`",
        f"- Missing rows: `{summary['missing_rows']}`",
        f"- Hash DB files: `{summary['hash_db_files']}`",
        "",
        "## Queue Counts",
    ]
    for key, value in summary["queue_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Risk Counts"])
    for key, value in summary["risk_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if summary.get("schema_warnings"):
        lines.extend(["", "## Schema Warnings"])
        for warning in summary["schema_warnings"]:
            lines.append(f"- `{warning}`")
    lines.extend(["", "## Outputs"])
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## Important",
            "",
            "No SQLite writes were performed. This is a pre-apply safety package.",
            "Real DB backup copies should be created immediately before staged apply, using the final `auto_apply_ready.csv` subset.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def _counter_rows(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, "count": value} for key, value in counter.most_common()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def _resolve_path(path: Path, project_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build backup/rollback manifest for hard-gate backfill plan.")
    parser.add_argument("--apply-plan-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--queue-filter")
    parser.add_argument("--no-db-hash", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> HardGateBackupManifestConfig:
    return HardGateBackupManifestConfig(
        apply_plan_csv=Path(args.apply_plan_csv),
        out_root=Path(args.out_root),
        project_root=Path(args.project_root),
        queue_filter=args.queue_filter,
        hash_db_files=not bool(args.no_db_hash),
    )


__all__ = [
    "HardGateBackupManifestConfig",
    "build_hard_gate_backup_manifest",
    "config_from_args",
    "parse_args",
]
