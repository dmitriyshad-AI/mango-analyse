from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
ARCHIVE_HINTS = (
    "before_",
    ".before_",
    "broken_",
    "benchmarks",
    "ab_tests",
    "external_m1",
    "asr_only",
    "overnight_",
    "sales_master_export_20260413",
    "sales_master_export_20260414",
    "sales_master_export_20260415",
)
DO_NOT_TOUCH_HINTS = (
    "2026-03-09--26",
    "ra_missing_all_20260506",
    "amocrm_runtime",
    "non_conversation_hard_gate",
    "transcript_quality_",
    ".env",
)


@dataclass(frozen=True)
class ProjectInventoryConfig:
    project_root: Path
    out_root: Path
    max_depth: int = 2
    replacement_artifact: str = "canonical_master_db_and_manifest"


def build_project_inventory(config: ProjectInventoryConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    out_root = _resolve_under_project(config.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    top_dirs = _top_level_sizes(project_root)
    runtime_dirs = _child_dir_sizes(project_root / "stable_runtime")
    db_rows = _db_inventory(project_root)
    archive_rows = _archive_candidates(project_root, top_dirs, runtime_dirs, db_rows, replacement_artifact=config.replacement_artifact)

    outputs = {
        "summary_json": out_root / "summary.json",
        "top_level_sizes_tsv": out_root / "top_level_sizes.tsv",
        "stable_runtime_sizes_tsv": out_root / "stable_runtime_sizes.tsv",
        "db_inventory_tsv": out_root / "db_inventory.tsv",
        "archive_candidates_dry_run_tsv": out_root / "archive_candidates_dry_run.tsv",
        "README_md": out_root / "README.md",
    }
    _write_tsv(outputs["top_level_sizes_tsv"], top_dirs)
    _write_tsv(outputs["stable_runtime_sizes_tsv"], runtime_dirs)
    _write_tsv(outputs["db_inventory_tsv"], db_rows)
    _write_tsv(outputs["archive_candidates_dry_run_tsv"], archive_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "read_only_inventory",
        "project_root": str(project_root),
        "replacement_artifact": config.replacement_artifact,
        "project_size_bytes": sum(row["size_bytes"] for row in top_dirs if row["path"] != "."),
        "top_level_entries": len(top_dirs),
        "stable_runtime_entries": len(runtime_dirs),
        "db_files": len(db_rows),
        "db_total_size_bytes": sum(row["size_bytes"] for row in db_rows),
        "archive_candidate_rows": len(archive_rows),
        "archive_candidate_size_bytes": sum(row["size_bytes"] for row in archive_rows if row["safe_to_archive_after_master"] == "yes"),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["README_md"].write_text(_readme(summary), encoding="utf-8")
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _top_level_sizes(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(project_root.iterdir(), key=lambda p: p.name):
        rows.append(_path_size_row(path, project_root))
    return sorted(rows, key=lambda row: row["size_bytes"], reverse=True)


def _child_dir_sizes(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = [_path_size_row(path, root.parent) for path in sorted(root.iterdir(), key=lambda p: p.name) if path.is_dir()]
    return sorted(rows, key=lambda row: row["size_bytes"], reverse=True)


def _path_size_row(path: Path, project_root: Path) -> dict[str, Any]:
    size = _du_bytes(path)
    logical_size = _logical_bytes(path)
    rel = _rel(path, project_root)
    return {
        "path": rel,
        "type": "dir" if path.is_dir() else "file",
        "size_bytes": size,
        "size_human": _human(size),
        "logical_size_bytes": logical_size,
        "logical_size_human": _human(logical_size),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds"),
        "classification": _classification(rel),
    }


def _db_inventory(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(project_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DB_SUFFIXES:
            continue
        if ".git" in path.parts:
            continue
        rel = _rel(path, project_root)
        row_count = ""
        has_call_records = "false"
        db_error = ""
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as con:
                if con.execute("select 1 from sqlite_master where type='table' and name='call_records'").fetchone():
                    has_call_records = "true"
                    row_count = con.execute("select count(*) from call_records").fetchone()[0]
        except sqlite3.Error as exc:
            db_error = str(exc)
        size = path.stat().st_size
        rows.append(
            {
                "path": rel,
                "size_bytes": size,
                "size_human": _human(size),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds"),
                "has_call_records": has_call_records,
                "call_records_rows": row_count,
                "classification": _classification(rel),
                "db_error": db_error,
            }
        )
    return sorted(rows, key=lambda row: row["size_bytes"], reverse=True)


def _archive_candidates(
    project_root: Path,
    top_dirs: list[dict[str, Any]],
    runtime_dirs: list[dict[str, Any]],
    db_rows: list[dict[str, Any]],
    *,
    replacement_artifact: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in [*top_dirs, *runtime_dirs]:
        path = str(row["path"])
        classification = _classification(path)
        if classification not in {"candidate_after_master", "review_after_master"}:
            continue
        rows.append(
            {
                "path": path,
                "type": row["type"],
                "size_bytes": row["size_bytes"],
                "size_human": row["size_human"],
                "reason": classification,
                "safe_to_archive_after_master": "yes" if classification == "candidate_after_master" else "review",
                "requires_manual_approval": "yes",
                "replacement_artifact": replacement_artifact,
            }
        )
    for row in db_rows:
        path = str(row["path"])
        if ".before_" not in path and "before_" not in path and "broken_" not in path:
            continue
        rows.append(
            {
                "path": path,
                "type": "db_backup",
                "size_bytes": row["size_bytes"],
                "size_human": row["size_human"],
                "reason": "backup_db_candidate_after_master",
                "safe_to_archive_after_master": "yes",
                "requires_manual_approval": "yes",
                "replacement_artifact": replacement_artifact,
            }
        )
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["path"]] = row
    return sorted(deduped.values(), key=lambda row: row["size_bytes"], reverse=True)


def _classification(rel: str) -> str:
    if any(hint in rel for hint in DO_NOT_TOUCH_HINTS):
        return "do_not_touch_now"
    if any(hint in rel for hint in ARCHIVE_HINTS):
        return "candidate_after_master"
    if rel in {"telegram_exports (2)", "_local_archive_20260424", "2026-03-05-21-06-49-ч1", "2026-03-05-21-06-49-ч2", ".venv-asrbench"}:
        return "review_after_master"
    return "keep_or_review"


def _du_bytes(path: Path) -> int:
    du_size = _du_bytes_from_system(path)
    if du_size is not None:
        return du_size
    if path.is_file():
        stat = path.stat()
        return int(getattr(stat, "st_blocks", 0) or 0) * 512 or stat.st_size
    total = 0
    for root, dirs, files in os.walk(path):
        if ".git" in Path(root).parts and path.name != ".git":
            dirs[:] = []
            continue
        for name in files:
            file_path = Path(root) / name
            try:
                stat = file_path.stat()
                total += int(getattr(stat, "st_blocks", 0) or 0) * 512 or stat.st_size
            except OSError:
                pass
    return total


def _du_bytes_from_system(path: Path) -> int | None:
    try:
        proc = subprocess.run(
            ["du", "-sk", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = proc.stdout.splitlines()[0].split()[0] if proc.stdout.splitlines() else ""
    try:
        return int(first) * 1024
    except ValueError:
        return None


def _logical_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, dirs, files in os.walk(path):
        if ".git" in Path(root).parts and path.name != ".git":
            dirs[:] = []
            continue
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _readme(summary: dict[str, Any]) -> str:
    return (
        "# Project inventory\n\n"
        "Read-only inventory. No files were deleted or moved.\n\n"
        f"- DB files: `{summary['db_files']}`\n"
        f"- DB total size: `{_human(summary['db_total_size_bytes'])}`\n"
        f"- Archive candidate rows: `{summary['archive_candidate_rows']}`\n"
        f"- Candidate size after master: `{_human(summary['archive_candidate_size_bytes'])}`\n\n"
        f"Replacement artifact: `{summary['replacement_artifact']}`\n\n"
        "Use this only after canonical master validation passes. Archive/delete still requires manual approval.\n"
    )


def _human(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


def _resolve_under_project(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only project cleanup inventory.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--replacement-artifact", default="canonical_master_db_and_manifest")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ProjectInventoryConfig:
    return ProjectInventoryConfig(
        project_root=args.project_root,
        out_root=args.out_root,
        replacement_artifact=args.replacement_artifact,
    )
