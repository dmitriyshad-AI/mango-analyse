from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.transcript_quality_backfill import (
    TranscriptQualityBackfillConfig,
    run_transcript_quality_backfill,
)


@dataclass(frozen=True)
class HardGateStagedBackfillConfig:
    project_root: Path
    auto_apply_csv: Path
    out_root: Path
    mode: str = "dry-run"
    limit: int | None = None
    offset: int = 0
    create_backup: bool = True


def run_hard_gate_staged_backfill(config: HardGateStagedBackfillConfig) -> dict[str, Any]:
    if config.mode not in {"dry-run", "apply"}:
        raise ValueError("mode must be 'dry-run' or 'apply'")
    project_root = config.project_root.expanduser().resolve()
    auto_apply_csv = _resolve_path(project_root, config.auto_apply_csv)
    out_root = _resolve_path(project_root, config.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(auto_apply_csv)
    selected = rows[max(0, int(config.offset)) :]
    if config.limit is not None:
        selected = selected[: max(0, int(config.limit))]
    selected = [row for row in selected if _clean(row.get("queue")) == "auto_apply_ready"]

    selected_csv = out_root / "selected_auto_apply.csv"
    _write_csv(selected_csv, selected)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        db = _clean(row.get("db"))
        if db:
            grouped[db].append(row)

    db_summaries: list[dict[str, Any]] = []
    totals = Counter()
    missing_db_rows: list[dict[str, str]] = []
    for db, db_rows in sorted(grouped.items()):
        db_path = _resolve_path(project_root, Path(db))
        safe_name = _safe_name(db_path)
        db_out = out_root / "per_db" / safe_name
        db_out.mkdir(parents=True, exist_ok=True)
        db_csv = db_out / "candidates.csv"
        _write_csv(db_csv, db_rows)
        if not db_path.exists():
            for row in db_rows:
                missing_db_rows.append({**row, "missing_db": str(db_path)})
            db_summaries.append(
                {
                    "db": db,
                    "db_path": str(db_path),
                    "input_candidates": len(db_rows),
                    "missing_db": True,
                }
            )
            totals["missing_db_rows"] += len(db_rows)
            continue
        summary = run_transcript_quality_backfill(
            TranscriptQualityBackfillConfig(
                database_url=f"sqlite:///{db_path}",
                candidates_csv=db_csv,
                out_root=db_out,
                mode=config.mode,
                create_backup=config.create_backup,
            )
        )
        db_summary = {
            "db": db,
            "db_path": str(db_path),
            "input_candidates": summary["input_candidates"],
            "planned_updates": summary["planned_updates"],
            "applied_updates": summary["applied_updates"],
            "blocked_rows": summary["blocked_rows"],
            "already_applied": summary["already_applied"],
            "missing_rows": summary["missing_rows"],
            "errors": summary["errors"],
            "backup_path": summary.get("backup_path"),
            "out_root": str(db_out),
            "missing_db": False,
        }
        db_summaries.append(db_summary)
        for key in [
            "input_candidates",
            "planned_updates",
            "applied_updates",
            "blocked_rows",
            "already_applied",
            "missing_rows",
            "errors",
        ]:
            totals[key] += int(summary.get(key) or 0)

    _write_csv(out_root / "db_summary.csv", db_summaries)
    _write_csv(out_root / "missing_db_rows.csv", missing_db_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": config.mode,
        "project_root": str(project_root),
        "auto_apply_csv": str(auto_apply_csv),
        "out_root": str(out_root),
        "limit": config.limit,
        "offset": config.offset,
        "create_backup": config.create_backup,
        "selected_rows": len(selected),
        "selected_dbs": len(grouped),
        "counts": dict(totals),
        "risk_counts": dict(Counter(_clean(row.get("risk_level")) or "unknown" for row in selected)),
        "month_counts": dict(Counter(_clean(row.get("month")) or "unknown" for row in selected)),
        "outputs": {
            "selected_auto_apply_csv": str(selected_csv),
            "db_summary_csv": str(out_root / "db_summary.csv"),
            "missing_db_rows_csv": str(out_root / "missing_db_rows.csv"),
            "summary_json": str(out_root / "summary.json"),
            "report_markdown": str(out_root / "HARD_GATE_STAGED_BACKFILL_REPORT.md"),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "HARD_GATE_STAGED_BACKFILL_REPORT.md").write_text(_markdown_report(summary, db_summaries), encoding="utf-8")
    return summary


def _markdown_report(summary: dict[str, Any], db_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Hard Gate Staged Backfill Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Selected rows: `{summary['selected_rows']}`",
        f"- Selected DBs: `{summary['selected_dbs']}`",
        f"- Planned updates: `{summary['counts'].get('planned_updates', 0)}`",
        f"- Applied updates: `{summary['counts'].get('applied_updates', 0)}`",
        f"- Already applied: `{summary['counts'].get('already_applied', 0)}`",
        f"- Blocked rows: `{summary['counts'].get('blocked_rows', 0)}`",
        f"- Missing rows: `{summary['counts'].get('missing_rows', 0)}`",
        f"- Errors: `{summary['counts'].get('errors', 0)}`",
        "",
        "## DB Summary",
        "",
        "| DB | Input | Planned | Applied | Already | Blocked | Missing | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in db_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("db", "")),
                    str(row.get("input_candidates", 0)),
                    str(row.get("planned_updates", 0)),
                    str(row.get("applied_updates", 0)),
                    str(row.get("already_applied", 0)),
                    str(row.get("blocked_rows", 0)),
                    str(row.get("missing_rows", 0)),
                    str(row.get("errors", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


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


def _resolve_path(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _safe_name(path: Path) -> str:
    text = str(path)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[-180:]


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply hard-gate auto_apply_ready rows by SQLite DB in staged batches.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--auto-apply-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> HardGateStagedBackfillConfig:
    return HardGateStagedBackfillConfig(
        project_root=args.project_root,
        auto_apply_csv=args.auto_apply_csv,
        out_root=args.out_root,
        mode=args.mode,
        limit=args.limit,
        offset=args.offset,
        create_backup=not bool(args.no_backup),
    )


__all__ = ["HardGateStagedBackfillConfig", "run_hard_gate_staged_backfill", "parse_args", "config_from_args"]
