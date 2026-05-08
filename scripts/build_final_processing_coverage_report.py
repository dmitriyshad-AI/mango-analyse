from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


TERMINAL_RESOLVE = {"done", "skipped"}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _has_call_records(conn: sqlite3.Connection) -> bool:
    return bool(
        conn.execute(
            "select 1 from sqlite_master where type='table' and name='call_records'"
        ).fetchone()
    )


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)


def _read_baseline_db_list(path: Path, project_root: Path) -> list[Path]:
    rows: list[Path] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            db = str(row.get("db") or "").strip()
            if db:
                rows.append((project_root / db).resolve())
    return rows


def _read_exclusions(paths: list[Path]) -> dict[str, str]:
    exclusions: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = str(row.get("source_filename") or "").strip()
                if not name:
                    continue
                reason = str(row.get("exclusion_reason") or "excluded").strip() or "excluded"
                exclusions[name] = reason
    return exclusions


def _source_audio(source_dir: Path, start: date, end: date) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in source_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        meta = parse_filename_metadata(path.name)
        started_at = meta.get("started_at")
        if not isinstance(started_at, datetime):
            continue
        if not (start <= started_at.date() <= end):
            continue
        result[path.name] = {
            "source_filename": path.name,
            "source_file": str(path),
            "month": started_at.strftime("%Y-%m"),
            "started_at": started_at.isoformat(sep=" "),
        }
    return result


def _scan_db(db_path: Path, source_names: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    stats = {
        "db": "",
        "rows": 0,
        "source_hits": 0,
        "asr_hits": 0,
        "ra_hits": 0,
        "manual_hits": 0,
    }
    with _connect_ro(db_path) as conn:
        if not _has_call_records(conn):
            return states, stats
        for row in conn.execute(
            """
            select
                source_filename,
                transcription_status,
                resolve_status,
                analysis_status,
                coalesce(dead_letter_stage, '')
              from call_records
             where source_filename is not null
               and source_filename != ''
            """
        ):
            stats["rows"] += 1
            name = str(row[0] or "").strip()
            if name not in source_names:
                continue
            stats["source_hits"] += 1
            transcription_status = _norm(row[1])
            resolve_status = _norm(row[2])
            analysis_status = _norm(row[3])
            dead_letter_stage = str(row[4] or "").strip()
            asr_done = transcription_status == "done"
            full_ra = asr_done and resolve_status in TERMINAL_RESOLVE and analysis_status == "done"
            manual = asr_done and resolve_status == "manual" and not full_ra
            if asr_done:
                stats["asr_hits"] += 1
            if full_ra:
                stats["ra_hits"] += 1
            if manual:
                stats["manual_hits"] += 1
            state = states.setdefault(
                name,
                {
                    "asr_done": False,
                    "full_ra": False,
                    "manual_not_full_ra": False,
                    "dead_letter": False,
                    "evidence": [],
                },
            )
            if asr_done:
                state["asr_done"] = True
            if full_ra:
                state["full_ra"] = True
            if manual:
                state["manual_not_full_ra"] = True
            if dead_letter_stage:
                state["dead_letter"] = True
            if asr_done or full_ra or manual:
                state["evidence"].append(
                    {
                        "db": str(db_path),
                        "transcription_status": transcription_status,
                        "resolve_status": resolve_status,
                        "analysis_status": analysis_status,
                        "dead_letter_stage": dead_letter_stage,
                    }
                )
    return states, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict final processing coverage report.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-dir", default="2026-03-09--26")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-05-31")
    parser.add_argument("--baseline-included-dbs", required=True)
    parser.add_argument("--extra-db", action="append", default=[])
    parser.add_argument("--exclude-csv", action="append", default=[])
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    source_dir = (project_root / args.source_dir).resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    sources = _source_audio(source_dir, start, end)
    source_names = set(sources)
    exclusions = _read_exclusions([Path(item).expanduser().resolve() for item in args.exclude_csv])

    db_paths = _read_baseline_db_list(Path(args.baseline_included_dbs).expanduser().resolve(), project_root)
    db_paths.extend(Path(item).expanduser().resolve() for item in args.extra_db)
    unique_db_paths: list[Path] = []
    seen_db_paths: set[Path] = set()
    for path in db_paths:
        if path in seen_db_paths:
            continue
        seen_db_paths.add(path)
        if path.exists():
            unique_db_paths.append(path)

    aggregate: dict[str, dict[str, Any]] = {
        name: {
            "asr_done": False,
            "full_ra": False,
            "manual_not_full_ra": False,
            "dead_letter": False,
            "evidence": [],
        }
        for name in source_names
    }
    included_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for db_path in unique_db_paths:
        try:
            db_states, stats = _scan_db(db_path, source_names)
        except sqlite3.Error as exc:
            errors.append(f"{_rel(db_path, project_root)}: {exc}")
            continue
        stats["db"] = _rel(db_path, project_root)
        included_rows.append(stats)
        for name, state in db_states.items():
            agg = aggregate[name]
            agg["asr_done"] = bool(agg["asr_done"] or state["asr_done"])
            agg["full_ra"] = bool(agg["full_ra"] or state["full_ra"])
            agg["manual_not_full_ra"] = bool(
                agg["manual_not_full_ra"] or state["manual_not_full_ra"]
            )
            agg["dead_letter"] = bool(agg["dead_letter"] or state["dead_letter"])
            agg["evidence"].extend(state["evidence"])

    by_month: dict[str, Counter[str]] = {}
    missing_asr: list[str] = []
    missing_full_ra: list[str] = []
    manual_not_full_ra: list[str] = []
    asr_no_full_ra_non_manual: list[str] = []
    excluded_no_asr: list[str] = []

    for name in sorted(source_names):
        month = sources[name]["month"]
        counter = by_month.setdefault(month, Counter())
        state = aggregate[name]
        excluded = name in exclusions
        counter["source_audio"] += 1
        if excluded:
            counter["excluded"] += 1
        else:
            counter["actionable_source_audio"] += 1
        if state["asr_done"]:
            counter["asr_done"] += 1
        elif excluded:
            excluded_no_asr.append(name)
        else:
            counter["missing_asr"] += 1
            missing_asr.append(name)
        if state["full_ra"]:
            counter["full_ra"] += 1
        elif excluded:
            pass
        else:
            counter["missing_full_ra"] += 1
            missing_full_ra.append(name)
            if state["asr_done"] and state["manual_not_full_ra"]:
                counter["manual_not_full_ra"] += 1
                manual_not_full_ra.append(name)
            elif state["asr_done"]:
                counter["asr_no_full_ra_non_manual"] += 1
                asr_no_full_ra_non_manual.append(name)

    coverage_rows = []
    totals = Counter()
    for month in sorted(by_month):
        row = {"month": month, **by_month[month]}
        for key, value in row.items():
            if key != "month":
                totals[key] += int(value)
        coverage_rows.append(row)

    fieldnames = [
        "month",
        "source_audio",
        "excluded",
        "actionable_source_audio",
        "asr_done",
        "missing_asr",
        "full_ra",
        "missing_full_ra",
        "manual_not_full_ra",
        "asr_no_full_ra_non_manual",
    ]
    with (out_root / "coverage_by_month.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in coverage_rows:
            writer.writerow({field: row.get(field, 0) for field in fieldnames})
        writer.writerow({field: ("TOTAL" if field == "month" else totals.get(field, 0)) for field in fieldnames})

    with (out_root / "included_dbs.tsv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames_db = ["db", "rows", "source_hits", "asr_hits", "ra_hits", "manual_hits"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames_db, delimiter="\t")
        writer.writeheader()
        for row in included_rows:
            writer.writerow({field: row.get(field, 0) for field in fieldnames_db})

    outputs = {
        "missing_asr.txt": missing_asr,
        "missing_full_ra.txt": missing_full_ra,
        "manual_not_full_ra.txt": manual_not_full_ra,
        "asr_no_full_ra_non_manual.txt": asr_no_full_ra_non_manual,
        "excluded_no_asr.txt": excluded_no_asr,
    }
    for filename, names in outputs.items():
        (out_root / filename).write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "date_window": {"start": args.start_date, "end": args.end_date},
        "source_audio": totals["source_audio"],
        "excluded_no_asr": len(excluded_no_asr),
        "actionable_source_audio": totals["actionable_source_audio"],
        "asr_done": totals["asr_done"],
        "missing_asr_actionable": len(missing_asr),
        "full_ra": totals["full_ra"],
        "missing_full_ra_actionable": len(missing_full_ra),
        "manual_not_full_ra": len(manual_not_full_ra),
        "asr_no_full_ra_non_manual": len(asr_no_full_ra_non_manual),
        "included_db_count": len(included_rows),
        "errors": errors,
        "outputs": {key: str(out_root / key) for key in outputs},
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
