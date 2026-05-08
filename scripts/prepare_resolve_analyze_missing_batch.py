from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKIP_DIR_PARTS = {
    ".git",
    ".cache",
    ".pytest_cache",
    ".venv-asrbench",
    "tests",
    "test_sets",
    "test_runs",
    "_local_archive_20260424",
}
SKIP_SUBSTRINGS = (
    ".before_",
    "before_",
    "_before_",
    "backup",
    ".bak",
    "_bak",
    "broken",
    "ab_test",
    "benchmark",
    "test300",
)
MONTH_RE = re.compile(r"^(20\d\d-[01]\d)-[0-3]\d__")
TERMINAL_RESOLVE = {"done", "skipped"}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _should_skip_db(path: Path, root: Path, excluded: set[Path]) -> bool:
    resolved = path.resolve()
    if resolved in excluded:
        return True
    rel = path.relative_to(root)
    if any(part in SKIP_DIR_PARTS for part in rel.parts):
        return True
    lowered = str(rel).lower()
    return any(token in lowered for token in SKIP_SUBSTRINGS)


def _has_call_records(conn: sqlite3.Connection) -> bool:
    try:
        return bool(
            conn.execute(
                "select 1 from sqlite_master where type='table' and name='call_records'"
            ).fetchone()
        )
    except sqlite3.Error:
        return False


def _month_from_filename(name: str) -> str:
    match = MONTH_RE.match(name)
    if match:
        return match.group(1)
    if re.match(r"^20\d\d-[01]\d", name):
        return name[:7]
    return "unknown"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _candidate_score(
    *,
    resolve_status: str,
    analysis_status: str,
    dead_letter_stage: str,
    variants_len: int,
    transcript_len: int,
    has_gigaam: int,
    updated_at: str,
) -> tuple[int, str]:
    score = 0
    if not dead_letter_stage:
        score += 50
    else:
        score -= 200
    if resolve_status == "done":
        score += 1000
    elif resolve_status == "skipped":
        score += 900
    elif resolve_status == "manual":
        score -= 50
    elif resolve_status == "failed":
        score -= 20
    if analysis_status in {"pending", "failed", "in_progress"}:
        score += 10
    if has_gigaam:
        score += 80
    score += min(max(variants_len, 0) // 1000, 120)
    score += min(max(transcript_len, 0) // 1000, 80)
    return score, updated_at or ""


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)


def _copy_schema(template_db: Path, target_db: Path) -> list[str]:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    if target_db.exists():
        target_db.unlink()
    with _connect_ro(template_db) as src, sqlite3.connect(target_db) as dst:
        rows = src.execute(
            """
            select type, name, sql
              from sqlite_master
             where tbl_name = 'call_records'
               and sql is not null
               and type in ('table', 'index')
             order by case type when 'table' then 0 else 1 end, name
            """
        ).fetchall()
        for _, _, sql in rows:
            dst.execute(sql)
        dst.commit()
        columns = [row[1] for row in dst.execute("pragma table_info(call_records)")]
    return columns


def _source_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("pragma table_info(call_records)")}


def _sql_literal_default(column: str) -> Any:
    if column in {
        "transcription_status",
        "resolve_status",
        "analysis_status",
        "sync_status",
    }:
        return "pending"
    if column in {
        "transcribe_attempts",
        "resolve_attempts",
        "analyze_attempts",
        "sync_attempts",
    }:
        return 0
    if column in {"created_at", "updated_at"}:
        return datetime.now(timezone.utc).isoformat(sep=" ")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one consolidated Resolve+Analyze batch DB from unique calls that already "
            "have ASR but have no full R+A in any scanned working DB."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--exclude-db", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    target_db = out_root / f"{out_root.name}.db"
    excluded = {Path(item).expanduser().resolve() for item in args.exclude_db}

    if out_root.exists() and any(out_root.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty, use --force: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    db_paths = [
        path
        for path in sorted(project_root.rglob("*.db"))
        if path.is_file() and not _should_skip_db(path, project_root, excluded)
    ]

    states: dict[str, dict[str, Any]] = {}
    included_dbs: list[dict[str, Any]] = []
    errors: list[str] = []
    template_db: Path | None = None

    for db_path in db_paths:
        try:
            conn = _connect_ro(db_path)
        except sqlite3.Error as exc:
            errors.append(f"{_rel(db_path, project_root)}: open: {exc}")
            continue
        try:
            if not _has_call_records(conn):
                continue
            if template_db is None:
                template_db = db_path
            rel_db = _rel(db_path, project_root)
            rows_seen = asr_rows = ra_rows = 0
            for row in conn.execute(
                """
                select
                    id,
                    source_filename,
                    transcription_status,
                    resolve_status,
                    analysis_status,
                    coalesce(dead_letter_stage, ''),
                    length(coalesce(transcript_variants_json, '')),
                    length(coalesce(transcript_text, '')),
                    case when lower(coalesce(transcript_variants_json, '')) like '%giga%' then 1 else 0 end,
                    coalesce(updated_at, '')
                  from call_records
                 where source_filename is not null
                   and source_filename != ''
                """
            ):
                rows_seen += 1
                (
                    row_id,
                    source_filename,
                    transcription_status,
                    resolve_status_raw,
                    analysis_status_raw,
                    dead_letter_stage,
                    variants_len,
                    transcript_len,
                    has_gigaam,
                    updated_at,
                ) = row
                name = str(source_filename or "").strip()
                if not name:
                    continue
                transcription_done = _norm(transcription_status) == "done"
                resolve_status = _norm(resolve_status_raw)
                analysis_status = _norm(analysis_status_raw)
                full_ra = (
                    transcription_done
                    and resolve_status in TERMINAL_RESOLVE
                    and analysis_status == "done"
                )
                if transcription_done:
                    asr_rows += 1
                if full_ra:
                    ra_rows += 1

                state = states.setdefault(
                    name,
                    {
                        "month": _month_from_filename(name),
                        "asr_done": False,
                        "ra_done": False,
                        "candidate": None,
                    },
                )
                if transcription_done:
                    state["asr_done"] = True
                    score = _candidate_score(
                        resolve_status=resolve_status,
                        analysis_status=analysis_status,
                        dead_letter_stage=str(dead_letter_stage or ""),
                        variants_len=int(variants_len or 0),
                        transcript_len=int(transcript_len or 0),
                        has_gigaam=int(has_gigaam or 0),
                        updated_at=str(updated_at or ""),
                    )
                    candidate = {
                        "score": score,
                        "db_path": db_path,
                        "db": rel_db,
                        "id": int(row_id),
                        "source_filename": name,
                        "source_resolve_status": resolve_status,
                        "source_analysis_status": analysis_status,
                    }
                    previous = state.get("candidate")
                    if previous is None or candidate["score"] > previous["score"]:
                        state["candidate"] = candidate
                if full_ra:
                    state["ra_done"] = True
            if rows_seen:
                included_dbs.append(
                    {
                        "db": rel_db,
                        "rows": rows_seen,
                        "asr_rows": asr_rows,
                        "ra_rows": ra_rows,
                    }
                )
        except sqlite3.Error as exc:
            errors.append(f"{_rel(db_path, project_root)}: scan: {exc}")
        finally:
            conn.close()

    if template_db is None:
        raise SystemExit("No usable call_records DB found")

    selected = [
        state["candidate"]
        for state in states.values()
        if state.get("asr_done") and not state.get("ra_done") and state.get("candidate")
    ]
    selected.sort(key=lambda item: (states[item["source_filename"]]["month"], item["source_filename"]))

    columns = _copy_schema(template_db, target_db)
    insert_columns = [column for column in columns if column != "id"]
    placeholders = ", ".join("?" for _ in insert_columns)
    insert_sql = (
        f"insert into call_records ({', '.join(insert_columns)}) values ({placeholders})"
    )
    now = datetime.now(timezone.utc).isoformat(sep=" ")
    selected_by_month: Counter[str] = Counter()
    selected_by_source_db: Counter[str] = Counter()
    source_status_pairs: Counter[str] = Counter()

    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        grouped[item["db_path"]].append(item)

    with sqlite3.connect(target_db) as dst:
        for db_path, items in grouped.items():
            ids = [int(item["id"]) for item in items]
            item_by_id = {int(item["id"]): item for item in items}
            with _connect_ro(db_path) as src:
                src_columns = _source_columns(src)
                select_columns = [
                    column if column in src_columns else f"NULL as {column}"
                    for column in columns
                    if column != "id"
                ]
                for chunk_start in range(0, len(ids), 500):
                    chunk = ids[chunk_start : chunk_start + 500]
                    sql = (
                        f"select id, {', '.join(select_columns)} from call_records "
                        f"where id in ({', '.join('?' for _ in chunk)})"
                    )
                    for row in src.execute(sql, chunk):
                        row_id = int(row[0])
                        item = item_by_id[row_id]
                        values = dict(zip(insert_columns, row[1:]))
                        source_resolve = item["source_resolve_status"]
                        source_analysis = item["source_analysis_status"]

                        values["transcription_status"] = "done"
                        if source_resolve in TERMINAL_RESOLVE:
                            values["resolve_status"] = source_resolve
                        else:
                            values["resolve_status"] = "pending"
                            values["resolve_json"] = None
                            values["resolve_quality_score"] = None
                            values["resolve_attempts"] = 0

                        values["analysis_status"] = "pending"
                        values["analysis_json"] = None
                        values["analyze_attempts"] = 0
                        values["sync_status"] = "pending"
                        values["sync_attempts"] = 0
                        values["pipeline_stage"] = None
                        values["pipeline_worker_id"] = None
                        values["pipeline_claimed_at"] = None
                        values["analysis_worker_id"] = None
                        values["analysis_claimed_at"] = None
                        values["next_retry_at"] = None
                        values["dead_letter_stage"] = None
                        values["last_error"] = None
                        values["created_at"] = values.get("created_at") or now
                        values["updated_at"] = now

                        for column in insert_columns:
                            if column not in values:
                                values[column] = _sql_literal_default(column)
                        dst.execute(insert_sql, [values.get(column) for column in insert_columns])

                        month = states[item["source_filename"]]["month"]
                        selected_by_month[str(month)] += 1
                        selected_by_source_db[item["db"]] += 1
                        source_status_pairs[f"{source_resolve or 'none'}|{source_analysis or 'none'}"] += 1
        dst.commit()

    selected_csv = out_root / "selected_calls.tsv"
    included_csv = out_root / "included_dbs.tsv"
    with selected_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source_filename",
                "month",
                "source_db",
                "source_resolve_status",
                "source_analysis_status",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "source_filename": item["source_filename"],
                    "month": states[item["source_filename"]]["month"],
                    "source_db": item["db"],
                    "source_resolve_status": item["source_resolve_status"],
                    "source_analysis_status": item["source_analysis_status"],
                }
            )
    with included_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["db", "rows", "asr_rows", "ra_rows"], delimiter="\t")
        writer.writeheader()
        writer.writerows(included_dbs)

    manifest = {
        "generated_at": now,
        "project_root": str(project_root),
        "out_root": str(out_root),
        "db_path": str(target_db),
        "definition": {
            "dedupe_key": "source_filename",
            "selected": "ASR done in at least one scanned DB and no full R+A in any scanned DB",
            "full_ra": "transcription_status='done' and resolve_status in ('done','skipped') and analysis_status='done'",
        },
        "excluded_dbs": [str(path) for path in sorted(excluded)],
        "db_files_considered": len(db_paths),
        "included_db_count": len(included_dbs),
        "unique_filenames_seen": len(states),
        "selected_calls": len(selected),
        "selected_by_month": dict(sorted(selected_by_month.items())),
        "selected_by_source_db_top": selected_by_source_db.most_common(50),
        "source_status_pairs": dict(sorted(source_status_pairs.items())),
        "template_db": _rel(template_db, project_root),
        "selected_calls_tsv": str(selected_csv),
        "included_dbs_tsv": str(included_csv),
        "errors": errors,
        "pipeline": {
            "transcribe": False,
            "backfill_second_asr": False,
            "resolve": True,
            "analyze": True,
            "sync": False,
        },
    }
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
