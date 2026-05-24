from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


TERMINAL_RESOLVE = {"done", "skipped"}
CALL_RECORD_COLUMNS = [
    "id",
    "source_file",
    "source_filename",
    "source_call_id",
    "audio_codec",
    "sample_rate",
    "channels",
    "duration_sec",
    "phone",
    "manager_name",
    "direction",
    "started_at",
    "transcription_status",
    "resolve_status",
    "analysis_status",
    "sync_status",
    "transcribe_attempts",
    "resolve_attempts",
    "analyze_attempts",
    "sync_attempts",
    "pipeline_stage",
    "pipeline_worker_id",
    "pipeline_claimed_at",
    "analysis_worker_id",
    "analysis_claimed_at",
    "next_retry_at",
    "dead_letter_stage",
    "transcript_manager",
    "transcript_client",
    "transcript_text",
    "transcript_variants_json",
    "resolve_json",
    "resolve_quality_score",
    "analysis_json",
    "amocrm_contact_id",
    "amocrm_lead_id",
    "last_error",
    "created_at",
    "updated_at",
]


@dataclass(frozen=True)
class CanonicalMasterConfig:
    project_root: Path
    source_dir: Path
    included_dbs_tsv: Path
    out_root: Path
    excluded_no_asr_txt: Path | None = None
    start_date: date = date(2025, 1, 1)
    end_date: date = date(2026, 5, 31)
    mode: str = "dry-run"
    canonical_db_name: str = "canonical_calls_master.db"
    expected_source_audio: int | None = 64867
    expected_excluded_no_asr: int | None = 35
    expected_actionable_source_audio: int | None = 64832
    expected_asr_done_actionable: int | None = 64832
    expected_full_ra_actionable: int | None = 64832


def build_canonical_master_preview(config: CanonicalMasterConfig) -> dict[str, Any]:
    if config.mode not in {"dry-run", "write"}:
        raise ValueError("Unsupported mode. Use dry-run or write.")

    project_root = config.project_root.resolve()
    source_dir = _resolve_under_project(config.source_dir, project_root)
    out_root = _resolve_under_project(config.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sources = _source_audio(source_dir, config.start_date, config.end_date)
    source_names = set(sources)
    exclusions = _read_name_list(
        _resolve_under_project(config.excluded_no_asr_txt, project_root)
        if config.excluded_no_asr_txt
        else None
    )
    db_paths = _read_included_dbs(_resolve_under_project(config.included_dbs_tsv, project_root), project_root)

    all_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    db_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for db_index, db_path in enumerate(db_paths):
        try:
            candidates, stats = _scan_db(db_path, db_index=db_index, source_names=source_names, project_root=project_root)
        except sqlite3.Error as exc:
            errors.append(f"{_rel(db_path, project_root)}: {exc}")
            continue
        db_rows.append(stats)
        for name, rows in candidates.items():
            all_candidates[name].extend(rows)

    selected_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    selected_by_name: dict[str, dict[str, Any] | None] = {}
    by_month: dict[str, Counter[str]] = defaultdict(Counter)
    selected_by_db: Counter[str] = Counter()
    duplicate_source_names = 0

    for name in sorted(source_names):
        source = sources[name]
        month = source["month"]
        counter = by_month[month]
        counter["source_audio"] += 1
        excluded = name in exclusions
        if excluded:
            counter["excluded_no_asr"] += 1
        else:
            counter["actionable_source_audio"] += 1

        candidates = all_candidates.get(name, [])
        if len(candidates) > 1:
            duplicate_source_names += 1
            conflict_rows.append(_conflict_row(source, candidates, project_root=project_root))

        selected = _select_best_candidate(candidates)
        selected_by_name[name] = selected
        status = _canonical_status(selected, excluded=excluded)
        counter[status] += 1
        if selected:
            if _is_asr_done(selected):
                counter["asr_done"] += 1
            selected_by_db[_clean(selected.get("provenance_db"))] += 1
        selected_rows.append(_preview_row(source, selected, status=status, excluded=excluded))

    coverage_rows = _coverage_rows(by_month)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": config.mode,
        "project_root": str(project_root),
        "source_dir": str(source_dir),
        "included_dbs_tsv": str(_resolve_under_project(config.included_dbs_tsv, project_root)),
        "excluded_no_asr_txt": str(_resolve_under_project(config.excluded_no_asr_txt, project_root)) if config.excluded_no_asr_txt else "",
        "date_window": {"start": config.start_date.isoformat(), "end": config.end_date.isoformat()},
        "source_audio": len(sources),
        "excluded_no_asr": len([name for name in source_names if name in exclusions]),
        "actionable_source_audio": len([name for name in source_names if name not in exclusions]),
        "selected_records": sum(1 for row in selected_rows if row["canonical_status"] != "missing"),
        "asr_done_actionable": sum(
            1 for row in selected_rows if row["is_actionable"] == "true" and row["transcription_status"] == "done"
        ),
        "full_ra_actionable": sum(
            1 for row in selected_rows if row["is_actionable"] == "true" and row["is_full_ra"] == "true"
        ),
        "missing_asr_actionable": sum(
            1 for row in selected_rows if row["is_actionable"] == "true" and row["transcription_status"] != "done"
        ),
        "missing_full_ra_actionable": sum(
            1 for row in selected_rows if row["is_actionable"] == "true" and row["is_full_ra"] != "true"
        ),
        "duplicate_source_names_with_candidates": duplicate_source_names,
        "included_db_count": len(db_paths),
        "dbs_with_selected_records": len(selected_by_db),
        "errors": errors,
        "validation": {},
        "outputs": {},
    }
    summary["validation"] = _validation(summary, config)

    outputs = {
        "summary_json": out_root / "summary.json",
        "canonical_preview_csv": out_root / "canonical_preview.csv",
        "coverage_by_month_tsv": out_root / "coverage_by_month.tsv",
        "db_scan_summary_tsv": out_root / "db_scan_summary.tsv",
        "selected_by_db_tsv": out_root / "selected_by_db.tsv",
        "duplicate_conflicts_csv": out_root / "duplicate_conflicts.csv",
        "README_md": out_root / "README.md",
    }
    if config.mode == "write":
        outputs["canonical_db"] = out_root / config.canonical_db_name
    _write_csv(outputs["canonical_preview_csv"], selected_rows)
    _write_tsv(outputs["coverage_by_month_tsv"], coverage_rows)
    _write_tsv(outputs["db_scan_summary_tsv"], db_rows)
    _write_tsv(
        outputs["selected_by_db_tsv"],
        [{"db": db, "selected_records": count} for db, count in selected_by_db.most_common()],
    )
    _write_csv(outputs["duplicate_conflicts_csv"], conflict_rows)
    if config.mode == "write":
        db_summary = _write_canonical_db(
            outputs["canonical_db"],
            summary=summary,
            config=config,
            project_root=project_root,
            source_dir=source_dir,
            sources=sources,
            exclusions=exclusions,
            db_paths=db_paths,
            db_rows=db_rows,
            all_candidates=all_candidates,
            selected_by_name=selected_by_name,
        )
        summary["canonical_db"] = db_summary
    outputs["README_md"].write_text(_readme(summary), encoding="utf-8")
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _source_audio(source_dir: Path, start: date, end: date) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in source_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        meta = parse_filename_metadata(path.name)
        started_at = meta.get("started_at")
        if not isinstance(started_at, datetime):
            continue
        if not (start <= started_at.date() <= end):
            continue
        stat = path.stat()
        result[path.name] = {
            "source_filename": path.name,
            "source_file": str(path.resolve()),
            "month": started_at.strftime("%Y-%m"),
            "started_at": started_at.isoformat(sep=" "),
            "phone_from_filename": meta.get("phone"),
            "manager_from_filename": meta.get("manager_name"),
            "source_call_id_from_filename": meta.get("source_call_id"),
            "audio_size_bytes": stat.st_size,
            "audio_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds"),
        }
    return result


def _write_canonical_db(
    db_path: Path,
    *,
    summary: dict[str, Any],
    config: CanonicalMasterConfig,
    project_root: Path,
    source_dir: Path,
    sources: dict[str, dict[str, Any]],
    exclusions: set[str],
    db_paths: list[Path],
    db_rows: list[dict[str, Any]],
    all_candidates: dict[str, list[dict[str, Any]]],
    selected_by_name: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    build_id = f"canonical_master_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    with sqlite3.connect(tmp_path) as con:
        con.execute("pragma journal_mode=delete")
        con.execute("pragma synchronous=normal")
        _create_canonical_schema(con)
        _insert_build_row(con, build_id=build_id, summary=summary, config=config, project_root=project_root, source_dir=source_dir)
        artifact_ids = _insert_source_artifacts(
            con,
            build_id=build_id,
            project_root=project_root,
            source_dir=source_dir,
            config=config,
            db_paths=db_paths,
            db_rows=db_rows,
        )
        canonical_ids = _insert_canonical_calls(
            con,
            build_id=build_id,
            sources=sources,
            exclusions=exclusions,
            selected_by_name=selected_by_name,
            all_candidates=all_candidates,
        )
        _insert_provenance(
            con,
            build_id=build_id,
            canonical_ids=canonical_ids,
            all_candidates=all_candidates,
            selected_by_name=selected_by_name,
            artifact_ids=artifact_ids,
        )
        _insert_exclusions(
            con,
            build_id=build_id,
            canonical_ids=canonical_ids,
            exclusions=exclusions,
            artifact_id=artifact_ids.get("excluded_no_asr_txt"),
        )
        _insert_quality_current(con, canonical_ids=canonical_ids, selected_by_name=selected_by_name)
        _insert_validation_results(con, summary)
        _create_canonical_indexes(con)
        con.commit()

    if db_path.exists():
        db_path.unlink()
    tmp_path.replace(db_path)
    return _validate_written_db(db_path, summary)


def _create_canonical_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        create table canonical_builds (
            build_id text primary key,
            schema_version text not null,
            created_at text not null,
            mode text not null,
            project_root text not null,
            source_dir text not null,
            included_dbs_tsv text not null,
            excluded_no_asr_txt text,
            date_start text not null,
            date_end text not null,
            summary_json text not null,
            validation_passed integer not null
        );

        create table source_artifacts (
            artifact_id integer primary key autoincrement,
            build_id text not null,
            artifact_type text not null,
            path text not null,
            size_bytes integer,
            mtime text,
            row_count integer,
            status_counts_json text,
            notes text
        );

        create table canonical_calls (
            canonical_call_id integer primary key autoincrement,
            build_id text not null,
            source_filename text not null unique,
            source_file text not null,
            month text,
            started_at text,
            audio_size_bytes integer,
            audio_mtime text,
            is_actionable integer not null,
            excluded_reason text,
            canonical_status text not null,
            selected_source_db text,
            selected_call_record_id integer,
            source_call_id text,
            phone text,
            manager_name text,
            duration_sec real,
            direction text,
            transcription_status text,
            resolve_status text,
            analysis_status text,
            sync_status text,
            dead_letter_stage text,
            transcript_text text,
            transcript_manager text,
            transcript_client text,
            transcript_variants_json text,
            resolve_json text,
            resolve_quality_score real,
            analysis_json text,
            amocrm_contact_id integer,
            amocrm_lead_id integer,
            last_error text,
            selected_updated_at text,
            has_transcript_text integer not null,
            has_transcript_variants_json integer not null,
            has_resolve_json integer not null,
            has_analysis_json integer not null,
            transcript_chars integer not null,
            analysis_json_chars integer not null,
            candidate_count integer not null,
            created_at text not null
        );

        create table call_record_provenance (
            provenance_id integer primary key autoincrement,
            canonical_call_id integer not null,
            build_id text not null,
            source_filename text not null,
            source_db text not null,
            source_db_abs text,
            source_row_id integer,
            source_file text,
            source_updated_at text,
            merge_role text not null,
            rank_json text not null,
            transcription_status text,
            resolve_status text,
            analysis_status text,
            sync_status text,
            is_asr_done integer not null,
            is_full_ra integer not null,
            transcript_chars integer not null,
            analysis_json_chars integer not null
        );

        create table call_exclusions (
            exclusion_id integer primary key autoincrement,
            canonical_call_id integer not null,
            build_id text not null,
            source_filename text not null,
            exclusion_type text not null,
            exclusion_status text not null,
            is_actionable integer not null,
            asr_required integer not null,
            reason_code text not null,
            source_artifact_id integer,
            policy_version text not null,
            notes text
        );

        create table call_quality_current (
            canonical_call_id integer primary key,
            call_type text,
            needs_review integer,
            review_reasons_json text,
            resolve_quality_score real,
            quality_flags_json text,
            transcript_quality_label text,
            transcript_quality_score integer,
            transcript_quality_reason_codes_json text,
            protected_live_dialogue integer,
            recommended_call_type text,
            recommended_contact_subtype text,
            quality_status text,
            updated_at text
        );

        create table validation_results (
            check_name text primary key,
            passed integer not null,
            expected text,
            actual text
        );
        """
    )


def _insert_build_row(
    con: sqlite3.Connection,
    *,
    build_id: str,
    summary: dict[str, Any],
    config: CanonicalMasterConfig,
    project_root: Path,
    source_dir: Path,
) -> None:
    con.execute(
        """
        insert into canonical_builds (
            build_id, schema_version, created_at, mode, project_root, source_dir,
            included_dbs_tsv, excluded_no_asr_txt, date_start, date_end,
            summary_json, validation_passed
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            build_id,
            "canonical_master_v1",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "write",
            str(project_root),
            str(source_dir),
            str(_resolve_under_project(config.included_dbs_tsv, project_root)),
            str(_resolve_under_project(config.excluded_no_asr_txt, project_root)) if config.excluded_no_asr_txt else "",
            config.start_date.isoformat(),
            config.end_date.isoformat(),
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
            1 if (summary.get("validation") or {}).get("passed") else 0,
        ),
    )


def _insert_source_artifacts(
    con: sqlite3.Connection,
    *,
    build_id: str,
    project_root: Path,
    source_dir: Path,
    config: CanonicalMasterConfig,
    db_paths: list[Path],
    db_rows: list[dict[str, Any]],
) -> dict[str, int]:
    artifact_ids: dict[str, int] = {}

    def insert_one(kind: str, path: Path, *, row_count: int | None = None, status: dict[str, Any] | None = None, notes: str = "") -> int:
        stat = path.stat() if path.exists() else None
        cur = con.execute(
            """
            insert into source_artifacts (
                build_id, artifact_type, path, size_bytes, mtime, row_count,
                status_counts_json, notes
            ) values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                build_id,
                kind,
                _rel(path, project_root),
                stat.st_size if stat else None,
                datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds") if stat else "",
                row_count,
                json.dumps(status or {}, ensure_ascii=False, sort_keys=True),
                notes,
            ),
        )
        return int(cur.lastrowid)

    artifact_ids["source_dir"] = insert_one("audio_dir", source_dir, notes="Canonical audio universe")
    artifact_ids["included_dbs_tsv"] = insert_one("coverage_included_dbs", _resolve_under_project(config.included_dbs_tsv, project_root))
    if config.excluded_no_asr_txt:
        artifact_ids["excluded_no_asr_txt"] = insert_one("excluded_no_asr_txt", _resolve_under_project(config.excluded_no_asr_txt, project_root))
    db_stats_by_path = {row["db"]: row for row in db_rows}
    for db_path in db_paths:
        rel = _rel(db_path, project_root)
        artifact_ids[f"db::{rel}"] = insert_one(
            "input_db",
            db_path,
            row_count=int(db_stats_by_path.get(rel, {}).get("rows") or 0),
            status=db_stats_by_path.get(rel, {}),
        )
    return artifact_ids


def _insert_canonical_calls(
    con: sqlite3.Connection,
    *,
    build_id: str,
    sources: dict[str, dict[str, Any]],
    exclusions: set[str],
    selected_by_name: dict[str, dict[str, Any] | None],
    all_candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    canonical_ids: dict[str, int] = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for name in sorted(sources):
        source = sources[name]
        selected = selected_by_name.get(name) or {}
        excluded = name in exclusions
        status = _canonical_status(selected or None, excluded=excluded)
        rows.append(
            (
                build_id,
                name,
                source["source_file"],
                source["month"],
                source["started_at"],
                source["audio_size_bytes"],
                source["audio_mtime"],
                0 if excluded else 1,
                "manager_manager_no_asr" if excluded else "",
                status,
                _clean(selected.get("provenance_db")),
                _safe_int(selected.get("id")),
                _clean(selected.get("source_call_id") or source.get("source_call_id_from_filename")),
                _clean(selected.get("phone") or source.get("phone_from_filename")),
                _clean(selected.get("manager_name") or source.get("manager_from_filename")),
                _safe_float(selected.get("duration_sec")),
                _clean(selected.get("direction")),
                _norm(selected.get("transcription_status")),
                _norm(selected.get("resolve_status")),
                _norm(selected.get("analysis_status")),
                _norm(selected.get("sync_status")),
                _clean(selected.get("dead_letter_stage")),
                _clean(selected.get("transcript_text")),
                _clean(selected.get("transcript_manager")),
                _clean(selected.get("transcript_client")),
                _clean(selected.get("transcript_variants_json")),
                _clean(selected.get("resolve_json")),
                _safe_float(selected.get("resolve_quality_score")),
                _clean(selected.get("analysis_json")),
                _safe_int(selected.get("amocrm_contact_id")),
                _safe_int(selected.get("amocrm_lead_id")),
                _clean(selected.get("last_error")),
                _clean(selected.get("updated_at")),
                1 if _clean(selected.get("transcript_text")) else 0,
                1 if _clean(selected.get("transcript_variants_json")) else 0,
                1 if _clean(selected.get("resolve_json")) else 0,
                1 if _clean(selected.get("analysis_json")) else 0,
                sum(len(_clean(selected.get(key))) for key in ("transcript_text", "transcript_manager", "transcript_client")),
                len(_clean(selected.get("analysis_json"))),
                len(all_candidates.get(name, [])),
                now,
            )
        )
    con.executemany(
        """
        insert into canonical_calls (
            build_id, source_filename, source_file, month, started_at, audio_size_bytes,
            audio_mtime, is_actionable, excluded_reason, canonical_status, selected_source_db,
            selected_call_record_id, source_call_id, phone, manager_name, duration_sec,
            direction, transcription_status, resolve_status, analysis_status, sync_status,
            dead_letter_stage, transcript_text, transcript_manager, transcript_client,
            transcript_variants_json, resolve_json, resolve_quality_score, analysis_json,
            amocrm_contact_id, amocrm_lead_id, last_error, selected_updated_at,
            has_transcript_text, has_transcript_variants_json, has_resolve_json,
            has_analysis_json, transcript_chars, analysis_json_chars, candidate_count,
            created_at
        ) values (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )
    for row in con.execute("select canonical_call_id, source_filename from canonical_calls"):
        canonical_ids[str(row[1])] = int(row[0])
    return canonical_ids


def _insert_provenance(
    con: sqlite3.Connection,
    *,
    build_id: str,
    canonical_ids: dict[str, int],
    all_candidates: dict[str, list[dict[str, Any]]],
    selected_by_name: dict[str, dict[str, Any] | None],
    artifact_ids: dict[str, int],
) -> None:
    rows = []
    for name, candidates in all_candidates.items():
        canonical_id = canonical_ids.get(name)
        if not canonical_id:
            continue
        selected = selected_by_name.get(name) or {}
        selected_key = (_clean(selected.get("provenance_db")), _clean(selected.get("id")))
        for candidate in sorted(candidates, key=_rank_score, reverse=True):
            candidate_key = (_clean(candidate.get("provenance_db")), _clean(candidate.get("id")))
            transcript_chars = sum(len(_clean(candidate.get(key))) for key in ("transcript_text", "transcript_manager", "transcript_client"))
            rows.append(
                (
                    canonical_id,
                    build_id,
                    name,
                    _clean(candidate.get("provenance_db")),
                    _clean(candidate.get("provenance_db_abs")),
                    _safe_int(candidate.get("id")),
                    _clean(candidate.get("source_file")),
                    _clean(candidate.get("updated_at")),
                    "selected_primary" if candidate_key == selected_key else "candidate_lost",
                    json.dumps(list(_rank_score(candidate)), ensure_ascii=False),
                    _norm(candidate.get("transcription_status")),
                    _norm(candidate.get("resolve_status")),
                    _norm(candidate.get("analysis_status")),
                    _norm(candidate.get("sync_status")),
                    1 if _is_asr_done(candidate) else 0,
                    1 if _is_full_ra(candidate) else 0,
                    transcript_chars,
                    len(_clean(candidate.get("analysis_json"))),
                )
            )
    con.executemany(
        """
        insert into call_record_provenance (
            canonical_call_id, build_id, source_filename, source_db, source_db_abs,
            source_row_id, source_file, source_updated_at, merge_role, rank_json,
            transcription_status, resolve_status, analysis_status, sync_status,
            is_asr_done, is_full_ra, transcript_chars, analysis_json_chars
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_exclusions(
    con: sqlite3.Connection,
    *,
    build_id: str,
    canonical_ids: dict[str, int],
    exclusions: set[str],
    artifact_id: int | None,
) -> None:
    rows = [
        (
            canonical_ids[name],
            build_id,
            name,
            "manager_manager",
            "accepted",
            0,
            0,
            "manager_manager_no_asr",
            artifact_id,
            "excluded_no_asr_v1",
            "Internal manager-manager call excluded from ASR/R+A coverage gaps.",
        )
        for name in sorted(exclusions)
        if name in canonical_ids
    ]
    con.executemany(
        """
        insert into call_exclusions (
            canonical_call_id, build_id, source_filename, exclusion_type,
            exclusion_status, is_actionable, asr_required, reason_code,
            source_artifact_id, policy_version, notes
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_quality_current(
    con: sqlite3.Connection,
    *,
    canonical_ids: dict[str, int],
    selected_by_name: dict[str, dict[str, Any] | None],
) -> None:
    rows = []
    for name, selected in selected_by_name.items():
        if not selected or name not in canonical_ids:
            continue
        analysis = _safe_json_object(selected.get("analysis_json"))
        quality_flags = _safe_dict(analysis.get("quality_flags"))
        review_reasons = analysis.get("review_reasons")
        tq = _safe_dict(quality_flags.get("transcript_quality_guardrails"))
        rows.append(
            (
                canonical_ids[name],
                _clean(quality_flags.get("call_type")),
                1 if _truthy(analysis.get("needs_review")) else 0,
                json.dumps(review_reasons if isinstance(review_reasons, list) else [], ensure_ascii=False),
                _safe_float(selected.get("resolve_quality_score")),
                json.dumps(quality_flags, ensure_ascii=False, sort_keys=True),
                _clean(quality_flags.get("transcript_quality_label") or tq.get("label")),
                _safe_int(quality_flags.get("transcript_quality_score") or tq.get("score")),
                json.dumps(quality_flags.get("transcript_quality_reason_codes") or tq.get("reason_codes") or [], ensure_ascii=False),
            1 if _truthy(quality_flags.get("transcript_quality_protected_live_dialogue") or tq.get("protected_live_dialogue")) else 0,
                _clean(quality_flags.get("transcript_quality_recommended_call_type") or tq.get("recommended_call_type")),
                _clean(quality_flags.get("transcript_quality_recommended_contact_subtype") or tq.get("recommended_contact_subtype")),
                _clean(quality_flags.get("quality_status") or ("needs_review" if _truthy(analysis.get("needs_review")) else "accepted")),
                _clean(selected.get("updated_at")),
            )
        )
    con.executemany(
        """
        insert into call_quality_current (
            canonical_call_id, call_type, needs_review, review_reasons_json,
            resolve_quality_score, quality_flags_json, transcript_quality_label,
            transcript_quality_score, transcript_quality_reason_codes_json,
            protected_live_dialogue, recommended_call_type,
            recommended_contact_subtype, quality_status, updated_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_validation_results(con: sqlite3.Connection, summary: dict[str, Any]) -> None:
    validation = summary.get("validation") if isinstance(summary.get("validation"), dict) else {}
    checks = validation.get("checks") if isinstance(validation.get("checks"), dict) else {}
    expected = validation.get("expected") if isinstance(validation.get("expected"), dict) else {}
    actual = {
        "source_audio": summary.get("source_audio"),
        "excluded_no_asr": summary.get("excluded_no_asr"),
        "actionable_source_audio": summary.get("actionable_source_audio"),
        "asr_done_actionable": summary.get("asr_done_actionable"),
        "full_ra_actionable": summary.get("full_ra_actionable"),
        "missing_asr_actionable": summary.get("missing_asr_actionable"),
        "missing_full_ra_actionable": summary.get("missing_full_ra_actionable"),
        "errors": summary.get("errors"),
    }
    rows = [
        (
            name,
            1 if passed else 0,
            json.dumps(expected, ensure_ascii=False, sort_keys=True),
            json.dumps(actual, ensure_ascii=False, sort_keys=True),
        )
        for name, passed in checks.items()
    ]
    con.executemany(
        "insert into validation_results (check_name, passed, expected, actual) values (?, ?, ?, ?)",
        rows,
    )


def _create_canonical_indexes(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        create index idx_canonical_calls_source_filename on canonical_calls(source_filename);
        create index idx_canonical_calls_status on canonical_calls(canonical_status);
        create index idx_canonical_calls_phone on canonical_calls(phone);
        create index idx_canonical_calls_month on canonical_calls(month);
        create index idx_provenance_call on call_record_provenance(canonical_call_id);
        create index idx_provenance_source on call_record_provenance(source_filename);
        create index idx_provenance_role on call_record_provenance(merge_role);
        create index idx_quality_call_type on call_quality_current(call_type);
        """
    )


def _validate_written_db(db_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        counts = {
            "canonical_calls": con.execute("select count(*) from canonical_calls").fetchone()[0],
            "full_ra": con.execute("select count(*) from canonical_calls where canonical_status = 'full_ra'").fetchone()[0],
            "excluded": con.execute("select count(*) from canonical_calls where canonical_status = 'excluded_manager_manager_no_asr'").fetchone()[0],
            "provenance_rows": con.execute("select count(*) from call_record_provenance").fetchone()[0],
            "selected_primary_rows": con.execute("select count(*) from call_record_provenance where merge_role = 'selected_primary'").fetchone()[0],
            "candidate_lost_rows": con.execute("select count(*) from call_record_provenance where merge_role = 'candidate_lost'").fetchone()[0],
            "exclusion_rows": con.execute("select count(*) from call_exclusions").fetchone()[0],
            "source_artifacts": con.execute("select count(*) from source_artifacts").fetchone()[0],
            "validation_failed": con.execute("select count(*) from validation_results where passed = 0").fetchone()[0],
        }
    checks = {
        "canonical_calls_match_source_audio": counts["canonical_calls"] == summary["source_audio"],
        "full_ra_matches_summary": counts["full_ra"] == summary["full_ra_actionable"],
        "excluded_matches_summary": counts["excluded"] == summary["excluded_no_asr"],
        "selected_primary_matches_actionable_records": counts["selected_primary_rows"] == summary["actionable_source_audio"],
        "exclusions_match_summary": counts["exclusion_rows"] == summary["excluded_no_asr"],
        "validation_results_all_passed": counts["validation_failed"] == 0,
    }
    return {
        "path": str(db_path),
        "size_bytes": db_path.stat().st_size,
        "counts": counts,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _read_included_dbs(path: Path, project_root: Path) -> list[Path]:
    dbs: list[Path] = []
    seen: set[Path] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            raw = _clean(row.get("db"))
            if not raw:
                continue
            db_path = _resolve_under_project(Path(raw), project_root)
            if db_path in seen:
                continue
            seen.add(db_path)
            if db_path.exists():
                dbs.append(db_path)
    return dbs


def _read_name_list(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _scan_db(
    db_path: Path,
    *,
    db_index: int,
    source_names: set[str],
    project_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {
        "db": _rel(db_path, project_root),
        "rows": 0,
        "source_hits": 0,
        "asr_hits": 0,
        "full_ra_hits": 0,
        "manual_hits": 0,
        "selected_possible": 0,
        "error": "",
    }
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as con:
        con.row_factory = sqlite3.Row
        if not _has_call_records(con):
            return found, stats
        columns = _table_columns(con, "call_records")
        select_columns = [column for column in CALL_RECORD_COLUMNS if column in columns]
        if "source_filename" not in select_columns:
            stats["error"] = "missing source_filename column"
            return found, stats
        sql = f"select {', '.join(select_columns)} from call_records where source_filename is not null and source_filename != ''"
        for row in con.execute(sql):
            stats["rows"] += 1
            payload = {column: row[column] if column in row.keys() else None for column in select_columns}
            name = _clean(payload.get("source_filename"))
            if name not in source_names:
                continue
            stats["source_hits"] += 1
            payload["provenance_db"] = _rel(db_path, project_root)
            payload["provenance_db_abs"] = str(db_path)
            payload["provenance_db_index"] = db_index
            payload["is_asr_done"] = _is_asr_done(payload)
            payload["is_full_ra"] = _is_full_ra(payload)
            payload["is_manual_not_full_ra"] = _is_manual(payload) and not _is_full_ra(payload)
            payload["rank_score"] = _rank_score(payload)
            if payload["is_asr_done"]:
                stats["asr_hits"] += 1
            if payload["is_full_ra"]:
                stats["full_ra_hits"] += 1
            if payload["is_manual_not_full_ra"]:
                stats["manual_hits"] += 1
            found[name].append(payload)
    stats["selected_possible"] = len(found)
    return found, stats


def _select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(candidates, key=_rank_score, reverse=True)[0]


def _rank_score(row: dict[str, Any]) -> tuple[Any, ...]:
    transcript_len = sum(len(_clean(row.get(key))) for key in ("transcript_text", "transcript_manager", "transcript_client"))
    return (
        1 if _is_full_ra(row) else 0,
        1 if _is_asr_done(row) else 0,
        1 if _clean(row.get("analysis_json")) else 0,
        1 if _clean(row.get("resolve_json")) else 0,
        transcript_len,
        _parse_dt_sort_key(row.get("updated_at")),
        -int(row.get("provenance_db_index") or 0),
        -int(row.get("id") or 0),
    )


def _preview_row(source: dict[str, Any], selected: dict[str, Any] | None, *, status: str, excluded: bool) -> dict[str, Any]:
    selected = selected or {}
    return {
        "source_filename": source["source_filename"],
        "source_file": source["source_file"],
        "month": source["month"],
        "started_at": source["started_at"],
        "audio_size_bytes": source["audio_size_bytes"],
        "audio_mtime": source["audio_mtime"],
        "is_actionable": "false" if excluded else "true",
        "excluded_reason": "manager_manager_no_asr" if excluded else "",
        "canonical_status": status,
        "canonical_db": _clean(selected.get("provenance_db")),
        "canonical_call_record_id": _clean(selected.get("id")),
        "source_call_id": _clean(selected.get("source_call_id") or source.get("source_call_id_from_filename")),
        "phone": _clean(selected.get("phone") or source.get("phone_from_filename")),
        "manager_name": _clean(selected.get("manager_name") or source.get("manager_from_filename")),
        "duration_sec": _clean(selected.get("duration_sec")),
        "direction": _clean(selected.get("direction")),
        "transcription_status": _norm(selected.get("transcription_status")),
        "resolve_status": _norm(selected.get("resolve_status")),
        "analysis_status": _norm(selected.get("analysis_status")),
        "sync_status": _norm(selected.get("sync_status")),
        "dead_letter_stage": _clean(selected.get("dead_letter_stage")),
        "is_full_ra": "true" if _is_full_ra(selected) else "false",
        "is_manual_not_full_ra": "true" if selected and _is_manual(selected) and not _is_full_ra(selected) else "false",
        "has_transcript_text": "true" if _clean(selected.get("transcript_text")) else "false",
        "has_transcript_variants_json": "true" if _clean(selected.get("transcript_variants_json")) else "false",
        "has_resolve_json": "true" if _clean(selected.get("resolve_json")) else "false",
        "has_analysis_json": "true" if _clean(selected.get("analysis_json")) else "false",
        "transcript_chars": sum(len(_clean(selected.get(key))) for key in ("transcript_text", "transcript_manager", "transcript_client")),
        "analysis_json_chars": len(_clean(selected.get("analysis_json"))),
        "updated_at": _clean(selected.get("updated_at")),
    }


def _conflict_row(source: dict[str, Any], candidates: list[dict[str, Any]], *, project_root: Path) -> dict[str, Any]:
    ranked = sorted(candidates, key=_rank_score, reverse=True)
    selected = ranked[0]
    return {
        "source_filename": source["source_filename"],
        "candidate_count": len(candidates),
        "selected_db": _clean(selected.get("provenance_db")),
        "selected_id": _clean(selected.get("id")),
        "selected_full_ra": "true" if _is_full_ra(selected) else "false",
        "candidate_dbs": " | ".join(_clean(row.get("provenance_db")) for row in ranked),
        "candidate_ids": " | ".join(_clean(row.get("id")) for row in ranked),
        "candidate_statuses": " | ".join(
            f"{_norm(row.get('transcription_status'))}/{_norm(row.get('resolve_status'))}/{_norm(row.get('analysis_status'))}"
            for row in ranked
        ),
    }


def _canonical_status(selected: dict[str, Any] | None, *, excluded: bool) -> str:
    if excluded:
        return "excluded_manager_manager_no_asr"
    if not selected:
        return "missing"
    if _is_full_ra(selected):
        return "full_ra"
    if _is_asr_done(selected):
        return "asr_only_or_manual"
    return "not_processed"


def _coverage_rows(by_month: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    fields = [
        "month",
        "source_audio",
        "excluded_no_asr",
        "actionable_source_audio",
        "asr_done",
        "full_ra",
        "missing",
        "not_processed",
        "asr_only_or_manual",
        "excluded_manager_manager_no_asr",
    ]
    rows = []
    totals = Counter()
    for month in sorted(by_month):
        row = {"month": month}
        for field in fields:
            if field == "month":
                continue
            value = by_month[month].get(field, 0)
            row[field] = value
            totals[field] += value
        rows.append(row)
    total_row = {"month": "TOTAL"}
    total_row.update({field: totals.get(field, 0) for field in fields if field != "month"})
    rows.append(total_row)
    return rows


def _validation(summary: dict[str, Any], config: CanonicalMasterConfig) -> dict[str, Any]:
    expected_source_audio = config.expected_source_audio
    expected_excluded = config.expected_excluded_no_asr
    expected_actionable = config.expected_actionable_source_audio
    expected_asr = config.expected_asr_done_actionable
    expected_full_ra = config.expected_full_ra_actionable
    checks = {
        "source_audio_matches_expected": (
            expected_source_audio is None or summary["source_audio"] == expected_source_audio
        ),
        "excluded_no_asr_matches_expected": (
            expected_excluded is None or summary["excluded_no_asr"] == expected_excluded
        ),
        "actionable_matches_expected": (
            expected_actionable is None or summary["actionable_source_audio"] == expected_actionable
        ),
        "asr_done_actionable_matches_expected": (
            expected_asr is None or summary["asr_done_actionable"] == expected_asr
        ),
        "full_ra_actionable_matches_expected": (
            expected_full_ra is None or summary["full_ra_actionable"] == expected_full_ra
        ),
        "no_missing_asr_actionable": summary["missing_asr_actionable"] == 0,
        "no_missing_full_ra_actionable": summary["missing_full_ra_actionable"] == 0,
        "no_scan_errors": not summary["errors"],
    }
    return {
        "expected": {
            "source_audio": expected_source_audio,
            "excluded_no_asr": expected_excluded,
            "actionable_source_audio": expected_actionable,
            "asr_done_actionable": expected_asr,
            "full_ra_actionable": expected_full_ra,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _is_asr_done(row: dict[str, Any]) -> bool:
    return _norm(row.get("transcription_status")) == "done"


def _is_full_ra(row: dict[str, Any]) -> bool:
    return (
        _is_asr_done(row)
        and _norm(row.get("resolve_status")) in TERMINAL_RESOLVE
        and _norm(row.get("analysis_status")) == "done"
    )


def _is_manual(row: dict[str, Any]) -> bool:
    return _is_asr_done(row) and _norm(row.get("resolve_status")) == "manual"


def _has_call_records(con: sqlite3.Connection) -> bool:
    return bool(con.execute("select 1 from sqlite_master where type='table' and name='call_records'").fetchone())


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"pragma table_info({table})")}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    validation = summary.get("validation") or {}
    return (
        "# Canonical master dry-run\n\n"
        "Read-only preview. Existing DBs and audio files were not modified.\n\n"
        f"- Source audio: `{summary['source_audio']}`\n"
        f"- Excluded manager-manager/no-ASR: `{summary['excluded_no_asr']}`\n"
        f"- Actionable audio: `{summary['actionable_source_audio']}`\n"
        f"- ASR done actionable: `{summary['asr_done_actionable']}`\n"
        f"- Full R+A actionable: `{summary['full_ra_actionable']}`\n"
        f"- Duplicate source names with DB candidates: `{summary['duplicate_source_names_with_candidates']}`\n"
        f"- Validation passed: `{validation.get('passed')}`\n\n"
        "Next safe step: inspect conflicts and only then build/write the actual canonical DB.\n"
    )


def _resolve_under_project(path: Path | None, project_root: Path) -> Path:
    if path is None:
        raise ValueError("Path is required")
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _norm(value: Any) -> str:
    return _clean(value).lower()


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _parse_dt_sort_key(value: Any) -> float:
    raw = _clean(value).replace("Z", "+00:00")
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0


def _safe_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "y", "да"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only canonical master preview from distributed call DBs.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-dir", type=Path, default=Path("product_data/audio_working_store_20260523_v1/by_filename"))
    parser.add_argument("--included-dbs-tsv", type=Path, required=True)
    parser.add_argument("--excluded-no-asr-txt", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-05-31")
    parser.add_argument("--mode", choices=["dry-run", "write"], default="dry-run")
    parser.add_argument("--canonical-db-name", default="canonical_calls_master.db")
    parser.add_argument("--expected-source-audio", type=int, default=64867)
    parser.add_argument("--expected-excluded-no-asr", type=int, default=35)
    parser.add_argument("--expected-actionable-source-audio", type=int, default=64832)
    parser.add_argument("--expected-asr-done-actionable", type=int, default=64832)
    parser.add_argument("--expected-full-ra-actionable", type=int, default=64832)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> CanonicalMasterConfig:
    return CanonicalMasterConfig(
        project_root=args.project_root,
        source_dir=args.source_dir,
        included_dbs_tsv=args.included_dbs_tsv,
        excluded_no_asr_txt=args.excluded_no_asr_txt,
        out_root=args.out_root,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        mode=args.mode,
        canonical_db_name=args.canonical_db_name,
        expected_source_audio=args.expected_source_audio,
        expected_excluded_no_asr=args.expected_excluded_no_asr,
        expected_actionable_source_audio=args.expected_actionable_source_audio,
        expected_asr_done_actionable=args.expected_asr_done_actionable,
        expected_full_ra_actionable=args.expected_full_ra_actionable,
    )
