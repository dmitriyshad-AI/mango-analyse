from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from mango_mvp.maintenance.canonical_master import (
    CanonicalMasterConfig,
    build_canonical_master_preview,
)


def _audio(path: Path, name: str) -> None:
    (path / name).write_bytes(b"audio")


def _db(path: Path, rows: list[dict[str, object]]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            create table call_records (
                id integer primary key,
                source_file text,
                source_filename text,
                source_call_id text,
                duration_sec real,
                phone text,
                manager_name text,
                direction text,
                started_at text,
                transcription_status text,
                resolve_status text,
                analysis_status text,
                sync_status text,
                transcript_text text,
                transcript_manager text,
                transcript_client text,
                transcript_variants_json text,
                resolve_json text,
                resolve_quality_score real,
                analysis_json text,
                dead_letter_stage text,
                updated_at text
            )
            """
        )
        for idx, row in enumerate(rows, start=1):
            con.execute(
                """
                insert into call_records (
                    id, source_file, source_filename, source_call_id, duration_sec, phone,
                    manager_name, direction, started_at, transcription_status, resolve_status,
                    analysis_status, sync_status, transcript_text, transcript_manager,
                    transcript_client, transcript_variants_json, resolve_json,
                    resolve_quality_score, analysis_json, dead_letter_stage, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idx,
                    row.get("source_file", f"/tmp/{row['source_filename']}"),
                    row["source_filename"],
                    row.get("source_call_id"),
                    row.get("duration_sec", 10),
                    row.get("phone", "+79160000000"),
                    row.get("manager_name", "Менеджер"),
                    row.get("direction", "outbound"),
                    row.get("started_at", "2025-01-01 10:00:00"),
                    row.get("transcription_status", "done"),
                    row.get("resolve_status", "done"),
                    row.get("analysis_status", "done"),
                    row.get("sync_status", "pending"),
                    row.get("transcript_text", "MANAGER: hello\nCLIENT: hello"),
                    row.get("transcript_manager", ""),
                    row.get("transcript_client", ""),
                    row.get("transcript_variants_json", "{}"),
                    row.get("resolve_json", "{}"),
                    row.get("resolve_quality_score", 1.0),
                    row.get("analysis_json", json.dumps({"quality_flags": {"call_type": "sales_call"}})),
                    row.get("dead_letter_stage"),
                    row.get("updated_at", "2025-01-01 10:00:00"),
                ),
            )
        con.commit()
    finally:
        con.close()


def _write_included(path: Path, dbs: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["db", "rows", "source_hits", "asr_hits", "ra_hits", "manual_hits"], delimiter="\t")
        writer.writeheader()
        for db in dbs:
            writer.writerow({"db": db.name, "rows": 0, "source_hits": 0, "asr_hits": 0, "ra_hits": 0, "manual_hits": 0})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, source_dir: Path, included: Path, out_root: Path, excluded: Path | None = None) -> CanonicalMasterConfig:
    return CanonicalMasterConfig(
        project_root=tmp_path,
        source_dir=source_dir,
        included_dbs_tsv=included,
        excluded_no_asr_txt=excluded,
        out_root=out_root,
        expected_source_audio=3,
        expected_excluded_no_asr=1 if excluded else 0,
        expected_actionable_source_audio=2 if excluded else 3,
        expected_asr_done_actionable=2 if excluded else 3,
        expected_full_ra_actionable=2 if excluded else 3,
    )


def test_canonical_master_preview_selects_full_ra_duplicate_and_keeps_excluded(tmp_path: Path) -> None:
    source_dir = tmp_path / "audio"
    source_dir.mkdir()
    _audio(source_dir, "2025-01-01__10-00-00__Анна__79160000001.mp3")
    _audio(source_dir, "2025-01-01__11-00-00__Анна__79160000002.mp3")
    _audio(source_dir, "2025-01-01__12-00-00__Анна__Олег.mp3")

    db1 = tmp_path / "db1.db"
    db2 = tmp_path / "db2.db"
    _db(
        db1,
        [
            {"source_filename": "2025-01-01__10-00-00__Анна__79160000001.mp3"},
            {
                "source_filename": "2025-01-01__11-00-00__Анна__79160000002.mp3",
                "transcription_status": "done",
                "resolve_status": "pending",
                "analysis_status": "pending",
                "analysis_json": "",
                "updated_at": "2025-01-02 10:00:00",
            },
        ],
    )
    _db(
        db2,
        [
            {
                "source_filename": "2025-01-01__11-00-00__Анна__79160000002.mp3",
                "transcription_status": "done",
                "resolve_status": "done",
                "analysis_status": "done",
                "updated_at": "2025-01-01 10:00:00",
            }
        ],
    )
    included = tmp_path / "included.tsv"
    _write_included(included, [db1, db2])
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("2025-01-01__12-00-00__Анна__Олег.mp3\n", encoding="utf-8")

    summary = build_canonical_master_preview(_config(tmp_path, source_dir, included, tmp_path / "out", excluded))

    assert summary["validation"]["passed"] is True
    assert summary["source_audio"] == 3
    assert summary["excluded_no_asr"] == 1
    assert summary["actionable_source_audio"] == 2
    assert summary["full_ra_actionable"] == 2
    assert summary["duplicate_source_names_with_candidates"] == 1

    rows = list(csv.DictReader((tmp_path / "out" / "canonical_preview.csv").open(encoding="utf-8-sig")))
    by_name = {row["source_filename"]: row for row in rows}
    assert by_name["2025-01-01__11-00-00__Анна__79160000002.mp3"]["canonical_db"] == "db2.db"
    assert by_name["2025-01-01__12-00-00__Анна__Олег.mp3"]["canonical_status"] == "excluded_manager_manager_no_asr"
    coverage = list(csv.DictReader((tmp_path / "out" / "coverage_by_month.tsv").open(encoding="utf-8"), delimiter="\t"))
    total = coverage[-1]
    assert total["month"] == "TOTAL"
    assert total["full_ra"] == "2"


def test_canonical_master_preview_is_read_only_and_reports_missing(tmp_path: Path) -> None:
    source_dir = tmp_path / "audio"
    source_dir.mkdir()
    _audio(source_dir, "2025-01-01__10-00-00__Анна__79160000001.mp3")
    _audio(source_dir, "2025-01-01__11-00-00__Анна__79160000002.mp3")
    _audio(source_dir, "2025-01-01__12-00-00__Анна__79160000003.mp3")

    db1 = tmp_path / "db1.db"
    _db(db1, [{"source_filename": "2025-01-01__10-00-00__Анна__79160000001.mp3"}])
    before = _sha256(db1)
    included = tmp_path / "included.tsv"
    _write_included(included, [db1])

    summary = build_canonical_master_preview(
        CanonicalMasterConfig(
            project_root=tmp_path,
            source_dir=source_dir,
            included_dbs_tsv=included,
            out_root=tmp_path / "out",
            expected_source_audio=3,
            expected_excluded_no_asr=0,
            expected_actionable_source_audio=3,
            expected_asr_done_actionable=3,
            expected_full_ra_actionable=3,
        )
    )

    assert _sha256(db1) == before
    assert summary["validation"]["passed"] is False
    assert summary["missing_asr_actionable"] == 2
    assert summary["missing_full_ra_actionable"] == 2
    assert not (tmp_path / "out" / "canonical_calls_master.db").exists()


def test_canonical_master_write_creates_db_with_provenance_and_exclusions(tmp_path: Path) -> None:
    source_dir = tmp_path / "audio"
    source_dir.mkdir()
    _audio(source_dir, "2025-01-01__10-00-00__Анна__79160000001.mp3")
    _audio(source_dir, "2025-01-01__11-00-00__Анна__79160000002.mp3")
    _audio(source_dir, "2025-01-01__12-00-00__Анна__Олег.mp3")

    db1 = tmp_path / "db1.db"
    db2 = tmp_path / "db2.db"
    _db(
        db1,
        [
            {"source_filename": "2025-01-01__10-00-00__Анна__79160000001.mp3"},
            {
                "source_filename": "2025-01-01__11-00-00__Анна__79160000002.mp3",
                "transcription_status": "done",
                "resolve_status": "pending",
                "analysis_status": "pending",
                "analysis_json": "",
                "updated_at": "2025-01-02 10:00:00",
            },
        ],
    )
    _db(
        db2,
        [
            {
                "source_filename": "2025-01-01__11-00-00__Анна__79160000002.mp3",
                "transcription_status": "done",
                "resolve_status": "done",
                "analysis_status": "done",
                "updated_at": "2025-01-01 10:00:00",
            }
        ],
    )
    db1_before = _sha256(db1)
    db2_before = _sha256(db2)
    included = tmp_path / "included.tsv"
    _write_included(included, [db1, db2])
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("2025-01-01__12-00-00__Анна__Олег.mp3\n", encoding="utf-8")

    out_root = tmp_path / "out"
    summary = build_canonical_master_preview(
        CanonicalMasterConfig(
            project_root=tmp_path,
            source_dir=source_dir,
            included_dbs_tsv=included,
            excluded_no_asr_txt=excluded,
            out_root=out_root,
            mode="write",
            expected_source_audio=3,
            expected_excluded_no_asr=1,
            expected_actionable_source_audio=2,
            expected_asr_done_actionable=2,
            expected_full_ra_actionable=2,
        )
    )

    assert _sha256(db1) == db1_before
    assert _sha256(db2) == db2_before
    assert summary["mode"] == "write"
    assert summary["validation"]["passed"] is True
    assert summary["canonical_db"]["passed"] is True
    canonical_db = out_root / "canonical_calls_master.db"
    assert canonical_db.exists()

    con = sqlite3.connect(canonical_db)
    con.row_factory = sqlite3.Row
    try:
        assert con.execute("select count(*) from canonical_calls").fetchone()[0] == 3
        assert con.execute("select count(distinct source_filename) from canonical_calls").fetchone()[0] == 3
        assert con.execute("select count(*) from canonical_calls where is_actionable = 1").fetchone()[0] == 2
        assert con.execute("select count(*) from canonical_calls where canonical_status = 'full_ra'").fetchone()[0] == 2
        assert con.execute("select count(*) from call_exclusions").fetchone()[0] == 1
        assert con.execute("select count(*) from source_artifacts where artifact_type = 'input_db'").fetchone()[0] == 2

        duplicate = con.execute(
            """
            select selected_source_db, canonical_status
            from canonical_calls
            where source_filename = '2025-01-01__11-00-00__Анна__79160000002.mp3'
            """
        ).fetchone()
        assert duplicate["selected_source_db"] == "db2.db"
        assert duplicate["canonical_status"] == "full_ra"

        assert con.execute(
            """
            select count(*) from call_record_provenance
            where source_filename = '2025-01-01__11-00-00__Анна__79160000002.mp3'
              and merge_role = 'selected_primary'
            """
        ).fetchone()[0] == 1
        assert con.execute(
            """
            select count(*) from call_record_provenance
            where source_filename = '2025-01-01__11-00-00__Анна__79160000002.mp3'
              and merge_role = 'candidate_lost'
            """
        ).fetchone()[0] == 1

        excluded_row = con.execute(
            """
            select canonical_status, is_actionable, excluded_reason, transcription_status
            from canonical_calls
            where source_filename = '2025-01-01__12-00-00__Анна__Олег.mp3'
            """
        ).fetchone()
        assert excluded_row["canonical_status"] == "excluded_manager_manager_no_asr"
        assert excluded_row["is_actionable"] == 0
        assert excluded_row["excluded_reason"] == "manager_manager_no_asr"
        assert excluded_row["transcription_status"] == ""
    finally:
        con.close()
