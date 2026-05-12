from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from mango_mvp.maintenance.project_inventory import (
    ProjectInventoryConfig,
    build_project_inventory,
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_project_inventory_reports_db_files_and_archive_candidates(tmp_path: Path) -> None:
    runtime = tmp_path / "stable_runtime"
    runtime.mkdir()
    candidate = runtime / "jan_asr_only_20260501"
    candidate.mkdir()
    db_path = candidate / "calls.before_requeue.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute("create table call_records (id integer primary key, source_filename text)")
        con.execute("insert into call_records (source_filename) values ('a.mp3')")
        con.commit()
    finally:
        con.close()

    protected = runtime / "ra_missing_all_20260506"
    protected.mkdir()
    (protected / "active.db").write_bytes(b"not sqlite")

    replacement_artifact = "stable_runtime/canonical_master_20260509_v1/canonical_calls_master.db"
    summary = build_project_inventory(
        ProjectInventoryConfig(
            project_root=tmp_path,
            out_root=tmp_path / "inventory",
            replacement_artifact=replacement_artifact,
        )
    )

    assert summary["db_files"] == 2
    assert summary["archive_candidate_rows"] >= 1
    assert summary["replacement_artifact"] == replacement_artifact
    db_rows = _read_tsv(tmp_path / "inventory" / "db_inventory.tsv")
    by_path = {row["path"]: row for row in db_rows}
    assert by_path["stable_runtime/jan_asr_only_20260501/calls.before_requeue.db"]["has_call_records"] == "true"
    assert by_path["stable_runtime/ra_missing_all_20260506/active.db"]["classification"] == "do_not_touch_now"

    archive_rows = _read_tsv(tmp_path / "inventory" / "archive_candidates_dry_run.tsv")
    candidate_row = next(row for row in archive_rows if row["path"] == "stable_runtime/jan_asr_only_20260501")
    assert candidate_row["replacement_artifact"] == replacement_artifact
