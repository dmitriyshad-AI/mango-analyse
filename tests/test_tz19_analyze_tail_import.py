from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.import_tz19_analyze_tail_results import ImportConfig, import_tail_results


PROMPT_SHA = "p" * 64


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    db, manifest, blacklist, results = seed_import_case(tmp_path, [1], [result_row(1)])

    summary = import_tail_results(ImportConfig(db, manifest, blacklist, (results,), expect_prompt_sha256=PROMPT_SHA))

    assert summary["mode"] == "dry_run"
    assert summary["counters"]["updated"] == 1
    row = fetch_call(db, 1)
    assert json.loads(row["analysis_json"])["analysis_meta"]["analysis_prompt_version"] == "v6"


def test_apply_requires_backup_to(tmp_path: Path) -> None:
    db, manifest, blacklist, results = seed_import_case(tmp_path, [1], [result_row(1)])

    with pytest.raises(RuntimeError, match="requires --backup-to"):
        import_tail_results(
            ImportConfig(db, manifest, blacklist, (results,), apply=True, expect_prompt_sha256=PROMPT_SHA)
        )


def test_apply_updates_only_whitelist_and_second_run_is_idempotent(tmp_path: Path) -> None:
    db, manifest, blacklist, results = seed_import_case(tmp_path, [1], [result_row(1)])
    backup = tmp_path / "backup.sqlite"

    first = import_tail_results(
        ImportConfig(db, manifest, blacklist, (results,), apply=True, backup_to=backup, expect_prompt_sha256=PROMPT_SHA)
    )
    second = import_tail_results(
        ImportConfig(db, manifest, blacklist, (results,), expect_prompt_sha256=PROMPT_SHA)
    )

    assert backup.exists()
    assert first["counters"]["updated"] == 1
    assert second["counters"]["updated"] == 0
    assert second["counters"]["skipped_same"] == 1
    row = fetch_call(db, 1)
    assert row["analysis_status"] == "done"
    assert row["has_analysis_json"] == 1
    assert row["analysis_json_chars"] == len(result_payload())
    assert row["last_error"] is None
    assert row["phone"] == "+79990000001"
    assert row["transcript_chars"] == 123


def test_rejects_outside_manifest_blacklist_bad_rows_and_transcript_mismatch(tmp_path: Path) -> None:
    rows = [
        result_row(1),
        result_row(2),
        result_row(3, status="failed"),
        result_row(4, payload="{bad"),
        result_row(5, payload=json.dumps({"analysis_schema_version": "v1", "analysis_meta": {"analysis_prompt_version": "v7", "analysis_model": "gpt-5.4-mini"}})),
        result_row(6, payload=json.dumps({"analysis_schema_version": "v2", "analysis_meta": {"analysis_prompt_version": "v6", "analysis_model": "gpt-5.4-mini"}})),
        result_row(7, transcript_chars=999),
        result_row(999),
    ]
    db, manifest, blacklist, results = seed_import_case(
        tmp_path,
        [1, 3, 4, 5, 6, 7],
        rows,
        blacklist_ids=[2],
    )

    summary = import_tail_results(ImportConfig(db, manifest, blacklist, (results,), expect_prompt_sha256=PROMPT_SHA))

    assert summary["counters"]["updated"] == 1
    assert summary["counters"]["rejected_blacklist"] == 1
    assert summary["counters"]["rejected_not_done"] == 1
    assert summary["counters"]["rejected_bad_json"] == 2
    assert summary["counters"]["rejected_meta"] == 1
    assert summary["counters"]["rejected_transcript_changed"] == 1
    assert summary["counters"]["rejected_not_in_manifest"] == 1


def test_rejects_duplicate_conflict_without_writing(tmp_path: Path) -> None:
    conflict_payload = json.dumps(
        {
            "analysis_schema_version": "v2",
            "analysis_meta": {"analysis_prompt_version": "v7", "analysis_model": "gpt-5.4-mini"},
            "history_summary": "other",
        },
        ensure_ascii=False,
    )
    db, manifest, blacklist, results = seed_import_case(
        tmp_path,
        [1],
        [result_row(1), result_row(1, payload=conflict_payload)],
    )

    summary = import_tail_results(ImportConfig(db, manifest, blacklist, (results,), expect_prompt_sha256=PROMPT_SHA))

    assert summary["counters"]["rejected_duplicate_conflict"] == 1
    assert summary["counters"]["updated"] == 0
    assert json.loads(fetch_call(db, 1)["analysis_json"])["analysis_meta"]["analysis_prompt_version"] == "v6"


def test_same_duplicate_is_counted_once(tmp_path: Path) -> None:
    row = result_row(1)
    db, manifest, blacklist, results = seed_import_case(tmp_path, [1], [row, row])

    summary = import_tail_results(ImportConfig(db, manifest, blacklist, (results,), expect_prompt_sha256=PROMPT_SHA))

    assert summary["counters"]["skipped_duplicate_same"] == 1
    assert summary["counters"]["updated"] == 1


def test_manifest_prompt_sha_mismatch_blocks_before_db_write(tmp_path: Path) -> None:
    db, manifest, blacklist, results = seed_import_case(tmp_path, [1], [result_row(1)])

    with pytest.raises(RuntimeError, match="prompt_sha256 mismatch"):
        import_tail_results(ImportConfig(db, manifest, blacklist, (results,), expect_prompt_sha256="wrong"))


def seed_import_case(
    tmp_path: Path,
    manifest_ids: list[int],
    result_rows: list[dict],
    *,
    blacklist_ids: list[int] | None = None,
) -> tuple[Path, Path, Path, Path]:
    db = tmp_path / "calls.sqlite"
    manifest = tmp_path / "manifest.json"
    blacklist = tmp_path / "blacklist_77.txt"
    results = tmp_path / "results.jsonl.gz"
    seed_db(db, sorted(set(manifest_ids + [999])))
    manifest.write_text(
        json.dumps(
            {
                "rows": len(manifest_ids),
                "prompt_version": "v7",
                "prompt_sha256": PROMPT_SHA,
                "calls": [{"canonical_call_id": cid} for cid in manifest_ids],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    blacklist.write_text("\n".join(str(cid) for cid in (blacklist_ids or [])) + "\n", encoding="utf-8")
    with gzip.open(results, "wt", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return db, manifest, blacklist, results


def seed_db(path: Path, call_ids: list[int]) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE canonical_calls (
            canonical_call_id INTEGER PRIMARY KEY,
            phone TEXT,
            analysis_json TEXT,
            analysis_status TEXT,
            analysis_json_chars INTEGER,
            has_analysis_json INTEGER,
            last_error TEXT,
            transcript_chars INTEGER
        );
        """
    )
    old_payload = json.dumps(
        {"analysis_schema_version": "v2", "analysis_meta": {"analysis_prompt_version": "v6", "analysis_model": "old"}},
        ensure_ascii=False,
    )
    con.executemany(
        "INSERT INTO canonical_calls VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                cid,
                f"+7999000{cid:04d}",
                old_payload,
                "done",
                len(old_payload),
                1,
                "old error",
                123,
            )
            for cid in call_ids
        ],
    )
    con.commit()
    con.close()


def result_payload() -> str:
    return json.dumps(
        {
            "analysis_schema_version": "v2",
            "analysis_meta": {"analysis_prompt_version": "v7", "analysis_model": "gpt-5.4-mini"},
            "history_summary": "ok",
        },
        ensure_ascii=False,
    )


def result_row(
    call_id: int,
    *,
    status: str = "done",
    payload: str | None = None,
    transcript_chars: int = 123,
) -> dict:
    return {
        "canonical_call_id": call_id,
        "analysis_status": status,
        "analysis_json": result_payload() if payload is None else payload,
        "transcript_chars": transcript_chars,
        "last_error": None,
    }


def fetch_call(db: Path, call_id: int) -> sqlite3.Row:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return con.execute("SELECT * FROM canonical_calls WHERE canonical_call_id=?", (call_id,)).fetchone()
    finally:
        con.close()
