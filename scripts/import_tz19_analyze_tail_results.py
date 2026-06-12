#!/usr/bin/env python3
"""Import TZ-19 analyze tail results into canonical calls DB.

Default mode is dry-run. Real write requires --apply and --backup-to.
The script updates only analysis columns and rejects rows outside the
manifest whitelist or inside blacklist_77.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_UPDATE_COLUMNS = (
    "analysis_json",
    "analysis_status",
    "analysis_json_chars",
    "has_analysis_json",
    "last_error",
)
DEFAULT_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/"
    "stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
)
DEFAULT_MANIFEST = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/"
    "analyze_tail_20260612/data/manifest.json"
)
DEFAULT_BLACKLIST = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/"
    "analyze_tail_20260612/blacklist_77.txt"
)
DEFAULT_RESULTS = tuple(
    Path(
        "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/"
        f"analyze_tail_20260612/results_part{idx}.jsonl.gz"
    )
    for idx in range(1, 5)
)


@dataclass(frozen=True)
class ImportConfig:
    db: Path
    manifest: Path
    blacklist: Path
    results: tuple[Path, ...]
    apply: bool = False
    backup_to: Path | None = None
    expect_prompt_version: str = "v7"
    expect_model_substr: str = "mini"
    expect_prompt_sha256: str | None = None
    report_out: Path | None = None


def import_tail_results(config: ImportConfig) -> Mapping[str, Any]:
    cfg = normalize_config(config)
    manifest = read_manifest(cfg.manifest)
    allowed_ids = manifest_call_ids(manifest)
    blacklist_ids = read_int_set(cfg.blacklist)
    if not allowed_ids:
        raise RuntimeError("manifest contains no allowed call ids")
    if allowed_ids & blacklist_ids:
        raise RuntimeError("manifest intersects blacklist")
    if cfg.expect_prompt_sha256 and str(manifest.get("prompt_sha256")) != cfg.expect_prompt_sha256:
        raise RuntimeError("manifest prompt_sha256 mismatch")
    if str(manifest.get("prompt_version", "")) != cfg.expect_prompt_version:
        raise RuntimeError("manifest prompt_version mismatch")
    if cfg.apply and cfg.backup_to is None:
        raise RuntimeError("--apply requires --backup-to")

    records, duplicate_report = load_unique_records(cfg.results)
    if cfg.apply and cfg.backup_to is not None:
        cfg.backup_to.parent.mkdir(parents=True, exist_ok=True)
        if cfg.backup_to.exists():
            raise RuntimeError(f"backup already exists: {cfg.backup_to}")
        shutil.copy2(cfg.db, cfg.backup_to)

    con = sqlite3.connect(cfg.db if cfg.apply else f"file:{cfg.db}?mode=ro", uri=not cfg.apply)
    con.row_factory = sqlite3.Row
    try:
        before = prompt_counts(con, allowed_ids, cfg.expect_prompt_version)
        counters: dict[str, int] = {
            "read": 0,
            "updated": 0,
            "skipped_same": 0,
            "skipped_duplicate_same": duplicate_report["same"],
            "rejected_duplicate_conflict": duplicate_report["conflict"],
            "rejected_not_in_manifest": 0,
            "rejected_blacklist": 0,
            "rejected_not_done": 0,
            "rejected_bad_json": 0,
            "rejected_meta": 0,
            "rejected_missing_row": 0,
            "rejected_transcript_changed": 0,
        }
        rejected_ids: dict[str, list[int]] = {key: list(value) for key, value in duplicate_report["ids"].items()}
        conflict_ids = set(duplicate_report["conflict_ids"])

        for record in records:
            counters["read"] += 1
            cid = int(record.get("canonical_call_id") or 0)
            if cid in conflict_ids:
                add_rejected_id(rejected_ids, "duplicate_conflict", cid)
                continue
            if cid in blacklist_ids:
                counters["rejected_blacklist"] += 1
                add_rejected_id(rejected_ids, "blacklist", cid)
                continue
            if cid not in allowed_ids:
                counters["rejected_not_in_manifest"] += 1
                add_rejected_id(rejected_ids, "not_in_manifest", cid)
                continue
            payload = str(record.get("analysis_json") or "").strip()
            if record.get("analysis_status") != "done" or not payload:
                counters["rejected_not_done"] += 1
                add_rejected_id(rejected_ids, "not_done", cid)
                continue
            try:
                doc = json.loads(payload)
            except json.JSONDecodeError:
                counters["rejected_bad_json"] += 1
                add_rejected_id(rejected_ids, "bad_json", cid)
                continue
            if not isinstance(doc, dict) or doc.get("analysis_schema_version") != "v2":
                counters["rejected_bad_json"] += 1
                add_rejected_id(rejected_ids, "bad_json", cid)
                continue
            meta = doc.get("analysis_meta") if isinstance(doc.get("analysis_meta"), dict) else {}
            if str(meta.get("analysis_prompt_version", "")) != cfg.expect_prompt_version:
                counters["rejected_meta"] += 1
                add_rejected_id(rejected_ids, "meta", cid)
                continue
            if cfg.expect_model_substr not in str(meta.get("analysis_model", "")):
                counters["rejected_meta"] += 1
                add_rejected_id(rejected_ids, "meta", cid)
                continue
            row = con.execute(
                "SELECT analysis_json, transcript_chars FROM canonical_calls WHERE canonical_call_id=?",
                (cid,),
            ).fetchone()
            if row is None:
                counters["rejected_missing_row"] += 1
                add_rejected_id(rejected_ids, "missing_row", cid)
                continue
            expected_chars = record.get("transcript_chars")
            if expected_chars is not None and int(row["transcript_chars"] or 0) != int(expected_chars):
                counters["rejected_transcript_changed"] += 1
                add_rejected_id(rejected_ids, "transcript_changed", cid)
                continue
            if (row["analysis_json"] or "") == payload:
                counters["skipped_same"] += 1
                continue
            if cfg.apply:
                con.execute(
                    """
                    UPDATE canonical_calls
                    SET analysis_json=?,
                        analysis_status='done',
                        analysis_json_chars=?,
                        has_analysis_json=1,
                        last_error=NULL
                    WHERE canonical_call_id=?
                    """,
                    (payload, len(payload), cid),
                )
            counters["updated"] += 1
        if cfg.apply:
            con.commit()
        after = prompt_counts(con, allowed_ids, cfg.expect_prompt_version)
    finally:
        con.close()

    summary = {
        "schema_version": "tz19_tail_import_report_v1",
        "mode": "apply" if cfg.apply else "dry_run",
        "db": str(cfg.db),
        "manifest": str(cfg.manifest),
        "results": [str(path) for path in cfg.results],
        "allowed_ids": len(allowed_ids),
        "blacklist_ids": len(blacklist_ids),
        "allowed_update_columns": list(ALLOWED_UPDATE_COLUMNS),
        "manifest_prompt_sha256": manifest.get("prompt_sha256"),
        "expected_prompt_sha256": cfg.expect_prompt_sha256,
        "before": before,
        "after": after,
        "counters": counters,
        "rejected_ids": {key: values[:50] for key, values in sorted(rejected_ids.items())},
        "safety": {
            "write_crm": False,
            "write_tallanto": False,
            "write_amo": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
    }
    if cfg.report_out:
        cfg.report_out.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def normalize_config(config: ImportConfig) -> ImportConfig:
    return ImportConfig(
        db=config.db.expanduser().resolve(strict=False),
        manifest=config.manifest.expanduser().resolve(strict=False),
        blacklist=config.blacklist.expanduser().resolve(strict=False),
        results=tuple(path.expanduser().resolve(strict=False) for path in config.results),
        apply=config.apply,
        backup_to=config.backup_to.expanduser().resolve(strict=False) if config.backup_to else None,
        expect_prompt_version=config.expect_prompt_version,
        expect_model_substr=config.expect_model_substr,
        expect_prompt_sha256=config.expect_prompt_sha256,
        report_out=config.report_out.expanduser().resolve(strict=False) if config.report_out else None,
    )


def load_unique_records(paths: Sequence[Path]) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    first_by_id: dict[int, Mapping[str, Any]] = {}
    payload_by_id: dict[int, str] = {}
    duplicate_same = 0
    duplicate_conflict = 0
    duplicate_ids: dict[str, list[int]] = {"duplicate_same": [], "duplicate_conflict": []}
    conflict_ids: set[int] = set()
    for record in iter_results(paths):
        cid = int(record.get("canonical_call_id") or 0)
        payload = str(record.get("analysis_json") or "")
        if cid not in first_by_id:
            first_by_id[cid] = record
            payload_by_id[cid] = payload
            continue
        if payload_by_id[cid] == payload:
            duplicate_same += 1
            duplicate_ids["duplicate_same"].append(cid)
            continue
        duplicate_conflict += 1
        duplicate_ids["duplicate_conflict"].append(cid)
        conflict_ids.add(cid)
    return (
        [first_by_id[cid] for cid in sorted(first_by_id)],
        {
            "same": duplicate_same,
            "conflict": duplicate_conflict,
            "ids": duplicate_ids,
            "conflict_ids": sorted(conflict_ids),
        },
    )


def iter_results(paths: Sequence[Path]) -> Iterable[Mapping[str, Any]]:
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise RuntimeError(f"result line is not object in {path}")
                yield data


def read_manifest(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("manifest must be an object")
    return data


def manifest_call_ids(manifest: Mapping[str, Any]) -> set[int]:
    calls = manifest.get("calls")
    if not isinstance(calls, list):
        raise RuntimeError("manifest.calls must be a list")
    ids = set()
    for item in calls:
        if not isinstance(item, dict):
            raise RuntimeError("manifest.calls item must be an object")
        ids.add(int(item["canonical_call_id"]))
    if int(manifest.get("rows") or 0) != len(ids):
        raise RuntimeError("manifest rows mismatch unique call ids")
    return ids


def read_int_set(path: Path) -> set[int]:
    values: set[int] = set()
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item.isdigit():
            values.add(int(item))
    return values


def prompt_counts(con: sqlite3.Connection, allowed_ids: set[int], prompt_version: str) -> Mapping[str, int]:
    if not allowed_ids:
        return {"rows": 0, "prompt_version_rows": 0}
    con.execute("DROP TABLE IF EXISTS tz19_allowed_ids")
    con.execute("CREATE TEMP TABLE tz19_allowed_ids(call_id INTEGER PRIMARY KEY)")
    con.executemany("INSERT INTO tz19_allowed_ids(call_id) VALUES (?)", [(item,) for item in sorted(allowed_ids)])
    row = con.execute(
        """
        SELECT COUNT(*) AS rows,
               SUM(
                 CASE
                   WHEN analysis_json IS NOT NULL
                    AND json_valid(analysis_json)
                    AND json_extract(analysis_json, '$.analysis_meta.analysis_prompt_version') = ?
                   THEN 1 ELSE 0
                 END
               ) AS prompt_version_rows
        FROM canonical_calls c
        JOIN tz19_allowed_ids a ON a.call_id = c.canonical_call_id
        """,
        (prompt_version,),
    ).fetchone()
    return {"rows": int(row["rows"] or 0), "prompt_version_rows": int(row["prompt_version_rows"] or 0)}


def add_rejected_id(container: dict[str, list[int]], key: str, cid: int) -> None:
    container.setdefault(key, []).append(cid)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--results", nargs="+", type=Path, default=list(DEFAULT_RESULTS))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-to", type=Path, default=None)
    parser.add_argument("--expect-prompt-version", default="v7")
    parser.add_argument("--expect-model-substr", default="mini")
    parser.add_argument("--expect-prompt-sha256", default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = import_tail_results(
            ImportConfig(
                db=args.db,
                manifest=args.manifest,
                blacklist=args.blacklist,
                results=tuple(args.results),
                apply=bool(args.apply),
                backup_to=args.backup_to,
                expect_prompt_version=args.expect_prompt_version,
                expect_model_substr=args.expect_model_substr,
                expect_prompt_sha256=args.expect_prompt_sha256,
                report_out=args.report_out,
            )
        )
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    rejected_total = sum(value for key, value in summary["counters"].items() if key.startswith("rejected_"))
    return 1 if rejected_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
