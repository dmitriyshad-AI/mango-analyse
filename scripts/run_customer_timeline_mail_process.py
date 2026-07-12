#!/usr/bin/env python3
"""Build one mail-only Customer Timeline increment from downloaded archives."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mail_archive import CANONICAL_MAIL_ARCHIVE_DB  # noqa: E402
from scripts.build_customer_timeline_nightly_dv2_sources import build_mail_increment  # noqa: E402
from scripts.run_customer_timeline_mail_download import (  # noqa: E402
    CANONICAL_RELATIVE_ROOT,
    archive_stats,
    atomic_write_json,
    exclusive_lock,
    runtime_identity,
    sha256_file,
    utc_now,
)


def parse_time(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp_without_timezone")
    return parsed.astimezone(timezone.utc)


def load_success_manifest(
    path: Path, *, expected_runtime: Mapping[str, str], max_age_hours: float
) -> Mapping[str, Any]:
    if not path.is_file():
        raise RuntimeError("download_manifest_missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError("download_manifest_not_ok")
    if payload.get("truncated") is not False or int(payload.get("errors") or 0) != 0:
        raise RuntimeError("download_manifest_incomplete")
    reports = payload.get("mailbox_reports")
    if not isinstance(reports, Mapping) or set(reports) != {"inbox", "sent"}:
        raise RuntimeError("download_manifest_mailboxes_incomplete")
    if any(not isinstance(item, Mapping) or item.get("status") != "ok" for item in reports.values()):
        raise RuntimeError("download_manifest_mailbox_failed")
    if payload.get("runtime") != expected_runtime:
        raise RuntimeError("download_manifest_runtime_mismatch")
    age = datetime.now(timezone.utc) - parse_time(payload.get("finished_at"))
    if age < timedelta(0) or age > timedelta(hours=max_age_hours):
        raise RuntimeError("download_manifest_stale")
    return payload


def read_cursor(timeline_db: Path, *, bootstrap: str | None, overlap_seconds: int) -> datetime:
    if not timeline_db.is_file():
        if not bootstrap:
            raise RuntimeError("timeline_db_missing_and_no_bootstrap")
        return parse_time(bootstrap) - timedelta(seconds=overlap_seconds)
    uri = timeline_db.as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as con:
            row = con.execute(
                """
                SELECT last_cursor_ts
                FROM ingestion_cursors
                WHERE tenant_id = ? AND source_system = ?
                """,
                ("foton", "mail_archive_stage2"),
            ).fetchone()
    except sqlite3.Error as exc:
        if not bootstrap:
            raise RuntimeError("mail_cursor_unavailable_and_no_bootstrap") from exc
        row = None
    if row and row[0]:
        cursor = parse_time(row[0])
    elif bootstrap:
        cursor = parse_time(bootstrap)
    else:
        raise RuntimeError("mail_cursor_missing_and_no_bootstrap")
    return cursor - timedelta(seconds=overlap_seconds)


def safe_archive_paths(
    *, data_root: Path, download_manifest: Mapping[str, Any]
) -> list[Path]:
    canonical_root = (data_root / CANONICAL_RELATIVE_ROOT).resolve()
    paths = [(data_root / CANONICAL_MAIL_ARCHIVE_DB).resolve()]
    paths.extend(Path(str(item)).resolve() for item in download_manifest["archive_db_paths"])
    unique: list[Path] = []
    for path in paths:
        if canonical_root not in path.parents:
            raise RuntimeError("archive_db_outside_canonical_root")
        if path not in unique:
            unique.append(path)
    if any(not path.is_file() for path in unique):
        raise RuntimeError("archive_db_missing")
    return unique


def db_inventory(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    result = []
    for path in paths:
        stats = archive_stats(path)
        result.append(
            {
                "path": str(path),
                "exists": stats["exists"],
                "sha256": stats["sha256"],
                "mtime": path.stat().st_mtime if path.exists() else None,
                "message_count": stats["message_count"],
            }
        )
    return result


def staging_root_for(*, state_dir: Path, timeline_db: Path) -> Path:
    state_dir = state_dir.resolve()
    timeline_db = timeline_db.resolve()
    staging_root = state_dir.parent
    if staging_root.name != "staging" or staging_root.parent.name != ".codex_local":
        raise RuntimeError("mail_state_dir_not_under_codex_staging")
    if staging_root not in timeline_db.parents:
        raise RuntimeError("timeline_db_outside_codex_staging")
    lowered = "/".join(part.casefold() for part in timeline_db.parts)
    forbidden = ("stable_runtime", "customer_timeline_prod_", "/product_data/customer_timeline/")
    if any(token in lowered for token in forbidden):
        raise RuntimeError("timeline_db_matches_forbidden_prod_path")
    return staging_root


def _execute_locked(args: argparse.Namespace) -> Mapping[str, Any]:
    code_root = Path(args.code_root).resolve()
    data_root = Path(args.data_root).resolve()
    state_dir = Path(args.state_dir).resolve()
    timeline_db = Path(args.timeline_db).resolve()
    staging_root = staging_root_for(state_dir=state_dir, timeline_db=timeline_db)
    runtime = runtime_identity(code_root)
    download_path = state_dir / "mail_download_manifest.json"
    started_at = utc_now()
    download = load_success_manifest(
        download_path,
        expected_runtime=runtime,
        max_age_hours=args.max_download_age_hours,
    )
    archive_paths = safe_archive_paths(data_root=data_root, download_manifest=download)
    cursor_start = read_cursor(
        timeline_db,
        bootstrap=args.bootstrap_cursor,
        overlap_seconds=args.overlap_seconds,
    )
    process_dir = state_dir / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = process_dir / "mail_archive_stage2_incremental.jsonl"
    builder_manifest = process_dir / "mail_increment_builder_manifest.json"
    builder = build_mail_increment(
        data_root,
        out_jsonl=out_jsonl,
        manifest_path=builder_manifest,
        since=cursor_start,
        text_limit=args.text_limit,
        timeline_db=timeline_db,
        archive_db_paths=archive_paths,
    )
    config_path = process_dir / "mail_incremental_config.json"
    config = {
        "schema_version": "mail_incremental_config_v1",
        "runtime": runtime,
        "timeline_db": str(timeline_db),
        "allowed_root": str(staging_root),
        "journal_path": str(process_dir / "mail_incremental_journal.jsonl"),
        "tenant_id": "foton",
        "safety_margin_seconds": args.overlap_seconds,
        "sources": [
            {
                "name": "mail_archive_stage2_incremental",
                "source_system": "mail_archive_stage2",
                "path": str(out_jsonl),
                "source_ref": "mail_pipeline:mail_archive_stage2",
                "normalizer": "mail_archive_stage2",
                "required": True,
            }
        ],
    }
    atomic_write_json(config_path, config)
    manifest_path = state_dir / "mail_process_manifest.json"
    manifest = {
        "schema_version": "mail_process_manifest_v1",
        "status": "ok",
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime": runtime,
        "download_manifest": str(download_path),
        "download_manifest_sha256": sha256_file(download_path),
        "cursor_source": "ingestion_cursors.mail_archive_stage2",
        "cursor_start_with_overlap": cursor_start.isoformat(),
        "overlap_seconds": args.overlap_seconds,
        "archive_databases": db_inventory(archive_paths),
        "builder_manifest": str(builder_manifest),
        "builder_manifest_sha256": sha256_file(builder_manifest),
        "rows_written": int(builder.get("rows_written") or 0),
        "linked_rows": int(builder.get("linked_rows") or 0),
        "pending_rows": int(builder.get("pending_rows") or 0),
        "max_event_at": builder.get("max_event_at"),
        "output_jsonl": str(out_jsonl),
        "output_sha256": sha256_file(out_jsonl),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "write_external_systems": False,
        "writes_prod_db": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def execute(args: argparse.Namespace) -> Mapping[str, Any]:
    state_dir = Path(args.state_dir).resolve()
    with exclusive_lock(state_dir / "mail_pipeline.lock"):
        return _execute_locked(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", default=str(ROOT))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--state-dir", default=str(ROOT / ".codex_local/staging/mail_pipeline"))
    parser.add_argument(
        "--timeline-db",
        default=str(ROOT / ".codex_local/staging/customer_timeline_staging.sqlite"),
    )
    parser.add_argument("--bootstrap-cursor")
    parser.add_argument("--overlap-seconds", type=int, default=300)
    parser.add_argument("--max-download-age-hours", type=float, default=4.0)
    parser.add_argument("--text-limit", type=int, default=1200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = execute(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        if str(exc) == "mail_download_already_running":
            print(json.dumps({"status": "already_running", "stop_reason": "already_running"}, sort_keys=True))
            return 75
        print(json.dumps({"status": "failed", "error": type(exc).__name__}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
