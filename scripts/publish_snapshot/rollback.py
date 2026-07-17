#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publish_snapshot.common import (
    add_common_args,
    finish_cli,
    foreign_key_check,
    load_config,
    quick_check,
    remove_sidecars,
    replace_sqlite_verified,
    report_base,
    sha256_file,
    sidecar_paths,
    table_counts,
    user_version,
    wal_checkpoint_truncate,
)


def validate_rollback_source(cfg, backup_db: Path, *, expected_sha256: str | None = None) -> tuple[dict, bool]:
    backup_sha256 = sha256_file(backup_db)
    backup_root = cfg.backup_root
    location_valid = bool(
        backup_root
        and backup_db.parent.parent == backup_root
        and backup_db.parent.name.startswith("pre_flip_backup_")
        and backup_db.name == cfg.prod_db.name
    )
    sidecars = {str(path): path.stat().st_size for path in sidecar_paths(backup_db) if path.exists()}
    expected = str(expected_sha256 or "").strip().lower()
    backup_counts = table_counts(backup_db, cfg.count_tables)
    count_tables_complete = set(cfg.count_tables).issubset(backup_counts)
    validation = {
        "backup_sha256": backup_sha256,
        "backup_quick_check": quick_check(backup_db),
        "backup_foreign_key_check_rows": len(foreign_key_check(backup_db)),
        "backup_counts": backup_counts,
        "backup_count_tables_complete": count_tables_complete,
        "backup_user_version": user_version(backup_db),
        "backup_sidecars": sidecars,
        "backup_location_valid": location_valid,
        "expected_sha256": expected or None,
        "expected_sha256_match": not expected or backup_sha256 == expected,
    }
    ok = (
        validation["backup_quick_check"] == "ok"
        and validation["backup_foreign_key_check_rows"] == 0
        and not sidecars
        and location_valid
        and count_tables_complete
        and validation["expected_sha256_match"] is True
    )
    return validation, ok


def rollback(
    config_path: Path,
    *,
    backup_db: Path,
    execute: bool,
    expected_sha256: str | None = None,
) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "rollback")
    prod_db = cfg.prod_db
    backup_db = backup_db.expanduser()
    try:
        backup_db = backup_db.resolve(strict=True)
        validation, source_ok = validate_rollback_source(cfg, backup_db, expected_sha256=expected_sha256)
    except Exception as exc:
        report.update(
            {
                "execute": execute,
                "prod_db": str(prod_db),
                "backup_db": str(backup_db.resolve(strict=False)),
                "status": "blocked_rollback_source",
                "backup_validation_exception": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        return report, False
    validation["expected_sha256_required"] = bool(execute)
    if execute and not validation["expected_sha256"]:
        source_ok = False
    report.update({"execute": execute, "prod_db": str(prod_db), "backup_db": str(backup_db), **validation})
    if not source_ok:
        report["status"] = "blocked_rollback_source"
        return report, False
    if not execute:
        report["status"] = "dry_run_validated"
        return report, True
    try:
        backup_checkpoint = wal_checkpoint_truncate(backup_db)
        backup_removed_sidecars = remove_sidecars(backup_db, execute=True)
    except Exception as exc:
        report.update(
            {
                "status": "blocked_checkpoint",
                "exception": {"type": type(exc).__name__, "message": str(exc), "attempt": 0},
            }
        )
        return report, False
    removed_sidecars = remove_sidecars(prod_db, execute=True)
    tmp_target = prod_db.with_suffix(prod_db.suffix + ".rollback")
    shutil.copy2(backup_db, tmp_target)
    replacement = replace_sqlite_verified(tmp_target, prod_db)
    ok = bool(replacement["ok"])
    report.update(
        {
            "status": "ok" if ok else "failed",
            "sha256": replacement["sha256"],
            "quick_check": replacement["quick_check"],
            "backup_checkpoint": backup_checkpoint,
            "backup_removed_sidecars": backup_removed_sidecars,
            "removed_sidecars": removed_sidecars,
            "post_replace_verification": replacement,
        }
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback stable Customer Timeline DB path to a previous snapshot/backup.")
    add_common_args(parser)
    parser.add_argument("--backup-db", type=Path, required=True)
    parser.add_argument("--expected-backup-sha256")
    parser.add_argument("--execute", action="store_true", help="Actually replace prod DB with backup.")
    args = parser.parse_args()
    report, ok = rollback(
        args.config,
        backup_db=args.backup_db,
        execute=args.execute,
        expected_sha256=args.expected_backup_sha256,
    )
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
