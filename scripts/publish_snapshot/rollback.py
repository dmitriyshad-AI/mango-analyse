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
    lsof_holders,
    quick_check,
    remove_sidecars,
    render_command,
    replace_sqlite_verified,
    report_base,
    run_command,
    sha256_file,
    sidecar_paths,
    split_ignored_lsof_holders,
    table_counts,
    user_version,
    wal_checkpoint_truncate,
)
from scripts.publish_snapshot.flip import wait_process_pattern_counts


def _non_ignored_holders(prod_db: Path, cfg) -> tuple[list[str], list[str]]:
    all_holders = lsof_holders(prod_db)
    return split_ignored_lsof_holders(
        all_holders,
        ignored_command_prefixes=tuple(str(item) for item in cfg.raw.get("ignored_lsof_command_prefixes") or ()),
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
    restart_readers: bool = False,
) -> tuple[dict, bool]:
    """Restore prod_db from a verified pre_flip_backup.

    Symmetric with flip(): stop readers -> lsof check -> replace -> (optionally)
    restart readers -> caller runs reader_smoke separately. Readers are always
    stopped before the file is touched (same as flip), but ``start_command`` only
    runs when the caller passes ``restart_readers=True`` -- by default rollback
    must not change whatever state Wappi/readers were already in (master TZ 11.5).
    """
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
                "restart_readers": restart_readers,
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
    report.update(
        {
            "execute": execute,
            "restart_readers": restart_readers,
            "prod_db": str(prod_db),
            "backup_db": str(backup_db),
            **validation,
        }
    )
    if not source_ok:
        report["status"] = "blocked_rollback_source"
        return report, False
    if not execute:
        report["status"] = "dry_run_validated"
        return report, True

    stop_results = []
    for reader in cfg.readers:
        command = reader.get("stop_command")
        if command:
            worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
            result = run_command(render_command(command, {"db": prod_db}), cwd=worktree, timeout=int(reader.get("stop_timeout_seconds") or 120))
            result["name"] = reader.get("name")
            stop_results.append(result)
    failed_stops = [result for result in stop_results if result.get("rc") != 0]
    if failed_stops:
        report.update(
            {
                "status": "blocked_reader_stop",
                "stop_results": stop_results,
                "failed_stops": failed_stops,
            }
        )
        return report, False
    holders, ignored_holders = _non_ignored_holders(prod_db, cfg)
    if holders:
        report.update(
            {
                "status": "blocked_lsof",
                "stop_results": stop_results,
                "holders": holders,
                "ignored_holders": ignored_holders,
            }
        )
        return report, False

    try:
        backup_checkpoint = wal_checkpoint_truncate(backup_db)
        backup_removed_sidecars = remove_sidecars(backup_db, execute=True)
    except Exception as exc:
        report.update(
            {
                "status": "blocked_checkpoint",
                "stop_results": stop_results,
                "exception": {"type": type(exc).__name__, "message": str(exc), "attempt": 0},
            }
        )
        return report, False
    tmp_target = prod_db.with_suffix(prod_db.suffix + ".rollback")
    shutil.copy2(backup_db, tmp_target)

    pre_replace_holders, pre_replace_ignored_holders = _non_ignored_holders(prod_db, cfg)
    if pre_replace_holders:
        tmp_target.unlink(missing_ok=True)
        report.update(
            {
                "status": "blocked_lsof_before_replace",
                "stop_results": stop_results,
                "holders": pre_replace_holders,
                "ignored_holders": ignored_holders,
                "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
                "backup_checkpoint": backup_checkpoint,
                "backup_removed_sidecars": backup_removed_sidecars,
            }
        )
        return report, False

    removed_sidecars = remove_sidecars(prod_db, execute=True)
    replacement = replace_sqlite_verified(tmp_target, prod_db)
    ok = bool(replacement["ok"])

    start_results = []
    post_start_process_checks = []
    skipped_start = []
    if ok:
        for reader in cfg.readers:
            command = reader.get("start_command")
            if not restart_readers:
                if command:
                    skipped_start.append({"name": reader.get("name"), "reason": "restart_readers_flag_not_set"})
                continue
            if command:
                worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
                result = run_command(render_command(command, {"db": prod_db}), cwd=worktree, timeout=int(reader.get("start_timeout_seconds") or 120))
                result["name"] = reader.get("name")
                start_results.append(result)
                ok = ok and result.get("rc") == 0
            patterns = tuple(str(item) for item in reader.get("process_patterns") or ())
            if patterns:
                process_check = dict(
                    wait_process_pattern_counts(
                        patterns,
                        timeout=int(reader.get("start_timeout_seconds") or 120),
                    )
                )
                process_check["name"] = reader.get("name")
                post_start_process_checks.append(process_check)
                ok = ok and bool(process_check.get("ok"))

    report.update(
        {
            "status": "ok" if ok else "failed",
            "sha256": replacement["sha256"],
            "quick_check": replacement["quick_check"],
            "stop_results": stop_results,
            "ignored_lsof_holders": ignored_holders,
            "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
            "backup_checkpoint": backup_checkpoint,
            "backup_removed_sidecars": backup_removed_sidecars,
            "removed_sidecars": removed_sidecars,
            "start_results": start_results,
            "skipped_start": skipped_start,
            "post_start_process_checks": post_start_process_checks,
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
    parser.add_argument(
        "--restart-readers",
        action="store_true",
        help=(
            "Restart configured reader services (e.g. Wappi draft loop) after a successful rollback. "
            "Default (omitted) stops readers for the restore and leaves them stopped afterward -- pass "
            "this flag only when that service was intentionally running before the rollback."
        ),
    )
    args = parser.parse_args()
    report, ok = rollback(
        args.config,
        backup_db=args.backup_db,
        execute=args.execute,
        expected_sha256=args.expected_backup_sha256,
        restart_readers=args.restart_readers,
    )
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
