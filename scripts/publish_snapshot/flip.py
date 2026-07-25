#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publish_snapshot.common import (
    add_common_args,
    backup_plan_report,
    copy_verified,
    finish_cli,
    lsof_holders,
    load_config,
    remove_sidecars,
    replace_sqlite_verified,
    render_command,
    report_base,
    run_command,
    sha256_file,
    split_ignored_lsof_holders,
    wal_checkpoint_truncate,
)


def process_pattern_counts(patterns: Sequence[str]) -> list[dict[str, object]]:
    normalized = tuple(str(pattern).strip() for pattern in patterns if str(pattern).strip())
    if not normalized:
        return []
    proc = subprocess.run(["ps", "-axo", "pid=,command="], text=True, capture_output=True)
    if proc.returncode != 0:
        return [
            {
                "pattern": pattern,
                "count": -1,
                "matches": [],
                "error": proc.stderr.strip(),
            }
            for pattern in normalized
        ]
    lines = [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]
    return [
        {
            "pattern": pattern,
            "count": sum(1 for line in lines if pattern in line),
            "matches": [line for line in lines if pattern in line][:10],
        }
        for pattern in normalized
    ]


def wait_process_pattern_counts(patterns: Sequence[str], *, timeout: int) -> Mapping[str, object]:
    deadline = time.monotonic() + max(1, timeout)
    last: list[dict[str, object]] = process_pattern_counts(patterns)
    while time.monotonic() <= deadline:
        if last and all(int(item.get("count") or 0) == 1 for item in last):
            return {"ok": True, "checks": last}
        time.sleep(1)
        last = process_pattern_counts(patterns)
    return {"ok": bool(last) and all(int(item.get("count") or 0) == 1 for item in last), "checks": last}


def _non_ignored_holders(prod_db: Path, cfg) -> tuple[list[str], list[str]]:
    all_holders = lsof_holders(prod_db)
    return split_ignored_lsof_holders(
        all_holders,
        ignored_command_prefixes=tuple(str(item) for item in cfg.raw.get("ignored_lsof_command_prefixes") or ()),
    )


def flip(config_path: Path, *, snapshot_db: Path, execute: bool, restart_readers: bool = False) -> tuple[dict, bool]:
    """Atomically swap prod_db to snapshot_db.

    Service restart is a separate, explicit decision from the data swap (see
    Foton/codex_artifacts/ETAP6_publish_plan.md and master TZ 11.6): readers are
    always stopped before the swap (to release locks), but a reader's
    ``start_command`` only runs when the caller passes ``restart_readers=True``.
    Without that flag, flip leaves every reader stopped after a successful swap,
    e.g. to keep the Wappi draft loop off until OWNER-GATE 5A is granted.
    """
    cfg = load_config(config_path)
    report = report_base(cfg, "flip")
    prod_db = cfg.prod_db
    snapshot_db = snapshot_db.expanduser().resolve(strict=True)
    report.update(
        {
            "execute": execute,
            "restart_readers": restart_readers,
            "prod_db": str(prod_db),
            "snapshot_db": str(snapshot_db),
        }
    )
    if not execute:
        report["status"] = "dry_run"
        return report, True
    backup_check = backup_plan_report(
        prod_db,
        cfg.backup_root,
        cfg.backup_async_copy_root,
        required_bytes=prod_db.stat().st_size,
    )
    if not backup_check["ok"]:
        return {**report, "status": "blocked_backup_preflight", "backup": backup_check}, False

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
        return {
            **report,
            "status": "blocked_reader_stop",
            "stop_results": stop_results,
            "failed_stops": failed_stops,
        }, False
    holders, ignored_holders = _non_ignored_holders(prod_db, cfg)
    if holders:
        return {
            **report,
            "status": "blocked_lsof",
            "stop_results": stop_results,
            "holders": holders,
            "ignored_holders": ignored_holders,
        }, False

    try:
        prod_checkpoint = wal_checkpoint_truncate(prod_db)
        snapshot_checkpoint = wal_checkpoint_truncate(snapshot_db)
        snapshot_removed_sidecars = remove_sidecars(snapshot_db, execute=True)
    except Exception as exc:
        return {
            **report,
            "status": "blocked_checkpoint",
            "stop_results": stop_results,
            "exception": {"type": type(exc).__name__, "message": str(exc), "attempt": 0},
        }, False

    backup_root = cfg.backup_root
    if backup_root is None:
        return {**report, "status": "blocked_backup_root_missing", "backup": backup_check}, False
    async_root = cfg.backup_async_copy_root
    if async_root is None:
        return {**report, "status": "blocked_backup_async_copy_root_missing", "backup": backup_check}, False
    backup_dir = backup_root / ("pre_flip_backup_" + report["generated_at"].replace(":", "").replace("+", "Z"))
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_db = backup_dir / prod_db.name
    backup_copy = copy_verified(prod_db, backup_db)
    backup_sha = str(backup_copy["target_sha256"])
    async_backup_dir = async_root / backup_dir.name
    async_backup_db = async_backup_dir / prod_db.name
    async_backup_copy = copy_verified(backup_db, async_backup_db)
    tmp_target = prod_db.with_suffix(prod_db.suffix + ".new")
    shutil.copy2(snapshot_db, tmp_target)
    pre_replace_holders, pre_replace_ignored_holders = _non_ignored_holders(prod_db, cfg)
    if pre_replace_holders:
        tmp_target.unlink(missing_ok=True)
        return {
            **report,
            "status": "blocked_lsof_before_replace",
            "stop_results": stop_results,
            "holders": pre_replace_holders,
            "ignored_holders": ignored_holders,
            "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
            "backup_db": str(backup_db),
            "backup_copy": backup_copy,
            "async_backup_db": str(async_backup_db),
            "async_backup_copy": async_backup_copy,
            "backup_sha256": backup_sha,
            "prod_checkpoint": prod_checkpoint,
            "snapshot_checkpoint": snapshot_checkpoint,
            "snapshot_removed_sidecars": snapshot_removed_sidecars,
        }, False
    removed_sidecars = remove_sidecars(prod_db, execute=True)
    replacement = replace_sqlite_verified(tmp_target, prod_db)
    if not replacement["ok"]:
        return {
            **report,
            "status": "failed_post_replace_verification",
            "stop_results": stop_results,
            "backup_db": str(backup_db),
            "backup_copy": backup_copy,
            "async_backup_db": str(async_backup_db),
            "async_backup_copy": async_backup_copy,
            "backup_sha256": backup_sha,
            "prod_checkpoint": prod_checkpoint,
            "snapshot_checkpoint": snapshot_checkpoint,
            "snapshot_removed_sidecars": snapshot_removed_sidecars,
            "removed_sidecars": removed_sidecars,
            "ignored_lsof_holders": ignored_holders,
            "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
            "post_replace_verification": replacement,
        }, False
    new_sha = str(replacement["sha256"])
    ok = True

    start_results = []
    post_start_process_checks = []
    skipped_start = []
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
            "stop_results": stop_results,
            "start_results": start_results,
            "skipped_start": skipped_start,
            "post_start_process_checks": post_start_process_checks,
            "backup_db": str(backup_db),
            "backup_copy": backup_copy,
            "async_backup_db": str(async_backup_db),
            "async_backup_copy": async_backup_copy,
            "backup_sha256": backup_sha,
            "new_sha256": new_sha,
            "prod_checkpoint": prod_checkpoint,
            "snapshot_checkpoint": snapshot_checkpoint,
            "snapshot_removed_sidecars": snapshot_removed_sidecars,
            "removed_sidecars": removed_sidecars,
            "ignored_lsof_holders": ignored_holders,
            "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
            "quick_check": replacement["quick_check"],
            "post_replace_verification": replacement,
        }
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomically flip stable Customer Timeline DB path to a built snapshot.")
    add_common_args(parser)
    parser.add_argument("--snapshot-db", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually stop readers and replace prod DB.")
    parser.add_argument(
        "--restart-readers",
        action="store_true",
        help=(
            "Restart configured reader services (e.g. Wappi draft loop) after a successful flip. "
            "Default (omitted) stops readers for the swap and leaves them stopped afterward -- pass "
            "this flag only when starting that service is separately approved (e.g. OWNER-GATE 5A)."
        ),
    )
    args = parser.parse_args()
    report, ok = flip(args.config, snapshot_db=args.snapshot_db, execute=args.execute, restart_readers=args.restart_readers)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
