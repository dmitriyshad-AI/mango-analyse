#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

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
    quick_check,
    remove_sidecars,
    render_command,
    report_base,
    run_command,
    sha256_file,
    split_ignored_lsof_holders,
)


def flip(config_path: Path, *, snapshot_db: Path, execute: bool) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "flip")
    prod_db = cfg.prod_db
    snapshot_db = snapshot_db.expanduser().resolve(strict=True)
    report.update({"execute": execute, "prod_db": str(prod_db), "snapshot_db": str(snapshot_db)})
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
    all_holders = lsof_holders(prod_db)
    holders, ignored_holders = split_ignored_lsof_holders(
        all_holders,
        ignored_command_prefixes=tuple(str(item) for item in cfg.raw.get("ignored_lsof_command_prefixes") or ()),
    )
    if holders:
        return {
            **report,
            "status": "blocked_lsof",
            "stop_results": stop_results,
            "holders": holders,
            "ignored_holders": ignored_holders,
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
    removed_sidecars = remove_sidecars(prod_db, execute=True)
    tmp_target = prod_db.with_suffix(prod_db.suffix + ".new")
    shutil.copy2(snapshot_db, tmp_target)
    os.replace(tmp_target, prod_db)
    new_sha = sha256_file(prod_db)
    ok = quick_check(prod_db) == "ok"

    start_results = []
    for reader in cfg.readers:
        command = reader.get("start_command")
        if command:
            worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
            result = run_command(render_command(command, {"db": prod_db}), cwd=worktree, timeout=int(reader.get("start_timeout_seconds") or 120))
            result["name"] = reader.get("name")
            start_results.append(result)
            ok = ok and result.get("rc") == 0
    report.update(
        {
            "status": "ok" if ok else "failed",
            "stop_results": stop_results,
            "start_results": start_results,
            "backup_db": str(backup_db),
            "backup_copy": backup_copy,
            "async_backup_db": str(async_backup_db),
            "async_backup_copy": async_backup_copy,
            "backup_sha256": backup_sha,
            "new_sha256": new_sha,
            "removed_sidecars": removed_sidecars,
            "ignored_lsof_holders": ignored_holders,
            "quick_check": quick_check(prod_db),
        }
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomically flip stable Customer Timeline DB path to a built snapshot.")
    add_common_args(parser)
    parser.add_argument("--snapshot-db", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually stop readers and replace prod DB.")
    args = parser.parse_args()
    report, ok = flip(args.config, snapshot_db=args.snapshot_db, execute=args.execute)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
