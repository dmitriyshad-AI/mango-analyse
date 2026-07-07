#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publish_snapshot.common import (
    add_common_args,
    disk_report,
    finish_cli,
    foreign_key_check,
    git_head,
    git_status_short,
    classify_publish_worktree_status,
    load_config,
    quick_check,
    report_base,
    separate_filesystem_report,
    schema_diff,
    table_counts,
)


def build_report(config_path: Path) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "preflight")
    staging = cfg.staging_db
    prod = cfg.prod_db
    snapshot_root = cfg.snapshot_root
    required = max(staging.stat().st_size if staging.exists() else 0, prod.stat().st_size if prod.exists() else 0) * cfg.required_free_copies
    readers = []
    ok = True
    for reader in cfg.readers:
        worktree = Path(str(reader.get("worktree") or "")).expanduser().resolve(strict=False)
        status = git_status_short(worktree) if str(worktree) else None
        classified_status = classify_publish_worktree_status(status)
        reader_report = {
            "name": reader.get("name"),
            "worktree": str(worktree) if str(worktree) else "",
            "git_head": git_head(worktree) if str(worktree) else None,
            "git_status_clean": status == "",
            "clean_for_publish": classified_status["clean_for_publish"],
            "git_status_short": status,
            "tracked_blockers": classified_status["tracked_blockers"],
            "untracked_code_blockers": classified_status["untracked_code_blockers"],
            "untracked_allowed": classified_status["untracked_allowed"],
            "has_stop_command": bool(reader.get("stop_command")),
            "has_start_command": bool(reader.get("start_command")),
            "has_smoke_command_or_internal": bool(reader.get("smoke_command")) or bool(cfg.control_customers),
        }
        if not reader_report["clean_for_publish"]:
            ok = False
        if not reader_report["has_stop_command"] or not reader_report["has_start_command"]:
            ok = False
        readers.append(reader_report)
    diff = schema_diff(prod, staging)
    disk = disk_report(snapshot_root.parent if snapshot_root.parent.exists() else Path.cwd(), required)
    backup = separate_filesystem_report(prod, cfg.backup_root, required_bytes=prod.stat().st_size if prod.exists() else 0)
    report.update(
        {
            "staging_db": str(staging),
            "prod_db": str(prod),
            "snapshot_root": str(snapshot_root),
            "quick_check": {"prod": quick_check(prod), "staging": quick_check(staging)},
            "foreign_key_check": {"prod_rows": len(foreign_key_check(prod)), "staging_rows": len(foreign_key_check(staging))},
            "schema_diff": diff,
            "counts": {"prod": table_counts(prod, cfg.count_tables), "staging": table_counts(staging, cfg.count_tables)},
            "disk": disk,
            "backup": backup,
            "readers": readers,
        }
    )
    ok = ok and disk["ok"] and report["quick_check"]["prod"] == "ok" and report["quick_check"]["staging"] == "ok"
    ok = ok and bool(backup["ok"])
    ok = ok and report["foreign_key_check"]["prod_rows"] == 0 and report["foreign_key_check"]["staging_rows"] == 0
    return report, bool(ok)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight Customer Timeline snapshot publication.")
    add_common_args(parser)
    args = parser.parse_args()
    report, ok = build_report(args.config)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
