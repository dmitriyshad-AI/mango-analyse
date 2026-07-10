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

from scripts.publish_snapshot.common import add_common_args, finish_cli, load_config, quick_check, remove_sidecars, report_base, sha256_file


def rollback(config_path: Path, *, backup_db: Path, execute: bool) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "rollback")
    prod_db = cfg.prod_db
    backup_db = backup_db.expanduser().resolve(strict=True)
    report.update({"execute": execute, "prod_db": str(prod_db), "backup_db": str(backup_db)})
    if not execute:
        report["status"] = "dry_run"
        return report, True
    removed_sidecars = remove_sidecars(prod_db, execute=True)
    tmp_target = prod_db.with_suffix(prod_db.suffix + ".rollback")
    shutil.copy2(backup_db, tmp_target)
    os.replace(tmp_target, prod_db)
    ok = quick_check(prod_db) == "ok"
    report.update(
        {
            "status": "ok" if ok else "failed",
            "sha256": sha256_file(prod_db),
            "quick_check": quick_check(prod_db),
            "removed_sidecars": removed_sidecars,
        }
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback stable Customer Timeline DB path to a previous snapshot/backup.")
    add_common_args(parser)
    parser.add_argument("--backup-db", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually replace prod DB with backup.")
    args = parser.parse_args()
    report, ok = rollback(args.config, backup_db=args.backup_db, execute=args.execute)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
