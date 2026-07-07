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
    finish_cli,
    foreign_key_check,
    git_head,
    load_config,
    quick_check,
    report_base,
    sha256_file,
    table_counts,
    user_version,
    vacuum_into,
    wal_checkpoint_truncate,
    write_json,
)


def build_snapshot(config_path: Path, *, execute: bool, snapshot_name: str | None = None) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "build_snapshot")
    snapshot_name = snapshot_name or "prod_" + report["generated_at"].replace(":", "").replace("+", "Z")
    snapshot_dir = cfg.snapshot_root / snapshot_name
    snapshot_db = snapshot_dir / "customer_timeline.sqlite"
    report.update({"snapshot_dir": str(snapshot_dir), "snapshot_db": str(snapshot_db), "execute": execute})
    if not execute:
        report["status"] = "dry_run"
        return report, True
    checkpoint = wal_checkpoint_truncate(cfg.staging_db)
    vacuum_into(cfg.staging_db, snapshot_db)
    manifest = {
        "schema_version": "customer_timeline_snapshot_build_manifest_v1",
        "built_at": report["generated_at"],
        "package_name": cfg.package_name,
        "writer_git_head": git_head(Path.cwd()),
        "source_staging_db": str(cfg.staging_db),
        "snapshot_db": str(snapshot_db),
        "sha256": sha256_file(snapshot_db),
        "size_bytes": snapshot_db.stat().st_size,
        "quick_check": quick_check(snapshot_db),
        "foreign_key_check_rows": len(foreign_key_check(snapshot_db)),
        "user_version": user_version(snapshot_db),
        "counts": table_counts(snapshot_db, cfg.count_tables),
        "control_customers": list(cfg.control_customers),
        "wal_checkpoint": checkpoint,
    }
    write_json(snapshot_dir / "build_manifest.json", manifest)
    report["manifest"] = manifest
    ok = manifest["quick_check"] == "ok" and manifest["foreign_key_check_rows"] == 0
    report["status"] = "ok" if ok else "failed"
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable Customer Timeline publication snapshot.")
    add_common_args(parser)
    parser.add_argument("--execute", action="store_true", help="Actually checkpoint staging and build snapshot.")
    parser.add_argument("--snapshot-name")
    args = parser.parse_args()
    report, ok = build_snapshot(args.config, execute=args.execute, snapshot_name=args.snapshot_name)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
