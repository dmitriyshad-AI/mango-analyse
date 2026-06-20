#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mango_mvp.customer_timeline.mail_stage2_ingest import (
    MailStage2IngestConfig,
    apply_stage2_mail_ingest,
    create_timeline_backup,
    dry_run_stage2_mail_ingest,
    restore_timeline_backup,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe Stage2 mail ingest procedure for customer_timeline test DBs.",
    )
    parser.add_argument("command", choices=("backup", "dry-run", "apply", "restore"))
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--identity-db", required=True, type=Path)
    parser.add_argument("--event-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--backup-manifest", type=Path)
    parser.add_argument("--tenant-id", default="mango")
    parser.add_argument("--source-ref", default="mail_stage2_fresh_relink_20260621")
    parser.add_argument("--text-max-chars", type=int, default=6000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backup-label")
    return parser


def make_config(args: argparse.Namespace) -> MailStage2IngestConfig:
    return MailStage2IngestConfig(
        timeline_db_path=args.timeline_db,
        allowed_root=args.allowed_root,
        identity_db_path=args.identity_db,
        event_jsonl_paths=tuple(args.event_jsonl),
        out_dir=args.out_dir,
        backup_root=args.backup_root,
        tenant_id=args.tenant_id,
        source_ref=args.source_ref,
        text_max_chars=args.text_max_chars,
        limit=args.limit,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = make_config(args)
    if args.command == "backup":
        report = create_timeline_backup(config, label=args.backup_label)
    elif args.command == "dry-run":
        report = dry_run_stage2_mail_ingest(config)
    elif args.command == "apply":
        if args.backup_manifest is None:
            print("apply requires --backup-manifest created by backup command", file=sys.stderr)
            return 2
        report = apply_stage2_mail_ingest(config, backup_manifest_path=args.backup_manifest)
    elif args.command == "restore":
        if args.backup_manifest is None:
            print("restore requires --backup-manifest created by backup command", file=sys.stderr)
            return 2
        report = restore_timeline_backup(config, backup_manifest_path=args.backup_manifest)
    else:  # pragma: no cover - argparse protects this.
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
