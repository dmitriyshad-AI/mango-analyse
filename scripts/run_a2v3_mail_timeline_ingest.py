#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mango_mvp.customer_timeline.a2_mail_ingest import (
    DEFAULT_A2V3_INPUT,
    DEFAULT_PROD_TIMELINE_DB,
    DEFAULT_TALLANTO_IDENTITY_DB,
    A2V3MailIngestConfig,
    apply_a2v3_mail_ingest,
    build_local_client_review,
    create_test_db_backup,
    plan_a2v3_mail_ingest,
    prod_readonly_check,
    validate_a2v3_mail_ingest,
    verify_test_db,
    write_foton_report,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A2-v3 100 email timeline ingest test runner.")
    parser.add_argument("command", choices=("init-db", "validate", "backup", "apply", "run-all"))
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_A2V3_INPUT)
    parser.add_argument("--prod-timeline-db", type=Path, default=DEFAULT_PROD_TIMELINE_DB)
    parser.add_argument("--tallanto-identity-db", type=Path, default=DEFAULT_TALLANTO_IDENTITY_DB)
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--source-ref", default="a2v3_100_review_20260701")
    parser.add_argument("--backup-manifest", type=Path)
    parser.add_argument(
        "--foton-report",
        type=Path,
        default=Path("/Users/dmitrijfabarisov/Claude Projects/Foton/ingest_100_report.md"),
    )
    return parser


def make_config(args: argparse.Namespace) -> A2V3MailIngestConfig:
    return A2V3MailIngestConfig(
        input_jsonl=args.input_jsonl,
        prod_timeline_db=args.prod_timeline_db,
        timeline_db_path=args.timeline_db,
        allowed_root=args.allowed_root,
        out_dir=args.out_dir,
        tallanto_identity_db=args.tallanto_identity_db,
        tenant_id=args.tenant_id,
        source_ref=args.source_ref,
    )


def init_test_db(config: A2V3MailIngestConfig) -> dict[str, object]:
    store = CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=config.allowed_root)
    try:
        open_result = store.open_result
    finally:
        store.close()
    return {
        "mode": "init-db",
        "test_db_path": str(config.timeline_db_path),
        "open_result": open_result.to_json_dict(),
    }


def run_all(config: A2V3MailIngestConfig, *, foton_report: Path) -> dict[str, object]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    prod_before = prod_readonly_check(config.prod_timeline_db)
    init_report = init_test_db(config)
    validate = validate_a2v3_mail_ingest(config)
    backup = create_test_db_backup(config, label="a2v3_100")
    first_apply = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))
    second_apply = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))
    plans, _ = plan_a2v3_mail_ingest(config)
    review = build_local_client_review(config, plans)
    test_db_verification = verify_test_db(config)
    prod_after = prod_readonly_check(config.prod_timeline_db)
    if prod_before["sha256_before"] != prod_after["sha256_after"]:
        raise RuntimeError("prod timeline sha256 changed during A2-v3 ingest test")
    write_foton_report(
        foton_report,
        prod_check_before=prod_before,
        prod_check_after=prod_after,
        validate_report=validate,
        first_apply_report=first_apply,
        second_apply_report=second_apply,
        review_report=review,
        test_db_verification=test_db_verification,
        test_db_path=config.timeline_db_path,
    )
    return {
        "mode": "run-all",
        "init": init_report,
        "validate": validate,
        "backup": backup,
        "first_apply": first_apply,
        "second_apply": second_apply,
        "test_db_verification": test_db_verification,
        "client_review": review,
        "prod_sha256_unchanged": True,
        "foton_report": str(foton_report),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = make_config(args)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "init-db":
        report = init_test_db(config)
    elif args.command == "validate":
        report = validate_a2v3_mail_ingest(config)
    elif args.command == "backup":
        init_test_db(config)
        report = create_test_db_backup(config, label="a2v3_100")
    elif args.command == "apply":
        if args.backup_manifest is None:
            print("apply requires --backup-manifest", file=sys.stderr)
            return 2
        report = apply_a2v3_mail_ingest(config, backup_manifest_path=args.backup_manifest)
    elif args.command == "run-all":
        report = run_all(config, foton_report=args.foton_report)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
