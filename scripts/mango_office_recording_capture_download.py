#!/usr/bin/env python3
"""Controlled Mango recording downloader from a Stage 6 capture plan.

This command is productization-only. It may download audio only under the
product appliance root when --execute is supplied. It never writes runtime DBs,
does not start ASR/R+A, and does not write to CRM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mango_office_client import (  # noqa: E402
    DEFAULT_MANGO_BASE_URL,
    MangoOfficeCredentials,
)
from mango_mvp.productization.mango_recordings import MangoRecordingDownloader  # noqa: E402
from mango_mvp.productization.recording_capture_download import (  # noqa: E402
    audit_recording_capture_download,
    run_recording_capture_download,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_SOURCE_PLAN = f"{DEFAULT_PRODUCT_ROOT}/recording_capture_dry_run/recording_capture_plan_stage6.jsonl"
DEFAULT_DOWNLOAD_DIR = f"{DEFAULT_PRODUCT_ROOT}/recording_capture_downloads"
DEFAULT_RECORDINGS_DIR = f"{DEFAULT_DOWNLOAD_DIR}/recordings"
DEFAULT_DOWNLOAD_MANIFEST = f"{DEFAULT_DOWNLOAD_DIR}/recording_download_manifest.jsonl"
DEFAULT_RUN_AUDIT = f"{DEFAULT_DOWNLOAD_DIR}/recording_download_stage7_audit.json"
DEFAULT_VERIFY_AUDIT = f"{DEFAULT_DOWNLOAD_DIR}/recording_download_stage7_verify_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env_file()
    args = parse_args(argv)
    try:
        if args.command == "run":
            report = run_command(args)
            out = args.out
        elif args.command == "audit":
            out = args.out
            report = audit_recording_capture_download(
                download_manifest_path=Path(args.download_manifest),
                product_root=Path(args.product_root),
                recordings_dir=Path(args.recordings_dir),
                out_path=Path(out),
            )
        else:
            raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"recording capture download failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"out": str(Path(out).resolve(strict=False)), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"].get("validation_ok", True) else 1


def run_command(args: argparse.Namespace) -> dict:
    downloader = None
    if args.execute:
        api_key = args.api_key or os.getenv("MANGO_OFFICE_API_KEY")
        api_salt = args.api_salt or os.getenv("MANGO_OFFICE_API_SALT")
        if not api_key or not api_salt:
            raise ValueError("MANGO_OFFICE_API_KEY and MANGO_OFFICE_API_SALT are required when --execute is supplied")
        downloader = MangoRecordingDownloader(
            credentials=MangoOfficeCredentials(api_key=api_key, api_salt=api_salt),
            base_url=args.base_url,
            timeout_sec=args.timeout_sec,
            link_retries=args.link_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
        )
    return dict(
        run_recording_capture_download(
            source_plan_manifest_path=Path(args.source_plan),
            product_root=Path(args.product_root),
            recordings_dir=Path(args.recordings_dir),
            download_manifest_path=Path(args.download_manifest),
            out_path=Path(args.out),
            downloader=downloader,
            execute=args.execute,
            limit=args.limit,
            manager_ref=args.manager_ref,
            sleep_sec=args.sleep_sec,
        )
    )


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled Mango recording downloader from Stage 6 manifest.")
    parser.add_argument("--product-root", default=DEFAULT_PRODUCT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--source-plan", default=DEFAULT_SOURCE_PLAN)
    run.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    run.add_argument("--download-manifest", default=DEFAULT_DOWNLOAD_MANIFEST)
    run.add_argument("--out", default=DEFAULT_RUN_AUDIT)
    run.add_argument("--execute", action="store_true", help="Actually download selected recordings.")
    run.add_argument("--limit", type=int)
    run.add_argument("--manager-ref")
    run.add_argument("--sleep-sec", type=float, default=1.5)
    run.add_argument("--base-url", default=os.getenv("MANGO_OFFICE_BASE_URL", DEFAULT_MANGO_BASE_URL))
    run.add_argument("--api-key")
    run.add_argument("--api-salt")
    run.add_argument("--timeout-sec", type=int, default=60)
    run.add_argument("--link-retries", type=int, default=8)
    run.add_argument("--rate-limit-sleep-sec", type=float, default=30.0)

    audit = sub.add_parser("audit")
    audit.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    audit.add_argument("--download-manifest", default=DEFAULT_DOWNLOAD_MANIFEST)
    audit.add_argument("--out", default=DEFAULT_VERIFY_AUDIT)
    return parser.parse_args(argv)


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
