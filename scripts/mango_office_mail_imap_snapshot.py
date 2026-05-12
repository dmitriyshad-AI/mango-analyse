#!/usr/bin/env python3
"""Build a read-only IMAP mailbox/header snapshot for productization.

The script reads only folder metadata and message headers via BODY.PEEK. It
does not download bodies, attachments, send mail, delete/move mail, or write CRM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mail_imap_snapshot import (  # noqa: E402
    MailImapCredentials,
    MailImapSnapshotConfig,
    build_mail_imap_snapshot,
)


DEFAULT_PRODUCT_ROOT = "_local_archive_mango_api_downloads_20260507/product_appliance"
DEFAULT_HOST = "mail.hosting.reg.ru"
DEFAULT_PORT = 993
DEFAULT_PASSWORD_ENV = "MAIL_IMAP_PASSWORD"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    password = os.environ.get(args.password_env, "")
    email_address = args.email or os.environ.get("MAIL_IMAP_EMAIL", "")
    host = args.host or os.environ.get("MAIL_IMAP_HOST", DEFAULT_HOST)
    port = args.port or int(os.environ.get("MAIL_IMAP_PORT", DEFAULT_PORT))

    if not email_address:
        print("MAIL IMAP snapshot failed: missing --email or MAIL_IMAP_EMAIL.", file=sys.stderr)
        return 2
    if not password:
        print(
            f"MAIL IMAP snapshot failed: missing password env {args.password_env}.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir or default_out_dir(args.account_label))
    try:
        report = build_mail_imap_snapshot(
            credentials=MailImapCredentials(
                host=host,
                port=port,
                email_address=email_address,
                password=password,
            ),
            config=MailImapSnapshotConfig(
                out_dir=out_dir,
                since_days=args.since_days,
                header_sample_limit_per_mailbox=args.header_sample_limit_per_mailbox,
                account_label=args.account_label,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL IMAP snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "email": report["email"],
                "host": report["host"],
                "port": report["port"],
                "since_days": report["since_days"],
                "mailbox_count": report["mailbox_count"],
                "header_sample_rows": report["header_sample_rows"],
                "total_since": report["total_since"],
                "total_messages": report["total_messages"],
                "errors": report["errors"],
                "paths": report["paths"],
                "safety": report["safety"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not report["errors"] else 1


def default_out_dir(account_label: str) -> str:
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in account_label)
    suffix = datetime.now().strftime("%Y%m%d")
    return f"{DEFAULT_PRODUCT_ROOT}/mail_snapshots/{safe_label}_{suffix}_dry_run"


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only IMAP mail snapshot.")
    parser.add_argument("--host", help=f"Defaults to MAIL_IMAP_HOST or {DEFAULT_HOST}.")
    parser.add_argument("--port", type=int, help=f"Defaults to MAIL_IMAP_PORT or {DEFAULT_PORT}.")
    parser.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    parser.add_argument(
        "--password-env",
        default=DEFAULT_PASSWORD_ENV,
        help=f"Environment variable that contains the mailbox password. Default: {DEFAULT_PASSWORD_ENV}.",
    )
    parser.add_argument("--account-label", default="regru_edu")
    parser.add_argument("--out-dir")
    parser.add_argument("--since-days", type=int, default=30)
    parser.add_argument("--header-sample-limit-per-mailbox", type=int, default=10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
