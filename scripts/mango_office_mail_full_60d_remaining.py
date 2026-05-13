#!/usr/bin/env python3
"""Run a controlled read-only IMAP archive ingest window excluding known mail."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mail_archive import (  # noqa: E402
    MailArchiveIngestConfig,
    MailArchivePreflightConfig,
    MailArchiveVerificationConfig,
    build_mail_archive_ingest,
    build_mail_archive_preflight,
    guard_external_handoffs_output,
    guard_git_ignored_output,
    guard_not_stable_runtime,
    verify_mail_archive_pilot,
)
from mango_mvp.productization.mail_imap_snapshot import (  # noqa: E402
    ImapLibClient,
    MailImapCredentials,
    parse_mailbox_list_line,
)


DEFAULT_HOST = "mail.hosting.reg.ru"
DEFAULT_PORT = 993
DEFAULT_PASSWORD_ENV = "MAIL_IMAP_PASSWORD"
DEFAULT_ACCOUNT_LABEL = "regru_edu"
DEFAULT_ROOT = (
    "_external_handoffs/mail_archive_2026-05-12/"
    "regru_edu/full_60d_remaining_20260513"
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    archive_dir = out_root / "archive"
    batch_reports_dir = out_root / "batch_reports"
    try:
        guard_not_stable_runtime(out_root, "full mail run output directory")
        guard_external_handoffs_output(out_root, "full mail run output directory")
        guard_git_ignored_output(out_root, "full mail run output directory")
    except ValueError as exc:
        print(f"Full mail run blocked: {exc}", file=sys.stderr)
        return 2

    password = os.environ.get(args.password_env, "")
    if not password:
        print(f"Full mail run blocked: missing password env {args.password_env}.", file=sys.stderr)
        return 2
    email_address = args.email or os.environ.get("MAIL_IMAP_EMAIL", "")
    if not email_address:
        print("Full mail run blocked: missing --email or MAIL_IMAP_EMAIL.", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    batch_reports_dir.mkdir(parents=True, exist_ok=True)

    excluded_sha256s = load_excluded_message_sha256s(args.exclude_archive_db or [])
    plan = build_plan(
        host=args.host,
        port=args.port,
        email_address=email_address,
        password=password,
        since_days=args.since_days,
        older_than_days=args.older_than_days,
        max_messages=args.max_messages,
        include_mailboxes=set(args.mailbox or []),
    )
    plan["excluded_control_sha256_count"] = len(excluded_sha256s)
    plan["archive_dir"] = str(archive_dir)
    plan["batch_reports_dir"] = str(batch_reports_dir)
    write_json(out_root / "full_60d_plan.json", plan)

    blocking = [
        window
        for window in plan["windows"]
        if int(window["message_count"]) > int(args.max_messages)
    ]
    if blocking:
        write_json(out_root / "full_60d_blocking_windows.json", {"windows": blocking})
        print(
            f"Full mail run blocked: {len(blocking)} windows exceed max_messages.",
            file=sys.stderr,
        )
        return 2
    if args.plan_only:
        print_summary(plan, status="planned")
        return 0

    credentials = MailImapCredentials(
        host=args.host,
        port=args.port,
        email_address=email_address,
        password=password,
    )
    manifest: dict[str, Any] = {
        "schema_version": "mail_full_60d_remaining_run_v1",
        "created_at": utc_now(),
        "account_label": args.account_label,
        "archive_dir": str(archive_dir),
        "planned_window_count": len(plan["windows"]),
        "planned_message_count": plan["planned_message_count"],
        "since_days": args.since_days,
        "older_than_days": args.older_than_days,
        "excluded_control_sha256_count": len(excluded_sha256s),
        "batch_reports": [],
        "totals": {
            "messages_found_since": 0,
            "messages_attempted": 0,
            "messages_inserted_or_seen": 0,
            "messages_excluded_by_sha256": 0,
            "raw_eml_written": 0,
            "attachments_written": 0,
            "text_files_written": 0,
            "errors": 0,
        },
        "safety": {
            "readonly_select": True,
            "fetch_uses_body_peek": True,
            "send_mail": False,
            "delete_or_move_mail": False,
            "write_crm": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_ra": False,
            "stable_runtime_writes": False,
        },
    }
    manifest_path = out_root / "full_60d_run_manifest.json"

    for index, window in enumerate(plan["windows"], start=1):
        batch_id = str(window["batch_id"])
        batch_max_messages = int(args.max_messages)
        preflight = build_mail_archive_preflight(
            MailArchivePreflightConfig(
                out_dir=archive_dir,
                mailbox=str(window["mailbox_raw"]),
                since_days=int(window["since_days"]),
                before_days=window["before_days"],
                max_messages=batch_max_messages,
                account_label=args.account_label,
                host=args.host,
                port=args.port,
                email_address=email_address,
                password_env_name=args.password_env,
                password_env_present=True,
                identity_db_path=Path(args.identity_db) if args.identity_db else None,
                allow_large_batch=True,
            )
        )
        write_json(batch_reports_dir / f"{batch_id}_preflight.json", preflight)
        if not preflight.get("preflight_pass"):
            manifest["batch_reports"].append(
                {"batch_id": batch_id, "status": "blocked", "preflight_pass": False}
            )
            write_json(manifest_path, manifest)
            print(f"Batch blocked by preflight: {batch_id}", file=sys.stderr)
            return 2

        report = build_mail_archive_ingest(
            credentials=credentials,
            config=MailArchiveIngestConfig(
                out_dir=archive_dir,
                mailbox=str(window["mailbox_raw"]),
                mailbox_label=str(window["mailbox"]),
                since_days=int(window["since_days"]),
                before_days=window["before_days"],
                max_messages=batch_max_messages,
                account_label=args.account_label,
                internal_domains=tuple(args.internal_domain or []),
                extracted_text_max_chars=args.extracted_text_max_chars,
                exclude_message_sha256s=tuple(excluded_sha256s),
            ),
        )
        write_json(batch_reports_dir / f"{batch_id}_ingest.json", report)
        verification = verify_mail_archive_pilot(
            MailArchiveVerificationConfig(
                archive_dir=archive_dir,
                expected_max_messages=batch_max_messages,
            )
        )
        write_json(batch_reports_dir / f"{batch_id}_verification.json", verification)

        errors = len(report.get("errors") or [])
        complete = is_batch_complete(
            report,
            verification,
            max_messages=batch_max_messages,
        )
        planned_message_count = int(window["message_count"])
        live_message_count = int(report.get("messages_found_since") or 0)
        batch_summary = {
            "batch_id": batch_id,
            "index": index,
            "status": "ok" if complete else "failed",
            "mailbox": window["mailbox"],
            "since_days": window["since_days"],
            "before_days": window["before_days"],
            "planned_message_count": planned_message_count,
            "messages_found_since": live_message_count,
            "new_messages_seen_after_plan": max(0, live_message_count - planned_message_count),
            "messages_missing_after_plan": max(0, planned_message_count - live_message_count),
            "max_messages": batch_max_messages,
            "messages_attempted": report.get("messages_attempted"),
            "messages_inserted_or_seen": report.get("messages_inserted_or_seen"),
            "messages_excluded_by_sha256": report.get("messages_excluded_by_sha256"),
            "errors": errors,
            "verification_pass": verification.get("verification_pass"),
        }
        manifest["batch_reports"].append(batch_summary)
        for key in manifest["totals"]:
            if key == "errors":
                manifest["totals"][key] += errors
            else:
                manifest["totals"][key] += int(report.get(key) or 0)
        write_json(manifest_path, manifest)
        print(
            f"[{index}/{len(plan['windows'])}] {batch_id}: "
            f"found={report.get('messages_found_since')} "
            f"inserted={report.get('messages_inserted_or_seen')} "
            f"excluded={report.get('messages_excluded_by_sha256')} "
            f"errors={errors}",
            flush=True,
        )
        if not complete:
            print(f"Full mail run stopped at failed batch: {batch_id}", file=sys.stderr)
            return 1

    manifest["completed_at"] = utc_now()
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    print_summary(manifest, status="completed")
    return 0


def build_plan(
    *,
    host: str,
    port: int,
    email_address: str,
    password: str,
    since_days: int,
    older_than_days: int,
    max_messages: int,
    include_mailboxes: set[str],
) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    windows: list[dict[str, Any]] = []
    folder_counts: list[dict[str, Any]] = []
    imap = ImapLibClient(host=host, port=port)
    try:
        imap.login(email_address, password)
        list_status, boxes = imap.list()
        if list_status != "OK":
            raise RuntimeError(f"IMAP LIST failed: {list_status}")
        for raw_line in boxes or []:
            parsed = parse_mailbox_list_line(raw_line)
            flags = set(parsed.get("flags") or [])
            mailbox = str(parsed.get("name") or "")
            mailbox_raw = str(parsed.get("name_raw") or mailbox)
            if "\\Noselect" in flags:
                continue
            if include_mailboxes and mailbox not in include_mailboxes:
                continue
            select_status, _ = imap.select(mailbox_raw, readonly=True)
            if select_status != "OK":
                folder_counts.append(
                    {"mailbox": mailbox, "mailbox_raw": mailbox_raw, "select_status": select_status}
                )
                continue
            total = 0
            for age in range(int(since_days), int(older_than_days), -1):
                since_imap = (today - timedelta(days=age)).strftime("%d-%b-%Y")
                before_days = age - 1 if age > 1 else None
                before_imap = (
                    (today - timedelta(days=before_days)).strftime("%d-%b-%Y")
                    if before_days is not None
                    else ""
                )
                criteria = ["SINCE", since_imap]
                if before_imap:
                    criteria.extend(["BEFORE", before_imap])
                search_status, search_data = imap.search(None, *criteria)
                if search_status != "OK":
                    raise RuntimeError(f"IMAP SEARCH failed for {mailbox}: {search_status}")
                count = count_search_ids(search_data)
                total += count
                if count:
                    windows.append(
                        {
                            "batch_id": batch_id(mailbox, age, before_days),
                            "mailbox": mailbox,
                            "mailbox_raw": mailbox_raw,
                            "since_days": age,
                            "before_days": before_days,
                            "since_imap": since_imap,
                            "before_imap": before_imap,
                            "message_count": count,
                            "exceeds_max_messages": count > int(max_messages),
                        }
                    )
            folder_counts.append(
                {
                    "mailbox": mailbox,
                    "mailbox_raw": mailbox_raw,
                    "select_status": "OK",
                    "since_days": since_days,
                    "message_count": total,
                }
            )
            imap.close()
    finally:
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass
    return {
        "schema_version": "mail_full_60d_remaining_plan_v1",
        "created_at": utc_now(),
        "today_utc": today.isoformat(),
        "since_days": since_days,
        "older_than_days": older_than_days,
        "max_messages": max_messages,
        "folder_count": len(folder_counts),
        "planned_window_count": len(windows),
        "planned_message_count": sum(int(window["message_count"]) for window in windows),
        "max_window_count": max((int(window["message_count"]) for window in windows), default=0),
        "folder_counts": folder_counts,
        "windows": windows,
        "privacy": {
            "downloads_raw_mail": False,
            "contains_password": False,
            "contains_raw_personal_values": False,
        },
    }


def load_excluded_message_sha256s(paths: Sequence[str]) -> set[str]:
    values: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        uri = f"file:{quote(str(path.resolve(strict=False)), safe='/:')}?mode=ro"
        with sqlite3.connect(uri, uri=True) as con:
            for row in con.execute("SELECT sha256 FROM messages"):
                sha256 = str(row[0] or "").strip()
                if len(sha256) == 64 and all(ch in "0123456789abcdef" for ch in sha256):
                    values.add(sha256)
    return values


def is_batch_complete(
    report: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    max_messages: int,
) -> bool:
    errors = len(report.get("errors") or [])
    found = int(report.get("messages_found_since") or 0)
    attempted = int(report.get("messages_attempted") or 0)
    inserted_or_seen = int(report.get("messages_inserted_or_seen") or 0)
    excluded = int(report.get("messages_excluded_by_sha256") or 0)
    return (
        errors == 0
        and found <= int(max_messages)
        and attempted == found
        and inserted_or_seen + excluded == attempted
        and verification.get("verification_pass") is True
    )


def count_search_ids(search_data: Sequence[bytes]) -> int:
    return len((search_data[0] or b"").split()) if search_data else 0


def batch_id(mailbox: str, since_days: int, before_days: int | None) -> str:
    label = safe_label(mailbox)[:50] or "mailbox"
    mailbox_hash = hashlib.sha256(mailbox.encode("utf-8")).hexdigest()[:8]
    if before_days is None:
        return f"{label}_{mailbox_hash}_d{since_days:03d}_open"
    return f"{label}_{mailbox_hash}_d{since_days:03d}_to_d{before_days:03d}"


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report: Mapping[str, Any], *, status: str) -> None:
    keys = (
        "schema_version",
        "created_at",
        "completed_at",
        "status",
        "today_utc",
        "folder_count",
        "since_days",
        "older_than_days",
        "planned_window_count",
        "planned_message_count",
        "max_window_count",
        "excluded_control_sha256_count",
        "archive_dir",
        "totals",
        "safety",
        "privacy",
    )
    payload = {key: report[key] for key in keys if key in report}
    payload["run_status"] = status
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--email")
    parser.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV)
    parser.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    parser.add_argument("--out-root", default=DEFAULT_ROOT)
    parser.add_argument("--identity-db")
    parser.add_argument("--since-days", type=int, default=60)
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=0,
        help="Only process windows older than this many days. Example: 60 with --since-days 180 processes 180..60 days ago.",
    )
    parser.add_argument("--max-messages", type=int, default=250)
    parser.add_argument("--mailbox", action="append")
    parser.add_argument("--exclude-archive-db", action="append")
    parser.add_argument("--internal-domain", action="append", default=["kmipt.ru"])
    parser.add_argument("--extracted-text-max-chars", type=int, default=250_000)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
