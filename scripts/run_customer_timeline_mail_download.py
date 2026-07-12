#!/usr/bin/env python3
"""Download INBOX and Sent into stable read-only IMAP archives."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mail_imap_snapshot import (  # noqa: E402
    ImapLibClient,
    parse_mailbox_list_line,
)
from scripts.mango_office_mail_archive import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    load_dotenv_file,
)

DEFAULT_SECRET = Path.home() / ".mango_secrets/mail_imap_edu_kmipt.env"
CANONICAL_RELATIVE_ROOT = Path("_external_handoffs/mail_archive_canonical_20260711")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_stats(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {"exists": False, "message_count": 0, "waterline": None, "sha256": None}
    with sqlite3.connect(path) as con:
        count, waterline = con.execute(
            "SELECT COUNT(*), MAX(message_date_iso) FROM messages"
        ).fetchone()
    return {
        "exists": True,
        "message_count": int(count or 0),
        "waterline": waterline,
        "sha256": sha256_file(path),
    }


def runtime_identity(root: Path) -> Mapping[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "worktree": git("rev-parse", "--show-toplevel"),
    }


def discover_required_mailboxes(
    *, host: str, port: int, email_address: str, password: str, sent_name: str
) -> Mapping[str, Mapping[str, str]]:
    imap = ImapLibClient(host=host, port=port)
    try:
        login_status, _ = imap.login(email_address, password)
        if login_status != "OK":
            raise RuntimeError("IMAP LOGIN failed")
        list_status, raw_boxes = imap.list()
        if list_status != "OK":
            raise RuntimeError("IMAP LIST failed")
        boxes = [parse_mailbox_list_line(line) for line in raw_boxes or []]
    finally:
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass

    def exactly_one(label: str, predicate: Any) -> Mapping[str, str]:
        matches = [box for box in boxes if predicate(box)]
        if len(matches) != 1:
            raise RuntimeError(f"mailbox_{label}_match_count={len(matches)}")
        return {"name": str(matches[0]["name"]), "raw": str(matches[0]["name_raw"])}

    return {
        "inbox": exactly_one("inbox", lambda box: str(box.get("name", "")).casefold() == "inbox"),
        "sent": exactly_one("sent", lambda box: str(box.get("name", "")) == sent_name),
    }


@contextmanager
def exclusive_lock(path: Path) -> Iterator[Mapping[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("mail_download_already_running") from exc
        yield {"path": str(path), "exclusive": True}


def run_ingest(
    *, root: Path, dotenv: Path, mailbox: Mapping[str, str], out_dir: Path, since_days: int
) -> tuple[int, Mapping[str, Any]]:
    command = [
        sys.executable,
        str(root / "scripts/mango_office_mail_archive.py"),
        "ingest",
        "--dotenv",
        str(dotenv),
        "--mailbox",
        mailbox["raw"],
        "--mailbox-label",
        mailbox["name"],
        "--since-days",
        str(since_days),
        "--max-messages",
        "0",
        "--allow-large-batch",
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"errors": [{"error": "ingest_output_not_json"}]}
    return completed.returncode, payload


def execute(args: argparse.Namespace) -> Mapping[str, Any]:
    root = Path(args.code_root).resolve()
    data_root = Path(args.data_root).resolve()
    state_dir = Path(args.state_dir).resolve()
    dotenv = Path(args.dotenv).expanduser().resolve()
    identity = runtime_identity(root)
    started_at = utc_now()
    manifest_path = state_dir / "mail_download_manifest.json"
    cursor_path = state_dir / "mail_download_cursor.json"
    lock_path = state_dir / "mail_pipeline.lock"

    if not args.apply:
        return {
            "schema_version": "mail_download_manifest_v1",
            "status": "dry_run",
            "started_at": started_at,
            "finished_at": utc_now(),
            "runtime": identity,
            "mailboxes": ["INBOX", args.sent_mailbox_name],
            "overlap_days": args.since_days,
            "max_messages": None,
            "network_calls": False,
            "write_external_systems": False,
            "paths": {"manifest": str(manifest_path), "cursor": str(cursor_path)},
        }

    if not dotenv.is_file():
        raise RuntimeError("mail_secret_file_missing")
    if stat.S_IMODE(dotenv.stat().st_mode) & 0o077:
        raise RuntimeError("mail_secret_file_permissions_too_open")
    for name in ("MAIL_IMAP_EMAIL", "MAIL_IMAP_PASSWORD", "MAIL_IMAP_HOST", "MAIL_IMAP_PORT"):
        os.environ.pop(name, None)
    load_dotenv_file(dotenv)
    email_address = os.environ.get("MAIL_IMAP_EMAIL", "")
    password = os.environ.get("MAIL_IMAP_PASSWORD", "")
    host = os.environ.get("MAIL_IMAP_HOST", DEFAULT_HOST)
    port = int(os.environ.get("MAIL_IMAP_PORT", DEFAULT_PORT))
    if not email_address or not password:
        raise RuntimeError("mail_credentials_missing")

    with exclusive_lock(lock_path) as lock:
        discovered = discover_required_mailboxes(
            host=host,
            port=port,
            email_address=email_address,
            password=password,
            sent_name=args.sent_mailbox_name,
        )
        reports: dict[str, Any] = {}
        archive_paths: list[str] = []
        errors = 0
        truncated = False
        for key in ("inbox", "sent"):
            out_dir = data_root / CANONICAL_RELATIVE_ROOT / "incoming/regru_edu" / key
            db_path = out_dir / "mail_archive.sqlite"
            before = archive_stats(db_path)
            rc, ingest = run_ingest(
                root=root,
                dotenv=dotenv,
                mailbox=discovered[key],
                out_dir=out_dir,
                since_days=args.since_days,
            )
            after = archive_stats(db_path)
            report_errors = len(ingest.get("errors") or [])
            report_truncated = bool(ingest.get("selection_truncated"))
            errors += int(rc != 0) + report_errors
            truncated = truncated or report_truncated
            archive_paths.append(str(db_path))
            reports[key] = {
                "mailbox": discovered[key]["name"],
                "status": "ok" if rc == 0 and not report_errors and not report_truncated else "failed",
                "searched": int(ingest.get("messages_found_since") or 0),
                "attempted": int(ingest.get("messages_attempted") or 0),
                "inserted": max(0, int(after["message_count"]) - int(before["message_count"])),
                "duplicates": max(
                    0,
                    int(ingest.get("messages_inserted_or_seen") or 0)
                    - max(0, int(after["message_count"]) - int(before["message_count"])),
                ),
                "errors": report_errors + int(rc != 0),
                "truncated": report_truncated,
                "archive_db": str(db_path),
                "waterline_before": before["waterline"],
                "waterline_after": after["waterline"],
                "message_count_before": before["message_count"],
                "message_count_after": after["message_count"],
                "sha256_after": after["sha256"],
            }

        status = "ok" if errors == 0 and not truncated and len(reports) == 2 else "failed"
        manifest = {
            "schema_version": "mail_download_manifest_v1",
            "status": status,
            "started_at": started_at,
            "finished_at": utc_now(),
            "runtime": identity,
            "cursor_kind": "overlap_waterline_sha",
            "overlap_days": args.since_days,
            "max_messages": None,
            "mailbox_reports": reports,
            "archive_db_paths": archive_paths,
            "errors": errors,
            "truncated": truncated,
            "lock": lock,
            "write_external_systems": False,
            "paths": {"manifest": str(manifest_path), "cursor": str(cursor_path)},
        }
        atomic_write_json(manifest_path, manifest)
        if status == "ok":
            atomic_write_json(
                cursor_path,
                {
                    "schema_version": "mail_download_cursor_v1",
                    "cursor_kind": "overlap_waterline_sha",
                    "last_success_at": manifest["finished_at"],
                    "runtime": identity,
                    "mailboxes": {
                        key: {
                            "waterline": reports[key]["waterline_after"],
                            "message_count": reports[key]["message_count_after"],
                            "sha256": reports[key]["sha256_after"],
                        }
                        for key in ("inbox", "sent")
                    },
                },
            )
        return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform read-only IMAP downloads.")
    parser.add_argument("--code-root", default=str(ROOT))
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--state-dir", default=str(ROOT / ".codex_local/staging/mail_pipeline"))
    parser.add_argument("--dotenv", default=str(DEFAULT_SECRET))
    parser.add_argument("--sent-mailbox-name", default="Sent")
    parser.add_argument("--since-days", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = execute(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        if str(exc) == "mail_download_already_running":
            print(json.dumps({"status": "already_running", "stop_reason": "already_running"}, sort_keys=True))
            return 75
        print(json.dumps({"status": "failed", "error": type(exc).__name__}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
