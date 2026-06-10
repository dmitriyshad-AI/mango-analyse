from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mango_mvp.customer_profile.builder import CustomerProfileBuilder, CustomerProfileBuildOptions


_LIVE_JOURNAL_COPY_SOURCE = Path.home() / ".mango_local" / ("draft" + "_loop") / "journal.jsonl"
DEFAULT_QUIET_MINUTES = 30


def parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def detect_quiet_dialogs(
    journal_rows: Iterable[Mapping[str, Any]],
    now_utc: datetime,
    quiet_minutes: int = DEFAULT_QUIET_MINUTES,
) -> list[tuple[str, str]]:
    now = parse_iso_datetime(now_utc)
    if now is None:
        raise ValueError("now_utc must be a valid timezone-aware datetime")
    last: dict[tuple[str, str], datetime] = {}
    for row in journal_rows:
        if not isinstance(row, Mapping):
            continue
        profile_id = clean_text(row.get("profile_id"))
        chat_id = clean_text(row.get("chat_id"))
        ts = parse_iso_datetime(row.get("created_at"))
        if not profile_id or not chat_id or ts is None:
            continue
        key = (profile_id, chat_id)
        last[key] = max(last.get(key, ts), ts)
    threshold = timedelta(minutes=quiet_minutes)
    return [key for key, ts in last.items() if (now - ts) >= threshold]


def read_journal_jsonl(path: Path) -> tuple[list[Mapping[str, Any]], int]:
    if _same_path(path, _LIVE_JOURNAL_COPY_SOURCE):
        raise ValueError("--from-journal must point to a copied journal.jsonl, not the live journal")
    rows: list[Mapping[str, Any]] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            if isinstance(parsed, Mapping):
                rows.append(parsed)
            else:
                bad_rows += 1
    return rows, bad_rows


def select_customer_ids_since(timeline_db: Path, *, tenant_id: str, since: datetime) -> list[str]:
    since_utc = parse_iso_datetime(since)
    if since_utc is None:
        raise ValueError("since must be a valid ISO timestamp")
    con = _connect_read_only(timeline_db)
    try:
        rows = con.execute(
            """
            SELECT customer_id, created_at
            FROM timeline_events
            WHERE tenant_id = ?
              AND customer_id IS NOT NULL
              AND customer_id <> ''
              AND created_at IS NOT NULL
            ORDER BY created_at, customer_id
            """,
            (tenant_id,),
        ).fetchall()
    finally:
        con.close()
    selected: list[str] = []
    for row in rows:
        created_at = parse_iso_datetime(row["created_at"])
        if created_at and created_at > since_utc:
            selected.append(str(row["customer_id"]))
    return sorted(dict.fromkeys(selected))


def collect_journal_dialog_activity(journal_rows: Iterable[Mapping[str, Any]]) -> tuple[dict[tuple[str, str], datetime], int]:
    last: dict[tuple[str, str], datetime] = {}
    ignored = 0
    for row in journal_rows:
        if not isinstance(row, Mapping):
            ignored += 1
            continue
        profile_id = clean_text(row.get("profile_id"))
        chat_id = clean_text(row.get("chat_id"))
        ts = parse_iso_datetime(row.get("created_at"))
        if not profile_id or not chat_id or ts is None:
            ignored += 1
            continue
        key = (profile_id, chat_id)
        last[key] = max(last.get(key, ts), ts)
    return last, ignored


def resolve_quiet_dialog_customer_ids(
    timeline_db: Path,
    *,
    tenant_id: str,
    quiet_pairs: Sequence[tuple[str, str]],
    last_seen: Mapping[tuple[str, str], datetime],
    now_utc: datetime,
) -> tuple[list[str], list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    now = parse_iso_datetime(now_utc)
    if now is None:
        raise ValueError("now_utc must be a valid timezone-aware datetime")
    con = _connect_read_only(timeline_db)
    selected: list[str] = []
    matched: list[Mapping[str, Any]] = []
    unmatched: list[Mapping[str, Any]] = []
    ambiguous: list[Mapping[str, Any]] = []
    try:
        for journal_profile_id, chat_id in quiet_pairs:
            customer_ids = _customer_ids_for_chat(con, tenant_id=tenant_id, chat_id=chat_id)
            last_at = last_seen.get((journal_profile_id, chat_id))
            item = {
                "journal_profile_id": journal_profile_id,
                "chat_id": chat_id,
                "last_created_at": last_at.isoformat() if last_at else None,
                "quiet_for_minutes": int((now - last_at).total_seconds() // 60) if last_at else None,
            }
            if len(customer_ids) == 1:
                customer_id = customer_ids[0]
                selected.append(customer_id)
                matched.append({**item, "customer_id": customer_id})
            elif len(customer_ids) > 1:
                ambiguous.append({**item, "customer_ids": customer_ids})
            else:
                unmatched.append(item)
    finally:
        con.close()
    return list(dict.fromkeys(selected)), matched, unmatched, ambiguous


def refresh_since(
    *,
    timeline_db: Path,
    profiles_db: Path,
    master_calls_db: Path | None,
    tenant_id: str,
    since: datetime,
    build_id: str | None = None,
) -> Mapping[str, Any]:
    selected = select_customer_ids_since(timeline_db, tenant_id=tenant_id, since=since)
    build = rebuild_selected_profiles(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=master_calls_db,
        tenant_id=tenant_id,
        customer_ids=selected,
        build_id=build_id,
    )
    return {
        "mode": "since",
        "tenant_id": tenant_id,
        "timeline_db": str(timeline_db),
        "profiles_db": str(profiles_db),
        "master_calls_db": str(master_calls_db) if master_calls_db else None,
        "since": since.astimezone(timezone.utc).isoformat(),
        "selected_customer_ids": selected,
        "selected_customer_count": len(selected),
        "build": build,
    }


def refresh_from_journal(
    *,
    timeline_db: Path,
    profiles_db: Path,
    master_calls_db: Path | None,
    tenant_id: str,
    journal_path: Path,
    now_utc: datetime,
    quiet_minutes: int = DEFAULT_QUIET_MINUTES,
    build_id: str | None = None,
) -> Mapping[str, Any]:
    rows, bad_rows = read_journal_jsonl(journal_path)
    last_seen, ignored_rows = collect_journal_dialog_activity(rows)
    quiet_pairs = detect_quiet_dialogs(rows, now_utc, quiet_minutes=quiet_minutes)
    selected, matched, unmatched, ambiguous = resolve_quiet_dialog_customer_ids(
        timeline_db,
        tenant_id=tenant_id,
        quiet_pairs=quiet_pairs,
        last_seen=last_seen,
        now_utc=now_utc,
    )
    build = rebuild_selected_profiles(
        timeline_db=timeline_db,
        profiles_db=profiles_db,
        master_calls_db=master_calls_db,
        tenant_id=tenant_id,
        customer_ids=selected,
        build_id=build_id,
    )
    return {
        "mode": "from_journal",
        "tenant_id": tenant_id,
        "timeline_db": str(timeline_db),
        "profiles_db": str(profiles_db),
        "master_calls_db": str(master_calls_db) if master_calls_db else None,
        "journal_path": str(journal_path),
        "journal_rows_loaded": len(rows),
        "journal_rows_bad": bad_rows,
        "journal_rows_ignored": ignored_rows,
        "quiet_minutes": quiet_minutes,
        "quiet_pairs": [
            {
                "journal_profile_id": profile_id,
                "chat_id": chat_id,
                "last_created_at": last_seen.get((profile_id, chat_id)).isoformat()
                if last_seen.get((profile_id, chat_id))
                else None,
            }
            for profile_id, chat_id in quiet_pairs
        ],
        "matched_quiet_pairs": matched,
        "unmatched_quiet_pairs": unmatched,
        "ambiguous_quiet_pairs": ambiguous,
        "selected_customer_ids": selected,
        "selected_customer_count": len(selected),
        "build": build,
    }


def rebuild_selected_profiles(
    *,
    timeline_db: Path,
    profiles_db: Path,
    master_calls_db: Path | None,
    tenant_id: str,
    customer_ids: Sequence[str],
    build_id: str | None = None,
) -> Mapping[str, Any] | None:
    selected = tuple(dict.fromkeys(clean_text(item) for item in customer_ids if clean_text(item)))
    if not selected:
        return None
    return CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_calls_db,
            tenant_id=tenant_id,
            customer_ids=selected,
            build_id=build_id,
        )
    ).build()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh deterministic customer profiles from timeline events.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--profiles-db", required=True)
    parser.add_argument("--master-calls-db")
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--build-id")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--since", help="Rebuild customers with timeline_events.created_at greater than this ISO timestamp.")
    mode.add_argument("--from-journal", help="Read a copied journal.jsonl and rebuild quiet matched dialogs.")
    parser.add_argument("--now-utc", help="ISO timestamp for journal quiet detection. Defaults to current UTC time.")
    parser.add_argument("--quiet-minutes", type=int, default=DEFAULT_QUIET_MINUTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timeline_db = Path(args.timeline_db)
    profiles_db = Path(args.profiles_db)
    master_calls_db = Path(args.master_calls_db) if args.master_calls_db else None
    if args.since:
        since = parse_iso_datetime(args.since)
        if since is None:
            raise SystemExit("--since must be a valid ISO timestamp")
        report = refresh_since(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_calls_db,
            tenant_id=args.tenant_id,
            since=since,
            build_id=args.build_id,
        )
    else:
        now = parse_iso_datetime(args.now_utc) if args.now_utc else datetime.now(timezone.utc)
        if now is None:
            raise SystemExit("--now-utc must be a valid ISO timestamp")
        report = refresh_from_journal(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_calls_db,
            tenant_id=args.tenant_id,
            journal_path=Path(args.from_journal),
            now_utc=now,
            quiet_minutes=args.quiet_minutes,
            build_id=args.build_id,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _customer_ids_for_chat(con: sqlite3.Connection, *, tenant_id: str, chat_id: str) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT customer_id
        FROM identity_links
        WHERE tenant_id = ?
          AND customer_id IS NOT NULL
          AND customer_id <> ''
          AND (
            (link_type = 'telegram_user_id' AND link_value = ?)
            OR (link_type = 'channel_session_id' AND link_value = ?)
          )
        ORDER BY customer_id
        """,
        (tenant_id, chat_id, f"telegram:{chat_id}"),
    ).fetchall()
    return [str(row["customer_id"]) for row in rows]


def _connect_read_only(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except FileNotFoundError:
        return left.expanduser().absolute() == right.expanduser().absolute()


def clean_text(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
