#!/usr/bin/env python3
"""Idempotent Mango Office capture staging.

This command is productization-only. It writes a capture manifest and optional
audio files under the selected output directory. It does not write runtime DBs,
does not start ASR/R+A and does not write to CRM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.capture_staging import (  # noqa: E402
    CaptureManifestStore,
    audit_capture_manifest,
    stage_capture_events,
)
from mango_mvp.productization.contracts import Direction, TelephonyCallEvent, TenantRef  # noqa: E402
from mango_mvp.productization.mango_office import MangoOfficePayloadMapper  # noqa: E402
from mango_mvp.productization.mango_office_client import (  # noqa: E402
    DEFAULT_MANGO_BASE_URL,
    MangoOfficeClient,
    MangoOfficeCredentials,
)
from mango_mvp.productization.mango_recordings import MangoRecordingDownloader  # noqa: E402


DEFAULT_OUT_ROOT = "_local_archive_mango_capture_stage_20260507"


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env_file()
    args = parse_args(argv)
    try:
        summary = run_capture_stage(args)
    except Exception as exc:
        print(f"capture stage failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("stage", {}).get("failed", 0) == 0 else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an idempotent Mango capture staging manifest.")
    parser.add_argument("--tenant", default="foton")
    parser.add_argument("--from-report", help="Use an existing missing-vs-audio JSON report instead of polling.")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--since", help="ISO datetime poll start.")
    parser.add_argument("--until", help="ISO datetime poll end.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", help="Recording directory. Defaults to <out-root>/recordings.")
    parser.add_argument("--manifest", help="Capture manifest JSONL. Defaults to <out-root>/capture_manifest.jsonl.")
    parser.add_argument("--audit-out", help="Optional audit JSON path. Defaults to <out-root>/capture_audit.json.")
    parser.add_argument("--base-url", default=os.getenv("MANGO_OFFICE_BASE_URL", DEFAULT_MANGO_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("MANGO_OFFICE_API_KEY"))
    parser.add_argument("--api-salt", default=os.getenv("MANGO_OFFICE_API_SALT"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=1.5)
    parser.add_argument("--link-retries", type=int, default=8)
    parser.add_argument("--rate-limit-sleep-sec", type=float, default=30.0)
    parser.add_argument("--timeout-sec", type=int, default=60)
    return parser.parse_args(argv)


def run_capture_stage(args: argparse.Namespace) -> Mapping[str, Any]:
    out_root = Path(args.out_root)
    recordings_dir = Path(args.out_dir) if args.out_dir else out_root / "recordings"
    manifest_path = Path(args.manifest) if args.manifest else out_root / "capture_manifest.jsonl"
    audit_path = Path(args.audit_out) if args.audit_out else out_root / "capture_audit.json"
    out_root.mkdir(parents=True, exist_ok=True)

    events = load_events(args)
    if args.limit is not None:
        events = events[: args.limit]

    downloader = None
    if not args.dry_run:
        if not args.api_key or not args.api_salt:
            raise ValueError("MANGO_OFFICE_API_KEY and MANGO_OFFICE_API_SALT are required")
        downloader = MangoRecordingDownloader(
            credentials=MangoOfficeCredentials(api_key=args.api_key, api_salt=args.api_salt),
            base_url=args.base_url,
            timeout_sec=args.timeout_sec,
            link_retries=args.link_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
        )

    manifest_store = CaptureManifestStore(manifest_path)
    stage_summary = stage_capture_events(
        events=events,
        manifest_store=manifest_store,
        recordings_dir=recordings_dir,
        downloader=downloader,
        dry_run=args.dry_run,
        sleep_sec=args.sleep_sec,
    )
    audit = audit_capture_manifest(manifest_path=manifest_path, recordings_dir=recordings_dir)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "stage": stage_summary.to_json_dict(),
        "audit": audit,
        "audit_path": str(audit_path),
    }


def load_events(args: argparse.Namespace) -> list[TelephonyCallEvent]:
    tenant = TenantRef(args.tenant)
    if args.from_report:
        return events_from_missing_report(tenant=tenant, path=Path(args.from_report))

    if not args.api_key or not args.api_salt:
        raise ValueError("MANGO_OFFICE_API_KEY and MANGO_OFFICE_API_SALT are required")
    since, until = resolve_window(args)
    client = MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key=args.api_key, api_salt=args.api_salt),
        base_url=args.base_url,
        timeout_sec=args.timeout_sec,
    )
    mapper = MangoOfficePayloadMapper()
    rows = client.poll_call_history(since=since, until=until)
    events = [mapper.from_payload(tenant=tenant, payload=row) for row in rows]
    return sorted(events, key=lambda event: (event.started_at, event.provider_call_id))


def events_from_missing_report(tenant: TenantRef, path: Path) -> list[TelephonyCallEvent]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in data.get("missing", []) if row.get("recording_ref")]
    events = [event_from_report_row(tenant, row) for row in rows]
    return sorted(events, key=lambda event: (event.started_at, event.provider_call_id))


def event_from_report_row(tenant: TenantRef, row: Mapping[str, Any]) -> TelephonyCallEvent:
    started_at = parse_datetime(str(row["started_at_utc"]))
    duration = row.get("duration_seconds")
    ended_at = None
    if duration is not None:
        try:
            ended_at = started_at + timedelta(seconds=int(duration))
        except (TypeError, ValueError):
            ended_at = None
    return TelephonyCallEvent(
        tenant=tenant,
        provider="mango",
        provider_call_id=str(row["provider_call_id"]),
        started_at=started_at,
        ended_at=ended_at,
        direction=parse_direction(row.get("direction")),
        client_phone=optional_str(row.get("client_phone")),
        manager_ref=optional_str(row.get("manager_ref")),
        recording_ref=optional_str(row.get("recording_ref")),
        raw_payload={},
    )


def resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    until = parse_datetime(args.until) if args.until else datetime.now(timezone.utc)
    since = parse_datetime(args.since) if args.since else until - timedelta(hours=args.hours)
    if until <= since:
        raise ValueError("--until must be later than --since")
    return since, until


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_direction(value: Any) -> Direction:
    try:
        return Direction(str(value or "").strip().lower())
    except ValueError:
        return Direction.UNKNOWN


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
