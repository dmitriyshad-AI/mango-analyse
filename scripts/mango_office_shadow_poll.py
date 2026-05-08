#!/usr/bin/env python3
"""Read-only Mango Office shadow polling report.

This script does not download audio, does not write runtime DBs, does not start
ASR/R+A and does not write to AMO/Tallanto.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.capture import (  # noqa: E402
    CaptureDecision,
    CapturePlanner,
    InMemorySeenCallStore,
)
from mango_mvp.productization.contracts import TenantRef  # noqa: E402
from mango_mvp.productization.mango_office import MangoOfficePayloadMapper  # noqa: E402
from mango_mvp.productization.mango_office_client import (  # noqa: E402
    DEFAULT_MANGO_BASE_URL,
    MangoOfficeClient,
    MangoOfficeCredentials,
)
from mango_mvp.productization.payload_archive import write_shadow_poll_raw_payload_jsonl  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
    except Exception as exc:
        print(f"mango shadow poll failed: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(f"{text}\n", encoding="utf-8")
    else:
        print(text)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="Build a read-only Mango Office shadow polling report.")
    parser.add_argument("--tenant", required=True, help="Logical tenant id, e.g. foton.")
    parser.add_argument("--hours", type=float, default=2.0, help="Lookback window in hours.")
    parser.add_argument("--since", help="ISO datetime override for poll start.")
    parser.add_argument("--until", help="ISO datetime override for poll end.")
    parser.add_argument("--base-url", default=os.getenv("MANGO_OFFICE_BASE_URL", DEFAULT_MANGO_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("MANGO_OFFICE_API_KEY"))
    parser.add_argument("--api-salt", default=os.getenv("MANGO_OFFICE_API_SALT"))
    parser.add_argument("--seen-keys", help="Optional newline file with already seen event keys.")
    parser.add_argument("--allow-metadata-only", action="store_true", help="Do not require recording refs.")
    parser.add_argument("--raw-payload-jsonl", help="Optional JSONL path for raw Mango stats rows.")
    parser.add_argument("--out", help="Optional JSON report path.")
    return parser.parse_args(argv)


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def build_report(args: argparse.Namespace) -> Mapping[str, Any]:
    if not args.api_key or not args.api_salt:
        raise ValueError("MANGO_OFFICE_API_KEY and MANGO_OFFICE_API_SALT are required")

    since, until = resolve_window(args)
    tenant = TenantRef(args.tenant)
    client = MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key=args.api_key, api_salt=args.api_salt),
        base_url=args.base_url,
    )
    mapper = MangoOfficePayloadMapper()
    rows = client.poll_call_history(since=since, until=until)
    raw_payload_rows = write_shadow_poll_rows_if_requested(args, rows, since=since, until=until)

    events = []
    errors = []
    for index, row in enumerate(rows):
        try:
            events.append(mapper.from_payload(tenant=tenant, payload=row))
        except Exception as exc:
            errors.append({"index": index, "error": str(exc), "payload": row})

    planner = CapturePlanner(
        seen_store=InMemorySeenCallStore(read_seen_keys(args.seen_keys)),
        require_recording=not args.allow_metadata_only,
    )
    decisions = planner.plan_batch(events)

    return {
        "tenant_id": tenant.tenant_id,
        "window": {"since": since.isoformat(), "until": until.isoformat()},
        "source": {"provider": "mango", "base_url": args.base_url},
        "raw_payload_archive": raw_payload_rows,
        "counts": summarize(decisions=decisions, normalization_errors=errors),
        "decisions": [decision_to_dict(decision) for decision in decisions],
        "normalization_errors": errors,
    }


def write_shadow_poll_rows_if_requested(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    since: datetime,
    until: datetime,
) -> Mapping[str, Any]:
    out_path = getattr(args, "raw_payload_jsonl", None)
    if not out_path:
        return {"enabled": False}
    written = write_shadow_poll_raw_payload_jsonl(
        rows=rows,
        out_path=Path(out_path),
        tenant_id=args.tenant,
        provider="mango",
        base_url=args.base_url,
        since=since.isoformat(),
        until=until.isoformat(),
    )
    return {"enabled": True, "path": str(out_path), "rows": written}


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


def read_seen_keys(path: Optional[str]) -> Iterable[str]:
    if not path:
        return ()
    seen_path = Path(path)
    if not seen_path.exists():
        return ()
    return tuple(line.strip() for line in seen_path.read_text(encoding="utf-8").splitlines() if line.strip())


def summarize(
    decisions: Sequence[CaptureDecision],
    normalization_errors: Sequence[Mapping[str, Any]],
) -> Mapping[str, int]:
    counts = {
        "source_rows": len(decisions) + len(normalization_errors),
        "normalized_events": len(decisions),
        "normalization_errors": len(normalization_errors),
        "enqueue_shadow_capture": 0,
        "skip_duplicate": 0,
        "skip_no_recording": 0,
    }
    for decision in decisions:
        counts[decision.action.value] = counts.get(decision.action.value, 0) + 1
    return counts


def decision_to_dict(decision: CaptureDecision) -> Mapping[str, Any]:
    candidate = asdict(decision.candidate) if decision.candidate else None
    if candidate is not None:
        candidate["started_at"] = decision.candidate.started_at.isoformat()
        candidate["direction"] = decision.candidate.direction.value
    return {
        "action": decision.action.value,
        "reason": decision.reason,
        "event": {
            "event_key": decision.event.event_key,
            "provider_call_id": decision.event.provider_call_id,
            "started_at": decision.event.started_at.isoformat(),
            "ended_at": decision.event.ended_at.isoformat() if decision.event.ended_at else None,
            "duration_seconds": decision.event.duration_seconds,
            "direction": decision.event.direction.value,
            "client_phone": decision.event.client_phone,
            "manager_ref": decision.event.manager_ref,
            "recording_ref": decision.event.recording_ref,
            "recording_url": decision.event.recording_url,
        },
        "candidate": candidate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
