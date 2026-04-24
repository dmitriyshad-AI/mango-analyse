from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll subset DB and print progress snapshots with ETA."
    )
    parser.add_argument("--db", required=True, help="SQLite DB path")
    parser.add_argument("--interval-sec", type=int, default=300, help="Polling interval")
    parser.add_argument(
        "--eta-stage",
        choices=["resolve", "analysis"],
        default="analysis",
        help="Stage to use for ETA calculation",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=12,
        help="How many recent samples to keep for rate/ETA",
    )
    return parser.parse_args()


def fetch_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    counters: dict[str, dict[str, int]] = {}
    for col in ("resolve_status", "analysis_status"):
        rows = conn.execute(
            f"SELECT COALESCE({col}, 'NULL'), COUNT(*) FROM call_records GROUP BY {col}"
        ).fetchall()
        counters[col] = {str(k): int(v) for k, v in rows}
    total = int(conn.execute("SELECT COUNT(*) FROM call_records").fetchone()[0] or 0)
    return {"total": total, **counters}


def stage_done_equivalent(stage: str, counts: dict[str, Any]) -> int:
    block = counts[f"{stage}_status"]
    if stage == "resolve":
        return int(block.get("done", 0)) + int(block.get("skipped", 0)) + int(block.get("manual", 0))
    return int(block.get("done", 0))


def stage_in_progress(stage: str, counts: dict[str, Any]) -> int:
    return int(counts[f"{stage}_status"].get("in_progress", 0))


def stage_remaining(stage: str, counts: dict[str, Any]) -> int:
    total = int(counts["total"])
    return max(0, total - stage_done_equivalent(stage, counts))


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def estimate_eta(samples: deque[tuple[float, int]], remaining: int) -> tuple[float | None, float | None]:
    if len(samples) < 2:
        return None, None
    t0, d0 = samples[0]
    t1, d1 = samples[-1]
    dt = max(0.0, t1 - t0)
    dd = d1 - d0
    if dt <= 0 or dd <= 0:
        return None, None
    rate = dd / dt
    return remaining / rate if rate > 0 else None, rate


def main() -> int:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    conn = sqlite3.connect(db_path)
    history: deque[tuple[float, int]] = deque(maxlen=max(2, args.history_points))

    while True:
        counts = fetch_counts(conn)
        now = time.time()
        done_equiv = stage_done_equivalent(args.eta_stage, counts)
        remaining = stage_remaining(args.eta_stage, counts)
        in_progress = stage_in_progress(args.eta_stage, counts)
        history.append((now, done_equiv))
        eta_sec, rate = estimate_eta(history, remaining)

        payload = {
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "db": str(db_path),
            "total": int(counts["total"]),
            "resolve": counts["resolve_status"],
            "analysis": counts["analysis_status"],
            "eta_stage": args.eta_stage,
            "done_equivalent": done_equiv,
            "remaining": remaining,
            "in_progress": in_progress,
            "rate_items_per_min": round(rate * 60, 2) if rate is not None else None,
            "eta": format_eta(eta_sec),
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)

        if args.eta_stage == "analysis" and remaining == 0 and in_progress == 0:
            break
        if args.eta_stage == "resolve" and remaining == 0 and in_progress == 0:
            break
        time.sleep(max(1, int(args.interval_sec)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
