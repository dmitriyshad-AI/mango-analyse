#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


DEFAULT_TIMELINE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango-analyse-tz16/product_data/customer_profiles/tz16_profiles_v7_20260612/customer_timeline.sqlite"
)
DEFAULT_MASTER_CALLS_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
)
DEFAULT_RERUN_PACKAGE = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611"
)
DEFAULT_OUT = Path(
    "/Users/dmitrijfabarisov/Projects/Mango-analyse-tz16/product_data/customer_profiles/tz16_profiles_v7_20260612/rerun_tail_report.json"
)
ACTIVE_AMO_EXCLUDED_STATUSES = ("Закрыто и не реализовано", "Успешно")
RERUN_SINCE = date(2025, 6, 1)
MIN_DURATION_SEC = 60
CURRENT_PROMPT_VERSION = "v7"


@dataclass(frozen=True)
class RerunTailConfig:
    timeline_db: Path
    master_calls_db: Path
    rerun_package: Path
    out: Path | None = None
    as_of: date = date(2026, 6, 12)


@dataclass(frozen=True)
class CallRow:
    call_id: int
    started_at: str
    duration_sec: int
    analysis_status: str
    transcript_chars: int
    prompt_version: str


def compute_rerun_tail(config: RerunTailConfig) -> Mapping[str, Any]:
    config = RerunTailConfig(
        timeline_db=config.timeline_db.expanduser().resolve(strict=False),
        master_calls_db=config.master_calls_db.expanduser().resolve(strict=False),
        rerun_package=config.rerun_package.expanduser().resolve(strict=False),
        out=config.out.expanduser().resolve(strict=False) if config.out else None,
        as_of=config.as_of,
    )
    zone = load_zone(config.timeline_db)
    target_ids = read_int_set(config.rerun_package / "data" / "ids_all.txt")
    blacklist_ids = read_int_set(config.rerun_package / "blacklist_77.txt")
    manifest = read_json(config.rerun_package / "data" / "manifest.json")
    execution_estimate = estimate_execution(config.rerun_package)
    rows = load_call_rows(config.master_calls_db, sorted(zone["call_ids"]))
    row_ids = {row.call_id for row in rows}
    missing_in_master = sorted(zone["call_ids"] - row_ids)
    target_ids_in_zone = target_ids & zone["call_ids"]
    blacklist_ids_in_zone = blacklist_ids & zone["call_ids"]

    current_v7 = [row for row in rows if row.prompt_version == CURRENT_PROMPT_VERSION]
    current_not_v7 = [row for row in rows if row.prompt_version != CURRENT_PROMPT_VERSION]
    eligible_tail = [
        row
        for row in current_not_v7
        if row.call_id not in blacklist_ids
        and row.analysis_status == "done"
        and row.duration_sec >= MIN_DURATION_SEC
        and parse_date(row.started_at) >= RERUN_SINCE
        and row.transcript_chars > 0
    ]
    non_blacklist_old_tail = [
        row
        for row in current_not_v7
        if row.call_id not in blacklist_ids and row.analysis_status == "done" and row.transcript_chars > 0
    ]
    ge60_non_blacklist_old_tail = [row for row in non_blacklist_old_tail if row.duration_sec >= MIN_DURATION_SEC]
    ge60_old_tail_with_blacklist = [
        row
        for row in current_not_v7
        if row.analysis_status == "done" and row.transcript_chars > 0 and row.duration_sec >= MIN_DURATION_SEC
    ]
    tail_reasons = classify_tail_reasons(current_not_v7, target_ids=target_ids, blacklist_ids=blacklist_ids)
    result = {
        "schema_version": "tz16_rerun_tail_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "timeline_db": str(config.timeline_db),
            "master_calls_db": str(config.master_calls_db),
            "rerun_package": str(config.rerun_package),
            "as_of": config.as_of.isoformat(),
            "active_amo_excluded_statuses": list(ACTIVE_AMO_EXCLUDED_STATUSES),
            "rerun_since": RERUN_SINCE.isoformat(),
            "min_duration_sec": MIN_DURATION_SEC,
        },
        "zone": {
            "active_amo_customers": zone["active_amo_customers"],
            "strong_tallanto_student_customers": zone["strong_tallanto_student_customers"],
            "union_customers": zone["union_customers"],
            "zone_call_count": len(zone["call_ids"]),
            "zone_calls_missing_in_master": len(missing_in_master),
        },
        "v7_first_slice": {
            "target_ids_total": len(target_ids),
            "target_ids_in_zone": len(target_ids_in_zone),
            "blacklist_ids_total": len(blacklist_ids),
            "blacklist_ids_in_zone": len(blacklist_ids_in_zone),
            "zone_calls_current_v7": len(current_v7),
            "zone_calls_current_not_v7": len(current_not_v7),
            "target_ids_current_v7": sum(1 for row in rows if row.call_id in target_ids and row.prompt_version == CURRENT_PROMPT_VERSION),
            "target_ids_current_not_v7": sum(1 for row in rows if row.call_id in target_ids and row.prompt_version != CURRENT_PROMPT_VERSION),
        },
        "tail": {
            "old_summary_calls": len(current_not_v7),
            "old_summary_transcript_chars": sum(row.transcript_chars for row in current_not_v7),
            "non_blacklist_old_summary_calls": len(non_blacklist_old_tail),
            "non_blacklist_old_summary_transcript_chars": sum(row.transcript_chars for row in non_blacklist_old_tail),
            "ge60_old_summary_calls_including_blacklist": len(ge60_old_tail_with_blacklist),
            "ge60_old_summary_transcript_chars_including_blacklist": sum(
                row.transcript_chars for row in ge60_old_tail_with_blacklist
            ),
            "ge60_non_blacklist_old_summary_calls": len(ge60_non_blacklist_old_tail),
            "ge60_non_blacklist_old_summary_transcript_chars": sum(row.transcript_chars for row in ge60_non_blacklist_old_tail),
            "eligible_second_slice_calls": len(eligible_tail),
            "eligible_second_slice_transcript_chars": sum(row.transcript_chars for row in eligible_tail),
            "eligible_second_slice_duration_sec": sum(row.duration_sec for row in eligible_tail),
            "reasons": tail_reasons,
            "reason_quadrants": tail_reason_quadrants(current_not_v7, blacklist_ids=blacklist_ids),
            "by_recency": bucket_recency(current_not_v7, as_of=config.as_of),
            "by_duration": bucket_duration(current_not_v7),
        },
        "estimate": {
            "same_filter_second_slice": estimate_slice(eligible_tail, execution_estimate, manifest),
            "ge60_old_tail_including_blacklist": estimate_slice(ge60_old_tail_with_blacklist, execution_estimate, manifest),
            "ge60_old_tail_excluding_blacklist": estimate_slice(ge60_non_blacklist_old_tail, execution_estimate, manifest),
            "all_non_blacklist_old_tail": estimate_slice(non_blacklist_old_tail, execution_estimate, manifest),
        },
        "observed_first_slice": {
            "manifest_rows": int(manifest.get("rows") or 0),
            "manifest_transcript_chars_sum": int(manifest.get("transcript_chars_sum") or 0),
            "execution_estimate": execution_estimate,
        },
        "safety": {
            "write_crm": False,
            "write_tallanto": False,
            "write_amo": False,
            "run_asr": False,
            "run_resolve_analyze": False,
            "llm_calls": 0,
        },
    }
    if config.out:
        write_json(config.out, result)
    return result


def load_zone(timeline_db: Path) -> Mapping[str, Any]:
    con = connect_read_only(timeline_db)
    try:
        active_amo = {
            str(row[0])
            for row in con.execute(
                f"""
                SELECT DISTINCT customer_id
                FROM customer_opportunities
                WHERE source_system = 'amocrm_snapshot'
                  AND opportunity_type = 'amo_deal'
                  AND COALESCE(closed_at, '') = ''
                  AND status NOT IN ({",".join("?" for _ in ACTIVE_AMO_EXCLUDED_STATUSES)})
                """,
                ACTIVE_AMO_EXCLUDED_STATUSES,
            ).fetchall()
        }
        strong_tallanto = {
            str(row[0])
            for row in con.execute(
                """
                SELECT DISTINCT customer_id
                FROM identity_links
                WHERE source_system = 'tallanto_snapshot'
                  AND link_type = 'tallanto_student_id'
                  AND match_class = 'strong_unique'
                """
            ).fetchall()
        }
        zone_customers = active_amo | strong_tallanto
        call_ids: set[int] = set()
        if zone_customers:
            for batch in chunked(sorted(zone_customers), 800):
                placeholders = ",".join("?" for _ in batch)
                rows = con.execute(
                    f"""
                    SELECT source_id
                    FROM timeline_events
                    WHERE event_type = 'mango_call'
                      AND source_system = 'mango_processed_summary'
                      AND customer_id IN ({placeholders})
                    """,
                    tuple(batch),
                ).fetchall()
                call_ids.update(int(str(row[0])) for row in rows if str(row[0]).isdigit())
    finally:
        con.close()
    return {
        "active_amo_customers": len(active_amo),
        "strong_tallanto_student_customers": len(strong_tallanto),
        "union_customers": len(zone_customers),
        "call_ids": call_ids,
    }


def load_call_rows(master_calls_db: Path, call_ids: Sequence[int]) -> list[CallRow]:
    if not call_ids:
        return []
    con = connect_read_only(master_calls_db)
    con.row_factory = sqlite3.Row
    try:
        table = "canonical_calls" if table_exists(con, "canonical_calls") else "call_records"
        id_column = "canonical_call_id" if has_column(con, table, "canonical_call_id") else "id"
        con.execute("CREATE TEMP TABLE tz16_zone_call_ids(call_id INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO tz16_zone_call_ids(call_id) VALUES (?)", [(int(item),) for item in call_ids])
        rows = con.execute(
            f"""
            SELECT CAST({id_column} AS INTEGER) AS call_id,
                   COALESCE(started_at, '') AS started_at,
                   COALESCE(duration_sec, 0) AS duration_sec,
                   COALESCE(analysis_status, '') AS analysis_status,
                   COALESCE(transcript_chars, LENGTH(COALESCE(transcript_text, '')), 0) AS transcript_chars,
                   CASE
                     WHEN analysis_json IS NOT NULL
                      AND json_valid(analysis_json)
                     THEN COALESCE(json_extract(analysis_json, '$.analysis_meta.analysis_prompt_version'), '')
                     ELSE ''
                   END AS prompt_version
            FROM {table}
            JOIN tz16_zone_call_ids z ON z.call_id = CAST({id_column} AS INTEGER)
            ORDER BY CAST({id_column} AS INTEGER)
            """
        ).fetchall()
    finally:
        con.close()
    return [
        CallRow(
            call_id=int(row["call_id"]),
            started_at=str(row["started_at"] or ""),
            duration_sec=int(row["duration_sec"] or 0),
            analysis_status=str(row["analysis_status"] or ""),
            transcript_chars=int(row["transcript_chars"] or 0),
            prompt_version=str(row["prompt_version"] or ""),
        )
        for row in rows
    ]


def classify_tail_reasons(
    rows: Iterable[CallRow],
    *,
    target_ids: set[int],
    blacklist_ids: set[int],
) -> Mapping[str, int]:
    counters = {
        "blacklist_preserved": 0,
        "targeted_but_not_current_v7": 0,
        "before_2025_06_01": 0,
        "below_60_sec": 0,
        "not_done_or_empty_transcript": 0,
        "eligible_not_in_first_slice": 0,
    }
    for row in rows:
        started = parse_date(row.started_at)
        if row.call_id in blacklist_ids:
            counters["blacklist_preserved"] += 1
        elif row.call_id in target_ids:
            counters["targeted_but_not_current_v7"] += 1
        elif started < RERUN_SINCE:
            counters["before_2025_06_01"] += 1
        elif row.duration_sec < MIN_DURATION_SEC:
            counters["below_60_sec"] += 1
        elif row.analysis_status != "done" or row.transcript_chars <= 0:
            counters["not_done_or_empty_transcript"] += 1
        else:
            counters["eligible_not_in_first_slice"] += 1
    return counters


def tail_reason_quadrants(rows: Iterable[CallRow], *, blacklist_ids: set[int]) -> Mapping[str, Mapping[str, int]]:
    keys = (
        "recent_ge_2025_06_01_ge_60_blacklist",
        "recent_ge_2025_06_01_ge_60_non_blacklist",
        "recent_ge_2025_06_01_lt_60",
        "old_lt_2025_06_01_ge_60",
        "old_lt_2025_06_01_lt_60",
    )
    counters = {key: {"calls": 0, "transcript_chars": 0} for key in keys}
    for row in rows:
        recent = parse_date(row.started_at) >= RERUN_SINCE
        ge60 = row.duration_sec >= MIN_DURATION_SEC
        if recent and ge60 and row.call_id in blacklist_ids:
            key = "recent_ge_2025_06_01_ge_60_blacklist"
        elif recent and ge60:
            key = "recent_ge_2025_06_01_ge_60_non_blacklist"
        elif recent:
            key = "recent_ge_2025_06_01_lt_60"
        elif ge60:
            key = "old_lt_2025_06_01_ge_60"
        else:
            key = "old_lt_2025_06_01_lt_60"
        counters[key]["calls"] += 1
        counters[key]["transcript_chars"] += row.transcript_chars
    return counters


def bucket_recency(rows: Iterable[CallRow], *, as_of: date) -> Mapping[str, int]:
    buckets = {
        "0_30_days": 0,
        "31_90_days": 0,
        "91_180_days": 0,
        "181_365_days": 0,
        "366_plus_days": 0,
        "unknown_date": 0,
    }
    for row in rows:
        started = parse_date(row.started_at)
        if started == date.min:
            buckets["unknown_date"] += 1
            continue
        age_days = (as_of - started).days
        if age_days <= 30:
            buckets["0_30_days"] += 1
        elif age_days <= 90:
            buckets["31_90_days"] += 1
        elif age_days <= 180:
            buckets["91_180_days"] += 1
        elif age_days <= 365:
            buckets["181_365_days"] += 1
        else:
            buckets["366_plus_days"] += 1
    return buckets


def bucket_duration(rows: Iterable[CallRow]) -> Mapping[str, int]:
    buckets = {
        "0_14_sec": 0,
        "15_29_sec": 0,
        "30_59_sec": 0,
        "60_119_sec": 0,
        "120_299_sec": 0,
        "300_599_sec": 0,
        "600_plus_sec": 0,
    }
    for row in rows:
        seconds = row.duration_sec
        if seconds < 15:
            buckets["0_14_sec"] += 1
        elif seconds < 30:
            buckets["15_29_sec"] += 1
        elif seconds < 60:
            buckets["30_59_sec"] += 1
        elif seconds < 120:
            buckets["60_119_sec"] += 1
        elif seconds < 300:
            buckets["120_299_sec"] += 1
        elif seconds < 600:
            buckets["300_599_sec"] += 1
        else:
            buckets["600_plus_sec"] += 1
    return buckets


def estimate_execution(package_root: Path) -> Mapping[str, Any]:
    elapsed = 0.0
    done = 0
    total = 0
    files: list[Mapping[str, Any]] = []
    for path in sorted(package_root.glob("ab_summary_part*.json")):
        data = read_json(path)
        for model in data.get("models") or []:
            model_elapsed = model.get("elapsed_sec")
            metrics = model.get("metrics") or {}
            model_done = int(metrics.get("done") or 0)
            model_total = int(metrics.get("total") or 0)
            if model_elapsed is None:
                continue
            elapsed += float(model_elapsed)
            done += model_done
            total += model_total
            files.append(
                {
                    "file": path.name,
                    "elapsed_sec": round(float(model_elapsed), 3),
                    "done": model_done,
                    "total": model_total,
                }
            )
    wall_seconds = max((float(item["elapsed_sec"]) for item in files), default=0.0)
    return {
        "source": "ab_summary_part*_continuation/retry elapsed_sec",
        "elapsed_sec_sum": round(elapsed, 3),
        "parallel_wall_elapsed_sec_max": round(wall_seconds, 3),
        "done_sum": done,
        "total_sum": total,
        "seconds_per_done_call": round(elapsed / done, 6) if done else None,
        "calls_per_parallel_wall_hour": round(done / (wall_seconds / 3600), 3) if wall_seconds and done else None,
        "files_used": files,
    }


def estimate_slice(
    eligible_tail: Sequence[CallRow],
    execution_estimate: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    seconds_per_call = execution_estimate.get("seconds_per_done_call")
    transcript_chars = sum(row.transcript_chars for row in eligible_tail)
    manifest_rows = int(manifest.get("rows") or 0)
    manifest_chars = int(manifest.get("transcript_chars_sum") or 0)
    estimated_sec_by_calls = None
    estimated_sec_by_chars = None
    estimated_parallel_wall_hours_by_throughput = None
    if isinstance(seconds_per_call, (float, int)):
        estimated_sec_by_calls = float(seconds_per_call) * len(eligible_tail)
        chars_per_call = manifest_chars / manifest_rows if manifest_rows else 0
        seconds_per_char = float(seconds_per_call) / chars_per_call if chars_per_call else 0
        estimated_sec_by_chars = seconds_per_char * transcript_chars if seconds_per_char else None
    calls_per_parallel_wall_hour = execution_estimate.get("calls_per_parallel_wall_hour")
    if isinstance(calls_per_parallel_wall_hour, (float, int)) and calls_per_parallel_wall_hour:
        estimated_parallel_wall_hours_by_throughput = len(eligible_tail) / float(calls_per_parallel_wall_hour)
    return {
        "basis": "rough estimate from M1 first-slice observed elapsed_sec; no LLM calls were made",
        "calls": len(eligible_tail),
        "transcript_chars": transcript_chars,
        "estimated_serial_seconds_by_calls": round_or_none(estimated_sec_by_calls),
        "estimated_serial_hours_by_calls": round_or_none(estimated_sec_by_calls / 3600 if estimated_sec_by_calls else None),
        "estimated_4_parallel_wall_hours_by_calls": round_or_none(estimated_sec_by_calls / 4 / 3600 if estimated_sec_by_calls else None),
        "estimated_parallel_wall_hours_by_observed_throughput": round_or_none(
            estimated_parallel_wall_hours_by_throughput
        ),
        "estimated_serial_seconds_by_chars": round_or_none(estimated_sec_by_chars),
        "estimated_serial_hours_by_chars": round_or_none(estimated_sec_by_chars / 3600 if estimated_sec_by_chars else None),
        "estimated_4_parallel_wall_hours_by_chars": round_or_none(estimated_sec_by_chars / 4 / 3600 if estimated_sec_by_chars else None),
    }


def parse_date(value: str) -> date:
    if not value:
        return date.min
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return date.min


def read_int_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    values: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item.isdigit():
            values.add(int(item))
    return values


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"items": data}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=False)
    uri = f"file:{quote(str(resolved), safe='/:')}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in con.execute(f"PRAGMA table_info({table})").fetchall())


def chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def round_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TZ-16 read-only rerun tail sizing.")
    parser.add_argument("--timeline-db", type=Path, default=DEFAULT_TIMELINE_DB)
    parser.add_argument("--master-calls-db", type=Path, default=DEFAULT_MASTER_CALLS_DB)
    parser.add_argument("--rerun-package", type=Path, default=DEFAULT_RERUN_PACKAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--as-of", default=date.today().isoformat())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = compute_rerun_tail(
        RerunTailConfig(
            timeline_db=args.timeline_db,
            master_calls_db=args.master_calls_db,
            rerun_package=args.rerun_package,
            out=args.out,
            as_of=date.fromisoformat(args.as_of),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
