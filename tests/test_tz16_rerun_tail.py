from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.compute_tz16_rerun_tail import (
    RerunTailConfig,
    bucket_duration,
    bucket_recency,
    compute_rerun_tail,
    estimate_execution,
    load_zone,
)


def test_load_zone_matches_active_amo_and_strong_tallanto_formula(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.sqlite"
    seed_timeline(timeline)

    zone = load_zone(timeline)

    assert zone["active_amo_customers"] == 1
    assert zone["strong_tallanto_student_customers"] == 1
    assert zone["union_customers"] == 2
    assert zone["call_ids"] == {101, 102}


def test_load_zone_opens_timeline_under_path_with_space(tmp_path: Path) -> None:
    root = tmp_path / "timeline with space"
    root.mkdir()
    timeline = root / "timeline.sqlite"
    seed_timeline(timeline)

    zone = load_zone(timeline)

    assert zone["call_ids"] == {101, 102}


def test_compute_rerun_tail_counts_v7_blacklist_and_eligible_tail(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.sqlite"
    master = tmp_path / "master.sqlite"
    package = tmp_path / "package"
    out = tmp_path / "out.json"
    seed_timeline(timeline)
    seed_master(master)
    seed_package(package)

    summary = compute_rerun_tail(RerunTailConfig(timeline, master, package, out=out))

    assert out.exists()
    assert summary["zone"]["zone_call_count"] == 2
    assert summary["v7_first_slice"]["target_ids_total"] == 2
    assert summary["v7_first_slice"]["blacklist_ids_total"] == 1
    assert summary["v7_first_slice"]["zone_calls_current_v7"] == 1
    assert summary["tail"]["old_summary_calls"] == 1
    assert summary["tail"]["eligible_second_slice_calls"] == 0
    assert summary["tail"]["ge60_old_summary_calls_including_blacklist"] == 1
    assert summary["tail"]["reasons"]["blacklist_preserved"] == 1
    assert summary["tail"]["reason_quadrants"]["recent_ge_2025_06_01_ge_60_blacklist"]["calls"] == 1
    assert summary["safety"]["llm_calls"] == 0


def test_tail_buckets_are_stable() -> None:
    from scripts.compute_tz16_rerun_tail import CallRow
    from datetime import date

    rows = [
        CallRow(1, "2026-06-01T00:00:00+00:00", 10, "done", 100, ""),
        CallRow(2, "2026-02-01T00:00:00+00:00", 60, "done", 100, ""),
        CallRow(3, "2025-08-01T00:00:00+00:00", 200, "done", 100, ""),
        CallRow(4, "2024-01-01T00:00:00+00:00", 700, "done", 100, ""),
        CallRow(5, "", 1300, "done", 100, ""),
    ]

    assert bucket_recency(rows, as_of=date(2026, 6, 12)) == {
        "0_30_days": 1,
        "31_90_days": 0,
        "91_180_days": 1,
        "181_365_days": 1,
        "366_plus_days": 1,
        "unknown_date": 1,
    }
    assert bucket_duration(rows) == {
        "0_14_sec": 1,
        "15_29_sec": 0,
        "30_59_sec": 0,
        "60_119_sec": 1,
        "120_299_sec": 1,
        "300_599_sec": 0,
        "600_plus_sec": 2,
    }


def test_estimate_execution_uses_only_files_with_elapsed(tmp_path: Path) -> None:
    (tmp_path / "ab_summary_part1.json").write_text(
        json.dumps({"models": [{"metrics": {"done": 10, "total": 10}}]}),
        encoding="utf-8",
    )
    (tmp_path / "ab_summary_part1_continuation.json").write_text(
        json.dumps({"models": [{"elapsed_sec": 25.0, "metrics": {"done": 5, "total": 6}}]}),
        encoding="utf-8",
    )

    estimate = estimate_execution(tmp_path)

    assert estimate["done_sum"] == 5
    assert estimate["total_sum"] == 6
    assert estimate["seconds_per_done_call"] == 5.0
    assert estimate["parallel_wall_elapsed_sec_max"] == 25.0
    assert estimate["calls_per_parallel_wall_hour"] == 720.0
    assert estimate["files_used"][0]["file"] == "ab_summary_part1_continuation.json"


def seed_timeline(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customer_opportunities (
          opportunity_id TEXT PRIMARY KEY,
          tenant_id TEXT,
          customer_id TEXT,
          opportunity_type TEXT,
          source_system TEXT,
          source_id TEXT,
          title TEXT,
          status TEXT,
          opened_at TEXT,
          closed_at TEXT,
          confidence REAL,
          record_hash TEXT,
          record_json TEXT
        );
        CREATE TABLE identity_links (
          link_id TEXT PRIMARY KEY,
          tenant_id TEXT,
          customer_id TEXT,
          link_type TEXT,
          link_value TEXT,
          source_system TEXT,
          source_ref TEXT,
          match_class TEXT,
          confidence REAL,
          first_seen_at TEXT,
          last_seen_at TEXT,
          record_hash TEXT,
          record_json TEXT
        );
        CREATE TABLE timeline_events (
          event_id TEXT PRIMARY KEY,
          dedupe_key TEXT,
          tenant_id TEXT,
          customer_id TEXT,
          opportunity_id TEXT,
          event_type TEXT,
          event_at TEXT,
          source_system TEXT,
          source_id TEXT,
          source_ref TEXT,
          direction TEXT,
          match_status TEXT,
          confidence REAL,
          importance INTEGER,
          subject TEXT,
          text_preview TEXT,
          summary TEXT,
          created_at TEXT,
          record_hash TEXT,
          record_json TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO customer_opportunities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("opp1", "foton", "cust-amo", "amo_deal", "amocrm_snapshot", "1", "", "В работе", "", "", 1, "h", "{}"),
            (
                "opp2",
                "foton",
                "cust-closed",
                "amo_deal",
                "amocrm_snapshot",
                "2",
                "",
                "Закрыто и не реализовано",
                "",
                "",
                1,
                "h",
                "{}",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO identity_links VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("link1", "foton", "cust-tallanto", "tallanto_student_id", "s1", "tallanto_snapshot", "ref", "strong_unique", 1, "", "", "h", "{}"),
            ("link2", "foton", "cust-amb", "tallanto_student_id", "s2", "tallanto_snapshot", "ref", "ambiguous", 1, "", "", "h", "{}"),
        ],
    )
    con.executemany(
        "INSERT INTO timeline_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("e1", "d1", "foton", "cust-amo", "", "mango_call", "2026-01-01T00:00:00+00:00", "mango_processed_summary", "101", "mango:101", "system", "strong_unique", 1, 0, "", "", "", "", "h", "{}"),
            ("e2", "d2", "foton", "cust-tallanto", "", "mango_call", "2026-01-02T00:00:00+00:00", "mango_processed_summary", "102", "mango:102", "system", "strong_unique", 1, 0, "", "", "", "", "h", "{}"),
            ("e3", "d3", "foton", "cust-closed", "", "mango_call", "2026-01-03T00:00:00+00:00", "mango_processed_summary", "103", "mango:103", "system", "strong_unique", 1, 0, "", "", "", "", "h", "{}"),
        ],
    )
    con.commit()
    con.close()


def seed_master(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE canonical_calls (
          canonical_call_id INTEGER PRIMARY KEY,
          started_at TEXT,
          duration_sec INTEGER,
          analysis_status TEXT,
          transcript_chars INTEGER,
          transcript_text TEXT,
          analysis_json TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO canonical_calls VALUES (?,?,?,?,?,?,?)",
        [
            (101, "2026-01-01T00:00:00+00:00", 300, "done", 1000, "", json.dumps({"analysis_meta": {"analysis_prompt_version": "v7"}})),
            (102, "2026-01-02T00:00:00+00:00", 300, "done", 1000, "", json.dumps({"analysis_meta": {"analysis_prompt_version": "v6"}})),
        ],
    )
    con.commit()
    con.close()


def seed_package(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    (root / "data" / "ids_all.txt").write_text("101\n102\n", encoding="utf-8")
    (root / "blacklist_77.txt").write_text("102\n", encoding="utf-8")
    (root / "data" / "manifest.json").write_text(
        json.dumps({"rows": 2, "transcript_chars_sum": 2000}),
        encoding="utf-8",
    )
    (root / "ab_summary_part1_continuation.json").write_text(
        json.dumps({"models": [{"elapsed_sec": 20.0, "metrics": {"done": 2, "total": 2}}]}),
        encoding="utf-8",
    )
