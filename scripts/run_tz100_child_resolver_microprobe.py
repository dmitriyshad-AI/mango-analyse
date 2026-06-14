#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from mango_mvp.customer_profile.builder import CustomerProfileBuilder, CustomerProfileBuildOptions
from mango_mvp.customer_profile.child_resolver_llm import DEFAULT_STOPLIST_PATH, load_shared_phone_stoplist
from mango_mvp.utils.phone import normalize_phone


DEFAULT_SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_profiles_after_tail_20260613")
DEFAULT_MASTER_CALLS_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
)


@dataclass(frozen=True)
class Candidate:
    profile_id: str
    profile_hash: str
    primary_phone: str
    child_slots: int
    named_slots: int
    unnamed_slots: int
    has_merge_marker: bool
    is_shared_phone: bool
    source_event_count: int


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve(strict=False)
    profiles_db = Path(args.profiles_db).expanduser().resolve(strict=False) if args.profiles_db else source_root / "customer_profiles.sqlite"
    timeline_db = Path(args.timeline_db).expanduser().resolve(strict=False) if args.timeline_db else source_root / "customer_timeline.sqlite"
    master_calls_db = Path(args.master_calls_db).expanduser().resolve(strict=False)
    stoplist_path = Path(args.stoplist).expanduser().resolve(strict=False)
    out_root = Path(args.out_root or default_out_root()).expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)

    stoplist = load_shared_phone_stoplist(stoplist_path, required=True)
    candidates = load_candidates(profiles_db, stoplist=stoplist)
    selected = select_microprobe_candidates(candidates, limit=int(args.limit))
    if not selected:
        raise SystemExit("No TZ100 microprobe candidates found")

    trace_path = out_root / "llm_child_resolver_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    before_db = out_root / "customer_profiles_before.sqlite"
    after_db = out_root / "customer_profiles_after.sqlite"

    old_env = os.environ.copy()
    try:
        os.environ["PROFILE_CHILD_MERGE_BY_TRAIT"] = "0"
        os.environ["PROFILE_LLM_CHILD_RESOLVER"] = "0"
        before_report = build_profiles(
            timeline_db=timeline_db,
            profiles_db=before_db,
            master_calls_db=master_calls_db,
            customer_ids=[item.profile_id for item in selected],
            build_id="tz100_microprobe_before",
        )

        os.environ["PROFILE_LLM_CHILD_RESOLVER"] = "1"
        os.environ["PROFILE_LLM_CHILD_RESOLVER_PROVIDER"] = str(args.provider)
        os.environ["PROFILE_LLM_CHILD_RESOLVER_MODEL"] = str(args.model)
        os.environ["PROFILE_LLM_CHILD_RESOLVER_REASONING"] = str(args.reasoning)
        os.environ["PROFILE_LLM_CHILD_RESOLVER_MAX_CONCURRENCY"] = str(args.max_concurrency)
        os.environ["PROFILE_LLM_CHILD_RESOLVER_STOPLIST"] = str(stoplist_path)
        os.environ["PROFILE_LLM_CHILD_RESOLVER_TRACE_PATH"] = str(trace_path)
        os.environ["PROFILE_LLM_CHILD_RESOLVER_PROJECT_ROOT"] = str(Path.cwd())
        os.environ["LLM_CACHE_DIR"] = str(out_root / "llm_cache")
        after_report = build_profiles(
            timeline_db=timeline_db,
            profiles_db=after_db,
            master_calls_db=master_calls_db,
            customer_ids=[item.profile_id for item in selected],
            build_id="tz100_microprobe_after",
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    before_metrics = child_metrics(before_db)
    after_metrics = child_metrics(after_db)
    trace_events = read_trace_events(trace_path)
    selected_public = [
        {
            "profile_hash": item.profile_hash,
            "child_slots_before_source": item.child_slots,
            "named_slots_source": item.named_slots,
            "unnamed_slots_source": item.unnamed_slots,
            "has_merge_marker_source": item.has_merge_marker,
            "is_shared_phone_candidate": item.is_shared_phone,
            "source_event_count": item.source_event_count,
        }
        for item in selected
    ]
    summary = {
        "schema_version": "tz100_child_resolver_microprobe_v1",
        "out_root": str(out_root),
        "profiles_db_source": str(profiles_db),
        "timeline_db": str(timeline_db),
        "master_calls_db": str(master_calls_db),
        "model": args.model,
        "provider": args.provider,
        "reasoning": args.reasoning,
        "limit_requested": int(args.limit),
        "selected_count": len(selected),
        "selected_profiles": selected_public,
        "shared_phone_candidate_included": any(item.is_shared_phone for item in selected),
        "before_report": before_report,
        "after_report": after_report,
        "before_child_metrics": before_metrics,
        "after_child_metrics": after_metrics,
        "trace_path": str(trace_path),
        "trace_events": trace_events,
        "safety": {
            "write_amo": False,
            "write_tallanto": False,
            "write_crm": False,
            "run_asr": False,
            "run_resolve_analyze": False,
            "full_7512_run": False,
            "output_ignored_by_git": True,
            "names_in_report": False,
            "phones_in_report": False,
        },
    }
    write_json(out_root / "summary.json", summary)
    write_json(out_root / "selected_profiles_anonymized.json", selected_public)
    print(json.dumps({"out_root": str(out_root), "selected_count": len(selected), "trace_path": str(trace_path)}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TZ100 child resolver microprobe on 15-20 existing profile families.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--profiles-db")
    parser.add_argument("--timeline-db")
    parser.add_argument("--master-calls-db", default=str(DEFAULT_MASTER_CALLS_DB))
    parser.add_argument("--stoplist", default=str(DEFAULT_STOPLIST_PATH))
    parser.add_argument("--out-root")
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--provider", default="codex_cli")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--max-concurrency", type=int, default=2)
    return parser


def default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("product_data/customer_profiles") / f"tz100_microprobe_{stamp}"


def build_profiles(
    *,
    timeline_db: Path,
    profiles_db: Path,
    master_calls_db: Path,
    customer_ids: Sequence[str],
    build_id: str,
) -> Mapping[str, Any]:
    return CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=profiles_db,
            master_calls_db=master_calls_db,
            customer_ids=tuple(customer_ids),
            build_id=build_id,
        )
    ).build()


def load_candidates(profiles_db: Path, *, stoplist: set[str]) -> list[Candidate]:
    con = connect_read_only(profiles_db)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            WITH slots AS (
              SELECT profile_id, child_key,
                     MAX(CASE WHEN field = 'child_name' THEN 1 ELSE 0 END) AS has_name
              FROM profile_fields
              WHERE superseded_by = ''
                AND child_key <> ''
                AND field IN ('child_name', 'grade', 'subject')
              GROUP BY profile_id, child_key
            ),
            profile_slot_counts AS (
              SELECT profile_id,
                     COUNT(*) AS child_slots,
                     SUM(CASE WHEN has_name = 1 THEN 1 ELSE 0 END) AS named_slots,
                     SUM(CASE WHEN has_name = 0 THEN 1 ELSE 0 END) AS unnamed_slots
              FROM slots
              GROUP BY profile_id
            ),
            merge_markers AS (
              SELECT DISTINCT profile_id
              FROM profile_fields
              WHERE superseded_by = '' AND field = 'child_slot_merge_candidate'
            )
            SELECT cp.profile_id, cp.primary_phone, cp.source_event_count,
                   psc.child_slots, psc.named_slots, psc.unnamed_slots,
                   CASE WHEN mm.profile_id IS NOT NULL THEN 1 ELSE 0 END AS has_merge_marker
            FROM profile_slot_counts psc
            JOIN customer_profiles cp ON cp.profile_id = psc.profile_id
            LEFT JOIN merge_markers mm ON mm.profile_id = psc.profile_id
            WHERE psc.child_slots >= 2 OR psc.unnamed_slots >= 1
            ORDER BY cp.source_event_count DESC, cp.profile_id
            """
        ).fetchall()
    finally:
        con.close()
    return [
        Candidate(
            profile_id=str(row["profile_id"]),
            profile_hash=hash_text(str(row["profile_id"]))[:12],
            primary_phone=str(row["primary_phone"] or ""),
            child_slots=int(row["child_slots"] or 0),
            named_slots=int(row["named_slots"] or 0),
            unnamed_slots=int(row["unnamed_slots"] or 0),
            has_merge_marker=bool(row["has_merge_marker"]),
            is_shared_phone=phone_in_stoplist(str(row["primary_phone"] or ""), stoplist),
            source_event_count=int(row["source_event_count"] or 0),
        )
        for row in rows
    ]


def select_microprobe_candidates(candidates: Sequence[Candidate], *, limit: int) -> list[Candidate]:
    selected: list[Candidate] = []

    def add(items: Sequence[Candidate], count: int) -> None:
        for item in items:
            if len(selected) >= limit or count <= 0:
                return
            if item.profile_id in {existing.profile_id for existing in selected}:
                continue
            selected.append(item)
            count -= 1

    shared = [item for item in candidates if item.is_shared_phone]
    merge_markers = [item for item in candidates if item.has_merge_marker and not item.is_shared_phone]
    named_nameless = [item for item in candidates if item.named_slots >= 1 and item.unnamed_slots >= 1 and not item.is_shared_phone]
    nameless_only = [item for item in candidates if item.unnamed_slots >= 1 and item.named_slots == 0 and not item.is_shared_phone]
    general = [item for item in candidates if not item.is_shared_phone]
    add(shared, 1)
    add(merge_markers, max(3, limit // 3))
    add(named_nameless, max(5, limit // 3))
    add(nameless_only, max(2, limit // 6))
    add(general, limit - len(selected))
    return selected[:limit]


def child_metrics(profiles_db: Path) -> Mapping[str, Any]:
    con = connect_read_only(profiles_db)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            WITH slots AS (
              SELECT profile_id, child_key,
                     MAX(CASE WHEN field = 'child_name' THEN 1 ELSE 0 END) AS has_name
              FROM profile_fields
              WHERE superseded_by = ''
                AND child_key <> ''
                AND field IN ('child_name', 'grade', 'subject')
              GROUP BY profile_id, child_key
            ),
            profile_slot_counts AS (
              SELECT profile_id,
                     COUNT(*) AS child_slots,
                     SUM(CASE WHEN has_name = 1 THEN 1 ELSE 0 END) AS named_slots,
                     SUM(CASE WHEN has_name = 0 THEN 1 ELSE 0 END) AS unnamed_slots
              FROM slots
              GROUP BY profile_id
            )
            SELECT COUNT(*) AS profiles_with_children,
                   SUM(child_slots) AS child_slots_total,
                   SUM(named_slots) AS named_slots_total,
                   SUM(unnamed_slots) AS unnamed_slots_total,
                   SUM(CASE WHEN child_slots >= 2 THEN 1 ELSE 0 END) AS profiles_with_2plus_children,
                   SUM(CASE WHEN unnamed_slots >= 1 THEN 1 ELSE 0 END) AS profiles_with_unnamed,
                   SUM(CASE WHEN named_slots >= 1 AND unnamed_slots >= 1 THEN 1 ELSE 0 END) AS profiles_named_plus_unnamed
            FROM profile_slot_counts
            """
        ).fetchone()
    finally:
        con.close()
    return {key: int(rows[key] or 0) for key in rows.keys()}


def read_trace_events(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    result: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            result.append(payload)
    return result


def phone_in_stoplist(phone: str, stoplist: set[str]) -> bool:
    normalized = normalize_phone(phone)
    if not normalized:
        return False
    stop_last10 = {item[-10:] for item in stoplist if len(item) >= 10}
    return normalized in stoplist or normalized[-10:] in stop_last10


def connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=False)
    uri = f"file:{quote(str(resolved), safe='/:')}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
