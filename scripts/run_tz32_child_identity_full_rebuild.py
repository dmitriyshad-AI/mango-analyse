#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_profile.builder import CustomerProfileBuilder, CustomerProfileBuildOptions, connect_read_only
from mango_mvp.customer_profile.child_identity_dedup_llm import (
    DEFAULT_STOPLIST_PATH,
    PROMPT_VERSION,
    ChildResolverCase,
    build_child_resolver_cases,
    normalize_full_name,
)
from mango_mvp.customer_profile.store import sha256_file


DEFAULT_SOURCE_ROOT = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_profiles_after_tail_20260613"
)
DEFAULT_MASTER_CALLS_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
)
CHILD_FIELDS = ("child_name", "grade", "subject")
RUN_CONFIG_COMPARE_KEYS = (
    "schema_version",
    "out_root",
    "timeline_db",
    "timeline_db_sha256",
    "master_calls_db",
    "master_calls_db_sha256",
    "prompt_version",
    "provider",
    "model",
    "reasoning",
    "max_concurrency",
    "max_retries",
    "request_timeout_seconds",
    "stoplist_path",
    "profile_phone_index_enabled",
    "profile_child_merge_by_trait",
    "write_amo",
    "write_tallanto",
    "write_crm",
    "run_asr",
    "run_resolve_analyze",
    "llm_cache_enabled",
    "codex_cli_command",
    "codex_home",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve(strict=False)
    timeline_db = Path(args.timeline_db).expanduser().resolve(strict=False) if args.timeline_db else source_root / "customer_timeline.sqlite"
    master_calls_db = Path(args.master_calls_db).expanduser().resolve(strict=False)
    stoplist_path = Path(args.stoplist).expanduser().resolve(strict=False)
    out_root = Path(args.out_root or default_out_root()).expanduser().resolve(strict=False)

    before_db = out_root / "customer_profiles_before_regex.sqlite"
    after_db = out_root / "customer_profiles_after_llm.sqlite"
    trace_path = out_root / "llm_child_resolver_trace.anonymized.jsonl"
    name_diagnostics_raw_path = out_root / "name_review_diagnostics.raw.local.jsonl"
    cache_dir = out_root / "llm_cache"
    config_path = out_root / "run_config.json"

    timeline_sha = sha256_file(timeline_db)
    master_sha = sha256_file(master_calls_db)
    run_config = {
        "schema_version": "tz32_child_identity_full_rebuild_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "timeline_db": str(timeline_db),
        "timeline_db_sha256": timeline_sha,
        "master_calls_db": str(master_calls_db),
        "master_calls_db_sha256": master_sha,
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "model": args.model,
        "reasoning": args.reasoning,
        "max_concurrency": int(args.max_concurrency),
        "max_retries": int(args.max_retries),
        "request_timeout_seconds": float(args.timeout_seconds),
        "llm_cache_enabled": True,
        "codex_cli_command": str(args.codex_cli_command),
        "codex_home": str(args.codex_home or ""),
        "stoplist_path": str(stoplist_path),
        "profile_phone_index_enabled": True,
        "profile_child_merge_by_trait": False,
        "write_amo": False,
        "write_tallanto": False,
        "write_crm": False,
        "run_asr": False,
        "run_resolve_analyze": False,
    }
    run_config = prepare_out_root_and_config(out_root, config_path, run_config)

    before_report_path = out_root / "before_regex_build_report.json"
    after_report_path = out_root / "after_llm_build_report.json"
    if not before_report_path.exists():
        before_report = build_profiles(
            timeline_db=timeline_db,
            profiles_db=before_db,
            master_calls_db=master_calls_db,
            build_id="tz32_before_regex",
            env={
                "PROFILE_CHILD_MERGE_BY_TRAIT": "0",
                "PROFILE_LLM_CHILD_RESOLVER": "0",
                "PROFILE_PHONE_INDEX": "1",
                "LLM_CACHE_ENABLED": "1",
                "CODEX_CLI_COMMAND": str(args.codex_cli_command),
                "PROFILE_LLM_CHILD_RESOLVER_CODEX_HOME": str(args.codex_home or ""),
            },
        )
        write_json(before_report_path, before_report)
    else:
        before_report = read_json(before_report_path)

    if not after_report_path.exists():
        after_report = build_profiles(
            timeline_db=timeline_db,
            profiles_db=after_db,
            master_calls_db=master_calls_db,
            build_id="tz32_after_llm_v5",
            env={
                "PROFILE_CHILD_MERGE_BY_TRAIT": "0",
                "PROFILE_LLM_CHILD_RESOLVER": "1",
                "PROFILE_LLM_CHILD_RESOLVER_PROVIDER": str(args.provider),
                "PROFILE_LLM_CHILD_RESOLVER_MODEL": str(args.model),
                "PROFILE_LLM_CHILD_RESOLVER_REASONING": str(args.reasoning),
                "PROFILE_LLM_CHILD_RESOLVER_MAX_CONCURRENCY": str(args.max_concurrency),
                "PROFILE_LLM_CHILD_RESOLVER_MAX_RETRIES": str(args.max_retries),
                "PROFILE_LLM_CHILD_RESOLVER_TIMEOUT_SECONDS": str(args.timeout_seconds),
                "PROFILE_LLM_CHILD_RESOLVER_STOPLIST": str(stoplist_path),
                "PROFILE_LLM_CHILD_RESOLVER_TRACE_PATH": str(trace_path),
                "PROFILE_LLM_CHILD_RESOLVER_NAME_DIAGNOSTICS_PATH": str(name_diagnostics_raw_path),
                "PROFILE_LLM_CHILD_RESOLVER_PROJECT_ROOT": str(Path.cwd()),
                "LLM_CACHE_DIR": str(cache_dir),
                "LLM_CACHE_ENABLED": "1",
                "CODEX_CLI_COMMAND": str(args.codex_cli_command),
                "PROFILE_LLM_CHILD_RESOLVER_CODEX_HOME": str(args.codex_home or ""),
                "PROFILE_PHONE_INDEX": "1",
            },
        )
        write_json(after_report_path, after_report)
    else:
        after_report = read_json(after_report_path)

    raw_cases = load_raw_child_resolver_cases(timeline_db=timeline_db, master_calls_db=master_calls_db)
    raw_case_by_id = {case.case_id: case for case in raw_cases}
    trace_events = dedupe_trace_events(read_jsonl(trace_path))
    raw_trace_events = [raw_trace_event(raw_case_by_id.get(str(event.get("case_id") or "")), event) for event in trace_events]
    raw_trace_by_case_id = {str(event.get("case_id") or ""): event for event in raw_trace_events}
    raw_name_events = read_jsonl(name_diagnostics_raw_path)
    write_jsonl(out_root / "raw_diagnostics.anonymized.jsonl", trace_events)
    write_jsonl(out_root / "raw_diagnostics.raw.local.jsonl", raw_trace_events)
    write_jsonl(out_root / "name_review_diagnostics.anonymized.jsonl", [anonymize_name_diagnostic(row) for row in raw_name_events])
    low_events = [event for event in trace_events if event_has_confidence(event, "low")]
    low_multi_named = [event for event in low_events if input_named_count(event) >= 2]
    high_applied = [event for event in trace_events if event.get("accepted") and event_all_confidence(event, "high") and event_has_merge(event)]
    incompatible = [event for event in trace_events if str(event.get("error_code") or "").startswith("incompatible_")]
    write_jsonl(out_root / "low_confidence_queue.anonymized.jsonl", low_events)
    write_jsonl(out_root / "low_confidence_multi_named_queue.anonymized.jsonl", low_multi_named)
    write_jsonl(out_root / "high_confidence_applied_merges.anonymized.jsonl", high_applied)
    write_jsonl(out_root / "incompatible_grades_queue.anonymized.jsonl", incompatible)
    write_jsonl(out_root / "low_confidence_queue.raw.local.jsonl", raw_events_for(low_events, raw_trace_by_case_id))
    write_jsonl(out_root / "low_confidence_multi_named_queue.raw.local.jsonl", raw_events_for(low_multi_named, raw_trace_by_case_id))
    write_jsonl(out_root / "high_confidence_applied_merges.raw.local.jsonl", raw_events_for(high_applied, raw_trace_by_case_id))
    write_jsonl(out_root / "incompatible_grades_queue.raw.local.jsonl", raw_events_for(incompatible, raw_trace_by_case_id))

    before_slots = child_slot_counts(before_db)
    after_slots = child_slot_counts(after_db)
    llm_case_profile_ids = sorted({case.profile_id for case in raw_cases})
    comparison = compare_slot_counts(before_slots, after_slots, llm_case_profile_ids)
    trace_summary = summarize_trace(trace_events)
    phone_index = phone_index_status(after_db)
    cache_files = list((cache_dir / "child_resolver_v1").glob("*.json")) if cache_dir.exists() else []
    report = {
        **run_config,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "before_db": str(before_db),
        "after_db": str(after_db),
        "before_report": before_report,
        "after_report": after_report,
        "comparison": comparison,
        "trace_summary": trace_summary,
        "phone_index": phone_index,
        "cache_files": len(cache_files),
        "paths": {
            "trace_anonymized": str(trace_path),
            "raw_diagnostics_anonymized": str(out_root / "raw_diagnostics.anonymized.jsonl"),
            "raw_diagnostics_raw_local": str(out_root / "raw_diagnostics.raw.local.jsonl"),
            "name_review_diagnostics_anonymized": str(out_root / "name_review_diagnostics.anonymized.jsonl"),
            "name_review_diagnostics_raw_local": str(name_diagnostics_raw_path),
            "low_confidence_queue_anonymized": str(out_root / "low_confidence_queue.anonymized.jsonl"),
            "low_confidence_queue_raw_local": str(out_root / "low_confidence_queue.raw.local.jsonl"),
            "low_confidence_multi_named_queue_anonymized": str(out_root / "low_confidence_multi_named_queue.anonymized.jsonl"),
            "low_confidence_multi_named_queue_raw_local": str(out_root / "low_confidence_multi_named_queue.raw.local.jsonl"),
            "high_confidence_applied_merges_anonymized": str(out_root / "high_confidence_applied_merges.anonymized.jsonl"),
            "high_confidence_applied_merges_raw_local": str(out_root / "high_confidence_applied_merges.raw.local.jsonl"),
            "incompatible_grades_queue_anonymized": str(out_root / "incompatible_grades_queue.anonymized.jsonl"),
            "incompatible_grades_queue_raw_local": str(out_root / "incompatible_grades_queue.raw.local.jsonl"),
        },
    }
    write_json(out_root / "summary.json", report)
    print(json.dumps(small_stdout_report(report), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TZ32 full child identity dedup rebuild.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--timeline-db")
    parser.add_argument("--master-calls-db", default=str(DEFAULT_MASTER_CALLS_DB))
    parser.add_argument("--stoplist", default=str(DEFAULT_STOPLIST_PATH))
    parser.add_argument("--out-root")
    parser.add_argument("--provider", default="codex_cli")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument("--codex-home", default="")
    return parser


def default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("product_data/customer_profiles") / f"tz32_child_identity_full_v5_{stamp}"


def prepare_out_root_and_config(out_root: Path, config_path: Path, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    if out_root.exists():
        existing_files = [item for item in out_root.iterdir() if item.name != ".DS_Store"]
        if config_path.exists():
            existing = dict(read_json(config_path))
            merged = {**candidate, **existing}
            mismatches = []
            for key in RUN_CONFIG_COMPARE_KEYS:
                if merged.get(key) != candidate.get(key):
                    mismatches.append((key, merged.get(key), candidate.get(key)))
            if mismatches:
                details = "; ".join(f"{key}: existing={old!r} current={new!r}" for key, old, new in mismatches)
                raise RuntimeError(f"out-root already has incompatible run_config.json: {details}")
            if any(key not in existing for key in candidate):
                merged["metadata_backfilled_at"] = datetime.now().isoformat(timespec="seconds")
                write_json(config_path, merged)
            return merged
        if existing_files:
            raise RuntimeError(f"out-root exists and is not empty but has no run_config.json: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(config_path, candidate)
    return candidate


def build_profiles(
    *,
    timeline_db: Path,
    profiles_db: Path,
    master_calls_db: Path,
    build_id: str,
    env: Mapping[str, str],
) -> Mapping[str, Any]:
    old_env = os.environ.copy()
    try:
        os.environ.update(env)
        return CustomerProfileBuilder(
            CustomerProfileBuildOptions(
                timeline_db=timeline_db,
                profiles_db=profiles_db,
                master_calls_db=master_calls_db,
                build_id=build_id,
            )
        ).build()
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def child_slot_counts(db_path: Path) -> dict[str, Mapping[str, int]]:
    con = sqlite3.connect(db_path)
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
            )
            SELECT profile_id,
                   COUNT(*) AS child_slots,
                   SUM(CASE WHEN has_name = 1 THEN 1 ELSE 0 END) AS named_slots,
                   SUM(CASE WHEN has_name = 0 THEN 1 ELSE 0 END) AS unnamed_slots
            FROM slots
            GROUP BY profile_id
            """
        ).fetchall()
    finally:
        con.close()
    return {
        str(row["profile_id"]): {
            "child_slots": int(row["child_slots"] or 0),
            "named_slots": int(row["named_slots"] or 0),
            "unnamed_slots": int(row["unnamed_slots"] or 0),
        }
        for row in rows
    }


def load_raw_child_resolver_cases(*, timeline_db: Path, master_calls_db: Path) -> list[ChildResolverCase]:
    builder = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=timeline_db,
            profiles_db=Path("/dev/null"),
            master_calls_db=master_calls_db,
            build_id="tz32_case_reconstruction",
        )
    )
    timeline = connect_read_only(timeline_db)
    timeline.row_factory = sqlite3.Row
    try:
        profile_ids = builder._select_profile_ids(timeline)
        profiles = builder._load_profile_snapshots(timeline, profile_ids)
        fields = list(builder._fields_from_timeline(timeline, profile_ids))
        if master_calls_db:
            fields.extend(builder._fields_from_master_calls(timeline, profile_ids))
        profile_phones = {profile.profile_id: profile.primary_phone for profile in profiles}
        return build_child_resolver_cases(fields, profile_phones=profile_phones)
    finally:
        timeline.close()


def compare_slot_counts(
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
    profile_ids: Sequence[str],
) -> Mapping[str, Any]:
    before_distribution: Counter[int] = Counter()
    after_distribution: Counter[int] = Counter()
    fixed_to_non_ambiguous = 0
    reduced = 0
    increased = 0
    unchanged = 0
    removed_slots = 0
    for profile_id in profile_ids:
        before_row = before.get(profile_id, {})
        after_row = after.get(profile_id, {})
        before_count = int(before_row.get("child_slots", 0))
        after_count = int(after_row.get("child_slots", 0))
        before_distribution[before_count] += 1
        after_distribution[after_count] += 1
        if after_count < before_count:
            reduced += 1
            removed_slots += before_count - after_count
        elif after_count > before_count:
            increased += 1
        else:
            unchanged += 1
        if after_count < 2 and int(after_row.get("unnamed_slots", 0)) == 0:
            fixed_to_non_ambiguous += 1
    return {
        "ambiguous_profiles_before_regex": len(profile_ids),
        "comparison_scope": "raw_child_resolver_cases_before_regex",
        "profiles_reduced_child_slots": reduced,
        "profiles_fixed_to_non_ambiguous": fixed_to_non_ambiguous,
        "profiles_increased_child_slots": increased,
        "profiles_unchanged_child_slots": unchanged,
        "child_slots_removed_total": removed_slots,
        "children_per_family_before_distribution": dict(sorted(before_distribution.items())),
        "children_per_family_after_distribution": dict(sorted(after_distribution.items())),
        "before_totals": aggregate_slots(before, profile_ids),
        "after_totals": aggregate_slots(after, profile_ids),
        "false_merges_manual_gate": "not_computed_by_script",
    }


def aggregate_slots(rows: Mapping[str, Mapping[str, int]], profile_ids: Sequence[str]) -> Mapping[str, int]:
    result = Counter()
    for profile_id in profile_ids:
        row = rows.get(profile_id, {})
        result["child_slots"] += int(row.get("child_slots", 0))
        result["named_slots"] += int(row.get("named_slots", 0))
        result["unnamed_slots"] += int(row.get("unnamed_slots", 0))
        if int(row.get("child_slots", 0)) >= 2:
            result["profiles_with_2plus_children"] += 1
        if int(row.get("unnamed_slots", 0)) >= 1:
            result["profiles_with_unnamed"] += 1
    result["profiles"] = len(profile_ids)
    return dict(sorted(result.items()))


def summarize_trace(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    accepted = sum(1 for event in events if event.get("accepted") is True)
    failed = [event for event in events if event.get("accepted") is not True]
    errors = Counter(str(event.get("error_code") or "") for event in failed)
    confidence_children = Counter()
    confidence_events = Counter()
    for event in events:
        children = event.get("model_children") if isinstance(event.get("model_children"), list) else []
        values = [
            str(child.get("merge_confidence") or "missing").strip().lower()
            for child in children
            if isinstance(child, Mapping)
        ]
        if not values:
            values = ["missing"]
        for value in values:
            confidence_children[value if value in {"high", "low"} else "missing"] += 1
        confidence_events["low" if "low" in values else "high" if all(value == "high" for value in values) else "missing"] += 1
    return {
        "events": len(events),
        "accepted": accepted,
        "failed_soft": len(failed),
        "failed_soft_by_code": dict(sorted(errors.items())),
        "cache_hits": sum(1 for event in events if event.get("cache_hit") is True),
        "children_by_confidence": dict(sorted(confidence_children.items())),
        "events_by_confidence_signal": dict(sorted(confidence_events.items())),
        "low_confidence_events": sum(1 for event in events if event_has_confidence(event, "low")),
        "low_confidence_multi_named_events": sum(
            1 for event in events if event_has_confidence(event, "low") and input_named_count(event) >= 2
        ),
        "high_confidence_applied_merge_events": sum(
            1 for event in events if event.get("accepted") and event_all_confidence(event, "high") and event_has_merge(event)
        ),
    }


def phone_index_status(db_path: Path) -> Mapping[str, Any]:
    con = sqlite3.connect(db_path)
    try:
        columns = {str(row[1]) for row in con.execute("PRAGMA table_info(customer_profiles)").fetchall()}
        indexes = {str(row[1]) for row in con.execute("PRAGMA index_list(customer_profiles)").fetchall()}
        filled = 0
        total = 0
        if "primary_phone_norm" in columns:
            total, filled = con.execute(
                "SELECT COUNT(*), SUM(CASE WHEN COALESCE(primary_phone_norm, '') <> '' THEN 1 ELSE 0 END) FROM customer_profiles"
            ).fetchone()
    finally:
        con.close()
    return {
        "primary_phone_norm_column": "primary_phone_norm" in columns,
        "idx_customer_profiles_phone_norm": "idx_customer_profiles_phone_norm" in indexes,
        "profiles_total": int(total or 0),
        "profiles_with_primary_phone_norm": int(filled or 0),
    }


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def raw_events_for(events: Sequence[Mapping[str, Any]], raw_trace_by_case_id: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [raw_trace_by_case_id.get(str(event.get("case_id") or ""), event) for event in events]


def raw_trace_event(case: ChildResolverCase | None, event: Mapping[str, Any]) -> Mapping[str, Any]:
    if case is None:
        return dict(event)
    payload = dict(event)
    placeholder_to_name = case_placeholder_to_name(case)
    payload["input_mentions"] = [
        {
            "mention_id": mention.mention_id,
            "child_key": mention.child_key,
            "name": mention.child_name,
            "name_norm": normalize_full_name(mention.child_name),
            "grades": list(mention.grades),
            "subjects": list(mention.subjects),
            "brand": mention.brand,
            "event_at": mention.event_at.isoformat(timespec="seconds") if mention.event_at else "",
            "source_ref": mention.source_ref,
        }
        for mention in case.mentions
    ]
    children = []
    for child in event.get("model_children", []) if isinstance(event.get("model_children"), list) else []:
        if not isinstance(child, Mapping):
            continue
        item = dict(child)
        item["canonical_name"] = demask_name(item.get("canonical_name"), placeholder_to_name)
        item["name_variants"] = [demask_name(value, placeholder_to_name) for value in item.get("name_variants", [])]
        children.append(item)
    payload["model_children"] = children
    return payload


def case_placeholder_to_name(case: ChildResolverCase) -> Mapping[str, str]:
    by_norm: dict[str, str] = {}
    placeholders: dict[str, str] = {}
    for mention in case.mentions:
        norm = normalize_full_name(mention.child_name)
        if not norm:
            continue
        if norm not in by_norm:
            placeholder = f"child_name_{len(by_norm) + 1}"
            by_norm[norm] = placeholder
            placeholders[placeholder] = mention.child_name
    return placeholders


def demask_name(value: Any, placeholder_to_name: Mapping[str, str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return placeholder_to_name.get(text, text)


def dedupe_trace_events(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_case: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id:
            by_case[case_id] = row
    return [by_case[key] for key in sorted(by_case)]


def event_has_confidence(event: Mapping[str, Any], expected: str) -> bool:
    children = event.get("model_children") if isinstance(event.get("model_children"), list) else []
    return any(
        isinstance(child, Mapping) and str(child.get("merge_confidence") or "").strip().lower() == expected
        for child in children
    )


def event_all_confidence(event: Mapping[str, Any], expected: str) -> bool:
    children = event.get("model_children") if isinstance(event.get("model_children"), list) else []
    values = [
        str(child.get("merge_confidence") or "").strip().lower()
        for child in children
        if isinstance(child, Mapping)
    ]
    return bool(values) and all(value == expected for value in values)


def event_has_merge(event: Mapping[str, Any]) -> bool:
    children = event.get("model_children") if isinstance(event.get("model_children"), list) else []
    return any(
        isinstance(child, Mapping)
        and isinstance(child.get("mention_ids"), list)
        and len([item for item in child["mention_ids"] if str(item or "").strip()]) >= 2
        for child in children
    )


def input_named_count(event: Mapping[str, Any]) -> int:
    mentions = event.get("input_mentions") if isinstance(event.get("input_mentions"), list) else []
    return sum(1 for mention in mentions if isinstance(mention, Mapping) and mention.get("has_name"))


def anonymize_name_diagnostic(row: Mapping[str, Any]) -> Mapping[str, Any]:
    name_map: dict[str, str] = {}

    def mask(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        key = text.casefold()
        if key not in name_map:
            name_map[key] = f"child_name_{len(name_map) + 1}"
        return name_map[key]

    payload = dict(row)
    grouped = []
    for group in payload.get("grouped_name_spellings", []) if isinstance(payload.get("grouped_name_spellings"), list) else []:
        if not isinstance(group, Mapping):
            continue
        item = dict(group)
        item["name_norm"] = mask(item.get("name_norm"))
        item["spellings"] = [mask(value) for value in item.get("spellings", []) if str(value or "").strip()]
        grouped.append(item)
    payload["grouped_name_spellings"] = grouped
    mentions = []
    for mention in payload.get("input_mentions", []) if isinstance(payload.get("input_mentions"), list) else []:
        if not isinstance(mention, Mapping):
            continue
        item = dict(mention)
        item["name"] = mask(item.get("name"))
        item["name_norm"] = mask(item.get("name_norm"))
        mentions.append(item)
    payload["input_mentions"] = mentions
    children = []
    for child in payload.get("model_children", []) if isinstance(payload.get("model_children"), list) else []:
        if not isinstance(child, Mapping):
            continue
        item = dict(child)
        item["canonical_name"] = mask(item.get("canonical_name"))
        item["name_variants"] = [mask(value) for value in item.get("name_variants", []) if str(value or "").strip()]
        children.append(item)
    payload["model_children"] = children
    return payload


def small_stdout_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    child_slot_merge = report.get("after_report", {}).get("child_slot_merge", {})
    return {
        "out_root": report.get("out_root"),
        "timeline_db_sha256": report.get("timeline_db_sha256"),
        "master_calls_db_sha256": report.get("master_calls_db_sha256"),
        "settings": {
            "provider": report.get("provider"),
            "model": report.get("model"),
            "reasoning": report.get("reasoning"),
            "max_concurrency": report.get("max_concurrency"),
            "prompt_version": report.get("prompt_version"),
        },
        "families": {
            "ambiguous_profiles_before_regex": report.get("comparison", {}).get("ambiguous_profiles_before_regex"),
            "llm_cases_total": child_slot_merge.get("llm_cases_total"),
            "llm_calls_total": child_slot_merge.get("llm_calls_total"),
            "llm_calls_total_semantics": "scoped non-stoplist resolver cases without cache hit; retry attempts are not separately instrumented",
            "accepted": report.get("trace_summary", {}).get("accepted"),
            "failed_soft": report.get("trace_summary", {}).get("failed_soft"),
            "failed_soft_by_code": report.get("trace_summary", {}).get("failed_soft_by_code"),
        },
        "comparison": {
            "profiles_reduced_child_slots": report.get("comparison", {}).get("profiles_reduced_child_slots"),
            "profiles_fixed_to_non_ambiguous": report.get("comparison", {}).get("profiles_fixed_to_non_ambiguous"),
            "child_slots_removed_total": report.get("comparison", {}).get("child_slots_removed_total"),
        },
        "paths": report.get("paths"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
