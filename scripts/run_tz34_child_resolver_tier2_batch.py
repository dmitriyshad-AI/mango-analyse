#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_profile.child_identity_dedup_llm import (
    DEFAULT_STOPLIST_PATH,
    ESCALATION_PROMPT_VERSION,
    PROMPT_VERSION,
    ChildResolverCase,
    ChildResolverFamilyResult,
    first_joint_child_name,
    name_review_reason,
)

from run_tz32_child_identity_full_rebuild import (
    aggregate_slots,
    anonymize_name_diagnostic,
    build_profiles,
    child_slot_counts,
    compare_slot_counts,
    dedupe_trace_events,
    load_raw_child_resolver_cases,
    raw_events_for,
    raw_trace_event,
    read_json,
    read_jsonl,
    summarize_trace,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "tz34_child_resolver_escalation_v1"
DEFAULT_SEARCH_ROOT = Path("product_data/customer_profiles")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_out_root = resolve_baseline_out_root(args.baseline_out_root)
    baseline_summary = read_json(baseline_out_root / "summary.json")
    out_root = Path(args.out_root or default_out_root()).expanduser().resolve(strict=False)
    stoplist_path = Path(args.stoplist).expanduser().resolve(strict=False)
    if not stoplist_path.exists():
        raise RuntimeError(f"shared phone stoplist is required: {stoplist_path}")

    run_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "run_tier2" if args.run_tier2 else "plan_only",
        "out_root": str(out_root),
        "baseline_out_root": str(baseline_out_root),
        "timeline_db": str(baseline_summary.get("timeline_db") or ""),
        "timeline_db_sha256": str(baseline_summary.get("timeline_db_sha256") or ""),
        "master_calls_db": str(baseline_summary.get("master_calls_db") or ""),
        "master_calls_db_sha256": str(baseline_summary.get("master_calls_db_sha256") or ""),
        "tier1_prompt_version": PROMPT_VERSION,
        "tier1_provider": args.provider,
        "tier1_model": args.model,
        "tier1_reasoning": args.reasoning,
        "tier1_max_concurrency": int(args.max_concurrency),
        "tier1_timeout_seconds": float(args.timeout_seconds),
        "tier2_prompt_version": ESCALATION_PROMPT_VERSION,
        "tier2_model": args.escalation_model,
        "tier2_reasoning": args.escalation_reasoning,
        "tier2_max_concurrency": int(args.escalation_max_concurrency),
        "tier2_timeout_seconds": float(args.escalation_timeout_seconds),
        "stoplist_path": str(stoplist_path),
        "profile_child_merge_by_trait": False,
        "profile_phone_index_enabled": True,
        "write_amo": False,
        "write_tallanto": False,
        "write_crm": False,
        "run_asr": False,
        "run_resolve_analyze": False,
        "codex_cli_command": str(args.codex_cli_command),
        "codex_home": str(args.codex_home or ""),
        "cache_only": bool(args.cache_only),
    }
    prepare_out_root(out_root, run_config, resume=args.resume)

    timeline_db = Path(str(baseline_summary.get("timeline_db") or "")).expanduser().resolve(strict=False)
    master_calls_db = Path(str(baseline_summary.get("master_calls_db") or "")).expanduser().resolve(strict=False)
    raw_cases = load_raw_child_resolver_cases(timeline_db=timeline_db, master_calls_db=master_calls_db)
    case_by_id = {case.case_id: case for case in raw_cases}
    plan = write_plan_artifacts(out_root=out_root, baseline_out_root=baseline_out_root, case_by_id=case_by_id)
    report: dict[str, Any] = {
        **run_config,
        "baseline": baseline_overview(baseline_summary),
        "tier2_plan": plan,
        "paths": {
            **plan["paths"],
        },
    }

    if args.run_tier2:
        actual = run_escalation_rebuild(
            args=args,
            out_root=out_root,
            baseline_out_root=baseline_out_root,
            baseline_summary=baseline_summary,
            stoplist_path=stoplist_path,
            case_by_id=case_by_id,
        )
        report.update(actual)
        report["paths"] = {**report["paths"], **actual["paths"]}
    else:
        report["tier2_run"] = {
            "status": "not_started",
            "reason": "run Tier-2 only with explicit --run-tier2 after Dmitry approval",
        }

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(out_root / "summary.json", report)
    print(json.dumps(stdout_report(report), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run TZ34 child resolver Tier-2 escalation.")
    parser.add_argument("--baseline-out-root", default="")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--run-tier2", action="store_true", help="Actually run gpt-5.5 Tier-2 escalation.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing TZ34 out-root.")
    parser.add_argument("--provider", default="codex_cli")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--escalation-model", default="gpt-5.5")
    parser.add_argument("--escalation-reasoning", default="high")
    parser.add_argument("--escalation-max-concurrency", type=int, default=2)
    parser.add_argument("--escalation-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--stoplist", default=str(DEFAULT_STOPLIST_PATH))
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument("--codex-home", default="")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use only existing child resolver cache; cache misses become fail-soft without an LLM call.",
    )
    parser.add_argument(
        "--no-seed-tier1-cache",
        action="store_true",
        help="Do not copy TZ32 cache into the TZ34 out-root before --run-tier2.",
    )
    return parser


def resolve_baseline_out_root(value: str) -> Path:
    if value:
        path = Path(value).expanduser().resolve(strict=False)
        if not (path / "summary.json").exists():
            raise RuntimeError(f"baseline summary.json not found: {path}")
        return path
    candidates = sorted(DEFAULT_SEARCH_ROOT.glob("tz32_child_identity_full_v5_*/summary.json"))
    if not candidates:
        raise RuntimeError("no TZ32 baseline summary found; pass --baseline-out-root")
    return candidates[-1].parent.resolve(strict=False)


def default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("product_data/customer_profiles") / f"tz34_child_escalation_v6_{stamp}"


def prepare_out_root(out_root: Path, run_config: Mapping[str, Any], *, resume: bool) -> None:
    config_path = out_root / "run_config.json"
    if out_root.exists():
        existing_files = [item for item in out_root.iterdir() if item.name != ".DS_Store"]
        if existing_files and not resume:
            raise RuntimeError(f"out-root exists and is not empty; pass --resume or choose a new path: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and resume:
        existing = read_json(config_path)
        mismatches = [
            key
            for key in ("schema_version", "baseline_out_root", "timeline_db_sha256", "master_calls_db_sha256")
            if existing.get(key) != run_config.get(key)
        ]
        if mismatches:
            raise RuntimeError(f"out-root resume config mismatch: {', '.join(mismatches)}")
    write_json(config_path, run_config)


def baseline_overview(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "out_root": summary.get("out_root"),
        "timeline_db_sha256": summary.get("timeline_db_sha256"),
        "master_calls_db_sha256": summary.get("master_calls_db_sha256"),
        "comparison": summary.get("comparison"),
        "trace_summary": summary.get("trace_summary"),
        "after_child_slot_merge": summary.get("after_report", {}).get("child_slot_merge", {}),
    }


def write_plan_artifacts(
    *,
    out_root: Path,
    baseline_out_root: Path,
    case_by_id: Mapping[str, ChildResolverCase],
) -> Mapping[str, Any]:
    baseline_summary = read_json(baseline_out_root / "summary.json")
    trace_path = Path(
        str(
            baseline_summary.get("paths", {}).get("trace_anonymized")
            or baseline_out_root / "llm_child_resolver_trace.anonymized.jsonl"
        )
    )
    raw_trace_path = Path(
        str(
            baseline_summary.get("paths", {}).get("raw_diagnostics_raw_local")
            or baseline_out_root / "raw_diagnostics.raw.local.jsonl"
        )
    )
    queue_path = Path(
        str(
            baseline_summary.get("paths", {}).get("low_confidence_multi_named_queue_anonymized")
            or baseline_out_root / "low_confidence_multi_named_queue.anonymized.jsonl"
        )
    )
    trace_events = dedupe_trace_events(read_jsonl(trace_path))
    raw_trace_events = read_jsonl(raw_trace_path)
    if not raw_trace_events:
        raw_trace_events = [
            raw_trace_event(case_by_id.get(str(event.get("case_id") or "")), event) for event in trace_events
        ]
    raw_trace_by_case_id = {str(event.get("case_id") or ""): event for event in raw_trace_events}
    queue_events = read_jsonl(queue_path)
    queue_ids = {str(event.get("case_id") or "") for event in queue_events if str(event.get("case_id") or "")}
    trace_by_case_id = {str(event.get("case_id") or ""): event for event in trace_events}

    trigger_events: list[Mapping[str, Any]] = []
    call_candidate_events: list[Mapping[str, Any]] = []
    trigger_joint_events: list[Mapping[str, Any]] = []
    queue_joint_events: list[Mapping[str, Any]] = []
    queue_not_trigger_events: list[Mapping[str, Any]] = []
    for case_id in sorted(queue_ids):
        case = case_by_id.get(case_id)
        event = trace_by_case_id.get(case_id)
        if case is not None and event is not None and first_joint_child_name(case):
            queue_joint_events.append(event)
    trigger_ids: set[str] = set()
    for event in trace_events:
        case_id = str(event.get("case_id") or "")
        case = case_by_id.get(case_id)
        if case is None:
            continue
        result = result_from_trace_event(case, event)
        if name_review_reason(case, result) != "low_confidence_multi_named":
            continue
        trigger_ids.add(case_id)
        trigger_events.append(event)
        if first_joint_child_name(case):
            trigger_joint_events.append(event)
        else:
            call_candidate_events.append(event)
    for case_id in sorted(queue_ids - trigger_ids):
        event = trace_by_case_id.get(case_id)
        if event is not None:
            queue_not_trigger_events.append(event)

    write_jsonl(out_root / "tier2_candidates.anonymized.jsonl", call_candidate_events)
    write_jsonl(out_root / "tier2_candidates.raw.local.jsonl", raw_events_for(call_candidate_events, raw_trace_by_case_id))
    write_jsonl(out_root / "tier2_trigger_all.anonymized.jsonl", trigger_events)
    write_jsonl(out_root / "tier2_trigger_all.raw.local.jsonl", raw_events_for(trigger_events, raw_trace_by_case_id))
    write_jsonl(out_root / "joint_mentions.anonymized.jsonl", trigger_joint_events)
    write_jsonl(out_root / "joint_mentions.raw.local.jsonl", raw_events_for(trigger_joint_events, raw_trace_by_case_id))
    write_jsonl(out_root / "queue_joint_mentions.anonymized.jsonl", queue_joint_events)
    write_jsonl(out_root / "queue_joint_mentions.raw.local.jsonl", raw_events_for(queue_joint_events, raw_trace_by_case_id))
    write_jsonl(out_root / "queue_rows_not_production_trigger.anonymized.jsonl", queue_not_trigger_events)
    write_jsonl(
        out_root / "queue_rows_not_production_trigger.raw.local.jsonl",
        raw_events_for(queue_not_trigger_events, raw_trace_by_case_id),
    )
    trigger_not_in_queue_events = [event for event in trigger_events if str(event.get("case_id") or "") not in queue_ids]
    write_jsonl(out_root / "trigger_not_in_tz32_queue.anonymized.jsonl", trigger_not_in_queue_events)
    write_jsonl(
        out_root / "trigger_not_in_tz32_queue.raw.local.jsonl",
        raw_events_for(trigger_not_in_queue_events, raw_trace_by_case_id),
    )

    return {
        "tz32_queue_file_rows": len(queue_events),
        "production_trigger_candidates": len(trigger_events),
        "tier2_call_candidates_after_joint": len(call_candidate_events),
        "trigger_joint_mentions": len(trigger_joint_events),
        "queue_file_joint_mentions": len(queue_joint_events),
        "queue_rows_not_production_trigger": len(queue_not_trigger_events),
        "trigger_not_in_tz32_queue": len(trigger_not_in_queue_events),
        "trigger_definition": 'name_review_reason(case, result) == "low_confidence_multi_named"',
        "paths": {
            "tier2_candidates_anonymized": str(out_root / "tier2_candidates.anonymized.jsonl"),
            "tier2_candidates_raw_local": str(out_root / "tier2_candidates.raw.local.jsonl"),
            "tier2_trigger_all_anonymized": str(out_root / "tier2_trigger_all.anonymized.jsonl"),
            "tier2_trigger_all_raw_local": str(out_root / "tier2_trigger_all.raw.local.jsonl"),
            "joint_mentions_anonymized": str(out_root / "joint_mentions.anonymized.jsonl"),
            "joint_mentions_raw_local": str(out_root / "joint_mentions.raw.local.jsonl"),
            "queue_joint_mentions_anonymized": str(out_root / "queue_joint_mentions.anonymized.jsonl"),
            "queue_joint_mentions_raw_local": str(out_root / "queue_joint_mentions.raw.local.jsonl"),
            "queue_rows_not_production_trigger_anonymized": str(out_root / "queue_rows_not_production_trigger.anonymized.jsonl"),
            "queue_rows_not_production_trigger_raw_local": str(out_root / "queue_rows_not_production_trigger.raw.local.jsonl"),
            "trigger_not_in_tz32_queue_anonymized": str(out_root / "trigger_not_in_tz32_queue.anonymized.jsonl"),
            "trigger_not_in_tz32_queue_raw_local": str(out_root / "trigger_not_in_tz32_queue.raw.local.jsonl"),
        },
    }


def result_from_trace_event(case: ChildResolverCase, event: Mapping[str, Any]) -> ChildResolverFamilyResult:
    children = []
    for child in event.get("model_children", []) if isinstance(event.get("model_children"), list) else []:
        if not isinstance(child, Mapping):
            continue
        children.append(
            {
                "child_id": str(child.get("child_id") or ""),
                "canonical_name": str(child.get("canonical_name") or ""),
                "name_variants": list(child.get("name_variants") or []),
                "grades": list(child.get("grades") or []),
                "subjects": list(child.get("subjects") or []),
                "mention_ids": list(child.get("mention_ids") or []),
                "merge_confidence": str(child.get("merge_confidence") or ""),
            }
        )
    applied = event.get("applied_child_keys") if isinstance(event.get("applied_child_keys"), Mapping) else {}
    return ChildResolverFamilyResult(
        case_id=case.case_id,
        profile_id=case.profile_id,
        accepted=event.get("accepted") is True,
        mention_to_child_key={str(key): str(value) for key, value in applied.items()},
        raw_response={"children": children},
        error_code=str(event.get("error_code") or ""),
        error_detail=str(event.get("error_detail") or ""),
        cache_hit=event.get("cache_hit") is True,
    )


def run_escalation_rebuild(
    *,
    args: argparse.Namespace,
    out_root: Path,
    baseline_out_root: Path,
    baseline_summary: Mapping[str, Any],
    stoplist_path: Path,
    case_by_id: Mapping[str, ChildResolverCase],
) -> Mapping[str, Any]:
    timeline_db = Path(str(baseline_summary.get("timeline_db") or "")).expanduser().resolve(strict=False)
    master_calls_db = Path(str(baseline_summary.get("master_calls_db") or "")).expanduser().resolve(strict=False)
    after_db = out_root / "customer_profiles_after_escalation.sqlite"
    after_report_path = out_root / "after_escalation_build_report.json"
    trace_path = out_root / "llm_child_resolver_trace.anonymized.jsonl"
    name_diagnostics_raw_path = out_root / "name_review_diagnostics.raw.local.jsonl"
    cache_dir = out_root / "llm_cache"
    if not args.no_seed_tier1_cache:
        seed_tier1_cache(baseline_out_root / "llm_cache", cache_dir)

    if not after_report_path.exists():
        after_report = build_profiles(
            timeline_db=timeline_db,
            profiles_db=after_db,
            master_calls_db=master_calls_db,
            build_id="tz34_after_escalation_v6",
            env={
                "PROFILE_CHILD_MERGE_BY_TRAIT": "0",
                "PROFILE_LLM_CHILD_RESOLVER": "1",
                "PROFILE_LLM_CHILD_RESOLVER_ESCALATION": "1",
                "PROFILE_LLM_CHILD_RESOLVER_PROVIDER": str(args.provider),
                "PROFILE_LLM_CHILD_RESOLVER_MODEL": str(args.model),
                "PROFILE_LLM_CHILD_RESOLVER_REASONING": str(args.reasoning),
                "PROFILE_LLM_CHILD_RESOLVER_MAX_CONCURRENCY": str(args.max_concurrency),
                "PROFILE_LLM_CHILD_RESOLVER_MAX_RETRIES": str(args.max_retries),
                "PROFILE_LLM_CHILD_RESOLVER_TIMEOUT_SECONDS": str(args.timeout_seconds),
                "PROFILE_LLM_CHILD_RESOLVER_CACHE_ONLY": "1" if args.cache_only else "0",
                "PROFILE_LLM_CHILD_RESOLVER_ESCALATION_MODEL": str(args.escalation_model),
                "PROFILE_LLM_CHILD_RESOLVER_ESCALATION_REASONING": str(args.escalation_reasoning),
                "PROFILE_LLM_CHILD_RESOLVER_ESCALATION_MAX_CONCURRENCY": str(args.escalation_max_concurrency),
                "PROFILE_LLM_CHILD_RESOLVER_ESCALATION_TIMEOUT_SECONDS": str(args.escalation_timeout_seconds),
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

    trace_events = dedupe_trace_events(read_jsonl(trace_path))
    raw_trace_events = [raw_trace_event(case_by_id.get(str(event.get("case_id") or "")), event) for event in trace_events]
    raw_trace_by_case_id = {str(event.get("case_id") or ""): event for event in raw_trace_events}
    raw_name_events = read_jsonl(name_diagnostics_raw_path)
    write_jsonl(out_root / "raw_diagnostics.anonymized.jsonl", trace_events)
    write_jsonl(out_root / "raw_diagnostics.raw.local.jsonl", raw_trace_events)
    write_jsonl(out_root / "name_review_diagnostics.anonymized.jsonl", [anonymize_name_diagnostic(row) for row in raw_name_events])

    resolved_high = [event for event in trace_events if event.get("accepted") and event.get("escalated")]
    joint = [event for event in trace_events if event.get("error_code") == "joint_mention"]
    still_low = [event for event in trace_events if event.get("manual_review_reason") == "escalation_low_confidence"]
    validation_failed = [event for event in trace_events if event.get("manual_review_reason") == "escalation_validation_failed"]
    manual = [
        event
        for event in trace_events
        if str(event.get("manual_review_reason") or "") in {"joint_mention", "escalation_low_confidence", "escalation_validation_failed"}
    ]
    write_queue(out_root, "tier2_resolved_high", resolved_high, raw_trace_by_case_id)
    write_queue(out_root, "tier3_joint_mentions", joint, raw_trace_by_case_id)
    write_queue(out_root, "tier3_still_low", still_low, raw_trace_by_case_id)
    write_queue(out_root, "tier3_validation_failed", validation_failed, raw_trace_by_case_id)
    write_queue(out_root, "manual_review_queue", manual, raw_trace_by_case_id)

    profile_ids = sorted({case.profile_id for case in case_by_id.values()})
    raw_db = Path(str(baseline_summary.get("before_db") or ""))
    off_db = Path(str(baseline_summary.get("after_db") or ""))
    raw_slots = child_slot_counts(raw_db)
    off_slots = child_slot_counts(off_db)
    on_slots = child_slot_counts(after_db)
    comparison = {
        "raw_vs_escalation_off": baseline_summary.get("comparison"),
        "raw_vs_escalation_on": compare_slot_counts(raw_slots, on_slots, profile_ids),
        "off_vs_escalation_on": compare_slot_counts(off_slots, on_slots, profile_ids),
        "totals": {
            "raw": aggregate_slots(raw_slots, profile_ids),
            "escalation_off": aggregate_slots(off_slots, profile_ids),
            "escalation_on": aggregate_slots(on_slots, profile_ids),
        },
    }
    cache_files = list((cache_dir / "child_resolver_v1").glob("*.json")) if cache_dir.exists() else []
    return {
        "tier2_run": {
            "status": "finished",
            "after_db": str(after_db),
            "after_report": after_report,
            "trace_summary": summarize_trace(trace_events),
            "escalation_summary": after_report.get("child_slot_merge", {}),
            "resolved_high": len(resolved_high),
            "still_low": len(still_low),
            "validation_failed": len(validation_failed),
            "joint_mentions": len(joint),
            "manual_review": len(manual),
            "cache_files": len(cache_files),
        },
        "comparison": comparison,
        "paths": {
            "after_db": str(after_db),
            "after_report": str(after_report_path),
            "trace_anonymized": str(trace_path),
            "raw_diagnostics_anonymized": str(out_root / "raw_diagnostics.anonymized.jsonl"),
            "raw_diagnostics_raw_local": str(out_root / "raw_diagnostics.raw.local.jsonl"),
            "name_review_diagnostics_anonymized": str(out_root / "name_review_diagnostics.anonymized.jsonl"),
            "name_review_diagnostics_raw_local": str(name_diagnostics_raw_path),
            "tier2_resolved_high_anonymized": str(out_root / "tier2_resolved_high.anonymized.jsonl"),
            "tier2_resolved_high_raw_local": str(out_root / "tier2_resolved_high.raw.local.jsonl"),
            "tier3_joint_mentions_anonymized": str(out_root / "tier3_joint_mentions.anonymized.jsonl"),
            "tier3_joint_mentions_raw_local": str(out_root / "tier3_joint_mentions.raw.local.jsonl"),
            "tier3_still_low_anonymized": str(out_root / "tier3_still_low.anonymized.jsonl"),
            "tier3_still_low_raw_local": str(out_root / "tier3_still_low.raw.local.jsonl"),
            "tier3_validation_failed_anonymized": str(out_root / "tier3_validation_failed.anonymized.jsonl"),
            "tier3_validation_failed_raw_local": str(out_root / "tier3_validation_failed.raw.local.jsonl"),
            "manual_review_queue_anonymized": str(out_root / "manual_review_queue.anonymized.jsonl"),
            "manual_review_queue_raw_local": str(out_root / "manual_review_queue.raw.local.jsonl"),
        },
    }


def seed_tier1_cache(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def write_queue(
    out_root: Path,
    name: str,
    events: Sequence[Mapping[str, Any]],
    raw_trace_by_case_id: Mapping[str, Mapping[str, Any]],
) -> None:
    write_jsonl(out_root / f"{name}.anonymized.jsonl", events)
    write_jsonl(out_root / f"{name}.raw.local.jsonl", raw_events_for(events, raw_trace_by_case_id))


def stdout_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    tier2_run = report.get("tier2_run", {})
    child_slot_merge = tier2_run.get("escalation_summary", {}) if isinstance(tier2_run, Mapping) else {}
    return {
        "out_root": report.get("out_root"),
        "mode": report.get("mode"),
        "timeline_db_sha256": report.get("timeline_db_sha256"),
        "settings": {
            "tier1_model": report.get("tier1_model"),
            "tier1_reasoning": report.get("tier1_reasoning"),
            "tier2_model": report.get("tier2_model"),
            "tier2_reasoning": report.get("tier2_reasoning"),
            "tier2_max_concurrency": report.get("tier2_max_concurrency"),
            "tier2_prompt_version": report.get("tier2_prompt_version"),
        },
        "plan": {
            "tz32_queue_file_rows": report.get("tier2_plan", {}).get("tz32_queue_file_rows"),
            "production_trigger_candidates": report.get("tier2_plan", {}).get("production_trigger_candidates"),
            "tier2_call_candidates_after_joint": report.get("tier2_plan", {}).get("tier2_call_candidates_after_joint"),
            "trigger_joint_mentions": report.get("tier2_plan", {}).get("trigger_joint_mentions"),
            "queue_rows_not_production_trigger": report.get("tier2_plan", {}).get("queue_rows_not_production_trigger"),
        },
        "tier2_run": {
            "status": tier2_run.get("status") if isinstance(tier2_run, Mapping) else None,
            "resolved_high": tier2_run.get("resolved_high") if isinstance(tier2_run, Mapping) else None,
            "still_low": tier2_run.get("still_low") if isinstance(tier2_run, Mapping) else None,
            "validation_failed": tier2_run.get("validation_failed") if isinstance(tier2_run, Mapping) else None,
            "joint_mentions": tier2_run.get("joint_mentions") if isinstance(tier2_run, Mapping) else None,
            "manual_review": tier2_run.get("manual_review") if isinstance(tier2_run, Mapping) else None,
            "llm_calls_total": child_slot_merge.get("llm_calls_total"),
            "llm_escalation_calls_total": child_slot_merge.get("llm_escalation_calls_total"),
        },
        "paths": report.get("paths"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
