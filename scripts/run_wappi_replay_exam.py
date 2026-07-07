#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase
from mango_mvp.replay_exam.judge_executor import (
    CodexReplayJudgeRunner,
    build_replay_judge_requests,
    execute_replay_judge_requests,
    write_replay_judge_payloads,
)
from mango_mvp.replay_exam.provider_adapter import (
    RealReplayDraftProvider,
    assert_real_replay_cases_safe,
    assert_real_replay_output_path,
    assert_scrubbed_cases_path,
)
from mango_mvp.replay_exam.pseudonymizer import kb_contact_allowlist
from mango_mvp.replay_exam.pii_scan import scan_paths
from mango_mvp.replay_exam.runner import load_cases, run_replay_exam, write_replay_outputs


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_progress(path: Path, *, total_cases: int, completed_ids: set[str], max_bot_calls: int | None) -> None:
    payload = {
        "schema_version": "wappi_replay_progress_v1",
        "total_cases": total_cases,
        "done_cases": len(completed_ids),
        "completed_exam_ids": sorted(completed_ids),
        "max_bot_calls": max_bot_calls,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _fake_provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
    del context
    return BotReplayResult(route="draft_for_manager", bot_text=f"Черновик по запросу: {case.client_message}", safety_flags=("draft_only",))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline Wappi replay exam over scrubbed cases.")
    parser.add_argument("--set", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--fake-provider", action="store_true", help="Use deterministic fake provider for pipeline smoke tests.")
    parser.add_argument("--real-provider", action="store_true", help="Use the real draft provider over scrubbed replay cases.")
    parser.add_argument("--allow-llm-calls", action="store_true", help="Required with --real-provider; confirms external model calls are allowed.")
    parser.add_argument("--snapshot", type=Path, help="Knowledge snapshot for --real-provider.")
    parser.add_argument("--provider-cache-dir", type=Path, default=Path("~/.mango_local/replay_exam/provider_cache").expanduser())
    parser.add_argument("--allow-non-chat-only", action="store_true", help="Allow non-chat_only segments in real-provider replay.")
    parser.add_argument("--resume", action="store_true", help="Resume from replay_results.partial.jsonl in --out-dir.")
    parser.add_argument("--max-bot-calls", type=int, help="Stop after at most this many pending replay cases.")
    parser.add_argument("--run-judge", action="store_true", help="Run replay_judge_v1 after machine gate.")
    parser.add_argument("--allow-judge-llm-calls", action="store_true", help="Required with --run-judge; confirms judge model calls are allowed.")
    parser.add_argument("--max-judge-calls", type=int, help="Stop judge after at most this many clean chat_only cases.")
    parser.add_argument("--judge-seed", default="replay_judge_v1")
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--judge-reasoning", default="medium")
    args = parser.parse_args()
    if args.fake_provider == args.real_provider:
        raise SystemExit("Choose exactly one provider mode: --fake-provider or --real-provider.")
    if args.run_judge:
        if not args.allow_judge_llm_calls:
            raise SystemExit("--run-judge requires explicit --allow-judge-llm-calls.")
        if args.max_judge_calls is None or args.max_judge_calls < 1:
            raise SystemExit("--run-judge requires positive --max-judge-calls.")
    if args.real_provider:
        if not args.allow_llm_calls:
            raise SystemExit("--real-provider requires explicit --allow-llm-calls.")
        if args.snapshot is None:
            raise SystemExit("--real-provider requires --snapshot.")
        if args.parallel > 2:
            raise SystemExit("--real-provider parallelism is capped at 2 for safety.")
        assert_scrubbed_cases_path(args.set)
        assert_real_replay_output_path(args.out_dir)
    cases = load_cases(args.set)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "replay_results.partial.jsonl"
    progress_path = out_dir / "progress.json"
    existing_rows = _read_jsonl(partial_path) if args.resume else []
    completed_ids = {str(row.get("turn_id") or row.get("exam_id") or "") for row in existing_rows if str(row.get("turn_id") or row.get("exam_id") or "")}
    pending_cases = [case for case in cases if case.turn_id not in completed_ids]
    if args.max_bot_calls is not None:
        if args.max_bot_calls < 1:
            raise SystemExit("--max-bot-calls must be positive.")
        pending_cases = pending_cases[: args.max_bot_calls]
    _write_progress(progress_path, total_cases=len(cases), completed_ids=completed_ids, max_bot_calls=args.max_bot_calls)
    if args.fake_provider:
        provider = _fake_provider
        parallel = args.parallel
    else:
        assert_real_replay_cases_safe(cases, allow_non_chat_only=args.allow_non_chat_only)
        provider = RealReplayDraftProvider(snapshot_path=args.snapshot, cache_dir=args.provider_cache_dir)
        parallel = args.parallel
    def progress_callback(dialog_rows: Sequence[Mapping[str, object]]) -> None:
        _append_jsonl(partial_path, dialog_rows)
        for row in dialog_rows:
            turn_id = str(row.get("turn_id") or row.get("exam_id") or "")
            if turn_id:
                completed_ids.add(turn_id)
        _write_progress(progress_path, total_cases=len(cases), completed_ids=completed_ids, max_bot_calls=args.max_bot_calls)

    new_rows = run_replay_exam(pending_cases, provider, parallel_dialogs=parallel, progress_callback=progress_callback)
    rows = [*existing_rows, *new_rows]
    write_replay_outputs(out_dir, rows)
    judge_requests = []
    if args.run_judge:
        judge_requests = build_replay_judge_requests(
            cases,
            rows,
            seed=args.judge_seed,
            max_judge_calls=args.max_judge_calls,
        )
        write_replay_judge_payloads(out_dir, judge_requests)
    allowlist: tuple[str, ...] = ()
    if args.snapshot is not None:
        allowlist = kb_contact_allowlist(args.snapshot)
    pii_findings = scan_paths([args.set, out_dir], allowlist=allowlist)
    pii_report = {"schema_version": "wappi_replay_pii_scan_v2", "leak_count": len(pii_findings), "findings": pii_findings}
    (out_dir / "pii_scan_v2.json").write_text(
        json.dumps(pii_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if pii_findings:
        raise SystemExit(f"PII scan failed: {len(pii_findings)} findings; see {out_dir / 'pii_scan_v2.json'}")
    if args.run_judge:
        from mango_mvp.channels.subscription_llm_parts.codex_exec import CodexExecConfig

        judge_runner = CodexReplayJudgeRunner(
            config=CodexExecConfig(model=args.judge_model, reasoning_effort=args.judge_reasoning),
        )
        execute_replay_judge_requests(out_dir, judge_requests, runner=judge_runner)
        post_judge_findings = scan_paths([out_dir / "judge_results.jsonl"], allowlist=allowlist)
        if post_judge_findings:
            raise SystemExit(f"PII scan failed after judge: {len(post_judge_findings)} findings; see {out_dir / 'judge_results.jsonl'}")
    print(f"replay_results={out_dir / 'replay_results.jsonl'}")
    print(f"replay_summary={out_dir / 'replay_summary.json'}")
    if args.run_judge:
        print(f"judge_results={out_dir / 'judge_results.jsonl'}")
        print(f"judge_key={out_dir / 'judge_key.jsonl'}")
    print(f"progress={progress_path}")
    print(f"pii_scan={out_dir / 'pii_scan_v2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
