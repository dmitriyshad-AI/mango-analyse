#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase
from mango_mvp.replay_exam.provider_adapter import (
    RealReplayDraftProvider,
    assert_real_replay_cases_safe,
    assert_real_replay_output_path,
    assert_scrubbed_cases_path,
)
from mango_mvp.replay_exam.runner import load_cases, run_replay_exam, write_replay_outputs


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
    args = parser.parse_args()
    if args.fake_provider == args.real_provider:
        raise SystemExit("Choose exactly one provider mode: --fake-provider or --real-provider.")
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
    if args.fake_provider:
        provider = _fake_provider
        parallel = args.parallel
    else:
        assert_real_replay_cases_safe(cases, allow_non_chat_only=args.allow_non_chat_only)
        provider = RealReplayDraftProvider(snapshot_path=args.snapshot, cache_dir=args.provider_cache_dir)
        parallel = args.parallel
    rows = run_replay_exam(cases, provider, parallel_dialogs=parallel)
    write_replay_outputs(args.out_dir, rows)
    print(f"replay_results={args.out_dir / 'replay_results.jsonl'}")
    print(f"replay_summary={args.out_dir / 'replay_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
