#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase
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
    args = parser.parse_args()
    if not args.fake_provider:
        raise SystemExit("Only --fake-provider is implemented until live provider adapter is explicitly approved.")
    rows = run_replay_exam(load_cases(args.set), _fake_provider, parallel_dialogs=args.parallel)
    write_replay_outputs(args.out_dir, rows)
    print(f"replay_results={args.out_dir / 'replay_results.jsonl'}")
    print(f"replay_summary={args.out_dir / 'replay_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
