#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mango_mvp.replay_exam.m1_adapter import build_replay_m1_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--live-head", default="")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source-head", type=Path)
    parser.add_argument("--pii-report", type=Path)
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--retention-manifest", type=Path)
    parser.add_argument("--max-bot-calls", type=int)
    parser.add_argument("--max-judge-calls", type=int)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("--command requires at least one token")
    budgets = {}
    if args.max_bot_calls is not None:
        budgets["max_bot_calls"] = args.max_bot_calls
    if args.max_judge_calls is not None:
        budgets["max_judge_calls"] = args.max_judge_calls
    path = build_replay_m1_manifest(
        set_path=args.set,
        out_path=args.out,
        command=args.command,
        live_head=args.live_head,
        snapshot_path=args.snapshot,
        bundle_path=args.bundle,
        source_head_path=args.source_head,
        pii_report_path=args.pii_report,
        raw_manifest_path=args.raw_manifest,
        retention_manifest_path=args.retention_manifest,
        budgets=budgets,
    )
    print(f"manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
