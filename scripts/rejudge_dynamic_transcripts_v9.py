#!/usr/bin/env python3
"""Offline v9 re-judge for saved dynamic simulator transcripts.

This script intentionally does not use run_telegram_dynamic_client_sim
``--transcripts-in`` because that path re-attaches current context facts. v9
re-judge must read the saved per-turn fields as-is and write a sidecar result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_telegram_dynamic_client_sim as sim


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-judge saved dynamic transcripts with judge prompt v9.")
    parser.add_argument("--transcripts", type=Path, required=True, help="Existing dynamic_dialog_transcripts.jsonl.")
    parser.add_argument("--scenarios", type=Path, required=True, help="Scenario file containing the original judge_spec row.")
    parser.add_argument("--out", type=Path, default=None, help="Output jsonl. Default: judge_results_v9.jsonl next to transcripts.")
    parser.add_argument("--brand", choices=("all", "foton", "unpk"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge-mode", choices=("codex", "fake"), default="codex")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--judge-reasoning", default="high")
    parser.add_argument("--timeout-sec", type=int, default=180)
    args = parser.parse_args(argv)

    sim_input = sim.load_dynamic_sim_input(args.scenarios)
    dialogs = [
        dialog
        for dialog in sim.load_transcripts(args.transcripts)
        if args.brand == "all" or dialog.get("brand") == args.brand
    ]
    if args.limit > 0:
        dialogs = dialogs[: args.limit]

    call_counter = sim.LlmCallCounter()
    model_args = argparse.Namespace(
        judge_mode=args.judge_mode,
        model=args.model,
        judge_reasoning=args.judge_reasoning,
        timeout_sec=args.timeout_sec,
        llm_call_counter=call_counter,
    )
    judge_model = sim.build_judge_model(model_args)
    rows = []
    for dialog in dialogs:
        rows.append(
            sim.judge_dialog(
                judge_model,
                sim_input.judge_spec,
                dialog.get("persona") or {},
                dialog.get("turns") or [],
                dialog_id=str(dialog.get("dialog_id") or ""),
                brand=str(dialog.get("brand") or ""),
                judge_prompt_version="v9",
                run_status=str(dialog.get("run_status") or "completed"),
            )
        )

    out = args.out or (args.transcripts.parent / "judge_results_v9.jsonl")
    sim.write_jsonl(out, rows)
    print(
        json.dumps(
            {
                "ok": True,
                "transcripts": str(args.transcripts),
                "out": str(out),
                "dialogs": len(rows),
                "llm_calls": dict(call_counter.snapshot()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
