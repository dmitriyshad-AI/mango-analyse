#!/usr/bin/env python3
"""Print-only ADR-003 live swap/rollback checklist.

This helper intentionally does not stop processes, start bots, edit env files,
or write to external systems. It only prints commands for a human operator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo(path: str) -> str:
    return f'git -C "{path}"'


def main() -> int:
    parser = argparse.ArgumentParser(description="Print ADR-003 deploy swap/rollback checklist.")
    parser.add_argument("--candidate-worktree", default="/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff")
    parser.add_argument("--previous-worktree", required=True)
    parser.add_argument("--previous-head", required=True)
    parser.add_argument("--previous-screen", required=True)
    parser.add_argument("--allow-same-worktree", action="store_true")
    parser.add_argument("--live-pid", default="60227")
    parser.add_argument("--heartbeat", default=".codex_local/telegram_pilot_bots/runtime/public_pilot_bots_heartbeat.json")
    args = parser.parse_args()

    candidate = str(Path(args.candidate_worktree).expanduser())
    previous = str(Path(args.previous_worktree).expanduser())
    heartbeat = str(Path(args.heartbeat))
    if candidate == previous and not args.allow_same_worktree:
        print(
            "Refusing to print rollback plan: previous-worktree equals candidate-worktree. "
            "Pass a separate known-good rollback worktree, or use --allow-same-worktree only "
            "with a separately verified previous HEAD.",
            file=sys.stderr,
        )
        return 2

    print("# ADR-003 deploy swap dry-run")
    print("# This script is print-only. It performs no live action.")
    print()
    print("## 1. Fresh freeze checks")
    print(f"{_repo(candidate)} status --short --branch")
    print(f"{_repo(candidate)} rev-parse HEAD")
    print(f"ps -p {args.live_pid} -o pid=,ppid=,stat=,lstart=,command=")
    print(f"lsof -a -p {args.live_pid} -d cwd")
    print(
        "python3 - <<'PY'\n"
        "import json\n"
        f"p='{candidate}/{heartbeat}'\n"
        "data=json.load(open(p))\n"
        "summary=data.get('summary') if isinstance(data.get('summary'), dict) else {}\n"
        "def pick(*keys):\n"
        "    for key in keys:\n"
        "        if key in data: return data.get(key)\n"
        "        if key in summary: return summary.get(key)\n"
        "    return None\n"
        "print({\n"
        " 'status': pick('status'),\n"
        " 'last_cycle_at': pick('last_cycle_at'),\n"
        " 'pid': pick('pid'),\n"
        " 'effective_profile': pick('effective_profile'),\n"
        " 'draft_path': pick('draft_path'),\n"
        " 'model': pick('model'),\n"
        " 'reasoning_effort': pick('reasoning_effort'),\n"
        " 'snapshot': pick('snapshot','snapshot_path'),\n"
        "})\n"
        "print('profile_selfcheck', pick('profile_selfcheck'))\n"
        "PY"
    )
    print()
    print("## 2. Candidate smoke, no live write")
    print(f'cd "{candidate}"')
    print("PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_p0_text_hygiene.py tests/test_subscription_llm_draft_provider.py -k 'p0 or payment or refund or tax or text_hygiene or roles_read'")
    print("PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_public_pilot_bots.py -k 'profile or heartbeat or direct_path'")
    print()
    print("## 3. Stop current live process only after explicit owner confirmation")
    print(f"# screen -S <current_screen_name> -X quit")
    print(f"# or kill {args.live_pid}")
    print()
    print("## 4. Start candidate only after explicit owner confirmation")
    print('# screen -dmS mango_public_pilot_bots_<new_sha> bash -lc "cd <candidate-worktree> && python3 scripts/run_telegram_public_pilot_bots.py --env-file /dev/null --mode poll --brand all"')
    print()
    print("## 5. Rollback skeleton")
    print(f'cd "{previous}"')
    print(f"{_repo(previous)} rev-parse HEAD")
    print(f"{_repo(previous)} cat-file -e {args.previous_head}^{{commit}}")
    print(f'candidate_head=$({_repo(candidate)} rev-parse HEAD)')
    print(f'test "$candidate_head" != "{args.previous_head}"')
    print(f"# expected previous HEAD: {args.previous_head}")
    print(f"# previous screen: {args.previous_screen}")
    print("# start previous known-good command/screen recorded in freeze")
    print("# verify heartbeat pid/cwd/profile/snapshot after rollback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
