#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mango_mvp.replay_exam.m1_adapter import build_replay_m1_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("--command requires at least one token")
    path = build_replay_m1_manifest(set_path=args.set, out_path=args.out, command=args.command)
    print(f"manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
