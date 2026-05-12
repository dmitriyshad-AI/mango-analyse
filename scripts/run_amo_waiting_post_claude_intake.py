#!/usr/bin/env python3
"""Build the post-Claude command center for AMO waiting work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.amo_waiting_post_claude_intake import (  # noqa: E402
    DEFAULT_AMO_WAITING_POST_CLAUDE_INTAKE_ROOT,
    build_amo_waiting_post_claude_intake,
)


DEFAULT_RESULT_DIR = "audits/_results/2026-05-11_amo_waiting_autonomous_work_v1"
DEFAULT_WAITING_ROOT = "stable_runtime/amo_waiting_autonomous_work_20260511_v1"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = build_amo_waiting_post_claude_intake(
        result_dir=Path(args.result_dir),
        waiting_root=Path(args.waiting_root),
        out_root=Path(args.out_root),
        tunnel_host=args.tunnel_host,
        tunnel_port=args.tunnel_port,
        check_tunnel=not args.skip_tunnel_check,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["summary"]["network_dry_run_allowed"] else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post-Claude intake gate for AMO waiting autonomous work.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--waiting-root", default=DEFAULT_WAITING_ROOT)
    parser.add_argument("--out-root", default=str(DEFAULT_AMO_WAITING_POST_CLAUDE_INTAKE_ROOT))
    parser.add_argument("--tunnel-host", default="127.0.0.1")
    parser.add_argument("--tunnel-port", type=int, default=15432)
    parser.add_argument("--skip-tunnel-check", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
