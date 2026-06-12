#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.existing_clients.amo_step1_snapshot import (
    DEFAULT_ENV_PATH,
    DEFAULT_OUT_ROOT,
    DEFAULT_PAGE_LIMIT,
    DEFAULT_SLEEP_SEC,
    AmoMcpClient,
    build_amo_step1_snapshot,
    read_mcp_env,
)
from mango_mvp.existing_clients.run_roots import cli_run_out_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TZ-14 Step 1 read-only AMO snapshot and duplicate detector.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--mcp-env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--transport", choices=("curl", "urllib"), default="curl")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--page-limit", type=int, default=DEFAULT_PAGE_LIMIT)
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--max-contacts", type=int, default=None)
    parser.add_argument("--max-leads", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--pilot-pages",
        type=int,
        default=None,
        help="Limit both contacts and leads pagination for a small read-only pilot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_pages = args.pilot_pages if args.pilot_pages is not None else args.max_pages
    generated_at = datetime.now(timezone.utc)
    config = read_mcp_env(args.mcp_env)
    config = replace(
        config,
        transport=args.transport,
        timeout_seconds=args.timeout_seconds or config.timeout_seconds,
    )
    summary = build_amo_step1_snapshot(
        project_root=args.project_root,
        out_root=cli_run_out_root(project_root=args.project_root, out_root=args.out_root, generated_at=generated_at),
        client=AmoMcpClient(config),
        page_limit=args.page_limit,
        sleep_sec=args.sleep_sec,
        max_contacts=args.max_contacts,
        max_leads=args.max_leads,
        max_pages=max_pages,
        progress_every=args.progress_every,
    )
    print(json.dumps({"status": "ok", "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
