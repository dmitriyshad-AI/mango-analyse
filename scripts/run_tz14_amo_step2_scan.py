#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.existing_clients.amo_step1_snapshot import (
    DEFAULT_ENV_PATH,
    DEFAULT_PAGE_LIMIT,
    DEFAULT_SLEEP_SEC,
    AmoMcpClient,
    read_mcp_env,
)
from mango_mvp.existing_clients.amo_step2_scan import (
    DEFAULT_OUT_ROOT,
    DEFAULT_PROFILES_DB,
    NewLeadScanOptions,
    build_step2_scan,
    parse_datetime,
)
from mango_mvp.existing_clients.run_roots import cli_run_out_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TZ-14 Step 2 read-only AMO new-lead scan.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--profiles-db", type=Path, default=DEFAULT_PROFILES_DB)
    parser.add_argument("--mcp-env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--transport", choices=("curl", "urllib"), default="curl")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--since", help="Timezone-aware ISO datetime. Defaults to the last 15 minutes.")
    parser.add_argument("--page-limit", type=int, default=DEFAULT_PAGE_LIMIT)
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-leads", type=int, default=None)
    parser.add_argument("--callback-requests", type=Path, default=None)
    parser.add_argument("--enable-amo-notes", action="store_true", help="Fail-closed placeholder; live write is not implemented.")
    parser.add_argument("--enable-amo-tasks", action="store_true", help="Fail-closed placeholder; live write is not implemented.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    since = parse_datetime(args.since) if args.since else None
    if args.since and since is None:
        raise SystemExit("--since must be an ISO datetime or epoch timestamp")
    generated_at = datetime.now(timezone.utc)
    config = read_mcp_env(args.mcp_env)
    config = replace(
        config,
        transport=args.transport,
        timeout_seconds=args.timeout_seconds or config.timeout_seconds,
    )
    summary = build_step2_scan(
        NewLeadScanOptions(
            project_root=args.project_root,
            out_root=cli_run_out_root(project_root=args.project_root, out_root=args.out_root, generated_at=generated_at),
            profiles_db=args.profiles_db,
            since=since,
            client=AmoMcpClient(config),
            page_limit=args.page_limit,
            sleep_sec=args.sleep_sec,
            max_pages=args.max_pages,
            max_leads=args.max_leads,
            callback_requests_path=args.callback_requests,
            enable_amo_notes=args.enable_amo_notes,
            enable_amo_tasks=args.enable_amo_tasks,
            generated_at=generated_at,
        )
    )
    print(json.dumps({"status": "ok", "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
