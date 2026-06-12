#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.existing_clients.amo_step1_snapshot import DEFAULT_ENV_PATH, AmoMcpClient, read_mcp_env
from mango_mvp.existing_clients.amo_step3_contact_cards import (
    DEFAULT_AMO_SNAPSHOT_DB,
    DEFAULT_OUT_ROOT,
    DEFAULT_PROFILES_DB,
    ContactCardOptions,
    build_contact_card_stage_a,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TZ-14 Step 3A dry-run AMO contact card package.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--profiles-db", type=Path, default=DEFAULT_PROFILES_DB)
    parser.add_argument("--amo-snapshot-db", type=Path, default=DEFAULT_AMO_SNAPSHOT_DB)
    parser.add_argument("--stage-a-families", type=int, default=20)
    parser.add_argument("--mcp-env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--transport", choices=("curl", "urllib"), default="curl")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--skip-live-field-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = None
    if not args.skip_live_field_check:
        config = read_mcp_env(args.mcp_env)
        config = replace(
            config,
            transport=args.transport,
            timeout_seconds=args.timeout_seconds or config.timeout_seconds,
        )
        client = AmoMcpClient(config)
    summary = build_contact_card_stage_a(
        ContactCardOptions(
            project_root=args.project_root,
            out_root=args.out_root,
            profiles_db=args.profiles_db,
            amo_snapshot_db=args.amo_snapshot_db,
            client=client,
            stage_a_families=args.stage_a_families,
            generated_at=datetime.now(timezone.utc),
        )
    )
    print(json.dumps({"status": "ok", "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
