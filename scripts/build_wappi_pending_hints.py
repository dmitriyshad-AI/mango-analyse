#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.customer_timeline.wappi_history_import import (
    WappiPairCustomerResolver,
    build_readonly_wappi_client,
    load_wappi_pairs,
    profiles_from_phase1_config,
)
from mango_mvp.customer_timeline.wappi_pending_hints import (
    build_pending_hints,
    load_pending_wappi_chats,
    validate_human_decisions,
    write_hint_pack,
)
from mango_mvp.integrations.amo_wappi_auto_resolver import (
    DEFAULT_AMO_MCP_ENV_PATH,
    DEFAULT_STOPLIST_PATH,
    build_amo_auto_resolver,
)
from mango_mvp.integrations.amo_wappi_phase1 import (
    AMO_WAPPI_ENV_FILE,
    DEFAULT_AMO_WAPPI_CONFIG_PATH,
    AmoWappiPhase1Config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build read-only human-review hints for pending Wappi chats.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--timeline-db", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--tenant-id", default="foton")
    build.add_argument("--expect-pending-chats", type=int, default=328)
    build.add_argument("--wappi-env-file", type=Path, default=AMO_WAPPI_ENV_FILE)
    build.add_argument("--phase1-config", type=Path, default=DEFAULT_AMO_WAPPI_CONFIG_PATH)
    build.add_argument("--pairs-file", type=Path, default=Path.home() / ".mango_secrets" / "draft_loop_pairs.json")
    build.add_argument("--auto-pairs-file", type=Path, default=Path.home() / ".mango_secrets" / "draft_loop_auto_pairs.json")
    build.add_argument("--amo-mcp-env-file", type=Path, default=DEFAULT_AMO_MCP_ENV_PATH)
    build.add_argument("--shared-phone-stoplist", type=Path, default=DEFAULT_STOPLIST_PATH)
    build.add_argument("--list-request-limit", type=int, default=100)
    build.add_argument("--messages-per-chat", type=int, default=5)
    build.add_argument("--sleep-seconds", type=float, default=0.2)
    build.add_argument("--amo-pause-seconds-per-call", type=float, default=1.05)
    build.add_argument("--review-limit", type=int, default=50)

    validate = sub.add_parser("validate-decisions")
    validate.add_argument("--hints-jsonl", type=Path, required=True)
    validate.add_argument("--decisions-csv", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate-decisions":
        print(json.dumps(validate_human_decisions(args.hints_jsonl, args.decisions_csv), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    pending = load_pending_wappi_chats(args.timeline_db, tenant_id=args.tenant_id)
    if len(pending) != args.expect_pending_chats:
        raise SystemExit(f"pending Wappi chat count changed: expected={args.expect_pending_chats}, actual={len(pending)}")
    phase1 = AmoWappiPhase1Config.from_file(args.phase1_config)
    profiles = profiles_from_phase1_config(phase1)
    resolver = WappiPairCustomerResolver.from_store(
        args.timeline_db,
        tenant_id=args.tenant_id,
        pairs=load_wappi_pairs(args.pairs_file, args.auto_pairs_file),
        amo_auto_resolver=build_amo_auto_resolver(
            amo_mcp_env_file=args.amo_mcp_env_file,
            shared_phone_stoplist=args.shared_phone_stoplist,
            user_agent="mango-wappi-pending-hints/1.0",
            require_known_brand=True,
        ),
    )
    rows, summary = build_pending_hints(
        client=build_readonly_wappi_client(args.wappi_env_file),
        profiles=profiles,
        resolver=resolver,
        pending_chats=pending,
        list_request_limit=args.list_request_limit,
        messages_per_chat=args.messages_per_chat,
        sleep_seconds=args.sleep_seconds,
        amo_pause_seconds_per_call=args.amo_pause_seconds_per_call,
    )
    files = write_hint_pack(args.out_dir, rows, summary, review_limit=args.review_limit)
    print(json.dumps({"pending_chats": len(pending), "counts": summary, "files": files}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
