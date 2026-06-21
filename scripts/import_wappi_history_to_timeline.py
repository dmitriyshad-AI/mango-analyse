#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from mango_mvp.customer_timeline.wappi_history_import import (
    WappiFetchLimits,
    WappiHistoryImportConfig,
    run_wappi_history_import,
    write_json_report,
)
from mango_mvp.integrations.amo_wappi_phase1 import AMO_WAPPI_ENV_FILE, DEFAULT_AMO_WAPPI_CONFIG_PATH


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        report = run_wappi_history_import(config)
        if config.out_path:
            write_json_report(config.out_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - compact CLI error without secrets.
        print(f"wappi history import failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read Wappi Telegram/Max history and import it into a local test customer_timeline SQLite DB. "
            "Defaults to dry-run; use --apply to write only the provided local DB."
        )
    )
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--env-file", type=Path, default=AMO_WAPPI_ENV_FILE)
    parser.add_argument("--phase1-config", type=Path, default=DEFAULT_AMO_WAPPI_CONFIG_PATH)
    parser.add_argument("--pairs-file", type=Path, default=Path.home() / ".mango_secrets" / "draft_loop_pairs.json")
    parser.add_argument("--auto-pairs-file", type=Path, default=Path.home() / ".mango_secrets" / "draft_loop_auto_pairs.json")
    parser.add_argument("--amo-auto-resolver", action="store_true", help="Resolve chats without static pairs through read-only AMO MCP.")
    parser.add_argument("--amo-mcp-env-file", type=Path, default=Path.home() / ".mango_secrets" / "foton_crm_readonly_mcp_connector.env")
    parser.add_argument("--shared-phone-stoplist", type=Path, default=Path.home() / ".mango_secrets" / "shared_phones_stoplist.json")
    parser.add_argument("--chat-limit-per-profile", type=int, default=50)
    parser.add_argument("--messages-per-chat", type=int, default=100)
    parser.add_argument("--message-limit-total", type=int, default=2000)
    parser.add_argument("--request-limit-total", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--show-all-chats", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--actor", default="wappi_history_timeline_import")
    parser.add_argument("--idempotency-key")
    parser.add_argument("--out", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> WappiHistoryImportConfig:
    return WappiHistoryImportConfig(
        timeline_db=Path(args.timeline_db),
        allowed_root=Path(args.allowed_root),
        tenant_id=args.tenant_id,
        env_file=args.env_file,
        phase1_config=args.phase1_config,
        pairs_file=args.pairs_file,
        auto_pairs_file=args.auto_pairs_file,
        amo_auto_resolver_enabled=bool(args.amo_auto_resolver),
        amo_mcp_env_file=args.amo_mcp_env_file,
        shared_phone_stoplist=args.shared_phone_stoplist,
        apply=bool(args.apply),
        actor=args.actor,
        idempotency_key=args.idempotency_key,
        out_path=args.out,
        limits=WappiFetchLimits(
            chat_limit_per_profile=args.chat_limit_per_profile,
            messages_per_chat=args.messages_per_chat,
            message_limit_total=args.message_limit_total,
            request_limit_total=args.request_limit_total,
            page_size=args.page_size,
            sleep_seconds=args.sleep_seconds,
            show_all_chats=bool(args.show_all_chats),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
