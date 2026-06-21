#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib import parse as url_parse

from mango_mvp.channels.subscription_llm import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION,
    SubscriptionLlmDraftProvider,
)
from mango_mvp.integrations.amo_wappi_phase1 import (
    AI_OFFICE_ENV_FILE,
    AMO_WAPPI_ENV_FILE,
    DEFAULT_AMO_WAPPI_CONFIG_PATH,
    AiOfficeAmoNoteClient,
    AiOfficeClientConfig,
    AmoWappiPhase1Config,
    WappiClientConfig,
    WappiPhase1Client,
    _json_http_request,
    load_env_file,
)
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy
from mango_mvp.integrations.draft_loop import (
    DEFAULT_DRAFT_LOOP_DIR,
    DEFAULT_STOP_PATH,
    AmoWappiDraftLoop,
    DraftLoopConfig,
    DraftLoopKey,
    DraftLoopProfile,
    DraftLoopState,
    WappiHistoryMessage,
    build_draft_loop_config_fingerprint,
    load_pairs_file,
    load_profiles_file,
)
from mango_mvp.integrations.amo_wappi_auto_resolver import (
    DEFAULT_STOPLIST_PATH,
    DRAFT_LOOP_AUTO_RESOLVER_ENV,
    build_amo_auto_resolver,
)
from mango_mvp.pilot_context_assembly import build_pilot_context_payload


DEFAULT_PROFILES_PATH = Path.home() / ".mango_secrets" / "amo_wappi_profiles.json"
DEFAULT_PAIRS_PATH = Path.home() / ".mango_secrets" / "draft_loop_pairs.json"
DEFAULT_AUTO_PAIRS_PATH = Path.home() / ".mango_secrets" / "draft_loop_auto_pairs.json"
DEFAULT_RETRO_DIR = Path.home() / ".mango_local" / "draft_loop_inventory"
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Wappi, build bot drafts, and write AMO draft notes for explicit test pairs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run exactly one polling cycle.")
    mode.add_argument("--loop", action="store_true", help="Run polling loop.")
    parser.add_argument("--interval-sec", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true", default=True, help="Do not POST AMO notes. Default.")
    parser.add_argument("--live-write", action="store_true", help="Allow AMO note POST after all allowlist checks.")
    parser.add_argument("--env-file", type=Path, default=AMO_WAPPI_ENV_FILE)
    parser.add_argument("--ai-office-env-file", type=Path, default=AI_OFFICE_ENV_FILE)
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--auto-pairs-file", type=Path, default=DEFAULT_AUTO_PAIRS_PATH)
    parser.add_argument("--phase1-config", type=Path, default=DEFAULT_AMO_WAPPI_CONFIG_PATH)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_DRAFT_LOOP_DIR)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_PATH)
    parser.add_argument("--chat-limit", type=int, default=50)
    parser.add_argument("--amo-mcp-env-file", type=Path, default=Path("~/.mango_secrets/foton_crm_readonly_mcp_connector.env").expanduser())
    parser.add_argument("--shared-phone-stoplist", type=Path, default=DEFAULT_STOPLIST_PATH)
    parser.add_argument("--retro-report", nargs="?", const="", default=None, help="Write an offline bot-vs-manager report outside the repo and exit.")
    parser.add_argument("--retro-lookback-hours", type=int, default=48)
    parser.add_argument("--retro-limit", type=int, default=30)
    parser.add_argument("--manager-outgoing-visible", choices=("unknown", "yes", "no"), default="unknown")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning", default="xhigh")
    parser.add_argument("--timeout-sec", type=int, default=240)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DraftLoopConfig:
    profiles = load_profiles_file(args.profiles_file)
    pairs = load_pairs_file(args.pairs_file)
    allowed: set[str] = set()
    if args.phase1_config.expanduser().exists():
        phase1 = AmoWappiPhase1Config.from_file(args.phase1_config)
        allowed.update(phase1.allowed_test_lead_ids)
    allowed.update(str(pair.lead_id) for pair in pairs.values())
    auto_pairs_path = getattr(args, "auto_pairs_file", DEFAULT_AUTO_PAIRS_PATH).expanduser()
    if auto_pairs_path.exists():
        allowed.update(str(pair.lead_id) for pair in load_pairs_file(auto_pairs_path, default_source="auto").values())
    if not allowed:
        allowed.update(str(pair.lead_id) for pair in pairs.values())
    local_dir = args.local_dir.expanduser()
    visibility = None
    if args.manager_outgoing_visible == "yes":
        visibility = True
    elif args.manager_outgoing_visible == "no":
        visibility = False
    raw_chat_limit = int(getattr(args, "chat_limit", 50) or 0)
    chat_limit = 0 if raw_chat_limit <= 0 else max(1, min(raw_chat_limit, 100))
    return DraftLoopConfig(
        profiles=profiles,
        pairs=pairs,
        auto_pairs_path=auto_pairs_path,
        allowed_test_lead_ids=frozenset(allowed),
        state_path=local_dir / "state.json",
        journal_path=local_dir / "journal.jsonl",
        manager_edit_log_path=local_dir / "manager_edits.jsonl",
        heartbeat_path=local_dir / "heartbeat.json",
        stop_path=args.stop_file.expanduser(),
        manager_outgoing_visible=visibility,
        chat_limit=chat_limit,
        config_fingerprint=build_draft_loop_config_fingerprint(
            getattr(args, "snapshot", DEFAULT_SNAPSHOT),
            gold_pack_version=DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION,
        ),
    )


def build_context_builder(snapshot_path: Path):
    def _build(
        key: DraftLoopKey,
        history: list[str] | tuple[str, ...],
        client_message: str,
        brand: str,
        *,
        channel: str = "telegram",
        dialogue_memory: Mapping[str, Any] | None = None,
        current_message_id: str = "",
    ) -> Mapping[str, Any]:
        return build_pilot_context_payload(
            current_text=client_message,
            snapshot_path=snapshot_path,
            active_brand=brand,
            recent_messages=tuple(history)[-10:],
            dialogue_memory=dialogue_memory or {},
            session_id=f"amo_draft_loop:{brand}:{key.profile_id}:{key.chat_id}",
            channel=f"wappi_{str(channel or 'telegram').strip().casefold()}",
            channel_thread_id=key.value,
            channel_user_id=key.chat_id,
            current_message_id=current_message_id,
            dialogue_contract_pipeline_enabled=True,
            sends_client_replies=False,
            debug_impersonation_enabled=False,
            crm_context={},
        )

    return _build


def build_safe_transport(ai_office_config: AiOfficeClientConfig, wappi_config: WappiClientConfig) -> DefaultDenyTransport:
    ai_office_host = url_parse.urlparse(ai_office_config.base_url).netloc.casefold()
    wappi_host = url_parse.urlparse(wappi_config.base_url).netloc.casefold()
    return DefaultDenyTransport(
        _json_http_request,
        policy=SafeTransportPolicy(
            wappi_hosts=frozenset(host for host in (wappi_host,) if host),
            amo_read_hosts=frozenset(),
            ai_office_hosts=frozenset(host for host in (ai_office_host,) if host),
        ),
    )


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def build_auto_resolver(args: argparse.Namespace):
    if not _truthy(os.getenv(DRAFT_LOOP_AUTO_RESOLVER_ENV)):
        return None
    return build_amo_auto_resolver(
        amo_mcp_env_file=args.amo_mcp_env_file,
        shared_phone_stoplist=args.shared_phone_stoplist,
        user_agent="mango-draft-loop-auto-resolver/1.0",
    )


def build_runner(args: argparse.Namespace) -> AmoWappiDraftLoop:
    load_env_file(args.env_file)
    if args.ai_office_env_file.expanduser().exists():
        load_env_file(args.ai_office_env_file)
    os.environ.setdefault(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)
    config = build_config(args)
    ai_office_config = AiOfficeClientConfig.from_env()
    wappi_config = WappiClientConfig.from_env()
    transport = build_safe_transport(ai_office_config, wappi_config)
    bot_provider = SubscriptionLlmDraftProvider(
        model=args.model,
        reasoning_effort=args.reasoning,
        timeout_sec=args.timeout_sec,
        cache_dir=None,
        codex_isolated=True,
    )
    return AmoWappiDraftLoop(
        config=config,
        wappi_client=WappiPhase1Client(wappi_config, transport=transport),
        amo_client=AiOfficeAmoNoteClient(ai_office_config, transport=transport),
        bot_provider=bot_provider,
        context_builder=build_context_builder(args.snapshot),
        state=DraftLoopState(config.state_path),
        auto_resolver=build_auto_resolver(args),
    )


def main() -> int:
    args = parse_args()
    dry_run = not bool(args.live_write)
    runner = build_runner(args)
    if args.retro_report is not None:
        report = runner.build_retro_report(lookback_hours=args.retro_lookback_hours, limit=args.retro_limit)
        if args.retro_report:
            target = Path(args.retro_report).expanduser()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            target = DEFAULT_RETRO_DIR / f"retro_compare_{stamp}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"retro_report": str(target), **dict(report.get("summary") or {})}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.loop:
        summary = runner.run_once(dry_run=dry_run)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    interval = max(5, int(args.interval_sec))
    while True:
        summary = runner.run_once(dry_run=dry_run)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
