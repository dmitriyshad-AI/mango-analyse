#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mango_mvp.integrations.amo_wappi_phase1 import WappiClientConfig, WappiPhase1Client, _json_http_request, load_env_file
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy
from mango_mvp.integrations.draft_loop import DraftLoopProfile, load_profiles_file
from mango_mvp.replay_exam.exporter import RAW_ROOT, assert_raw_output_path, export_recent_dialogs

DEFAULT_PROFILES_FILE = Path.home() / ".mango_secrets" / "amo_wappi_profiles.json"
DEFAULT_ENV_FILE = Path.home() / ".mango_secrets" / "amo_wappi.env"
DEFAULT_REPLAY_PROFILE_IDS = (
    "ec2eed50-b55f",
    "18b255b8-7a67",
    "2952990f-9e4c",
    "152b441d-81a2",
)
MOSCOW_TZ = ZoneInfo("Europe/Moscow")


def _profile_filter(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def _selected_profiles(path: Path, profile_ids: set[str]) -> list[DraftLoopProfile]:
    profiles = load_profiles_file(path)
    selected = [profiles[profile_id] for profile_id in profile_ids if profile_id in profiles]
    missing = sorted(profile_ids - set(profiles))
    if missing:
        raise SystemExit(f"Replay profile ids are missing from {path}: {', '.join(missing)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Export recent Wappi dialogs for local replay exam.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES_FILE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--profile-ids", default=",".join(DEFAULT_REPLAY_PROFILE_IDS))
    parser.add_argument("--per-profile", type=int, default=25)
    parser.add_argument("--chat-page-limit", type=int, default=100)
    parser.add_argument("--max-chat-pages", type=int, default=20)
    parser.add_argument("--message-page-limit", type=int, default=100)
    parser.add_argument("--max-message-pages", type=int, default=50)
    parser.add_argument("--allow-live-wappi-read", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now(MOSCOW_TZ).strftime("%Y%m%d_%H%M%S")
    out_root = assert_raw_output_path(args.raw_root.expanduser() / f"wappi_replay_raw_{stamp}" / "manifest.json").parent
    if not args.allow_live_wappi_read:
        raise SystemExit("Refusing live Wappi read without --allow-live-wappi-read and owner confirmation.")
    load_env_file(args.env_file)
    profiles = _selected_profiles(args.profiles_file.expanduser(), _profile_filter(args.profile_ids))
    transport = DefaultDenyTransport(
        _json_http_request,
        policy=SafeTransportPolicy.wappi_read_only(),
    )
    client = WappiPhase1Client(WappiClientConfig.from_env(), transport=transport)
    manifest = export_recent_dialogs(
        client,
        profiles=profiles,
        raw_root=out_root,
        per_profile=args.per_profile,
        chat_page_limit=args.chat_page_limit,
        max_chat_pages=args.max_chat_pages,
        message_page_limit=args.message_page_limit,
        max_message_pages=args.max_message_pages,
    )
    manifest_path = out_root / "manifest.json"
    # Keep stdout free of raw message content and secret values.
    print(f"raw_manifest={manifest_path}")
    print(f"dialog_count={manifest['dialog_count']}")
    print(f"message_count={manifest['message_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
