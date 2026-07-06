from __future__ import annotations

"""Source-level bot visibility policy for Customer Timeline.

Raw mail archives, including `mail_archive_stage2`, are manager-only until the
separate E4b opening step explicitly promotes vetted chunks under semantic
regression review. E1 may compute candidate diagnostics, but raw mail sources
must not write bot-visible chunks directly.
"""

import os
from pathlib import Path
from typing import Optional

from mango_mvp.customer_timeline.ids import normalize_key


CUSTOMER_TIMELINE_SOURCE_POLICY_VERSION = "customer_timeline_source_policy_v1"
MAIL_STAGE2_BOT_VISIBLE_ENV = "CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE"
MAIL_STAGE2_BOT_VISIBLE_ALLOW_TEST_PATHS_ENV = "CUSTOMER_TIMELINE_E4B_MAIL_STAGE2_BOT_VISIBLE_ALLOW_TEST_PATHS"
CHANNEL_HISTORY_BOT_VISIBLE_ENV = "CUSTOMER_TIMELINE_E4B_CHANNEL_HISTORY_BOT_VISIBLE"
CHANNEL_HISTORY_BOT_VISIBLE_ALLOW_TEST_PATHS_ENV = "CUSTOMER_TIMELINE_E4B_CHANNEL_HISTORY_BOT_VISIBLE_ALLOW_TEST_PATHS"
MAIL_STAGE2_SOURCE_SYSTEM = "mail_archive_stage2"
TELEGRAM_HISTORY_SOURCE_SYSTEM = "telegram_history"
WAPPI_TELEGRAM_SOURCE_SYSTEM = "wappi_telegram"
WAPPI_MAX_SOURCE_SYSTEM = "wappi_max"
MANGO_PROCESSED_SOURCE_SYSTEM = "mango_processed_summary"
CHANNEL_HISTORY_SOURCE_SYSTEMS = frozenset(
    {
        TELEGRAM_HISTORY_SOURCE_SYSTEM,
        WAPPI_TELEGRAM_SOURCE_SYSTEM,
        WAPPI_MAX_SOURCE_SYSTEM,
    }
)
_TRUTHY_VALUES = {"1", "true", "yes", "on", "да", "y"}

BOT_FORBIDDEN_SOURCE_SYSTEMS = frozenset(
    {
        "mail_archive",
        MAIL_STAGE2_SOURCE_SYSTEM,
        "channel_snapshot",
        TELEGRAM_HISTORY_SOURCE_SYSTEM,
        WAPPI_TELEGRAM_SOURCE_SYSTEM,
        WAPPI_MAX_SOURCE_SYSTEM,
        MANGO_PROCESSED_SOURCE_SYSTEM,
        "amo_events_created_at",
        "amo_leads_updated_at",
        "amo_contacts_updated_at",
        "amocrm_event",
    }
)


def mail_stage2_bot_visible_enabled(value: object = None, *, timeline_db_path: Path | str | None = None) -> bool:
    if value is None:
        value = os.getenv(MAIL_STAGE2_BOT_VISIBLE_ENV)
    if str(value or "").strip().casefold() not in _TRUTHY_VALUES:
        return False
    if str(os.getenv(MAIL_STAGE2_BOT_VISIBLE_ALLOW_TEST_PATHS_ENV) or "").strip().casefold() in _TRUTHY_VALUES:
        return True
    return _is_e4b_staging_path(timeline_db_path)


def channel_history_bot_visible_enabled(value: object = None, *, timeline_db_path: Path | str | None = None) -> bool:
    if value is None:
        value = os.getenv(CHANNEL_HISTORY_BOT_VISIBLE_ENV)
    if str(value or "").strip().casefold() not in _TRUTHY_VALUES:
        return False
    if str(os.getenv(CHANNEL_HISTORY_BOT_VISIBLE_ALLOW_TEST_PATHS_ENV) or "").strip().casefold() in _TRUTHY_VALUES:
        return True
    return _is_e4b_staging_path(timeline_db_path)


def is_bot_forbidden_source_system(source_system: Optional[str], *, timeline_db_path: Path | str | None = None) -> bool:
    if not source_system:
        return False
    normalized = normalize_key(source_system, "source_system")
    if normalized == MAIL_STAGE2_SOURCE_SYSTEM and mail_stage2_bot_visible_enabled(timeline_db_path=timeline_db_path):
        return False
    if normalized in CHANNEL_HISTORY_SOURCE_SYSTEMS and channel_history_bot_visible_enabled(timeline_db_path=timeline_db_path):
        return False
    return normalized in BOT_FORBIDDEN_SOURCE_SYSTEMS


def assert_bot_context_chunk_source_policy(
    *,
    source_system: Optional[str],
    allowed_for_bot: bool,
    requires_manager_review: bool,
    timeline_db_path: Path | str | None = None,
) -> None:
    if not is_bot_forbidden_source_system(source_system, timeline_db_path=timeline_db_path):
        return
    if allowed_for_bot or not requires_manager_review:
        raise ValueError(
            f"{normalize_key(source_system, 'source_system')} bot context chunks must be "
            "stored with allowed_for_bot=False and requires_manager_review=True"
        )


def _is_e4b_staging_path(path: Path | str | None) -> bool:
    if path is None:
        return False
    resolved = Path(path).expanduser().resolve(strict=False)
    parts = tuple(part.casefold() for part in resolved.parts)
    if any("customer_timeline_prod_" in part for part in parts):
        return False
    for index, part in enumerate(parts[:-1]):
        if part == ".codex_local" and parts[index + 1] == "staging":
            return True
    return False
