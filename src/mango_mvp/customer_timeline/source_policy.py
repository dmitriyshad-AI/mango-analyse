from __future__ import annotations

"""Source-level bot visibility policy for Customer Timeline.

Raw mail archives, including `mail_archive_stage2`, are manager-only until the
separate E4b opening step explicitly promotes vetted chunks under semantic
regression review. E1 may compute candidate diagnostics, but raw mail sources
must not write bot-visible chunks directly.
"""

import os
from pathlib import Path
from typing import Any, Mapping, Optional

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
BOT_SAFE_SUMMARY_SOURCE_SYSTEM = "customer_timeline_bot_safe_summary"
BOT_SAFE_SUMMARY_CHUNK_TYPE = "bot_safe_summary"
BOT_SAFE_SUMMARY_SCHEMA_VERSION = "customer_timeline_bot_safe_summary_v1"
BOT_SAFE_SUMMARY_ACTOR = "customer_timeline_bot_safe_summary_builder"
PURCHASE_HISTORY_SOURCE_SYSTEM = "customer_purchases_v1"
PURCHASE_HISTORY_CHUNK_TYPE = "purchase_history"
PURCHASE_HISTORY_PROJECTION_VERSION = "customer_purchases_v1_bot_neutral_v1"
PURCHASE_HISTORY_PROJECTION_OWNER = "stage5_money_ingest"
PURCHASE_HISTORY_SEMANTIC_SCOPE = "historical_payment_brand_neutral"
PURCHASE_HISTORY_BOT_TEXT = (
    "В истории клиента есть подтверждённая входящая оплата; "
    "её бренд и текущий доступ не подтверждены."
)
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


def is_non_contentful_call_record(value: Mapping[str, Any]) -> bool:
    """One structural interpretation of an explicitly non-conversational call."""

    record = value.get("record") if isinstance(value.get("record"), Mapping) else {}
    call = record.get("call") if isinstance(record.get("call"), Mapping) else {}
    analysis = record.get("call_analysis") if isinstance(record.get("call_analysis"), Mapping) else {}
    metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
    payloads = (value, record, call, analysis, metadata)
    contentful_values = {
        str(payload.get("contentful", "")).strip().casefold()
        for payload in payloads
    }
    call_types = {
        str(payload.get(field, "")).strip().casefold()
        for payload in payloads
        for field in ("call_type", "subject")
    }
    return bool(contentful_values & {"0", "0.0", "false", "нет", "no", "non_conversation"}) or (
        "non_conversation" in call_types
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


def assert_canonical_bot_projection_writer(
    *,
    source_system: Optional[str],
    metadata: Mapping[str, Any],
    actor: str,
) -> None:
    """Reject self-declared canonical owner proof from any other writer."""

    if source_system == BOT_SAFE_SUMMARY_SOURCE_SYSTEM:
        claims_owner_proof = any(
            key in metadata
            for key in ("client_safe_provenance", "projection_owner", "projection_version")
        )
        if not claims_owner_proof:
            return
        if actor != BOT_SAFE_SUMMARY_ACTOR:
            raise ValueError("canonical bot-safe summaries require their projection owner actor")
        return
    if source_system != PURCHASE_HISTORY_SOURCE_SYSTEM:
        return
    claims_owner_proof = any(
        key in metadata
        for key in ("client_safe_provenance", "projection_owner", "projection_version")
    )
    if not claims_owner_proof:
        return
    if actor != PURCHASE_HISTORY_PROJECTION_OWNER:
        raise ValueError("canonical purchase history requires the Stage5 projection owner actor")


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
