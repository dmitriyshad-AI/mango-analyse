from __future__ import annotations

"""Source-level bot visibility policy for Customer Timeline.

Raw mail archives, including `mail_archive_stage2`, are manager-only until the
separate E4b opening step explicitly promotes vetted chunks under semantic
regression review. E1 may compute candidate diagnostics, but raw mail sources
must not write bot-visible chunks directly.
"""

from typing import Optional

from mango_mvp.customer_timeline.ids import normalize_key


CUSTOMER_TIMELINE_SOURCE_POLICY_VERSION = "customer_timeline_source_policy_v1"

BOT_FORBIDDEN_SOURCE_SYSTEMS = frozenset(
    {
        "mail_archive",
        "mail_archive_stage2",
        "channel_snapshot",
        "telegram_history",
        "amo_events_created_at",
        "amo_leads_updated_at",
        "amo_contacts_updated_at",
        "amocrm_event",
    }
)


def is_bot_forbidden_source_system(source_system: Optional[str]) -> bool:
    if not source_system:
        return False
    return normalize_key(source_system, "source_system") in BOT_FORBIDDEN_SOURCE_SYSTEMS


def assert_bot_context_chunk_source_policy(
    *,
    source_system: Optional[str],
    allowed_for_bot: bool,
    requires_manager_review: bool,
) -> None:
    if not is_bot_forbidden_source_system(source_system):
        return
    if allowed_for_bot or not requires_manager_review:
        raise ValueError(
            f"{normalize_key(source_system, 'source_system')} bot context chunks must be "
            "stored with allowed_for_bot=False and requires_manager_review=True"
        )
