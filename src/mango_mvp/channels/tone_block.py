from __future__ import annotations

import os
from typing import Any, Mapping

TONE_SELL_PROMPT_ENV = "TELEGRAM_TONE_SELL_PROMPT"
TONE_RICH_FORMAT_ENV = "TELEGRAM_TONE_RICH_FORMAT"
DIRECT_PATH_PILOT_CONFIG_ENV = "TELEGRAM_DIRECT_PATH_PILOT_CONFIG"
DIRECT_PATH_PILOT_CONFIG_VERSION = "pilot_gold_v1"


def truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def sell_prompt_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        for key in (TONE_SELL_PROMPT_ENV, "tone_sell_prompt_enabled"):
            if key in context:
                return truthy_value(context.get(key))
    return truthy_value(os.getenv(TONE_SELL_PROMPT_ENV))


def tone_rich_format_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        for key in (TONE_RICH_FORMAT_ENV, "tone_rich_format_enabled"):
            if key in context:
                return truthy_value(context.get(key))
    if TONE_RICH_FORMAT_ENV in os.environ:
        return truthy_value(os.getenv(TONE_RICH_FORMAT_ENV))
    if isinstance(context, Mapping):
        profile = str(context.get(DIRECT_PATH_PILOT_CONFIG_ENV) or context.get("direct_path_pilot_config") or "").strip()
        if profile == DIRECT_PATH_PILOT_CONFIG_VERSION:
            return True
    return str(os.getenv(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip() == DIRECT_PATH_PILOT_CONFIG_VERSION
