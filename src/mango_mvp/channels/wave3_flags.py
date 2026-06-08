"""Feature flag helpers for Wave 3 grounded-fact controls."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


PILOT_CONFIG_ENV = "TELEGRAM_DIRECT_PATH_PILOT_CONFIG"
PILOT_CONFIG_VERSION = "pilot_gold_v1"

WAVE3_ENV = "TELEGRAM_WAVE3"
PARTIAL_ANSWER_FLOOR_ENV = "TELEGRAM_PARTIAL_ANSWER_FLOOR"
CALC_OVER_GROUNDED_ENV = "TELEGRAM_CALC_OVER_GROUNDED"
PROMISE_VS_FACT_ENV = "TELEGRAM_PROMISE_VS_FACT"
SOFT_REDACT_GRADED_ENV = "TELEGRAM_SOFT_REDACT_GRADED"
SCOPE_ADDRESSED_ENV = "TELEGRAM_SCOPE_ADDRESSED"
PER_CLAUSE_GATE_ENV = "TELEGRAM_PER_CLAUSE_GATE"
CLOSED_WORLD_NEGATIVE_ENV = "TELEGRAM_CLOSED_WORLD_NEGATIVE"
ELLIPSIS_RESOLVE_ENV = "TELEGRAM_ELLIPSIS_RESOLVE"

PILOT_GOLD_WAVE1_FLAGS = frozenset(
    {
        "TELEGRAM_NUMBER_GATE_SCOPE_AWARE",
        "TELEGRAM_VERIFIER_HANDOFF_CLAIMS",
    }
)


def truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on", "да", "вкл"}


def explicit_flag_value(context: Mapping[str, Any] | None, env_name: str) -> bool | None:
    if isinstance(context, Mapping) and env_name in context:
        return truthy_value(context.get(env_name))
    raw = os.getenv(env_name)
    if raw is not None:
        return truthy_value(raw)
    return None


def pilot_gold_config_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        for key in (PILOT_CONFIG_ENV, "direct_path_pilot_config", "pilot_config"):
            value = str(context.get(key) or "").strip()
            if value:
                return value == PILOT_CONFIG_VERSION
    return str(os.getenv(PILOT_CONFIG_ENV) or "").strip() == PILOT_CONFIG_VERSION


def pilot_gold_wave1_flag_enabled(context: Mapping[str, Any] | None, env_name: str) -> bool:
    explicit = explicit_flag_value(context, env_name)
    if explicit is not None:
        return explicit
    return env_name in PILOT_GOLD_WAVE1_FLAGS and pilot_gold_config_enabled(context)


def wave3_flag_enabled(context: Mapping[str, Any] | None, env_name: str) -> bool:
    explicit = explicit_flag_value(context, env_name)
    if explicit is not None:
        return explicit
    return bool(explicit_flag_value(context, WAVE3_ENV))
