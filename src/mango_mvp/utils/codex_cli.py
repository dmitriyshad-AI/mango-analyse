from __future__ import annotations

import os


CODEX_SERVICE_TIER_ENV = "MANGO_CODEX_SERVICE_TIER"
DEFAULT_CODEX_SERVICE_TIER = "flex"


def codex_service_tier() -> str:
    return str(os.getenv(CODEX_SERVICE_TIER_ENV) or DEFAULT_CODEX_SERVICE_TIER).strip()


def append_codex_service_tier(cmd: list[str]) -> None:
    service_tier = codex_service_tier()
    if service_tier:
        cmd.extend(["-c", f'service_tier="{service_tier}"'])
