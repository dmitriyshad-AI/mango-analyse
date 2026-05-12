from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


TENANT_CONFIG_SCHEMA_VERSION = "tenant_config_v1"


@dataclass(frozen=True)
class TenantConfigLoadResult:
    path: str
    sha256: str
    config: Mapping[str, Any]


def load_tenant_config(path: str | Path | None) -> TenantConfigLoadResult | None:
    """Load non-secret tenant policy config for quality gates.

    This deliberately accepts plain JSON only. Credentials and OAuth tokens must
    stay in separate secret stores/env files.
    """
    if not path:
        return None
    config_path = Path(path).expanduser().resolve()
    raw = config_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    validate_tenant_config(payload)
    return TenantConfigLoadResult(
        path=str(config_path),
        sha256=hashlib.sha256(raw).hexdigest(),
        config=payload,
    )


def validate_tenant_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != TENANT_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"tenant config schema_version must be {TENANT_CONFIG_SCHEMA_VERSION!r}")
    tenant_id = clean(payload.get("tenant_id"))
    if not tenant_id:
        raise ValueError("tenant config requires tenant_id")
    for section in ("business", "crm", "privacy", "quality"):
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"tenant config requires object section: {section}")
    crm = payload.get("crm") or {}
    if not isinstance(crm.get("protected_fields"), list):
        raise ValueError("tenant config crm.protected_fields must be a list")
    if not isinstance(crm.get("target_fields"), list):
        raise ValueError("tenant config crm.target_fields must be a list")
    secrets = find_secret_like_keys(payload)
    if secrets:
        raise ValueError(f"tenant config must not contain secrets: {', '.join(sorted(secrets))}")


def tenant_config_summary(load_result: TenantConfigLoadResult | None) -> Mapping[str, Any]:
    if load_result is None:
        return {
            "loaded": False,
            "path": "",
            "sha256": "",
            "tenant_id": "",
            "schema_version": "",
        }
    return {
        "loaded": True,
        "path": load_result.path,
        "sha256": load_result.sha256,
        "tenant_id": clean(load_result.config.get("tenant_id")),
        "schema_version": clean(load_result.config.get("schema_version")),
    }


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def find_secret_like_keys(value: object, prefix: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = clean(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if any(token in key_text.casefold() for token in ("secret", "token", "password", "oauth")):
                result.add(path)
            result.update(find_secret_like_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(find_secret_like_keys(item, f"{prefix}[{index}]"))
    return result


__all__ = [
    "TENANT_CONFIG_SCHEMA_VERSION",
    "TenantConfigLoadResult",
    "load_tenant_config",
    "tenant_config_summary",
    "validate_tenant_config",
]
