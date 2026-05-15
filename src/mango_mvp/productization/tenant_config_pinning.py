from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from mango_mvp.productization.tenant_config import TenantConfigLoadResult, clean, load_tenant_config


EXPECTED_TENANT_CONFIG_SHA256 = "9de1e6363171ea619cdd52055ce16f0b2b71c499a6d00fd88b2f55e70711c288"
EXPECTED_TENANT_CONFIG_SCHEMA_VERSION = "tenant_config_v1"
EXPECTED_TENANT_ID = "foton"


def check_tenant_config_pin(
    load_result: TenantConfigLoadResult | None,
    *,
    expected_sha256: str = EXPECTED_TENANT_CONFIG_SHA256,
    expected_tenant_id: str = EXPECTED_TENANT_ID,
    expected_schema_version: str = EXPECTED_TENANT_CONFIG_SCHEMA_VERSION,
) -> dict[str, Any]:
    if load_result is None:
        return {
            "passed": False,
            "reason": "tenant_config_not_loaded",
            "expected_sha256": expected_sha256,
            "actual_sha256": "",
            "path": "",
            "tenant_id": "",
            "schema_version": "",
        }

    tenant_id = clean(load_result.config.get("tenant_id"))
    schema_version = clean(load_result.config.get("schema_version"))
    actual_sha256 = clean(load_result.sha256)
    reason = "ok"
    passed = True
    if actual_sha256 != expected_sha256:
        passed = False
        reason = "tenant_config_sha256_mismatch"
    elif tenant_id != expected_tenant_id:
        passed = False
        reason = "tenant_config_tenant_id_mismatch"
    elif schema_version != expected_schema_version:
        passed = False
        reason = "tenant_config_schema_version_mismatch"

    return {
        "passed": passed,
        "reason": reason,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "path": clean(load_result.path),
        "tenant_id": tenant_id,
        "schema_version": schema_version,
    }


def tenant_config_pin_check_off() -> dict[str, Any]:
    return {
        "passed": True,
        "reason": "tenant_config_pin_check_off",
        "expected_sha256": "",
        "actual_sha256": "",
        "path": "",
        "tenant_id": "",
        "schema_version": "",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check and print tenant config pin metadata.")
    parser.add_argument("--print-current", action="store_true")
    parser.add_argument("--path", default="")
    args = parser.parse_args(argv)

    if not args.print_current:
        parser.error("--print-current is required")
    if not args.path:
        parser.error("--path is required for --print-current")

    result = load_tenant_config(Path(args.path))
    if result is None:
        parser.error("tenant config was not loaded")
    payload = {
        "path": result.path,
        "sha256": result.sha256,
        "tenant_id": clean(result.config.get("tenant_id")),
        "schema_version": clean(result.config.get("schema_version")),
        "constant_to_update": "EXPECTED_TENANT_CONFIG_SHA256",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "EXPECTED_TENANT_CONFIG_SCHEMA_VERSION",
    "EXPECTED_TENANT_CONFIG_SHA256",
    "EXPECTED_TENANT_ID",
    "check_tenant_config_pin",
    "tenant_config_pin_check_off",
]


if __name__ == "__main__":
    raise SystemExit(main())
