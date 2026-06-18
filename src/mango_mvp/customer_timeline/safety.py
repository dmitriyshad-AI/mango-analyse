from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


CUSTOMER_TIMELINE_SAFETY_SCHEMA_VERSION = "customer_timeline_safety_v1"


def customer_timeline_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": CUSTOMER_TIMELINE_SAFETY_SCHEMA_VERSION,
        "read_only_source_systems": True,
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "live_send": False,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "runtime_db_writes": False,
        "mutate_stable_runtime": False,
        "stable_runtime_writes": False,
        "delete_source_artifacts": False,
        "store_raw_files_in_sqlite": False,
        "identity_conflicts_auto_merge": False,
        "old_to_new_customer_id_mapping_required": True,
        "brand_blocks_identity_merge": False,
    }


def blocked_live_actions() -> tuple[str, ...]:
    return (
        "write_crm",
        "write_tallanto",
        "send_email",
        "send_messenger",
        "live_send",
        "run_asr",
        "run_ra",
        "write_runtime_db",
        "runtime_db_writes",
        "mutate_stable_runtime",
        "stable_runtime_writes",
        "delete_source_artifacts",
    )


def assert_customer_timeline_safety_contract(contract: Mapping[str, Any]) -> None:
    for action in blocked_live_actions():
        if contract.get(action) is not False:
            raise ValueError(f"customer timeline safety requires {action}=False")
    if contract.get("read_only_source_systems") is not True:
        raise ValueError("customer timeline safety requires read_only_source_systems=True")
    if contract.get("store_raw_files_in_sqlite") is not False:
        raise ValueError("customer timeline safety requires store_raw_files_in_sqlite=False")
    if contract.get("identity_conflicts_auto_merge") is not False:
        raise ValueError("customer timeline safety requires identity_conflicts_auto_merge=False")
    if contract.get("old_to_new_customer_id_mapping_required") is not True:
        raise ValueError("customer timeline safety requires old_to_new_customer_id_mapping_required=True")
    if contract.get("brand_blocks_identity_merge") is not False:
        raise ValueError("customer timeline safety requires brand_blocks_identity_merge=False")


def is_stable_runtime_path(path: Path | str) -> bool:
    return any(part.casefold() == "stable_runtime" for part in Path(path).parts)


def guard_customer_timeline_output_path(path: Path | str, allowed_root: Path | str) -> Path:
    resolved = Path(path).resolve(strict=False)
    root = Path(allowed_root).resolve(strict=False)
    if is_stable_runtime_path(resolved):
        raise ValueError(f"customer timeline output must not be under stable_runtime: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"customer timeline output must stay under allowed root: {root}") from exc
    return resolved
