#!/usr/bin/env python3
"""Derive deterministic customer_timeline signals.

Dry-run is the default and opens customer_timeline.sqlite in SQLite read-only
mode. Use --apply to write only to that local timeline DB.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline import (
    DEFAULT_HOT_LEAD_SILENCE_DAYS,
    DERIVED_SIGNAL_RECOMPUTE_SCHEMA_VERSION,
    CustomerTimelineSQLiteStore,
    recompute_customer_signals,
)
from mango_mvp.customer_timeline.ids import normalize_key, optional_text, require_timezone
from mango_mvp.customer_timeline.safety import assert_customer_timeline_safety_contract, customer_timeline_safety_contract
from mango_mvp.customer_timeline.store import guard_customer_timeline_sqlite_path


DERIVE_CUSTOMER_TIMELINE_SIGNALS_CLI_SCHEMA_VERSION = "derive_customer_timeline_signals_cli_v1"


@dataclass(frozen=True)
class DeriveCustomerTimelineSignalsConfig:
    timeline_db: Path
    allowed_root: Path
    tenant_id: str
    customer_id: Optional[str] = None
    apply: bool = False
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hot_lead_silence_days: int = DEFAULT_HOT_LEAD_SILENCE_DAYS
    limit: int = 500
    actor: str = "derive_customer_timeline_signals"

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        db_path = guard_customer_timeline_sqlite_path(Path(self.timeline_db).expanduser())
        require_timezone(self.as_of, "as_of")
        if self.hot_lead_silence_days <= 0:
            raise ValueError("hot_lead_silence_days must be positive")
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "timeline_db", db_path)
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "customer_id", optional_text(self.customer_id))


def run_derive_customer_timeline_signals(config: DeriveCustomerTimelineSignalsConfig) -> Mapping[str, Any]:
    assert_customer_timeline_safety_contract(customer_timeline_safety_contract())
    store = (
        CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root)
        if config.apply
        else CustomerTimelineSQLiteStore.open_read_only(config.timeline_db, allowed_root=config.allowed_root)
    )
    try:
        customer_ids = (config.customer_id,) if config.customer_id else _list_customer_ids(store, config.tenant_id, config.limit)
        results = [
            recompute_customer_signals(
                store,
                config.tenant_id,
                customer_id,
                as_of=config.as_of,
                apply=config.apply,
                hot_lead_silence_days=config.hot_lead_silence_days,
                actor=config.actor,
            )
            for customer_id in customer_ids
        ]
    finally:
        store.close()
    return {
        "schema_version": DERIVE_CUSTOMER_TIMELINE_SIGNALS_CLI_SCHEMA_VERSION,
        "recompute_schema_version": DERIVED_SIGNAL_RECOMPUTE_SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run",
        "write_applied": bool(config.apply),
        "tenant_id": config.tenant_id,
        "customer_id": config.customer_id,
        "as_of": config.as_of.isoformat(),
        "hot_lead_silence_days": config.hot_lead_silence_days,
        "customers": len(results),
        "summary": _merge_results(results),
        "results": [result.to_json_dict() for result in results],
        "safety": {
            **dict(customer_timeline_safety_contract()),
            "write_customer_timeline_sqlite": bool(config.apply),
            "network_calls": False,
            "llm_calls": False,
        },
    }


def _list_customer_ids(store: CustomerTimelineSQLiteStore, tenant_id: str, limit: int) -> tuple[str, ...]:
    remaining = int(limit)
    cursor: Optional[str] = None
    customer_ids: list[str] = []
    while remaining > 0:
        page = store.list_customers(tenant_id, limit=min(remaining, 500), cursor=cursor)
        items = tuple(page["items"])
        if not items:
            break
        customer_ids.extend(str(item["customer_id"]) for item in items)
        remaining = int(limit) - len(customer_ids)
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return tuple(customer_ids)


def _merge_results(results: Sequence[Any]) -> Mapping[str, Any]:
    status_counts: dict[str, int] = {}
    signal_type_counts: dict[str, int] = {}
    write_status_counts: dict[str, int] = {}
    for result in results:
        _merge_counts(status_counts, result.status_counts)
        _merge_counts(signal_type_counts, result.signal_type_counts)
        _merge_counts(write_status_counts, result.write_status_counts)
    return {
        "signals_total": sum(status_counts.values()),
        "status_counts": status_counts,
        "signal_type_counts": signal_type_counts,
        "write_status_counts": write_status_counts,
    }


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def parse_as_of(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require_timezone(parsed, "as_of")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive deterministic customer_timeline signals. Defaults to dry-run.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--customer-id")
    parser.add_argument("--as-of", help="Timezone-aware ISO datetime. Defaults to current UTC time.")
    parser.add_argument("--hot-lead-silence-days", type=int, default=DEFAULT_HOT_LEAD_SILENCE_DAYS)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--actor", default="derive_customer_timeline_signals")
    parser.add_argument("--apply", action="store_true", help="Write only to the configured local customer_timeline.sqlite.")
    return parser


def config_from_args(args: argparse.Namespace) -> DeriveCustomerTimelineSignalsConfig:
    return DeriveCustomerTimelineSignalsConfig(
        timeline_db=Path(args.timeline_db),
        allowed_root=Path(args.allowed_root),
        tenant_id=args.tenant_id,
        customer_id=args.customer_id,
        apply=bool(args.apply),
        as_of=parse_as_of(args.as_of),
        hot_lead_silence_days=int(args.hot_lead_silence_days),
        limit=int(args.limit),
        actor=args.actor,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_derive_customer_timeline_signals(config_from_args(args))
    except Exception as exc:  # noqa: BLE001 - compact CLI error for operators.
        print(f"derive customer timeline signals failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
