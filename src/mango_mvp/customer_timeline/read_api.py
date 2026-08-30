from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from mango_mvp.customer_timeline.ids import normalize_key, require_text, require_timezone
from mango_mvp.customer_timeline.next_step_resolver import (
    resolve_customer_manager_action,
    resolve_customer_next_step,
)
from mango_mvp.customer_timeline.safety import blocked_live_actions, guard_customer_timeline_output_path
from mango_mvp.customer_timeline.source_policy import (
    PURCHASE_HISTORY_BOT_TEXT,
    PURCHASE_HISTORY_CHUNK_TYPE,
    PURCHASE_HISTORY_SOURCE_SYSTEM,
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    UNRESOLVED_CONFLICT_STATUSES,
    customer_timeline_sqlite_safety_contract,
    guard_customer_timeline_sqlite_path,
)
from mango_mvp.customer_timeline.tallanto_finished_grade import (
    FAMILY_LINK_SCOPE_OUT,
    family_link_timeline_scope_state,
    family_timeline_scope_decision,
)


CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION = "customer_timeline_read_api_v1"
READ_API_ROUTES = (
    "/health",
    "/summary",
    "/customers",
    "/customer",
    "/customer/timeline",
    "/customer/bot-context",
    "/search",
    "/conflicts",
)


@dataclass(frozen=True)
class CustomerTimelineReadApiConfig:
    timeline_db: Path
    allowed_root: Path

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).resolve(strict=False)
        db_path = guard_customer_timeline_sqlite_path(self.timeline_db)
        db_path = guard_customer_timeline_output_path(db_path, root)
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "timeline_db", db_path)


class CustomerTimelineReadApi:
    """Read-only facade for customer_timeline.sqlite.

    UI, channel bots and future HTTP routers should call this facade instead of
    reading SQLite tables directly. It opens the DB in SQLite read-only mode and
    does not expose mutation methods.
    """

    def __init__(self, store: CustomerTimelineSQLiteStore) -> None:
        if not store.read_only:
            raise ValueError("CustomerTimelineReadApi requires a read-only store")
        self.store = store

    @classmethod
    def open(cls, config: CustomerTimelineReadApiConfig) -> "CustomerTimelineReadApi":
        store = CustomerTimelineSQLiteStore.open_read_only(config.timeline_db, allowed_root=config.allowed_root)
        return cls(store)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "CustomerTimelineReadApi":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def health(self) -> Mapping[str, Any]:
        summary = self.store.summary()
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "status": "ok" if summary.get("validation_ok") else "blocked",
            "read_only": True,
            "db_path": summary["db_path"],
            "fts_enabled": bool(summary.get("fts_enabled")),
            "validation_ok": bool(summary.get("validation_ok")),
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def summary(self, tenant_id: str, *, recent_limit: int = 10) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        store_summary = self.store.summary()
        tenant_counts = self._tenant_counts(tenant)
        recent_runs = self.store.list_ingestion_runs(tenant, limit=bounded_limit(recent_limit, default=10, max_limit=100))
        recent_conflicts = self.list_conflicts(tenant, limit=bounded_limit(recent_limit, default=10, max_limit=100))
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "endpoint": "GET /summary",
            "tenant_id": tenant,
            "read_only": True,
            "summary": {
                "validation_ok": bool(store_summary.get("validation_ok")),
                "tenant_counts": tenant_counts,
                "events_without_customer": store_summary.get("soft_integrity", {}).get("events_without_customer", 0),
                "event_customer_missing": store_summary.get("soft_integrity", {}).get("event_customer_missing", 0),
                "bot_chunks_blocked_for_bot": store_summary.get("soft_integrity", {}).get("bot_chunks_blocked_for_bot", 0),
                "open_conflicts": self._count(
                    "timeline_conflicts",
                    "tenant_id = ? AND status IN (?, ?)",
                    (tenant, *sorted(UNRESOLVED_CONFLICT_STATUSES)),
                ),
                "recent_ingestion_runs": len(recent_runs["items"]),
            },
            "store": store_summary,
            "recent_ingestion_runs": recent_runs,
            "recent_conflicts": recent_conflicts,
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def list_customers(
        self,
        tenant_id: str,
        *,
        q: Optional[str] = None,
        identity_status: Optional[str] = None,
        updated_since: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        page_limit = bounded_limit(limit, default=50, max_limit=200)
        result = _operational_visible_page(
            fetch_page=lambda page_cursor: self.store.list_customers(
                tenant,
                q=q,
                identity_status=identity_status,
                updated_since=updated_since,
                limit=max(page_limit, 50),
                cursor=page_cursor,
            ),
            customer_id_of=lambda item: str(item.get("customer_id") or ""),
            out_of_scope_for_customer_ids=lambda customer_ids: (
                self._operational_out_of_scope_customer_ids(tenant, customer_ids)
            ),
            limit=page_limit,
            cursor=cursor,
        )
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "endpoint": "GET /customers",
            "tenant_id": tenant,
            "items": [project_customer(item) for item in result["items"]],
            "next_cursor": result.get("next_cursor"),
            "redaction": redaction_summary(bot_safe=False),
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def customer_profile(
        self,
        tenant_id: str,
        customer_id: str,
        *,
        event_limit: int = 25,
        bot_context_limit: int = 25,
        include_children: bool = True,
        as_of: Optional[datetime] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        customer = self.store.get_customer(tenant, customer_id)
        if customer is None:
            return {
                "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
                "endpoint": "GET /customer",
                "tenant_id": tenant,
                "customer_id": customer_id,
                "found": False,
                "safety": customer_timeline_read_api_safety_contract(),
            }
        links = self.store.list_identity_links(tenant, customer_id=customer["customer_id"], limit=200)
        opportunities = self._records(
            "customer_opportunities",
            "tenant_id = ? AND customer_id = ?",
            (tenant, customer["customer_id"]),
            order_by="opened_at DESC, opportunity_id",
            limit=100,
        )
        events = self.store.list_events_by_customer(
            tenant,
            customer["customer_id"],
            include_artifacts=include_children,
            include_signals=include_children,
            limit=bounded_limit(event_limit, default=25, max_limit=200),
        )
        last_event_at = self._last_event_at(tenant, customer["customer_id"])
        snapshot_as_of = last_event_at or first_text(customer, ("last_seen_at", "updated_at", "created_at"))
        snapshot_dt = parse_iso_datetime(snapshot_as_of)
        bot_context = self.bot_context(
            tenant,
            customer["customer_id"],
            allowed_only=False,
            as_of=as_of,
            limit=bounded_limit(bot_context_limit, default=25, max_limit=200),
        )
        conflicts = self.list_conflicts(tenant, customer_id=customer["customer_id"], limit=100)
        signals = self.store.list_signals_by_customer(
            tenant,
            customer["customer_id"],
            active_at=snapshot_dt,
            limit=100,
        )
        next_step_events = self.store.list_events_by_customer(
            tenant,
            customer["customer_id"],
            sort="asc",
            include_artifacts=False,
            include_signals=False,
            limit=500,
        )
        mappings = self._customer_id_mappings_for_profile(tenant, customer["customer_id"])
        readiness = {
            "events": len(events["items"]),
            "identity_links": len(links),
            "opportunities": len(opportunities),
            "signals": len(signals),
            "bot_context_chunks": len(bot_context["items"]),
            "bot_allowed_chunks": bot_context["summary"]["allowed_chunks"],
            "bot_review_required_chunks": bot_context["summary"]["review_required_chunks"],
            "open_conflicts": conflicts["summary"]["open_conflicts"],
            "safe_for_automatic_bot": bot_context["summary"]["allowed_chunks"] > 0 and conflicts["summary"]["open_conflicts"] == 0,
        }
        next_step_resolution = resolve_customer_next_step(
            next_step_events["items"],
            readiness=readiness,
            conflicts=conflicts["items"],
            customer_id=customer["customer_id"],
        ).to_informational_json_dict()
        manager_action = asdict(
            resolve_customer_manager_action(
                self.store._con,  # noqa: SLF001 - the read facade owns this read-only connection.
                tenant_id=tenant,
                customer_id=customer["customer_id"],
                as_of=as_of or self.store._clock(),  # noqa: SLF001 - one store clock owns the cutoff.
            )
        )
        manager_action["readiness_reason_codes"] = list(manager_action["readiness_reason_codes"])
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "endpoint": "GET /customer",
            "tenant_id": tenant,
            "customer_id": customer["customer_id"],
            "snapshot_as_of": snapshot_as_of,
            "last_event_at": last_event_at,
            "found": True,
            "customer": project_customer(customer),
            "customer_id_mappings": [project_customer_id_mapping(item) for item in mappings],
            "identity_links": [project_identity_link(item, audience="ui") for item in links],
            "opportunities": [project_opportunity(item) for item in opportunities],
            "manager_projection": project_manager_projection(
                customer=customer,
                links=links,
                opportunities=opportunities,
                events=events["items"],
                next_step_resolution=next_step_resolution,
                manager_action=manager_action,
            ),
            "timeline": {
                **events,
                "items": [project_event(item, include_artifacts=include_children, include_signals=include_children) for item in events["items"]],
            },
            "signals": [project_signal(item) for item in signals],
            "bot_context": bot_context,
            "conflicts": conflicts,
            "readiness": readiness,
            "next_step_resolution": next_step_resolution,
            "redaction": redaction_summary(bot_safe=False),
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def _last_event_at(self, tenant_id: str, customer_id: str) -> Optional[str]:
        page = self.store.list_events_by_customer(
            tenant_id,
            customer_id,
            sort="desc",
            include_artifacts=False,
            include_signals=False,
            limit=1,
        )
        items = page.get("items") or ()
        if not items:
            return None
        return str(items[0].get("event_at") or "") or None

    def _customer_id_mappings_for_profile(self, tenant_id: str, customer_id: str) -> tuple[Mapping[str, Any], ...]:
        by_id: dict[str, Mapping[str, Any]] = {}
        for item in self.store.list_customer_id_mappings(tenant_id, new_customer_id=customer_id, limit=500):
            by_id[str(item.get("mapping_id") or f"new:{len(by_id)}")] = item
        for item in self.store.list_customer_id_mappings(tenant_id, old_customer_id=customer_id, limit=500):
            by_id[str(item.get("mapping_id") or f"old:{len(by_id)}")] = item
        return tuple(by_id[key] for key in sorted(by_id))

    def customer_timeline(
        self,
        tenant_id: str,
        customer_id: str,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        sort: str = "desc",
        event_types: Sequence[str] = (),
        source_systems: Sequence[str] = (),
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_artifacts: bool = True,
        include_signals: bool = True,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        result = self.store.list_events_by_customer(
            tenant,
            customer_id,
            event_types=event_types,
            source_systems=source_systems,
            since=since,
            until=until,
            sort=sort,
            include_artifacts=include_artifacts,
            include_signals=include_signals,
            limit=bounded_limit(limit, default=50, max_limit=200),
            cursor=cursor,
        )
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "endpoint": "GET /customer/timeline",
            "tenant_id": tenant,
            "customer_id": customer_id,
            "items": [project_event(item, include_artifacts=include_artifacts, include_signals=include_signals) for item in result["items"]],
            "next_cursor": result.get("next_cursor"),
            "redaction": redaction_summary(bot_safe=False),
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def bot_context(
        self,
        tenant_id: str,
        customer_id: str,
        *,
        allowed_only: bool = True,
        as_of: Optional[datetime] = None,
        limit: int = 50,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        normalized_customer_id = require_text(customer_id, "customer_id")
        evaluated_at = as_of or self.store._clock()  # noqa: SLF001 - one store clock owns the read cutoff.
        require_timezone(evaluated_at, "as_of")
        out_of_scope = (
            allowed_only
            and normalized_customer_id
            in self._operational_out_of_scope_customer_ids(tenant, (normalized_customer_id,))
        )
        allowed_clauses = ["tenant_id = ?"]
        allowed_params: list[Any] = [tenant]
        self.store._append_chunk_filters(  # noqa: SLF001 - one canonical bot-safe boundary.
            allowed_clauses,
            allowed_params,
            tenant_id=tenant,
            customer_id=normalized_customer_id,
            opportunity_id=None,
            since=None,
            until=evaluated_at,
            allowed_for_bot=True,
        )
        if allowed_only:
            clauses = list(allowed_clauses)
            params = list(allowed_params)
        else:
            clauses = ["tenant_id = ?"]
            params = [tenant]
            clauses.append("customer_id = ?")
            params.append(normalized_customer_id)
        page_limit = bounded_limit(limit, default=50, max_limit=200)
        raw_limit = min(page_limit * 4, 500) if allowed_only else page_limit
        raw_items = [] if out_of_scope else self._records(
            "bot_context_chunks",
            " AND ".join(clauses),
            tuple(params),
            order_by="event_at DESC, created_at DESC, ordinal, chunk_id",
            limit=raw_limit,
        )
        visible_items = _dedupe_bot_context_items(raw_items)[:page_limit] if allowed_only else raw_items
        total_chunks = self._count("bot_context_chunks", "tenant_id = ? AND customer_id = ?", (tenant, normalized_customer_id))
        allowed_chunks = 0 if out_of_scope else self._count(
            "bot_context_chunks", " AND ".join(allowed_clauses), tuple(allowed_params)
        )
        review_required_chunks = self._count(
            "bot_context_chunks",
            "tenant_id = ? AND customer_id = ? AND requires_manager_review = 1",
            (tenant, normalized_customer_id),
        )
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "endpoint": "GET /customer/bot-context",
            "tenant_id": tenant,
            "customer_id": customer_id,
            "as_of": evaluated_at.isoformat(),
            "allowed_only": allowed_only,
            "out_of_scope": out_of_scope,
            "items": [project_bot_context(item, audience="bot" if allowed_only else "ui") for item in visible_items],
            "summary": {
                "out_of_scope": out_of_scope,
                "visible_chunks": len(visible_items),
                "total_chunks": total_chunks,
                "allowed_chunks": allowed_chunks,
                "review_required_chunks": review_required_chunks,
                "blocked_chunks": total_chunks - allowed_chunks,
            },
            "redaction": redaction_summary(bot_safe=allowed_only),
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def search(
        self,
        tenant_id: str,
        query: str,
        *,
        customer_id: Optional[str] = None,
        scopes: Sequence[str] = ("events", "bot_context", "signals"),
        allowed_for_bot: Optional[bool] = None,
        as_of: Optional[datetime] = None,
        limit: int = 25,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        evaluated_at = as_of or self.store._clock()  # noqa: SLF001 - one store clock owns the read cutoff.
        require_timezone(evaluated_at, "as_of")
        if allowed_for_bot is True:
            scopes = ("bot_context",)
        page_limit = bounded_limit(limit, default=25, max_limit=100)
        result = _operational_visible_page(
            fetch_page=lambda page_cursor: self.store.search_timeline(
                tenant,
                query,
                customer_id=customer_id,
                scopes=scopes,
                allowed_for_bot=allowed_for_bot,
                until=evaluated_at if allowed_for_bot is True else None,
                limit=max(page_limit, 50),
                cursor=page_cursor,
            ),
            customer_id_of=search_hit_customer_id,
            out_of_scope_for_customer_ids=lambda customer_ids: (
                self._operational_out_of_scope_customer_ids(tenant, customer_ids)
            ),
            limit=page_limit,
            cursor=cursor,
        )
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "endpoint": "GET /search",
            "tenant_id": tenant,
            "customer_id": customer_id,
            "as_of": evaluated_at.isoformat(),
            "result": {
                **result,
                "items": [
                    project_search_hit(item, bot_safe=allowed_for_bot is True)
                    for item in result["items"]
                ],
            },
            "redaction": redaction_summary(bot_safe=allowed_for_bot is True),
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def _operational_out_of_scope_customer_ids(
        self,
        tenant_id: str,
        customer_ids: Sequence[str],
    ) -> frozenset[str]:
        """Resolve scope only for the requested page/profile, never the whole tenant."""
        selected_customer_ids = tuple(
            sorted({str(value) for value in customer_ids if str(value)})
        )
        if not selected_customer_ids:
            return frozenset()
        tables = {
            str(row[0])
            for row in self.store._con.execute(  # noqa: SLF001 - read-only facade schema check.
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('family_members_v1','family_links_v1')"
            )
        }
        if tables != {"family_members_v1", "family_links_v1"}:
            return frozenset()
        families_by_customer: dict[str, set[str]] = {}
        for row in self.store._con.execute(  # noqa: SLF001 - one local read boundary.
            "SELECT family_id,customer_id FROM family_members_v1 WHERE tenant_id=? "
            "AND customer_id IN (SELECT value FROM json_each(?))",
            (tenant_id, json.dumps(selected_customer_ids, ensure_ascii=False)),
        ):
            families_by_customer.setdefault(str(row["customer_id"]), set()).add(str(row["family_id"]))
        selected_family_ids = tuple(
            sorted({family_id for values in families_by_customer.values() for family_id in values})
        )
        if not selected_family_ids:
            return frozenset()
        family_customer_ids = tuple(
            sorted({
                str(row["customer_id"])
                for row in self.store._con.execute(  # noqa: SLF001 - indexed family boundary.
                    "SELECT customer_id FROM family_members_v1 WHERE tenant_id=? "
                    "AND family_id IN (SELECT value FROM json_each(?))",
                    (tenant_id, json.dumps(selected_family_ids, ensure_ascii=False)),
                )
            })
        )
        if not family_customer_ids:
            return frozenset()
        states_by_family: dict[str, list[str]] = {}
        for row in self.store._con.execute(  # noqa: SLF001 - one local read boundary.
            "SELECT family_id,record_json,grades_json FROM family_links_v1 WHERE tenant_id=? "
            "AND customer_id IN (SELECT value FROM json_each(?))",
            (tenant_id, json.dumps(family_customer_ids, ensure_ascii=False)),
        ):
            family_id = str(row["family_id"])
            if family_id not in selected_family_ids:
                continue
            try:
                record = json.loads(str(row["record_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                record = {}
            if not isinstance(record, Mapping):
                record = {}
            if not isinstance(record.get("student_types"), list) and not isinstance(record.get("grades"), list):
                try:
                    grades = json.loads(str(row["grades_json"] or "[]"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    grades = None
                if isinstance(grades, list):
                    record = {**record, "grades": grades}
            states_by_family.setdefault(family_id, []).append(
                family_link_timeline_scope_state(record)
            )
        excluded: set[str] = set()
        for customer_id, family_ids in families_by_customer.items():
            states = [
                state
                for family_id in family_ids
                for state in states_by_family.get(family_id, ())
            ]
            if states and family_timeline_scope_decision(states) == FAMILY_LINK_SCOPE_OUT:
                excluded.add(customer_id)
        return frozenset(excluded)

    def list_conflicts(
        self,
        tenant_id: str,
        *,
        customer_id: Optional[str] = None,
        status: Optional[str] = None,
        conflict_type: Optional[str] = None,
        limit: int = 50,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        normalized_customer_id = require_text(customer_id, "customer_id") if customer_id else None
        normalized_status = normalize_key(status, "status") if status else None
        normalized_conflict_type = normalize_key(conflict_type, "conflict_type") if conflict_type else None
        if customer_id:
            page_limit = bounded_limit(limit, default=50, max_limit=200)
            complete_window = list(
                self.store.list_conflicts_by_customer(
                    tenant,
                    normalized_customer_id,
                    statuses=(normalized_status,) if normalized_status else (),
                    conflict_types=(normalized_conflict_type,) if normalized_conflict_type else (),
                    limit=page_limit + 1,
                )
            )
            items = complete_window[:page_limit]
        else:
            clauses = ["tenant_id = ?", "json_valid(record_json)"]
            params: list[Any] = [tenant]
            if normalized_status:
                clauses.append("status = ?")
                params.append(normalized_status)
            if normalized_conflict_type:
                clauses.append("conflict_type = ?")
                params.append(normalized_conflict_type)
            items = self._records(
                "timeline_conflicts",
                " AND ".join(clauses),
                tuple(params),
                order_by="created_at DESC, conflict_id",
                limit=bounded_limit(limit, default=50, max_limit=200),
            )
        if normalized_customer_id and len(complete_window) <= page_limit:
            summary = conflict_summary_from_complete_customer_items(items)
        else:
            summary = self._global_conflict_summary(
                tenant,
                customer_id=normalized_customer_id,
                status=normalized_status,
                conflict_type=normalized_conflict_type,
            )
        summary["recent_window"] = {
            "returned": len(items),
            "by_type": count_by(items, "conflict_type"),
            "by_status": count_by(items, "status"),
            "by_severity": count_by(items, "severity"),
        }
        return {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "endpoint": "GET /conflicts",
            "tenant_id": tenant,
            "customer_id": customer_id,
            "items": [project_conflict(item) for item in items],
            "summary": summary,
            "redaction": redaction_summary(bot_safe=False),
            "safety": customer_timeline_read_api_safety_contract(),
        }

    def _global_conflict_summary(
        self,
        tenant_id: str,
        *,
        customer_id: Optional[str],
        status: Optional[str],
        conflict_type: Optional[str],
    ) -> dict[str, Any]:
        clauses = ["conflict.tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if customer_id:
            refs = self.store.conflict_entity_refs_for_customer(tenant_id, customer_id)
            clauses.append(
                "json_valid(conflict.record_json) AND EXISTS ("
                "SELECT 1 FROM json_each(conflict.record_json, '$.entity_refs') AS ref "
                "WHERE _mango_canonical_identity_ref(CAST(ref.value AS TEXT)) "
                "IN (SELECT CAST(value AS TEXT) FROM json_each(?))"
                ")"
            )
            params.append(json.dumps(refs, ensure_ascii=False, separators=(",", ":")))
        if status:
            clauses.append("conflict.status = ?")
            params.append(status)
        if conflict_type:
            clauses.append("conflict.conflict_type = ?")
            params.append(conflict_type)
        where_sql = " AND ".join(clauses)
        grouped_rows = self.store._con.execute(  # noqa: SLF001 - one read-only aggregate boundary.
            f"SELECT conflict.conflict_type, conflict.severity, conflict.status, COUNT(*) AS count, "
            "SUM(CASE WHEN json_valid(conflict.record_json) THEN 0 ELSE 1 END) AS invalid_count "
            f"FROM timeline_conflicts AS conflict WHERE {where_sql} "
            "GROUP BY conflict.conflict_type, conflict.severity, conflict.status",
            tuple(params),
        ).fetchall()
        total = 0
        open_conflicts = 0
        malformed_conflict_payload_count = 0
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        open_by_severity: dict[str, int] = {}
        for item in grouped_rows:
            count = int(item["count"])
            conflict_type_key = str(item["conflict_type"] or "unknown")
            severity_key = str(item["severity"] or "unknown")
            status_key = str(item["status"] or "unknown")
            total += count
            malformed_conflict_payload_count += int(item["invalid_count"] or 0)
            by_type[conflict_type_key] = by_type.get(conflict_type_key, 0) + count
            by_status[status_key] = by_status.get(status_key, 0) + count
            by_severity[severity_key] = by_severity.get(severity_key, 0) + count
            if status_key in UNRESOLVED_CONFLICT_STATUSES:
                open_conflicts += count
                open_by_severity[severity_key] = open_by_severity.get(severity_key, 0) + count
        if customer_id:
            affected_customer_count = int(total > 0)
            open_affected_customer_count = int(open_conflicts > 0)
        else:
            ref_rows = self.store._con.execute(  # noqa: SLF001 - exact read-only refs.
                f"SELECT DISTINCT conflict.status, CAST(ref.value AS TEXT) AS entity_ref "
                "FROM timeline_conflicts AS conflict "
                "CROSS JOIN json_each(CASE WHEN json_valid(conflict.record_json) "
                "THEN conflict.record_json ELSE '{\"entity_refs\":[]}' END, '$.entity_refs') AS ref "
                f"WHERE {where_sql}",
                tuple(params),
            ).fetchall()
            refs_by_status: dict[str, list[str]] = {}
            for item in ref_rows:
                refs_by_status.setdefault(str(item["status"]), []).append(str(item["entity_ref"]))
            affected_by_status = self.store.conflict_affected_customer_ids_by_status(
                tenant_id,
                refs_by_status,
            )
            affected_customer_count = len(set().union(*affected_by_status.values())) if affected_by_status else 0
            open_affected_customer_count = len(
                set().union(
                    *(affected_by_status.get(status, frozenset()) for status in UNRESOLVED_CONFLICT_STATUSES)
                )
            )
        return {
            "total": total,
            "open_conflicts": open_conflicts,
            "affected_customer_count": affected_customer_count,
            "open_affected_customer_count": open_affected_customer_count,
            "malformed_conflict_payload_count": malformed_conflict_payload_count,
            "by_type": dict(sorted(by_type.items())),
            "by_status": dict(sorted(by_status.items())),
            "by_severity": dict(sorted(by_severity.items())),
            "open_by_severity": dict(sorted(open_by_severity.items())),
        }

    def _tenant_counts(self, tenant_id: str) -> Mapping[str, int]:
        tables = (
            "customer_identities",
            "identity_links",
            "customer_opportunities",
            "timeline_events",
            "event_artifacts",
            "derived_signals",
            "bot_context_chunks",
            "ingestion_runs",
            "timeline_conflicts",
            "audit_log",
        )
        return {table: self._count(table, "tenant_id = ?", (tenant_id,)) for table in tables}

    def _records(
        self,
        table: str,
        where_sql: str,
        params: Sequence[Any],
        *,
        order_by: str,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        if table not in READ_API_TABLES:
            raise ValueError(f"unsupported read API table: {table}")
        effective_where = self._read_where(table, where_sql)
        json_guard = " AND json_valid(record_json)=1" if table == "bot_context_chunks" else ""
        selected_columns = (
            "record_json,chunk_id,event_id,source_system,chunk_type"
            if table == "bot_context_chunks"
            else "record_json"
        )
        rows = self.store._con.execute(  # noqa: SLF001 - read facade wraps store internals for callers.
            f"SELECT {selected_columns} FROM {table} WHERE ({effective_where}){json_guard} "
            f"ORDER BY {order_by} LIMIT ?",
            (*params, bounded_limit(limit, default=50, max_limit=500)),
        ).fetchall()
        if table == "bot_context_chunks":
            result: list[Mapping[str, Any]] = []
            for row in rows:
                payload = json.loads(row["record_json"])
                if not isinstance(payload, Mapping):
                    continue
                trusted = dict(payload)
                for key in ("chunk_id", "event_id", "source_system", "chunk_type"):
                    trusted[key] = row[key]
                result.append(trusted)
            return result
        return [json.loads(row["record_json"]) for row in rows]

    def _count(self, table: str, where_sql: str, params: Sequence[Any]) -> int:
        if table not in READ_API_TABLES:
            raise ValueError(f"unsupported read API table: {table}")
        effective_where = self._read_where(table, where_sql)
        row = self.store._con.execute(  # noqa: SLF001 - read facade wraps store internals for callers.
            f"SELECT COUNT(*) AS value FROM {table} WHERE {effective_where}",
            tuple(params),
        ).fetchone()
        return int(row["value"] if row else 0)

    def _read_where(self, table: str, where_sql: str) -> str:
        if table in {"timeline_events", "bot_context_chunks"} and self.store._table_has_column(table, "superseded_by"):  # noqa: SLF001
            return f"({where_sql}) AND superseded_by IS NULL"
        return where_sql


READ_API_TABLES = {
    "customer_identities",
    "identity_links",
    "customer_opportunities",
    "timeline_events",
    "event_artifacts",
    "derived_signals",
    "bot_context_chunks",
    "ingestion_runs",
    "timeline_conflicts",
    "audit_log",
}


def route_customer_timeline_request(api: CustomerTimelineReadApi, method: str, raw_path: str) -> tuple[int, Mapping[str, Any]]:
    parsed = urlparse(raw_path)
    route = parsed.path.rstrip("/") or "/"
    query = parse_qs(parsed.query)
    if method.upper() != "GET":
        return 405, {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "route": route,
            "error": "method_not_allowed_read_only_api",
            "read_only": True,
            "safety": customer_timeline_read_api_safety_contract(),
        }
    try:
        if route == "/health":
            return 200, api.health()
        if route == "/summary":
            return 200, api.summary(required_query(query, "tenant_id"), recent_limit=query_int(query, "limit", 10))
        if route == "/customers":
            return 200, api.list_customers(
                required_query(query, "tenant_id"),
                q=query_scalar(query, "q"),
                identity_status=query_scalar(query, "identity_status"),
                updated_since=parse_datetime(query_scalar(query, "updated_since")),
                limit=query_int(query, "limit", 50),
                cursor=query_scalar(query, "cursor"),
            )
        if route == "/customer":
            return 200, api.customer_profile(
                required_query(query, "tenant_id"),
                required_query(query, "customer_id"),
                event_limit=query_int(query, "event_limit", 25),
                bot_context_limit=query_int(query, "bot_context_limit", 25),
                as_of=parse_datetime(query_scalar(query, "as_of")),
            )
        if route == "/customer/timeline":
            return 200, api.customer_timeline(
                required_query(query, "tenant_id"),
                required_query(query, "customer_id"),
                limit=query_int(query, "limit", 50),
                cursor=query_scalar(query, "cursor"),
                sort=query_scalar(query, "sort", "desc") or "desc",
                event_types=query_list(query, "event_type"),
                source_systems=query_list(query, "source_system"),
                since=parse_datetime(query_scalar(query, "since")),
                until=parse_datetime(query_scalar(query, "until")),
                include_artifacts=query_bool(query, "include_artifacts", True),
                include_signals=query_bool(query, "include_signals", True),
            )
        if route == "/customer/bot-context":
            return 200, api.bot_context(
                required_query(query, "tenant_id"),
                required_query(query, "customer_id"),
                allowed_only=query_bool(query, "allowed_only", True),
                as_of=parse_datetime(query_scalar(query, "as_of")),
                limit=query_int(query, "limit", 50),
            )
        if route == "/search":
            return 200, api.search(
                required_query(query, "tenant_id"),
                required_query(query, "q"),
                customer_id=query_scalar(query, "customer_id"),
                scopes=tuple(query_list(query, "scope")) or ("events", "bot_context", "signals"),
                allowed_for_bot=query_bool_or_none(query, "allowed_for_bot"),
                as_of=parse_datetime(query_scalar(query, "as_of")),
                limit=query_int(query, "limit", 25),
            )
        if route == "/conflicts":
            return 200, api.list_conflicts(
                required_query(query, "tenant_id"),
                customer_id=query_scalar(query, "customer_id"),
                status=query_scalar(query, "status"),
                conflict_type=query_scalar(query, "conflict_type"),
                limit=query_int(query, "limit", 50),
            )
    except ValueError as exc:
        return 400, {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "route": route,
            "error": "bad_request",
            "detail": str(exc),
            "safety": customer_timeline_read_api_safety_contract(),
        }
    return 404, {
        "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
        "route": route,
        "error": "route_not_found",
        "implemented_routes": list(READ_API_ROUTES),
        "safety": customer_timeline_read_api_safety_contract(),
    }


def build_customer_timeline_read_report(
    *,
    config: CustomerTimelineReadApiConfig,
    tenant_id: str,
    customer_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 25,
    out_path: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    out = guard_customer_timeline_output_path(out_path, config.allowed_root) if out_path else None
    report_as_of = generated_at or datetime.now(timezone.utc)
    with CustomerTimelineReadApi.open(config) as api:
        report: dict[str, Any] = {
            "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
            "report_kind": "customer_timeline_read_report",
            "generated_at": report_as_of.isoformat(),
            "health": api.health(),
            "summary": api.summary(tenant_id, recent_limit=limit),
            "customer_profile": (
                api.customer_profile(tenant_id, customer_id, event_limit=limit, as_of=report_as_of)
                if customer_id
                else None
            ),
            "search": (
                api.search(tenant_id, query, customer_id=customer_id, limit=limit, as_of=report_as_of)
                if query
                else None
            ),
            "safety": customer_timeline_read_api_safety_contract(),
        }
    report["validation_ok"] = bool(report["health"].get("validation_ok")) and bool(report["summary"]["summary"].get("validation_ok"))
    if out:
        if out == config.timeline_db:
            raise ValueError("read report output path must not overwrite timeline DB")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = CustomerTimelineReadApiConfig(timeline_db=Path(args.timeline_db), allowed_root=Path(args.allowed_root))
        report = build_customer_timeline_read_report(
            config=config,
            tenant_id=args.tenant_id,
            customer_id=args.customer_id,
            query=args.query,
            limit=args.limit,
            out_path=Path(args.out) if args.out else None,
        )
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if not args.out:
            print(text)
        return 0 if report["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - compact CLI-facing error.
        print(f"customer timeline read report failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only report from customer_timeline.sqlite.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--customer-id")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out")
    return parser


def customer_timeline_read_api_safety_contract() -> Mapping[str, Any]:
    sqlite_flags = {
        key: value
        for key, value in customer_timeline_sqlite_safety_contract().items()
        if "raw_payload" not in key
    }
    flags = {
        **sqlite_flags,
        "schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
        "read_only_api": True,
        "read_only_db_connection": True,
        "write_product_timeline_db": False,
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "live_send": False,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "stable_runtime_writes": False,
        "network_calls": False,
        "subprocess_calls": False,
    }
    return {
        **flags,
        "ok": all(flags.get(action) is False for action in blocked_live_actions()),
        "blocked_live_actions": list(blocked_live_actions()),
    }


def bounded_limit(value: Optional[int], *, default: int, max_limit: int) -> int:
    if value is None:
        return default
    limit = int(value)
    if limit < 1:
        raise ValueError("limit must be positive")
    return min(limit, max_limit)


def count_by(items: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def conflict_summary_from_complete_customer_items(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact customer summary after a limit+1 query proved completeness."""
    by_type = dict(sorted(count_by(items, "conflict_type").items()))
    by_status = dict(sorted(count_by(items, "status").items()))
    by_severity = dict(sorted(count_by(items, "severity").items()))
    open_items = [item for item in items if str(item.get("status") or "unknown") in UNRESOLVED_CONFLICT_STATUSES]
    open_by_severity = dict(sorted(count_by(open_items, "severity").items()))
    total = len(items)
    open_conflicts = len(open_items)
    return {
        "total": total,
        "open_conflicts": open_conflicts,
        "affected_customer_count": int(total > 0),
        "open_affected_customer_count": int(open_conflicts > 0),
        "malformed_conflict_payload_count": 0,
        "by_type": by_type,
        "by_status": by_status,
        "by_severity": by_severity,
        "open_by_severity": open_by_severity,
    }


def project_customer(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "customer_id": item.get("customer_id"),
        "tenant_id": item.get("tenant_id"),
        "identity_status": item.get("identity_status"),
        "display_name": item.get("display_name"),
        "primary_phone": mask_phone(item.get("primary_phone")),
        "primary_email": mask_email(item.get("primary_email")),
        "first_seen_at": item.get("first_seen_at"),
        "last_seen_at": item.get("last_seen_at"),
        "touch_count": item.get("touch_count"),
        "summary": safe_mapping(item.get("summary")),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def project_identity_link(item: Mapping[str, Any], *, audience: str) -> Mapping[str, Any]:
    value = item.get("link_value")
    link_type = str(item.get("link_type") or "")
    projected_value: Optional[str]
    if audience == "bot":
        projected_value = None
    elif link_type == "phone" or "phone" in link_type:
        projected_value = mask_phone(value)
    elif link_type == "email":
        projected_value = mask_email(value)
    else:
        projected_value = mask_identifier(value)
    return {
        "link_id": item.get("link_id"),
        "customer_id": item.get("customer_id"),
        "link_type": link_type,
        "link_value": projected_value,
        "source_system": item.get("source_system"),
        "match_class": item.get("match_class"),
        "confidence": item.get("confidence"),
        "first_seen_at": item.get("first_seen_at"),
        "last_seen_at": item.get("last_seen_at"),
    }


def project_opportunity(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "opportunity_id": item.get("opportunity_id"),
        "customer_id": item.get("customer_id"),
        "opportunity_type": item.get("opportunity_type"),
        "source_system": item.get("source_system"),
        "title": item.get("title"),
        "status": item.get("status"),
        "opened_at": item.get("opened_at"),
        "closed_at": item.get("closed_at"),
        "confidence": item.get("confidence"),
    }


def project_manager_projection(
    *,
    customer: Mapping[str, Any],
    links: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    next_step_resolution: Mapping[str, Any] | None = None,
    manager_action: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    phone_values = sorted(
        {
            str(item.get("link_value"))
            for item in links
            if item.get("link_type") in {"phone", "mango_client_phone"} and item.get("link_value") not in (None, "")
        }
    )
    if customer.get("primary_phone") not in (None, ""):
        phone_values.append(str(customer.get("primary_phone")))
    phone_values = sorted(set(phone_values))
    amo_contact_ids = sorted(
        {
            str(item.get("link_value"))
            for item in links
            if item.get("link_type") == "amo_contact_id" and item.get("link_value") not in (None, "")
        }
    )
    amo_lead_ids = {
        str(item.get("link_value"))
        for item in links
        if item.get("link_type") == "amo_lead_id" and item.get("link_value") not in (None, "")
    }
    amo_lead_ids.update(
        str(item.get("source_id"))
        for item in opportunities
        if item.get("source_system") == "amocrm_snapshot" and item.get("source_id") not in (None, "")
    )
    for event in events:
        record = event.get("record") if isinstance(event.get("record"), Mapping) else {}
        contact_id = first_text(record, ("canonical_amocrm_contact_id",))
        lead_id = first_text(record, ("canonical_amocrm_lead_id",))
        if contact_id:
            amo_contact_ids.append(contact_id)
        if lead_id:
            amo_lead_ids.add(lead_id)
    return {
        "schema_version": "customer_profile_manager_projection_v2",
        "audience": "manager_internal",
        "primary_phone": str(customer.get("primary_phone") or ""),
        "primary_email": str(customer.get("primary_email") or ""),
        "phone_numbers": phone_values,
        "amo_contact_ids": sorted(set(amo_contact_ids)),
        "amo_lead_ids": sorted(amo_lead_ids),
        "identity_links": [
            project_identity_link_manager(item)
            for item in links
            if item.get("link_type") in {"amo_contact_id", "amo_lead_id"}
        ],
        "opportunities": [project_opportunity_manager(item) for item in opportunities if item.get("source_system") == "amocrm_snapshot"],
        "next_step_resolution": dict(next_step_resolution or {}),
        "manager_action": dict(manager_action or {}),
    }


def project_identity_link_manager(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "link_id": item.get("link_id"),
        "customer_id": item.get("customer_id"),
        "link_type": item.get("link_type"),
        "link_value": item.get("link_value"),
        "source_system": item.get("source_system"),
        "match_class": item.get("match_class"),
        "confidence": item.get("confidence"),
        "first_seen_at": item.get("first_seen_at"),
        "last_seen_at": item.get("last_seen_at"),
    }


def project_opportunity_manager(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "opportunity_id": item.get("opportunity_id"),
        "customer_id": item.get("customer_id"),
        "opportunity_type": item.get("opportunity_type"),
        "source_system": item.get("source_system"),
        "source_id": item.get("source_id"),
        "title": item.get("title"),
        "status": item.get("status"),
        "opened_at": item.get("opened_at"),
        "closed_at": item.get("closed_at"),
        "confidence": item.get("confidence"),
    }


def project_event(
    item: Mapping[str, Any],
    *,
    include_artifacts: bool = True,
    include_signals: bool = True,
) -> Mapping[str, Any]:
    projected: dict[str, Any] = {
        "event_id": item.get("event_id"),
        "tenant_id": item.get("tenant_id"),
        "customer_id": item.get("customer_id"),
        "opportunity_id": item.get("opportunity_id"),
        "event_type": item.get("event_type"),
        "event_at": item.get("event_at"),
        "source_system": item.get("source_system"),
        "direction": item.get("direction"),
        "participants": [project_participant(participant) for participant in item.get("participants") or ()],
        "actor_name": item.get("actor_name"),
        "actor_ref": mask_identifier(item.get("actor_ref")),
        "subject": item.get("subject"),
        "text_preview": item.get("text_preview"),
        "summary": item.get("summary"),
        "stage_before": item.get("stage_before"),
        "stage_after": item.get("stage_after"),
        "importance": item.get("importance"),
        "match_status": item.get("match_status"),
        "confidence": item.get("confidence"),
        "allowed_for_bot": False,
        "requires_manager_review": True,
        "created_at": item.get("created_at"),
    }
    if include_artifacts:
        projected["artifacts"] = [project_artifact(artifact) for artifact in item.get("artifacts") or ()]
    if include_signals:
        projected["signals"] = [project_signal(signal) for signal in item.get("signals") or ()]
    record = item.get("record") if isinstance(item.get("record"), Mapping) else {}
    call_analysis = record.get("call_analysis") if isinstance(record.get("call_analysis"), Mapping) else {}
    if call_analysis:
        projected["call_analysis"] = safe_nested_value(call_analysis)
        projected["call_type"] = record.get("call_type") or call_analysis.get("call_type")
        projected["call_history_eligible"] = bool(record.get("call_history_eligible"))
    return projected


def project_participant(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "role": item.get("role"),
        "ref": mask_identifier(item.get("ref")),
        "name": item.get("name"),
        "channel": item.get("channel"),
    }


def project_artifact(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "artifact_id": item.get("artifact_id"),
        "tenant_id": item.get("tenant_id"),
        "event_id": item.get("event_id"),
        "artifact_type": item.get("artifact_type"),
        "sha256": item.get("sha256"),
        "size_bytes": item.get("size_bytes"),
        "mime_type": item.get("mime_type"),
        "source_system": item.get("source_system"),
        "extraction_status": item.get("extraction_status"),
        "created_at": item.get("created_at"),
        "has_path": bool(item.get("path")),
    }


def project_signal(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "signal_id": item.get("signal_id"),
        "tenant_id": item.get("tenant_id"),
        "customer_id": item.get("customer_id"),
        "opportunity_id": item.get("opportunity_id"),
        "event_id": item.get("event_id"),
        "signal_type": item.get("signal_type"),
        "severity": item.get("severity"),
        "status": item.get("status") or "active",
        "expires_at": item.get("expires_at"),
        "confidence": item.get("confidence"),
        "evidence_text": item.get("evidence_text"),
        "recommended_action": item.get("recommended_action"),
        "allowed_for_bot": False,
        "requires_manager_review": bool(item.get("requires_manager_review")),
        "created_at": item.get("created_at"),
    }


def project_bot_context(item: Mapping[str, Any], *, audience: str) -> Mapping[str, Any]:
    neutral_purchase = audience == "bot" and (
        item.get("source_system") == PURCHASE_HISTORY_SOURCE_SYSTEM
        and item.get("chunk_type") == PURCHASE_HISTORY_CHUNK_TYPE
    )
    payload = {
        "chunk_id": item.get("chunk_id"),
        "customer_id": item.get("customer_id") if audience != "bot" else None,
        "opportunity_id": item.get("opportunity_id") if audience != "bot" else None,
        "event_id": item.get("event_id") if audience != "bot" else None,
        "source_system": item.get("source_system"),
        "chunk_type": item.get("chunk_type"),
        "text": PURCHASE_HISTORY_BOT_TEXT if neutral_purchase else item.get("text"),
        "summary": PURCHASE_HISTORY_BOT_TEXT if neutral_purchase else item.get("summary"),
        "event_at": None if neutral_purchase else item.get("event_at"),
        "freshness_score": item.get("freshness_score"),
        "relevance_tags": list(item.get("relevance_tags") or ()),
        "allowed_for_bot": bool(item.get("allowed_for_bot")),
        "requires_manager_review": bool(item.get("requires_manager_review")),
    }
    next_step_status = _project_next_step_status(item)
    if next_step_status:
        payload["next_step_status"] = next_step_status
    return payload


def _dedupe_bot_context_items(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in items:
        identity = _bot_context_source_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result


def _bot_context_source_identity(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    source_identity = ""
    for key in ("message_sha256", "source_message_sha256", "source_id", "event_id"):
        value = str(item.get(key) or metadata.get(key) or "").strip()
        if value:
            source_identity = f"{key}:{value}"
            break
    if not source_identity:
        source_ref = str(item.get("source_ref") or "").strip()
        source_identity = f"source_ref:{source_ref}" if source_ref else f"chunk_id:{item.get('chunk_id') or ''}"
    return (
        str(item.get("customer_id") or ""),
        str(item.get("source_system") or ""),
        str(item.get("chunk_type") or ""),
        source_identity,
    )


def _project_next_step_status(item: Mapping[str, Any]) -> str:
    metadata = item.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    next_step = metadata.get("next_step")
    if not isinstance(next_step, Mapping):
        return ""
    status = str(next_step.get("status") or "").strip().casefold()
    return status if status in {"active", "needs_manager_review", "empty"} else ""


def project_conflict(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "conflict_id": item.get("conflict_id"),
        "tenant_id": item.get("tenant_id"),
        "conflict_type": item.get("conflict_type"),
        "severity": item.get("severity"),
        "status": item.get("status"),
        "created_at": item.get("created_at"),
        "resolved_at": item.get("resolved_at"),
        "entity_refs": [mask_identifier(ref) for ref in item.get("entity_refs") or ()],
        "summary": item.get("summary"),
    }


def project_customer_id_mapping(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "mapping_id": item.get("mapping_id"),
        "tenant_id": item.get("tenant_id"),
        "old_customer_id": item.get("old_customer_id"),
        "new_customer_id": item.get("new_customer_id"),
        "mapping_kind": item.get("mapping_kind"),
        "resolution_status": item.get("resolution_status"),
        "reason": item.get("reason"),
        "source_refs": list(item.get("source_refs") or ()),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def project_search_hit(item: Mapping[str, Any], *, bot_safe: bool = False) -> Mapping[str, Any]:
    scope = item.get("scope")
    record = item.get("record") or {}
    if scope == "event":
        projected = project_event(record, include_artifacts=False, include_signals=False)
    elif scope == "bot_context":
        projected = project_bot_context(record, audience="bot")
    elif scope == "signal":
        projected = project_signal(record)
    else:
        projected = {}
    return {
        "scope": scope,
        "id": item.get("id"),
        "event_at": (
            None
            if bot_safe
            and scope == "bot_context"
            and record.get("source_system") == PURCHASE_HISTORY_SOURCE_SYSTEM
            and record.get("chunk_type") == PURCHASE_HISTORY_CHUNK_TYPE
            else item.get("event_at")
        ),
        "record": projected,
        "highlight": None if bot_safe else item.get("highlight"),
    }


def _operational_visible_page(
    *,
    fetch_page: Callable[[Optional[str]], Mapping[str, Any]],
    customer_id_of: Callable[[Mapping[str, Any]], str],
    out_of_scope_for_customer_ids: Callable[[Sequence[str]], frozenset[str]],
    limit: int,
    cursor: Optional[str],
) -> Mapping[str, Any]:
    """Apply the operational scope before the public page limit and cursor boundary."""
    visible: list[Mapping[str, Any]] = []
    page_cursor = cursor
    while True:
        page = fetch_page(page_cursor)
        raw_items = list(page.get("items") or ())
        out_of_scope = out_of_scope_for_customer_ids(
            tuple(customer_id_of(item) for item in raw_items)
        )
        raw_offset = int(page_cursor or "0")  # fetch_page validates the cursor first.
        for index, item in enumerate(raw_items):
            if customer_id_of(item) in out_of_scope:
                continue
            visible.append(item)
            if len(visible) == limit:
                has_more = index + 1 < len(raw_items) or page.get("next_cursor") is not None
                return {
                    **page,
                    "items": visible,
                    "next_cursor": str(raw_offset + index + 1) if has_more else None,
                }
        next_cursor = page.get("next_cursor")
        if next_cursor is None:
            return {**page, "items": visible, "next_cursor": None}
        page_cursor = str(next_cursor)


def search_hit_customer_id(item: Mapping[str, Any]) -> str:
    record = item.get("record") if isinstance(item.get("record"), Mapping) else {}
    return str(record.get("customer_id") or "")


def first_text(item: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def safe_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value if str(key).casefold() not in FORBIDDEN_OUTPUT_KEYS}
    return {}


def safe_nested_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe_nested_value(value[key]) for key in value if str(key).casefold() not in FORBIDDEN_OUTPUT_KEYS}
    if isinstance(value, (list, tuple)):
        return [safe_nested_value(item) for item in value]
    return value


def mask_phone(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) <= 4:
        return "***"
    return f"+***{digits[-4:]}"


def mask_email(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if "@" not in text:
        return mask_identifier(text)
    name, domain = text.split("@", 1)
    first = name[:1] or "*"
    return f"{first}***@{domain}"


def mask_identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if len(text) <= 6:
        return "***"
    return f"{text[:3]}***{text[-3:]}"


def redaction_summary(*, bot_safe: bool) -> Mapping[str, Any]:
    return {
        "source_payload_removed": True,
        "artifact_paths_removed": True,
        "storage_rows_removed": True,
        "hashes_internal": True,
        "bot_safe": bot_safe,
    }


FORBIDDEN_OUTPUT_KEYS = {
    "raw_payload",
    "provider_raw_payload",
    "webhook_payload",
    "telegram_update",
    "telegram_raw_update",
    "crm_dialog_payload",
    "email_raw_body",
    "raw_body",
    "raw_file",
    "file_bytes",
    "attachment_bytes",
    "audio_bytes",
    "record_json",
    "record_hash",
    "path",
    "source_path",
    "raw_eml_path",
    "audio_path",
    "transcript_path",
}


def query_scalar(query: Mapping[str, Sequence[str]], key: str, default: Optional[str] = None) -> Optional[str]:
    value = query.get(key)
    if not value:
        return default
    text = str(value[0]).strip()
    return text or default


def required_query(query: Mapping[str, Sequence[str]], key: str) -> str:
    return require_text(query_scalar(query, key), key)


def query_int(query: Mapping[str, Sequence[str]], key: str, default: int) -> int:
    value = query_scalar(query, key)
    return default if value is None else int(value)


def query_bool(query: Mapping[str, Sequence[str]], key: str, default: bool) -> bool:
    value = query_scalar(query, key)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "y"}


def query_bool_or_none(query: Mapping[str, Sequence[str]], key: str) -> Optional[bool]:
    value = query_scalar(query, key)
    if value is None:
        return None
    return value.casefold() in {"1", "true", "yes", "y"}


def query_list(query: Mapping[str, Sequence[str]], key: str) -> tuple[str, ...]:
    values = query.get(key) or ()
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return tuple(result)


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    require_timezone(parsed, "datetime")
    return parsed.astimezone(timezone.utc)


__all__ = [
    "CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION",
    "READ_API_ROUTES",
    "CustomerTimelineReadApi",
    "CustomerTimelineReadApiConfig",
    "build_customer_timeline_read_report",
    "build_parser",
    "customer_timeline_read_api_safety_contract",
    "main",
    "route_customer_timeline_request",
]
