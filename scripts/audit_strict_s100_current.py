#!/usr/bin/env python3
"""Run the newest frozen strict S100 once through the current read contract."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


HARNESS_SHA256 = "446b61cd968d8dd6d7ac47f02978d9751cf428e59af718992d654dc604139e57"
POOL_SHA256 = "988c8a05c41eb8f9b9f2080d16424c8c3cd7932b8218b45916c9e3550ab3590a"
SCHEMA_VERSION = "customer_timeline_strict_s100_current_tz_v1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_strict_s100_harness", path)
    if spec is None or spec.loader is None:
        raise SystemExit("STOP: frozen S100 harness cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_compatibility(harness: ModuleType, repo: Path) -> Callable[[], None]:
    source_root = repo / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from mango_mvp.customer_timeline import freshness, manager_dossier as manager
    from mango_mvp.customer_timeline.next_step_resolver import (
        is_meaningful_manager_action,
        load_manager_action_read_snapshot,
    )
    from mango_mvp.customer_timeline.source_policy import (
        BOT_SAFE_SUMMARY_CHUNK_TYPE,
        BOT_SAFE_SUMMARY_SOURCE_SYSTEM,
        PURCHASE_HISTORY_CHUNK_TYPE,
        PURCHASE_HISTORY_SOURCE_SYSTEM,
    )
    from mango_mvp.customer_timeline.store import (
        _canonical_identity_conflict_ref,
        register_timeline_record_integrity_sql_functions,
    )
    from mango_mvp.customer_timeline.temporal import register_temporal_sql_functions

    original_build = manager.build_customer_dossier
    original_expected_sources = harness._expected_dossier_sources
    missing = object()
    originals = {
        "family_scope": getattr(manager, "_family_scope", missing),
        "meaningful_next_step": getattr(manager, "_meaningful_next_step", missing),
        "load_snapshot": getattr(manager, "load_manager_dossier_conflict_snapshot", missing),
        "active_deals_field": harness._active_deals_field,
        "open_read_api": harness._open_immutable_read_api,
        "future_pairs": getattr(freshness, "FUTURE_EVENT_AT_ALLOWED_PAIRS", missing),
        "eventless_pairs": getattr(freshness, "EVENTLESS_BOT_CONTEXT_ALLOWED_PAIRS", missing),
        "event_visible": getattr(freshness, "event_visible_at_sql", missing),
    }
    loaded_snapshot: Any = None

    def family_scope(
        con: sqlite3.Connection, *, tenant_id: str, customer_id: str,
        as_of: datetime, blocked_customer_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], bool]:
        if loaded_snapshot is None:
            raise RuntimeError("manager dossier snapshot must be loaded before family scope")
        members = loaded_snapshot.family_customer_ids_by_customer.get(customer_id) or (customer_id,)
        return members, any(member in blocked_customer_ids for member in members)

    def load_snapshot(
        con: sqlite3.Connection, *, tenant_id: str,
        customer_ids: Sequence[str], as_of: datetime,
    ) -> Any:
        nonlocal loaded_snapshot
        snapshot = load_manager_action_read_snapshot(
            con, tenant_id=tenant_id, customer_ids=customer_ids, as_of=as_of,
        )
        loaded_snapshot = snapshot
        return SimpleNamespace(
            blocked_customer_ids=snapshot.conflict_customer_ids,
            read_snapshot=snapshot,
        )

    def build_dossier(*args: Any, conflict_snapshot: Any = None, **kwargs: Any) -> Any:
        kwargs["manager_action_read_snapshot"] = (
            conflict_snapshot.read_snapshot if conflict_snapshot is not None else None
        )
        return original_build(*args, **kwargs)

    def event_visible_at_sql(cutoff: str, alias: str) -> tuple[str, tuple[str, ...]]:
        return (
            f"julianday({alias}.event_at)<=julianday(?) "
            f"AND {alias}.match_status IN ('strong_unique','manual')",
            (cutoff,),
        )

    def expected_sources(
        con: sqlite3.Connection, **kwargs: Any,
    ) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
        result = dict(original_expected_sources(con, **kwargs))
        tenant_id = kwargs["tenant_id"]
        cutoff = kwargs["cutoff"]
        family_ids = tuple(kwargs["family_customer_ids"] or (kwargs["customer_id"],))
        placeholders = ",".join("?" for _ in family_ids)
        record_source = kwargs["record_source"]
        visible_sql, visible_params = kwargs["event_visible_at_sql"](cutoff.isoformat(), "event")
        rows = harness._rows(
            con,
            f"SELECT event.event_id,event.event_at,event.event_type,event.source_system,event.subject,"
            f"event.summary,event.text_preview,event.record_hash FROM timeline_events AS event "
            f"WHERE event.tenant_id=? AND event.customer_id IN ({placeholders}) AND {visible_sql} "
            "AND event.match_status IN ('strong_unique','manual') "
            "AND COALESCE(event.superseded_by,'')='' "
            "AND COALESCE(json_extract(event.record_json,'$.metadata.pending_attribution'),0) NOT IN (1,'true') "
            "AND event.event_type!='tallanto_attendance' "
            "AND event.source_system!='tallanto_attendance_api' "
            "ORDER BY event.event_at DESC,event.event_id DESC LIMIT 12",
            (tenant_id, *family_ids, *visible_params),
        )
        result["chronology"] = {}
        for row in rows:
            if not kwargs["event_summary_for_manager"](row):
                continue
            source = record_source("timeline_events", row["event_id"])
            result["chronology"][source] = harness._source_ref(
                source, row, table="timeline_events", at_key="event_at",
            )
        return result

    def active_deals_field(
        con: sqlite3.Connection, tenant_id: str, customer_id: str,
        dossier_rows: Sequence[Any], is_active_deal_at: Any, cutoff: datetime,
    ) -> Mapping[str, Any]:
        if loaded_snapshot is None:
            raise RuntimeError("manager dossier snapshot must be loaded before active deals")
        family_ids = loaded_snapshot.family_customer_ids_by_customer.get(customer_id) or (customer_id,)
        placeholders = ",".join("?" for _ in family_ids)
        opportunities = harness._rows(
            con,
            f"SELECT opportunity_id,customer_id,opportunity_type,source_system,source_id,status,"
            f"opened_at,closed_at,record_hash,record_json FROM customer_opportunities "
            f"WHERE tenant_id=? AND customer_id IN ({placeholders}) "
            "AND (opened_at IS NULL OR julianday(opened_at)<=julianday(?))",
            (tenant_id, *family_ids, cutoff.isoformat()),
        )
        lead_ids = sorted({str(row.get("source_id") or "") for row in opportunities if row.get("source_id")})
        owners: dict[str, set[str]] = {}
        if lead_ids:
            for row in con.execute(
                "SELECT link_value,customer_id FROM identity_links WHERE tenant_id=? "
                "AND link_type='amo_lead_id' AND match_class IN ('strong_unique','manual') "
                "AND (first_seen_at IS NULL OR julianday(first_seen_at)<=julianday(?)) "
                "AND link_value IN (SELECT value FROM json_each(?))",
                (tenant_id, cutoff.isoformat(), json.dumps(lead_ids, ensure_ascii=False)),
            ):
                owners.setdefault(str(row["link_value"]), set()).add(str(row["customer_id"]))
        active = [] if customer_id in loaded_snapshot.conflict_customer_ids else [
            row for row in opportunities
            if owners.get(str(row.get("source_id") or "")) == {str(row["customer_id"])}
            and is_active_deal_at(row, as_of=cutoff)
        ]
        expected = {
            f"customer_opportunities:{row['opportunity_id']}": harness._source_ref(
                f"customer_opportunities:{row['opportunity_id']}", row,
                table="customer_opportunities", at_key="opened_at",
            )
            for row in active
        }
        displayed = harness._dossier_row_field(
            dossier_rows, expected_sources=expected,
            reason_present="dedicated_family_active_deals_projection",
            reason_absent="canonical_family_active_deal_predicate_zero",
        )
        projection_matches = sorted(str(row.source) for row in dossier_rows) == sorted(expected)
        data_present = harness._state(
            "known" if active else "absent",
            "canonical_family_active_deal_predicate" if active else "canonical_family_active_deal_predicate_zero",
            tuple(expected.values()), count=len(active),
        )
        return {
            **displayed,
            "data_present": data_present,
            "dossier_displayed": displayed,
            "projection_matches_storage": projection_matches,
            "current_contract": "family-scoped active AMO deals with one exact lead owner",
        }

    def open_immutable_read_api(store_type: Any, api_type: Any, db: Path) -> Any:
        class ImmutableCurrentStore(store_type):
            def _connect(self) -> sqlite3.Connection:
                con = sqlite3.connect(self.db_path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=15)
                con.row_factory = sqlite3.Row
                con.execute("PRAGMA query_only=ON")
                con.execute("PRAGMA foreign_keys=ON")
                register_temporal_sql_functions(con)
                con.create_function(
                    "_mango_canonical_identity_ref", 1, _canonical_identity_conflict_ref,
                    deterministic=True,
                )
                register_timeline_record_integrity_sql_functions(con)
                return con

        return api_type(ImmutableCurrentStore(db, allowed_root=db.parent, read_only=True))

    manager._family_scope = family_scope
    manager._meaningful_next_step = is_meaningful_manager_action
    manager.load_manager_dossier_conflict_snapshot = load_snapshot
    manager.build_customer_dossier = build_dossier
    harness._active_deals_field = active_deals_field
    harness._expected_dossier_sources = expected_sources
    harness._open_immutable_read_api = open_immutable_read_api
    freshness.FUTURE_EVENT_AT_ALLOWED_PAIRS = frozenset()
    freshness.EVENTLESS_BOT_CONTEXT_ALLOWED_PAIRS = frozenset({
        (BOT_SAFE_SUMMARY_SOURCE_SYSTEM, BOT_SAFE_SUMMARY_CHUNK_TYPE),
        (PURCHASE_HISTORY_SOURCE_SYSTEM, PURCHASE_HISTORY_CHUNK_TYPE),
    })
    freshness.event_visible_at_sql = event_visible_at_sql

    def restore() -> None:
        for name, original in (
            ("_family_scope", originals["family_scope"]),
            ("_meaningful_next_step", originals["meaningful_next_step"]),
            ("load_manager_dossier_conflict_snapshot", originals["load_snapshot"]),
        ):
            if original is missing:
                delattr(manager, name)
            else:
                setattr(manager, name, original)
        manager.build_customer_dossier = original_build
        harness._active_deals_field = originals["active_deals_field"]
        harness._expected_dossier_sources = original_expected_sources
        harness._open_immutable_read_api = originals["open_read_api"]
        for name, original in (
            ("FUTURE_EVENT_AT_ALLOWED_PAIRS", originals["future_pairs"]),
            ("EVENTLESS_BOT_CONTEXT_ALLOWED_PAIRS", originals["eventless_pairs"]),
            ("event_visible_at_sql", originals["event_visible"]),
        ):
            if original is missing:
                delattr(freshness, name)
            else:
                setattr(freshness, name, original)

    return restore


def _identity_suppression_safe(row: Mapping[str, Any]) -> bool:
    if row.get("reason_code") != "identity_conflict_open":
        return False
    fields = row.get("fields") or {}
    simple_names = ("family", "money", "signals", "objections", "chronology")
    dedicated_names = ("active_deals", "attendance")
    if not all(isinstance(fields.get(name), Mapping) for name in (*simple_names, *dedicated_names)):
        return False
    simple_empty = all(
        (fields[name].get("state") in {"absent", "conflict"})
        and bool(fields[name].get("reason_code"))
        and int(fields[name].get("count") or 0) == 0
        for name in simple_names
    )
    dedicated_empty = all(
        isinstance(fields[name].get("dossier_displayed"), Mapping)
        and fields[name]["dossier_displayed"].get("state") in {"absent", "conflict"}
        and bool(fields[name]["dossier_displayed"].get("reason_code"))
        and int(fields[name]["dossier_displayed"].get("count") or 0) == 0
        for name in dedicated_names
    )
    return simple_empty and dedicated_empty


def _dossier_provenance_exact(row: Mapping[str, Any], field_name: str) -> bool:
    if _identity_suppression_safe(row):
        return True
    field = (row.get("fields") or {}).get(field_name) or {}
    return field.get("state") == "absent" or bool(
        field.get("all_sources_resolved") is True
        and field.get("projection_matches_storage") is True
        and len(field.get("provenance") or ())
        == int(field.get("resolved_count") or 0)
        == int(field.get("count") or 0)
    )


def _current_regrade(report: Mapping[str, Any], safety: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = list(report.get("rows") or ())
    summary = report.get("summary") or {}
    raw_checks = summary.get("checks") or {}
    status_counts: dict[str, int] = {}
    critical: dict[str, int] = {}
    false_ready: list[str] = []
    product_ready = 0
    for row in rows:
        status = str(row.get("action_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        if row.get("found") is not True:
            critical["missing_customer"] = critical.get("missing_customer", 0) + 1
        validation = row.get("strict_validation") or {}
        facts = validation.get("validator_facts") or {}
        if facts.get("product_readiness_ready") is True:
            product_ready += 1
            if validation.get("ready") is not True:
                false_ready.append(str(row.get("customer_sha256") or ""))
        owner = (row.get("context") or {}).get("owner_validation") or {}
        for key in (
            "missing_chunk_rows", "missing_event_rows", "foreign_chunk_owners",
            "foreign_event_owners", "duplicate_visible_chunk_ids", "invalid_unlinked_chunks",
            "source_mismatches", "time_mismatches", "invalid_event_times",
            "cutoff_violations", "superseded_events",
        ):
            value = int(owner.get(key) or 0)
            if value:
                critical[key] = critical.get(key, 0) + value
    formal_checks = dict(raw_checks.get("formal") or {})
    data_checks = dict(raw_checks.get("data") or {})
    semantic_checks = dict(raw_checks.get("semantic") or {})
    runtime_checks = dict(raw_checks.get("runtime") or {})
    semantic_checks.pop("structured_responsible_and_due_provenance_valid", None)
    semantic_checks["all_dossier_rows_provenance_exact"] = all(
        _dossier_provenance_exact(row, field_name)
        for row in rows
        for field_name in ("family", "money", "signals", "objections", "chronology")
    )
    semantic_checks["active_deals_and_attendance_projection_exact"] = all(
        _identity_suppression_safe(row)
        or (row.get("fields") or {}).get(field_name, {}).get("projection_matches_storage") is True
        for row in rows
        for field_name in ("active_deals", "attendance")
    )
    semantic_checks["identity_conflict_suppression_safe"] = all(
        _identity_suppression_safe(row)
        for row in rows
        if row.get("reason_code") == "identity_conflict_open"
    )
    semantic_checks["no_false_ready"] = not false_ready
    runtime_checks.update({
        "database_unchanged": safety.get("database_unchanged") is True,
        "network_attempts_zero": int(safety.get("network_attempts") or 0) == 0,
        "unexpected_subprocesses_zero": int(safety.get("unexpected_subprocesses") or 0) == 0,
        "adapter_unchanged": safety.get("adapter_unchanged") is True,
    })
    formal = bool(formal_checks) and all(value is True for value in formal_checks.values())
    data = bool(data_checks) and all(value is True for value in data_checks.values()) and not critical
    semantic = bool(semantic_checks) and all(value is True for value in semantic_checks.values())
    runtime = bool(runtime_checks) and all(value is True for value in runtime_checks.values())
    verdicts = {
        "formal": "PASS" if formal else "FAIL",
        "data": "PASS" if data else "FAIL",
        "semantic": "PASS" if semantic else "FAIL",
        "business": "NOT_EVALUATED_REQUIRES_30_HUMAN_DOSSIERS",
        "runtime": "PASS" if runtime else "FAIL",
    }
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "product_ready_count": product_ready,
        "ready_denominator": len(rows),
        "false_ready_count": len(false_ready),
        "false_ready_customer_sha256": false_ready,
        "critical_error_count": sum(critical.values()),
        "critical_error_reason_counts": dict(sorted(critical.items())),
        "checks": {
            "formal": formal_checks,
            "data": data_checks,
            "semantic": semantic_checks,
            "runtime": runtime_checks,
        },
        "replaced_upstream_check": {
            "structured_responsible_and_due_provenance_valid": "no_false_ready",
            "dossier_projection_exactness": "identity_conflict_suppression_safe",
        },
        "verdicts": verdicts,
        "verdict": "PASS" if all(verdicts[name] == "PASS" for name in ("formal", "data", "semantic", "runtime")) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-db-sha256", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--cutoff", required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve(strict=True)
    db = args.db.expanduser().resolve(strict=True)
    pool = args.pool.expanduser().resolve(strict=True)
    harness_path = args.harness.expanduser().resolve(strict=True)
    cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise SystemExit("STOP: --cutoff must be timezone-aware")
    if _sha_file(harness_path) != HARNESS_SHA256 or _sha_file(pool) != POOL_SHA256:
        raise SystemExit("STOP: frozen S100 harness or pool SHA256 mismatch")
    original_run = subprocess.run
    adapter_path = Path(__file__).resolve()
    adapter_rel = adapter_path.relative_to(repo)
    tracked_adapter = original_run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", str(adapter_rel)],
        check=False, capture_output=True, text=True,
    )
    if tracked_adapter.returncode != 0:
        raise SystemExit("STOP: strict S100 adapter must be committed")
    tracked_status = original_run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if tracked_status:
        raise SystemExit("STOP: tracked code differs from HEAD")
    adapter_sha = _sha_file(adapter_path)
    actual_db_sha = _sha_file(db)
    if actual_db_sha != args.expected_db_sha256:
        raise SystemExit("STOP: candidate SQLite SHA256 mismatch")
    harness = _load(harness_path)
    restore_compatibility = _install_compatibility(harness, repo)
    original_write = harness._write_atomic
    harness._write_atomic = lambda *_args, **_kwargs: None
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    network_attempts: list[str] = []
    unexpected_commands: list[str] = []

    def deny_network(*args: Any, **_kwargs: Any) -> Any:
        network_attempts.append(str(args[:1] or "network"))
        raise RuntimeError("network denied during strict S100")

    def guarded_run(command: Any, *run_args: Any, **run_kwargs: Any) -> Any:
        parts = [command] if isinstance(command, str) else [str(value) for value in command]
        allowed = parts[:1] == ["lsof"] or parts == ["git", "-C", str(repo), "rev-parse", "HEAD"]
        if not allowed:
            unexpected_commands.append(" ".join(parts))
            raise RuntimeError("unexpected subprocess during strict S100")
        return original_run(command, *run_args, **run_kwargs)

    socket.socket = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    subprocess.run = guarded_run
    try:
        report = harness._run(argparse.Namespace(
            repo=repo, db=db, pool=pool, out=args.out, tenant_id=args.tenant_id,
            cutoff=cutoff.isoformat(), expected_count=100,
        ))
    finally:
        restore_compatibility()
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        subprocess.run = original_run
        harness._write_atomic = original_write
    post_db_sha = _sha_file(db)
    post_adapter_sha = _sha_file(adapter_path)
    safety = {
        **dict(report.get("safety") or {}),
        "network_denied": True,
        "network_attempts": len(network_attempts),
        "unexpected_subprocesses": len(unexpected_commands),
        "database_unchanged": bool(
            report["database"]["unchanged"] and post_db_sha == actual_db_sha
        ),
        "database_sha256_before": actual_db_sha,
        "database_sha256_after": post_db_sha,
        "tracked_code_clean": True,
        "adapter_unchanged": post_adapter_sha == adapter_sha,
        "report_write_only": True,
    }
    final = {
        **dict(report),
        "schema_version": SCHEMA_VERSION,
        "audit_harness_sha256": HARNESS_SHA256,
        "adapter_sha256": adapter_sha,
        "database_sha256": actual_db_sha,
        "safety": safety,
        "s100_current_tz_verdict": _current_regrade(report, safety),
        "limitations": [
            *(report.get("limitations") or ()),
            "Business usefulness is evaluated separately on 30 human-reviewed dossiers.",
        ],
    }
    original_write(args.out.expanduser().resolve(strict=False), final)
    print(json.dumps(final["s100_current_tz_verdict"], ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if final["s100_current_tz_verdict"]["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
