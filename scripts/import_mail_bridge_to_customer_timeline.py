#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mango_mvp.customer_timeline import (  # noqa: E402
    CanonicalReadonlyTimelineConfig,
    CustomerTimelineReadApi,
    CustomerTimelineReadApiConfig,
    CustomerTimelineSQLiteStore,
    MailMessageNormalizer,
    TimelineImportService,
    TimelineSourceRecord,
    build_canonical_readonly_customer_timeline,
    stable_customer_id,
    stable_digest,
)


SCHEMA_VERSION = "mail_fresh_relink_customer_timeline_bridge_v1"
DEFAULT_TENANT_ID = "foton"
DEFAULT_MAIN_PROJECT_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")
DEFAULT_FRESH_RELINK_ROOT = Path(
    "/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-20/regru_edu/"
    "stage2_customer_relink_contacts_20260620"
)
DEFAULT_EMAIL_FALLBACK_AT = "2026-06-21T00:00:00+00:00"


@dataclass(frozen=True)
class FreshRelinkDecision:
    message_sha256: str
    decision: str
    reason: str
    tallanto_id: Optional[str]
    authority: str
    source_csv: str
    line_number: int

    @property
    def resolved(self) -> bool:
        return bool(self.tallanto_id)


@dataclass(frozen=True)
class PreparedEmailRecords:
    records: tuple[TimelineSourceRecord, ...]
    counts: Mapping[str, int]
    decision_counts: Mapping[str, int]
    authority_counts: Mapping[str, int]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_bridge(args)
    except Exception as exc:  # noqa: BLE001 - operator-facing CLI.
        print(f"mail bridge import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    if args.report_out:
        report_out = Path(args.report_out).expanduser().resolve(strict=False)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if report.get("validation_ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load stage2 email events through fresh-only relink into an isolated customer_timeline DB. "
            "Sources are opened read-only; AMO/Tallanto/CRM are never written."
        )
    )
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    parser.add_argument("--data-project-root", default=str(DEFAULT_MAIN_PROJECT_ROOT))
    parser.add_argument("--fresh-relink-root", default=str(DEFAULT_FRESH_RELINK_ROOT))
    parser.add_argument("--identity-db")
    parser.add_argument("--corpus-events")
    parser.add_argument("--delta-events")
    parser.add_argument("--corpus-decisions")
    parser.add_argument("--delta-decisions")
    parser.add_argument("--out-root")
    parser.add_argument("--timeline-db")
    parser.add_argument("--report-out")
    parser.add_argument("--apply", action="store_true", help="Write into the test timeline DB. Without this flag only email dry-run runs.")
    parser.add_argument("--seed-timeline-db", help="Copy an existing read-only canonical timeline DB into the test out-root before email import.")
    parser.add_argument("--build-calls", action="store_true", help="First build canonical read-only call timeline into the same test DB.")
    parser.add_argument("--max-call-events-per-contact", type=int, default=0)
    parser.add_argument("--email-limit", type=int)
    parser.add_argument("--skip-repeat-check", action="store_true")
    parser.add_argument("--generated-at", default="2026-06-21T00:00:00+00:00")
    return parser


def run_bridge(args: argparse.Namespace) -> Mapping[str, Any]:
    tenant_id = str(args.tenant_id)
    data_project_root = Path(args.data_project_root).expanduser().resolve(strict=False)
    fresh_relink_root = Path(args.fresh_relink_root).expanduser().resolve(strict=False)
    out_root = (
        Path(args.out_root).expanduser().resolve(strict=False)
        if args.out_root
        else data_project_root / "product_data" / "customer_timeline" / "canonical_readonly_email_bridge_20260621"
    )
    timeline_db = Path(args.timeline_db).expanduser().resolve(strict=False) if args.timeline_db else out_root / "customer_timeline.sqlite"
    out_root.mkdir(parents=True, exist_ok=True)
    generated_at = parse_datetime(args.generated_at)

    identity_db = Path(args.identity_db).expanduser().resolve(strict=False) if args.identity_db else default_identity_db(fresh_relink_root)
    event_paths = (
        Path(args.corpus_events).expanduser().resolve(strict=False) if args.corpus_events else default_corpus_events(data_project_root),
        Path(args.delta_events).expanduser().resolve(strict=False) if args.delta_events else default_delta_events(data_project_root),
    )
    decision_paths = (
        Path(args.corpus_decisions).expanduser().resolve(strict=False)
        if args.corpus_decisions
        else fresh_relink_root / "corpus_27009" / "mail_stage2_customer_relink_preview_decisions.csv",
        Path(args.delta_decisions).expanduser().resolve(strict=False)
        if args.delta_decisions
        else fresh_relink_root / "delta_3084" / "mail_stage2_customer_relink_preview_decisions.csv",
    )

    assert_existing_files((*event_paths, *decision_paths, identity_db))

    canonical_report: Mapping[str, Any] | None = None
    seed_timeline_db = Path(args.seed_timeline_db).expanduser().resolve(strict=False) if args.seed_timeline_db else None
    if seed_timeline_db and args.build_calls:
        raise ValueError("--seed-timeline-db and --build-calls are mutually exclusive")
    if seed_timeline_db:
        if not args.apply:
            raise ValueError("--seed-timeline-db requires --apply so the copied DB is actually used")
        if not seed_timeline_db.exists() or not seed_timeline_db.is_file():
            raise FileNotFoundError(seed_timeline_db)
        if timeline_db.exists():
            raise FileExistsError(f"target timeline DB already exists; choose a fresh out-root: {timeline_db}")
        copy_sqlite_seed(seed_timeline_db, timeline_db)
        canonical_report = {
            "mode": "seed_existing_canonical_timeline_db",
            "seed_timeline_db": str(seed_timeline_db),
            "seed_summary": summarize_timeline_db(timeline_db),
        }
    elif args.build_calls:
        canonical_report = build_canonical_readonly_customer_timeline(
            CanonicalReadonlyTimelineConfig(
                project_root=data_project_root,
                out_root=out_root,
                timeline_db=timeline_db,
                tenant_id=tenant_id,
                generated_at=generated_at,
                max_call_events_per_contact=max(0, int(args.max_call_events_per_contact or 0)),
            )
        )

    tallanto_hash_index, hash_index_report = load_tallanto_hash_index(identity_db)
    decisions, decision_report = load_fresh_relink_decisions(decision_paths, tallanto_hash_index)
    existing_customer_map = load_existing_tallanto_customer_map(timeline_db, tenant_id=tenant_id) if timeline_db.exists() else {}
    prepared = build_email_source_records(
        event_paths,
        decisions=decisions,
        tenant_id=tenant_id,
        existing_customer_by_tallanto_id=existing_customer_map,
        email_limit=args.email_limit,
    )

    store = CustomerTimelineSQLiteStore(timeline_db, allowed_root=out_root)
    try:
        before_summary = store.summary()
        service = TimelineImportService(store)
        email_report = service.import_records(
            prepared.records,
            normalizer=MailMessageNormalizer(tenant_id=tenant_id),
            tenant_id=tenant_id,
            source_ref="mail_stage2_fresh_relink_bridge_20260621",
            idempotency_key=stable_digest(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_paths": [str(path) for path in event_paths],
                    "decision_paths": [str(path) for path in decision_paths],
                    "identity_db": str(identity_db),
                    "email_limit": args.email_limit,
                }
            ),
            dry_run=not args.apply,
            actor="mail_fresh_relink_bridge",
        )
        repeat_report = None
        if args.apply and not args.skip_repeat_check:
            repeat_report = service.import_records(
                prepared.records,
                normalizer=MailMessageNormalizer(tenant_id=tenant_id),
                tenant_id=tenant_id,
                source_ref="mail_stage2_fresh_relink_bridge_20260621",
                idempotency_key=stable_digest(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "repeat": True,
                        "event_paths": [str(path) for path in event_paths],
                        "decision_paths": [str(path) for path in decision_paths],
                        "identity_db": str(identity_db),
                        "email_limit": args.email_limit,
                    }
                ),
                dry_run=False,
                actor="mail_fresh_relink_bridge_idempotency_check",
            )
        after_summary = store.summary()
        quick_check = store._con.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        store.close()

    read_api_samples = build_read_api_samples(timeline_db, out_root, tenant_id=tenant_id) if args.apply else []
    report = {
        "schema_version": SCHEMA_VERSION,
        "validation_ok": email_report.validation_ok and quick_check == "ok",
        "mode": "apply" if args.apply else "dry_run",
        "paths": {
            "timeline_db": str(timeline_db),
            "out_root": str(out_root),
            "identity_db": str(identity_db),
            "event_jsonl": [str(path) for path in event_paths],
            "fresh_relink_decisions_csv": [str(path) for path in decision_paths],
        },
        "fresh_relink": {
            "source": "fresh-only stage2_customer_relink_contacts_20260620",
            "union_used": False,
            "hash_index": hash_index_report,
            "decision_report": decision_report,
        },
        "email_prepare": {
            "records": len(prepared.records),
            "counts": dict(prepared.counts),
            "decision_counts": dict(prepared.decision_counts),
            "authority_counts": dict(prepared.authority_counts),
        },
        "canonical_calls": redact_canonical_report(canonical_report),
        "email_import_report": email_report.to_json_dict(),
        "email_repeat_import_report": repeat_report.to_json_dict() if repeat_report else None,
        "store_summary_before": before_summary,
        "store_summary_after": after_summary,
        "quick_check": quick_check,
        "read_api_samples": read_api_samples,
        "safety": {
            "read_only_sources": True,
            "write_test_timeline_db": bool(args.apply),
            "write_crm": False,
            "write_tallanto": False,
            "send_email": False,
            "use_union_relink": False,
            "inline_customer_id_used": False,
            "allowed_for_bot_gate": "mail/channel allowed_for_bot=True rejected during normalization",
        },
    }
    write_json(out_root / "mail_fresh_relink_bridge_report.json", report)
    return report


def default_identity_db(fresh_relink_root: Path) -> Path:
    external_handoffs = fresh_relink_root.parents[2]
    sibling = external_handoffs / "tallanto_contacts_export_2026-06-20" / "identity_map" / "tallanto_email_identity_map.sqlite"
    if sibling.exists():
        return sibling
    return (
        DEFAULT_FRESH_RELINK_ROOT.parents[2]
        / "tallanto_contacts_export_2026-06-20"
        / "identity_map"
        / "tallanto_email_identity_map.sqlite"
    )


def default_corpus_events(data_project_root: Path) -> Path:
    return (
        data_project_root
        / "_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/"
        "stage2_email_ingest_20260620/stage2_full_corpus_events.jsonl"
    )


def default_delta_events(data_project_root: Path) -> Path:
    return (
        data_project_root
        / "_external_handoffs/mail_archive_2026-06-20/regru_edu/incremental_20260513_to_20260620/"
        "stage2_delta_ingest_20260621/stage2_delta_full_events.jsonl"
    )


def assert_existing_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists() or not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source files are missing: {missing}")


def copy_sqlite_seed(source_db: Path, target_db: Path) -> None:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, target_db)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(source_db) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, Path(str(target_db) + suffix))


def load_tallanto_hash_index(identity_db: Path) -> tuple[dict[str, str], Mapping[str, Any]]:
    raw: dict[str, set[str]] = defaultdict(set)
    with sqlite3.connect(f"file:{identity_db}?mode=ro", uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        rows = con.execute(
            """
            SELECT tallanto_id
            FROM identity_candidates
            WHERE tallanto_id IS NOT NULL AND tallanto_id != ''
            """
        ).fetchall()
    for row in rows:
        tallanto_id = str(row["tallanto_id"]).strip()
        for variant in (tallanto_id, f"tallanto:{tallanto_id}", f"tallanto:student:{tallanto_id}"):
            raw[sha16(variant)].add(tallanto_id)
    resolved = {key: next(iter(values)) for key, values in raw.items() if len(values) == 1}
    ambiguous = {key: sorted(values) for key, values in raw.items() if len(values) > 1}
    return resolved, {
        "identity_db": str(identity_db),
        "candidates": len(rows),
        "hashes_resolved": len(resolved),
        "hashes_ambiguous": len(ambiguous),
        "read_only_sqlite": True,
    }


def load_fresh_relink_decisions(
    decision_paths: Sequence[Path],
    tallanto_hash_index: Mapping[str, str],
) -> tuple[dict[str, FreshRelinkDecision], Mapping[str, Any]]:
    decisions: dict[str, FreshRelinkDecision] = {}
    counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()
    duplicate_message_sha256 = 0
    for path in decision_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                message_sha256 = str(row.get("message_sha256") or "").strip()
                if not message_sha256:
                    continue
                decision = str(row.get("decision") or "").strip()
                reason = str(row.get("reason") or "").strip()
                tallanto_id, authority = resolve_tallanto_id_from_decision(row, tallanto_hash_index)
                item = FreshRelinkDecision(
                    message_sha256=message_sha256,
                    decision=decision,
                    reason=reason,
                    tallanto_id=tallanto_id,
                    authority=authority,
                    source_csv=str(path),
                    line_number=int(str(row.get("line_number") or "0") or 0),
                )
                existing = decisions.get(message_sha256)
                if existing is not None:
                    duplicate_message_sha256 += 1
                    if existing.resolved or not item.resolved:
                        continue
                decisions[message_sha256] = item
                counts[decision] += 1
                authority_counts[authority] += 1
    return decisions, {
        "decisions_loaded": len(decisions),
        "decision_counts": dict(counts),
        "authority_counts": dict(authority_counts),
        "duplicate_message_sha256": duplicate_message_sha256,
    }


def resolve_tallanto_id_from_decision(row: Mapping[str, str], tallanto_hash_index: Mapping[str, str]) -> tuple[Optional[str], str]:
    decision = str(row.get("decision") or "").strip()
    tallanto_hash = str(row.get("tallanto_id_hash") or "").strip()
    if decision == "linked" and tallanto_hash and tallanto_hash in tallanto_hash_index:
        return tallanto_hash_index[tallanto_hash], "fresh_relink_tallanto_id_hash"
    old_hash = str(row.get("old_customer_id_hash") or "").strip()
    if decision == "already_linked" and old_hash and old_hash in tallanto_hash_index:
        return tallanto_hash_index[old_hash], "fresh_relink_old_customer_id_hash"
    if decision in {"linked", "already_linked"}:
        return None, "fresh_relink_unresolved_hash"
    return None, "fresh_relink_unmatched"


def load_existing_tallanto_customer_map(timeline_db: Path, *, tenant_id: str) -> dict[str, str]:
    if not timeline_db.exists():
        return {}
    grouped: dict[str, set[str]] = defaultdict(set)
    with sqlite3.connect(f"file:{timeline_db}?mode=ro", uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        tables = {row["name"] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "identity_links" not in tables:
            return {}
        for row in con.execute(
            """
            SELECT link_value, customer_id
            FROM identity_links
            WHERE tenant_id = ? AND link_type = 'tallanto_student_id'
            """,
            (tenant_id,),
        ):
            grouped[str(row["link_value"])].add(str(row["customer_id"]))
    return {tallanto_id: next(iter(customer_ids)) for tallanto_id, customer_ids in grouped.items() if len(customer_ids) == 1}


def build_email_source_records(
    event_paths: Sequence[Path],
    *,
    decisions: Mapping[str, FreshRelinkDecision],
    tenant_id: str,
    existing_customer_by_tallanto_id: Mapping[str, str],
    email_limit: Optional[int] = None,
) -> PreparedEmailRecords:
    records: list[TimelineSourceRecord] = []
    counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()
    for path in event_paths:
        for row in iter_jsonl(path):
            message_sha256 = str(row.get("message_sha256") or "").strip()
            if not message_sha256:
                counts["missing_message_sha256"] += 1
                continue
            decision = decisions.get(message_sha256)
            if decision is None:
                counts["missing_fresh_relink_decision"] += 1
                decision = FreshRelinkDecision(
                    message_sha256=message_sha256,
                    decision="missing_decision",
                    reason="message_sha256_not_found_in_fresh_relink_csv",
                    tallanto_id=None,
                    authority="missing_decision",
                    source_csv="",
                    line_number=0,
                )
            payload = sanitize_email_event_payload(
                row,
                tenant_id=tenant_id,
                decision=decision,
                existing_customer_by_tallanto_id=existing_customer_by_tallanto_id,
            )
            records.append(
                TimelineSourceRecord(
                    source_system="mail_archive",
                    source_ref=f"mail:{message_sha256}",
                    payload=payload,
                    source_path=str(path),
                )
            )
            counts["records"] += 1
            counts["resolved"] += int(bool(payload.get("resolved_customer_id")))
            counts["pending_attribution"] += int(not bool(payload.get("resolved_customer_id")))
            decision_counts[decision.decision] += 1
            authority_counts[decision.authority] += 1
            if email_limit and len(records) >= email_limit:
                return PreparedEmailRecords(tuple(records), dict(counts), dict(decision_counts), dict(authority_counts))
    return PreparedEmailRecords(tuple(records), dict(counts), dict(decision_counts), dict(authority_counts))


def sanitize_email_event_payload(
    row: Mapping[str, Any],
    *,
    tenant_id: str,
    decision: FreshRelinkDecision,
    existing_customer_by_tallanto_id: Mapping[str, str],
) -> dict[str, Any]:
    message_sha256 = str(row.get("message_sha256") or decision.message_sha256)
    customer_id = None
    customer_exists = False
    if decision.tallanto_id:
        customer_id = existing_customer_by_tallanto_id.get(decision.tallanto_id)
        customer_exists = bool(customer_id)
        if not customer_id:
            customer_id = stable_customer_id(
                tenant_id=tenant_id,
                source_ref=f"tallanto:student:{decision.tallanto_id}",
            )
    payload: dict[str, Any] = {
        "message_sha256": message_sha256,
        "allowed_for_bot": parse_bool(row.get("allowed_for_bot")),
        "requires_manager_review": True,
        "date_first": row.get("date_first") or row.get("date_last") or DEFAULT_EMAIL_FALLBACK_AT,
        "date_last": row.get("date_last") or row.get("date_first") or DEFAULT_EMAIL_FALLBACK_AT,
        "thread_id": row.get("thread_id"),
        "subject": row.get("subject"),
        "summary": extract_thread_summary(row.get("thread_summary")),
        "summary_status": row.get("summary_status"),
        "brand": row.get("brand"),
        "channel": row.get("channel"),
        "source_system": row.get("source_system") or "mail_archive",
        "source_event_type": row.get("event_type"),
        "relink_decision": decision.decision,
        "relink_reason": decision.reason,
        "relink_authority": decision.authority,
        "relink_source_csv": decision.source_csv,
        "relink_line_number": decision.line_number,
        "inline_customer_id_hash": sha16(str(row.get("customer_id"))) if row.get("customer_id") else None,
    }
    if customer_id:
        payload["resolved_customer_id"] = customer_id
        payload["resolved_tallanto_id"] = decision.tallanto_id
        payload["resolved_customer_exists"] = customer_exists
    return {key: value for key, value in payload.items() if value not in (None, "")}


def extract_thread_summary(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("client_safe_summary", "summary", "short_summary", "last_message_summary", "text"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:1000]
    return str(value or "").strip()


def iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                continue
            yield dict(payload)


def build_read_api_samples(timeline_db: Path, allowed_root: Path, *, tenant_id: str) -> list[Mapping[str, Any]]:
    sample_customer_ids: list[str] = []
    with sqlite3.connect(f"file:{timeline_db}?mode=ro", uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        rows = con.execute(
            """
            SELECT customer_id, COUNT(*) AS event_count, MIN(event_at) AS first_event_at, MAX(event_at) AS last_event_at
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = 'mail_archive'
            GROUP BY customer_id
            ORDER BY event_count DESC, customer_id
            LIMIT 5
            """,
            (tenant_id,),
        ).fetchall()
        sample_customer_ids = [str(row["customer_id"]) for row in rows]
    samples: list[Mapping[str, Any]] = []
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=timeline_db, allowed_root=allowed_root)) as api:
        for customer_id in sample_customer_ids:
            profile = api.customer_profile(tenant_id, customer_id, event_limit=5, bot_context_limit=5, include_children=False)
            samples.append(
                {
                    "customer_id": customer_id,
                    "found": bool(profile.get("found")),
                    "events": profile.get("readiness", {}).get("events", 0),
                    "identity_links": profile.get("readiness", {}).get("identity_links", 0),
                    "open_conflicts": profile.get("readiness", {}).get("open_conflicts", 0),
                    "event_types": [item.get("event_type") for item in profile.get("timeline", {}).get("items", [])],
                }
            )
    return samples


def summarize_timeline_db(timeline_db: Path) -> Mapping[str, Any]:
    counts: dict[str, int] = {}
    with sqlite3.connect(f"file:{timeline_db}?mode=ro", uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        quick_check = con.execute("PRAGMA quick_check").fetchone()[0]
        for table in ("customer_identities", "identity_links", "timeline_events", "bot_context_chunks", "timeline_conflicts", "ingestion_runs"):
            try:
                counts[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.Error:
                counts[table] = -1
    return {
        "path": str(timeline_db),
        "size_bytes": timeline_db.stat().st_size if timeline_db.exists() else 0,
        "quick_check": quick_check,
        "counts": counts,
    }


def redact_canonical_report(report: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if report is None:
        return None
    if report.get("mode") == "seed_existing_canonical_timeline_db":
        return dict(report)
    return {
        "schema_version": report.get("schema_version"),
        "timeline_db": report.get("paths", {}).get("timeline_db") if isinstance(report.get("paths"), Mapping) else None,
        "source_counts": report.get("source_counts"),
        "event_counts": report.get("event_counts"),
        "imported_counts": report.get("imported_counts"),
        "write_status_counts": report.get("write_status_counts"),
        "safety": report.get("safety"),
    }


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "да"}


def sha16(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
