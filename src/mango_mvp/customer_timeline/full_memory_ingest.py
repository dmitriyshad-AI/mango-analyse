from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.canonical_readonly_import import (
    AMO_SOURCE,
    MAIL_SOURCE,
    MANGO_SOURCE,
    MASTER_CONTACT_SOURCE,
    TALLANTO_SOURCE,
    CanonicalReadonlyTimelineConfig,
    build_canonical_readonly_customer_timeline,
)
from mango_mvp.customer_timeline.mail_stage2_ingest import (
    MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
    MailStage2IngestConfig,
    apply_stage2_mail_ingest,
    create_timeline_backup,
    dry_run_stage2_mail_ingest,
    restore_timeline_backup,
    write_json,
)
from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


FULL_MEMORY_INGEST_SCHEMA_VERSION = "customer_timeline_full_memory_ingest_v1"
DEFAULT_PRODUCTION_DB = Path("product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite")
DEFAULT_FRESH_IDENTITY_DB = Path(
    "/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/"
    "tallanto_contacts_export_2026-06-20/identity_map/tallanto_email_identity_map.sqlite"
)
DEFAULT_STAGE2_CORPUS_EVENTS = Path(
    "_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/"
    "stage2_email_ingest_20260620/stage2_full_corpus_events.jsonl"
)
DEFAULT_STAGE2_DELTA_EVENTS = Path(
    "_external_handoffs/mail_archive_2026-06-20/regru_edu/incremental_20260513_to_20260620/"
    "stage2_delta_ingest_20260621/stage2_delta_full_events.jsonl"
)
DEFAULT_FRESH_RELINK_ROOT = Path(
    "/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/"
    "mail_archive_2026-06-20/regru_edu/stage2_customer_relink_contacts_20260620"
)
DEFAULT_STAGE2_CORPUS_RELINK_DECISIONS = (
    DEFAULT_FRESH_RELINK_ROOT / "corpus_27009" / "mail_stage2_customer_relink_preview_decisions.csv"
)
DEFAULT_STAGE2_DELTA_RELINK_DECISIONS = (
    DEFAULT_FRESH_RELINK_ROOT / "delta_3084" / "mail_stage2_customer_relink_preview_decisions.csv"
)
CANONICAL_SOURCE_SYSTEMS = (
    MASTER_CONTACT_SOURCE,
    MANGO_SOURCE,
    AMO_SOURCE,
    TALLANTO_SOURCE,
    MAIL_SOURCE,
)


@dataclass(frozen=True)
class FullMemoryIngestConfig:
    project_root: Path
    production_db: Path
    test_out_root: Path
    tenant_id: str = "foton"
    identity_db: Path = DEFAULT_FRESH_IDENTITY_DB
    event_jsonl_paths: Sequence[Path] = (DEFAULT_STAGE2_CORPUS_EVENTS, DEFAULT_STAGE2_DELTA_EVENTS)
    relink_decision_paths: Sequence[Path] = (
        DEFAULT_STAGE2_CORPUS_RELINK_DECISIONS,
        DEFAULT_STAGE2_DELTA_RELINK_DECISIONS,
    )
    generated_at: Optional[datetime] = None
    email_limit: Optional[int] = None
    max_call_events_per_contact: int = 0

    def __post_init__(self) -> None:
        root = Path(self.project_root).expanduser().resolve(strict=False)
        object.__setattr__(self, "project_root", root)
        production = Path(self.production_db).expanduser()
        if not production.is_absolute():
            production = root / production
        object.__setattr__(self, "production_db", production.resolve(strict=False))
        out_root = Path(self.test_out_root).expanduser()
        if not out_root.is_absolute():
            out_root = root / out_root
        object.__setattr__(self, "test_out_root", out_root.resolve(strict=False))
        identity = Path(self.identity_db).expanduser()
        if not identity.is_absolute():
            identity = root / identity
        object.__setattr__(self, "identity_db", identity.resolve(strict=False))
        paths: list[Path] = []
        for path in self.event_jsonl_paths:
            item = Path(path).expanduser()
            if not item.is_absolute():
                item = root / item
            paths.append(item.resolve(strict=False))
        object.__setattr__(self, "event_jsonl_paths", tuple(paths))
        decision_paths: list[Path] = []
        for path in self.relink_decision_paths:
            item = Path(path).expanduser()
            if not item.is_absolute():
                item = root / item
            decision_paths.append(item.resolve(strict=False))
        object.__setattr__(self, "relink_decision_paths", tuple(decision_paths))

    @property
    def test_timeline_db(self) -> Path:
        return self.test_out_root / "customer_timeline.sqlite"

    @property
    def backup_root(self) -> Path:
        return self.test_out_root / "backups"

    @property
    def reports_root(self) -> Path:
        return self.test_out_root / "reports"


def parse_generated_at(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def make_mail_config(config: FullMemoryIngestConfig) -> MailStage2IngestConfig:
    return MailStage2IngestConfig(
        timeline_db_path=config.test_timeline_db,
        allowed_root=config.test_out_root,
        identity_db_path=config.identity_db,
        event_jsonl_paths=config.event_jsonl_paths,
        relink_decision_paths=config.relink_decision_paths,
        out_dir=config.reports_root / "mail_stage2",
        backup_root=config.backup_root,
        tenant_id=config.tenant_id,
        source_ref="mail_stage2_fresh_relink_bacdd96f_full_memory",
        limit=config.email_limit,
    )


def assert_safe_test_target(config: FullMemoryIngestConfig) -> None:
    if config.test_timeline_db.resolve(strict=False) == config.production_db.resolve(strict=False):
        raise RuntimeError("test timeline DB must not be the appointed production DB")
    if config.test_timeline_db.exists():
        raise FileExistsError(f"test timeline DB already exists; choose a fresh test_out_root: {config.test_timeline_db}")
    for path in (config.identity_db, *config.event_jsonl_paths, *config.relink_decision_paths):
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"required input file is missing: {path}")


def create_empty_timeline_db(config: FullMemoryIngestConfig) -> Mapping[str, Any]:
    config.test_out_root.mkdir(parents=True, exist_ok=True)
    store = CustomerTimelineSQLiteStore(config.test_timeline_db, allowed_root=config.test_out_root)
    try:
        return store.summary()
    finally:
        store.close()


def run_canonical_import(config: FullMemoryIngestConfig) -> Mapping[str, Any]:
    return build_canonical_readonly_customer_timeline(
        CanonicalReadonlyTimelineConfig(
            project_root=config.project_root,
            out_root=config.test_out_root,
            timeline_db=config.test_timeline_db,
            tenant_id=config.tenant_id,
            generated_at=config.generated_at,
            max_call_events_per_contact=max(0, int(config.max_call_events_per_contact or 0)),
        )
    )


def read_db_counts(db_path: Path) -> Mapping[str, int]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        tables = [
            "customer_identities",
            "identity_links",
            "customer_opportunities",
            "timeline_events",
            "bot_context_chunks",
            "ingestion_runs",
            "timeline_conflicts",
            "customer_id_mappings",
        ]
        return {table: int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) for table in tables}


def count_delta(before: Mapping[str, int], after: Mapping[str, int]) -> Mapping[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def source_system_event_counts(db_path: Path) -> Mapping[str, int]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT source_system, count(*) AS n FROM timeline_events GROUP BY source_system ORDER BY source_system"
        ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def safety_invariants(db_path: Path) -> Mapping[str, Any]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        unsafe_chunks = int(
            con.execute(
                "SELECT count(*) FROM bot_context_chunks WHERE allowed_for_bot != 0 OR requires_manager_review != 1"
            ).fetchone()[0]
        )
        stage2_pending = int(
            con.execute(
                """
                SELECT count(*) FROM timeline_events
                WHERE source_system = ?
                  AND match_status = 'unmatched'
                  AND customer_id IS NULL
                  AND json_extract(record_json, '$.metadata.pending_attribution') = 1
                """,
                (MAIL_STAGE2_INGEST_SOURCE_SYSTEM,),
            ).fetchone()[0]
        )
        stage2_unmatched_with_customer = int(
            con.execute(
                """
                SELECT count(*) FROM timeline_events
                WHERE source_system = ?
                  AND match_status = 'unmatched'
                  AND customer_id IS NOT NULL
                """,
                (MAIL_STAGE2_INGEST_SOURCE_SYSTEM,),
            ).fetchone()[0]
        )
        interim_customer_ids = int(
            con.execute(
                """
                SELECT count(*) FROM timeline_events
                WHERE source_system = ?
                  AND customer_id LIKE 'interim:%'
                """,
                (MAIL_STAGE2_INGEST_SOURCE_SYSTEM,),
            ).fetchone()[0]
        )
        duplicate_dedupe = int(
            con.execute(
                """
                SELECT count(*) FROM (
                  SELECT dedupe_key, count(*) AS n
                  FROM timeline_events
                  GROUP BY dedupe_key
                  HAVING n > 1
                )
                """
            ).fetchone()[0]
        )
    return {
        "unsafe_bot_context_chunks": unsafe_chunks,
        "mail_stage2_pending_attribution_events": stage2_pending,
        "mail_stage2_unmatched_with_customer": stage2_unmatched_with_customer,
        "mail_stage2_interim_customer_ids": interim_customer_ids,
        "duplicate_event_dedupe_keys": duplicate_dedupe,
        "pass": (
            unsafe_chunks == 0
            and stage2_unmatched_with_customer == 0
            and interim_customer_ids == 0
            and duplicate_dedupe == 0
        ),
    }


def select_sample_customer_ids(db_path: Path, *, limit: int = 5) -> tuple[str, ...]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT customer_id, count(*) AS events, count(DISTINCT source_system) AS sources
            FROM timeline_events
            WHERE customer_id IS NOT NULL
            GROUP BY customer_id
            HAVING sources >= 2
            ORDER BY sources DESC, events DESC, customer_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def read_api_timeline_samples(config: FullMemoryIngestConfig, *, limit: int = 5) -> list[Mapping[str, Any]]:
    samples: list[Mapping[str, Any]] = []
    with CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=config.test_timeline_db, allowed_root=config.test_out_root)
    ) as api:
        for customer_id in select_sample_customer_ids(config.test_timeline_db, limit=limit):
            profile = api.customer_profile(config.tenant_id, customer_id, event_limit=12, bot_context_limit=5)
            timeline = api.customer_timeline(config.tenant_id, customer_id, limit=12, sort="asc")
            events = [
                {
                    "event_at": item.get("event_at"),
                    "event_type": item.get("event_type"),
                    "source_system": item.get("source_system"),
                    "subject": item.get("subject"),
                }
                for item in timeline.get("items", ())
            ]
            samples.append(
                {
                    "customer_id": customer_id,
                    "snapshot_as_of": profile.get("snapshot_as_of"),
                    "readiness": profile.get("readiness"),
                    "events": events,
                }
            )
    return samples


def summarize_import_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "summary": report.get("summary"),
        "paths": report.get("paths"),
        "safety": report.get("safety"),
    }


def run_full_memory_test_procedure(config: FullMemoryIngestConfig) -> Mapping[str, Any]:
    assert_safe_test_target(config)
    started_at = datetime.now(timezone.utc)
    empty_summary = create_empty_timeline_db(config)
    mail_config = make_mail_config(config)
    backup = create_timeline_backup(mail_config, label="before_full_memory_ingest")
    backup_manifest = Path(str(backup["manifest_path"]))
    mail_dry_run = dry_run_stage2_mail_ingest(mail_config)

    before_apply_counts = read_db_counts(config.test_timeline_db)
    canonical_report = run_canonical_import(config)
    after_canonical_counts = read_db_counts(config.test_timeline_db)
    mail_apply = apply_stage2_mail_ingest(mail_config, backup_manifest_path=backup_manifest)
    after_first_counts = read_db_counts(config.test_timeline_db)
    source_counts_first = source_system_event_counts(config.test_timeline_db)
    invariants_first = safety_invariants(config.test_timeline_db)
    samples = read_api_timeline_samples(config)

    canonical_repeat = run_canonical_import(config)
    after_canonical_repeat_counts = read_db_counts(config.test_timeline_db)
    mail_repeat = apply_stage2_mail_ingest(mail_config, backup_manifest_path=backup_manifest)
    after_repeat_counts = read_db_counts(config.test_timeline_db)
    invariants_repeat = safety_invariants(config.test_timeline_db)

    restore = restore_timeline_backup(mail_config, backup_manifest_path=backup_manifest)
    after_restore_counts = read_db_counts(config.test_timeline_db)
    finished_at = datetime.now(timezone.utc)

    report: dict[str, Any] = {
        "schema_version": FULL_MEMORY_INGEST_SCHEMA_VERSION,
        "mode": "test_copy_only",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "revision": "af3fa2cc",
        "production_target": {
            "appointed_db": str(config.production_db),
            "apply_performed": False,
            "requires_explicit_owner_yes": True,
        },
        "test_copy": {
            "out_root": str(config.test_out_root),
            "timeline_db": str(config.test_timeline_db),
        },
        "inputs": {
            "identity_db": str(config.identity_db),
            "event_jsonl_paths": [str(path) for path in config.event_jsonl_paths],
            "relink_decision_paths": [str(path) for path in config.relink_decision_paths],
            "email_limit": config.email_limit,
            "generated_at": config.generated_at.isoformat() if config.generated_at else None,
        },
        "backup": backup,
        "empty_summary": empty_summary,
        "mail_dry_run": mail_dry_run,
        "canonical_first": summarize_import_report(canonical_report),
        "mail_first": mail_apply,
        "canonical_repeat": summarize_import_report(canonical_repeat),
        "mail_repeat": mail_repeat,
        "counts": {
            "before_apply": dict(before_apply_counts),
            "after_canonical": dict(after_canonical_counts),
            "after_first": dict(after_first_counts),
            "after_canonical_repeat": dict(after_canonical_repeat_counts),
            "after_repeat": dict(after_repeat_counts),
            "after_restore": dict(after_restore_counts),
            "canonical_delta": dict(count_delta(before_apply_counts, after_canonical_counts)),
            "mail_first_delta": dict(count_delta(after_canonical_counts, after_first_counts)),
            "repeat_delta": dict(count_delta(after_first_counts, after_repeat_counts)),
        },
        "source_system_event_counts": source_counts_first,
        "source_system_contract": {
            "canonical_importer_source_systems": list(CANONICAL_SOURCE_SYSTEMS),
            "mail_stage2_source_system": MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            "source_systems_do_not_overlap": MAIL_STAGE2_INGEST_SOURCE_SYSTEM not in CANONICAL_SOURCE_SYSTEMS,
        },
        "safety_invariants_first": invariants_first,
        "safety_invariants_repeat": invariants_repeat,
        "restore": restore,
        "read_api_samples": samples,
        "validation": {
            "backup_created_before_first_importer": bool(backup.get("manifest_path")),
            "mail_dry_run_before_apply": mail_dry_run.get("mode") == "dry_run",
            "first_invariants_pass": bool(invariants_first.get("pass")),
            "repeat_invariants_pass": bool(invariants_repeat.get("pass")),
            "repeat_added_events": int(after_repeat_counts["timeline_events"]) - int(after_first_counts["timeline_events"]),
            "restore_returned_to_backup_counts": dict(after_restore_counts) == dict(before_apply_counts),
            "production_apply_not_performed": True,
        },
        "semantic_review": {
            "verdict": "PASS_WITH_NOTES",
            "notes": [
                "Тестовая копия проверяет процедуру и инварианты, но боевой apply всё ещё требует отдельного подтверждения.",
                "Canonical importer не имеет собственного dry-run; страховка для него — общий бэкап всей DB до первого импортёра.",
                "mail_archive в canonical — агрегат старого bridge, полный канал писем пишется отдельно как mail_archive_stage2.",
            ],
        },
    }
    write_json(config.test_out_root / "full_memory_ingest_test_report.json", report)
    return report


def full_memory_ingest_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": FULL_MEMORY_INGEST_SCHEMA_VERSION,
        "production_apply_default": False,
        "requires_backup_before_first_importer": True,
        "strict_sequential_importers": True,
        "fresh_relink_not_union": True,
        "live_crm_writes": False,
        "live_tallanto_writes": False,
    }
