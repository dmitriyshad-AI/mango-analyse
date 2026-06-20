from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityLinkType,
    IdentityMatchClass,
    IdentityStatus,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.ids import normalize_key
from mango_mvp.customer_timeline.ingestion import compact_text
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.productization.mail_archive import (
    build_stage2_value_brand_scope,
    clean_text,
    decide_stage2_event_customer_relink,
    load_stage2_mail_events,
    load_tallanto_customer_address_book,
    read_safe_stage2_event_text,
    stable_value_hash,
    stage2_event_external_emails,
    stage2_event_phone_signals,
)


MAIL_STAGE2_TIMELINE_INGEST_SCHEMA_VERSION = "mail_stage2_timeline_ingest_v1"
MAIL_STAGE2_INGEST_SOURCE_SYSTEM = "mail_archive_stage2"


@dataclass(frozen=True)
class MailStage2IngestConfig:
    timeline_db_path: Path
    allowed_root: Path
    identity_db_path: Path
    event_jsonl_paths: Sequence[Path]
    out_dir: Path
    relink_decision_paths: Sequence[Path] = ()
    backup_root: Optional[Path] = None
    tenant_id: str = "mango"
    source_ref: str = "mail_stage2_fresh_relink_20260621"
    text_max_chars: int = 6000
    limit: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db_path", Path(self.timeline_db_path).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "identity_db_path", Path(self.identity_db_path).expanduser())
        object.__setattr__(
            self,
            "event_jsonl_paths",
            tuple(Path(path).expanduser() for path in self.event_jsonl_paths),
        )
        object.__setattr__(
            self,
            "relink_decision_paths",
            tuple(Path(path).expanduser() for path in self.relink_decision_paths),
        )
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        if self.backup_root is not None:
            object.__setattr__(self, "backup_root", Path(self.backup_root).expanduser())
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))


@dataclass(frozen=True)
class PlannedMailStage2Event:
    batch: Mapping[str, Any]
    event: TimelineEvent
    customer: Optional[CustomerIdentity]
    identity_links: tuple[IdentityLink, ...]
    opportunity: Optional[CustomerOpportunity]
    chunk: Optional[BotContextChunk]
    decision: Mapping[str, Any]


@dataclass(frozen=True)
class MailStage2RelinkDecision:
    message_sha256: str
    decision: str
    reason: str
    tallanto_id: Optional[str]
    authority: str
    source_csv: str
    line_number: int
    tallanto_id_hash: str = ""
    signal_kind: str = ""
    signal_value_sha256: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.tallanto_id)

    def to_decision_mapping(self) -> Mapping[str, Any]:
        return {
            "message_sha256": self.message_sha256,
            "decision": "linked" if self.resolved else self.decision,
            "reason": self.reason,
            "tallanto_id": self.tallanto_id or "",
            "tallanto_id_hash": self.tallanto_id_hash,
            "signal_kind": self.signal_kind,
            "signal_value_sha256": self.signal_value_sha256,
            "authority": self.authority,
            "source_csv": self.source_csv,
            "line_number": self.line_number,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_event_at(event: Mapping[str, Any]) -> datetime:
    raw = clean_text(event.get("date_iso") or event.get("date_first") or event.get("date") or event.get("message_date_iso"))
    if raw:
        text = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw)
            except (TypeError, ValueError, IndexError, OverflowError):
                parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _event_direction(event: Mapping[str, Any]) -> TimelineDirection:
    kind = clean_text(event.get("message_kind") or event.get("direction") or event.get("folder")).casefold()
    if any(token in kind for token in ("sent", "out", "исход")):
        return TimelineDirection.OUTBOUND
    return TimelineDirection.INBOUND


def _message_sha(event: Mapping[str, Any]) -> str:
    value = clean_text(event.get("message_sha256") or event.get("sha"))
    if value:
        return value
    fallback = {
        "source_file": clean_text(event.get("_source_file")),
        "line_number": int(event.get("_line_number") or 0),
        "subject": clean_text(event.get("subject")),
        "date": clean_text(event.get("date_iso") or event.get("date_first") or event.get("date")),
    }
    return hashlib.sha256(json.dumps(fallback, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _sha16(value: str) -> str:
    return hashlib.sha256(clean_text(value).encode("utf-8")).hexdigest()[:16]


def _safe_text_summary(event: Mapping[str, Any], *, max_chars: int) -> tuple[str, str]:
    text, status = read_safe_stage2_event_text(event.get("extracted_text_path"), max_chars=max_chars)
    if text:
        return text, status
    fallback = clean_text(event.get("summary") or event.get("text_preview") or event.get("subject"))
    return fallback, status


def _customer_from_fresh_relink(
    *,
    tenant_id: str,
    tallanto_id: str,
    address_book: Mapping[str, Any],
    source_ref: str,
    event_at: datetime,
    customer_id: Optional[str] = None,
) -> CustomerIdentity:
    clients = address_book.get("clients") or {}
    client = clients.get(tallanto_id) or {}
    emails = sorted(str(item) for item in client.get("emails", set()) if item)
    common_emails = set(client.get("common_emails", set()))
    primary_email = next((email for email in emails if email not in common_emails), emails[0] if emails else None)
    phones = sorted(str(item) for item in client.get("phones", set()) if item)
    common_phones = set(client.get("common_phones", set()))
    primary_phone = next((phone for phone in phones if phone not in common_phones), phones[0] if phones else None)
    names = sorted(str(item) for item in client.get("names", set()) if item)
    return CustomerIdentity(
        tenant_id=tenant_id,
        customer_id=customer_id or f"tallanto:{tallanto_id}",
        identity_status=IdentityStatus.STRONG,
        display_name=names[0] if names else None,
        primary_email=primary_email,
        primary_phone=primary_phone,
        source_ref=source_ref,
        first_seen_at=event_at,
        last_seen_at=event_at,
        touch_count=1,
        summary={
            "source": "mail_stage2_fresh_relink",
            "tallanto_id_hash": stable_value_hash(tallanto_id)[:16],
        },
        metadata={
            "identity_source": "fresh_relink_bacdd96f",
            "candidate_keys_count": len(client.get("candidate_keys", set())),
        },
        created_at=event_at,
        updated_at=event_at,
    )


def _build_links(
    *,
    tenant_id: str,
    customer_id: str,
    tallanto_id: str,
    decision: Mapping[str, Any],
    source_ref: str,
    event_at: datetime,
) -> tuple[IdentityLink, ...]:
    links: list[IdentityLink] = [
        IdentityLink(
            tenant_id=tenant_id,
            customer_id=customer_id,
            link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
            link_value=tallanto_id,
            source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            source_ref=source_ref,
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.95,
            evidence={"decision": "fresh_relink", "message_sha256": clean_text(decision.get("message_sha256"))},
            first_seen_at=event_at,
            last_seen_at=event_at,
        )
    ]
    signal_kind = clean_text(decision.get("signal_kind"))
    signal_value_sha = clean_text(decision.get("signal_value_sha256"))
    if signal_kind in {"email", "phone"} and signal_value_sha:
        # The raw matched value is intentionally not reconstructed from the hash here.
        # The authoritative Tallanto id is the actual customer key; the signal hash remains audit evidence.
        links[0] = IdentityLink(
            tenant_id=tenant_id,
            customer_id=customer_id,
            link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
            link_value=tallanto_id,
            source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            source_ref=source_ref,
            match_class=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.95,
            evidence={
                "decision": "fresh_relink",
                "signal_kind": signal_kind,
                "signal_value_sha256": signal_value_sha,
            },
            first_seen_at=event_at,
            last_seen_at=event_at,
        )
    return tuple(links)


def prepare_stage2_events(
    config: MailStage2IngestConfig,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Mapping[str, set[str]]]]:
    events = load_stage2_mail_events(config.event_jsonl_paths)
    if config.limit is not None:
        events = events[: max(0, int(config.limit))]
    prepared: list[dict[str, Any]] = []
    for event in events:
        emails = stage2_event_external_emails(
            event,
            internal_domains=("foton", "unpk", "cdpofoton.ru", "edu.mipt.ru", "kmipt.ru"),
        )
        phones = stage2_event_phone_signals(event, max_chars=config.text_max_chars)
        prepared.append({"event": event, "emails": emails, "phones": phones})
    return prepared, load_tallanto_customer_address_book(config.identity_db_path), build_stage2_value_brand_scope(prepared)


def load_tallanto_hash_index(identity_db_path: Path) -> tuple[dict[str, str], Mapping[str, Any]]:
    raw: dict[str, set[str]] = defaultdict(set)
    with sqlite3.connect(f"file:{identity_db_path}?mode=ro", uri=True, timeout=15) as con:
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
        tallanto_id = clean_text(row["tallanto_id"])
        if not tallanto_id:
            continue
        for variant in (tallanto_id, f"tallanto:{tallanto_id}", f"tallanto:student:{tallanto_id}"):
            raw[_sha16(variant)].add(tallanto_id)
    resolved = {key: next(iter(values)) for key, values in raw.items() if len(values) == 1}
    ambiguous = {key: sorted(values) for key, values in raw.items() if len(values) > 1}
    return resolved, {
        "identity_db": str(identity_db_path),
        "candidates": len(rows),
        "hashes_resolved": len(resolved),
        "hashes_ambiguous": len(ambiguous),
        "read_only_sqlite": True,
    }


def resolve_tallanto_id_from_relink_row(
    row: Mapping[str, str],
    tallanto_hash_index: Mapping[str, str],
) -> tuple[Optional[str], str]:
    decision = clean_text(row.get("decision"))
    tallanto_hash = clean_text(row.get("tallanto_id_hash"))
    if decision == "linked" and tallanto_hash and tallanto_hash in tallanto_hash_index:
        return tallanto_hash_index[tallanto_hash], "fresh_relink_tallanto_id_hash"
    old_hash = clean_text(row.get("old_customer_id_hash"))
    if decision == "already_linked" and old_hash and old_hash in tallanto_hash_index:
        return tallanto_hash_index[old_hash], "fresh_relink_old_customer_id_hash"
    if decision in {"linked", "already_linked"}:
        return None, "fresh_relink_unresolved_hash"
    return None, "fresh_relink_unmatched"


def load_relink_decisions(
    decision_paths: Sequence[Path],
    *,
    tallanto_hash_index: Mapping[str, str],
) -> tuple[dict[str, MailStage2RelinkDecision], Mapping[str, Any]]:
    decisions: dict[str, MailStage2RelinkDecision] = {}
    decision_counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()
    duplicate_message_sha256 = 0
    for path in decision_paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                message_sha = clean_text(row.get("message_sha256"))
                if not message_sha:
                    continue
                tallanto_id, authority = resolve_tallanto_id_from_relink_row(row, tallanto_hash_index)
                item = MailStage2RelinkDecision(
                    message_sha256=message_sha,
                    decision=clean_text(row.get("decision")),
                    reason=clean_text(row.get("reason")),
                    tallanto_id=tallanto_id,
                    authority=authority,
                    source_csv=str(path),
                    line_number=int(clean_text(row.get("line_number")) or "0"),
                    tallanto_id_hash=clean_text(row.get("tallanto_id_hash")),
                    signal_kind=clean_text(row.get("signal_kind")),
                    signal_value_sha256=clean_text(row.get("signal_value_sha256")),
                )
                existing = decisions.get(message_sha)
                if existing is not None:
                    duplicate_message_sha256 += 1
                    if existing.resolved or not item.resolved:
                        continue
                decisions[message_sha] = item
                decision_counts[item.decision] += 1
                authority_counts[item.authority] += 1
    return decisions, {
        "decisions_loaded": len(decisions),
        "decision_counts": dict(sorted(decision_counts.items())),
        "authority_counts": dict(sorted(authority_counts.items())),
        "duplicate_message_sha256": duplicate_message_sha256,
    }


def load_existing_tallanto_customer_map(timeline_db_path: Path, *, tenant_id: str) -> dict[str, str]:
    path = Path(timeline_db_path)
    if not path.exists():
        return {}
    grouped: dict[str, set[str]] = defaultdict(set)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15) as con:
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
            grouped[clean_text(row["link_value"])].add(str(row["customer_id"]))
    return {tallanto_id: next(iter(customer_ids)) for tallanto_id, customer_ids in grouped.items() if len(customer_ids) == 1}


def relink_decision_for_stage2_event(
    *,
    message_sha: str,
    fallback_decision: Mapping[str, Any],
    relink_decisions: Mapping[str, MailStage2RelinkDecision],
) -> Mapping[str, Any]:
    if not relink_decisions:
        return fallback_decision
    decision = relink_decisions.get(message_sha)
    if decision is None:
        return {
            "message_sha256": message_sha,
            "decision": "unmatched",
            "reason": "message_sha256_not_found_in_relink_decisions",
            "tallanto_id": "",
            "authority": "missing_decision",
        }
    return decision.to_decision_mapping()


def plan_stage2_mail_ingest(config: MailStage2IngestConfig) -> tuple[list[PlannedMailStage2Event], Mapping[str, int]]:
    prepared, address_book, value_brand_scope = prepare_stage2_events(config)
    tallanto_hash_index, hash_index_report = load_tallanto_hash_index(config.identity_db_path)
    relink_decisions, relink_report = load_relink_decisions(
        config.relink_decision_paths,
        tallanto_hash_index=tallanto_hash_index,
    )
    existing_customer_by_tallanto_id = load_existing_tallanto_customer_map(config.timeline_db_path, tenant_id=config.tenant_id)
    plans: list[PlannedMailStage2Event] = []
    counters = {
        "input_events": len(prepared),
        "linked": 0,
        "unmatched": 0,
        "chunks": 0,
        "fallback_sha": 0,
        "relink_decisions_loaded": int(relink_report.get("decisions_loaded", 0)),
        "relink_existing_customer_map": len(existing_customer_by_tallanto_id),
        "relink_hashes_resolved": int(hash_index_report.get("hashes_resolved", 0)),
    }
    for item in prepared:
        event = item["event"]
        event_at = _parse_event_at(event)
        message_sha = _message_sha(event)
        if not clean_text(event.get("message_sha256") or event.get("sha")):
            counters["fallback_sha"] += 1
        source_file = clean_text(event.get("_source_file")) or "stage2_mail_events.jsonl"
        source_ref = f"mail_stage2:{source_file}:{int(event.get('_line_number') or 0)}:{message_sha[:16]}"
        text, text_status = _safe_text_summary(event, max_chars=config.text_max_chars)
        event_for_fresh_relink = dict(event)
        event_for_fresh_relink["customer_id"] = ""
        fallback_decision = decide_stage2_event_customer_relink(
            event_for_fresh_relink,
            emails=item["emails"],
            phones=item["phones"],
            address_book=address_book,
            value_brand_scope=value_brand_scope,
        )
        decision = relink_decision_for_stage2_event(
            message_sha=message_sha,
            fallback_decision=fallback_decision,
            relink_decisions=relink_decisions,
        )
        subject = compact_text(event.get("subject"), limit=160) or "Email message"
        record = {
            "message_sha256": message_sha,
            "source_file": source_file,
            "line_number": int(event.get("_line_number") or 0),
            "subject_hash": stable_value_hash(event.get("subject"))[:16] if event.get("subject") else "",
            "brand_signal": clean_text(event.get("brand") or event.get("brand_signal") or "unknown"),
            "brand_source": clean_text(event.get("brand_source") or event.get("brand_src") or event.get("brand_note")),
            "extracted_text_path": clean_text(event.get("extracted_text_path")),
            "text_read_status": text_status,
            "fresh_relink_decision": {
                "decision": clean_text(decision.get("decision")),
                "reason": clean_text(decision.get("reason")),
                "tallanto_id_hash": clean_text(decision.get("tallanto_id_hash")),
                "signal_kind": clean_text(decision.get("signal_kind")),
                "signal_value_sha256": clean_text(decision.get("signal_value_sha256")),
                "authority": clean_text(decision.get("authority")),
                "source_csv": clean_text(decision.get("source_csv")),
                "old_customer_id_hash": stable_value_hash(event.get("customer_id"))[:16] if event.get("customer_id") else "",
            },
        }
        if decision.get("decision") == "linked" and clean_text(decision.get("tallanto_id")):
            tallanto_id = clean_text(decision.get("tallanto_id"))
            existing_customer_id = existing_customer_by_tallanto_id.get(tallanto_id)
            customer = None
            customer_id = existing_customer_id
            if not customer_id:
                customer = _customer_from_fresh_relink(
                    tenant_id=config.tenant_id,
                    tallanto_id=tallanto_id,
                    address_book=address_book,
                    source_ref=source_ref,
                    event_at=event_at,
                )
                customer_id = customer.customer_id
            links = _build_links(
                tenant_id=config.tenant_id,
                customer_id=customer_id,
                tallanto_id=tallanto_id,
                decision=decision,
                source_ref=source_ref,
                event_at=event_at,
            )
            opportunity = CustomerOpportunity(
                tenant_id=config.tenant_id,
                customer_id=customer_id,
                opportunity_type=OpportunityType.MAIL_THREAD,
                source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
                source_id=clean_text(event.get("thread_id") or event.get("conversation_id")) or message_sha,
                title=subject,
                status="observed",
                opened_at=event_at,
                confidence=0.8,
                evidence={"message_sha256": message_sha, "fresh_relink": True},
            )
            timeline_event = TimelineEvent(
                tenant_id=config.tenant_id,
                customer_id=customer_id,
                opportunity_id=opportunity.opportunity_id,
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=event_at,
                source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
                source_id=message_sha,
                source_ref=source_ref,
                direction=_event_direction(event),
                subject=subject,
                text_preview=compact_text(text, limit=240),
                summary=compact_text(text, limit=500),
                match_status=IdentityMatchClass.STRONG_UNIQUE,
                confidence=0.95,
                record=record,
                metadata={"pending_attribution": False, "fresh_relink": True},
                created_at=event_at,
            )
            chunk = None
            if text:
                chunk = BotContextChunk(
                    tenant_id=config.tenant_id,
                    customer_id=customer_id,
                    opportunity_id=opportunity.opportunity_id,
                    event_id=timeline_event.event_id,
                    source_ref=source_ref,
                    source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
                    chunk_type="email_message",
                    text=text,
                    summary=compact_text(text, limit=500),
                    event_at=event_at,
                    freshness_score=0.7,
                    relevance_tags=("email", "manager_only"),
                    allowed_for_bot=False,
                    requires_manager_review=True,
                    metadata={"fresh_relink": True, "message_sha256": message_sha},
                    created_at=event_at,
                )
                counters["chunks"] += 1
            counters["linked"] += 1
            plans.append(
                PlannedMailStage2Event(
                    batch=item,
                    event=timeline_event,
                    customer=customer,
                    identity_links=links,
                    opportunity=opportunity,
                    chunk=chunk,
                    decision=decision,
                )
            )
            continue
        timeline_event = TimelineEvent(
            tenant_id=config.tenant_id,
            customer_id=None,
            event_type=TimelineEventType.EMAIL_MESSAGE,
            event_at=event_at,
            source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            source_id=message_sha,
            source_ref=source_ref,
            direction=_event_direction(event),
            subject=subject,
            text_preview=compact_text(text, limit=240),
            summary=compact_text(text, limit=500),
            match_status=IdentityMatchClass.UNMATCHED,
            confidence=0.0,
            record=record,
            metadata={
                "pending_attribution": True,
                "pending_reason": clean_text(decision.get("reason")) or "unmatched",
                "fresh_relink": True,
            },
            created_at=event_at,
        )
        counters["unmatched"] += 1
        plans.append(
            PlannedMailStage2Event(
                batch=item,
                event=timeline_event,
                customer=None,
                identity_links=(),
                opportunity=None,
                chunk=None,
                decision=decision,
            )
        )
    return plans, counters


def existing_event_dedupe_keys(db_path: Path) -> set[str]:
    path = Path(db_path)
    if not path.exists():
        return set()
    with sqlite3.connect(path) as con:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='timeline_events'"
        ).fetchone()
        if not exists:
            return set()
        return {str(row[0]) for row in con.execute("SELECT dedupe_key FROM timeline_events WHERE dedupe_key IS NOT NULL")}


def input_fingerprint(config: MailStage2IngestConfig) -> str:
    payload = {
        "schema_version": MAIL_STAGE2_TIMELINE_INGEST_SCHEMA_VERSION,
        "identity_db": file_sha256(config.identity_db_path),
        "event_jsonl": [(str(path), file_sha256(path)) for path in config.event_jsonl_paths],
        "relink_decisions": [(str(path), file_sha256(path)) for path in config.relink_decision_paths],
        "tenant_id": config.tenant_id,
        "source_ref": config.source_ref,
        "limit": config.limit,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def create_timeline_backup(config: MailStage2IngestConfig, *, label: Optional[str] = None) -> Mapping[str, Any]:
    db_path = config.timeline_db_path
    if not db_path.exists():
        raise FileNotFoundError(f"timeline DB does not exist, cannot create backup: {db_path}")
    backup_root = config.backup_root or (config.allowed_root / "backups")
    backup_dir = backup_root / f"{label or 'mail_stage2'}_{_now_stamp()}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_db = backup_dir / db_path.name
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source, sqlite3.connect(backup_db) as target:
        source.backup(target)
    manifest = {
        "schema_version": MAIL_STAGE2_TIMELINE_INGEST_SCHEMA_VERSION,
        "kind": "timeline_backup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db_path": str(db_path.resolve(strict=False)),
        "backup_db_path": str(backup_db.resolve(strict=False)),
        "backup_sha256": file_sha256(backup_db),
        "source_sha256_after_backup": file_sha256(db_path),
        "tenant_id": config.tenant_id,
        "source_ref": config.source_ref,
    }
    manifest_path = backup_dir / "backup_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def validate_backup_manifest(config: MailStage2IngestConfig, manifest_path: Path) -> Mapping[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "timeline_backup":
        raise ValueError("backup manifest has wrong kind")
    source_db = Path(str(manifest.get("source_db_path", ""))).resolve(strict=False)
    target_db = config.timeline_db_path.resolve(strict=False)
    if source_db != target_db:
        raise ValueError(f"backup source DB mismatch: {source_db} != {target_db}")
    backup_db = Path(str(manifest.get("backup_db_path", ""))).expanduser()
    if not backup_db.exists():
        raise FileNotFoundError(f"backup DB not found: {backup_db}")
    expected_sha = clean_text(manifest.get("backup_sha256"))
    actual_sha = file_sha256(backup_db)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("backup DB sha256 mismatch")
    return manifest


def dry_run_stage2_mail_ingest(config: MailStage2IngestConfig) -> Mapping[str, Any]:
    plans, counters = plan_stage2_mail_ingest(config)
    existing = existing_event_dedupe_keys(config.timeline_db_path)
    seen: set[str] = set()
    duplicate_in_input = 0
    would_create = 0
    would_skip_existing = 0
    for plan in plans:
        key = plan.event.dedupe_key
        if key in seen:
            duplicate_in_input += 1
            continue
        seen.add(key)
        if key in existing:
            would_skip_existing += 1
        else:
            would_create += 1
    report = {
        "schema_version": MAIL_STAGE2_TIMELINE_INGEST_SCHEMA_VERSION,
        "mode": "dry_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeline_db_path": str(config.timeline_db_path),
        "identity_db_path": str(config.identity_db_path),
        "relink_decision_paths": [str(path) for path in config.relink_decision_paths],
        "identity_mode": "fresh_relink_decision_join_bacdd96f" if config.relink_decision_paths else "fresh_relink_bacdd96f",
        "union_identity_db_used": False,
        "input_hash": input_fingerprint(config),
        "counts": {
            **counters,
            "planned_events": len(plans),
            "would_create_events": would_create,
            "would_skip_existing_events": would_skip_existing,
            "duplicate_events_in_input": duplicate_in_input,
        },
        "gates": {
            "channel_chunks_allowed_for_bot": False,
            "unmatched_pending_attribution": True,
            "bot_visible_unmatched_chunks": 0,
            "requires_backup_for_apply": True,
        },
    }
    write_json(config.out_dir / "dry_run_report.json", report)
    return report


def apply_stage2_mail_ingest(config: MailStage2IngestConfig, *, backup_manifest_path: Path) -> Mapping[str, Any]:
    backup_manifest = validate_backup_manifest(config, backup_manifest_path)
    plans, counters = plan_stage2_mail_ingest(config)
    existing = existing_event_dedupe_keys(config.timeline_db_path)
    seen: set[str] = set()
    selected: list[PlannedMailStage2Event] = []
    duplicate_in_input = 0
    skipped_existing = 0
    for plan in plans:
        key = plan.event.dedupe_key
        if key in seen:
            duplicate_in_input += 1
            continue
        seen.add(key)
        if key in existing:
            skipped_existing += 1
            continue
        selected.append(plan)

    run_report: dict[str, Any] = {
        "schema_version": MAIL_STAGE2_TIMELINE_INGEST_SCHEMA_VERSION,
        "mode": "apply",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeline_db_path": str(config.timeline_db_path),
        "identity_db_path": str(config.identity_db_path),
        "relink_decision_paths": [str(path) for path in config.relink_decision_paths],
        "identity_mode": "fresh_relink_decision_join_bacdd96f" if config.relink_decision_paths else "fresh_relink_bacdd96f",
        "union_identity_db_used": False,
        "input_hash": input_fingerprint(config),
        "backup_manifest_path": str(Path(backup_manifest_path)),
        "backup_sha256": backup_manifest.get("backup_sha256"),
        "counts": {
            **counters,
            "planned_events": len(plans),
            "selected_new_events": len(selected),
            "skipped_existing_events": skipped_existing,
            "duplicate_events_in_input": duplicate_in_input,
            "created_events": 0,
            "created_chunks": 0,
            "pending_attribution_events": 0,
        },
    }
    store = CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=config.allowed_root)
    try:
        run = store.start_ingestion_run(
            tenant_id=config.tenant_id,
            source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            source_ref=config.source_ref,
            run_kind="mail_stage2_apply",
            idempotency_key=f"{config.source_ref}:{run_report['input_hash']}",
            input_hash=run_report["input_hash"],
            metadata={
                "identity_mode": "fresh_relink_bacdd96f",
                "backup_manifest_path": str(Path(backup_manifest_path)),
            },
            actor="mail_stage2_ingest_procedure",
        )
        with store.bulk_write():
            for plan in selected:
                if plan.customer is not None:
                    store.upsert_customer(plan.customer, actor="mail_stage2_ingest_procedure", ingestion_run_id=run.run_id)
                for link in plan.identity_links:
                    store.upsert_identity_link(
                        link,
                        actor="mail_stage2_ingest_procedure",
                        ingestion_run_id=run.run_id,
                    )
                if plan.opportunity is not None:
                    store.upsert_opportunity(
                        plan.opportunity,
                        actor="mail_stage2_ingest_procedure",
                        ingestion_run_id=run.run_id,
                    )
                event_result = store.upsert_event(
                    plan.event,
                    actor="mail_stage2_ingest_procedure",
                    ingestion_run_id=run.run_id,
                )
                if event_result.created:
                    run_report["counts"]["created_events"] += 1
                if plan.event.metadata.get("pending_attribution"):
                    run_report["counts"]["pending_attribution_events"] += 1
                if plan.chunk is not None:
                    chunk_result = store.upsert_bot_context_chunk(
                        plan.chunk,
                        actor="mail_stage2_ingest_procedure",
                        ingestion_run_id=run.run_id,
                    )
                    if chunk_result.created:
                        run_report["counts"]["created_chunks"] += 1
        store.finish_ingestion_run(
            run.run_id,
            status="completed",
            accepted_count=run_report["counts"]["created_events"],
            rejected_count=run_report["counts"]["skipped_existing_events"] + run_report["counts"]["duplicate_events_in_input"],
            output_ref=str(config.out_dir / "apply_report.json"),
            metadata={"counts": dict(run_report["counts"])},
            actor="mail_stage2_ingest_procedure",
        )
    finally:
        store.close()
    write_json(config.out_dir / "apply_report.json", run_report)
    return run_report


def restore_timeline_backup(config: MailStage2IngestConfig, *, backup_manifest_path: Path) -> Mapping[str, Any]:
    manifest = validate_backup_manifest(config, backup_manifest_path)
    backup_db = Path(str(manifest["backup_db_path"]))
    target = config.timeline_db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(target) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    shutil.copy2(backup_db, target)
    report = {
        "schema_version": MAIL_STAGE2_TIMELINE_INGEST_SCHEMA_VERSION,
        "mode": "restore",
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "timeline_db_path": str(target),
        "backup_manifest_path": str(Path(backup_manifest_path)),
        "restored_sha256": file_sha256(target),
    }
    write_json(config.out_dir / "restore_report.json", report)
    return report
