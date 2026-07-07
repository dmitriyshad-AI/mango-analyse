from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.a2_mail_ingest import (
    A2V3_MAIL_SOURCE_SYSTEM,
    _chunk_text,
)
from mango_mvp.customer_timeline.canonical_readonly_import import infer_offline_brand
from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    IdentityMatchClass,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.ids import (
    normalize_email,
    normalize_identity_value,
    normalize_key,
    stable_digest,
)
from mango_mvp.customer_timeline.ingestion import compact_text
from mango_mvp.customer_timeline.safety import (
    guard_customer_timeline_output_path,
    is_customer_timeline_prod_path,
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    parse_datetime,
)

MAIL_LINK_ENRICH_SCHEMA_VERSION = "mail_link_enrich_v1"
MAIL_LINK_ENRICH_SOURCE_REF = "mail_link_enrich:stage2_pending"
PHONE_LINK_TYPES = ("phone", "primary_phone", "mango_client_phone", "whatsapp_phone")
EMAIL_LINK_TYPES = ("email", "primary_email")
OWN_DOMAINS = {"kmipt.ru", "cdpofoton.ru", "foton.school", "amocrm.ru", "amocrm.com"}
OWN_EMAILS = {"edu@kmipt.ru"}
HOTLINE_PHONE_DIGITS = {"88000000000", "88005553535", "74951234567"}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+7|7|8)\s*(?:\(?\d{3,4}\)?[\s.-]*)\d{2,3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)"
)
Participant = tuple[str, str, str]


@dataclass(frozen=True)
class MailLinkEnrichConfig:
    timeline_db: Path
    allowed_root: Path
    out_dir: Path
    tenant_id: str = "foton"
    apply: bool = False
    max_events: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db", Path(self.timeline_db).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        if self.max_events is not None and self.max_events < 0:
            raise ValueError("max_events must not be negative")


@dataclass(frozen=True)
class LinkDecision:
    outcome: str
    reason: str
    customer_id: Optional[str] = None
    method: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_source: Optional[str] = None
    candidate_customer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContactResult:
    contact_email: str | None
    contact_phone: str | None
    contact_name: str | None
    contact_source: str | None
    contact_missing: bool
    contact_ambiguous: bool
    contact_reason: str
    external_recipient_count: int


def run_mail_link_enrich(config: MailLinkEnrichConfig) -> Mapping[str, Any]:
    timeline_db, allowed_root = _validated_paths(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    before = _snapshot_counts(timeline_db)
    targets = _load_target_events(timeline_db, tenant_id=config.tenant_id, max_events=config.max_events)
    archive_cache: dict[Path, sqlite3.Connection] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter(target_events=len(targets))
    try:
        with sqlite3.connect(f"file:{timeline_db}?mode=ro", uri=True) as ro:
            ro.row_factory = sqlite3.Row
            ro.execute("PRAGMA query_only=ON")
            for row in targets:
                decision = _plan_event(row, ro, archive_cache)
                decisions.append(_decision_report(row, decision))
                counters[f"planned.{decision.outcome}"] += 1
                if decision.reason:
                    counters[f"reason.{decision.reason}"] += 1
    finally:
        for con in archive_cache.values():
            con.close()

    apply_report: Mapping[str, Any] = {}
    if config.apply:
        apply_report = _apply_decisions(
            timeline_db,
            allowed_root=allowed_root,
            tenant_id=config.tenant_id,
            targets=targets,
            decisions=decisions,
            out_dir=config.out_dir,
        )
    after = _snapshot_counts(timeline_db)
    report = {
        "schema_version": MAIL_LINK_ENRICH_SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run",
        "timeline_db": str(timeline_db),
        "allowed_root": str(allowed_root),
        "target_events": len(targets),
        "counts": dict(counters),
        "apply": apply_report,
        "before": before,
        "after": after,
        "safety": {
            "writes_prod_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "sends_messages": False,
            "allowed_for_bot_before": before["allowed_for_bot_total"],
            "allowed_for_bot_after": after["allowed_for_bot_total"],
            "allowed_for_bot_changed": before["allowed_for_bot_total"] != after["allowed_for_bot_total"],
            "mail_stage2_allowed_for_bot_before": before["mail_stage2_allowed_for_bot"],
            "mail_stage2_allowed_for_bot_after": after["mail_stage2_allowed_for_bot"],
            "mail_stage2_allowed_for_bot_changed": before["mail_stage2_allowed_for_bot"] != after["mail_stage2_allowed_for_bot"],
        },
    }
    (config.out_dir / "mail_link_enrich_decisions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in decisions) + ("\n" if decisions else ""),
        encoding="utf-8",
    )
    _write_json(config.out_dir / ("mail_link_enrich_apply_report.json" if config.apply else "mail_link_enrich_dry_run_report.json"), report)
    return report


def _validated_paths(config: MailLinkEnrichConfig) -> tuple[Path, Path]:
    allowed_root = Path(config.allowed_root).expanduser().resolve(strict=False)
    timeline_db = guard_customer_timeline_output_path(config.timeline_db, allowed_root)
    if is_customer_timeline_prod_path(timeline_db):
        raise ValueError(f"mail_link_enrich refuses prod DB path: {timeline_db}")
    guard_customer_timeline_output_path(config.out_dir, allowed_root)
    return timeline_db, allowed_root


def _load_target_events(db_path: Path, *, tenant_id: str, max_events: Optional[int]) -> list[sqlite3.Row]:
    sql = """
        SELECT *
        FROM timeline_events
        WHERE tenant_id = ?
          AND source_system = ?
          AND match_status = 'unmatched'
          AND (customer_id IS NULL OR customer_id = '')
          AND json_extract(record_json, '$.metadata.pending_attribution') = 1
          AND json_extract(record_json, '$.metadata.pending_reason') IS NULL
        ORDER BY event_at, event_id
    """
    params: list[Any] = [tenant_id, A2V3_MAIL_SOURCE_SYSTEM]
    if max_events is not None:
        sql += " LIMIT ?"
        params.append(max_events)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        return list(con.execute(sql, params))


def _plan_event(
    row: sqlite3.Row,
    timeline_ro: sqlite3.Connection,
    archive_cache: dict[Path, sqlite3.Connection],
) -> LinkDecision:
    payload = _event_payload(row)
    source_payload = _source_payload(payload)
    archive_db = _archive_db_for_source_payload(source_payload)
    raw_message = _load_archive_message(archive_db, str(row["source_id"]), archive_cache) if archive_db else {}
    contact = _contact_from_archive_row(str(row["direction"]), raw_message)
    email = normalize_email(contact.contact_email)
    phone = _normalize_phone(contact.contact_phone)
    if phone:
        phone_resolution = _resolve_identity_value(
            timeline_ro,
            tenant_id=str(row["tenant_id"]),
            link_types=PHONE_LINK_TYPES,
            link_value=phone,
        )
        if phone_resolution["status"] == "strong":
            return LinkDecision(
                "strong",
                "strong_phone_identity_link",
                customer_id=phone_resolution["customer_id"],
                method="phone_identity_link",
                contact_email=email or None,
                contact_phone=phone,
                contact_source=contact.contact_source,
                candidate_customer_ids=tuple(phone_resolution["candidate_customer_ids"]),
            )
        if phone_resolution["status"] in {"ambiguous", "blocked"}:
            return LinkDecision(
                "blocked",
                f"phone_{phone_resolution['reason']}",
                contact_email=email or None,
                contact_phone=phone,
                contact_source=contact.contact_source,
                candidate_customer_ids=tuple(phone_resolution["candidate_customer_ids"]),
            )
    if email:
        email_resolution = _resolve_identity_value(
            timeline_ro,
            tenant_id=str(row["tenant_id"]),
            link_types=EMAIL_LINK_TYPES,
            link_value=email,
        )
        if email_resolution["status"] in {"strong", "ambiguous", "blocked"}:
            return LinkDecision(
                "weak_email",
                f"email_{email_resolution['reason']}",
                customer_id=None,
                method="email_identity_link_weak",
                contact_email=email,
                contact_phone=phone or None,
                contact_source=contact.contact_source,
                candidate_customer_ids=tuple(email_resolution["candidate_customer_ids"]),
            )
    if contact.contact_ambiguous:
        return LinkDecision("blocked", contact.contact_reason, contact_source=contact.contact_source)
    if contact.contact_missing:
        return LinkDecision("unmatched", contact.contact_reason)
    return LinkDecision(
        "unmatched",
        "no_strong_identity_match",
        contact_email=email or None,
        contact_phone=phone or None,
        contact_source=contact.contact_source,
    )


def _event_payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        parsed = json.loads(str(row["record_json"] or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _source_payload(event_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    record = event_payload.get("record")
    if isinstance(record, Mapping):
        payload = record.get("payload")
        if isinstance(payload, Mapping):
            return payload
    return {}


def _archive_db_for_source_payload(source_payload: Mapping[str, Any]) -> Optional[Path]:
    raw_db = str(source_payload.get("source_db") or "").strip()
    if raw_db:
        db_path = Path(raw_db).expanduser()
        if db_path.exists() and db_path.is_file():
            return db_path
    raw = str(source_payload.get("source_file") or "").strip()
    if not raw:
        return None
    source_path = Path(raw).expanduser()
    for parent in (source_path.parent, *source_path.parents):
        candidate = parent / "archive" / "mail_archive.sqlite"
        if candidate.exists():
            return candidate
    return None


def _load_archive_message(
    archive_db: Path,
    message_sha: str,
    archive_cache: dict[Path, sqlite3.Connection],
) -> Mapping[str, Any]:
    db = archive_db.resolve(strict=False)
    con = archive_cache.get(db)
    if con is None:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        archive_cache[db] = con
    msg = con.execute("SELECT * FROM messages WHERE sha256 = ?", (message_sha,)).fetchone()
    if msg is None:
        return {}
    participants = [
        dict(row)
        for row in con.execute(
            """
            SELECT header_name, display_name, email_normalized, domain
            FROM message_participants
            WHERE message_sha256 = ?
            """,
            (message_sha,),
        )
    ]
    text = ""
    extracted = Path(str(msg["extracted_text_path"] or ""))
    if not extracted.is_absolute():
        extracted = db.parent / extracted
    if extracted.exists() and extracted.is_file():
        text = extracted.read_text(encoding="utf-8", errors="ignore")[:65536]
    return {"message": dict(msg), "participants": participants, "text": text}


def _contact_from_archive_row(direction: str, raw_message: Mapping[str, Any]) -> ContactResult:
    participants = raw_message.get("participants") if isinstance(raw_message.get("participants"), Sequence) else []

    def tup(item: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(item.get("display_name") or ""),
            str(item.get("email_normalized") or ""),
            str(item.get("domain") or ""),
        )

    from_participant = None
    to_participants: list[tuple[str, str, str]] = []
    cc_participants: list[tuple[str, str, str]] = []
    for item in participants:
        if not isinstance(item, Mapping):
            continue
        header = str(item.get("header_name") or "").strip().lower()
        if header == "from":
            from_participant = tup(item)
        elif header == "to":
            to_participants.append(tup(item))
        elif header == "cc":
            cc_participants.append(tup(item))
    return _resolve_customer_contact(
        direction=str(direction or ""),
        from_participant=from_participant,
        to_participants=to_participants,
        cc_participants=cc_participants,
        raw_text=_signature_text(str(raw_message.get("text") or "")),
    )


def _resolve_customer_contact(
    *,
    direction: str,
    from_participant: Participant | None,
    to_participants: Sequence[Participant],
    cc_participants: Sequence[Participant],
    raw_text: str = "",
) -> ContactResult:
    to_external = [_clean_participant(item) for item in to_participants if _is_external_participant(item)]
    cc_external = [_clean_participant(item) for item in cc_participants if _is_external_participant(item)]
    external_recipients = [item for item in [*to_external, *cc_external] if item and item[1]]
    quoted_emails = {email.casefold() for email in EMAIL_RE.findall(raw_text or "")}

    if direction == "inbound":
        from_clean = _clean_participant(from_participant)
        if from_clean and _is_external_participant(from_clean):
            phone = _single_external_phone(raw_text) if raw_text else None
            return _contact_result(
                email=from_clean[1],
                phone=phone,
                name=from_clean[0],
                source="header_from",
                reason="inbound_external_from",
                external_count=len(external_recipients),
            )
        quoted = _quoted_match([*to_external, *cc_external], quoted_emails)
        if quoted:
            return _contact_result(
                email=quoted[1],
                phone=None,
                name=quoted[0],
                source="quoted_header",
                reason="quoted_email_matches_envelope",
                external_count=len(external_recipients),
            )
        return _contact_missing("inbound_no_external_from", len(external_recipients))

    if len(external_recipients) > 1:
        return ContactResult(None, None, None, None, False, True, "multiple_external_recipients", len(external_recipients))
    if len(external_recipients) == 1:
        participant = external_recipients[0]
        source = "header_to" if participant in to_external else "header_cc"
        return _contact_result(
            email=participant[1],
            phone=None,
            name=participant[0],
            source=source,
            reason="outbound_single_external_recipient",
            external_count=1,
        )
    quoted = _quoted_match([_clean_participant(from_participant)], quoted_emails)
    if quoted:
        return _contact_result(
            email=quoted[1],
            phone=None,
            name=quoted[0],
            source="quoted_header",
            reason="quoted_email_matches_envelope",
            external_count=0,
        )
    return _contact_missing("outbound_no_external_recipient", 0)


def _clean_participant(value: Participant | None) -> Participant | None:
    if not value:
        return None
    name, email, domain = value
    clean_email = str(email or "").strip().lower()
    clean_domain = str(domain or _domain_of(clean_email)).strip().lower()
    return (str(name or "").strip(), clean_email, clean_domain)


def _domain_of(email: str) -> str:
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[-1].strip().lower()


def _is_external_participant(value: Participant | None) -> bool:
    participant = _clean_participant(value)
    if not participant:
        return False
    _, email, domain = participant
    return bool(email) and email not in OWN_EMAILS and domain not in OWN_DOMAINS


def _quoted_match(candidates: Sequence[Participant | None], quoted_emails: set[str]) -> Participant | None:
    for candidate in candidates:
        participant = _clean_participant(candidate)
        if participant and participant[1].casefold() in quoted_emails:
            return participant
    return None


def _single_external_phone(text: str) -> str | None:
    phones: list[str] = []
    seen: set[str] = set()
    for raw in PHONE_RE.findall(text or ""):
        digits = re.sub(r"\D", "", raw)
        if digits.startswith("8"):
            digits = "7" + digits[1:]
        if digits.startswith("7") and len(digits) == 11 and digits not in HOTLINE_PHONE_DIGITS and digits not in seen:
            seen.add(digits)
            phones.append("+" + digits)
    return phones[0] if len(phones) == 1 else None


def _contact_result(
    *,
    email: str | None,
    phone: str | None,
    name: str | None,
    source: str,
    reason: str,
    external_count: int,
) -> ContactResult:
    return ContactResult(
        contact_email=email or None,
        contact_phone=phone or None,
        contact_name=name or None,
        contact_source=source,
        contact_missing=False,
        contact_ambiguous=False,
        contact_reason=reason,
        external_recipient_count=external_count,
    )


def _contact_missing(reason: str, external_count: int) -> ContactResult:
    return ContactResult(None, None, None, None, True, False, reason, external_count)


def _signature_text(text: str) -> str:
    """Keep only the plausible signature tail for phone extraction.

    The contact resolver may treat a single phone in raw_text as a contact
    signal. For mail-link enrich this must never come from the business body
    or quoted history, so we pass only a short non-quoted footer.
    """
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue
        if line.startswith(">"):
            continue
        if lower.startswith(("от:", "from:", "sent:", "to:", "кому:", "subject:", "тема:")):
            continue
        if "исходное сообщение" in lower or "original message" in lower:
            break
        lines.append(line)
    return "\n".join(lines[-12:])[:2000]


def _normalize_phone(value: Any) -> str:
    if not value:
        return ""
    try:
        return normalize_identity_value("phone", value)
    except ValueError:
        return ""


def _resolve_identity_value(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    link_types: Sequence[str],
    link_value: str,
) -> Mapping[str, Any]:
    if not link_value:
        return {"status": "unmatched", "reason": "missing_value", "candidate_customer_ids": ()}
    placeholders = ",".join("?" for _ in link_types)
    rows = con.execute(
        f"""
        SELECT l.customer_id, l.match_class, c.identity_status
        FROM identity_links l
        LEFT JOIN customer_identities c
          ON c.tenant_id = l.tenant_id AND c.customer_id = l.customer_id
        WHERE l.tenant_id = ?
          AND l.link_type IN ({placeholders})
          AND l.link_value = ?
          AND l.customer_id IS NOT NULL
          AND l.customer_id != ''
        """,
        (tenant_id, *link_types, link_value),
    ).fetchall()
    customer_ids = sorted({str(row["customer_id"]) for row in rows if row["customer_id"]})
    if not customer_ids:
        return {"status": "unmatched", "reason": "no_identity_link", "candidate_customer_ids": ()}
    if len(customer_ids) != 1:
        return {"status": "ambiguous", "reason": "multiple_customers", "candidate_customer_ids": tuple(customer_ids)}
    if any(str(row["match_class"]) == IdentityMatchClass.AMBIGUOUS.value for row in rows):
        return {"status": "ambiguous", "reason": "ambiguous_identity_link", "candidate_customer_ids": tuple(customer_ids)}
    if any(str(row["identity_status"] or "").lower() == "ambiguous" for row in rows):
        return {"status": "blocked", "reason": "customer_identity_ambiguous", "candidate_customer_ids": tuple(customer_ids)}
    return {
        "status": "strong",
        "reason": "unique_identity_link",
        "customer_id": customer_ids[0],
        "candidate_customer_ids": tuple(customer_ids),
    }


def _decision_report(row: sqlite3.Row, decision: LinkDecision) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "message_sha256": str(row["source_id"]),
        "outcome": decision.outcome,
        "reason": decision.reason,
        "method": decision.method,
        "customer_id": decision.customer_id,
        "contact_email_hash": _hash_optional(decision.contact_email),
        "contact_phone_hash": _hash_optional(decision.contact_phone),
        "contact_source": decision.contact_source,
        "candidate_customer_count": len(decision.candidate_customer_ids),
    }


def _apply_decisions(
    db_path: Path,
    *,
    allowed_root: Path,
    tenant_id: str,
    targets: Sequence[sqlite3.Row],
    decisions: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> Mapping[str, Any]:
    by_event = {str(row["event_id"]): row for row in targets}
    counters: Counter[str] = Counter()
    changed_event_ids: list[str] = []
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        run = store.start_ingestion_run(
            tenant_id=tenant_id,
            source_system=A2V3_MAIL_SOURCE_SYSTEM,
            source_ref=MAIL_LINK_ENRICH_SOURCE_REF,
            run_kind="mail_link_enrich",
            idempotency_key=stable_digest(
                {
                    "schema_version": MAIL_LINK_ENRICH_SCHEMA_VERSION,
                    "events": [item["event_id"] for item in decisions],
                    "decisions": [
                        {
                            "event_id": item["event_id"],
                            "outcome": item["outcome"],
                            "reason": item["reason"],
                            "customer_id": item.get("customer_id"),
                        }
                        for item in decisions
                    ],
                }
            ),
            input_hash=stable_digest({"target_event_ids": [str(row["event_id"]) for row in targets]}),
            metadata={"target_events": len(targets)},
            actor="mail_link_enrich",
        )
        with store.bulk_write():
            for decision in decisions:
                row = by_event[str(decision["event_id"])]
                event = _updated_event_from_decision(row, decision)
                before_hash = str(row["record_hash"])
                result = store.upsert_event(event, actor="mail_link_enrich", ingestion_run_id=run.run_id)
                if result.record_hash != before_hash:
                    counters["updated_events"] += 1
                    changed_event_ids.append(event.event_id)
                else:
                    counters["unchanged_events"] += 1
                if decision["outcome"] == "strong" and not _event_has_chunk(store._con, event.event_id):
                    chunk = _chunk_for_event(row, event, decision)
                    if chunk is not None:
                        chunk_result = store.upsert_bot_context_chunk(
                            chunk,
                            actor="mail_link_enrich",
                            ingestion_run_id=run.run_id,
                        )
                        if chunk_result.created:
                            counters["created_chunks"] += 1
                _upsert_fact_for_decision(store._con, event, decision)
        store.finish_ingestion_run(
            run.run_id,
            status="completed",
            accepted_count=int(counters["updated_events"] + counters["created_chunks"]),
            rejected_count=0,
            output_ref=str(out_dir / "mail_link_enrich_apply_report.json"),
            metadata={"counts": dict(counters), "changed_event_ids": changed_event_ids[:100]},
            actor="mail_link_enrich",
        )
    return {"counts": dict(counters), "changed_event_ids_sample": changed_event_ids[:20]}


def _updated_event_from_decision(row: sqlite3.Row, decision: Mapping[str, Any]) -> TimelineEvent:
    payload = _event_payload(row)
    record = dict(payload.get("record") or {})
    metadata = dict(payload.get("metadata") or {})
    source_payload = dict(_source_payload(payload))
    brand = _brand_for_event(row, source_payload=source_payload)
    source_payload["brand"] = brand
    source_payload["contact_email_hash"] = decision.get("contact_email_hash")
    source_payload["contact_phone_hash"] = decision.get("contact_phone_hash")
    source_payload["mail_link_enrich_outcome"] = decision["outcome"]
    source_payload["mail_link_enrich_reason"] = decision["reason"]
    if decision["outcome"] == "strong":
        source_payload["customer_id"] = decision.get("customer_id")
    record["payload"] = source_payload
    metadata["brand"] = brand
    metadata["mail_link_enrich"] = {
        "schema_version": MAIL_LINK_ENRICH_SCHEMA_VERSION,
        "outcome": decision["outcome"],
        "reason": decision["reason"],
        "method": decision.get("method"),
        "contact_source": decision.get("contact_source"),
        "candidate_customer_count": decision.get("candidate_customer_count"),
    }
    metadata["fresh_relink"] = True
    if decision["outcome"] == "strong":
        metadata["pending_attribution"] = False
        metadata.pop("pending_reason", None)
        customer_id = str(decision["customer_id"])
        match_status = IdentityMatchClass.STRONG_UNIQUE
        confidence = 0.92
    elif decision["outcome"] == "weak_email":
        metadata["pending_attribution"] = True
        metadata["pending_reason"] = "weak_email_only"
        customer_id = None
        match_status = IdentityMatchClass.UNMATCHED
        confidence = 0.0
    elif decision["outcome"] == "blocked":
        metadata["pending_attribution"] = True
        metadata["pending_reason"] = decision["reason"]
        customer_id = None
        match_status = IdentityMatchClass.AMBIGUOUS
        confidence = 0.0
    else:
        metadata["pending_attribution"] = True
        metadata["pending_reason"] = decision["reason"]
        customer_id = None
        match_status = IdentityMatchClass.UNMATCHED
        confidence = 0.0
    return TimelineEvent(
        tenant_id=str(row["tenant_id"]),
        customer_id=customer_id,
        event_id=str(row["event_id"]),
        event_type=TimelineEventType(str(row["event_type"])),
        event_at=parse_datetime(str(row["event_at"]), "event_at"),
        source_system=str(row["source_system"]),
        source_id=str(row["source_id"]),
        source_ref=str(row["source_ref"] or row["source_id"]),
        direction=TimelineDirection(str(row["direction"])),
        match_status=match_status,
        confidence=confidence,
        importance=int(row["importance"] or 0),
        subject=row["subject"],
        text_preview=row["text_preview"],
        summary=row["summary"],
        record=record,
        metadata=metadata,
        created_at=parse_datetime(str(row["created_at"]), "created_at"),
    )


def _brand_for_event(row: sqlite3.Row, *, source_payload: Mapping[str, Any]) -> str:
    explicit = _normalize_brand(source_payload.get("brand"))
    if explicit != "unknown":
        return explicit
    text = {
        "subject": row["subject"],
        "summary": row["summary"],
        "text_preview": row["text_preview"],
        "full_clean_text": source_payload.get("full_clean_text"),
    }
    return _normalize_brand(infer_offline_brand(text))


def _normalize_brand(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "мфти"}:
        return "unpk"
    return "unknown"


def _event_has_chunk(con: sqlite3.Connection, event_id: str) -> bool:
    return bool(
        con.execute(
            "SELECT 1 FROM bot_context_chunks WHERE event_id = ? AND source_system = ? LIMIT 1",
            (event_id, A2V3_MAIL_SOURCE_SYSTEM),
        ).fetchone()
    )


def _chunk_for_event(row: sqlite3.Row, event: TimelineEvent, decision: Mapping[str, Any]) -> Optional[BotContextChunk]:
    source_payload = _source_payload(_event_payload(row))
    row_for_chunk = {
        "full_clean_text": source_payload.get("full_clean_text") or "",
        "summary_payload": {"summary": row["summary"]},
        "summary": row["summary"],
        "brand": event.metadata.get("brand"),
        "message_sha256": row["source_id"],
        "quality": {"memory_status": "needs_summary_later"},
    }
    text, overflow = _chunk_text(row_for_chunk, rich=True)
    if not text:
        text = compact_text(row["summary"] or row["text_preview"] or row["subject"], limit=1200) or ""
    if not text:
        return None
    return BotContextChunk(
        tenant_id=event.tenant_id,
        customer_id=str(decision["customer_id"]),
        event_id=event.event_id,
        source_system=A2V3_MAIL_SOURCE_SYSTEM,
        source_ref=event.source_ref,
        chunk_type="email_message",
        text=text,
        summary=compact_text(row["summary"], limit=500),
        event_at=event.event_at,
        freshness_score=0.7,
        relevance_tags=("email", "mail_archive_stage2", "manager_only", str(event.metadata.get("brand") or "unknown")),
        allowed_for_bot=False,
        requires_manager_review=True,
        metadata={
            "message_sha256": row["source_id"],
            "mail_link_enrich": True,
            "thread_context_overflow": overflow,
            "bot_eligible_candidate": False,
            "bot_gate_reason": "manager_review_until_owner_opening",
        },
        created_at=event.created_at,
    )


def _upsert_fact_for_decision(con: sqlite3.Connection, event: TimelineEvent, decision: Mapping[str, Any]) -> None:
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
    ).fetchone():
        return
    existing = con.execute(
        "SELECT message_sha256 FROM a2v3_mail_event_facts WHERE message_sha256 = ?",
        (event.source_id,),
    ).fetchone()
    row = {
        "event_id": event.event_id,
        "tenant_id": event.tenant_id,
        "customer_id": event.customer_id,
        "message_sha256": event.source_id,
        "event_type_detail": "other",
        "money_direction": None,
        "amount_kind": None,
        "amount_rub": None,
        "money_amounts_rub_json": "[]",
        "amount_uncertain": 0,
        "email_brand": event.metadata.get("brand") or "unknown",
        "email_brand_source": "mail_link_enrich_infer",
        "customer_brand": event.metadata.get("brand") or "unknown",
        "customer_brand_source": "mail_link_enrich_event_brand",
        "customer_brand_reason": "event_level_brand_only",
        "contact_email": None,
        "contact_phone": None,
        "contact_name": None,
        "student_name": None,
        "grade": None,
        "subject_area": None,
        "memory_status": "needs_summary_later",
        "client_safe": 1,
        "client_safe_reason": "not_evaluated",
        "client_safe_policy_version": "cs_v1",
        "sensitivity_tags_json": "[]",
        "bot_visible": 0,
        "bot_gate_reason": "manager_review_until_owner_opening",
        "identity_outcome": decision["outcome"],
        "identity_reason": decision["reason"],
        "created_at": event.created_at.isoformat(),
    }
    if existing:
        con.execute(
            """
            UPDATE a2v3_mail_event_facts
            SET event_id=:event_id, customer_id=:customer_id, email_brand=:email_brand,
                customer_brand=:customer_brand, customer_brand_source=:customer_brand_source,
                customer_brand_reason=:customer_brand_reason, bot_visible=0,
                bot_gate_reason=:bot_gate_reason, identity_outcome=:identity_outcome,
                identity_reason=:identity_reason
            WHERE message_sha256=:message_sha256
            """,
            row,
        )
    else:
        con.execute(
            """
            INSERT INTO a2v3_mail_event_facts (
              event_id, tenant_id, customer_id, message_sha256, event_type_detail,
              money_direction, amount_kind, amount_rub, money_amounts_rub_json,
              amount_uncertain, email_brand, email_brand_source, customer_brand,
              customer_brand_source, customer_brand_reason, contact_email, contact_phone,
              contact_name, student_name, grade, subject_area, memory_status,
              client_safe, client_safe_reason, client_safe_policy_version, sensitivity_tags_json,
              bot_visible, bot_gate_reason, identity_outcome, identity_reason, created_at
            )
            VALUES (
              :event_id, :tenant_id, :customer_id, :message_sha256, :event_type_detail,
              :money_direction, :amount_kind, :amount_rub, :money_amounts_rub_json,
              :amount_uncertain, :email_brand, :email_brand_source, :customer_brand,
              :customer_brand_source, :customer_brand_reason, :contact_email, :contact_phone,
              :contact_name, :student_name, :grade, :subject_area, :memory_status,
              :client_safe, :client_safe_reason, :client_safe_policy_version, :sensitivity_tags_json,
              :bot_visible, :bot_gate_reason, :identity_outcome, :identity_reason, :created_at
            )
            """,
            row,
        )


def _snapshot_counts(db_path: Path) -> Mapping[str, Any]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        return {
            "timeline_events": _count(con, "timeline_events"),
            "bot_context_chunks": _count(con, "bot_context_chunks"),
            "allowed_for_bot_total": int(con.execute("SELECT count(*) FROM bot_context_chunks WHERE allowed_for_bot=1").fetchone()[0]),
            "mail_stage2_allowed_for_bot": int(
                con.execute(
                    "SELECT count(*) FROM bot_context_chunks WHERE source_system=? AND allowed_for_bot=1",
                    (A2V3_MAIL_SOURCE_SYSTEM,),
                ).fetchone()[0]
            ),
            "pending_null_reason": int(
                con.execute(
                    """
                    SELECT count(*)
                    FROM timeline_events
                    WHERE source_system = ?
                      AND match_status = 'unmatched'
                      AND (customer_id IS NULL OR customer_id = '')
                      AND json_extract(record_json, '$.metadata.pending_attribution') = 1
                      AND json_extract(record_json, '$.metadata.pending_reason') IS NULL
                    """,
                    (A2V3_MAIL_SOURCE_SYSTEM,),
                ).fetchone()[0]
            ),
        }


def _count(con: sqlite3.Connection, table: str) -> int:
    try:
        return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def _hash_optional(value: Optional[str]) -> Optional[str]:
    return stable_digest({"value": value})[:16] if value else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


__all__ = [
    "MAIL_LINK_ENRICH_SCHEMA_VERSION",
    "MailLinkEnrichConfig",
    "run_mail_link_enrich",
]
