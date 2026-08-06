from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email import policy
from email.parser import BytesHeaderParser
from functools import lru_cache
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
    open_family_identity_conflict_customer_ids,
    parse_datetime,
)

MAIL_LINK_ENRICH_SCHEMA_VERSION = "mail_link_enrich_v1"
MAIL_LINK_ENRICH_SOURCE_REF = "mail_link_enrich:stage2_pending"
PHONE_LINK_TYPES = ("phone", "primary_phone", "mango_client_phone", "whatsapp_phone")
EMAIL_LINK_TYPES = ("email", "primary_email")
MAIL_IDENTITY_SOURCE_SYSTEMS = {"mail_archive", A2V3_MAIL_SOURCE_SYSTEM}
TRUSTED_EMAIL_IDENTITY_SOURCES = {
    "tallanto_snapshot",
    "tallanto_historical_snapshot",
    "amocrm_snapshot",
    "master_contacts_snapshot",
}
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
    reconsider_pending: bool = False
    revalidate_existing_strong: bool = False
    fallback_archive_dbs: tuple[Path, ...] = ()
    tallanto_identity_dbs: tuple[Path, ...] = ()
    aggregate_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db", Path(self.timeline_db).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        object.__setattr__(
            self,
            "fallback_archive_dbs",
            tuple(Path(path).expanduser() for path in self.fallback_archive_dbs),
        )
        object.__setattr__(
            self,
            "tallanto_identity_dbs",
            tuple(Path(path).expanduser() for path in self.tallanto_identity_dbs),
        )
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
    brand_context_authorized: bool = False
    customer_brand: str = "unknown"
    family_id: str = ""
    # D2 rule 7: which trusted source produced a "strong"/"family_strong"
    # decision (e.g. "amocrm_snapshot", "tallanto_snapshot",
    # "tallanto_historical_snapshot") -- used only to build the
    # exact_amo/exact_tallanto/thread_propagation/ambiguous/unmatched
    # breakdown in the report, never persisted onto the event itself.
    source_system_hint: str = ""


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
    config.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.out_dir.chmod(0o700)
    before = _snapshot_counts(timeline_db)
    targets = _load_target_events(
        timeline_db,
        tenant_id=config.tenant_id,
        max_events=config.max_events,
        reconsider_pending=config.reconsider_pending,
        revalidate_existing_strong=config.revalidate_existing_strong,
    )
    archive_cache: dict[Path, sqlite3.Connection] = {}
    participant_cache: dict[Path, Mapping[str, list[Mapping[str, Any]]]] = {}
    message_id_cache: dict[Path, Mapping[str, tuple[str, ...]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter(target_events=len(targets))
    historical_identity_summary: Mapping[str, Any] = {"databases": 0, "usable_email_values": 0}
    try:
        with sqlite3.connect(f"file:{timeline_db}?mode=ro", uri=True) as ro:
            ro.row_factory = sqlite3.Row
            ro.execute("PRAGMA query_only=ON")
            identity_index = _load_identity_index(ro, tenant_id=config.tenant_id)
            family_ids = _load_persisted_family_ids(ro, tenant_id=config.tenant_id)
            historical_email_index, historical_identity_summary = _load_historical_tallanto_email_index(
                ro,
                tenant_id=config.tenant_id,
                identity_dbs=config.tallanto_identity_dbs,
                family_ids=family_ids,
            )
            customer_brands = _load_trusted_customer_brands(ro, tenant_id=config.tenant_id)
            for row in targets:
                decision = _plan_event(
                    row,
                    ro,
                    archive_cache,
                    participant_cache,
                    identity_index,
                    message_id_cache,
                    family_ids=family_ids,
                    historical_email_index=historical_email_index,
                    fallback_archive_dbs=config.fallback_archive_dbs,
                )
                decision = _block_existing_customer_conflict(row, decision)
                decision = _preserve_existing_strong_without_explicit_conflict(row, decision)
                decision = _annotate_brand_authorization(row, decision, customer_brands)
                decisions.append(_decision_report(row, decision))
                counters[f"planned.{decision.outcome}"] += 1
                counters[f"transition.{str(row['match_status'])}.{decision.outcome}"] += 1
                counters[
                    f"transition_reason.{str(row['match_status'])}.{decision.outcome}.{decision.reason or 'none'}"
                ] += 1
                existing_customer_id = str(row["customer_id"] or "")
                if existing_customer_id and decision.outcome != "strong":
                    relation = (
                        "existing_in_candidates"
                        if existing_customer_id in set(decision.candidate_customer_ids)
                        else "existing_not_in_candidates"
                    )
                    counters[f"transition_relation.{decision.outcome}.{decision.reason}.{relation}"] += 1
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
            aggregate_only=config.aggregate_only,
        )
    after = _snapshot_counts(timeline_db)
    report = {
        "schema_version": MAIL_LINK_ENRICH_SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run",
        "timeline_db": str(timeline_db),
        "allowed_root": str(allowed_root),
        "target_events": len(targets),
        "reconsider_pending": config.reconsider_pending,
        "revalidate_existing_strong": config.revalidate_existing_strong,
        "fallback_archive_db_count": len(config.fallback_archive_dbs),
        "historical_tallanto_identity": historical_identity_summary,
        "counts": dict(counters),
        # D2 rule 7: exact AMO / exact Tallanto / thread propagation /
        # ambiguous / unmatched breakdown, separate from the much more
        # granular `counts.reason.*`/`counts.planned.*` keys above.
        "breakdown": _mail_link_enrich_breakdown(decisions),
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
    if not config.aggregate_only:
        decisions_path = config.out_dir / "mail_link_enrich_decisions.jsonl"
        decisions_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in decisions) + ("\n" if decisions else ""),
            encoding="utf-8",
        )
        decisions_path.chmod(0o600)
    report_path = config.out_dir / ("mail_link_enrich_apply_report.json" if config.apply else "mail_link_enrich_dry_run_report.json")
    _write_json(report_path, report)
    report_path.chmod(0o600)
    return report


def _validated_paths(config: MailLinkEnrichConfig) -> tuple[Path, Path]:
    allowed_root = Path(config.allowed_root).expanduser().resolve(strict=False)
    timeline_db = guard_customer_timeline_output_path(config.timeline_db, allowed_root)
    if is_customer_timeline_prod_path(timeline_db):
        raise ValueError(f"mail_link_enrich refuses prod DB path: {timeline_db}")
    guard_customer_timeline_output_path(config.out_dir, allowed_root)
    return timeline_db, allowed_root


def _load_target_events(
    db_path: Path,
    *,
    tenant_id: str,
    max_events: Optional[int],
    reconsider_pending: bool,
    revalidate_existing_strong: bool,
) -> list[sqlite3.Row]:
    sql = """
        SELECT *
        FROM timeline_events
        WHERE tenant_id = ?
          AND source_system = ?
          AND (
            (
              (match_status = 'unmatched' OR (? = 1 AND match_status = 'ambiguous'))
              AND (customer_id IS NULL OR customer_id = '')
              AND json_extract(record_json, '$.metadata.pending_attribution') = 1
              AND (
                json_extract(record_json, '$.metadata.pending_reason') IS NULL
                OR ? = 1
              )
            )
            OR (
              ? = 1
              AND match_status = 'strong_unique'
              AND customer_id IS NOT NULL AND customer_id != ''
            )
            OR (
              (? = 1 OR ? = 1)
              AND match_status = 'ambiguous'
              AND (customer_id IS NULL OR customer_id = '')
              AND json_extract(record_json, '$.metadata.mail_link_enrich.outcome') = 'family_strong'
            )
          )
        ORDER BY event_at, event_id
    """
    params: list[Any] = [
        tenant_id,
        A2V3_MAIL_SOURCE_SYSTEM,
        int(reconsider_pending),
        int(reconsider_pending),
        int(revalidate_existing_strong),
        int(revalidate_existing_strong),
        int(reconsider_pending),
    ]
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
    participant_cache: dict[Path, Mapping[str, list[Mapping[str, Any]]]],
    identity_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    message_id_cache: dict[Path, Mapping[str, tuple[str, ...]]],
    *,
    fallback_archive_dbs: Sequence[Path] = (),
    family_ids: Mapping[str, str] | None = None,
    historical_email_index: Mapping[str, Sequence[str]] | None = None,
) -> LinkDecision:
    payload = _event_payload(row)
    source_payload = _source_payload(payload)
    archive_db = _archive_db_for_source_payload(source_payload)
    archive_dbs = [path for path in (archive_db, *fallback_archive_dbs) if path and path.is_file()]
    raw_message: Mapping[str, Any] = {}
    for candidate in dict.fromkeys(archive_dbs):
        raw_message = _load_archive_message(
            candidate,
            str(row["source_id"]),
            archive_cache,
            participant_cache,
        )
        if raw_message:
            break
    existing_link_outcome = str(
        ((payload.get("metadata") or {}).get("mail_link_enrich") or {}).get("outcome") or ""
    )
    if not raw_message and (
        str(row["match_status"]) == IdentityMatchClass.STRONG_UNIQUE.value
        or existing_link_outcome == "family_strong"
    ):
        return LinkDecision(
            "not_revalidated_archive_missing",
            "not_revalidated_archive_missing",
            customer_id=str(row["customer_id"] or "") or None,
        )
    contact = _contact_from_archive_row(str(row["direction"]), raw_message)
    email = normalize_email(contact.contact_email)
    phone = _normalize_phone(contact.contact_phone)
    phone_resolution = (
        _resolve_identity_value(
            timeline_ro,
            tenant_id=str(row["tenant_id"]),
            link_types=PHONE_LINK_TYPES,
            link_value=phone,
            identity_index=identity_index,
        )
        if phone
        else _empty_identity_resolution()
    )
    email_resolution = (
        _resolve_identity_value(
            timeline_ro,
            tenant_id=str(row["tenant_id"]),
            link_types=EMAIL_LINK_TYPES,
            link_value=email,
            identity_index=identity_index,
        )
        if email
        else _empty_identity_resolution()
    )
    if email:
        email_resolution = _with_historical_tallanto_email(
            email_resolution,
            historical_customer_ids=(historical_email_index or {}).get(email, ()),
        )
    phone_ids = set(phone_resolution["candidate_customer_ids"])
    email_ids = set(email_resolution["candidate_customer_ids"])
    candidates = tuple(sorted(phone_ids | email_ids))
    trusted_email_ids = set(email_resolution.get("external_strong_customer_ids", ()))
    trusted_email_sources = set(email_resolution.get("external_strong_source_systems", ()))
    trusted_email_id = str(email_resolution.get("customer_id") or "")
    if not (
        email_resolution["status"] == "strong"
        and trusted_email_ids == {trusted_email_id}
        and trusted_email_sources & TRUSTED_EMAIL_IDENTITY_SOURCES
    ):
        trusted_email_id = ""
    family_candidates = tuple(
        sorted(
            set(phone_resolution.get("trusted_family_customer_ids", ()))
            | set(email_resolution.get("trusted_family_customer_ids", ()))
        )
    )
    _family_customer_id, family_id = _family_customer_for_candidates(
        family_candidates,
        family_ids=family_ids or {},
        preferred_customer_ids=tuple(
            sorted(
                set(phone_resolution.get("amo_customer_ids", ()))
                | set(email_resolution.get("amo_customer_ids", ()))
            )
        ),
    )

    if family_id:
        return LinkDecision(
            "family_strong",
            "strong_tallanto_family_identity_link",
            method="tallanto_family_identity_link",
            contact_email=email or None,
            contact_phone=phone or None,
            contact_source=contact.contact_source,
            candidate_customer_ids=candidates,
            family_id=family_id,
        )

    if email_resolution.get("reason") == "historical_tallanto_email_multiple_customers":
        return LinkDecision(
            "blocked",
            "historical_tallanto_email_cross_family",
            contact_email=email or None,
            contact_phone=phone or None,
            contact_source=contact.contact_source,
            candidate_customer_ids=candidates,
        )

    if phone_ids and email_ids and phone_ids.isdisjoint(email_ids):
        return LinkDecision(
            "blocked",
            "phone_email_customer_conflict",
            contact_email=email or None,
            contact_phone=phone or None,
            contact_source=contact.contact_source,
            candidate_customer_ids=candidates,
        )
    if phone:
        if phone_resolution["status"] == "strong":
            if email_resolution["status"] in {"ambiguous", "blocked"}:
                return LinkDecision(
                    "blocked",
                    f"email_{email_resolution['reason']}",
                    contact_email=email or None,
                    contact_phone=phone,
                    contact_source=contact.contact_source,
                    candidate_customer_ids=candidates,
                )
            return LinkDecision(
                "strong",
                "strong_phone_identity_link",
                customer_id=phone_resolution["customer_id"],
                method="phone_identity_link",
                contact_email=email or None,
                contact_phone=phone,
                contact_source=contact.contact_source,
                candidate_customer_ids=candidates,
                source_system_hint=_source_hint_from_systems(phone_resolution.get("external_strong_source_systems", ())),
            )
        if phone_resolution["status"] in {"ambiguous", "blocked"} and trusted_email_id in phone_ids:
            return LinkDecision(
                "strong",
                "shared_phone_strong_email_intersection",
                customer_id=trusted_email_id,
                method="phone_email_identity_intersection",
                contact_email=email or None,
                contact_phone=phone,
                contact_source=contact.contact_source,
                candidate_customer_ids=candidates,
                source_system_hint=_source_hint_from_systems(trusted_email_sources),
            )
        if phone_resolution["status"] in {"ambiguous", "blocked"}:
            return LinkDecision(
                "blocked",
                f"phone_{phone_resolution['reason']}",
                contact_email=email or None,
                contact_phone=phone,
                contact_source=contact.contact_source,
                candidate_customer_ids=candidates,
            )
    if email:
        if trusted_email_id:
            historical = "tallanto_historical_snapshot" in trusted_email_sources
            return LinkDecision(
                "strong",
                (
                    "strong_historical_tallanto_email_identity_link"
                    if historical
                    else "strong_external_email_identity_link"
                ),
                customer_id=email_resolution["customer_id"],
                method=(
                    "historical_tallanto_email_identity_link"
                    if historical
                    else "external_email_identity_link"
                ),
                contact_email=email,
                contact_phone=phone or None,
                contact_source=contact.contact_source,
                candidate_customer_ids=tuple(email_resolution["candidate_customer_ids"]),
                source_system_hint=_source_hint_from_systems(trusted_email_sources),
            )
        if email_resolution["status"] in {"strong", "weak", "ambiguous", "blocked"}:
            thread_decision = _compatible_thread_decision(
                _thread_anchor_decision(
                    row,
                    raw_message,
                    timeline_ro,
                    message_id_cache,
                ),
                candidate_customer_ids=candidates,
                contact=contact,
            )
            if thread_decision is not None:
                return thread_decision
        # D2 rule 4: an email shared by candidates who resolve to more than
        # one distinct known family is a positive, evidenced conflict --
        # more specific/actionable than the generic "weak_email" fallback
        # below, and (like weak_email) it never picks a customer_id, so it
        # is never a first-match. Emails whose candidates have no family
        # evidence at all fall through to the existing weak_email outcome
        # unchanged (we do not invent a conflict from missing data).
        if email_resolution["status"] == "ambiguous" and _spans_multiple_known_families(
            email_resolution["candidate_customer_ids"], family_ids or {}
        ):
            return LinkDecision(
                "blocked",
                "email_multiple_families_conflict",
                customer_id=None,
                method="email_identity_link_ambiguous_families",
                contact_email=email,
                contact_phone=phone or None,
                contact_source=contact.contact_source,
                candidate_customer_ids=tuple(email_resolution["candidate_customer_ids"]),
            )
        if email_resolution["status"] != "unmatched":
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
    thread_decision = _compatible_thread_decision(
        _thread_anchor_decision(
            row,
            raw_message,
            timeline_ro,
            message_id_cache,
        ),
        candidate_customer_ids=candidates,
        contact=contact,
    )
    if thread_decision is not None:
        return thread_decision
    if phone_resolution["status"] == "weak":
        return LinkDecision(
            "unmatched",
            "phone_non_authoritative_identity_link",
            contact_email=email or None,
            contact_phone=phone or None,
            contact_source=contact.contact_source,
            candidate_customer_ids=tuple(phone_resolution["candidate_customer_ids"]),
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


def _thread_anchor_decision(
    row: sqlite3.Row,
    raw_message: Mapping[str, Any],
    timeline_ro: sqlite3.Connection,
    message_id_cache: dict[Path, Mapping[str, tuple[str, ...]]],
) -> LinkDecision | None:
    archive_db_raw = raw_message.get("archive_db")
    message = raw_message.get("message")
    if not archive_db_raw or not isinstance(message, Mapping):
        return None
    archive_db = Path(str(archive_db_raw)).resolve(strict=False)
    reference_ids = _message_reference_ids(archive_db, message)
    if not reference_ids:
        return None
    message_ids = message_id_cache.get(archive_db)
    if message_ids is None:
        message_ids = _load_message_id_index(archive_db)
        message_id_cache[archive_db] = message_ids
    source_ids = sorted(
        {
            source_id
            for reference_id in reference_ids
            for source_id in message_ids.get(reference_id, ())
            if source_id != str(row["source_id"])
        }
    )
    if not source_ids:
        return None
    placeholders = ",".join("?" for _ in source_ids)
    anchors = timeline_ro.execute(
        f"""
        SELECT e.customer_id
        FROM timeline_events e
        JOIN customer_identities c
          ON c.tenant_id = e.tenant_id AND c.customer_id = e.customer_id
        WHERE e.tenant_id = ?
          AND e.source_system = ?
          AND e.source_id IN ({placeholders})
          AND e.match_status = 'strong_unique'
          AND c.identity_status = 'strong'
          AND COALESCE(json_extract(e.record_json, '$.metadata.mail_link_enrich.method'), '') != 'thread_header_identity_link'
        """,
        (str(row["tenant_id"]), A2V3_MAIL_SOURCE_SYSTEM, *source_ids),
    ).fetchall()
    customer_ids = sorted({str(anchor["customer_id"]) for anchor in anchors if anchor["customer_id"]})
    if not customer_ids:
        return None
    if len(customer_ids) > 1:
        return LinkDecision(
            "blocked",
            "thread_customer_conflict",
            candidate_customer_ids=tuple(customer_ids),
        )
    return LinkDecision(
        "strong",
        "strong_thread_header_identity_link",
        customer_id=customer_ids[0],
        method="thread_header_identity_link",
        contact_source="rfc_message_thread",
        candidate_customer_ids=tuple(customer_ids),
        source_system_hint="thread",
    )


def _compatible_thread_decision(
    decision: LinkDecision | None,
    *,
    candidate_customer_ids: Sequence[str],
    contact: ContactResult,
) -> LinkDecision | None:
    if decision is None:
        return None
    if decision.outcome == "strong" and not candidate_customer_ids:
        return None
    if decision.outcome == "strong" and len(set(candidate_customer_ids)) > 1:
        return LinkDecision(
            "blocked",
            "thread_contact_ambiguous",
            contact_email=contact.contact_email,
            contact_phone=contact.contact_phone,
            contact_source=contact.contact_source,
            candidate_customer_ids=tuple(sorted({*candidate_customer_ids, *decision.candidate_customer_ids})),
        )
    if decision.outcome == "strong" and candidate_customer_ids and decision.customer_id not in candidate_customer_ids:
        return LinkDecision(
            "blocked",
            "thread_contact_customer_conflict",
            contact_email=contact.contact_email,
            contact_phone=contact.contact_phone,
            contact_source=contact.contact_source,
            candidate_customer_ids=tuple(sorted({*candidate_customer_ids, *decision.candidate_customer_ids})),
        )
    return replace(
        decision,
        contact_email=contact.contact_email,
        contact_phone=contact.contact_phone,
        contact_source=decision.contact_source or contact.contact_source,
    )


def _block_existing_customer_conflict(row: sqlite3.Row, decision: LinkDecision) -> LinkDecision:
    existing_customer_id = str(row["customer_id"] or "")
    if decision.outcome == "family_strong" and existing_customer_id:
        if existing_customer_id in decision.candidate_customer_ids:
            return replace(
                decision,
                outcome="strong",
                reason="existing_customer_confirmed_within_family",
                customer_id=existing_customer_id,
            )
        return LinkDecision(
            "blocked",
            "existing_customer_family_conflict",
            contact_email=decision.contact_email,
            contact_phone=decision.contact_phone,
            contact_source=decision.contact_source,
            candidate_customer_ids=tuple(
                sorted({existing_customer_id, *(decision.candidate_customer_ids or ())})
            ),
        )
    if decision.outcome == "strong" and existing_customer_id and decision.customer_id != existing_customer_id:
        return LinkDecision(
            "blocked",
            "existing_customer_identity_conflict",
            contact_email=decision.contact_email,
            contact_phone=decision.contact_phone,
            contact_source=decision.contact_source,
            candidate_customer_ids=tuple(
                sorted({existing_customer_id, *(decision.candidate_customer_ids or ())})
            ),
        )
    return decision


def _preserve_existing_strong_without_explicit_conflict(
    row: sqlite3.Row,
    decision: LinkDecision,
) -> LinkDecision:
    existing_customer_id = str(row["customer_id"] or "")
    if str(row["match_status"]) != IdentityMatchClass.STRONG_UNIQUE.value or not existing_customer_id:
        return decision
    if decision.outcome == "strong" or decision.outcome.startswith("not_revalidated_"):
        return decision
    reason = decision.reason
    if (
        "conflict" in reason
        or reason == "cross_brand_signal"
        or reason.endswith(("multiple_customers", "ambiguous_identity_link"))
    ):
        return decision
    return LinkDecision(
        "not_revalidated_no_explicit_conflict",
        "not_revalidated_no_explicit_conflict",
        customer_id=existing_customer_id,
        contact_email=decision.contact_email,
        contact_phone=decision.contact_phone,
        contact_source=decision.contact_source,
        candidate_customer_ids=decision.candidate_customer_ids,
    )


def _load_message_id_index(archive_db: Path) -> Mapping[str, tuple[str, ...]]:
    with sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        columns = {str(row[1]) for row in con.execute("PRAGMA table_info(messages)")}
        if "message_id" not in columns:
            return {}
        by_message_id: dict[str, set[str]] = {}
        for row in con.execute("SELECT sha256, message_id FROM messages WHERE message_id IS NOT NULL"):
            normalized = _normalize_message_id(row["message_id"])
            if normalized:
                by_message_id.setdefault(normalized, set()).add(str(row["sha256"]))
        return {key: tuple(sorted(values)) for key, values in by_message_id.items()}


def _message_reference_ids(archive_db: Path, message: Mapping[str, Any]) -> tuple[str, ...]:
    sha = str(message.get("sha256") or "")
    candidates: list[Path] = []
    raw_path = str(message.get("raw_eml_path") or "").strip()
    if raw_path:
        candidate = Path(raw_path).expanduser()
        candidates.append(candidate if candidate.is_absolute() else archive_db.parent / candidate)
    if sha:
        candidates.append(archive_db.parent / "raw_eml" / sha[:2] / f"{sha}.eml")
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return ()
    return _message_reference_ids_from_file(source)


@lru_cache(maxsize=131072)
def _message_reference_ids_from_file(source: Path) -> tuple[str, ...]:
    lines: list[bytes] = []
    remaining = 65536
    try:
        with source.open("rb") as handle:
            while remaining > 0:
                line = handle.readline(min(remaining, 8192))
                if not line:
                    break
                lines.append(line)
                remaining -= len(line)
                if line in {b"\n", b"\r\n"}:
                    break
    except OSError:
        return ()
    header_bytes = b"".join(lines)
    parsed = BytesHeaderParser(policy=policy.default).parsebytes(header_bytes)
    values = [
        *parsed.get_all("In-Reply-To", []),
        *parsed.get_all("References", []),
    ]
    return tuple(
        dict.fromkeys(
            normalized
            for value in values
            for token in (re.findall(r"<([^<>]+)>", str(value)) or str(value).split())
            if (normalized := _normalize_message_id(token))
        )
    )


def _normalize_message_id(value: Any) -> str:
    return str(value or "").strip().strip("<>").casefold()


def _load_trusted_customer_brands(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, str]:
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_customer_brand_profiles'"
    ).fetchone():
        return {}
    rows = con.execute(
        """
        SELECT customer_id, brand
        FROM a2v3_customer_brand_profiles
        WHERE tenant_id = ?
          AND source = 'customer_history'
          AND reason IN ('single_known_brand_in_history', 'dominant_brand_history')
        """,
        (tenant_id,),
    ).fetchall()
    return {
        str(row["customer_id"]): brand
        for row in rows
        if (brand := _normalize_brand(row["brand"])) in {"foton", "unpk"}
    }


def _annotate_brand_authorization(
    row: sqlite3.Row,
    decision: LinkDecision,
    customer_brands: Mapping[str, str],
) -> LinkDecision:
    if decision.outcome != "strong" or not decision.customer_id:
        return decision
    customer_brand = customer_brands.get(decision.customer_id, "unknown")
    mail_brand = _brand_for_event(row, source_payload=_source_payload(_event_payload(row)))
    return replace(
        decision,
        brand_context_authorized=(
            customer_brand in {"foton", "unpk"}
            and mail_brand in {"foton", "unpk"}
            and customer_brand == mail_brand
        ),
        customer_brand=customer_brand,
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
        return record
    return {}


def _archive_db_for_source_payload(source_payload: Mapping[str, Any]) -> Optional[Path]:
    raw_db = str(
        source_payload.get("stage2_enrich_archive_db")
        or source_payload.get("archive_db")
        or source_payload.get("source_db")
        or ""
    ).strip()
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
    participant_cache: dict[Path, Mapping[str, list[Mapping[str, Any]]]],
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
    by_message = participant_cache.get(db)
    if by_message is None:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in con.execute(
            "SELECT message_sha256, header_name, display_name, email_normalized, domain FROM message_participants"
        ):
            grouped.setdefault(str(row["message_sha256"]), []).append(dict(row))
        participant_cache[db] = grouped
        by_message = grouped
    participants = by_message.get(message_sha, [])
    message = dict(msg)
    text = ""
    extracted_raw = str(message.get("extracted_text_path") or "")
    extracted = Path(extracted_raw).expanduser() if extracted_raw else Path()
    if extracted_raw and not extracted.is_absolute():
        extracted = db.parent / extracted
    if not extracted_raw or not extracted.is_file():
        extracted = db.parent / "extracted_text" / f"{message_sha}.txt"
    if extracted.is_file():
        text = extracted.read_text(encoding="utf-8", errors="ignore")[:65536]
    return {
        "archive_db": str(db),
        "message": message,
        "participants": participants,
        "text": text,
    }


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
    effective_direction = str(direction or "")
    if _is_external_participant(from_participant):
        effective_direction = "inbound"
    elif from_participant and any(
        _is_external_participant(item) for item in (*to_participants, *cc_participants)
    ):
        effective_direction = "outbound"
    return _resolve_customer_contact(
        direction=effective_direction,
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
    identity_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    if not link_value:
        return {"status": "unmatched", "reason": "missing_value", "candidate_customer_ids": ()}
    if identity_index is not None:
        rows = [row for link_type in link_types for row in identity_index.get((link_type, link_value), ())]
    else:
        placeholders = ",".join("?" for _ in link_types)
        rows = con.execute(
            f"""
            SELECT l.customer_id, l.match_class, l.source_system, c.identity_status
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
    amo_customer_ids = {
        str(row["customer_id"])
        for row in rows
        if str(row["match_class"]) == IdentityMatchClass.STRONG_UNIQUE.value
        and str(row["source_system"]) == "amocrm_snapshot"
    }
    trusted_family_customer_ids = {
        str(row["customer_id"])
        for row in rows
        if str(row["source_system"]) in TRUSTED_EMAIL_IDENTITY_SOURCES
        and str(row["match_class"]) in {
            IdentityMatchClass.STRONG_UNIQUE.value,
            IdentityMatchClass.AMBIGUOUS.value,
        }
    }
    if not customer_ids:
        return {"status": "unmatched", "reason": "no_identity_link", "candidate_customer_ids": ()}
    if len(customer_ids) != 1:
        return {
            "status": "ambiguous",
            "reason": "multiple_customers",
            "candidate_customer_ids": tuple(customer_ids),
            "amo_customer_ids": tuple(amo_customer_ids),
            "trusted_family_customer_ids": tuple(trusted_family_customer_ids),
        }
    if any(str(row["match_class"]) == IdentityMatchClass.AMBIGUOUS.value for row in rows):
        return {
            "status": "ambiguous",
            "reason": "ambiguous_identity_link",
            "candidate_customer_ids": tuple(customer_ids),
            "amo_customer_ids": tuple(amo_customer_ids),
            "trusted_family_customer_ids": tuple(trusted_family_customer_ids),
        }
    strong_customer_ids = {
        str(row["customer_id"])
        for row in rows
        if str(row["match_class"]) == IdentityMatchClass.STRONG_UNIQUE.value
    }
    external_strong_customer_ids = {
        str(row["customer_id"])
        for row in rows
        if str(row["match_class"]) == IdentityMatchClass.STRONG_UNIQUE.value
        and str(row["source_system"]) not in MAIL_IDENTITY_SOURCE_SYSTEMS
    }
    external_strong_source_systems = {
        str(row["source_system"])
        for row in rows
        if str(row["match_class"]) == IdentityMatchClass.STRONG_UNIQUE.value
        and str(row["source_system"]) not in MAIL_IDENTITY_SOURCE_SYSTEMS
    }
    identity_statuses = {str(row["identity_status"] or "").lower() for row in rows}
    exact_trusted_email = (
        bool(set(link_types) & set(EMAIL_LINK_TYPES))
        and external_strong_customer_ids == {customer_ids[0]}
        and bool(external_strong_source_systems & TRUSTED_EMAIL_IDENTITY_SOURCES)
    )
    if identity_statuses != {"strong"} and not exact_trusted_email:
        return {"status": "blocked", "reason": "customer_identity_not_strong", "candidate_customer_ids": tuple(customer_ids)}
    if not strong_customer_ids:
        return {
            "status": "weak",
            "reason": "non_authoritative_identity_link",
            "candidate_customer_ids": tuple(customer_ids),
            "external_strong_customer_ids": tuple(external_strong_customer_ids),
            "external_strong_source_systems": tuple(external_strong_source_systems),
            "amo_customer_ids": tuple(amo_customer_ids),
            "trusted_family_customer_ids": tuple(trusted_family_customer_ids),
        }
    return {
        "status": "strong",
        "reason": "unique_identity_link",
        "customer_id": customer_ids[0],
        "candidate_customer_ids": tuple(customer_ids),
        "external_strong_customer_ids": tuple(external_strong_customer_ids),
        "external_strong_source_systems": tuple(external_strong_source_systems),
        "amo_customer_ids": tuple(amo_customer_ids),
        "trusted_family_customer_ids": tuple(trusted_family_customer_ids),
    }


def _with_historical_tallanto_email(
    current: Mapping[str, Any],
    *,
    historical_customer_ids: Sequence[str],
) -> Mapping[str, Any]:
    if (
        current.get("trusted_family_customer_ids")
        or current.get("status") in {"ambiguous", "blocked"}
        or current.get("external_strong_source_systems")
    ):
        return current
    historical = tuple(sorted(set(historical_customer_ids)))
    if not historical:
        return current
    if len(historical) == 1:
        return {
            "status": "strong",
            "reason": "historical_tallanto_email_identity_link",
            "customer_id": historical[0],
            "candidate_customer_ids": historical,
            "external_strong_customer_ids": historical,
            "external_strong_source_systems": ("tallanto_historical_snapshot",),
            "amo_customer_ids": (),
            "trusted_family_customer_ids": historical,
        }
    return {
        "status": "ambiguous",
        "reason": "historical_tallanto_email_multiple_customers",
        "candidate_customer_ids": historical,
        "external_strong_customer_ids": (),
        "external_strong_source_systems": ("tallanto_historical_snapshot",),
        "amo_customer_ids": (),
        "trusted_family_customer_ids": historical,
    }


def _family_customer_for_candidates(
    candidate_customer_ids: Sequence[str],
    *,
    family_ids: Mapping[str, str],
    preferred_customer_ids: Sequence[str] = (),
) -> tuple[str, str]:
    candidates = tuple(sorted(set(candidate_customer_ids)))
    if len(candidates) < 2:
        return "", ""
    families = {family_ids.get(customer_id, "") for customer_id in candidates}
    if len(families) != 1 or "" in families:
        return "", ""
    del preferred_customer_ids
    return "", next(iter(families))


def _source_hint_from_systems(systems: Sequence[str]) -> str:
    """Pick one representative trusted source system label for the D2 rule 7
    breakdown (exact_amo / exact_tallanto). Prefers the more specific/fresh
    sources first; falls back to whatever is present so the hint is never
    silently empty when a strong source system exists.
    """
    values = set(systems)
    for preferred in ("amocrm_snapshot", "tallanto_snapshot", "tallanto_historical_snapshot", "master_contacts_snapshot"):
        if preferred in values:
            return preferred
    return next(iter(sorted(values)), "")


def _spans_multiple_known_families(candidate_customer_ids: Sequence[str], family_ids: Mapping[str, str]) -> bool:
    """True only when candidates carry *positive* evidence of 2+ distinct
    known families -- never when family membership is simply unknown for
    some/all candidates (that stays the existing, more conservative
    weak_email outcome; D2 rule 4 asks for a conflict only when we actually
    know it spans multiple families, not merely because we cannot prove it
    is one).
    """
    known_families = {family_ids[customer_id] for customer_id in candidate_customer_ids if family_ids.get(customer_id)}
    return len(known_families) > 1


def _decision_provenance(decision: LinkDecision) -> str:
    """D2 rule 7: exact_amo / exact_tallanto / thread_propagation / family /
    ambiguous / unmatched / other breakdown bucket for one decision.
    """
    if decision.outcome == "family_strong":
        return "family"
    if decision.outcome == "strong":
        if decision.method == "thread_header_identity_link":
            return "thread_propagation"
        hint = decision.source_system_hint
        if hint == "amocrm_snapshot":
            return "exact_amo"
        if hint in {"tallanto_snapshot", "tallanto_historical_snapshot", "master_contacts_snapshot"}:
            return "exact_tallanto"
        return "other_strong"
    if decision.outcome in {"weak_email", "blocked"}:
        return "ambiguous"
    if decision.outcome == "unmatched":
        return "unmatched"
    return "not_revalidated"


def _mail_link_enrich_breakdown(decisions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """D2 rule 7: report breakdown separated by provenance, computed from the
    already-built per-event decision reports (decision["provenance"], set in
    _decision_report) so it always matches exactly what was decided/applied.
    """
    counts: Counter[str] = Counter(str(item.get("provenance") or "other") for item in decisions)
    keys = (
        "exact_amo",
        "exact_tallanto",
        "thread_propagation",
        "family",
        "ambiguous",
        "unmatched",
        "other_strong",
        "not_revalidated",
    )
    return {key: int(counts.get(key, 0)) for key in keys}


def _load_persisted_family_ids(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, str]:
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='family_members_v1'"
    ).fetchone()
    if exists is None:
        return {}
    blocked = open_family_identity_conflict_customer_ids(con, tenant_id)
    return {
        str(row["customer_id"]): str(row["family_id"])
        for row in con.execute(
            """
            SELECT customer_id, family_id
            FROM family_members_v1
            WHERE tenant_id = ? AND membership_status = 'confident' AND confidence = 'high'
            """,
            (tenant_id,),
        )
        if str(row["customer_id"] or "") not in blocked
        and str(row["customer_id"] or "")
        and str(row["family_id"] or "")
    }


def _load_identity_index(
    con: sqlite3.Connection, *, tenant_id: str
) -> Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]:
    link_types = tuple(dict.fromkeys((*PHONE_LINK_TYPES, *EMAIL_LINK_TYPES)))
    placeholders = ",".join("?" for _ in link_types)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in con.execute(
        f"""
        SELECT l.link_type, l.link_value, l.customer_id, l.match_class, l.source_system, c.identity_status
        FROM identity_links l
        LEFT JOIN customer_identities c
          ON c.tenant_id = l.tenant_id AND c.customer_id = l.customer_id
        WHERE l.tenant_id = ?
          AND l.link_type IN ({placeholders})
          AND l.customer_id IS NOT NULL
          AND l.customer_id != ''
        """,
        (tenant_id, *link_types),
    ):
        grouped.setdefault((str(row["link_type"]), str(row["link_value"])), []).append(dict(row))
    return grouped


def _load_historical_tallanto_email_index(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    identity_dbs: Sequence[Path],
    family_ids: Mapping[str, str],
) -> tuple[Mapping[str, Sequence[str]], Mapping[str, Any]]:
    tallanto_customers: dict[str, set[str]] = {}
    for row in con.execute(
        """
        SELECT l.link_value, l.customer_id
        FROM identity_links l
        JOIN customer_identities c
          ON c.tenant_id = l.tenant_id AND c.customer_id = l.customer_id
        WHERE l.tenant_id = ?
          AND l.link_type = 'tallanto_student_id'
          AND l.source_system = 'tallanto_snapshot'
          AND l.match_class = 'strong_unique'
          AND c.identity_status = 'strong'
        """,
        (tenant_id,),
    ):
        tallanto_customers.setdefault(str(row["link_value"]), set()).add(str(row["customer_id"]))
    email_customers: dict[str, set[str]] = {}
    databases = 0
    source_email_values = 0
    for raw_path in dict.fromkeys(Path(path).expanduser().resolve(strict=False) for path in identity_dbs):
        if not raw_path.is_file():
            raise FileNotFoundError(f"Tallanto identity DB is missing: {raw_path}")
        with sqlite3.connect(f"file:{raw_path}?mode=ro", uri=True, timeout=15) as identity_con:
            identity_con.row_factory = sqlite3.Row
            identity_con.execute("PRAGMA query_only=ON")
            if str(identity_con.execute("PRAGMA quick_check").fetchone()[0]) != "ok":
                raise RuntimeError(f"Tallanto identity DB failed quick_check: {raw_path}")
            rows = identity_con.execute(
                """
                SELECT l.value, c.tallanto_id
                FROM identity_links l
                JOIN identity_values v
                  ON v.kind = l.kind AND v.value = l.value
                JOIN identity_candidates c
                  ON c.candidate_key = l.candidate_key
                WHERE l.kind = 'email'
                  AND v.match_class = 'strong_unique'
                  AND v.candidate_count = 1
                  AND c.tallanto_id IS NOT NULL
                  AND c.tallanto_id != ''
                """
            ).fetchall()
        databases += 1
        source_email_values += len(rows)
        for row in rows:
            customers = tallanto_customers.get(str(row["tallanto_id"]), ())
            if customers:
                email_customers.setdefault(normalize_email(row["value"]), set()).update(customers)
    result = {email: tuple(sorted(customers)) for email, customers in email_customers.items() if email}
    single_values = sum(1 for customers in result.values() if len(customers) == 1)
    same_family_values = sum(
        1
        for customers in result.values()
        if len(customers) > 1
        and len({family_ids.get(customer_id, "") for customer_id in customers}) == 1
        and all(family_ids.get(customer_id) for customer_id in customers)
    )
    return result, {
        "databases": databases,
        "source_email_values": source_email_values,
        "usable_email_values": len(result),
        "single_customer_values": single_values,
        "same_family_values": same_family_values,
        "cross_family_values": len(result) - single_values - same_family_values,
        "read_only": True,
    }


def _empty_identity_resolution() -> Mapping[str, Any]:
    return {
        "status": "unmatched",
        "reason": "missing_value",
        "candidate_customer_ids": (),
        "external_strong_customer_ids": (),
        "external_strong_source_systems": (),
        "amo_customer_ids": (),
        "trusted_family_customer_ids": (),
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
        "brand_context_authorized": decision.brand_context_authorized,
        "customer_brand": decision.customer_brand,
        "family_id": decision.family_id,
        # D2 rule 7: exact_amo / exact_tallanto / thread_propagation / family
        # / ambiguous / unmatched breakdown bucket for this one decision.
        "provenance": _decision_provenance(decision),
    }


def _apply_decisions(
    db_path: Path,
    *,
    allowed_root: Path,
    tenant_id: str,
    targets: Sequence[sqlite3.Row],
    decisions: Sequence[Mapping[str, Any]],
    out_dir: Path,
    aggregate_only: bool,
) -> Mapping[str, Any]:
    by_event = {str(row["event_id"]): row for row in targets}
    counters: Counter[str] = Counter()
    changed_event_ids: list[str] = []
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        events_with_chunks = {
            str(row[0])
            for row in store._con.execute(
                "SELECT event_id FROM bot_context_chunks WHERE source_system = ? AND event_id IS NOT NULL AND superseded_by IS NULL",
                (A2V3_MAIL_SOURCE_SYSTEM,),
            )
        }
        with store.bulk_write():
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
            for decision in decisions:
                row = by_event[str(decision["event_id"])]
                if str(decision["outcome"]).startswith("not_revalidated_"):
                    counters["not_revalidated_events"] += 1
                    continue
                event = _updated_event_from_decision(row, decision)
                before_hash = str(row["record_hash"])
                result = store.upsert_event(event, actor="mail_link_enrich", ingestion_run_id=run.run_id)
                if result.record_hash != before_hash:
                    counters["updated_events"] += 1
                    changed_event_ids.append(event.event_id)
                else:
                    counters["unchanged_events"] += 1
                if decision["outcome"] == "strong" and event.event_id not in events_with_chunks:
                    chunk = _chunk_for_event(row, event, decision)
                    if chunk is not None:
                        chunk_result = store.upsert_bot_context_chunk(
                            chunk,
                            actor="mail_link_enrich",
                            ingestion_run_id=run.run_id,
                        )
                        if chunk_result.created:
                            counters["created_chunks"] += 1
                        events_with_chunks.add(event.event_id)
                elif decision["outcome"] != "strong" and event.event_id in events_with_chunks:
                    counters["revoked_chunks"] += store.revoke_bot_context_chunks_for_event(
                        event.tenant_id,
                        event_id=event.event_id,
                        source_system=A2V3_MAIL_SOURCE_SYSTEM,
                        reason="identity_revalidation",
                        actor="mail_link_enrich",
                        ingestion_run_id=run.run_id,
                    )
                    events_with_chunks.discard(event.event_id)
                _upsert_fact_for_decision(store._con, event, decision)
            store.finish_ingestion_run(
                run.run_id,
                status="completed",
                accepted_count=int(counters["updated_events"] + counters["created_chunks"]),
                rejected_count=0,
                output_ref=str(out_dir / "mail_link_enrich_apply_report.json"),
                metadata=(
                    {"counts": dict(counters)}
                    if aggregate_only
                    else {"counts": dict(counters), "changed_event_ids": changed_event_ids[:100]}
                ),
                actor="mail_link_enrich",
            )
    report: dict[str, Any] = {"counts": dict(counters)}
    if not aggregate_only:
        report["changed_event_ids_sample"] = changed_event_ids[:20]
    return report


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
    elif decision["outcome"] == "family_strong":
        source_payload["customer_id"] = None
    record["payload"] = source_payload
    metadata["brand"] = brand
    metadata["brand_context_authorized"] = bool(decision.get("brand_context_authorized"))
    metadata["family_id"] = str(decision.get("family_id") or "")
    metadata["mail_link_enrich"] = {
        "schema_version": MAIL_LINK_ENRICH_SCHEMA_VERSION,
        "outcome": decision["outcome"],
        "reason": decision["reason"],
        "method": decision.get("method"),
        "contact_source": decision.get("contact_source"),
        "candidate_customer_count": decision.get("candidate_customer_count"),
    }
    metadata["fresh_relink"] = True
    metadata.pop("family_attributed", None)
    if decision["outcome"] == "strong":
        metadata["pending_attribution"] = False
        metadata.pop("pending_reason", None)
        customer_id = str(decision["customer_id"])
        match_status = IdentityMatchClass.STRONG_UNIQUE
        confidence = 0.92
    elif decision["outcome"] == "family_strong":
        metadata["pending_attribution"] = False
        metadata["family_attributed"] = True
        customer_id = None
        match_status = IdentityMatchClass.AMBIGUOUS
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
            "brand_context_authorized": bool(event.metadata.get("brand_context_authorized")),
        },
        created_at=event.created_at,
    )


def _upsert_fact_for_decision(con: sqlite3.Connection, event: TimelineEvent, decision: Mapping[str, Any]) -> None:
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
    ).fetchone():
        return
    existing = con.execute(
        "SELECT customer_id, bot_visible, bot_gate_reason FROM a2v3_mail_event_facts WHERE message_sha256 = ?",
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
        "customer_brand": "unknown",
        "customer_brand_source": "unresolved",
        "customer_brand_reason": "email_brand_is_not_customer_brand",
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
        same_customer = bool(existing["customer_id"] and event.customer_id) and (
            str(existing["customer_id"]) == str(event.customer_id)
        )
        row["bot_visible"] = int(existing["bot_visible"] or 0) if same_customer else 0
        row["bot_gate_reason"] = (
            existing["bot_gate_reason"] if same_customer else "identity_changed_requires_a2_revalidation"
        )
        con.execute(
            """
            UPDATE a2v3_mail_event_facts
            SET event_id=:event_id, customer_id=:customer_id, email_brand=:email_brand,
                customer_brand=:customer_brand, customer_brand_source=:customer_brand_source,
                customer_brand_reason=:customer_brand_reason, bot_visible=:bot_visible,
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
