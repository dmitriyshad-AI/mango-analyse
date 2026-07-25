from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.p0_recall_spec import hard_codes_from_text
from mango_mvp.customer_timeline.derived_signals import _is_access_event, _is_active_deal
from mango_mvp.customer_timeline.freshness import (
    MANAGER_REQUIRED_SOURCE_SYSTEMS,
    manager_freshness_gate,
    source_freshness_rows,
)
from mango_mvp.customer_timeline.next_step_resolver import (
    NEXT_STEP_STATUS_ACTIVE,
    NEXT_STEP_STATUS_EMPTY,
    _event_text,
    _is_non_closing_service_event,
    resolve_customer_next_step,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import customer_entity_ref_values, customer_timeline_readonly_uri


MANAGER_DOSSIER_SCHEMA_VERSION = "customer_timeline_manager_dossier_v1"
INTEREST_MARKER_RE = re.compile(r"\b(?:интересу\w*|рассматрива\w*|хот(?:им|им\s+бы|им\s+посмотреть))\b", re.I)
PAIN_MARKER_RE = re.compile(r"\b(?:не\s+успева\w*|сложн\w*|провалил\w*|провал\w*|пережива\w*)\b", re.I)
INTEREST_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"математик\w*|физик\w*|информатик\w*|программировани\w*|русск\w+\s+язык\w*|английск\w+\s+язык\w*|"
    r"егэ|огэ|олимпиад\w*|курс\w*|заняти\w*|групп\w*|лагер\w*|школ\w*|смен\w*|интенсив\w*|"
    r"подготовк\w*|очно\w*|онлайн\w*|выездн\w*|летн\w*|годов\w*"
    r")\b",
    re.I,
)
CONTACT_RE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}|"
    r"(?<!\d)(?:(?:\+7|8|7)\s*)?\(?\d{3,4}\)?[\s.-]*\d{2,3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)",
    re.I,
)
EMAIL_SUMMARY_REVIEW_NEEDED_RE = re.compile(r"^\s*Требуется\s+ручная\s+проверка\s+модельной\s+выжимки\b", re.I)
WHITESPACE_RE = re.compile(r"\s+")
SPEECH_FILLER_RE = re.compile(
    r"^(?:(?:ну|ээ+|э+|эм+|мм+|вот|значит|как\s+бы|то\s+есть|скажем|короче)\b[\s,.;:–—-]*)+",
    re.I,
)
SPEECH_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s+(?:"
    r"Можете|можете|можно|подскажите|скажите|скиньте|пришлите|"
    r"сколько\s+будет|сч[её]т\s+скинуть|как\s+там|сейчас"
    r")\b",
    re.I,
)
PRODUCT_KEYS = {
    "products_of_interest",
    "product_of_interest",
    "продукты интереса",
    "интересы",
    "interest",
    "interests",
}
MANAGER_OUTREACH_SIGNAL_TYPES = (
    "client_returned",
    "callback_due",
    "deal_stalling",
    "season_return_candidate",
)
MANAGER_OUTREACH_RISK_SIGNAL_TYPES = ("paid_no_access", "duplicate_contact")
MANAGER_KNOWN_BRANDS = frozenset({"foton", "unpk"})
OWNER50_SIGNAL_PRIORITY = {
    "callback_due": 0,
    "client_returned": 0,
    "deal_stalling": 1,
    "season_return_candidate": 2,
}
OWNER50_HARD_REASONS = frozenset({
    "identity_not_strong", "brand_not_exactly_one_known", "open_identity_conflict",
    "family_ambiguous", "durable_p0_history", "durable_opt_out",
    "meaningful_outbound_after_evidence",
})
OWNER50_STAFF_TEST_RE = re.compile(
    r"\b(?:staff|employee|test|system|сотрудник\w*|тестов\w*|служебн\w*|системн\w*)\b",
    re.I,
)
OWNER50_GRADUATE_RE = re.compile(r"(?<!\d)11(?!\d)|\bвыпускник\w*", re.I)
MANAGER_OPTOUT_PHRASES = (
    "не пишите",
    "больше не пишите",
    "перестаньте писать",
    "не звоните",
    "больше не звоните",
    "не надо мне звонить",
    "не беспокойте",
    "не связывайтесь",
    "удалите номер",
    "отпишите меня",
    "хочу отписаться",
    "не хочу получать рассылку",
)


@dataclass(frozen=True)
class DossierMarker:
    kind: str
    text: str
    source: str


@dataclass(frozen=True)
class DossierRow:
    section: str
    text: str
    source: str


@dataclass(frozen=True)
class CustomerDossier:
    tenant_id: str
    customer_id: str
    display_name: str
    brand: str
    phone: str
    email: str
    actuality_header: str = ""
    family: tuple[DossierRow, ...] = field(default_factory=tuple)
    money: tuple[DossierRow, ...] = field(default_factory=tuple)
    signals: tuple[DossierRow, ...] = field(default_factory=tuple)
    next_step: str = ""
    next_step_source: str = ""
    objections: tuple[DossierRow, ...] = field(default_factory=tuple)
    chronology: tuple[DossierRow, ...] = field(default_factory=tuple)
    interests: tuple[DossierMarker, ...] = field(default_factory=tuple)
    pains: tuple[DossierMarker, ...] = field(default_factory=tuple)


def build_customer_dossier(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    canonical_calls: Mapping[str, str] | None = None,
    actuality_header: str = "",
) -> CustomerDossier:
    con.row_factory = sqlite3.Row
    customer = con.execute(
        """
        SELECT customer_id, tenant_id, display_name, primary_phone, primary_email, record_json
        FROM customer_identities
        WHERE tenant_id = ? AND customer_id = ?
        """,
        (tenant_id, customer_id),
    ).fetchone()
    if customer is None:
        raise ValueError(f"customer not found: {customer_id}")
    customer_record = _safe_json(customer["record_json"])
    brands = [
        str(item).strip().casefold()
        for item in (_mapping(customer_record.get("metadata")).get("brands") or ())
        if str(item).strip()
    ]
    opportunities = con.execute(
        """
        SELECT opportunity_id, record_json
        FROM customer_opportunities
        WHERE tenant_id = ? AND customer_id = ?
        ORDER BY opened_at DESC, opportunity_id
        """,
        (tenant_id, customer_id),
    ).fetchall()
    events = con.execute(
        """
        SELECT event_id, event_at, source_id, source_ref, event_type, record_json
        FROM timeline_events
        WHERE tenant_id = ?
          AND customer_id = ?
          AND event_type = 'mango_call'
          AND match_status = 'strong_unique'
          AND (superseded_by IS NULL OR superseded_by = '')
        ORDER BY event_at DESC, event_id DESC
        LIMIT 100
        """,
        (tenant_id, customer_id),
    ).fetchall()
    interests: list[DossierMarker] = []
    pains: list[DossierMarker] = []
    for value in _product_interest_values(customer["record_json"], opportunities):
        interests.append(DossierMarker(kind="interest", text=f"Из данных: {value}", source="products_of_interest"))
    call_texts = canonical_calls or {}
    for event in events:
        client_text = _lookup_canonical_client_text(event, call_texts)
        if not client_text.strip():
            continue
        source = f"mango_call:{event['source_id']}"
        interests.extend(_markers_from_client_text(client_text, INTEREST_MARKER_RE, kind="interest", label="Интерес из звонка", source=source))
        pains.extend(_markers_from_client_text(client_text, PAIN_MARKER_RE, kind="pain", label="Боль из звонка", source=source))
    signals = _signal_rows(con, tenant_id=tenant_id, customer_id=customer_id)
    next_step, next_step_source = _next_step_for_dossier(
        con, tenant_id=tenant_id, customer_id=customer_id, signals=signals
    )
    return CustomerDossier(
        tenant_id=str(customer["tenant_id"]),
        customer_id=str(customer["customer_id"]),
        display_name=_clean_text(customer["display_name"]),
        brand=brands[0] if len(brands) == 1 else "",
        phone=_clean_text(customer["primary_phone"]),
        email=_clean_text(customer["primary_email"]),
        actuality_header=actuality_header,
        family=tuple(
            _family_rows(
                con,
                tenant_id=tenant_id,
                customer_id=customer_id,
                active_brand=brands[0] if len(brands) == 1 else "",
            )
        ),
        money=tuple(_money_rows(con, tenant_id=tenant_id, customer_id=customer_id)),
        signals=tuple(signals),
        next_step=next_step,
        next_step_source=next_step_source,
        objections=tuple(_objection_rows(con, tenant_id=tenant_id, customer_id=customer_id)),
        chronology=tuple(_chronology_rows(con, tenant_id=tenant_id, customer_id=customer_id, limit=12)),
        interests=tuple(_dedupe_markers(interests, limit=8)),
        pains=tuple(_dedupe_markers(pains, limit=8)),
    )


def build_manager_dossier_workbook(
    *,
    timeline_db: Path | str,
    allowed_root: Path | str,
    out_xlsx: Path | str,
    tenant_id: str = "foton",
    customer_ids: Sequence[str] | None = None,
    canonical_calls_db: Path | str | None = None,
    reconcile_json: Path | str | None = None,
    limit: int = 50,
    enforce_freshness: bool = True,
    enforce_outreach_eligibility: bool = False,
) -> Mapping[str, Any]:
    db = Path(timeline_db).expanduser().resolve(strict=False)
    out = _guard_local_dossier_output_path(out_xlsx, allowed_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    canonical_calls, canonical_warning = _load_canonical_calls_fail_soft(canonical_calls_db)
    reconcile = _read_json(Path(reconcile_json).expanduser()) if reconcile_json else {}
    with _connect_ro(db) as con:
        ids = (
            tuple(_full_dossier_segment_customer_ids(con, tenant_id=tenant_id, limit=limit))
            if customer_ids is None
            else tuple(customer_ids)
        )
        freshness = _source_freshness(con, tenant_id=tenant_id)
        freshness_gate = manager_freshness_gate(freshness)
        if enforce_freshness and not freshness_gate["passed"]:
            reasons = ", ".join(
                f"{item['source_system']}:{item['reason']}" for item in freshness_gate["blockers"]
            )
            raise RuntimeError(f"manager freshness gate failed: {reasons}")
        segment_total = _full_dossier_segment_count(con, tenant_id=tenant_id)
        actuality_header = _actuality_header(freshness, reconcile)
        dossiers: list[CustomerDossier] = []
        missing_customer_ids: list[str] = []
        exclusion_counts: Counter[str] = Counter()
        for customer_id in ids:
            if enforce_outreach_eligibility:
                eligibility = manager_outreach_eligibility(
                    con,
                    tenant_id=tenant_id,
                    customer_id=customer_id,
                )
                if not eligibility["eligible"]:
                    exclusion_counts.update(eligibility["reasons"])
                    continue
            try:
                dossiers.append(
                    build_customer_dossier(
                        con,
                        tenant_id=tenant_id,
                        customer_id=customer_id,
                        canonical_calls=canonical_calls,
                        actuality_header=actuality_header,
                    )
                )
            except ValueError:
                missing_customer_ids.append(customer_id)
    _write_workbook(out, dossiers)
    summary = {
        "schema_version": MANAGER_DOSSIER_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "requested_customers": len(ids),
        "customers": len(dossiers),
        "missing_customer_ids_count": len(missing_customer_ids),
        "missing_customer_ids_sample": missing_customer_ids[:10],
        "outreach_eligibility_enforced": bool(enforce_outreach_eligibility),
        "outreach_exclusion_counts": dict(exclusion_counts),
        "full_dossier_segment_total": segment_total,
        "interests_total": sum(len(item.interests) for item in dossiers),
        "pains_total": sum(len(item.pains) for item in dossiers),
        "family_rows_total": sum(len(item.family) for item in dossiers),
        "money_rows_total": sum(len(item.money) for item in dossiers),
        "signals_total": sum(len(item.signals) for item in dossiers),
        "objections_total": sum(len(item.objections) for item in dossiers),
        "chronology_rows_total": sum(len(item.chronology) for item in dossiers),
        "next_step_rows_total": sum(1 for item in dossiers if item.next_step),
        "missing_next_step_rows_total": sum(1 for item in dossiers if not item.next_step),
        "canonical_calls_loaded": len(canonical_calls),
        "canonical_calls_warning": canonical_warning,
        "actuality_header": actuality_header,
        "source_freshness_top": freshness[:12],
        "freshness_gate": freshness_gate,
        "reconcile_status": reconcile.get("status") if reconcile else "missing",
        "out_xlsx": str(out),
        "safety": {
            "source_open_mode": "sqlite_mode_ro",
            "write_crm": False,
            "write_tallanto": False,
            "send_messages": False,
            "pii_scope": "local_codex_local_only",
        },
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_owner50_family_workbook(
    *,
    timeline_db: Path | str,
    allowed_root: Path | str,
    out_xlsx: Path | str,
    tenant_id: str = "foton",
    limit: int = 50,
    as_of: datetime | None = None,
) -> Mapping[str, Any]:
    """Build the owner-only family outreach queue without external writes."""
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    out = _guard_local_dossier_output_path(out_xlsx, allowed_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with _connect_ro(Path(timeline_db).expanduser().resolve(strict=False)) as con:
        candidates, control = _owner50_family_rows(con, tenant_id=tenant_id, as_of=now)
        candidates.sort(key=lambda row: row["rank_key"])
        selected = candidates[: max(0, int(limit))]
        _enrich_owner50_selected_rows(con, selected, tenant_id=tenant_id)
    selected_ids = {row["family_id"] for row in selected}
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        control.append((row["family_id"], "selected", row["rank_reason"]))
    control.extend(
        (row["family_id"], "outside_limit", row["rank_reason"])
        for row in candidates
        if row["family_id"] not in selected_ids
    )
    _write_owner50_workbook(out, selected, control)
    return {
        "families": len(selected),
        "candidate_families": len(candidates),
        "excluded_families": sum(status == "excluded" for _, status, _ in control),
        "out_xlsx": str(out),
        "sheets": ("Кому писать", "Доказательства", "Контроль"),
        "write_external": False,
    }


def load_canonical_call_client_texts(path: Path | str | None) -> Mapping[str, str]:
    if path is None:
        return {}
    db = Path(path).expanduser().resolve(strict=False)
    if not db.exists():
        return {}
    with _connect_ro(db) as con:
        rows = con.execute("SELECT canonical_call_id, transcript_client FROM canonical_calls").fetchall()
    return {str(row[0]): str(row[1] or "") for row in rows}


def _load_canonical_calls_fail_soft(path: Path | str | None) -> tuple[Mapping[str, str], str]:
    if path is None:
        return {}, ""
    db = Path(path).expanduser().resolve(strict=False)
    if not db.exists():
        return {}, f"canonical calls DB not found, continuing without call quotes: {db}"
    try:
        return load_canonical_call_client_texts(db), ""
    except (sqlite3.Error, OSError) as exc:
        return {}, f"canonical calls DB unavailable, continuing without call quotes: {type(exc).__name__}"


def _guard_local_dossier_output_path(path: Path | str, allowed_root: Path | str) -> Path:
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    root = Path(allowed_root).resolve(strict=False)
    relative = resolved.relative_to(root)
    if not relative.parts or relative.parts[0] != ".codex_local":
        raise ValueError("manager dossier output contains PII and must stay under .codex_local")
    return resolved


def _lookup_canonical_client_text(event: sqlite3.Row, canonical_calls: Mapping[str, str]) -> str:
    for key in _canonical_call_candidate_keys(event):
        value = canonical_calls.get(key)
        if value:
            return value
    return ""


def _canonical_call_candidate_keys(event: sqlite3.Row) -> tuple[str, ...]:
    keys: list[str] = []
    record = _safe_json(event["record_json"])
    nested_record = record.get("record") if isinstance(record, Mapping) and isinstance(record.get("record"), Mapping) else {}
    canonical_call_id = _clean_text(
        (record.get("canonical_call_id") or nested_record.get("canonical_call_id")) if isinstance(record, Mapping) else None
    )
    if canonical_call_id:
        keys.append(canonical_call_id)
    for raw in (event["source_id"], event["source_ref"]):
        text = _clean_text(raw)
        if not text:
            continue
        keys.append(text)
        if text.startswith("call:"):
            keys.append(text.removeprefix("call:"))
        if ":" in text:
            keys.append(text.split(":", 1)[0])
    return tuple(_dedupe_texts(keys))


def _connect_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(customer_timeline_readonly_uri(path), uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def manager_outreach_eligibility(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    signal_id: str | None = None,
    as_of: datetime | None = None,
) -> Mapping[str, Any]:
    """Fail closed before a customer reaches a proactive manager list."""
    con.row_factory = sqlite3.Row
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    required_tables = ("customer_identities", "derived_signals", "timeline_events", "timeline_conflicts", "family_links_v1")
    missing = [table for table in required_tables if not _table_exists(con, table)]
    if missing:
        return {"eligible": False, "reasons": tuple(f"safety_table_missing:{table}" for table in missing)}

    identity = con.execute(
        "SELECT identity_status, record_json FROM customer_identities WHERE tenant_id=? AND customer_id=?",
        (tenant_id, customer_id),
    ).fetchone()
    reasons: list[str] = []
    identity_record = _safe_json(identity["record_json"]) if identity else {}
    brands = {
        str(item).strip().casefold()
        for item in (_mapping(identity_record.get("metadata")).get("brands") or ())
        if str(item).strip()
    }
    if not identity or str(identity["identity_status"] or "") != "strong":
        reasons.append("identity_not_strong")
    if len(brands) != 1 or not brands.issubset(MANAGER_KNOWN_BRANDS):
        reasons.append("brand_not_exactly_one_known")

    signal_clauses = [
        "tenant_id=?", "customer_id=?", "status='active'",
        f"signal_type IN ({','.join('?' for _ in MANAGER_OUTREACH_SIGNAL_TYPES)})",
        "(expires_at IS NULL OR expires_at='' OR julianday(expires_at)>=julianday(?))",
    ]
    signal_params: list[Any] = [tenant_id, customer_id, *MANAGER_OUTREACH_SIGNAL_TYPES, now.isoformat()]
    if signal_id:
        signal_clauses.append("signal_id=?")
        signal_params.append(signal_id)
    signal = con.execute(
        f"SELECT signal_id, event_id, signal_type, created_at, record_json FROM derived_signals "
        f"WHERE {' AND '.join(signal_clauses)} ORDER BY created_at DESC, signal_id LIMIT 1",
        tuple(signal_params),
    ).fetchone()
    if signal is None:
        reasons.append("no_active_outreach_signal")

    refs = customer_entity_ref_values(customer_id)
    open_conflict = con.execute(
        "SELECT 1 FROM timeline_conflicts c WHERE c.tenant_id=? AND c.status IN ('open','active') "
        "AND json_valid(c.record_json) AND EXISTS (SELECT 1 FROM json_each(c.record_json,'$.entity_refs') r "
        f"WHERE CAST(r.value AS TEXT) IN ({','.join('?' for _ in refs)})) LIMIT 1",
        (tenant_id, *refs),
    ).fetchone()
    if open_conflict:
        reasons.append("open_identity_conflict")
    family_risk = con.execute(
        "SELECT 1 FROM family_links_v1 WHERE tenant_id=? AND customer_id=? "
        "AND (COALESCE(status,'')!='confident' OR COALESCE(confidence,'') NOT IN ('high','medium')) LIMIT 1",
        (tenant_id, customer_id),
    ).fetchone()
    if family_risk:
        reasons.append("family_ambiguous")
    risk_signal = con.execute(
        f"SELECT signal_type FROM derived_signals WHERE tenant_id=? AND customer_id=? AND status='active' "
        f"AND signal_type IN ({','.join('?' for _ in MANAGER_OUTREACH_RISK_SIGNAL_TYPES)}) "
        "AND (expires_at IS NULL OR expires_at='' OR julianday(expires_at)>=julianday(?)) LIMIT 1",
        (tenant_id, customer_id, *MANAGER_OUTREACH_RISK_SIGNAL_TYPES, now.isoformat()),
    ).fetchone()
    if risk_signal:
        reasons.append(f"active_risk_signal:{risk_signal['signal_type']}")

    evidence_at: datetime | None = None
    signal_created_at: datetime | None = None
    if signal is not None:
        signal_created_at = _parse_iso_datetime(signal["created_at"])
        signal_record = _safe_json(signal["record_json"])
        event_id = _clean_text(signal["event_id"] or signal_record.get("event_id"))
        if event_id:
            event = con.execute(
                "SELECT event_at,event_type,match_status,superseded_by FROM timeline_events "
                "WHERE tenant_id=? AND customer_id=? AND event_id=? LIMIT 1",
                (tenant_id, customer_id, event_id),
            ).fetchone()
            if event is None:
                reasons.append("signal_evidence_not_owned")
            elif event["superseded_by"]:
                reasons.append("signal_evidence_superseded")
            elif str(event["event_type"] or "") == "mango_call" and str(event["match_status"] or "") != "strong_unique":
                reasons.append("signal_evidence_ambiguous_call")
            else:
                evidence_at = _parse_iso_datetime(event["event_at"])
        elif str(signal["signal_type"]) == "season_return_candidate":
            evidence_at = _parse_iso_datetime(_mapping(signal_record.get("metadata")).get("last_purchase_at"))
            if evidence_at is None:
                reasons.append("signal_evidence_missing")
            elif not _season_purchase_matches(
                con,
                tenant_id=tenant_id,
                customer_id=customer_id,
                evidence_at=evidence_at,
            ):
                reasons.append("season_purchase_not_confirmed")
            elif _has_active_customer_access(con, tenant_id=tenant_id, customer_id=customer_id):
                reasons.append("active_access_or_learning")
        else:
            reasons.append("signal_evidence_missing")

    scan_from = min(filter(None, (signal_created_at, now - timedelta(days=30))), default=now - timedelta(days=30))
    event_rows = con.execute(
        "SELECT event_id,event_at,event_type,source_system,source_id,source_ref,subject,text_preview,summary,direction,record_json "
        "FROM timeline_events WHERE tenant_id=? AND customer_id=? AND (superseded_by IS NULL OR superseded_by='') "
        "AND julianday(event_at)>=julianday(?) ORDER BY event_at,event_id",
        (tenant_id, customer_id, scan_from.isoformat()),
    ).fetchall()
    outbound_cutoff = max(filter(None, (evidence_at, now - timedelta(days=30))), default=now - timedelta(days=30))
    for row in event_rows:
        event = dict(row)
        stored = _safe_json(row["record_json"])
        event["record"] = _mapping(stored.get("record"))
        event["metadata"] = _mapping(stored.get("metadata"))
        text = _event_text(event)
        if (
            str(row["direction"] or "").casefold() == "outbound"
            and (_parse_iso_datetime(row["event_at"]) or now) > outbound_cutoff
            and not _is_non_closing_service_event(event)
        ):
            reasons.append("meaningful_outbound_after_evidence")
    # ponytail: block historical hard risks until a structured resolution/opt-in field exists.
    reasons.extend(_durable_contact_risks(con, tenant_id=tenant_id, customer_id=customer_id))
    unique_reasons = tuple(dict.fromkeys(reasons))
    return {
        "eligible": not unique_reasons,
        "reasons": unique_reasons,
        "signal_id": str(signal["signal_id"]) if signal else None,
        "signal_type": str(signal["signal_type"]) if signal else None,
        "brand": next(iter(brands)) if len(brands) == 1 else None,
    }


def _season_purchase_matches(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    evidence_at: datetime,
) -> bool:
    if not _table_exists(con, "customer_purchases_v1"):
        return False
    row = con.execute(
        "SELECT SUM(total_in) AS total_in, SUM(total_out) AS total_out, "
        "SUM(deals_cnt) AS deals_cnt, MAX(last_purchase_at) AS last_purchase_at "
        "FROM customer_purchases_v1 WHERE tenant_id=? AND customer_id=? AND money_kind='fact'",
        (tenant_id, customer_id),
    ).fetchone()
    stored_at = _parse_iso_datetime(row["last_purchase_at"]) if row else None
    return bool(
        row
        and float(row["total_in"] or 0) > 0
        and float(row["total_out"] or 0) == 0
        and int(row["deals_cnt"] or 0) > 0
        and stored_at
        and stored_at.date() == evidence_at.date()
    )


def _has_active_customer_access(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> bool:
    rows = con.execute(
        "SELECT event_id,event_at,event_type,source_system,source_id,source_ref,subject,text_preview,summary,direction,record_json "
        "FROM timeline_events WHERE tenant_id=? AND customer_id=? "
        "AND (superseded_by IS NULL OR superseded_by='') ORDER BY event_at DESC,event_id DESC",
        (tenant_id, customer_id),
    ).fetchall()
    for row in rows:
        event = dict(row)
        stored = _safe_json(row["record_json"])
        event["record"] = _mapping(stored.get("record"))
        event["metadata"] = _mapping(stored.get("metadata"))
        if _is_access_event(event):
            return True
    return False


def _durable_contact_risks(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> tuple[str, ...]:
    rows = con.execute(
        "SELECT event_id,event_at,event_type,source_system,source_id,source_ref,subject,text_preview,summary,direction,record_json "
        "FROM timeline_events WHERE tenant_id=? AND customer_id=? "
        "AND (superseded_by IS NULL OR superseded_by='') ORDER BY event_at DESC,event_id DESC",
        (tenant_id, customer_id),
    ).fetchall()
    risks: list[str] = []
    for row in rows:
        event = dict(row)
        stored = _safe_json(row["record_json"])
        event["record"] = _mapping(stored.get("record"))
        event["metadata"] = _mapping(stored.get("metadata"))
        text = _event_text(event)
        if hard_codes_from_text(text):
            risks.append("durable_p0_history")
        if any(phrase in text for phrase in MANAGER_OPTOUT_PHRASES):
            risks.append("durable_opt_out")
    return tuple(dict.fromkeys(risks))


def _owner50_family_rows(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    as_of: datetime,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    required = ("family_members_v1", "family_links_v1", "customer_identities", "customer_opportunities",
                "derived_signals", "timeline_events", "timeline_conflicts")
    missing = [table for table in required if not _table_exists(con, table)]
    if missing:
        raise RuntimeError(f"owner50 safety tables missing: {', '.join(missing)}")
    types = tuple(OWNER50_SIGNAL_PRIORITY)
    signals_by_family: dict[str, list[sqlite3.Row]] = {}
    for row in con.execute(
        f"""
        SELECT member.family_id, signal.*, identity.display_name, identity.primary_phone, identity.primary_email
        FROM derived_signals AS signal
        JOIN family_members_v1 AS member
          ON member.tenant_id=signal.tenant_id AND member.customer_id=signal.customer_id
        JOIN customer_identities AS identity
          ON identity.tenant_id=signal.tenant_id AND identity.customer_id=signal.customer_id
        WHERE signal.tenant_id=? AND signal.status='active'
          AND signal.signal_type IN ({','.join('?' for _ in types)})
          AND (signal.expires_at IS NULL OR signal.expires_at='' OR julianday(signal.expires_at)>=julianday(?))
        ORDER BY CASE signal.signal_type WHEN 'callback_due' THEN 0 WHEN 'client_returned' THEN 0
                                         WHEN 'deal_stalling' THEN 1 ELSE 2 END,
          signal.created_at DESC, member.family_id, signal.signal_id
        """,
        (tenant_id, *types, as_of.isoformat()),
    ):
        signals_by_family.setdefault(str(row["family_id"]), []).append(row)
    # ponytail: universe = every family in family_members_v1, not only families that
    # already have a tracked signal, so signal-less families flow through the same
    # classification and land in "Контроль" instead of silently vanishing.
    all_family_ids = {
        str(row["family_id"])
        for row in con.execute(
            "SELECT DISTINCT family_id FROM family_members_v1 WHERE tenant_id=?",
            (tenant_id,),
        )
    }
    candidates: list[dict[str, Any]] = []
    control: list[tuple[str, str, str]] = []
    for family_id in sorted(all_family_ids | signals_by_family.keys()):
        signals = signals_by_family.get(family_id, [])
        members = con.execute(
            """SELECT member.*, identity.display_name, identity.primary_phone, identity.primary_email,
                      identity.record_json AS identity_record_json
               FROM family_members_v1 AS member LEFT JOIN customer_identities AS identity
                 ON identity.tenant_id=member.tenant_id AND identity.customer_id=member.customer_id
               WHERE member.tenant_id=? AND member.family_id=? ORDER BY member.customer_id""",
            (tenant_id, family_id),
        ).fetchall()
        member_ids = tuple(str(row["customer_id"]) for row in members)
        placeholders = ",".join("?" for _ in member_ids)
        reasons: list[str] = []
        parent_members = [row for row in members if str(row["reason"]) == "exact_amo_parent_name_and_phone_or_email"]
        contact_member = members[0] if len(members) == 1 else parent_members[0] if len(parent_members) == 1 else None
        if len(members) > 1 and not parent_members:
            reasons.append("commercial_parent_missing")
        elif len(parent_members) > 1:
            reasons.append("commercial_parent_ambiguous")
        parent_brand = ""
        for member in members:
            if str(member["membership_status"]) not in {"confident", "singleton"} or str(member["confidence"]) not in {"high", "medium"}:
                reasons.append("family_ambiguous")
            record = _safe_json(member["identity_record_json"])
            metadata = _mapping(record.get("metadata"))
            no_contact = (record.get("no_contact"), record.get("opt_out"), metadata.get("no_contact"),
                          metadata.get("opt_out"), metadata.get("do_not_contact"))
            if any(str(value).strip().casefold() in {"1", "true", "yes", "да"} for value in no_contact) or str(
                metadata.get("contact_allowed", "true")
            ).casefold() in {"0", "false", "no", "нет"}:
                reasons.append("structured_no_contact")
            roles = [member["customer_id"], member["display_name"], *(metadata.get(key) for key in ("role", "kind", "type", "tags"))]
            if OWNER50_STAFF_TEST_RE.search(" ".join(_plain_values(roles))):
                reasons.append("staff_test_system")
            gate = manager_outreach_eligibility(con, tenant_id=tenant_id, customer_id=str(member["customer_id"]), as_of=as_of)
            reasons.extend(
                reason
                for reason in gate["reasons"]
                if reason in OWNER50_HARD_REASONS and reason != "brand_not_exactly_one_known"
                or reason.startswith(("active_risk_signal:", "safety_table_missing:"))
            )
            if contact_member is not None and str(member["customer_id"]) == str(contact_member["customer_id"]):
                parent_brand = str(gate.get("brand") or "")
        if contact_member is not None and not (
            _clean_text(contact_member["primary_phone"]) or _clean_text(contact_member["primary_email"])
        ):
            reasons.append("contact_missing")
        children = con.execute(
            "SELECT canonical_name,grades_json,subjects_json,brand,status,confidence "
            "FROM family_links_v1 WHERE tenant_id=? AND family_id=? ORDER BY canonical_name",
            (tenant_id, family_id),
        ).fetchall()
        if not children:
            reasons.append("child_missing")
        if any(str(row["status"]) != "confident" or str(row["confidence"]) not in {"high", "medium"} for row in children):
            reasons.append("child_ambiguous")
        child_texts = _dedupe_texts(
            f"{_clean_text(row['canonical_name'])} ({_join_list_json(row['grades_json'])}; {_join_list_json(row['subjects_json'])})"
            for row in children
            if _clean_text(row["canonical_name"])
        )
        if any(OWNER50_GRADUATE_RE.search(text) for text in child_texts):
            reasons.append("grade_11_or_graduate")
        child_brands = {str(row["brand"]).casefold() for row in children if _clean_text(row["brand"])}
        if len(child_brands) != 1 or (parent_brand and parent_brand not in child_brands):
            reasons.append("brand_ambiguous")
        if reasons:
            control.append((family_id, "excluded", ", ".join(dict.fromkeys(reasons))))
            continue
        opportunities = con.execute(
            f"SELECT opportunity_id,customer_id,opportunity_type,title,status,closed_at,record_json FROM customer_opportunities WHERE tenant_id=? "
            f"AND customer_id IN ({placeholders}) ORDER BY closed_at IS NULL DESC,opened_at DESC",
            (tenant_id, *member_ids),
        ).fetchall()
        product_offers = _product_interest_values(None, opportunities)
        historical_interests = _dedupe_texts([*product_offers, *(_clean_text(row["title"]) for row in opportunities)])
        purchase = con.execute(
            f"SELECT SUM(total_in) total_in,SUM(total_out) total_out,SUM(deals_cnt) deals_cnt,MAX(last_purchase_at) last_purchase_at "
            f"FROM customer_purchases_v1 WHERE tenant_id=? AND customer_id IN ({placeholders}) AND money_kind='fact'",
            (tenant_id, *member_ids),
        ).fetchone() if _table_exists(con, "customer_purchases_v1") else None
        payment_history = bool(purchase and float(purchase["total_in"] or 0) > float(purchase["total_out"] or 0)
                               and int(purchase["deals_cnt"] or 0) > 0)
        specific_offer = bool(product_offers)
        child_tokens = {token[:5] for row in children for token in re.findall(
            r"[a-zа-яё0-9]+", f"{_join_list_json(row['grades_json'])} {_join_list_json(row['subjects_json'])}".casefold()
        ) if token != "класс"}
        offer_tokens = {token[:5] for token in re.findall(r"[a-zа-яё0-9]+", " ".join(product_offers).casefold())}
        child_fit = bool(specific_offer and child_tokens & offer_tokens)
        rejected: list[str] = []
        for signal in signals:
            customer_id = str(signal["customer_id"])
            signal_type = str(signal["signal_type"])
            signal_record = _safe_json(signal["record_json"])
            if signal_type == "season_return_candidate" and any(
                _has_active_customer_access(con, tenant_id=tenant_id, customer_id=value) for value in member_ids
            ):
                rejected.append("active_access_or_learning")
                continue
            if signal_type == "season_return_candidate":
                purchase_at = _parse_iso_datetime(_mapping(signal_record.get("metadata")).get("last_purchase_at"))
                if purchase_at is None or not _season_purchase_matches(
                    con, tenant_id=tenant_id, customer_id=customer_id, evidence_at=purchase_at
                ):
                    rejected.append("season_purchase_not_confirmed")
                    continue
            if signal_type == "deal_stalling" and not any(
                str(row["customer_id"]) == customer_id and not row["closed_at"] and _is_active_deal(dict(row))
                for row in opportunities
            ):
                rejected.append("active_deal_missing")
                continue
            gate = manager_outreach_eligibility(con, tenant_id=tenant_id, customer_id=customer_id,
                                                signal_id=str(signal["signal_id"]), as_of=as_of)
            if not gate["eligible"]:
                rejected.extend(gate["reasons"])
                continue
            evidence_text = _clean_text(signal_record.get("evidence_text"))
            if not evidence_text:
                rejected.append("signal_evidence_text_missing")
                continue
            event = (
                con.execute(
                    "SELECT event_at,summary,source_system,source_id FROM timeline_events "
                    "WHERE tenant_id=? AND customer_id=? AND event_id=?",
                    (tenant_id, customer_id, signal["event_id"]),
                ).fetchone()
                if signal["event_id"]
                else None
            )
            evidence_at = _parse_iso_datetime(event["event_at"] if event else signal["created_at"]) or as_of
            due = signal_type == "callback_due"
            fresh_intent = signal_type == "client_returned"
            rank_reason = (
                f"tier={OWNER50_SIGNAL_PRIORITY[signal_type]}; due={int(due)}; fresh_intent={int(fresh_intent)}; "
                f"specific_offer={int(specific_offer)}; child_fit={int(child_fit)}; payment_history={int(payment_history)}"
            )
            evidence = [("signal", evidence_text, f"derived_signals:{signal['signal_id']}")]
            if event:
                evidence.append(("event", _clean_text(event["summary"]) or evidence_text, f"timeline_events:{signal['event_id']}"))
            for opportunity in opportunities:
                offer_evidence = _dedupe_texts([
                    *_product_interest_values(None, (opportunity,)), _clean_text(opportunity["title"])
                ])
                if offer_evidence:
                    evidence.append(("offer", "; ".join(offer_evidence), f"customer_opportunities:{opportunity['opportunity_id']}"))
            if child_texts:
                evidence.append(("child", "; ".join(child_texts), f"family_links_v1:{family_id}"))
            if payment_history:
                evidence.append(("payment", f"Вход: {_format_money(purchase['total_in'])}; последнее: {purchase['last_purchase_at']}", "customer_purchases_v1"))
            candidates.append({
                "family_id": family_id, "name": _clean_text(contact_member["display_name"]),
                "phone": _clean_text(contact_member["primary_phone"]), "email": _clean_text(contact_member["primary_email"]),
                "member_ids": member_ids,
                "brand": next(iter(child_brands)), "signal_type": signal_type, "evidence_text": evidence_text,
                "expires_at": _clean_text(signal["expires_at"]),
                "historical_interest": "; ".join(historical_interests[:3]),
                "offer": "Уточнить актуальный интерес; затем подобрать продукт из действующей базы знаний.",
                "children": "; ".join(child_texts),
                "payment": f"{_format_money(purchase['total_in'])}; {int(purchase['deals_cnt'] or 0)} сделок" if purchase else "",
                "rank_reason": rank_reason,
                "rank_key": (OWNER50_SIGNAL_PRIORITY[signal_type], -int(due), -int(fresh_intent),
                             -int(specific_offer), -int(child_fit), -int(payment_history), -evidence_at.timestamp(), family_id),
                "evidence": evidence,
            })
            break
        else:
            control.append((family_id, "excluded", ", ".join(dict.fromkeys(rejected)) or "no_active_outreach_signal"))
    return candidates, control


def _enrich_owner50_selected_rows(
    con: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
    *,
    tenant_id: str,
) -> None:
    freshness = source_freshness_rows(con, tenant_id=tenant_id)
    freshness_text = "; ".join(
        f"{row['source_system']}: {row.get('cursor_at') or row.get('max_event_at') or 'нет даты'}"
        for row in freshness
        if row.get("source_system") in {
            "amocrm_snapshot", "tallanto_snapshot", "tallanto_attendance",
            "mail_archive_stage2", "wappi_telegram", "wappi_max", "mango_processed_summary",
        }
    )
    for row in rows:
        member_ids = tuple(str(value) for value in row.get("member_ids") or () if value)
        placeholders = ",".join("?" for _ in member_ids)
        if not placeholders:
            continue
        messages = {}
        for direction in ("inbound", "outbound"):
            event = con.execute(
                f"SELECT event_at,source_system,summary,text_preview FROM timeline_events "
                f"WHERE tenant_id=? AND customer_id IN ({placeholders}) AND direction=? "
                "AND (superseded_by IS NULL OR superseded_by='') ORDER BY event_at DESC LIMIT 1",
                (tenant_id, *member_ids, direction),
            ).fetchone()
            messages[direction] = event
        inbound = messages.get("inbound")
        outbound = messages.get("outbound")
        source = _clean_text(inbound["source_system"] if inbound else "")
        row["channel"] = {
            "wappi_telegram": "Telegram через Wappi",
            "wappi_max": "MAX через Wappi",
            "telegram_history": "Telegram",
            "mail_archive_stage2": "Email",
            "mango_processed_summary": "Телефон",
        }.get(source, source or ("Email" if row.get("email") else "Телефон"))
        row["last_inbound"] = _clean_text((inbound["summary"] or inbound["text_preview"]) if inbound else "")
        row["last_outbound"] = _clean_text((outbound["summary"] or outbound["text_preview"]) if outbound else "")
        attendance = con.execute(
            f"SELECT MAX(event_at) FROM timeline_events WHERE tenant_id=? AND customer_id IN ({placeholders}) "
            "AND event_type='tallanto_attendance' AND match_status IN ('strong_unique','manual') "
            "AND json_extract(record_json, '$.record.attendance_confirmed') = 1 "
            "AND (superseded_by IS NULL OR superseded_by='')",
            (tenant_id, *member_ids),
        ).fetchone()
        row["attendance"] = f"Последнее подтверждённое посещение: {attendance[0]}" if attendance and attendance[0] else "Нет подтверждённых посещений"
        row["next_step"] = {
            "callback_due": "Связаться по просроченному обещанию и закрыть вопрос клиента.",
            "client_returned": "Ответить на последнее входящее сообщение по его реальному вопросу.",
            "deal_stalling": "Уточнить, актуален ли интерес, и согласовать следующий шаг.",
            "season_return_candidate": "Уточнить текущую учебную задачу ребёнка и интерес к новому сезону.",
        }.get(str(row.get("signal_type")), "Проверить историю и согласовать следующий шаг.")
        row["freshness"] = freshness_text


def _full_dossier_segment_customer_ids(con: sqlite3.Connection, *, tenant_id: str, limit: int) -> list[str]:
    sql = """
        SELECT e.customer_id
        FROM timeline_events e
        JOIN customer_identities ci
          ON ci.tenant_id=e.tenant_id AND ci.customer_id=e.customer_id
        WHERE e.tenant_id = ?
          AND e.customer_id IS NOT NULL
          AND e.customer_id != ''
          AND (e.superseded_by IS NULL OR e.superseded_by = '')
          AND ci.identity_status='strong'
          AND COALESCE(json_array_length(json_extract(ci.record_json, '$.metadata.brands')), 0)=1
          AND LOWER(json_extract(ci.record_json, '$.metadata.brands[0]')) IN ('foton','unpk')
        GROUP BY e.customer_id
        HAVING SUM(e.event_type = 'mango_call' AND e.match_status = 'strong_unique') > 0
           AND SUM(e.event_type = 'email_message') > 0
        ORDER BY MAX(e.event_at) DESC, e.customer_id
    """
    params: tuple[Any, ...]
    if limit > 0:
        sql += " LIMIT ?"
        params = (tenant_id, int(limit))
    else:
        params = (tenant_id,)
    return [str(row[0]) for row in con.execute(sql, params).fetchall()]


def _full_dossier_segment_count(con: sqlite3.Connection, *, tenant_id: str) -> int:
    row = con.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT e.customer_id
          FROM timeline_events e
          JOIN customer_identities ci
            ON ci.tenant_id=e.tenant_id AND ci.customer_id=e.customer_id
          WHERE e.tenant_id = ?
            AND e.customer_id IS NOT NULL
            AND e.customer_id != ''
            AND (e.superseded_by IS NULL OR e.superseded_by = '')
            AND ci.identity_status='strong'
            AND COALESCE(json_array_length(json_extract(ci.record_json, '$.metadata.brands')), 0)=1
            AND LOWER(json_extract(ci.record_json, '$.metadata.brands[0]')) IN ('foton','unpk')
          GROUP BY e.customer_id
          HAVING SUM(e.event_type = 'mango_call' AND e.match_status = 'strong_unique') > 0
             AND SUM(e.event_type = 'email_message') > 0
        )
        """,
        (tenant_id,),
    ).fetchone()
    return int(row[0] or 0) if row else 0


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _read_json(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _source_freshness(con: sqlite3.Connection, *, tenant_id: str = "foton") -> list[Mapping[str, Any]]:
    if not _table_exists(con, "timeline_events"):
        return []
    return source_freshness_rows(
        con,
        tenant_id=tenant_id,
        expected_sources=MANAGER_REQUIRED_SOURCE_SYSTEMS,
    )


def _actuality_header(freshness: Sequence[Mapping[str, Any]], reconcile: Mapping[str, Any]) -> str:
    cursor_text = "; ".join(
        f"{_display_freshness_source(row.get('source_system'))}={row.get('cursor_at') or 'нет курсора'}"
        for row in freshness[:8]
    ) or "нет данных"
    cursor_checked_text = "; ".join(
        f"{_display_freshness_source(row.get('source_system'))}={row.get('cursor_updated_at') or 'нет проверки'}"
        for row in freshness[:8]
    ) or "нет данных"
    event_text = "; ".join(
        f"{_display_freshness_source(row.get('source_system'))}={row.get('max_event_at')}"
        for row in freshness[:8]
    ) or "нет данных"
    imported_text = "; ".join(
        f"{_display_freshness_source(row.get('source_system'))}={row.get('imported_at') or 'нет успешного импорта'}"
        for row in freshness[:8]
    ) or "нет данных"
    status = str(reconcile.get("status") or "")
    if status == "checked":
        reconcile_text = (
            f"{reconcile.get('generated_at')}; "
            f"{reconcile.get('customers_changed')} расхождений из {reconcile.get('customers_checked')}; "
            f"snapshot_stale={reconcile.get('snapshot_stale')}"
        )
    elif reconcile:
        reconcile_text = f"не проводилась ({reconcile.get('reason') or status or 'unknown'})"
    else:
        reconcile_text = "не проводилась"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (
        f"Данные: cursor_at по источникам: {cursor_text}; "
        f"cursor checked_at отдельно: {cursor_checked_text}; "
        f"imported_at отдельно: {imported_text}; "
        f"max event_at отдельно: {event_text}; собрано {generated_at}; "
        f"сверка с живым AMO: {reconcile_text}"
    )


def _display_freshness_source(source: Any) -> str:
    mapping = {
        "amocrm_snapshot": "AMO снимок",
        "amocrm_event": "AMO события",
        "amocrm_price_readonly": "AMO цены",
        "mango_processed_summary": "сводки звонков",
        "mail_archive": "архив почты",
        "mail_archive_stage2": "письма",
        "tallanto_crm_call": "Tallanto платежи",
        "master_contacts_snapshot": "сводка контактов",
        "tallanto_snapshot": "Tallanto снимок",
        "telegram_history": "Telegram история",
    }
    return mapping.get(str(source or ""), _clean_text(source) or "неизвестный источник")


def _family_rows(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    active_brand: str,
) -> list[DossierRow]:
    if not _table_exists(con, "family_links_v1"):
        return []
    rows = con.execute(
        """
        SELECT canonical_name, name_variants_json, grades_json, subjects_json, brand, status, confidence, reason
        FROM family_links_v1
        WHERE tenant_id = ? AND customer_id = ?
        ORDER BY status, confidence DESC, canonical_name
        """,
        (tenant_id, customer_id),
    ).fetchall()
    result: list[DossierRow] = []
    for row in rows:
        variants = _join_list_json(row["name_variants_json"])
        grades = _join_list_json(row["grades_json"])
        subjects = _join_list_json(row["subjects_json"])
        quality = f"{row['status']}/{row['confidence']}"
        text = f"{_clean_text(row['canonical_name'])}"
        details = []
        if variants and variants != text:
            details.append(f"варианты: {variants}")
        if grades:
            details.append(f"класс: {grades}")
        if subjects:
            details.append(f"предметы: {subjects}")
        if row["brand"]:
            details.append(f"бренд: {row['brand']}")
            if active_brand and str(row["brand"]).casefold() != active_brand.casefold():
                details.append("исторический другой бренд — не переносить в текущее предложение")
        if str(row["status"]) != "confident" or str(row["confidence"]) not in {"high", "medium"}:
            details.append("уточнить семейную связь")
        if details:
            text += " (" + "; ".join(details) + ")"
        result.append(DossierRow("Семья", text, f"family_links_v1:{quality}:{row['reason']}"))
    return result


def _money_rows(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> list[DossierRow]:
    if not _table_exists(con, "customer_purchases_v1"):
        return []
    rows = con.execute(
        """
        SELECT period, money_kind, total_in, total_out, deals_cnt, last_purchase_at, computability
        FROM customer_purchases_v1
        WHERE tenant_id = ? AND customer_id = ?
        ORDER BY period, money_kind
        """,
        (tenant_id, customer_id),
    ).fetchall()
    result: list[DossierRow] = []
    labels = {"fact": "факт оплат", "plan": "план сделок"}
    for row in rows:
        kind = str(row["money_kind"] or "plan")
        label = labels.get(kind, kind)
        text = (
            f"{label}, период {row['period']}: вход {_format_money(row['total_in'])}; "
            f"списания/расход {_format_money(row['total_out'])}; сделок {int(row['deals_cnt'] or 0)}"
        )
        if row["last_purchase_at"]:
            text += f"; последнее событие {row['last_purchase_at']}"
        if row["computability"]:
            text += f"; вычислимость {row['computability']}"
        result.append(DossierRow("Деньги", text, "customer_purchases_v1"))
    return result


def _signal_rows(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> list[DossierRow]:
    if not _table_exists(con, "derived_signals"):
        return []
    rows = con.execute(
        """
        SELECT signal_type, severity, expires_at, confidence, requires_manager_review, record_json
        FROM derived_signals
        WHERE tenant_id = ? AND customer_id = ? AND status = 'active'
          AND (expires_at IS NULL OR expires_at = '' OR julianday(expires_at) >= julianday('now'))
        ORDER BY CASE severity
                   WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4
                 END,
                 expires_at, signal_type
        LIMIT 12
        """,
        (tenant_id, customer_id),
    ).fetchall()
    result: list[DossierRow] = []
    for row in rows:
        record = _safe_json(row["record_json"])
        action = _clean_text(record.get("recommended_action") or record.get("action") or "")
        evidence = _clean_text(record.get("evidence_text") or record.get("reason") or "")
        label = _signal_label(str(row["signal_type"] or ""))
        parts = [label]
        if row["severity"]:
            parts.append(f"важность: {row['severity']}")
        if row["expires_at"]:
            parts.append(f"до: {row['expires_at']}")
        if action and _meaningful_next_step(action):
            parts.append(f"рекомендация: {action}")
        if evidence:
            parts.append(f"основание: {evidence}")
        result.append(DossierRow("Сигналы", "; ".join(parts), f"derived_signals:{row['signal_type']}"))
    return result


def _next_step_from_signals(signals: Sequence[DossierRow]) -> str:
    for signal in signals:
        match = re.search(r"рекомендация:\s*([^;]+)", signal.text)
        if not match:
            continue
        value = _clean_text(match.group(1))
        if _meaningful_next_step(value):
            return value
    return ""


def _next_step_for_dossier(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    signals: Sequence[DossierRow],
) -> tuple[str, str]:
    rows = con.execute(
        """
        SELECT event_id, customer_id, event_at, event_type, source_system, source_id,
               source_ref, subject, summary, text_preview, direction, record_json
        FROM timeline_events
        WHERE tenant_id = ? AND customer_id = ?
          AND (superseded_by IS NULL OR superseded_by = '')
          AND (event_type != 'mango_call' OR match_status = 'strong_unique')
        ORDER BY event_at DESC, event_id DESC
        LIMIT 500
        """,
        (tenant_id, customer_id),
    ).fetchall()
    events: list[Mapping[str, Any]] = []
    for row in rows:
        stored = _safe_json(row["record_json"])
        event = dict(row)
        event["record"] = dict(stored["record"]) if isinstance(stored.get("record"), Mapping) else {}
        event["metadata"] = dict(stored["metadata"]) if isinstance(stored.get("metadata"), Mapping) else {}
        event["stage_before"] = stored.get("stage_before")
        event["stage_after"] = stored.get("stage_after")
        events.append(event)
    conflicts: list[Mapping[str, Any]] = []
    if _table_exists(con, "timeline_conflicts"):
        customer_refs = set(customer_entity_ref_values(customer_id))
        for row in con.execute(
            "SELECT conflict_type, status, record_json FROM timeline_conflicts WHERE tenant_id = ? AND status = 'open'",
            (tenant_id,),
        ).fetchall():
            record = dict(_safe_json(row["record_json"]))
            entity_refs = {str(item) for item in (record.get("entity_refs") or ())}
            if customer_refs.isdisjoint(entity_refs):
                continue
            record.setdefault("conflict_type", row["conflict_type"])
            record.setdefault("status", row["status"])
            conflicts.append(record)
    resolved = resolve_customer_next_step(
        events,
        readiness={"open_conflicts": len(conflicts)},
        conflicts=conflicts,
        customer_id=customer_id,
    )
    if resolved.status == NEXT_STEP_STATUS_ACTIVE and _meaningful_next_step(resolved.action):
        return resolved.display_text, "timeline_events"
    if resolved.status != NEXT_STEP_STATUS_EMPTY:
        return "", ""
    fallback = _next_step_from_signals(signals)
    return fallback, "derived_signals" if fallback else ""


def _meaningful_next_step(value: str) -> bool:
    text = value.casefold()
    if not text or text in {"уточнить у менеджера", "связаться с клиентом", "позвонить клиенту"}:
        return False
    if "посмотреть историю" in text:
        return False
    return len(text.split()) >= 3


def _objection_rows(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> list[DossierRow]:
    if not _table_exists(con, "customer_objections_v1"):
        return []
    rows = con.execute(
        """
        SELECT source_channel, objection_type, quote_preview, budget_hint_rub, price_sensitivity, confidence, speaker
        FROM customer_objections_v1
        WHERE tenant_id = ?
          AND customer_id = ?
          AND speaker = 'client'
        ORDER BY confidence DESC, extracted_at DESC
        LIMIT 12
        """,
        (tenant_id, customer_id),
    ).fetchall()
    result: list[DossierRow] = []
    for row in rows:
        quote = _safe_marker_phrase(row["quote_preview"], re.compile(r".+"))
        text = f"{row['objection_type']}: {quote}"
        if row["budget_hint_rub"]:
            text += f"; бюджет: {_format_money(row['budget_hint_rub'])}"
        if row["price_sensitivity"]:
            text += f"; чувствительность к цене: {row['price_sensitivity']}"
        result.append(DossierRow("Возражения", text, f"customer_objections_v1:{row['source_channel']}:{row['confidence']}"))
    return result


def _chronology_rows(con: sqlite3.Connection, *, tenant_id: str, customer_id: str, limit: int) -> list[DossierRow]:
    customer_ids = _family_scope_customer_ids(
        con,
        tenant_id=tenant_id,
        customer_id=customer_id,
    )
    placeholders = ",".join("?" for _ in customer_ids)
    rows = con.execute(
        f"""
        SELECT event.event_at, event.event_type, event.source_system, event.subject,
               event.summary, event.text_preview, event.record_json, event.customer_id,
               identity.display_name AS source_customer_name
        FROM timeline_events AS event
        LEFT JOIN customer_identities AS identity
          ON identity.tenant_id = event.tenant_id AND identity.customer_id = event.customer_id
        WHERE event.tenant_id = ?
          AND event.customer_id IN ({placeholders})
          AND (event.superseded_by IS NULL OR event.superseded_by = '')
          AND (event.event_type != 'mango_call' OR event.match_status = 'strong_unique')
        ORDER BY event.event_at DESC, event.event_id DESC
        LIMIT ?
        """,
        (tenant_id, *customer_ids, int(limit)),
    ).fetchall()
    result: list[DossierRow] = []
    for row in rows:
        summary = _event_summary_for_manager(row)
        if not summary:
            continue
        text = f"{row['event_at']} [{row['event_type']}] {summary}"
        if len(customer_ids) > 1:
            member = _clean_text(row["source_customer_name"]) or str(row["customer_id"])
            text = f"{text} [карточка: {member}]"
        result.append(
            DossierRow(
                "Хронология",
                text,
                f"{row['source_system']}:{row['customer_id']}",
            )
        )
    return result


def _family_scope_customer_ids(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
) -> tuple[str, ...]:
    if not _table_exists(con, "family_members_v1"):
        return (customer_id,)
    root = con.execute(
        """
        SELECT family_id, membership_status
        FROM family_members_v1
        WHERE tenant_id = ? AND customer_id = ?
        """,
        (tenant_id, customer_id),
    ).fetchone()
    if root is None or str(root["membership_status"] or "") not in {"confident", "singleton"}:
        return (customer_id,)
    members = tuple(
        str(row["customer_id"])
        for row in con.execute(
            """
            SELECT customer_id
            FROM family_members_v1
            WHERE tenant_id = ? AND family_id = ?
              AND membership_status IN ('confident', 'singleton')
            ORDER BY customer_id
            """,
            (tenant_id, root["family_id"]),
        )
    )
    if customer_id not in members or not 1 <= len(members) <= 8:
        return (customer_id,)
    return members


def _event_summary_for_manager(row: sqlite3.Row) -> str:
    event_type = str(row["event_type"] or "")
    subject = _clean_text(row["subject"])
    summary = _clean_text(row["summary"]) or _clean_text(row["text_preview"])
    if event_type == "email_message":
        if EMAIL_SUMMARY_REVIEW_NEEDED_RE.search(summary):
            summary = f"Письмо «{subject or 'без темы'}»: полный текст в базе."
        elif summary:
            summary = f"{summary} Полный текст в базе."
        elif subject:
            summary = f"Письмо «{subject}»: полный текст в базе."
    elif event_type == "tallanto_attendance" and subject:
        summary = f"Списание за занятие: {subject}."
    return summary


def _signal_label(signal_type: str) -> str:
    labels = {
        "client_returned": "клиент вернулся",
        "callback_due": "нужно перезвонить",
        "deal_stalling": "сделка зависла",
        "hot_streak": "горячая серия касаний",
        "season_return_candidate": "сезонный возврат",
    }
    return labels.get(signal_type, signal_type)


def _format_money(value: Any) -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        amount = 0.0
    return f"{amount:,.0f} руб.".replace(",", " ")


def _join_list_json(raw: Any) -> str:
    value = _json_any(raw) if isinstance(raw, str) else raw
    if isinstance(value, list):
        return ", ".join(_clean_text(item) for item in value if _clean_text(item))
    return _clean_text(value)


def _json_any(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _product_interest_values(customer_record_json: str | None, opportunities: Sequence[sqlite3.Row]) -> tuple[str, ...]:
    values: list[str] = []
    values.extend(_recursive_product_values(_safe_json(customer_record_json)))
    for row in opportunities:
        payload = _safe_json(row["record_json"])
        values.extend(_recursive_product_values(payload))
        values.extend(_plain_values(payload.get("product_context") if isinstance(payload, Mapping) else None))
    return tuple(_dedupe_texts(_safe_phrase(value) for value in values if _safe_phrase(value)))


def _recursive_product_values(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).strip().casefold()
            if key_text in PRODUCT_KEYS:
                result.extend(_plain_values(nested))
            else:
                result.extend(_recursive_product_values(nested))
    elif isinstance(value, list):
        for item in value:
            result.extend(_recursive_product_values(item))
    return result


def _plain_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in (
            "title", "name", "course", "group", "filial", "subject", "subject_name",
            "format", "class", "value",
        ):
            result.extend(_plain_values(value.get(key)))
        return result
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_plain_values(item))
        return result
    return []


def _markers_from_client_text(text: str, pattern: re.Pattern[str], *, kind: str, label: str, source: str) -> list[DossierMarker]:
    markers: list[DossierMarker] = []
    seen: set[str] = set()
    for sentence in _sentences(text):
        if not pattern.search(sentence):
            continue
        phrase = _safe_marker_phrase(sentence, pattern)
        if kind == "interest" and not INTEREST_CONTEXT_RE.search(phrase):
            continue
        if not phrase or phrase.casefold() in seen:
            continue
        seen.add(phrase.casefold())
        markers.append(DossierMarker(kind=kind, text=f"{label}: {phrase}", source=source))
    return markers


def _sentences(text: str) -> list[str]:
    compact = _clean_text(text)
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?…])\s+|\n+", compact)
    return [part.strip(" -–—\t") for part in parts if part.strip(" -–—\t")]


def _safe_phrase(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = CONTACT_RE.sub("[contact]", text)
    text = text.strip(" .;:,")
    return text[:220]


def _safe_marker_phrase(value: Any, pattern: re.Pattern[str]) -> str:
    text = CONTACT_RE.sub("[contact]", _clean_text(value))
    if not text:
        return ""
    text = _trim_to_marker_window(text, pattern)
    text = SPEECH_FILLER_RE.sub("", text).strip(" -–—,.;:")
    text = _collapse_repeated_words(text)
    text = _trim_to_first_meaningful_clause(text)
    text = _trim_to_word_boundary(text, 160)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?…":
        text += "."
    return text


def _trim_to_marker_window(text: str, pattern: re.Pattern[str], *, after: int = 110) -> str:
    match = _select_marker_match(text, pattern)
    if match is None:
        return text
    start = _marker_window_start(text, match.start())
    end = min(len(text), match.end() + after)
    window = text[start:end].strip(" -–—,.;:")
    return window


def _select_marker_match(text: str, pattern: re.Pattern[str]) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    for index, match in enumerate(matches):
        token = match.group(0).casefold()
        tail = text[match.end() : match.end() + 80].casefold()
        if token.startswith("хотел") and "звон" in tail and index + 1 < len(matches):
            continue
        return match
    return matches[0]


def _marker_window_start(text: str, marker_start: int) -> int:
    prefix = text[:marker_start]
    words = list(re.finditer(r"\b[А-Яа-яA-Za-zЁё]{1,8}\b", prefix))
    if not words:
        return marker_start
    previous = words[-1]
    gap = prefix[previous.end() :].strip(" ,.;:–—-")
    if gap:
        return marker_start
    if previous.group(0).casefold() in {"нас", "нам", "мы", "мне", "меня"}:
        return previous.start()
    return marker_start


def _trim_to_first_meaningful_clause(text: str) -> str:
    boundary = SPEECH_CLAUSE_BOUNDARY_RE.search(text)
    if boundary and boundary.start() >= 12:
        text = text[: boundary.start()]
    text = re.split(r"\s+(?:но|а|и)\s+", text, maxsplit=1, flags=re.I)[0]
    return text.strip(" -–—,.;:")


def _trim_to_word_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.strip(" -–—,.;:")
    chunk = text[:limit].rstrip()
    cut = max(chunk.rfind(" "), chunk.rfind(","), chunk.rfind(";"), chunk.rfind("."))
    if cut >= int(limit * 0.55):
        chunk = chunk[:cut]
    return chunk.strip(" -–—,.;:")


def _collapse_repeated_words(text: str) -> str:
    parts = text.split()
    result: list[str] = []
    previous = ""
    for part in parts:
        key = part.strip(" ,.;:!?").casefold()
        if key and key == previous:
            continue
        result.append(part)
        previous = key
    return " ".join(result)


def _clean_text(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(value or "").replace("\u00a0", " ")).strip()


def _safe_json(value: str | None) -> Mapping[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _dedupe_texts(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _dedupe_markers(values: Sequence[DossierMarker], *, limit: int) -> list[DossierMarker]:
    result: list[DossierMarker] = []
    seen: set[str] = set()
    for item in values:
        key = item.text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _write_owner50_workbook(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    control: Sequence[tuple[str, str, str]],
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    outreach = wb.active
    outreach.title = "Кому писать"
    evidence = wb.create_sheet("Доказательства")
    checks = wb.create_sheet("Контроль")
    outreach.append(
        (
            "Ранг",
            "family_id",
            "Бренд",
            "Кому",
            "Телефон",
            "Email",
            "Канал",
            "Сигнал",
            "Основание",
            "Срок",
            "Исторический интерес",
            "Актуальное предложение",
            "Следующий шаг",
            "Ребёнок/класс",
            "Последнее входящее",
            "Последний ответ",
            "Посещения",
            "Оплаты",
            "Свежесть источников",
            "Формула ранга",
        )
    )
    evidence.append(("family_id", "Ранг", "Тип", "Доказательство", "Источник"))
    checks.append(("family_id", "Статус", "Причина/контроль"))
    for row in rows:
        outreach.append(
            (
                row["rank"],
                row["family_id"],
                row["brand"],
                row["name"],
                row["phone"],
                row["email"],
                row.get("channel", ""),
                row["signal_type"],
                row["evidence_text"],
                row["expires_at"],
                row.get("historical_interest", ""),
                row["offer"],
                row.get("next_step", ""),
                row["children"],
                row.get("last_inbound", ""),
                row.get("last_outbound", ""),
                row.get("attendance", ""),
                row["payment"],
                row.get("freshness", ""),
                row["rank_reason"],
            )
        )
        for kind, text, source in row["evidence"]:
            evidence.append((row["family_id"], row["rank"], kind, text, source))
    for item in sorted(control):
        checks.append(item)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for column in ws.columns:
            letter = column[0].column_letter
            ws.column_dimensions[letter].width = min(80, max(12, *(len(str(cell.value or "")) for cell in column)))
    wb.save(path)


def _write_workbook(path: Path, dossiers: Sequence[CustomerDossier]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    overview = wb.active
    overview.title = "Оглавление"
    overview.append(("customer_id", "Имя", "Бренд", "Семья", "Сигналы", "Следующий шаг", "Интересов", "Болей", "Возражений", "Хронология"))
    overview.freeze_panes = "A2"
    for cell in overview[1]:
        cell.font = Font(bold=True)
    for index, dossier in enumerate(dossiers, start=1):
        sheet_name = f"Клиент {index}"
        overview.append(
            (
                dossier.customer_id,
                dossier.display_name,
                dossier.brand,
                len(dossier.family),
                len(dossier.signals),
                dossier.next_step,
                len(dossier.interests),
                len(dossier.pains),
                len(dossier.objections),
                len(dossier.chronology),
            )
        )
        ws = wb.create_sheet(sheet_name)
        ws.append(("Раздел", "Значение", "Откуда"))
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
        if dossier.actuality_header:
            ws.append(("Актуальность", dossier.actuality_header, _display_source("timeline_events/reconcile")))
        ws.append(("Кто", dossier.display_name, _display_source("customer_identities")))
        ws.append(("Бренд", dossier.brand or "Не определён однозначно", _display_source("customer_identities")))
        ws.append(("Контакт", f"{dossier.phone} {dossier.email}".strip(), _display_source("customer_identities")))
        for row in dossier.family:
            ws.append((row.section, row.text, _display_source(row.source)))
        for row in dossier.money:
            ws.append((row.section, row.text, _display_source(row.source)))
        for row in dossier.signals:
            ws.append((row.section, row.text, _display_source(row.source)))
        ws.append(
            (
                "Следующий шаг",
                dossier.next_step or "Не определён: менеджеру нужно выбрать действие после проверки истории.",
                _display_source(dossier.next_step_source) if dossier.next_step else "Требует решения менеджера",
            )
        )
        for row in dossier.objections:
            ws.append((row.section, row.text, _display_source(row.source)))
        for item in dossier.interests:
            ws.append(("Интересы", item.text, _display_source(item.source)))
        for item in dossier.pains:
            ws.append(("Боли", item.text, _display_source(item.source)))
        for row in dossier.chronology:
            ws.append((row.section, row.text, _display_source(row.source)))
        for column, width in {"A": 18, "B": 90, "C": 28}.items():
            ws.column_dimensions[column].width = width
    wb.save(path)


def _display_source(source: str) -> str:
    text = str(source or "")
    if text.startswith("family_links_v1"):
        return "Семейная карта"
    if text.startswith("customer_purchases_v1"):
        return "Деньги из staging"
    if text.startswith("derived_signals"):
        return "Сигнал Customer Timeline"
    if text.startswith("customer_objections_v1"):
        return "Клиентское возражение"
    if text.startswith("mango_call"):
        return "Клиентская реплика из звонка"
    if text == "products_of_interest":
        return "Данные клиента/сделки"
    mapping = {
        "timeline_events/reconcile": "Шапка актуальности",
        "customer_identities": "Карточка клиента",
        "mango_processed_summary": "Сводка звонка",
        "mail_archive_stage2": "Письмо",
        "mail_archive": "Письмо",
        "amocrm_snapshot": "AMO read-only",
        "amocrm_price_readonly": "AMO read-only",
        "amocrm_event": "AMO read-only",
        "master_contacts_snapshot": "Сводка контакта",
        "tallanto_snapshot": "Tallanto staging",
        "tallanto_crm_call": "Tallanto staging",
        "telegram_history": "Telegram история",
    }
    return mapping.get(text, text or "Источник не указан")
