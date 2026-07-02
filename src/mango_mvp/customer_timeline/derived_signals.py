from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.contracts import DerivedSignal, SignalSeverity, SignalStatus
from mango_mvp.customer_timeline.ids import normalize_key, optional_text, require_text, require_timezone, stable_signal_id
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


DERIVED_SIGNAL_RECOMPUTE_SCHEMA_VERSION = "customer_timeline_derived_signals_v1"
SIGNAL_RULES_VERSION = "sg_v1"
PAID_NO_ACCESS_SIGNAL = "paid_no_access"
HOT_LEAD_SILENT_SIGNAL = "hot_lead_silent_7d"
DUPLICATE_CONTACT_SIGNAL = "duplicate_contact"
CLIENT_RETURNED_SIGNAL = "client_returned"
CALLBACK_DUE_SIGNAL = "callback_due"
DEAL_STALLING_SIGNAL = "deal_stalling"
HOT_STREAK_SIGNAL = "hot_streak"
SEASON_RETURN_SIGNAL = "season_return_candidate"
DEFAULT_HOT_LEAD_SILENCE_DAYS = 7
DEFAULT_RETURN_SILENCE_DAYS = 60
DEFAULT_CALLBACK_DUE_DAYS = 3
DEFAULT_DEAL_STALL_DAYS = 14
DEFAULT_HOT_STREAK_DAYS = 7
DEFAULT_SEASON_RETURN_DAYS = 180
SIGNAL_TTL_DAYS: Mapping[str, int] = {
    PAID_NO_ACCESS_SIGNAL: 90,
    HOT_LEAD_SILENT_SIGNAL: 30,
    DUPLICATE_CONTACT_SIGNAL: 180,
    CLIENT_RETURNED_SIGNAL: 30,
    CALLBACK_DUE_SIGNAL: 14,
    DEAL_STALLING_SIGNAL: 30,
    HOT_STREAK_SIGNAL: 14,
    SEASON_RETURN_SIGNAL: 30,
}
MANAGED_SIGNAL_TYPES = (PAID_NO_ACCESS_SIGNAL, HOT_LEAD_SILENT_SIGNAL, DUPLICATE_CONTACT_SIGNAL)
SG_V1_SIGNAL_TYPES = (
    CLIENT_RETURNED_SIGNAL,
    CALLBACK_DUE_SIGNAL,
    DEAL_STALLING_SIGNAL,
    HOT_STREAK_SIGNAL,
    SEASON_RETURN_SIGNAL,
)

INTEREST_MARKERS = (
    "заявк",
    "интерес",
    "интересует",
    "хочу",
    "подбер",
    "стоим",
    "цен",
    "курс",
    "обуч",
    "занят",
    "запис",
    "пробн",
    "егэ",
    "огэ",
    "летн",
)
CALLBACK_MARKERS = ("перезвон", "свяж", "напиш", "созвон", "позвон")
CALLBACK_PROMISE_MARKERS = (
    "перезвоню",
    "перезвоним",
    "позвоню",
    "позвоним",
    "напишу",
    "напишем",
    "свяжемся",
    "созвонимся",
)
ACTIVE_DEAL_STATUSES = ("актив", "observed", "open", "new", "в работе", "первичный контакт", "переговор")
PAYMENT_IN_MARKERS = ("in", "поступ", "оплат", "приход", "зачисл")
PAYMENT_OUT_MARKERS = ("out", "refund", "возврат", "отмен", "cancel")
ACTIVE_ABONEMENT_MARKERS = ("active", "актив", "действ", "открыт")
INACTIVE_ABONEMENT_MARKERS = ("closed", "закры", "отмен", "cancel", "expired", "истек")
DUPLICATE_CONFLICT_TYPES = {
    "ambiguous_identity",
    "shared_amo_contact",
    "shared_amo_contact_across_customers",
    "shared_amo_lead",
    "shared_amo_lead_across_customers",
}


@dataclass(frozen=True)
class DerivedSignalInputs:
    tenant_id: str
    customer_id: str
    events: Sequence[Mapping[str, Any]]
    conflicts: Sequence[Mapping[str, Any]] = ()
    as_of: Optional[datetime] = None
    hot_lead_silence_days: int = DEFAULT_HOT_LEAD_SILENCE_DAYS

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "customer_id", require_text(self.customer_id, "customer_id"))
        if self.as_of is not None:
            require_timezone(self.as_of, "as_of")
        if self.hot_lead_silence_days <= 0:
            raise ValueError("hot_lead_silence_days must be positive")
        object.__setattr__(self, "events", tuple(dict(item) for item in self.events))
        object.__setattr__(self, "conflicts", tuple(dict(item) for item in self.conflicts))


@dataclass(frozen=True)
class CustomerSignalRecomputeResult:
    tenant_id: str
    customer_id: str
    apply: bool
    signals: tuple[DerivedSignal, ...]
    write_status_counts: Mapping[str, int]
    status_counts: Mapping[str, int]
    signal_type_counts: Mapping[str, int]

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": DERIVED_SIGNAL_RECOMPUTE_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "customer_id": self.customer_id,
            "apply": self.apply,
            "signals": [signal.to_json_dict() for signal in self.signals],
            "write_status_counts": dict(self.write_status_counts),
            "status_counts": dict(self.status_counts),
            "signal_type_counts": dict(self.signal_type_counts),
        }


def derive_active_signals(inputs: DerivedSignalInputs) -> tuple[DerivedSignal, ...]:
    events = tuple(sorted(inputs.events, key=lambda item: (_event_at(item).isoformat(), str(item.get("event_id") or ""))))
    signals: list[DerivedSignal] = []
    signals.extend(_derive_paid_no_access(inputs.tenant_id, inputs.customer_id, events))
    hot_lead = _derive_hot_lead_silent(inputs.tenant_id, inputs.customer_id, events, inputs.as_of, inputs.hot_lead_silence_days)
    if hot_lead is not None:
        signals.append(hot_lead)
    signals.extend(_derive_duplicate_contact(inputs.tenant_id, inputs.customer_id, inputs.conflicts))
    return tuple(signals)


def recompute_customer_signals(
    store: Any,
    tenant_id: str,
    customer_id: str,
    *,
    as_of: datetime,
    apply: bool = False,
    hot_lead_silence_days: int = DEFAULT_HOT_LEAD_SILENCE_DAYS,
    actor: str = "derived_signal_recompute",
) -> CustomerSignalRecomputeResult:
    require_timezone(as_of, "as_of")
    tenant = normalize_key(tenant_id, "tenant_id")
    customer = require_text(customer_id, "customer_id")
    events = _list_all_customer_events(store, tenant, customer)
    conflicts = store.list_conflicts_by_customer(tenant, customer, limit=500)
    current = store.list_signals_by_customer(tenant, customer, signal_types=MANAGED_SIGNAL_TYPES, limit=500)
    current_by_id = {require_text(item.get("signal_id"), "signal_id"): item for item in current}
    active_candidates = derive_active_signals(
        DerivedSignalInputs(
            tenant_id=tenant,
            customer_id=customer,
            events=events,
            conflicts=conflicts,
            as_of=as_of,
            hot_lead_silence_days=hot_lead_silence_days,
        )
    )

    desired: list[DerivedSignal] = []
    desired_ids: set[str] = set()
    for candidate in active_candidates:
        status = SignalStatus.STALE if candidate.expires_at and candidate.expires_at <= as_of else SignalStatus.ACTIVE
        existing = current_by_id.get(candidate.signal_id)
        desired_signal = _replace_signal_lifecycle(
            candidate,
            signal_id=candidate.signal_id,
            status=status,
            created_at=_existing_created_at(existing) or candidate.created_at,
            metadata_extra={"lifecycle_reason": "candidate_active" if status == SignalStatus.ACTIVE else "candidate_expired"},
        )
        desired.append(desired_signal)
        desired_ids.add(require_text(desired_signal.signal_id, "signal_id"))

    for signal_id, existing in current_by_id.items():
        if signal_id in desired_ids:
            continue
        existing_signal = _signal_from_payload(existing)
        status = SignalStatus.STALE if existing_signal.expires_at and existing_signal.expires_at <= as_of else SignalStatus.RESOLVED
        desired.append(
            _replace_signal_lifecycle(
                existing_signal,
                signal_id=signal_id,
                status=status,
                created_at=existing_signal.created_at,
                metadata_extra={
                    "lifecycle_reason": "expired" if status == SignalStatus.STALE else "predicate_resolved",
                    "lifecycle_as_of": as_of.isoformat(),
                },
            )
        )

    write_status_counts: dict[str, int] = {}
    if apply:
        for signal in desired:
            result = store.upsert_signal(signal, actor=actor)
            write_status_counts[result.status] = write_status_counts.get(result.status, 0) + 1

    return CustomerSignalRecomputeResult(
        tenant_id=tenant,
        customer_id=customer,
        apply=bool(apply),
        signals=tuple(sorted(desired, key=lambda item: (item.signal_type, item.signal_id or ""))),
        write_status_counts=write_status_counts,
        status_counts=_count_by_signal_attr(desired, "status"),
        signal_type_counts=_count_by_signal_attr(desired, "signal_type"),
    )


def derive_sg_v1_signals(
    *,
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]] = (),
    purchases: Mapping[str, Any] | None = None,
    as_of: datetime,
) -> tuple[DerivedSignal, ...]:
    require_timezone(as_of, "as_of")
    tenant = normalize_key(tenant_id, "tenant_id")
    customer = require_text(customer_id, "customer_id")
    ordered = tuple(
        sorted(
            (event for event in events if _event_at(event) <= as_of),
            key=lambda item: (_event_at(item), str(item.get("event_id") or "")),
        )
    )
    signals: list[DerivedSignal] = []
    returned = _derive_client_returned(tenant, customer, ordered, as_of)
    if returned:
        signals.append(returned)
    callback = _derive_callback_due(tenant, customer, ordered, as_of)
    if callback:
        signals.append(callback)
    stalling = _derive_deal_stalling(tenant, customer, ordered, opportunities, as_of)
    if stalling:
        signals.append(stalling)
    hot_streak = _derive_hot_streak(tenant, customer, ordered, as_of)
    if hot_streak:
        signals.append(hot_streak)
    season = _derive_season_return(tenant, customer, purchases or {}, as_of)
    if season:
        signals.append(season)
    return tuple(signals)


def backfill_sg_v1_signals(
    db_path: Path | str,
    *,
    allowed_root: Path | str,
    tenant_id: str = "foton",
    as_of: datetime,
    apply: bool = True,
) -> Mapping[str, Any]:
    require_timezone(as_of, "as_of")
    db = guard_customer_timeline_output_path(db_path, allowed_root)
    _require_existing_db(db)
    with _connect_existing_db(db, writable=False) as con:
        con.row_factory = sqlite3.Row
        loaded = _load_sg_v1_inputs(con, tenant_id=tenant_id)
    signals: list[DerivedSignal] = []
    for customer_id, payload in loaded.items():
        signals.extend(
            derive_sg_v1_signals(
                tenant_id=tenant_id,
                customer_id=customer_id,
                events=payload["events"],
                opportunities=payload["opportunities"],
                purchases=payload.get("purchases") or {},
                as_of=as_of,
            )
        )
    write_status_counts: Counter[str] = Counter()
    lifecycle_status_counts: Counter[str] = Counter()
    if apply:
        desired_ids = {require_text(signal.signal_id, "signal_id") for signal in signals}
        with CustomerTimelineSQLiteStore(db, allowed_root=allowed_root) as store:
            with store.bulk_write():
                for signal in signals:
                    result = store.upsert_signal(signal, actor="sg_v1_backfill")
                    write_status_counts[result.status] += 1
                for existing in _list_existing_sg_v1_signals(store._con, tenant_id=tenant_id):
                    signal_id = require_text(existing.get("signal_id"), "signal_id")
                    if signal_id in desired_ids:
                        continue
                    existing_signal = _signal_from_payload(existing)
                    if existing_signal.status != SignalStatus.ACTIVE:
                        lifecycle_status_counts[existing_signal.status.value] += 1
                        continue
                    status = (
                        SignalStatus.STALE
                        if existing_signal.expires_at and existing_signal.expires_at <= as_of
                        else SignalStatus.RESOLVED
                    )
                    lifecycle_status_counts[status.value] += 1
                    result = store.upsert_signal(
                        _replace_signal_lifecycle(
                            existing_signal,
                            signal_id=signal_id,
                            status=status,
                            created_at=existing_signal.created_at,
                            metadata_extra={
                                "lifecycle_reason": "expired" if status == SignalStatus.STALE else "predicate_resolved",
                                "rules_version": SIGNAL_RULES_VERSION,
                            },
                        ),
                        actor="sg_v1_backfill",
                    )
                    write_status_counts[result.status] += 1
    return {
        "schema_version": DERIVED_SIGNAL_RECOMPUTE_SCHEMA_VERSION,
        "rules_version": SIGNAL_RULES_VERSION,
        "apply": bool(apply),
        "customers_scanned": len(loaded),
        "signals": len(signals),
        "signal_type_counts": dict(Counter(signal.signal_type for signal in signals)),
        "status_counts": dict(Counter(signal.status.value for signal in signals)),
        "lifecycle_status_counts": dict(lifecycle_status_counts),
        "write_status_counts": dict(write_status_counts),
        "as_of": as_of.isoformat(),
    }


def _list_existing_sg_v1_signals(con: sqlite3.Connection, *, tenant_id: str) -> tuple[Mapping[str, Any], ...]:
    placeholders = ",".join("?" for _ in SG_V1_SIGNAL_TYPES)
    rows = con.execute(
        f"""
        SELECT record_json
        FROM derived_signals
        WHERE tenant_id = ?
          AND signal_type IN ({placeholders})
        """,
        (tenant_id, *SG_V1_SIGNAL_TYPES),
    ).fetchall()
    payloads = []
    for row in rows:
        payload = _safe_record_json(row["record_json"])
        if payload:
            payloads.append(payload)
    return tuple(payloads)


def signal_expires_at(signal_type: str, event_at: datetime) -> datetime:
    require_timezone(event_at, "event_at")
    ttl_days = SIGNAL_TTL_DAYS[normalize_key(signal_type, "signal_type")]
    return event_at + timedelta(days=ttl_days)


def _safe_record_json(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _direction(event: Mapping[str, Any]) -> str:
    return str(event.get("direction") or "").strip().lower()


def _event_text(event: Mapping[str, Any], *, include_thread_context: bool = True) -> str:
    record = _record(event)
    values = [
        event.get("subject"),
        event.get("text_preview"),
        event.get("summary"),
        record.get("summary"),
        record.get("full_clean_text"),
    ]
    if include_thread_context:
        values.append(record.get("thread_context"))
    return _joined_lower(*values)


def _callback_promise_text(event: Mapping[str, Any]) -> str:
    if _direction(event) != "outbound":
        return ""
    return _event_text(event, include_thread_context=False)


def _is_active_deal(opportunity: Mapping[str, Any]) -> bool:
    if str(opportunity.get("opportunity_type") or "") != "amo_deal":
        return False
    status = _joined_lower(opportunity.get("status"))
    if not status:
        return True
    if any(marker in status for marker in ("закры", "lost", "won", "успеш", "оплата получена")):
        return False
    return any(marker in status for marker in ACTIVE_DEAL_STATUSES) or status not in {"closed", "lost", "won"}


def _load_sg_v1_inputs(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, Mapping[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"events": [], "opportunities": [], "purchases": {}})
    superseded_filter = "AND superseded_by IS NULL" if _has_column(con, "timeline_events", "superseded_by") else ""
    for row in con.execute(
        f"""
        SELECT tenant_id, customer_id, event_id, event_type, event_at, direction,
               source_system, source_id, subject, text_preview, summary, record_json
        FROM timeline_events
        WHERE tenant_id = ?
          AND customer_id IS NOT NULL
          {superseded_filter}
        ORDER BY customer_id ASC, event_at ASC, event_id ASC
        """,
        (tenant_id,),
    ):
        payload = _safe_record_json(row["record_json"])
        grouped[str(row["customer_id"])]["events"].append(
            {
                "tenant_id": row["tenant_id"],
                "customer_id": row["customer_id"],
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "event_at": row["event_at"],
                "direction": row["direction"],
                "source_system": row["source_system"],
                "source_id": row["source_id"],
                "subject": row["subject"],
                "text_preview": row["text_preview"],
                "summary": row["summary"],
                "record": payload.get("record") or {},
            }
        )
    for row in con.execute(
        """
        SELECT tenant_id, customer_id, opportunity_id, opportunity_type, status,
               opened_at, closed_at, record_json
        FROM customer_opportunities
        WHERE tenant_id = ? AND customer_id IS NOT NULL
        ORDER BY customer_id ASC, opened_at ASC, opportunity_id ASC
        """,
        (tenant_id,),
    ):
        payload = _safe_record_json(row["record_json"])
        grouped[str(row["customer_id"])]["opportunities"].append(
            {
                "tenant_id": row["tenant_id"],
                "customer_id": row["customer_id"],
                "opportunity_id": row["opportunity_id"],
                "opportunity_type": row["opportunity_type"],
                "status": row["status"],
                "opened_at": row["opened_at"],
                "closed_at": row["closed_at"],
                "record": payload,
            }
        )
    if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='customer_purchases_v1'").fetchone():
        for row in con.execute(
            """
            SELECT tenant_id, customer_id, period, deals_cnt, last_purchase_at, computability
            FROM customer_purchases_v1
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ):
            grouped[str(row["customer_id"])]["purchases"] = {
                "period": row["period"],
                "deals_cnt": int(row["deals_cnt"] or 0),
                "last_purchase_at": row["last_purchase_at"],
                "computability": row["computability"],
            }
    return grouped


def _has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in con.execute(f"PRAGMA table_info({table})").fetchall())


def _require_existing_db(db: Path) -> None:
    if not db.exists() or not db.is_file():
        raise FileNotFoundError(f"customer timeline DB does not exist: {db}")


def _connect_existing_db(db: Path, *, writable: bool) -> sqlite3.Connection:
    if writable:
        return sqlite3.connect(db)
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def _derive_client_returned(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
    as_of: datetime,
) -> Optional[DerivedSignal]:
    inbound = [event for event in events if _direction(event) == "inbound"]
    if len(inbound) < 2:
        return None
    latest = inbound[-1]
    latest_at = _event_at(latest)
    latest_key = (latest_at, str(latest.get("event_id") or ""))
    previous_events = [
        event for event in events
        if (_event_at(event), str(event.get("event_id") or "")) < latest_key
    ]
    if not previous_events:
        return None
    previous_at = max(_event_at(event) for event in previous_events)
    if latest_at - previous_at < timedelta(days=DEFAULT_RETURN_SILENCE_DAYS):
        return None
    event_id = require_text(latest.get("event_id"), "event_id")
    return DerivedSignal(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_id=event_id,
        source_event_ids=(event_id,),
        signal_type=CLIENT_RETURNED_SIGNAL,
        severity=SignalSeverity.MEDIUM,
        confidence=0.84,
        evidence_text=f"Клиент вернулся после паузы {DEFAULT_RETURN_SILENCE_DAYS}+ дней.",
        recommended_action="Посмотреть историю и ответить с учётом прошлого запроса клиента.",
        status=SignalStatus.ACTIVE,
        expires_at=signal_expires_at(CLIENT_RETURNED_SIGNAL, latest_at),
        metadata={"rules_version": SIGNAL_RULES_VERSION},
        created_at=latest_at,
    )


def _derive_callback_due(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
    as_of: datetime,
) -> Optional[DerivedSignal]:
    callback_events = [
        event for event in events
        if any(marker in _callback_promise_text(event) for marker in CALLBACK_PROMISE_MARKERS)
    ]
    if not callback_events:
        return None
    candidate = callback_events[-1]
    candidate_at = _event_at(candidate)
    if as_of < candidate_at + timedelta(days=DEFAULT_CALLBACK_DUE_DAYS):
        return None
    if any(_event_at(event) > candidate_at and _direction(event) in {"inbound", "outbound"} for event in events):
        return None
    event_id = require_text(candidate.get("event_id"), "event_id")
    return DerivedSignal(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_id=event_id,
        source_event_ids=(event_id,),
        signal_type=CALLBACK_DUE_SIGNAL,
        severity=SignalSeverity.MEDIUM,
        confidence=0.78,
        evidence_text="В истории есть обещание связаться, после него нет нового касания 3+ дня.",
        recommended_action="Проверить, нужен ли обещанный контакт с клиентом.",
        requires_manager_review=True,
        status=SignalStatus.ACTIVE,
        expires_at=signal_expires_at(CALLBACK_DUE_SIGNAL, candidate_at),
        metadata={"rules_version": SIGNAL_RULES_VERSION},
        created_at=candidate_at,
    )


def _derive_deal_stalling(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    as_of: datetime,
) -> Optional[DerivedSignal]:
    if not events or not any(_is_active_deal(item) for item in opportunities):
        return None
    latest = events[-1]
    latest_at = _event_at(latest)
    if as_of < latest_at + timedelta(days=DEFAULT_DEAL_STALL_DAYS):
        return None
    event_id = require_text(latest.get("event_id"), "event_id")
    return DerivedSignal(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_id=event_id,
        source_event_ids=(event_id,),
        signal_type=DEAL_STALLING_SIGNAL,
        severity=SignalSeverity.MEDIUM,
        confidence=0.8,
        evidence_text="Есть активная сделка, но нет событий 14+ дней.",
        recommended_action="Проверить актуальность сделки и следующий шаг.",
        requires_manager_review=True,
        status=SignalStatus.ACTIVE,
        expires_at=signal_expires_at(DEAL_STALLING_SIGNAL, latest_at),
        metadata={"rules_version": SIGNAL_RULES_VERSION},
        created_at=latest_at,
    )


def _derive_hot_streak(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
    as_of: datetime,
) -> Optional[DerivedSignal]:
    recent_inbound = [
        event for event in events
        if _direction(event) == "inbound" and timedelta(0) <= as_of - _event_at(event) <= timedelta(days=DEFAULT_HOT_STREAK_DAYS)
    ]
    if len(recent_inbound) < 2:
        return None
    source_events = tuple(require_text(event.get("event_id"), "event_id") for event in recent_inbound[-2:])
    latest_at = _event_at(recent_inbound[-1])
    return DerivedSignal(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_id=source_events[-1],
        source_event_ids=source_events,
        signal_type=HOT_STREAK_SIGNAL,
        severity=SignalSeverity.HIGH,
        confidence=0.86,
        evidence_text="Клиент написал 2+ раза за последние 7 дней.",
        recommended_action="Ответить быстро: клиент активно вовлечён.",
        status=SignalStatus.ACTIVE,
        expires_at=signal_expires_at(HOT_STREAK_SIGNAL, latest_at),
        metadata={"rules_version": SIGNAL_RULES_VERSION},
        created_at=latest_at,
    )


def _derive_season_return(
    tenant_id: str,
    customer_id: str,
    purchases: Mapping[str, Any],
    as_of: datetime,
) -> Optional[DerivedSignal]:
    if int(purchases.get("deals_cnt") or 0) <= 0 or not purchases.get("last_purchase_at"):
        return None
    last_purchase_at = _parse_datetime(purchases["last_purchase_at"], "last_purchase_at")
    if as_of < last_purchase_at + timedelta(days=DEFAULT_SEASON_RETURN_DAYS):
        return None
    period_start = as_of.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    period = period_start.strftime("%Y-%m")
    signal_id = stable_signal_id(
        tenant_id=tenant_id,
        customer_id=customer_id,
        signal_type=SEASON_RETURN_SIGNAL,
        source_event_ids=(),
        evidence_text=f"{SEASON_RETURN_SIGNAL}:{customer_id}:{last_purchase_at.date().isoformat()}:{period}",
    )
    return DerivedSignal(
        tenant_id=tenant_id,
        customer_id=customer_id,
        signal_id=signal_id,
        signal_type=SEASON_RETURN_SIGNAL,
        severity=SignalSeverity.LOW,
        confidence=0.72,
        evidence_text="Клиент покупал в прошлом сезоне, стоит проверить повторный интерес.",
        recommended_action="Проверить, актуально ли предложить следующий сезон/курс.",
        status=SignalStatus.ACTIVE,
        expires_at=period_start + timedelta(days=SIGNAL_TTL_DAYS[SEASON_RETURN_SIGNAL]),
        metadata={
            "rules_version": SIGNAL_RULES_VERSION,
            "last_purchase_at": last_purchase_at.isoformat(),
            "period": period,
        },
        created_at=period_start,
    )


def _list_all_customer_events(store: Any, tenant_id: str, customer_id: str) -> tuple[Mapping[str, Any], ...]:
    events: list[Mapping[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        page = store.list_events_by_customer(tenant_id, customer_id, sort="asc", limit=500, cursor=cursor)
        events.extend(page.get("items") or ())
        cursor = optional_text(page.get("next_cursor"))
        if not cursor:
            return tuple(events)


def _signal_from_payload(payload: Mapping[str, Any]) -> DerivedSignal:
    return DerivedSignal(
        tenant_id=payload["tenant_id"],
        customer_id=payload.get("customer_id"),
        opportunity_id=payload.get("opportunity_id"),
        event_id=payload.get("event_id"),
        source_event_ids=tuple(payload.get("source_event_ids") or ()),
        signal_type=payload["signal_type"],
        severity=payload["severity"],
        evidence_text=payload["evidence_text"],
        signal_id=payload.get("signal_id"),
        confidence=payload.get("confidence"),
        recommended_action=payload.get("recommended_action"),
        requires_manager_review=bool(payload.get("requires_manager_review")),
        status=payload.get("status") or SignalStatus.ACTIVE,
        expires_at=_optional_datetime(payload.get("expires_at"), "expires_at"),
        metadata=payload.get("metadata") or {},
        created_at=_parse_datetime(payload.get("created_at"), "created_at"),
    )


def _replace_signal_lifecycle(
    signal: DerivedSignal,
    *,
    signal_id: str,
    status: SignalStatus,
    created_at: datetime,
    metadata_extra: Mapping[str, Any],
) -> DerivedSignal:
    metadata = {**dict(signal.metadata), **dict(metadata_extra)}
    return replace(
        signal,
        signal_id=signal_id,
        status=status,
        created_at=created_at,
        metadata=metadata,
    )


def _existing_created_at(existing: Optional[Mapping[str, Any]]) -> Optional[datetime]:
    if not existing or not existing.get("created_at"):
        return None
    return _parse_datetime(existing["created_at"], "created_at")


def _optional_datetime(value: Any, field_name: str) -> Optional[datetime]:
    if not optional_text(value):
        return None
    return _parse_datetime(value, field_name)


def _count_by_signal_attr(signals: Sequence[DerivedSignal], attr: str) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for signal in signals:
        value = getattr(signal, attr)
        if hasattr(value, "value"):
            value = value.value
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _derive_paid_no_access(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
) -> tuple[DerivedSignal, ...]:
    access_events = tuple(event for event in events if _is_access_event(event))
    signals: list[DerivedSignal] = []
    for payment in events:
        if not _is_paid_payment(payment):
            continue
        if any(_access_matches_payment(access, payment) for access in access_events):
            continue
        event_id = require_text(payment.get("event_id"), "event_id")
        event_at = _event_at(payment)
        payment_ref = _payment_ref(payment)
        signals.append(
            DerivedSignal(
                tenant_id=tenant_id,
                customer_id=customer_id,
                event_id=event_id,
                source_event_ids=(event_id,),
                signal_type=PAID_NO_ACCESS_SIGNAL,
                severity=SignalSeverity.HIGH,
                confidence=0.93,
                evidence_text=f"Оплата Tallanto {payment_ref} есть, активный доступ не найден.",
                recommended_action="Проверить оплату в Tallanto и выдать клиенту доступ или объяснить задержку.",
                requires_manager_review=True,
                status=SignalStatus.ACTIVE,
                expires_at=signal_expires_at(PAID_NO_ACCESS_SIGNAL, event_at),
                metadata={
                    "source": "deterministic_tallanto_events",
                    "payment_ref": payment_ref,
                    "access_predicate": "active_abonement_visits_left_or_matching_most_class",
                },
                created_at=event_at,
            )
        )
    return tuple(signals)


def _derive_hot_lead_silent(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
    as_of: Optional[datetime],
    silence_days: int,
) -> Optional[DerivedSignal]:
    if as_of is None:
        return None
    interest_events = tuple(event for event in events if _is_interest_event(event))
    if not interest_events:
        return None
    latest_interest = max(interest_events, key=lambda item: (_event_at(item), str(item.get("event_id") or "")))
    interest_at = _event_at(latest_interest)
    if as_of < interest_at + timedelta(days=silence_days):
        return None
    if any(_is_touch_event(event) and _event_at(event) > interest_at for event in events):
        return None
    event_id = require_text(latest_interest.get("event_id"), "event_id")
    return DerivedSignal(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_id=event_id,
        source_event_ids=(event_id,),
        signal_type=HOT_LEAD_SILENT_SIGNAL,
        severity=SignalSeverity.MEDIUM,
        confidence=0.82,
        evidence_text=f"Горячий интерес был {interest_at.date().isoformat()}, касаний нет {silence_days}+ дней.",
        recommended_action="Связаться с клиентом и уточнить, актуален ли интерес.",
        requires_manager_review=False,
        status=SignalStatus.ACTIVE,
        expires_at=signal_expires_at(HOT_LEAD_SILENT_SIGNAL, interest_at),
        metadata={
            "source": "deterministic_timeline_events",
            "silence_days": silence_days,
            "as_of": as_of.isoformat(),
        },
        created_at=interest_at,
    )


def _derive_duplicate_contact(
    tenant_id: str,
    customer_id: str,
    conflicts: Sequence[Mapping[str, Any]],
) -> tuple[DerivedSignal, ...]:
    signals: list[DerivedSignal] = []
    for conflict in sorted(conflicts, key=lambda item: str(item.get("conflict_id") or "")):
        conflict_type = normalize_key(conflict.get("conflict_type"), "conflict_type")
        status = normalize_key(conflict.get("status") or "open", "conflict_status")
        if status not in {"open", "active"}:
            continue
        if not _is_duplicate_contact_conflict(conflict_type):
            continue
        if not _conflict_mentions_customer(conflict, customer_id):
            continue
        conflict_id = require_text(conflict.get("conflict_id"), "conflict_id")
        created_at = _parse_datetime(conflict.get("created_at"), "created_at")
        signals.append(
            DerivedSignal(
                tenant_id=tenant_id,
                customer_id=customer_id,
                signal_type=DUPLICATE_CONTACT_SIGNAL,
                severity=SignalSeverity.MEDIUM,
                confidence=0.9,
                evidence_text=f"Открытый конфликт дубля контакта: {conflict_id}.",
                recommended_action="Проверить дубль контакта/сделки и выбрать корректную карточку перед ответом клиенту.",
                requires_manager_review=True,
                status=SignalStatus.ACTIVE,
                expires_at=signal_expires_at(DUPLICATE_CONTACT_SIGNAL, created_at),
                metadata={
                    "source": "timeline_conflicts",
                    "conflict_id": conflict_id,
                    "conflict_type": conflict_type,
                },
                created_at=created_at,
            )
        )
    return tuple(signals)


def _is_paid_payment(event: Mapping[str, Any]) -> bool:
    if event.get("event_type") != "tallanto_payment":
        return False
    record = _record(event)
    amount = _number(record.get("amount") or record.get("cost") or record.get("payment_summa"))
    if amount is None or amount <= 0:
        return False
    status_text = _joined_lower(record.get("payment_direction"), record.get("payment_status"), record.get("payment_type"), event.get("summary"))
    return any(marker in status_text for marker in PAYMENT_IN_MARKERS) and not any(
        marker in status_text for marker in PAYMENT_OUT_MARKERS
    )


def _is_access_event(event: Mapping[str, Any]) -> bool:
    record = _record(event)
    event_type = str(event.get("event_type") or "")
    if event_type == "tallanto_abonement":
        visits_left = _number(record.get("visits_left") or record.get("num_visit_left"))
        if visits_left is not None:
            return visits_left > 0
        status_text = _joined_lower(record.get("status"), event.get("summary"))
        return any(marker in status_text for marker in ACTIVE_ABONEMENT_MARKERS) and not any(
            marker in status_text for marker in INACTIVE_ABONEMENT_MARKERS
        )
    module = str(record.get("module") or "").strip().lower()
    return event_type == "tallanto_group" or module == "most_class"


def _access_matches_payment(access: Mapping[str, Any], payment: Mapping[str, Any]) -> bool:
    access_record = _record(access)
    payment_record = _record(payment)
    payment_abonement_id = optional_text(payment_record.get("abonement_id") or payment_record.get("most_abonements_id"))
    access_abonement_id = optional_text(access_record.get("abonement_id") or access_record.get("most_abonements_id"))
    if payment_abonement_id and access_abonement_id:
        return payment_abonement_id == access_abonement_id
    payment_class_id = optional_text(payment_record.get("class_id") or payment_record.get("most_class_id"))
    access_class_id = optional_text(access_record.get("class_id") or access_record.get("most_class_id"))
    if payment_class_id and access_class_id:
        return payment_class_id == access_class_id
    return not payment_abonement_id and not payment_class_id


def _is_interest_event(event: Mapping[str, Any]) -> bool:
    if not _is_touch_event(event):
        return False
    text = _joined_lower(event.get("subject"), event.get("text_preview"), event.get("summary"))
    return any(marker in text for marker in INTEREST_MARKERS)


def _is_touch_event(event: Mapping[str, Any]) -> bool:
    return str(event.get("direction") or "").strip().lower() in {"inbound", "outbound"}


def _is_duplicate_contact_conflict(conflict_type: str) -> bool:
    return (
        conflict_type in DUPLICATE_CONFLICT_TYPES
        or "shared_amo_contact" in conflict_type
        or "shared_amo_lead" in conflict_type
        or "ambiguous_identity" in conflict_type
    )


def _conflict_mentions_customer(conflict: Mapping[str, Any], customer_id: str) -> bool:
    refs = tuple(str(item) for item in conflict.get("entity_refs") or ())
    return any(customer_id == ref or customer_id in ref for ref in refs)


def _record(event: Mapping[str, Any]) -> Mapping[str, Any]:
    record = event.get("record")
    return record if isinstance(record, Mapping) else {}


def _event_at(event: Mapping[str, Any]) -> datetime:
    return _parse_datetime(event.get("event_at") or event.get("created_at"), "event_at")


def _parse_datetime(value: Any, field_name: str) -> datetime:
    text = require_text(value, field_name)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    require_timezone(parsed, field_name)
    return parsed


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _payment_ref(payment: Mapping[str, Any]) -> str:
    record = _record(payment)
    return (
        optional_text(record.get("payment_id"))
        or optional_text(record.get("source_id"))
        or optional_text(payment.get("source_id"))
        or require_text(payment.get("event_id"), "event_id")
    )


def _joined_lower(*values: Any) -> str:
    return " ".join(str(value or "").casefold() for value in values if value is not None)
