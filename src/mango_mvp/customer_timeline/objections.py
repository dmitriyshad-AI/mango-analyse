from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path


OBJECTION_EXTRACTOR_VERSION = "ob_v1"
OBJECTION_SCHEMA_VERSION = "customer_timeline_objections_v1"
OBJECTION_TYPES = ("price", "schedule", "trust", "competitor", "child_refusal", "other")
PRICE_SENSITIVITY_ORDER = {"low": 0, "medium": 1, "high": 2}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}", re.I)
DOMAIN_RE = re.compile(r"\b[\w.-]+\.[a-zа-я]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+7|8|7)\s*)?\(?\d{3,4}\)?[\s.-]*\d{2,3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)"
)
USERNAME_RE = re.compile(r"(?<![\w.])@[a-z0-9_][a-z0-9_.-]{2,}", re.I)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
SERVICE_ID_RE = re.compile(
    r"\b(?:id|uid|user|client|lead|lead_id|chat|chat_id|profile|profile_id|message|message_id|msg|contact|contact_id)"
    r"[\s:_-]*[a-z0-9][a-z0-9_-]{5,}\b",
    re.I,
)
ADDRESS_RE = re.compile(
    r"\b(?:адрес|г\.|город|ул\.|улица|проспект|пр-т|дом|д\.|кв\.)\s*[:№#-]?\s*[^,.;\n]{1,80}",
    re.I,
)
PERSON_LABEL_RE = re.compile(
    r"\b(?P<label>меня зовут|это|мама|папа|родитель|ученик|ученица|реб[её]нок)\s+"
    r"(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})",
)
LEADING_NAME_RE = re.compile(r"^[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}(?=,|\s)")
AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[ \u00a0]?\d{3})+|\d{4,6})(?:\s*(?:руб|₽|р\.))?", re.I)
AMOUNT_THOUSAND_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\s*[-–]\s*\d{1,3})?\s*(?:тыс|к)(?![а-яa-z])", re.I)

MARKERS: Mapping[str, tuple[str, ...]] = {
    "price": (
        "дорого",
        "не потян",
        "скидк",
        "рассроч",
        "долями",
        "за такие деньги",
        "стоимость",
        "цена",
        "бюджет",
        "сколько стоит",
    ),
    "schedule": (
        "не успеваем",
        "время не подходит",
        "не подходит время",
        "далеко возить",
        "далеко ехать",
        "расписание не",
    ),
    "trust": (
        "подумаем",
        "посоветуемся",
        "сомневаемся",
        "не уверены",
        "обсудим",
    ),
    "competitor": (
        "занимаемся в",
        "выбрали другое",
        "другая школа",
        "репетитор",
        "у конкур",
    ),
    "child_refusal": (
        "ребёнок не хочет",
        "ребенок не хочет",
        "сын не хочет",
        "дочь не хочет",
        "не хочет заниматься",
    ),
}
HIGH_PRICE_MARKERS = ("дорого", "не потян", "за такие деньги", "скидк")
MEDIUM_PRICE_MARKERS = ("стоимость", "цена", "рассроч", "долями", "бюджет", "сколько стоит")


@dataclass(frozen=True)
class ObjectionExtraction:
    objection_type: str
    quote_preview: str
    budget_hint_rub: int | None
    price_sensitivity: str


def extract_objections_from_text(text: str) -> tuple[ObjectionExtraction, ...]:
    normalized = _normalize_text(text)
    if not normalized:
        return ()
    results: list[ObjectionExtraction] = []
    for objection_type in OBJECTION_TYPES:
        if objection_type == "other":
            continue
        marker = _first_marker(normalized, MARKERS[objection_type])
        if not marker:
            continue
        quote = _quote_preview(_mask_preview(text), marker)
        budget = _budget_hint_rub(quote or text) if objection_type == "price" else None
        results.append(
            ObjectionExtraction(
                objection_type=objection_type,
                quote_preview=quote[:120],
                budget_hint_rub=budget,
                price_sensitivity=_price_sensitivity(normalized) if objection_type == "price" else "low",
            )
        )
    return tuple(results)


def backfill_customer_objections_v1(
    db_path: Path | str,
    *,
    allowed_root: Path | str,
    tenant_id: str = "foton",
    apply: bool = True,
    as_of: datetime | None = None,
) -> Mapping[str, Any]:
    db = guard_customer_timeline_output_path(db_path, allowed_root)
    _require_existing_db(db)
    computed_at = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    with _connect_existing_db(db, writable=apply) as con:
        con.row_factory = sqlite3.Row
        candidates = _load_objection_candidate_events(con, tenant_id=tenant_id)
        rows = []
        for event in candidates:
            text = _event_text(event)
            for extraction in extract_objections_from_text(text):
                rows.append(
                    {
                        "tenant_id": str(event["tenant_id"]),
                        "customer_id": str(event["customer_id"]),
                        "source_event_id": str(event["event_id"]),
                        "source_channel": _source_channel(event),
                        "objection_type": extraction.objection_type,
                        "quote_preview": extraction.quote_preview[:120],
                        "budget_hint_rub": extraction.budget_hint_rub,
                        "price_sensitivity": extraction.price_sensitivity,
                        "extracted_at": computed_at,
                        "extractor_version": OBJECTION_EXTRACTOR_VERSION,
                    }
                )
        if apply:
            _ensure_objection_tables(con)
            con.execute(
                "DELETE FROM customer_objections_v1 WHERE tenant_id = ? AND extractor_version = ?",
                (tenant_id, OBJECTION_EXTRACTOR_VERSION),
            )
            _upsert_objection_rows(con, rows)
            con.execute(
                "DELETE FROM customer_objection_summary_v1 WHERE tenant_id = ? AND extractor_version = ?",
                (tenant_id, OBJECTION_EXTRACTOR_VERSION),
            )
            _refresh_objection_summary(con, tenant_id=tenant_id, extracted_at=computed_at)
            con.commit()
        return {
            "schema_version": OBJECTION_SCHEMA_VERSION,
            "apply": bool(apply),
            "candidate_events": len(candidates),
            "objections": len(rows),
            "objection_type_counts": dict(Counter(row["objection_type"] for row in rows)),
            "price_sensitivity_counts": dict(Counter(row["price_sensitivity"] for row in rows)),
            "extractor_version": OBJECTION_EXTRACTOR_VERSION,
        }


def _ensure_objection_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS customer_objections_v1 (
          tenant_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          source_event_id TEXT NOT NULL,
          source_channel TEXT NOT NULL,
          objection_type TEXT NOT NULL,
          quote_preview TEXT NOT NULL,
          budget_hint_rub INTEGER,
          price_sensitivity TEXT NOT NULL,
          extracted_at TEXT NOT NULL,
          extractor_version TEXT NOT NULL,
          PRIMARY KEY (tenant_id, customer_id, source_event_id, objection_type)
        );
        CREATE INDEX IF NOT EXISTS idx_customer_objections_v1_customer
          ON customer_objections_v1(tenant_id, customer_id, objection_type);
        CREATE TABLE IF NOT EXISTS customer_objection_summary_v1 (
          tenant_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          top_objections_json TEXT NOT NULL,
          max_price_sensitivity TEXT NOT NULL,
          last_budget_signal_json TEXT,
          extracted_at TEXT NOT NULL,
          extractor_version TEXT NOT NULL,
          PRIMARY KEY (tenant_id, customer_id)
        );
        """
    )


def _load_objection_candidate_events(con: sqlite3.Connection, *, tenant_id: str) -> list[sqlite3.Row]:
    superseded_filter = "AND superseded_by IS NULL" if _has_column(con, "timeline_events", "superseded_by") else ""
    return list(
        con.execute(
            f"""
            SELECT tenant_id, customer_id, event_id, event_type, source_system, event_at,
                   subject, text_preview, summary, record_json
            FROM timeline_events
            WHERE tenant_id = ?
              AND customer_id IS NOT NULL
              AND event_type IN ('email_message', 'mango_call', 'call_transcript')
              {superseded_filter}
            ORDER BY event_at ASC, event_id ASC
            """,
            (tenant_id,),
        )
    )


def _event_text(row: sqlite3.Row) -> str:
    record = _safe_json_object(row["record_json"]).get("record") or {}
    values = [
        row["subject"],
        row["text_preview"],
        row["summary"],
        record.get("full_clean_text"),
        record.get("thread_context"),
        record.get("summary"),
        record.get("topic"),
    ]
    return "\n".join(str(value) for value in values if value)


def _source_channel(row: sqlite3.Row) -> str:
    return "email" if str(row["event_type"]) == "email_message" else "call"


def _upsert_objection_rows(con: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    con.executemany(
        """
        INSERT INTO customer_objections_v1 (
          tenant_id, customer_id, source_event_id, source_channel, objection_type,
          quote_preview, budget_hint_rub, price_sensitivity, extracted_at, extractor_version
        )
        VALUES (
          :tenant_id, :customer_id, :source_event_id, :source_channel, :objection_type,
          :quote_preview, :budget_hint_rub, :price_sensitivity, :extracted_at, :extractor_version
        )
        ON CONFLICT(tenant_id, customer_id, source_event_id, objection_type) DO UPDATE SET
          source_channel = excluded.source_channel,
          quote_preview = excluded.quote_preview,
          budget_hint_rub = excluded.budget_hint_rub,
          price_sensitivity = excluded.price_sensitivity,
          extracted_at = excluded.extracted_at,
          extractor_version = excluded.extractor_version
        """,
        list(rows),
    )


def _refresh_objection_summary(con: sqlite3.Connection, *, tenant_id: str, extracted_at: str) -> None:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in con.execute(
        """
        SELECT *
        FROM customer_objections_v1
        WHERE tenant_id = ?
        ORDER BY extracted_at ASC, source_event_id ASC
        """,
        (tenant_id,),
    ):
        grouped[str(row["customer_id"])].append(row)
    rows = []
    for customer_id, items in grouped.items():
        type_counts = Counter(str(item["objection_type"]) for item in items)
        max_sensitivity = max((str(item["price_sensitivity"]) for item in items), key=lambda item: PRICE_SENSITIVITY_ORDER[item])
        budget_items = [item for item in items if item["budget_hint_rub"] is not None]
        last_budget = None
        if budget_items:
            item = budget_items[-1]
            last_budget = {
                "source_event_id": item["source_event_id"],
                "budget_hint_rub": int(item["budget_hint_rub"]),
                "quote_preview": item["quote_preview"],
            }
        rows.append(
            {
                "tenant_id": tenant_id,
                "customer_id": customer_id,
                "top_objections_json": json.dumps(type_counts.most_common(), ensure_ascii=False),
                "max_price_sensitivity": max_sensitivity,
                "last_budget_signal_json": json.dumps(last_budget, ensure_ascii=False) if last_budget else None,
                "extracted_at": extracted_at,
                "extractor_version": OBJECTION_EXTRACTOR_VERSION,
            }
        )
    if rows:
        con.executemany(
            """
            INSERT INTO customer_objection_summary_v1 (
              tenant_id, customer_id, top_objections_json, max_price_sensitivity,
              last_budget_signal_json, extracted_at, extractor_version
            )
            VALUES (
              :tenant_id, :customer_id, :top_objections_json, :max_price_sensitivity,
              :last_budget_signal_json, :extracted_at, :extractor_version
            )
            ON CONFLICT(tenant_id, customer_id) DO UPDATE SET
              top_objections_json = excluded.top_objections_json,
              max_price_sensitivity = excluded.max_price_sensitivity,
              last_budget_signal_json = excluded.last_budget_signal_json,
              extracted_at = excluded.extracted_at,
              extractor_version = excluded.extractor_version
            """,
            rows,
        )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def _first_marker(text: str, markers: Sequence[str]) -> str:
    for marker in markers:
        if marker in text:
            return marker
    return ""


def _quote_preview(text: str, marker: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    index = cleaned.casefold().find(marker)
    if index < 0:
        return cleaned[:120]
    start = max(0, index - 45)
    end = min(len(cleaned), index + 75)
    return cleaned[start:end].strip()


def _budget_hint_rub(text: str) -> int | None:
    amounts = []
    for match in AMOUNT_THOUSAND_RE.finditer(text):
        value = int(match.group(1)) * 1000
        if 1_000 <= value <= 2_000_000:
            amounts.append(value)
    for match in AMOUNT_RE.finditer(text):
        raw = re.sub(r"\D", "", match.group(1))
        if raw:
            value = int(raw)
            if 1_000 <= value <= 2_000_000:
                amounts.append(value)
    return amounts[0] if amounts else None


def _price_sensitivity(text: str) -> str:
    if any(marker in text for marker in HIGH_PRICE_MARKERS):
        return "high"
    if any(marker in text for marker in MEDIUM_PRICE_MARKERS):
        return "medium"
    return "low"


def _mask_preview(text: str) -> str:
    value = EMAIL_RE.sub("[email]", text)
    value = DOMAIN_RE.sub("[domain]", value)
    value = PHONE_RE.sub("[phone]", value)
    value = USERNAME_RE.sub("[username]", value)
    value = UUID_RE.sub("[id]", value)
    value = SERVICE_ID_RE.sub("[id]", value)
    value = ADDRESS_RE.sub("[address]", value)
    value = PERSON_LABEL_RE.sub(lambda match: f"{match.group('label')} [name]", value)
    value = LEADING_NAME_RE.sub("[name]", value)
    return value


def _safe_json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in con.execute(f"PRAGMA table_info({table})").fetchall())


def _require_existing_db(db: Path) -> None:
    if not db.exists() or not db.is_file():
        raise FileNotFoundError(f"customer timeline DB does not exist: {db}")


def _connect_existing_db(db: Path, *, writable: bool) -> sqlite3.Connection:
    if writable:
        return sqlite3.connect(db)
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
