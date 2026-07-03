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
LEGACY_OBJECTION_EXTRACTOR_VERSIONS: tuple[str, ...] = ()
OBJECTION_SCHEMA_VERSION = "customer_timeline_objections_v1"
OBJECTION_TYPES = ("price", "schedule", "trust", "competitor", "child_refusal", "other")
PRICE_SENSITIVITY_ORDER = {"low": 0, "medium": 1, "high": 2}
CALL_MATCH_COVERAGE_GATE = 0.70
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
QUOTE_HEADER_RE = re.compile(
    r"^\s*(-{2,}\s*original message|-{2,}\s*пересылаемое|исходное сообщение|>+|on .+ wrote:|"
    r"(?:в\s+)?(?:пн|вт|ср|чт|пт|сб|вс|понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)"
    r".{0,120}(?:написал|написала|написал\(а\)|пишет|wrote)\s*:|"
    r".{0,140}(?:написал|написала|написал\(а\)|wrote)\s*:|"
    r"\d{1,2}[./]\d{1,2}[./]\d{2,4}.*(пишет|написал|написала|написал\(а\))|"
    r"\d{1,2}\s+[а-яё]{3,}\.?\s+\d{2,4}.{0,120}(?:пишет|написал|написала|написал\(а\))|"
    r"от кого:|кому:|отправлено:|sent:|from:\s)",
    re.I,
)
INLINE_QUOTE_HEADER_RE = re.compile(
    r"(?is)\b(?:пн|вт|ср|чт|пт|сб|вс|понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)"
    r"\s*,?\s*\d{1,2}\s+[а-яё]{3,}\.?\s+\d{2,4}.{0,180}(?:учебный\s+центр|<[^>]+>|написал|написала|wrote)\s*:?.*$|"
    r"\b\d{1,2}\s+[а-яё]{3,}\.?\s+\d{2,4}.{0,180}(?:учебный\s+центр|<[^>]+>|написал|написала|wrote)\s*:?.*$"
)
SIGNATURE_DIVIDER_RE = re.compile(r"^\s*--\s*$")
FOOTER_HINT_RE = re.compile(r"(с уважением|best regards|данное сообщение.*конфиденц|отписаться|unsubscribe|©)", re.I)
PRICE_BLOCK_HEADER_RE = re.compile(
    r"^\s*(?:стоимость(?:\s+(?:обучения|курса|занятий))?|условия\s+оплаты|порядок\s+оплаты|"
    r"варианты\s+оплаты|оплата\s+обучения|предоплата|полная\s+стоимость|рассрочка|долями)\s*[:：]?\s*$",
    re.I,
)
NON_CLIENT_EMAIL_TEMPLATE_RE = re.compile(
    r"(?is)^\s*(?:```|\[\d+\])?\s*[*_]*\s*(?:"
    r"-{5,}.{0,500}(?:вы\s+записаны|ниже\s+вы\s+можете|стоимость\s+обучения)|"
    r"(?:летн(?:яя|ие)\s+выездн(?:ая|ые)\s+школ(?:а|ы)|зимн(?:яя|ие)\s+выездн(?:ая|ые)\s+школ(?:а|ы))"
    r".{0,900}(?:вы\s+записаны|отправляем\s+вам\s+информацию|стоимость|при\s+оплате|акции)|"
    r"подготовительные\s+(?:(?:очные|онлайн)\s+)?(?:онлайн-)?курсы.{0,500}(?:вы\s+(?:записаны|интересовались)|предлагаем\s+вам\s+записаться|"
    r"ниже\s+вы\s+можете\s+ознакомиться|стоимость\s+обучения|условия\s+оплаты)|"
    r"\*?подготовительные\s+(?:(?:очные|онлайн)\s+)?(?:онлайн-)?курсы.{0,500}(?:квитанция|оферт|стоимость|ждем\s+вас)|"
    r"добрый\s+день!.{0,500}(?:вы\s+записаны|ниже\s+направляем\s+основную\s+информацию|условия\s+оплаты)|"
    r"добрый\s+день!.{0,500}(?:отправляем\s+вам\s+информацию|информация\s+об\s+индивидуальных\s+занятиях|"
    r"ниже\s+вы\s+можете\s+ознакомиться|вы\s+записаны\s+в\s+список\s+желающих)|"
    r"здравствуйте!.{0,500}(?:вы\s+записаны|ниже\s+вы\s+можете\s+ознакомиться|стоимость\s+обучения|условия\s+оплаты)|"
    r"здравствуйте,?\s+у\s+нас\s+открыта\s+предзапись.{0,900}(?:дополнительная\s+скидка|при\s+оплате|стоимости)|"
    r"учебный\s+центр.{0,500}(?:отправляем\s+вам\s+информацию|стоимость|скидк)|"
    r"отправляем\s+вам\s+информацию\s+по\s+(?:летним|зимним)\s+выездным\s+школам|"
    r"вы\s+записаны\s+в\s+список\s+желающих\s+обучаться|"
    r"информация\s+об\s+индивидуальных\s+занятиях\s+с\s+нашими\s+преподавателями|"
    r"доброе\s+утро!.{0,300}лицевой\s+счет\s+не\s+указываете|"
    r"физмат\s+направление.{0,900}(?:ai-репетитор|егэ|олимпиад)|"
    r"для\s+вас\s+открыт\s+пробный\s+доступ|"
    r"онлайн-летн(?:яя|ей)\s+школ(?:а|ы).{0,500}(?:ваша\s+запись|условия\s+оплаты|формат\s*:\s*онлайн)"
    r")"
)
EMBEDDED_OUTBOUND_TEMPLATE_RE = re.compile(
    r"(?im)^\s*(?:```|\[\d+\])?\s*(?:"
    r"подготовительные\s+(?:(?:очные|онлайн)\s+)?(?:онлайн-)?курсы\b|"
    r"\*?подготовительные\s+(?:(?:очные|онлайн)\s+)?(?:онлайн-)?курсы\b|"
    r"добрый\s+день!\s+спасибо\.?\s+получено\.?\s+стоимость\b|"
    r"учебный\s+центр\b|"
    r"стоимость\s+смены\b|"
    r"онлайн-летн(?:яя|ей)\s+школ(?:а|ы)\b"
    r")"
)
NON_CLIENT_PHRASE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("отправляем вам информацию",),
    ("вы записаны в список желающих",),
    ("вы записаны", "стоимость"),
    ("не смогли до вас дозвониться",),
    ("актуальна ли ваша запись",),
    ("скидка действует до", "далее будет дороже"),
    ("действует акция", "стоимость составляет"),
    ("лицевой счет не указываете",),
    ("лицевой счет только для бюджетных организаций",),
    ("дорогой родитель", "телеграмм-канал"),
    ("физмат направление",),
    ("не заполнили анкету", "потребуется ли трансфер"),
    ("будете оплачивать из личных средств", "рассрочку"),
    ("прислали вам информацию", "напоминаем"),
    ("предлагаем услуги",),
    ("рекламных интеграций",),
    ("сотрудничество в сфере недвижимости",),
    ("цифровой рубль",),
    ("мои услуги стоят",),
    ("информация об индивидуальных занятиях",),
)
OUTBOUND_PRICE_OFFER_RE = re.compile(
    r"(?is)\b(?:"
    r"стоимость\s+(?:за|обучения|курса|составляет|смены)|"
    r"общая\s+стоимость|полная\s+стоимость|"
    r"квитанц(?:ия|ию)|оферт\w*|производя\s+оплату|"
    r"при\s+оплате.{0,80}скидк|"
    r"скидк\w+\s+(?:действует|составляет|предоставляется)|"
    r"можем\s+предложить\s+оформить\s+рассроч|"
    r"отправляем\s+вам\s+информацию|ниже\s+направляем|ниже\s+вы\s+можете|"
    r"вы\s+записаны|ждем\s+вас\s+на\s+наших\s+курсах|"
    r"пришлите\s+подтверждение\s+оплаты\s+ответным\s+письмом"
    r")"
)
CLIENT_PRICE_INTENT_RE = re.compile(
    r"(?is)\b(?:"
    r"дорого|не\s+потян|за\s+такие\s+деньги|бюджет|"
    r"сколько\s+стоит|какая\s+(?:цена|стоимость)|какая\s+стоимость\s+будет|"
    r"(?:подскажите|скажите|уточните|интересует|можно\s+ли|можем\s+ли|получить|предусматривается)"
    r".{0,120}(?:скидк|рассроч|долями|снижени\w+\s+цен|стоимост|цен[ауые])|"
    r"есть\s+ли.{0,80}(?:у\s+вас|для\s+нас|для\s+многодетн|скидк)|"
    r"как\s+(?:мне|нам|преподавателю|сотруднику|получить).{0,120}скидк|"
    r"скиньте\s+пожалуйста.{0,80}стоимост|"
    r"если\s+(?:цена|стоимость).{0,80}(?:посильн|ниже|уменьш)"
    r")"
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


@dataclass(frozen=True)
class ObjectionSourceText:
    event: sqlite3.Row
    text: str
    speaker: str
    direction: str
    confidence: str
    source_kind: str


@dataclass(frozen=True)
class CanonicalCallText:
    transcript_client: str
    direction: str


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
    canonical_calls_db_path: Path | str | None = None,
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
        canonical_calls = _load_canonical_call_texts(canonical_calls_db_path)
        source_texts, metrics = _objection_source_texts(candidates, canonical_calls=canonical_calls)
        rows = []
        for source in source_texts:
            for extraction in extract_objections_from_text(source.text):
                if source.source_kind == "email_inbound" and not _email_objection_allowed(source.text, extraction):
                    metrics["email_objections_skipped_non_client_price"] = (
                        int(metrics.get("email_objections_skipped_non_client_price") or 0) + 1
                    )
                    continue
                rows.append(
                    {
                        "tenant_id": str(source.event["tenant_id"]),
                        "customer_id": str(source.event["customer_id"]),
                        "source_event_id": str(source.event["event_id"]),
                        "source_channel": _source_channel(source.event),
                        "objection_type": extraction.objection_type,
                        "quote_preview": extraction.quote_preview[:120],
                        "budget_hint_rub": extraction.budget_hint_rub,
                        "price_sensitivity": extraction.price_sensitivity,
                        "speaker": source.speaker,
                        "direction": source.direction,
                        "confidence": source.confidence,
                        "extracted_at": computed_at,
                        "extractor_version": OBJECTION_EXTRACTOR_VERSION,
                    }
                )
        coverage_gate_passed = bool(metrics["call_match_coverage"] >= CALL_MATCH_COVERAGE_GATE)
        if apply:
            _ensure_objection_tables(con)
            versions = tuple(dict.fromkeys((OBJECTION_EXTRACTOR_VERSION, *LEGACY_OBJECTION_EXTRACTOR_VERSIONS)))
            placeholders = ",".join("?" for _ in versions)
            con.execute(
                f"DELETE FROM customer_objections_v1 WHERE tenant_id = ? AND extractor_version IN ({placeholders})",
                (tenant_id, *versions),
            )
            _upsert_objection_rows(con, rows)
            con.execute(
                "DELETE FROM customer_objection_summary_v1 WHERE tenant_id = ?",
                (tenant_id,),
            )
            _refresh_objection_summary(con, tenant_id=tenant_id, extracted_at=computed_at)
            _record_objection_run(
                con,
                tenant_id=tenant_id,
                extracted_at=computed_at,
                metrics=metrics,
                crm_objections_enabled=coverage_gate_passed,
            )
            con.commit()
        return {
            "schema_version": OBJECTION_SCHEMA_VERSION,
            "apply": bool(apply),
            "candidate_events": len(candidates),
            **metrics,
            "coverage_gate_passed": coverage_gate_passed,
            "objections": len(rows),
            "objection_type_counts": dict(Counter(row["objection_type"] for row in rows)),
            "price_sensitivity_counts": dict(Counter(row["price_sensitivity"] for row in rows)),
            "speaker_counts": dict(Counter(row["speaker"] for row in rows)),
            "confidence_counts": dict(Counter(row["confidence"] for row in rows)),
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
          speaker TEXT NOT NULL DEFAULT 'unknown',
          direction TEXT NOT NULL DEFAULT 'unknown',
          confidence TEXT NOT NULL DEFAULT 'low',
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
        CREATE TABLE IF NOT EXISTS customer_objection_extraction_runs_v1 (
          tenant_id TEXT NOT NULL,
          extractor_version TEXT NOT NULL,
          extracted_at TEXT NOT NULL,
          call_events_total INTEGER NOT NULL,
          call_events_matched INTEGER NOT NULL,
          call_events_with_client_transcript INTEGER NOT NULL,
          call_match_coverage REAL NOT NULL,
          crm_objections_enabled INTEGER NOT NULL,
          metrics_json TEXT NOT NULL,
          PRIMARY KEY (tenant_id, extractor_version)
        );
        """
    )
    columns = {str(row[1]) for row in con.execute("PRAGMA table_info(customer_objections_v1)").fetchall()}
    for name, ddl in {
        "speaker": "ALTER TABLE customer_objections_v1 ADD COLUMN speaker TEXT NOT NULL DEFAULT 'unknown'",
        "direction": "ALTER TABLE customer_objections_v1 ADD COLUMN direction TEXT NOT NULL DEFAULT 'unknown'",
        "confidence": "ALTER TABLE customer_objections_v1 ADD COLUMN confidence TEXT NOT NULL DEFAULT 'low'",
    }.items():
        if name not in columns:
            con.execute(ddl)
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_customer_objections_v1_crm_eligible
          ON customer_objections_v1(tenant_id, customer_id, speaker, confidence)
        """
    )


def _load_objection_candidate_events(con: sqlite3.Connection, *, tenant_id: str) -> list[sqlite3.Row]:
    superseded_filter = "AND superseded_by IS NULL" if _has_column(con, "timeline_events", "superseded_by") else ""
    return list(
        con.execute(
            f"""
            SELECT tenant_id, customer_id, event_id, event_type, source_system, event_at,
                   source_id, direction, subject, text_preview, summary, record_json
                   , confidence
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


def _email_head_text(row: sqlite3.Row) -> str:
    record = _safe_json_object(row["record_json"]).get("record") or {}
    source = str(record.get("full_clean_text") or "").strip()
    if not source:
        source = str(row["text_preview"] or "").strip()
    head = _strip_embedded_outbound_template(_strip_quoted_email_tail(source))
    if _looks_like_non_client_email_head(head):
        return ""
    return head


def _objection_source_texts(
    events: Sequence[sqlite3.Row],
    *,
    canonical_calls: Mapping[str, CanonicalCallText],
) -> tuple[list[ObjectionSourceText], dict[str, Any]]:
    sources: list[ObjectionSourceText] = []
    metrics: Counter[str] = Counter()
    for event in events:
        event_type = str(event["event_type"])
        direction = str(event["direction"] or "unknown").lower()
        if event_type == "email_message":
            metrics["email_events_total"] += 1
            if direction != "inbound":
                metrics["email_events_skipped_non_client"] += 1
                continue
            text = _email_head_text(event)
            if text.strip():
                metrics["email_events_client_source"] += 1
                sources.append(
                    ObjectionSourceText(
                        event=event,
                        text=text,
                        speaker="client",
                        direction=direction,
                        confidence="high",
                        source_kind="email_inbound",
                    )
                )
            continue
        if event_type in {"mango_call", "call_transcript"}:
            metrics["call_events_total"] += 1
            canonical = canonical_calls.get(str(event["source_id"]))
            if canonical is None:
                metrics["call_events_unmatched"] += 1
                continue
            metrics["call_events_matched"] += 1
            if not canonical.transcript_client.strip():
                metrics["call_events_without_client_transcript"] += 1
                continue
            metrics["call_events_with_client_transcript"] += 1
            sources.append(
                ObjectionSourceText(
                    event=event,
                    text=canonical.transcript_client,
                    speaker="client",
                    direction=direction or canonical.direction or "unknown",
                    confidence=_call_source_confidence(event, canonical),
                    source_kind="call_transcript_client",
                )
            )
    call_total = int(metrics["call_events_total"])
    call_matched = int(metrics["call_events_matched"])
    result = {key: int(value) for key, value in metrics.items()}
    result["call_match_coverage"] = round(call_matched / call_total, 6) if call_total else 1.0
    result["source_texts"] = len(sources)
    return sources, result


def _load_canonical_call_texts(path: Path | str | None) -> Mapping[str, CanonicalCallText]:
    if path is None:
        return {}
    db = Path(path).expanduser().resolve(strict=False)
    if not db.exists() or not db.is_file():
        raise FileNotFoundError(f"canonical calls DB does not exist: {db}")
    uri = f"{db.as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT canonical_call_id, transcript_client, direction
            FROM canonical_calls
            """
        ).fetchall()
    result: dict[str, CanonicalCallText] = {}
    for row in rows:
        result[str(row["canonical_call_id"])] = CanonicalCallText(
            transcript_client=str(row["transcript_client"] or ""),
            direction=str(row["direction"] or "unknown").lower(),
        )
    return result


def _strip_quoted_email_tail(text: str) -> str:
    lines: list[str] = []
    in_context = False
    for line in str(text or "").replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if QUOTE_HEADER_RE.match(stripped) or SIGNATURE_DIVIDER_RE.match(stripped):
            in_context = True
        if in_context:
            continue
        if stripped.startswith(">"):
            continue
        if PRICE_BLOCK_HEADER_RE.match(stripped) and len("\n".join(lines)) > 80:
            in_context = True
            continue
        if FOOTER_HINT_RE.search(stripped) and len("\n".join(lines)) > 120:
            in_context = True
            continue
        lines.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return INLINE_QUOTE_HEADER_RE.sub("", cleaned).strip()


def _looks_like_non_client_email_head(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    head = normalized[:2000].casefold()
    if NON_CLIENT_EMAIL_TEMPLATE_RE.search(head[:1200]):
        return True
    return any(all(phrase in head for phrase in group) for group in NON_CLIENT_PHRASE_GROUPS)


def _strip_embedded_outbound_template(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    match = EMBEDDED_OUTBOUND_TEMPLATE_RE.search(value)
    if match and match.start() == 0:
        return ""
    if match and match.start() > 0:
        return value[:match.start()].strip()
    return value


def _email_objection_allowed(text: str, extraction: ObjectionExtraction) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if _looks_like_non_client_email_head(normalized):
        return False
    if extraction.objection_type != "price":
        return True
    client_intent = CLIENT_PRICE_INTENT_RE.search(normalized)
    outbound_offer = OUTBOUND_PRICE_OFFER_RE.search(normalized)
    if outbound_offer and (client_intent is None or outbound_offer.start() < client_intent.start()):
        return False
    if client_intent:
        return True
    return False


def _call_source_confidence(event: sqlite3.Row, canonical: CanonicalCallText) -> str:
    try:
        event_confidence = float(event["confidence"] or 0)
    except (KeyError, TypeError, ValueError):
        event_confidence = 0.0
    transcript_len = len(str(canonical.transcript_client or "").strip())
    if event_confidence >= 0.9 and transcript_len >= 120:
        return "high"
    if event_confidence >= 0.75 and transcript_len >= 60:
        return "medium"
    return "low"


def _source_channel(row: sqlite3.Row) -> str:
    return "email" if str(row["event_type"]) == "email_message" else "call"


def _upsert_objection_rows(con: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    con.executemany(
        """
        INSERT INTO customer_objections_v1 (
          tenant_id, customer_id, source_event_id, source_channel, objection_type,
          quote_preview, budget_hint_rub, price_sensitivity, speaker, direction, confidence,
          extracted_at, extractor_version
        )
        VALUES (
          :tenant_id, :customer_id, :source_event_id, :source_channel, :objection_type,
          :quote_preview, :budget_hint_rub, :price_sensitivity, :speaker, :direction, :confidence,
          :extracted_at, :extractor_version
        )
        ON CONFLICT(tenant_id, customer_id, source_event_id, objection_type) DO UPDATE SET
          source_channel = excluded.source_channel,
          quote_preview = excluded.quote_preview,
          budget_hint_rub = excluded.budget_hint_rub,
          price_sensitivity = excluded.price_sensitivity,
          speaker = excluded.speaker,
          direction = excluded.direction,
          confidence = excluded.confidence,
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
          AND speaker = 'client'
          AND confidence = 'high'
          AND extractor_version = ?
        ORDER BY extracted_at ASC, source_event_id ASC
        """,
        (tenant_id, OBJECTION_EXTRACTOR_VERSION),
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


def _record_objection_run(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    extracted_at: str,
    metrics: Mapping[str, Any],
    crm_objections_enabled: bool,
) -> None:
    con.execute(
        """
        INSERT INTO customer_objection_extraction_runs_v1 (
          tenant_id, extractor_version, extracted_at, call_events_total,
          call_events_matched, call_events_with_client_transcript, call_match_coverage,
          crm_objections_enabled, metrics_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tenant_id, extractor_version) DO UPDATE SET
          extracted_at = excluded.extracted_at,
          call_events_total = excluded.call_events_total,
          call_events_matched = excluded.call_events_matched,
          call_events_with_client_transcript = excluded.call_events_with_client_transcript,
          call_match_coverage = excluded.call_match_coverage,
          crm_objections_enabled = excluded.crm_objections_enabled,
          metrics_json = excluded.metrics_json
        """,
        (
            tenant_id,
            OBJECTION_EXTRACTOR_VERSION,
            extracted_at,
            int(metrics.get("call_events_total") or 0),
            int(metrics.get("call_events_matched") or 0),
            int(metrics.get("call_events_with_client_transcript") or 0),
            float(metrics.get("call_match_coverage") or 0.0),
            1 if crm_objections_enabled else 0,
            json.dumps(dict(metrics), ensure_ascii=False, sort_keys=True),
        ),
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
        return _trim_preview_to_word_boundary(cleaned[:120])
    start = max(0, index - 45)
    if start > 0 and not cleaned[start - 1].isspace():
        next_space = cleaned.find(" ", start)
        if next_space != -1 and next_space < index:
            start = next_space + 1
    return _trim_preview_to_word_boundary(cleaned[start : start + 120])


def _trim_preview_to_word_boundary(text: str) -> str:
    preview = str(text or "").strip()
    if len(preview) <= 1:
        return preview
    if len(preview) >= 120 and not preview[-1].isspace():
        last_space = preview.rfind(" ")
        if last_space >= 80:
            preview = preview[:last_space]
    return preview.strip()


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
