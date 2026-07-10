from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path


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
    phone: str
    email: str
    actuality_header: str = ""
    family: tuple[DossierRow, ...] = field(default_factory=tuple)
    money: tuple[DossierRow, ...] = field(default_factory=tuple)
    signals: tuple[DossierRow, ...] = field(default_factory=tuple)
    next_step: str = ""
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
    return CustomerDossier(
        tenant_id=str(customer["tenant_id"]),
        customer_id=str(customer["customer_id"]),
        display_name=_clean_text(customer["display_name"]),
        phone=_clean_text(customer["primary_phone"]),
        email=_clean_text(customer["primary_email"]),
        actuality_header=actuality_header,
        family=tuple(_family_rows(con, tenant_id=tenant_id, customer_id=customer_id)),
        money=tuple(_money_rows(con, tenant_id=tenant_id, customer_id=customer_id)),
        signals=tuple(signals),
        next_step=_next_step_from_signals(signals),
        objections=tuple(_objection_rows(con, tenant_id=tenant_id, customer_id=customer_id)),
        chronology=tuple(_chronology_rows(con, tenant_id=tenant_id, customer_id=customer_id, limit=40)),
        interests=tuple(_dedupe_markers(interests, limit=8)),
        pains=tuple(_dedupe_markers(pains, limit=8)),
    )


def build_manager_dossier_workbook(
    *,
    timeline_db: Path | str,
    allowed_root: Path | str,
    out_xlsx: Path | str,
    tenant_id: str = "foton",
    customer_ids: Sequence[str] = (),
    canonical_calls_db: Path | str | None = None,
    reconcile_json: Path | str | None = None,
    limit: int = 50,
) -> Mapping[str, Any]:
    db = Path(timeline_db).expanduser().resolve(strict=False)
    out = _guard_local_dossier_output_path(out_xlsx, allowed_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    canonical_calls, canonical_warning = _load_canonical_calls_fail_soft(canonical_calls_db)
    reconcile = _read_json(Path(reconcile_json).expanduser()) if reconcile_json else {}
    with _connect_ro(db) as con:
        ids = tuple(customer_ids) or tuple(_full_dossier_segment_customer_ids(con, tenant_id=tenant_id, limit=limit))
        freshness = _source_freshness(con)
        segment_total = _full_dossier_segment_count(con, tenant_id=tenant_id)
        actuality_header = _actuality_header(freshness, reconcile)
        dossiers: list[CustomerDossier] = []
        missing_customer_ids: list[str] = []
        for customer_id in ids:
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
        "full_dossier_segment_total": segment_total,
        "interests_total": sum(len(item.interests) for item in dossiers),
        "pains_total": sum(len(item.pains) for item in dossiers),
        "family_rows_total": sum(len(item.family) for item in dossiers),
        "money_rows_total": sum(len(item.money) for item in dossiers),
        "signals_total": sum(len(item.signals) for item in dossiers),
        "objections_total": sum(len(item.objections) for item in dossiers),
        "chronology_rows_total": sum(len(item.chronology) for item in dossiers),
        "next_step_rows_total": sum(1 for item in dossiers if item.next_step),
        "canonical_calls_loaded": len(canonical_calls),
        "canonical_calls_warning": canonical_warning,
        "actuality_header": actuality_header,
        "source_freshness_top": freshness[:12],
        "reconcile_status": reconcile.get("status") if reconcile else "missing",
        "out_xlsx": str(out),
        "safety": {
            "source_open_mode": "sqlite_mode_ro_immutable",
            "write_crm": False,
            "write_tallanto": False,
            "send_messages": False,
            "pii_scope": "local_codex_local_only",
        },
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


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
    uri = f"{path.as_uri()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def _full_dossier_segment_customer_ids(con: sqlite3.Connection, *, tenant_id: str, limit: int) -> list[str]:
    sql = """
        SELECT customer_id
        FROM timeline_events
        WHERE tenant_id = ?
          AND customer_id IS NOT NULL
          AND customer_id != ''
        GROUP BY customer_id
        HAVING SUM(event_type = 'mango_call') > 0
           AND SUM(event_type = 'email_message') > 0
        ORDER BY MAX(event_at) DESC, customer_id
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
          SELECT customer_id
          FROM timeline_events
          WHERE tenant_id = ?
            AND customer_id IS NOT NULL
            AND customer_id != ''
          GROUP BY customer_id
          HAVING SUM(event_type = 'mango_call') > 0
             AND SUM(event_type = 'email_message') > 0
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


def _source_freshness(con: sqlite3.Connection) -> list[Mapping[str, Any]]:
    if not _table_exists(con, "timeline_events"):
        return []
    return [
        dict(row)
        for row in con.execute(
            """
            SELECT source_system, MAX(event_at) AS max_event_at, COUNT(*) AS events
            FROM timeline_events
            GROUP BY source_system
            ORDER BY max_event_at DESC, source_system
            """
        ).fetchall()
    ]


def _actuality_header(freshness: Sequence[Mapping[str, Any]], reconcile: Mapping[str, Any]) -> str:
    freshness_text = "; ".join(f"{_display_freshness_source(row.get('source_system'))}={row.get('max_event_at')}" for row in freshness[:8])
    if not freshness_text:
        freshness_text = "нет данных"
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
    return f"Данные: staging max event_at по источникам: {freshness_text}; собрано {generated_at}; сверка с живым AMO: {reconcile_text}"


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


def _family_rows(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> list[DossierRow]:
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
            f"возвраты/исход {_format_money(row['total_out'])}; сделок {int(row['deals_cnt'] or 0)}"
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
        ORDER BY severity DESC, expires_at, signal_type
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
    rows = con.execute(
        """
        SELECT event_at, event_type, source_system, subject, summary, text_preview, record_json
        FROM timeline_events
        WHERE tenant_id = ?
          AND customer_id = ?
          AND (superseded_by IS NULL OR superseded_by = '')
        ORDER BY event_at DESC, event_id DESC
        LIMIT ?
        """,
        (tenant_id, customer_id, int(limit)),
    ).fetchall()
    result: list[DossierRow] = []
    for row in rows:
        summary = _event_summary_for_manager(row)
        if not summary:
            continue
        text = f"{row['event_at']} [{row['event_type']}] {summary}"
        result.append(DossierRow("Хронология", text, str(row["source_system"] or "")))
    return result


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
        for key in ("title", "name", "subject", "format", "class", "value"):
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


def _write_workbook(path: Path, dossiers: Sequence[CustomerDossier]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    overview = wb.active
    overview.title = "Оглавление"
    overview.append(("customer_id", "Имя", "Семья", "Сигналы", "Следующий шаг", "Интересов", "Болей", "Возражений", "Хронология"))
    overview.freeze_panes = "A2"
    for cell in overview[1]:
        cell.font = Font(bold=True)
    for index, dossier in enumerate(dossiers, start=1):
        sheet_name = f"Клиент {index}"
        overview.append(
            (
                dossier.customer_id,
                dossier.display_name,
                len(dossier.family),
                len(dossier.signals),
                "да" if dossier.next_step else "",
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
        ws.append(("Контакт", f"{dossier.phone} {dossier.email}".strip(), _display_source("customer_identities")))
        for row in dossier.family:
            ws.append((row.section, row.text, _display_source(row.source)))
        for row in dossier.money:
            ws.append((row.section, row.text, _display_source(row.source)))
        for row in dossier.signals:
            ws.append((row.section, row.text, _display_source(row.source)))
        if dossier.next_step:
            ws.append(("Следующий шаг", dossier.next_step, _display_source("derived_signals")))
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
