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
INTEREST_MARKER_RE = re.compile(r"\b(?:интересу\w*|рассматрива\w*|хот(?:им|ел[аи]?|им\s+бы|им\s+посмотреть))\b", re.I)
PAIN_MARKER_RE = re.compile(r"\b(?:не\s+успева\w*|сложн\w*|провалил\w*|провал\w*|пережива\w*)\b", re.I)
CONTACT_RE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}|"
    r"(?<!\d)(?:(?:\+7|8|7)\s*)?\(?\d{3,4}\)?[\s.-]*\d{2,3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)",
    re.I,
)
WHITESPACE_RE = re.compile(r"\s+")
SPEECH_FILLER_RE = re.compile(
    r"^(?:(?:ну|ээ+|э+|эм+|мм+|вот|значит|как\s+бы|то\s+есть|скажем|короче)\b[\s,.;:–—-]*)+",
    re.I,
)
SPEECH_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s+(?:"
    r"Можете|можете|можно|подскажите|скажите|скиньте|пришлите|"
    r"сколько\s+будет|сч[её]т\s+скинуть|как\s+там"
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
class CustomerDossier:
    tenant_id: str
    customer_id: str
    display_name: str
    phone: str
    email: str
    interests: tuple[DossierMarker, ...] = field(default_factory=tuple)
    pains: tuple[DossierMarker, ...] = field(default_factory=tuple)


def build_customer_dossier(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    canonical_calls: Mapping[str, str] | None = None,
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
    return CustomerDossier(
        tenant_id=str(customer["tenant_id"]),
        customer_id=str(customer["customer_id"]),
        display_name=_clean_text(customer["display_name"]),
        phone=_clean_text(customer["primary_phone"]),
        email=_clean_text(customer["primary_email"]),
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
    limit: int = 50,
) -> Mapping[str, Any]:
    db = Path(timeline_db).expanduser().resolve(strict=False)
    out = _guard_local_dossier_output_path(out_xlsx, allowed_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    canonical_calls = load_canonical_call_client_texts(canonical_calls_db) if canonical_calls_db else {}
    with _connect_ro(db) as con:
        ids = tuple(customer_ids) or tuple(_full_dossier_segment_customer_ids(con, tenant_id=tenant_id, limit=limit))
        dossiers = [build_customer_dossier(con, tenant_id=tenant_id, customer_id=customer_id, canonical_calls=canonical_calls) for customer_id in ids]
    _write_workbook(out, dossiers)
    summary = {
        "schema_version": MANAGER_DOSSIER_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "customers": len(dossiers),
        "interests_total": sum(len(item.interests) for item in dossiers),
        "pains_total": sum(len(item.pains) for item in dossiers),
        "canonical_calls_loaded": len(canonical_calls),
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
        raise FileNotFoundError(f"canonical calls DB does not exist: {db}")
    with _connect_ro(db) as con:
        rows = con.execute("SELECT canonical_call_id, transcript_client FROM canonical_calls").fetchall()
    return {str(row[0]): str(row[1] or "") for row in rows}


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
    overview.append(("customer_id", "Имя", "Интересов", "Болей"))
    overview.freeze_panes = "A2"
    for cell in overview[1]:
        cell.font = Font(bold=True)
    for index, dossier in enumerate(dossiers, start=1):
        sheet_name = f"Клиент {index}"
        overview.append((dossier.customer_id, dossier.display_name, len(dossier.interests), len(dossier.pains)))
        ws = wb.create_sheet(sheet_name)
        ws.append(("Раздел", "Значение", "Источник"))
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.append(("Кто", dossier.display_name, "customer_identities"))
        ws.append(("Контакт", f"{dossier.phone} {dossier.email}".strip(), "customer_identities"))
        for item in dossier.interests:
            ws.append(("Интересы", item.text, item.source))
        for item in dossier.pains:
            ws.append(("Боли", item.text, item.source))
        for column, width in {"A": 18, "B": 90, "C": 28}.items():
            ws.column_dimensions[column].width = width
    wb.save(path)
