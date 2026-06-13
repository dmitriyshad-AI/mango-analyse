from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from mango_mvp.utils.phone import last10, normalize_phone


CRM_SUMMARY_MAX_CHARS = 1200
EMPTY_PROFILE_SUMMARY = "CRM-выжимка профиля: данных пока недостаточно. Активных полей профиля нет."

_SERVICE_KEY_RE = re.compile(
    r"\b(?:source_id|source_ref|field_id|profile_id|customer_id|tenant_id|build_id|dedupe_key)\b\s*(?::|=|\s)\s*[^\s;,.)]+",
    re.IGNORECASE,
)
_PHONE_LIKE_RE = re.compile(r"(?<!\d)(?:\+?7|8|9)(?:[\s().-]*\d){9,14}(?!\d)")
_LEGAL_FRAGMENT_RE = re.compile(
    r"\b(?:ИНН|КПП|ОГРН|ОГРНИП|БИК|р/с|к/с|расчетный\s+счет|корр\.?\s+счет)\s*[:№#-]*\s*[\w./\\-]+",
    re.IGNORECASE,
)
_LEGAL_FIELD_MARKERS = (
    "inn",
    "kpp",
    "ogrn",
    "bik",
    "legal",
    "requisites",
    "bank_account",
    "passport",
    "snils",
    "инн",
    "кпп",
    "огрн",
    "бик",
    "юр",
    "реквиз",
    "паспорт",
    "снилс",
)
_TOUCH_FIELDS = {
    "brand_touch",
    "channel_touch",
    "touch_history",
    "touch_counter",
    "call_count",
    "message_count",
    "mango_call_count",
    "telegram_message_count",
    "whatsapp_message_count",
}


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    tenant_id: str
    display_name: str
    primary_phone: str
    source_event_count: int
    last_event_at: str
    active_fields: tuple[Mapping[str, Any], ...]


def render_crm_summary_from_db(
    profiles_db: Path | str,
    *,
    phone: str | None = None,
    profile_id: str | None = None,
    max_chars: int = CRM_SUMMARY_MAX_CHARS,
) -> str:
    profiles = load_profiles_for_summary(profiles_db, phone=phone, profile_id=profile_id)
    if not profiles:
        return _missing_profile_summary(phone=phone, profile_id=profile_id, max_chars=max_chars)
    if len(profiles) == 1:
        return render_crm_summary(profiles[0], max_chars=max_chars)

    selector = f" по телефону {mask_phone(phone)}" if phone else ""
    text = (
        f"Найдено несколько профилей{selector}: {len(profiles)}. "
        "CRM-выжимку не формирую автоматически: выберите профиль вручную."
    )
    return _fit(text, max_chars)


def load_profiles_for_summary(
    profiles_db: Path | str,
    *,
    phone: str | None = None,
    profile_id: str | None = None,
) -> list[ProfileRecord]:
    if bool(phone) == bool(profile_id):
        raise ValueError("Specify exactly one of phone or profile_id")
    with _connect_read_only(Path(profiles_db)) as con:
        con.row_factory = sqlite3.Row
        if profile_id:
            row = con.execute(
                """
                SELECT profile_id, tenant_id, COALESCE(display_name, '') AS display_name,
                       COALESCE(primary_phone, '') AS primary_phone,
                       source_event_count, COALESCE(last_event_at, '') AS last_event_at
                FROM customer_profiles
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()
            return [_load_profile(con, row)] if row else []

        matched_ids = _profile_ids_by_phone(con, phone or "")
        return [_load_profile(con, row) for row in _profile_rows(con, matched_ids)]


def render_crm_summary(profile: ProfileRecord, *, max_chars: int = CRM_SUMMARY_MAX_CHARS) -> str:
    active_fields = tuple(row for row in profile.active_fields if not _is_forbidden_field(str(row.get("field") or "")))
    header = _header_line(profile, active_fields)
    if not active_fields:
        return _fit(f"{header}\n{EMPTY_PROFILE_SUMMARY}", max_chars)

    lines = [header]
    for line in (
        _tallanto_line(active_fields),
        _next_step_line(active_fields),
        _touches_line(active_fields),
    ):
        if line:
            lines.append(line)

    if len(lines) == 1:
        lines.append(EMPTY_PROFILE_SUMMARY)
    return _fit("\n".join(lines), max_chars)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview deterministic CRM profile summary without CRM writes.")
    parser.add_argument("--profiles-db", required=True, help="Path to customer_profiles.sqlite.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--phone", help="Find profile by primary phone; output masks the phone.")
    selector.add_argument("--profile-id", help="Preview a known profile_id.")
    parser.add_argument("--out", help="Optional preview text output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = render_crm_summary_from_db(
        args.profiles_db,
        phone=args.phone,
        profile_id=args.profile_id,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=False)
    uri = f"file:{quote(str(resolved), safe='/:')}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _profile_ids_by_phone(con: sqlite3.Connection, phone: str) -> list[str]:
    target = normalize_phone(phone)
    target_tail = last10(phone)
    if not target and not target_tail:
        return []
    if os.getenv("PROFILE_PHONE_INDEX", "0") == "1" and _has_customer_profile_column(con, "primary_phone_norm"):
        rows = con.execute(
            """
            SELECT profile_id
            FROM customer_profiles
            WHERE primary_phone_norm = ?
               OR (? != '' AND primary_phone_norm LIKE ?)
            ORDER BY profile_id
            """,
            (target or "", target_tail or "", f"%{target_tail}" if target_tail else ""),
        ).fetchall()
        return [str(row["profile_id"]) for row in rows]
    rows = con.execute(
        """
        SELECT profile_id, COALESCE(primary_phone, '') AS primary_phone
        FROM customer_profiles
        ORDER BY profile_id
        """
    ).fetchall()
    matched: list[str] = []
    for row in rows:
        stored = str(row["primary_phone"] or "")
        if normalize_phone(stored) == target or (target_tail and last10(stored) == target_tail):
            matched.append(str(row["profile_id"]))
    return matched


def _has_customer_profile_column(con: sqlite3.Connection, column_name: str) -> bool:
    return any(str(row["name"]) == column_name for row in con.execute("PRAGMA table_info(customer_profiles)").fetchall())


def _profile_rows(con: sqlite3.Connection, profile_ids: Sequence[str]) -> list[sqlite3.Row]:
    if not profile_ids:
        return []
    placeholders = ",".join("?" for _ in profile_ids)
    return con.execute(
        f"""
        SELECT profile_id, tenant_id, COALESCE(display_name, '') AS display_name,
               COALESCE(primary_phone, '') AS primary_phone,
               source_event_count, COALESCE(last_event_at, '') AS last_event_at
        FROM customer_profiles
        WHERE profile_id IN ({placeholders})
        ORDER BY profile_id
        """,
        tuple(profile_ids),
    ).fetchall()


def _load_profile(con: sqlite3.Connection, row: sqlite3.Row) -> ProfileRecord:
    fields = con.execute(
        """
        SELECT field_id, field, value, child_key, brand, source_system, event_at, quote
        FROM profile_fields
        WHERE profile_id = ? AND COALESCE(superseded_by, '') = ''
        ORDER BY event_at DESC, field, child_key, field_id
        """,
        (row["profile_id"],),
    ).fetchall()
    return ProfileRecord(
        profile_id=str(row["profile_id"]),
        tenant_id=str(row["tenant_id"]),
        display_name=str(row["display_name"] or ""),
        primary_phone=str(row["primary_phone"] or ""),
        source_event_count=int(row["source_event_count"] or 0),
        last_event_at=str(row["last_event_at"] or ""),
        active_fields=tuple(dict(item) for item in fields),
    )


def _header_line(profile: ProfileRecord, fields: Sequence[Mapping[str, Any]]) -> str:
    name = _latest_value(fields, "parent_name") or _sanitize_value(profile.display_name) or "имя не указано"
    phone = mask_phone(profile.primary_phone) if profile.primary_phone else "телефон не указан"
    children = _children_text(fields) or "нет активных данных"
    return _clip(f"Клиент: {name}; телефон: {phone}; дети: {children}.", 360)


def _children_text(fields: Sequence[Mapping[str, Any]]) -> str:
    child_keys = sorted(
        {
            str(row.get("child_key") or "child_1")
            for row in fields
            if str(row.get("field") or "") in {"child_name", "grade", "subject"}
        }
    )
    children: list[str] = []
    for child_key in child_keys:
        name = _latest_value(fields, "child_name", child_key=child_key)
        grade = _latest_value(fields, "grade", child_key=child_key)
        subject = _latest_value(fields, "subject", child_key=child_key)
        parts = [part for part in (name or "ребенок", _format_grade(grade), subject) if part]
        children.append(" - ".join(parts))
    return "; ".join(children[:4])


def _tallanto_line(fields: Sequence[Mapping[str, Any]]) -> str:
    groups = _latest_values(fields, "tallanto_group", limit=3)
    balances = _latest_values(fields, "tallanto_balance", limit=2)
    payment = _latest_row(fields, "payment_fact")
    parts: list[str] = []
    if groups:
        parts.append(f"группа {', '.join(groups)}")
    if balances:
        parts.append(f"баланс {', '.join(balances)}")
    if payment:
        value = _sanitize_value(payment.get("value"))
        if value:
            parts.append(f"последняя оплата {value}")
    return _clip(f"Tallanto: {'; '.join(parts)}.", 300) if parts else ""


def _next_step_line(fields: Sequence[Mapping[str, Any]]) -> str:
    rows = _sorted_rows(fields, "next_step")
    items: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = _clip(_sanitize_value(row.get("value")), 240)
        if not value or value in seen:
            continue
        seen.add(value)
        date = _date_only(row.get("event_at"))
        items.append(f"{date} - {value}" if date else value)
        if len(items) >= 2:
            break
    return _clip(f"Договоренности: {'; '.join(items)}.", 420) if items else ""


def _touches_line(fields: Sequence[Mapping[str, Any]]) -> str:
    touch_rows = [
        row
        for row in fields
        if str(row.get("field") or "") in _TOUCH_FIELDS or "touch" in str(row.get("field") or "").lower()
    ]
    items: list[str] = []
    for row in touch_rows:
        items.extend(_touch_items(row))
        if len(items) >= 4:
            break
    items = _dedupe(items)[:4]
    return _clip(f"Касания: {'; '.join(items)}.", 300) if items else ""


def _touch_items(row: Mapping[str, Any]) -> list[str]:
    value = _sanitize_value(row.get("value"))
    data = _json_mapping(value)
    if data:
        return _touch_items_from_mapping(row, data)
    label = _touch_label(row, None)
    return [f"{label}: {value}"] if value else []


def _touch_items_from_mapping(row: Mapping[str, Any], data: Mapping[str, Any]) -> list[str]:
    period = _period_text(data)
    channels = data.get("channels") if isinstance(data.get("channels"), Mapping) else None
    if channels:
        return [
            _touch_counter_text(row, str(channel), count, period=period, data=data)
            for channel, count in sorted(channels.items())
            if _sanitize_value(count)
        ]

    channel = data.get("channel")
    count = data.get("count") or data.get("touch_count") or data.get("messages") or data.get("calls")
    if channel or count:
        return [_touch_counter_text(row, str(channel or ""), count or "", period=period, data=data)]

    items: list[str] = []
    for key in ("calls", "messages", "telegram", "whatsapp", "mango_call"):
        if key in data and _sanitize_value(data.get(key)):
            items.append(_touch_counter_text(row, key, data.get(key), period=period, data=data))
    return items


def _touch_counter_text(
    row: Mapping[str, Any],
    channel: str,
    count: Any,
    *,
    period: str,
    data: Mapping[str, Any],
) -> str:
    label = _touch_label(row, channel or None, brand=str(data.get("brand") or row.get("brand") or ""))
    count_text = _sanitize_value(count)
    return f"{label}: {count_text}{f' ({period})' if period else ''}"


def _touch_label(row: Mapping[str, Any], channel: str | None, *, brand: str | None = None) -> str:
    brand_text = _brand_label(brand if brand is not None else str(row.get("brand") or ""))
    channel_text = _channel_label(channel or _channel_from_field(str(row.get("field") or "")))
    parts = [part for part in (brand_text, channel_text) if part]
    return "/".join(parts) if parts else "касания"


def _period_text(data: Mapping[str, Any]) -> str:
    start = _date_only(data.get("first_at") or data.get("from") or data.get("start_at"))
    finish = _date_only(data.get("last_at") or data.get("to") or data.get("end_at"))
    if start and finish:
        return f"{start}..{finish}"
    return start or finish


def _channel_from_field(field: str) -> str:
    lowered = field.lower()
    if "telegram" in lowered:
        return "telegram"
    if "whatsapp" in lowered:
        return "whatsapp"
    if "mango" in lowered or "call" in lowered:
        return "звонки"
    if "message" in lowered:
        return "сообщения"
    return ""


def _channel_label(value: str) -> str:
    lowered = str(value or "").strip().lower()
    return {
        "call": "звонки",
        "calls": "звонки",
        "mango_call": "звонки",
        "message": "сообщения",
        "messages": "сообщения",
    }.get(lowered, lowered)


def _brand_label(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value and value != "unknown" else ""


def _latest_row(fields: Sequence[Mapping[str, Any]], field: str, *, child_key: str | None = None) -> Mapping[str, Any] | None:
    rows = _sorted_rows(fields, field, child_key=child_key)
    return rows[0] if rows else None


def _latest_value(fields: Sequence[Mapping[str, Any]], field: str, *, child_key: str | None = None) -> str:
    row = _latest_row(fields, field, child_key=child_key)
    return _sanitize_value(row.get("value")) if row else ""


def _latest_values(fields: Sequence[Mapping[str, Any]], field: str, *, limit: int) -> list[str]:
    values: list[str] = []
    for row in _sorted_rows(fields, field):
        value = _sanitize_value(row.get("value"))
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _sorted_rows(
    fields: Sequence[Mapping[str, Any]],
    field: str,
    *,
    child_key: str | None = None,
) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in fields
        if str(row.get("field") or "") == field
        and (child_key is None or str(row.get("child_key") or "child_1") == child_key)
    ]
    return sorted(rows, key=lambda row: (str(row.get("event_at") or ""), str(row.get("field_id") or "")), reverse=True)


def _format_grade(value: str) -> str:
    value = _sanitize_value(value)
    if not value:
        return ""
    return value if "класс" in value.lower() else f"{value} класс"


def _sanitize_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = _SERVICE_KEY_RE.sub("", text)
    text = _LEGAL_FRAGMENT_RE.sub("", text)
    text = _PHONE_LIKE_RE.sub(lambda match: mask_phone(match.group(0)), text)
    return text.strip(" ;,.-")


def _json_mapping(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _is_forbidden_field(field: str) -> bool:
    lowered = field.lower()
    return any(marker in lowered for marker in _LEGAL_FIELD_MARKERS) or lowered in {
        "source_id",
        "source_ref",
        "field_id",
        "profile_id",
        "customer_id",
        "tenant_id",
    }


def _date_only(value: Any) -> str:
    text = str(value or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    return ""


def mask_phone(value: Any) -> str:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) <= 4:
        return "***"
    return f"+***{digits[-4:]}"


def _missing_profile_summary(*, phone: str | None, profile_id: str | None, max_chars: int) -> str:
    if phone:
        selector = f" Профиль по телефону {mask_phone(phone)} не найден."
    elif profile_id:
        selector = " Профиль с указанным идентификатором не найден."
    else:
        selector = ""
    return _fit(EMPTY_PROFILE_SUMMARY + selector, max_chars)


def _dedupe(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _clip(value: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3].rstrip() + "..."


def _fit(value: str, max_chars: int) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return _clip(value, max_chars)


if __name__ == "__main__":
    raise SystemExit(main())
