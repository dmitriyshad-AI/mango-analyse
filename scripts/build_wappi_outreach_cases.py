#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "wappi_outreach_case_packets_v1"
WAPPI_FIELDS = (
    "profile_id", "chat_id", "brand", "channel", "peer_names", "n_messages", "n_inbound", "n_outbound",
    "last_inbound_ts", "last_outbound_ts", "last_is_inbound", "messages",
)

SHEET_SCHEMA_VERSION = "wappi_outreach_sheet_rows_v1"
# Канон A:AB из audits/_inbox/wappi_total_outreach_strategy_20260822/STRATEGY_SOURCE_DRAFT.md
SHEET_COLUMNS = (
    "Ключ чата", "Статус", "Теплота", "Бренд и канал", "Открыть", "Клиент", "Последний контакт",
    "Что хотел", "Покупка/запись 2026/27", "Следующий шаг", "Что предложить", "Почему сейчас",
    "Сообщение 1", "Дата 1", "Сообщение 2", "Дата 2", "Сообщение 3", "Дата 3", "Сообщение 4",
    "Дата 4", "Сообщение 5", "Дата 5", "Подпись", "Решение РОПа", "Результат", "Доказательства",
    "Актуальность", "Проверено на",
)
MANUAL_COLUMNS = (23, 24)  # X:Y — колонки РОПа; CLI их не генерирует и не перезаписывает
# Схема листа фиксирована: пишем ровно два блока, X:Y между ними не трогаем.
WRITE_RANGES = ("A:W", "Z:AB")
WRITE_SLICES = {"A:W": slice(0, 23), "Z:AB": slice(25, 28)}
ANCHOR_COLUMN = "A"
VERDICT_COLUMNS = {
    "status": 1, "heat": 2, "past_interest": 7, "purchase_2026_27": 8, "next_step": 9, "offer": 10,
    "why_now": 11, "message_1": 12, "date_1": 13, "message_2": 14, "date_2": 15, "message_3": 16,
    "date_3": 17, "message_4": 18, "date_4": 19, "message_5": 20, "date_5": 21, "signature": 22,
    "freshness": 26,
}
# Структурная схема, а не смысл: закрытый список допустимых значений двух управляющих колонок.
SEND_STATUS = "ОТПРАВИТЬ ПОСЛЕ ПРОВЕРКИ ЧАТА"
STATUS_VALUES = (SEND_STATUS, "МЕНЕДЖЕР СЕГОДНЯ", "ПРОВЕРКА", "НЕ ПИСАТЬ")
FRESHNESS_VALUES = ("АКТУАЛЬНО", "ОБНОВИТЬ", "СТОП ДЕЙСТВУЕТ")
class VerdictContentInvariantError(ValueError): pass
SEND_IDENTITY_STATUSES = frozenset({"OVERLAY_EXACT_SINGLE_LEAD", "REGISTRY_EXACT_SINGLE_LEAD"})
ROP_DECISIONS = ("", "ОК", "ПРАВИТЬ", "НЕ ПИСАТЬ")
CAMPAIGN_RESULTS = ("", "ОТПРАВЛЕНО-1", "ОТПРАВЛЕНО-2", "ОТПРАВЛЕНО-3", "ОТПРАВЛЕНО-4",
                    "ОТПРАВЛЕНО-5", "ОТВЕТИЛ", "КУПИЛ", "СТОП")

AMO_BASE = "https://educent.amocrm.ru"
MOSCOW = ZoneInfo("Europe/Moscow")
STAMP_FORMAT = "%d.%m.%Y %H:%M"
# Сохранённый лист «Звонки»: 16 колонок, значимы шесть.
EVIDENCE_SCHEMA_VERSION = "wappi_outreach_calls_sheet_v1"
EVIDENCE_FIELDS = {"report_row": 0, "call_at_moscow": 1, "phone": 6, "summary": 9, "next_step": 12, "transcript": 15}
EVIDENCE_WIDTH = 16
BRAND_TITLES = {"foton": "Фотон", "unpk": "УНПК МФТИ"}
CHANNEL_TITLES = {"telegram": "Telegram", "max": "Max"}


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(f"JSON array contains a non-object record: {path}")
        return [dict(row) for row in payload]
    for name in ("candidates", "records", "items", "rows"):
        rows = payload.get(name) if isinstance(payload, Mapping) else None
        if isinstance(rows, list):
            if not all(isinstance(row, Mapping) for row in rows):
                raise ValueError(f"JSON wrapper contains a non-object record: {path}")
            return [dict(row) for row in rows]
    raise ValueError(f"unsupported JSON payload: {path}")


def _calls_evidence_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, Mapping)
            or payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or payload.get("source_sheet") != "Звонки"
            or not isinstance(payload.get("rows"), list)):
        raise ValueError(f"calls evidence schema mismatch: {path}")
    rows = [dict(row) for row in payload["rows"] if isinstance(row, Mapping)]
    if len(rows) != len(payload["rows"]) or any(
        not isinstance(row.get("values"), list) or len(row["values"]) != EVIDENCE_WIDTH for row in rows
    ):
        raise ValueError(f"calls evidence needs exactly {EVIDENCE_WIDTH} columns per row: {path}")
    return rows


def _one_by_key(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        profile_id, chat_id = str(row.get("profile_id") or "").strip(), str(row.get("chat_id") or "").strip()
        if not profile_id or not chat_id:
            raise ValueError(f"{label} row needs both profile_id and chat_id: {profile_id!r}:{chat_id!r}")
        key = f"{profile_id}:{chat_id}"
        if key in result:
            raise ValueError(f"duplicate {label} key: {key}")
        result[key] = dict(row)
    return result


def _seq(value: Any) -> list[Any]:
    """Списковое поле, которое в сырье может прийти скаляром — строка не рассыпается на символы."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _ids(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _embedded_ids(row: Mapping[str, Any], name: str) -> list[str]:
    embedded = row.get("record", {}).get("_embedded", {}) if isinstance(row.get("record"), Mapping) else {}
    values = embedded.get(name, []) if isinstance(embedded, Mapping) else []
    return _ids(item.get("id") for item in values if isinstance(item, Mapping))


def _phone_digits(value: Any) -> str:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) == 11 and digits.startswith("8"):
        return "7" + digits[1:]
    return "7" + digits if len(digits) == 10 else digits


def _parsed_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso).replace(microsecond=0)
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%d.%m.%Y %H:%M:%S").replace(microsecond=0)
    except ValueError:
        return None


def _epoch(value: Any) -> int:
    try:
        stamp = int(value or 0)
    except (TypeError, ValueError):
        return 0
    while stamp > 10 ** 11:  # миллисекунды в некоторых выгрузках Wappi
        stamp //= 1000
    return stamp


def _call_join(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """Ключ join сайдкара: normalized_phone + call_at (UTC без TZ) → Europe/Moscow, точность секунда."""
    phone, stamp = _phone_digits(row.get("normalized_phone")), _parsed_datetime(row.get("call_at"))
    if not phone or stamp is None:
        return None
    moscow = (stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp).astimezone(MOSCOW)
    return phone, moscow.replace(tzinfo=None).isoformat()


def _evidence_join(row: Mapping[str, Any]) -> tuple[str, str] | None:
    values = row.get("values")
    if not isinstance(values, (list, tuple)) or len(values) != EVIDENCE_WIDTH:
        return None
    phone = _phone_digits(values[EVIDENCE_FIELDS["phone"]])
    stamp = _parsed_datetime(values[EVIDENCE_FIELDS["call_at_moscow"]])
    if not phone or stamp is None:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(MOSCOW).replace(tzinfo=None)
    return phone, stamp.isoformat()


def _link_class(base: Mapping[str, Any] | None, overlay: Mapping[str, Any] | None) -> tuple[str, bool]:
    if overlay is not None:
        if not str(overlay.get("amo_contact_id") or "").strip():
            return "OVERLAY_MISSING_CONTACT", False
        return "OVERLAY", True
    if base is None:
        return "REGISTRY_MISSING", False
    if base.get("status") == "LINK_EXTRACTION_GAP":
        return "REGISTRY_GAP", False
    if base.get("status") == "LINK_CONFLICT":
        return "REGISTRY_CONFLICT", False
    if not str(base.get("amo_contact_id") or "").strip():
        return "REGISTRY_EXTRACTED_MISSING_CONTACT", False
    return "REGISTRY", True


def build_case_packets(
    *,
    candidates_path: Path,
    wappi_registry_path: Path,
    wappi_link_delta_path: Path | None,
    amo_contacts_path: Path,
    amo_leads_path: Path,
    amo_events_path: Path,
    tallanto_path: Path,
    calls_sidecar_path: Path,
    calls_evidence_path: Path | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    paths = {
        "candidates": candidates_path,
        "wappi_registry": wappi_registry_path,
        "wappi_link_delta": wappi_link_delta_path,
        "amo_contacts": amo_contacts_path,
        "amo_leads": amo_leads_path,
        "amo_events": amo_events_path,
        "tallanto": tallanto_path,
        "calls_sidecar": calls_sidecar_path,
        "calls_evidence": calls_evidence_path,
    }
    candidates = _one_by_key(_records(candidates_path), label="candidate")
    registry = _one_by_key(_records(wappi_registry_path), label="registry")
    link_delta = _one_by_key(_records(wappi_link_delta_path), label="link delta")
    contacts, leads = _records(amo_contacts_path), _records(amo_leads_path)
    events, tallanto = _records(amo_events_path), _records(tallanto_path)
    calls, call_evidence = _records(calls_sidecar_path), _calls_evidence_records(calls_evidence_path)

    contacts_by_id = defaultdict(list)
    leads_by_id = defaultdict(list)
    lead_ids_by_contact = defaultdict(set)
    events_by_entity = defaultdict(list)
    tallanto_by_contact = defaultdict(list)
    tallanto_by_id = defaultdict(list)
    calls_by_contact = defaultdict(list)
    for row in contacts:
        contacts_by_id[str(row.get("entity_id") or row.get("id") or "")].append(row)
    for row in leads:
        lead_id = str(row.get("entity_id") or row.get("id") or "")
        leads_by_id[lead_id].append(row)
        for contact_id in _embedded_ids(row, "contacts"):
            lead_ids_by_contact[contact_id].add(lead_id)
    for row in events:
        events_by_entity[(str(row.get("entity_type") or ""), str(row.get("entity_id") or ""))].append(row)
    for row in tallanto:
        tallanto_by_contact[str(row.get("amo_contact_id") or "")].append(row)
        tallanto_by_id[str(row.get("tallanto_id") or "")].append(row)
    for row in calls:
        for contact_id in _ids([row.get("amo_contact_id"), *_seq(row.get("amo_contact_ids"))]):
            calls_by_contact[contact_id].append(row)

    # Точный уникальный join звонка и его расшифровки: телефон + секунда Europe/Moscow, без окна и эвристик.
    calls_by_join = defaultdict(list)
    calls_unparsed = 0
    for row in calls:
        join = _call_join(row)
        if join is None:
            calls_unparsed += 1
        else:
            calls_by_join[join].append(row)
    evidence_by_join = defaultdict(list)
    for row in call_evidence:
        join = _evidence_join(row)
        if join is not None:
            evidence_by_join[join].append(row)
    attached: dict[tuple[str, str], dict[str, Any]] = {}
    unmatched = ambiguous = 0
    for join, rows in evidence_by_join.items():
        peers = calls_by_join.get(join, ())
        if not peers:
            unmatched += len(rows)
        elif len(rows) == 1 and len(peers) == 1:
            values = rows[0]["values"]
            attached[join] = {"call_id": str(peers[0].get("call_id") or ""),
                              **{name: values[index] for name, index in EVIDENCE_FIELDS.items()}}
        else:
            ambiguous += len(rows)
    joinable = sum(len(rows) for rows in evidence_by_join.values())

    def evidence_for(row: Mapping[str, Any]) -> list[dict[str, Any]]:
        found = attached.get(_call_join(row) or ("", ""))
        return [found] if found else []

    keys = sorted(candidates)
    if offset < 0 or (limit is not None and limit < 0):
        raise ValueError("offset and limit must be non-negative")
    selected_keys = keys[offset : None if limit is None else offset + limit]
    cases: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)
    for key in selected_keys:
        candidate = candidates[key]
        base_link, delta_link = registry.get(key), link_delta.get(key)
        # Исправляющая дельта заменяет идентичность целиком. Старые IDs остаются только
        # в отдельном wappi_registry для аудита и не попадают в effective/LLM-контекст.
        effective = dict(delta_link) if delta_link is not None else dict(base_link or {})
        link_class, verified = _link_class(base_link, delta_link)
        reported_contact_ids = _ids([effective.get("amo_contact_id")])
        contact_ids = reported_contact_ids if verified else []
        contact_rows = [row for item in contact_ids for row in contacts_by_id[item]]
        lead_ids = _ids(
            [*_seq(effective.get("amo_lead_ids")), effective.get("amo_lead_id")]
            + [lead_id for item in contact_ids for lead_id in lead_ids_by_contact[item]]
            + [lead_id for row in contact_rows for lead_id in _embedded_ids(row, "leads")]
        ) if verified else []
        lead_record_conflicts = _ids(
            lead_id for lead_id in lead_ids
            if leads_by_id[lead_id]
            and all(_embedded_ids(row, "contacts") and not (set(_embedded_ids(row, "contacts")) & set(contact_ids))
                    for row in leads_by_id[lead_id])
        )
        safe_lead_ids = [lead_id for lead_id in lead_ids if lead_id not in lead_record_conflicts]
        if verified:
            final_kind = ("EXACT_SINGLE_LEAD" if len(safe_lead_ids) == 1 else
                          "EXACT_MULTIPLE_LEADS" if len(safe_lead_ids) > 1 else "EXACT_CONTACT_ONLY")
            link_class = f"{link_class}_{final_kind}"
        lead_rows = [
            row for lead_id in safe_lead_ids for row in leads_by_id[lead_id]
            if not _embedded_ids(row, "contacts") or set(_embedded_ids(row, "contacts")) & set(contact_ids)
        ]
        customer_ids = _ids([row.get("customer_id") for row in [*contact_rows, *lead_rows]])
        event_rows = [
            *[row for item in contact_ids for row in events_by_entity[("contact", item)]],
            *[row for item in safe_lead_ids for row in events_by_entity[("lead", item)]],
        ]
        # Overlay полностью заменяет старую идентичность: старые Tallanto IDs нельзя переносить
        # к исправленному AMO-контакту.
        tallanto_ids = _ids(_seq(effective.get("tallanto_ids"))) if verified else []
        tallanto_matches = [
            *[row for item in contact_ids for row in tallanto_by_contact[item]],
            *[row for item in tallanto_ids for row in tallanto_by_id[item]
              if not str(row.get("amo_contact_id") or "").strip()
              or str(row.get("amo_contact_id")) in contact_ids],
        ]
        tallanto_rows = list({str(row.get("source_id") or row.get("tallanto_id") or _digest(row)): row
                              for row in tallanto_matches}.values())
        linked_calls = list({str(row.get("call_id") or _digest(row)): row
                             for item in contact_ids for row in calls_by_contact[item]}.values())
        exact_calls = [row for row in linked_calls if row.get("match_status") == "EXACT_AMO_CONTACT"]
        family_calls = [row for row in linked_calls if row.get("match_status") == "SHARED_PHONE_CONTEXT"]
        other_calls = [row for row in linked_calls if row not in exact_calls and row not in family_calls]
        if len(linked_calls) != len(exact_calls) + len(family_calls) + len(other_calls):
            raise RuntimeError(f"linked calls balance failed: {key}")
        status_counts[link_class] += 1
        raw = {
            "wappi_conversation": {name: candidate.get(name) for name in WAPPI_FIELDS},
            "wappi_registry": base_link,
            "wappi_link_delta": delta_link,
            "wappi_effective_link": effective,
            "amo_contacts": contact_rows,
            "amo_leads": lead_rows,
            "amo_events": event_rows,
            "tallanto": tallanto_rows,
            "calls_exact": [
                {"context_only": False, "link": row, "evidence": evidence_for(row)} for row in exact_calls
            ],
            "calls_family": [
                {"context_only": True, "link": row, "evidence": evidence_for(row)} for row in family_calls
            ],
            "calls_unclassified": [
                {"context_only": True, "link": row, "evidence": evidence_for(row)} for row in other_calls
            ],
        }
        cases.append({
            "key": key,
            "profile_id": str(candidate.get("profile_id") or ""),
            "chat_id": str(candidate.get("chat_id") or ""),
            "identity": {
                "status": link_class,
                "amo_contact_ids": contact_ids,
                "reported_amo_contact_ids": reported_contact_ids,
                "amo_lead_ids": safe_lead_ids,
                "reported_amo_lead_ids": lead_ids,
                "amo_lead_record_conflict_ids": lead_record_conflicts,
                "amo_records_current": bool(contact_rows) and bool(lead_rows),
                "customer_ids": customer_ids,
                "tallanto_ids": _ids(row.get("tallanto_id") for row in tallanto_rows),
                "registry_status": (base_link or {}).get("status"),
            },
            "source_fingerprint": _digest(raw),
            "raw_linked_facts": raw,
        })

    source_hashes = {name: _file_digest(path) for name, path in paths.items()}
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _digest(source_hashes),
        "source_sha256": source_hashes,
        "offset": offset,
        "limit": limit,
        "total_candidates": len(keys),
        "case_count": len(cases),
        "source_row_counts": {
            "candidates": len(candidates), "wappi_registry": len(registry), "wappi_link_delta": len(link_delta),
            "amo_contacts": len(contacts), "amo_leads": len(leads), "amo_events": len(events),
            "tallanto": len(tallanto), "calls_sidecar": len(calls), "calls_evidence": len(call_evidence),
        },
        "link_class_counts": dict(sorted(status_counts.items())),
        "calls_evidence_diagnostics": {
            "sidecar_rows": len(calls),
            "sidecar_unparsed": calls_unparsed,
            "rows": len(call_evidence),
            "unparsed": len(call_evidence) - joinable,
            "matched_exact": len(attached),
            "unmatched": unmatched,
            "ambiguous": ambiguous,
        },
        "cases": cases,
    }
    diagnostics = result["calls_evidence_diagnostics"]
    if diagnostics["rows"] != sum(diagnostics[k] for k in ("unparsed", "matched_exact", "unmatched", "ambiguous")):
        raise RuntimeError("calls evidence balance failed")
    if result["case_count"] != sum(result["link_class_counts"].values()):
        raise RuntimeError("closed balance failed")
    return result


def _amo_url(identity: Mapping[str, Any]) -> str:
    """Ссылка только на числовой ID AMO — иначе пусто, чтобы менеджер не открывал чужую карточку."""
    leads = [item for item in identity.get("amo_lead_ids") or () if str(item).isdigit()]
    contacts = [item for item in identity.get("amo_contact_ids") or () if str(item).isdigit()]
    if len(leads) == 1:
        return f"{AMO_BASE}/leads/detail/{leads[0]}"
    return f"{AMO_BASE}/contacts/detail/{contacts[0]}" if len(contacts) == 1 else ""


def _brand_channel(chat: Mapping[str, Any], *, key: str) -> str:
    brand = BRAND_TITLES.get(str(chat.get("brand") or "").strip().lower())
    channel = CHANNEL_TITLES.get(str(chat.get("channel") or "").strip().lower())
    if brand is None or channel is None:
        raise ValueError(f"unknown brand/channel for {key}: {chat.get('brand')!r}/{chat.get('channel')!r}")
    return f"{brand} · {channel}"


def _client_name(case: Mapping[str, Any]) -> str:
    """Единственное точное имя AMO; иначе имена собеседника из чата; иначе пусто — без выдумки."""
    names = _ids(row.get("name") for row in case["raw_linked_facts"]["amo_contacts"])
    if len(names) == 1:
        return names[0]
    return ", ".join(_ids(_seq(case["raw_linked_facts"]["wappi_conversation"].get("peer_names"))))


def _last_contact(chat: Mapping[str, Any]) -> str:
    inbound, outbound = _epoch(chat.get("last_inbound_ts")), _epoch(chat.get("last_outbound_ts"))
    stamp = max(inbound, outbound)
    if stamp <= 0:
        return ""
    side = "клиент" if inbound >= outbound else "мы"
    return f"{datetime.fromtimestamp(stamp, MOSCOW).strftime(STAMP_FORMAT)} · последним: {side}"


def _validate_verdict(key: str, verdict: Any) -> dict[str, str]:
    if not isinstance(verdict, Mapping):
        raise ValueError(f"verdict must be an object: {key}")
    missing = sorted(set(VERDICT_COLUMNS) - set(verdict))
    unknown = sorted(set(verdict) - set(VERDICT_COLUMNS))
    if missing or unknown:
        raise ValueError(f"verdict schema mismatch for {key}: missing={missing} unknown={unknown}")
    if not all(isinstance(value, str) for value in verdict.values()):
        raise ValueError(f"verdict fields must be strings: {key}")
    if verdict["status"] not in STATUS_VALUES:
        raise ValueError(f"status outside schema for {key}: {verdict['status']!r}")
    if verdict["freshness"] not in FRESHNESS_VALUES:
        raise ValueError(f"freshness outside schema for {key}: {verdict['freshness']!r}")
    if verdict["status"] == SEND_STATUS and verdict["freshness"] != "АКТУАЛЬНО":
        raise ValueError(f"send status requires current sources for {key}")
    messages = [verdict[f"message_{index}"] for index in range(1, 6)]
    dates = [verdict[f"date_{index}"] for index in range(1, 6)]
    if verdict["status"] == "НЕ ПИСАТЬ":
        return {name: "" if name.startswith(("message_", "date_")) or name == "signature" else value
                for name, value in verdict.items()}
    if any(bool(message.strip()) != bool(date.strip()) for message, date in zip(messages, dates)):
        raise VerdictContentInvariantError(f"message/date mismatch for {key}")
    if verdict["status"] == "ПРОВЕРКА" and any(
        value.strip() for value in [*messages, *dates, verdict["signature"]]
    ):
        raise VerdictContentInvariantError(f"non-send status carries campaign text for {key}")
    return dict(verdict)


def _error_verdict() -> dict[str, str]:
    verdict = {name: "" for name in VERDICT_COLUMNS}
    verdict.update(status="ПРОВЕРКА", next_step="Повторить смысловой анализ", freshness="ОБНОВИТЬ")
    return verdict


def _verdicts_by_key(
    rows: Iterable[Mapping[str, Any]], *, allowed: set[str], required: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    result: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("key") or "").strip()
        if not key or key not in allowed:
            raise ValueError(f"unknown verdict key: {key!r}")
        if key in seen:
            raise ValueError(f"duplicate verdict key: {key}")
        seen.add(key)
        if set(row) != {"key", "verdict"}:
            errors[key] = "VERDICT_INVALID"
            continue
        try:
            result[key] = _validate_verdict(key, row["verdict"])
        except ValueError:
            errors[key] = "VERDICT_INVALID"
    errors.update({key: "VERDICT_MISSING" for key in required - seen})
    return result, errors


def _previous_verdicts(
    previous_sheet: Mapping[str, Any] | None,
) -> dict[str, tuple[str, dict[str, Any], str | None]]:
    if previous_sheet is None:
        return {}
    if previous_sheet.get("schema_version") != SHEET_SCHEMA_VERSION:
        raise ValueError(f"previous sheet schema mismatch: {previous_sheet.get('schema_version')!r}")
    result: dict[str, tuple[str, dict[str, Any], str | None]] = {}
    for row in previous_sheet.get("rows") or ():
        key, source = str(row.get("key") or ""), str(row.get("source_fingerprint") or "")
        if not key or not source or key in result:
            raise ValueError(f"previous sheet row is unusable: {key!r}")
        sticky_status = None
        if row.get("manual_status_override"):
            values = row.get("values") or {}
            a_w = values.get("A:W") if isinstance(values, Mapping) else None
            rendered = str(a_w[1]) if isinstance(a_w, list) and len(a_w) > 1 else ""
            if rendered not in STATUS_VALUES:
                raise ValueError(f"previous sheet manual status is unusable: {key!r}")
            sticky_status = rendered
        try:
            verdict = _validate_verdict(key, row.get("verdict"))
        except VerdictContentInvariantError:
            source, verdict = "", _error_verdict()
        result[key] = (source, verdict, sticky_status)
    return result


def _manual_state_by_key(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if set(row) != {"key", "rop_decision", "result"}:
            raise ValueError("manual state row must hold exactly key+rop_decision+result")
        key = str(row["key"] or "").strip()
        decision, outcome = str(row["rop_decision"] or "").strip(), str(row["result"] or "").strip()
        if not key or key in result:
            raise ValueError(f"duplicate or empty manual state key: {key!r}")
        if decision not in ROP_DECISIONS or outcome not in CAMPAIGN_RESULTS:
            raise ValueError(f"manual state outside schema for {key}: {decision!r}/{outcome!r}")
        result[key] = {"rop_decision": decision, "result": outcome}
    return result


def _status_after_manual_state(proposed: str, state: Mapping[str, str]) -> str:
    outcome, decision = state.get("result", ""), state.get("rop_decision", "")
    if outcome in {"КУПИЛ", "СТОП"} or decision == "НЕ ПИСАТЬ":
        return "НЕ ПИСАТЬ"
    if outcome == "ОТВЕТИЛ":
        return "МЕНЕДЖЕР СЕГОДНЯ"
    if outcome in CAMPAIGN_RESULTS[1:6] or decision == "ПРАВИТЬ":
        return "ПРОВЕРКА"
    return proposed


def build_sheet_rows(
    payload: Mapping[str, Any],
    verdict_rows: Iterable[Mapping[str, Any]],
    *,
    checked_at: str,
    previous_sheet: Mapping[str, Any] | None = None,
    manual_state_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Собрать канонические строки листа из пакетов и ровно одного вердикта LLM на ключ.

    Возвращаются только физически записываемые блоки A:W и Z:AB плюс якорь по колонке A.
    X:Y (решение РОПа и результат) в значениях отсутствуют, поэтому писателю нечем их затереть.
    Если `previous_sheet` задан и отпечаток источника ключа не менялся, вердикт переиспользуется
    без нового вызова LLM; изменившийся или отсутствующий источник требует свежего вердикта.
    """
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"case payload schema mismatch: {payload.get('schema_version')!r}")
    if not str(checked_at or "").strip():
        raise ValueError("checked_at must be a non-empty stamp")
    source_by_key = {str(case["key"]): case["source_fingerprint"] for case in payload["cases"]}
    previous = _previous_verdicts(previous_sheet)
    reusable = {key: verdict for key, (source, verdict, _) in previous.items()
                if source_by_key.get(key) == source}
    sticky_statuses = {key: status for key, (_, _, status) in previous.items() if status}
    required = set(source_by_key) - set(reusable)
    verdicts, analysis_errors = _verdicts_by_key(verdict_rows, allowed=set(source_by_key), required=required)
    manual_state = _manual_state_by_key(manual_state_rows)

    rows: list[dict[str, Any]] = []
    for case in sorted(payload["cases"], key=lambda item: str(item["key"])):
        key, identity = str(case["key"]), case["identity"]
        chat = case["raw_linked_facts"]["wappi_conversation"]
        used_reusable = key not in verdicts and key in reusable and key not in analysis_errors
        verdict = verdicts.get(key) or (reusable.get(key) if used_reusable else None) or _error_verdict()
        state = manual_state.get(key, {})
        rendered_status = _status_after_manual_state(verdict["status"], state)
        rendered_freshness = verdict["freshness"]
        if state.get("result") in {"КУПИЛ", "СТОП"} or state.get("rop_decision") == "НЕ ПИСАТЬ":
            rendered_freshness = "СТОП ДЕЙСТВУЕТ"
        if key not in manual_state and key in sticky_statuses:
            rendered_status = sticky_statuses[key]
            if rendered_status == "НЕ ПИСАТЬ":
                rendered_freshness = "СТОП ДЕЙСТВУЕТ"
        if rendered_status == SEND_STATUS:
            missing_call_evidence = any(not item["evidence"] for item in case["raw_linked_facts"]["calls_exact"])
            exact_identity = (
                identity["status"] in SEND_IDENTITY_STATUSES
                and len(identity["amo_contact_ids"]) == 1
                and len(identity["amo_lead_ids"]) == 1
                and identity["amo_records_current"]
            )
            if not exact_identity or case["raw_linked_facts"]["calls_unclassified"] or missing_call_evidence:
                analysis_errors[key] = "CALL_EVIDENCE_MISSING" if missing_call_evidence else "UNSAFE_SEND"
                verdict = _error_verdict()
                rendered_status = _status_after_manual_state(verdict["status"], manual_state.get(key, {}))
                rendered_freshness = verdict["freshness"]
        verdict_fingerprint = _digest(verdict)
        cells: list[str | None] = [None] * len(SHEET_COLUMNS)
        for name, index in VERDICT_COLUMNS.items():
            cells[index] = verdict[name]
        if rendered_status in {"ПРОВЕРКА", "НЕ ПИСАТЬ"}:
            for name in [*(f"message_{i}" for i in range(1, 6)), *(f"date_{i}" for i in range(1, 6)), "signature"]:
                cells[VERDICT_COLUMNS[name]] = ""
        cells[1] = rendered_status
        cells[26] = rendered_freshness
        cells[0] = key
        cells[3] = _brand_channel(chat, key=key)
        cells[4] = _amo_url(identity)
        cells[5] = _client_name(case)
        cells[6] = _last_contact(chat)
        evidence = [
            f"identity={identity['status']}",
            f"amo_contact_ids={','.join(identity['amo_contact_ids'])}",
            f"amo_lead_ids={','.join(identity['amo_lead_ids'])}",
            f"source_fingerprint={case['source_fingerprint']}",
            f"verdict_fingerprint={verdict_fingerprint}",
        ]
        if key in analysis_errors:
            evidence.append(f"analysis_error={analysis_errors[key]}")
        cells[25] = " · ".join(evidence)
        cells[27] = checked_at
        values = {name: cells[WRITE_SLICES[name]] for name in WRITE_RANGES}
        if any(value is None for block in values.values() for value in block):
            raise RuntimeError(f"written block has an unfilled cell: {key}")
        rows.append({
            "key": key,
            "anchor": {"column": ANCHOR_COLUMN, "value": key},
            "source_fingerprint": case["source_fingerprint"],
            "verdict_fingerprint": verdict_fingerprint,
            # Штамп проверки (AB) намеренно вне отпечатка: перепроверка без изменений не «меняет» строку.
            "row_fingerprint": _digest([case["source_fingerprint"], verdict_fingerprint,
                                        values["A:W"], values["Z:AB"][:-1]]),
            "reused_verdict": used_reusable,
            "manual_status_override": rendered_status != verdict["status"],
            "analysis_error": analysis_errors.get(key),
            "verdict": verdict,
            "values": values,
        })

    if len(rows) != len(source_by_key) or len({row["key"] for row in rows}) != len(rows):
        raise RuntimeError("closed balance failed")
    reused = sum(1 for row in rows if row["reused_verdict"])
    sheet_status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        sheet_status_counts[row["values"]["A:W"][1]] += 1
    if len(rows) != sum(sheet_status_counts.values()):
        raise RuntimeError("sheet status balance failed")
    return {
        "schema_version": SHEET_SCHEMA_VERSION,
        "case_schema_version": payload["schema_version"],
        "source_fingerprint": payload["source_fingerprint"],
        "checked_at": checked_at,
        "columns": list(SHEET_COLUMNS),
        "column_count": len(SHEET_COLUMNS),
        "manual_columns": {
            "range": "X:Y",
            "indexes": list(MANUAL_COLUMNS),
            "headers": [SHEET_COLUMNS[index] for index in MANUAL_COLUMNS],
            "policy": "preserve",
            "excluded_from_output": True,
            "allowed_decisions": list(ROP_DECISIONS),
            "allowed_results": list(CAMPAIGN_RESULTS),
        },
        "write_ranges": list(WRITE_RANGES),
        "row_count": len(rows),
        "status_counts": dict(sorted(sheet_status_counts.items())),
        "analysis_errors": {"count": len(analysis_errors), "by_key": dict(sorted(analysis_errors.items()))},
        "reuse": {"new": len(rows) - reused, "reused": reused, "required_keys": sorted(required)},
        "rows": rows,
    }


INPUT_ARGS = (
    "candidates", "wappi_registry", "wappi_link_delta", "amo_contacts", "amo_leads", "amo_events",
    "tallanto", "calls_sidecar", "calls_evidence", "verdicts", "previous_sheet", "manual_state",
)


def _check_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Входы строго read-only: ни один выход не может совпасть со входом или с другим выходом."""
    input_items = [(Path(getattr(args, name)), f"--{name.replace('_', '-')}")
                   for name in INPUT_ARGS if getattr(args, name) is not None]
    missing = [f"{flag}={path}" for path, flag in input_items if not path.exists()]
    if missing:
        parser.error(f"input does not exist: {', '.join(missing)}")
    inputs = {path.resolve(): flag for path, flag in input_items}
    seen: list[tuple[Path, str]] = []
    for flag, path in (("--out", args.out), ("--sheet-out", args.sheet_out)):
        if path is None:
            continue
        resolved = Path(path).resolve()
        same_input = next((input_flag for input_path, input_flag in input_items
                           if Path(path).exists() and os.path.samefile(path, input_path)), None)
        if resolved in inputs or same_input:
            parser.error(f"{flag} must not overwrite input {same_input or inputs[resolved]}: {path}")
        same_output = next((other_flag for other_path, other_flag in seen
                            if str(resolved).casefold() == str(other_path).casefold()
                            or (Path(path).exists() and other_path.exists() and os.path.samefile(path, other_path))), None)
        if same_output:
            parser.error(f"{flag} must differ from {same_output}: {path}")
        seen.append((resolved, flag))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic local Wappi outreach case packets.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--wappi-registry", required=True, type=Path)
    parser.add_argument("--wappi-link-delta", type=Path)
    parser.add_argument("--amo-contacts", required=True, type=Path)
    parser.add_argument("--amo-leads", required=True, type=Path)
    parser.add_argument("--amo-events", required=True, type=Path)
    parser.add_argument("--tallanto", required=True, type=Path)
    parser.add_argument("--calls-sidecar", required=True, type=Path)
    parser.add_argument("--calls-evidence", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verdicts", type=Path, help="JSON/JSONL list of {key, verdict} records")
    parser.add_argument("--sheet-out", type=Path)
    parser.add_argument("--checked-at", help="verification stamp for column AB (no clock is read)")
    parser.add_argument("--previous-sheet", type=Path, help="prior --sheet-out payload; reuses unchanged verdicts")
    parser.add_argument("--manual-state", type=Path, help="current Google X:Y rows: key+rop_decision+result")
    args = parser.parse_args()
    sheet_args = (args.verdicts, args.sheet_out, args.checked_at)
    if any(sheet_args) and not all(sheet_args):
        parser.error("--verdicts, --sheet-out and --checked-at are required together")
    if args.previous_sheet is not None and not all(sheet_args):
        parser.error("--previous-sheet needs --verdicts, --sheet-out and --checked-at")
    if args.manual_state is not None and not all(sheet_args):
        parser.error("--manual-state needs --verdicts, --sheet-out and --checked-at")
    _check_paths(parser, args)

    payload = build_case_packets(
        candidates_path=args.candidates,
        wappi_registry_path=args.wappi_registry,
        wappi_link_delta_path=args.wappi_link_delta,
        amo_contacts_path=args.amo_contacts,
        amo_leads_path=args.amo_leads,
        amo_events_path=args.amo_events,
        tallanto_path=args.tallanto,
        calls_sidecar_path=args.calls_sidecar,
        calls_evidence_path=args.calls_evidence,
        offset=args.offset,
        limit=args.limit,
    )
    # Сначала собираем и проверяем всё, и только потом пишем: ошибка вердикта не оставляет частичный выход.
    outputs = [(args.out, _json_bytes(payload, indent=2))]
    sheet = None
    if args.verdicts:
        previous = json.loads(args.previous_sheet.read_text(encoding="utf-8")) if args.previous_sheet else None
        sheet = build_sheet_rows(payload, _records(args.verdicts), checked_at=args.checked_at,
                                 previous_sheet=previous, manual_state_rows=_records(args.manual_state))
        outputs.append((args.sheet_out, _json_bytes(sheet, indent=2)))
    for path, data in outputs:
        _write_atomic(path, data)

    print(f"out={args.out} cases={payload['case_count']} fingerprint={payload['source_fingerprint']}")
    if sheet is not None:
        print(f"sheet={args.sheet_out} rows={sheet['row_count']} write_ranges={','.join(sheet['write_ranges'])} "
              f"new={sheet['reuse']['new']} reused={sheet['reuse']['reused']} "
              f"errors={sheet['analysis_errors']['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
