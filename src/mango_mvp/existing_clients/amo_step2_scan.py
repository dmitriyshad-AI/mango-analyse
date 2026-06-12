from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from mango_mvp.customer_profile.crm_summary import ProfileRecord, load_profiles_for_summary
from mango_mvp.existing_clients.amo_step1_snapshot import (
    DEFAULT_PAGE_LIMIT,
    DEFAULT_SLEEP_SEC,
    AmoMcpClient,
    embedded_items,
    epoch_to_dt,
    extract_contact_phones,
    fetch_amo_collection,
    guard_output_root,
)
from mango_mvp.utils.phone import normalize_phone


AMO_STEP2_SCHEMA_VERSION = "tz14_amo_step2_scan_v1"
DEFAULT_PROFILES_DB = Path("product_data/customer_profiles/tz12_working_batch3/customer_profiles.sqlite")
DEFAULT_OUT_ROOT = Path("product_data/customer_profiles/tz14_amo_step2_scan")
CALLBACK_INTENT = "callback"
CALLBACK_CHANNEL = "Telegram"


@dataclass(frozen=True)
class NewLeadScanOptions:
    project_root: Path
    out_root: Path = DEFAULT_OUT_ROOT
    profiles_db: Path = DEFAULT_PROFILES_DB
    since: datetime | None = None
    client: AmoMcpClient | None = None
    page_limit: int = DEFAULT_PAGE_LIMIT
    sleep_sec: float = DEFAULT_SLEEP_SEC
    max_pages: int | None = None
    max_leads: int | None = None
    callback_requests_path: Path | None = None
    enable_amo_notes: bool = False
    enable_amo_tasks: bool = False
    generated_at: datetime | None = None


def build_step2_scan(options: NewLeadScanOptions) -> Mapping[str, Any]:
    if options.enable_amo_notes or options.enable_amo_tasks:
        raise PermissionError("TZ-14 Step 2 live AMO writes are disabled in this implementation")
    if options.page_limit < 1 or options.page_limit > 50:
        raise ValueError("page_limit must be between 1 and 50")
    if options.sleep_sec < 0.5:
        raise ValueError("sleep_sec must be at least 0.5 seconds for AMO reads")
    if options.client is None:
        raise ValueError("client is required for AMO read-only scan")

    project_root = options.project_root.expanduser().resolve(strict=False)
    out_root = _resolve_out_root(project_root, options.out_root)
    guard_output_root(project_root=project_root, out_root=out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    now = options.generated_at or datetime.now(timezone.utc)
    since = _require_aware(options.since or now - timedelta(minutes=15), "since")
    journal = Step2Journal(out_root / "step2_scan_journal.sqlite")
    try:
        leads, contact_cache, lead_pages = fetch_recent_leads_with_contacts(
            client=options.client,
            since=since,
            page_limit=options.page_limit,
            sleep_sec=options.sleep_sec,
            max_pages=options.max_pages,
            max_leads=options.max_leads,
        )
        note_result = build_family_note_drafts(
            leads=leads,
            contact_cache=contact_cache,
            profiles_db=options.profiles_db,
            journal=journal,
            generated_at=now,
        )
        task_result = build_callback_task_drafts(
            callback_requests_path=options.callback_requests_path,
            journal=journal,
            generated_at=now,
        )
        outputs = write_step2_outputs(
            out_root=out_root,
            note_rows=note_result["rows"],
            task_rows=task_result["rows"],
            skipped_rows=[*note_result["skipped_rows"], *task_result["skipped_rows"]],
        )
        summary = {
            "schema_version": AMO_STEP2_SCHEMA_VERSION,
            "generated_at": now.isoformat(timespec="seconds"),
            "since": since.isoformat(timespec="seconds"),
            "out_root": str(out_root),
            "profiles_db": str(options.profiles_db),
            "read_only": True,
            "write_crm": False,
            "enable_amo_notes": False,
            "enable_amo_tasks": False,
            "page_limit": options.page_limit,
            "sleep_sec": options.sleep_sec,
            "lead_pages": lead_pages,
            "leads_seen": len(leads),
            "contacts_fetched": len(contact_cache),
            "counts": {
                **note_result["counts"],
                **task_result["counts"],
            },
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        _write_json(outputs["summary_json"], summary)
        return summary
    finally:
        journal.close()


def fetch_recent_leads_with_contacts(
    *,
    client: AmoMcpClient,
    since: datetime,
    page_limit: int,
    sleep_sec: float,
    max_pages: int | None = None,
    max_leads: int | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]], int]:
    since = _require_aware(since, "since")
    params = {
        "with": "contacts",
        "filter[created_at][from]": int(since.timestamp()),
        "order[created_at]": "asc",
    }
    leads, pages = fetch_amo_collection(
        client,
        path="leads",
        embedded_key="leads",
        params=params,
        page_limit=page_limit,
        sleep_sec=sleep_sec,
        max_items=max_leads,
        max_pages=max_pages,
        collection_name="",
        checkpoint_root=None,
    )
    contact_cache: dict[str, Mapping[str, Any]] = {}
    for lead in leads:
        for ref in embedded_items(lead, "contacts"):
            contact_id = str(ref.get("id") or "").strip()
            if not contact_id or contact_id in contact_cache:
                continue
            payload = client.amo_api_get(path=f"contacts/{contact_id}", params={}, limit=1)
            contact = _first_embedded(payload, "contacts")
            if contact is None and isinstance(payload, Mapping) and payload.get("id"):
                contact = payload
            if isinstance(contact, Mapping):
                contact_cache[contact_id] = dict(contact)
    return leads, contact_cache, pages


def build_family_note_drafts(
    *,
    leads: Sequence[Mapping[str, Any]],
    contact_cache: Mapping[str, Mapping[str, Any]],
    profiles_db: Path,
    journal: "Step2Journal",
    generated_at: datetime,
) -> Mapping[str, Any]:
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    counts = {
        "family_note_drafts": 0,
        "known_family_notes": 0,
        "common_phone_notes": 0,
        "leads_without_phone": 0,
        "leads_without_profile": 0,
        "note_drafts_skipped_existing": 0,
    }
    for lead in leads:
        lead_id = _safe_id(lead.get("id"))
        if not lead_id:
            continue
        action_key = f"family_note:{lead_id}"
        if journal.has_action(action_key):
            counts["note_drafts_skipped_existing"] += 1
            skipped.append(_skip_row("family_note", lead_id, "already_in_journal"))
            continue
        phones = _lead_phones(lead, contact_cache)
        if not phones:
            counts["leads_without_phone"] += 1
            skipped.append(_skip_row("family_note", lead_id, "no_phone"))
            continue
        profiles = _profiles_for_phones(profiles_db, phones)
        distinct_profiles = _dedupe_profiles(profiles)
        if not distinct_profiles:
            counts["leads_without_profile"] += 1
            skipped.append(_skip_row("family_note", lead_id, "phone_not_in_profile_db"))
            continue
        if len(distinct_profiles) > 1:
            note_text = common_phone_note_text(len(distinct_profiles))
            review_class = "common_phone"
            counts["common_phone_notes"] += 1
        else:
            note_text = known_family_note_text(distinct_profiles[0])
            review_class = "known_family"
            counts["known_family_notes"] += 1
        payload_hash = _sha256(note_text)
        journal.record(
            action_key=action_key,
            action_type="family_note",
            entity_id=lead_id,
            status="dry_run",
            payload_hash=payload_hash,
        )
        rows.append(
            {
                "action_key": action_key,
                "lead_id": lead_id,
                "review_class": review_class,
                "profile_match_count": str(len(distinct_profiles)),
                "note_text": note_text,
                "payload_sha256": payload_hash,
                "generated_at": generated_at.isoformat(timespec="seconds"),
                "live_write": "false",
            }
        )
    counts["family_note_drafts"] = len(rows)
    return {"rows": rows, "skipped_rows": skipped, "counts": counts}


def build_callback_task_drafts(
    *,
    callback_requests_path: Path | None,
    journal: "Step2Journal",
    generated_at: datetime,
) -> Mapping[str, Any]:
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    counts = {
        "callback_requests_seen": 0,
        "callback_task_drafts": 0,
        "callback_requests_malformed": 0,
        "callback_requests_without_intent": 0,
        "callback_task_drafts_skipped_existing": 0,
        "callback_requests_deduped": 0,
    }
    if callback_requests_path is None:
        return {"rows": rows, "skipped_rows": skipped, "counts": counts}

    candidates: dict[str, dict[str, Any]] = {}
    for line_no, payload in _read_jsonl(callback_requests_path):
        if payload is None:
            counts["callback_requests_malformed"] += 1
            skipped.append(_skip_row("callback_task", f"line:{line_no}", "malformed_json"))
            continue
        counts["callback_requests_seen"] += 1
        text = _safe_text(payload.get("text"))
        if not is_callback_request(text):
            counts["callback_requests_without_intent"] += 1
            skipped.append(_skip_row("callback_task", f"line:{line_no}", "no_callback_intent"))
            continue
        created_at = parse_datetime(payload.get("created_at")) or generated_at
        chat_id = _safe_text(payload.get("chat_id")) or f"unknown:{line_no}"
        action_key = callback_action_key(chat_id=chat_id, created_at=created_at)
        draft = callback_task_draft(payload, created_at=created_at, generated_at=generated_at, action_key=action_key)
        previous = candidates.get(action_key)
        if previous:
            counts["callback_requests_deduped"] += 1
            if int(draft["complete_till_ts"]) >= int(previous["complete_till_ts"]):
                candidates[action_key] = draft
        else:
            candidates[action_key] = draft

    for action_key, draft in sorted(candidates.items()):
        if journal.has_action(action_key):
            counts["callback_task_drafts_skipped_existing"] += 1
            skipped.append(_skip_row("callback_task", action_key, "already_in_journal"))
            continue
        payload_hash = _sha256(json.dumps(draft, ensure_ascii=False, sort_keys=True))
        draft["payload_sha256"] = payload_hash
        draft["live_write"] = "false"
        journal.record(
            action_key=action_key,
            action_type="callback_task",
            entity_id=_safe_text(draft.get("lead_id") or draft.get("contact_id") or action_key),
            status="dry_run",
            payload_hash=payload_hash,
        )
        rows.append({key: str(value) for key, value in draft.items()})
    counts["callback_task_drafts"] = len(rows)
    return {"rows": rows, "skipped_rows": skipped, "counts": counts}


def known_family_note_text(profile: ProfileRecord) -> str:
    parent = _latest_value(profile.active_fields, "parent_name") or _safe_text(profile.display_name)
    parent_text = f"семья {parent}" if parent else "известная семья"
    children = _children_note_text(profile.active_fields)
    return _clip(
        f"Телефон известен: {parent_text}. Дети: {children}. Уточните, о ком разговор.",
        900,
    )


def common_phone_note_text(profile_count: int) -> str:
    return (
        f"Общий телефон: найдено несколько семейных профилей ({profile_count}). "
        "Не подставляйте имя автоматически; откройте карточки и уточните, о ком разговор."
    )


def is_callback_request(text: str) -> bool:
    lowered = text.casefold()
    if not lowered.strip():
        return False
    positive = ("перезвон", "позвоните", "позвонить мне", "свяжитесь", "наберите")
    negative = ("я перезвоню", "я сам перезвон", "мы перезвоним", "менеджер перезвонит")
    return any(item in lowered for item in positive) and not any(item in lowered for item in negative)


def callback_action_key(*, chat_id: str, created_at: datetime) -> str:
    created_at = _require_aware(created_at, "created_at")
    return f"callback_task:{chat_id}:{created_at.date().isoformat()}:{CALLBACK_INTENT}"


def callback_task_draft(
    payload: Mapping[str, Any],
    *,
    created_at: datetime,
    generated_at: datetime,
    action_key: str,
) -> dict[str, str]:
    text = _safe_text(payload.get("text"))
    brand = _safe_text(payload.get("brand")) or "unknown"
    quote = _clip(text.replace("\n", " "), 180)
    summary = _safe_text(payload.get("summary")) or _one_line_intent(text)
    complete_till = infer_callback_deadline(text, created_at)
    return {
        "action_key": action_key,
        "task_type": "call",
        "lead_id": _safe_id(payload.get("lead_id")),
        "contact_id": _safe_id(payload.get("contact_id")),
        "chat_id": _safe_text(payload.get("chat_id")),
        "message_id": _safe_text(payload.get("message_id")),
        "brand": brand,
        "text": _clip(
            f'[БОТ] Клиент просит перезвонить. Бренд: {brand}. Суть: {summary}. Фраза: "{quote}". Канал: {CALLBACK_CHANNEL}',
            900,
        ),
        "complete_till_iso": complete_till.isoformat(timespec="seconds"),
        "complete_till_ts": str(int(complete_till.timestamp())),
        "generated_at": generated_at.isoformat(timespec="seconds"),
    }


def infer_callback_deadline(text: str, created_at: datetime) -> datetime:
    created_at = _require_aware(created_at, "created_at")
    lowered = text.casefold()
    if "завтра" in lowered:
        return _at_local_time(created_at + timedelta(days=1), time(10, 0))
    after_hour = re.search(r"после\s+(\d{1,2})(?:[:.]\d{2})?\s*(?:час|ч)?", lowered)
    if after_hour:
        hour = min(23, int(after_hour.group(1)) + 1)
        candidate = _at_local_time(created_at, time(hour, 0))
        if candidate <= created_at:
            candidate = candidate + timedelta(days=1)
        return candidate
    if "вечер" in lowered:
        candidate = _at_local_time(created_at, time(18, 0))
        return candidate if candidate > created_at else candidate + timedelta(days=1)
    if any(token in lowered for token in ("сейчас", "поскорее", "срочно")):
        return next_business_time(created_at + timedelta(hours=1))
    if created_at.hour < 9 or created_at.hour >= 21:
        return _at_local_time(created_at + timedelta(days=1), time(9, 30))
    return next_business_time(created_at + timedelta(hours=2))


def next_business_time(value: datetime) -> datetime:
    value = _require_aware(value, "value")
    if value.hour < 9:
        return _at_local_time(value, time(9, 30))
    if value.hour >= 21:
        return _at_local_time(value + timedelta(days=1), time(9, 30))
    return value


class Step2Journal:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._con = sqlite3.connect(path)
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS action_journal (
              action_key TEXT PRIMARY KEY,
              action_type TEXT NOT NULL,
              entity_id TEXT NOT NULL,
              status TEXT NOT NULL,
              payload_hash TEXT NOT NULL DEFAULT '',
              recorded_at TEXT NOT NULL
            )
            """
        )
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    def has_action(self, action_key: str) -> bool:
        row = self._con.execute("SELECT 1 FROM action_journal WHERE action_key = ?", (action_key,)).fetchone()
        return row is not None

    def record(
        self,
        *,
        action_key: str,
        action_type: str,
        entity_id: str,
        status: str,
        payload_hash: str = "",
    ) -> None:
        self._con.execute(
            """
            INSERT OR IGNORE INTO action_journal (
              action_key, action_type, entity_id, status, payload_hash, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                action_key,
                action_type,
                entity_id,
                status,
                payload_hash,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._con.commit()


def write_step2_outputs(
    *,
    out_root: Path,
    note_rows: Sequence[Mapping[str, str]],
    task_rows: Sequence[Mapping[str, str]],
    skipped_rows: Sequence[Mapping[str, str]],
) -> Mapping[str, Path]:
    outputs = {
        "note_drafts_csv": out_root / "family_note_drafts.csv",
        "callback_task_drafts_csv": out_root / "callback_task_drafts.csv",
        "skipped_csv": out_root / "skipped.csv",
        "summary_json": out_root / "summary.json",
    }
    _write_csv(outputs["note_drafts_csv"], note_rows)
    _write_csv(outputs["callback_task_drafts_csv"], task_rows)
    _write_csv(outputs["skipped_csv"], skipped_rows)
    return outputs


def _lead_phones(lead: Mapping[str, Any], contact_cache: Mapping[str, Mapping[str, Any]]) -> list[str]:
    phones: list[str] = []
    for ref in embedded_items(lead, "contacts"):
        contact_id = _safe_id(ref.get("id"))
        contact = contact_cache.get(contact_id)
        if not contact:
            continue
        for phone in extract_contact_phones(contact):
            normalized = normalize_phone(phone)
            if normalized and normalized not in phones:
                phones.append(normalized)
    return phones


def _profiles_for_phones(profiles_db: Path, phones: Sequence[str]) -> list[ProfileRecord]:
    profiles: list[ProfileRecord] = []
    for phone in phones:
        profiles.extend(load_profiles_for_summary(profiles_db, phone=phone))
    return profiles


def _dedupe_profiles(profiles: Sequence[ProfileRecord]) -> list[ProfileRecord]:
    result: dict[str, ProfileRecord] = {}
    for profile in profiles:
        result.setdefault(profile.profile_id, profile)
    return [result[key] for key in sorted(result)]


def _children_note_text(fields: Sequence[Mapping[str, Any]]) -> str:
    child_keys = sorted(
        {
            str(row.get("child_key") or "child_1")
            for row in fields
            if str(row.get("field") or "") in {"child_name", "grade", "subject"}
        }
    )
    children: list[str] = []
    for child_key in child_keys:
        name = _latest_value(fields, "child_name", child_key=child_key) or "ребенок"
        grade = _latest_value(fields, "grade", child_key=child_key)
        subject = _latest_value(fields, "subject", child_key=child_key)
        brand = _latest_brand(fields, child_key=child_key)
        parts = [name]
        if grade:
            parts.append(f"{grade} кл")
        if subject:
            parts.append(subject)
        if brand:
            parts.append(brand)
        children.append(" ".join(parts))
    return "; ".join(children[:4]) if children else "нет активных данных"


def _latest_value(fields: Sequence[Mapping[str, Any]], field: str, *, child_key: str | None = None) -> str:
    rows = [
        row
        for row in fields
        if str(row.get("field") or "") == field and (child_key is None or str(row.get("child_key") or "child_1") == child_key)
    ]
    rows.sort(key=lambda row: str(row.get("event_at") or ""), reverse=True)
    return _safe_text(rows[0].get("value")) if rows else ""


def _latest_brand(fields: Sequence[Mapping[str, Any]], *, child_key: str) -> str:
    rows = [row for row in fields if str(row.get("child_key") or "child_1") == child_key]
    rows.sort(key=lambda row: str(row.get("event_at") or ""), reverse=True)
    for row in rows:
        brand = _safe_text(row.get("brand")).casefold()
        if brand == "foton":
            return "[Фотон]"
        if brand == "unpk":
            return "[УНПК]"
    return "[—]" if rows else ""


def _first_embedded(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    embedded = payload.get("_embedded")
    if not isinstance(embedded, Mapping):
        return None
    rows = embedded.get(key)
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    return None


def _read_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any] | None]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None
                continue
            yield line_no, payload if isinstance(payload, Mapping) else None


def parse_datetime(value: Any) -> datetime | None:
    text = _safe_text(value)
    if not text:
        return None
    if text.isdigit():
        return epoch_to_dt(text)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _at_local_time(value: datetime, target: time) -> datetime:
    return value.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)


def _one_line_intent(text: str) -> str:
    cleaned = _clip(text.replace("\n", " "), 100)
    return cleaned or "просьба перезвонить"


def _resolve_out_root(project_root: Path, out_root: Path) -> Path:
    out_root = out_root.expanduser()
    if not out_root.is_absolute():
        out_root = project_root / out_root
    return out_root.resolve(strict=False)


def _skip_row(action_type: str, entity_id: str, reason: str) -> dict[str, str]:
    return {"action_type": action_type, "entity_id": entity_id, "reason": reason}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clip(value: str, max_chars: int) -> str:
    text = _safe_text(value)
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_id(value: Any) -> str:
    return _safe_text(value)
