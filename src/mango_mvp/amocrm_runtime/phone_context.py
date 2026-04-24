from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.utils.phone import normalize_phone

settings = get_settings()

CONTACTS_CSV_NAME = 'master_contacts_ru.csv'
CALLS_CSV_NAME = 'master_calls_ru.csv'
CANONICAL_EXPORT_POINTER_NAME = 'CANONICAL_EXPORT.txt'


@dataclass(frozen=True)
class PhoneContext:
    phone: str
    source_dir: str
    contact_row: Optional[dict[str, str]]
    call_rows: list[dict[str, str]]
    call_ids: list[str]
    first_call_at: Optional[str]
    last_call_at: Optional[str]
    manager_history: list[str]
    interest_summary: str
    objections_summary: str
    current_sales_temperature: str
    recommended_next_step: str
    follow_up_due_at: Optional[str]
    history_summary: str
    chronology: str
    tallanto_id: str
    tallanto_match_status: str


_CACHE: dict[str, Any] = {
    'contacts_path': None,
    'contacts_mtime_ns': None,
    'calls_path': None,
    'calls_mtime_ns': None,
    'contacts_by_phone': {},
    'calls_by_phone': {},
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    return text


def _parse_dt(value: str) -> tuple[int, str]:
    candidate = _safe_text(value)
    if not candidate:
        return (0, '')
    normalized = candidate.replace('T', ' ')
    for fmt in (
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%d.%m.%Y %H:%M',
        '%d.%m.%Y',
    ):
        try:
            parsed = datetime.strptime(normalized, fmt)
            return (int(parsed.timestamp()), candidate)
        except ValueError:
            continue
    return (0, candidate)


def _latest_export_dir() -> Path:
    stable_runtime = Path(settings.source_workspace_root) / 'stable_runtime'
    pointer_path = stable_runtime / CANONICAL_EXPORT_POINTER_NAME
    if pointer_path.exists():
        raw_target = _safe_text(pointer_path.read_text(encoding='utf-8'))
        if raw_target:
            candidate = Path(raw_target)
            if not candidate.is_absolute():
                candidate = stable_runtime / raw_target
            candidate = candidate.resolve()
            if candidate.is_dir() and (candidate / CONTACTS_CSV_NAME).exists() and (candidate / CALLS_CSV_NAME).exists():
                return candidate
    candidates: list[Path] = []
    for path in stable_runtime.glob('sales_master_export_*review_accepted'):
        if not path.is_dir():
            continue
        if (path / CONTACTS_CSV_NAME).exists() and (path / CALLS_CSV_NAME).exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f'No sales export directory with {CONTACTS_CSV_NAME} and {CALLS_CSV_NAME} found under {stable_runtime}.'
        )
    candidates.sort(key=lambda item: (item.stat().st_mtime_ns, item.name), reverse=True)
    return candidates[0]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _reload_cache_if_needed() -> None:
    export_dir = _latest_export_dir()
    contacts_path = export_dir / CONTACTS_CSV_NAME
    calls_path = export_dir / CALLS_CSV_NAME
    contacts_mtime_ns = contacts_path.stat().st_mtime_ns
    calls_mtime_ns = calls_path.stat().st_mtime_ns
    if (
        _CACHE['contacts_path'] == str(contacts_path)
        and _CACHE['calls_path'] == str(calls_path)
        and _CACHE['contacts_mtime_ns'] == contacts_mtime_ns
        and _CACHE['calls_mtime_ns'] == calls_mtime_ns
    ):
        return

    contacts_by_phone: dict[str, dict[str, str]] = {}
    for row in _load_csv_rows(contacts_path):
        phone = normalize_phone(row.get('Телефон клиента'))
        if not phone:
            continue
        contacts_by_phone[phone] = row

    calls_by_phone: dict[str, list[dict[str, str]]] = {}
    for row in _load_csv_rows(calls_path):
        phone = normalize_phone(row.get('Телефон клиента'))
        if not phone:
            continue
        calls_by_phone.setdefault(phone, []).append(row)

    for phone, rows in calls_by_phone.items():
        rows.sort(key=lambda item: _parse_dt(item.get('Дата и время звонка', ''))[0], reverse=True)

    _CACHE.update(
        {
            'contacts_path': str(contacts_path),
            'contacts_mtime_ns': contacts_mtime_ns,
            'calls_path': str(calls_path),
            'calls_mtime_ns': calls_mtime_ns,
            'contacts_by_phone': contacts_by_phone,
            'calls_by_phone': calls_by_phone,
        }
    )


def _unique_non_empty(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _summarize_interests(call_rows: list[dict[str, str]], contact_row: Optional[dict[str, str]]) -> str:
    candidates = []
    if contact_row is not None:
        candidates.extend(_safe_text(contact_row.get('Продукты интереса')).split(' | '))
        candidates.append(_safe_text(contact_row.get('Рекомендуемый продукт')))
    for row in call_rows[:12]:
        candidates.extend(_safe_text(row.get('Продукты интереса')).split(' | '))
        candidates.extend(_safe_text(row.get('Предметы интереса')).split(' | '))
        candidates.append(_safe_text(row.get('Рекомендуемый продукт')))
    return ' | '.join(_unique_non_empty(candidates))


def _summarize_objections(call_rows: list[dict[str, str]], contact_row: Optional[dict[str, str]]) -> str:
    candidates = []
    if contact_row is not None:
        candidates.extend(_safe_text(contact_row.get('Возражения')).split(' | '))
    for row in call_rows[:12]:
        candidates.extend(_safe_text(row.get('Возражения')).split(' | '))
        candidates.extend(_safe_text(row.get('Коммерческие ограничения')).split(' | '))
    return ' | '.join(_unique_non_empty(candidates))


def get_phone_context(phone: str) -> Optional[PhoneContext]:
    normalized_phone = normalize_phone(phone)
    if not normalized_phone:
        return None

    _reload_cache_if_needed()
    contact_row = _CACHE['contacts_by_phone'].get(normalized_phone)
    call_rows = list(_CACHE['calls_by_phone'].get(normalized_phone, []))
    if contact_row is None and not call_rows:
        return None

    call_rows_sorted_oldest = sorted(call_rows, key=lambda item: _parse_dt(item.get('Дата и время звонка', ''))[0])
    first_call_at = _safe_text(call_rows_sorted_oldest[0].get('Дата и время звонка')) if call_rows_sorted_oldest else _safe_text(contact_row.get('Первый звонок') if contact_row else '') or None
    last_call_at = _safe_text(call_rows[0].get('Дата и время звонка')) if call_rows else _safe_text(contact_row.get('Последний звонок') if contact_row else '') or None
    manager_history = _unique_non_empty([row.get('Менеджер', '') for row in call_rows_sorted_oldest])
    call_ids = [row.get('ID звонка', '') for row in call_rows_sorted_oldest if _safe_text(row.get('ID звонка'))]

    export_dir = Path(_CACHE['contacts_path']).parent if _CACHE['contacts_path'] else _latest_export_dir()
    return PhoneContext(
        phone=normalized_phone,
        source_dir=str(export_dir),
        contact_row=contact_row,
        call_rows=call_rows,
        call_ids=call_ids,
        first_call_at=first_call_at,
        last_call_at=last_call_at,
        manager_history=manager_history,
        interest_summary=_summarize_interests(call_rows, contact_row),
        objections_summary=_summarize_objections(call_rows, contact_row),
        current_sales_temperature=_safe_text(contact_row.get('Приоритет лида') if contact_row else '') or _safe_text(call_rows[0].get('Приоритет лида') if call_rows else ''),
        recommended_next_step=_safe_text(contact_row.get('Следующий шаг') if contact_row else '') or _safe_text(call_rows[0].get('Следующий шаг') if call_rows else ''),
        follow_up_due_at=_safe_text(contact_row.get('Рекомендуемая дата следующего контакта') if contact_row else '') or _safe_text(call_rows[0].get('Рекомендуемая дата следующего контакта') if call_rows else '') or None,
        history_summary=_safe_text(contact_row.get('Краткая история общения') if contact_row else '') or _safe_text(call_rows[0].get('Краткое резюме разговора') if call_rows else ''),
        chronology=_safe_text(contact_row.get('Хронология общения (последние 5 касаний)') if contact_row else ''),
        tallanto_id=_safe_text(contact_row.get('ID Tallanto') if contact_row else ''),
        tallanto_match_status=_safe_text(contact_row.get('Статус матчинга Tallanto') if contact_row else ''),
    )


def get_all_known_phones() -> list[str]:
    _reload_cache_if_needed()
    phones = set(_CACHE['contacts_by_phone']) | set(_CACHE['calls_by_phone'])
    return sorted(phones)
