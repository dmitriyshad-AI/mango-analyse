# TZ139 Work C — DerivedSignal lifecycle

Дата: 2026-06-18.
Ветка: `codex/tz139-customer-timeline`.
База до Work C: `ab4cf72`.
Коммиты Work C:

- `75cb27e` — C0: lifecycle fields and SQLite migration.
- `f15e86c` — C1: deterministic signal derivation rules.
- `73a5b79` — C2: recompute/upsert lifecycle, read filters, CLI.

## Что сделано

### C0

- `src/mango_mvp/customer_timeline/contracts.py:135` добавлен `SignalStatus(active/resolved/stale)`.
- `src/mango_mvp/customer_timeline/contracts.py:572` в `DerivedSignal` добавлены `status` и `expires_at`.
- `src/mango_mvp/customer_timeline/store.py:33` migration id поднят до `20260618_002_derived_signal_status`.
- `src/mango_mvp/customer_timeline/store.py:478` таблица `derived_signals` создаётся с `status/expires_at`.
- `src/mango_mvp/customer_timeline/store.py:635` добавлена идемпотентная миграция старых DB через `PRAGMA table_info` + `ALTER TABLE` только если колонок нет.
- `src/mango_mvp/customer_timeline/store.py:647` добавлен индекс `(tenant_id, customer_id, status, expires_at)`.

Критичный инвариант: `src/mango_mvp/customer_timeline/ids.py:202` `stable_signal_id` не менялся; `status/expires_at` не входят в `signal_id`.

### C1

- `src/mango_mvp/customer_timeline/derived_signals.py:96` добавлен deterministic `derive_active_signals`.
- `paid_no_access`: Tallanto payment без доступа; доступ определяется через активный `tallanto_abonement.visits_left > 0` или matching `most_class`/`tallanto_group`.
- `hot_lead_silent_7d`: входящий/исходящий touch с интересом, затем нет нового touch по любому каналу `N=7` дней.
- `duplicate_contact`: открытые `shared_amo_contact/shared_amo_lead/ambiguous_identity` conflicts.
- Все сигналы имеют `severity`, `recommended_action`, `requires_manager_review`, `expires_at` от source `event_at/created_at`.

### C2

- `src/mango_mvp/customer_timeline/derived_signals.py:107` добавлен `recompute_customer_signals`.
- Пересчёт reread events/conflicts из timeline DB, сравнивает с текущими managed signals, upsert-ит активные и auto-closes исчезнувшие predicates в `resolved`/`stale`.
- Read-modify-preserve: при обновлении существующего signal сохраняется `signal_id` и `created_at`.
- `src/mango_mvp/customer_timeline/store.py:1394` добавлен `list_signals_by_customer` с active-filter.
- `src/mango_mvp/customer_timeline/store.py:1432` добавлен `list_conflicts_by_customer`.
- `src/mango_mvp/customer_timeline/read_api.py:184` карточка читает только active/non-expired signals.
- `src/mango_mvp/customer_timeline/store.py:1885` event children signals тоже фильтруются active/non-expired.
- `src/mango_mvp/customer_timeline/store.py:2177` search по signals фильтрует active/non-expired.
- `scripts/derive_customer_timeline_signals.py:57` добавлен CLI: dry-run по умолчанию, `--apply` пишет только в timeline SQLite.

## Probe числа

Локальный `/tmp` probe на 3 искусственных клиентах:

```json
{
  "dry_run_summary": {
    "signals_total": 3,
    "status_counts": {"active": 3},
    "signal_type_counts": {
      "duplicate_contact": 1,
      "hot_lead_silent_7d": 1,
      "paid_no_access": 1
    },
    "write_status_counts": {}
  },
  "first_apply_summary": {
    "signals_total": 3,
    "status_counts": {"active": 3},
    "write_status_counts": {"created": 3}
  },
  "resolve_apply_summary": {
    "signals_total": 3,
    "status_counts": {"resolved": 3},
    "write_status_counts": {"updated": 3}
  },
  "repeat_apply_summary": {
    "signals_total": 3,
    "status_counts": {"resolved": 3},
    "write_status_counts": {"duplicate": 3}
  }
}
```

Вывод probe: повторный derive не плодит дубли; авто-закрытие сработало для всех трёх типов; повтор после auto-close стал duplicate.

## STOP

Work C завершён и остановлен на регрейд Claude #1. Work D/E/F не начинались.
