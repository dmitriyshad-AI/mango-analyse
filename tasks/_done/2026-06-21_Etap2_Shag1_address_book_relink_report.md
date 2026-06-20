# Этап 2, Шаг 1: адресная книга клиента + дозапривязка писем

Дата отчёта: 2026-06-20  
Ветка: `codex/etap2-step1-address-book`  
Worktree: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf`

## Read-only карта

- Источник identity: `/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite`.
- Таблицы: `identity_candidates` = 18126, `identity_values` = 36765, `identity_links` = 39392.
- Дубли из `identity_values.match_class='duplicate'`: email = 1171, phone = 1276.
- Дубли `tallanto_id`: 109 ID, затронута 221 строка. Поэтому новый слой агрегирует по `tallanto_id`, но блокирует авто-привязку для duplicate `tallanto_id`.
- Текущий handoff писем: `/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/customer_history_handoff_full_all_mail/mail_customer_history_handoff.sqlite`.
- Текущий ключ старого pipeline: `candidate_key`; новый preview добавляет канон `tallanto_id`, не мутируя старые таблицы.

## Что реализовано

- Новый безопасный preview-слой `build_mail_customer_relink_preview`.
- Новый CLI:

```bash
PYTHONPATH=src python3 scripts/mango_office_mail_archive.py customer-relink-preview \
  --mail-handoff-db <mail_customer_history_handoff.sqlite> \
  --identity-db <tallanto_email_identity_map.sqlite> \
  --classification-index <classification_index.csv> \
  --out-dir <_external_handoffs/.../customer_relink_preview>
```

- Выходной артефакт: отдельная БД `mail_customer_relink_preview.sqlite`, старые `mail_customer_history_handoff.sqlite` и `mail_mango_bridge_preview.sqlite` не меняются.
- Live Tallanto lookup поддержан только явным флагом `--live-tallanto-lookup`; в текущем прогоне не включался.

## Реальный preview

Артефакт:

`/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-21/regru_edu/customer_relink_preview_full_all_mail/mail_customer_relink_preview_report.json`

Было:

- `ready`: 29711
- `manual_review`: 9644
- `linkable_messages`: 39355
- `ready_share`: 75.49%

Стало в offline-preview:

- новых связок: 62
- `ready`: 29773
- остаток `manual_review`: 9582
- `ready_share`: 75.65%
- добавлено в книгу learned values: 54

Остаток unmatched:

- `classification_outbound_campaign`: 3578
- `classification_bulk_newsletter`: 2737
- `identity_value_missing`: 2660
- `no_phone_signal`: 310
- `classification_internal`: 198
- `classification_service_notification`: 54
- `classification_spam_or_empty`: 26
- `duplicate_identity_value`: 17
- `classification_bounce`: 1
- `text_missing`: 1

## NEG / предохранители

- Общий email/телефон (`identity_values.match_class='duplicate'`) не склеивается.
- Семейный общий контакт не разрешается через ФИО: ФИО не используется как самостоятельный доказательный сигнал.
- Cross-brand общий телефон/почта блокируется как `brand_conflict`, если бренд есть в candidate data.
- Duplicate `tallanto_id` блокируется для авто-привязки.
- Live Tallanto: 0/2+ контактов, ошибка или brand mismatch не дают link.
- Повторный запуск не увеличивает число learned rows.
- Tallanto/AMO/timeline/YAML не пишутся.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_mail_archive.py`  
  Результат: `54 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_canonical_readonly_import.py tests/test_customer_timeline_canonical_readonly_triage.py`  
  Результат: `17 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`  
  Результат: `3435 passed, 5 skipped, 1 warning`

## Остаточные риски

- В текущей identity DB нет бренда для большинства реальных строк; код умеет использовать `Филиал` в новых identity maps, но старый snapshot часто остаётся с `brand_scope=[]`.
- Offline-прирост меньше потолка 214, потому что рассылки/кампании и не-real correspondence заблокированы по `classification_index`.
- Live Tallanto read-only не запускался в этом прогоне: нужен отдельный запуск с подтверждённым доступом/лимитами, если архитектор хочет добрать контакты после 12.05.
