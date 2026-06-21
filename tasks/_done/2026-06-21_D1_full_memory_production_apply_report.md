# D1 full memory production apply — 2026-06-21

## Контекст

Задача: выполнить реальный apply полной памяти клиента по ТЗ `2026-06-21_TZ_D1_boevoy_ingest_polnoy_pamyati.md` после явного разрешения Дмитрия.

Кодовая ревизия: `main` с фиксом relink-JOIN писем `d7b3950`; перед apply добавлен безопасный CLI-режим `apply-production` коммитом `63a1b0b`.

Live-write во внешние системы: не выполнялся. AMO/Tallanto/CRM не писались.

## Боевая БД

Путь:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite
```

Размер после apply: около 2.3 GB.

Отчёт скрипта:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/full_memory_ingest_production_apply_report.json
```

## Backup до первого импортёра

Backup создан до canonical/importer apply:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/backups/before_full_memory_production_apply_20260621T000245Z/backup_manifest.json
```

Backup sha256:

```text
6ded6b21307974d96bddce6d0a930bc0aef0c0f93556595c4668c2b26815b631
```

Restore не выполнялся: данные оставлены в боевой БД.

## Каналы

События по `source_system`:

| source_system | events |
|---|---:|
| `mango_processed_summary` | 71 962 |
| `mail_archive_stage2` | 30 093 |
| `master_contacts_snapshot` | 16 901 |
| `tallanto_snapshot` | 16 901 |
| `amocrm_snapshot` | 16 277 |
| `mail_archive` | 4 168 |

Итого `timeline_events`: 156 302.

Прочие объекты после первого apply:

| table | rows |
|---|---:|
| `bot_context_chunks` | 107 586 |
| `customer_identities` | 17 851 |
| `customer_opportunities` | 21 650 |
| `identity_links` | 82 085 |
| `customer_id_mappings` | 18 164 |
| `timeline_conflicts` | 601 |

## Письма

mail_stage2 использовал fresh relink decisions через JOIN по `message_sha256`, не старый `customer_id` из JSONL.

| metric | count |
|---|---:|
| input/planned events | 30 093 |
| linked | 22 491 |
| pending/unmatched | 7 602 |
| created mail chunks | 22 397 |
| relink decisions loaded | 30 093 |
| interim customer ids | 0 |
| unmatched with customer_id | 0 |

Доля привязки: 22 491 / 30 093 = 74.7%.

## Идемпотентность

Повторный полный прогон сразу после apply:

```json
{
  "timeline_events": 0,
  "bot_context_chunks": 0,
  "customer_identities": 0,
  "customer_opportunities": 0,
  "identity_links": 0,
  "customer_id_mappings": 0,
  "timeline_conflicts": 0,
  "ingestion_runs": 0
}
```

`repeat_added_events`: 0.

## Safety-инварианты

```json
{
  "unsafe_bot_context_chunks": 0,
  "mail_stage2_pending_attribution_events": 7602,
  "mail_stage2_unmatched_with_customer": 0,
  "mail_stage2_interim_customer_ids": 0,
  "duplicate_event_dedupe_keys": 0,
  "pass": true
}
```

`allowed_for_bot` для каналовых chunks не открыт.

## Ленты клиентов

Read API samples сохранены в `full_memory_ingest_production_apply_report.json`.

Проверенные примеры:

1. `customer:2a659050096d861b285fc2d808fe1d37` — 12 событий в sample, Mango calls, 343 manager-review chunks.
2. `customer:23ad22e65931a1553b943fc1645b0aac` — 12 событий в sample, Mango calls, 328 manager-review chunks.
3. `customer:547fdefd9b1fd4a5ef60ee0943e47068` — mail_stage2 письма по расписанию/очной школе, 204 manager-review chunks.
4. `customer:0c7ebd6c18aab4a139a7128e6935149c` — mail_stage2 + Mango calls, 181 manager-review chunks.
5. `customer:008978b883ab0af2340cc4868d3f6732` — mail_stage2 + Mango calls, 164 manager-review chunks.

Все samples имеют `safe_for_automatic_bot=false`, потому что chunks manager-review only.

## Остаточный риск данных

У `mail_archive_stage2` найдено 788 / 30 093 писем с `event_at <= 1970-01-02`. Это означает, что часть дат писем не распарсилась и попала в начало хронологии. На привязку и safety это не влияет, но для идеальной временной ленты писем нужен отдельный фикс парсинга дат stage2.

## Проверки

Перед live apply:

```text
26 passed
3479 passed, 5 skipped, 1 warning
```

После live apply:

- Боевая БД существует.
- Backup manifest существует.
- `production_apply_performed=true`.
- `restore_performed=false`.
- Прямой SQLite-контроль подтвердил counts и safety.

## Операционные примечания

Было два безопасных ранних стопа до настоящего apply:

1. Не найден stage2 JSONL в отдельном worktree `Mango_main_merge`.
2. Canonical source paths требовали `project_root=/Users/dmitrijfabarisov/Projects/Mango analyse`.

Настоящий production apply выполнен только в основной путь `Mango analyse/product_data/...`, указанный выше.
