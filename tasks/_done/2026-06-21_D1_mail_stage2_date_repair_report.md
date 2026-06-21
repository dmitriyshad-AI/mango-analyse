# D1 mail_stage2 date repair — 2026-06-21

## Что исправлено

В боевой `customer_timeline` после полного ingest было 788 событий `mail_archive_stage2` с `event_at=1970-01-01`. Это ломало чистую хронологию писем.

Исправлено:

- parser `mail_stage2_ingest._parse_event_at` теперь учитывает `date_last`;
- добавлен воспроизводимый скрипт `scripts/repair_mail_stage2_event_dates.py`;
- в боевой БД обновлены 788 событий, связанные `bot_context_chunks.event_at` и `customer_opportunities.opened_at`;
- в `record_json` добавлена метка `mail_stage2_date_repair` с источником даты.

## Боевая БД

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite
```

## Backup

Перед update создан backup:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/backups/before_mail_stage2_date_repair_20260621T002242Z/backup_manifest.json
```

Backup sha256:

```text
09fbc56687204038437858b48324cd6415861efce3463628fe06655ca3795a69
```

## Числа

Dry-run до apply:

| source | updates |
|---|---:|
| `stage2_date_last` | 421 |
| `archive_first_ingested_at_no_message_date` | 367 |
| total | 788 |

Пояснение: 367 событий — в основном Drafts без почтового `Date` header и без `message_date_iso`; для них использован архивный `first_ingested_at`, а в `record_json` явно записан источник `archive_first_ingested_at_no_message_date`.

Проверка после apply:

| check | value |
|---|---:|
| `remaining_1970_mail_stage2_events` | 0 |
| `date_repair_marked_events` | 788 |
| `unsafe_mail_stage2_bot_context_chunks` | 0 |

Runtime-отчёт:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/mail_stage2_date_repair_report.json
```

## Тесты

```text
tests/test_customer_timeline_mail_stage2_ingest.py: 5 passed
full pytest: 3480 passed, 5 skipped, 1 warning
```

## Смысловая проверка

Verdict: PASS_WITH_NOTES.

Что стало лучше: 1970-событий в mail_stage2 больше нет; письма больше не всплывают в начале ленты как артефакт эпохи Unix.

Остаточный нюанс: 367 no-date Drafts получили не исходную дату письма, а дату попадания в архив. Это честно помечено в `record_json`; более точной даты в доступных source-of-truth не найдено.
