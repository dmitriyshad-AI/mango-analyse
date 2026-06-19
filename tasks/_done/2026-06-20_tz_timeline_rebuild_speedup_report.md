# Отчёт: малое ускорение пересборки Customer Timeline

Дата: 2026-06-20  
Ветка: `codex/tz139-customer-timeline-integrate`  
ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-20_TZ_uskorenie_peresborki_timeline.md`

## Что сделано

- FTS-синхронизация на время канонического bulk-импорта отключается через `defer_fts_sync=True`.
- После записи всех строк FTS строится одним детерминированным проходом для:
  - `timeline_event_fts`;
  - `bot_context_chunk_fts`.
- После rebuild выполняется FTS `integrity-check`; в отчёт сборки пишутся FTS-счётчики.
- Добавлен кэш нормализованных источников:
  - ключ = `source_name + sha256(источника) + normalizer_version`;
  - кэшируется нормализованный batch источника до фильтрации по `known_phones`;
  - смена `normalizer_version` инвалидирует все источники;
  - изменение одного файла инвалидирует только соответствующий источник.
- В `import_report.json` и `coverage_report.json` добавлен блок `performance`:
  - `parse_time_seconds`;
  - `normalize_time_seconds`;
  - `write_time_seconds`;
  - `fts_time_seconds`;
  - `total_time_seconds`;
  - `source_cache`;
  - `fts_counts`.

## Контрольный замер

Команда: `scripts/build_canonical_readonly_customer_timeline.py` на тех же read-only источниках, `--max-call-events-per-contact 1`, `generated_at=2026-06-20T00:00:00+00:00`.

Артефакты:

- До: `product_data/customer_timeline/perf_before_20260620_tz_timeline_speed/`
- После, холодный кэш: `product_data/customer_timeline/perf_after_cold_20260620_tz_timeline_speed/`
- После, тёплый кэш: `product_data/customer_timeline/perf_after_warm_20260620_tz_timeline_speed/`
- Кэш: `product_data/customer_timeline/perf_source_cache_20260620_tz_timeline_speed/`

Итог по времени:

| Прогон | real | parse | normalize | write | fts | total из отчёта |
|---|---:|---:|---:|---:|---:|---:|
| До | 422.76s | нет разбивки | нет разбивки | нет разбивки | нет разбивки | нет разбивки |
| После, холодный кэш | 144.76s | 15.07s | 5.32s | 107.34s | 12.40s | 143.33s |
| После, тёплый кэш | 124.78s | 3.81s | 4.97s | 96.87s | 14.54s | 123.31s |

Кэш в тёплом прогоне: `master_contacts`, `master_calls`, `amo_contacts`, `amo_deals`, `mail_bridge` = `hit`.

## Идентичность данных

Табличные счётчики совпали в `before`, `after_cold`, `after_warm`:

- `customer_identities`: 16901
- `identity_links`: 59594
- `customer_opportunities`: 4011
- `timeline_events`: 71148
- `bot_context_chunks`: 30128
- `timeline_conflicts`: 601
- `customer_id_mappings`: 18164
- `derived_signals`: 0
- `event_artifacts`: 0
- `ingestion_runs`: 1
- `audit_log`: 200940
- `timeline_event_fts`: 71148
- `bot_context_chunk_fts`: 30128

Контрольные FTS-выборки совпали для запросов: `стоимость`, `математика`, `расписание`.

## Тесты

- Точечные NEG: `4 passed`.
- Customer Timeline suite: `192 passed`.
- Полный pytest: `3365 passed, 5 skipped, 1 warning in 50.84s`.

## Остаточный риск

Оптимизация не трогает single-writer и не распараллеливает запись. После FTS/cache основной остаток времени остаётся в `write_time`; это ожидаемая зона Этапа 2.

Live-write, AMO, Tallanto, CRM, ASR, Resolve+Analyze не запускались.
