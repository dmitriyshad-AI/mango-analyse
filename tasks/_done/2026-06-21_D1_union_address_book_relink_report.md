# D1 union address book relink — report

Дата: 2026-06-20.
Ветка: `codex/etap2-step1-address-book`.
ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_Etap2_obedinennaya_kniga_relink.md`.

## Что сделано

- Добавлена отдельная сборка union identity map: `identity-map-union`.
- Union строится до классификации значений: old students TSV + fresh contacts CSV -> один row-set -> дедуп по `tallanto_id` -> пересчёт `identity_values`.
- Один и тот же `tallanto_id` из old+fresh становится одной карточкой `candidate_key=tallanto:<id>`.
- Дубли `tallanto_id` внутри исходной выгрузки остаются отдельными candidate и блокируют авто-привязку как раньше.
- ФИО-конфликт под одним ID помечается `id_identity_conflict` в `candidate_json` и агрегируется в отчёте.
- Tallanto/AMO/timeline не писались; все артефакты в `_external_handoffs/`.

## Артефакты

- Union sqlite:
  `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/tallanto_identity_map_union_20260620/tallanto_identity_map_union_20260620.sqlite`
- Union report:
  `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/tallanto_identity_map_union_20260620/tallanto_identity_map_union_report.json`
- Corpus preview:
  `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-20/regru_edu/stage2_customer_relink_union_20260620/corpus_27009/mail_stage2_customer_relink_preview_report.json`
- Delta preview:
  `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-20/regru_edu/stage2_customer_relink_union_20260620/delta_3084/mail_stage2_customer_relink_preview_report.json`

## Входы

| Источник | Строк |
|---|---:|
| old students 2026-05-12 TSV | 18 126 |
| fresh Contacts 20.06.2026 CSV | 11 860 |

Fresh CSV прочитан csv-aware через сконвертированный CSV: 11 860 записей, не физические строки OLE/XLS.

## Union identity map

| Метрика | Значение |
|---|---:|
| Входных строк всего | 29 986 |
| Карточек после дедупа по `tallanto_id` | 18 931 |
| Схлопнуто по `tallanto_id` | 11 055 |
| Дублирующихся `tallanto_id` после union | 110 |
| `id_identity_conflict` | 83 |
| Email values strong_unique / duplicate | 17 435 / 1 323 |
| Phone values strong_unique / duplicate | 18 068 / 1 326 |
| SQLite sha256 | `a64da1ce6011f7a9778f22993f69377ffb9c28807804933aa5181a063eade743` |

## Relink results

Сравнение с fresh-only baseline `bacdd96f`:

| Набор | Fresh-only linked | Union linked | Дельта |
|---|---:|---:|---:|
| Corpus 27 009 | 21 095 / 78.10% | 20 556 / 76.11% | -539 |
| Delta 3 084 | 1 764 / 57.20% | 1 729 / 56.06% | -35 |

Union preview относительно исходно уже привязанных событий:

| Набор | Уже было linked | Новые links union | Итого linked |
|---|---:|---:|---:|
| Corpus 27 009 | 20 444 | 112 | 20 556 |
| Delta 3 084 | 1 134 | 595 | 1 729 |

Вклад новых union links:

| Набор | old_only | fresh_only | both |
|---|---:|---:|---:|
| Corpus | 0 | 37 | 75 |
| Delta | 0 | 560 | 35 |

## Почему KPI ТЗ не выполнен

Формальный код union работает по ТЗ, но строгий пересчёт общих значений на объединённой книге снижает автопривязку:

- из 651 fresh-only corpus links union теряет 539;
- причины потерь: `duplicate_identity_value` 291, `cross_brand_signal` 139, `identity_value_missing` 109;
- диагностика sample показала механизм: fresh-only email был уникален в свежей книге, но в old+fresh union тот же email/телефон встречается под другим `tallanto_id`; по предохранителю ТЗ `один email/телефон у >=2 разных tallanto_id -> не склеивать`.

То есть требование “пересчитать общие на union” и критерий “не ниже bacdd96f” конфликтуют на реальных данных. Чтобы вернуть 78.10%+, нужно отдельное бизнес-решение, например правило current/fresh-prefer для конфликтов old-vs-fresh. Я его не добавлял, потому что ТЗ прямо запрещает склеивать общие email/телефоны.

## Unmatched reasons

Corpus:

- `cross_brand_signal`: 2 663
- `identity_value_missing`: 2 489
- `duplicate_identity_value`: 1 286
- `multiple_identity_targets`: 15

Delta:

- `identity_value_missing`: 1 215
- `no_identity_signal`: 75
- `duplicate_identity_value`: 34
- `cross_brand_signal`: 30
- `multiple_identity_targets`: 1

## Примеры новых привязок без ПДн

| Набор | Сырьё | signal | signal_hash | tallanto_hash | bucket | brand |
|---|---|---|---|---|---|---|
| corpus | candidates.jsonl:53 | phone | `e358ae62a22f9963` | `2819aa1a47a803ed` | both | unknown |
| corpus | candidates.jsonl:55 | phone | `e358ae62a22f9963` | `2819aa1a47a803ed` | both | unknown |
| corpus | candidates.jsonl:235 | email | `1ca6998d20aa14a8` | `8ffb6c0d3371ea36` | both | unknown |
| corpus | candidates.jsonl:356 | email | `ada4f20b4341b9c0` | `03e2ae04aecd04a1` | both | unknown |
| corpus | candidates.jsonl:791 | email | `6074c4a23e533ee8` | `4196f9bc7d2a5bfd` | fresh_only | unknown |
| corpus | candidates.jsonl:982 | email | `16f302c0a6fc51fe` | `33a4d26d0b4184f4` | fresh_only | unknown |
| corpus | candidates.jsonl:1035 | email | `ed18ab8d85eb89e5` | `d0f5b9f3d386032b` | both | unpk |
| delta | candidates.jsonl:1 | email | `c378b8e3663f7d7c` | `8de9bcd846fffb3e` | fresh_only | unpk |
| delta | candidates.jsonl:2 | email | `c378b8e3663f7d7c` | `8de9bcd846fffb3e` | fresh_only | unpk |
| delta | candidates.jsonl:7 | email | `963e40317812e3e2` | `8ab6ef8f3e567efb` | fresh_only | unknown |
| delta | candidates.jsonl:8 | email | `ebf8730f84c0356b` | `8ab6ef8f3e567efb` | fresh_only | unpk |
| delta | candidates.jsonl:9 | email | `d9dcf93b9cbc52c6` | `2a9d744e28ca246c` | fresh_only | unpk |

## Проверки

- NEG: same `tallanto_id` in old+fresh -> one candidate, not duplicate.
- NEG: shared email across different IDs -> duplicate, no auto-link.
- NEG: duplicate `tallanto_id` inside one source remains blocking.
- NEG: fresh CSV comma parsing works.
- Idempotency: repeated corpus decisions CSV hash unchanged: `b82bfe6a0c152cbbf7bf661903a2d940fe04d61df0a9796cfaf1366abf03f878`.
- Idempotency: repeated delta decisions CSV hash unchanged: `a9bec32c6b50cf2973046c7565b07ef8935e03ff500300cf45a90894b6499d84`.
- Full pytest: `3440 passed, 5 skipped, 1 warning in 48.48s`.

## Semantic review

`formal_pass`: да.

`semantic_pass`: `PASS_WITH_NOTES`.

Причина: strict-union безопасно удерживает общие email/телефоны и cross-brand, но не достигает KPI “>= bacdd96f”. Это не баг теста, а бизнес-конфликт между полнотой автопривязки и строгим запретом склеивать один контактный сигнал на несколько `tallanto_id`.

Рекомендация архитектору: принять strict-union как безопасный диагностический артефакт или дать отдельное ТЗ на current/fresh-prefer rule для старых конфликтов с явным NEG на семьи/однофамильцев.
