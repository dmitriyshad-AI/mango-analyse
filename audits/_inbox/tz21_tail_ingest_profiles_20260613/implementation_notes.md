# TZ-21 implementation notes

Дата: 2026-06-13

## Что сделано

- Блок А уже зафиксирован отдельным отчётом: `tasks/_done/2026-06-13_TZ21_blockA_tail_import_report.md`.
- Хвост 3,439 был влит в canonical DB с обязательным бэкапом и идемпотентной проверкой.
- Блок Б пересобрал профили в новой ignored-папке:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_profiles_after_tail_20260613/`
- Перед полной сборкой сборщик выполнил микропробу на 5 профилях.
- Полная сборка построила 18,399 профилей и отдельную идемпотентную копию.
- Найден и исправлен технический дефект: SQLite read-only URI ломался на пути проекта с пробелом. Теперь профильный builder и скрипты TZ-16/TZ-21 используют `mode=ro&immutable=1`.

## Ключевые артефакты

- Summary: `product_data/customer_profiles/tz21_profiles_after_tail_20260613/summary.json`
- До/после: `product_data/customer_profiles/tz21_profiles_after_tail_20260613/tz21_comparison.json`
- Tail after TZ-21: `product_data/customer_profiles/tz21_profiles_after_tail_20260613/rerun_tail_after_tz21.json`

Все эти файлы лежат под ignored `product_data/customer_profiles/` и не должны попадать в git.

## Основные счётчики

- Canonical v7: 22,679 -> 26,118, дельта +3,439.
- Zone v7: 16,797 -> 20,236, дельта +3,439.
- Zone old not-v7: 29,390 -> 25,951, дельта -3,439.
- Длинный non-blacklist хвост после вливания: 0.
- Остались только 56 длинных blacklist-звонков в зоне, их этот блок не трогал.
- Blacklist rows with v7: 0.

## Профили

- Profiles: 18,399 -> 18,399.
- Fields total: 190,614 -> 190,670.
- Active fields: 117,823 -> 117,702.
- Superseded fields: 72,791 -> 72,968.
- Profiles with 2+ children: 4,411 -> 4,410.
- Merge-candidate profiles: 824 -> 830.
- Merge-candidate markers: 835 -> 841.
- Full build time: 36.576 sec; total script elapsed: 95.982 sec.
- Idempotence: `content_signature_equal=true`.

## Safety

- AMO/CRM write: no.
- Tallanto write: no.
- ASR: no.
- Resolve+Analyze: no.
- LLM calls: 0.
