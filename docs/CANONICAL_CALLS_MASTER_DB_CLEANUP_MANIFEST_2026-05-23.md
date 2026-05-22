# Canonical Calls Master DB Cleanup Manifest — 2026-05-23

## Решение

Старые файлы `canonical_calls_master.db` удалены из рабочих `stable_runtime/canonical_master_*` папок переносом в корзину macOS. Папки сборок, `summary.json` и прочие текстовые артефакты оставлены на месте как история/audit trail.

Текущая защищенная база осталась на месте:

`stable_runtime/canonical_master_20260521_after_mango_update_v1/canonical_calls_master.db`

Источник истины для текущего runtime:

`stable_runtime/CURRENT_RUNTIME.json`

## Что защищено

| Статус | Путь | Причина |
|---|---|---|
| protected_current | `stable_runtime/canonical_master_20260521_after_mango_update_v1/canonical_calls_master.db` | Текущая база из `CURRENT_RUNTIME.json`; `65 974` звонков всего, `65 939` actionable, missing ASR/R+A = `0/0`. |
| protected_current | `stable_runtime/CURRENT_RUNTIME.json` | Машинный контракт текущего runtime. |
| protected_current | `stable_runtime/CANONICAL_EXPORT.txt` | Указатель активного export. |
| protected_current | `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance/` | Активный export, связанный с текущей canonical DB. |

## Что удалено из рабочей папки переносом в корзину

Корзина:

`/Users/dmitrijfabarisov/.Trash/mango_canonical_calls_master_db_cleanup_20260523_001331/`

| Исходный путь | Размер | Состояние | Куда перенесено |
|---|---:|---|---|
| `stable_runtime/canonical_master_20260509_v1/canonical_calls_master.db` | ~1.4G | старый слой | `/Users/dmitrijfabarisov/.Trash/mango_canonical_calls_master_db_cleanup_20260523_001331/canonical_master_20260509_v1__canonical_calls_master.db` |
| `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db` | ~1.4G | старый слой | `/Users/dmitrijfabarisov/.Trash/mango_canonical_calls_master_db_cleanup_20260523_001331/canonical_master_20260510_after_quality_backfill_v1__canonical_calls_master.db` |
| `stable_runtime/canonical_master_20260516_after_mango_update_v1/canonical_calls_master.db` | ~1.4G | старый слой | `/Users/dmitrijfabarisov/.Trash/mango_canonical_calls_master_db_cleanup_20260523_001331/canonical_master_20260516_after_mango_update_v1__canonical_calls_master.db` |
| `stable_runtime/canonical_master_20260517_after_mango_asr_only_v1/canonical_calls_master.db` | ~1.5G | промежуточный ASR-only слой, не текущий accepted runtime | `/Users/dmitrijfabarisov/.Trash/mango_canonical_calls_master_db_cleanup_20260523_001331/canonical_master_20260517_after_mango_asr_only_v1__canonical_calls_master.db` |

## Что было исправлено перед удалением

Убраны старые жесткие ссылки на `canonical_master_20260509/20260510/20260516/20260517` из активных `src/` и `scripts/` путей.

Измененные скрипты теперь берут текущую canonical DB из `stable_runtime/CURRENT_RUNTIME.json`:

- `scripts/build_post_backfill_amo_ready_export.py`
- `scripts/build_insight_readiness_from_canonical.py`
- `scripts/build_audio_store_downstream_projection.py`
- `scripts/build_canonical_after_mango_update.py`

Также обновлена проверка источника в:

- `scripts/build_deal_aware_stage709_review.py`

Она больше не считает текущим только старый путь `canonical_master_20260510_after_quality_backfill_v1`, а сверяется с `CURRENT_RUNTIME.json`.

## Проверки после переноса

Команды/результаты:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile \
  scripts/build_post_backfill_amo_ready_export.py \
  scripts/build_insight_readiness_from_canonical.py \
  scripts/build_audio_store_downstream_projection.py \
  scripts/build_canonical_after_mango_update.py \
  scripts/build_deal_aware_stage709_review.py
```

Результат: OK.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_call_processing_readiness.py \
  tests/test_productization_current_runtime_operator_status.py
```

Результат: `8 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_current_runtime.py \
  --out /tmp/CURRENT_RUNTIME_check_after_canonical_cleanup.json
```

Результат: `validation_ok=true`, `blocked=0`, `warnings=0`, `canonical_actionable_calls=65939`, missing ASR/R+A = `0/0`.

```bash
rg -n "canonical_master_202605(09|10|16|17)|stable_runtime/canonical_master_202605" src scripts -S
```

Результат: нет совпадений.

```bash
find stable_runtime -maxdepth 3 -name canonical_calls_master.db -type f -print
```

Результат: осталась только текущая база `canonical_master_20260521_after_mango_update_v1/canonical_calls_master.db`.

## Как восстановить, если понадобится

Пример для восстановления `20260516`:

```bash
mv "/Users/dmitrijfabarisov/.Trash/mango_canonical_calls_master_db_cleanup_20260523_001331/canonical_master_20260516_after_mango_update_v1__canonical_calls_master.db" \
   "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260516_after_mango_update_v1/canonical_calls_master.db"
```

Аналогично для остальных трех файлов.

## Ограничения

- Удалены только старые `.db` файлы, не папки сборок целиком.
- Документы и старые отчеты могут по-прежнему упоминать старые пути как историю проекта; это не рабочие ссылки.
- Если очистить корзину macOS, быстрый откат через `mv` будет невозможен.
