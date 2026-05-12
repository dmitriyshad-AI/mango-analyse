# SaaS Sanitized Real Demo Audit

Дата: 2026-05-09

## Решение

Для B2B-презентаций нужен не fake-only demo, а sanitized real demo: реальные
строки product DB и CRM snapshots, но без персональных/операционных
идентификаторов.

## Реализовано

| Area | Artifact | Result |
|---|---|---|
| Exporter | `src/mango_mvp/productization/sanitized_real_demo.py` | Создает отдельный demo product root из source product DB. |
| CLI | `scripts/mango_office_sanitized_real_demo.py` | Запускает exporter с `--source-product-root`, `--source-product-db`, `--demo-product-root`, `--replace`. |
| Tests | `tests/test_productization_sanitized_real_demo.py` | Проверяет копирование структуры, маскирование, CLI и отказ от runtime/stable sources. |
| Runbook | `src/mango_mvp/productization/appliance_command_surface.py` | Добавлена команда `build_sanitized_real_demo`. |

## Что сохраняется реальным

- количество звонков и capture rows;
- статусы capture/scheduler;
- даты/время и длительность;
- наличие/отсутствие recording id;
- распределение по менеджерам в обезличенных кодах;
- CRM/Tallanto mapping structure, если есть snapshots.

## Что маскируется

- телефоны;
- имена менеджеров;
- CRM owner names/emails/ids;
- provider call ids;
- event keys;
- recording ids/audio refs/recording URLs;
- raw payload refs/source refs;
- CRM/Tallanto entity ids and names;
- job input/output refs and result payloads.

## Safety

- `stable_runtime` не читается и не изменяется.
- Runtime DB filenames вроде `mango_mvp.db` и `ai_office.db` запрещены как source.
- Audio не копируется.
- ASR/R+A не запускаются.
- AMO/Tallanto/CRM live calls и writes не выполняются.
- Output пишется только в отдельный demo product root.

## Команда

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_sanitized_real_demo.py \
  --source-product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --source-product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --demo-product-root _local_archive_mango_api_downloads_20260507/sanitized_real_demo_appliance \
  --replace
```

После этого dashboard:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py \
  --product-root _local_archive_mango_api_downloads_20260507/sanitized_real_demo_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/sanitized_real_demo_appliance/mango_product_appliance.sqlite \
  serve --host 127.0.0.1 --port 8765
```

## Verification

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_sanitized_real_demo.py \
  tests/test_productization_appliance_command_surface.py \
  tests/test_productization_product_api.py \
  tests/test_productization_product_api_http.py

PYTHONPYCACHEPREFIX=/private/tmp/mango_pycache_sanitized_demo PYTHONPATH=src python3 -m py_compile \
  src/mango_mvp/productization/sanitized_real_demo.py \
  src/mango_mvp/productization/appliance_command_surface.py \
  scripts/mango_office_sanitized_real_demo.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

- focused productization tests: `21 passed`;
- full productization suite: `305 passed, 1 warning`;
- compile check: passed.

Local product DB dry-run:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_sanitized_real_demo.py \
  --source-product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --source-product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --demo-product-root _local_archive_mango_api_downloads_20260507/sanitized_real_demo_appliance \
  --replace
```

Result:

- `validation_ok=true`;
- `product_calls=297`;
- `capture_inbox_items=21`;
- `job_runs=5`;
- `snapshots_written=0`, because the source product root has no local
  `crm_snapshots/amocrm_entities.*` or `crm_snapshots/tallanto_entities.*`;
- `warnings=3`, all from pending owner mappings already present in the source
  product DB shape, not from sanitizer copy failures.

## Self-audit fixes

- Added a guard that refuses overlapping source/demo roots. This prevents
  accidental replacement of the source product DB when `--replace` is used.
- Added target-schema filtering while copying rows. Extra columns from a future
  source product DB are skipped and reported instead of being copied into the
  sanitized demo output.
- Added tests for overlapping root refusal and extra source column skipping.

## Остаточные ограничения

- Это demo из product DB, а не из runtime processing DB. Значит качество
  транскриптов/анализа не улучшается и не проверяется этим шагом.
- Если source product DB еще не содержит достаточно реальных capture rows, сначала
  нужен safe Mango shadow/capture ingest в product DB.
- Для внешней демонстрации все равно нужен ручной human review финального demo
  root перед показом клиенту.
