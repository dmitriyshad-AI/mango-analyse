# SaaS CRM Entity Resolver Stage 6 Audit

Дата: 2026-05-09

Scope: SaaS/productization ветка. Resolver не читает live amoCRM, не пишет CRM,
не трогает `stable_runtime`, не запускает ASR/R+A.

## What changed

Добавлен read-only resolver между product calls и CRM writeback preview:

```text
src/mango_mvp/productization/crm_entity_resolver.py
scripts/mango_office_crm_entity_resolver.py
```

Resolver берет:

- product DB `product_calls`;
- локальный CRM snapshot под product root;
- телефон клиента, извлеченный из `source_filename`;
- нормализует телефоны;
- строит exact phone match.

Поддержанные snapshot formats:

- JSON list;
- JSON object с ключом `entities`, `items`, `contacts`, `leads` или `rows`;
- JSONL;
- CSV.

Минимальные поля snapshot:

```json
{
  "entity_id": "501",
  "entity_type": "lead",
  "phone": "+79990000000"
}
```

## Resolution actions

- `RESOLVE_CRM_ENTITY`: один точный CRM entity match по телефону;
- `BLOCK_NO_CALL_PHONE`: у звонка не удалось извлечь телефон;
- `BLOCK_NO_CRM_MATCH`: по телефону нет entity в snapshot;
- `BLOCK_AMBIGUOUS_CRM_MATCH`: по телефону найдено несколько CRM entities.

Ambiguous matches intentionally block preview readiness. Автоматически выбирать
одну из нескольких сделок/контактов нельзя.

## Writeback preview integration

`crm_writeback_preview_v1` теперь принимает `crm_snapshot_path`.

Если snapshot передан или найден Product API default path:

```text
crm_snapshots/amocrm_entities.json
crm_snapshots/amocrm_entities.jsonl
crm_snapshots/amocrm_entities.csv
config/amocrm_entities.json
```

preview получает:

- `crm_entity_id`;
- `crm_entity_type`;
- compact `crm_resolution`;
- structured resolver summary.

Это снимает blocker `BLOCK_MISSING_CRM_ENTITY`, но не снимает
`BLOCK_MISSING_INSIGHT`. Insight payload должен прийти позже из processing/ROP
ветки.

## CLI

Read-only resolver:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_crm_entity_resolver.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --crm-snapshot _local_archive_mango_api_downloads_20260507/product_appliance/crm_snapshots/amocrm_entities.json
```

Preview with resolver:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_crm_writeback_preview.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --crm-snapshot _local_archive_mango_api_downloads_20260507/product_appliance/crm_snapshots/amocrm_entities.json \
  --stage batch_10
```

## Safety audit

Resolver safety contract:

- `product_db_writes=false`;
- `runtime_db_writes=false`;
- `stable_runtime_writes=false`;
- `live_crm_reads=false`;
- `write_crm=false`;
- `write_tallanto=false`;
- `run_asr=false`;
- `run_ra=false`.

CRM snapshot must stay under product root and is refused under `stable_runtime`.

## Remaining gap

Resolver currently uses local snapshot only. Это правильно для текущего этапа:
мы можем тестировать matching и preview без live CRM mutation. Позже можно
добавить отдельный guarded read-only AMO snapshot export, который будет сохранять
`crm_snapshots/amocrm_entities.json` под product root.
