# SaaS Phase 4-6 Completion Report

Дата: 2026-05-09

Scope: SaaS/productization ветка. `stable_runtime` DB/audio/transcripts не
менялись, ASR/R+A не запускались, AMO/Tallanto/CRM live writes не выполнялись.
Фаза 7 намеренно не трогалась, потому что слой обработки разговоров сейчас
стабилизируется в отдельном диалоге.

## Phase 4. Mango capture from shadow to controlled ingest

Status: complete for controlled ingest v1.

Что добавлено:

- `controlled_capture_ingest_v1` contract;
- read-only plan mode для shadow poll reports;
- apply mode, который пишет только `capture_inbox_items` в product DB;
- классификация:
  - `INGEST_ENQUEUE_CAPTURE`;
  - `SKIP_DUPLICATE_CAPTURE_INBOX`;
  - `SKIP_DUPLICATE_PRODUCT_CALL`;
  - `WAIT_DELAYED_RECORDING`;
  - `SKIP_NO_RECORDING`;
  - `BLOCK_POLICY`;
  - `BLOCK_MISSING_EVENT_KEY`;
- grace-window для delayed recordings;
- CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_controlled_capture_ingest.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --report _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_outputs/shadow_poll_job.json \
  plan
```

Safety:

- no audio download;
- no ASR/R+A;
- no runtime DB writes;
- no CRM/Tallanto writes.

## Phase 5. Processing orchestration bridge

Status: complete for dry-run lifecycle bridge v1.

Что добавлено:

- `processing_lifecycle_v1` contract;
- lifecycle report from product capture inbox to ASR handoff readiness;
- явные состояния:
  - `CANDIDATE_ASR_HANDOFF_DRY_RUN`;
  - `SKIP_ALREADY_IN_HANDOFF_MANIFEST`;
  - `WAIT_RECORDING_ASSET`;
  - `WAIT_RECORDING_DOWNLOAD`;
  - `WAIT_ASSET_READY`;
  - `BLOCK_CAPTURE_STATUS`;
  - `BLOCK_MISSING_RECORDING_REF`;
  - `BLOCK_DUPLICATE_PROVIDER_CALL_ID`;
  - `BLOCK_DUPLICATE_RECORDING_ID`;
- усиленный idempotency check в `processing_handoff.py`:
  - duplicate `queue_item_id`;
  - duplicate `provider_call_id`;
  - duplicate `recording_id`;
- CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_processing_lifecycle.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --no-asset-db \
  --no-handoff-manifest
```

Safety:

- bridge is report-only;
- no auto-trigger;
- no ASR/R+A;
- no writes to `stable_runtime`;
- no CRM writes.

## Phase 6. Controlled CRM writeback

Status: complete for preview/approval contract v1.

Что добавлено:

- `crm_writeback_preview_v1` contract;
- AMO writeback preview diff;
- staged queue:
  - `batch_10`;
  - `batch_50`;
  - `batch_300`;
  - `full`;
- policy gates:
  - no automatic lead close;
  - no delete/merge contacts;
  - no direct client messages;
  - no overwrite of non-empty fields without safe mode;
- rollback plan contract;
- Product API `/writeback/previews` теперь возвращает structured preview;
- CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_crm_writeback_preview.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --stage batch_10
```

Current limitation:

- preview пока блокирует реальные product calls как `BLOCK_MISSING_CRM_ENTITY`,
  потому что product DB еще не хранит resolved `crm_entity_id`;
- preview также блокирует `BLOCK_MISSING_INSIGHT`, пока Phase 7/processing layer
  не даст готовые insight payloads;
- это ожидаемое состояние, а не ошибка: live write path остается выключенным.

## Verification

Targeted checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_controlled_capture_ingest.py \
  tests/test_productization_processing_lifecycle.py \
  tests/test_productization_processing_handoff.py \
  tests/test_productization_crm_writeback_preview.py \
  tests/test_productization_product_api.py \
  tests/test_productization_product_api_http.py
```

Full productization suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

## Audit result

Фазы 4-6 закрыты как безопасный productization layer:

- ingest умеет объяснять, что делать с новыми Mango calls;
- processing bridge умеет показывать готовность к handoff без запуска обработки;
- CRM layer умеет показывать preview/gates/rollback без live write.

Remaining gaps:

- Phase 7 надо отложить до стабилизации обработки разговоров;
- для полноценного CRM preview нужен отдельный resolver `crm_entity_id`;
- для клиентского demo нужен polished UI и обезличенный demo tenant.
