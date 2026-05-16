# Audio Store Runtime Contract 2026-05-16

## Назначение

Новый audio-store нужен, чтобы downstream-слои не зависели от множества старых папок с дублями аудио. Он не меняет текущий accepted runtime сам по себе.

## Актуальный store

`product_data/canonical_audio_store_20260516_v1/`

Покрытие:

- `64 867` звонков из текущего canonical master;
- `267` новых Mango-записей из очереди на будущий ASR/R+A;
- всего `65 134` актуальных аудио;
- `0` пропущенных файлов;
- `0` ошибок проверки.

## Как downstream должен читать аудио

Источник истины для перехода:

`product_data/canonical_audio_store_20260516_v1/audio_store_mapping.csv`

Для новых обработок не нужно искать файл в старых папках. Нужно брать `canonical_audio_path` из mapping/projection.

Готовые проекции:

- `product_data/canonical_audio_store_20260516_v1/downstream_projection/canonical_calls_audio_store_projection.csv`
- `product_data/canonical_audio_store_20260516_v1/downstream_projection/new_mango_processing_handoff_audio_store.jsonl`

Проверочный отчёт:

`product_data/canonical_audio_store_20260516_v1/downstream_projection/downstream_projection_report.json`

## Безопасная команда пересборки проекции

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_audio_store_downstream_projection.py
```

Эта команда:

- читает текущий canonical DB только read-only;
- читает очередь новых Mango-записей;
- пишет только в `product_data/canonical_audio_store_20260516_v1/downstream_projection/`;
- не запускает ASR;
- не запускает Resolve+Analyze;
- не пишет в AMO/CRM/Tallanto;
- не меняет `stable_runtime`.

## Проверка ASR worker pack без запуска ASR

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_asr_worker_pack.py \
  --product-root . \
  --source-manifest product_data/canonical_audio_store_20260516_v1/downstream_projection/new_mango_processing_handoff_audio_store.jsonl \
  --pack-root product_data/canonical_audio_store_20260516_v1/downstream_projection/asr_worker_pack_dry_run \
  --pack-manifest product_data/canonical_audio_store_20260516_v1/downstream_projection/asr_worker_pack_dry_run/asr_worker_input_manifest.jsonl \
  --out product_data/canonical_audio_store_20260516_v1/downstream_projection/asr_worker_pack_dry_run/audit.json \
  --dry-run --mode hardlink
```

Ожидаемый результат текущей версии: `267/267`, `blocked=0`, `validation_ok=true`.

## Что пока нельзя делать

Без отдельного решения нельзя:

- удалять старые аудиофайлы;
- переносить старые аудиофайлы в quarantine;
- менять `stable_runtime` DB/source paths;
- запускать ASR/R+A по новой очереди;
- пересобирать canonical master/downstream;
- писать в AMO/CRM/Tallanto.

## Когда можно будет чистить дубли

Только после отдельного этапа:

1. downstream-пайплайн подтверждённо читает аудио через `audio_store_mapping.csv` / projection;
2. новый ASR/R+A проход по 267 звонкам выполнен и принят;
3. downstream-слои пересобраны и проверены;
4. audit pack подтверждает, что старые пути больше не нужны;
5. Дмитрий отдельно подтверждает физический перенос/удаление.

Текущий список кандидатов:

`product_data/canonical_audio_store_20260516_v1/exact_duplicate_cleanup_candidates.csv`

В нём все строки сейчас должны оставаться `safe_to_delete_or_move_now=false`.
