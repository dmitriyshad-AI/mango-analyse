# Customer Timeline Import Runbook

Дата: 2026-05-12

## Назначение

`scripts/customer_timeline_import.py` переносит локальные read-only выгрузки в изолированную продуктовую БД `customer_timeline.sqlite`.

Команда нужна для безопасного шага:

1. прочитать локальный источник;
2. нормализовать записи в контракты единой истории клиента;
3. показать preview-отчет;
4. при явном `--apply` записать только в продуктовую timeline-БД.

## Что команда не делает

- не пишет в AMO;
- не пишет в Tallanto;
- не отправляет письма и сообщения;
- не запускает ASR;
- не запускает Resolve + Analyze;
- не скачивает аудио;
- не меняет `stable_runtime`;
- не пишет runtime-БД обработки звонков.

## Поддерживаемые источники

- `amocrm_snapshot`;
- `tallanto_snapshot`;
- `channel_snapshot`;
- `mail_archive`;
- `mango_processed_summary`.

Форматы локальных файлов: `.json`, `.jsonl`, `.csv`.

SQLite-источники читаются только через `mode=ro` и требуют `--sqlite-table`.

## Dry-run preview

```bash
PYTHONPATH=src python3 scripts/customer_timeline_import.py \
  --tenant-id foton \
  --source-kind tallanto_snapshot \
  --source-path /path/to/students.csv \
  --allowed-root /path/to/product_root \
  --timeline-db /path/to/product_root/customer_timeline/customer_timeline.sqlite \
  --source-ref students_export_2026-05-12 \
  --csv-encoding cp1251 \
  --csv-delimiter '\t' \
  --out /path/to/product_root/reports/customer_timeline_preview.json
```

В этом режиме БД не создается и не меняется. Отчет показывает `writes_planned`, конфликты, source inventory до/после и safety flags.

## Apply

```bash
PYTHONPATH=src python3 scripts/customer_timeline_import.py \
  --tenant-id foton \
  --source-kind tallanto_snapshot \
  --source-path /path/to/students.csv \
  --allowed-root /path/to/product_root \
  --timeline-db /path/to/product_root/customer_timeline/customer_timeline.sqlite \
  --source-ref students_export_2026-05-12 \
  --idempotency-key students_export_2026-05-12_v1 \
  --csv-encoding cp1251 \
  --csv-delimiter '\t' \
  --out /path/to/product_root/reports/customer_timeline_apply.json \
  --apply
```

`--apply` пишет только в указанную `customer_timeline.sqlite`. Повторный запуск с тем же `--idempotency-key` идемпотентен.

## SQLite source example

```bash
PYTHONPATH=src python3 scripts/customer_timeline_import.py \
  --tenant-id foton \
  --source-kind mail_archive \
  --source-path /path/to/mail_archive.sqlite \
  --allowed-root /path/to/product_root \
  --timeline-db /path/to/product_root/customer_timeline/customer_timeline.sqlite \
  --source-ref mail_archive_2026-05-12 \
  --sqlite-table messages \
  --sqlite-source-ref-column message_id \
  --out /path/to/product_root/reports/mail_timeline_preview.json
```

## Как читать отчет

- `summary` - короткий итог для оператора.
- `source` - что прочитано и изменился ли источник.
- `normalization` - сколько объектов единой истории получилось.
- `writes` - что будет записано или уже записано.
- `conflicts` - неоднозначные совпадения клиентов, которые не склеиваются автоматически.
- `errors` - строки, которые не удалось нормализовать.
- `safety` - подтверждение, что live-записи и тяжелая обработка отключены.
