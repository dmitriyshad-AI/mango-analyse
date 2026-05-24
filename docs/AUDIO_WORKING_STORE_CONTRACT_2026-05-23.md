# Audio Working Store Contract

Дата: 2026-05-23

## Назначение

Единая рабочая папка аудиозаписей для текущей canonical DB:

`product_data/audio_working_store_20260523_v1/`

Она собрана из runtime 2026-05-21 и затем стала текущим runtime-источником:

- canonical DB: `stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`;
- active export: `stable_runtime/sales_master_export_20260523_audio_working_store_v1`;
- runtime contract: `stable_runtime/CURRENT_RUNTIME.json`.

## Что внутри

- `audio/` - один файл на один уникальный SHA-256 аудио.
- `by_filename/` - совместимый слой с исходными именами файлов; это символические ссылки на `audio/`, не копии.
- `manifests/call_audio_mapping.csv` - связь каждого `canonical_call_id` с новым рабочим аудио.
- `manifests/unique_audio_manifest.csv` - список уникальных аудио.
- `manifests/summary.json` - сводка сборки и проверки.

## Результат сборки

- строк в canonical DB: `65 974`;
- привязано к рабочей папке: `65 974`;
- уникальных аудио по SHA-256: `65 974`;
- точных дублей по SHA-256: `0`;
- отсутствующих исходных файлов: `0`;
- несовпадений размера: `0`;
- неожиданных расширений: `0`;
- способ материализации: `hardlink` для `65 974` файлов.

## Безопасность

Текущая canonical DB больше не читает старые аудио-пути: все `source_file` указывают на `product_data/audio_working_store_20260523_v1/audio/`.

Старые mp3-копии перенесены в корзину только после SHA-256 проверки покрытия. Manifest cleanup:

`docs/AUDIO_WORKING_STORE_CLEANUP_MANIFEST_2026-05-23.md`

Новый рабочий путь можно использовать в новых задачах через:

`product_data/audio_working_store_20260523_v1/by_filename/`

если скрипту нужны исходные имена файлов, или через:

`product_data/audio_working_store_20260523_v1/manifests/call_audio_mapping.csv`

если скрипту нужен надежный путь по `canonical_call_id`.

## Старые аудио-папки

Старые mp3-копии из основных аудио-источников уже перенесены в корзину по manifest. Не-аудио служебные файлы старых папок оставлены, чтобы не ломать исторические отчеты и product appliance.

Не создавать новые Mango-аудио в старых папках. Для новых задач использовать отдельный update-batch, затем добавлять аудио в текущий working store через manifest.
