# Audio working store cleanup manifest - 2026-05-23

## Итог

Создана и включена единая рабочая папка аудиозаписей:

`product_data/audio_working_store_20260523_v1/`

Текущий runtime переключен на canonical DB, где все `source_file` указывают на новый audio store:

`stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`

Старые аудио-копии перенесены в корзину macOS, не удалены безвозвратно:

`/Users/dmitrijfabarisov/.Trash/MangoAnalyse_audio_cleanup_20260522T230404Z`

## Проверка покрытия перед переносом

Проверка выполнялась по SHA-256 содержимого, не по имени файла.

| Источник | Аудиофайлов | Покрыто новым store | Не покрыто |
|---|---:|---:|---:|
| `2026-03-09--26` | 64 867 | 64 867 | 0 |
| `2026-03-05-21-06-49-ч1` | 2 628 | 2 628 | 0 |
| `2026-03-05-21-06-49-ч2` | 2 347 | 2 347 | 0 |
| `_local_archive_mango_api_downloads_20260507` | 1 172 | 1 172 | 0 |
| `product_data/canonical_audio_store_20260516_v1/audio` | 65 138 | 65 138 | 0 |
| `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/audio` | 848 | 848 | 0 |

Перед cleanup в новый store были дополнительно добавлены orphan-аудио:

- 337 файлов из старых локальных источников;
- 9 файлов из `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/audio`.

Orphan manifest:

`product_data/audio_working_store_20260523_v1/manifests/orphan_audio_manifest.csv`

## Что перенесено в корзину

Машинный CSV-манифест:

`docs/AUDIO_WORKING_STORE_OLD_AUDIO_MOVED_2026-05-23.csv`

JSON summary:

`docs/AUDIO_WORKING_STORE_OLD_AUDIO_CLEANUP_SUMMARY_2026-05-23.json`

Кандидатов-аудиофайлов было проверено: `137025` (включая дополнительную проверку двух малых batch-папок 16 мая).
Непокрытых файлов: `0`.
Записей переноса в манифесте: `70702`.

Перенесены только аудиофайлы или audio-only директории. Не-аудио артефакты, HTML, SQLite, JSON, логи и product appliance структура оставлены на месте.

Дополнительно после глобальной проверки вне нового store были найдены и перенесены две малые Mango batch audio-папки:

- `product_data/mango_incremental_4_asr_ra_20260516_v1/audio` - 4 файла;
- `product_data/mango_new_21_asr_ra_20260516_v1/audio` - 21 файл.

Extra summary:

`docs/AUDIO_WORKING_STORE_OLD_AUDIO_MOVED_EXTRA_2026-05-23.json`


## Состояние после cleanup

- `2026-03-09--26`: mp3 = `0`, остались 36 HTML-файлов.
- `2026-03-05-21-06-49-ч1`: mp3 = `0`, остался 1 HTML-файл.
- `2026-03-05-21-06-49-ч2`: mp3 = `0`, остался 1 HTML-файл.
- `_local_archive_mango_api_downloads_20260507`: mp3 = `0`, product appliance и служебные файлы оставлены.
- `product_data/canonical_audio_store_20260516_v1/audio`: директория перенесена в корзину.
- `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/audio`: mp3 = `0`, папка оставлена пустой для исторического контракта batch.
- `product_data/mango_incremental_4_asr_ra_20260516_v1/audio`: директория перенесена в корзину.
- `product_data/mango_new_21_asr_ra_20260516_v1/audio`: директория перенесена в корзину.

## Проверка runtime после cleanup

`stable_runtime/CURRENT_RUNTIME.json` пересобран после переноса.

Итог:

- `validation_ok=true`;
- `blocked=0`;
- actionable calls = `65939`;
- missing ASR = `0`;
- missing Resolve+Analyze = `0`.

Проверка canonical DB:

- строк в `canonical_calls`: `65974`;
- ссылок на новый audio store: `65974`;
- ссылок на старые audio-папки: `0`;
- missing/size mismatch: `0`.

## Безопасные проверки

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_audio_store_projection.py \
  tests/test_productization_call_processing_readiness.py \
  tests/test_productization_current_runtime_operator_status.py
```

Результат: `10 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_audio_store_downstream_projection.py --no-verify-checksum
```

Результат: `validation_ok=true`, unresolved rows = `0`, target missing/size mismatch = `0`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
```

Результат: `1695 tests collected`.

## Откат

Если потребуется откатить перенос старых audio-копий, вернуть содержимое из:

`/Users/dmitrijfabarisov/.Trash/MangoAnalyse_audio_cleanup_20260522T230404Z`

Но текущий runtime уже не зависит от этих старых путей; все актуальные ссылки идут через `product_data/audio_working_store_20260523_v1/audio/`.
