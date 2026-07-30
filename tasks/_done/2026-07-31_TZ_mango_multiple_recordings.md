> DONE 2026-07-31 01:56 | ветка codex/calls-dialogue-m1-20260730 | codex

> TAKE 2026-07-31 01:16 | ветка codex/calls-dialogue-m1-20260730 | codex

Ветка: codex/calls-dialogue-m1-20260730
Зоны: src/mango_mvp/productization/contracts.py, src/mango_mvp/productization/mango_office.py, src/mango_mvp/productization/mango_recordings.py, src/mango_mvp/productization/capture_staging.py, src/mango_mvp/customer_timeline/calls_two_processes.py, tests/test_productization_mango_office.py, tests/test_productization_mango_recordings.py, tests/test_productization_capture_staging.py, tests/test_mango_calls_two_processes.py, docs/RUNBOOK.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_mango_office.py tests/test_productization_mango_recordings.py tests/test_productization_capture_staging.py tests/test_mango_calls_two_processes.py
Семантический-аудит: нет

# ТЗ: несколько записей одного звонка Mango

## Цель

Не терять части звонка, когда Mango возвращает несколько recording id. Набор
частей должен накапливаться без удаления, а неоднозначный порядок должен закрыто
останавливать ASR одного логического звонка.

## Требования

1. Mapper принимает scalar, CSV, brackets и list, очищает пустые и повторные id,
   сохраняет порядок.
2. Контракт звонка и manifest хранят полный tuple recording ids; старые scalar
   строки manifest читаются без миграции.
3. Один id скачивается старым путём; несколько скачиваются по отдельности и
   получают `multiple_recordings_needs_review`, без догадки о порядке.
4. Ошибка одной части не публикует звонок в ASR; уже скачанные части сохраняются
   и повторно не скачиваются.
5. Повтор идентичного события не скачивает заново. Более короткий поздний снимок
   API не удаляет уже найденные recording id.
6. Один provider_call_id остаётся одной строкой ingest и одним звонком downstream.
7. Capture повторно опрашивает свежие manifest events и не фильтрует их внешним
   known-call слоем; одиночный звонок идёт в ASR только после окна стабилизации.

## Приёмка

- scalar-путь обратно совместим;
- list/CSV/brackets дают одинаковый ordered tuple;
- duplicate ids не скачиваются дважды;
- две части скачаны отдельно и не попали в ASR;
- part failure fail-closed, успешная часть сохранена для повтора;
- повтор без изменений даёт zero download;
- late-added и shrinking snapshots дают монотонный полный набор;
- свежая одиночная запись не попадает в metadata до стабилизации;
- тесты зелёные, реальный Mango/ASR/R+A/runtime не запускается.

## СТОП

- Не запускать Mango API, Process A/B, ASR, Resolve+Analyze.
- Не менять существующий runtime manifest/audio.
- Не создавать отдельные downstream call records для частей.

## Бритва

Новый механизм до 150 добавленных строк нетестового кода; новых feature flags и
зависимостей нет. Автоматическую склейку без временных границ не добавлять.
