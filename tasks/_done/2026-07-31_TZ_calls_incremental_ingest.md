> DONE 2026-07-31 02:30 | ветка codex/calls-dialogue-m1-20260730 | codex

> TAKE 2026-07-31 01:57 | ветка codex/calls-dialogue-m1-20260730 | codex

Ветка: codex/calls-dialogue-m1-20260730
Зоны: src/mango_mvp/customer_timeline/calls_two_processes.py, src/mango_mvp/services/ingest.py, src/mango_mvp/services/transcribe.py, tests/test_mango_calls_two_processes.py, tests/test_ingest.py, tests/test_ingest_filename_parse.py, docs/RUNBOOK.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_two_processes.py tests/test_ingest_filename_parse.py
Семантический-аудит: нет

# ТЗ: инкрементальный ingest звонков

## Цель

На неизменных данных второй Process A не должен снова готовить и перечитывать все старые аудиофайлы. Новые и незавершённые строки должны продолжать обрабатываться существующими worker-стадиями.

## Требования

1. `prepare_ingest_inputs` read-only читает уже известные `source_call_id` из working SQLite и не добавляет их повторно в metadata.
2. При отсутствии DB/table поведение совместимо: доступны все стабилизированные downloaded manifest entries.
3. Ingest с `metadata_csv` обходит только названные в ней файлы; header-only CSV даёт zero processed, а без CSV сохраняется старый полный обход.
4. Повторный запуск не удаляет и не перезаписывает старые аудио/DB.
5. Нерешённые multi, свежие stabilizing и missing audio остаются видимыми в skipped counters.

## Приёмка

- второй prepare на DB с существующим source_call_id: zero audio_files;
- pending/failed строки уже существующего CallRecord не дублируются, их подхватывают workers из DB;
- metadata с одним именем не сканирует соседний файл;
- header-only metadata: processed=0;
- вызов без metadata сохраняет прежнее поведение;
- реальный runtime/ASR/R+A не запускать.

## СТОП

- Не запускать Process A/B, Mango API, ASR или R+A.
- Не менять и не удалять реальные working SQLite/audio.

## Бритва

После независимого аудита граница расширена до нового механизма в 150 добавленных строк нетестового кода: дешёвый SQL не различает реальные хвосты, будущие повторы и зависшие блокировки. Новых файлов механизма, флагов и зависимостей нет.
