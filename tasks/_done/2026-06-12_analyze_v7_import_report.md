# Analyze v7 import report — 2026-06-12

## Итог

Вливание перепрогона `analyze` v7 в каноническую базу завершено.

База:

```text
stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db
```

Вход:

```text
/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611/results_filtered.jsonl.gz
```

Бэкап из ТЗ проверен и использован как уже существующий:

```text
stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db.backup_before_v7_20260612
```

Новый бэкап не создавался: запись шла с `--apply --no-backup`.

Пересборка профилей не запускалась.

## Предпроверки

- `results_filtered.jsonl.gz`: 22 679 строк.
- `blacklist_77.txt`: 77 непустых ID. `wc -l` показывает 76 из-за отсутствия завершающего перевода строки.
- База перед импортом: `PRAGMA quick_check` -> `ok`.
- Hot journal после открытия базы штатно откатился; после приёмки `canonical_calls_master.db-journal` отсутствует.

## Сухой прогон

Команда без `--apply`:

```text
read=22679
updated=22679
skipped_same=0
rejected_not_done=0
rejected_missing_row=0
rejected_transcript_changed=0
rejected_bad_json=0
rejected_meta=0
```

Счётчики полностью совпали с ожиданием.

## Запись

Команда:

```text
import_analyze_results.py --apply --no-backup
```

Результат:

```text
read=22679
updated=22679
skipped_same=0
rejected_not_done=0
rejected_missing_row=0
rejected_transcript_changed=0
rejected_bad_json=0
rejected_meta=0
```

Скрипт обновлял только поля выжимки:

- `analysis_json`
- `analysis_status`
- `analysis_json_chars`
- `has_analysis_json`
- `last_error`

ASR, Resolve+Analyze и пересборка профилей не запускались.

## Приёмка

Повторный прогон без `--apply`:

```text
read=22679
updated=0
skipped_same=22679
rejected_not_done=0
rejected_missing_row=0
rejected_transcript_changed=0
rejected_bad_json=0
rejected_meta=0
```

Проверка SQLite:

```text
PRAGMA quick_check;
ok
```

Счётчик v7:

```text
json_valid(analysis_json)
and json_extract(analysis_json, '$.analysis_meta.analysis_prompt_version') = 'v7'
=> 22679
```

Текстовый контроль маркера:

```text
analysis_json like '%"analysis_prompt_version": "v7"%'
=> 22679
```

Диапазон ID среди импортированных v7 строк:

```text
min=12608
max=65972
```

## Blacklist 77

77 ID из `blacklist_77.txt` не попали в `results_filtered.jsonl.gz`, поэтому старые канонические выжимки по ним сохранены.

Причина по логам и данным пакета:

- Технически М1 завершил все 22 756 строк: финальные `ab_summary_part*.json` показывают `done=5689` на каждую из 4 частей, `failed=0`, `not_done=0`, `empty_json=0`, `meta_v7_mini=5689`.
- До retry были только 3 технических сбоя: два timeout 180 сек и один `model at capacity`; retry их закрыл.
- 77 blacklist-строк не были техническими ошибками. Это смысловые ошибки модели `gpt-5.4-mini`: новая выжимка была `done/v7/mini`, но ложно сводила живой разговор к шаблону `Содержательного диалога не было: автоответчик, IVR, голосовой ассистент или технический недозвон`.
- Паттерн повторяется: 74 из 77 формулировок используют шаблон `автоответчик, IVR, голосовой ассистент или технический недозвон`, ещё 3 — `менеджер оставил сообщение на автоответчике`.
- 44 из 77 имеют `transcript_chars >= 5000`; 35 из 77 >= 10000; 12 из 77 >= 20000. Но проблема не только в длине: встречаются и короткие транскрипты.

Пример `canonical_call_id=19055`:

- `transcript_chars=55049`.
- Транскрипт живой: есть реплики менеджера и клиента, обсуждение онлайн-подготовки к ЕГЭ, предметы, уровни групп, цена, скидка, дальнейший звонок.
- Старая каноническая выжимка это отражает.
- Новая v7-mini выжимка ошибочно сказала: `Содержательного диалога не было: автоответчик, IVR, голосовой ассистент или технический недозвон`.

Вывод: blacklist 77 — не проблема импорта и не проблема SQLite. Это отдельный класс смысловой ошибки модели/промпта на распознавании живого диалога. Эти 77 строк правильно исключены из импорта, старые выжимки сохранены.

## Статус

- Импорт завершён.
- База цела.
- Идемпотентность подтверждена.
- `quick_check=ok`.
- `analysis_prompt_version=v7`: 22 679 строк.
- Профили клиентов не пересобирались.
