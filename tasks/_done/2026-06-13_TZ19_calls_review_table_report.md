# TZ-19 calls review table report

Дата: 2026-06-13

## Что сделано

Собрана Excel-таблица для ручного анализа Analyse v7 звонков Дмитрием.

Файл:

`/Users/dmitrijfabarisov/Claude Projects/Foton/tz19_calls_review_table_2026-06-13.xlsx`

Сводка рядом с файлом:

`/Users/dmitrijfabarisov/Claude Projects/Foton/tz19_calls_review_table_2026-06-13.summary.json`

В git добавлен только скрипт сборки и тесты. Excel-файл лежит вне репозитория.

## Scope decision

ТЗ просило Analyse v7 на 22,679 строк. Текущая canonical DB после TZ-21 уже содержит 26,118 v7-строк. Чтобы выполнить ТЗ буквально, скрипт по умолчанию использует `scope=baseline_22679`: текущие v7 минус tail manifest TZ-21 на 3,439 строк.

## Workbook sheets

- `Все`: 22,679 строк.
- `blacklist-77`: 0 строк.
- `длинные`: 2,796 строк, порог `transcript_chars > 10000`.

У всех листов закреплены заголовки и включён автофильтр.

## Counts

- Current v7 total in DB: 26,118.
- Tail manifest excluded: 3,439.
- Exported rows: 22,679.
- Blacklist ids loaded: 77.
- Blacklist rows inside exported v7 scope: 0.
- Long rows: 2,796.

Resolve breakdown:

- нашёл одного: 9,249.
- неоднозначно: 13,383.
- не нашёл: 47.

Brand/branch best-effort breakdown:

- unknown: 20,339.
- unpk: 1,918.
- foton: 289.
- mixed: 133.

## PII controls

- Raw workbook is outside git under `/Users/dmitrijfabarisov/Claude Projects/Foton/`.
- Open columns do not include raw phone numbers or FIO.
- Phone is represented as mask plus grouped hash, for example `ph_xxxx_xxxx_xxxx`.
- Email/phone regex scan over generated workbook: `email_hits=0`, `phone_hits=0`.
- Full `history_summary` is not exported; table uses shorter columnar fields.

## Safety

- Canonical DB access: read-only.
- ASR: not run.
- Analyse/R+A: not run.
- AMO/CRM/Tallanto writes: none.
- LLM calls: 0.

## Tests

- Targeted: `2 passed in 0.13s`.
- Full pytest: `3102 passed, 2 skipped, 1 warning in 46.13s`.

Warning is existing environment noise from urllib3/LibreSSL.
