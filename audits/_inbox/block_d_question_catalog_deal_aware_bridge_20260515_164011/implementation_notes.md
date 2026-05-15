# Implementation Notes

Дата: 2026-05-15
Блок: D
Статус: выполнено без live-запуска

## Что сделано

- `extract_call_questions()` теперь сохраняет явные source-ключи:
  - `call_id`;
  - `recording_id`;
  - `moment_id`;
  - `source_row_index`;
  - `source_kind`;
  - `source_table`;
  - `source_id_raw`.
- Добавлен `src/mango_mvp/question_catalog/source_index.py`.
- Добавлен скрипт `scripts/build_question_catalog_source_index.py`.
- `deal_quality_gate` получил optional-вход `question_catalog_source_index`.
- Без индекса поведение gate не меняется.
- С индексом gate блокирует первые рискованные конфликты.

## Ограничение scope

Блок D не делает большой rebuild каталога вопросов и не запускает реальные runtime-данные. Проверка сделана только на unit/fake-данных.
