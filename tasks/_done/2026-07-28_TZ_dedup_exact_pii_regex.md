> DONE 2026-07-28 17:03 | ветка main | codex

> TAKE 2026-07-28 16:57 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/subscription_llm_parts/provider.py, tests/test_direct_path_semantic_frame_shadow.py, tests/fixtures/adr003_runtime_channel_regex_snapshot.json, tests/fixtures/adr003_direct_path_text_patterns_snapshot.json, docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_path_semantic_frame_shadow.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: нет

# ТЗ: убрать точные дубли шаблонов телефона и почты

## Цель

Сохранить локальные имена `_SEMANTIC_FRAME_PHONE_RE` и
`_SEMANTIC_FRAME_EMAIL_RE`, но ссылаться на уже импортированные канонические
объекты `_A2_PHONE_RE` и `_CLIENT_EMAIL_RE` из `support.py`.

## Ограничения

- Поведение маскирования должно совпасть побайтово.
- Новые шаблоны, флаги, файлы рабочего кода и зависимости запрещены.
- Другие дубли, брендовые правила, P0 и Wappi не трогать.
- Изменение снимков ADR-003 допустимо только как удаление двух точных дублей и
  должно быть объяснено в документе моратория.

## СТОП

Остановиться без изменения кода, если шаблоны или флаги regex не совпадают,
целевые тесты красные до правки либо рабочее дерево получает чужие изменения.

## Приёмка

- Локальные имена являются теми же объектами, что канонические regex.
- Телефон и email маскируются как раньше.
- Целевые тесты и полный безопасный pytest зелёные.
- В рабочем коде удалено больше строк, чем добавлено.
