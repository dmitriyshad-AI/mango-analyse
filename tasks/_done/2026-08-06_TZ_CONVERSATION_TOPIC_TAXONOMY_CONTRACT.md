> DONE 2026-08-06 09:48 | ветка codex/conversation-topic-taxonomy-contract | codex

> TAKE 2026-08-06 09:30 | ветка codex/conversation-topic-taxonomy-contract | codex

Ветка: codex/conversation-topic-taxonomy-contract
Зоны: src/mango_mvp/channels/conversation_intent_plan.py, tests/test_conversation_intent_plan.py, tests/test_subscription_llm_draft_provider.py, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_conversation_intent_plan.py tests/test_draft_prompt_builder.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# ТЗ: единый идентификатор темы преподавателя

## Проблема

Живой `build_conversation_intent_plan()` возвращает для намерения `teacher`
идентификатор `theme:017_teachers`, которого нет в канонической taxonomy.
Канонический идентификатор уже существует: `theme:017_teacher_method`.

## Образ результата и бизнес-польза

Вопрос клиента о преподавателе получает допустимый `topic_id`, поэтому готовый
план не отбрасывается следующими проверками taxonomy и менеджер получает
содержательный черновик по правильной теме.

## Минимальное решение

1. Заменить только устаревший идентификатор в существующем сопоставлении.
2. Добавить сквозной тест через `build_conversation_intent_plan()` с реальным
   вопросом о преподавателе и проверкой канонического ID.
3. Не добавлять новый классификатор, regex, feature flag, модуль или зависимость.

## Приёмка

- вопрос «Кто будет преподавать физику?» распознаётся как `teacher`;
- `topic_id == theme:017_teacher_method`;
- `topic_id` входит в канонический набор taxonomy;
- отрицательный контроль со старым ID краснеет;
- целевые тесты зелёные;
- отдельный смысловой аудит подтверждает, что бизнес-смысл темы не изменён.

## СТОП

- если `teacher` уже нормализуется на канонический ID позже по живому пути;
- если исправление требует нового слоя или более 20 строк нетестового кода;
- если изменение маршрутизации выходит за тему преподавателя.

## Результат

- Живой план и тестовая фабрика используют `theme:017_teacher_method`.
- Сквозной тест подтверждает намерение, точный ID и членство в taxonomy.
- Мутация на прежний ID красит тест.
- Целевой набор: 397 passed.
- Полный набор: 4955 passed, 3 skipped, 10 известных падений чистого main.
- Производственный код: +1/-1; новых файлов, флагов и зависимостей нет.
