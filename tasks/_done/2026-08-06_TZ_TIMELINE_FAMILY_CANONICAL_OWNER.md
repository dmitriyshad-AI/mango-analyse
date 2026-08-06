> DONE 2026-08-06 05:41 | ветка codex/timeline-family-canonical-owner-20260806 | codex

> TAKE 2026-08-06 05:13 | ветка codex/timeline-family-canonical-owner-20260806 | codex

Ветка: codex/timeline-family-canonical-owner-20260806
Зоны: src/mango_mvp/customer_timeline/family_graph.py, tests/test_customer_timeline_family_graph.py, docs/worktrees_registry.md, docs/MANGO_CURRENT_STATE_V7.md, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_family_graph.py tests/test_customer_timeline_store.py tests/test_bot_safe_runtime_context.py tests/test_customer_timeline_manager_dossier.py
Семантический-аудит: да

# ТЗ: семейный граф не отменяет канонического владельца Tallanto student ID

## Проблема

`authoritative_tallanto_student_owners()` уже выбирает единственного точного
владельца student ID и оставляет `None` только для прямого конфликта или двух
точных владельцев. Но `_resolve_family_assignments()` повторно собирает
`customers_by_id` из исторических `ambiguous`-ссылок и при `len(owners) > 1`
помечает конфликтными всех, включая канонического владельца.

На фиксированном staging-клоне среди клиентов открытых `ambiguous_identity`
441 имеют `conflict/low`: у 376 единственная причина — слабая историческая
ссылка на тот же student ID у другого клиента. Существующий переоценщик затем
закрывает `0 из 1 397`, потому что канонический владелец уже ошибочно объявлен
небезопасным.

## Образ результата и бизнес-польза

Историческая слабая ссылка не скрывает у менеджера точного ученика, его семью и
оплаты. Ошибочный старый держатель остаётся конфликтным. Два точных владельца и
прямой `tallanto_identity_conflict` по-прежнему блокируют обоих. В черновик не
может попасть событие другого ребёнка.

## Рассмотренные варианты

1. Ослабить prompt-гейт — отклонено: маскирует дефект семейного графа.
2. Добавить отдельный repair/resolver — отклонено: дублирует канонического owner.
3. Выбранный минимум: в существующей проверке нескольких держателей не
   объявлять канонического владельца unsafe из-за слабых исторических ссылок.

## Реализация

1. Переиспользовать переданный в `_resolve_family_assignments()` словарь
   `exact_tallanto_owners`; не создавать новый resolver, таблицу или флаг.
2. Если canonical owner задан, ошибочные держатели уже блокируются строками
   выше, а canonical owner не блокируется повторно.
3. Если canonical owner отсутствует или равен `None`, несколько держателей
   остаются конфликтом.

## Приёмка

1. Новый тест: canonical exact owner + historical ambiguous holder одного
   student ID. Canonical owner не `conflict`, historical holder — `conflict`.
2. Сквозной семейный тест: два точных разных ребёнка одной семьи остаются одной
   семьёй даже при слабой старой ссылке на ID одного ребёнка; контактный конфликт
   переоценивается, чужое событие не переезжает.
3. Существующие тесты двух exact owners, direct conflict, diminutive duplicate и
   mixed-family остаются зелёными.
4. На клоне: сухой прогноз для 1 и 10 затронутых клиентов, затем один полный
   derived rebuild; `COUNT(timeline_events)` неизменен, FK/quick_check зелёные.
5. Показать было -> стало: `conflict/low`, open AI/SFP, blocked customers,
   prompt money на фиксированном обезличенном наборе. Любой чужой chunk = STOP.

## СТОП

- canonical owner выбирается при двух exact/manual владельцах;
- прямой `tallanto_identity_conflict` перестаёт блокировать;
- общий телефон используется без точного student ID;
- требуется изменение prompt-гейта, публикация или запись во внешнюю систему;
- `COUNT(timeline_events)` меняется либо появляется чужой chunk.

## Бритва

Один production-файл, без новых файлов кода, флагов, таблиц и зависимостей.
Ожидаемый production diff — до 5 строк.

## Результат

- production diff: `+2/-2` в одном существующем resolver, новых механизмов нет;
- отрицательный контроль со старой строкой краснеет на различающихся family ID;
- точечные проверки: `64 passed`; полный заявленный набор: `310 passed`;
- весь pytest: `4 959 passed, 3 skipped, 10 failed`; 10 падений относятся к
  прежнему ADR-003 snapshot и отсутствующему/неактуальному KB-каталогу;
- полный derived rebuild: `conflict 9 251 -> 8 639`, `confident 4 277 -> 4 719`,
  `singleton 21 319 -> 21 489`, закрыто `284 из 1 397` контактных конфликтов;
- повторный полный проход: закрыто `0`, числа не изменились;
- события: `486 114 -> 486 114`, `quick_check=ok`, FK-ошибок нет;
- бизнес-проверка: exact-владелец и семья восстановлены, чужой держатель остался
  конфликтным; однако разрешённая история оплат попала в prompt у `0 из 10`,
  а на отдельной десятке 8 семей блокирует соседний
  `tallanto_identity_ambiguous`; поэтому готовность всей памяти не заявляется.
