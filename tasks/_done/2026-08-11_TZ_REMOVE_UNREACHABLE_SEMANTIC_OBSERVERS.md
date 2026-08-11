> DONE 2026-08-11 20:16 | ветка codex/semantic-routing-cleanup-wave2-20260811 | codex

> TAKE 2026-08-11 19:25 | ветка codex/semantic-routing-cleanup-wave2-20260811 | codex

Ветка: codex/semantic-routing-cleanup-wave2-20260811
Зоны: src/mango_mvp/channels/subscription_llm_parts/, scripts/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_adr003_semantic_reading_trace.py tests/test_subscription_llm_parts_facade.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# ТЗ: удалить невызванные смысловые наблюдатели и regex-ремонтники

## Проблема

После первой волны модель уже является владельцем смысла на рабочем direct-path,
но `policy_routing.py` всё ещё содержит несколько старых цепочек. Они угадывают
намерение или переписывают ответ по словам, хотя рабочий provider их не вызывает.
Пассивные фасадные импорты, экспорт и тесты делают этот код похожим на живой.

Graphify на точном SHA скопировал исходники, но построил пустой граф; поэтому он
не используется как доказательство отсутствия вызовов. Доказательство строится
по AST-графу, сырому `rg`, реальному `build_draft()` и точечным тестам.

## Образ результата и бизнес-польза

- Рабочий путь и фактический текст черновика не меняются.
- Модель остаётся единственным владельцем смысла; мёртвые regex-решения не могут
  случайно вернуться через фасад или новый вызов.
- P0, проверка фактов, личности, ПДн и финальный output-floor не меняются.
- Удалено больше рабочего кода, чем добавлено; новых runtime-файлов, флагов,
  зависимостей и LLM-вызовов нет.

## Бритва: три варианта

1. Удалить только доказанно невызванные цепочки и пассивные импорты — выбран.
2. Удалить весь `conversation_intent_plan` — запрещено: он участвует в подборе
   фактов и prompt, а связанного Wappi replay пока нет.
3. Переписать мёртвые цепочки новым модельным вызовом — запрещено как лишняя
   архитектура: рабочий путь уже получает SemanticFrame от основной модели.

## Точный срез

1. Удалить невызванные `autonomy_scope_precision`-ремонтники адреса, Fix1b,
   старый verified-informational helper, compact-plan helper и их закрытые
   вспомогательные цепочки.
2. Удалить `apply_known_context_redundant_question_guard`, `apply_reask_read_trace`,
   `apply_roles_read_trace` и только те helper/константы, которые замкнуты на них.
3. Удалить пассивные импорты и re-export этих символов из `provider.py`,
   `post_layers.py` и `subscription_llm_parts/__init__.py`.
4. Удалить тесты, которые проверяют только отсутствие уже невызванного поведения;
   живые provider/floor/semantic-frame тесты сохранить.
5. Механически обновить frozen snapshots, regex-бюджет и историческую карту.

## Приёмка

- Сырой поиск не находит удалённых символов в `src/` и `scripts/`.
- Рабочая цепочка `SubscriptionLlmDraftProvider.build_draft()` и все передаваемые
  callbacks остаются на месте.
- Точечный набор, façade-контракт, moratorium и collect-only зелёные.
- До/после: production LOC, runtime regex и direct-path text patterns уменьшаются.
- Независимые architect, breaker, business и cleaner reviews не находят изменения
  маршрута, фактов или текста рабочего черновика.

## Стоп-условия

- Не удалять `conversation_intent_plan`, model/fact retriever и provider callbacks.
- Не удалять P0/output/fact/identity/PII floors даже при малом числе прямых ссылок.
- Не менять Customer Timeline, Calls, runtime, AMO/Tallanto и внешние сообщения.
- Если символ имеет фактического production caller или требуется как callback,
  исключить его из среза, а не заменять заплаткой.

## Результат 2026-08-11

- Удалены 46 верхнеуровневых узлов: 37 функций и 9 констант. Новых
  исполняемых узлов нет.
- Из профиля убраны ложно «включённые» `route_templates`, `reask_read`, `roles_read`,
  `off_topic`, `intent_actions`, `rewrite_quality`, `post_semantics`. Остались только
  `sense_seats`, `slots_gsf`, `live_status_read`, `fact_select_read`, у которых есть
  фактические production-потребители.
- Целиком удалён невызываемый `TELEGRAM_READING_APPLY_CLASSES`-реестр; ADR-003
  runner, red-switch plan, runbook и тесты приведены к фактическому контракту.
- Код `src/`: `+6/-797`, net `-791`; операционные scripts: `+11/-84`, net `-73`.
  Весь tracked diff: `+471/-2006`, net `-1535`.
- Direct-path inventory: `735 -> 701`; удалены 26 inline regex-вызовов. Runtime
  `re.compile` snapshot не изменился: `163`; новых regex нет.
- Целевой набор: `358 passed`; collect-only: `4918 collected`. Полный прогон:
  `4903 passed`, `3 skipped`, `12 failed`; один новый timeout сразу перепроверен
  отдельно (`1 passed`), остальные 11 падений по списку совпадают с базой
  `81105c08` и относятся к sandbox/KB-data/вложенному worktree.
- Architect, breaker, business reviewer и cleaner после двух раундов дали `PASS`.
  Их первые BLOCK по ложному профилю, хвосту импортов, provenance карты и
  устаревшему runbook исправлены в том же корне.
- Новых runtime-файлов, feature flags, зависимостей, LLM-вызовов, live-write
  и отправок клиентам нет.
