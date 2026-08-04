> DONE 2026-08-04 13:56 | ветка main | codex

> TAKE 2026-08-04 07:38 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/semantic_roles.py, src/mango_mvp/channels/conversation_intent_plan.py, src/mango_mvp/channels/dialogue_memory.py, src/mango_mvp/channels/few_shot_reference.py, src/mango_mvp/channels/telegram_pilot_context_builder.py, src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/channels/subscription_llm_parts/provider.py, src/mango_mvp/channels/subscription_llm_parts/policy_routing.py, src/mango_mvp/channels/subscription_llm_parts/post_layers.py, src/mango_mvp/channels/subscription_llm_parts/support.py, tests/test_regex_map_selective_integration.py, tests/test_subscription_llm_draft_provider.py, tests/test_direct_path_semantic_frame_shadow.py, tests/test_adr003_regex_understanding_moratorium.py, tests/test_telegram_public_pilot_bots.py, tests/fixtures/adr003_direct_path_text_patterns_snapshot.json, tests/fixtures/adr003_runtime_channel_regex_snapshot.json, docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md, docs/adr003_understanding_map.yaml, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_subscription_llm_draft_provider.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# Выборочная приёмка ветки regex-map

## Образ результата и бизнес-польза

Бот понимает естественно сформулированную потребность в цене через уже
существующий вызов модели и выбирает подтверждённый факт без подсказки старой
регулярки. Он не теряет уже прочитанные слоты и не превращает менеджерский
маршрут или содержательный ответ в ложное прощание. P0, наличие мест, привязка
слотов и финальные защитные полы не ослабляются.

## Донор и доказанный инвентарь

- Донор: `codex/regex-map-claude-code-20260731@cfa13826`.
- Прямое слияние запрещено: донор отстаёт от `main`, содержит около 70 тысяч
  строк исследовательских артефактов и откатывает более строгий P0-контракт.
- Существующий `_apply_tone_close_frame_veto` уже защищает покупательские
  сигналы через `deal_stage/payment_readiness/requested_action`; второй
  классификатор tone-close не строить.
- Существующие P0/output/availability floors сохраняются без изменений.

## Три варианта

1. Слить четыре коммита донора: отвергнуто как откат, дублирование и мусор.
2. Построить общий ручной словарь и грамматику цены: отвергнуто после аудита,
   потому что каждая новая формулировка требует очередной заплатки и добавляет
   около 185 строк кода.
3. Включить уже существующий model-driven retriever в пилотном профиле и
   оставить старый keyword-путь только явным операционным откатом: выбран
   минимальный вариант без нового вызова модели, флага или классификатора.

## Реализация

1. Включить существующие `TELEGRAM_RETRIEVER_MODEL_DRIVEN` и
   `TELEGRAM_ASSUMED_SCOPE_GUARD` в `pilot_gold_v1`. Вызов модели уже есть и
   выполняется до генерации; дополнительных LLM-вызовов не добавлять.
2. В model-driven prompt не передавать `primary_intent`, `answer_topics` и
   `required_fact_keys`, полученные старым regex-слоем. Передавать только
   подтверждённые слоты; иначе модель будет повторять ошибку регулярки.
3. Пустой или аварийный результат model-driven retriever считать безопасным
   отсутствием подтверждённого факта. Не запускать после него keyword-отбор:
   он доказанно передаёт нерелевантные факты. Старый путь доступен только при
   явном операционном откате model-driven режима.
4. Передать текущий контекст в существующий
   `_semantic_reading_slots_from_payload`, но сохранить подтверждение слотов
   историей и защиту от неоднозначности.
5. Не принимать произвольную модельную `metadata`: служебные гейты и трассы
   собираются локально только из нормализованных физических полей ответа.
6. Добавить в тот же semantic frame строгий JSON boolean
   `open_question_unanswered`. Невалидное или отсутствующее поле не должно
   разрешать новый самостоятельный ответ.
7. Использовать это поле только как запрет tone-close. Оно не определяет P0,
   intent, route и не создаёт нового смыслового хозяина.
8. Сделать маршруты `manager_only` и `draft_for_manager` монотонными:
   tone-close может оставить диагностическую метку, но не менять маршрут,
   текст или safety flags.
9. Вернуть сквозные тесты существующего frame-veto, потерянные старым merge.
10. Не менять состав regex-снимка и бюджеты: эта ветка не добавляет и не
   расширяет регулярки понимания. Механическое обновление координат разрешено,
   если набор 832 устойчивых `row_id` и все смысловые поля совпадают.

## Не принимать

- `conversation_state` и второй хозяин tone-close;
- отключение availability floor моделью;
- `require_history_support=False`;
- двусторонний model-led off-topic;
- output probe без потребителя и тревоги;
- prompt-дубли, старые снапшоты, карты и `artifacts/**`.

## Приёмка

- Естественный денежный вопрос через настоящий `build_draft` получает только
  актуальный клиентский факт конкретного продукта, если он есть; отсутствие
  такого факта не превращается в цену другого продукта или истёкшую цену.
- В model-driven retriever prompt отсутствуют старые `primary_intent`,
  `answer_topics` и `required_fact_keys`.
- Для вопроса об адресе модель не передаёт ценовой факт генератору.
- Пустая декларация, пустой выбор и ошибка модели не включают keyword-отбор и
  не передают генератору нерелевантные факты; явный откат меняет способ отбора.
- Контекстный `slots_gsf` работает, но предмет без подтверждения не
  записывается.
- Raw payload -> normalizer -> `build_draft`: открытый вопрос сохраняет
  исходный черновик; nested/invalid поле не подделывает floor.
- `manager_only` и `draft_for_manager` после tone-close неизменны.
- «Ок, беру» и «Хорошо, оформляйте» сохраняют содержательный черновик через
  существующий frame-veto.
- P0 и обещание неподтверждённых мест не ослаблены.
- P0 останавливается до retriever-вызова и остаётся `manager_only`.
- Мораторий, целевые тесты и полный регресс зелёные относительно известных
  средовых падений.

## Бритва

- Новых вызовов модели: 0.
- Новых feature flags: 0.
- Новых зависимостей: 0.
- Новый production-файл, regex-словарь или второй классификатор запрещены.
- Нет live, сети, внешних записей и runtime-изменений.

## СТОП

- Остановиться, если исправление требует снять P0-, availability-, brand-,
  number- или fact-floor.
- Остановиться, если полезный результат требует нового вызова модели, нового
  feature flag или прямого слияния артефактов донора.
- Не менять live/runtime и не запускать внешние записи.
