# ТЗ-119: фикс P0-самоотключения стража + микро-перезамер

Дата: 2026-06-16
Ветка: `codex/tz119-assumed-scope-guard-main`
Статус: `formal_pass`, `semantic_pass: PASS_WITH_NOTES`

## Что исправлено

### 1. Страж больше не самоотключается от самого факта MODEL_P0

Файл:

- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`

Было:

- `_direct_path_assumed_scope_p0_active()` считал P0 активным, если в metadata просто есть `direct_path_model_p0`.
- При `TELEGRAM_DIRECT_PATH_MODEL_P0=1` этот блок есть почти всегда, поэтому страж уходил в `skipped_p0_or_risk` даже на низком риске.

Стало:

- учитывается фактический риск:
  - `result.risk_level` high/p0/critical;
  - P0/high-risk safety flags;
  - `direct_path_model_p0.is_p0=true`;
  - `direct_path_model_p0.risk_level` high/p0/critical;
  - непустой реальный `p0_kind`, кроме `none/no/false/low/normal`;
  - активный P0-latch/risk flags в памяти.

### 2. Клиентский переспрос стража сужен до реальных параметров риска

Файл:

- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`

Стало:

- для мягкого выбора фактов `product/product_family` остаётся в scope-логике;
- но выходной клиентский страж больше не делает переспрос из-за широкого `product=курс`;
- переспрос возможен только по:
  - классу;
  - предмету;
  - формату.

Причина: микро-замер первой ON-версии показал плохой переспрос «про курс», который ухудшал диалог.

### 3. Исправлена кривая фраза «Для данные ребёнка»

Файл:

- `src/mango_mvp/channels/subscription_llm_parts/post_layers.py`

Стало:

- sanitizer меняет `Для данные ребёнка ...` на нормальное `Для ребёнка ...`;
- защита ПДн не ослаблена: неупомянутое имя всё равно удаляется.

## Тесты

Точечный набор после фикса:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_subscription_llm_draft_provider.py::test_tz119_model_driven_requires_assumed_scope_guard \
  tests/test_subscription_llm_draft_provider.py::test_tz119_assumed_scope_guard_reasks_without_manager_handoff \
  tests/test_subscription_llm_draft_provider.py::test_tz119_assumed_scope_guard_checks_low_risk_model_p0_metadata \
  tests/test_subscription_llm_draft_provider.py::test_tz119_confirmed_slot_quote_prevents_reask_on_ellipsis \
  tests/test_subscription_llm_draft_provider.py::test_tz119_assumed_scope_guard_skips_p0_risk \
  tests/test_subscription_llm_draft_provider.py::test_direct_path_model_p0_benign_messages_stay_autonomous \
  tests/test_subscription_llm_draft_provider.py::test_direct_path_model_p0_payment_dispute_routes_before_gate_and_replaces_sales_text \
  tests/test_subscription_llm_draft_provider.py::test_pii_relation_stopwords_flag_still_masks_unmentioned_name \
  tests/test_subscription_llm_draft_provider.py::test_direct_path_output_sanitizer_masks_unmentioned_child_name
```

Результат:

- `9 passed`

Расширенный точечный набор:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_subscription_llm_draft_provider.py \
  -k 'tz119 or direct_path_model_p0 or pii_relation_stopwords or output_sanitizer_masks_unmentioned_child_name'
```

Результат:

- `16 passed, 456 deselected`

Контроль sanitizer:

- `test_pii_relation_stopwords_flag_still_masks_unmentioned_name`: `1 passed`

## Микро-набор

Файл сценариев:

- `runs/tz119_assumed_scope_guard_20260616_micro/scenarios/tz119_micro_guard_20260616.jsonl`

Состав:

- `autonomy_unpk_real_002`
- `autonomy_unpk_real_012`
- 4 синтетических POS: класс/предмет/формат не подтверждены клиентом
- 3 ellipsis/NEG из gain:
  - `gain_fact_p03_unpk_physics_schedule_ellipsis`
  - `gain_fact_p14_foton_online_price_ellipsis`
  - `gain_fact_p15_unpk_grade_ellipsis`
- 1 NEG с подтверждённым классом:
  - `tz119_neg_confirmed_grade_price`
- 1 P0:
  - `tz119_p0_payment_dispute`

Папки прогонов:

- OFF: `runs/tz119_assumed_scope_guard_20260616_micro/off_micro`
- ON после финального фикса: `runs/tz119_assumed_scope_guard_20260616_micro/on_micro_v2`
- direct-smoke: `runs/tz119_assumed_scope_guard_20260616_micro/direct_guard_smoke_v2.json`

Флаги OFF:

- `TELEGRAM_ASSUMED_SCOPE_GUARD=0`
- `TELEGRAM_RETRIEVER_MODEL_DRIVEN=0`
- `TELEGRAM_RETRIEVER_NEED_SHADOW=0`

Флаги ON:

- `TELEGRAM_ASSUMED_SCOPE_GUARD=1`
- `TELEGRAM_RETRIEVER_MODEL_DRIVEN=1`
- `TELEGRAM_RETRIEVER_NEED_SHADOW=1`

Общие флаги:

- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- `TELEGRAM_DIRECT_PATH_MODEL_P0=1`
- `TELEGRAM_DEAL_ACTION_DECISION=1`
- `TELEGRAM_HANDOFF_TRACE=1`
- `DIALOGUE_CONTRACT_DEBUG_TRACE=1`
- `--parallel 4`
- `--judge-prompt-version v9.1`

Полные сеты не запускались.

## Микро-замер: OFF vs ON

| Метрика | OFF | ON v2 | Вывод |
|---|---:|---:|---|
| Диалогов | 11 | 11 | одинаково |
| Ходов | 21 | 21 | одинаково |
| FAIL | 0 | 0 | не выросло |
| Hard gate failures | 0 | 0 | не выросло |
| Verdict | PASS 6 / PASS_WITH_NOTES 5 | PASS 7 / PASS_WITH_NOTES 4 | не хуже |
| Over-handoff | 20/21 = 0.952 | 20/21 = 0.952 | не вырос |
| P0 hard fail | 0 | 0 | P0 цел |
| Brand leak | 0 | 0 | бренд цел |

Trace стража:

| Trace | OFF | ON v2 |
|---|---:|---:|
| `assumed_scope_guard_turns` | 0 | 21 |
| `assumed_scope_guard_actions.pass` | 0 | 13 |
| `assumed_scope_guard_actions.skipped_non_self_route` | 0 | 7 |
| `assumed_scope_guard_actions.unknown` | 0 | 1 |
| `assumed_scope_guard_actions.skipped_p0_or_risk` | 0 | 0 |
| `asserted_assumed_slot_count` | 0 | 0 |

Главный вывод:

- прежнее массовое `skipped_p0_or_risk` исчезло;
- страж теперь реально доходит до проверки низкорисковых ходов;
- в динамическом микро-наборе модель не дала текстов, где нужно было переписать ответ через `reask_assumed_parameter`;
- поэтому POS-срабатывание дополнительно проверено прямым smoke-кейсом ниже.

## Direct-smoke POS/NEG/P0

Файл:

- `runs/tz119_assumed_scope_guard_20260616_micro/direct_guard_smoke_v2.json`

Результаты:

1. POS: есть неподтверждённые `grade=4`, `format=онлайн`, metadata `direct_path_model_p0` низкого риска.
   - было утверждение цены под 4 класс;
   - страж вернул `reask_assumed_parameter`;
   - цена `29 750 ₽` удалена из ответа;
   - маршрут не повышен до менеджера.

2. NEG: класс подтверждён клиентской цитатой.
   - страж вернул `pass`;
   - текст не изменён;
   - лишнего переспроса нет.

3. P0: реальный высокий риск.
   - страж вернул `skipped_p0_or_risk`;
   - P0-маршрут `manager_only` сохранён.

## Смысловая проверка микро-набора

Что хорошо:

- `over_handoff` не вырос.
- P0 и бренд не сломались.
- `skipped_p0_or_risk` больше не маскирует весь профиль `MODEL_P0=1`.
- Кривая фраза `Для данные ребёнка` больше не воспроизводится на sanitizer-тесте.
- Переспрос `про курс` после сужения output-guard больше не появляется.

Остаточные замечания:

- Динамический симулятор не умеет напрямую подкладывать боту «CRM знает класс, но клиент его не называл», поэтому real POS-срабатывание стража доказано direct-smoke и unit-тестом, а не живым многоходовым симулятором.
- В `on_micro_v2` `gain_fact_p14_foton_online_price_ellipsis` остаётся `PASS_WITH_NOTES`: бот осторожно просит уточнить курс/класс, полезность ниже идеала.
- В `gain_fact_p15_unpk_grade_ellipsis` остаётся `PASS_WITH_NOTES`: бот отвечает по 10 классу, но всё ещё просит уточнить тему/предмет.
- Это не блокер для ТЗ-119, но важно для отдельного трека ellipsis/usefulness.

## Вердикт

`formal_pass`: кодовый фикс и тесты прошли.

`semantic_pass: PASS_WITH_NOTES`: микро-замер показывает, что флаг больше не самоотключается, верхние метрики не ухудшились, P0/бренд целы. Но динамический микро-набор не является полной финальной приёмкой и не доказывает все POS-кейсы без direct-smoke.

Рекомендация:

- отдать этот отчёт на регрейд Claude #1;
- если регрейд принимает, следующим шагом можно запускать финальную полную приёмку на больших сетах;
- отдельно завести трек улучшения ellipsis-полезности по `p14/p15`, не смешивая его с ТЗ-119.

