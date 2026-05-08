# Sales Insight Knowledge Base Plan - 2026-05-07

## Цель

Построить не простой FAQ, а `AI Sales Playbook`: базу продающих реакций на основе реальных разговоров, клиентских цепочек и коммерческих outcomes.

Финальный продукт этого направления должен отвечать на вопросы:

- какие сигналы клиенты реально подают в процессе продажи;
- как менеджеры на них отвечают;
- какие ответы коррелируют с оплатой, продолжением обучения, возвратом или потерей;
- какие реакции стоит внедрить в скрипты, обучение отдела продаж, ROP-пакеты и будущего Telegram-бота;
- где менеджеры теряют клиентов из-за слабого next step, плохой отработки возражений или неправильного тона.

## Главный архитектурный принцип

Единица анализа - не отдельный звонок и не простая пара `вопрос -> ответ`.

Правильная единица анализа:

```text
клиентская цепочка -> скрытая стадия продажи -> sales moment -> реакция менеджера -> outcome
```

Где `sales moment` - это конкретный коммерчески значимый фрагмент коммуникации:

- вопрос клиента;
- возражение;
- сомнение;
- buying signal;
- сравнение с альтернативой;
- запрос цены/расписания/формата;
- жалоба или сервисный риск;
- сигнал действующего клиента;
- момент, где менеджер должен был зафиксировать следующий шаг.

## Почему нельзя анализировать 60k+ звонков линейно

У нас больше 60 тысяч звонков, но полезность убывает:

1. Первые 500-1000 содержательных звонков дадут большую часть частых вопросов и возражений.
2. Следующие несколько тысяч расширят покрытие по продуктам, менеджерам, источникам и редким сценариям.
3. Дальше новая польза будет приходить в основном из редких сегментов, провальных цепочек и unusual outcomes.

Поэтому стратегия должна быть `progressive + utility-driven`, а не `все звонки подряд через дорогую модель`.

## Клиентская цепочка как основа

Основная сущность: `client_identity`.

Ключи сопоставления:

- нормализованный телефон;
- AMO contact id;
- AMO lead ids;
- Tallanto student/contact ids;
- Telegram dialog/contact when available;
- email when available;
- fuzzy-доказательства только как низкоуверенные связи.

Для каждого клиента строится timeline:

```text
client_identity
  -> call events
  -> AMO deals/stage changes/tasks/notes
  -> Tallanto payments/enrollments/groups
  -> Telegram/email events where matched
  -> derived hidden sales stages
  -> outcomes by time window
```

Важно: если у одного телефона несколько сделок, несколько учеников или несколько периодов обучения, это не ошибка. Это отдельная часть модели: `client_identity -> opportunity/thread`.

## Скрытые стадии продажи

Нам нужны не только CRM-статусы, а inferred stages из фактической коммуникации:

- `new_request`: новая заявка/первый контакт;
- `discovery`: выяснение класса, предмета, цели, формата;
- `offer_explained`: менеджер объяснил продукт/программу;
- `price_discussion`: обсуждение цены, скидки, рассрочки;
- `objection_handling`: клиент сомневается или возражает;
- `materials_sent`: договорились отправить материалы/ссылку/КП;
- `decision_wait`: клиент думает, советуется, обещает вернуться;
- `payment_intent`: просит ссылку, счет, места, старт;
- `paid_or_enrolled`: оплатил/попал в Tallanto как ученик;
- `existing_client_service`: действующий клиент, сервисный вопрос;
- `reactivation`: попытка вернуть старого/остывшего клиента;
- `lost_or_stalled`: явный отказ или длинная пауза без следующего шага.

Каждый sales moment должен знать стадию цепочки, иначе один и тот же ответ может быть хорошим в одном контексте и плохим в другом.

## Данные, которые нужно получить

### 1. Client Chain Layer

Поля:

- `client_key`
- `normalized_phone`
- `amo_contact_ids`
- `amo_lead_ids`
- `tallanto_ids`
- `first_seen_at`
- `last_seen_at`
- `touch_count`
- `manager_count`
- `deal_count`
- `has_payment`
- `has_active_learning`
- `final_known_outcome`
- `identity_confidence`
- `identity_conflict_flags`

### 2. Chain Event Layer

Поля:

- `event_id`
- `client_key`
- `opportunity_key`
- `event_type`: `call`, `amo_stage`, `amo_task`, `payment`, `enrollment`, `telegram`, `email`
- `event_at`
- `source_system`
- `source_id`
- `manager`
- `raw_status`
- `derived_stage_before`
- `derived_stage_after`
- `evidence`

### 3. Sales Moment Layer

Поля:

- `moment_id`
- `client_key`
- `opportunity_key`
- `call_id`
- `source_filename`
- `timestamp_start`
- `timestamp_end`
- `client_signal_type`
- `client_signal_category`
- `client_quote`
- `manager_response_quote`
- `topic`
- `product`
- `subject`
- `grade`
- `format`
- `price_or_schedule_context`
- `hidden_stage`
- `moment_importance`
- `extraction_confidence`

### 4. Response Quality Layer

Оценивать не одним баллом, а рубрикой:

- `factual_correctness`
- `completeness`
- `persuasiveness`
- `personalization`
- `objection_handling`
- `next_step_clarity`
- `empathy_tone`
- `sales_discipline`
- `risk_flags`
- `overall_quality_score`
- `what_manager_did_well`
- `what_manager_missed`
- `ideal_reaction`
- `ideal_answer_example`
- `avoid_using_when`

### 5. Outcome Layer

Outcome должен быть привязан ко времени, а не просто к финальному статусу сделки.

Поля:

- `outcome_1d`
- `outcome_7d`
- `outcome_14d`
- `outcome_30d`
- `outcome_60d`
- `eventual_outcome`
- `paid_amount`
- `payment_date`
- `enrollment_date`
- `lost_reason`
- `is_existing_client`
- `is_reopen_candidate`
- `next_touch_after_moment_at`
- `time_to_next_touch_hours`
- `time_to_payment_hours`
- `outcome_confidence`

## Категории клиентских сигналов

Стартовая таксономия:

- `price_question`
- `price_objection`
- `discount_or_installment_question`
- `schedule_question`
- `format_question_online_offline`
- `location_question`
- `teacher_question`
- `program_question`
- `level_fit_question`
- `exam_or_olympiad_goal`
- `trust_question`
- `competitor_comparison`
- `child_motivation_concern`
- `parent_decision_delay`
- `spouse_or_family_approval`
- `not_relevant_now`
- `already_learning_elsewhere`
- `ready_to_pay`
- `materials_request`
- `callback_request`
- `technical_or_access_issue`
- `existing_client_progress`
- `complaint_or_service_risk`

Таксономия должна быть версионированной. Новые категории добавляются только если они часто повторяются или коммерчески важны.

## Стратегия выборки при 60k+ звонках

### Wave 0. Data readiness

Цель: проверить, что можно строить client chains и outcome links.

Объем: без LLM или с минимальным LLM, только метаданные и готовый Analyze.

Результат:

- количество уникальных телефонов;
- распределение touch_count;
- сколько клиентов имеют AMO/Tallanto outcome;
- сколько клиентов имеют несколько сделок;
- сколько звонков содержательные по `call_type`;
- сколько звонков являются non-conversation и не должны идти в дорогой анализ.

### Wave 1. Stratified pilot

Объем: 300-500 клиентских цепочек, не звонков.

Выборка должна покрывать:

- paid / lost / existing client / in work;
- разные менеджеры;
- разные продукты/предметы/классы;
- разные источники/UTM where available;
- разные длины цепочек: 1 звонок, 2-3 касания, 4+ касания;
- свежие и исторические периоды;
- high-value и спорные кейсы.

Цель: проверить, что extractor реально выделяет sales moments и не превращает все в мусорный FAQ.

### Wave 2. High-utility expansion

Объем: 2k-5k клиентских цепочек.

Приоритеты:

- цепочки с известным outcome;
- цепочки с оплатой после нескольких касаний;
- цепочки с проигрышем после сильного интереса;
- клиенты с несколькими сделками;
- повторные клиенты;
- длинные разговоры с ценой/возражениями;
- разные менеджеры по одной и той же категории сигнала.

Цель: получить первые устойчивые паттерны и сравнения ответов.

### Wave 3. Novelty-driven coverage

Объем: до тех пор, пока появляется новая польза.

После каждой партии считать:

- `new_signal_category_rate`
- `new_topic_rate`
- `new_objection_pattern_rate`
- `new_answer_pattern_rate`
- `new_high_quality_answer_rate`
- `new_bad_pattern_rate`
- `outcome_linked_coverage`

Останавливать массовую обработку категории, если новизна падает ниже порога, например:

```text
new_answer_pattern_rate < 3-5% на 500 новых цепочек
и нет роста качества playbook items
```

### Wave 4. Rare and strategic cases

Отдельно добирать редкие, но важные сегменты:

- дорогие продукты;
- летние школы/лагеря;
- олимпиады;
- 10-11 классы;
- сильные price objections;
- жалобы;
- клиенты, которые вернулись после долгой паузы;
- топовые и слабые менеджеры;
- каналы/UTM с высокой или низкой конверсией.

## Utility Score для выбора следующей партии

Каждой клиентской цепочке дать приоритет:

```text
utility_score =
  known_outcome_weight
+ commercial_signal_weight
+ multi_touch_weight
+ multi_deal_weight
+ manager_diversity_weight
+ source_diversity_weight
+ recency_weight
+ high_value_product_weight
+ unresolved_or_lost_weight
+ novelty_gap_weight
- non_conversation_weight
- duplicate_pattern_penalty
```

Так мы не тратим дорогие модели на 20 тысяч одинаковых недозвонов и повторяющихся коротких звонков.

## Модельный каскад

### Bulk extraction

- Модель: `gpt-5.4-mini` или `gpt-5.4`
- Reasoning: `low/medium`
- Задача: извлечь sales moments, цитаты, категории, темы, next step.

### Quality scoring

- Модель: `gpt-5.5`
- Reasoning: минимум `medium`
- Задача: оценить качество ответа, что пропущено, идеальную реакцию.

### Hard cases

- Модель: `gpt-5.5`
- Reasoning: `high`
- Задача: длинные цепочки, противоречия outcome, спорные категории, высокоценные клиенты.

### Synthesis

- Модель: `gpt-5.5`
- Reasoning: `medium/high`
- Задача: объединить повторяющиеся моменты в playbook items и готовые ответы для РОПа/бота.

## Защита от ложных выводов

LLM-score не считать истиной. Он является экспертной оценкой, но outcome зависит от множества факторов.

Минимальные confounders:

- source/UTM;
- менеджер;
- продукт;
- предмет;
- класс;
- сезон;
- новый клиент или действующий;
- количество касаний до момента;
- скорость следующего контакта;
- цена/рассрочка;
- филиал/формат;
- наличие нескольких сделок.

Отчеты должны разделять:

- observed correlation;
- LLM recommendation;
- ROP-approved recommendation;
- production bot answer.

## Итоговый мини-продукт

Название: `AI Sales Playbook`.

Состав:

### 1. ROP workbook `.xlsx`

Листы:

- `Сигналы клиентов`
- `Лучшие ответы`
- `Ошибки менеджеров`
- `Возражения и реакции`
- `Скрытые стадии продажи`
- `Гипотезы и корреляции`
- `Что внедрить в скрипты`
- `Что дать TG-боту первым`

### 2. Machine-readable KB

Форматы:

- `sales_moments.jsonl`
- `client_chains.jsonl`
- `response_scores.jsonl`
- `playbook_items.jsonl`
- later: `parquet` or DB tables.

### 3. Bot-ready retrieval base

Каждый item:

- `intent/category`
- `allowed_context`
- `avoid_context`
- `ideal_answer`
- `short_answer`
- `diagnostic_questions`
- `recommended_next_step`
- `evidence_count`
- `positive_outcome_rate`
- `risk_flags`
- `approved_by_human`

### 4. ROP methodology `.md`

- как читать результаты;
- какие ответы внедрить;
- какие ошибки отслеживать;
- как валидировать новые playbook items;
- как не путать корреляцию с причинностью.

## План реализации

### Фаза 1. Data audit для insight-layer

1. Найти все DB, входящие в strict coverage v5.
2. Построить список всех звонков с `analysis_status=done` и `resolve_status in done/skipped`.
3. Посчитать распределение по `quality_flags.call_type`.
4. Посчитать уникальные телефоны и touch_count.
5. Найти телефоны с несколькими сделками/AMO ids/Tallanto ids.
6. Проверить, какие outcome-поля доступны из AMO/Tallanto snapshots.
7. Собрать `insight_data_readiness_report`.

### Фаза 2. Client Chain MVP

8. Сделать нормализатор phone identity.
9. Собрать `client_identity` по телефону.
10. Сгруппировать звонки в timeline по клиенту.
11. Подтянуть AMO/Tallanto context, где уже есть локальные snapshots/exports.
12. Ввести `opportunity_key` для случаев нескольких сделок на один телефон.
13. Посчитать hidden-stage candidates по цепочке.
14. Сохранить `client_chains_v1.jsonl` и audit `.xlsx`.

### Фаза 3. Outcome Linker MVP

15. Определить outcome taxonomy: paid, enrolled, existing_client, lost, stalled, in_work, reopened, unknown.
16. Привязать payment/enrollment/deal outcome к client/opportunity.
17. Считать outcome windows: 1/7/14/30/60 дней после звонка и после sales moment.
18. Помечать низкую уверенность при конфликте нескольких сделок/учеников.
19. Сохранить `outcome_links_v1.jsonl`.

### Фаза 4. Sales Moment Extractor Pilot

20. Выбрать 300-500 клиентских цепочек stratified sampling, не просто 500 звонков.
21. Для каждой цепочки подать модели компактный timeline + релевантные transcript snippets.
22. Извлечь sales moments с цитатами клиента и менеджера.
23. Проверить JSON schema и confidence.
24. Собрать ручной review pack на 50-100 моментов для проверки качества.
25. Исправить taxonomy/schema/prompts.

### Фаза 5. Quality Scoring Pilot

26. На извлеченных moments прогнать rubric scoring.
27. Для важных/сложных моментов использовать `gpt-5.5 medium/high`.
28. Сформировать `ideal_reaction` и `ideal_answer_example`.
29. Отделить `manager actually said` от `recommended ideal answer`.
30. Сравнить score с outcome windows.

### Фаза 6. High-utility Expansion

31. Посчитать utility_score для всех клиентских цепочек.
32. Выбрать 2k-5k наиболее полезных цепочек.
33. Запустить extraction/scoring каскадом.
34. После каждой партии считать novelty metrics.
35. Остановить или сузить категории, где новизна исчерпана.

### Фаза 7. Playbook Synthesis

36. Сгруппировать moments по category/topic/context.
37. Выделить лучшие реальные ответы менеджеров.
38. Сравнить с LLM ideal answers.
39. Сгенерировать playbook items с evidence и constraints.
40. Отдать РОПу xlsx на валидацию.
41. Пометить items как `approved`, `needs_edit`, `rejected`.

### Фаза 8. Bot-ready KB

42. Конвертировать approved playbook items в retrieval records.
43. Добавить allowed/avoid context.
44. Добавить короткие и длинные варианты ответа.
45. Добавить диагностические вопросы перед продажным ответом.
46. Подготовить bot prompt contract: как использовать KB, как не выдумывать, когда эскалировать менеджеру.
47. Сделать offline simulation на исторических client signals.

### Фаза 9. Sales Analytics и гипотезы

48. Посчитать факторы, связанные с outcome: next step, скорость follow-up, тип возражения, ответ на цену, менеджер, источник, stage.
49. Разделить корреляции на устойчивые и сомнительные.
50. Подготовить ROP action plan: что изменить в скриптах, обучении, контроле качества и TG-боте.

## Acceptance criteria первого этапа

Первый этап считается успешным, если есть:

- `client_chains_v1` хотя бы по всем телефонам из обработанных звонков;
- отчет по числу уникальных клиентов, касаний, outcome coverage и конфликтов identity;
- pilot extraction на 300-500 клиентских цепочках;
- не менее 500-1000 sales moments с валидной схемой;
- ручной review pack для РОПа;
- первые 30-50 playbook candidate items;
- понятный ответ: какие категории уже насыщены, а где нужно добирать данные.

## Первые файлы/модули, которые стоит добавить

Предлагаемый write scope для этого диалога:

```text
src/mango_mvp/insights/
  contracts.py
  phone_identity.py
  client_chains.py
  outcome_linker.py
  sampling.py
  sales_moments.py
  scoring.py
  playbook.py

scripts/
  build_insight_readiness_report.py
  build_client_chains.py
  extract_sales_moments_pilot.py
  score_sales_moments_pilot.py
  build_sales_playbook_pack.py

docs/
  SALES_INSIGHT_KNOWLEDGE_BASE_PLAN_2026-05-07.md
```

Не трогать в этом потоке:

- SaaS capture/productization modules второго диалога;
- AMO writeback;
- ASR/R+A runtime DB;
- live Mango API credentials;
- UI redesign files, если второй диалог ими владеет.

## Рекомендованный следующий шаг

Начать с `build_insight_readiness_report.py`:

1. Взять strict coverage v5 как источник DB list.
2. Просканировать только terminal analyzed calls.
3. Посчитать client-level statistics по нормализованным телефонам.
4. Посчитать распределение call_type/topic/outcome availability.
5. Сформировать `.json` + `.xlsx` readiness report.

После этого станет ясно, сколько реально клиентских цепочек, сколько из них имеют outcome, и какую первую stratified sample брать для extractor pilot.
