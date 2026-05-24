# ТЗ: диалоговая память уровня веб-ассистента и skill для классов ошибок

Дата: 2026-05-23

Статус: `draft_ready_for_review`

Назначение: следующий этап доработок Telegram-ботов Фотона и УНПК МФТИ после блока `TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`.

Главная идея: перестать чинить отдельные ответы точечными патчами и построить слой, который делает бота устойчивым в многоходовом диалоге: он помнит уже сказанное, понимает последний прямой вопрос, не переспросит известное, держит один бренд, ведет клиента к следующему шагу и превращает каждую ошибку тестов/пилота в класс проблемы.

## 1. Контекст и диагноз

Предыдущий этап дал важную инфраструктуру:

- единый журнал пилота и daily report;
- feedback import;
- review queues;
- answer quality rewriter;
- `polish_sales`-режим под feature flag;
- динамические тесты `v8_targeted16`;
- статичные проверки `v6/v5`;
- P0/brand/fact guards;
- отдельные smoke-прогоны с `0 FAIL` по узким наборам.

Но ключевая проблема осталась: бот все еще может терять контекст на 2-3 ходе диалога, отвечать рядом с вопросом, звучать шаблонно, лишний раз отдавать менеджеру или не делать понятный следующий шаг.

Почему это происходит:

1. Бот получает недостаточно структурированное состояние диалога.
2. История реплик не превращается в устойчивые слоты: класс, предмет, формат, цель, вопрос, обещания бота.
3. Rewriter пытается чинить уже слабый первый ответ.
4. Targeted-тесты постепенно стали набором для настройки, а не честным финальным замером.
5. Ошибки тестов пока фиксируются как отдельные случаи, а не как классы проблем.

Метафора "память уровня веб-версии ChatGPT" означает не копирование архитектуры OpenAI/Anthropic, а внедрение тех же принципов на нашем уровне:

- хранить недавние реплики;
- хранить краткое состояние диалога;
- явно помнить, что уже известно;
- явно помнить, на какой вопрос еще не ответили;
- не передавать модели лишний шум;
- не давать модели менять бренд или придумывать факты;
- иметь отдельный слой проверки качества ответа.

## 2. Цель этапа

Сделать Telegram-ботов более похожими на живого внимательного консультанта, не ослабляя безопасность:

- бот помнит уже сказанные класс, предмет, формат, цель, город, интерес;
- бот отвечает на последний прямой вопрос клиента;
- бот не спрашивает повторно известные данные;
- бот не забывает, что сам обещал или спросил;
- бот делает один конкретный следующий шаг;
- бот не уходит к менеджеру там, где есть подтвержденный безопасный факт;
- бот не выдумывает конкретику, если факта нет;
- P0, бренд-разделение и запрет неподтвержденных обещаний остаются жесткими gate-правилами;
- каждый FAIL/PASS_WITH_NOTES превращается в запись о классе проблемы и план исправления.

## 3. Не делать в этом этапе

Не входит в этот этап:

- live-write в AMO, CRM или Tallanto;
- запись в `stable_runtime`;
- ASR;
- Resolve+Analyze;
- полный v8 как первый шаг;
- кросс-сессионная долговременная память клиента между разными днями;
- хранение raw PII в prompt или открытых отчетах;
- fine-tuning модели;
- большой SaaS-рефакторинг;
- перенос ботов на M1 Pro;
- замена P0-логики более "человечной" формулировкой.

Важно: сначала делаем память внутри текущей Telegram-сессии. Кросс-сессионная память клиента по AMO/Tallanto/телефонии/email - отдельный этап с отдельным P0/privacy-решением.

## 4. Источники правды перед реализацией

Перед началом реализации прочитать:

1. `AGENTS.md`
2. `docs/CURRENT_STATE.md`
3. `docs/DECISIONS_LOG.md`
4. `docs/ROADMAP.md`
5. `docs/RUNBOOK.md`
6. `docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`
7. последние audit packs:
   - `audits/_inbox/answer_quality_llm_rewriter_smoke2_*`
   - `audits/_inbox/telegram_pilot_journal_dialogue_strategy_20260523_021602`
   - `audits/_inbox/telegram_targeted16_static_v6_v5_review_20260523`
8. актуальные v8/v6/v5 файлы:
   - `/Users/dmitrijfabarisov/Claude Projects/Foton/v8_dynamic_sim_2026-05-22/v8_targeted16_2026-05-22.jsonl`
   - `/Users/dmitrijfabarisov/Claude Projects/Foton/v8_dynamic_sim_2026-05-22/MEGA_autonomy_tests_v6_2026-05-22.jsonl`
   - `/Users/dmitrijfabarisov/Claude Projects/Foton/v8_dynamic_sim_2026-05-22/MEGA_multitopic_batch_v5_2026-05-22.jsonl`
9. текущие файлы генерации ответа:
   - `src/mango_mvp/channels/telegram_pilot_context_builder.py`
   - `src/mango_mvp/channels/subscription_llm.py`
   - `src/mango_mvp/channels/answer_quality_rewriter.py`
   - `src/mango_mvp/channels/few_shot_reference.py`
   - `scripts/run_telegram_dynamic_client_sim.py`
   - `scripts/run_telegram_public_pilot_bots.py`

Чат использовать только как дополнительный контекст.

## 5. Блок A. Skill `bot-failure-class-review`

### 5.1. Зачем

Сейчас, когда тест падает, есть риск чинить конкретный текст или конкретный кейс. Это дает временное улучшение, но не лечит класс проблемы.

Нужен отдельный Codex skill, который при каждом FAIL/PASS_WITH_NOTES заставляет смотреть шире:

- что именно сломалось;
- какой это класс проблемы;
- есть ли похожие случаи;
- где корень: память, факт, route, prompt, rewriter, тест, судья, бизнес-правило;
- какой системный фикс нужен;
- какой regression test/gate/checklist нужно добавить.

### 5.2. Где создать

Создать новый skill:

```text
/Users/dmitrijfabarisov/.codex/skills/bot-failure-class-review/SKILL.md
```

Опциональные ресурсы:

```text
/Users/dmitrijfabarisov/.codex/skills/bot-failure-class-review/references/failure_class_record_template.md
/Users/dmitrijfabarisov/.codex/skills/bot-failure-class-review/references/review_rubric.md
```

Не создавать лишние README/CHANGELOG внутри skill.

### 5.3. Когда использовать

Skill должен срабатывать при задачах:

- "разбери FAIL";
- "почему тест упал";
- "что означает PASS_WITH_NOTES";
- "бот опять ответил шаблонно";
- "прогони v8/v6/v5 и объясни классы проблем";
- "что чинить после тестов";
- "преврати ошибку бота в системное правило".

### 5.4. Workflow skill

Для каждого проблемного кейса:

1. Прочитать полный transcript, verdict, route, flags, used facts.
2. Выписать симптом одним предложением.
3. Определить класс проблемы.
4. Проверить, единичный ли это случай или повторяемый класс.
5. Найти 2-5 похожих кейсов в текущем audit pack, если возможно.
6. Сформулировать root cause:
   - нет памяти слотов;
   - не распознан прямой вопрос;
   - нет бизнес-правила;
   - факт есть, но не выбран;
   - факт отсутствует;
   - тест устарел;
   - судья ошибся;
   - rewriter ухудшил ответ;
   - safety guard слишком строгий;
   - P0/brand guard сработал верно.
7. Предложить решение на уровне класса, а не конкретной фразы.
8. Указать, что добавить:
   - тест;
   - semantic gate;
   - rule в prompt/context builder;
   - правку KB;
   - ручной контроль;
   - уточнение у Дмитрия/РОПа.
9. Записать результат в `failure_classes.md` текущего audit pack.

### 5.5. Стартовый список классов проблем

Обязательные классы:

- `ignored_question` - бот не ответил на прямой вопрос;
- `context_loss` - бот забыл класс/предмет/формат/цель;
- `reasked_known_data` - бот спросил то, что уже известно;
- `templated_opening` - шаблонное начало;
- `over_handoff` - менеджер там, где можно было ответить фактом;
- `weak_next_step` - нет понятного следующего шага;
- `single_topic_answer_to_multitopic_question` - ответил только на одну часть составного вопроса;
- `missing_next_step` - ответ есть, но клиент не понимает, что делать дальше;
- `assumed_unstated_need` - бот приписал клиенту цель/потребность;
- `kb_voice` - ответ звучит как выдержка из базы, не как консультант;
- `commitment_drift` - бот забыл свое обещание/вопрос;
- `price_fix_question` - нет сильного сценария "как зафиксировать цену";
- `fact_selection_wrong_scope` - выбран факт не того класса/продукта/формата;
- `stale_test_expectation` - тест ждет старую политику;
- `judge_issue` - судья ошибся;
- `business_rule_missing` - нужна позиция Дмитрия/РОПа.

### 5.6. Артефакты skill

После каждого значимого прогона должен появляться:

```text
audits/_inbox/<run>/failure_classes.md
```

И общий реестр:

```text
docs/BOT_FAILURE_CLASSES_REGISTRY.md
```

Реестр должен хранить:

- id класса;
- описание;
- первый найденный пример;
- повторяемость;
- root cause;
- статус: `open`, `in_progress`, `fixed`, `accepted_risk`, `test_issue`;
- связанный тест или gate;
- где исправлено.

## 6. Блок B. DialogueMemory

### 6.1. Зачем

Это главный структурный слой этапа. Он должен дать боту рабочую память внутри текущего диалога.

Не просто "добавить больше истории в prompt", а создать объект состояния, который обновляется после каждой реплики и подается генератору ответа в сжатом виде.

### 6.2. Минимальная схема

Создать модуль:

```text
src/mango_mvp/channels/dialogue_memory.py
```

Минимальная структура:

```text
DialogueMemory:
  schema_version
  session_id
  active_brand
  turns
  known_slots
  open_question
  answered_questions
  last_bot_commitments
  sales_stage
  risk_flags
  handoff_state
  fact_refs
  route_history
  updated_at
```

### 6.3. Поля

`active_brand`:

- задается каналом: `foton` или `unpk`;
- не может быть изменен репликой клиента;
- не может быть изменен CRM/Tallanto;
- если клиент упоминает другой бренд, это не смена бренда, а сигнал `cross_brand_mention`.

`turns`:

- последние 6-10 реплик;
- хранить роль: `client` / `bot` / `system_test`;
- в prompt передавать только последние 2-4 реплики плюс summary;
- в открытые отчеты маскировать PII.

`known_slots`:

- `grade`;
- `subject`;
- `format`;
- `goal`;
- `city_or_location`;
- `course_type`;
- `camp_shift`;
- `payment_interest`;
- `trial_interest`;
- `parent_or_student`;
- `client_known_from_crm`;
- `student_known_from_crm`;
- источник слота: `user_text`, `crm_readonly`, `tallanto_readonly`, `test_mode`, `bot_inference`;
- confidence.

`open_question`:

- последний прямой вопрос клиента, на который бот еще не ответил;
- тип: price, schedule, trial, installment, address, camp, platform, tax, matkap, off_topic, p0, other;
- обязательный вход для answer-first проверки.

`last_bot_commitments`:

- что бот обещал: "передам менеджеру", "менеджер проверит", "подберем вариант", "уточним";
- запрещено обещать конкретную дату/время связи без подтвержденного факта;
- если обещание требует действия человека, должен быть handoff-event/flag.

`sales_stage`:

- `cold`;
- `interest`;
- `qualification`;
- `offer`;
- `objection`;
- `closing`;
- `handoff_required`.

`risk_flags`:

- refund;
- complaint;
- legal_threat;
- payment_dispute;
- contract_claim;
- pii_request;
- cross_brand_mention;
- unsupported_promise_risk;
- off_topic;

Если есть P0-флаг, продажная логика и polish отключаются.

`handoff_state`:

- `none`;
- `suggested`;
- `required`;
- `completed_event_created`;

### 6.4. Правила обновления памяти

Перед генерацией ответа:

1. Загрузить память по `brand + chat/session`.
2. Добавить новую реплику клиента.
3. Извлечь слоты детерминированно.
4. Определить прямой вопрос.
5. Обновить risk flags.
6. Сформировать `dialogue_memory_view` для prompt.

После генерации и post-filter:

1. Добавить ответ бота в turns.
2. Отметить, закрыт ли `open_question`.
3. Добавить used fact refs.
4. Записать route и flags.
5. Записать commitments.
6. Обновить sales_stage.
7. Сохранить snapshot в journal.

### 6.5. Что нельзя делать

- Не давать LLM напрямую менять память.
- Не хранить raw PII в открытых отчетах.
- Не переносить memory между брендами.
- Не использовать память для обхода P0.
- Не считать CRM-факт клиентским фактом автоматически.
- Не использовать старую историю клиента как разрешение ответить на спорную тему.

### 6.6. Persistence

На первом шаге:

- для симуляторов и unit-тестов допускается in-memory store;
- для Telegram-пилота сохранять snapshot памяти в существующий `TelegramPilotSQLiteStore`;
- не создавать новый источник правды, если можно расширить текущий store.

Нужные таблицы/поля:

- `dialogue_memory_snapshot`;
- `dialogue_memory_events`;
- `memory_schema_version`;
- `known_slots_json`;
- `open_question_json`;
- `commitments_json`;
- `sales_stage`;
- `risk_flags_json`.

## 7. Блок C. Memory-aware context builder

### 7.1. Цель

`telegram_pilot_context_builder` должен отдавать генератору не только KB/CRM context, но и ясное состояние диалога.

### 7.2. Новый объект в prompt context

Добавить:

```text
dialogue_memory_view:
  active_brand
  recent_turns
  known_slots
  open_question
  last_bot_commitments
  sales_stage
  risk_flags
  already_asked
  facts_already_used
  do_not_ask_again
  next_best_action_hint
```

### 7.3. Правила prompt

В prompt для генерации ответа добавить порядок:

1. Сначала ответь на `open_question`, если это безопасно.
2. Не спрашивай поля из `known_slots`.
3. Если факта нет, дай честный частичный ответ без цифр/дат/обещаний.
4. Задай один следующий вопрос или предложи один следующий шаг.
5. Не меняй active_brand.
6. Не делай P0-темы более "теплыми" за счет обещаний или признания вины.

### 7.4. Защита от шаблона

Context builder должен передавать разнообразные warm few-shot примеры только если:

- бренд совпадает;
- тема совпадает;
- required facts доступны;
- пример не содержит P0;
- пример не требует отсутствующих слотов;
- пример не содержит устаревшие цены/сроки.

## 8. Блок D. Open-question / answer-first gate

### 8.1. Зачем

Главная проблема тестов - `ignored_question`: клиент спрашивает одно, бот отвечает рядом.

Нужен отдельный gate:

- понять последний прямой вопрос;
- проверить, закрывает ли ответ этот вопрос;
- если не закрывает, переписать или заблокировать как `needs_review`.

### 8.2. Правила

Если вопрос безопасный и факт есть:

- ответ должен содержать прямой ответ в первых 1-2 предложениях.

Если факт частичный:

- ответ должен сказать, что известно;
- честно сказать, что требует проверки;
- задать один уточняющий вопрос.

Если P0:

- не отвечать по сути спорной темы;
- передать менеджеру;
- не собирать PII;
- не извиняться от лица компании;
- не обещать исход.

Если вопрос составной:

- ответить на все безопасные подпункты;
- P0-подпункт переводит весь вопрос в handoff;
- безопасные части можно кратко закрыть, но без спорной конкретики.

### 8.3. Детектор

Сначала детерминированный:

- вопросительные слова;
- `?`;
- ключевые intent-слова;
- короткие клиентские фразы: "а цена?", "как оплатить?", "куда ехать?", "вы кто?", "можно зафиксировать?".

LLM-детектор можно добавить позже под flag, но он не должен сам принимать финальное решение по P0.

## 9. Блок E. Бизнес-правило "зафиксировать цену / оформить по текущей"

### 9.1. Почему нужно

В тестах и живых диалогах часто возникает вопрос:

- "как зафиксировать цену?";
- "что нужно, чтобы оформить?";
- "можно записаться по текущей?";
- "что сделать, пока цена не выросла?".

Без явного правила бот либо слишком осторожничает, либо обещает действие, которого сам не может выполнить.

### 9.2. Черновик правила

Если бренд, продукт, класс/формат и текущая цена подтверждены:

- бот может назвать текущую цену;
- бот может сказать, что цена скоро повысится, без точной даты, если это подтвержденная политика;
- бот может предложить передать заявку менеджеру, чтобы оформить по текущим условиям;
- бот не может обещать место, бронь, договор, оплату, скидку или запись без проверки менеджера/системы.

Если не хватает класса/предмета/формата:

- бот задает один вопрос, который реально нужен для оформления.

Если клиент спрашивает "что от меня нужно":

- бот кратко просит недостающий безопасный слот;
- если все слоты есть, говорит, что передаст менеджеру заявку с уже собранными данными.

### 9.3. Что уточнить у Дмитрия/РОПа

Перед включением в более автономный режим нужно подтвердить:

- что именно считается "заявкой";
- можно ли боту писать "передам заявку менеджеру";
- какие поля нужны менеджеру для первого оформления;
- можно ли писать "оформить по текущим условиям" без обещания брони;
- есть ли ситуации, где цена фиксируется только после оплаты.

До уточнения использовать осторожную формулу:

```text
Могу передать менеджеру запрос на оформление по текущим условиям. Чтобы он сразу сориентировался, подскажите ...
```

## 10. Блок F. LLM-rewriter: только как страховка

### 10.1. Текущее решение

LLM-rewriter остается под feature flag.

Он не должен быть главным источником качества. Сначала надо улучшить:

- DialogueMemory;
- open_question;
- known_slots;
- first-pass prompt;
- few-shot selection.

### 10.2. Когда включать

Только по триггерам:

- ответ не закрывает прямой вопрос;
- reasked_known_data;
- over_handoff при наличии факта;
- weak_next_step;
- templated_opening;
- kb_voice;
- single_topic_answer_to_multitopic_question.

Не включать:

- P0;
- legal/refund/complaint/payment dispute;
- cross-brand risk;
- если факты конфликтуют;
- если ответ уже прошел semantic checks.

### 10.3. После rewrite

Обязательно повторить:

- P0 guard;
- brand guard;
- fact guard;
- unsupported promise guard;
- no PII/debug leak;
- route consistency.

Если rewrite ухудшил безопасность - откат к исходному безопасному ответу или manager_only.

## 11. Блок G. Honest holdout

### 11.1. Проблема

`v8_targeted16` уже стал dev-набором. На нем можно быстро видеть, стало ли лучше, но нельзя считать его честной финальной оценкой.

### 11.2. Что создать

Создать закрытый holdout:

```text
product_data/bot_eval/holdout_dialogues_20260523/
```

Минимум:

- 20 диалогов Фотон;
- 20 диалогов УНПК;
- 5-10 cross-brand/off-topic/adversarial;
- 5-10 P0;
- 5-10 многоходовых с "запомни класс/предмет/формат";
- 5-10 по "зафиксировать цену / оформить".

Каждый кейс пометить:

```text
do_not_tune_against: true
```

### 11.3. Правило

Нельзя подгонять prompt/regex/rewriter под конкретные holdout-кейсы.

Если кейс использован для настройки - удалить его из holdout и заменить новым.

## 12. Блок H. Тесты

### 12.1. Unit-тесты DialogueMemory

Добавить тесты:

- извлечение класса;
- извлечение предмета;
- извлечение формата;
- не перезаписывать brand;
- не спрашивать известный слот;
- open_question ставится и закрывается;
- commitments сохраняются;
- P0 замораживает sales_stage;
- PII маскируется в отчетах.

### 12.2. Integration-тесты context builder

Проверить:

- `dialogue_memory_view` входит в context;
- prompt получает known_slots;
- prompt получает open_question;
- `do_not_ask_again` формируется;
- facts_already_used не ломает brand isolation.

### 12.3. Rewriter/gate тесты

Проверить:

- answer-first gate ловит ignored_question;
- multi-topic gate ловит один ответ на два вопроса;
- over_handoff gate ловит manager-only при наличии факта;
- rewrite не срабатывает на P0;
- rewrite проходит повторные guards.

### 12.4. Dynamic tests

Порядок:

1. малый targeted smoke 4-6 кейсов;
2. `v8_targeted16`;
3. статичные `v6/v5`;
4. честный holdout;
5. полный v8 только отдельным следующим этапом.

## 13. Метрики acceptance

### 13.1. Hard gates

Должно быть:

- P0 safety: 100%;
- brand purity: 100%;
- no unsupported price/date/discount/seat promise: 100%;
- no PII/debug/source leak: 100%;
- no cross-brand conditions in client answers: 100%.

Любое нарушение hard gate блокирует этап.

### 13.2. Quality targets на targeted

Цели:

- `answered_direct_question_rate >= 80%`;
- `no_reask_known_data_rate >= 90%`;
- `commitment_kept_rate >= 90%`;
- `over_handoff_rate` ниже baseline минимум на 50%;
- `weak_next_step_rate` ниже baseline минимум на 50%;
- `templated_opening_rate` ниже baseline минимум на 30%.

### 13.3. Quality targets на holdout

Цели:

- улучшение не хуже, чем на targeted, с допустимым разрывом;
- если targeted сильно лучше holdout, считать это подгонкой;
- human tone использовать как индикатор, но не как единственный gate;
- финальный semantic verdict: минимум `PASS_WITH_NOTES`, без блокеров.

## 14. Audit pack

После реализации создать:

```text
audits/_inbox/dialogue_memory_failure_skills_YYYYMMDD_HHMMSS/
```

Внутри:

- `implementation_notes.md`;
- `changed_files.txt`;
- `test_output.txt`;
- `semantic_review.md`;
- `risk_review.md`;
- `backward_compatibility.md`;
- `failure_classes.md`;
- `memory_schema.md`;
- `targeted16_summary.md`;
- `holdout_summary.md`, если holdout запускался;
- `known_limitations.md`.

## 15. Порядок реализации

### Phase 0. Preflight

- Проверить `git status --short`.
- Не трогать чужие изменения.
- Зафиксировать baseline из последних smoke/audit packs.
- Создать/обновить `docs/BOT_FAILURE_CLASSES_REGISTRY.md`.

### Phase 1. Skill

- Создать `bot-failure-class-review`.
- Добавить компактный `SKILL.md`.
- Добавить шаблон записи класса проблемы.
- Проверить skill на 3 уже известных кейсах:
  - ignored_question;
  - over_handoff;
  - context_loss/reasked_known_data.

### Phase 2. DialogueMemory schema

- Создать `dialogue_memory.py`.
- Добавить dataclasses/schema.
- Добавить JSON serialization.
- Добавить in-memory store для тестов.
- Расширить SQLite store snapshot-ами памяти.

### Phase 3. Memory extraction/update

- Реализовать детерминированное извлечение слотов.
- Реализовать open_question detector.
- Реализовать commitments detector.
- Реализовать sales_stage update.
- Реализовать risk flag update.

### Phase 4. Context builder integration

- Передавать `dialogue_memory_view` в контекст генерации.
- Обновить prompt builder/provider: answer-first, do-not-ask-known, one-next-step.
- Не ломать существующий funnel layer, а продолжить его: funnel остается бизнес-логикой, memory становится состоянием диалога.

### Phase 5. Answer-first gate и бизнес-правило фиксации цены

- Добавить gate для прямого вопроса.
- Добавить rule для "зафиксировать цену / оформить".
- Добавить многоходовые regression-тесты.

### Phase 6. Rewriter integration

- Подключить memory-aware вход в deterministic/LLM rewriter.
- LLM-rewriter оставить выключенным по умолчанию, включать только флагом.
- После rewrite повторять guards.

### Phase 7. Tests and semantic review

- Запустить точечные unit/integration tests.
- Запустить targeted smoke.
- Запустить `v8_targeted16`.
- Запустить статичные `v6/v5`, если targeted не дал hard fail.
- Запустить holdout, если готов.
- Создать audit pack.
- Провести `business-semantic-review`.

### Phase 8. Docs

Если реализация прошла:

- обновить `docs/CURRENT_STATE.md`;
- обновить `docs/ROADMAP.md`;
- обновить `docs/RUNBOOK.md`;
- обновить `docs/DECISIONS_LOG.md`;
- добавить команды запуска новых тестов.

## 16. Риски

### 16.1. Память начнет врать

Риск: бот запомнит не то или сделает вывод, которого клиент не говорил.

Защита:

- source/confidence для каждого слота;
- не использовать LLM как прямой writer памяти;
- `assumed_unstated_need` как отдельный detector;
- audit по memory snapshots.

### 16.2. Память смешает бренды

Риск: УНПК-диалог подтянет факт Фотона.

Защита:

- active_brand immutable;
- brand-specific memory namespace;
- brand guard после каждого rewrite.

### 16.3. Rewriter станет новым источником ошибок

Риск: переписанный ответ станет теплым, но менее точным.

Защита:

- rewrite только по флагу;
- повторные guards;
- fail-closed;
- сравнение original vs rewritten в journal.

### 16.4. Подгонка под targeted16

Риск: targeted16 пройдет, живой бот останется слабым.

Защита:

- honest holdout;
- не тюнить по holdout;
- failure classes вместо точечных фраз.

### 16.5. Рост задержки

Риск: память + rewrite замедлят бот.

Защита:

- deterministic memory fast path;
- LLM-rewrite только по флагу;
- latency в journal;
- отдельный concurrency smoke.

## 17. Решения, которые нужны от Дмитрия/РОПа

Не блокируют Phase 1-4, но нужны для сильного Phase 5:

1. Что именно можно писать по "зафиксировать цену":
   - "передам заявку менеджеру";
   - "оформить по текущим условиям";
   - "цена фиксируется после оплаты";
   - "цена фиксируется после договора";
   - другое.
2. Какие слоты менеджеру реально нужны для первого действия:
   - класс;
   - предмет;
   - формат;
   - телефон;
   - имя родителя;
   - удобное время;
   - другое.
3. Можно ли боту в live-режиме писать "передам менеджеру" без фактического создания задачи в CRM.
4. Нужен ли отдельный статус "manager_handoff_event_created".
5. Кто владеет holdout-набором: Codex, Claude, Дмитрий, РОП.

## 18. Definition of Done

Этап считается выполненным, если:

- создан skill `bot-failure-class-review`;
- создан и используется `docs/BOT_FAILURE_CLASSES_REGISTRY.md`;
- DialogueMemory работает в симуляторе и Telegram-pilot context;
- бот не теряет класс/предмет/формат в многоходовом тесте;
- answer-first gate снижает `ignored_question`;
- бот не спрашивает известные данные;
- "зафиксировать цену" имеет безопасный сценарий;
- LLM-rewriter работает только под feature flag и после него повторяются guards;
- targeted16 улучшился без hard fails;
- статичные v6/v5 не показали реальный P0-регресс;
- создан audit pack;
- смысловая проверка дала `PASS` или `PASS_WITH_NOTES` без блокеров;
- документы состояния обновлены.

## 19. Первое минимальное внедрение

Если делать этап по частям, первый полезный срез:

1. `docs/BOT_FAILURE_CLASSES_REGISTRY.md`;
2. skill `bot-failure-class-review`;
3. `DialogueMemory` dataclass + in-memory store;
4. extraction `grade/subject/format/open_question`;
5. prompt/context injection;
6. запрет спрашивать известное;
7. unit tests;
8. малый dynamic smoke;
9. audit pack.

Это даст реальную пользу даже без полного LLM-rewriter.

## 20. Семантическая проверка ТЗ

Verdict: `PASS_WITH_NOTES`.

Что хорошо:

- этап лечит корневую проблему, а не отдельные фразы;
- память отделена от KB, CRM и long-term customer timeline;
- P0 и brand isolation остаются hard gate;
- targeted16 не считается финальной честной оценкой;
- ошибки тестов превращаются в реестр классов проблем;
- rewriter не становится главным источником качества.

Риски:

- без бизнес-решения по "зафиксировать цену" бот все равно будет осторожничать;
- если сделать memory через LLM-writer, появятся новые выдумки;
- если не создать holdout, можно снова подогнать систему под v8_targeted16;
- если skill будет слишком длинным, его перестанут реально использовать.

Следующий шаг:

- утвердить это ТЗ;
- затем реализовать Phase 0-4 как первый проверяемый срез;
- после первого targeted-прогона решить, нужен ли Phase 5-6 сразу или сначала уточнить бизнес-правило фиксации цены.
