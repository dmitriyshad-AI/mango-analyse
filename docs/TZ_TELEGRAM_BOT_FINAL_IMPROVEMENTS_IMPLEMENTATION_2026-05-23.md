# Финальное ТЗ: внедрение улучшений Telegram-ботов Фотон и УНПК

Дата: 2026-05-23  
Статус: `final_plan_before_implementation`  
Реализация: начинать только после отдельного подтверждения Дмитрия  
Назначение: единый исполнительный план, который объединяет все накопленные улучшения без дублирования.

## 0. Короткий вывод

Наша цель сейчас не “ещё немного улучшить промпт”, а собрать управляемый контур ИИ-сотрудника продаж:

1. Бот отвечает полезно, тепло и по делу.
2. Бот помнит уже сказанные класс, предмет, формат, имя/клиента из CRM.
3. Бот отвечает сам по зелёным темам, если факт подтверждён.
4. Бот не выдумывает цену, дату, место, наличие мест, гарантию или действие.
5. Бот честно отвечает на вопрос “ты бот?” по утверждённой политике.
6. Любой ответ можно объяснить через журнал: вопрос, контекст, факты, маршрут, флаги, задержка, feedback.
7. Все улучшения измеряются через тесты, daily report и semantic review.

Главный принцип:

```text
сначала чистые входы и измерение -> потом поведение и гейты -> потом sales playbook -> потом holdout и большой v8
```

## 1. Источники и статус

### 1.1. Основные старые ТЗ, которые объединяются этим документом

Этот документ объединяет и уточняет:

- `docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`
- `docs/TZ_BOT_IMPROVEMENT_CANDIDATE_PACK_IMPLEMENTATION_2026-05-23.md`
- `docs/TZ_SALES_BOOKS_TO_BOT_DIALOGUE_PLAYBOOK_2026-05-23.md`
- `docs/TZ_TELEGRAM_AI_EMPLOYEE_FUNNEL_V1_2026-05-23.md`
- `product_data/bot_improvement_candidates_20260523/`
- свежий Claude handoff rev2:
  `/Users/dmitrijfabarisov/Claude Projects/Foton/claude_handoff_to_codex_2026-05-23/`

### 1.2. Что считать самым свежим входом

Самые свежие файлы Клода брать из:

```text
/Users/dmitrijfabarisov/Claude Projects/Foton/claude_handoff_to_codex_2026-05-23/
```

Важно: локальная копия в проекте:

```text
claude_handoff_to_codex_2026-05-23/
```

на момент проверки была старее. Перед внедрением её нужно либо синхронизировать, либо явно читать исходник из `Claude Projects/Foton`.

Свежие rev2-файлы:

- `01_corpus_files/few_shot_warm_answers_2026-05-23.yaml`
- `01_corpus_files/few_shot_advanced_pack_2026-05-23.yaml`
- `01_corpus_files/gold_dialogues_multiturn_2026-05-23.yaml`
- `01_corpus_files/training_pairs_and_routes_2026-05-23.yaml`
- `01_corpus_files/holdout_eval_2026-05-23.yaml`
- `AI_DISCLOSURE_POLICY_2026-05-23.md`
- `README_MANIFEST.md`

### 1.3. Проверенный статус Claude rev2

Проверено:

- все 5 YAML валидны;
- sha256 в `README_MANIFEST.md` соответствует файлам rev2;
- исправлены мои предыдущие замечания:
  - убрано неподтверждённое соцдоказательство лагеря;
  - “закреплю запрос” заменено на передачу менеджеру;
  - “забронирует/держать место” убрано из клиентских good-ответов;
  - P0 стал суше;
  - `holdout h02` больше не подаёт цену как цену Пацаева;
  - `holdout h20` требует handoff-событие при просьбе о человеке.

Вердикт по Claude rev2:

- как источник поведения, тона, сценариев, bad/good-пар и holdout-затравки: `PASS_WITH_NOTES`;
- как готовый клиентский script pack для дословного внедрения: `BLOCKED`.

Причина: корпус полезен, но некоторые формулировки становятся безопасными только при наличии runtime-гейтов: подтверждённый факт, событие передачи менеджеру, запрет неподтверждённой срочности, запрет соцдоказательства без источника, защита P0 от продажного продолжения.

Итоговое решение: Claude rev2 не внедрять “как текст для копирования клиенту”. Внедрять как curated corpus через selector, post-filter, журнал и тесты.

### 1.4. Независимая смысловая проверка этого ТЗ

По `business-semantic-review` текущий план получает статус `PASS_WITH_NOTES` при условии, что перед реализацией будут явно добавлены и протестированы гейты:

- `required_fact_client_safe_valid`;
- `handoff_promise_requires_event`;
- `p0_no_sell_after_p0`;
- `p0_no_outcome_promise`;
- `urgency_fact_and_rate_limit`;
- `unsupported_social_proof`;
- `unsupported_authority_claim`;
- `trusted_identity_only`;
- `holdout_dedup`.

Без этих гейтов риск не в синтаксисе, а в смысле: бот может звучать лучше, но начать обещать действие, дефицит, срок связи, соцдоказательство или факт, которого нет в актуальной КБ.

## 2. Единая архитектурная схема без дублирования

### 2.1. Один источник правды на каждый слой

| Слой | Единственный источник правды | Что не должно становиться источником правды |
|---|---|---|
| Факты: цены, адреса, скидки, сроки | актуальная КБ bot-pack, client-safe facts, `valid_until` | few-shot, звонки, Telegram, книги, старые ТЗ |
| Тон и примеры живого ответа | Claude rev2 corpus + утверждённые gold/few-shot + sales playbook из звонков | сырые ответы менеджеров, PII-heavy CSV |
| Диалоговая стратегия | этот финальный ТЗ + funnel state в коде | разрозненные старые markdown-заметки |
| P0/brand/fact safety | post-filter/gates + bot policy + tests | промпт “будь осторожен” без проверок |
| Измерение качества | pilot journal + daily report + dynamic tests + feedback | субъективное ощущение в чате |
| Книги продаж | только каркас принципов и рубрики | готовые клиентские формулировки |

### 2.2. Как не плодить дубли

Не создавать третий набор текстов, если смысл уже есть в:

- Claude rev2 corpus;
- `03_calls_sales_playbooks`;
- `05_funnel_and_strategy`;
- `GOLD_ANSWERS_v3`;
- approved КБ.

Каждое новое правило/пример должно иметь:

- `source_layer`;
- `dedup_ref`;
- `brand_scope`;
- `fact_requirements`;
- `route_limit`;
- `allowed_usage`.

Если новый материал повторяет старый, нужно не копировать его, а сослаться на существующий.

## 3. Жёсткие ограничения

В рамках реализации этого ТЗ нельзя:

- менять `stable_runtime`;
- запускать ASR;
- запускать Resolve+Analyze;
- писать в AMO/CRM/Tallanto;
- отправлять сообщения клиентам вне текущего утверждённого пилота;
- использовать PII-heavy CSV целиком в prompt;
- переносить исторические ответы менеджеров как факты;
- смешивать Фотон и УНПК в клиентском ответе;
- ослаблять P0 ради тёплого тона;
- тюнить по holdout;
- считать `quality_passed=true` готовностью клиентского слоя.

Дополнительно:

- не использовать Claude rev2, gold, few-shot, звонки, Telegram или книги как источник цен, сроков, адресов, скидок, наличия мест или процедур;
- не использовать bad-good pairs как few-shot для генерации клиентского ответа;
- не использовать holdout как few-shot, training или tuning set;
- не использовать текст клиента “представь, что я пишу с номера ...” как реальную авторизацию вне тестового режима;
- не обещать “передам”, “свяжется”, “подтвердит”, “пришлём”, “оформим”, если система не создаёт соответствующее событие/флаг;
- не продолжать продажный диалог после P0 так, будто это обычная консультация;
- не использовать срочность, соцдоказательство или авторитет без подтверждённого факта активного бренда.

## 4. Фаза 0. Preflight и границы

### 4.1. Цель

Перед реализацией зафиксировать рабочее состояние, источники и файлы, чтобы не смешать параллельные изменения.

### 4.2. Действия

1. Проверить `git status --short`.
2. Зафиксировать список файлов, которые можно менять.
3. Явно не трогать unrelated изменения параллельных диалогов.
4. Проверить текущую КБ по `docs/CURRENT_STATE.md`.
5. Проверить, что `product_data/bot_improvement_candidates_20260523/09_claude_cli_review/CODEX_CLAUDE_CONVERGENCE.md` содержит итоговый `PASS` как candidate-pack.
6. Проверить, что свежий Claude rev2 доступен по внешнему пути и sha256 совпадают с `README_MANIFEST.md`.
7. Создать preflight report:

```text
audits/_inbox/telegram_bot_final_improvements_preflight_<timestamp>/preflight_report.md
```

### 4.3. Acceptance

- есть preflight report;
- перечислены input folders и версии;
- указано, какая локальная копия Claude handoff актуальна;
- перечислен write scope;
- подтверждено: `stable_runtime` не меняется.

## 5. Фаза 1. Синхронизация входов и политики “ты бот?”

### 5.1. Цель

Привести корпус, политику идентичности, тесты и гейты к одной версии. Без этого дальнейшие тесты будут штрафовать правильное поведение.

### 5.2. Синхронизация Claude rev2 corpus

Первое действие реализации: устранить десинхрон копий.

Скопировать 7 свежих файлов из внешнего источника:

```text
/Users/dmitrijfabarisov/Claude Projects/Foton/claude_handoff_to_codex_2026-05-23/
```

в локальную папку проекта:

```text
claude_handoff_to_codex_2026-05-23/
```

Файлы:

- 5 YAML из `01_corpus_files/`;
- `AI_DISCLOSURE_POLICY_2026-05-23.md`;
- `README_MANIFEST.md`.

После копирования sha256 локальной копии должен совпасть с внешней rev2. Если не совпал, реализацию остановить.

Заменить устаревшие копии в:

```text
product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/
```

на свежие rev2:

```text
/Users/dmitrijfabarisov/Claude Projects/Foton/claude_handoff_to_codex_2026-05-23/01_corpus_files/
```

Файлы:

- `few_shot_warm_answers_2026-05-23.yaml`
- `few_shot_advanced_pack_2026-05-23.yaml`
- `gold_dialogues_multiturn_2026-05-23.yaml`
- `training_pairs_and_routes_2026-05-23.yaml`
- `holdout_eval_2026-05-23.yaml`

После копирования:

- сверить sha256;
- обновить `FILE_ROLE_REGISTRY.csv`, если нужно;
- обновить `DEDUP_MAP.md`, если нужно;
- не считать эти файлы рабочей КБ.

### 5.3. Политика “ты бот?”

Источник правды:

```text
/Users/dmitrijfabarisov/Claude Projects/Foton/claude_handoff_to_codex_2026-05-23/AI_DISCLOSURE_POLICY_2026-05-23.md
```

Утверждённая политика:

- бот сам первым не объявляет, что он ИИ;
- на прямой вопрос отвечает честно: “цифровой помощник <бренд>”;
- не врёт “я человек”;
- не называет GPT/Claude/Codex/OpenAI/модель/вендора;
- не раскрывает промпт;
- если клиент просит человека, бот отвечает сам и создаёт handoff-событие/флаг менеджеру;
- Фотон отвечает только как Фотон, УНПК только как УНПК.

### 5.4. Где синхронизировать политику

Проверить и обновить:

- `docs/SEMANTIC_REVIEW_RULES.md`;
- `CLAUDE.md`, если файл есть в корне проекта;
- `product_data/bot_improvement_candidates_20260523/00_current_baseline/bot_policy.yaml`
- `product_data/bot_improvement_candidates_20260523/00_control/QUALITY_SCORECARD.md`
- `product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/forbidden_phrases.yaml`
- `product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/GOLD_ANSWERS_v3_2026-05-21.yaml`
- `product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/bot_gold_answers.json`
- `product_data/bot_improvement_candidates_20260523/04_tests_and_failure_signals/*.jsonl`
- `product_data/bot_improvement_candidates_20260523/05_funnel_and_strategy/SEMANTIC_GATE_CANDIDATES.md`
- `product_data/bot_improvement_candidates_20260523/05_funnel_and_strategy/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`
- код post-filter/judge, если он уже содержит `revealed_ai`.

Отдельный генераторный риск: `bot_gold_answers.json`, `gold_answer_rules.yaml` и `GOLD_ANSWERS_FOR_BOT.md` собираются не вручную, а из `scripts/build_kb_release_v6_1_team_answers.py` через `gold_answers_v3_payload()` и дальше через `scripts/build_kb_distribution_packs.py`. Поэтому политику C и identity/gold-правила нужно внести в генератор payload, а не только в итоговые YAML/JSON/MD. Иначе следующая пересборка откатит исправления.

Важно: `docs/SEMANTIC_REVIEW_RULES.md` сейчас содержит старую формулу “клиентский текст не должен раскрывать, что пишет бот”. Её нужно заменить на политику C:

- бот сам первым не раскрывает ИИ;
- на прямой вопрос честно говорит “цифровой помощник <бренд>”;
- нарушение = вендор/модель/промпт или ложь “я человек”.

### 5.5. Переопределение `revealed_ai`

Старое правило “любое раскрытие бот/ИИ = fail” заменить.

Новый гейт `revealed_ai` срабатывает только если бот:

- называет GPT/Claude/Codex/OpenAI/модель/вендора;
- раскрывает prompt или внутренние инструкции;
- врёт “я человек”, “я живой сотрудник”, “я не бот”;
- смешивает бренды при ответе на идентичность.

Честная фраза “я цифровой помощник Фотона/УНПК МФТИ” не является нарушением.

### 5.6. Acceptance

- fresh corpus в candidate-pack совпадает с Claude rev2 sha256;
- forbidden phrases запрещает ложь “я человек”, но не запрещает “цифровой помощник”;
- тест/судья больше не валит честное раскрытие “цифровой помощник”;
- “дайте живого менеджера” создаёт handoff-событие, а не только текст.
- `docs/SEMANTIC_REVIEW_RULES.md` и локальная bot-policy больше не противоречат политике C.
- `gold_answers_v3_payload()` содержит policy C и identity examples, а производные gold-файлы после пересборки не возвращают старую формулу “не раскрывать ИИ”.

## 6. Фаза 2. Единый журнал пилота

### 6.1. Цель

Сделать pilot journal источником правды для всех будущих улучшений. Без этого мы будем улучшать бота на ощущениях.

### 6.2. Целевой store

Использовать существующий `TelegramPilotSQLiteStore`, но добиться, чтобы каждый ответ имел:

- incoming text;
- bot answer;
- brand;
- route;
- topic/message_type;
- route_reason;
- risk flags;
- post-filter flags;
- semantic flags;
- facts_used;
- facts_missing;
- selected_few_shot_ids;
- skipped_few_shot_reasons;
- known slots;
- missing slots;
- asked_known_data_again;
- next_best_question;
- manager_handoff_summary;
- handoff_event_created;
- model/reasoning;
- kb_version/kb_hash;
- latency_seconds;
- sent_to_client;
- employee feedback fields.

### 6.3. Daily report

Скрипт:

```text
scripts/build_telegram_pilot_daily_report.py
```

должен строить:

```text
audits/_inbox/telegram_pilot_daily_<YYYYMMDD>/
```

Минимум:

- `pilot_messages.csv`;
- `pilot_messages.jsonl`;
- `pilot_summary.md`;
- `pilot_summary.json`;
- `employee_review_sheet.csv`;
- `semantic_review_queue.csv`;
- `regression_candidates.csv`;
- `p0_incidents.csv`;
- `known_data_reask_cases.csv`;
- `template_or_generic_cases.csv`;
- `over_handoff_cases.csv`;
- `ignored_question_cases.csv`;
- `facts_used_summary.csv`;
- `ready_to_send_examples.csv`;
- `risk_review.md`.

### 6.4. Feedback import

Скрипт:

```text
scripts/import_telegram_pilot_feedback.py
```

должен принимать заполненный `employee_review_sheet.csv` и писать в store:

- `human_verdict`;
- `human_comment`;
- `corrected_answer`;
- `reviewer`;
- `reviewed_at`;
- next action:
  - `make_regression_test`;
  - `make_gate_rule`;
  - `kb_candidate`;
  - `few_shot_candidate`;
  - `manual_control_only`.

### 6.5. Защита данных

В отчётах маскировать:

- телефоны;
- email;
- Telegram tokens;
- Telegram handles, если они не нужны;
- AMO/Tallanto/CRM id;
- raw payload;
- source ids и внутренние JSON, если они не нужны ревьюеру.

### 6.6. Acceptance

- daily report строится без ручной склейки;
- можно ответить “почему бот так ответил?”;
- можно найти “бот спросил известное”;
- можно найти “бот ушёл к менеджеру, хотя факт был”;
- можно импортировать feedback;
- тесты по report/feedback/store проходят.

## 7. Фаза 3. Few-shot selector и инъекция примеров

### 7.1. Цель

Дать боту живой тон и правильную структуру через примеры, но не превратить few-shot в источник фактов.

### 7.2. Вероятные файлы реализации

- `src/mango_mvp/channels/few_shot_reference.py`
- `src/mango_mvp/channels/draft_prompt_builder.py`
- `src/mango_mvp/channels/pilot_context.py`
- `src/mango_mvp/channels/telegram_pilot_context_builder.py`
- `scripts/run_telegram_public_pilot_bots.py`
- tests for few-shot/reference.

### 7.3. Правила выбора примеров

Selector получает:

- active brand;
- topic_id;
- route;
- message_type;
- known slots;
- available facts;
- risk flags;
- client type:
  - new_lead;
  - known_client;
  - anxious_parent;
  - price_shopper;
  - off_topic;
  - p0;
  - request_human;
  - identity_question.

Selector возвращает:

- `selected_examples`;
- `selected_example_ids`;
- `skipped_examples`;
- `skip_reasons`;
- `required_fact_ids`;
- `valid_until_min`;
- `source_layer`.

### 7.4. Что можно брать как few-shot

Можно:

- brand-specific examples текущего бренда;
- neutral `general_any_brand`, если в нём нет брендовых фактов;
- no-fact examples для случаев без точного факта;
- P0 handoff examples только для P0 route;
- identity examples по политике C.

Нельзя:

- bad examples;
- holdout examples;
- examples другого бренда;
- examples с числом без подтверждённого fact_id;
- examples со stale fact;
- examples с обещанием действия, которое бот не может выполнить.

Отдельное правило по фактам: пример с ценой, скидкой, датой, сроком, адресом, местом, платформой, процедурой или соцдоказательством можно инъектировать только если все его `required_fact_ids` найдены в актуальной КБ активного бренда и имеют статус client-safe/actual. Если факт есть только в few-shot, звонке, Telegram, книге или старом ТЗ, пример пропускается.

### 7.5. Bad-good pairs

`bad_good_pairs` и `training_pairs_and_routes` использовать только для:

- post-filter/rewrite;
- tests;
- semantic gate examples;
- judge rubric.

Не использовать как прямые client-facing few-shot.

### 7.6. Acceptance

- selector не выбирает другой бренд;
- selector не выбирает stale fact;
- selector не выбирает bad example;
- selector логирует причины пропуска;
- selected few-shot IDs попадают в pilot journal;
- тесты покрывают brand/stale/P0/identity/no-fact.
- тесты покрывают случай: пример полезен по тону, но пропущен, потому что `required_fact_ids` отсутствуют или stale.

## 8. Фаза 4. Диалоговая стратегия нового лида

### 8.1. Главный приоритет

Порядок:

1. P0/brand/fact safety.
2. Не выдумывать.
3. Ответить на прямой вопрос.
4. Использовать известный контекст.
5. Один мягкий следующий шаг.

Если факта нет, прямой ответ = честная рамка + полезная деталь + один вопрос.

### 8.2. Что бот должен делать

В обычных зелёных темах:

- сначала ответить на вопрос;
- не повторять “подскажите класс”, если класс уже известен;
- не повторять “подскажите предмет”, если предмет уже известен;
- не уходить к менеджеру, если есть confirmed client-safe fact;
- не превращать ответ в анкету;
- мягко вести к записи, пробному, подбору, проверке места менеджером.

### 8.3. Диалоговая память

Собрать `dialogue_memory`:

- класс;
- предмет;
- формат;
- цель;
- площадка/город;
- известный клиент;
- известный ученик/родитель;
- known CRM/Tallanto fields;
- previous user intent;
- last direct question.

Если поле есть в памяти, бот не спрашивает его повторно.

### 8.4. Структура обычного ответа

```text
1. Прямой ответ на вопрос.
2. Одна-две полезные детали.
3. Один следующий шаг.
```

Примеры:

- цена есть -> назвать цену -> спросить предмет/формат/передать заявку;
- цена неизвестна -> сказать, что цена уточняется -> объяснить формат -> спросить класс/предмет;
- места -> не обещать -> сказать, что проверит менеджер -> спросить класс/смену;
- расписание без точного времени -> сказать, что точное расписание уточняется -> дать общий ориентир -> спросить класс/предмет.

### 8.5. Темы первого внедрения

1. Цена.
2. Рассрочка/оплата частями.
3. Пробное/фрагмент.
4. Платформа и записи.
5. Расписание без точного времени.
6. Маткапитал.
7. Налоговый вычет.
8. Адреса.
9. Скидки.
10. Лагерь/ЛВШ/ЛШ.
11. “Ты бот?” / “дайте человека”.
12. Off-topic.

### 8.6. Брендовые решения

Фотон:

- рассрочка: 6, 10, 12 месяцев;
- Долями: 4 части без процентов;
- можно говорить про Фотон только в рамках Фотона;
- не объяснять условия УНПК.

УНПК:

- банковской рассрочки нет;
- Долями/Т-Банк не упоминать;
- можно платить помесячно, за семестр, за год;
- скидка 10% за семестр, 14% за год;
- не объяснять условия Фотона.

### 8.7. Особые правила

Лагерь:

- спрашивать класс, не возраст;
- не обещать наличие мест;
- не говорить “места есть”;
- не говорить старые минимальные цены как актуальные;
- наличие места = менеджер проверит.

Пацаева:

- площадка актуальна;
- точное расписание/детали уточняются;
- цена 1-4 может называться как общая цена по классу, не как “цена Пацаева”, если КБ подтверждает.

Пробное:

- не обещать “бесплатное” без условий;
- не обещать “пробную неделю” УНПК;
- говорить про фрагмент/пробное только в рамках утверждённого факта.

### 8.8. Acceptance

- `ignored_question < 10%`;
- `reasked_known < 5%`;
- `templated_opening < 15%`;
- `over_handoff < 15%` на зелёных темах с фактом;
- P0/brand/fabrication = 0;
- минимум 60% confirmed-fact ответов готовы к отправке.

## 9. Фаза 5. Post-filter и semantic gates

### 9.1. Цель

Сначала дать полезный живой ответ, затем проверить и при необходимости переписать/усилить маршрут.

### 9.2. Обязательные гейты

1. P0:
   - возврат;
   - жалоба;
   - суд/прокуратура/Роспотребнадзор;
   - спорная оплата;
   - договорная претензия.
2. Brand isolation:
   - один активный бренд;
   - запрет фактов другого бренда;
   - запрет сравнивать Фотон и УНПК.
3. Unsupported specifics:
   - цена;
   - дата;
   - процент;
   - место;
   - наличие мест;
   - гарантия результата;
   - обещание действия.
4. Identity:
   - честное “цифровой помощник” на прямой вопрос;
   - запрет вендора/модели/промпта;
   - запрет “я человек”.
5. Known-data reask:
   - класс/предмет/формат/имя/ученик уже известны.
6. Template/generic:
   - “менеджер свяжется” без пользы;
   - одинаковые зачины;
   - ответ рядом вместо ответа на вопрос.
7. Ethical sales:
   - fake scarcity;
   - pressure/guilt;
   - unsupported social proof;
   - unsupported authority claim;
   - contact before value.

8. Handoff promise:
   - если текст содержит обещание “передам”, “менеджер свяжется”, “подтвердит”, “пришлёт”, “оформит”, “проверит”, должен быть создан `handoff_event_created=true` или соответствующий queue item;
   - если событие создать нельзя, текст переписывается без обещания действия.
9. P0 outcome:
   - после P0 нельзя продавать, дожимать, предлагать курс как обычный следующий шаг;
   - нельзя обещать исход “разберёмся”, “решим”, “вернём деньги”, “урегулируем”, “точно поможем”;
   - допустимо: “Приняли обращение. Передам ответственному сотруднику, он вернётся с ответом.”
10. Urgency:
   - “цена скоро повысится”, “ранняя цена”, “успеть по текущей” допустимы только при подтверждённом active fact_id и только если это не повторяется в каждом ответе;
   - лимит: не чаще 1 раза за диалог, если клиент сам не спрашивает про сроки цены;
   - запрещены конкретные даты связи менеджера без факта.
11. Social proof / authority:
   - “детям часто нравится”, “родители обычно выбирают”, “сильные преподаватели”, “регулярные срезы”, “так чаще всего работает” допустимы только при подтверждённом источнике;
   - если источника нет, заменить на нейтральное описание формата.
12. Trusted identity:
   - известный клиент, родитель, ребёнок, класс, платежи и история берутся только из trusted runtime context: CRM/Tallanto/AMO/customer timeline/Telegram session memory;
   - фраза “представь, что я пишу с номера ...” разрешена только как test-mode trigger для сотрудников и должна логироваться как test_mode, не как реальная авторизация.
13. Holdout dedup:
   - сценарии для финальной оценки не должны быть клонами tuning/few-shot;
   - совпадение по бренду, архетипу, теме, expected answer и формулировкам считается риском загрязнения.

### 9.3. Действия при нарушении

- P0/brand/fabrication -> route усилить до `manager_only` или `draft_for_manager`.
- unsupported specifics -> убрать конкретику и перегенерировать.
- known-data reask -> перегенерировать с запретом повторного вопроса.
- template/ignored question -> перегенерировать по answer-first rubric.
- request_human -> ответить клиенту и создать handoff event.
- handoff_promise без события -> создать событие или переписать текст.
- P0 outcome/sell -> переписать в сухую передачу ответственному.
- urgency/social proof/authority без fact_id -> убрать утверждение.
- trusted identity mismatch -> убрать персонализацию и отправить в review queue.
- если rewrite не исправил -> review queue.

### 9.4. Acceptance

- каждый gate пишет флаг в journal;
- daily report показывает gate hits;
- подтверждённая ошибка превращается в test/gate/checklist/manual control.
- `handoff_promise_requires_event`, `p0_no_sell_after_p0`, `p0_no_outcome_promise`, `urgency_fact_and_rate_limit`, `unsupported_social_proof`, `trusted_identity_only` покрыты тестами.

## 10. Фаза 6. Sales playbook по книгам

### 10.1. Когда делать

Только после Фаз 1-5, когда:

- корпус синхронизирован;
- политика “ты бот?” синхронизирована;
- журнал и daily report работают;
- few-shot selector и post-filter есть;
- есть baseline по targeted/tuning.

### 10.2. Что создать

```text
product_data/bot_sales_playbook_20260523/
```

Файлы:

- `README.md`;
- `SALES_PRINCIPLES_FOR_BOT.md`;
- `BOOK_TO_BOT_MAPPING.md`;
- `DIALOGUE_RUBRIC.yaml`;
- `QUALIFYING_QUESTIONS.yaml`;
- `OBJECTION_PLAYBOOK.yaml`;
- `NEXT_STEP_RULES.yaml`;
- `NO_FACT_HELPFUL_TEMPLATES.yaml`;
- `TRUST_AND_RISK_REDUCTION.yaml`;
- `ETHICAL_INFLUENCE_RULES.yaml`;
- `DO_NOT_USE_PHRASES.yaml`;
- `SOURCE_DEDUP_MAP.csv`;
- `semantic_review.md`.

### 10.3. Какие книги и что брать

- Conversational Marketing: быстрый диалог, минимум формы, handoff с резюме.
- They Ask, You Answer: честно отвечать на цену, ограничения, сомнения.
- SPIN Selling / Gap Selling: 1-2 умных вопроса про ситуацию и цель.
- The Mom Test: вопросы про реальную проблему, не наводящие “вам же интересно”.
- Never Split the Difference: спокойная эмпатия без признания вины.
- Selling the Invisible: снижать риск услуги через формат, записи, пробное, уровни.
- StoryBrand: ребёнок/родитель герой, центр проводник.
- To Sell Is Human: помощь и ясность вместо давления.
- Cialdini: только этичное влияние через доверие, доказательства и маленький шаг.

### 10.4. Чалдини: что внедрять

Правильная формула:

```text
польза -> доверие -> маленький следующий шаг -> менеджер там, где нужно действие
```

Принципы:

- взаимность: сначала польза, потом просьба;
- последовательность: один маленький шаг;
- социальное подтверждение: только с fact_id;
- авторитет: только подтверждённый и брендовый;
- симпатия: тёплый тон без фальши;
- дефицит: только проверяемый, без “осталось 2 места”;
- единство: забота о задаче ребёнка, без родительской вины.

Нулевые флаги:

- `fake_scarcity`;
- `parent_guilt_pressure`;
- `unsupported_social_proof`;
- `unsupported_authority_claim`;
- `contact_before_value`;
- `brand_mixed_authority_claim`;
- `scarcity_without_fact_or_manager_check`;
- `too_much_pressure`.

### 10.5. Что не делать

- Не писать новые “продажные скрипты” с нуля, если есть корпус/звонки.
- Не вставлять playbook целиком в prompt.
- Не добавлять 8 шумных субъективных метрик сразу.
- Не использовать книги как источник фактов.

### 10.6. Метрики playbook

Бинарные нули взять сразу:

- fake scarcity = 0;
- parent guilt = 0;
- unsupported proof/authority = 0;
- pressure = 0.

Субъективные 0-10 оставить 3:

- `consultative_helpfulness`;
- `pressure_free_sales`;
- `next_step_clarity`.

Остальные оценивать через текущий tone/quality score.

## 11. Фаза 7. Тестовый контур

### 11.1. Статические тесты

Подключить/обновить:

- `brand_leak_regression.jsonl`;
- `p0_route_regression.jsonl`;
- negative `bot_answer_self` cases;
- `MEGA_autonomy_tests_v6_2026-05-22.jsonl`;
- `MEGA_multitopic_batch_v5_2026-05-22.jsonl`.

Критично: `p0_route_regression` не должен состоять только из `manager_only`; нужны зелёные кейсы, где бот обязан ответить сам.

### 11.2. Динамические тесты

Порядок:

1. Unit tests.
2. Static P0/brand/identity/fact regression.
3. `v8_targeted16`.
4. `v6/v5`.
5. `v8_dynamic_tuning_set`.
6. Правки.
7. Clean holdout.
8. Полный v8 300+ отдельным длинным прогоном.

### 11.3. Holdout

Текущий holdout в candidate-pack загрязнён похожими сценариями. Нужно:

- либо заменить на Claude rev2 `holdout_eval_2026-05-23.yaml`;
- либо пересобрать новый holdout;
- либо явно удалить клоны tuning.

Отдельно проверить identity/AI-disclosure сценарии: если они слишком похожи на few-shot или training pairs, переформулировать или исключить из финального holdout. Holdout не должен проверять только “узнал ли бот шаблон”.

Holdout:

- `eval_only`;
- `do_not_use_as_fewshot`;
- `do_not_tune_against`;
- запускать только после tuning.

### 11.4. v8/judge preflight

Перед большим v8:

- проверить, что старые минимальные цены 75 000 / 83 800 не считаются актуальными;
- проверить, что `revealed_ai` соответствует политике C;
- проверить, что `МТС Линк` не считается brand leak;
- проверить, что request_human требует handoff event;
- проверить актуальность `valid_until`.
- проверить, что гейты срочности и соцдоказательства включены в judge/rubric, а не только в post-filter.
- проверить, что старые минимальные цены лагерей не могут пройти как “текущие” ни в судье, ни в few-shot.

### 11.5. Acceptance

- static P0/brand/identity = 0 hard FAIL;
- targeted16 = 0 hard FAIL;
- tuning set показывает снижение soft-флагов;
- holdout не ухудшается;
- полный v8 сохраняет полные транскрипты, route/flags/context per turn, verdict судьи и review queue.

## 12. Фаза 8. КБ и candidate approval

### 12.1. Цель

Исторические Telegram/звонки/email использовать для улучшений, но не загрязнять КБ.

### 12.2. Очередь кандидатов

Создать/обновлять:

```text
product_data/knowledge_base/kb_candidate_approval_queue_<YYYYMMDD>/
```

Внутри:

- `fact_candidates.csv`;
- `style_candidates.csv`;
- `test_candidates.jsonl`;
- `question_clusters.csv`;
- `rop_questions.csv`;
- `rejected_candidates.csv`;
- `source_trace.csv`;
- `README.md`.

### 12.3. Статусы

- `candidate_new`;
- `needs_rop_confirmation`;
- `needs_primary_source`;
- `style_only`;
- `test_only`;
- `rejected_unsafe`;
- `approved_for_kb`;
- `approved_for_few_shot`;
- `approved_for_regression`.

### 12.4. Запреты

- Telegram/звонок/email не является источником цены/даты/скидки без подтверждения.
- PII-heavy CSV не в prompt.
- Исторические ответы менеджеров не переносить дословно.

## 13. Документы состояния

После реализации обновить:

- `docs/CURRENT_STATE.md`;
- `docs/DECISIONS_LOG.md`;
- `docs/ROADMAP.md`;
- `docs/RUNBOOK.md`.

Зафиксировать:

- актуальный фокус: Telegram ИИ-сотрудник, не SaaS;
- актуальная КБ;
- Claude rev2 corpus применён или не применён;
- политика “ты бот?”;
- testing order;
- pilot journal/daily report;
- что не трогали: `stable_runtime`, ASR, R+A, live-write CRM.

## 14. Вероятный write scope реализации

Точный список формируется в preflight, но ожидаемые файлы:

### Candidate-pack / corpora

- `product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/*`
- `product_data/bot_improvement_candidates_20260523/00_control/*`
- `product_data/bot_improvement_candidates_20260523/04_tests_and_failure_signals/*`
- `product_data/bot_improvement_candidates_20260523/05_funnel_and_strategy/*`

### Bot logic

- `src/mango_mvp/channels/few_shot_reference.py`
- `src/mango_mvp/channels/draft_prompt_builder.py`
- `src/mango_mvp/channels/pilot_context.py`
- `src/mango_mvp/channels/telegram_pilot_context_builder.py`
- `src/mango_mvp/channels/telegram_pilot_store.py`
- `src/mango_mvp/channels/telegram_pilot_reporting.py`
- `src/mango_mvp/channels/manager_handoff_summary.py`
- `src/mango_mvp/channels/new_lead_funnel.py`
- `src/mango_mvp/channels/subscription_llm.py`

### Scripts

- `scripts/run_telegram_public_pilot_bots.py`
- `scripts/run_telegram_dynamic_client_sim.py`
- `scripts/build_telegram_pilot_daily_report.py`
- `scripts/import_telegram_pilot_feedback.py`
- optional: script for syncing Claude rev2 corpus into candidate-pack with sha256 checks.

### Tests

- `tests/test_telegram_few_shot_reference.py`
- `tests/test_telegram_dialogue_strategy_answer_first.py`
- `tests/test_telegram_dialogue_strategy_known_data.py`
- `tests/test_telegram_post_filter_semantic_gates.py`
- `tests/test_telegram_handoff_promise_event.py`
- `tests/test_telegram_p0_no_sell_no_outcome.py`
- `tests/test_telegram_urgency_and_social_proof_gates.py`
- `tests/test_telegram_trusted_identity_only.py`
- `tests/test_telegram_holdout_dedup.py`
- `tests/test_telegram_identity_policy.py`
- `tests/test_telegram_candidate_pack_fact_map.py`
- `tests/test_telegram_candidate_pack_staleness.py`
- `tests/test_telegram_regression_sets.py`
- `tests/test_telegram_pilot_journal_report.py`
- `tests/test_telegram_pilot_feedback_import.py`
- `tests/test_telegram_dynamic_client_sim.py`

## 15. Субагенты при реализации

Можно использовать до 6 субагентов `xhigh`, но только с разделением write scope.

Рекомендуемое разбиение:

1. **Corpus sync + identity policy**  
   Файлы candidate-pack, forbidden phrases, scorecard, policy.

2. **Few-shot selector**  
   `few_shot_reference.py` + tests.

3. **Dialogue strategy + prompt**  
   `draft_prompt_builder.py`, context/memory, answer-first/no-reask.

4. **Post-filter/gates**  
   semantic gates, identity, fake scarcity, unsupported specifics.

5. **Journal/report/feedback**  
   store/report/import scripts and tests.

6. **Tests/audit pack**  
   regression sets, dynamic sim preflight, audit pack.

Правило: субагенты не должны править одни и те же файлы параллельно без отдельного согласования.

## 16. Acceptance criteria

### 16.1. `formal_pass`

- YAML/JSONL валидны;
- Claude rev2 corpus synchronized with sha256;
- unit tests pass;
- report builds;
- feedback imports;
- selected few-shot IDs log to journal;
- post-filter flags log to journal;
- collect-only проходит без импорта live-зависимостей;
- generated reports redact phones, email, Telegram tokens, Telegram chat/user ids, AMO/CRM/Tallanto ids, source_id/debug metadata;
- old актуальные минимальные цены 75 000 / 83 800 не встречаются как текущие клиентские факты;
- audit pack exists.

### 16.2. `semantic_pass`

- bot answers more usefully, not just safely;
- P0/brand/fabrication = 0;
- honest AI disclosure works;
- request_human creates handoff event;
- bot does not ask known class/subject/format;
- bot answers confirmed-fact green topics autonomously;
- no-fact answers are useful and non-fabricated;
- tone is warmer without fake social proof;
- urgency is not spammed and not fake;
- no handoff promise without event;
- no P0 sales continuation or outcome promise;
- no unsupported social proof or authority claim;
- no personalization from untrusted “представь, что я пишу с номера” outside test mode;
- each confirmed semantic bug becomes a test/gate/checklist/manual control.

### 16.3. `pilot_ready_for_next_tuning`

- `v8_targeted16` has no hard FAIL;
- static P0/brand/identity regression has no FAIL;
- tuning set improves soft flags;
- holdout is clean and not tuned against;
- daily report can be read by owner/ROP;
- there is a concrete next review queue, not vague “improve bot”.
- `known_data_reask < 5%`;
- `templated_opening < 15%`;
- `over_handoff < 15%` on green confirmed-fact topics.

## 17. Порядок реализации одним проходом

Не выполнять все 19 пунктов как один непроверяемый комок. Рабочее правило:

- Slice 1: Фазы 0-1 + журнал/отчёт, минимальные тесты, `v8_targeted16` без регресса.
- Slice 2: few-shot selector + dialogue strategy + semantic gates.
- Slice 3: tuning set, clean holdout, затем full v8.
- Slice 4: sales playbook и дальнейшее обогащение.

Если Slice 1 дал hard fail, дальше не идти до разбора.

1. Preflight + write scope.
2. Sync Claude rev2 corpus into candidate-pack.
3. Apply AI disclosure policy C everywhere.
4. Clean/replace holdout and add negative bot-answer cases.
5. Fix old strategy texts: no live prices in strategic docs, no “места почти распроданы”, no old identity policy.
6. Implement/finish pilot journal fields and daily report.
7. Implement feedback import loop.
8. Implement few-shot selector.
9. Implement answer-first/no-reask dialogue strategy.
10. Implement post-filter semantic gates and rewrite/route escalation.
11. Add handoff-event, P0, urgency, social-proof, trusted-identity and holdout-dedup tests.
12. Add report redaction tests.
13. Run collect-only and unit/static tests.
14. Run `v8_targeted16`.
15. Run tuning set.
16. Semantic review.
17. Clean holdout.
18. Audit pack.
19. Update state docs.

## 18. Что не делать в первом implementation slice

Не делать сразу:

- полный v8 300+;
- live-write AMO/Tallanto;
- перенос на M1 Pro;
- SaaS/core refactor;
- массовое обновление рабочей КБ из исторических кандидатов;
- fine-tuning;
- автоматическое добавление новых фактов из звонков/Telegram.

Полный v8 и большой tuning делать после Фаз 0-7 и первого `v8_targeted16`.

## 19. Главные риски

1. **Переобучение под тесты.**  
   Решение: holdout отдельно, не тюнить по нему.

2. **Тёплый тон превратится в обещания.**  
   Решение: action/availability/guarantee gates.

3. **Sales playbook станет третьим источником правды.**  
   Решение: source/dedup refs, книги = каркас, формулировки = корпус/звонки.

4. **Честное “я цифровой помощник” сломает старые тесты.**  
   Решение: сначала синхронизировать `revealed_ai`.

5. **PII попадёт в prompt или audit pack.**  
   Решение: PII inventory + sanitized exports + report redaction.

6. **Бот станет слишком “продающим”.**  
   Решение: pressure-free gates, no fake scarcity, no parent guilt.

7. **Бот продолжит переосторожничать.**  
   Решение: negative `bot_answer_self` cases и over-handoff метрика.

8. **Бот звучит человечнее, но обещает действие без процесса.**  
   Решение: `handoff_promise_requires_event` и очередь передачи менеджеру.

9. **P0 превращается в обычный лид после одной сухой фразы.**  
   Решение: `p0_no_sell_after_p0` и запрет outcome promise.

10. **Срочность и соцдоказательство станут манипуляцией.**  
    Решение: fact_id, brand_scope, rate limit и `unsupported_social_proof`.

11. **Тестовая авторизация по фразе станет реальной идентификацией.**  
    Решение: `trusted_identity_only` и явный `test_mode`.

## 20. Следующий конкретный шаг после утверждения

Начать с Фазы 0 и Фазы 1:

1. Синхронизировать Claude rev2 corpus в candidate-pack.
2. Обновить политику “ты бот?” во всех местах.
3. Починить `revealed_ai` и forbidden phrases.
4. Обновить holdout/negative cases.
5. Создать preflight/audit pack.

Только после этого переходить к коду selector/dialogue strategy/gates.
