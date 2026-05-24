# ТЗ: внедрение результатов из папки улучшений в Telegram-ботов и КБ

Дата: 2026-05-23  
Статус: `draft_for_discussion`  
Реализация: не начинать до отдельного подтверждения Дмитрия  
Основной источник: `product_data/bot_improvement_candidates_20260523/`

## 1. Цель

Внедрить полезные результаты из папки улучшений так, чтобы боты Фотона и УНПК стали:

- более полезными для новых клиентов;
- более тёплыми и живыми по тону;
- лучше помнили контекст диалога;
- меньше уходили к менеджеру там, где есть подтверждённый факт;
- не теряли P0-безопасность, бренд-разделение и фактологическую строгость.

Главная продуктовая цель: приблизить бота к роли внутреннего ИИ-сотрудника продаж, который помогает клиенту выбрать следующий шаг, но не выдумывает факты и не берёт на себя рискованные действия.

## 2. Что считаем источником

Candidate-pack:

```text
product_data/bot_improvement_candidates_20260523/
```

Финальный статус candidate-pack зафиксирован:

```text
product_data/bot_improvement_candidates_20260523/09_claude_cli_review/CODEX_CLAUDE_CONVERGENCE.md
```

Вердикт: `PASS` именно как candidate-pack, не как рабочая КБ.

## 3. Жёсткие ограничения

Нельзя в рамках этого ТЗ:

- применять всю папку целиком;
- переносить исторические ответы менеджеров как факты;
- загружать PII-heavy CSV целиком в промпты;
- менять `stable_runtime`;
- запускать ASR / Resolve+Analyze;
- писать в AMO/CRM/Tallanto;
- отправлять сообщения клиентам вне текущего утверждённого Telegram-пилота;
- расширять автономность P0-тем;
- смешивать Фотон и УНПК в одном клиентском ответе.

Любой факт можно использовать в клиентском ответе только если:

- он относится к активному бренду;
- есть `actual_fact_id` в `FACT_KEY_TO_FACT_ID_MAP.csv` или в актуальной КБ;
- факт client-safe;
- факт не устарел по `STALENESS_CALENDAR.md`;
- route policy разрешает автономный ответ или явно требует manager/draft route.

## 4. Связь с уже выполненным ТЗ

Это ТЗ продолжает, а не дублирует:

```text
docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md
```

Уже сделанный журнал пилота, daily report, feedback import, route/fact/flag logging и динамический симулятор считаются фундаментом.

В этом ТЗ основной фокус: внедрить лучшие материалы из candidate-pack в поведение бота и в тестовый контур.

## 5. Входные файлы candidate-pack

### 5.1. Управляющие файлы

- `00_control/PRIMARY_SECONDARY_SOURCES.md`
- `00_control/READY_FOR_NEXT_TZ_CHECKLIST.md`
- `00_control/PRIORITIZED_IMPLEMENTATION_BACKLOG.csv`
- `00_control/QUALITY_SCORECARD.md`
- `00_control/FACT_KEY_TO_FACT_ID_MAP.csv`
- `00_control/STALENESS_CALENDAR.md`
- `00_control/PII_RISK_INVENTORY.md`
- `00_control/SHARING_EXPORT_GUIDE.md`
- `00_control/FILE_ROLE_REGISTRY.csv`
- `00_control/DEDUP_MAP.md`

### 5.2. Тон и примеры

- `01_gold_and_few_shot/few_shot_warm_answers_2026-05-23.yaml`
- `01_gold_and_few_shot/few_shot_advanced_pack_2026-05-23.yaml`
- `01_gold_and_few_shot/gold_dialogues_multiturn_2026-05-23.yaml`
- `01_gold_and_few_shot/forbidden_phrases.yaml`
- `01_gold_and_few_shot/GOLD_ANSWERS_v3_2026-05-21.yaml`

### 5.3. Исторические кандидаты

- `02_historical_channels/*`
- `03_calls_sales_playbooks/*`
- `08_pilot_feedback_and_sales_insight/*`

Использовать только как:

- стиль;
- сценарии;
- частые вопросы;
- candidate facts для РОП/KB approval;
- тестовые сигналы.

Не использовать как источник истины.

### 5.4. Тестовые материалы

- `04_tests_and_failure_signals/brand_leak_regression.jsonl`
- `04_tests_and_failure_signals/p0_route_regression.jsonl`
- `04_tests_and_failure_signals/v8_dynamic_tuning_set_20260523.jsonl`
- `04_tests_and_failure_signals/v8_dynamic_holdout_set_20260523.jsonl`
- `04_tests_and_failure_signals/v8_targeted16_2026-05-22.jsonl`
- `04_tests_and_failure_signals/MEGA_autonomy_tests_v6_2026-05-22.jsonl`
- `04_tests_and_failure_signals/MEGA_multitopic_batch_v5_2026-05-22.jsonl`

## 6. Этап 0. Preflight и границы работы

Перед реализацией:

1. Проверить `git status --short`.
2. Зафиксировать список файлов, которые можно менять.
3. Не трогать unrelated изменения параллельных диалогов.
4. Проверить, что текущая рабочая КБ и bot-pack совпадают с `docs/CURRENT_STATE.md`.
5. Проверить, что candidate-pack существует и его `CODEX_CLAUDE_CONVERGENCE.md` содержит финальный `PASS`.
6. Проверить, что `FACT_KEY_TO_FACT_ID_MAP.csv` валиден:
   - 0 missing;
   - 0 cross-brand fact mismatch;
   - `runtime_context_required` только для CRM/runtime контекста.
7. Проверить `STALENESS_CALENDAR.md`:
   - ближайшие даты устаревания;
   - нельзя использовать факт после `valid_until` без обновления.

Acceptance:

- есть preflight report;
- явно перечислены файлы реализации;
- нет изменений `stable_runtime`;
- нет raw PII в промптах/логах.

## 7. Этап 1. Безопасная загрузка few-shot и gold-примеров

### 7.1. Цель

Сделать так, чтобы бот получал 2-3 релевантных живых примера под тему и бренд, но только если эти примеры безопасны по фактам.

### 7.2. Что реализовать

Создать или доработать слой:

```text
src/mango_mvp/channels/few_shot_reference.py
```

Логика:

1. На вход:
   - active brand;
   - detected topic;
   - route;
   - available facts;
   - known slots;
   - risk flags.
2. Выбрать примеры из:
   - `few_shot_warm_answers`;
   - `few_shot_advanced_pack`;
   - `gold_dialogues_multiturn`.
3. Отфильтровать:
   - другой бренд;
   - P0-несоответствие route;
   - наличие запрещённых фраз;
   - отсутствие fact_id для факто-нагруженного примера;
   - устаревший `valid_until`;
   - примеры, где route policy требует manager/draft, а текущий ответ автономный.
4. Вернуть:
   - `selected_examples`;
   - `skipped_examples`;
   - `skip_reasons`;
   - `fact_ids_required`;
   - `valid_until_min`.

### 7.3. Правила отбора

Приоритет примеров:

1. Совпал бренд + тема + route.
2. Совпал бренд + смежная тема + route.
3. `general_any_brand` только если пример не содержит брендовых фактов.
4. Bad→good пары использовать для post-filter/rewrite, а не как прямые примеры клиентского ответа.

Запрет:

- не выбирать пример только потому, что он красиво звучит;
- не выбирать пример с числом, если это число не связано с актуальным fact_id;
- не выбирать пример УНПК для Фотона и наоборот.

Acceptance:

- есть unit-тесты на brand isolation;
- есть unit-тесты на stale fact rejection;
- есть unit-тесты на forbidden phrases;
- есть unit-тесты на `crm.customer_loaded` как runtime context, а не KB fact;
- в лог решения пишутся выбранные few-shot IDs и причины пропуска.

## 8. Этап 2. Диалоговая стратегия: answer-first + context memory

### 8.1. Цель

Убрать главные soft-проблемы:

- `ignored_question`;
- `templated_opening`;
- `over_handoff`;
- `reasked_known`;
- слабый следующий шаг.

### 8.2. Правило приоритетов

Порядок:

1. P0/brand/fact safety.
2. Не выдумывать.
3. Ответить на прямой вопрос.
4. Не спрашивать уже известное.
5. Один мягкий следующий шаг.

Если факта нет, прямой ответ должен звучать так:

- честно обозначить рамку;
- не называть неподтверждённую конкретику;
- дать полезный ориентир;
- задать один вопрос.

### 8.3. Что доработать

В `draft_prompt_builder.py` и/или соседнем dialogue strategy слое:

1. Перед генерацией сформировать `dialogue_memory`:
   - класс;
   - предмет;
   - формат;
   - цель;
   - город/площадка;
   - known client;
   - known parent/student fields.
2. В prompt явно запретить повторно спрашивать известные поля.
3. В prompt добавить структуру ответа:
   - первое смысловое предложение отвечает на вопрос;
   - затем 1-2 полезных детали;
   - затем один следующий шаг.
4. Подключить selected few-shot из этапа 1.
5. Для no-fact случаев использовать `no_fact_helpful_template`, а не пустой handoff.
6. Для P0 оставить сухой и твёрдый handoff без сбора данных.

### 8.4. Темы первого внедрения

В первую очередь:

1. Цена.
2. Рассрочка/оплата частями.
3. Пробное/фрагмент занятия.
4. Платформа и записи.
5. Расписание без точного времени.
6. Маткапитал.
7. Налоговый вычет.
8. Адреса.
9. Скидки с условиями.
10. Лагерь/ЛВШ/ЛШ.

### 8.5. Acceptance

- `ignored_question < 10%` на targeted/holdout.
- `reasked_known < 5%`.
- `templated_opening < 15%`.
- `over_handoff < 15%` на зелёных темах с подтверждённым фактом.
- P0/brand/fabrication = 0.
- На confirmed-fact темах минимум 60% ответов реально готовы к отправке.

## 9. Этап 3. Post-filter и semantic gates

### 9.1. Цель

Безопасность должна быть фильтром после попытки дать живой полезный ответ, а не причиной всегда писать сухой шаблон.

### 9.2. Gates

Обязательные проверки:

1. P0:
   - возврат;
   - жалоба;
   - суд/прокуратура/Роспотребнадзор;
   - спорная оплата;
   - договорная претензия.
2. Brand isolation:
   - один активный бренд;
   - запрещённые слова другого бренда;
   - запрет сравнивать Фотон и УНПК.
3. Unsupported specifics:
   - цена;
   - дата;
   - процент;
   - место;
   - наличие мест;
   - гарантия результата;
   - обещание действия.
4. Forbidden phrases:
   - из `forbidden_phrases.yaml`;
   - партнёрские/cross-brand referral фразы.
5. Known-data reask:
   - бот не спрашивает класс/предмет/формат/имя, если они уже есть.
6. Template detector:
   - слишком общий ответ;
   - “менеджер свяжется” без пользы;
   - одинаковые зачины.

### 9.3. Действия при нарушении

- P0/brand/fabrication: route усилить до manager_only или draft_for_manager.
- Unsupported specifics: убрать неподтверждённую конкретику и перегенерировать.
- Reasked known data: перегенерировать с запретом спрашивать известное.
- Template/ignored question: перегенерировать с answer-first rubric.
- Если повторная генерация не исправила — отправить в review queue.

Acceptance:

- каждый gate пишет флаг в journal;
- каждый сработавший gate виден в daily report;
- подтверждённая ошибка превращается в тест или checklist item.

## 10. Этап 4. КБ-обогащение только через approval

### 10.1. Цель

Использовать исторические каналы для улучшения КБ, но не загрязнить её устаревшими или неофициальными фактами.

### 10.2. Что делать

Создать отдельную очередь:

```text
product_data/knowledge_base/kb_candidate_approval_queue_YYYYMMDD/
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

Кандидаты брать из:

- `02_historical_channels/`;
- `03_calls_sales_playbooks/`;
- `08_pilot_feedback_and_sales_insight/`;
- daily report feedback.

### 10.3. Статусы кандидатов

Использовать статусы:

- `candidate_new`;
- `needs_rop_confirmation`;
- `needs_primary_source`;
- `style_only`;
- `test_only`;
- `rejected_unsafe`;
- `approved_for_kb`;
- `approved_for_few_shot`;
- `approved_for_regression`.

### 10.4. Что нельзя

- Не добавлять факт в рабочую КБ без `approved_for_kb`.
- Не использовать Telegram/звонок как единственный источник цены/даты/скидки.
- Не добавлять менеджерский ответ в few-shot, если он содержит PII, старую цену, cross-brand или обещание действия.

Acceptance:

- есть human-readable approval queue;
- есть source trace;
- есть rejected list с причинами;
- есть DIFF перед любым изменением рабочей КБ.

## 11. Этап 5. Тестовый контур

### 11.1. Статические тесты

Подключить:

- `brand_leak_regression.jsonl`;
- `p0_route_regression.jsonl`;
- `MEGA_autonomy_tests_v6_2026-05-22.jsonl`;
- `MEGA_multitopic_batch_v5_2026-05-22.jsonl`.

Важно: эти файлы могут быть candidate-тестами. Перед подключением к CI проверить формат runner.

### 11.2. Динамические тесты

Использовать:

- `v8_targeted16_2026-05-22.jsonl` — быстрый smoke;
- `v8_dynamic_tuning_set_20260523.jsonl` — для тюнинга;
- `v8_dynamic_holdout_set_20260523.jsonl` — только финальная проверка.

Запрет:

- не тюнить prompt/few-shot по holdout;
- не объявлять успех только по targeted16;
- не скрывать soft-флаги за общим PASS.

### 11.3. Порядок прогонов

1. Unit-тесты few-shot/reference/filter.
2. Unit-тесты dialogue strategy.
3. Static P0/brand regression.
4. `v8_targeted16`.
5. `v6/v5`.
6. `v8_dynamic_tuning_set`.
7. Правки.
8. `v8_dynamic_holdout_set`.

Полный v8 300+ диалогов запускать отдельно после этого, с resume и полными транскриптами.

## 12. Этап 6. Daily feedback loop

### 12.1. Цель

Каждый реальный промах пилота должен попадать в цикл улучшения.

### 12.2. Что добавить в daily report

В daily report должны быть отдельные очереди:

- `ready_to_send_examples`;
- `needs_small_edit`;
- `rewrite_required`;
- `unsafe_or_p0`;
- `reasked_known_data`;
- `too_template`;
- `ignored_direct_question`;
- `over_handoff`;
- `candidate_new_fact`;
- `candidate_new_test`;
- `candidate_new_few_shot`.

### 12.3. Feedback import

При импорте feedback сотрудника:

- если `wrong_fact` → candidate fact review;
- если `unsafe` → P0/semantic gate review;
- если `too_robotic` → few-shot/style review;
- если `asked_known_data` → known-data regression;
- если `ignored_question` → answer-first regression;
- если `useful` → candidate gold/few-shot.

Acceptance:

- ежедневный report показывает не только ошибки, но и хорошие ответы;
- хорошие ответы можно отправить в approval queue;
- плохие ответы автоматически становятся кандидатами на тест/gate.

## 13. Метрики качества

Обязательные нули:

- P0-промахи: 0.
- Brand leak: 0.
- Unsupported price/date/discount/place: 0.
- Сбор данных в возврате/жалобе/суде: 0.
- Раскрытие `бот/ИИ/GPT/Claude/Codex`: 0.

Целевые показатели:

- `ignored_question < 10%`;
- `templated_opening < 15%`;
- `over_handoff < 15%` на confirmed-fact green topics;
- `reasked_known < 5%`;
- tone average >= 75;
- ready-to-send >= 70% на свежем holdout;
- ready-to-send >= 60% на confirmed-fact темах;
- next_step_offered растёт без роста P0/brand/fabrication.

## 14. Тесты, которые нужно добавить или обновить

Минимум:

- `tests/test_telegram_few_shot_reference.py`;
- `tests/test_telegram_dialogue_strategy_answer_first.py`;
- `tests/test_telegram_dialogue_strategy_known_data.py`;
- `tests/test_telegram_post_filter_semantic_gates.py`;
- `tests/test_telegram_candidate_pack_fact_map.py`;
- `tests/test_telegram_candidate_pack_staleness.py`;
- `tests/test_telegram_regression_sets.py`;
- `tests/test_telegram_pilot_journal_report.py`;
- `tests/test_telegram_pilot_feedback_import.py`.

Обязательные кейсы:

- Фотон цена с подтверждённым фактом → автономно, прямо, с одним шагом.
- УНПК цена с подтверждённым фактом → автономно, без Фотона.
- Фотон рассрочка → 6/10/12 + Долями, без обещания одобрения.
- УНПК оплата → помесячно/семестр/год, без Т-Банк/Долями.
- Нет точной цены → полезно без числа.
- Клиент уже дал класс → не спрашивать класс.
- Клиент уже дал предмет → не спрашивать предмет.
- P0 в середине обычного диалога → manager_only.
- Лагерь → спрашивать класс, не возраст; не обещать места.
- Off-topic → мягко вернуть к теме активного бренда.
- “Кто вы?” → честная брендовая формулировка “цифровой помощник <бренда>, не живой оператор”, без vendor/model/prompt.
- Forbidden phrase → filter/rewrite.
- Cross-brand historical phrase → rejected.
- Stale fact → не использовать.

## 15. Документы и audit pack

После реализации создать:

```text
audits/_inbox/bot_improvement_candidate_pack_implementation_<timestamp>/
```

Минимум:

- `implementation_notes.md`;
- `changed_files.txt`;
- `test_output.txt`;
- `semantic_review.md`;
- `risk_review.md`;
- `backward_compatibility.md`;
- `candidate_pack_diff.md`;
- `before_after_quality_metrics.md`;
- `v8_targeted16_result.md`;
- `holdout_result.md`, если запускался holdout.

Обновить при необходимости:

- `docs/CURRENT_STATE.md`;
- `docs/DECISIONS_LOG.md`;
- `docs/ROADMAP.md`;
- `docs/RUNBOOK.md`;

## 16. Definition of done

### formal_pass

- новые тесты проходят;
- JSONL/CSV/YAML валидны;
- few-shot selector работает и логирует причины выбора/пропуска;
- post-filter/gates пишут флаги;
- daily report видит новые флаги;
- audit pack создан.

### semantic_pass

- ответы стали полезнее, а не просто безопаснее;
- нет P0/brand/fabrication;
- бот не спрашивает известное в контрольных сценариях;
- бот даёт прямой ответ при подтверждённом факте;
- no-fact ответы полезны, но без выдумок;
- тон теплее и менее шаблонный;
- выбранные few-shot не содержат запрещённых фраз;
- каждый новый подтверждённый промах стал тестом, gate rule, checklist item или ручным контролем.

### pilot_ready для следующего шага

- `v8_targeted16` без hard FAIL;
- static P0/brand regression без FAIL;
- holdout показывает улучшение по `ignored_question`, `templated_opening`, `over_handoff`, `reasked_known`;
- daily report может показать Дмитрию и РОПу, что именно улучшилось;
- есть список следующих точечных правок, а не общая фраза “надо улучшать бота”.

## 17. Рекомендуемый порядок реализации

1. Preflight + точный список файлов.
2. Few-shot reference selector.
3. Forbidden phrases + staleness/fact map gates.
4. Answer-first + no-reask-known strategy.
5. Over-handoff rewrite для зелёных тем.
6. No-fact helpful templates.
7. Static P0/brand regression.
8. `v8_targeted16`.
9. Tuning set.
10. Semantic review.
11. Holdout.
12. Audit pack.

## 18. Что обсудить перед стартом реализации

1. Внедряем всё одним блоком или режем на 2-3 коммита?
2. Можно ли сразу подключать few-shot selector к live pilot, или сначала только к simulator/dry-run?
3. Нужен ли отдельный sanitized export для РОПа до внедрения?
4. Кто проставляет `due` в open questions/review queue?
5. Какой уровень модели использовать для dynamic tests после внедрения: high или xhigh?
