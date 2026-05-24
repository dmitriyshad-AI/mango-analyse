# PROBLEMS_LOG — анализ качества ответов бота

Пин: git `649c9397`. Классы: `ROOT` (системное) / `PATCH` (частный фикс, не генерализуется) /
`SAFETY_OK` (правильный safety-guard) / `DEAD_CONFLICT`. Evidence — file:line по копиям в снимке.

> Гипотеза «генерацию не настроили» — ЧАСТИЧНО ОПРОВЕРГНУТА: промпт инструктирует answer-first,
> тепло, анти-шаблон, и в него заходят intent-plan/memory/few-shot/funnel. Проблема глубже — ниже.

## Найдено самим (draft_prompt_builder.py)

### P1 [ROOT] Перегруз промпта: ~140 строк правил до контекста и сообщения
`build_draft_prompt` (стр. 140-283) — стена инструкций: безопасность, автономность, полезность,
playbook, gold, контекст, память, intent-plan, воронка, тон, таксономия, JSON-схема, спец-правила тем,
РОП. Известное поведение LLM: при таком объёме конкурирующих правил модель надёжно выполняет
повторённые SAFETY/структурные требования и **скатывается в безопасное/шаблонное**, недооценивая
answer-first/тёплость, зашитые в середине. Размер промпта сам душит качество. → укоротить, поднять
answer-first + few-shot наверх, безопасность — коротким жёстким блоком.

### P2 [ROOT] В промпте ОДНА LLM-задача делает всё сразу
Модель в одном вызове: классифицирует topic_id из закрытой таксономии + message_type + route +
question_parts + draft_text + quality_notes + checklist (схема, стр. 236-259). Классификационное
давление тянет draft к «шаблону темы». Генерация смешана с маршрутизацией и разметкой.

### P3 [ROOT/risk] Факты захардкожены в текст промпта (рассинхрон с КБ)
Стр. 188-191: правила Фотон «6,10,12 + Долями», УНПК «10%/14%, без Т-Банка» вписаны в промпт прозой,
а не берутся только из снапшота КБ. Источник правды раздваивается: правка КБ не меняет промпт.
То же — `SAFE_SCHEDULE_TEMPLATE_TEXT` (стр. 15-19) — захардкоженный «ответ».

### P4 [ROOT] Захардкоженные safe-template как «ответ» (корень шаблонности)
`SAFE_SCHEDULE_TEMPLATE` здесь + (из outline subscription_llm) `_cross_brand_safe_template`,
`_missing_fact_helpful_template`, `_operational_specificity_guarded_result`, и по транскриптам:
`unpk_installment_approved_fallback`, `camp_safe_template`, `trial_safe_template`,
`direct_process_safe_template`. Это канцелярит, которым guard'ы ЗАМЕНЯЮТ живой черновик ради
«безопасности» → ровно отсюда дословные повторы (templated_opening 14/16) и пропуск под-вопроса
(шаблон не отвечает на уточнение). Чем больше тем-шаблонов добавляют, тем хуже шаблонность. Это
двигатель whack-a-mole.

### P5 [observation] Промпт сам по себе разумный (answer-first, тёплость, анти-шаблон есть)
Стр. 163-229 — инструкции про прямой ответ, важность последней реплики, не-анкета, тёплый тон,
не повторять зачины. Значит дефект не в отсутствии инструкций, а в (а) их перегрузе и (б) пост-слое,
который их перебивает шаблонами.

## Гипотеза для проверки субагентами
Качество пытаются добыть детерминированными правилами + канцелярит-шаблонами поверх одного
перегруженного LLM-вызова. Safety правилами достигается, КАЧЕСТВО — нет: шаблон-подстановки сами
рождают templated/ignored, а каждый новый фикс = новый шаблон/ветка = больше жёсткости. Это
структурный whack-a-mole, не дисциплинарный.

---

## Находки субагентов (по компонентам, с evidence)

### subscription_llm.py (guard-стек + шаблоны)
- P6 [ROOT] ~10 per-topic safe-template функций ЗАМЕНЯЮТ черновик на ЗЕЛЁНЫХ темах: `_terminal_safe_template`(2809), `_camp_safe_template`(3275), `_trial_safe_template`(3358), `_pricing_safe_template`(3757), `_discount_safe_template`(2630), `_installment_safe_template`(2682), `_matkap_safe_template`(2727), `_tax_safe_template`(3249), `_docs_safe_template`(2977), `_teacher_safe_template`(3015), `_missing_fact_helpful_template`(3050). ~90 захардкоженных строк. Это и есть источник templated_opening/ignored_question.
- P7 [ROOT] 20+ точек ПЕРЕЗАПИСИ клиентского текста против ~7 чистых BLOCK-гардов. Перекос в рискованную перезапись.
- P8 [ROOT, главный архитектурный баг] Весь guard-стек прогоняется ДВАЖДЫ (build_draft 704-730), `answer_quality_rewriter` стоит ПОСЕРЕДИНЕ (713) → второй проход шаблонов затирает улучшенный рерайтером ответ. `skip_quality_template_overwrite`(1373) дырявый: не защищает от `_missing_fact_helpful_template`, refund/legal/complaint, known_context_redundant_question.
- P9 [PATCH] 70+ хардкод-сценарных веток; `_soften_current_price_deadline_text`(3456) — ~20 regex, чинящих СВОЙ же искажённый текст (патч на патч).
- P10 [DEAD_CONFLICT] `apply_funnel_policy_guard`(2174) дублирует p0-ветку autonomy_matrix(1763); два расходящихся набора verified-template (1166 vs 1863); двойной identity-матч в terminal (2818 и 2850).

### Сборка контекста (pilot_context / telegram_pilot_context_builder / customer_context_for_draft)
- P11 [ROOT] Факты выбираются по совпадению слов/чисел (`_fact_match_score` 812-864), НЕ по resolved-интенту → wrong-scope (цена лагеря на вопрос про онлайн-лето).
- P12 [ROOT, конкретный баг] `_ALLOWED_CONTEXT_KEYS` (draft_prompt_builder) молча роняет `read_only_customer_context`, который раннер выставляет (run_telegram_public_pilot_bots 609) — до промпта доходит только summary. Генерация недополучает CRM-контекст.
- P13 [PATCH] few-shot путь относительный → зависит от CWD; вне корня lru_cache вернёт {} и примеры исчезнут.
- P14 [DEAD_CONFLICT] `customer_context_for_draft.py` — мёртвая параллельная ветка: живой раннер зовёт `build_read_only_crm_context`, не её. Норм. бренд-нормализация продублирована 3×.

### Смысл/память/воронка (conversation_intent_plan / dialogue_memory / new_lead_funnel / signals)
- P15 [ROOT] 100% подстрочный/regex матч, нет семантики. «сдавал огэ»→grade=9 ошибочно; новая формулировка без маркера → general_consultation.
- P16 [ROOT] confidence фиктивна (lookup, не вероятность); низкая уверенность НЕ откатывается на буквальный вопрос, а НАСЛЕДУЕТ прежний интент + тянет прежний продукт → движок assumed_unstated_need.
- P17 [ROOT] Память теряет/устаревает: текущее сообщение мёржится override=True conf 0.95 → «а у брата 9?» перезапишет grade; turns обрезаны до 10 → слот из 11-го хода назад теряется → повторный вопрос; open_question обновляется только если новое сообщение — вопрос → follow-up без «?» оставляет старый open_question → зацикливание. Прямая причина «ломается на 2-3 ходу».
- P18 [ROOT] `answer_quality_rewriter` ИГНОРИРУЕТ `intent_plan.direct_question`: `_answers_direct_question` начинается с `del direct_question`(856) и переклассифицирует своими regex. Прямой вопрос детектится ДВАЖДЫ разными движками → «план поймал, ответ игнорирует».
- P19 [ROOT/SAFETY] 3-4 рассинхронизированных P0-детектора; мошенн*/незаконн*/«жаловаться» (стем жалов≠жалоб) пропущены ВЕЗДЕ; «возврат к теме» чищен в intent_plan, но не в new_lead_funnel/signals → benign→P0 там.

### Качество/few-shot (answer_quality_rewriter / few_shot_reference)
- P20 [PATCH] Детерминированный rewrite = ~13 хардкод-веток с фиксированными строками; `wrong_scope_fact_selected` — единственная ветка (online-summer).
- P21 [DEAD_CONFLICT] Слой против `safe_template_repeated` САМ плодит повторяющиеся шаблоны (фиксированные строки на каждый триггер), self-repeat-чек смотрит только на прошлые реплики из контекста, не на свой вывод.
- P22 [ROOT] Нет вариации few-shot: `examples[:2]` — один и тот же пример каждый раз на тему+бренд → модель воспроизводит зачин → templated.
- P23 [gap] LLM-rewrite не получает тёплый корпус как style-reference (знает правила, не видит эталон тона). holdout/eval-only нигде не исключён из few-shot (риск утечки, если eval в тех же YAML).

### Измерение (run_telegram_dynamic_client_sim + тесты)
- P24 [ROOT, смертельно для метрики] Судье в confirmed_facts ПОДСОВЫВАЕТСЯ СОБСТВЕННЫЙ safe-template бота (run_sim 697-701: `_is_verified_client_safe_template`) → освобождён от fabrication; сухой P0-хендофф освобождён от штрафа по тону. Бот «проходит», подменив ответ канцеляритом — подстановка шаблона СТРУКТУРНО ПООЩРЯЕТСЯ замером.
- P25 [PATCH] ~130 тестов в test_subscription_llm_draft_provider, ~85-90% — замороженные per-incident проверки конкретных строк (один инцидент → одна ветка → один assert). Тесты ЗАКРЕПЛЯЮТ шаблонное поведение (assert, что бот ДОЛЖЕН выдать canned-шаблон).
- P26 [ROOT] Чистого holdout нет нигде в тестах/симе — только тюнинг-сет. Нет метрики «менеджер отправил бы без правок» на свежих данных. Нет теста, ловящего регресс качества на НЕвиденных входах.
