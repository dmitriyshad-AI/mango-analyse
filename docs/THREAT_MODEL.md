# Threat Model

Дата: 2026-05-10

Статус: актуализировано после Stage15 v11 frozen gate.

## Scope

Этот threat model используется для независимого аудита bot-safe / controlled allowlist слоя и downstream-экспортов, которые опираются на результаты анализа звонков.

В scope:

- `bot_safe_answer`;
- controlled bot allowlist;
- ROP/manager-assist knowledge base;
- sanitizer output;
- independent detector output;
- frozen adversarial corpus;
- CRM/writeback input, если он содержит transcript-derived историю общения.

Вне scope:

- live ASR/R+A execution;
- live CRM/AMO/Tallanto writes;
- сырые аудиофайлы;
- ручная бизнес-политика компании, если она еще не утверждена владельцем.

## Current Release Baseline

Фактический актуальный слой на 2026-05-10:

- Stage15 gate: `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`;
- frozen corpus: `stable_runtime/bot_safety_frozen_corpus_20260510_v3_frozen_gate/bot_safety_adversarial_cases.jsonl`;
- frozen corpus validation: `stable_runtime/bot_safety_frozen_corpus_validation_20260510_v4_frozen_gate/summary.json`;
- knowledge base: `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v11_frozen_gate/`;
- ROP pack: `stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v11_frozen_gate/`;
- canonical DB after quality backfill: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`;
- phone-chain report: `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/`.

На этом слое:

- fixed-point sanitizer реализован;
- independent detector реализован и не должен импортировать regex-правила sanitizer;
- ASR-tolerance cases включены во frozen corpus;
- frozen validation прошла с `0` failures;
- Stage15 v11 прошел;
- autonomous bot production остается заблокированным до ROP-review over-sanitization queue.

## Personal Data

### Definition

Нельзя выпускать в bot-safe слой или CRM-ready публичный текст:

- ФИО клиента, ребенка, родителя;
- телефоны, email, messenger handles;
- одиночные русские имена в клиентском контексте;
- фамилии и отчества после role/placeholders: `ученик Гамзяков`, `ученик Алексеевичу`;
- фамилии педагогов: `преподаватель Лукина`, `будет вести Кондрашова`;
- фамилии после слов `фамилия`, `фамилию`, `фамилии`.

### Adversarial Examples

- `ученик Гамазяков`;
- `ученик Гамзяков`;
- `преподаватель Пасынкова`;
- `будет вести Кондрашова`;
- `назовите фамилию Николаев`;
- `напишите в телеграм @client_name`;
- `почта ivan.petrov@example.com`;
- `номер 8 916 123-45-67`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- `tests/test_knowledge_base.py`;
- `tests/test_bot_safety_detector.py`;
- `tests/test_bot_safety_frozen_corpus.py`.

## Locations

### Definition

Нельзя выпускать без tenant config:

- города, метро, улицы, переулки, проспекты;
- кабинеты и аудитории;
- локальные кампусы/объекты: КПМ, ФТИ, корпус ФТИ;
- филиалы, если они не подтверждены текущей компанией.

### Adversarial Examples

- `Сухаревская`;
- `Долгопрудный`;
- `Скорняжный переулок`;
- `Корняжный переулок`;
- `Пацаева 7 корпус 1 кабинет 49`;
- `кабинет 324`;
- `КПМ`;
- `ФТИ`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `location`;
- `tests/test_bot_safety_frozen_corpus.py`.

## Money

### Definition

Нельзя выпускать:

- конкретные цены;
- скидки и проценты;
- персональные рассрочки;
- конкретные условия ранней оплаты;
- платежные провайдеры в контексте инструкции к оплате;
- обещания возврата денег.

### Adversarial Examples

- `7900 за 4 занятия`;
- `семестр за 88000`;
- `год целиком за 147000`;
- `при ранней оплате 78400`;
- `скидка 20%`;
- `оплата через Альфа`;
- `оформим Яндекс Сплит`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `money`;
- `test_unsafe_naked_5digit_amount_blocked`.

## Deadlines

### Definition

Нельзя выпускать без актуализации:

- `до конца дня`, `до конца недели`, `до конца каникул`;
- `до 15 числа`, `17 числа`;
- календарные даты и интервалы;
- обещания удержать бронь.

### Adversarial Examples

- `акция до конца дня`;
- `можно записаться до 17 числа`;
- `бронь держим до 15 числа`;
- `перезвоним сегодня вечером`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `deadline`;
- Stage15 gate risk counts.

## Promises

### Definition

Нельзя выпускать как обещание бота:

- `вернемся сегодня`;
- `перезвоним до конца дня`;
- `компенсируем занятие`;
- `возместим`;
- `проверим и сразу напишем`.

### Adversarial Examples

- `мы компенсируем это занятие`;
- `вернемся с ответом сегодня`;
- `менеджер точно перезвонит до конца дня`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `promise`.

## Brand Tenant

### Definition

Tenant-specific сущности допустимы только после явной настройки компании. Для SaaS-safe слоя они считаются риском или требуют tenant config.

### Adversarial Examples

- `Фотон`;
- `МФТИ`;
- `ФТИ`;
- `КПМ`;
- ASR-варианты брендов.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus ASR-tolerance layer.

## CRM Writeback Relevance

### Definition

Нельзя автоматически писать в CRM history/follow-up слой строки, которые не являются полезным контекстом для менеджера продаж:

- wrong-number / wrong-direction;
- wrong-person / identity mismatch: собеседник не тот человек, контакт не подтвердился, путаница с именем;
- no-content / no-dialogue;
- технические IVR и служебные проверки;
- сторонний B2B/vendor/government/marketing/HR запрос;
- запросы без EdTech-intent и без понятного следующего действия;
- CRM-текст с пустой историей или обрезкой через `...`.

Такие строки должны уходить в non-ready/manual-review, а не в AMO-ready.

### Adversarial Examples

- `клиент позвонил по вакансии программиста и попросил HR`;
- `представитель интегратора АМА CRM звонил по виджету`;
- `служебная проверка телефонии Мангофис / колл-трекинг`;
- `номер общий и он не знает, кто ранее звонил`;
- `клиент ошибся номером / вышел на не ту организацию`;
- `контакт не подтвердился, на линии была не та Светлана, произошла путаница с именем`;
- `обсуждение программы, интереса к продукту и следующих шагов не состоялось`;
- `деталей о классе, предмете, продукте и цене не зафиксировано`;
- `автоматическое сообщение о подборе менеджеров по продажам, нажать 1`;
- `историк и журналист искал контакты фонда`;
- `CRM summary ends with ...`.

### Matcher Reference

- detector: `src/mango_mvp/quality/crm_writeback_quality_detector.py`;
- corpus validator: `src/mango_mvp/quality/crm_writeback_frozen_corpus.py`;
- gate runner: `scripts/run_crm_writeback_quality_gate.py`;
- export builder integration: `scripts/build_post_backfill_amo_ready_export.py`.

### Test Reference

- `tests/fixtures/crm_writeback_relevance_frozen_corpus.jsonl`;
- `tests/test_crm_writeback_quality_detector.py`;
- `tests/test_crm_writeback_frozen_corpus.py`;
- `tests/test_crm_writeback_quality_gate.py`;
- `tests/test_post_backfill_amo_ready_export.py`.

## Over-Sanitization

### Definition

Это не safety leak, но качество ответа недостаточно для автономного бота.

Примеры:

- повтор placeholder-а несколько раз подряд;
- ответ стал слишком общим;
- рядом несколько generic replacements без полезного смысла.

Ожидаемая реакция:

- не считать P0/P1 утечкой;
- отправлять в `over_sanitization_candidates`;
- не выпускать в autonomous bot без ROP-review.

## Exit Criterion

Controlled allowlist / manager-assist слой можно считать готовым, если:

- frozen corpus validation: `0` P0/P1 failures;
- Stage15 gate: `passed=true`;
- `crm_quality_writeback_ready=true`;
- `bot_allowlist_export_ready=true`;
- `bot_autonomous_production_ready=false`, пока over-sanitization queue не разобрана;
- independent detector показывает `0` P0/P1 на release allowlist;
- `fixpoint_not_reached=0`;
- Claude/GPT-аудит используется как periodic monitoring, а не бесконечный release blocker.

## CRM Writeback v5 Product Gate Addendum (2026-05-10)

Новые обязательные классы перед live AMO writeback:

- `A4_service_existing_client_not_new_lead`: service/existing-client/technical calls не должны попадать в live sales writeback как новые лиды. Они остаются в истории клиента и manual/context queue.
- `B6_C9_orphan_or_ambiguous_amo_entity`: live writeback требует ровно один известный AMO contact id или отдельную явно одобренную create-contact policy.
- `C1_F5_population_recall_counter`: detector green не считается доказательством чистой популяции без независимого population marker counter.
- `C8_F8_corpus_self_validation_loop`: frozen corpus не может состоять только из прошлых Claude/GPT findings; нужны forward/random/negative-overblock layers.
- `C12_history_duplication`: дублирование CRM-полей является UX-risk для старых contact-only стадий и fail-live risk для новых contact/deal writeback стадий. Разные AI-поля должны отвечать на разные вопросы, а не повторять один и тот же текст.
- `F2_tenant_config_required`: industry, products, CRM protected/target fields, orphan policy и privacy policy должны быть tenant-level config, а не hardcoded правилами Фотона.
- `C10_third_party_pii_policy`: телефоны третьих лиц hard-redact; имена третьих лиц для internal CRM пока warn/report, для SaaS/bot-safe должны стать hard-redact/block.

Live writeback может быть разрешен только если Stage15 quality gate и CRM writeback quality gate оба passed, а `population_recall.passed_for_live=true`.

## CRM Writeback v6 Wrong-Person Addendum (2026-05-11)

Stage55 dry-run показал отдельный класс риска внутри A1/A3: строка может выглядеть как `sales_call`, но фактически это wrong-person / identity mismatch. Признаки:

- `контакт не подтвердился`;
- `путаница с именем`;
- `на линии был/была не тот/не та человек`;
- `обсуждение программы, интереса к продукту и следующих шагов не состоялось`.

Такие строки нельзя писать в AMO manager-assist поля как полезную историю продаж. Они должны блокироваться до live writeback или уходить в manual-review. Важно не overblock-ать валидное EdTech-возражение `это не та программа`, если клиент реально обсуждал ученика, предмет, курс или альтернативный формат.

### Matcher Reference

- detector: `src/mango_mvp/quality/crm_writeback_quality_detector.py`;
- population marker: `src/mango_mvp/quality/crm_writeback_population_recall.py`;
- frozen corpus layer: `stage55_local_dryrun_regression` + `negative_overblock_guard`.

### Test Reference

- `tests/test_crm_writeback_quality_detector.py`;
- `tests/test_crm_writeback_population_recall.py`;
- `tests/fixtures/crm_writeback_relevance_frozen_corpus.jsonl`;
- Stage54 gate: `stable_runtime/amo_live_stage54_20260511_v1/stage54_quality_gate/summary.json`.

## CRM Manager-Assist Text Quality Addendum (2026-05-10)

Этот раздел относится к CRM-internal manager-assist полям, прежде всего `Авто история общения`, `Последняя AI-сводка`, `AI-рекомендованный следующий шаг`.

### Q1 Lossy Ellipsis Truncation

CRM writeback не должен выпускать автоматическую обрезку через `...` / `…` в содержательных полях. Если текст слишком длинный, система должна структурно сжать его, использовать full-history storage, или заблокировать строку до настройки textarea/field capacity.

### Q2 Duplicate Label + Count Artifacts

CRM writeback не должен смешивать raw-label и count-label как два факта: `летний лагерь | летний лагерь: 14`, `математика | математика: 3`. Агрегаты должны выводиться в каноническом виде: `летний лагерь (14 касаний)` или без счетчиков в карточке менеджера.

### Q3 Weak / Stale Objection Labels

Слабые ярлыки `время`, `доверие`, `цена`, `неактуально`, `неудобно` нельзя выводить как актуальные возражения без evidence и recency. Нужно разделять `актуальные ограничения` и `исторические возражения`.

### Q4 Next-Step Consistency

`Следующий шаг`, `AI-приоритет`, `Вероятность продажи` и `Возражения` должны быть непротиворечивыми. Если next step означает `не беспокоить`, `отменить`, `ждать обращения`, это должно отражаться в priority/follow-up policy или уходить в manual-review.

### Q5 Manager UX Budget

AMO contact-card должен иметь compact manager summary без потери смысла. Полная история допустима, но должна жить в full-history поле/слое, а не маскироваться обрезкой.

### Q6 Post-Writeback Readback Gate

После каждого live stage система должна читать записанные поля обратно из AMO и запускать CRM text quality gate. Следующий stage разрешается только при зеленом readback gate.

### Q7 Cross-Field Information Uniqueness

Каждое AI-поле в контакте или сделке должно содержать уникальный смысл:

- `AI-рекомендованный следующий шаг` содержит только действие менеджера;
- `Последняя AI-сводка` / `AI-сводка по сделке` содержит краткое состояние;
- `Авто история общения` / `AI-история по сделке` содержит таймлайн без дословного повторения сводки;
- Tallanto-поля содержат только учебный/финансовый контекст;
- риск/предупреждение содержит только причину осторожности.

Если один и тот же абзац или почти тот же набор фактов повторяется в двух разных AI-полях, строка должна блокироваться до пересборки payload. Это не косметика: менеджер не должен читать одно и то же несколько раз в карточке.

### Test Reference

- future: `src/mango_mvp/quality/crm_text_quality_detector.py`;
- future: `tests/test_crm_text_quality_detector.py`;
- current evidence: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T141007Z/amo_live_stage20_fields_after_writeback.json`;
- current plan: `docs/CRM_TEXT_QUALITY_STAGE20_PLAN_2026-05-10.md`.

## CRM Text Quality Claude Refinements (2026-05-10)

После независимого аудита `crm_text_quality_stage20_20260510_v1` добавлены уточнения к Q1-Q6.

### Q1b Internal Ellipsis Is A Blocker

Любой `...` / `…` внутри CRM target AI fields является fail-live для новых stage writeback, даже если поле не заканчивается на `...`. Источники включают `_history_line`, `_call_gist`, `_compose_last_summary` и любые будущие summarization/truncation helpers.

### Q3a Weak Filler Labels

`время`, `доверие`, `цена`, `неудобно` являются слабыми ярлыками. Они допустимы только как warning/report или при наличии текстового evidence и recency. Их нельзя выводить как самостоятельные актуальные возражения.

### Q3b Historical Strong Negative Labels

`неактуально`, `отказ`, `не интересно`, `не беспокоить`, `отменить запись` являются сильными негативными/closure сигналами. Если они исторические, они должны иметь дату/evidence или переноситься в `Исторические возражения`. Если они актуальные, строка должна получать closure/manual-review policy, а не sales follow-up.

### Q4b Recommended Follow-Up Date Semantics

`Рекомендуемая дата следующего контакта` не может быть автоматически равна дате анализа для всех строк. Дата должна выводиться из next-step semantics, CRM task policy, explicit customer timing, или оставаться пустой/manual-review.

### Q4c Vague Next Step Marker

`связаться позже`, `связаться в мае`, `вернуться при изменении решения`, `ждать обращения`, `ждать выбора дат/решения клиента` не являются достаточным action для менеджера. Такие строки должны получать уточняющий manager-action или manual-review/low-priority policy.

### Q4d Lost Lead / Competitor Purchase Conflict

Если CRM-текст содержит закрывающий сигнал `уже купили/приобрели/оплатили у другого поставщика`, `выбрали другой лагерь/школу/центр`, `интерес закрыт покупкой у конкурента`, `дальнейшее продолжение сделки не требуется`, строка не может одновременно иметь активный следующий шаг вроде `перезвонить`, `отправить материалы`, `продолжить`, warm/hot priority или существенную вероятность продажи. Такой конфликт должен блокировать live writeback до пересчета next-step/priority или ручного review.

### Q4e Passive Customer Return Conflict

Если клиент явно сказал `сам свяжется`, `когда определимся`, `не предлагать активно`, `ждать обращения клиента`, CRM-текст не должен превращать это в активный менеджерский next step `перезвонить` / `отправить материалы` без даты и основания. Такие строки должны либо получить passive/low-touch policy, либо уйти в manual review.

### Q4f Explicit No Next Step Conflict

Если CRM-текст прямо говорит `договоренности о следующем шаге не было`, `следующий шаг не согласован`, `клиент отказался от дальнейшего интереса`, строка не может одновременно получать активный следующий шаг `перезвонить`, `отправить материалы`, warm/hot priority или высокую вероятность. Такой конфликт блокирует live writeback до пересчета статуса.

### Q4g Completed Payment / Closed Deal Conflict

Если CRM-текст или внешний CRM/deal context содержит `чек прислан`, `оплата поступила`, `платеж подтвержден`, `сделка закрыта/оплачена`, то AI-поля контакта или сделки не должны просить клиента снова `оплатить`, `прислать оплату`, `отправить чек`. Для контакта это либо post-sale/service context, либо задача менеджеру проверить документы; для сделки это должно отражаться в конкретной сделке.

### Q4h Stale Source Action

Если последний содержательный звонок старше 30 дней, AI не должен механически переносить старый следующий шаг (`перезвонить`, `отправить материалы`, `уточнить`, `соединить`, `выслать`, `передать`) как свежую задачу на сегодня. Такая строка должна стать reactivation/manual-review, либо получить новое основание из актуального AMO/Tallanto/мессенджер-контекста.

Если последний содержательный звонок старше 90 дней, любой непустой активный next-step считается небезопасным без отдельного свежего сигнала.

### Q4i Wrong Person / Identity Mismatch

Если звонок содержит `контакт не подтвердился`, `путаница с именем`, `на линии была не та ...`, `обсуждение программы не состоялось`, строка не должна попадать в live writeback как полезная история продаж. Это blocker, даже если старый анализ пометил звонок как `sales_call`.

### Q4j Active Client Loss Reason

Если AMO-сделка закрыта как `Закрыто и не реализовано`, но в причине отказа стоит `действующий клиент`, `текущий клиент`, `действующий ученик`, это не обычный потерянный лид. Такой кейс означает, что реальная работа, оплата или обучение могут идти в другой карточке/сделке/по другому телефону.

Правило:

- не писать активный продажный следующий шаг в закрытую сделку;
- не считать это проигранной продажей или холодным отказом;
- отправлять строку в manual/entity-resolution review: найти актуальную карточку контакта, активную сделку, связанный телефон, Tallanto-ученика, платежи и занятия.

### Q4k Terminal Loss Reason Taxonomy

Закрытые AMO-сделки с причиной отказа нельзя обрабатывать одним общим правилом `потерянный лид`. Причина закрытия определяет, что именно должна сделать система:

- `Дубль`, `объединены карточки` -> `duplicate_entity_resolution`: искать каноническую карточку/сделку и писать AI-контекст только туда.
- `Действующий клиент`, `текущий клиент`, `действующий ученик` -> `active_client_entity_resolution`: искать актуальную карточку, активную сделку, другой телефон, Tallanto-ученика, оплаты и занятия.
- `Не актуально`, `ушел к конкурентам`, `выбрали репетитора` -> `lost_or_not_actual`: не создавать активный следующий шаг без нового входящего сигнала.
- `Дорого` -> `lost_or_not_actual`: закрытый лид по цене не должен получать активный следующий шаг без новой акции/свежего сигнала.
- `Архив`, `нет связи`, `недозвон` -> `no_contact_archive`: не превращать старый недозвон в свежую задачу продаж.
- `Спам`, `Тест` -> `invalid_or_test_no_action`: не реальный лид, AI-writeback запрещен.
- `Перспектива`, `Перспектива (не подошло расписание)` -> `future_prospect_reactivation`: нужен отдельный сценарий реактивации с датой/условием, а не обычная продажная рекомендация.
- `Закрыли группу (мы)` -> `company_side_unavailable`: причина на стороне компании, сначала проверить альтернативы.
- `Возврат` -> `refund_or_postsale_service_review`: сервисно-финансовый контекст, не sales-next-step.
- `Выпускник` -> `graduate_or_alumni`: отдельная политика выпускников/повторных продаж.
- `Не оставлял заявку`, `не тот клиент` -> `no_application_wrong_direction`: блокировать sales-writeback, пока не подтверждено, что это целевой клиент.
- `Не квал`, `нецелевой`, `Не подходит формат`, неподходящий филиал/география, tenant-specific out-of-scope причины вроде `ШД Жако` -> `not_qualified_or_out_of_scope`: только review, не автоматический следующий шаг.
- `Другое` -> `ambiguous_other_manual_review`: ручная проверка, потому что причина закрытия не объясняет корректное действие.
- Пустая причина отказа при статусе `Закрыто и не реализовано` -> `terminal_lost_without_loss_reason_requires_manual_review`: ручная проверка, потому что система не имеет права угадывать причину закрытия.

Эта таксономия должна жить в коде как общий policy-layer, а не как набор телефонов или частных строк из аудита. При появлении новой причины отказа она добавляется в эту таблицу классов и в регрессионный тест.

### Q6b AMO Field Capacity Mismatch

Readback mismatch после live writeback является blocker даже если исходный payload прошел preflight. Класс риска: CRM-поле физически хранит меньше текста, чем отправляет система. Пример из Stage51: `Последняя AI-сводка` в AMO имела тип `text` и резала длинную сводку примерно до short-text лимита, тогда как `Авто история общения` имела тип `textarea` и хранила полный manager context.

Правило:

- `Последняя AI-сводка` и `AI-рекомендованный следующий шаг` должны быть AMO `textarea`; если они снова станут `text`, live writeback должен блокироваться или поле должно получать только compact value, не длиннее безопасного лимита short-text поля.
- Другие AMO `text`-поля должны получать только compact value, не длиннее безопасного лимита short-text поля.
- Полная история общения должна храниться только в long-text/full-history поле (`Авто история общения`) или отдельном full-history слое.
- Следующий live stage запрещен, если post-writeback readback gate видит `readback_value_mismatch`.

### Test Reference

- `tests/test_amo_writeback_guards.py`;
- `tests/test_amo_readback_gate.py`;
- `tests/test_crm_text_quality_detector.py`;
- Stage51 evidence: `stable_runtime/amo_live_stage51_20260512_v1/readback_after_live/summary.json`;
- Stage51 repair pack: `audits/_inbox/amo_stage51_summary_repair_preflight_20260512_v1/`.
