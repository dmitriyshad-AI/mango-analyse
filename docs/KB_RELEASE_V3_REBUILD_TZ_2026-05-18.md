# ТЗ: пересборка базы знаний v3 для Telegram-пилота

Дата: 2026-05-18  
Статус: готово к смысловому ревью и дальнейшей реализации  
Основание: handoff Claude `claude_to_codex_v3_handoff_2026-05-17`

## 1. Контекст и проблема

В проекте уже есть технически рабочий слой базы знаний v2:

- `product_data/knowledge_base/kb_release_20260517_v2`
- `product_data/knowledge_base/kb_release_20260517_v2_handoff_for_claude_and_team`
- сборщик: `scripts/build_kb_release_v2_from_claude_and_codex.py`
- тесты базы и Telegram Stage 6 проходят: локальный узкий набор даёт `60 passed`

Но глубокая проверка handoff Клода и текущих артефактов v2 показала: архитектурные правила в целом правильные, а содержательная часть базы знаний сломана. Это означает, что система умеет фильтровать, маршрутизировать и запрещать опасные вещи, но сама база фактов недостаточно пригодна для точных ответов.

Подтверждённые проблемы v2:

1. В `facts_registry` нет большинства ключевых цен и числовых условий. Из 25 контрольных чисел handoff найдено только 4: `16900`, `27720`, `18800`, `34400`. Не найдены `44600`, `74500`, `49000`, `82000`, `29750`, `47250`, `98000`, `75000`, `120000`, `89900`, `83800`, `3900`, `6900`, `23000`, `18900`, `94500`, `33000`, `50000`, `11900`, `56500`, `94000`.
2. Сложные словари вроде `prices_regular_2026_27`, `discounts`, `lvsh_mendeleevo_2026` были превращены в слишком общие записи. В v2 есть 3 price-записи по `prices_regular_2026_27` с пустым `fact_text`.
3. `forbidden_to_say` попал в факты как разрешённый клиентский текст. Найдено 11 строк с `forbidden_to_say` в `fact_id/fact_key`, у всех есть `client_safe_text` и `allowed_for_client_answer=True`.
4. `internal_only_for_number` не соблюдён. Номера лицензий и старый номер `70369` попали в разрешённые клиентские строки. Найдено 7 allowed client rows с номерами лицензий или старым номером.
5. Сломана связка фактов с источниками: 252 из 395 фактов имеют `source_id`, которого нет в `source_registry`; 198 из 204 уникальных `source_id` в фактах отсутствуют в реестре источников.
6. `approval_queue_for_rop_v2.csv` содержит 133 строки, но не покрывает бизнес-факты поштучно. По ключевым типам там только `price=5`, `discount=4`, `schedule=1`.
7. Stage 6 по v2 стал безопасным, но не достаточно содержательным: `became_more_substantive` около 7/20 вместо целевых 15-18/20.
8. Live-скрипт Telegram-пилота пока не подключает базу знаний через общий builder. В `scripts/telegram_manager_draft_pilot.py:143` используется `build_pilot_context(...)`, а не `build_telegram_pilot_context(..., snapshot_path=...)`.

Вывод: v2 нельзя считать финальной базой знаний для Telegram-пилота. Нужна воспроизводимая v3-пересборка.

## 2. Цель

Собрать v3-базу знаний так, чтобы:

1. Все проверенные факты из handoff Клода были представлены в машинном виде.
2. Каждое важное число было отдельным фактом, а не потерянной частью словаря.
3. Бот использовал только факты активного бренда: отдельно Фотон, отдельно УНПК.
4. Запрещённые фразы и внутренние данные не могли попасть в клиентский черновик.
5. Каждый факт имел проверяемую ссылку на источник.
6. РОП получил удобную очередь утверждения фактов, а не технический мусор.
7. Stage 6 доказал, что ответы стали содержательнее без потери безопасности.
8. Live Telegram-пилот мог явно использовать v3 snapshot в режиме черновиков для менеджера.

## 3. Источник истины

Основной вход для v3:

`claude_to_codex_v3_handoff_2026-05-17/`

Файлы:

- `README.md`
- `REBUILD_REQUIREMENTS_FROM_CLAUDE.md`
- `CHANGELOG_FINAL.md`
- `CHANGELOG_v3_after_team_answers.md`
- `OPEN_QUESTIONS_FOR_TEAM.md`
- `brand_rules.yaml`
- `bot_policy.yaml`
- `facts_for_bot_FOTON.yaml`
- `facts_for_bot_UNPK.yaml`
- `facts_internal_only.yaml`

Дополнительные Stage 6 fixtures:

`/Users/dmitrijfabarisov/Claude Projects/Foton/kb_release_v2_claude_layer_2026-05-17/stage6_fixtures/`

Дополнительные ответы Дмитрия/команды:

`/Users/dmitrijfabarisov/Claude Projects/Foton/Questions_for_Team_kb_v2_2026-05-17.docx`

Эти ответы важнее старых open-question статусов, если они явно закрывают вопрос. Свежие ответы q14/q15 из handoff Клода от 2026-05-18 важнее старых строк в `CHANGELOG_FINAL.md` и `REBUILD_REQUIREMENTS_FROM_CLAUDE.md`. При конфликте с `OPEN_QUESTIONS_FOR_TEAM.md` использовать свежий `OPEN_QUESTIONS_FOR_TEAM.md`, `CHANGELOG_v3_after_team_answers.md` и актуальные YAML, но сохранять осторожность: если ответ говорит “пока нет данных”, факт не становится точным клиентским фактом.

Важно: в `bot_policy.yaml` блок `open_questions` может содержать устаревшие q14/q15. v3-сборщик не должен считать этот блок главным источником статуса вопроса. Для q14/q15 главными являются актуальный `OPEN_QUESTIONS_FOR_TEAM.md`, комментарии и структура в `facts_for_bot_UNPK.yaml`, а также `Codex_v3_TZ_Review_2026-05-18.md`.

Старые папки не использовать как источник правды:

- `Mango_Bot_Knowledge_Base_FINAL_2026-05-17/`
- `Claude Mango_Bot_Knowledge_Base_FINAL_2026-05-17/`
- `kb_release_20260517_v2` как источник содержательных фактов

Текущий `kb_release_20260517_v2` можно использовать только как технический пример формата и как объект регрессионных проверок.

## 4. Границы работ

Можно менять:

- `scripts/build_kb_release_v2_from_claude_and_codex.py`, если выбрано расширение текущего сборщика;
- предпочтительно создать новый сборщик `scripts/build_kb_release_v3_from_claude_handoff.py`;
- `src/mango_mvp/knowledge_base/*`, если нужна общая функция нормализации фактов;
- `src/mango_mvp/channels/telegram_pilot_context_builder.py`;
- `scripts/run_telegram_stage6_kb_eval.py`;
- `scripts/telegram_manager_draft_pilot.py`, только для безопасного подключения snapshot по env-переменной;
- тесты в `tests/`;
- новые v3-артефакты в `product_data/knowledge_base/kb_release_20260518_v3/`;
- audit pack в `audits/_inbox/`.

Нельзя:

- менять `stable_runtime`;
- запускать ASR;
- запускать Resolve+Analyze;
- писать в AMO/CRM/Tallanto;
- отправлять сообщения клиентам;
- удалять старые v1/v2-артефакты;
- исправлять открытые вопросы Клода догадками;
- вручную редактировать исходные YAML Клода как способ “починить” сборку.

## 5. Техническое решение

### 5.1. Новый v3-сборщик

Рекомендуемое решение: создать отдельный сборщик:

`scripts/build_kb_release_v3_from_claude_handoff.py`

Причина: v2-сборщик уже используется тестами и артефактами. Латать его в лоб рискованно. v3 должен быть воспроизводимым отдельным блоком, с отдельным run-id и отдельными выходными папками.

Новый сборщик может переиспользовать безопасные функции из v2, но должен иметь явные v3-правила.

Выходы:

- `product_data/knowledge_base/kb_release_20260518_v3/kb_release_v3_snapshot.json`
- `product_data/knowledge_base/kb_release_20260518_v3/facts_registry.csv`
- `product_data/knowledge_base/kb_release_20260518_v3/facts_registry.jsonl`
- `product_data/knowledge_base/kb_release_20260518_v3/facts_registry.yaml`
- `product_data/knowledge_base/kb_release_20260518_v3/source_registry.csv`
- `product_data/knowledge_base/kb_release_20260518_v3/source_registry.json`
- `product_data/knowledge_base/kb_release_20260518_v3/approval_queue_for_rop_v3.csv`
- `product_data/knowledge_base/kb_release_20260518_v3/quality_report.json`
- `product_data/knowledge_base/kb_release_20260518_v3/QUALITY_REPORT.md`
- `product_data/knowledge_base/kb_release_20260518_v3/README.md`

Отдельно собрать handoff:

`product_data/knowledge_base/kb_release_20260518_v3_handoff_for_claude_and_team/`

### 5.2. Реестр источников

В v3 каждый факт обязан ссылаться на `source_id`, который есть в `source_registry`.

Минимально добавить источники:

- `claude_layer_v3:brand_rules`
- `claude_layer_v3:bot_policy`
- `claude_layer_v3:facts_for_bot_FOTON`
- `claude_layer_v3:facts_for_bot_UNPK`
- `claude_layer_v3:facts_internal_only`
- `claude_layer_v3:open_questions`
- `claude_layer_v3:rebuild_requirements`

Для каждого источника:

- `source_id`
- `source_kind`
- `title`
- `path`
- `sha256`
- `brand`
- `freshness_status`
- `source_status`

Если факт невозможно привязать к первичному источнику из Google Drive или сайта, он должен ссылаться хотя бы на конкретный Claude YAML, из которого был извлечён.

Запрещено создавать fallback вида `source:claude_layer:<section>`, если такой `source_id` не добавлен в registry.

### 5.3. Атомарная развёртка фактов

Главное правило: каждое число, срок, процент, дата, цена, лимит, скидка, промокод, параметр группы, смена лагеря или условие рассрочки должно стать отдельным фактом.

Пример: блок

`prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester = 44600`

должен стать отдельной записью:

- `fact_id`: стабильный и человекочитаемый;
- `fact_key`: полный путь в YAML;
- `fact_type`: `price`;
- `brand`: `foton` или `unpk`;
- `product`: например `regular_courses_offline_5_11`;
- `fact_text`: непустой текст для менеджера;
- `client_safe_text`: текст, который можно показать клиенту, если факт разрешён;
- `structured_value`: JSON с числом, валютой, периодом, классом, форматом, сроком действия;
- `source_id`: существующий в `source_registry`;
- `source_sha256`: заполнен;
- `freshness_status`;
- `allowed_for_client_answer`;
- `usable_for_precise_answer`;
- `requires_manager_confirmation`;
- `route_policy`;
- `linked_open_question`, если факт спорный.

Списки и диапазоны:

- `[47250, 52500]` не превращать в одну пустую строку;
- либо создать факт с `amount_min` и `amount_max`;
- либо создать два связанных факта `range_min` и `range_max`;
- текст должен объяснять, что это диапазон.

### 5.4. Запрещённые и внутренние поля

`forbidden_to_say`:

- не создаёт фактов;
- не попадает в `fact_text`;
- не попадает в `client_safe_text`;
- собирается в отдельный список пост-фильтра;
- если такая фраза появляется в `draft_text`, маршрут становится `manager_only`, причина `brand_separation_violation`.

`internal_only_for_number: true`:

- `number`, `date`, `holder` не попадают в `client_safe_text`;
- могут попасть только в `manager_check_text`, `internal_text` или `brand_internal_note`;
- клиентская фраза должна быть общей: “У нас есть лицензия на образовательную деятельность”.

Любое поле с суффиксом `_internal`:

- не попадает в `client_safe_text`;
- не создаёт `allowed_for_client_answer=True`;
- используется только для менеджера или внутренних проверок.

### 5.5. Открытые вопросы

Открытые вопросы из `OPEN_QUESTIONS_FOR_TEAM.md` нельзя закрывать самостоятельно.

Часть вопросов закрыта ответами из `Questions_for_Team_kb_v2_2026-05-17.docx` и последующим handoff Клода от 2026-05-18. Их нужно перенести в v3 как решения, но не смешивать с догадками.

Закрытые или уточнённые ответы:

- q3: формула связи брендов остаётся как есть: “Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра.”
- q8: второй и последующие предметы: УНПК МФТИ - 20%, онлайн Фотон - 30%.
- q11: модульных курсов М9/М11 сейчас нет; это были прошлые интенсивы. Нельзя публиковать матрицу модульных курсов как действующий продукт.
- тема 11, договор: заменить обещание “в течение рабочего дня” на “в ближайшие дни”.
- тема 12, справка: не уточнять тип справки; писать, что отправим в течение 10 дней, но постараемся сделать раньше.
- тема 17, преподаватели: имена не называть в общих ответах. Использовать только общие регалии. Преподавателя сообщают, когда человек уже учится в конкретной группе.
- q7: скидка сотрудникам МФТИ - 10% всегда; скидки не суммируются, применяется наибольшая.
- q9: “Долями” делит оплату на 4 части. Отдельно есть рассрочка для клиентов на 6 и 12 месяцев, стоимость курса может возрасти; условия индивидуальны и зависят от банка, поэтому вопрос направляется менеджеру.
- q12: возврат всегда отдаётся менеджеру; клиенту говорить, что менеджер свяжется по телефону.
- лицензии: 4 номера подтверждены.
- справка КНД 1151158: все 4 юрлица могут выдавать справку, так как у всех есть лицензия.
- q10: официальный адрес - “Скорняжный”, только так.
- q5 marketing: `@unpkmfti` действительно используется как Telegram-аккаунт Фотона, когда-нибудь будет изменён; отдельный канал выездных школ - `@kmiptlvs`.
- q5_it: в AMO и Tallanto воронки не разделены по брендам.
- Tallanto brand matching: сейчас однозначности нет. МФТИ и Пацаева - точно УНПК МФТИ; Москва и онлайн есть в обоих брендах, чаще Фотон, но много исключений.
- q4.2_it: выбираем два отдельных Telegram-бота.
- q13: дат ЗВШ Менделеево на 2026/27 пока нет; всех записываем в лист ожидания.
- q14: УНПК очно 49 000 / 82 000 относится к 5-11 классам.
- q15: УНПК онлайн 41 800 / 69 900 относится только к олимпиадной подготовке Физтех для 9 и 11 классов по будням. Для других классов нет подтверждённого предложения и цены, нужен менеджер.

Остающиеся спорные или осторожные факты должны иметь:

- `status` или `freshness_status`: `needs_owner_confirmation`;
- `allowed_for_client_answer: false`;
- `usable_for_precise_answer: false`;
- `route_policy`: `draft_for_manager` или `manager_handoff_only`;
- `linked_open_question`: например `promocodes` или другой реально открытый вопрос;
- понятный `manager_check_text`.

Особо проверить:

- q14 как закрытый факт: УНПК очно, 5-11 классы, 49 000 / 82 000;
- q15 как закрытый, но узкий факт: УНПК онлайн, олимпиадная подготовка Физтех, только 9 и 11 классы, 41 800 / 69 900;
- промокоды преподавателей и Флоктори;
- точную бренд-привязку CRM/Tallanto по Москва/онлайн, потому что текущая CRM/Tallanto структура не даёт надёжного разделения.

## 6. Обязательные содержательные факты

В v3 должны быть отражены 16 пунктов из handoff Клода:

1. Долями: после ответа команды считать текущей клиентской формулировкой “делит оплату на 4 части”. Старую формулировку handoff про “Долями Плюс 3/6/10 месяцев, 0-16,9%” не использовать как клиентский факт без нового подтверждения.
2. Рассрочка: есть варианты 6 и 12 месяцев, но стоимость курса может возрасти, условия индивидуальны и зависят от банка. Клиентский вопрос про рассрочку направлять менеджеру, не обещать автоматическое одобрение и не называть точные банковские условия как гарантированные.
3. ЗВШ Менделеево: точных дат на 2026/27 пока нет; клиенту можно говорить только про лист ожидания. Не называть даты и цены как точный факт.
4. Подписанты 2026: Кузнецова А.Е. для Фотона, Харламов М.Ю. для АНО УНПК. Только internal.
5. Шкала удержаний при возврате: только internal; клиенту не объяснять расчёт возврата, всегда manager_only.
6. Преподаватели: в общих ответах не называть ФИО, только регалии. Имена можно использовать только внутренне или после привязки клиента к конкретной группе.
7. q14: УНПК очно 49 000 / 82 000 как проверенный факт для 5-11 классов.
8. q15: УНПК онлайн 41 800 / 69 900 как проверенный факт только для олимпиадной подготовки Физтех для 9 и 11 классов по будням. Для других классов - handoff менеджеру.
9. Лицензии 4 юрлиц подтверждены, но клиенту только общий факт наличия лицензии. Номера, даты и юрлица не показывать в `client_safe_text`.
10. Все 4 юрлица могут выдавать справку КНД 1151158; клиенту можно говорить, что справку подготовим, но без раскрытия юрлица.
11. Маткапитал и налоговый вычет для обоих брендов.
12. Новые продукты Фотон: индивидуальные занятия 3 900 / 6 900 / 23 000; модульные М9/М11 не считать действующим продуктом до отдельного подтверждения.
13. Новые продукты УНПК: ФизТех-олимпиада 33 000 / 50 000; дошкольники Пацаева 11 900 / 56 500 / 94 000.
14. Скидки: второй предмет УНПК 20%, онлайн Фотон 30%, сотрудникам МФТИ 10%, скидки не суммируются, применяется наибольшая; refer_a_friend и другие скидки держать по фактам из YAML/ответов, спорные - в очередь РОПа.
15. НДС-льгота: internal.
16. Промокоды преподавателей и Флоктори: internal или `needs_owner_confirmation` до подтверждения маркетингом.
17. Справка: не спрашивать тип, обещать отправку в течение 10 дней, с фразой “постараемся сделать раньше”.
18. Договор: “менеджер пришлёт в ближайшие дни”, без обещания “в течение рабочего дня”.
19. Адрес Фотон Москва: “Скорняжный”, не “Скоряжный”.

## 7. Очередь РОПа v3

Создать:

`approval_queue_for_rop_v3.csv`

Цель: 400+ строк. Если корректная атомарная развёртка даёт меньше 400, не создавать фиктивные строки. Вместо этого quality report должен явно показать фактическое количество и список недостающих тем. Но без отдельного решения Дмитрия такой результат считается не пройденным.

Обязательные колонки:

- `priority`
- `approval_item_id`
- `item_type`
- `topic`
- `fact_id_ref`
- `brand`
- `product`
- `manager_text`
- `suggested_decision`
- `rop_question`
- `source_id`
- `linked_open_question`
- `risk_notes`

Очередь должна включать:

- все цены 2026/27;
- скидки;
- промокоды;
- дедлайны;
- ЛВШ и ЗВШ;
- параметры программ;
- контакты и безопасные фразы связи;
- интенсивы;
- рассрочку;
- лимиты налогового вычета;
- маткапитал;
- спорные факты по открытым вопросам.
- ответы из `Questions_for_Team_kb_v2_2026-05-17.docx`, особенно договор, справка, преподаватели, скидки, Долями, возврат, адрес, CRM/Tallanto brand isolation.

## 8. Telegram и Stage 6

### 8.1. Единый путь контекста

Сейчас Stage 6 и live Telegram могут использовать разные пути.

Нужно:

1. В `scripts/telegram_manager_draft_pilot.py` добавить безопасную поддержку env-переменной `TELEGRAM_PILOT_KB_SNAPSHOT_PATH`.
2. Если переменная задана, live-pilot context builder должен использовать `build_telegram_pilot_context(..., snapshot_path=..., active_brand=...)`.
3. Добавить env `TELEGRAM_PILOT_ACTIVE_BRAND`, допустимые значения: `foton`, `unpk`, `unknown`.
4. Поддержать два отдельных Telegram-бота как целевую архитектуру: отдельный бот Фотон и отдельный бот УНПК. Бренд определяется ботом/каналом входа, а не догадкой из AMO/Tallanto.
5. По умолчанию поведение не менять: если snapshot path не задан, live остаётся без KB.
6. Всё равно сохраняется режим “черновик менеджеру”, без отправки клиенту.

Важно: CRM/Tallanto не являются надёжным источником активного бренда. В AMO и Tallanto сейчас нет полного разделения по брендам. МФТИ и Пацаева можно считать УНПК; Москва и онлайн неоднозначны. Поэтому для клиентского ответа активный бренд должен приходить из выбранного Telegram-бота или другой явной точки входа.

### 8.2. Stage 6 fixtures

Использовать fixtures Клода:

- `stage6_fixtures_FOTON.jsonl`
- `stage6_fixtures_UNPK.jsonl`

Перед использованием fixtures нужно обновить ожидаемые ответы с учётом `Questions_for_Team_kb_v2_2026-05-17.docx`:

- `foton_03_installment_tbank`: больше не ожидать “Долями Плюс 3/6/10”. Долями = 4 части; рассрочка 6/12 месяцев индивидуальна и должна вести к менеджеру.
- `unpk_03_payment_monthly`: оставить осторожный ответ про варианты оплаты, без Фотона, Т-Банка и Долями.
- `foton_07_lvsh_mendeleevo` и `unpk_07_lvsh_mendeleevo`: если речь именно про ЗВШ, не ожидать точные даты; если речь про ЛВШ, проверять только подтверждённые смены ЛВШ. Для ЗВШ 2026/27 ожидаем “лист ожидания”.
- `foton_05_tax_deduction` и `unpk_05_tax_deduction`: справка до 10 дней, без уточнения типа справки.
- любые teacher fixtures: не ожидать ФИО преподавателей в клиентском draft.

Добавить проверку:

- `expected_route`;
- `expected_in_draft`;
- `forbidden_in_draft`;
- `expected_high_risk_flag`;
- `brand_separation_violation`;
- `unsupported_numeric_promise`;
- отсутствие служебных маркеров: `source_id`, `fact_id`, `freshness`, `AMO`, `Tallanto`, `GPT`, `Claude`, `Codex`, “я бот”, “я ИИ”.

Целевые метрики:

- `brand_separation_violation = 0`;
- `unsupported_numeric_promises = 0`;
- `high_risk_route_relaxed = 0`;
- `baseline_manager_only_relaxed = 0`;
- `invalid_topic_id = 0`;
- `expected_route_hit = 100%` для P0;
- `became_more_substantive >= 15/20`.

Реальный `provider=codex` запускать только если это не нарушает текущие ограничения и нет live-отправки. Если есть сомнения, сначала прогнать fake/smoke и подготовить команду для отдельного подтверждения.

## 9. Тесты

Создать новый файл:

`tests/test_kb_release_v3_import.py`

Минимальные тесты:

1. `test_v3_expands_nested_numeric_blocks_into_atomic_facts`
   Проверяет, что ключевые числа стали отдельными фактами с непустым `fact_text`, `client_safe_text`, `fact_type`, `brand`, `source_id`.

2. `test_v3_keeps_forbidden_to_say_out_of_facts_and_post_filter_blocks_it`
   Проверяет, что `forbidden_to_say` не создаёт клиентские факты, но попадает в пост-фильтр.

3. `test_v3_internal_only_for_number_keeps_license_numbers_out_of_client_safe_text`
   Проверяет, что `Л035`, `50Л01`, `№77753`, `70369`, `АНО ДПО` отсутствуют в allowed client text.

4. `test_v3_all_fact_source_ids_exist_in_source_registry`
   Проверяет 0 orphan facts.

5. `test_v3_open_or_cautious_questions_stay_needs_owner_confirmation`
   Проверяет q11/q13, промокоды и другие реально открытые или осторожные факты.

   После ответов команды тест нужно учитывать: q7/q8/q14/q15 закрыты, q11 и q13 не становятся точными клиентскими фактами. q11 означает “модульных курсов сейчас нет”, q13 означает “дат ЗВШ нет, лист ожидания”.

6. `test_v3_q14_q15_closed_with_correct_scope`
   Проверяет, что q14 создаёт проверенный факт УНПК очно 5-11 классы 49 000 / 82 000, а q15 создаёт проверенный факт только для УНПК онлайн олимпиадной подготовки Физтех 9 и 11 классы 41 800 / 69 900. Любые другие онлайн-классы УНПК должны уходить к менеджеру.

7. `test_v3_approval_queue_contains_atomic_business_items`
   Проверяет структуру очереди и наличие price/discount/promocode/deadline/lvsh/program/installment/tax/matkap.

8. `test_v3_brand_scope_blocks_cross_brand_prices`
   Для active_brand `foton` не попадают УНПК-цены; для active_brand `unpk` не попадают Фотон-цены.

9. `test_v3_preserves_brand_rules_and_bot_policy_contract`
   Проверяет, что approved формула связи брендов, `forbidden_client_mentions`, `pilot_mode`, `post_filter_draft_text` не сломаны.

10. `test_v3_stage6_fixture_expected_and_forbidden_terms_are_enforced`
   Проверяет fixtures Клода на expected/forbidden terms.

11. `test_telegram_manager_draft_pilot_can_use_kb_snapshot_when_env_set`
   Проверяет, что live context builder может использовать snapshot, но не отправляет клиенту.

12. `test_v3_team_answers_override_open_questions_without_overpromising`
   Проверяет ответы из `.docx`: договор “в ближайшие дни”, справка “до 10 дней”, преподаватели без ФИО, возврат manager_only, адрес “Скорняжный”, Долями 4 части, два отдельных Telegram-бота.

12. `test_v3_active_brand_does_not_come_from_ambiguous_crm_or_tallanto`
   Проверяет, что при неоднозначной CRM/Tallanto-привязке точные брендовые факты не используются, если active_brand не задан явно.

Расширить существующие:

- `tests/test_telegram_stage6_kb_eval.py`
- `tests/test_telegram_pilot_context_builder.py`
- `tests/test_bot_policy_v2.py`, если нужен общий post-filter.

Безопасные команды:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider \
  tests/test_kb_release_v3_import.py \
  tests/test_telegram_stage6_kb_eval.py \
  tests/test_telegram_pilot_context_builder.py \
  tests/test_bot_policy_v2.py
```

Регрессионно:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider \
  tests/test_knowledge_base.py \
  tests/test_kc_context.py \
  tests/test_manager_answer_playbook.py \
  tests/test_kb_release_v2_import.py \
  tests/test_telegram_stage6_kb_eval.py \
  tests/test_telegram_pilot_eval_pack.py \
  tests/test_telegram_pilot_context_builder.py \
  tests/test_bot_policy_v2.py \
  tests/test_bot_safety_detector.py \
  tests/test_bot_safety_frozen_corpus.py
```

## 10. Acceptance criteria

Работа считается выполненной, если:

1. Создан v3-релиз в отдельной папке, v2 не перезаписан.
2. Все 25 контрольных чисел grep-находимы в `facts_registry`.
3. Нет пустых `fact_text` у значимых фактов.
4. Нет `forbidden_to_say` в `client_safe_text`.
5. Нет allowed client text с номерами лицензий, `70369`, `Л035`, `50Л01`, `№77753`, `АНО ДПО`.
6. Все `source_id` фактов существуют в `source_registry`.
7. У всех source-записей для Claude YAML есть sha256.
8. Все оставшиеся открытые вопросы помечены `needs_owner_confirmation` и не разрешены для клиентского точного ответа.
9. Ответы из `Questions_for_Team_kb_v2_2026-05-17.docx` перенесены в v3 без расширения смысла:
   - формула брендов оставлена;
   - Фотон онлайн второй предмет 30%, УНПК 20%;
   - сотрудники МФТИ 10%, скидки не суммируются;
   - Долями 4 части, рассрочка 6/12 месяцев только через менеджера;
   - возврат manager_only;
   - справка до 10 дней;
   - договор в ближайшие дни;
   - преподаватели без ФИО в общих ответах;
   - Скорняжный;
	   - два отдельных Telegram-бота;
	   - CRM/Tallanto не используются для автоматического определения бренда.
10. Ответы q14/q15 из handoff Клода от 2026-05-18 перенесены без расширения смысла:
    - УНПК очно 49 000 / 82 000 только для 5-11 классов;
    - УНПК онлайн 41 800 / 69 900 только для олимпиадной подготовки Физтех для 9 и 11 классов;
    - по другим онлайн-классам УНПК бот не называет эти цены и передаёт менеджеру.
11. `approval_queue_for_rop_v3.csv` содержит 400+ осмысленных пунктов или quality report явно блокирует релиз из-за меньшего числа.
12. Stage 6 fixtures дают:
    - `brand_separation_violation = 0`;
    - `unsupported_numeric_promises = 0`;
    - `expected_route_hit = 100%` для P0;
    - `became_more_substantive >= 15/20`.
12. Live Telegram pilot умеет использовать KB snapshot через env, но по умолчанию остаётся совместимым.
13. Audit pack создан и содержит:
    - `implementation_notes.md`;
    - `changed_files.txt`;
    - `test_output.txt`;
    - `risk_review.md`;
    - `backward_compatibility.md`;
    - `quality_report_summary.md`;
    - `stage6_summary.md`;
    - `claude_handoff_response.md`.

## 11. Использование субагентов

При реализации использовать до 6 субагентов с `xhigh`:

1. Проверка parser/facts: атомарная развёртка YAML.
2. Проверка brand safety: Фотон/УНПК, forbidden phrases, лицензии.
3. Проверка source registry и sha256.
4. Проверка approval queue v3.
5. Проверка Stage 6 fixtures и Telegram path.
6. Финальный независимый audit pack review.

Субагенты не должны писать в одни и те же файлы одновременно. Если им поручается код, нужно заранее разделить зоны ответственности.

## 12. Порядок реализации

1. Preflight: `git status --short`, зафиксировать незакоммиченные входы.
2. Создать v3-тесты с ожидаемыми падениями.
3. Создать новый v3-сборщик или v3-режим текущего сборщика.
4. Реализовать source registry v3.
5. Реализовать атомарную развёртку фактов.
6. Реализовать запреты `forbidden_to_say`, `_internal`, `internal_only_for_number`.
7. Реализовать open-question handling.
8. Реализовать approval_queue v3.
9. Реализовать quality gates.
10. Подключить Stage 6 fixtures.
11. Добавить env-подключение KB snapshot в live Telegram pilot.
12. Собрать v3-релиз.
13. Прогнать тесты.
14. Собрать audit pack.
15. Подготовить папку для Клода.
16. Только после ревью Клода и Дмитрия решать вопрос о коммите и следующем пилотном запуске.

## 13. Что я бы сделал на месте Дмитрия

Я бы не запускал Telegram-пилот с точными ценами на текущей v2-базе. Сначала нужно закрыть v3-пересборку. При этом не надо останавливать весь проект: можно держать пилот в режиме осторожных черновиков без точных фактов, но полноценные содержательные ответы должны ждать v3.

Главный приоритет: не “ещё больше документов”, а правильная машинная развёртка уже собранной базы. Сейчас самая большая потеря качества не в отсутствии данных, а в том, что данные лежат в YAML правильно, но превращаются в плохой `facts_registry`.
