# Аудит качества ASR/Resolve/Analyze и план безопасных правок

Дата: 2026-05-09.

Контекст: повторная проверка после аудита `stable_runtime/claude_telegram_bot_transcript_audit_20260509/CLAUDE_AUDIT_RESULT.md`.

## Краткий вывод

Проблема не сводится к "модель 5.4-mini недостаточно умная" и не решается повторным ASR.

Основная причина - слабые жесткие guardrails между слоями:

1. ASR иногда возвращает автоответчик, голосовую почту или артефакты тишины как текст.
2. Resolve часто `skipped` для таких звонков и не является смысловым классификатором "был ли живой клиентский диалог".
3. Analyze имеет слишком узкий набор `non_conversation`-маркеров.
4. Если Analyze не поставил `non_conversation`, readiness считает звонок содержательным только по `call_type`.
5. Pilot extraction, LLM review, Knowledge Base и ROP pack дальше воспринимают этот звонок как валидный sales/service moment.

Итог: недозвоны и автоответчики иногда становятся "риском потери выручки", "черновиком для бота" или "процессной проблемой", хотя коммерчески это другой класс данных.

## Проверенные источники

- `src/mango_mvp/services/analyze.py`
- `src/mango_mvp/services/resolve.py`
- `src/mango_mvp/services/transcribe.py`
- `src/mango_mvp/insights/readiness.py`
- `src/mango_mvp/insights/pilot_extraction.py`
- `src/mango_mvp/insights/llm_review.py`
- `src/mango_mvp/insights/knowledge_base.py`
- `src/mango_mvp/insights/rop_validation_pack.py`
- `stable_runtime/final_processing_coverage_report_20260507_v5/summary.json`
- `stable_runtime/insight_readiness_report_20260507/calls_terminal_analyzed.csv`
- `stable_runtime/pilot_sales_moments_20260507/sales_moments.csv`
- `stable_runtime/sales_insight_knowledge_base_20260507_v2/enriched_reviews.csv`
- `stable_runtime/rop_validation_pack_20260507_v1/rop_validation.csv`
- `stable_runtime/claude_telegram_bot_transcript_audit_20260509/CLAUDE_AUDIT_RESULT.md`

## Масштаб по текущим артефактам

По `stable_runtime/final_processing_coverage_report_20260507_v5/summary.json`:

- Исходных аудио в окне 2025-01-01 - 2026-05-31: 64 867.
- Исключено из ASR: 35.
- Actionable audio: 64 832.
- ASR done: 64 832.
- Full R+A: 64 832.

По `stable_runtime/insight_readiness_report_20260507/summary.json`:

- Terminal analyzed calls: 64 832.
- Contentful calls: 51 429.
- Non-conversation calls: 13 403.
- Unique client phones: 15 924.

Эвристический повторный скан `calls_terminal_analyzed.csv`:

- `history_summary` с маркерами недозвона/голосовой почты: 3 810.
- `history_summary` с ASR-артефактами: 5.
- `contentful=True` + маркеры недозвона/голосовой почты/артефакта: 3 416.
- Из них `technical_call`: 1 866.
- Из них `service_call`: 1 250.
- Из них `sales_call`: 288.
- Из них `existing_client_progress`: 12.
- `contentful=True` + подозрительный текст + заполненный `next_step`: 2 011.

Важно: 3 416 - это не финальное число ошибок. Это candidate set для dry-run/backfill. Часть может оказаться реальными звонками, где недозвон упоминается в истории контакта. Поэтому исправление должно быть двухступенчатым: сначала безопасный классификатор с причинами, потом dry-run, потом выборочная проверка.

По `stable_runtime/pilot_sales_moments_20260507/sales_moments.csv`:

- Sales moments всего: 2 734.
- Moments с no-live/artifact-маркерами в полях: 148.
- Moments с `service_call`/`technical_call` и no-live/artifact-маркерами: 51.
- `non_conversation` среди moments: 0.

По `stable_runtime/sales_insight_knowledge_base_20260507_v2/enriched_reviews.csv`:

- Review всего: 2 734.
- `answer_pattern=no_live_contact_or_voicemail`: 44.
- Из них ошибочно `commercial_usefulness=revenue_leakage_risk`: 20.
- Bot-ready/needs-validation no-live: 0.
- `ideal_answer_example` с brand-risk: 20.
- `ideal_answer_example` с деньгами/скидками/рассрочкой/возвратом/дедлайном: 661.
- Bot-ready/needs-validation с такими коммерческими условиями: 630.

По `stable_runtime/rop_validation_pack_20260507_v1/rop_validation.csv`:

- Combined rows: 731.
- Строки с no-live-маркерами: 41.
- Строки с ASR-артефактами: 1.
- `Риск потери выручки` + no-live/artifact: 29.
- `P0` + no-live/artifact: 8.
- `Идеальный ответ` с brand-risk: 3.
- `Идеальный ответ` с деньгами/скидками/рассрочкой/возвратом/дедлайном: 178.
- Bot-candidate rows с такими коммерческими условиями: 94.
- Top-answer rows с ответом менеджера короче 200 символов: 23.

## Подтвержденная цепочка дефекта

### 1. ASR

Whisper/GigaAM иногда дают текст там, где живого разговора нет:

- `Абонент не ответил...`
- `Вызываемый абонент не отвечает... голосовой почтовый ящик...`
- `Субтитры сделал DimaTorzok`
- `Kim Kim Kim...`
- `Olá...`
- `Продолжение следует...`

Двойной ASR помогает выбрать лучший вариант, но не является надежным детектором "живой диалог / не живой диалог".

### 2. Resolve

На проблемных примерах статус часто `resolve_status=skipped`. Это нормальное техническое состояние для части звонков, но нельзя ожидать, что Resolve исправит смысловую классификацию.

Resolve должен остаться слоем сборки/сверки вариантов ASR, а не единственной защитой от no-dialogue.

### 3. Analyze

В `STRONG_NON_CONVERSATION_MARKERS` сейчас не хватает реальных формулировок:

- `абонент не ответил`
- `вызываемый абонент не отвечает`
- `голосовой почтовый ящик`
- `голосовая почта`
- `оставить сообщение`
- `после звукового сигнала`
- `не можем оставить голосовое сообщение`

Также есть две усиливающие ошибки:

- `_candidate_next_step_action()` видит слово `перезвон` в автоответчике и ставит `Перезвонить клиенту`.
- `_detect_preferred_channel()` видит `почт` в `голосовой почтовый ящик` и ставит `email`.

### 4. Readiness

`CallCandidate.contentful` сейчас фактически означает: `call_type != non_conversation`.

Это слишком слабое правило. Если Analyze ошибся и поставил `service_call`/`technical_call`, readiness считает звонок содержательным, даже если `history_summary` прямо говорит "живого диалога не было".

### 5. Pilot extraction

`_is_contentful()` в `pilot_extraction.py` также опирается на `contentful/call_type`, не проверяя transcript/history markers.

В результате no-live звонки попадают в `sales_moments.csv`.

### 6. LLM review

LLM review часто видит, что диалога не было, но схема заставляет его дать `ideal_answer_example`.

Для no-live нужна отдельная ветка: `not_a_sales_moment`, без идеального ответа для менеджера и без seed для бота.

### 7. Knowledge Base

`classify_answer_pattern()` умеет поставить `no_live_contact_or_voicemail`, но `commercial_usefulness()` сначала проверяет `score < 55 and lost/follow_up`, и только потом process patterns.

Из-за порядка условий no-live с низкой оценкой становится `revenue_leakage_risk`.

### 8. ROP pack

`revenue_risks` берутся по `commercial_usefulness=revenue_leakage_risk`, поэтому no-live может попасть в `Риски потери выручки`.

Сортировка и priority label частично пытаются понизить no-live, но это поздно: строка уже попала не в тот лист.

## Дополнительные проблемы, найденные сверх предыдущего сообщения

### A. Ложный `email` из голосовой почты

Пример из readiness:

`Абонент недоступен, звонок перенаправлен на голосовую почту... Контакты: канал: email.`

Причина: `_detect_preferred_channel()` ищет подстроку `почт`, поэтому `голосовая почта` и `голосовой почтовый ящик` превращаются в email.

Риск: в CRM/таблицах может появляться ложный preferred channel.

### B. Ложный next step из автоответчика

Пример:

`абонент сейчас не может ответить и просит перезвонить позднее` -> `next_step=Перезвонить позднее`, `follow_up_score=70`.

Причина: next-step heuristics не отличают системный голос от договоренности с клиентом.

Риск: недозвон выглядит как согласованный следующий шаг.

### C. `service_call`/`technical_call` переоцениваются как contentful

Большая часть подозрительных contentful-звонков попала именно в `technical_call` и `service_call`.

Причина: голосовая почта, почтовый ящик, ссылка, сообщение, "оставить" и похожие слова пересекаются с сервисной/технической лексикой.

Риск: сервисная аналитика и очередь РОПа загрязняются no-dialogue.

### D. `sales_call` тоже иногда заражен no-live

Эвристический скан нашел 288 `sales_call` с no-live-маркерами в history summary.

Вероятное объяснение: в тексте рядом могут быть продуктовые/продажные слова из шаблонной речи менеджера или LLM-summary, хотя живого клиента нет.

Риск: reactivation/opportunity выборки получают недостоверные sales moments.

### E. Top answers и bot seeds смешивают разные критерии

23 top-answer строки имеют короткий ответ менеджера (<200 символов). Это не всегда ошибка, но для "лучших ответов для скриптов" короткое `хорошо, перезвоню` редко является обучающим примером.

Риск: РОП видит слишком много слабых/коротких примеров в листе, который должен быть эталонным.

### F. Bot seeds содержат протухающие коммерческие условия

В KB v2 найдено 630 bot-ready/needs-validation строк с деньгами, скидками, рассрочками, возвратом или дедлайнами в `ideal_answer_example`.

Риск: Telegram-бот начнет обещать старые цены, акции, условия возврата или сроки.

### G. Brand normalization нужна не только в bot layer

Brand-risk найден не только в расшифровках, но и в `ideal_answer_example`.

Риск: бот или скрипты могут использовать неправильное название центра.

### H. Требуется backfill не только Excel, но и производных слоев

Если поправить только ROP Excel, ошибка останется в:

- `analysis_json`;
- `calls_terminal_analyzed.csv`;
- `client_chains.csv`;
- `pilot_sales_moments.csv`;
- `llm_review` outputs;
- `sales_insight_knowledge_base`;
- будущих AMO/writeback и bot exports.

Правильное исправление: код генераторов + безопасный backfill/regen всех производных артефактов.

## Что нужно изменить

### P0. Общий классификатор no-live/no-dialogue

Сделать единый модуль, например:

`src/mango_mvp/quality/non_conversation.py`

Функции:

- `detect_non_conversation_signals(text) -> NonConversationSignals`
- `is_strong_no_live(text) -> bool`
- `is_asr_artifact(text) -> bool`
- `has_substantive_live_dialogue(text) -> bool`
- `explain_non_conversation(text) -> list[str]`

Этот модуль должны использовать:

- Analyze;
- readiness;
- pilot_extraction;
- llm_review;
- knowledge_base;
- rop_validation_pack;
- future AMO/bot exports.

Ключевой принцип: не размазывать regex по разным файлам.

### P0. Analyze guardrails

Изменить:

- `STRONG_NON_CONVERSATION_MARKERS`;
- `_is_non_conversation()`;
- `_detect_call_type()`;
- `_candidate_next_step_action()`;
- `_detect_preferred_channel()`;
- `_normalize_analysis()`.

Правила:

1. Сильный no-live + нет содержательного живого диалога => `call_type=non_conversation`.
2. No-live не может иметь `follow_up_score > 0`.
3. No-live не может иметь `lead_priority=warm/hot`.
4. No-live не может иметь `next_step`, кроме отдельного process flag `callback_needed_after_no_answer`, не используемого как договоренность с клиентом.
5. `голосовая почта` / `почтовый ящик` не может давать `preferred_channel=email`.

### P0. Readiness contentful v2

`contentful` должен учитывать:

- `call_type`;
- `quality_flags`;
- `history_summary`;
- `transcript_text` при доступности;
- no-live/artifact reasons.

Новые поля в `calls_terminal_analyzed.csv`:

- `contentful_v2`;
- `contentfulness_reason`;
- `non_conversation_reason_codes`;
- `asr_artifact_flag`;
- `no_live_flag`;
- `live_dialogue_evidence_level`.

Старое `contentful` можно оставить для совместимости, но новые генераторы должны читать `contentful_v2`.

### P0. Pilot extraction gate

`_is_contentful()` должен:

- исключать no-live/artifact по общему классификатору;
- не выбирать звонки с `live_dialogue_evidence_level=none`;
- не выбирать `sales_moment_excerpt` для bot seeds без флага достаточного контекста.

### P0. LLM review schema

Добавить допустимый исход:

- `review_status=not_a_sales_moment`;
- `not_a_sales_moment_reason`;
- `ideal_answer_example=""`;
- `bot_seed_status=exclude_no_dialogue`;

LLM не должен быть вынужден сочинять "идеальный ответ" на автоответчик.

### P0. Knowledge Base

Изменить порядок в `commercial_usefulness()`:

1. Сначала hard-exclude `no_live_contact_or_voicemail`.
2. Потом bot/process/revenue rules.

Для no-live:

- `commercial_usefulness=process_no_live_contact`;
- не `revenue_leakage_risk`;
- не `playbook_candidate`;
- не bot seed.

Также добавить sanitizer:

- `ideal_answer_for_manager`;
- `safe_answer_for_bot`;
- `bot_safety_flags`;
- `brand_risk_flag`;
- `money_or_discount_flag`;
- `legal_or_refund_flag`;
- `deadline_or_promise_flag`;
- `personal_data_flag`.

### P0. ROP pack

Добавить отдельный лист:

- `Недозвоны / non-conversation`

Исключить no-live из:

- `Риски потери выручки`;
- `ТОП ответы для скриптов`;
- `Черновики для бота`.

В `Сводка` явно показывать:

- сколько строк исключено как no-live;
- сколько строк исключено из bot seeds по цене/скидке/юридическим условиям;
- сколько строк содержит brand-risk.

### P1. Backfill для 60k+ без повторного ASR

Повторный ASR не нужен на первом этапе.

Нужен скрипт:

`scripts/backfill_non_conversation_quality_v2.py`

Режимы:

- `--dry-run`: ничего не пишет, считает кандидатов и причины.
- `--write-sidecar`: пишет sidecar CSV/JSONL без изменения DB.
- `--apply`: обновляет `analysis_json.quality_flags` и совместимые поля только после dry-run approval.

Sidecar output:

- `source_db`;
- `source_filename`;
- `old_call_type`;
- `new_call_type_candidate`;
- `old_contentful`;
- `new_contentful_candidate`;
- `reason_codes`;
- `risk_level`;
- `requires_manual_review`;
- `suggested_action`.

Сначала применять sidecar-подход, а не массовую запись в SQLite.

### P1. Regeneration plan

После кода и dry-run:

1. Собрать `non_conversation_backfill_candidates_v1.csv`.
2. Проверить выборку:
   - 50 high-confidence no-live;
   - 50 borderline;
   - 50 старых contentful, которые классификатор НЕ должен трогать.
3. Применить backfill только для high-confidence.
4. Пересобрать readiness v2.
5. Пересобрать outcome linkage v2.
6. Пересобрать pilot sales moments v2.
7. Прогнать LLM-review только на новых/затронутых moments, а не на все 2 734, если moment-id и source filename позволяют кешировать.
8. Пересобрать KB v3.
9. Пересобрать ROP validation pack v2.
10. Сравнить v1/v2:
    - сколько строк ушло из revenue risk;
    - сколько строк ушло из bot seeds;
    - сколько top answers осталось;
    - не просели ли реальные содержательные sales/service examples.

### P1. Regression tests

Добавить тесты:

- `tests/test_non_conversation_quality.py`
- `tests/test_analyze.py`
- `tests/test_analysis_schema.py`
- `tests/test_insight_readiness.py`
- `tests/test_pilot_extraction.py`
- `tests/test_knowledge_base.py`
- `tests/test_rop_validation_pack.py`

Обязательные кейсы:

1. `Абонент не ответил... голосовое сообщение оставить нельзя` => `non_conversation`.
2. `Вызываемый абонент не отвечает... голосовой почтовый ящик... после звукового сигнала` => `non_conversation`.
3. `Субтитры сделал DimaTorzok` => ASR artifact, exclude.
4. `Kim Kim Kim...` => ASR artifact, exclude.
5. `Olá... Абонент сейчас не может ответить` => no-live/artifact, exclude.
6. `Попробуйте перезвонить позднее` из автоответчика не создает `next_step`.
7. `голосовой почтовый ящик` не создает `preferred_channel=email`.
8. Живой technical call с `ссылка не работает` не должен стать `non_conversation`.
9. Живой service call с `оплата/расписание/перенос занятия` не должен стать `non_conversation`.
10. Живой transfer call с `оставайтесь на линии` не должен стать `non_conversation`.
11. No-live не может попасть в `revenue_leakage_risk`.
12. No-live не может попасть в `bot_knowledge_seeds`.
13. Bot-safe answer не содержит рублей, скидок, дат акций, возврата и неправильного бренда.

### P1. Quality gates "ничего не стало хуже"

Перед применением:

- Все существующие тесты проходят.
- Новые regression tests проходят.
- Dry-run показывает не только количество candidates, но и примеры по каждому reason code.
- На ручной проверке borderline sample не более 5-10% ложных no-live.

После применения:

- Количество `sales_call` не должно резко упасть без объяснения.
- Количество contentful не должно падать больше, чем объем high-confidence no-live.
- Реальные service/technical calls из существующих тестов остаются contentful.
- Top answers не должны очищаться до слишком маленького объема.
- Bot seeds не должны содержать:
  - неправильный бренд;
  - конкретные рублевые суммы;
  - скидки/проценты;
  - дедлайны акций;
  - юридические обещания возврата;
  - личные имена клиентов без плейсхолдера.

### P2. Data model

В перспективе лучше не перетирать старое значение, а хранить:

- `call_type_original`;
- `call_type_v2`;
- `contentful_original`;
- `contentful_v2`;
- `quality_version`;
- `quality_reason_codes`.

Для текущего локального проекта можно начать с `analysis_json.quality_flags` + sidecar CSV.

## Почему не нужен повторный ASR

ASR уже выполнен по всем actionable 64 832 звонкам. Проблема не в отсутствии текста, а в интерпретации/маршрутизации текста.

Повторный ASR:

- дорогой по времени;
- не гарантирует исчезновение автоответчиков;
- не исправит downstream-логику `contentful`, `commercial_usefulness`, `ROP pack`.

Нужен replay/backfill от Analyze-derived layers:

- reclassify no-live/artifacts;
- regenerate readiness;
- regenerate pilot/KB/ROP;
- не трогать аудио и базовые transcript variants.

## Рекомендуемый порядок выполнения

1. Реализовать общий `non_conversation` quality module.
2. Покрыть модуль тестами на реальные тексты из аудита.
3. Подключить модуль в Analyze.
4. Подключить модуль в readiness и pilot extraction.
5. Исправить Knowledge Base no-live ordering.
6. Исправить ROP pack: отдельный лист no-live, исключение из revenue/bot/top.
7. Добавить bot safety sanitizer.
8. Сделать dry-run backfill по всем 64 832 звонкам.
9. Проверить sample кандидатов.
10. Применить high-confidence backfill.
11. Пересобрать readiness/outcome/pilot/KB/ROP.
12. Провести сравнительный аудит v1/v2.

## Что пока не делать

- Не запускать повторный ASR.
- Не править Excel вручную.
- Не менять массово SQLite без dry-run и sidecar.
- Не отдавать текущий ROP pack v1 как основу Telegram-бота.
- Не использовать `ideal_answer_example` как bot seed без отдельного safe-answer sanitizer.

## Итоговый статус

Текущий pipeline ценен и масштабно обработал все actionable звонки, но слой отбора "что является содержательным sales/service моментом" требует усиления.

Главные новые находки после повторного анализа:

1. Ошибка заметна уже на readiness-уровне: 3 416 contentful candidates с no-live/artifact признаками.
2. Есть отдельный ложный `email` из `голосовой почты`.
3. Есть массовый ложный `next_step` из автоответчика.
4. В KB no-live может стать `revenue_leakage_risk` из-за порядка условий.
5. Bot seed безопасность требует отдельной версии ответа, а не переиспользования `ideal_answer_example`.
6. Исправлять нужно не только ROP Excel, а весь производный слой от Analyze до ROP/бота.
