# ТЗ: Полная обработка Telegram-истории УНПК МФТИ для обогащения базы знаний и customer timeline

Дата: 2026-05-21

Статус: ТЗ к реализации. Реализацию запускать отдельным блоком после подтверждения Дмитрия.

## 1. Цель

Полностью обработать историю Telegram УНПК МФТИ из `TP UNPK DataExport_2026-05-21`, включая тексты, PDF, фото, документы и голосовые, и превратить ее в безопасный read-only пакет кандидатов:

- для обогащения базы знаний;
- для правил работы Telegram-бота;
- для новых тестов;
- для менеджерских черновиков;
- для сопоставления Telegram-диалогов с клиентами;
- для будущего слоя customer timeline.

Важно: результат не должен автоматически менять базу знаний, CRM, Tallanto, AMO или Telegram. Все найденное из Telegram считается кандидатом, а не утвержденным фактом.

## 2. Жесткие ограничения

Нельзя:

- писать в AMO, CRM, Tallanto, Telegram;
- отправлять сообщения клиентам;
- менять базу знаний автоматически;
- менять `stable_runtime` DB/audio/transcripts;
- запускать старые batch/start/run-ui скрипты;
- переносить raw Telegram export в git;
- коммитить персональные CSV/XLSX, медиа, токены, session-файлы;
- считать ответ менеджера истиной без ROP/KB approval.

Разрешено в рамках этого блока:

- читать `TP UNPK DataExport_2026-05-21`;
- содержательно анализировать PDF, фото, документы и голосовые из этого export;
- выполнять локальный OCR/извлечение текста/ASR только в изолированную папку `product_data/telegram_history_analysis/unpk_full_history_enrichment_20260521/`;
- использовать результаты только как masked candidates;
- читать текущую KB, bot rules, autonomy matrix, Tallanto/AMO локальные exports, если они уже лежат в проекте или `_external_handoffs/`.

## 3. Источники

Основной источник:

- `TP UNPK DataExport_2026-05-21/result.json`
- вложения внутри `TP UNPK DataExport_2026-05-21/chats/`

Сравнивать с:

- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/facts_registry.jsonl`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/bot_policy.yaml`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/brand_rules.yaml`
- `docs/TELEGRAM_BOT_AUTONOMY_MATRIX_V1_2026-05-21.md`
- `docs/TELEGRAM_UNPK_HISTORY_AUDIT_2026-05-21.md`
- `_external_handoffs/tallanto_students_export_2026-05-12/Ученики.csv`
- доступные локальные AMO/CRM exports, если они уже есть в проекте и читаются read-only.

## 4. Главный результат

Создать папку:

`product_data/telegram_history_analysis/unpk_full_history_enrichment_20260521/`

Внутри должны быть:

1. `full_message_inventory.csv`
2. `full_session_inventory.csv`
3. `attachment_inventory.csv`
4. `attachment_text_extracts.jsonl`
5. `voice_transcripts.jsonl`
6. `customer_question_candidates.csv`
7. `kb_fact_candidates.csv`
8. `bot_rule_candidates.csv`
9. `faq_candidates.csv`
10. `manager_draft_phrase_candidates.csv`
11. `new_smoke_tests_candidates.jsonl`
12. `p0_route_cases.csv`
13. `conflicts_with_current_kb.csv`
14. `coverage_against_current_kb.csv`
15. `identity_link_candidates.csv`
16. `dialog_customer_match_report.csv`
17. `customer_timeline_candidates.jsonl`
18. `privacy_masking_report.md`
19. `coverage_summary.md`
20. `README.md`

Создать audit pack:

`audits/_inbox/telegram_unpk_full_history_enrichment_20260521/`

Внутри:

- `implementation_notes.md`
- `data_inventory.md`
- `media_processing_report.md`
- `identity_matching_report.md`
- `kb_coverage_report.md`
- `semantic_review.md`
- `risk_review.md`
- `changed_files.txt`
- `test_output.txt`

## 5. Этапы реализации

### Этап 1. Git и границы

1. Проверить `git status --short`.
2. Зафиксировать, какие файлы уже изменены не этим блоком.
3. Не трогать чужие изменения.
4. Все новые результаты писать только в новую папку `product_data/telegram_history_analysis/unpk_full_history_enrichment_20260521/` и новый audit pack.

### Этап 2. Полная инвентаризация истории

Обработать все сообщения из `result.json`, не выборку.

Посчитать:

- чаты;
- сообщения;
- роли;
- период;
- сессии;
- вложения;
- PDF;
- фото;
- документы;
- голосовые;
- пустые сообщения;
- системные сообщения;
- пересланные сообщения;
- реакции;
- edited/reply.

Сохранить:

- `full_message_inventory.csv`
- `full_session_inventory.csv`
- `attachment_inventory.csv`
- `data_inventory.md`

### Этап 3. Извлечение содержимого вложений

Для PDF:

- извлечь текст структурно, если PDF текстовый;
- если PDF скан, применить OCR;
- сохранить только masked summary/extract, не raw полный документ, если документ содержит персональные данные.

Для фото:

- выполнить OCR;
- отдельно классифицировать тип: чек, договор, скриншот, расписание, справка, рекламный материал, неизвестно;
- не сохранять распознанные персональные данные в публичные отчеты.

Для документов:

- `.docx`, `.doc`, `.pdf` читать локально;
- извлекать только полезные факты-кандидаты и masked фрагменты.

Для голосовых:

- выполнить ASR локально, только для Telegram export;
- не использовать `stable_runtime` ASR;
- не писать результаты в `stable_runtime`;
- сохранять transcript только masked;
- если ASR не доступен или качество плохое, помечать `transcript_status=failed/low_confidence`.

Сохранить:

- `attachment_text_extracts.jsonl`
- `voice_transcripts.jsonl`
- `media_processing_report.md`

### Этап 4. Полное выделение клиентских вопросов и сценариев

По всей истории и извлеченным вложениям выделить:

- клиентские вопросы;
- уточнения;
- жалобы;
- возвраты;
- оплату;
- документы;
- СФР/маткапитал;
- скидки;
- расписание;
- лагеря/смены;
- доступы/ссылки/домашние задания;
- брендовые смешения УНПК/Фотон/ЦДПО;
- не по теме.

Каждый кейс должен иметь:

- `dialog_id`;
- `session_id`;
- `message_id`;
- `date`;
- `source_type`: text/pdf/photo/doc/voice;
- `masked_client_id`;
- `question_text`;
- `context_before`;
- `manager_answer`;
- `theme_id`;
- `risk_level`;
- `recommended_route`;
- `needed_fact`;
- `source_ref`;
- `confidence`;
- `notes`.

Важно: в первом полном проходе реальные Telegram-кейсы не помечать как автономные. Только:

- `manager_only`;
- `draft_for_manager`;
- `candidate_for_future_autonomy_after_approval`.

### Этап 5. Сравнение с текущей KB и правилами бота

Для каждого вопроса/кейса определить:

- `covered`;
- `partially_covered`;
- `missing_fact`;
- `missing_rule`;
- `missing_template`;
- `conflict_with_kb`;
- `live_status_only`;
- `manager_only_by_policy`;
- `brand_ambiguous`.

Сверять против:

- facts registry;
- bot policy;
- brand rules;
- autonomy matrix;
- текущих smoke tests, если они есть.

Сохранить:

- `coverage_against_current_kb.csv`
- `conflicts_with_current_kb.csv`
- `kb_coverage_report.md`

### Этап 6. Кандидаты на обогащение KB

Собрать только кандидаты, не изменяя KB.

`kb_fact_candidates.csv`:

- `candidate_id`;
- `brand`;
- `fact_type`;
- `candidate_fact_text`;
- `source_dialog_ref`;
- `source_message_ref`;
- `source_type`;
- `evidence_summary`;
- `risk_level`;
- `confidence`;
- `requires_rop_confirmation`;
- `requires_document_confirmation`;
- `can_be_client_safe_after_approval`;
- `why_needed`;
- `existing_kb_status`;
- `notes`.

Обязательные категории:

- оплата: QR, квитанция, назначение платежа, чек;
- частичная оплата/разбивка платежа;
- оформление скидки;
- маткапитал/СФР;
- договоры/документы;
- доступы/ссылки/ЛК/чаты/ДЗ/записи;
- расписание/наличие мест/лист ожидания;
- лагеря/смены;
- брендовая путаница;
- возвраты/жалобы как manager-only правила.

### Этап 7. Кандидаты в правила бота

`bot_rule_candidates.csv`:

- P0-гейты;
- правила автономности;
- правила передачи менеджеру;
- правила brand gate;
- правила PII;
- правила статуса оплаты/документов;
- правила вложений.

Минимальные обязательные правила:

- возврат всегда `manager_only`;
- юридическая угроза всегда `manager_only`;
- жалоба/негатив всегда `manager_only` или ручной черновик;
- статус оплаты/чека/документов/СФР только после проверенного контекста;
- УНПК/Фотон/ЦДПО не смешивать;
- не просить лишние персональные данные в Telegram;
- не использовать вложения как факт без проверки.

### Этап 8. Кандидаты в FAQ и менеджерские черновики

`faq_candidates.csv`:

- вопрос;
- безопасный краткий ответ-кандидат;
- какие факты нужны;
- какие факты уже есть;
- чего не хватает;
- можно ли после approval использовать клиенту.

`manager_draft_phrase_candidates.csv`:

- исходный вопрос;
- хорошая формулировка менеджера;
- почему полезна;
- что надо подтвердить;
- что нельзя переносить;
- reusable pattern.

Все реальные ответы менеджеров по умолчанию:

- `safe_for_bot=no`;
- `needs_rop_confirmation=yes`.

### Этап 9. Identity matching

Сопоставить Telegram-диалоги с клиентами read-only.

Источники признаков:

- phone из текста;
- email из текста;
- Telegram username;
- Telegram user_id;
- имя Telegram;
- контакты из Telegram export;
- Tallanto `Ученики.csv`: телефон, email, Telegram, Telegram ID, подписка в Telegram;
- локальные AMO/CRM exports, если доступны read-only.

Классы совпадений:

- `strong_unique`;
- `probable`;
- `ambiguous`;
- `unmatched`;
- `conflict`.

Запрещено:

- считать совпадение по одному имени `strong_unique`;
- писать результаты в CRM/Tallanto;
- выводить персональные данные в публичные отчеты.

Сохранить:

- `identity_link_candidates.csv`
- `dialog_customer_match_report.csv`
- `identity_matching_report.md`

### Этап 10. Customer timeline candidates

Сформировать read-only события:

- `telegram_dialog`;
- `telegram_session`;
- `telegram_message`;
- `telegram_attachment`;
- `telegram_question`;
- `telegram_manager_answer`;
- `telegram_identity_link`;
- `telegram_kb_gap`;
- `telegram_p0_case`.

Формат:

`customer_timeline_candidates.jsonl`

Каждая строка:

- `event_id`;
- `event_type`;
- `event_time`;
- `masked_customer_ref`;
- `identity_confidence`;
- `source_channel=telegram_unpk`;
- `source_ref`;
- `summary`;
- `risk_flags`;
- `kb_candidate_refs`;
- `route`;
- `raw_text_stored=false`;
- `pii_masked=true`.

### Этап 11. Тесты и проверки

Нужны минимум:

- CSV/JSONL parse tests;
- no raw email/phone/TG handle/raw Telegram ID in public outputs;
- no `bot_answer_self_for_pilot` in real Telegram-derived first-pass rows;
- P0 cases route to `manager_only`;
- brand ambiguity not autonomous;
- payment/status/docs not autonomous;
- media extraction outputs masked;
- identity matching does not create `strong_unique` from name-only match;
- idempotency: повторный запуск дает стабильные IDs and counts;
- no writes to AMO/Tallanto/CRM/Telegram;
- no writes to `stable_runtime`.

### Этап 12. Semantic review

Обязателен отдельный смысловой аудит.

Проверить:

- не смешаны ли УНПК и Фотон;
- не объявлены ли Telegram-ответы менеджеров истиной;
- нет ли опасных цен/дат/мест/скидок без approval;
- нет ли автономности по P0;
- нет ли персональных данных;
- полезны ли кандидаты для РОПа и базы знаний;
- есть ли реальные новые кандидаты или только сценарии.

Вердикты:

- `formal_pass`;
- `semantic_pass`;
- `pilot_ready`;
- `production_ready`.

Для этого блока допустимый финальный статус: `formal_pass` + `semantic_pass_for_candidates`.

Нельзя писать `готово к использованию` для бота, пока кандидаты не утверждены.

## 6. Субагенты для реализации

При реализации использовать до 6 субагентов с `xhigh`.

Разделение:

1. Субагент 1: полный Telegram parser, роли, сессии, message inventory.
2. Субагент 2: PDF/photo/doc extraction, OCR, attachment inventory.
3. Субагент 3: voice ASR, transcript quality, voice summary.
4. Субагент 4: KB/bot policy/brand rules coverage and conflicts.
5. Субагент 5: identity matching with Tallanto/AMO/local exports.
6. Субагент 6: semantic/privacy QA, P0 gates, final audit.

Ограничение для субагентов:

- не менять `stable_runtime`;
- не писать в CRM/Tallanto/Telegram/AMO;
- не менять KB;
- не коммитить raw exports;
- писать только в assigned output folders.

## 7. Критерии приемки

Работа считается выполненной только если:

- обработаны все сообщения из `result.json`;
- все вложения проинвентаризированы;
- все читаемые PDF/doc/photo/voice обработаны или помечены понятной причиной отказа;
- создан полный набор output-файлов;
- создан audit pack;
- нет raw PII в публичных CSV/JSONL/MD;
- нет автономных route для реальных Telegram-derived строк;
- все KB candidates имеют `requires_rop_confirmation`;
- identity links имеют confidence class;
- customer timeline candidates сформированы;
- все тесты/валидации пройдены;
- semantic review не нашел блокеров;
- итоговый отчет честно разделяет: новые факты-кандидаты, новые сценарии, новые формулировки, пробелы, конфликты.

## 8. Что делать при блокерах

Не останавливаться на первом затруднении.

Если OCR/ASR не работает для части файлов:

- обработать все остальные;
- проблемные файлы занести в `media_processing_report.md`;
- указать `failed_reason`;
- сохранить counts.

Если identity matching неоднозначный:

- не гадать;
- ставить `ambiguous`;
- объяснять конфликт.

Если KB conflict найден:

- не исправлять KB;
- занести в `conflicts_with_current_kb.csv`;
- предложить ROP question.

Если персональные данные не удается безопасно замаскировать:

- не включать текст в public output;
- сохранить только source ref и reason.

## 9. Финальный отчет Дмитрию

В конце реализации дать короткий отчет:

- сколько сообщений обработано;
- сколько вложений обработано;
- сколько PDF/photo/doc/voice успешно извлечено;
- сколько вопросов найдено;
- сколько KB candidates;
- сколько rule candidates;
- сколько FAQ candidates;
- сколько draft phrase candidates;
- сколько identity links по классам;
- сколько customer timeline events;
- какие P0 темы найдены;
- какие raw artifacts не коммитить;
- какие файлы безопасны для коммита;
- что требует решения РОПа.
