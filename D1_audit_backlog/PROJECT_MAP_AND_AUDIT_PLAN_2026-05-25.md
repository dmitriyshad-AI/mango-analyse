# Карта проекта Mango Analyse + что Claude может проверить. 2026-05-25 (read-only)

Репозиторий — не «бот», а платформа: 267 py-модулей в src (~190k строк), 258 тест-файлов.
Ниже: крупные части (описание + состояние + покрытие тестами) и классификация по тому,
что я (Claude) могу проверить сам, для чего написать готовые тесты Кодексу, где — только совет.

## Крупные части (описание / состояние)

1. channels/ (27.8k строк, 15 тестов «channels» + 16 «telegram») — ТЕЛЕГРАМ-БОТ-ЧЕРНОВИК.
   Распознавание, извлечение фактов, P0, held-состояние, шаблоны, генерация, маршрут, отчётность.
   Состояние: активная разработка; round-5 = FAIL0/hard-gate0/recall7-7, «отправил бы» 21%, слой человечности в работе. ЭТО наш текущий фокус.

2. quality/ (13.1k, тесты transcript/hard/crm/bot) — ДЕТЕКТОРЫ КАЧЕСТВА И ГЕЙТЫ.
   bot_safety_detector, crm_text_quality_detector, transcript_quality_* (много), hard_gate_*.
   Состояние: рабочая QA-машина для транскриптов, CRM-текста и безопасности бота. Детерминированные детекторы.

3. amocrm_runtime/ (6.8k, тесты amo/tallanto) — ИНТЕГРАЦИЯ AMO/TALLANTO.
   auth, db, deals, tallanto_api/matching/ranking, phone_context, deal_dossier.
   Состояние: чтение + сопоставление; запись закрыта гейтами (проверено: dry-run + флаги).

4. customer_timeline/ (14.9k, 16 тестов customer) — СВЯЗАННАЯ ИСТОРИЯ КЛИЕНТА (фундамент).
   ingestion, read_api, approval_workspace, canonical_readonly_import, safety (PII).
   Состояние: read-only импорт + ручная приёмка; основа для реактивации/timeline.

5. deal_aware/ (5.4k, 11 тестов deal) — КЛАССИФИКАЦИЯ И ЗАКРЫТИЕ СДЕЛОК.
   deal_state_classifier, deal_quality_gate, deal_text_builder, deal_writeback, amo_rollback.
   Состояние: черновики закрытия + защищённая запись (гейт Дмитрия), есть откат.

6. services/ (9.3k, тесты analyze/llm) — КОНВЕЙЕР ОБРАБОТКИ ЗВОНКОВ.
   transcribe(ASR), analyze, resolve, ingest, sync_amocrm, worker, export_*.
   Состояние: ASR и запись в AMO под гейтами; sync_amocrm dry-run.

7. productization/ (43.5k, 85 тестов!) — SAAS/ОПЕРАЦИИ (самая большая).
   ASR worker pipeline (под approval-гейтами), AMO-дедупликация, запись звонков, scheduler,
   tenant isolation, mango_office, capture. Состояние: много ops/pipeline; опасное (ASR, запись) гейтировано.

8. question_catalog/ (7.6k, 15 тестов question) — КАТАЛОГ ВОПРОСОВ/ТЕМ.
   classifier, theme_assigner_llm, rop_questionnaire, calibration_metrics, extractors.
   Состояние: классификация вопрос→тема, калибровка.

9. insights/ (5.7k) — ИЗВЛЕЧЕНИЕ/АНАЛИТИКА: outcome_linker, phone_identity, readiness, sanitizers, rop_validation_pack.

10. knowledge_base/ (3.9k, 8 тестов kb) — МАШИНЕРИЯ БАЗЫ ЗНАНИЙ: fact_registry, answer_registry, kc_context, drive_inventory.

11. clients/ (255) — API-клиенты amocrm/ollama. utils/ (361) — audio/phone/filename. maintenance/ (1.5k) — инвентаризация.

Гейты безопасности: 54 модуля ссылаются на dry_run/approval/write-гейты → опасное (запись, ASR) закрыто по дизайну.

## Классификация

### A. Могу проверить САМ (read-only, детерминированно/бизнес-смыслово) + написать офлайн-тесты
- channels/ бот целиком (делаю): распознавание, извлечение, P0, held, человечность, KB-гигиена, бренды, маршрут.
- knowledge_base/ + product_data KB: консистентность фактов, рассинхрон снимков, скидки-условия, valid_until, бренд-утечки, служебные маркеры. (частично сделано)
- quality/ детекторы bot_safety_detector + crm_text_quality_detector + non_conversation: это детерминированные детекторы — соберу бизнес-банки фраз (как P0-перифразы) и измерю recall/precision сам.
- question_catalog/ classifier + theme_assigner: соберу размеченный банк «вопрос→ожидаемая тема» и измерю точность классификации офлайн.
- insights/sanitizers: банк «вход→должно быть вычищено» (PII/служебное) — проверю сам.

### B. Могу написать ГОТОВЫЕ тесты для других диалогов Кодекса (они интегрируют и гоняют в своей среде)
- amocrm_runtime/ tallanto_matching + deal_attribution + crm_entity_resolver + phone_context:
  тест-пакет на синтетических записях (правильное сопоставление; НЕ матчить разных людей) + контракт «без записи без гейта».
- deal_aware/ writeback + quality_gate + rollback: приёмочные тесты «запись только при approval/dry_run выключен», «откат восстанавливает», «низкое качество → не пишем».
- customer_timeline/ identity resolution + safety: бизнес-тесты «верное связывание личности», «PII не утекает в клиентский/preview слой», «импорт строго read-only».
- services/ resolve/analyze/sanitize: банки «вход→ожидаемый разбор/чистка»; контракт sync_amocrm dry-run.
- productization/ контракты-гейты: «ASR не исполняется без approval-record», «tenant isolation не пересекает арендаторов», «capture идемпотентен». (контрактные тесты, без живого ASR)
- crm_writeback quality/recall детекторы (quality/): банк «текст→должно/не должно пройти в CRM».

### C. Могу только ДАТЬ СОВЕТ (сам не протестирую — нужна живая среда/данные)
- Живые интеграции: реальные AMO/Tallanto/Mango Office API, ASR (аудио), scheduler runtime, скачивание записей — нет доступа/кредов/аудио, и трогать нельзя. Могу читать код на контракты безопасности (гейты записи, идемпотентность, обработка ошибок) и советовать.
- Генерация end-to-end на gpt-5.5 (живое качество ответов) — нет модели.
- Производительность/масштаб, миграции БД, деплой/инфра.

## Рекомендованный приоритет новых проверок (по бизнес-риску)
1. quality/bot_safety_detector + crm_text_quality_detector — гейтят то, что доходит до клиента/CRM. (A — соберу банки сам)
2. amocrm_runtime/tallanto_matching + customer_timeline identity — неверное сопоставление = не тот клиент привязан (реальный вред). (B — тест-пакет Кодексу)
3. deal_aware writeback-гейты — запись в CRM самый рискованный шаг. (B — контрактные тесты)
4. question_catalog классификатор тем — влияет на маршрут/факты. (A — размеченный банк)
5. insights/sanitizers — чтобы PII/служебное не утекало. (A)

Дальше — productization контракты-гейты (B) и советы по живым интеграциям (C).
