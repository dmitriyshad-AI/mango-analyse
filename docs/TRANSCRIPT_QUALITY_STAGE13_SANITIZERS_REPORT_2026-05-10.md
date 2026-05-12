# Stage 13: Bot/ROP Sanitizers Report

Дата: 2026-05-10

## Цель

Сделать downstream-артефакты для РОПа и будущего Telegram-бота безопасными после quality backfill:

- не использовать один и тот же `ideal_answer_example` как скрипт менеджера и как ответ бота;
- нормализовать брендовые ASR-искажения;
- убрать из bot-safe ответов конкретные цены, скидки, дедлайны, возвраты, рассрочки и персональные данные;
- оставить raw/manager-facing ответ для аудита и обучения менеджеров;
- не допустить no-live/voicemail/IVR/ASR-мусор в revenue risks, top answers и bot seeds.

AMO/CRM writeback не выполнялся.

## Что изменено в коде

### Sanitizer layer

Файл: `src/mango_mvp/insights/sanitizers.py`

Добавлен единый deterministic sanitizer:

- `sanitize_answer(..., mode="manager")` — нормализует менеджерский идеальный ответ, но сохраняет смысл для РОПа;
- `sanitize_answer(..., mode="bot")` — строит безопасный ответ для бота без конкретных коммерческих/юридических/персональных обещаний;
- `sanitize_customer_text(...)` — чистит клиентские цитаты/вопросы;
- `has_brand_risk`, `has_money_or_terms_risk`, `has_personal_data_risk` — acceptance checks для baseline;
- `flag_booleans(...)` — отдельные флаги риска по категориям.

Покрываемые категории:

- brand artifacts: `НПК МФТИ`, `УНФК`, `МПК`, `Чебенцентр`, `Черный центр`, похожие ASR-искажения;
- prices/discounts: суммы, проценты, скидки, акции, промокоды, раннее бронирование;
- installment/payment terms: рассрочка, кредит, частями, предоплата, платежи;
- legal/refund: возвраты, договор, оферта, гарантии, юридические обещания;
- deadlines/promises: до даты, сегодня/завтра, на этой неделе, в течение N часов/дней;
- PII: email, телефон, ФИО/имя-отчество.

### Knowledge Base

Файл: `src/mango_mvp/insights/knowledge_base.py`

Добавлены поля в enriched layer:

- `ideal_answer_manager_sanitized`;
- `bot_safe_answer`;
- `customer_question_sanitized`, `customer_quote_sanitized`, `manager_quote_sanitized`;
- `sanitizer_flags`;
- `bot_safety_status`, `bot_safety_blocked`;
- `brand_risk_flag`, `money_or_discount_flag`, `installment_flag`, `legal_or_refund_flag`, `deadline_or_promise_flag`, `personal_data_flag`.

Изменена логика `bot_seed_status`:

- строки без safety-замен и с высоким качеством могут быть `ready_for_bot_draft`;
- строки, где sanitizer что-то заменял, по умолчанию уходят в `needs_rop_validation`;
- строки с unresolved safety risk блокируются;
- fallback/dry-run и no-live остаются исключенными.

### ROP validation pack

Файл: `src/mango_mvp/insights/rop_validation_pack.py`

В ROP pack добавлены отдельные колонки:

- `Идеальный ответ для менеджера`;
- `Безопасный ответ для бота`;
- `Статус sanitizer`;
- `Флаги sanitizer`;
- отдельные риск-флаги по бренду, цене/скидке, рассрочке, договору/возврату, сроку/обещанию, ПДн.

Bot drafts теперь строятся из `bot_safe_answer`, а не из raw `ideal_answer_example`.

### Baseline

Файл: `src/mango_mvp/quality/transcript_quality_baseline.py`

Baseline теперь различает:

- raw/manager ideal risks как нагрузку на sanitizer;
- bot-safe residual risks как acceptance gate.

Hard gate для stage 13: в bot-safe слоях остаточные риски должны быть 0.

## Новые артефакты

Knowledge Base:

- `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v3_stage13_sanitized/`
- `sales_insight_knowledge_base.xlsx`
- `enriched_reviews.csv`
- `bot_knowledge_seeds.csv`
- `best_answers.csv`
- `rop_coaching_queue.csv`

ROP pack:

- `stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v3_stage13_sanitized/`
- `ROP_validation_pack_v1.xlsx`
- `rop_validation.csv`
- `bot_knowledge_drafts.csv`

Baseline:

- `stable_runtime/transcript_quality_baseline_after_quality_backfill_20260510_v3_stage13_sanitized/`
- `summary.json`
- `BASELINE_REPORT.md`

## Ключевые метрики

До stage 13, baseline v2:

- `kb_bot_ready_money_or_terms`: 552
- `kb_ideal_answer_brand_risk`: 13
- `rop_bot_candidate_money_or_terms`: 85
- `rop_p0_no_live_or_artifact`: 0
- `rop_revenue_risk_no_live_or_artifact`: 0

После stage 13, baseline v3:

- `kb_bot_ready_money_or_terms`: 0
- `kb_ideal_answer_brand_risk`: 0
- `kb_bot_safe_answer_brand_risk`: 0
- `kb_bot_safe_answer_personal_data_risk`: 0
- `rop_bot_candidate_money_or_terms`: 0
- `rop_bot_safe_answer_brand_risk`: 0
- `rop_bot_safe_answer_personal_data_risk`: 0
- `rop_p0_no_live_or_artifact`: 0
- `rop_revenue_risk_no_live_or_artifact`: 0

Raw/source нагрузка, которую sanitizer теперь успешно чистит:

- `raw_ideal_answer_brand_risk`: 75
- `raw_ideal_answer_money_or_terms`: 1 923

Интерпретация: исходные идеальные ответы часто содержат коммерческие/брендовые/сроковые формулировки, но в bot-safe слой они больше не протекают.

## Проверка CSV bot-safe слоя

Отдельный скан CSV показал:

- `bot_knowledge_seeds.csv`: 300 строк, brand risk 0, money/terms risk 0, personal data risk 0;
- `bot_knowledge_drafts.csv`: 250 строк, brand risk 0, money/terms risk 0, personal data risk 0.

## Тесты

Прогнаны:

- targeted regression: `12 passed`;
- broader transcript/insight suite: `73 passed, 1 warning`;
- full suite: `675 passed, 1 warning`.

Единственное предупреждение внешнее: `urllib3` сообщает о LibreSSL в системном Python. На stage 13 это не влияет.

## Что важно помнить

- `ideal_answer_example` не удаляется и остается raw/source полем для аудита.
- Для менеджеров используется `ideal_answer_manager_sanitized`.
- Для будущего Telegram-бота используется только `bot_safe_answer` после ROP/human approval.
- Stage 13 не делает базу бота production-ready автоматически: он делает ее безопасной как черновой материал.
- Messenger/email context по-прежнему не учтен, поэтому ROP должен трактовать выводы как гипотезы по звонкам, а не как окончательную оценку всей работы менеджера.

## Следующий этап

Stage 14 после этого был выполнен отдельным сравнительным пакетом:

- пакет: `stable_runtime/transcript_quality_stage14_comparison_20260510_v1/`;
- отчет: `docs/TRANSCRIPT_QUALITY_STAGE14_COMPARISON_REPORT_2026-05-10.md`;
- следующий практический этап теперь stage 15: подключить hard gates в постоянный pipeline.
