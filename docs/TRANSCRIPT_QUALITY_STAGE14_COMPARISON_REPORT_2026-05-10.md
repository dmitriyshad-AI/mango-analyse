# Stage 14: Quality Comparison v2/v3 Report

Дата: 2026-05-10

## Цель

Проверить, что stage 13 sanitizer реально улучшил downstream-артефакты и не сломал предыдущие защиты:

- сравнить v2/v3 baseline;
- подтвердить отсутствие no-live/voicemail/ASR-artifact в ROP revenue/P0 и bot outputs;
- подтвердить отсутствие конкретных цен, скидок, сроков, возвратов, рассрочек, brand artifacts и ПДн в bot-safe ответах;
- собрать audit sample для GPT/Claude/РОПа;
- отдельно вынести over-sanitization кандидатов, где ответ может быть безопасным, но слишком общим.

AMO/CRM writeback не выполнялся.

## Что добавлено

Код:

- `src/mango_mvp/quality/stage14_quality_comparison.py`
- `scripts/build_transcript_quality_stage14_comparison.py`
- `tests/test_transcript_quality_stage14_comparison.py`

Также stage 14 выявил blind spots в stage 13 sanitizer. После read-only аудита субагентов sanitizer был усилен и stage 13 артефакты пересобраны заново:

- числовые даты: `15.05`, `10 апреля`;
- дни недели и относительные сроки: `до пятницы`, `в понедельник`, `на этой неделе`;
- время занятий/слотов: `10:00-12:00`;
- словесные проценты: `10 процентов`;
- сокращенные суммы: `50к`;
- booking/deadline promises: `бронь`, `держим`, `забронируем`;
- одиночные имена в bot-safe ответах: `Мария`, `Михаил`, `Егор`, и похожие частые формы.

## Артефакты

Stage 14 root:

- `stable_runtime/transcript_quality_stage14_comparison_20260510_v1/`

Основные файлы:

- `stage14_quality_comparison.xlsx` — удобный workbook для проверки;
- `summary.json` — machine-readable отчет;
- `STAGE14_QUALITY_COMPARISON_REPORT.md` — auto-generated report;
- `metric_delta.csv` — сравнение метрик v2/v3;
- `audit_sample.csv` — 200 строк для GPT/Claude/РОП-аудита;
- `over_sanitization_candidates.csv` — 250 кандидатов на проверку полезности;
- `residual_risk_sample.csv` — остаточные bot-safe риски, сейчас пустой;
- `bot_seed_before_after_sample.csv` — before/after bot draft sample;
- `AUDIT_PROMPT_FOR_CLAUDE_OR_GPT.md` — готовый prompt для внешнего аудита.

## Acceptance Result

Stage 14 acceptance: `passed=true`.

Hard checks:

- required KB columns present: `true`;
- required ROP columns present: `true`;
- bot seed safe columns present: `true`;
- no residual bot-safe risks: `true`;
- KB no-live revenue risk: `0`;
- ROP P0 no-live/artifact: `0`;
- ROP revenue no-live/artifact: `0`;
- KB bot-ready money/terms: `0`;
- ROP bot-candidate money/terms: `0`;
- bot-ready rows missing safe answer: `0`;
- audit sample built: `200` rows.

## Key Deltas

| Metric | v2 | v3 | Delta |
|---|---:|---:|---:|
| `kb_bot_ready_money_or_terms` | 552 | 0 | -552 |
| `kb_ideal_answer_brand_risk` | 13 | 0 | -13 |
| `rop_bot_candidate_money_or_terms` | 85 | 0 | -85 |
| `rop_p0_no_live_or_artifact` | 0 | 0 | 0 |
| `rop_revenue_risk_no_live_or_artifact` | 0 | 0 | 0 |
| `kb_bot_safe_answer_brand_risk` | n/a | 0 | n/a |
| `kb_bot_safe_answer_personal_data_risk` | n/a | 0 | n/a |
| `rop_bot_safe_answer_brand_risk` | n/a | 0 | n/a |
| `rop_bot_safe_answer_personal_data_risk` | n/a | 0 | n/a |

Raw/source нагрузка после усиления sanitizer:

- `kb_raw_ideal_answer_brand_risk`: 75;
- `kb_raw_ideal_answer_money_or_terms`: 1 923.

Интерпретация: raw manager/source слой остается рискованным, но bot-safe слой очищен. Это ожидаемо и правильно: raw нужен для аудита, bot-safe нужен для будущего бота.

## Audit Sample

Собрано 200 уникальных строк без дублей `moment_id`:

- `coverage_filler`: 49;
- `money_terms_sanitized`: 34;
- `brand_sanitized`: 25;
- `bot_ready_clean_no_changes`: 20;
- `legal_deadline_sanitized`: 19;
- `rop_revenue_risk`: 14;
- `installment_sanitized`: 13;
- `rop_top_answer`: 13;
- `rop_bot_draft`: 7;
- `personal_data_sanitized`: 6.

Что проверять во внешнем аудите:

- безопасен ли `bot_safe_answer`;
- не стал ли ответ слишком общим;
- понятен ли `ideal_answer_manager_sanitized` для РОПа;
- не попали ли no-live/voicemail/IVR/ASR-мусор в коммерческие очереди;
- нужны ли ручные rewrite для отдельных bot-safe ответов.

## Over-Sanitization Queue

Собрано 250 кандидатов.

Причина у всех: `bot_answer_many_generic_markers`.

Это не список ошибок. Это очередь для проверки полезности: sanitizer намеренно стал жестче, поэтому часть bot-safe ответов может быть безопасной, но слишком общей. Перед production Telegram-ботом эту очередь нужно выборочно проверить GPT/Claude/РОПом.

## Дополнительный Hidden-Risk Scan

После усиления sanitizer отдельный скан `bot_knowledge_seeds.csv` и `bot_knowledge_drafts.csv` показал 0 попаданий по:

- weekday terms;
- numeric dates;
- month dates;
- exact time ranges;
- spoken percent;
- `50к`-style money shorthand;
- booking/deadline words;
- common risk functions: brand/money/PII.

## Честная оценка

Stage 14 доказывает, что safety gate стал значительно сильнее и воспроизводимее. Это все еще не доказывает, что база бота идеальна по смыслу: полезность и естественность bot-safe ответов нужно проверять через audit sample и over-sanitization queue.

Качество близко к хорошему production-draft уровню для внутреннего РОП/методистского контура. Для автономного Telegram-бота это не финал: нужен stage 15 permanent gate + ROP-approved golden dataset.

## Следующий этап

Stage 15:

- подключить stage13/stage14 gates в постоянный pipeline;
- запретить экспорт KB/ROP/bot/CRM, если hard checks не пройдены;
- добавить runbook и smoke-команды;
- определить allowlist полей для future Telegram bot export;
- оставить raw поля только в audit/internal layers, не в bot/RAG export.
