# Stage 12 Downstream Rebuild Report

Дата: 2026-05-10 Moscow

## Статус

Этап 12 выполнен для read-only downstream-слоев после transcript-quality hard-gate/backfill. Live-запись в AMO/Tallanto не выполнялась.

## Что пересобрано

1. Canonical calls master
   - Артефакт: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/`
   - Canonical DB: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`
   - Source audio: `64 867`
   - Исключено без ASR: `35`
   - Actionable calls: `64 832`
   - ASR done actionable: `64 832`
   - Full R+A actionable: `64 832`
   - Missing ASR actionable: `0`
   - Missing full R+A actionable: `0`
   - Validation: `passed=true`

2. Insight readiness / contact-layer readiness
   - Артефакт: `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/`
   - Terminal analyzed calls: `64 832`
   - Calls with phone: `63 788`
   - Calls without phone: `1 044`
   - Unique client phones: `15 924`
   - Client chains with contentful calls: `13 477`
   - Contentful calls after backfill: `46 153`
   - Non-conversation calls after backfill: `18 679`
   - В сравнении со старым readiness false-contentful слой уменьшен на `5 276` звонков.

3. Outcome linkage
   - Артефакт: `stable_runtime/outcome_linkage_report_after_quality_backfill_20260510_v1/`
   - Client chains: `15 924`
   - Chains with Tallanto outcome signal: `7 899`
   - Chains with AMO outcome signal: `312`
   - Strong outcome: `4 830`
   - Proxy outcome: `9 290`
   - Unknown outcome: `1 804`
   - Reactivation revenue candidates: `151`
   - Winner pattern candidates: `1 336`
   - Loss pattern candidates: `1 850`

4. Pilot sales moments
   - Артефакт: `stable_runtime/pilot_sales_moments_after_quality_backfill_20260510_v2/`
   - Pilot clients: `500`
   - Selected calls: `2 732`
   - Excluded no-live sales moment candidates: `6`
   - Sales moments: `2 726`
   - Unique phones in moments: `500`
   - Non-conversation in sales moments: `0`

5. LLM-review layer for sales moments
   - Артефакт: `stable_runtime/pilot_sales_moment_llm_review_after_quality_backfill_20260510_v2_hybrid_reuse/`
   - Reviews written: `2 726`
   - Reused trusted GPT-5.5 reviews by same source filename: `2 336`
   - Deterministic fallback rows needing live LLM-refresh: `390`
   - Trusted GPT coverage: `85.69%`
   - Ограничение: новый live `codex exec` review был заблокирован текущей Codex CLI авторизацией. Поэтому fallback строки не используются как готовые bot/ROP/best-answer материалы.

6. Sales insight knowledge base
   - Артефакт: `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v2_hybrid_reuse/`
   - Workbook: `sales_insight_knowledge_base.xlsx`
   - Source reviews: `2 726`
   - Trusted live/GPT reviews: `2 336`
   - Need live LLM-refresh: `390`
   - Ready bot drafts in raw enriched layer: `1 059`
   - Needs ROP validation in raw enriched layer: `952`
   - LLM-refresh queue: `llm_refresh_queue.csv`
   - Best answers / bot seeds / ROP patterns are built only from trusted GPT rows.

7. ROP validation pack
   - Артефакт: `stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v2_hybrid_reuse/`
   - Workbook: `ROP_validation_pack_v1.xlsx`
   - Source reviews: `2 726`
   - Reviewable business rows: `2 329`
   - Excluded from validation: `397`
   - Combined unique moments for ROP: `636`
   - Top script answers: `150`
   - Revenue leakage risks: `122`
   - Process problems: `244`
   - Bot knowledge drafts: `250`
   - No-live/artifact rows in ROP pack: `0`
   - P0 no-live/artifact rows: `0`
   - Revenue-risk no-live/artifact rows: `0`

8. Transcript quality baseline
   - Артефакт: `stable_runtime/transcript_quality_baseline_after_quality_backfill_20260510_v2_hybrid_reuse/`
   - Report: `BASELINE_REPORT.md`
   - Readiness suspicious contentful by history: `361`
   - Suspicious contentful with next step: `188`
   - False email-from-voice-mail candidates: `16`
   - KB no-live revenue risk: `0`
   - ROP P0 no-live/artifact: `0`
   - ROP revenue-risk no-live/artifact: `0`

## Кодовые защиты, добавленные в рамках этапа

1. `pilot_extraction.py`
   - Добавлен фильтр no-live/technical voicemail кандидатов до сборки sales moments.
   - Исключенные кандидаты пишутся в `excluded_sales_moment_candidates.csv`.

2. `knowledge_base.py`
   - Добавлен trust-layer: `trusted_llm_review` vs `needs_live_llm_refresh`.
   - `dry_run`/deterministic fallback строки не попадают в best answers, bot seeds, ROP coaching, pattern matrix.
   - Добавлена очередь `llm_refresh_queue.csv`.
   - ASR-artifact markers вроде `Продолжение следует...` классифицируются как `no_live_contact_or_voicemail`, а не как риск выручки.

3. `rop_validation_pack.py`
   - ROP pack строится только из trusted business rows.
   - No-live / voicemail / ASR-artifact / fallback строки исключаются из проверки и выносятся в отдельный лист `Исключено из проверки`.

## Тесты

- Targeted downstream tests: `70 passed, 1 warning`.
- Full test suite: `669 passed, 1 warning`.

## Что намеренно не сделано в этапе 12

1. Live AMO writeback не выполнялся.
2. `АКТУАЛЬНО_AMO_ready.xlsx` не перезаписывался: старый root-файл считается stale, а новый AMO-ready нужно формировать после stage 13 sanitizers и staged writeback policy.
3. 390 fallback sales moments не используются как финальная база для бота/РОПа до восстановления Codex CLI auth или запуска через API key.

## Остаточные риски

1. В readiness еще есть `361` suspicious contentful строк по history-маркерам. Это не значит, что все ошибочные, но это очередь для stage 14 comparison / stage 15 cleanup.
2. `16` false email-from-voice-mail candidates требуют отдельной проверки, чтобы не переносить фейковые email/action claims в CRM или bot KB.
3. В bot candidates есть строки с деньгами/условиями (`552` в KB baseline, `85` в ROP pack). Это не ошибка, но перед использованием в Telegram-боте нужен stage 13 sanitizer: placeholders для цен, скидок, дедлайнов, возвратов и рассрочек.
4. Brand-risk в ideal answers: `13` в KB baseline, `1` в ROP pack. Нужен stage 13 brand normalizer.

## Вывод

Этап 12 доведен до безопасного read-only состояния: downstream-слои пересобраны, no-live/voicemail/ASR-artifact ошибки не попадают в ROP revenue risks, bot seeds и best answers защищены от fallback-строк, все тесты проходят. Следующий логичный этап: stage 13 bot/ROP sanitizers.
