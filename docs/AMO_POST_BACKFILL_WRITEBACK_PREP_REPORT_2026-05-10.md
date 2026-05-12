# AMO Post-Backfill Writeback Prep Report

Дата: 2026-05-10

## Итог

Подготовлен новый AMO-ready слой из post-backfill canonical DB и phone-chain context.

Новый export:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v1/`

Ключевые файлы:

- `master_calls_ru.csv`
- `master_contacts_ru.csv`
- `amo_export_ready_ru.csv`
- `master_contacts_ru.xlsx`
- `amo_export_ready_ru.xlsx`
- `master_export_pack_ru.xlsx`
- `summary.json`

Audit pack для Claude Code:

- `audits/_inbox/amo_post_backfill_writeback_20260510_v4_product_gate/`

## Метрики

По `summary.json` нового export:

- master calls: `64 832`
- master contacts: `15 923`
- AMO-ready rows: `5 667`
- manual-review rows: `6 999`
- CRM-contentful calls: `44 242`
- CRM-non-conversation / low-value calls: `20 590`
- Stage15 passed: `true`
- CRM quality writeback ready: `true`

## Что изменено

1. Добавлен builder:

   - `scripts/build_post_backfill_amo_ready_export.py`

   Он читает:

   - `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`
   - `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/client_chains.csv`
   - `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`

2. Builder не использует старый `sales_master_export_20260424...` слой.

3. История контакта строится только по CRM-contentful звонкам.

4. Из AMO-ready исключаются obvious low-value cases через отдельный `crm_writeback_quality_detector`, а не через локальные regex внутри builder-а:

   - wrong-number;
   - no-dialogue;
   - no-learning-request;
   - no meaningful conversation;
   - “сам не обращался / услуги не нужны”;
   - “содержательного диалога не состоялось”;
   - IVR / автоинформатор / сторонняя организация;
   - out-of-domain B2B, например логистика/перевозки;
   - no-content pickup, где запрос и интерес не определены;
   - технические/маркетинговые/провайдерские обращения;
   - поставки оборудования, подарки, сувениры, опросы, спецоператоры и похожие сторонние B2B-запросы.

5. Raw callback-телефоны внутри AI-текстов для AMO заменяются на `[PHONE]`.

6. Старые агрегированные `objections_top` из phone-chain больше не попадают в CRM writeback, чтобы не переносить low-value метки из несодержательных касаний в реальные карточки.

7. Добавлен отдельный CRM writeback quality gate:

   - `src/mango_mvp/quality/crm_writeback_quality_detector.py`
   - `src/mango_mvp/quality/crm_writeback_frozen_corpus.py`
   - `scripts/run_crm_writeback_quality_gate.py`
   - `tests/fixtures/crm_writeback_relevance_frozen_corpus.jsonl`

   Gate full-scan'ит весь AMO-ready слой, проверяет frozen corpus, пустую историю и CRM-поля, заканчивающиеся `...`.

8. `scripts/write_amo_ready_contacts.py` получил `--offline-preview`, чтобы можно было собрать payload preview без AMO runtime/tunnel.

9. Синхронизированы:

   - `CLAUDE.md`
   - `docs/THREAT_MODEL.md`
   - `docs/AI_WORKFLOW.md`
   - `.claude/commands/audit.md`
   - `audits/README.md`

## Dry-Run Статус

Live AMO dry-run не выполнен, потому что shared DB tunnel недоступен:

- `127.0.0.1:15432` не слушает;
- текущий AMO runtime через `.env.private` ожидает PostgreSQL;
- без tunnel невозможно получить OAuth connection из runtime DB.

Выполнен offline-preview на 100 строк:

- `stable_runtime/amocrm_runtime/contact_writebacks/20260510T115127Z/contact_writeback_summary.json`
- `offline_preview_rows=100`
- `failed=0`
- live writes: `0`

Offline-preview не проверяет наличие контакта в AMO. Он проверяет только payload, который был бы подготовлен к записи.

## Sanity Checks

В audit pack v4:

- `ready_empty_history=0`
- `ready_history_ends_with_ellipsis=0`
- `ready_chronology_ends_with_ellipsis=0`
- `crm_writeback_quality_blocking_rows=0`
- `crm_writeback_frozen_corpus_rows=49`
- `crm_writeback_frozen_corpus_passed=true`
- `v3_block_or_review_still_ready_count=0`
- `ready_phone_redaction_needed_rows=0`
- `ready_bad_phone_placeholder_rows=0`
- `ready_known_callback_phone_literal_rows=0`
- `known_bad_audit_phones_still_ready=[]`
- `claude_v2_bad_phones_still_ready=[]`

## Что Не Сделано

1. Live AMO dry-run не выполнен.

   Причина: не поднят shared PostgreSQL tunnel / AMO runtime. Контрольная попытка dry-run на 25 строк:

   - `stable_runtime/amocrm_runtime/contact_writebacks/20260510T112733Z/contact_writeback_summary.json`
   - `failed=25`
   - ошибка: `connection refused` на `127.0.0.1:15432`

2. Live AMO writeback не выполнялся.

## Следующий Шаг

1. При необходимости запустить повторный Claude Code аудит вручную из корня репозитория:

   ```bash
   claude -p --model opus --effort high --permission-mode acceptEdits "/audit audits/_inbox/amo_post_backfill_writeback_20260510_v4_product_gate"
   ```

   Если slash-command не сработает, использовать прямой prompt из инструкции проекта.

   Примечание: из Codex sandbox Claude CLI не запускается из-за `EPERM` на `/Users/dmitrijfabarisov/.claude.json`; команду нужно выполнить из обычного терминала пользователя.

2. После `PASS` или приемлемого `PASS_WITH_LIMITATIONS`:

   - поднять shared DB tunnel;
   - выполнить настоящий AMO dry-run через `scripts/write_amo_ready_contacts.py` без `--offline-preview`;
   - только после просмотра отчета идти в staged live writeback.

## Claude v1 Fix Status

Claude Code audit v1 дал `PASS_WITH_LIMITATIONS`. Все ограничения закрыты в v2:

- F-001: `stable_runtime/CANONICAL_EXPORT.txt` переключен на `sales_master_export_20260510_after_quality_backfill_v1`.
- F-002: low-value exclusion расширен; полный scan по `amo_export_ready_ru.csv` показывает `ready_low_value_red_flag_rows=0`.
- F-003: raw callback-phone literals внутри CRM text полей редактируются; полный scan показывает `ready_phone_redaction_needed_rows=0`.

Claude Code audit v2 снова дал `PASS_WITH_LIMITATIONS`: F-001/F-003 закрыты, F-002 downgraded до P2, но нашел 13 новых out-of-domain/no-content строк.

Claude Code audit v3 показал, что v2→v3 снова был слишком узким: конкретные v2 телефоны исчезли, но тот же класс F-002 продолжил протекать в AMO-ready; добавлен процессный finding F-006 о narrow-fix-then-regress.

В v4 исправление переведено в более общий слой:

- builder вызывает независимый `crm_writeback_quality_detector`;
- добавлен frozen corpus на 49 CRM-writeback relevance cases;
- quality gate full-scan'ит все `5 667` AMO-ready строк;
- все `36` строк Claude v3 с решением `block/needs_review` отсутствуют в AMO-ready;
- 2 строки с обрезанным `...` переведены в manual-review;
- `crm_writeback_quality_blocking_rows=0`.
