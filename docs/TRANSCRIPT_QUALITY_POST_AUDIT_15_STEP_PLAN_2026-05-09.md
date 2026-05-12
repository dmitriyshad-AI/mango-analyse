# Transcript Quality Post-Audit 15-Step Plan

Дата: 2026-05-09.

## Зачем этот план

Этот файл фиксирует актуальный порядок работ после сверки независимых аудитов GPT и Claude по hard-gate `non_conversation`.

Обновление policy от 2026-05-09: production-контур остается в GPT-only режиме. Claude используется как внешний аудит/регрессионная проверка, но не является обязательным блокером для будущих auto-apply решений.

Цель: безопасно улучшить качество обработки всех исторических звонков без повторного ASR, не потерять живые клиентские диалоги и не пустить voicemail/IVR/ASR-мусор в ROP-пакеты, Knowledge Base, bot seeds и CRM.

## Текущий verified state

Пакет аудита:

`stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/`

Факты сверки:

- `audit_items`: 200
- GPT decisions: 200
- Claude decisions: 200
- missing GPT: 0
- missing Claude: 0
- GPT: `safe_apply=186`, `keep_current=13`, `manual_review=1`
- Claude: `safe_apply=188`, `keep_current=7`, `manual_review=5`
- strict consensus auto-apply: `185`
- GPT-only auto-apply после deterministic safeguards: `186`
- GPT-only not auto-apply: `14`

Consensus queues:

- `consensus_auto_apply`: 185
- `consensus_keep_current`: 7
- `manual_review`: 5
- `disagreement_review`: 3

Ключевое production-правило после сверки:

**Автоматически применять можно случаи, где GPT дал `safe_apply`, а текущие deterministic safeguards также не блокируют применение. Claude не является production-блокером: его решения сохраняются как справочный аудит, но auto-apply не ждёт Claude-консенсуса.**

Практическая причина: GPT и Claude дали близкие результаты на проверке 200 кейсов, а основной продуктовый контур уже вызывает GPT. Дополнительное обязательное ожидание Claude усложняет pipeline и замедляет массовую обработку без пропорционального выигрыша.

## Рабочие артефакты

- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_consensus_comparison.csv`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_consensus_auto_apply_candidates.csv`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_consensus_review_queue.csv`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_consensus_summary.json`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_gpt_apply_plan.csv`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_gpt_blocked_apply_plan.csv`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_gpt_summary.json`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/audit_gpt_policy.json`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/CLAUDE answer/claude_decisions.jsonl`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/gpt_audit_decisions.jsonl`

## Что уже реализовано до этого плана

1. Создан и подключен `src/mango_mvp/quality/non_conversation.py`.
2. Добавлены live-safeguards v4 против ложного перевода живых диалогов в `non_conversation`.
3. `services/analyze.py` использует guardrail version `non_conversation_v4_live_safeguards`.
4. Добавлены regression-тесты по transfer-to-voicemail, service dialogue after IVR, short live opt-out и ASR-junk cases.
5. Проведен full dry-run по 64 832 fully processed звонкам: `would_update=5404`, `unchanged=59428`, `parse_errors=0`.
6. Проведены независимые GPT и Claude аудиты на 200 hard-gate кандидатах.
7. Выполнена per-case consensus-сверка GPT/Claude.

## 15 шагов дальше

### 1. Зафиксировать consensus GPT/Claude

Статус: `done`.

Что сделано:

- найден правильный Claude JSONL на 200 кейсов;
- выполнено сравнение с GPT JSONL;
- собраны strict auto-apply и review queues.

Критерий готовности: `185` strict consensus auto-apply и `15` non-auto подтверждены в CSV/JSON.

### 2. Синхронизировать policy auto-apply

Статус: `done`.

Что сделать:

- формально зашить правило: auto-apply разрешен только при `gpt_decision=safe_apply` + прохождение deterministic v4/v5 safeguards;
- запретить auto-apply для `gpt_manual_review`, `gpt_keep_current`, `reanalyze_required` и deterministic manual/protected-live блокировок;
- обновить apply/backfill скрипты так, чтобы Claude-safe не был достаточным основанием, а Claude-disagreement не блокировал GPT-safe при зеленых deterministic safeguards.

Результат: единый GPT-only policy gate перед любым массовым backfill.

Факт выполнения: `transcript_quality_backfill` теперь пропускает hard-gate кандидата при `gpt_decision=safe_apply`, `policy_queue=gpt_auto_apply` и high-confidence deterministic guardrail. Claude-колонки сохраняются в metadata, но не являются блокером. Сформированы `audit_gpt_apply_plan.csv` и `audit_gpt_blocked_apply_plan.csv`.

### 3. Проверить 15 non-auto кейсов и reason taxonomy

Статус: `done`.

Что сделать:

- разобрать 7 `consensus_keep_current`, 5 `manual_review`, 3 `disagreement_review`;
- убедиться, что все они блокируются safeguards;
- добавить недостающие reason codes, если какой-то риск не объясняется текущими правилами.

Результат: список блокирующих паттернов и регрессионные фикстуры.

Факт выполнения: 15 non-consensus кейсов разобраны. После перехода на GPT-only один из них становится допустимым auto-apply, потому что GPT дал `safe_apply` и deterministic слой зеленый. Production non-auto очередь теперь 14 кейсов: 13 `gpt_keep_current` и 1 `gpt_manual_review`. Решение: не расширять regex под эти кейсы без отдельного full dry-run, чтобы не ухудшить precision на очевидных no-live звонках.

### 4. Финализировать v5 safeguards

Статус: `done`.

Что сделать:

- если шаг 3 найдет недостающие правила, выпустить `non_conversation_v5_consensus_safeguards`;
- сохранить текущую v4 логику как baseline;
- добавить тесты на все новые паттерны.

Результат: deterministic слой, согласованный с аудитом.

Факт выполнения: отдельный `non_conversation_v5` не выпущен намеренно. По итогам GPT/Claude-аудита выбрана более безопасная архитектура: оставить `non_conversation_v4_live_safeguards` как deterministic baseline и добавить поверх него production policy `hard_gate_gpt_policy_v1`. Это снижает риск расширить regex и случайно отправить живые диалоги в `non_conversation`.

### 5. Повторить full dry-run по 64 832 звонкам

Статус: `done`.

Что сделать:

- запустить full dry-run уже на финальной v5/policy-логике;
- не писать в SQLite;
- собрать `would_update`, `protected_live_dialogues`, `manual_review`, `parse_errors`;
- сравнить с v4 dry-run (`5404 would_update`).

Результат: финальный список кандидатов перед backfill.

Факт выполнения: full dry-run запущен 2026-05-09 в `stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v5_gpt_policy_preview/`. Текущий covered corpus расширился до `68 771` terminal rows, потому что после baseline v4 часть RA-статусов дозавершилась. Итог: `would_update=5 761`, `unchanged=63 010`, `parse_errors=0`, `protected_live_dialogues=45 288`. Baseline v4 был `64 832 / 5 404 / 59 428 / 0 / 42 669`.

### 6. Построить apply-plan, а не сразу применять

Статус: `done`.

Что сделать:

- создать apply-plan CSV/JSONL по всем full-corpus кандидатам;
- разделить очереди: `auto_apply`, `keep_current`, `manual_review`, `reanalyze_required`, `blocked`;
- для каждой строки хранить `source_db`, `call_id`, `source_filename`, старый тип, новый тип, reason codes, confidence/source policy.

Результат: прозрачный список изменений до записи в БД.

Факт выполнения: создан full-corpus GPT-only apply-plan в `stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_phase6/`. Вход: `5 761` deterministic candidates из phase 5. Так как GPT-decisions по полному корпусу ещё не загружены, `auto_apply_ready=0`, `gpt_review_required=5 761`, `blocked=0`. Подготовлены `gpt_review_tasks.jsonl`, `gpt_decisions_template.jsonl`, prompt, risk/month summaries и priority chunks по 500 задач.

### 7. Сделать backup и rollback manifest

Статус: `done`.

Что сделать:

- перед любой записью сохранить список затрагиваемых SQLite DB;
- создать checksum/size manifest;
- подготовить rollback CSV: какие поля были до backfill.

Результат: можно откатить изменения без `git reset` и без потери данных.

Факт выполнения: создан phase 7 safety package в `stable_runtime/non_conversation_hard_gate_backup_manifest_20260509_phase7/`. Покрыто `5 761` строк из phase 6 и `25` SQLite DB. Для всех строк найден текущий row snapshot: `rollback_rows=5 761`, `missing_rows=0`, `schema_warnings=0`. Для БД зафиксированы size/mtime/SHA256, создан `backup_copy_plan.sh`, `rollback_snapshot.csv/jsonl` и restore notes. Реальные копии БД не создавались, чтобы не плодить гигабайты до финального `auto_apply_ready` subset; перед staged apply manifest нужно пересобрать/запустить backup по фактическому subset.


### 7.5. Выполнить GPT-review первого priority chunk

Статус: `done`.

Что сделано:

- создан hard-gate GPT-review runner через `codex exec`;
- исправлена проблема stale `CODEX_HOME` для nested Codex CLI;
- smoke-проверки на 10/20 задачах прошли без ошибок;
- первый priority chunk на `500` задачах обработан моделью `gpt-5.5` с `reasoning_effort=medium`;
- результат: `safe_apply=480`, `manual_review=11`, `keep_current=9`, errors=`0`;
- пересобран apply-plan: `auto_apply_ready=480`, `gpt_review_required=5261`, `manual_review=11`, `keep_current=9`;
- построен rollback manifest для `480` ready строк: `19` DB, `missing_rows=0`, `schema_warnings=0`.

Артефакты:

- `stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority01/`
- `stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_after_priority01_review/`
- `stable_runtime/non_conversation_hard_gate_backup_manifest_20260509_after_priority01_review/`
- `docs/TRANSCRIPT_QUALITY_PRIORITY01_GPT_REVIEW_REPORT_2026-05-09.md`
- `docs/TRANSCRIPT_QUALITY_STAGED_BACKFILL_REPORT_2026-05-09.md`

### 8. Staged apply на малой партии

Статус: `done`.

Что сделать:

- применить только `auto_apply` на 50-100 звонках;
- проверить SQL counts до/после;
- проверить 20 примеров вручную/LLM-аудитом;
- убедиться, что старые значения не удаляются, а сохраняются в quality/backfill metadata.

Результат: доказательство, что backfill технически безопасен.

Факт выполнения: staged apply на 100 строках выполнен 2026-05-09. Dry-run: planned=`100`, blocked=`0`, missing=`0`, errors=`0`. Apply: applied=`100`, validation ok=`100`, problems=`0`, backups созданы по 7 DB. Артефакт: `stable_runtime/non_conversation_hard_gate_staged_apply_20260509_stage100_apply/`.

### 9. Staged apply на средней партии

Статус: `done`.

Что сделать:

- применить 500-1000 auto-apply кандидатов;
- повторить валидацию распределений;
- проверить, что live sales/service/technical диалоги не уходят в `non_conversation`;
- проверить отсутствие падения downstream сборок.

Результат: масштабирование без регрессий.

Факт выполнения: применены все `safe_apply` строки из первых трех GPT-reviewed chunks. После priority01 применено `480/480`; после priority02 combined apply: selected=`980`, applied new=`500`, already_applied=`480`, blocked=`0`, missing=`0`, errors=`0`; после priority03 combined apply: selected=`1477`, applied new=`497`, already_applied=`980`, blocked=`0`, missing=`0`, errors=`0`. Post-apply validation: ok=`1477`, problems=`0`, verify dry-run: already_applied=`1477`. Артефакт: `stable_runtime/non_conversation_hard_gate_staged_apply_20260509_priority01_03_apply/`.

### 10. Full auto-apply backfill

Статус: `done`.

Что сделать:

- применить все финальные `auto_apply` full-corpus кандидаты;
- не трогать review/disagreement/keep_current;
- сохранить полный backfill report.

Результат: исторический корпус очищен от подтвержденных voicemail/IVR/no-live ошибок.

Факт выполнения: все `5 761` hard-gate кандидатов прошли GPT-review. Финальные очереди: `auto_apply_ready=5 672`, `manual_review=60`, `keep_current=29`, `gpt_review_required=0`. Все `5 672` safe строки применены в SQLite по 25 DB. Финальная валидация: ok=`5 672`, problems=`0`, verify dry-run: already_applied=`5 672`. Артефакт: `stable_runtime/non_conversation_hard_gate_staged_apply_20260509_priority01_12_apply/`.

### 11. Собрать reanalyze/manual-review очередь

Статус: `done`.

Что сделать:

- для всех borderline и disagreement кейсов собрать отдельную очередь;
- определить, что обрабатывает LLM, что Claude audit, что РОП;
- не смешивать эти строки с auto-apply.

Результат: спорные звонки не теряются и не портят авто-исправления.

Факт выполнения: собран финальный пакет `stable_runtime/non_conversation_hard_gate_manual_review_queue_20260509_final/`. Очереди: `manual_review=60`, `keep_current=29`, всего non-auto=`89`. Эти строки не применялись и отделены от `safe_apply=5 672`.

### 12. Пересобрать downstream слои

Статус: `done`.

Что сделать после backfill:

- readiness/contact-layer;
- outcome linkage;
- sales moments;
- Knowledge Base;
- ROP pack;
- bot seed candidates;
- AMO-ready exports, если они зависят от исправленных полей.

Результат: улучшение качества проявляется во всех рабочих таблицах, а не только в raw analysis.

Факт выполнения: пересобраны canonical master, insight readiness/contact-layer readiness, outcome linkage, pilot sales moments, Knowledge Base, ROP validation pack, bot seed candidates и transcript-quality baseline. Canonical validation: `64 832/64 832` actionable звонков имеют ASR и полный R+A, missing=`0`. Readiness после backfill: contentful=`46 153`, non-conversation=`18 679` (минус `5 276` false-contentful относительно старого readiness). Pilot moments: `2 726`, non-conversation in moments=`0`. KB/ROP построены с trust-layer: `2 336` trusted GPT-5.5 reviews, `390` fallback строк вынесены в `llm_refresh_queue` и не используются для bot/best/ROP. ROP pack: `636` уникальных моментов, P0 no-live/artifact=`0`, revenue-risk no-live/artifact=`0`. Full test suite: `669 passed`. Подробный отчёт: `docs/TRANSCRIPT_QUALITY_STAGE12_DOWNSTREAM_REBUILD_REPORT_2026-05-10.md`.

AMO-ready root export намеренно не перезаписывался: live writeback не выполнялся, старый `АКТУАЛЬНО_AMO_ready.xlsx` считается stale, новый AMO-ready нужно формировать после stage 13 sanitizers и staged writeback policy.

### 13. Добавить bot/ROP sanitizers

Статус: `done`.

Что сделать:

- нормализация бренда: ASR-варианты вроде `НПК МФТИ`, `УНФК МФТИ`, `Чебенцентр`, `Черный центр` не должны попадать в идеальные ответы;
- разделить `Идеальный ответ для менеджера` и `Безопасный ответ для бота`;
- заменить цены, скидки, дедлайны, возвраты, рассрочки, личные имена на flags/placeholders;
- no-live не должен попадать в `Риски потери выручки`, `ТОП ответы`, `Черновики для бота`.

Результат: ROP/Telegram bot artifacts становятся безопасными для использования.

Факт выполнения: добавлен deterministic sanitizer layer для Knowledge Base, ROP pack и baseline. Теперь `ideal_answer_example` остается raw/source полем для аудита, `ideal_answer_manager_sanitized` используется как менеджерский скрипт, а `bot_safe_answer` используется как единственный источник для черновиков Telegram-бота. Sanitizer нормализует brand artifacts (`НПК МФТИ`, `УНФК`, `Чебенцентр`, `Черный центр` и похожие ASR-искажения), убирает конкретные цены, скидки, дедлайны, возвраты, рассрочки, юридические обещания и персональные данные. Все строки с sanitizer-заменами переводятся минимум в `needs_rop_validation`; fallback/dry-run и no-live остаются исключенными.

Артефакты stage 13:

- `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v3_stage13_sanitized/`
- `stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v3_stage13_sanitized/`
- `stable_runtime/transcript_quality_baseline_after_quality_backfill_20260510_v3_stage13_sanitized/`
- подробный отчет: `docs/TRANSCRIPT_QUALITY_STAGE13_SANITIZERS_REPORT_2026-05-10.md`

Ключевые acceptance metrics после stage 13: `kb_bot_ready_money_or_terms=0`, `kb_ideal_answer_brand_risk=0`, `kb_bot_safe_answer_brand_risk=0`, `kb_bot_safe_answer_personal_data_risk=0`, `rop_bot_candidate_money_or_terms=0`, `rop_bot_safe_answer_brand_risk=0`, `rop_bot_safe_answer_personal_data_risk=0`, `rop_p0_no_live_or_artifact=0`, `rop_revenue_risk_no_live_or_artifact=0`. Raw/source нагрузка, которую sanitizer теперь чистит: `raw_ideal_answer_brand_risk=75`, `raw_ideal_answer_money_or_terms=1 923`. Full test suite: `675 passed`.

### 14. Сравнить качество v1/v2

Статус: `done`.

Что сделать:

- сколько voicemail/IVR ушло из revenue risks;
- сколько ASR-мусора ушло из bot seeds;
- сколько живых диалогов защищено safeguards;
- сколько строк ушло в manual review;
- выборочно проверить новую таблицу Claude/GPT-аудитом.

Результат: честная метрика улучшения и список остаточных рисков.

Факт выполнения: добавлен воспроизводимый stage14 comparison builder и собран пакет `stable_runtime/transcript_quality_stage14_comparison_20260510_v1/`. Сравнение v2/v3 подтвердило: `kb_bot_ready_money_or_terms 552 -> 0`, `rop_bot_candidate_money_or_terms 85 -> 0`, `kb_ideal_answer_brand_risk 13 -> 0`, no-live/artifact в ROP P0/revenue осталось `0`. После аудита субагентов усилен stage13 sanitizer по hidden-risk паттернам (`до пятницы`, `10 апреля`, `10:00-12:00`, `10 процентов`, `50к`, `бронь/держим`, одиночные имена); stage13 и stage14 артефакты пересобраны. Дополнительный scan `bot_knowledge_seeds.csv` и `bot_knowledge_drafts.csv` показал `0` residual hits по weekday/date/time/spoken-percent/money-k/booking и brand/money/PII risk functions.

Stage14 acceptance: `passed=true`. Собрано `audit_sample.csv` на `200` уникальных строк без дублей `moment_id` и `over_sanitization_candidates.csv` на `250` строк. Важно: `over_sanitization_candidates` — не список ошибок, а очередь проверки полезности, потому что sanitizer стал жестче и часть ответов может быть безопасной, но слишком общей. Workbook для проверки: `stable_runtime/transcript_quality_stage14_comparison_20260510_v2_stage15_hardened/stage14_quality_comparison.xlsx`. Подробный отчет: `docs/TRANSCRIPT_QUALITY_STAGE14_COMPARISON_REPORT_2026-05-10_V2_STAGE15_HARDENED.md`.

### 15. Включить новый quality gate в постоянный pipeline

Статус: `done`.

Что сделать:

- подключить финальный quality gate ко всем будущим Analyze/R+A прогонам;
- добавить stage gate перед ROP/KB/bot/CRM export;
- обновить docs/runbook;
- добавить smoke-тесты, чтобы future regressions ловились до экспорта в CRM/таблицы.

Результат: проблема исправляется не только исторически, но и для всех новых звонков.

Факт выполнения: добавлен permanent export gate `transcript_quality_stage15_export_gate_v2_hardened`. Он проверяет Stage14 acceptance, нулевые residual bot-safe риски, нулевые baseline risk counters, консистентность input roots, уникальность audit sample и безопасную allowlist-схему для bot/RAG export. Gate создает отдельный `bot_export_allowlist.csv`, в который не попадают raw/manager answer fields, телефоны, менеджеры, source filenames и другие внутренние ROP/CRM поля. Реальный прогон на текущем corpus: `passed=true`, `bot_export_allowlist_rows=473`, `blocked_bot_export_rows=0`, `stage14_residual_risk_rows=0`, `bot_export risk_counts brand/money/personal_data/additional=0/0/0/0`.

Артефакты stage 15:

- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/bot_export_allowlist.csv`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/EXPORT_GATE_RUNBOOK.md`
- подробный отчет: `docs/TRANSCRIPT_QUALITY_STAGE15_EXPORT_GATE_REPORT_2026-05-10.md`

Важно: `bot_allowlist_export_ready=true`, но `bot_autonomous_production_ready=false`, потому что `over_sanitization_candidates=250` требует ROP-проверки полезности перед автономным ботом. Это не safety failure, а управляемый usefulness gate. CRM quality-writeback ready означает только качество данных; live CRM writeback по-прежнему требует отдельный dry-run/staged/live-confirmation контур и теперь CLI live-write дополнительно требует `--quality-gate-summary` на Stage15 `summary.json`.

Дополнительное усиление в рамках stage 15: после read-only аудита субагентов добавлена независимая проверка adversarial bot-export рисков, которые не должны зависеть только от Stage13 sanitizer: словесные суммы (`пятьдесят тысяч рублей`), `50 т.р.`, Telegram handles, дополнительные одиночные имена учеников/родителей и расширенные brand artifacts. После этого KB/ROP/baseline/Stage14/Stage15 пересобраны в hardened-версии: `sales_insight_knowledge_base_after_quality_backfill_20260510_v4_stage15_hardened`, `rop_validation_pack_after_quality_backfill_20260510_v4_stage15_hardened`, `transcript_quality_baseline_after_quality_backfill_20260510_v4_stage15_hardened`, `transcript_quality_stage14_comparison_20260510_v2_stage15_hardened`. Финальный Stage15 снова зеленый: `blocked_bot_export_rows=0`, все independent risk counters = `0`.

## Что делать следующим шагом

Следующий практический шаг: ROP-usefulness review для `over_sanitization_candidates.csv` и сбор ROP-approved golden dataset.

Порядок выполнения:

1. РОП проверяет 250 строк `over_sanitization_candidates.csv`: безопасный ответ не стал ли слишком общим и сохранил ли коммерческий смысл.
2. По проверенным строкам формируем `bot_golden_answers_approved.csv`: вопрос/сигнал/идеальная реакция/безопасный ответ/когда не использовать.
3. После ROP approval пересобираем Stage15 gate с пустой или approved usefulness queue и переводим `bot_autonomous_production_ready` в true.
4. Для CRM/AMO отдельно запускаем staged writeback preview, не live-write.
5. Затем можно двигаться к Telegram bot/RAG export и product UI.

## Что не делать

- Не применять `188 Claude safe_apply` напрямую: production-источник решений теперь GPT-only.
- Не ждать обязательного Claude-консенсуса для будущих прогонов: Claude остается внешним аудитом, а не блокером pipeline.
- Не применять full-corpus `would_update` без нового policy gate.
- Не писать в SQLite без backup/rollback manifest.
- Не запускать повторный ASR: проблема решается на уровне Analyze/quality/backfill.
- Не править Excel вручную: исправлять генераторы и downstream pipeline.
