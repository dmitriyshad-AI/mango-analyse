# Project Commit Plan

Дата: 2026-05-22

Статус: план. Ничего не staged и не committed в рамках этой инвентаризации.

## 1. Принцип

Коммиты должны быть маленькими и логичными. Нельзя смешивать:

- bot/KB;
- runtime/stable_runtime;
- звонки;
- customer timeline;
- docs/prompts;
- audit packs;
- cleanup.

## 2. Предлагаемый порядок коммитов

### Commit 1. Parallel dialogue prompts

Цель: зафиксировать безопасные промпты для параллельных диалогов.

Файлы:

- `docs/PARALLEL_NEW_LEAD_SALES_FUNNEL_CONTEXT_2026-05-22.md`
- `docs/PARALLEL_NEW_LEAD_SALES_FUNNEL_PROMPT_2026-05-22.md`
- `docs/PARALLEL_SAAS_ARCHITECTURE_PROMPT_2026-05-22.md`
- `docs/PARALLEL_HISTORICAL_CHANNELS_KB_CANDIDATES_PROMPT_2026-05-22.md`
- `docs/PROJECT_GIT_AND_CLEANUP_MANIFEST_2026-05-22.md`
- `docs/PROJECT_COMMIT_PLAN_2026-05-22.md`

Риск: низкий, docs-only.

### Commit 2. Telegram pilot docs

Цель: зафиксировать ТЗ и инструкции пилота.

Файлы-кандидаты:

- `docs/TELEGRAM_PILOT_EMPLOYEE_TESTING_GUIDE_2026-05-21.md`
- `docs/TELEGRAM_PILOT_FEEDBACK_REGISTER_2026-05-21.md`
- `docs/TZ_TELEGRAM_PILOT_DIALOG_QUALITY_V1_2026-05-21.md`
- `docs/TZ_TELEGRAM_PILOT_FEEDBACK_AND_STABILIZATION_2026-05-21.md`
- `docs/TZ_TELEGRAM_PILOT_FEEDBACK_HUMAN_CONTEXT_V2_2026-05-21.md`
- `docs/TZ_TELEGRAM_GOLD_ANSWERS_V3_INTEGRATION_2026-05-22.md`

Риск: низкий/средний. Перед коммитом быстро сверить, что документы не содержат секреты и устаревшую "готовность к клиентскому потоку".

### Commit 3. Knowledge base v6.3 updates

Цель: зафиксировать актуальную базу знаний, gold answers и пакеты для бота/сотрудников.

Файлы-кандидаты:

- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers*/`
- `scripts/build_kb_release_v6_1_team_answers.py`
- `scripts/build_kb_distribution_packs.py`
- `tests/test_kb_release_v3_import.py`
- `tests/test_kb_gold_answers_v3.py`
- `tests/test_kb_foton_presentation_format_facts.py`

Перед коммитом:

- запустить точечные KB-тесты;
- сохранить audit pack;
- проверить `semantic_pass`.

Риск: высокий по бизнес-смыслу, но технически управляемый.

### Commit 4. Telegram bot behavior and feedback loop

Цель: зафиксировать runtime/logic улучшения бота.

Файлы-кандидаты:

- `src/mango_mvp/channels/draft_prompt_builder.py`
- `src/mango_mvp/channels/subscription_llm.py`
- `src/mango_mvp/channels/telegram_pilot_context_builder.py`
- `scripts/run_telegram_public_pilot_bots.py`
- `scripts/restart_telegram_public_pilot_bots.sh`
- `scripts/build_telegram_public_pilot_feedback_report.py`
- `scripts/run_telegram_dynamic_client_sim.py`
- `tests/test_draft_prompt_builder.py`
- `tests/test_subscription_llm_draft_provider.py`
- `tests/test_telegram_pilot_context_builder.py`
- `tests/test_telegram_public_pilot_bots.py`
- `tests/test_telegram_public_pilot_feedback_report.py`
- `tests/test_telegram_dynamic_client_sim.py`

Перед коммитом:

- точечные Telegram tests;
- быстрый smoke/dynamic subset;
- semantic review примеров.

Риск: высокий, потому что влияет на live-ботов.

### Commit 5. Customer timeline read-only layer

Цель: зафиксировать read-only customer timeline слой и triage.

Файлы-кандидаты:

- `.gitignore`
- `docs/TZ_CANONICAL_READONLY_CUSTOMER_TIMELINE_2026-05-21.md`
- `scripts/build_canonical_readonly_customer_timeline.py`
- `scripts/triage_canonical_readonly_customer_timeline.py`
- `src/mango_mvp/customer_timeline/__init__.py`
- `src/mango_mvp/customer_timeline/canonical_readonly_import.py`
- `src/mango_mvp/customer_timeline/canonical_readonly_triage.py`
- `tests/test_customer_timeline_canonical_readonly_import.py`
- `tests/test_customer_timeline_canonical_readonly_triage.py`

Перед коммитом:

- тесты customer timeline;
- подтвердить, что нет write в `stable_runtime`, AMO, Tallanto.

Риск: средний.

### Commit 6. Calls/runtime support docs and quality detector

Цель: зафиксировать изменения диалога по звонкам, если владелец блока подтверждает.

Файлы-кандидаты:

- `docs/MANGO_CALLS_UPDATE_RUNBOOK_2026-05-21.md`
- `docs/ASR_RUNTIME_CONTRACT_2026-05-21.md`
- `docs/ASR_RUNTIME_INCIDENT_AND_CLEANUP_RISK_REPORT_2026-05-21.md`
- `scripts/build_canonical_after_mango_update.py`
- `scripts/check_asr_runtime_contract.py`
- `scripts/run_crm_writeback_quality_gate.py`
- `src/mango_mvp/quality/crm_text_quality_detector.py`
- `tests/fixtures/crm_text_quality_cases.jsonl`
- `tests/test_crm_text_quality_detector.py`

Не включать `stable_runtime/*.sh` автоматически.

### Commit 7. Stable runtime script changes

Цель: только если отдельно подтверждено.

Файлы:

- `stable_runtime/CANONICAL_EXPORT.txt`
- `stable_runtime/*.sh`

Перед любым действием:

- выяснить источник изменений;
- получить подтверждение владельца runtime/звонков;
- не делать reset/checkout/clean;
- не коммитить как cleanup.

Риск: высокий.

## 3. Что пока не коммитить

- `.codex_local/**`
- `audits/_inbox/**` целиком без отбора;
- старые `product_data/gold_candidates...` все версии сразу;
- любые live logs;
- любые secret/env файлы;
- большие runtime artifacts.

## 4. Что можно сделать после коммитов

После фиксации основных блоков можно подготовить отдельный cleanup PR/коммит:

1. Добавить недостающие ignore-паттерны.
2. Перенести старые handoff/candidate пакеты в архивную структуру.
3. Удалить подтвержденные устаревшие папки.
4. Обновить `docs/NEXT_SAFE_CLEANUP_CANDIDATES_2026-05-12.md`.

Удаление только по отдельному подтвержденному списку путей.

