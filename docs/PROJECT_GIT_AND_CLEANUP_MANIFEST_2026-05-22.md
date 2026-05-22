# Project Git And Cleanup Manifest

Дата: 2026-05-22

Статус: read-only инвентаризация и план порядка. Файлы не удалялись, не переносились, `git reset/git clean/git checkout` не запускались.

## 1. Короткий вывод

Рабочее дерево сейчас не "грязное случайно", а содержит несколько независимых пластов работы:

1. Текущая база знаний и поведение Telegram-ботов.
2. Документы и промпты для параллельных диалогов.
3. Customer timeline read-only слой.
4. Feedback/reporting/dynamic simulation для Telegram-пилота.
5. Звонки/Mango/runtime служебные изменения.
6. Gold candidates из звонков.
7. Изменения в `stable_runtime`-скриптах, которые нельзя трогать без отдельного решения.

Главный риск: если сейчас "почистить" папку физически, можно удалить артефакты параллельных диалогов или текущей базы бота. Поэтому правильный порядок: сначала зафиксировать группы, потом коммитить/архивировать по отдельным блокам, а удаление делать только после явного подтверждения по manifest.

## 2. Текущее git-состояние

- Ветка: `codex/git-order-20260513`.
- Изменения в `product_data/knowledge_base`: основной объем текущей работы по базе бота.
- Изменения в `stable_runtime`: 33 tracked shell/runtime файла и `CANONICAL_EXPORT.txt`; зона повышенного риска.
- Новые документы в `docs`: ТЗ, промпты, отчеты и планы.
- Новые `product_data/gold_candidates...`: candidate-паки из звонков.
- Новые scripts/tests для Telegram pilot, customer timeline, dynamic sim и KB/gold.

## 3. Размеры крупных зон

На момент инвентаризации:

| Путь | Размер | Комментарий |
|---|---:|---|
| `stable_runtime` | 38G | Не трогать без отдельного подтверждения. |
| `product_data/knowledge_base` | 482M | Текущие релизы КБ и пакеты для бота/сотрудников. |
| `.codex_local` | 363M | Локальные логи/запуски; чаще всего не коммитить. |
| `audits/_inbox` | 322M | Audit packs; не удалять без решения, но можно позже архивировать. |
| `docs` | 3.6M | Рабочая правда проекта и ТЗ. |

Gold candidate-паки из звонков:

| Путь | Размер | Рекомендация |
|---|---:|---|
| `product_data/gold_dialogues_paid_after_calls_leonova_kozlova_20260521_v1` | 5.5M | KEEP/CANDIDATE до разбора. |
| `product_data/gold_candidates_paid_proxy_after_calls_leonova_kozlova_20260521_v2` | 6.7M | ARCHIVE_CANDIDATE после выбора v3. |
| `product_data/gold_candidates_paid_proxy_after_calls_leonova_kozlova_20260521_v3_with_leonova_supplement` | 7.4M | KEEP как актуальнее v2. |
| `product_data/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v1` | 11M | ARCHIVE_CANDIDATE после выбора v3. |
| `product_data/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v2_rebuilt_current_runtime` | 10M | ARCHIVE_CANDIDATE после выбора v3. |
| `product_data/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v3_rebuilt_current_runtime` | 11M | KEEP как актуальнее v1/v2. |

## 4. Группы изменений

### G1. Текущая база знаний Telegram-ботов

Пути:

- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack/`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack/`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_handoff_for_claude_and_team/`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_sources/facts/`
- `scripts/build_kb_release_v6_1_team_answers.py`
- `scripts/build_kb_distribution_packs.py`
- `tests/test_kb_release_v3_import.py`
- `tests/test_kb_gold_answers_v3.py`
- `tests/test_kb_foton_presentation_format_facts.py`

Рекомендация: KEEP. Коммитить отдельным коммитом после финальной проверки текущей КБ и semantic review.

Нельзя смешивать с cleanup, stable_runtime или SaaS docs.

### G2. Telegram pilot behavior/runtime

Пути:

- `scripts/run_telegram_public_pilot_bots.py`
- `scripts/restart_telegram_public_pilot_bots.sh`
- `scripts/build_telegram_public_pilot_feedback_report.py`
- `scripts/run_telegram_dynamic_client_sim.py`
- `src/mango_mvp/channels/draft_prompt_builder.py`
- `src/mango_mvp/channels/subscription_llm.py`
- `src/mango_mvp/channels/telegram_pilot_context_builder.py`
- `tests/test_draft_prompt_builder.py`
- `tests/test_subscription_llm_draft_provider.py`
- `tests/test_telegram_pilot_context_builder.py`
- `tests/test_telegram_public_pilot_bots.py`
- `tests/test_telegram_dynamic_client_sim.py`
- `tests/test_telegram_public_pilot_feedback_report.py`

Рекомендация: KEEP. Коммитить отдельным коммитом после текущего цикла бота.

### G3. Параллельные prompts и планы

Пути:

- `docs/PARALLEL_NEW_LEAD_SALES_FUNNEL_CONTEXT_2026-05-22.md`
- `docs/PARALLEL_NEW_LEAD_SALES_FUNNEL_PROMPT_2026-05-22.md`
- `docs/PARALLEL_SAAS_ARCHITECTURE_PROMPT_2026-05-22.md`
- `docs/PARALLEL_HISTORICAL_CHANNELS_KB_CANDIDATES_PROMPT_2026-05-22.md`
- `docs/BUSINESS_AUDIT_ACTION_REGISTER_2026-05-21.md`

Рекомендация: KEEP. Можно коммитить отдельным легким docs-коммитом.

### G4. Telegram pilot docs

Пути:

- `docs/TELEGRAM_PILOT_EMPLOYEE_TESTING_GUIDE_2026-05-21.md`
- `docs/TELEGRAM_PILOT_FEEDBACK_REGISTER_2026-05-21.md`
- `docs/TZ_TELEGRAM_PILOT_DIALOG_QUALITY_V1_2026-05-21.md`
- `docs/TZ_TELEGRAM_PILOT_FEEDBACK_AND_STABILIZATION_2026-05-21.md`
- `docs/TZ_TELEGRAM_PILOT_FEEDBACK_HUMAN_CONTEXT_V2_2026-05-21.md`
- `docs/TZ_TELEGRAM_GOLD_ANSWERS_V3_INTEGRATION_2026-05-22.md`

Рекомендация: KEEP. Можно объединить с G2 или отдельным docs-коммитом, но не смешивать с KB generated files.

### G5. Customer timeline read-only

Пути:

- `.gitignore` change: `product_data/customer_timeline/canonical_readonly_*/`
- `docs/TZ_CANONICAL_READONLY_CUSTOMER_TIMELINE_2026-05-21.md`
- `scripts/build_canonical_readonly_customer_timeline.py`
- `scripts/triage_canonical_readonly_customer_timeline.py`
- `src/mango_mvp/customer_timeline/__init__.py`
- `src/mango_mvp/customer_timeline/canonical_readonly_import.py`
- `src/mango_mvp/customer_timeline/canonical_readonly_triage.py`
- `tests/test_customer_timeline_canonical_readonly_import.py`
- `tests/test_customer_timeline_canonical_readonly_triage.py`

Рекомендация: KEEP, но отдельным коммитом. Не включать `timeline_primary_read_enabled`; не менять live-поведение бота без отдельного решения.

### G6. Calls/Mango/runtime support

Пути:

- `docs/MANGO_CALLS_UPDATE_RUNBOOK_2026-05-21.md`
- `docs/ASR_RUNTIME_CONTRACT_2026-05-21.md`
- `docs/ASR_RUNTIME_INCIDENT_AND_CLEANUP_RISK_REPORT_2026-05-21.md`
- `scripts/build_canonical_after_mango_update.py`
- `scripts/check_asr_runtime_contract.py`
- `scripts/run_crm_writeback_quality_gate.py`
- `src/mango_mvp/quality/crm_text_quality_detector.py`
- `tests/fixtures/crm_text_quality_cases.jsonl`
- `tests/test_crm_text_quality_detector.py`

Рекомендация: KEEP/REVIEW. Коммитить только после подтверждения диалога по звонкам. Не смешивать с Telegram-ботом.

### G7. Stable runtime tracked changes

Пути:

- `stable_runtime/CANONICAL_EXPORT.txt`
- `stable_runtime/*.sh`

Рекомендация: DO_NOT_TOUCH сейчас.

Причина: это зона, которую инструкции запрещают менять без отдельного подтверждения. Изменения небольшие по diff, но потенциально важные. Их нельзя удалять, откатывать или коммитить как "cleanup" без владельца блока звонков/runtime.

### G8. Gold candidates из звонков

Пути:

- `product_data/gold_candidates_paid_proxy_after_calls_*`
- `product_data/gold_dialogues_paid_after_calls_*`

Рекомендация:

- KEEP актуальные v3-папки;
- ARCHIVE_CANDIDATE старые v1/v2 после подтверждения;
- не коммитить все версии без решения, потому что это candidate data, а не исходный код.

### G9. Документы безопасности/угроз

Пути:

- `docs/THREAT_MODEL.md`

Рекомендация: REVIEW. Коммитить отдельно, если это результат security-блока. Не смешивать с bot/KB.

## 5. Устаревшее и временное

Ниже не список на удаление, а список на подтверждение.

## 5.1. Дополнительные наблюдения read-only ревью

Субагентская проверка подтвердила:

- рабочее дерево содержит около сотни изменённых tracked-файлов и много untracked-артефактов;
- новые `product_data/gold_*` папки являются candidate data и не должны автоматически попадать в git;
- активный Telegram polling сейчас нельзя трогать через cleanup;
- `docs/CURRENT_STATE.md` отстаёт от фактического runtime 2026-05-21 и требует отдельного обновления, но не в рамках физической уборки;
- `stable_runtime/*.sh` содержит массовые launcher/runtime изменения, которые нельзя коммитить как cleanup;
- `.venv-asrbench`, `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/`, `stable_runtime/messages35_asr_only_20260506/`, старые локальные архивы и Telegram exports могут быть lineage/runtime источниками, а не мусором.

### ARCHIVE_CANDIDATE

- Старые gold candidate версии, если актуальной считается v3:
  - `product_data/gold_candidates_paid_proxy_after_calls_leonova_kozlova_20260521_v2/`
  - `product_data/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v1/`
  - `product_data/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v2_rebuilt_current_runtime/`
- Старые внешние handoff-папки в корне проекта:
  - `claude_to_codex_v3_handoff_2026-05-17/`
- Старые TZ root-файлы `Mango_Analyse_TZ_*` после сверки, что они продублированы в `docs/`.

### IGNORE_CANDIDATE

- `.codex_local/`
- `audits/_inbox/` для больших локальных пакетов, если они еще не игнорируются.
- `product_data/customer_timeline/canonical_readonly_*/` уже добавлено в `.gitignore`.
- Будущие локальные dynamic simulation run outputs.

### DELETE_ONLY_AFTER_CONFIRMATION

Удалять можно только после отдельного "да, удалить именно эти пути":

- старые candidate-папки v1/v2 после выбора актуальной v3;
- старые external Claude handoff-папки после переноса в `audits/_inbox` или `docs/`;
- временные smoke/mega output-папки, если они продублированы в audit pack.

### DO_NOT_TOUCH

- `stable_runtime/**`
- `.venv-asrbench`
- `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/`
- `stable_runtime/messages35_asr_only_20260506/`
- `2026-03-09--26`
- `telegram_exports (2)`
- `_local_archive_mango_api_downloads_20260507/product_appliance`
- `.codex_local/telegram_pilot_bots/runtime/**`, пока боты запущены/тестируются.
- активный KB release `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers*`
- текущие audit packs последнего цикла бота.
 - будущие выходы параллельных диалогов:
  - `docs/historical_channels_kb_candidates_2026-05-22/`
  - `docs/new_lead_sales_funnel_2026-05-22/`
  - `docs/saas_architecture_parallel_2026-05-22/`

## 6. Минимальный безопасный порядок дальше

1. Дождаться, пока 3 параллельных диалога завершат текущие read-only задачи.
2. В этом диалоге сделать финальный `git status --short`.
3. Отдельно зафиксировать docs/prompts-коммит.
4. Отдельно зафиксировать KB/bot-коммит после текущего тестового цикла.
5. Отдельно зафиксировать customer timeline read-only коммит.
6. Stable runtime изменения вынести в отдельное решение с диалогом по звонкам.
7. Только после этого вернуться к физической уборке/архивации.
