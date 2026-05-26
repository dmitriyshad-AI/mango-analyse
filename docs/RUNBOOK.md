# Runbook

Дата обновления: 2026-05-23

Назначение: безопасные команды и рабочие правила для проекта.

## Перед любой работой

Проверить ветку и рабочую папку:

```bash
git branch --show-current
git status --short
```

Посмотреть актуальное состояние:

```bash
sed -n '1,220p' docs/CURRENT_STATE.md
sed -n '1,220p' docs/DECISIONS_LOG.md
sed -n '1,220p' docs/ROADMAP.md
```

## Безопасный сбор тестов

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
```

## Безопасный точечный запуск тестов

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q <tests>
```

Пример:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_post_backfill_amo_ready_export.py
```

## Смысловая проверка

Для базы знаний, Telegram/email-черновиков, CRM-текстов и клиентских ответов зеленые тесты дают только `formal_pass`.

Перед словами "готово к использованию" нужен `semantic_pass` по правилам:

```bash
sed -n '1,260p' docs/SEMANTIC_REVIEW_RULES.md
```

Для базы знаний запускать:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_kb_semantic_review.py \
  --release-dir product_data/knowledge_base/kb_release_20260520_v6_3_team_answers \
  --out-dir audits/_inbox/<block>/semantic_review
```

Если `semantic_pass=false`, блок не считается завершенным, даже если `quality_passed=true`.

## Разбор проблем бота по классам

Для FAIL/PASS_WITH_NOTES в Telegram-пилоте использовать skill:

```text
/Users/dmitrijfabarisov/.codex/skills/bot-failure-class-review/SKILL.md
```

Реестр классов:

```text
docs/BOT_FAILURE_CLASSES_REGISTRY.md
```

Правило: не чинить один пример как отдельную фразу, пока не понятно, это единичный случай, проблема теста или повторяемый класс.

## Что не запускать без отдельного подтверждения

- ASR;
- Resolve+Analyze по реальным данным;
- live AMO write;
- live CRM write;
- Tallanto write;
- массовые batch/start/run-ui скрипты;
- скрипты, которые пишут в `stable_runtime` как рабочее состояние;
- удаление или перенос runtime-папок.

## AMO Snapshot / Rollback

Live-запись deal-aware полей теперь обязана создать:

```text
pre_write_snapshot.jsonl
pre_write_snapshot.csv
rollback_manifest.json
live_write_report.csv
live_write_report.json
summary.json
```

Rollback по умолчанию запускать только в dry-run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/rollback_deal_aware_amo_fields.py \
  --live-run-root <live_run_root> \
  --dry-run
```

Реальный rollback запрещен без отдельного подтверждения Дмитрия. Для `--apply` нужен отдельный token:

```text
ROLLBACK_DEAL_AWARE_AMO_FIELDS
```

Первый live-микропилот после Блока A ограничен 1-5 сделками.

## Runtime

Текущие указатели можно читать:

```bash
sed -n '1,220p' stable_runtime/CURRENT_RUNTIME.json
cat stable_runtime/CANONICAL_EXPORT.txt
```

Но нельзя менять `stable_runtime` DB/audio/transcripts без отдельного подтверждения.

## Telegram Pilot

Токены и настройки локального пилота:

```text
/Users/dmitrijfabarisov/.codex/mango_telegram_pilot_bots.env
```

Безопасная проверка настроек без отправки клиентам:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_public_pilot_bots.py \
  --env-file /Users/dmitrijfabarisov/.codex/mango_telegram_pilot_bots.env \
  --mode getme \
  --brand all
```

Перезапуск локальных публичных ботов:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src bash scripts/restart_telegram_public_pilot_bots.sh
```

Важно: `--mode poll` может реально отправлять ответы в публичные Telegram-боты. Запускать только когда понятно, что сотрудники тестируют пилот.

Логи:

```text
.codex_local/telegram_pilot_bots/logs/
.codex_local/telegram_pilot_bots/runtime/
```

Единый store пилота:

```text
.codex_local/telegram_pilot/telegram_pilot.sqlite
```

Дневной отчёт из store:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_telegram_pilot_daily_report.py \
  --db .codex_local/telegram_pilot/telegram_pilot.sqlite \
  --date YYYY-MM-DD \
  --out-dir audits/_inbox/telegram_pilot_daily_YYYYMMDD
```

Импорт feedback сотрудников из `employee_review_sheet.csv`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/import_telegram_pilot_feedback.py \
  --db .codex_local/telegram_pilot/telegram_pilot.sqlite \
  --csv audits/_inbox/telegram_pilot_daily_YYYYMMDD/employee_review_sheet.csv \
  --actor nastya
```

База знаний для ботов:

```text
product_data/knowledge_base/kb_release_20260520_v6_3_team_answers
product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack
```

Каноничная сборка KB v6.3:

```text
docs/KB_BUILD_RUNBOOK_2026-05-26.md
```

Порядок быстрых проверок бота:

1. preflight актуальности v8-фактов;
2. точечные тесты DialogueMemory/answer-quality/journal;
3. `v8_targeted16`;
4. статичные `MEGA_autonomy_tests_v6` и `MEGA_multitopic_batch_v5`;
5. полный v8 только отдельным длинным прогоном с `--resume`, полными транскриптами и review queue.

Точечные тесты DialogueMemory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_dialogue_memory.py \
  tests/test_telegram_pilot_context_builder.py \
  tests/test_answer_quality_rewriter.py \
  tests/test_telegram_pilot_journal_report.py \
  tests/test_telegram_dynamic_client_sim.py \
  tests/test_subscription_llm_draft_provider.py
```

Быстрый параллельный запуск динамического симулятора:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios "/Users/dmitrijfabarisov/Claude Projects/Foton/v8_dynamic_sim_2026-05-22/v8_targeted16_2026-05-22.jsonl" \
  --out-dir audits/_inbox/telegram_dynamic_v8_targeted16_YYYYMMDD_HHMMSS \
  --parallel 2 \
  --resume
```

Для `codex`-режима рекомендуемый старт: `--parallel 2`. Если нет таймаутов и деградации, можно пробовать `--parallel 3`. Параллельность 10-50 для LLM-прогона не использовать как обычный acceptance: это скорее нагрузочный эксперимент, может давать нестабильность из-за лимитов Codex/модели.

Безопасный локальный smoke на одновременные обращения без Telegram-отправок:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_pilot_concurrency_smoke.py \
  --mode fake \
  --requests 50 \
  --concurrency 10 \
  --out-dir audits/_inbox/telegram_pilot_concurrency_smoke_YYYYMMDD
```

Если нужно проверить реальную LLM-задержку без отправки сообщений клиентам, можно запустить малый codex-smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_pilot_concurrency_smoke.py \
  --mode codex \
  --requests 10 \
  --concurrency 2 \
  --out-dir audits/_inbox/telegram_pilot_concurrency_codex_smoke_YYYYMMDD
```

## Audit Pack

После значимого блока создать:

```text
audits/_inbox/<block>_<timestamp>/
```

Минимальные файлы:

```text
implementation_notes.md
changed_files.txt
test_output.txt
semantic_review.md
risk_review.md
backward_compatibility.md
```

Для AMO/writeback блоков добавить:

```text
snapshot_contract.md
rollback_contract.md
dry_run_summary.md
readback_plan.md
live_write_status.md
```

Если live-write не запускался, явно написать:

```text
Live write was not executed.
```

## Коммиты

Коммитить только после:

1. понятного diff;
2. тестов;
3. audit pack;
4. проверки, что в коммит не попали runtime-артефакты;
5. проверки, что нет чужих unrelated изменений.

Не смешивать в один коммит:

- код разных блоков;
- документы и runtime-выгрузки;
- cleanup и функциональную реализацию;
- live отчеты и исходный код.

## Текущий порядок реализации

Актуальный порядок для этого диалога:

```text
Telegram pilot journal -> dialogue strategy -> docs -> v8_targeted16 -> static v6/v5 -> audit pack
```

Исторический порядок `G -> A -> PBF -> B -> C -> D -> E` по AMO/deal-aware/customer timeline считать фундаментом, но не текущим основным блоком.

## Customer Timeline Coverage

Безопасный read-only отчет покрытия:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/audit_customer_timeline_coverage.py \
  --deal-aware-candidates <path/to/deal_stage4_deal_candidates.csv> \
  --timeline-db <path/to/customer_timeline.sqlite> \
  --out-root <path/to/audit_output>
```

Нельзя писать output в `stable_runtime` без отдельного подтверждения.

Stage 4 preview может читать customer timeline только по явному флагу:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_deal_aware_stage4_preview.py \
  --customer-timeline-db <path/to/customer_timeline.sqlite> \
  --enable-customer-timeline-context
```

Этот контекст остается только в preview/report полях и не попадает в AMO payload.

## Навыки Codex

Для security-аудита использовать skills:

- `security-best-practices`;
- `security-threat-model`;
- `security-ownership-map`.

Для PDF:

- `pdf`;
- Python-среда: `/Users/dmitrijfabarisov/.codex/skill-venv/bin/python`.

Для notebooks:

- `jupyter-notebook`;
- Python-среда: `/Users/dmitrijfabarisov/.codex/skill-venv/bin/python`.

Для долговечных CLI:

- `cli-creator`.

Не использовать эти skills для простых вопросов и мелких правок.
