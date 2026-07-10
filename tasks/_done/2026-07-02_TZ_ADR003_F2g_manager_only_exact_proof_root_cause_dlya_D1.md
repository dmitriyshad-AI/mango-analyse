> DONE 2026-07-02 05:45 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 05:40 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, tests/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_manager_only_exact_proof_root_cause.py
Семантический-аудит: да

# TZ ADR-003 F2g: manager_only exact-proof root-cause report

## Контекст

F2f доказал: strict Ф3-кандидатов нет, но есть 2 строки `manager_only` с exact KB-proof.

Ф3 по железным правилам не имеет права понижать `manager_only`.

## Цель

Добавить report-only диагностику, которая объясняет, почему safe/exact-proof строки остались `manager_only`:

- что видел runtime retrieval;
- что было в `conversation_intent_plan`;
- что было в `answer_contract`;
- что сказал SemanticFrame;
- какие root-cause codes блокируют active.

## Scope

- Новый скрипт `scripts/report_adr003_manager_only_exact_proof_root_cause.py`.
- Новые тесты.
- Audit pack с пересчётом 36ea110.
- Никакой runtime-проводки, direct path, provider, profile, P0 floor/preblock.

## Инварианты

- Report-only: route/text не меняются.
- `manager_only` не становится active-кандидатом.
- Клиентские тексты не выводятся в markdown-отчёт.
- P0/money/danger и `check_availability` не трактуются как harmless context.

## Acceptance

- Реальный 36ea110 report показывает 2 `manager_only exact-proof` строки.
- Для обеих показано: runtime exact proof не был доставлен (`selected_exact_ids=[]`, `candidate_count=0`).
- Если frame сам говорит `manager_action`, это отдельная блокировка.
- Audit pack ПДн-чистый.
