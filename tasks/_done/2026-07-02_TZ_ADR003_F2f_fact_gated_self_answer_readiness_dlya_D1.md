> DONE 2026-07-02 04:27 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 04:23 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, tests/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_fact_gated_self_answer_readiness.py
Семантический-аудит: да

# TZ ADR-003 F2f: fact-gated self-answer readiness report

## Контекст

F2e доказал: на реальном 36ea110 есть 2 safe handoff по existence/format с точным KB-proof, но оба имеют текущий route `manager_only`.

Железное правило Ф3: active-понижение может трогать только `draft_for_manager`, не `manager_only`.

## Цель

Добавить report-only scorer, который отделяет:

- строгие Ф3-кандидаты: `draft_for_manager` + safe/self frame + exact product proof + нет P0/money/danger;
- exact-proof `manager_only`, которые нельзя понижать без отдельного policy/upstream решения;
- already-self exact proof;
- blocked/no-proof/danger.

## Scope

- Новый скрипт `scripts/report_adr003_fact_gated_self_answer_readiness.py`.
- Новые тесты.
- Audit pack с пересчётом 36ea110.
- Никакой runtime-проводки, direct path, provider, profile, P0 floor/preblock.

## Инварианты

- Report-only: route/text не меняются.
- `manager_only` не считается active-кандидатом даже при exact proof.
- `unknown/needs_slot` не считаются proof.
- P0/money/danger rows не считаются кандидатами.
- Текст не меняется и не генерируется.

## Acceptance

- Тесты доказывают strict draft candidate vs manager_only-needs-policy.
- На реальном 36ea110 report показывает, есть ли strict Ф3-кандидаты.
- Если strict Ф3-кандидатов 0, report явно говорит active NO-GO и почему.
- Audit pack ПДн-чистый.
