> DONE 2026-07-02 06:10 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 06:03 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, tests/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_exact_proof_injection_shadow.py
Семантический-аудит: да

# TZ ADR-003 F2h: exact-proof injection shadow report

## Контекст

F2g показал: 2 `manager_only + exact-proof` строки имеют свежий KB-факт, но runtime retrieval его не доставил.

## Цель

Добавить report-only shadow, который отвечает на вопрос:

> Если бы exact-proof факт был доставлен в runtime telemetry, достаточно ли этого для автономии?

## Scope

- Новый скрипт `scripts/report_adr003_exact_proof_injection_shadow.py`.
- Новые тесты.
- Audit pack с пересчётом 36ea110.
- Никакой runtime-проводки, direct path, provider, profile, P0 floor/preblock.

## Инварианты

- Report-only: route/text не меняются.
- Exact-proof injection — только гипотеза в отчёте, не runtime.
- `manager_only` не становится active-кандидатом.
- Fresh/client-safe proof проверяется по `valid_until`.
- Residual blockers остаются явными.

## Acceptance

- Реальный 36ea110 report показывает, достаточно ли одного exact proof.
- Если остаются route/frame/message_type/missing_facts blockers, active остаётся NO-GO.
- Audit pack ПДн-чистый.
