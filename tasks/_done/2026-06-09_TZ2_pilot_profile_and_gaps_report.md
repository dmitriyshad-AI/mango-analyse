# TZ2 pilot profile and gaps report

Дата: 2026-06-09
Ветка: main

## Read-only карта

- `pilot_gold_v1`: до правки включал direct path и gold pack, но не включал единым профилем `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER`, `TELEGRAM_OUTPUT_SANITIZER`, `TELEGRAM_NUMBER_GATE_SCOPE_AWARE`, `TELEGRAM_VERIFIER_HANDOFF_CLAIMS` и presale-подфлаги.
- `subscription_llm.py:_direct_path_legacy_context_fact_items`: legacy/upstream факты попадали в prompt без повторной проверки `brand`, `allowed_for_client_answer/client_safe`, `forbidden_for_client`, `internal_only`, `valid_until`.
- `apply_authoritative_output_gate`: direct path уже проходит финальный gate, но отдельной находки для promise-срока менеджера не было.
- `dialogue_contract_pipeline.py:number_gate_scope_aware_enabled`: скоуповый number-gate жил в отдельном модуле и не знал про `pilot_gold_v1`.

## Что изменено

Коммит: `d6416a28 Harden pilot profile direct path guards`

- `pilot_gold_v1` теперь в коде включает боевой профиль: direct path, gold pack, semantic verifier, output sanitizer, scope-aware number gate, verifier handoff claims и presale 1-4.
- Явный override конкретного флага через context/env сохраняется и пишется в `direct_path.pilot_profile_overrides`.
- Legacy/upstream факты direct path повторно фильтруются по active_brand/client_safe/valid_until перед попаданием в prompt.
- Добавлен gate finding `unsupported_manager_deadline_promise`: сроковые обещания действия менеджера переводят direct path в `draft_for_manager` с сохранением текста для менеджера; обычное «менеджер свяжется» без срока не трогается.
- `dialogue_contract_pipeline.number_gate_scope_aware_enabled` включает `TELEGRAM_NUMBER_GATE_SCOPE_AWARE` через `pilot_gold_v1`, если нет явного override.

## NEG и проверки

- `pilot_gold_v1` включает все профильные флаги.
- Явный override `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER=0` виден в metadata и реально выключает флаг.
- Без `pilot_gold_v1` дефолты остаются OFF.
- Upstream факты wrong brand / not client-safe / expired не попадают в direct prompt; валидный факт проходит.
- «Менеджер свяжется завтра утром» получает finding и downgrade_keep_text.
- «Менеджер свяжется» без срока проходит.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k 'pilot_gold_v1 or legacy_context_filters or manager_deadline_promise or presale or pii_echo or source_id or verifier_failsoft'`
  - `25 passed, 382 deselected`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_contract_pipeline.py -k 'number_scope_aware or number_gate or unsupported_product_number'`
  - `15 passed, 260 deselected`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `2840 passed, 2 skipped, 1 warning`

## Остаточный риск

- Симулятор и M1-регрейд не запускались по ТЗ. Это formal_pass; смысловую проверку клиентских эффектов должен закрыть следующий регрейд по сырью.
