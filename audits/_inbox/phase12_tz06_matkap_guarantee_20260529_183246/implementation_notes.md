# Phase 12 TZ-06 — matkap-guarantee

## Изменения

- В v2 safe-template dispatcher добавлен `matkap` spec с priority `40`.
- Producer переиспользует существующий `_matkap_safe_template(...)`.
- Маршрут соответствует legacy-логике: `manager_only` сохраняется, остальные случаи переводятся в `draft_for_manager`.
- Тексты не переписаны: используются утвержденные `MATKAP_REGIONAL_SAFE_TEXT`, `MATKAP_SFR_REVIEW_SAFE_TEXT`, `MATKAP_FEDERAL_TIMING_SAFE_TEXT`.

## Границы

- TZ-08/TZ-09 не трогались.
- Refund-latch E5_payment_02 не трогался.
- KB/source-YAML/jsonl не менялись.
- LLM-вызовы и 212-прогоны не запускались.
