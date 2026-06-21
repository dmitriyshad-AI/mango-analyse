# Implementation notes

N1 начат с проверки, как требовал промпт.

Фактический результат проверки: остаточных phone/email в сохранённых `bot_safe_summary` не найдено, поэтому код не менялся.

Проверенная тестовая БД:

`/tmp/mango_phase01_integration_final_20260621_102959/customer_timeline.sqlite`

Проверенные поля:

- `record_json.text`;
- `record_json.summary`;
- извлечённый фрагмент `Интерес: ...`.

Проверенные форматы:

- обычный email;
- обфусцированный email вида `name at domain dot ru`;
- телефон с `+7`, `7`, `8`;
- голый 10-значный телефон;
- телефон вида `999 123 45 67`.

Итог: фикс не нужен, потому что два слоя уже закрывают проверенный риск:

1. `_safe_interest_fragment` -> `_safe_fragment` -> `redact_text` + `LONG_DIGIT_TOKEN_RE`;
2. `scan_bot_safe_context_pii` перед подачей текста боту.
