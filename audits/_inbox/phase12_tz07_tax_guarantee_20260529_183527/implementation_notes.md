# Phase 12 TZ-07 — tax-guarantee

## Изменения

- В v2 safe-template dispatcher добавлен `tax` spec с priority `41`.
- Producer использует существующий `_tax_safe_template(...)`, но перед ним уточняет v2-разбор:
  - вопросы про решение/гарантию ФНС без суммы → `TAX_FNS_REVIEW_SAFE_TEXT`;
  - вопросы про оформление вычета по онлайн/дистанционному формату → `TAX_ONLINE_FORM_SAFE_TEXT`;
  - суммы/лимиты остаются в `TAX_AMOUNT_SAFE_TEXT`.
- Маршрут соответствует legacy-логике: `manager_only` сохраняется, остальные случаи переводятся в `draft_for_manager`.

## Границы

- Юр.номера лицензий не добавлялись.
- KB/source-YAML/jsonl не менялись.
- TZ-08/TZ-09 не трогались.
- Refund-latch E5_payment_02 не трогался.
- LLM-вызовы и 212-прогоны не запускались.
