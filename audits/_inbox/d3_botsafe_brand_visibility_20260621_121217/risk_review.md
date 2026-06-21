# Risk Review

Главный риск: подмешать в prompt память чужого бренда.

Защита:

- Runtime helper разрешает только `active_brand` или `unknown`.
- Если tags содержат известный бренд, отличный от `active_brand`, chunk скрывается.
- Direct path повторно применяет то же правило, даже если upstream context ошибочно смешал items.
- Output brand verifier остается backstop.

Метрики на backup test-copy:

- `unknown_contains_brand_marker=0`
- `foton_contains_unpk=0`
- `unpk_contains_foton=0`
- `legacy_source_ref_count=0`
- `duplicate_source_ref_groups=0`
- `raw_allowed_chunks=0`

Остаточный риск:

- Unknown chunks могут быть слишком общими и не всегда полезными. Это риск качества, не бренд-безопасности; его должен показать M1/ручной разбор черновиков.
