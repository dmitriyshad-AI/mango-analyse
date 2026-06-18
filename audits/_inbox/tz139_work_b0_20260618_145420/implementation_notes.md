# TZ139 Work B0 Implementation Notes

Дата: 2026-06-18

Статус: B0 выполнен, B1 не начинался. Стоп на регрейд Claude.

## Что сделано

- В `src/mango_mvp/customer_timeline/canonical_readonly_import.py:649` линк `mango_client_phone` теперь использует общий `phone_match_class` и `phone_confidence`.
- В `tests/test_customer_timeline_canonical_readonly_import.py:283` и `tests/test_customer_timeline_canonical_readonly_import.py:349` family-тесты теперь проверяют оба link type: `phone` и `mango_client_phone`.
- Для shared family phone оба link type должны быть `ambiguous` с confidence `0.55`.

## Дифф кратко

- Было: `phone` для семьи `ambiguous/0.55`, но `mango_client_phone` оставался `strong_unique/0.95`.
- Стало: `phone` и `mango_client_phone` для семьи используют одну политику `ambiguous/0.55`.
- Non-family `mango_client_phone` остается `strong_unique/0.95`.

## Read-Only

- Источник читался только как CSV + in-memory расчет.
- AMO/Tallanto/CRM writes: нет.
- `stable_runtime` writes: нет.
- ASR/R+A/analyze: нет.
- Писать разрешалось только в temp DB внутри pytest.
