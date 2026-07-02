# ADR-003 F2h Exact-Proof Injection Shadow

## Что сделано

Добавлен report-only scorer `scripts/report_adr003_exact_proof_injection_shadow.py`.

Он проверяет гипотезу: если exact KB-proof из F2f/F2g был бы доставлен в runtime telemetry, какие блокеры всё равно остались бы.

## Что это НЕ делает

- Не меняет route/text.
- Не подключается к direct path/provider/profile.
- Не включает флаги.
- Не понижает `manager_only`.
- Не инжектит факт в runtime.

## Реальный пересчёт 36ea110

См. `real_36ea110/adr003_exact_proof_injection_shadow_report.md`:

- manager-only exact-proof rows: 2;
- fresh client-safe proof after hypothetical injection: 2;
- evidence-only sufficient rows: 0;
- rows still blocked after injection: 2.

## Вывод

Одной доставки exact-proof факта недостаточно. Остаются:

- `route_is_manager_only`: 2;
- `message_type_context_update`: 2;
- `runtime_missing_live_or_operational_facts`: 2;
- frame-блокеры: low confidence у одной строки, manager_action/check_availability у другой.

Следующий безопасный шаг — не active, а отдельный reggrade Claude #1 и решение, какую shadow-гипотезу проверять дальше: retrieval/evidence delivery или frame calibration.
