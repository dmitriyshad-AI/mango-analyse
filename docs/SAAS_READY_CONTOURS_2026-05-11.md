# SaaS-ready contours

Дата: 2026-05-11
Цель: зафиксировать продуктовый подход, чтобы проект развивался как переносимый сервис, а не как локальный набор разовых скриптов.

## Tenant config

Уже есть базовая tenant-policy логика: защищенные поля, режимы writeback, safety gates. Следующий уровень - вынести в tenant config:

- CRM provider: `amocrm`, future adapters.
- Telephony provider: `mango`, future adapters.
- Allowed write fields по contact/deal.
- Protected fields, которые нельзя трогать ни при каком режиме.
- Brand dictionary, teacher/address/person policies.
- Bot/CRM/internal redaction policy отдельно.
- Quality thresholds: CRM text, bot safety, writeback relevance, entity resolution.

## Adapter boundaries

Для SaaS нельзя связывать бизнес-логику с AMO/Mango напрямую. Нужны стабильные интерфейсы:

- `TelephonyCaptureAdapter`: list calls, download recording, normalize metadata.
- `ASRAdapter`: transcribe, compare variants, expose confidence.
- `CRMReadAdapter`: contacts/deals lookup, field schema, duplicates.
- `CRMWriteAdapter`: dry-run payload, live write, readback.
- `LearningOutcomeAdapter`: Tallanto/current LMS/payment/outcome layer.

AMO и Mango остаются первыми реализациями этих интерфейсов, но не должны быть единственным форматом данных внутри системы.

## Safety gates

Перед любым live-write:

1. Exact input binding: gate input == writeback input.
2. Independent population recall: не только self-detected blockers.
3. Protected field guard.
4. Entity resolution guard: one surviving contact or explicit human resolution.
5. CRM text quality guard: no ellipsis, no duplicate count artifacts, no contradictory next step.
6. Dry-run with real runtime.
7. Explicit approval artifact.
8. Post-writeback readback gate.

## Current constraints

- SQLite остается приемлемым для локального batch/single-tenant режима.
- PostgreSQL нужен перед multi-tenant production, concurrent workers и hosted SaaS.
- Live Mango API/capture и AMO writeback должны идти через adapters and gates, а не через прямые одноразовые scripts.

## Не делать

- Не добавлять новых live-write shortcuts.
- Не строить новые отчеты от устаревшего April export.
- Не смешивать bot-safe sanitizer с CRM-internal sanitizer.
- Не использовать Claude/GPT audit как единственный gate без frozen corpus/population counter.
