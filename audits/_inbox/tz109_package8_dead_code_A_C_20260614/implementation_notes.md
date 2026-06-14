# TZ-109 Package 8 A+C implementation notes

Дата: 2026-06-14

## Что сделано

- Удалены три неиспользуемых Tallanto-модуля из `src/mango_mvp/amocrm_runtime/`.
- Добавлена заметка в `AGENTS.md` о замороженных legacy-слоях бота.

## Что не делали

- `subscription_llm_parts/monolith.py` не трогали.
- Пакеты 10/7 и единый brand resolver не трогали.
- Live AMO/Tallanto/CRM, ASR, Resolve+Analyze не запускали.
