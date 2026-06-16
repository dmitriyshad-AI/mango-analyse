# TZ-119 anti-dodumka guard implementation notes

Дата: 2026-06-16
Ветка: `codex/tz119-assumed-scope-guard`

## Что реализовано

- Добавлен флаг `TELEGRAM_ASSUMED_SCOPE_GUARD`, default OFF.
- Модельный режим ретривера `TELEGRAM_RETRIEVER_MODEL_DRIVEN` теперь активен только если одновременно включён `TELEGRAM_ASSUMED_SCOPE_GUARD`.
- Добавлен сбор провенанса слотов для direct path:
  - `confirmed_by_client` только при цитате клиента из `slot_provenance` / `slot_history` с source `dialogue_memory` или `memory_provenance`;
  - CRM/контекстные/бот-инференс слоты получают статус `assumed_from_context`.
- В hard scope фильтр фактов попадают только подтверждённые клиентом слоты.
- Неподтверждённые слоты не удаляются: они остаются как мягкая подсказка ранжирования и как явный статус в prompt.
- Retriever prompt и draft prompt при включённом флаге получают слоты в формате `{value, status}`.
- Добавлен мягкий страж финального direct path текста:
  - если клиентский ответ утверждает неподтверждённый класс/предмет/формат/продукт, бот задаёт один уточняющий вопрос;
  - маршрут не повышается к менеджеру;
  - P0/рисковые обращения страж не трогает.
- В `dynamic_summary.json` добавлен trace `assumed_scope_guard` внутри `fact_retrieval_trace`.

## Что не трогалось

- P0-пол и P0 latch.
- Brand guard и активный бренд канала.
- `claim_support` / `_claim_supported_by_facts`.
- Legacy rules engine / policy routing.
- Live AMO/Tallanto, ASR, Resolve+Analyze.

## Главный контракт

OFF = расширенный паритет с `main`.
ON = неподтверждённый класс/предмет/формат не режет факты жёстко, но не может быть представлен клиенту как подтверждённая истина.
