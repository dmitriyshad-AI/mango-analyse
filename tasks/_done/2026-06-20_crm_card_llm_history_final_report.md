# CRM card history final polish

Дата: 2026-06-20
Ветка: `codex/etap1-crm-card-assembler`

## Что изменено

- Сняты самодельные лимиты внутренней карточки/preview для `Последняя сводка`, `История общения`, `Следующий шаг`, `Tallanto`.
- Старые writeback-лимиты `MAX_NEXT_STEP_CHARS=800`, `MAX_LAST_SUMMARY_CHARS=1200`, `MAX_AUTO_HISTORY_CHARS=1600` заменены на консервативный лимит textarea-поля AMO `AMO_TEXTAREA_FIELD_CHAR_LIMIT=60000`.
- `История общения` теперь может собираться через кэшируемую LLM-сводку по хэшу входа; fallback не кэшируется как LLM-ответ.
- Почтовый стаб `Email handoff: N сообщений` исключён из истории.
- Сводчик чистит веб-мусор и служебные хвосты: `End of History`, `Apache 2.0 License`, `Итог: ...`, `Контакты: канал: ...`.
- Для `codex exec` добавлен `--ignore-user-config`, чтобы локальный `service_tier` в `~/.codex/config.toml` не ломал preview-сборку.

## Подтверждение поля AMO

Локальный контракт проекта подтверждает, что целевые длинные поля являются `textarea`:

- `docs/AMO_DEAL_FIELDS_AND_UNIQUENESS_CHECK_2026-05-12.md`: `Авто история общения`, `Последняя AI-сводка`, `AI-рекомендованный следующий шаг` = `textarea`.
- `tests/test_amo_writeback_guards.py`: guard блокирует не-`textarea`/api-only для этих полей.

Live-read AMO в этой задаче не выполнялся; live-write не выполнялся.

## Preview

Папка:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_llm_history_r2/`

Сборка:

- rows: 10
- ready_yes: 5
- ready_no: 5
- history_summary provider: `codex_cli`
- model: `gpt-5.4-mini`
- cache_hits: 9
- cache_misses: 1
- llm_calls: 1 при повторной сборке
- rule_fallbacks: 0
- errors: 0

Машинная проверка `История общения`:

- `Email handoff`: 0
- `Связанных писем`: 0
- `End of History`: 0
- `Apache 2.0 License`: 0
- `Итог:` / `Контакты:` служебными хвостами: 0
- сырьевые метки `mango_call` / `whatsapp_message` / `source_system`: 0
- `[сжато]` / `…`: 0
- старое имя `Последняя AI-сводка`: 0

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py tests/test_amo_writeback_guards.py`
  - `45 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `3380 passed, 5 skipped, 1 warning`

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- История стала компактной сводкой, а не сырой склейкой.
- В preview нет служебного мусора, email-заглушек, HTML/license-хвостов и маркеров обрезки.
- Поля остаются внутренними manager-only; live-write не выполнялся.

Остаточный риск:

- Preview собран на выборке 10 строк; нужен регрейд архитектора глазами на XLSX перед массовым применением.
- Лимит 60000 — консервативная защита для textarea writeback по локальному контракту; фактическую живую ёмкость AMO надо подтвердить readback-гейтом на следующем live-write этапе.
