# Отчёт: Этап 3, Фаза 1 — bot-safe выжимка в черновике бота

Дата: 2026-06-21.

Ветка: `codex/etap3-faza1-botsafe-bot`.

Что сделано:

- Подключён безопасный read-only слой customer timeline для черновиков Telegram/Wappi.
- Источник контекста строго ограничен: только `bot_context(allowed_only=True)`.
- Полный `customer_profile` боту не передаётся.
- Выжимки фильтруются по `active_brand` канала через `relevance_tags`.
- Добавлен флаг `TELEGRAM_BOT_SAFE_CRM_CONTEXT`, default OFF.
- Direct path prompt получил отдельный блок "Безопасная выжимка клиента".
- Добавлен PII/service-id scan до prompt и расширен выходной sanitizer.
- Исправлено read-only открытие SQLite из путей с пробелами и M1-бандлов на Яндекс.Диске.
- Подготовлен M1 target set для пары OFF/ON.

Боевые данные:

- DB: `product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
- Bot-safe chunks: 17,856.
- Brand tags: foton 1,290; unpk 4,017; unknown 12,549.

Тесты:

- `git diff --check` — passed.
- Targeted tests — 131 passed.
- Full pytest — 3493 passed, 5 skipped, 1 warning.
- Bundle fake smoke target set — completed, 1 dialog, 3 turns, PASS_WITH_NOTES.

M1:

- Target set: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/botsafe_crm_context_20260621/target_set.jsonl`.
- Target set SHA256: `1385c2c05fcd838531a27dc62d1f91190128a3012fd02eaf99505da25d60e4e3`.
- Нужно прогнать пару:
  - OFF: без `TELEGRAM_BOT_SAFE_CRM_CONTEXT`.
  - ON: `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`.

Acceptance focus for M1:

- Память помогает не переспрашивать уже известное.
- Нет утечки CRM/источников/служебных id/телефонов/email.
- Foton и UNPK не смешиваются.
- P0 уходит менеджеру.
- Нет автоответа клиенту.

Откат:

- Снять `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`.

Статус:

- formal_pass: да.
- semantic_pass: PASS_WITH_NOTES, потому что безопасность границ проверена, но эффект качества требует M1 OFF/ON замера по сырью.
