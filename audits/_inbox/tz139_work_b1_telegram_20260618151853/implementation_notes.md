# TZ139 Work B1: Telegram

# Что сделано

- `scripts/import_telegram_export_to_timeline.py:54` добавлен `identity_export_dir` и авто-подхват соседнего `*_max` / `*_with_contacts` для `dialogs.jsonl` с телефонами и username. Сообщения остаются из полного архива.
- `scripts/import_telegram_export_to_timeline.py:205` расширен Telegram normalizer: пишет `telegram_message` и `telegram_dialog`, создаёт `telegram_dialog` opportunity, сохраняет текстовые chunks как `allowed_for_bot=false`.
- `scripts/import_telegram_export_to_timeline.py:451` добавлен resolver phone+username: unique -> `strong_unique`, shared/family/ambiguous -> `ambiguous`, no match -> отдельная `unmatched` identity.
- `scripts/import_telegram_export_to_timeline.py:524` dry-run отчёт теперь показывает источник identity sidecar, counts по матчам и source unchanged.
- `src/mango_mvp/customer_timeline/store.py:54` расширен P0 scrub для вложенных Telegram raw/update/message ключей.
- `tests/test_import_telegram_export_to_timeline.py:187` и далее покрывают dialog+message events, unmatched, phone match, ambiguous phone, username match, sidecar contacts, idempotency.
- `tests/test_customer_timeline_store.py:454` проверяет физический SQLite dump: raw Telegram payload ключи не сохраняются.

# Как проверялось

См. `test_output.txt` и `realdata_report.md`.

# Что осталось

- B1 остановлен на ревью Клода. B2/B3 не начинал.
- В canonical DB на текущей связке полного архива + `_max` sidecar найдено 189 unique phone matches и 0 ambiguous; семейная ambiguous-логика покрыта тестом, но независимый регрейд по сырью остаётся за Клодом.
