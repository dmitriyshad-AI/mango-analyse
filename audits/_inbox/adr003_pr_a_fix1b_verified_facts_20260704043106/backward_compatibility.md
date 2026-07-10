# Обратная совместимость

- Default-OFF: без `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS=1` старое понижение сохраняется.
- Флаг не добавлен в `pilot_gold_v1` и не вносится в профильный default-on кортеж.
- Новых regex/marker helpers не добавлено; moratorium guard зелёный.
- `semantic_reading_trace` получает класс `fix1b` только при включённом флаге; allowlist reading-масок не расширен.
- P0/high-risk, unknown brand, topic policy, live-status и availability-promise floor остаются выше нового коридора.
