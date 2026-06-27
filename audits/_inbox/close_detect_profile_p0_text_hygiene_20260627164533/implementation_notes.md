# Что сделано

- Включён `TELEGRAM_TONE_CLOSE_DETECT` в канонический профиль `pilot_gold_v1`: флаг добавлен в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`, а `close_detect_enabled()` теперь читает профиль после явных override из context/env.
- Добавлена P0-гигиена текста direct-path за новым флагом `TELEGRAM_DIRECT_P0_TEXT_HYGIENE`, default OFF. Включённый слой заменяет опасные P0-черновики с обещанием возврата/продающим хвостом на нейтральный текст передачи менеджеру.
- P0-гигиена подключена в live direct-path после `_apply_direct_path_model_p0_route()` и до scope/fact guard. P0-маршрут, бренд-гейт и числовые проверки не менялись.

# Как проверялось

- Аудитор подтвердил ТЗ: closing-флаг нужно включать через профиль; P0-гигиена должна стоять в provider-level direct-path, а не в legacy хвосте.
- Новые тесты покрывают profile ON/explicit OFF, молчание close-detect на P0, default OFF для P0-гигиены, provider-level scrub, `payment_dispute` и presale refund policy.

# Что осталось

- Отдать Claude #1 на финальную сверку: closing-fix теперь реально включён профилем, P0-гигиена остаётся выключенной до отдельного регрейда.
- Не включать `TELEGRAM_DIRECT_P0_TEXT_HYGIENE` в профиль без отдельного semantic/regression PASS.
