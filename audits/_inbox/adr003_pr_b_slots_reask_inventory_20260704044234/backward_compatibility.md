# Обратная совместимость

- `TELEGRAM_SLOTS_REASK` default-OFF и не в профиле.
- Без флага `do_not_reask_slots` строится как раньше.
- Без `TELEGRAM_SEMANTIC_READING_CLASSES=slots_gsf` hidden slots не создаются.
- `to_prompt_view()` не меняет контракт: `semantic_reading_slots` наружу не отдаётся.
- `known_slots` и `client_confirmed_slots` не пополняются semantic reading values.
