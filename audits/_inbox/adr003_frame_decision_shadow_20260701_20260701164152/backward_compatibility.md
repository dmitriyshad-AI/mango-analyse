# Обратная совместимость

- Форматы: существующие ключи `semantic_frame` и legacy `semantic_frame_shadow` не менялись. Добавлен новый необязательный `frame_decision_shadow`.
- Потребители: старые потребители, которые не знают `frame_decision_shadow`, продолжают работать. Симулятор читает новый ключ только если он есть.
- Флаги: `TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW` default-OFF и не входит в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- Runtime: нет новых LLM-вызовов, нет новых внешних зависимостей.
