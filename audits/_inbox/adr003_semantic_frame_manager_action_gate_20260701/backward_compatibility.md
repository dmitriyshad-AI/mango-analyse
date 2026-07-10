# Backward compatibility

- OFF behavior: новый флаг default-OFF и не включен в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- При выключенном флаге цепочка возвращает результат как раньше.
- При включенном флаге без posthoc frame `ok` маршрут не меняется, пишется только trace.
- При включенном флаге существующие `draft_for_manager` и `manager_only` не понижаются.
- Клиентский текст не переписывается.
- Existing `frame_decision_shadow` продолжает работать и теперь видит route после manager-action gate.
