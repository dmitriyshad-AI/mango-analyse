# Risk Review

Основные риски и защита:

1. Утечка полного customer_profile.
   - Защита: новый runtime context вызывает только `CustomerTimelineReadApi.bot_context(..., allowed_only=True)`.
   - Дополнительно: в итоговом payload есть safety-флаги `customer_profile_included=false`, `raw_timeline_events_included=false`, `raw_ids_included=false`.

2. Смешение брендов.
   - Защита: active brand обязан быть `foton` или `unpk`; каждый chunk обязан иметь `relevance_tags` с `bot_safe` и точным active brand.
   - Unknown brand не допускается в prompt.

3. Утечка ПДн или служебных id.
   - Защита: pre-prompt scan в runtime context и direct path.
   - Дополнительно: output sanitizer удаляет `customer:`, `timeline_event:`, `bot_context_chunk:`, `botsafe:`.

4. Ложная уверенность бота на основе памяти.
   - Защита: prompt говорит использовать выжимку только для продолжения диалога; цены, даты и условия брать только из подтвержденных фактов.

5. Live-write риск.
   - Защита: изменения касаются draft context и dynamic sim; автоответ и запись в AMO/Tallanto не включались.

Остаточный риск:

- M1 должен подтвердить качество на сырье. Unit-тесты доказывают границы безопасности, но не доказывают, что менеджер будет чаще отправлять черновик без правки.
